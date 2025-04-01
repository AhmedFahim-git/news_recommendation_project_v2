from functools import partial
from pathlib import Path
import json
from datetime import datetime
from typing import Optional
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.adamw import AdamW
import torch.nn.functional as F
from tqdm import tqdm
from .config import (
    DEVICE,
    NUM_WORKERS,
)
from .data_utils import (
    ClassificationTrainDataset,
    rank_group_preds,
    FinalAttentionTrainDataset,
    final_attention_train_collate_fn,
    attention_attention_train_collate_fn,
)
from .batch_size_finder import (
    get_classification_train_batch_size,
    get_attention_train_batch_size,
    get_attention_attention_train_batch_size,
)
from .modeling_utils import ClassificationHead, FinalAttention, WeightedSumModel
from .data_model_helper import (
    get_classification_baseline_scores,
    get_cos_sim_final_score,
    get_cos_sim_scores,
    get_cos_sim_reduce_scores,
)
from .evaluation import score


class ClassificationModelTrainer:
    def __init__(
        self,
        model: ClassificationHead,
        train_embeddings: torch.Tensor,
        train_rev_index: np.ndarray,
        train_impression_len_list: np.ndarray,
        train_labels: np.ndarray,
        val_embeddings: torch.Tensor,
        val_rev_index: np.ndarray,
        val_impression_len_list: np.ndarray,
        val_labels: np.ndarray,
        log_dir: Optional[Path] = None,
        ckpt_dir: Optional[Path] = None,
        exp_name: str = "",
        rng: np.random.Generator = np.random.default_rng(1234),
    ):
        self.ckpt_dir = ckpt_dir
        self.log_dir = log_dir
        self.rng = rng
        self.exp_name = exp_name
        self.train_dataset = ClassificationTrainDataset(
            train_embeddings,
            train_rev_index,
            train_impression_len_list,
            train_labels,
            self.rng,
        )
        self.train_embeddings = train_embeddings
        self.train_rev_index = train_rev_index
        self.train_impression_len_list = train_impression_len_list
        self.train_labels = train_labels
        self.val_embeddings = val_embeddings
        self.val_rev_index = val_rev_index
        self.val_impression_len_list = val_impression_len_list
        self.val_labels = val_labels

        self.model = model

        self.optimizer = AdamW(self.model.parameters(), lr=1e-6)

        self.train_batch_size = (
            get_classification_train_batch_size(self.model, self.optimizer) // 2
        )

        print(
            f"Batch size for training Classification model is {self.train_batch_size}"
        )

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=NUM_WORKERS,
        )
        self.loss_fn = torch.nn.MarginRankingLoss(2)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=2
        )

    def train_one_epoch(self):
        self.model.train()
        losses, counts = [], []
        for pos_embeds, neg_embeds in tqdm(self.train_dataloader):
            self.optimizer.zero_grad()
            pos_res = self.model(pos_embeds.to(device=DEVICE)).squeeze()
            neg_res = self.model(neg_embeds.to(device=DEVICE)).squeeze()
            loss = self.loss_fn(
                pos_res, neg_res, torch.tensor([1], dtype=torch.int32, device=DEVICE)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()
            losses.append(loss.item())
            counts.append(len(pos_embeds))
        self.optimizer.zero_grad()
        train_epoch_loss = np.dot(losses, counts) / sum(counts)
        return train_epoch_loss

    def train(self, num_epochs: int):
        best_val_score = -np.inf
        for i in range(num_epochs):
            train_epoch_loss = self.train_one_epoch()
            train_eval_score = score(
                rank_group_preds(
                    get_classification_baseline_scores(
                        self.train_embeddings, self.model, self.train_rev_index
                    )["baseline_scores"],
                    self.train_impression_len_list,
                ),
                self.train_labels,
            )
            val_eval_score = score(
                rank_group_preds(
                    get_classification_baseline_scores(
                        self.val_embeddings, self.model, self.val_rev_index
                    )["baseline_scores"],
                    self.val_impression_len_list,
                ),
                self.val_labels,
            )

            print(
                i + 1,
                train_epoch_loss,
                "\nTrain Score:",
                train_eval_score,
                "\nVal Score:",
                val_eval_score,
            )
            if self.log_dir:
                self.log_dir.mkdir(parents=True, exist_ok=True)
                with open(self.log_dir / "train_classification_score.jsonl", "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "timestamp": datetime.now().isoformat(),
                                "exp_name": self.exp_name,
                                "epoch": i + 1,
                                "scores": train_eval_score,
                                "loss": train_epoch_loss,
                            }
                        )
                        + "\n"
                    )
                with open(self.log_dir / "eval_classification_score.jsonl", "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "timestamp": datetime.now().isoformat(),
                                "exp_name": self.exp_name,
                                "epoch": i + 1,
                                "scores": val_eval_score,
                            }
                        )
                        + "\n"
                    )
            mean_val_score = float(np.mean(list(val_eval_score.values())[:-1]))
            self.scheduler.step(mean_val_score)
            if self.ckpt_dir:
                self.ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save(self.model.state_dict(), self.ckpt_dir / f"Epoch_{i+1}.pt")
                if mean_val_score > best_val_score:
                    best_val_score = mean_val_score
                    torch.save(
                        self.model.state_dict(),
                        self.ckpt_dir / f"Best_model_{self.exp_name}.pt",
                    )
            self.train_dataset.reset()


class AttentionWeightTrainer:
    def __init__(
        self,
        attention_model: torch.nn.Module,
        weight_model: WeightedSumModel,
        train_history_rev_index: np.ndarray,
        train_history_len_list: np.ndarray,
        train_news_rev_index: np.ndarray,
        train_impression_len_list: np.ndarray,
        train_news_embeddings: torch.Tensor,
        train_classification_score: np.ndarray,
        train_labels: np.ndarray,
        val_history_rev_index: np.ndarray,
        val_history_len_list: np.ndarray,
        val_news_rev_index: np.ndarray,
        val_impression_len_list: np.ndarray,
        val_news_embeddings: torch.Tensor,
        val_classification_score: np.ndarray,
        val_labels: np.ndarray,
        log_dir: Optional[Path] = None,
        ckpt_dir: Optional[Path] = None,
        weight_ckpt_dir: Optional[Path] = None,
        exp_name: str = "",
        rng: np.random.Generator = np.random.default_rng(1234),
    ):
        self.rng = rng
        self.log_dir = log_dir
        self.exp_name = exp_name
        self.ckpt_dir = ckpt_dir
        self.weight_ckpt_dir = weight_ckpt_dir
        self.attention_model = attention_model
        self.weight_model = weight_model

        self.optimizer = AdamW(
            list(self.attention_model.parameters())
            + list(self.weight_model.parameters()),
            lr=1e-5,
        )

        self.loss_fn = torch.nn.MarginRankingLoss(2)
        self.train_batch_size = get_attention_train_batch_size(
            self.attention_model, self.optimizer
        )
        print(f"Batch size for training Attention model is {self.train_batch_size}")

        self.train_dataset = FinalAttentionTrainDataset(
            history_rev_index=train_history_rev_index,
            history_len_list=train_history_len_list,
            news_rev_index=train_news_rev_index,
            impression_len_list=train_impression_len_list,
            labels=train_labels,
            batch_size=self.train_batch_size,
            rng=self.rng,
        )
        self.train_news_embedding = train_news_embeddings
        self.train_classification_score = torch.tensor(train_classification_score)

        self.train_eval_score_func = partial(
            get_cos_sim_final_score,
            history_rev_index=train_history_rev_index,
            history_len_list=train_history_len_list,
            news_rev_index=train_news_rev_index,
            impression_len_list=train_impression_len_list,
            news_embeddings=train_news_embeddings,
            classification_score=train_classification_score,
            attention_model=self.attention_model,
            weight_model=self.weight_model,
        )
        self.train_labels = train_labels
        self.train_impression_len_list = train_impression_len_list

        self.val_eval_score_func = partial(
            get_cos_sim_final_score,
            history_rev_index=val_history_rev_index,
            history_len_list=val_history_len_list,
            news_rev_index=val_news_rev_index,
            impression_len_list=val_impression_len_list,
            news_embeddings=val_news_embeddings,
            classification_score=val_classification_score,
            attention_model=self.attention_model,
            weight_model=self.weight_model,
        )
        self.val_labels = val_labels
        self.val_impression_len_list = val_impression_len_list

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            pin_memory=True,
            num_workers=NUM_WORKERS,
            shuffle=False,
            collate_fn=final_attention_train_collate_fn,
        )

    def _get_val_score(self, res, impression_len_list, labels):
        grouped_preds = rank_group_preds(res, impression_len_list)
        return score(grouped_preds, labels)

    def train_one_epoch(self):
        self.attention_model.train()
        self.weight_model.train()

        losses, counts = [], []
        for (
            history_indices,
            history_attention_mask,
            history_rev_index,
            news_ind_pos_neg,
        ) in tqdm(self.train_dataloader):
            self.optimizer.zero_grad()
            model_input = self.train_news_embedding[
                history_indices
            ] * history_attention_mask.unsqueeze(-1)
            outputs = self.attention_model(
                model_input.to(DEVICE), history_attention_mask.to(DEVICE)
            )

            pos_neg_cos = F.cosine_similarity(
                outputs[history_rev_index.repeat(2).to(DEVICE)],
                self.train_news_embedding[news_ind_pos_neg].to(DEVICE),
            )
            res = self.weight_model(
                pos_neg_cos,
                self.train_classification_score[news_ind_pos_neg].to(DEVICE),
            )
            loss = self.loss_fn(
                *torch.chunk(res, 2),
                torch.tensor([1], device=DEVICE, dtype=torch.float32),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.attention_model.parameters())
                + list(self.weight_model.parameters()),
                max_norm=0.5,
            )
            self.optimizer.step()
            losses.append(loss.item())
            counts.append(len(history_rev_index))
        self.optimizer.zero_grad()
        train_epoch_loss = np.dot(losses, counts) / sum(counts)
        return train_epoch_loss

    def train(
        self,
        num_epochs,
    ):
        best_val_score = -np.inf
        for i in range(num_epochs):
            train_epoch_loss = self.train_one_epoch()
            train_eval_score = self._get_val_score(
                self.train_eval_score_func(),
                self.train_impression_len_list,
                self.train_labels,
            )
            val_eval_score = self._get_val_score(
                self.val_eval_score_func(),
                self.val_impression_len_list,
                self.val_labels,
            )

            print(
                i + 1,
                train_epoch_loss,
                "\nTrain Score:",
                train_eval_score,
                "\nVal Score:",
                val_eval_score,
            )
            if self.log_dir:
                self.log_dir.mkdir(parents=True, exist_ok=True)
                with open(self.log_dir / "train_final_history_score.jsonl", "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "timestamp": datetime.now().isoformat(),
                                "exp_name": self.exp_name,
                                "epoch": i + 1,
                                "scores": train_eval_score,
                                "loss": train_epoch_loss,
                            }
                        )
                        + "\n"
                    )
                with open(self.log_dir / "eval_final_history_score.jsonl", "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "timestamp": datetime.now().isoformat(),
                                "exp_name": self.exp_name,
                                "epoch": i + 1,
                                "scores": val_eval_score,
                            }
                        )
                        + "\n"
                    )
            mean_val_score = float(np.mean(list(val_eval_score.values())[:-1]))
            if self.ckpt_dir:
                self.ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    self.attention_model.state_dict(), self.ckpt_dir / f"Epoch_{i+1}.pt"
                )
                if mean_val_score > best_val_score:
                    torch.save(
                        self.attention_model.state_dict(),
                        self.ckpt_dir / f"Best_model_{self.exp_name}.pt",
                    )
            if self.weight_ckpt_dir:
                self.weight_ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    self.weight_model.state_dict(),
                    self.weight_ckpt_dir / f"Epoch_{i+1}.pt",
                )
                if mean_val_score > best_val_score:
                    torch.save(
                        self.weight_model.state_dict(),
                        self.weight_ckpt_dir / f"Best_model_{self.exp_name}.pt",
                    )
            if mean_val_score > best_val_score:
                best_val_score = mean_val_score
            self.train_dataset.reset()


class AttentionTrainer:
    def __init__(
        self,
        attention_model: torch.nn.Module,
        train_history_rev_index: np.ndarray,
        train_history_len_list: np.ndarray,
        train_news_rev_index: np.ndarray,
        train_impression_len_list: np.ndarray,
        train_news_embeddings: torch.Tensor,
        train_labels: np.ndarray,
        val_history_rev_index: np.ndarray,
        val_history_len_list: np.ndarray,
        val_news_rev_index: np.ndarray,
        val_impression_len_list: np.ndarray,
        val_news_embeddings: torch.Tensor,
        val_labels: np.ndarray,
        log_dir: Optional[Path] = None,
        ckpt_dir: Optional[Path] = None,
        exp_name: str = "",
        max_neg_ratio: Optional[float] = None,
        max_pos_ratio: Optional[float] = None,
        rng: np.random.Generator = np.random.default_rng(1234),
    ):
        self.rng = rng
        self.log_dir = log_dir
        self.exp_name = exp_name
        self.ckpt_dir = ckpt_dir
        self.attention_model = attention_model

        self.optimizer = AdamW(
            self.attention_model.parameters(),
            lr=1e-5,
        )

        self.loss_fn = torch.nn.MarginRankingLoss(2)
        self.train_batch_size = get_attention_train_batch_size(
            self.attention_model, self.optimizer
        )
        print(f"Batch size for training Attention model is {self.train_batch_size}")

        self.train_dataset = FinalAttentionTrainDataset(
            history_rev_index=train_history_rev_index,
            history_len_list=train_history_len_list,
            news_rev_index=train_news_rev_index,
            impression_len_list=train_impression_len_list,
            labels=train_labels,
            batch_size=self.train_batch_size,
            max_neg_raio=max_neg_ratio,
            max_pos_ratio=max_pos_ratio,
            rng=self.rng,
        )
        self.train_news_embedding = train_news_embeddings

        self.train_eval_score_func = partial(
            get_cos_sim_scores,
            history_rev_index=train_history_rev_index,
            history_len_list=train_history_len_list,
            news_rev_index=train_news_rev_index,
            impression_len_list=train_impression_len_list,
            news_embeddings=train_news_embeddings,
            model=self.attention_model,
        )
        self.train_labels = train_labels
        self.train_impression_len_list = train_impression_len_list

        self.val_eval_score_func = partial(
            get_cos_sim_scores,
            history_rev_index=val_history_rev_index,
            history_len_list=val_history_len_list,
            news_rev_index=val_news_rev_index,
            impression_len_list=val_impression_len_list,
            news_embeddings=val_news_embeddings,
            model=self.attention_model,
        )
        self.val_labels = val_labels
        self.val_impression_len_list = val_impression_len_list

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            pin_memory=True,
            num_workers=NUM_WORKERS,
            shuffle=False,
            collate_fn=final_attention_train_collate_fn,
        )

    def _get_val_score(self, res, impression_len_list, labels):
        grouped_preds = rank_group_preds(
            res.detach().cpu().numpy(), impression_len_list
        )
        return score(grouped_preds, labels)

    def train_one_epoch(self):
        self.attention_model.train()

        losses, counts = [], []
        for (
            history_indices,
            history_attention_mask,
            history_rev_index,
            news_ind_pos_neg,
        ) in tqdm(self.train_dataloader):
            self.optimizer.zero_grad()
            model_input = self.train_news_embedding[
                history_indices
            ] * history_attention_mask.unsqueeze(-1)
            outputs = self.attention_model(
                model_input.to(DEVICE), history_attention_mask.to(DEVICE)
            )

            res = F.cosine_similarity(
                outputs[history_rev_index.repeat(2).to(DEVICE)],
                self.train_news_embedding[news_ind_pos_neg].to(DEVICE),
            )

            loss = self.loss_fn(
                *torch.chunk(res, 2),
                torch.tensor([1], device=DEVICE, dtype=torch.float32),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.attention_model.parameters(),
                max_norm=0.5,
            )
            self.optimizer.step()
            losses.append(loss.item())
            counts.append(len(history_rev_index))
        self.optimizer.zero_grad()
        train_epoch_loss = np.dot(losses, counts) / sum(counts)
        return train_epoch_loss

    def train(
        self,
        num_epochs,
    ):
        best_val_score = -np.inf
        for i in range(num_epochs):
            train_epoch_loss = self.train_one_epoch()
            train_eval_score = self._get_val_score(
                self.train_eval_score_func(),
                self.train_impression_len_list,
                self.train_labels,
            )
            val_eval_score = self._get_val_score(
                self.val_eval_score_func(),
                self.val_impression_len_list,
                self.val_labels,
            )

            print(
                i + 1,
                train_epoch_loss,
                "\nTrain Score:",
                train_eval_score,
                "\nVal Score:",
                val_eval_score,
            )
            if self.log_dir:
                self.log_dir.mkdir(parents=True, exist_ok=True)
                with open(self.log_dir / "train_final_history_score.jsonl", "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "timestamp": datetime.now().isoformat(),
                                "exp_name": self.exp_name,
                                "epoch": i + 1,
                                "scores": train_eval_score,
                                "loss": train_epoch_loss,
                            }
                        )
                        + "\n"
                    )
                with open(self.log_dir / "eval_final_history_score.jsonl", "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "timestamp": datetime.now().isoformat(),
                                "exp_name": self.exp_name,
                                "epoch": i + 1,
                                "scores": val_eval_score,
                            }
                        )
                        + "\n"
                    )
            mean_val_score = float(np.mean(list(val_eval_score.values())[:-1]))
            if self.ckpt_dir:
                self.ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    self.attention_model.state_dict(), self.ckpt_dir / f"Epoch_{i+1}.pt"
                )
                if mean_val_score > best_val_score:
                    torch.save(
                        self.attention_model.state_dict(),
                        self.ckpt_dir / f"Best_model_{self.exp_name}.pt",
                    )
                    best_val_score = mean_val_score

            self.train_dataset.reset()


class AttentionReduceTrainer:
    def __init__(
        self,
        attention_model: torch.nn.Module,
        reduce_model: torch.nn.Module,
        train_history_rev_index: np.ndarray,
        train_history_len_list: np.ndarray,
        train_news_rev_index: np.ndarray,
        train_impression_len_list: np.ndarray,
        train_news_embeddings: torch.Tensor,
        train_labels: np.ndarray,
        val_history_rev_index: np.ndarray,
        val_history_len_list: np.ndarray,
        val_news_rev_index: np.ndarray,
        val_impression_len_list: np.ndarray,
        val_news_embeddings: torch.Tensor,
        val_labels: np.ndarray,
        log_dir: Optional[Path] = None,
        ckpt_dir: Optional[Path] = None,
        reduce_ckpt_dir: Optional[Path] = None,
        exp_name: str = "",
        max_neg_ratio: Optional[float] = None,
        max_pos_ratio: Optional[float] = None,
        rng: np.random.Generator = np.random.default_rng(1234),
    ):
        self.rng = rng
        self.log_dir = log_dir
        self.exp_name = exp_name
        self.ckpt_dir = ckpt_dir
        self.reduce_ckpt_dir = reduce_ckpt_dir
        self.attention_model = attention_model
        self.reduce_model = reduce_model

        self.optimizer = AdamW(
            list(self.attention_model.parameters())
            + list(self.reduce_model.parameters()),
            lr=1e-5,
        )

        self.loss_fn = torch.nn.MarginRankingLoss(2)
        self.train_batch_size = get_attention_train_batch_size(
            self.attention_model, self.optimizer
        )
        print(f"Batch size for training Attention model is {self.train_batch_size}")

        self.train_dataset = FinalAttentionTrainDataset(
            history_rev_index=train_history_rev_index,
            history_len_list=train_history_len_list,
            news_rev_index=train_news_rev_index,
            impression_len_list=train_impression_len_list,
            labels=train_labels,
            batch_size=self.train_batch_size,
            max_neg_raio=max_neg_ratio,
            max_pos_ratio=max_pos_ratio,
            rng=self.rng,
        )
        self.train_news_embedding = train_news_embeddings

        self.train_eval_score_func = partial(
            get_cos_sim_reduce_scores,
            history_rev_index=train_history_rev_index,
            history_len_list=train_history_len_list,
            news_rev_index=train_news_rev_index,
            impression_len_list=train_impression_len_list,
            news_embeddings=train_news_embeddings,
            attention_model=self.attention_model,
            reduce_model=self.reduce_model,
        )
        self.train_labels = train_labels
        self.train_impression_len_list = train_impression_len_list

        self.val_eval_score_func = partial(
            get_cos_sim_reduce_scores,
            history_rev_index=val_history_rev_index,
            history_len_list=val_history_len_list,
            news_rev_index=val_news_rev_index,
            impression_len_list=val_impression_len_list,
            news_embeddings=val_news_embeddings,
            attention_model=self.attention_model,
            reduce_model=self.reduce_model,
        )
        self.val_labels = val_labels
        self.val_impression_len_list = val_impression_len_list

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            pin_memory=True,
            num_workers=NUM_WORKERS,
            shuffle=False,
            collate_fn=final_attention_train_collate_fn,
        )

    def _get_val_score(self, res, impression_len_list, labels):
        grouped_preds = rank_group_preds(
            res.detach().cpu().numpy(), impression_len_list
        )
        return score(grouped_preds, labels)

    def train_one_epoch(self):
        self.attention_model.train()

        losses, counts = [], []
        for (
            history_indices,
            history_attention_mask,
            history_rev_index,
            news_ind_pos_neg,
        ) in tqdm(self.train_dataloader):
            self.optimizer.zero_grad()
            model_input = self.reduce_model(
                (
                    self.train_news_embedding[history_indices]
                    * history_attention_mask.unsqueeze(-1)
                ).to(DEVICE)
            )
            outputs = self.attention_model(
                model_input.to(DEVICE), history_attention_mask.to(DEVICE)
            )

            res = F.cosine_similarity(
                outputs[history_rev_index.repeat(2).to(DEVICE)],
                self.reduce_model(
                    self.train_news_embedding[news_ind_pos_neg].to(DEVICE)
                ),
            )

            loss = self.loss_fn(
                *torch.chunk(res, 2),
                torch.tensor([1], device=DEVICE, dtype=torch.float32),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.attention_model.parameters(),
                max_norm=0.5,
            )
            self.optimizer.step()
            losses.append(loss.item())
            counts.append(len(history_rev_index))
        self.optimizer.zero_grad()
        train_epoch_loss = np.dot(losses, counts) / sum(counts)
        return train_epoch_loss

    def train(
        self,
        num_epochs,
    ):
        best_val_score = -np.inf
        for i in range(num_epochs):
            train_epoch_loss = self.train_one_epoch()
            train_eval_score = self._get_val_score(
                self.train_eval_score_func(),
                self.train_impression_len_list,
                self.train_labels,
            )
            val_eval_score = self._get_val_score(
                self.val_eval_score_func(),
                self.val_impression_len_list,
                self.val_labels,
            )

            print(
                i + 1,
                train_epoch_loss,
                "\nTrain Score:",
                train_eval_score,
                "\nVal Score:",
                val_eval_score,
            )
            if self.log_dir:
                self.log_dir.mkdir(parents=True, exist_ok=True)
                with open(self.log_dir / "train_final_history_score.jsonl", "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "timestamp": datetime.now().isoformat(),
                                "exp_name": self.exp_name,
                                "epoch": i + 1,
                                "scores": train_eval_score,
                                "loss": train_epoch_loss,
                            }
                        )
                        + "\n"
                    )
                with open(self.log_dir / "eval_final_history_score.jsonl", "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "timestamp": datetime.now().isoformat(),
                                "exp_name": self.exp_name,
                                "epoch": i + 1,
                                "scores": val_eval_score,
                            }
                        )
                        + "\n"
                    )
            mean_val_score = float(np.mean(list(val_eval_score.values())[:-1]))
            if self.ckpt_dir:
                self.ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    self.attention_model.state_dict(), self.ckpt_dir / f"Epoch_{i+1}.pt"
                )
                if mean_val_score > best_val_score:
                    torch.save(
                        self.attention_model.state_dict(),
                        self.ckpt_dir / f"Best_model_{self.exp_name}.pt",
                    )
            if self.reduce_ckpt_dir:
                self.reduce_ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    self.reduce_model.state_dict(),
                    self.reduce_ckpt_dir / f"Epoch_{i+1}.pt",
                )
                if mean_val_score > best_val_score:
                    torch.save(
                        self.reduce_model.state_dict(),
                        self.reduce_ckpt_dir / f"Best_model_{self.exp_name}.pt",
                    )
            if mean_val_score > best_val_score:
                best_val_score = mean_val_score
            self.train_dataset.reset()


class AttentionAttentionTrainer:
    def __init__(
        self,
        db_name: str,
        token_attention_model: torch.nn.Module,
        final_attention_model: torch.nn.Module,
        train_history_rev_index: np.ndarray,
        train_history_len_list: np.ndarray,
        train_news_rev_index: np.ndarray,
        train_impression_len_list: np.ndarray,
        train_labels: np.ndarray,
        log_dir: Optional[Path] = None,
        token_ckpt_dir: Optional[Path] = None,
        final_attn_ckpt_dir: Optional[Path] = None,
        exp_name: str = "",
        max_neg_ratio: Optional[float] = None,
        max_pos_ratio: Optional[float] = None,
        rng: np.random.Generator = np.random.default_rng(1234),
    ):
        self.rng = rng
        self.log_dir = log_dir
        self.exp_name = exp_name
        self.token_ckpt_dir = token_ckpt_dir
        self.final_attn_ckpt_dir = final_attn_ckpt_dir
        self.token_attention_model = token_attention_model
        self.final_attention_model = final_attention_model

        self.optimizer = AdamW(
            list(self.token_attention_model.parameters())
            + list(self.final_attention_model.parameters()),
            lr=1e-4,
        )

        self.loss_fn = torch.nn.MarginRankingLoss(2)
        self.train_batch_size = get_attention_attention_train_batch_size(
            token_model=self.token_attention_model,
            final_attention=self.final_attention_model,
            optimizer=self.optimizer,
        )
        print(f"Batch size for training Attention model is {self.train_batch_size}")

        self.train_dataset = FinalAttentionTrainDataset(
            history_rev_index=train_history_rev_index,
            history_len_list=train_history_len_list,
            news_rev_index=train_news_rev_index,
            impression_len_list=train_impression_len_list,
            labels=train_labels,
            batch_size=self.train_batch_size,
            max_neg_raio=max_neg_ratio,
            max_pos_ratio=max_pos_ratio,
            rng=self.rng,
        )

        train_collate_fn = partial(
            attention_attention_train_collate_fn, db_name=db_name, rng=self.rng
        )

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            pin_memory=True,
            num_workers=NUM_WORKERS,
            shuffle=False,
            collate_fn=train_collate_fn,
        )

    def train_one_epoch(self):
        self.token_attention_model.train()
        self.final_attention_model.train()

        losses, counts = [], []
        batch_num = 0
        for (
            token_embeds,
            token_attn_mask,
            hist_indices,
            hist_attn_mask,
            news_ind_pos_neg,
        ) in tqdm(self.train_dataloader):
            batch_num += 1
            self.optimizer.zero_grad()
            first_res = self.token_attention_model(
                token_embeds.to(DEVICE), token_attn_mask.to(DEVICE)
            )
            second_res = first_res[hist_indices] * hist_attn_mask.unsqueeze(-1).to(
                DEVICE
            )
            outputs = self.final_attention_model(second_res, hist_attn_mask.to(DEVICE))
            # model_input = self.train_news_embedding[
            #     history_indices
            # ] * history_attention_mask.unsqueeze(-1)
            # outputs = self.attention_model(
            #     model_input.to(DEVICE), history_attention_mask.to(DEVICE)
            # )

            res = F.cosine_similarity(
                outputs.repeat((2, 1)).to(DEVICE),
                first_res[news_ind_pos_neg].to(DEVICE),
            )

            loss = self.loss_fn(
                *torch.chunk(res, 2),
                torch.tensor([1], device=DEVICE, dtype=torch.float32),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.token_attention_model.parameters())
                + list(self.final_attention_model.parameters()),
                max_norm=0.5,
            )
            self.optimizer.step()
            losses.append(loss.item())
            counts.append(len(hist_indices))
            if (batch_num % 5000) == 0:
                print(np.dot(losses, counts) / sum(counts))
                if self.token_ckpt_dir:
                    self.token_ckpt_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        self.token_attention_model.state_dict(),
                        self.token_ckpt_dir / f"Epoch_{1}_batch_{batch_num}.pt",
                    )
                if self.final_attn_ckpt_dir:
                    self.final_attn_ckpt_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        self.final_attention_model,
                        self.final_attn_ckpt_dir / f"Epoch_{1}_batch_{batch_num}.pt",
                    )
        self.optimizer.zero_grad()
        train_epoch_loss = np.dot(losses, counts) / sum(counts)
        return train_epoch_loss

    def train(
        self,
        num_epochs,
    ):
        for i in range(num_epochs):
            train_epoch_loss = self.train_one_epoch()
            # train_eval_score = self._get_val_score(
            #     self.train_eval_score_func(),
            #     self.train_impression_len_list,
            #     self.train_labels,
            # )
            # val_eval_score = self._get_val_score(
            #     self.val_eval_score_func(),
            #     self.val_impression_len_list,
            #     self.val_labels,
            # )

            print(
                i + 1,
                train_epoch_loss,
                # "\nTrain Score:",
                # train_eval_score,
                # "\nVal Score:",
                # val_eval_score,
            )
            if self.log_dir:
                self.log_dir.mkdir(parents=True, exist_ok=True)
                with open(self.log_dir / "train_final_history_score.jsonl", "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "timestamp": datetime.now().isoformat(),
                                "exp_name": self.exp_name,
                                "epoch": i + 1,
                                # "scores": train_eval_score,
                                "loss": train_epoch_loss,
                            }
                        )
                        + "\n"
                    )
                # with open(self.log_dir / "eval_final_history_score.jsonl", "a") as f:
                #     f.write(
                #         json.dumps(
                #             {
                #                 "timestamp": datetime.now().isoformat(),
                #                 "exp_name": self.exp_name,
                #                 "epoch": i + 1,
                #                 "scores": val_eval_score,
                #             }
                #         )
                #         + "\n"
                #     )
            # mean_val_score = float(np.mean(list(val_eval_score.values())[:-1]))
            if self.token_ckpt_dir:
                self.token_ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    self.token_attention_model.state_dict(),
                    self.token_ckpt_dir / f"Epoch_{i+1}.pt",
                )
            if self.final_attn_ckpt_dir:
                self.final_attn_ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    self.final_attention_model,
                    self.final_attn_ckpt_dir / f"Epoch_{i+1}.pt",
                )
                # if mean_val_score > best_val_score:
                #     torch.save(
                #         self.attention_model.state_dict(),
                #         self.ckpt_dir / f"Best_model_{self.exp_name}.pt",
                #     )
                #     best_val_score = mean_val_score

            self.train_dataset.reset()
