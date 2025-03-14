from pathlib import Path
from typing import Optional, Any
from functools import partial
from collections.abc import Iterable
import numpy as np
import torch
from .config import NewsDataset, DataSubset, NEWS_TEXT_MAXLEN
from .pipeline import PipelineComponent, check_req_keys
from .modeling_utils import (
    get_classification_head,
    get_weighted_sum_model,
    get_model_and_tokenizer,
    get_final_attention_model,
    get_embed_from_model,
    get_model_eval,
)
from .data_utils import (
    load_dataset,
    split_impressions_and_history,
    NewsTextDataset,
    eval_collate_fn,
    EmbeddingDataset,
)
from .data_model_helper import (
    get_embeddings,
    get_classification_baseline_scores,
    get_final_score,
)
from .trainer import ClassificationModelTrainer, AttentionWeightTrainer


class TransformData(PipelineComponent):
    required_keys = {"behaviors", "news_text_dict"}

    def _process_context(self, context_dict: dict[str, Any]) -> dict[str, Any]:

        behaviors = context_dict["behaviors"]
        news_text_dict = context_dict["news_text_dict"]
        new_context_dict = {
            "ImpressionID": behaviors["ImpressionID"],
            "news_text_dict": news_text_dict,
        }
        new_context_dict.update(
            split_impressions_and_history(
                behaviors["Impressions"], behaviors["History"]
            )
        )
        new_context_dict["history_bool"] = behaviors["History"].notna()
        return new_context_dict

    def transform(self, context_dict: dict[str, Any]) -> dict[str, Any]:
        check_req_keys(self.required_keys, context_dict)

        new_context_dict = self._process_context(context_dict)

        return new_context_dict


class EmbeddingsComponent(PipelineComponent):
    required_keys = {"news_list", "news_text_dict"}

    def __init__(self, model_path: str):
        self.model_path = model_path

    def transform(
        self,
        context_dict: dict[str, Any],
    ) -> dict[str, Any]:
        check_req_keys(self.required_keys, context_dict)

        new_context_dict = context_dict.copy()
        new_context_dict["news_embeddings"] = get_embeddings(
            self.model_path,
            new_context_dict["news_list"],
            new_context_dict["news_text_dict"],
        )

        return new_context_dict


class ClassificationComponent(PipelineComponent):
    required_keys = {"news_embeddings", "impression_rev_ind_array"}
    train_required_keys = required_keys | {"labels", "impression_len_list"}

    def __init__(
        self,
        model_path: Optional[Path] = None,
        log_dir: Optional[Path] = None,
        ckpt_dir: Optional[Path] = None,
        num_epochs: int = 5,
        rng=np.random.default_rng(1234),
    ):
        self.model = get_classification_head(model_path)
        self.num_epochs = num_epochs
        self.rng = rng
        self.log_dir = log_dir
        self.ckpt_dir = ckpt_dir

    def transform(self, context_dict: dict[str, Any]) -> dict[str, Any]:
        check_req_keys(self.required_keys, context_dict)
        new_context_dict = context_dict.copy()
        new_context_dict.update(
            get_classification_baseline_scores(
                new_context_dict["news_embeddings"],
                self.model,
                new_context_dict["impression_rev_ind_array"][0],
            )
        )

        return new_context_dict

    def train(
        self,
        context_dict: dict[str, Any],
        val_context_dict: Optional[dict[str, Any]] = None,
    ):
        assert val_context_dict, "We need the validation data"
        check_req_keys(self.train_required_keys, context_dict)
        check_req_keys(self.train_required_keys, val_context_dict)
        classification_trainer = ClassificationModelTrainer(
            self.model,
            context_dict["news_embeddings"],
            context_dict["impression_rev_ind_array"][0],
            context_dict["impression_len_list"],
            context_dict["labels"],
            val_context_dict["news_embeddings"],
            val_context_dict["impression_rev_ind_array"][0],
            val_context_dict["impression_len_list"],
            val_context_dict["labels"],
            self.log_dir,
            self.ckpt_dir,
            self.rng,
        )
        classification_trainer.train(self.num_epochs)
        if self.ckpt_dir:
            self.model = get_classification_head(self.ckpt_dir / "Best_model.pt")


class AttentionWeightComponent(PipelineComponent):
    required_keys = {
        "news_embeddings",
        "impression_rev_ind_array",
        "impression_len_list",
        "history_rev_ind_array",
        "history_len_list",
        "classification_preds",
        "history_bool",
    }
    train_required_keys = required_keys | {
        "labels",
    }

    def __init__(
        self,
        attention_model_path: Optional[Path] = None,
        weight_model_path: Optional[Path] = None,
        log_dir: Optional[Path] = None,
        ckpt_dir: Optional[Path] = None,
        weight_ckpt_dir: Optional[Path] = None,
        num_epochs=5,
        rng=np.random.default_rng(1234),
    ):
        self.attention_model = get_final_attention_model(attention_model_path)
        self.weight_model = get_weighted_sum_model(weight_model_path)
        self.num_epochs = num_epochs
        self.rng = rng
        self.log_dir = log_dir
        self.ckpt_dir = ckpt_dir
        self.weight_ckpt_dir = weight_ckpt_dir

    def transform(self, context_dict: dict[str, Any]) -> dict[str, Any]:
        check_req_keys(self.required_keys, context_dict)
        new_context_dict = context_dict.copy()
        new_context_dict.update(
            get_final_score(
                new_context_dict["history_rev_ind_array"][0],
                new_context_dict["history_len_list"],
                new_context_dict["impression_rev_ind_array"][0],
                new_context_dict["impression_len_list"],
                new_context_dict["news_embeddings"],
                new_context_dict["classification_preds"],
                new_context_dict["history_bool"],
                self.attention_model,
                self.weight_model,
            )
        )
        return new_context_dict

    def train(
        self,
        context_dict: dict[str, Any],
        val_context_dict: Optional[dict[str, Any]] = None,
    ):
        assert val_context_dict, "We need the validation data"
        check_req_keys(self.train_required_keys, context_dict)
        check_req_keys(self.train_required_keys, val_context_dict)
        print("History_bool sum", context_dict["history_bool"].sum())
        print("length of history len list", len(context_dict["history_len_list"]))

        imp_len_list = list(context_dict["impression_len_list"])
        val_imp_len_list = list(val_context_dict["impression_len_list"])
        attention_weight_trainer = AttentionWeightTrainer(
            self.attention_model,
            self.weight_model,
            context_dict["history_rev_ind_array"][0],
            context_dict["history_len_list"],
            context_dict["impression_rev_ind_array"][0][
                context_dict["history_bool"].repeat(imp_len_list)
            ],
            context_dict["impression_len_list"][context_dict["history_bool"]],
            context_dict["news_embeddings"],
            context_dict["classification_preds"],
            context_dict["labels"][context_dict["history_bool"]],
            val_context_dict["history_rev_ind_array"][0],
            val_context_dict["history_len_list"],
            val_context_dict["impression_rev_ind_array"][0][
                val_context_dict["history_bool"].repeat(val_imp_len_list)
            ],
            val_context_dict["impression_len_list"][val_context_dict["history_bool"]],
            val_context_dict["news_embeddings"],
            val_context_dict["classification_preds"],
            val_context_dict["labels"][val_context_dict["history_bool"]],
            self.log_dir,
            self.ckpt_dir,
            self.weight_ckpt_dir,
            self.rng,
        )
        attention_weight_trainer.train(self.num_epochs)
        if self.ckpt_dir:
            self.attention_model = get_final_attention_model(
                self.ckpt_dir / "Best_model.pt"
            )
        if self.weight_ckpt_dir:
            self.weight_model = get_weighted_sum_model(
                self.weight_ckpt_dir / "Best_model.pt"
            )
