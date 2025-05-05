from pathlib import Path
import os
import io
from typing import Optional, Any
import numpy as np
import torch
from azure.storage.blob import ContainerClient, BlobClient
from dotenv import load_dotenv
from .pipeline import PipelineComponent, check_req_keys
from .modeling_utils import (
    get_classification_head,
    get_weighted_sum_model,
    get_final_attention_model,
    get_new_attention_model,
    get_reducing_model,
    get_token_attn_model,
    get_latent_attention_model,
)
from .data_utils import (
    split_impressions_and_history,
)
from .data_model_helper import (
    get_embeddings,
    get_classification_baseline_scores,
    get_final_score,
    get_final_only_attention_score,
    get_final_only_reduce_attention_score,
    store_embeddings,
    apply_token_attn,
    get_final_second_attention_score,
)
from .trainer import (
    ClassificationModelTrainer,
    AttentionWeightTrainer,
    AttentionTrainer,
    AttentionReduceTrainer,
    AttentionAttentionTrainer,
)

load_dotenv()


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
        if "news_dataset" in context_dict:
            new_context_dict["news_dataset"] = context_dict["news_dataset"]
        if "db_name" in context_dict:
            new_context_dict["db_name"] = context_dict["db_name"]
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
        if self.model_path == "nvidia/NV-Embed-v2":
            embeds = get_embeddings(
                self.model_path,
                new_context_dict["news_list"],
                new_context_dict["news_text_dict"],
            )
            new_context_dict["query_news_embeddings"] = embeds[0]
            new_context_dict["news_embeddings"] = embeds[1]
        else:
            new_context_dict["news_embeddings"] = get_embeddings(
                self.model_path,
                new_context_dict["news_list"],
                new_context_dict["news_text_dict"],
            )

        return new_context_dict


class SaveEmbeddingComponent(PipelineComponent):
    required_keys = {"news_embeddings", "news_dataset"}

    def __init__(self, save_dir: Path):
        self.save_dir = save_dir
        # account_url = os.environ["ACCOUNT_URL"]
        # container_name = os.environ["CONTAINER_NAME"]
        # blob_sas_token = os.environ["BLOB_SAS_TOKEN"]

        # self.container = ContainerClient(
        #     account_url=account_url,
        #     container_name=container_name,
        #     credential=blob_sas_token,
        # )

    def transform(
        self,
        context_dict: dict[str, Any],
    ) -> dict[str, Any]:
        check_req_keys(self.required_keys, context_dict)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            context_dict["news_embeddings"],
            self.save_dir / f"{context_dict['news_dataset'].value}.pt",
        )
        # buffer = io.BytesIO()
        # torch.save(context_dict["news_embeddings"], buffer)
        # buffer.seek(0)
        # self.container.upload_blob(
        #     str(self.save_dir / f"{context_dict['news_dataset'].value}.pt"), buffer
        # )
        # buffer.close()
        if "query_news_embeddings" in context_dict:
            torch.save(
                context_dict["query_news_embeddings"],
                self.save_dir / f"query_{context_dict['news_dataset'].value}.pt",
            )
            # buffer = io.BytesIO()
            # torch.save(context_dict["query_news_embeddings"], buffer)
            # buffer.seek(0)
            # self.container.upload_blob(
            #     str(self.save_dir / f"query_{context_dict['news_dataset'].value}.pt"),
            #     buffer,
            # )
            # buffer.close()
        return context_dict


class LoadEmbeddingComponent(PipelineComponent):
    required_keys = {
        "news_dataset",
    }

    def __init__(self, save_dir: Path):
        self.save_dir = save_dir

    def transform(
        self,
        context_dict: dict[str, Any],
    ) -> dict[str, Any]:
        check_req_keys(self.required_keys, context_dict)
        context_dict["news_embeddings"] = torch.load(
            self.save_dir / f"{context_dict['news_dataset'].value}.pt",
            weights_only=True,
        )
        if (self.save_dir / f"query_{context_dict['news_dataset'].value}.pt").exists():
            context_dict["query_news_embeddings"] = torch.load(
                self.save_dir / f"query_{context_dict['news_dataset'].value}.pt",
                weights_only=True,
            )
        return context_dict


class ClassificationComponent(PipelineComponent):
    required_keys = {"news_embeddings", "impression_rev_ind_array"}
    train_required_keys = required_keys | {"labels", "impression_len_list"}

    def __init__(
        self,
        model_path: Optional[Path] = None,
        log_dir: Optional[Path] = None,
        ckpt_dir: Optional[Path] = None,
        num_epochs: int = 5,
        exp_name: str = "",
        rng=np.random.default_rng(1234),
    ):
        self.model = get_classification_head(model_path)
        if model_path is None:
            self.model_trained = False
        else:
            self.model_trained = True
        self.num_epochs = num_epochs
        self.exp_name = exp_name
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
        if self.model_trained:
            return
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
            self.exp_name,
            self.rng,
        )
        classification_trainer.train(self.num_epochs)
        if self.ckpt_dir:
            self.model = get_classification_head(
                self.ckpt_dir / f"Best_model_{self.exp_name}.pt"
            )


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
        exp_name: str = "",
        rng=np.random.default_rng(1234),
    ):
        self.attention_model = get_final_attention_model(attention_model_path)
        self.weight_model = get_weighted_sum_model(weight_model_path)
        self.num_epochs = num_epochs
        self.exp_name = exp_name
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
                # query_news_embeddings=new_context_dict.get("query_news_embeddings"),
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
            self.exp_name,
            self.rng,
        )
        attention_weight_trainer.train(self.num_epochs)
        if self.ckpt_dir:
            self.attention_model = get_final_attention_model(
                self.ckpt_dir / f"Best_model_{self.exp_name}.pt"
            )
        if self.weight_ckpt_dir:
            self.weight_model = get_weighted_sum_model(
                self.weight_ckpt_dir / f"Best_model_{self.exp_name}.pt"
            )


class AttentionComponent(PipelineComponent):
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
        log_dir: Optional[Path] = None,
        ckpt_dir: Optional[Path] = None,
        num_epochs=5,
        exp_name: str = "",
        max_neg_ratio: Optional[float] = None,
        max_pos_ratio: Optional[float] = None,
        rng=np.random.default_rng(1234),
    ):
        self.attention_model = get_final_attention_model(attention_model_path)
        # self.attention_model = get_latent_attention_model(attention_model_path)
        self.num_epochs = num_epochs
        self.exp_name = exp_name
        self.rng = rng
        self.log_dir = log_dir
        self.ckpt_dir = ckpt_dir
        self.max_neg_ratio = max_neg_ratio
        self.max_pos_ratio = max_pos_ratio

    def transform(self, context_dict: dict[str, Any]) -> dict[str, Any]:
        check_req_keys(self.required_keys, context_dict)
        new_context_dict = context_dict.copy()
        new_context_dict.update(
            get_final_only_attention_score(
                new_context_dict["history_rev_ind_array"][0],
                new_context_dict["history_len_list"],
                new_context_dict["impression_rev_ind_array"][0],
                new_context_dict["impression_len_list"],
                new_context_dict["news_embeddings"],
                new_context_dict["classification_preds"],
                new_context_dict["history_bool"],
                self.attention_model,
                # query_news_embeddings=new_context_dict.get("query_news_embeddings"),
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

        imp_len_list = list(context_dict["impression_len_list"])
        val_imp_len_list = list(val_context_dict["impression_len_list"])
        attention_weight_trainer = AttentionTrainer(
            attention_model=self.attention_model,
            train_history_rev_index=context_dict["history_rev_ind_array"][0],
            train_history_len_list=context_dict["history_len_list"],
            train_news_rev_index=context_dict["impression_rev_ind_array"][0][
                context_dict["history_bool"].repeat(imp_len_list)
            ],
            train_impression_len_list=context_dict["impression_len_list"][
                context_dict["history_bool"]
            ],
            train_news_embeddings=context_dict["news_embeddings"],
            train_labels=context_dict["labels"][context_dict["history_bool"]],
            val_history_rev_index=val_context_dict["history_rev_ind_array"][0],
            val_history_len_list=val_context_dict["history_len_list"],
            val_news_rev_index=val_context_dict["impression_rev_ind_array"][0][
                val_context_dict["history_bool"].repeat(val_imp_len_list)
            ],
            val_impression_len_list=val_context_dict["impression_len_list"][
                val_context_dict["history_bool"]
            ],
            val_news_embeddings=val_context_dict["news_embeddings"],
            val_labels=val_context_dict["labels"][val_context_dict["history_bool"]],
            log_dir=self.log_dir,
            ckpt_dir=self.ckpt_dir,
            exp_name=self.exp_name,
            max_neg_ratio=self.max_neg_ratio,
            max_pos_ratio=self.max_pos_ratio,
            rng=self.rng,
            # train_query_news_embeddings=context_dict.get("query_news_embeddings"),
            # val_query_news_embeddings=val_context_dict.get("query_news_embeddings"),
        )
        attention_weight_trainer.train(self.num_epochs)
        if self.ckpt_dir:
            self.attention_model = get_final_attention_model(
                self.ckpt_dir / f"Best_model_{self.exp_name}.pt"
            )


class NewAttentionReduceComponent(PipelineComponent):
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
        reduce_model_path: Optional[Path] = None,
        log_dir: Optional[Path] = None,
        ckpt_dir: Optional[Path] = None,
        reduce_ckpt_dir: Optional[Path] = None,
        num_epochs=5,
        exp_name: str = "",
        max_neg_ratio: Optional[float] = None,
        max_pos_ratio: Optional[float] = None,
        rng=np.random.default_rng(1234),
    ):
        self.attention_model = get_new_attention_model(attention_model_path)
        self.reduce_model = get_reducing_model(reduce_model_path)
        self.num_epochs = num_epochs
        self.exp_name = exp_name
        self.rng = rng
        self.log_dir = log_dir
        self.ckpt_dir = ckpt_dir
        self.reduce_ckpt_dir = reduce_ckpt_dir
        self.max_neg_ratio = max_neg_ratio
        self.max_pos_ratio = max_pos_ratio

    def transform(self, context_dict: dict[str, Any]) -> dict[str, Any]:
        check_req_keys(self.required_keys, context_dict)
        new_context_dict = context_dict.copy()
        new_context_dict.update(
            get_final_only_reduce_attention_score(
                history_rev_index=new_context_dict["history_rev_ind_array"][0],
                history_len_list=new_context_dict["history_len_list"],
                news_rev_index=new_context_dict["impression_rev_ind_array"][0],
                impression_len_list=new_context_dict["impression_len_list"],
                news_embeddings=new_context_dict["news_embeddings"],
                classification_score=new_context_dict["classification_preds"],
                history_bool=new_context_dict["history_bool"],
                attention_model=self.attention_model,
                reduce_model=self.reduce_model,
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

        imp_len_list = list(context_dict["impression_len_list"])
        val_imp_len_list = list(val_context_dict["impression_len_list"])
        attention_weight_trainer = AttentionReduceTrainer(
            attention_model=self.attention_model,
            reduce_model=self.reduce_model,
            train_history_rev_index=context_dict["history_rev_ind_array"][0],
            train_history_len_list=context_dict["history_len_list"],
            train_news_rev_index=context_dict["impression_rev_ind_array"][0][
                context_dict["history_bool"].repeat(imp_len_list)
            ],
            train_impression_len_list=context_dict["impression_len_list"][
                context_dict["history_bool"]
            ],
            train_news_embeddings=context_dict["news_embeddings"],
            train_labels=context_dict["labels"][context_dict["history_bool"]],
            val_history_rev_index=val_context_dict["history_rev_ind_array"][0],
            val_history_len_list=val_context_dict["history_len_list"],
            val_news_rev_index=val_context_dict["impression_rev_ind_array"][0][
                val_context_dict["history_bool"].repeat(val_imp_len_list)
            ],
            val_impression_len_list=val_context_dict["impression_len_list"][
                val_context_dict["history_bool"]
            ],
            val_news_embeddings=val_context_dict["news_embeddings"],
            val_labels=val_context_dict["labels"][val_context_dict["history_bool"]],
            log_dir=self.log_dir,
            ckpt_dir=self.ckpt_dir,
            reduce_ckpt_dir=self.reduce_ckpt_dir,
            exp_name=self.exp_name,
            max_neg_ratio=self.max_neg_ratio,
            max_pos_ratio=self.max_pos_ratio,
            rng=self.rng,
        )
        attention_weight_trainer.train(self.num_epochs)
        if self.ckpt_dir:
            self.attention_model = get_new_attention_model(
                self.ckpt_dir / f"Best_model_{self.exp_name}.pt"
            )
        if self.reduce_ckpt_dir:
            self.reduce_model = get_reducing_model(
                self.reduce_ckpt_dir / f"Best_model_{self.exp_name}.pt"
            )


class NewAttentionComponent(PipelineComponent):
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
        log_dir: Optional[Path] = None,
        ckpt_dir: Optional[Path] = None,
        num_epochs=5,
        exp_name: str = "",
        max_neg_ratio: Optional[float] = None,
        max_pos_ratio: Optional[float] = None,
        rng=np.random.default_rng(1234),
    ):
        self.attention_model = get_new_attention_model(attention_model_path)
        self.num_epochs = num_epochs
        self.exp_name = exp_name
        self.rng = rng
        self.log_dir = log_dir
        self.ckpt_dir = ckpt_dir
        self.max_neg_ratio = max_neg_ratio
        self.max_pos_ratio = max_pos_ratio

    def transform(self, context_dict: dict[str, Any]) -> dict[str, Any]:
        check_req_keys(self.required_keys, context_dict)
        new_context_dict = context_dict.copy()
        new_context_dict.update(
            get_final_only_attention_score(
                new_context_dict["history_rev_ind_array"][0],
                new_context_dict["history_len_list"],
                new_context_dict["impression_rev_ind_array"][0],
                new_context_dict["impression_len_list"],
                new_context_dict["news_embeddings"],
                new_context_dict["classification_preds"],
                new_context_dict["history_bool"],
                self.attention_model,
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

        imp_len_list = list(context_dict["impression_len_list"])
        val_imp_len_list = list(val_context_dict["impression_len_list"])
        attention_weight_trainer = AttentionTrainer(
            attention_model=self.attention_model,
            train_history_rev_index=context_dict["history_rev_ind_array"][0],
            train_history_len_list=context_dict["history_len_list"],
            train_news_rev_index=context_dict["impression_rev_ind_array"][0][
                context_dict["history_bool"].repeat(imp_len_list)
            ],
            train_impression_len_list=context_dict["impression_len_list"][
                context_dict["history_bool"]
            ],
            train_news_embeddings=context_dict["news_embeddings"],
            train_labels=context_dict["labels"][context_dict["history_bool"]],
            val_history_rev_index=val_context_dict["history_rev_ind_array"][0],
            val_history_len_list=val_context_dict["history_len_list"],
            val_news_rev_index=val_context_dict["impression_rev_ind_array"][0][
                val_context_dict["history_bool"].repeat(val_imp_len_list)
            ],
            val_impression_len_list=val_context_dict["impression_len_list"][
                val_context_dict["history_bool"]
            ],
            val_news_embeddings=val_context_dict["news_embeddings"],
            val_labels=val_context_dict["labels"][val_context_dict["history_bool"]],
            log_dir=self.log_dir,
            ckpt_dir=self.ckpt_dir,
            exp_name=self.exp_name,
            max_neg_ratio=self.max_neg_ratio,
            max_pos_ratio=self.max_pos_ratio,
            rng=self.rng,
        )
        attention_weight_trainer.train(self.num_epochs)
        if self.ckpt_dir:
            self.attention_model = get_new_attention_model(
                self.ckpt_dir / f"Best_model_{self.exp_name}.pt"
            )


class StoreEmbeddingsComponent(PipelineComponent):
    required_keys = {"news_list", "news_text_dict"}

    def __init__(self, model_path: str, db_name: str):
        self.model_path = model_path
        self.db_name = db_name

    def transform(
        self,
        context_dict: dict[str, Any],
    ) -> dict[str, Any]:
        check_req_keys(self.required_keys, context_dict)

        context_dict = context_dict.copy()
        store_embeddings(
            self.model_path,
            context_dict["news_list"],
            context_dict["news_text_dict"],
            self.db_name,
        )
        del context_dict["news_text_dict"]

        return context_dict


class AttentionAttentionComponent(PipelineComponent):
    required_keys = {
        "impression_rev_ind_array",
        "impression_len_list",
        "history_rev_ind_array",
        "history_len_list",
        "history_bool",
    }
    train_required_keys = required_keys | {
        "labels",
    }

    def __init__(
        self,
        db_name: str,
        token_attention_model_path: Optional[Path] = None,
        final_attention_model_path: Optional[Path] = None,
        log_dir: Optional[Path] = None,
        token_ckpt_dir: Optional[Path] = None,
        final_attn_ckpt_dir: Optional[Path] = None,
        num_epochs=5,
        exp_name: str = "",
        max_neg_ratio: Optional[float] = None,
        max_pos_ratio: Optional[float] = None,
        rng=np.random.default_rng(1234),
    ):
        self.db_name = db_name
        self.token_attention = get_token_attn_model(token_attention_model_path)
        self.final_attention = get_final_attention_model(final_attention_model_path)
        self.num_epochs = num_epochs
        self.exp_name = exp_name
        self.rng = rng
        self.log_dir = log_dir
        self.token_ckpt_dir = token_ckpt_dir
        self.final_attn_ckpt_dir = final_attn_ckpt_dir
        self.max_neg_ratio = max_neg_ratio
        self.max_pos_ratio = max_pos_ratio

    def transform(self, context_dict: dict[str, Any]):
        return context_dict

    def train(
        self,
        context_dict: dict[str, Any],
        val_context_dict: Optional[dict[str, Any]] = None,
    ):
        check_req_keys(self.train_required_keys, context_dict)
        imp_len_list = list(context_dict["impression_len_list"])
        attn_attn_trainer = AttentionAttentionTrainer(
            db_name=self.db_name,
            token_attention_model=self.token_attention,
            final_attention_model=self.final_attention,
            train_history_rev_index=context_dict["history_rev_ind_array"][0],
            train_history_len_list=context_dict["history_len_list"],
            train_news_rev_index=context_dict["impression_rev_ind_array"][0][
                context_dict["history_bool"].repeat(imp_len_list)
            ],
            train_impression_len_list=context_dict["impression_len_list"][
                context_dict["history_bool"]
            ],
            train_labels=context_dict["labels"][context_dict["history_bool"]],
            log_dir=self.log_dir,
            token_ckpt_dir=self.token_ckpt_dir,
            final_attn_ckpt_dir=self.final_attn_ckpt_dir,
            exp_name=self.exp_name,
            max_neg_ratio=self.max_neg_ratio,
            max_pos_ratio=self.max_pos_ratio,
            rng=self.rng,
        )
        attn_attn_trainer.train(self.num_epochs)


class TokenEmbeddingsComponent(PipelineComponent):
    required_keys = {"news_list", "db_name"}

    def __init__(
        self,
        model_path: Path,
    ):
        self.model_path = model_path

    def transform(
        self,
        context_dict: dict[str, Any],
    ) -> dict[str, Any]:
        check_req_keys(self.required_keys, context_dict)

        new_context_dict = context_dict.copy()
        new_context_dict["news_embeddings"] = apply_token_attn(
            self.model_path,
            new_context_dict["db_name"],
            len(new_context_dict["news_list"]),
        )

        return new_context_dict


class FinalAttentionComponent(PipelineComponent):
    required_keys = {
        "news_embeddings",
        "impression_rev_ind_array",
        "impression_len_list",
        "history_rev_ind_array",
        "history_len_list",
        "history_bool",
    }
    train_required_keys = required_keys | {
        "labels",
    }

    def __init__(
        self,
        attention_model_path: Optional[Path] = None,
        log_dir: Optional[Path] = None,
        ckpt_dir: Optional[Path] = None,
        num_epochs=5,
        exp_name: str = "",
        max_neg_ratio: Optional[float] = None,
        max_pos_ratio: Optional[float] = None,
        rng=np.random.default_rng(1234),
    ):
        self.attention_model = get_final_attention_model(attention_model_path)
        self.num_epochs = num_epochs
        self.exp_name = exp_name
        self.rng = rng
        self.log_dir = log_dir
        self.ckpt_dir = ckpt_dir
        self.max_neg_ratio = max_neg_ratio
        self.max_pos_ratio = max_pos_ratio

    def transform(self, context_dict: dict[str, Any]) -> dict[str, Any]:
        check_req_keys(self.required_keys, context_dict)
        new_context_dict = context_dict.copy()
        new_context_dict.update(
            get_final_second_attention_score(
                new_context_dict["history_rev_ind_array"][0],
                new_context_dict["history_len_list"],
                new_context_dict["impression_rev_ind_array"][0],
                new_context_dict["impression_len_list"],
                new_context_dict["news_embeddings"],
                new_context_dict["history_bool"],
                self.attention_model,
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

        imp_len_list = list(context_dict["impression_len_list"])
        val_imp_len_list = list(val_context_dict["impression_len_list"])
        attention_weight_trainer = AttentionTrainer(
            attention_model=self.attention_model,
            train_history_rev_index=context_dict["history_rev_ind_array"][0],
            train_history_len_list=context_dict["history_len_list"],
            train_news_rev_index=context_dict["impression_rev_ind_array"][0][
                context_dict["history_bool"].repeat(imp_len_list)
            ],
            train_impression_len_list=context_dict["impression_len_list"][
                context_dict["history_bool"]
            ],
            train_news_embeddings=context_dict["news_embeddings"],
            train_labels=context_dict["labels"][context_dict["history_bool"]],
            val_history_rev_index=val_context_dict["history_rev_ind_array"][0],
            val_history_len_list=val_context_dict["history_len_list"],
            val_news_rev_index=val_context_dict["impression_rev_ind_array"][0][
                val_context_dict["history_bool"].repeat(val_imp_len_list)
            ],
            val_impression_len_list=val_context_dict["impression_len_list"][
                val_context_dict["history_bool"]
            ],
            val_news_embeddings=val_context_dict["news_embeddings"],
            val_labels=val_context_dict["labels"][val_context_dict["history_bool"]],
            log_dir=self.log_dir,
            ckpt_dir=self.ckpt_dir,
            exp_name=self.exp_name,
            max_neg_ratio=self.max_neg_ratio,
            max_pos_ratio=self.max_pos_ratio,
            rng=self.rng,
        )
        attention_weight_trainer.train(self.num_epochs)
        if self.ckpt_dir:
            self.attention_model = get_new_attention_model(
                self.ckpt_dir / f"Best_model_{self.exp_name}.pt"
            )
