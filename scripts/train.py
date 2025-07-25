from pathlib import Path
from datetime import datetime
import json
import numpy as np
from news_rec_utils.pipeline import Pipeline
from news_rec_utils.components import (
    TransformData,
    EmbeddingsComponent,
    ClassificationComponent,
    AttentionWeightComponent,
    AttentionComponent,
    NewAttentionComponent,
)
from news_rec_utils.config import MODEL_PATH, NewsDataset
from news_rec_utils.data_utils import load_dataset
from news_rec_utils.evaluation import score


def main():
    data_dir = Path("data")
    log_dir = Path("logs")
    ckpt_root_dir = Path("models")
    exp_name = "ranking_loss_new_attn_individual_no_weight"

    rng = np.random.default_rng(1234)
    train_behaviors, train_news_feat_dict = load_dataset(
        data_dir,
        NewsDataset.MINDsmall_train,
        random_state=rng,
    )
    val_behaviors, val_news_feat_dict = load_dataset(
        data_dir,
        NewsDataset.MINDsmall_dev,
        random_state=rng,
    )

    transform_component = TransformData()
    embedding_component = EmbeddingsComponent(MODEL_PATH)

    classification_component = ClassificationComponent(
        # model_path=ckpt_root_dir / "classification_head" / "Best_model.pt",
        log_dir=log_dir,
        ckpt_dir=ckpt_root_dir / "classification_head",
        num_epochs=10,
        rng=rng,
        exp_name=exp_name,
    )
    # attention_component = AttentionWeightComponent(
    #     # attention_model_path=ckpt_root_dir / "attention_model" / "Best_model.pt",
    #     # weight_model_path=ckpt_root_dir / "weight_model" / "Best_model.pt",
    #     log_dir=log_dir,
    #     ckpt_dir=ckpt_root_dir / "attention_model",
    #     weight_ckpt_dir=ckpt_root_dir / "weight_model",
    #     num_epochs=10,
    #     exp_name=exp_name,
    #     rng=rng,
    # )
    # attention_only_component = AttentionComponent(
    #     log_dir=log_dir,
    #     ckpt_dir=ckpt_root_dir / "attention_model",
    #     num_epochs=10,
    #     exp_name=exp_name,
    #     rng=rng,
    # )

    # attention_only_component = AttentionComponent(
    #     log_dir=log_dir,
    #     ckpt_dir=ckpt_root_dir / "attention_model",
    #     num_epochs=10,
    #     exp_name=exp_name,
    #     # max_neg_ratio=1 / 5,
    #     rng=rng,
    # )

    new_attention = NewAttentionComponent(
        log_dir=log_dir,
        ckpt_dir=ckpt_root_dir / "new_attention_model",
        num_epochs=10,
        rng=rng,
    )

    train_pipeline = Pipeline(
        "train_small",
        [
            ("init_transform", transform_component),
            ("model_embed", embedding_component),
            ("classification", classification_component),
            # ("attention", attention_component),
            # ("only_attention", attention_only_component),
            ("new_attention", new_attention),
        ],
    )
    context_dict, val_context_dict = train_pipeline.train(
        context_dict={
            "behaviors": train_behaviors,
            **train_news_feat_dict,
        },
        val_context_dict={
            "behaviors": val_behaviors,
            **val_news_feat_dict,
        },
    )

    train_scores = score(context_dict["grouped_scores"], context_dict["labels"])
    assert val_context_dict
    val_scores = score(val_context_dict["grouped_scores"], val_context_dict["labels"])

    with open(log_dir / "final_scores.jsonl", "a") as f:
        f.write(
            json.dumps(
                {
                    "timestamp": datetime.now().isoformat(),
                    "exp_name": exp_name,
                    "train_scores": train_scores,
                    "val_scores": val_scores,
                }
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
