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
)
from news_rec_utils.config import MODEL_PATH, NewsDataset
from news_rec_utils.data_utils import load_dataset
from news_rec_utils.evaluation import score


def main():
    data_dir = Path("data")
    log_dir = Path("logs")
    ckpt_root_dir = Path("models")

    rng = np.random.default_rng(1234)
    train_behaviors, train_news_text_dict = load_dataset(
        data_dir, NewsDataset.MINDsmall_train, random_state=rng
    )
    val_behaviors, val_news_text_dict = load_dataset(
        data_dir, NewsDataset.MINDsmall_dev, random_state=rng
    )

    transform_component = TransformData()
    embedding_component = EmbeddingsComponent(MODEL_PATH)

    # val_pipeline = Pipeline(
    #     "validation_subset",
    #     [("init_transform", transform_component), ("model_embed", embedding_component)],
    #     use_cache=True,
    #     cache_dir=Path("cache"),
    # )
    # val_context_dict = val_pipeline.transform(
    #     {"behaviors": val_behaviors, "news_text_dict": val_news_text_dict}
    # )

    classification_component = ClassificationComponent(
        log_dir=log_dir,
        ckpt_dir=ckpt_root_dir / "classification_head",
        num_epochs=10,
        rng=rng,
    )
    attention_component = AttentionWeightComponent(
        log_dir=log_dir,
        ckpt_dir=ckpt_root_dir / "attention_model",
        weight_ckpt_dir=ckpt_root_dir / "weight_model",
        num_epochs=10,
        rng=rng,
    )

    train_pipeline = Pipeline(
        "train_small",
        [
            ("init_transform", transform_component),
            ("model_embed", embedding_component),
            ("classification", classification_component),
            ("attention", attention_component),
        ],
    )
    context_dict, val_context_dict = train_pipeline.train(
        context_dict={
            "behaviors": train_behaviors,
            "news_text_dict": train_news_text_dict,
        },
        val_context_dict={
            "behaviors": val_behaviors,
            "news_text_dict": val_news_text_dict,
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
                    "train_scores": train_scores,
                    "val_scores": val_scores,
                }
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
