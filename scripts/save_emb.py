from pathlib import Path
from datetime import datetime
import json
import numpy as np
from news_rec_utils.pipeline import Pipeline
from news_rec_utils.components import (
    TransformData,
    EmbeddingsComponent,
    SaveEmbeddingComponent,
    LoadEmbeddingComponent,
    # ClassificationComponent,
    # AttentionWeightComponent,
    # AttentionComponent,
)
from news_rec_utils.config import MODEL_PATH, NewsDataset
from news_rec_utils.data_utils import load_dataset
from news_rec_utils.evaluation import score


def main():
    data_dir = Path("data")
    # log_dir = Path("logs")
    save_dir = Path("embeddings")
    exp_name = "ranking_loss_attn"

    rng = np.random.default_rng(1234)
    train_behaviors, train_news_text_dict = load_dataset(
        data_dir, NewsDataset.MINDsmall_train, random_state=rng, num_samples=2000
    )
    val_behaviors, val_news_text_dict = load_dataset(
        data_dir, NewsDataset.MINDsmall_dev, random_state=rng, num_samples=2000
    )

    transform_component = TransformData()
    embedding_component = EmbeddingsComponent(MODEL_PATH)

    save_component = SaveEmbeddingComponent(save_dir)
    load_component = LoadEmbeddingComponent(save_dir)

    save_pipeline = Pipeline(
        "train_subset",
        [
            ("init_transform", transform_component),
            ("model_embed", embedding_component),
            ("save_embedding", save_component),
            # ("classification", classification_component),
            # ("attention", attention_component),
            # ("only_attention", attention_only_component),
        ],
    )

    load_pipeline = Pipeline(
        "train_subset",
        [("init_transform", transform_component), ("load_embedding", load_component)],
    )

    save_pipeline.transform(
        context_dict={
            "news_dataset": NewsDataset.MINDsmall_train,
            "behaviors": train_behaviors,
            "news_text_dict": train_news_text_dict,
        },
        val_context_dict={
            "news_dataset": NewsDataset.MINDsmall_dev,
            "behaviors": val_behaviors,
            "news_text_dict": val_news_text_dict,
        },
    )
    context_dict, val_context_dict = load_pipeline.transform(
        context_dict={
            "news_dataset": NewsDataset.MINDsmall_train,
            "behaviors": train_behaviors,
            "news_text_dict": train_news_text_dict,
        },
        val_context_dict={
            "news_dataset": NewsDataset.MINDsmall_dev,
            "behaviors": val_behaviors,
            "news_text_dict": val_news_text_dict,
        },
    )

    assert val_context_dict
    print(context_dict["news_embeddings"].shape)
    print(val_context_dict["news_embeddings"].shape)

    # train_scores = score(context_dict["grouped_scores"], context_dict["labels"])
    # val_scores = score(val_context_dict["grouped_scores"], val_context_dict["labels"])

    # with open(log_dir / "final_scores.jsonl", "a") as f:
    #     f.write(
    #         json.dumps(
    #             {
    #                 "timestamp": datetime.now().isoformat(),
    #                 "exp_name": exp_name,
    #                 "train_scores": train_scores,
    #                 "val_scores": val_scores,
    #             }
    #         )
    #         + "\n"
    #     )


if __name__ == "__main__":
    main()
