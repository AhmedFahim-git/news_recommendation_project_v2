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
    NewAttentionComponent,
    AttentionComponent,
    LoadEmbeddingComponent,
    NewAttentionReduceComponent,
    StoreEmbeddingsComponent,
    AttentionAttentionComponent,
)
from news_rec_utils.config import MODEL_PATH, NewsDataset, DataSubset
from news_rec_utils.data_utils import load_dataset
from news_rec_utils.evaluation import score


def main():
    # data_dir = Path("data")
    data_dir = Path("/content/drive/MyDrive/MIND_dataset")
    # log_dir = Path("logs")
    log_dir = Path("/content/drive/MyDrive/log_dir")
    # ckpt_root_dir = Path("models")
    ckpt_root_dir = Path("/content/drive/MyDrive/MIND_models_all")
    save_dir = Path("embeddings")
    exp_name = "attn_attn"
    # db_name = "mydb.sqlite"
    db_name = "/content/drive/MyDrive/my_db/mydb_dev.sqlite"

    rng = np.random.default_rng(1234)
    dev_behaviors, dev_news_text_dict = load_dataset(
        data_dir,
        NewsDataset.MINDsmall_dev,
        data_subset=DataSubset.WITH_HISTORY,
        random_state=rng,
        # num_samples=500,
    )
    # val_behaviors, val_news_text_dict = load_dataset(
    #     data_dir,
    #     NewsDataset.MINDsmall_dev,
    #     random_state=rng,
    # )

    transform_component = TransformData()
    store_emb_comp = StoreEmbeddingsComponent(MODEL_PATH, db_name=db_name)
    attn_attn_comp = AttentionAttentionComponent(
        db_name=db_name,
        log_dir=log_dir,
        token_ckpt_dir=ckpt_root_dir / "token_attn",
        final_attn_ckpt_dir=ckpt_root_dir / "final_attn",
        exp_name=exp_name,
        num_epochs=5,
        rng=rng,
    )
    # embedding_component = EmbeddingsComponent(MODEL_PATH)
    # load_component = LoadEmbeddingComponent(save_dir)

    # classification_component = ClassificationComponent(
    #     # model_path=ckpt_root_dir / "classification_head" / "Best_model.pt",
    #     log_dir=log_dir,
    #     ckpt_dir=ckpt_root_dir / "classification_head",
    #     num_epochs=10,
    #     rng=rng,
    #     exp_name=exp_name,
    # )
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

    # new_attention = NewAttentionComponent(
    #     log_dir=log_dir,
    #     ckpt_dir=ckpt_root_dir / "new_attention_model",
    #     num_epochs=5,
    #     rng=rng,
    # )
    # new_reduce_attention = NewAttentionReduceComponent(
    #     log_dir=log_dir,
    #     ckpt_dir=ckpt_root_dir / "new_attention_model",
    #     reduce_ckpt_dir=ckpt_root_dir / "reducing_model",
    #     num_epochs=2,
    #     rng=rng,
    # )

    train_pipeline = Pipeline(
        "train_subset",
        [
            ("init_transform", transform_component),
            ("store_comp", store_emb_comp),
            # ("attn_attn", attn_attn_comp),
            # ("load_embedding", load_component),
            # # ("model_embed", embedding_component),
            # ("classification", classification_component),
            # # ("attention", attention_component),
            # # ("only_attention", attention_only_component),
            # # ("new_attention", new_attention),
            # ("reduce_attenion", new_reduce_attention),
        ],
    )
    train_pipeline.train(
        context_dict={
            "news_dataset": NewsDataset.MINDsmall_dev,
            "behaviors": dev_behaviors,
            "news_text_dict": dev_news_text_dict,
        },
    )
    # context_dict, val_context_dict = train_pipeline.train(
    #     context_dict={
    #         "news_dataset": NewsDataset.MINDsmall_train,
    #         "behaviors": train_behaviors,
    #         "news_text_dict": train_news_text_dict,
    #     },
    # val_context_dict={
    #     "news_dataset": NewsDataset.MINDsmall_dev,
    #     "behaviors": val_behaviors,
    #     "news_text_dict": val_news_text_dict,
    # },
    # )

    # train_scores = score(context_dict["grouped_scores"], context_dict["labels"])
    # assert val_context_dict
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
