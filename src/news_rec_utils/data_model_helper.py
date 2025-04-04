from functools import partial
from typing import Iterable
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from .data_utils import (
    NewsTextDataset,
    eval_collate_fn,
    EmbeddingDataset,
    FinalAttentionEvalDataset,
    final_attention_eval_collate_fn,
    rank_group_preds,
    group_items,
)
from .config import (
    NEWS_TEXT_MAXLEN,
    NUM_WORKERS,
    DEVICE,
)
from .modeling_utils import (
    get_model_and_tokenizer,
    get_embed_from_model,
    get_model_eval,
    ClassificationHead,
    FinalAttention,
    WeightedSumModel,
    store_embed_from_model,
)
from .batch_size_finder import (
    get_classification_inference_batch_size,
    get_attention_inference_batch_size,
)


def get_embeddings(
    model_path: str, news_list: Iterable[str], news_text_dict: dict[str, str]
):
    news_text_dataset = NewsTextDataset(news_list, news_text_dict)
    model, tokenizer = get_model_and_tokenizer(model_path)
    text_collate_fn = partial(
        eval_collate_fn, tokenizer=tokenizer, max_len=NEWS_TEXT_MAXLEN
    )
    return get_embed_from_model(
        model, news_text_dataset, NEWS_TEXT_MAXLEN, text_collate_fn
    )


def get_reduced_dim_embeds(embeddings: torch.Tensor, model: torch.nn.Module):
    return model(embeddings.to(DEVICE)).detach().cpu()


def get_classification_preds(
    news_embeddings: torch.Tensor, model: ClassificationHead
) -> np.ndarray:
    batch_size = get_classification_inference_batch_size(model)
    print(f"Batch size for classification model inference {batch_size}")
    embedding_dataset = EmbeddingDataset(news_embeddings)
    dataloader = DataLoader(embedding_dataset, batch_size=batch_size, shuffle=False)
    return get_model_eval(dataloader, model).squeeze(dim=-1).numpy()


def get_classification_baseline_scores(
    news_embeddings: torch.Tensor, model: ClassificationHead, news_rev_index: np.ndarray
) -> dict[str, np.ndarray]:
    classification_preds = get_classification_preds(news_embeddings, model)
    return {
        "classification_preds": classification_preds,
        "baseline_scores": classification_preds[news_rev_index],
    }


def get_final_attention_eval(
    history_rev_index: np.ndarray,
    history_len_list: np.ndarray,
    news_embeddings: torch.Tensor,
    model: torch.nn.Module,
):
    attention_dataset = FinalAttentionEvalDataset(history_rev_index, history_len_list)
    batch_size = get_attention_inference_batch_size(model)
    print(f"Batch size for attention model inference {batch_size}")
    attention_dataloader = DataLoader(
        attention_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=partial(
            final_attention_eval_collate_fn, news_embeddings=news_embeddings
        ),
    )
    return get_model_eval(attention_dataloader, model)


def get_cos_sim_reduce_scores(
    history_rev_index: np.ndarray,
    history_len_list: np.ndarray,
    news_rev_index: np.ndarray,
    impression_len_list: np.ndarray,
    news_embeddings: torch.Tensor,
    attention_model: torch.nn.Module,
    reduce_model: torch.nn.Module,
):
    assert len(history_len_list) == len(
        impression_len_list
    ), "Number of rows should be consistent"
    assert sum(impression_len_list) == len(
        news_rev_index
    ), "Number of impressions should match length of impression list"
    news_embeddings = get_reduced_dim_embeds(news_embeddings, reduce_model)
    history_embeds = get_final_attention_eval(
        history_rev_index, history_len_list, news_embeddings, attention_model
    )
    # history_embeds = get_reduced_dim_embeds(history_embeds, reduce_model)
    grouped_rev_index = group_items(news_rev_index, impression_len_list)
    result_list = []
    for i, sub_list in enumerate(grouped_rev_index):
        result_list.append(
            F.cosine_similarity(
                news_embeddings[sub_list].to(DEVICE), history_embeds[i].to(DEVICE)
            )
        )
    return torch.cat(result_list)


def get_cos_sim_scores(
    history_rev_index: np.ndarray,
    history_len_list: np.ndarray,
    news_rev_index: np.ndarray,
    impression_len_list: np.ndarray,
    news_embeddings: torch.Tensor,
    model: torch.nn.Module,
):
    assert len(history_len_list) == len(
        impression_len_list
    ), "Number of rows should be consistent"
    assert sum(impression_len_list) == len(
        news_rev_index
    ), "Number of impressions should match length of impression list"
    history_embeds = get_final_attention_eval(
        history_rev_index, history_len_list, news_embeddings, model
    )
    grouped_rev_index = group_items(news_rev_index, impression_len_list)
    result_list = []
    for i, sub_list in enumerate(grouped_rev_index):
        result_list.append(
            F.cosine_similarity(
                news_embeddings[sub_list].to(DEVICE), history_embeds[i].to(DEVICE)
            )
        )
    return torch.cat(result_list)


def get_cos_sim_final_score(
    history_rev_index: np.ndarray,
    history_len_list: np.ndarray,
    news_rev_index: np.ndarray,
    impression_len_list: np.ndarray,
    news_embeddings: torch.Tensor,
    classification_score: np.ndarray,
    attention_model: torch.nn.Module,
    weight_model: WeightedSumModel,
) -> np.ndarray:
    return (
        weight_model(
            get_cos_sim_scores(
                history_rev_index,
                history_len_list,
                news_rev_index,
                impression_len_list,
                news_embeddings,
                attention_model,
            ),
            torch.tensor(classification_score[news_rev_index], device=DEVICE),
        )
        .detach()
        .cpu()
        .numpy()
    )


def get_final_score(
    history_rev_index: np.ndarray,
    history_len_list: np.ndarray,
    news_rev_index: np.ndarray,
    impression_len_list: np.ndarray,
    news_embeddings: torch.Tensor,
    classification_score: np.ndarray,
    history_bool: pd.Series,
    attention_model: torch.nn.Module,
    weight_model: WeightedSumModel,
):
    scores = classification_score[news_rev_index]
    imp_len_list = list(impression_len_list)

    history_score = get_cos_sim_final_score(
        history_rev_index,
        history_len_list,
        news_rev_index[history_bool.repeat(imp_len_list)],
        impression_len_list[history_bool],
        news_embeddings,
        classification_score,
        attention_model,
        weight_model,
    )

    scores[history_bool.repeat(imp_len_list)] = history_score
    grouped_scores = rank_group_preds(scores, impression_len_list)
    return {"scores": scores, "grouped_scores": grouped_scores}


def get_final_only_attention_score(
    history_rev_index: np.ndarray,
    history_len_list: np.ndarray,
    news_rev_index: np.ndarray,
    impression_len_list: np.ndarray,
    news_embeddings: torch.Tensor,
    classification_score: np.ndarray,
    history_bool: pd.Series,
    attention_model: torch.nn.Module,
):
    scores = classification_score[news_rev_index]
    imp_len_list = list(impression_len_list)

    history_score = (
        get_cos_sim_scores(
            history_rev_index,
            history_len_list,
            news_rev_index[history_bool.repeat(imp_len_list)],
            impression_len_list[history_bool],
            news_embeddings,
            attention_model,
        )
        .detach()
        .cpu()
        .numpy()
    )

    scores[history_bool.repeat(imp_len_list)] = history_score
    grouped_scores = rank_group_preds(scores, impression_len_list)
    return {"scores": scores, "grouped_scores": grouped_scores}


def get_final_only_reduce_attention_score(
    history_rev_index: np.ndarray,
    history_len_list: np.ndarray,
    news_rev_index: np.ndarray,
    impression_len_list: np.ndarray,
    news_embeddings: torch.Tensor,
    classification_score: np.ndarray,
    history_bool: pd.Series,
    attention_model: torch.nn.Module,
    reduce_model: torch.nn.Module,
):
    scores = classification_score[news_rev_index]
    imp_len_list = list(impression_len_list)

    history_score = (
        get_cos_sim_reduce_scores(
            history_rev_index,
            history_len_list,
            news_rev_index[history_bool.repeat(imp_len_list)],
            impression_len_list[history_bool],
            news_embeddings,
            attention_model=attention_model,
            reduce_model=reduce_model,
        )
        .detach()
        .cpu()
        .numpy()
    )

    scores[history_bool.repeat(imp_len_list)] = history_score
    grouped_scores = rank_group_preds(scores, impression_len_list)
    return {"scores": scores, "grouped_scores": grouped_scores}


def store_embeddings(
    model_path: str,
    news_list: Iterable[str],
    news_text_dict: dict[str, str],
    db_name: str,
):
    news_text_dataset = NewsTextDataset(news_list, news_text_dict)
    model, tokenizer = get_model_and_tokenizer(model_path)
    text_collate_fn = partial(
        eval_collate_fn, tokenizer=tokenizer, max_len=NEWS_TEXT_MAXLEN
    )
    store_embed_from_model(
        model, news_text_dataset, NEWS_TEXT_MAXLEN, text_collate_fn, db_name
    )
