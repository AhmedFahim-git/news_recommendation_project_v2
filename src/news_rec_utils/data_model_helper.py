from functools import partial
from typing import Iterable, Optional
import sqlite3
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
    TokenAttnEvalDataset,
    token_attention_eval_collate_fn,
)
from .config import (
    NEWS_TEXT_MAXLEN,
    NUM_WORKERS,
    DEVICE,
    QUERY_INSTRUCTION,
)
from .modeling_utils import (
    get_model_and_tokenizer,
    get_embed_from_model,
    get_model_eval,
    ClassificationHead,
    FinalAttention,
    WeightedSumModel,
    store_embed_from_model,
    get_token_attn_model,
    get_nvembed_model,
    get_nv_embeds,
)
from .batch_size_finder import (
    get_classification_inference_batch_size,
    get_attention_inference_batch_size,
    get_token_attention_inference_batch_size,
)


def get_embeddings(
    model_path: str, news_list: Iterable[str], news_text_dict: dict[str, str]
):
    if model_path == "nvidia/NV-Embed-v2":
        model = get_nvembed_model(model_path)
        text_list = [news_text_dict[i] for i in news_list]
        query_embeds = get_nv_embeds(model, text_list, "query").detach().cpu()
        passage_embeds = get_nv_embeds(model, text_list, "passage").detach().cpu()
        return query_embeds, passage_embeds
    model, tokenizer = get_model_and_tokenizer(model_path)
    text_collate_fn = partial(
        eval_collate_fn, tokenizer=tokenizer, max_len=NEWS_TEXT_MAXLEN
    )
    if model_path == "intfloat/multilingual-e5-large-instruct":
        query_news_text_dict = {
            k: QUERY_INSTRUCTION + v for k, v in news_text_dict.items()
        }
        query_news_text_dataset = NewsTextDataset(news_list, query_news_text_dict)
        passage_news_text_dataset = NewsTextDataset(news_list, news_text_dict)
        query_embeds = F.normalize(
            get_embed_from_model(
                model, query_news_text_dataset, NEWS_TEXT_MAXLEN, text_collate_fn
            ),
            p=2,
            dim=1,
        )
        passage_embeds = F.normalize(
            get_embed_from_model(
                model, passage_news_text_dataset, NEWS_TEXT_MAXLEN, text_collate_fn
            ),
            p=2,
            dim=1,
        )
        return query_embeds, passage_embeds
    news_text_dataset = NewsTextDataset(news_list, news_text_dict)
    return get_embed_from_model(
        model, news_text_dataset, NEWS_TEXT_MAXLEN, text_collate_fn
    )


def get_reduced_dim_embeds(embeddings: torch.Tensor, model: torch.nn.Module):
    return model(embeddings.to(DEVICE)).detach().cpu()


def get_classification_preds(
    news_embeddings: torch.Tensor, model: ClassificationHead
) -> np.ndarray:
    batch_size = 46336  # get_classification_inference_batch_size(model)
    print(f"Batch size for classification model inference {batch_size}")
    embedding_dataset = EmbeddingDataset(news_embeddings)
    dataloader = DataLoader(embedding_dataset, batch_size=batch_size, shuffle=False)
    return get_model_eval(dataloader, model).squeeze(dim=-1).numpy()


def get_classification_baseline_scores(
    news_embeddings: torch.Tensor, model: ClassificationHead, news_rev_index: np.ndarray
) -> dict[str, np.ndarray]:
    classification_preds = get_classification_preds(news_embeddings, model)
    print("Classification inference scores obtained")
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
    batch_size = get_attention_inference_batch_size(model) // 2
    # batch_size = 200
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
    # query_news_embeddings: Optional[torch.Tensor] = None,
):
    assert len(history_len_list) == len(
        impression_len_list
    ), "Number of rows should be consistent"
    assert sum(impression_len_list) == len(
        news_rev_index
    ), "Number of impressions should match length of impression list"
    # if isinstance(query_news_embeddings, torch.Tensor):
    #     history_embeds = get_final_attention_eval(
    #         history_rev_index, history_len_list, query_news_embeddings, model
    #     )
    # else:
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
    # query_news_embeddings: Optional[torch.Tensor] = None,
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
                # query_news_embeddings=query_news_embeddings,
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
    # query_news_embeddings: Optional[torch.Tensor] = None,
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
        # query_news_embeddings=query_news_embeddings,
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
    # query_news_embeddings: Optional[torch.Tensor] = None,
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
            # query_news_embeddings=query_news_embeddings,
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


def apply_token_attn(model_path, db_name, num_samples):
    model = get_token_attn_model(model_path=model_path)

    ds = TokenAttnEvalDataset(num_samples)
    conn = sqlite3.connect(db_name)

    collate_fn = partial(token_attention_eval_collate_fn, conn=conn)

    batch_size = get_token_attention_inference_batch_size(model) - 10
    print(batch_size)
    # batch_size = 50

    eval_dataloader = DataLoader(
        dataset=ds,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    res = get_model_eval(eval_dataloader, model)
    conn.close()

    return res


def get_final_second_attention_score(
    history_rev_index: np.ndarray,
    history_len_list: np.ndarray,
    news_rev_index: np.ndarray,
    impression_len_list: np.ndarray,
    news_embeddings: torch.Tensor,
    history_bool: pd.Series,
    attention_model: torch.nn.Module,
):
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

    scores = history_score
    grouped_scores = rank_group_preds(scores, impression_len_list)
    return {"scores": scores, "grouped_scores": grouped_scores}
