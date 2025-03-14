from abc import ABC, abstractmethod
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers.tokenization_utils_fast import (
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    BatchEncoding,
)
from collections.abc import Iterable
from .data_utils import split_impressions_pos_neg, group_items


class AbstractTextDataset(Dataset, ABC):
    def __init__(self, text_list: Iterable[str], news_text_dict: dict[str, str]):
        self.text_list = list(text_list)
        self.news_text_dict = news_text_dict

    def __len__(self):
        return len(self.text_list)

    @abstractmethod
    def __getitem__(self, idx):
        pass


def eval_collate_fn(
    input: Iterable[str],
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    max_len: int,
) -> BatchEncoding:
    return tokenizer(
        list(input),
        max_length=max_len,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )


class NewsTextDataset(AbstractTextDataset):
    def __getitem__(self, idx):
        return self.news_text_dict[self.text_list[idx]]


class EmbeddingDataset(Dataset):
    def __init__(self, embeds):
        self.embeds = embeds

    def __len__(self):
        return len(self.embeds)

    def __getitem__(self, idx):
        return self.embeds[idx]


class ClassificationTrainDataset(Dataset):
    def __init__(
        self,
        news_embeds: torch.Tensor,
        news_rev_index: np.ndarray,
        imp_counts: np.ndarray,
        labels: np.ndarray,
        rng: np.random.Generator,
    ):
        assert len(news_embeds) > 0, "We need the news embeddings for this dataset"
        self.news_embeds = news_embeds
        self.news_rev_index = news_rev_index
        self.imp_counts = imp_counts
        self.labels = labels
        self.rng = rng
        self.reset()

    def __len__(self):
        return len(self.pos_neg_indices)

    def __getitem__(self, idx):
        return (
            self.news_embeds[self.pos_neg_indices[idx, 0]],
            self.news_embeds[self.pos_neg_indices[idx, 1]],
        )

    def reset(self):
        pos_neg_indices = split_impressions_pos_neg(
            self.rng,
            grouped_news_rev_index=group_items(self.news_rev_index, self.imp_counts),
            labels=self.labels,
        )
        self.pos_neg_indices = pos_neg_indices[[0, 1]].T
