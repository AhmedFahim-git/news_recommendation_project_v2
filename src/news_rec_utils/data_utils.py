from pathlib import Path
from typing import Optional
import argparse
from typing import Any, Callable
from collections.abc import Sequence, Iterable
from abc import ABC, abstractmethod
import sqlite3
from contextlib import closing
import io
import pandas as pd
import numpy as np
from scipy.stats import rankdata
from tqdm import tqdm, trange
import torch
from torch.utils.data import Dataset
from transformers.tokenization_utils_fast import (
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    BatchEncoding,
)
from .config import DataSubset, NewsDataset, IMPRESSION_MAXLEN, NEWS_TEXT_MAXLEN


def load_dataset(
    data_dir: Path,
    news_dataset: NewsDataset,
    num_samples: Optional[int] = None,
    data_subset: Optional[DataSubset] = DataSubset.ALL,
    random_state: int | np.random.Generator = 1234,
):
    behaviors = pd.read_parquet(
        data_dir / "processed" / news_dataset.value / "behaviors.parquet",
        columns=["ImpressionID", "History", "Impressions"],
    )
    news_text_dict: dict[str, str] = (
        pd.read_parquet(
            data_dir / "processed" / news_dataset.value / "news_text.parquet"
        )
        .set_index("NewsID")["news_text"]
        .to_dict()
    )
    if data_subset == DataSubset.WITH_HISTORY:
        behaviors = behaviors[behaviors["History"].notna()].reset_index(drop=True)
    elif data_subset == DataSubset.WITHOUT_HISTORY:
        behaviors = behaviors[behaviors["History"].isna()].reset_index(drop=True)
    if num_samples and num_samples < len(behaviors):
        behaviors = behaviors.sample(
            n=num_samples, random_state=random_state, replace=False
        ).reset_index(drop=True)
        # behaviors = behaviors.iloc[:num_samples]

    return behaviors, news_text_dict


def read_data(
    data_dir: Path,
    news_dataset: NewsDataset,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Reading behaviors.tsv data")
    behaviors = pd.read_csv(
        data_dir / "raw" / news_dataset.value / "behaviors.tsv",
        sep="\t",
        header=None,
        names=["ImpressionID", "UserID", "Time", "History", "Impressions"],
        parse_dates=["Time"],
    )

    print("Reading news.tsv data")
    news = pd.read_csv(
        data_dir / "raw" / news_dataset.value / "news.tsv",
        sep="\t",
        header=None,
        names=[
            "NewsID",
            "Category",
            "SubCategory",
            "Title",
            "Abstract",
            "URL",
            "Title Entities",
            "Abstract Entities",
        ],
    )

    return behaviors, news


def split_impressions_and_history(
    impressions: Sequence[str], history: Sequence[str]
) -> dict[str, Any]:
    assert len(impressions) > 0, "No Impressions given"
    label_present = "-" in impressions[0]
    cur = 0
    position_dict = dict()
    news_list = []
    imp_rev_ind = []
    hist_rev_ind = []
    labels = []
    hist_len_list = []
    imp_len_list = []
    for i in trange(len(impressions), desc="Splitting impressions and history"):
        imp_row = impressions[i]
        hist_row = history[i]
        if hist_row:
            hist_sub_list = hist_row.split()
            hist_len_list.append(len(hist_sub_list))
            for news in hist_sub_list:
                if news not in position_dict:
                    position_dict[news] = cur
                    cur += 1
                    news_list.append(news)
                hist_rev_ind.append(position_dict[news])

        if label_present:
            news_sub_list, label = zip(
                *map(
                    lambda x: (x[0], int(x[1])), [k.split("-") for k in imp_row.split()]
                )
            )
            labels.append(label)
        else:
            news_sub_list = imp_row.split()
        imp_len_list.append(len(news_sub_list))
        for news in news_sub_list:
            if news not in position_dict:
                position_dict[news] = cur
                cur += 1
                news_list.append(news)

            imp_rev_ind.append(position_dict[news])
    return {
        "news_list": np.array(news_list),
        "impression_rev_ind_array": np.stack(
            [
                np.array(imp_rev_ind, dtype=np.int32),
                np.concatenate(
                    [[i] * n for i, n in enumerate(imp_len_list)], dtype=np.int32
                ),
            ]
        ),
        "impression_len_list": np.array(imp_len_list, dtype=np.int32),
        "history_rev_ind_array": np.stack(
            [
                np.array(hist_rev_ind, dtype=np.int32),
                np.concatenate(
                    [[i] * n for i, n in enumerate(hist_len_list)], dtype=np.int32
                ),
            ]
        ),
        "history_len_list": np.array(hist_len_list, dtype=np.int32),
        "labels": np.array(labels, dtype=object),
    }


def split_impressions(impressions: Sequence[str]):
    assert len(impressions) > 0, "No Impressions given"
    label_present = "-" in impressions[0]
    cur = 0
    pos_dict = dict()
    news_list = []
    rev_ind = []
    labels = []
    len_list = []
    for row in tqdm(impressions, desc="Splitting impressions"):
        if label_present:
            news_sub_list, label = zip(
                *map(lambda x: (x[0], int(x[1])), [k.split("-") for k in row.split()])
            )
            labels.append(label)
        else:
            news_sub_list = row.split()
        len_list.append(len(news_sub_list))
        for news in news_sub_list:
            if news not in pos_dict:
                pos_dict[news] = cur
                cur += 1
                news_list.append(news)

            rev_ind.append(pos_dict[news])
    return (
        np.array(news_list),
        np.stack(
            [
                np.array(rev_ind, dtype=np.int32),
                np.concatenate(
                    [[i] * n for i, n in enumerate(len_list)], dtype=np.int32
                ),
            ]
        ),
        np.array(len_list, dtype=np.int32),
        np.array(labels, dtype=object),
    )


def split_impressions_pos_neg(
    rng: np.random.Generator,
    grouped_news_rev_index: np.ndarray,
    labels: np.ndarray,
    max_neg_ratio: Optional[float] = None,
    max_pos_ratio: Optional[float] = None,
):
    pos_ind, neg_ind, len_list = [], [], []
    for i, row in enumerate(labels):
        temp_pos, temp_neg = [], []
        num_pos = sum(row)
        num_neg = len(row) - num_pos
        max_len = max(num_pos, num_neg)
        if max_neg_ratio or max_pos_ratio:
            if max_neg_ratio and (num_neg * max_neg_ratio > num_pos):
                max_len = int(num_pos / max_neg_ratio)
            elif max_pos_ratio and (num_pos * max_pos_ratio > num_neg):
                max_len = int(num_neg / max_pos_ratio)
        for j, label in enumerate(row):
            news_rev_ind = grouped_news_rev_index[i][j]
            if label == 0:
                temp_neg.append(news_rev_ind)
            else:
                temp_pos.append(news_rev_ind)
        if num_neg >= max_len:
            temp_neg = rng.choice(temp_neg, size=max_len, replace=False)
            temp_pos = rng.permutation(
                np.append(temp_pos, rng.choice(temp_pos, max_len - num_pos))
            )
        else:
            temp_pos = rng.choice(temp_pos, size=max_len, replace=False)
            temp_neg = rng.permutation(
                np.append(temp_neg, rng.choice(temp_neg, max_len - num_neg))
            )
        # else:
        #     temp_pos = rng.permutation(
        #         np.append(temp_pos, rng.choice(temp_pos, max_len - num_pos))
        #     )
        #     temp_neg = rng.permutation(
        #         np.append(temp_neg, rng.choice(temp_neg, max_len - num_neg))
        #     )

        pos_ind.extend(temp_pos.tolist())
        neg_ind.extend(temp_neg.tolist())
        len_list.append(max_len)
    return np.stack(
        [
            np.array(pos_ind, dtype=np.int32),
            np.array(neg_ind, dtype=np.int32),
            np.concatenate([[i] * n for i, n in enumerate(len_list)], dtype=np.int32),
        ]
    )


def expand_items(items: np.ndarray, rev_index: np.ndarray, imp_counts: np.ndarray):
    result_list = []

    cumsum_lengths = np.concatenate([[0], imp_counts.cumsum()])
    for i in range(len(imp_counts)):
        result_list.append(items[rev_index[cumsum_lengths[i] : cumsum_lengths[i + 1]]])
    return np.concatenate(result_list)


def group_items(
    items: np.ndarray,
    imp_counts: np.ndarray,
    func: Callable[[np.ndarray], np.ndarray] = lambda x: x,
):
    result_list = []

    cumsum_lengths = np.concatenate([[0], imp_counts.cumsum()])
    for i in range(len(imp_counts)):
        result_list.append(func(items[cumsum_lengths[i] : cumsum_lengths[i + 1]]))
    return np.array(result_list, dtype=object)


def rank_group_preds(pred_scores: np.ndarray, imp_counts: np.ndarray):
    return group_items(pred_scores, imp_counts, lambda x: rankdata(-x, method="dense"))


def get_data(
    data_dir: Path,
    news_dataset: NewsDataset,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    behaviors, news = read_data(data_dir, news_dataset)

    print("Getting list of target impressions for final filtering of news dataset")

    news = process_news(news)
    return behaviors, news


def process_news(news_df: pd.DataFrame) -> pd.DataFrame:
    print("Making news_text column for the news data")
    news_df["news_text"] = news_df.apply(
        lambda x: f"Title: {x['Title']}\nCategory: {x['Category']}\nSubCategory: {x['SubCategory']}\nAbstract: {x['Abstract']}",
        axis=1,
    )
    return news_df


def store_processed_data(data_dir: Path, news_dataset: NewsDataset) -> None:
    behaviors, news_text = get_data(data_dir, news_dataset)

    print("Saving datasets")
    (data_dir / "processed" / news_dataset.value).mkdir(parents=True, exist_ok=True)
    behaviors.to_parquet(
        data_dir / "processed" / news_dataset.value / "behaviors.parquet"
    )
    news_text.to_parquet(
        data_dir / "processed" / news_dataset.value / "news_text.parquet"
    )


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


class FinalAttentionEvalDataset(Dataset):
    def __init__(self, history_rev_index: np.ndarray, history_len_list: np.ndarray):
        self.group_history = group_items(history_rev_index, history_len_list)

    def __len__(self):
        return len(self.group_history)

    def __getitem__(self, idx):
        return self.group_history[idx]


class FinalAttentionTrainDataset(Dataset):
    def __init__(
        self,
        history_rev_index: np.ndarray,
        history_len_list: np.ndarray,
        news_rev_index: np.ndarray,
        impression_len_list: np.ndarray,
        labels: np.ndarray,
        batch_size: int,
        max_neg_raio: Optional[float] = None,
        max_pos_ratio: Optional[float] = None,
        history_maxlen: Optional[int] = None,
        rng=np.random.default_rng(1234),
    ):
        assert len(history_len_list) == len(
            impression_len_list
        ), "Number of rows should match between history and news"
        assert sum(impression_len_list) == len(
            news_rev_index
        ), "Number of impressions should match length of impression list"
        self.group_history = group_items(history_rev_index, history_len_list)
        self.batch_size = batch_size
        self.labels = labels
        self.news_rev_index = news_rev_index
        self.impression_len_list = impression_len_list
        self.rng = rng
        self.max_neg_ratio = max_neg_raio
        self.max_pos_ratio = max_pos_ratio
        self.history_maxlen = history_maxlen
        self.reset()

    def __len__(self):
        return len(self.pos_neg_indices)

    def __getitem__(self, idx):
        return (
            (
                self.group_history[self.pos_neg_indices[idx, 2]]
                if self.history_maxlen
                and (
                    len(self.group_history[self.pos_neg_indices[idx, 2]])
                    <= self.history_maxlen
                )
                else self.rng.choice(
                    self.group_history[self.pos_neg_indices[idx, 2]],
                    self.history_maxlen,
                    replace=False,
                    shuffle=False,
                )
            ),
            self.pos_neg_indices[idx, 0],
            self.pos_neg_indices[idx, 1],
        )

    def reset(self):
        permuted_index = self.rng.permutation(len(self.labels))
        pos_neg_indices = split_impressions_pos_neg(
            self.rng,
            grouped_news_rev_index=group_items(
                self.news_rev_index, self.impression_len_list
            )[permuted_index],
            labels=self.labels[permuted_index],
            max_neg_ratio=self.max_neg_ratio,
            max_pos_ratio=self.max_pos_ratio,
        )
        pos_neg_indices[2] = permuted_index[pos_neg_indices[2]]
        num_batches = -(pos_neg_indices.shape[1] // -self.batch_size)
        permuted_list = self.rng.permutation(num_batches - 1).tolist() + [
            num_batches - 1
        ]
        final_index = np.concatenate(
            [
                np.arange(i * self.batch_size, (i + 1) * self.batch_size)
                for i in permuted_list
            ]
        )[: pos_neg_indices.shape[1]]
        pos_neg_indices = pos_neg_indices[:, final_index]
        self.pos_neg_indices = pos_neg_indices.T


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


def pad_to_maxlen(
    grouped_items: np.ndarray | Sequence[np.ndarray] | Sequence[Sequence[int]],
) -> dict[str, np.ndarray]:
    len_list = list(map(len, grouped_items))
    max_len = max(len_list)
    indices = []
    attention_mask = []
    for i, length in enumerate(len_list):
        indices.append(
            np.pad(
                grouped_items[i] if max_len > length else grouped_items[i][:max_len],
                pad_width=(0, max_len - min(max_len, length)),
                mode="constant",
                constant_values=0,
            )
        )
        attention_mask.append(
            np.pad(
                np.ones(length, dtype=np.int32),
                pad_width=(0, max_len - length),
                mode="constant",
                constant_values=0,
            )
        )
    return {
        "indices": np.stack(indices, dtype=np.int32),
        "attention_mask": np.stack(attention_mask, dtype=np.int32),
    }


def tensor_pad_to_maxlen(
    grouped_items: Sequence[torch.Tensor],
) -> dict[str, torch.Tensor]:
    len_list = list(map(len, grouped_items))
    max_len = max(len_list)
    embeddings = []
    attention_mask = []
    for i, length in enumerate(len_list):
        embeddings.append(
            torch.nn.functional.pad(
                grouped_items[i],
                pad=(0, 0, 0, max_len - length),
                mode="constant",
                value=0,
            )
        )
        attention_mask.append(
            torch.nn.functional.pad(
                torch.ones(length, dtype=torch.int32),
                pad=(0, max_len - length),
                mode="constant",
                value=0,
            )
        )
    # final_embeddings = torch.stack(embeddings)
    return {
        "embeddings": torch.stack(embeddings),
        "attention_mask": torch.stack(attention_mask),
    }


def final_attention_eval_collate_fn(input, news_embeddings: torch.Tensor):
    padded_mask_history = pad_to_maxlen(input)
    indices = torch.tensor(padded_mask_history["indices"])
    attention_mask = torch.tensor(padded_mask_history["attention_mask"])
    return (
        news_embeddings[indices] * attention_mask.unsqueeze(-1),
        attention_mask,
    )


def final_attention_train_collate_fn(input):
    grouped_history, news_ind_pos, news_ind_neg = zip(*input)
    # print(type(grouped_history), type(grouped_history[0]))
    grouped_history_unique, grouped_history_rev_index = np.unique(
        [",".join(x.astype(str)) for x in grouped_history], return_inverse=True
    )
    grouped_history_unique = [
        [int(i) for i in x.split(",")] for x in grouped_history_unique
    ]
    # grouped_history_unique, grouped_history_rev_index = np.unique(
    #     np.array(list(map(list, grouped_history)), dtype=object),
    #     return_inverse=True,
    # )
    # if not hasattr(grouped_history_unique[0], "__len__"):
    #     grouped_history_unique = np.expand_dims(
    #         np.array(grouped_history_unique, dtype=np.int32), 0
    #     )
    padded_mask_history = pad_to_maxlen(grouped_history_unique)
    return (
        torch.tensor(padded_mask_history["indices"], dtype=torch.int32),
        torch.tensor(padded_mask_history["attention_mask"], dtype=torch.int32),
        torch.tensor(grouped_history_rev_index, dtype=torch.int32),
        torch.tensor(np.concatenate((news_ind_pos, news_ind_neg)), dtype=torch.int32),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Process news dataset and store the results."
    )

    # Argument for the data directory
    parser.add_argument(
        "data_dir",
        type=Path,
        help="Path to the directory containing data",
    )

    # Argument for selecting a dataset
    parser.add_argument(
        "news_dataset",
        choices=NewsDataset._member_names_,
        help="Select the news dataset",
    )

    args = parser.parse_args()

    # Ensure the data_dir is a valid directory
    if not args.data_dir.is_dir():
        parser.error(f"The path '{args.data_dir}' is not a valid directory.")

    # Convert dataset name to Enum
    dataset_enum = NewsDataset[args.news_dataset]

    # Call the processing function
    store_processed_data(args.data_dir, dataset_enum)


def get_embeds_from_db(conn, indices: Iterable):
    indices = ",".join([str(i + 1) for i in indices])

    # with closing(sqlite3.connect(db_name)) as conn:
    res = conn.execute(f"SELECT data from tensors where id in ({indices});").fetchall()
    tensors = []
    for i in res:
        f = io.BytesIO(i[0])
        tensors.append(torch.load(f, weights_only=True))
        f.close()
    # tensors = [torch.load(io.BytesIO(i[0]), weights_only=True) for i in res]
    final_dict = tensor_pad_to_maxlen(tensors)
    return final_dict


def attention_attention_train_collate_fn(input, conn):
    grouped_history, news_ind_pos, news_ind_neg = zip(*input)
    len_list = [len(i) for i in grouped_history]

    sum_list = list(np.cumsum(len_list))
    sum_list = sum_list + [sum_list[-1] + len(news_ind_pos)]

    all_indices = np.concatenate(
        list(grouped_history) + [news_ind_pos] + [news_ind_neg]
    )
    unique_indices, reverse_indices = np.unique(all_indices, return_inverse=True)
    final_dict = get_embeds_from_db(conn, unique_indices)

    rev_split = np.split(reverse_indices, sum_list)

    padded_history = pad_to_maxlen(rev_split[:-2])
    return (
        final_dict["embeddings"].to(dtype=torch.float32)[:, :NEWS_TEXT_MAXLEN],
        final_dict["attention_mask"].to(dtype=torch.int32)[:, :NEWS_TEXT_MAXLEN],
        torch.tensor(padded_history["indices"], dtype=torch.int32),
        torch.tensor(padded_history["attention_mask"], dtype=torch.int32),
        torch.tensor(np.concatenate(rev_split[-2:]), dtype=torch.int32),
    )


class TokenAttnEvalDataset(Dataset):
    def __init__(self, num_items: int):
        self.num_items = num_items

    def __len__(self):
        return self.num_items

    def __getitem__(self, idx):
        return idx


def token_attention_eval_collate_fn(input, conn):
    res_dict = get_embeds_from_db(conn, input)
    return res_dict["embeddings"].to(dtype=torch.float32), res_dict[
        "attention_mask"
    ].to(dtype=torch.int32)
