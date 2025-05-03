from typing import Optional
from pathlib import Path
import json
from concurrent.futures import ProcessPoolExecutor
from collections.abc import Sequence
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from .config import NUM_WORKERS


# Scoring functions adapted from https://github.com/msnews/MIND/blob/master/evaluate.py
def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2**y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def score_row(label_sub_rank):
    labels, sub_ranks, ind = label_sub_rank
    lt_len = float(len(labels))

    y_true = np.array(labels, dtype="float32")
    y_score = []

    for rank in sub_ranks:
        score_rslt = 1.0 / rank
        if score_rslt < 0 or score_rslt > 1:
            raise ValueError(
                "Line-{}: score_rslt should be int from 0 to {}".format(ind, lt_len)
            )
        y_score.append(score_rslt)

    auc = roc_auc_score(y_true, y_score)
    mrr = mrr_score(y_true, y_score)
    ndcg5 = ndcg_score(y_true, y_score, 5)
    ndcg10 = ndcg_score(y_true, y_score, 10)

    return auc, mrr, ndcg5, ndcg10


def score(
    preds_input: Sequence[Sequence[int]] | np.ndarray,
    labels_input: Sequence[Sequence[int]] | np.ndarray,
    imp_ids: Sequence[str] = [],
    debug_dir: Optional[Path] = None,
) -> dict[str, float]:
    pool_submit = zip(labels_input, preds_input, range(len(preds_input)))
    print("starting scoring")

    with ProcessPoolExecutor(NUM_WORKERS) as executor:
        aucs, mrrs, ndcg5s, ndcg10s = zip(
            *tqdm(executor.map(score_row, pool_submit), total=len(preds_input))
        )

    if debug_dir and (len(imp_ids) > 0):
        try:
            assert len(imp_ids) == len(
                preds_input
            ), "Number of impression ids should be same as the number of preds"
            debug_dir.mkdir(parents=True, exist_ok=True)
            with open(debug_dir / "debug_json.json") as f:
                json.dump(
                    {
                        "ImpressionID": list(imp_ids),
                        "auc": aucs,
                        "mrr": mrrs,
                        "ndcg5": ndcg5s,
                        "ndcg10": ndcg10s,
                        "preds": preds_input,
                        "labels": labels_input,
                    },
                    f,
                )
        except:
            print("Debug not possible due to error")
    return {
        "auc": np.mean(aucs).item(),
        "mrr": np.mean(mrrs).item(),
        "ndcg5": np.mean(ndcg5s).item(),
        "ndcg10": np.mean(ndcg10s).item(),
        "num_samples": len(preds_input),
    }
