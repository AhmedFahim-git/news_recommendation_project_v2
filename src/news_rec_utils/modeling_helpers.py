from contextlib import nullcontext
from tqdm import tqdm
from functools import partial
from typing import Type, Callable
from collections.abc import Iterable
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_fast import (
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    BatchEncoding,
)
from .modeling_utils import output_pool, ClassificationHead
from .config import TORCH_DTYPE, DEVICE, NUM_WORKERS, CLASSIFICATION_MODEL_BATCH_SIZE
from .batch_size_finder import get_text_inference_batch_size


def get_text_embed_eval(model: PreTrainedModel, input_dataloader: DataLoader):
    pool_fn = output_pool(model)
    text_embed_list = []
    cast_context = (
        torch.autocast(device_type="cuda", dtype=TORCH_DTYPE)
        if DEVICE.type == "cuda"
        else nullcontext()
    )
    with torch.no_grad(), cast_context:
        for inputs in tqdm(input_dataloader, desc="Embedding Text"):
            text_embed_list.append(
                pool_fn(
                    model(**inputs.to(DEVICE)).last_hidden_state,
                    inputs["attention_mask"],
                )
                .detach()
                .cpu()
            )
    return torch.concatenate(text_embed_list)


# See if we can use protocon for collate_fn type hint
def get_embed_from_model(
    model: PreTrainedModel,
    text_dataset: Dataset,
    text_maxlen: int,
    text_collate_fn: Callable[[Iterable[str]], BatchEncoding],
):
    text_batch_size = get_text_inference_batch_size(model, text_maxlen)
    print(f"Batch size for text of {text_maxlen}: {text_batch_size}")
    text_dataloader = DataLoader(
        text_dataset,
        batch_size=text_batch_size,
        collate_fn=text_collate_fn,
        shuffle=False,
        pin_memory=True,
        num_workers=NUM_WORKERS,
    )
    model.eval()
    return get_text_embed_eval(model, text_dataloader)


def get_classification_model_eval(
    embed_dataset: Dataset,
    model: ClassificationHead,
):
    embed_dataloader = DataLoader(
        embed_dataset, batch_size=CLASSIFICATION_MODEL_BATCH_SIZE, shuffle=False
    )

    result_list = []
    with torch.no_grad():
        for embed in embed_dataloader:
            result_list.append(model(embed.to(DEVICE)).detach().cpu().squeeze().numpy())
    return np.concatenate(result_list)
