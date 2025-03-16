from typing import Optional, Callable
from pathlib import Path
from collections.abc import Iterable
from contextlib import nullcontext
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers.modeling_outputs import (
    BaseModelOutputWithPooling,
    BaseModelOutputWithPast,
)
from transformers.tokenization_utils import BatchEncoding
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from tqdm import tqdm
from .config import (
    DEVICE,
    EMBEDDING_DIM,
    NUM_WORKERS,
    TORCH_DTYPE,
)
from .batch_size_finder import get_text_inference_batch_size


@torch.compile
def last_token_pool(
    last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
        ]


def output_pool(model) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    def get_last_embedding(
        last_hidden_states: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        return last_hidden_states[:, 0]

    if model.config.architectures[0] == "Qwen2ForCausalLM":
        return last_token_pool
    elif model.config.architectures[0] == "NewModel":
        return get_last_embedding
    else:
        return lambda x, y: x
    # Alternate implementation
    # with torch.no_grad():
    #     output = model(**dummy_text_inputs(1, 1, device=model.device)).to("cpu")
    # if isinstance(output, BaseModelOutputWithPast):
    #     return last_token_pool
    # elif isinstance(output, BaseModelOutputWithPooling):
    #     return get_last_embedding


def get_model_and_tokenizer(path: str, device=DEVICE):
    model: PreTrainedModel = AutoModel.from_pretrained(
        path,
        trust_remote_code=True,
        # unpad_inputs=True,
        # use_memory_efficient_attention=True,
        # torch_dtype=torch.float16,
    ).to(device)
    assert isinstance(model, PreTrainedModel), "Model is of different type"

    tokenizer = AutoTokenizer.from_pretrained(path)
    return model, tokenizer


class ClassificationHead(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.linear_1 = torch.nn.Linear(in_features=in_dim, out_features=hidden_dim)
        self.linear_2 = torch.nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.linear_3 = torch.nn.Linear(in_features=hidden_dim, out_features=out_dim)

    def forward(self, embeddings):
        embeddings = F.relu(self.linear_1(embeddings))
        embeddings = F.relu(self.linear_2(embeddings))
        return F.tanh(self.linear_3(embeddings))


def get_classification_head(model_path: Optional[Path] = None):
    model = ClassificationHead(
        in_dim=EMBEDDING_DIM, hidden_dim=EMBEDDING_DIM, out_dim=1
    )
    if model_path:
        model.load_state_dict(torch.load(model_path, weights_only=True))
    return model.to(DEVICE)


class WeightedSumModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, cos_sim, baseline):
        alpha = torch.sigmoid(self.alpha)
        return cos_sim * alpha + baseline * (1 - alpha)


def get_weighted_sum_model(model_path: Optional[Path] = None):
    model = WeightedSumModel()
    if model_path:
        model.load_state_dict(torch.load(model_path, weights_only=True))
    return model.to(DEVICE)


class FinalAttention(torch.nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.linear1 = torch.nn.Linear(embed_dim, embed_dim)
        self.dropout1 = torch.nn.Dropout(0.1)
        self.linear2 = torch.nn.Linear(embed_dim, embed_dim)
        self.dropout2 = torch.nn.Dropout(0.1)
        self.linear3 = torch.nn.Linear(embed_dim, embed_dim)
        self.dropout3 = torch.nn.Dropout(0.1)
        self.linear4 = torch.nn.Linear(embed_dim, hidden_dim)
        self.linear5 = torch.nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, embeddings: torch.Tensor, attention_mask: torch.Tensor):
        x = self.dropout1(F.relu(self.linear1(embeddings)))
        x = self.dropout2(F.relu(self.linear2(embeddings)))
        x = self.dropout3(F.relu(self.linear3(embeddings)))
        weights = self.linear4(x)
        weights = self.linear5(weights)
        weights = weights.squeeze(2)
        weights = torch.exp(weights) * attention_mask
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-10)
        weights = weights.unsqueeze(-1)
        return (x * weights).sum(dim=1)


# class FinalAttention(torch.nn.Module):
#     def __init__(self, embed_dim: int, hidden_dim: int):
#         super().__init__()
#         self.linear1 = torch.nn.Linear(embed_dim, embed_dim)
#         self.linear2 = torch.nn.Linear(embed_dim, hidden_dim)
#         self.linear3 = torch.nn.Linear(hidden_dim, 1, bias=False)

#     def forward(self, embeddings: torch.Tensor, attention_mask: torch.Tensor):
#         x = F.relu(self.linear1(embeddings))
#         weights = self.linear2(x)
#         weights = self.linear3(weights)
#         weights = weights.squeeze(2)
#         weights = torch.exp(weights) * attention_mask
#         weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-10)
#         weights = weights.unsqueeze(-1)
#         return (x * weights).sum(dim=1)


def get_final_attention_model(model_path: Optional[Path] = None):
    model = FinalAttention(embed_dim=EMBEDDING_DIM, hidden_dim=EMBEDDING_DIM)
    if model_path:
        model.load_state_dict(torch.load(model_path, weights_only=True))
    return model.to(DEVICE)


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


def get_model_eval(
    dataloader: DataLoader,
    model: torch.nn.Module,
) -> torch.Tensor:

    result_list = []
    model.eval()
    with torch.no_grad():
        for item in tqdm(dataloader, desc="Model Inference"):
            if isinstance(item, tuple):
                result_list.append(
                    model(*map(lambda x: x.to(DEVICE), item)).detach().cpu()
                )
            else:
                result_list.append(model(item.to(DEVICE)).detach().cpu())
    return torch.cat(result_list)
