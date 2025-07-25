from typing import Optional, Callable
from pathlib import Path
import os
from collections.abc import Iterable
from contextlib import nullcontext
import gc
import sqlite3
import io
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers.modeling_outputs import (
    BaseModelOutputWithPooling,
    BaseModelOutputWithPast,
)
from transformers.tokenization_utils import BatchEncoding
from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers.modeling_utils import PreTrainedModel
from tqdm import tqdm, trange
from .config import (
    DEVICE,
    EMBEDDING_DIM,
    NUM_WORKERS,
    NEWS_TEXT_MAXLEN,
    IMPRESSION_MAXLEN,
    QUERY_INSTRUCTION,
    TORCH_DTYPE,
    REDUCED_DIM,
    NUM_HIDDEN_LAYERS,
)
from .attention import NewAttention, MyEncoder
from .batch_size_finder import get_text_inference_batch_size, get_nv_embed_batch_size
from .latent_attention import LatentAttentionModel


# @torch.compile
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


def first_token_pool(last_hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    return last_hidden_states[:, 0]


def average_pool(
    last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def output_pool(model) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    def get_last_embedding(
        last_hidden_states: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        return last_hidden_states[:, 0]

    if model.config.architectures[0] == "Qwen2ForCausalLM":
        return last_token_pool
    elif model.config.architectures[0] == "NewModel":
        return first_token_pool
    elif model.config.architectures[0] == "XLMRobertaModel":
        return average_pool
    else:
        return first_token_pool
    # Alternate implementation
    # with torch.no_grad():
    #     output = model(**dummy_text_inputs(1, 1, device=model.device)).to("cpu")
    # if isinstance(output, BaseModelOutputWithPast):
    #     return last_token_pool
    # elif isinstance(output, BaseModelOutputWithPooling):
    #     return get_last_embedding


def get_nvembed_model(path: str, device=DEVICE):
    assert path == "nvidia/NV-Embed-v2"
    return AutoModel.from_pretrained("nvidia/NV-Embed-v2", trust_remote_code=True).to(
        device="cuda"
    )


def get_model_and_tokenizer(path: str, device=DEVICE):
    model: PreTrainedModel = AutoModel.from_pretrained(
        path,
        trust_remote_code=True,
        # unpad_inputs=True,
        # use_memory_efficient_attention=True,
        torch_dtype=torch.float16,
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
        return self.linear_3(embeddings)


class ClassificationHeadCatEmbed(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.cat_embed = torch.nn.Embedding(15, 128)
        self.linear_1 = torch.nn.Linear(in_features=in_dim, out_features=hidden_dim)
        self.linear_2 = torch.nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.linear_3 = torch.nn.Linear(in_features=hidden_dim, out_features=out_dim)

    def forward(self, embeddings):
        embeddings = self.add_embedding(embeddings)
        embeddings = F.relu(self.linear_1(embeddings))
        embeddings = F.relu(self.linear_2(embeddings))
        return self.linear_3(embeddings)

    def add_embedding(self, embeddings):
        cat_embeds = self.cat_embed(embeddings[..., -1].to(dtype=torch.int32))
        # subcat_embeds = self.subcat_embed(embeddings[..., -1].to(dtype=torch.int32))
        return torch.cat([embeddings[..., :-1], cat_embeds], dim=-1)


def get_classification_head(model_path: Optional[Path] = None):
    model = ClassificationHead(
        in_dim=EMBEDDING_DIM, hidden_dim=EMBEDDING_DIM, out_dim=1
    )
    # model = ClassificationHeadCatEmbed(
    #     in_dim=EMBEDDING_DIM, hidden_dim=EMBEDDING_DIM, out_dim=1
    # )
    if model_path:
        model.load_state_dict(torch.load(model_path, weights_only=True))
    return model.to(DEVICE)


def get_latent_attention_model(model_path: Optional[Path] = None):
    model = LatentAttentionModel()
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
    def __init__(self, reduced_dim: int, hidden_dim: int):
        super().__init__()

        # self.alpha = torch.nn.Parameter(torch.tensor(99999.9))

        # self.pos_emb_layer = torch.nn.Embedding(IMPRESSION_MAXLEN, 100)
        # self.in_layer = torch.nn.Linear(EMBEDDING_DIM + 100, reduced_dim)

        # self.in_layer = torch.nn.Linear(EMBEDDING_DIM, reduced_dim)
        self.linear1 = torch.nn.Linear(reduced_dim, hidden_dim)
        self.dropout1 = torch.nn.Dropout(0.1)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = torch.nn.Dropout(0.1)
        self.linear3 = torch.nn.Linear(hidden_dim, reduced_dim)
        self.linear4 = torch.nn.Linear(reduced_dim, hidden_dim)
        self.dropout3 = torch.nn.Dropout(0.1)
        self.linear5 = torch.nn.Linear(hidden_dim, reduced_dim, bias=False)
        # self.out_layer = torch.nn.Linear(reduced_dim, EMBEDDING_DIM)

    def forward(self, embeddings: torch.Tensor, attention_mask: torch.Tensor):
        # embeddings = embeddings[:, :IMPRESSION_MAXLEN]
        # attention_mask = attention_mask[:, :IMPRESSION_MAXLEN]

        # pos_weight = (
        #     torch.sigmoid(self.alpha)
        #     .pow(torch.arange(embeddings.shape[1], device=DEVICE))
        #     .unsqueeze(0)
        #     .unsqueeze(-1)
        # )
        # embeddings = embeddings * pos_weight

        # embeddings = torch.cat(
        #     [
        #         embeddings,
        #         self.pos_emb_layer(torch.arange(embeddings.shape[1], device=DEVICE))
        #         .unsqueeze(0)
        #         .expand(embeddings.shape[0], -1, -1),
        #     ],
        #     dim=-1,
        # )

        # x = self.in_layer(embeddings)
        x = self.dropout1(F.relu(self.linear1(embeddings)))
        x = self.dropout2(F.relu(self.linear2(x)))
        x = self.linear3(x)
        weights = self.dropout3(F.relu(self.linear4(x)))
        weights = self.linear5(weights)
        # weights = weights.squeeze(2)
        weights = torch.exp(weights) * attention_mask.unsqueeze(-1)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-10)
        # weights = weights.unsqueeze(-1)
        # return self.out_layer((x * weights).sum(dim=1))
        return (x * weights).sum(dim=1)


# class FinalAttention(torch.nn.Module):
#     def __init__(self, embed_dim: int, hidden_dim: int):
#         super().__init__()
#         self.linear1 = torch.nn.Linear(embed_dim, embed_dim)
#         self.dropout1 = torch.nn.Dropout(0.1)
#         self.linear2 = torch.nn.Linear(embed_dim, embed_dim)
#         self.dropout2 = torch.nn.Dropout(0.1)
#         self.linear3 = torch.nn.Linear(embed_dim, embed_dim)
#         self.dropout3 = torch.nn.Dropout(0.1)
#         self.linear4 = torch.nn.Linear(embed_dim, hidden_dim)
#         self.linear5 = torch.nn.Linear(hidden_dim, 1, bias=False)

#     def forward(self, embeddings: torch.Tensor, attention_mask: torch.Tensor):
#         x = self.dropout1(F.relu(self.linear1(embeddings)))
#         x = self.dropout2(F.relu(self.linear2(embeddings)))
#         x = self.dropout3(F.relu(self.linear3(embeddings)))
#         weights = self.linear4(x)
#         weights = self.linear5(weights)
#         weights = weights.squeeze(2)
#         weights = torch.exp(weights) * attention_mask
#         weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-10)
#         weights = weights.unsqueeze(-1)
#         return (x * weights).sum(dim=1)


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
    model = FinalAttention(reduced_dim=REDUCED_DIM, hidden_dim=4096)
    if model_path:
        model.load_state_dict(torch.load(model_path, weights_only=True))
        # model = torch.load(model_path, weights_only=False)
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
                    model(**inputs.to(DEVICE)).last_hidden_state.detach().cpu(),
                    inputs["attention_mask"].cpu(),
                )
            )
            gc.collect()
            torch.cuda.empty_cache()
    return torch.concatenate(text_embed_list)


# See if we can use protocon for collate_fn type hint
def get_embed_from_model(
    model: PreTrainedModel,
    text_dataset: Dataset,
    text_maxlen: int,
    text_collate_fn: Callable[[Iterable[str]], BatchEncoding],
):
    text_batch_size = int(
        get_text_inference_batch_size(model, text_maxlen) * 0.5
    )  # - 300
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


class EmbeddingWrapper(torch.nn.Module):
    def __init__(self, wrapped_model: torch.nn.Module):
        super().__init__()
        self.cat_embed = torch.nn.Embedding(15, 128)
        self.subcat_embed = torch.nn.Embedding(134, 128)
        self.wrapped_model = wrapped_model

    def forward(self, embeddings, *args, **kwargs):
        combined = self.add_embedding(embeddings)
        return self.wrapped_model(combined.to(dtype=torch.float32), *args, **kwargs)

    def add_embedding(self, embeddings):
        cat_embeds = self.cat_embed(embeddings[..., -2].to(dtype=torch.int32))
        subcat_embeds = self.subcat_embed(embeddings[..., -1].to(dtype=torch.int32))
        return torch.cat([embeddings[..., :-2], cat_embeds, subcat_embeds], dim=-1)


def get_embed_wrapped_model(wrap_model: torch.nn.Module):
    embed_wrapped_model = EmbeddingWrapper(wrap_model)
    embed_wrapped_model = embed_wrapped_model.to(device=DEVICE)
    return embed_wrapped_model


class ResizeWrapperModel(torch.nn.Module):
    def __init__(
        self,
        wrapped_model: torch.nn.Module,
        embed_dim=EMBEDDING_DIM,
        reduced_dim=REDUCED_DIM,
    ):
        super().__init__()
        self.bottleneck_in = torch.nn.Linear(embed_dim, reduced_dim)
        self.wrapped_model = wrapped_model
        self.bottleneck_out = torch.nn.Linear(reduced_dim, embed_dim)

    def forward(self, embeddings, *args, **kwargs):
        bottle_in = self.bottleneck_in(embeddings)
        wrap_out = self.wrapped_model(bottle_in, *args, **kwargs)
        return self.bottleneck_out(wrap_out)


def resize_wrap_model(wrap_model: torch.nn.Module):
    return ResizeWrapperModel(wrap_model).to(device=DEVICE)


def get_nv_embeds(model, texts: list[str], type: str):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    batch_size = get_nv_embed_batch_size(model) - 5
    if type == "query":
        instruction = QUERY_INSTRUCTION
    else:
        instruction = ""
    # res_list = []
    # for i in trange(0, len(texts), batch_size):
    #     sub_text = texts[i : i + batch_size]
    #     res_list.append(
    #         F.normalize(
    #             model.encode(
    #                 sub_text, instruction=instruction, max_length=NEWS_TEXT_MAXLEN
    #             ),
    #             p=2,
    #             dim=1,
    #         )
    #     )
    # return torch.concatenate(res_list)
    with torch.no_grad():
        res = model._do_encode(
            texts,
            batch_size=batch_size,
            instruction=instruction,
            max_length=NEWS_TEXT_MAXLEN,
            num_workers=NUM_WORKERS,
        )
    return F.normalize(res, p=2, dim=1)


def get_model_eval(
    dataloader: DataLoader,
    model: torch.nn.Module,
) -> torch.Tensor:

    result_list = []
    model.eval()
    with torch.no_grad():
        for item in tqdm(dataloader, desc="Model Inference"):
            if isinstance(item, tuple) or isinstance(item, list):
                result_list.append(
                    model(*map(lambda x: x.to(DEVICE), item)).detach().cpu()
                )
            else:
                result_list.append(model(item.to(DEVICE)).detach().cpu())
    return torch.cat(result_list)


def get_head_model(path: str, device=DEVICE):
    if path.endswith(".json"):
        my_config = AutoConfig.from_pretrained(path, trust_remote_code=True)

        my_model = AutoModel.from_config(my_config, trust_remote_code=True)
        return my_model.to(device)
    else:
        return AutoModel.from_pretrained(path, trust_remote_code=True).to(device)


def get_new_attention_model(model_path: Optional[Path] = None):
    # model = NewAttention(hidden_size=EMBEDDING_DIM)
    model = NewAttention(hidden_size=REDUCED_DIM)
    if model_path:
        model.load_state_dict(torch.load(model_path, weights_only=True))
    return model.to(DEVICE)


class ReducingModel(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.linear2 = torch.nn.Linear(output_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.linear(x))
        return self.linear2(x)


def get_reducing_model(model_path: Optional[Path] = None):
    model = ReducingModel(input_dim=EMBEDDING_DIM, output_dim=REDUCED_DIM)
    if model_path:
        model.load_state_dict(torch.load(model_path, weights_only=True))
    return model.to(DEVICE)


def store_text_embed_full_eval(
    model: PreTrainedModel, input_dataloader: DataLoader, db_name: str
):
    with torch.no_grad(), sqlite3.connect(db_name) as conn:
        conn.execute("DROP TABLE IF EXISTS tensors;")
        conn.execute("CREATE TABLE tensors (id INTEGER PRIMARY KEY, data BLOB)")
        for inputs in tqdm(input_dataloader, desc="Embedding Text"):
            res = model(**inputs.to(DEVICE)).last_hidden_state.detach().cpu()
            attn_mask = inputs["attention_mask"].cpu()
            for i in range(len(res)):
                save_tensor = res[i][attn_mask[i] == 1]
                buffer = io.BytesIO()
                torch.save(save_tensor, buffer)
                buffer.seek(0)
                conn.execute("INSERT INTO tensors (data) VALUES (?)", (buffer.read(),))
                buffer.close()
            gc.collect()
            torch.cuda.empty_cache()


# See if we can use protocon for collate_fn type hint
def store_embed_from_model(
    model: PreTrainedModel,
    text_dataset: Dataset,
    text_maxlen: int,
    text_collate_fn: Callable[[Iterable[str]], BatchEncoding],
    db_name: str,
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
    store_text_embed_full_eval(model, text_dataloader, db_name)


class FirstAttentionPoolFunc(torch.nn.Module):
    def __init__(
        self, pool_func, embedding_dim=EMBEDDING_DIM, num_layers=NUM_HIDDEN_LAYERS
    ):
        super().__init__()
        self.pool_func = pool_func
        self.encoder = MyEncoder(
            hidden_size=embedding_dim, num_hidden_layers=num_layers
        )

    def forward(self, embeddings, attention_mask):
        # x = torch.utils.checkpoint.checkpoint(
        #     self.encoder, (embeddings, attention_mask), use_reentrant=False
        # )
        x = self.encoder(embeddings, attention_mask)
        return self.pool_func(x, attention_mask)


def get_token_attn_model(model_path: Optional[Path] = None):
    model = FirstAttentionPoolFunc(
        pool_func=last_token_pool,
        embedding_dim=EMBEDDING_DIM,
        num_layers=NUM_HIDDEN_LAYERS,
    )
    if model_path:
        model.load_state_dict(torch.load(model_path, weights_only=True))
    return model.to(DEVICE)
