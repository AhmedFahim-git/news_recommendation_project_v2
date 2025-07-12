import torch
from .config import (
    DEVICE,
    EMBEDDING_DIM,
    IMPRESSION_MAXLEN,
    REDUCED_DIM,
    NEWS_TEXT_MAXLEN,
)


def dummy_text_inputs_outputs(
    batch_size: int, max_len: int, device: torch.device = DEVICE
):
    return {
        "inputs": {
            "input_ids": torch.ones((batch_size, max_len), dtype=torch.int).to(device),
            "attention_mask": torch.ones((batch_size, max_len), dtype=torch.int).to(
                device
            ),
            # "token_type_ids": torch.zeros((batch_size, max_len), dtype=torch.int).to(
            #     device
            # ),
        },
        "outputs": torch.ones((batch_size,), dtype=torch.float32, device=device),
    }


def dummy_classificaion_inputs_outputs(
    batch_size: int,
    # embedding_dim: int = EMBEDDING_DIM - 128 + 1,
    embedding_dim: int = EMBEDDING_DIM,
    device=DEVICE,
):
    return {
        "inputs": {"embeddings": torch.rand((batch_size, embedding_dim)).to(device)},
        "outputs": torch.ones((batch_size, 1), dtype=torch.float32, device=device),
    }


def dummy_attention_inputs_outputs(
    batch_size: int,
    max_len: int = IMPRESSION_MAXLEN,
    # embedding_dim: int = EMBEDDING_DIM - 128 + 1,
    embedding_dim: int = EMBEDDING_DIM,
    device=DEVICE,
):
    return {
        "inputs": {
            "embeddings": torch.rand((batch_size, max_len, embedding_dim)).to(device),
            "attention_mask": torch.ones((batch_size, max_len), dtype=torch.int32).to(
                device
            ),
        },
        "outputs": torch.ones(
            (batch_size, embedding_dim), dtype=torch.float32, device=device
        ),
    }


def dummy_attention_attention_inputs_outputs(
    batch_size: int,
    max_len=IMPRESSION_MAXLEN,
    news_text_maxlen=NEWS_TEXT_MAXLEN,
    embedding_dim=EMBEDDING_DIM,
    device=DEVICE,
):
    return {
        "inputs": {
            "token_attention": {
                "embeddings": torch.rand(
                    (
                        batch_size * max_len + 2 * batch_size,
                        news_text_maxlen,
                        embedding_dim,
                    )
                ).to(device),
                "attention_mask": torch.ones(
                    (batch_size * max_len + 2 * batch_size, news_text_maxlen),
                    dtype=torch.int32,
                ).to(device),
            },
            "final_attention": {
                "split_array": torch.tensor(
                    list(range(max_len, batch_size * max_len + 1, max_len))
                    + [batch_size * (max_len + 1) + 1]
                ),
                "attention_mask": torch.ones(
                    (batch_size, max_len), dtype=torch.int32
                ).to(device),
            },
        },
        "outputs": torch.ones(
            (batch_size, embedding_dim), dtype=torch.float32, device=device
        ),
    }


def dummy_token_attention_inputs_outputs(
    batch_size: int,
    news_text_maxlen=NEWS_TEXT_MAXLEN,
    embedding_dim=EMBEDDING_DIM,
    device=DEVICE,
):
    return {
        "inputs": {
            "embeddings": torch.rand(
                (
                    batch_size,
                    news_text_maxlen,
                    embedding_dim,
                )
            ).to(device),
            "attention_mask": torch.ones(
                (batch_size, news_text_maxlen),
                dtype=torch.int32,
            ).to(device),
        },
        "outputs": torch.ones(
            (batch_size, embedding_dim), dtype=torch.float32, device=device
        ),
    }
