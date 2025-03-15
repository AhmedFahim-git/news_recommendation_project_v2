import torch
from .config import DEVICE, EMBEDDING_DIM, IMPRESSION_MAXLEN


def dummy_text_inputs_outputs(
    batch_size: int, max_len: int, device: torch.device = DEVICE
):
    return {
        "inputs": {
            "input_ids": torch.ones((batch_size, max_len), dtype=torch.int).to(device),
            "attention_mask": torch.ones((batch_size, max_len), dtype=torch.int).to(
                device
            ),
            "token_type_ids": torch.zeros((batch_size, max_len), dtype=torch.int).to(
                device
            ),
        },
        "outputs": torch.ones((batch_size,), dtype=torch.float32, device=device),
    }


def dummy_classificaion_inputs_outputs(
    batch_size: int, embedding_dim: int = EMBEDDING_DIM, device=DEVICE
):
    return {
        "inputs": {"embeddings": torch.rand((batch_size, embedding_dim)).to(device)},
        "outputs": torch.ones((batch_size, 1), dtype=torch.float32, device=device),
    }


def dummy_attention_inputs_outputs(
    batch_size: int,
    max_len: int = IMPRESSION_MAXLEN,
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
