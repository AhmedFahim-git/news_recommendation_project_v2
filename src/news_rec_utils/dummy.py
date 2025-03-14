import torch
from .config import DEVICE


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
