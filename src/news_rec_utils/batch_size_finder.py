from functools import partial
import gc
from typing import Callable, Any
import torch
from .dummy import (
    dummy_text_inputs_outputs,
    dummy_classificaion_inputs_outputs,
    dummy_attention_inputs_outputs,
)


# Keys are ModelConfig_str(optimizer_type)_task_max_len. Value is batch size
BATCH_SIZES = dict()


def dummy_group_text_inference_func(
    model: torch.nn.Module,
    dummy_func,
    max_len: int,
    batch_size: int,
):
    with torch.no_grad():
        model(**dummy_func(batch_size=batch_size, max_len=max_len)["inputs"])


def dummy_inference_func(
    model: torch.nn.Module,
    dummy_func,
    batch_size: int,
):
    with torch.no_grad():
        model(**dummy_func(batch_size=batch_size)["inputs"])


def dummy_text_train_func(model, optimizer, dummy_func, batch_size):
    optimizer.zero_grad()
    # with torch.autocast("cuda", torch.float16):
    model(
        **dummy_func(batch_size=batch_size)["inputs"]
    ).last_hidden_state.min().backward()
    optimizer.zero_grad()


def dummy_train_func(model, optimizer, dummy_func, batch_size):
    optimizer.zero_grad()
    # with torch.autocast("cuda", torch.float16):
    model(**dummy_func(batch_size=batch_size)["inputs"]).min().backward()
    optimizer.zero_grad()


def check_batch_size(test_func, batch_size: int):
    success, error = False, False
    try:
        test_func(batch_size=batch_size)
        success = True
    except torch.cuda.OutOfMemoryError as e:
        success = False
    except Exception as e:
        error = True
        print(e)
        print("Return None for batch size")
    finally:
        gc.collect()
        torch.cuda.empty_cache()
    return success, error


def get_batch_size(test_func: Callable[[int], None]):
    low = 0
    high = 1
    not_even_one = True
    while True:
        success, error = check_batch_size(test_func, high)
        if error:
            return None

        if success:
            low, high = high, high * 2
            if low == 1:
                not_even_one = False
        elif not_even_one:
            print(
                "Even batch size 1 fits into memory, try lower max len. Returning None"
            )
            return None
        else:
            break
    while high - low > 1:
        mid = low + (high - low) // 2
        success, error = check_batch_size(test_func, mid)
        if error:
            return None
        if success:
            low = mid
        else:
            high = mid
    return low


def get_text_inference_batch_size(model: torch.nn.Module, max_len: int):
    # if model.device.type != "cuda":
    #     print("Model is on CPU. Not CUDA. Returning batch size of 5")
    #     return 5
    model_part = str(model.config if hasattr(model, "config") else model)
    task_type = "TEXT_INFERENCE"
    key = model_part + task_type + str(max_len)
    if key not in BATCH_SIZES:
        BATCH_SIZES[key] = get_batch_size(
            partial(
                dummy_group_text_inference_func,
                model=model,
                dummy_func=dummy_text_inputs_outputs,
                max_len=max_len,
            )
        )
    return BATCH_SIZES[key]


def get_classification_train_batch_size(model: torch.nn.Module, optimizer):
    # if model.device.type != "cuda":
    #     print("Model is on CPU. Not CUDA. Returning batch size of 5")
    #     return 5
    model_part = "classification"
    task_type = "TRAIN"
    key = model_part + task_type
    if key not in BATCH_SIZES:
        BATCH_SIZES[key] = get_batch_size(
            partial(
                dummy_train_func,
                model=model,
                optimizer=optimizer,
                dummy_func=dummy_classificaion_inputs_outputs,
            )
        )
    return BATCH_SIZES[key]


def get_classification_inference_batch_size(model: torch.nn.Module):
    # if model.device.type != "cuda":
    #     print("Model is on CPU. Not CUDA. Returning batch size of 5")
    #     return 5
    model_part = "classification"
    task_type = "INFERENCE"
    key = model_part + task_type
    if key not in BATCH_SIZES:
        BATCH_SIZES[key] = get_batch_size(
            partial(
                dummy_inference_func,
                model=model,
                dummy_func=dummy_classificaion_inputs_outputs,
            )
        )
    return BATCH_SIZES[key]


def get_attention_train_batch_size(model: torch.nn.Module, optimizer):
    # if model.device.type != "cuda":
    #     print("Model is on CPU. Not CUDA. Returning batch size of 5")
    #     return 5
    model_part = "attention"
    task_type = "TRAIN"
    key = model_part + task_type
    if key not in BATCH_SIZES:
        BATCH_SIZES[key] = get_batch_size(
            partial(
                dummy_train_func,
                model=model,
                optimizer=optimizer,
                dummy_func=dummy_attention_inputs_outputs,
            )
        )
    return BATCH_SIZES[key]


def get_attention_inference_batch_size(model: torch.nn.Module):
    # if model.device.type != "cuda":
    #     print("Model is on CPU. Not CUDA. Returning batch size of 5")
    #     return 5
    model_part = "attention"
    task_type = "INFERENCE"
    key = model_part + task_type
    if key not in BATCH_SIZES:
        BATCH_SIZES[key] = get_batch_size(
            partial(
                dummy_inference_func,
                model=model,
                dummy_func=dummy_attention_inputs_outputs,
            )
        )
    return BATCH_SIZES[key]
