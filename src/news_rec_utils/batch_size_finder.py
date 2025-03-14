from functools import partial
import gc
from typing import Callable, Any
import torch
from .dummy import dummy_text_inputs_outputs

# Keys are ModelConfig_str(optimizer_type)_task_max_len. Value is batch size
BATCH_SIZES = dict()


def dummy_inference_func(
    model: torch.nn.Module,
    dummy_func: Callable[[int, int, torch.device], dict[str, Any]],
    max_len: int,
    batch_size: int,
):
    with torch.no_grad():
        model(**dummy_func(batch_size, max_len, model.device)["inputs"])


def check_batch_size(test_func: Callable[[int], None], batch_size: int):
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
    if model.device.type != "cuda":
        print("Model is on CPU. Not CUDA. Returning batch size of 5")
        return 5
    model_part = str(model.config if hasattr(model, "config") else model)
    task_type = "TEXT_INFERENCE"
    key = model_part + task_type + str(max_len)
    if key not in BATCH_SIZES:
        BATCH_SIZES[key] = get_batch_size(
            partial(
                dummy_inference_func,
                model=model,
                dummy_func=dummy_text_inputs_outputs,
                max_len=max_len,
            )
        )
    return BATCH_SIZES[key]
