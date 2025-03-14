from abc import ABC, abstractmethod
from typing import Any, Optional
from collections.abc import Iterable
from pathlib import Path
import joblib


def check_req_keys(required_keys: set[str], context_dict: dict[str, Any]):
    for key in required_keys:
        assert key in context_dict, f"Required Key {key} is not present in context_dict"


class PipelineComponent(ABC):
    required_keys = set()
    train_required_keys = set()

    @abstractmethod
    def transform(
        self,
        context_dict: dict[str, Any],
    ) -> dict[str, Any]:
        pass

    def train(
        self,
        context_dict: dict[str, Any],
        val_context_dict: Optional[dict[str, Any]] = None,
    ) -> None:
        pass


class Pipeline:
    def __init__(
        self,
        name: str,
        steps: Iterable[tuple[str, PipelineComponent]],
        use_cache: bool = True,
        cache_dir: Path = Path("cache"),
    ):
        self.name = name
        self._steps = list(steps)
        self.use_cache = use_cache
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = cache_dir

    def _iterate_over_steps(
        self,
        context_dict: dict[str, Any],
        val_context_dict: Optional[dict[str, Any]] = None,
        training: bool = False,
    ) -> tuple[dict[str, Any], Optional[dict[str, Any]]]:
        for step_name, component in self._steps:
            print(f"Starting step {step_name}")
            cache_file_name = self.cache_dir / f"{self.name}_{step_name}.pkl.gz"
            if self.use_cache and cache_file_name.is_file():
                loaded_file = joblib.load(cache_file_name)
                context_dict = loaded_file["context_dict"]
                val_context_dict = loaded_file["context_dict"]
            else:
                if training:
                    component.train(context_dict, val_context_dict)
                context_dict = component.transform(context_dict)
                if val_context_dict:
                    val_context_dict = component.transform(val_context_dict)
                if self.use_cache:
                    joblib.dump(
                        {
                            "context_dict": context_dict,
                            "val_context_dict": val_context_dict,
                        },
                        cache_file_name,
                        compress=True,
                    )
            print(f"Completed step {step_name}")
        return context_dict, val_context_dict

    def transform(
        self,
        context_dict: dict[str, Any],
        val_context_dict: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, Any], Optional[dict[str, Any]]]:
        return self._iterate_over_steps(context_dict, val_context_dict, training=False)

    def train(
        self,
        context_dict: dict[str, Any],
        val_context_dict: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, Any], Optional[dict[str, Any]]]:
        return self._iterate_over_steps(context_dict, val_context_dict, training=True)
