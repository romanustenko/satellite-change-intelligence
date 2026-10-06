"""Small shared precision, distributed coordination, and reproducibility utilities."""

import os
import random
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader


def resolve_device(device: str | torch.device) -> torch.device:
    """Resolve cuda to its current visible index; never silently fall back to CPU."""
    resolved = torch.device(device)
    if resolved.type not in ("cpu", "cuda"):
        raise ValueError("Use cpu, cuda, or cuda:N.")
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable.")
        index = torch.cuda.current_device() if resolved.index is None else resolved.index
        if not 0 <= index < torch.cuda.device_count():
            raise ValueError(f"CUDA index {index} is not visible to this process.")
        resolved = torch.device("cuda", index)
    return resolved


def rank() -> int:
    """Return zero outside a process group."""
    return dist.get_rank() if dist.is_initialized() else 0


def world_size() -> int:
    """Return one outside a process group."""
    return dist.get_world_size() if dist.is_initialized() else 1


def unwrap(model: nn.Module) -> nn.Module:
    """Remove only the DDP wrapper for validation and portable checkpoints."""
    return model.module if isinstance(model, DistributedDataParallel) else model


def on_rank_zero(operation: Callable[[], Any]) -> Any:
    """Run a shared operation once and broadcast its result or failure to every rank."""
    if not dist.is_initialized():
        return operation()
    message: list[Any] = [None, None]
    if rank() == 0:
        try:
            message[0] = operation()
        except Exception as error:
            message[1] = f"{type(error).__name__}: {error}"
    dist.broadcast_object_list(message, src=0)
    if message[1] is not None:
        raise RuntimeError(f"Rank-zero operation failed: {message[1]}")
    return message[0]


def precision_context(device: torch.device, precision: str) -> Any:
    """Keep FP32 as default; explicit BF16 requires a supported CUDA device."""
    if precision == "fp32":
        return nullcontext()
    if precision != "bf16":
        raise ValueError("precision must be 'fp32' or 'bf16'.")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("bf16 precision requires a CUDA device with BF16 support.")
    with torch.cuda.device(device):
        if not torch.cuda.is_bf16_supported():
            raise ValueError("The selected CUDA device does not support BF16.")
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def seed_everything(seed: int) -> None:
    """Seed model/data RNGs and preserve the project's deterministic HPO policy."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def capture_rng(loader: DataLoader) -> dict[str, Any]:
    """Capture per-rank RNGs and the loader generator at an epoch boundary."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "loader": loader.generator.get_state() if loader.generator is not None else None,
    }


def restore_rng(state: dict[str, Any], loader: DataLoader) -> None:
    """Restore RNGs; persistent-worker mid-epoch continuation is not supported."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda"].cpu())
    if state["loader"] is not None:
        if loader.generator is None:
            raise ValueError("Resume requires a DataLoader generator matching the saved run.")
        loader.generator.set_state(state["loader"].cpu())


def gather_objects(value: Any) -> list[Any]:
    """Collect small per-rank metadata without reducing derived metrics."""
    values: list[Any] = [None] * world_size()
    if dist.is_initialized():
        dist.all_gather_object(values, value)
    else:
        values[0] = value
    return values
