"""Small shared helpers used by the active GCAT modules."""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch
from torch import nn


def set_seed(seed: int, deterministic: bool = False) -> None:
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    logging.info("Setting global random seed to %s (deterministic=%s)", seed, deterministic)


def create_activation(name: str) -> nn.Module:
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported activation: {name!r}")


def create_norm(name: str, hidden_dim: int) -> nn.Module:
    if name == "groupnorm":
        return nn.GroupNorm(4, hidden_dim)
    if name == "layernorm":
        return nn.LayerNorm(hidden_dim)
    raise ValueError(f"Unsupported normalization: {name!r}")
