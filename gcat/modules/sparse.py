"""Sparse masking primitive used by :mod:`gcat.modules.masking`."""

from __future__ import annotations

import torch


def simple_mask(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.sparse_coo_tensor(
        x.indices(),
        torch.where(mask[x.indices()[0], x.indices()[1]], 0.0, x.values()),
        x.shape,
        device=x.device,
    )
