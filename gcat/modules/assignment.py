"""Sparse assignment functions used by dynamic gene modules."""

from __future__ import annotations

import torch

try:
    from entmax import entmax15 as _package_entmax15
except ImportError:
    _package_entmax15 = None


def entmax15_exact(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Exact alpha=1.5 entmax using a sort-based threshold calculation."""
    if _package_entmax15 is not None:
        return _package_entmax15(logits, dim=dim)
    scaled = logits / 2.0
    scaled = scaled - scaled.max(dim=dim, keepdim=True).values
    sorted_values = torch.sort(scaled, dim=dim, descending=True).values
    range_shape = [1] * logits.dim()
    range_shape[dim] = logits.shape[dim]
    rho = torch.arange(
        1, logits.shape[dim] + 1, device=logits.device, dtype=logits.dtype
    ).view(range_shape)
    mean = sorted_values.cumsum(dim=dim) / rho
    mean_sq = sorted_values.square().cumsum(dim=dim) / rho
    ss = rho * (mean_sq - mean.square())
    delta = ((1.0 - ss) / rho).clamp_min(0.0)
    taus = mean - delta.sqrt()
    support_size = (taus <= sorted_values).sum(dim=dim, keepdim=True).clamp_min(1)
    tau_star = torch.gather(taus, dim, support_size.long() - 1)
    output = torch.clamp(scaled - tau_star, min=0.0).square()
    return output / output.sum(dim=dim, keepdim=True).clamp_min(1e-8)
