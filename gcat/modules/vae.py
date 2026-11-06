"""Variational latent layer and KL loss used by GCATFormer."""

from __future__ import annotations

import torch
from torch import nn


class VAELatentLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        latent_dim: int,
        *,
        kl_weight: float = 1.0,
        warmup_steps: int = 10000,
        loss_threshold: float = 5.0,
    ):
        super().__init__()
        self.hid_2mu = nn.Linear(hidden_dim, latent_dim)
        self.hid_2sigma = nn.Linear(hidden_dim, latent_dim)
        self.kl_weight = 0.0
        self.max_kl_weight = float(kl_weight)
        self.step_count = 0
        self.warmup_steps = int(warmup_steps)
        self.loss_threshold = float(loss_threshold)

    def _step_schedule(self) -> None:
        self.step_count += 1
        if self.step_count < self.warmup_steps:
            self.kl_weight += self.max_kl_weight / self.warmup_steps

    def forward(self, x_dict, sample: bool = True):
        hidden = x_dict["h"]
        mean = self.hid_2mu(hidden)
        # Encoder mean of the current forward; consumed by the optional
        # niche relation loss (never the sampled latent).
        self.last_mean = mean
        log_var = torch.clamp(self.hid_2sigma(hidden), -5, 5)
        if sample:
            std = (torch.exp(log_var) + 1e-4).sqrt()
            log_var = 2 * torch.log(std)
        else:
            std = torch.exp(0.5 * log_var)

        if self.training:
            latent = mean + std * torch.randn_like(std)
            kl_loss = -0.5 * (
                1 + log_var - mean.square() - log_var.exp()
            ).sum(1).mean() * self.kl_weight
            if kl_loss < self.loss_threshold:
                kl_loss = 0
            self._step_schedule()
        else:
            latent = mean
            kl_loss = 0
        return latent, kl_loss
