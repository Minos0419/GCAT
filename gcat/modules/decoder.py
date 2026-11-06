"""Negative-binomial decoder used by GCATFormer."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from .common import create_norm


class NegativeBinomialOutput(nn.Module):
    def __init__(self, hidden_dim: int, output_dim: int):
        super().__init__()
        self.hidden_layer = nn.Linear(hidden_dim, 128)
        self.mean_layer = nn.Linear(128, output_dim)
        self.log_dispersion = nn.Parameter(torch.randn(output_dim))

    def forward(
        self,
        latent: torch.Tensor,
        mean_residual: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = F.relu(self.hidden_layer(latent))
        log_mean = self.mean_layer(hidden)
        if mean_residual is not None:
            # Optional signed GRN residual on the log-mean pre-activation;
            # the dispersion branch is untouched.
            log_mean = log_mean + mean_residual.to(dtype=log_mean.dtype)
        mean = torch.exp(log_mean).clamp(min=1e-5, max=1e6)
        dispersion = torch.exp(self.log_dispersion).clamp(min=1e-4, max=1e4)
        return mean, dispersion.expand(latent.shape[0], -1)


class NegativeBinomialDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        dropout: float,
        normalization: str,
        batch_count: int = 0,
        dataset_count: int = 0,
        platform_count: int = 0,
        modality_count: int = 0,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.covariate_layers = nn.ModuleList()
        self.covariate_counts = {
            "batch": batch_count,
            "dataset": dataset_count,
            "platform": platform_count,
            "modality": modality_count,
        }

        for layer_index in range(num_layers - 1):
            layer_input_dim = hidden_dim if layer_index else input_dim
            self.layers.append(
                nn.Sequential(
                    nn.Linear(layer_input_dim, hidden_dim),
                    nn.PReLU(),
                    nn.Dropout(dropout),
                    create_norm(normalization, hidden_dim),
                )
            )
            covariates = nn.ModuleDict()
            for name, count in self.covariate_counts.items():
                if count > 0:
                    covariates[name] = nn.Sequential(
                        nn.Embedding(count, hidden_dim),
                        nn.PReLU(),
                        create_norm(normalization, hidden_dim),
                    )
            self.covariate_layers.append(covariates)

        self.output_layer = NegativeBinomialOutput(hidden_dim, output_dim)

    @staticmethod
    def _average_covariate(layer: nn.Sequential) -> torch.Tensor:
        embedding_sum = layer[0].weight.detach().sum(0, keepdim=True)
        return layer[2](layer[1](embedding_sum))

    def forward(self, x_dict, mean_residual: Optional[torch.Tensor] = None):
        hidden = x_dict["h"]
        for layer, covariates in zip(self.layers, self.covariate_layers):
            hidden = layer(hidden)
            for name, covariate_layer in covariates.items():
                if name in x_dict:
                    hidden = hidden + covariate_layer(x_dict[name])
                else:
                    hidden = hidden + self._average_covariate(covariate_layer)

        mean, dispersion = self.output_layer(hidden, mean_residual=mean_residual)
        return {
            "mean": mean,
            "disp": dispersion,
            "recon": mean,
            "latent": x_dict["h"],
        }
