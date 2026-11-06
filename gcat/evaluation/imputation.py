"""Metrics used during GCAT imputation evaluation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _pearson_corr_1d(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    y_true_centered = y_true - y_true.mean()
    y_pred_centered = y_pred - y_pred.mean()
    return torch.mean(
        torch.sum(y_true_centered * y_pred_centered)
        / torch.sqrt(torch.sum(y_true_centered.square()))
        / torch.sqrt(torch.sum(y_pred_centered.square()))
    )


def imputation_eval(
    pred_labels: torch.Tensor,
    true_labels: torch.Tensor,
    dim: int = 1,
) -> dict[str, float]:
    """Return the historical GCAT aggregate imputation metrics."""
    mse_values = []
    rmse_values = []
    mae_values = []
    corr_values = []
    cosine_values = []

    for index in range(true_labels.shape[dim]):
        true_vector = true_labels[index] if dim == 0 else true_labels[:, index]
        pred_vector = F.relu(pred_labels[index] if dim == 0 else pred_labels[:, index])
        mse = F.mse_loss(pred_vector, true_vector).item()
        mse_values.append(mse)
        rmse_values.append(float(np.sqrt(mse)))
        mae_values.append(F.l1_loss(pred_vector, true_vector).item())
        corr_values.append(_pearson_corr_1d(pred_vector, true_vector).item())
        cosine_values.append(F.cosine_similarity(pred_vector, true_vector, dim=0).item())

    return {
        "mse": float(np.mean(mse_values)),
        "rmse": float(np.mean(rmse_values)),
        "mae": float(np.mean(mae_values)),
        "corr": float(np.mean(corr_values)),
        "cos": float(np.mean(cosine_values)),
    }
