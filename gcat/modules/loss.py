"""Training losses used by GCATFormer."""

from __future__ import annotations

import torch
from torch import nn


def _to_dense_tensor(x: torch.Tensor) -> torch.Tensor:
    """Convert dense/sparse expression tensors to a dense tensor."""
    return x.to_dense() if hasattr(x, "to_dense") else x


def regulon_reference_activity(
    full_expression: torch.Tensor,
    assignment: torch.Tensor,
) -> torch.Tensor:
    """Compute the detached full-expression module/regulon score target."""
    full = _to_dense_tensor(full_expression).to(device=assignment.device)
    reference_assignment = assignment.detach()
    full_log = torch.log1p(full.clamp_min(0.0)).to(dtype=reference_assignment.dtype)
    denom = reference_assignment.abs().sum(dim=0).clamp_min(1.0)
    return (full_log @ reference_assignment) / denom.unsqueeze(0)


def regulon_activity_auxiliary_loss(
    predicted_activity: torch.Tensor,
    full_expression: torch.Tensor,
    assignment: torch.Tensor,
    *,
    source_mask: torch.Tensor | None = None,
    lambda_reg: float = 0.05,
    std_floor: float = 0.1,
) -> torch.Tensor:
    """MSE auxiliary loss for masked-input regulon/module activity.

    ``predicted_activity`` is the scalar readout of GCAT's encoder-updated
    module tokens.  The reference is computed from the unmasked ``x_seq``
    using the same gene-to-module assignment, after ``log1p``.  Per-module
    reference mean and standard deviation are used to standardize the target;
    the scalar readout predicts that standardized target directly.  The
    prediction is deliberately not divided by the reference standard
    deviation.  Assignment and target statistics are detached so this target
    cannot update the assignment through the full-expression branch.

    The optional ``source_mask`` can restrict supervised rows.  Normal GCAT
    training uses all ST and scRNA rows from the pre-random-mask ``x_seq``.
    Fold-held-out test genes have already been zeroed in ST ``x_seq``.
    """
    if lambda_reg == 0:
        return predicted_activity.new_tensor(0.0)
    if std_floor <= 0:
        raise ValueError(f"std_floor must be > 0, got {std_floor}")
    if predicted_activity.ndim != 2 or assignment.ndim != 2:
        raise ValueError(
            "predicted_activity and assignment must be 2D, got "
            f"{tuple(predicted_activity.shape)} and {tuple(assignment.shape)}"
        )

    full = _to_dense_tensor(full_expression).to(device=predicted_activity.device)
    if full.ndim != 2 or full.shape[1] != assignment.shape[0]:
        raise ValueError(
            "full_expression and assignment gene dimensions must match, got "
            f"{tuple(full.shape)} and {tuple(assignment.shape)}"
        )
    if predicted_activity.shape[0] != full.shape[0] or predicted_activity.shape[1] != assignment.shape[1]:
        raise ValueError(
            "predicted_activity shape must be (batch, modules), got "
            f"{tuple(predicted_activity.shape)} for full/assignment "
            f"{tuple(full.shape)}/{tuple(assignment.shape)}"
        )

    if source_mask is None:
        rows = torch.ones(full.shape[0], dtype=torch.bool, device=full.device)
    else:
        rows = torch.as_tensor(source_mask, device=full.device, dtype=torch.bool).reshape(-1)
        if rows.numel() != full.shape[0]:
            raise ValueError(
                f"source_mask length {rows.numel()} does not match batch size {full.shape[0]}"
            )
    if not bool(rows.any().item()):
        return predicted_activity.new_tensor(0.0)

    # Keep the existing full-expression module_expr definition as the target,
    # but detach assignment so the reference branch cannot train the module
    # projection parameters.
    reference_raw = regulon_reference_activity(full, assignment)
    reference_source = reference_raw[rows]
    pred_source = predicted_activity[rows]

    # Standardize each module using the selected full-expression cells.
    ref_mean = reference_source.mean(dim=0, keepdim=True)
    ref_std = reference_source.std(dim=0, keepdim=True, unbiased=False).clamp_min(
        float(std_floor)
    )
    target = ((reference_source - ref_mean) / ref_std).detach()
    prediction = pred_source
    finite = torch.isfinite(target) & torch.isfinite(prediction)
    if not bool(finite.any().item()):
        return predicted_activity.new_tensor(0.0)
    return ((prediction - target).square().masked_select(finite)).mean()


def masked_gene_standardized_mse(
    predicted_expression: torch.Tensor,
    full_expression: torch.Tensor,
    block_mask: torch.Tensor,
    *,
    seed_mask: torch.Tensor | None = None,
    source_mask: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Task-aligned loss for genes hidden across an ST neighborhood batch.

    Both prediction and truth are standardized across seed ST cells for each
    selected gene.  Their squared error is therefore a stable differentiable
    counterpart of the gene-wise PCC / z-scored RMSE used at evaluation.
    ``block_mask`` is produced only from platform-observed ST genes, so this
    objective never reads the outer-fold held-out truth.
    """

    pred = predicted_expression.float()
    truth = _to_dense_tensor(full_expression).to(device=pred.device, dtype=torch.float32)
    mask = torch.as_tensor(block_mask, device=pred.device, dtype=torch.bool)
    if pred.ndim != 2 or truth.shape != pred.shape or mask.shape != pred.shape:
        raise ValueError(
            "prediction, truth, and block_mask must share a 2D shape, got "
            f"{tuple(pred.shape)}, {tuple(truth.shape)}, and {tuple(mask.shape)}."
        )

    rows = torch.ones(pred.shape[0], dtype=torch.bool, device=pred.device)
    if seed_mask is not None:
        seeds = torch.as_tensor(seed_mask, device=pred.device, dtype=torch.bool).reshape(-1)
        if seeds.numel() != pred.shape[0]:
            raise ValueError("seed_mask length does not match batch size.")
        rows &= seeds
    if source_mask is not None:
        source = torch.as_tensor(source_mask, device=pred.device, dtype=torch.bool).reshape(-1)
        if source.numel() != pred.shape[0]:
            raise ValueError("source_mask length does not match batch size.")
        rows &= ~source
    mask &= rows.unsqueeze(1)

    selected_genes = mask.sum(dim=0) >= 2
    if not bool(selected_genes.any().item()):
        return pred.sum() * 0.0
    pred = pred[:, selected_genes]
    truth = truth[:, selected_genes]
    mask = mask[:, selected_genes]

    count = mask.sum(dim=0).to(dtype=torch.float32).clamp_min(1.0)
    pred_mean = (pred * mask).sum(dim=0) / count
    truth_mean = (truth * mask).sum(dim=0) / count
    pred_centered = (pred - pred_mean.unsqueeze(0)) * mask
    truth_centered = (truth - truth_mean.unsqueeze(0)) * mask
    pred_std = torch.sqrt(pred_centered.square().sum(dim=0) / count).clamp_min(eps)
    truth_std = torch.sqrt(truth_centered.square().sum(dim=0) / count).clamp_min(eps)
    pred_z = pred_centered / pred_std.unsqueeze(0)
    truth_z = truth_centered / truth_std.unsqueeze(0)
    return ((pred_z - truth_z).square() * mask).sum() / mask.sum().clamp_min(1)


class NegativeBinomialLoss(nn.Module):
    """NB imputation loss with observed-gene library scaling."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def _to_dense(x):
        return x.to_dense() if hasattr(x, "to_dense") else x

    @staticmethod
    def _crop_if_full_width(x, idx, original_width):
        if x.shape[1] == original_width:
            return x[:, idx]
        return x

    @staticmethod
    def _normalize_gene_mask(gene_mask, batch_size, num_genes, device):
        if gene_mask is None:
            return torch.ones(batch_size, num_genes, dtype=torch.bool, device=device)

        if not torch.is_tensor(gene_mask):
            gene_mask = torch.as_tensor(gene_mask, device=device)
        else:
            gene_mask = gene_mask.to(device)

        if gene_mask.dim() == 1:
            if gene_mask.dtype == torch.bool:
                if gene_mask.numel() != num_genes:
                    raise ValueError(
                        f"gene_mask length {gene_mask.numel()} does not match gene dim {num_genes}"
                    )
                return gene_mask.unsqueeze(0).expand(batch_size, -1)

            mask = torch.zeros(num_genes, dtype=torch.bool, device=device)
            mask[gene_mask.long()] = True
            return mask.unsqueeze(0).expand(batch_size, -1)

        if gene_mask.dim() == 2:
            if gene_mask.dtype == torch.bool:
                if gene_mask.shape != (batch_size, num_genes):
                    raise ValueError(
                        f"gene_mask shape {tuple(gene_mask.shape)} does not match {(batch_size, num_genes)}"
                    )
                return gene_mask

            mask = torch.zeros(batch_size, num_genes, dtype=torch.bool, device=device)
            mask.scatter_(1, gene_mask.long(), True)
            return mask

        raise ValueError(f"Unsupported gene_mask dim: {gene_mask.dim()}")

    @staticmethod
    def _nb_nll(mean, disp, truth, eps=1e-10):
        t1 = torch.lgamma(disp + eps) + torch.lgamma(truth + 1.0) - torch.lgamma(truth + disp + eps)
        t2 = (disp + truth) * torch.log1p(mean / (disp + eps)) + truth * (
            torch.log(disp + eps) - torch.log(mean + eps)
        )
        return t1 + t2

    def forward(self, out_dict, x_dict):
        eps = 1e-10

        mean = out_dict["mean"]
        disp = out_dict["disp"]
        original_width = mean.shape[1]

        truth = self._to_dense(x_dict["x_seq"]).to(mean.device)
        scale_input = self._to_dense(x_dict.get("masked_x_seq", x_dict["x_seq"])).to(mean.device)
        gene_mask = x_dict.get("gene_mask")

        if "input_gene_mask" in x_dict:
            idx = x_dict["input_gene_mask"]
            if not torch.is_tensor(idx):
                idx = torch.as_tensor(idx, device=mean.device)
            else:
                idx = idx.to(mean.device)

            mean = mean[:, idx]
            disp = disp[:, idx]
            truth = self._crop_if_full_width(truth, idx, original_width)
            scale_input = self._crop_if_full_width(scale_input, idx, original_width)

            if gene_mask is not None:
                if not torch.is_tensor(gene_mask):
                    gene_mask = torch.as_tensor(gene_mask, device=mean.device)
                else:
                    gene_mask = gene_mask.to(mean.device)
                if gene_mask.dim() == 1 and gene_mask.numel() == original_width:
                    gene_mask = gene_mask[idx]
                elif gene_mask.dim() == 2 and gene_mask.shape[1] == original_width:
                    gene_mask = gene_mask[:, idx]

        if truth.shape[1] != mean.shape[1]:
            raise ValueError(
                f"x_seq gene dim {truth.shape[1]} does not match decoder mean dim {mean.shape[1]}"
            )
        if scale_input.shape[1] != mean.shape[1]:
            raise ValueError(
                f"scale input gene dim {scale_input.shape[1]} does not match decoder mean dim {mean.shape[1]}"
            )

        B, G = truth.shape
        visible_mask = self._normalize_gene_mask(gene_mask, B, G, mean.device)

        visible_library = scale_input.masked_fill(~visible_mask, 0.0).sum(1, keepdim=True)
        pred_library = mean.masked_fill(~visible_mask, 0.0).sum(1, keepdim=True).clamp_min(eps)
        scale_factor = visible_library / pred_library

        scaled_mean = mean * scale_factor
        out_dict["pred"] = scaled_mean

        nb_final = self._nb_nll(scaled_mean.clamp_min(eps), disp, truth, eps=eps)
        nb_final = nb_final.masked_fill(~visible_mask, 0.0)
        return nb_final.sum(-1).mean()
