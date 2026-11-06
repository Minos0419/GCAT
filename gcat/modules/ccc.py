"""Cell-cell communication attention bias normalization."""

from __future__ import annotations

import torch


def prepare_cell_ccc_bias(
    ccc_bias,
    cell_tok: torch.Tensor,
    *,
    enabled: bool,
    bias_scale: torch.Tensor,
    program_edge_index=None,
    program_score=None,
    program_projection=None,
):
    if not enabled or ccc_bias is None:
        return None
    if cell_tok.dim() == 2:
        batch_size, cell_len = 1, cell_tok.shape[0]
    elif cell_tok.dim() == 3:
        batch_size, cell_len = cell_tok.shape[0], cell_tok.shape[1]
    else:
        raise ValueError(f"cell_tok must have dim 2 or 3, got shape {tuple(cell_tok.shape)}.")

    ccc_bias = torch.as_tensor(ccc_bias, device=cell_tok.device, dtype=cell_tok.dtype)
    if ccc_bias.dim() == 2:
        if tuple(ccc_bias.shape) != (cell_len, cell_len):
            raise ValueError(
                f"ccc_bias 2D shape must be ({cell_len},{cell_len}), got {tuple(ccc_bias.shape)}."
            )
        ccc_bias = ccc_bias.unsqueeze(0)
    elif ccc_bias.dim() == 3:
        if ccc_bias.shape[0] not in (1, batch_size) or tuple(ccc_bias.shape[1:]) != (cell_len, cell_len):
            raise ValueError(
                f"ccc_bias 3D shape must be (1|{batch_size},{cell_len},{cell_len}), got {tuple(ccc_bias.shape)}."
            )
    else:
        raise ValueError(f"ccc_bias must have dim 2 or 3, got shape {tuple(ccc_bias.shape)}.")
    if ccc_bias.shape[0] == 1 and batch_size > 1:
        ccc_bias = ccc_bias.expand(batch_size, -1, -1)

    positive_mask = ccc_bias > 0
    norm_bias = torch.zeros_like(ccc_bias)
    if bool(positive_mask.any().item()):
        pos_values = torch.log1p(ccc_bias[positive_mask].float())
        scale = torch.quantile(pos_values, 0.95).clamp_min(1e-6)
        norm_bias[positive_mask] = torch.clamp(pos_values / scale, min=0.0, max=1.0).to(norm_bias.dtype)
    eye = torch.eye(cell_len, device=norm_bias.device, dtype=torch.bool).unsqueeze(0)
    norm_bias = norm_bias.masked_fill(eye, 0.0)
    scalar_bias = norm_bias * bias_scale.to(device=norm_bias.device, dtype=norm_bias.dtype)

    if program_projection is None:
        return scalar_bias
    if program_edge_index is None or program_score is None:
        raise ValueError(
            "ccc_program_head_bias is enabled but program edge indices/scores are missing."
        )
    if batch_size != 1:
        raise ValueError(
            "Sparse CCC program edges currently require one sampled graph per forward; "
            f"got batch_size={batch_size}."
        )
    program_edge_index = torch.as_tensor(
        program_edge_index,
        device=cell_tok.device,
        dtype=torch.long,
    )
    program_score = torch.as_tensor(
        program_score,
        device=cell_tok.device,
        dtype=program_projection.weight.dtype,
    )
    if program_edge_index.dim() != 2 or program_edge_index.shape[0] != 2:
        raise ValueError(
            "ccc_program_edge_index must have shape (2,E), got "
            f"{tuple(program_edge_index.shape)}."
        )
    edge_count = int(program_edge_index.shape[1])
    if program_score.dim() != 2 or program_score.shape[0] != edge_count:
        raise ValueError(
            "ccc_program_score must have shape (E,P) aligned with edge_index; got "
            f"{tuple(program_score.shape)} for E={edge_count}."
        )
    if program_score.shape[1] != program_projection.in_features:
        raise ValueError(
            f"ccc_program_score P={program_score.shape[1]} does not match projection "
            f"input={program_projection.in_features}."
        )
    if edge_count == 0:
        return scalar_bias.unsqueeze(1).expand(-1, program_projection.out_features, -1, -1)
    if bool((program_edge_index < 0).any().item()) or bool((program_edge_index >= cell_len).any().item()):
        raise ValueError("ccc_program_edge_index contains an out-of-range local cell index.")

    edge_head_delta = program_projection(program_score).to(dtype=cell_tok.dtype)  # (E,H)
    delta_bias = cell_tok.new_zeros((1, program_projection.out_features, cell_len, cell_len))
    receiver, sender = program_edge_index[0], program_edge_index[1]
    delta_bias[0, :, receiver, sender] = edge_head_delta.transpose(0, 1)
    return scalar_bias.unsqueeze(1) + delta_bias
