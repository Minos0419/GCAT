"""Full-gene and cell self-attention building blocks."""

from __future__ import annotations

import math
import re
from typing import Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .common import create_activation, create_norm


def _apply_norm(norm: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if isinstance(norm, (nn.GroupNorm, nn.BatchNorm1d)):
        if x.dim() == 2:
            return norm(x)
        if x.dim() == 3:
            return norm(x.transpose(1, 2)).transpose(1, 2)
    return norm(x)


def _build_attn_bias(attn_mask, batch_size, q_len, k_len, device, dtype):
    if not torch.is_tensor(attn_mask):
        attn_mask = torch.tensor(attn_mask, device=device)
    else:
        attn_mask = attn_mask.to(device)

    if attn_mask.dim() == 1:
        attn_mask = attn_mask.unsqueeze(0)

    if attn_mask.dim() == 2:
        if attn_mask.shape[1] == k_len and attn_mask.shape[0] in (1, batch_size):
            attn_mask = attn_mask[:, None, None, :]
        elif attn_mask.shape[0] == q_len and attn_mask.shape[1] == k_len:
            attn_mask = attn_mask[None, None, :, :]
        else:
            raise ValueError(f"Unsupported 2D attn_mask shape: {tuple(attn_mask.shape)}")
    elif attn_mask.dim() == 3:
        if attn_mask.shape[0] in (1, batch_size) and attn_mask.shape[1] == q_len and attn_mask.shape[2] == k_len:
            attn_mask = attn_mask[:, None, :, :]
        else:
            raise ValueError(f"Unsupported 3D attn_mask shape: {tuple(attn_mask.shape)}")
    elif attn_mask.dim() != 4:
        raise ValueError(f"Unsupported attn_mask dim: {attn_mask.dim()}")

    if attn_mask.dtype == torch.bool:
        invalid = ~attn_mask
    elif torch.is_floating_point(attn_mask):
        if attn_mask.max() <= 1 and attn_mask.min() >= 0:
            invalid = attn_mask <= 0
        else:
            return attn_mask.to(dtype=dtype)
    else:
        invalid = attn_mask <= 0

    bias = torch.zeros_like(attn_mask, dtype=dtype, device=device)
    return bias.masked_fill(invalid, torch.finfo(dtype).min)


def rotate_every_two(x: torch.Tensor) -> torch.Tensor:
    d_last = x.size(-1)
    if d_last < 2:
        return x
    main_d = d_last if d_last % 2 == 0 else d_last - 1
    x_main = x[..., :main_d]
    x_tail = x[..., main_d:] if main_d < d_last else None
    x1 = x_main[..., ::2]
    x2 = x_main[..., 1::2]
    x_rot = torch.stack((-x2, x1), dim=-1).flatten(-2)
    if x_tail is not None:
        x_rot = torch.cat([x_rot, x_tail], dim=-1)
    return x_rot


def theta_shift(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    while sin.dim() < x.dim():
        sin = sin.unsqueeze(0)
        cos = cos.unsqueeze(0)
    return x * cos + rotate_every_two(x) * sin


def mix_flow_attention_scores(
    base_scores: torch.Tensor,
    flow_scores: torch.Tensor,
    flow_gate: torch.Tensor,
    flow_gate_mask: torch.Tensor,
) -> torch.Tensor:
    """Stage-4 directed-flow score mixing.

    Implements ``S_mix = S_base + gate * G * (S_flow - S_base)`` in float32.
    With ``gate = 0`` this is exactly ``S_base``; with ``gate = 1`` it reduces
    to the conceptual ``G * S_flow + (1 - G) * S_base`` replacement rule.
    """
    base = base_scores.float()
    flow = flow_scores.float()
    gate = flow_gate.float().view(1, -1, 1, 1)
    gate_mask = flow_gate_mask.float().unsqueeze(0).unsqueeze(0)
    return base + gate * gate_mask * (flow - base)


class SimpleRelPosEncoding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.rope_dim = dim if dim % 2 == 0 else max(dim - 1, 0)
        if self.rope_dim > 0:
            inv_freq = 1.0 / (10000 ** torch.linspace(0, 1, self.rope_dim // 2))
        else:
            inv_freq = torch.empty(0)
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len: int, device=None, dtype=None):
        if device is None:
            device = self.inv_freq.device
        if dtype is None:
            dtype = self.inv_freq.dtype if self.inv_freq.numel() > 0 else torch.float32
        if self.rope_dim == 0:
            sin = torch.zeros(seq_len, self.dim, device=device, dtype=dtype)
            cos = torch.ones(seq_len, self.dim, device=device, dtype=dtype)
            return sin, cos

        pos = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        sinusoid = torch.einsum("i,j->ij", pos, self.inv_freq.to(device))
        sin_half = sinusoid.sin()
        cos_half = sinusoid.cos()
        sin = torch.stack([sin_half, sin_half], dim=-1).flatten(-2)
        cos = torch.stack([cos_half, cos_half], dim=-1).flatten(-2)
        if self.rope_dim < self.dim:
            pad = self.dim - self.rope_dim
            sin = F.pad(sin, (0, pad), value=0.0)
            cos = F.pad(cos, (0, pad), value=1.0)
        return sin.to(dtype=dtype), cos.to(dtype=dtype)


def _valid_coord_mask_tensor(coord: torch.Tensor) -> torch.Tensor:
    # Normalized ST coordinates can be negative; only the (-1, -1) sentinel
    # and non-finite values should be treated as missing coordinates.
    coord = coord[..., :2]
    sentinel = torch.full_like(coord, -1.0)
    return torch.isfinite(coord).all(dim=-1) & (~torch.isclose(coord, sentinel).all(dim=-1))


def _resolve_spatial_pos(x_dict, pos=None):
    if pos is not None:
        return pos
    coord = x_dict.get("coord")
    if coord is None:
        return None
    if not torch.is_tensor(coord):
        coord = torch.as_tensor(coord)
    if coord.numel() == 0 or not bool(_valid_coord_mask_tensor(coord).any().item()):
        return None
    return coord


def _has_source_flag(x_dict) -> bool:
    source = x_dict.get("source")
    if source is None:
        return False
    if torch.is_tensor(source):
        return bool(source.reshape(-1)[0].item()) if source.numel() > 0 else False
    if isinstance(source, np.ndarray):
        return bool(source.reshape(-1)[0]) if source.size > 0 else False
    if isinstance(source, (list, tuple)):
        return bool(source[0]) if source else False
    return bool(source)


def _parse_layer_spec(layer_spec, num_layers: int) -> list[int]:
    if num_layers <= 0:
        return []
    if layer_spec is None:
        return [num_layers - 1]
    if isinstance(layer_spec, int):
        idx = layer_spec if layer_spec >= 0 else num_layers + layer_spec
        return [max(0, min(num_layers - 1, idx))]
    if isinstance(layer_spec, (list, tuple)):
        out = []
        for item in layer_spec:
            out.extend(_parse_layer_spec(item, num_layers))
        return sorted(set(out))

    spec = str(layer_spec).strip().lower()
    if spec in {"", "last"}:
        return [num_layers - 1]
    if spec == "all":
        return list(range(num_layers))
    if spec == "none":
        return []
    last_k = re.fullmatch(r"last(\d+)", spec)
    if last_k:
        # "last2" = the final two layers; clamped into [1, num_layers].
        count = max(1, min(int(last_k.group(1)), num_layers))
        return list(range(num_layers - count, num_layers))

    result = set()
    for chunk in spec.split(","):
        token = chunk.strip()
        if not token:
            continue
        range_sep = ":" if ":" in token else ("-" if "-" in token else None)
        if range_sep is not None:
            start_s, end_s = token.split(range_sep, 1)
            start = int(start_s)
            end = int(end_s)
            if start < 0:
                start += num_layers
            if end < 0:
                end += num_layers
            lo, hi = sorted((start, end))
            for idx in range(lo, hi + 1):
                if 0 <= idx < num_layers:
                    result.add(idx)
        else:
            idx = int(token)
            if idx < 0:
                idx += num_layers
            if 0 <= idx < num_layers:
                result.add(idx)
    return sorted(result)


class CellSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        qkv_bias: bool = True,
        cell_self_attn_backend: str = "flash",
        cell_gate_mode: str = "full",
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})."
            )
        if cell_self_attn_backend not in {"flash", "normal"}:
            raise ValueError(
                f"Unsupported cell_self_attn_backend='{cell_self_attn_backend}'. "
                "Supported: ['flash', 'normal']"
            )
        if cell_gate_mode not in {"full", "none"}:
            raise ValueError(
                f"Unsupported cell_gate_mode='{cell_gate_mode}'. Supported: ['full', 'none']"
            )
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.cell_self_attn_backend = cell_self_attn_backend
        self.cell_gate_mode = cell_gate_mode
        self.q_proj = nn.Linear(hidden_size, hidden_size + num_heads, bias=qkv_bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)

        # MOD: keep the distance-bias scale frozen so spatial bias stays a
        # fixed prior in the SDPA variant.
        self.beta = nn.Parameter(torch.ones(num_heads) * 0.1, requires_grad=False)
        self.relpos = SimpleRelPosEncoding(self.head_dim)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)
    def _sdpa_fallback_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attn_bias_4d: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attn_bias_4d,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=False,
        ).permute(0, 2, 1, 3).contiguous()

    def _normalize_pos(self, pos, batch_size, q_len, k_len, device, dtype):
        if pos is None:
            return None
        if not torch.is_tensor(pos):
            pos = torch.as_tensor(pos, device=device, dtype=dtype)
        else:
            pos = pos.to(device=device, dtype=dtype)

        if pos.dim() == 2:
            if pos.shape[0] != q_len:
                return None
            pos = pos.unsqueeze(0).expand(batch_size, -1, -1)
        elif pos.dim() == 3:
            if pos.shape[0] == 1 and batch_size > 1:
                pos = pos.expand(batch_size, -1, -1)
            elif pos.shape[0] != batch_size:
                return None
        else:
            return None

        if pos.shape[1] != q_len or q_len != k_len:
            return None
        return pos

    def _normalize_dist_bias(self, dist_bias, batch_size, q_len, k_len, device, dtype):
        if dist_bias is None:
            return None
        if not torch.is_tensor(dist_bias):
            dist_bias = torch.as_tensor(dist_bias, device=device, dtype=dtype)
        else:
            dist_bias = dist_bias.to(device=device, dtype=dtype)

        if dist_bias.dim() == 2:
            if dist_bias.shape != (q_len, k_len):
                return None
            dist_bias = dist_bias.unsqueeze(0).expand(batch_size, -1, -1)
        elif dist_bias.dim() == 3:
            if dist_bias.shape[0] == 1 and batch_size > 1:
                dist_bias = dist_bias.expand(batch_size, -1, -1)
            elif dist_bias.shape[0] != batch_size:
                return None
            if dist_bias.shape[1:] != (q_len, k_len):
                return None
        else:
            return None

        return torch.nan_to_num(dist_bias, nan=0.0, posinf=0.0, neginf=0.0)

    def _prepare_self_bias(
        self,
        bias: Optional[torch.Tensor],
        batch_size: int,
        q_len: int,
        k_len: int,
        device: torch.device,
        dtype: torch.dtype,
        name: str,
    ) -> Optional[torch.Tensor]:
        if bias is None:
            return None
        if not torch.is_tensor(bias):
            raise TypeError(f"{name} must be torch.Tensor or None, got {type(bias)}.")
        if bias.dtype == torch.bool:
            raise TypeError(f"{name} must be additive float bias, not bool mask.")

        if bias.dim() == 2:
            if tuple(bias.shape) != (q_len, k_len):
                raise ValueError(
                    f"{name} 2D shape must be ({q_len},{k_len}), got {tuple(bias.shape)}."
                )
            bias = bias.unsqueeze(0).unsqueeze(0)
        elif bias.dim() == 3:
            if bias.shape[0] not in (1, batch_size) or tuple(bias.shape[1:]) != (q_len, k_len):
                raise ValueError(
                    f"{name} 3D shape must be (B,Q,K) with B in (1,{batch_size}) and "
                    f"(Q,K)=({q_len},{k_len}), got {tuple(bias.shape)}."
                )
            bias = bias.unsqueeze(1)
        elif bias.dim() == 4:
            pass
        else:
            raise ValueError(
                f"{name} must have dim 2/3/4, got shape {tuple(bias.shape)}."
            )

        if bias.shape[0] not in (1, batch_size):
            raise ValueError(
                f"{name} batch dim must be 1 or {batch_size}, got {bias.shape[0]}."
            )
        if bias.shape[1] not in (1, self.num_heads):
            raise ValueError(
                f"{name} head dim must be 1 or {self.num_heads}, got {bias.shape[1]}."
            )
        if tuple(bias.shape[2:]) != (q_len, k_len):
            raise ValueError(
                f"{name} last dims must be (Q,K)=({q_len},{k_len}), got {tuple(bias.shape[2:])}."
            )
        return bias.to(device=device, dtype=dtype).expand(batch_size, self.num_heads, q_len, k_len).contiguous()

    def _run_backend_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attn_bias_4d: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # MOD: SDPA-only variant. We keep the backend argument for config
        # compatibility, but both "flash" and "normal" use the same torch SDPA core.
        return self._sdpa_fallback_attention(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attn_bias_4d=attn_bias_4d,
        )

    def _prepare_flow_scores(
        self,
        flow_scores,
        flow_gate_mask,
        flow_gate,
        q_len: int,
        device: torch.device,
    ):
        """Validate optional Stage-4 program-level flow attention inputs.

        ``flow_scores`` are precomputed directed program-compatibility scores
        ``(H, N, N)`` (or ``(1, H, N, N)``) with ``[query/receiver,
        key/sender]`` as the last two dims; ``flow_gate_mask`` is the dense
        directed adjacency ``(N, N)`` with ``mask[receiver, sender] = 1``;
        ``flow_gate`` is the per-head zero-initialized interpolation gate.
        Score math runs in float32 for FP16 stability.
        """
        provided = [
            value is not None
            for value in (flow_scores, flow_gate_mask, flow_gate)
        ]
        if not any(provided):
            return None
        if not all(provided):
            raise ValueError(
                "flow attention requires flow_scores, flow_gate_mask and "
                "flow_gate to be provided together."
            )
        if not torch.is_tensor(flow_scores):
            raise TypeError(f"flow_scores must be a torch.Tensor, got {type(flow_scores)}.")
        if flow_scores.dim() == 3:
            flow_scores = flow_scores.unsqueeze(0)
        expected = (1, self.num_heads, q_len, q_len)
        if flow_scores.dim() != 4 or tuple(flow_scores.shape) != expected:
            raise ValueError(
                f"flow_scores must have shape {expected} or {expected[1:]}, "
                f"got {tuple(flow_scores.shape)}."
            )
        scores = flow_scores.to(device=device, dtype=torch.float32)
        if not torch.is_tensor(flow_gate_mask):
            raise TypeError(f"flow_gate_mask must be a torch.Tensor, got {type(flow_gate_mask)}.")
        if flow_gate_mask.dim() != 2 or tuple(flow_gate_mask.shape) != (q_len, q_len):
            raise ValueError(
                f"flow_gate_mask must have shape ({q_len},{q_len}), got {tuple(flow_gate_mask.shape)}."
            )
        gate_mask = flow_gate_mask.to(device=device, dtype=torch.float32)
        if not torch.is_tensor(flow_gate):
            raise TypeError(f"flow_gate must be a torch.Tensor, got {type(flow_gate)}.")
        if flow_gate.dim() != 1 or flow_gate.numel() != self.num_heads:
            raise ValueError(
                f"flow_gate must have shape ({self.num_heads},), got {tuple(flow_gate.shape)}."
            )
        gate = flow_gate.to(device=device, dtype=torch.float32)
        return {"scores": scores, "gate": gate, "mask": gate_mask}

    def forward(
        self,
        x: torch.Tensor,
        pos=None,
        dist_bias=None,
        cell_attn_bias=None,
        attn_mask=None,
        apply_ccc_bias: bool = False,
        output_attentions: bool = False,
        return_pair_rep: bool = False,
        flow_scores=None,
        flow_gate_mask=None,
        flow_gate=None,
    ):
        squeeze_batch = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_batch = True
        elif x.dim() != 3:
            raise ValueError(f"Expected x rank 2 or 3, got shape {tuple(x.shape)}")

        bsz, q_len, _ = x.shape
        query_plus_gate = self.q_proj(x)
        query_states, gate_score = torch.split(
            query_plus_gate,
            [self.hidden_size, self.num_heads],
            dim=-1,
        )
        key_states = self.k_proj(x)
        value_states = self.v_proj(x)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        sin_q, cos_q = self.relpos(q_len, device=x.device, dtype=query_states.dtype)
        query_states = theta_shift(query_states, sin_q, cos_q)
        key_states = theta_shift(key_states, sin_q, cos_q)

        attn_bias_4d = None
        dist_matrix = self._normalize_dist_bias(
            dist_bias,
            bsz,
            q_len,
            q_len,
            x.device,
            query_states.dtype,
        )
        if dist_matrix is None:
            pos_norm = self._normalize_pos(pos, bsz, q_len, q_len, x.device, x.dtype)
            if pos_norm is not None:
                valid_batch = _valid_coord_mask_tensor(pos_norm).any(dim=1)
                if bool(valid_batch.any().item()):
                    p = 1 if self.spatial_dist_metric == "l1" else 2
                    # Match the script-side switchable L1/L2 distance metric in the internal fallback path.
                    dist_matrix = torch.cdist(pos_norm, pos_norm, p=p)
                    dist_matrix = dist_matrix * valid_batch.view(bsz, 1, 1).to(dist_matrix.dtype)

        if dist_matrix is not None:
            beta = F.softplus(self.beta).view(1, self.num_heads, 1, 1)
            attn_bias_4d = -beta * dist_matrix.unsqueeze(1)

        if attn_mask is not None:
            mask_bias = _build_attn_bias(
                attn_mask,
                bsz,
                q_len,
                q_len,
                query_states.device,
                query_states.dtype,
            )
            mask_bias = self._prepare_self_bias(
                mask_bias,
                batch_size=bsz,
                q_len=q_len,
                k_len=q_len,
                device=query_states.device,
                dtype=query_states.dtype,
                name="cell_self_attn_mask_bias",
            )
            attn_bias_4d = mask_bias if attn_bias_4d is None else (attn_bias_4d + mask_bias)
        # Fixed spatial/mask bias is a prior. CCC bias is added after this detach
        # so an optional trainable CCC scale can still receive gradients.
        if attn_bias_4d is not None:
            attn_bias_4d = attn_bias_4d.detach()
        if apply_ccc_bias and cell_attn_bias is not None:
            ccc_bias_4d = self._prepare_self_bias(
                cell_attn_bias,
                batch_size=bsz,
                q_len=q_len,
                k_len=q_len,
                device=query_states.device,
                dtype=query_states.dtype,
                name="cell_ccc_bias",
            )
            attn_bias_4d = ccc_bias_4d if attn_bias_4d is None else (attn_bias_4d + ccc_bias_4d)
        flow_inputs = self._prepare_flow_scores(
            flow_scores=flow_scores,
            flow_gate_mask=flow_gate_mask,
            flow_gate=flow_gate,
            q_len=q_len,
            device=query_states.device,
        )
        attn_scores = None
        if flow_inputs is None:
            attn_output = self._run_backend_attention(
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                attn_bias_4d=attn_bias_4d,
            )
        else:
            # Stage-4 flow-directed mixing: one attention operation over the
            # mixed scores; the value path is the original projected V.
            base_scores = torch.matmul(query_states, key_states.transpose(-1, -2)) * self.scale
            if attn_bias_4d is not None:
                base_scores = base_scores + attn_bias_4d
            attn_scores = mix_flow_attention_scores(
                base_scores=base_scores,
                flow_scores=flow_inputs["scores"],
                flow_gate=flow_inputs["gate"],
                flow_gate_mask=flow_inputs["mask"],
            )
            attn_weights_32 = F.softmax(attn_scores, dim=-1)
            attn_weights_32 = self.attn_dropout(attn_weights_32)
            attn_output = torch.matmul(
                attn_weights_32.to(value_states.dtype),
                value_states,
            )
            attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
            # pair_rep feeds the fp16 pair head; match the base-path dtype.
            attn_scores = attn_scores.to(value_states.dtype)

        pair_rep = None
        attn_weights = None
        if (output_attentions or return_pair_rep) and attn_scores is None:
            attn_scores = torch.matmul(query_states, key_states.transpose(-1, -2)) * self.scale
            if attn_bias_4d is not None:
                attn_scores = attn_scores + attn_bias_4d
        if return_pair_rep:
            pair_rep = torch.clone(attn_scores)
            pair_rep = torch.nan_to_num(pair_rep, nan=0.0, posinf=0.0, neginf=0.0)
            pair_rep = pair_rep.permute(0, 2, 3, 1).contiguous()
        if output_attentions:
            attn_weights = F.softmax(attn_scores.float(), dim=-1).to(attn_scores.dtype)
            attn_weights = torch.nan_to_num(attn_weights)
            attn_weights = self.attn_dropout(attn_weights)

        if self.cell_gate_mode == "full":
            gate = torch.sigmoid(gate_score).unsqueeze(-1)
            attn_output = attn_output * gate

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        attn_output = self.proj_dropout(attn_output)

        if squeeze_batch:
            attn_output = attn_output.squeeze(0)
            if output_attentions:
                attn_weights = attn_weights.squeeze(0)
            if return_pair_rep and pair_rep is not None:
                pair_rep = pair_rep.squeeze(0)

        if output_attentions and return_pair_rep:
            return attn_output, attn_weights, pair_rep
        if output_attentions:
            return attn_output, attn_weights
        if return_pair_rep:
            return attn_output, pair_rep
        return attn_output


class CellAttentionLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        activation: str = "gelu",
        norm: str = "layernorm",
        norm_first: bool = True,
        cell_self_attn_backend: str = "flash",
        cell_gate_mode: str = "full",
        ffn_mult: int = 2,
    ):
        super().__init__()
        self.self_attn = CellSelfAttention(
            hidden_size=embed_dim,
            num_heads=num_heads,
            attn_dropout=dropout,
            proj_dropout=dropout,
            cell_self_attn_backend=cell_self_attn_backend,
            cell_gate_mode=cell_gate_mode,
        )
        self._ff_block = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * ffn_mult),
            create_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * ffn_mult, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = create_norm(norm, embed_dim)
        self.norm2 = create_norm(norm, embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.norm_first = norm_first

    def forward(
        self,
        x: torch.Tensor,
        pos=None,
        dist_bias=None,
        cell_attn_bias=None,
        attn_mask=None,
        apply_ccc_bias: bool = False,
        output_attentions: bool = False,
        return_pair_rep: bool = False,
        flow_scores=None,
        flow_gate_mask=None,
        flow_gate=None,
    ):
        if self.norm_first:
            attn_in = _apply_norm(self.norm1, x)
            attn_out = self.self_attn(
                attn_in,
                pos=pos,
                dist_bias=dist_bias,
                cell_attn_bias=cell_attn_bias,
                attn_mask=attn_mask,
                apply_ccc_bias=apply_ccc_bias,
                output_attentions=output_attentions,
                return_pair_rep=return_pair_rep,
                flow_scores=flow_scores,
                flow_gate_mask=flow_gate_mask,
                flow_gate=flow_gate,
            )
            if output_attentions and return_pair_rep:
                attn_out, attn, pair_rep = attn_out
            elif output_attentions:
                attn_out, attn = attn_out
                pair_rep = None
            elif return_pair_rep:
                attn_out, pair_rep = attn_out
                attn = None
            else:
                attn = None
                pair_rep = None
            x = x + self.dropout1(attn_out)
            x = x + self._ff_block(_apply_norm(self.norm2, x))
        else:
            attn_out = self.self_attn(
                x,
                pos=pos,
                dist_bias=dist_bias,
                cell_attn_bias=cell_attn_bias,
                attn_mask=attn_mask,
                apply_ccc_bias=apply_ccc_bias,
                output_attentions=output_attentions,
                return_pair_rep=return_pair_rep,
                flow_scores=flow_scores,
                flow_gate_mask=flow_gate_mask,
                flow_gate=flow_gate,
            )
            if output_attentions and return_pair_rep:
                attn_out, attn, pair_rep = attn_out
            elif output_attentions:
                attn_out, attn = attn_out
                pair_rep = None
            elif return_pair_rep:
                attn_out, pair_rep = attn_out
                attn = None
            else:
                attn = None
                pair_rep = None
            x = _apply_norm(self.norm1, x + self.dropout1(attn_out))
            x = _apply_norm(self.norm2, x + self._ff_block(x))

        if output_attentions and return_pair_rep:
            return x, attn, pair_rep
        if output_attentions:
            return x, attn
        if return_pair_rep:
            return x, pair_rep
        return x


def _resolve_dist_bias(x_dict, dist_bias=None):
    if dist_bias is not None:
        return dist_bias
    return x_dict.get("dist_bias", None)


def _resolve_pair_labels(x_dict, pair_labels=None):
    if pair_labels is not None:
        return pair_labels
    return x_dict.get("pair_labels", None)

class _AdapterAttention(nn.Module):
    def __init__(self, d=1024, heads=4, dropout=0.1, backend="flash"):
        super().__init__()
        if d % heads != 0:
            raise ValueError(f"d ({d}) must be divisible by heads ({heads}).")
        self.heads = heads
        self.head_dim = d // heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.dropout = dropout
        self.backend = backend
        self.q_proj = nn.Linear(d, d, bias=True)
        self.k_proj = nn.Linear(d, d, bias=True)
        self.v_proj = nn.Linear(d, d, bias=True)
        self.out_proj = nn.Linear(d, d, bias=True)

    def _prepare_attn_bias(
        self,
        bias: Optional[torch.Tensor],
        batch: int,
        q_len: int,
        k_len: int,
        device: torch.device,
        dtype: torch.dtype,
        name: str,
    ) -> Optional[torch.Tensor]:
        if bias is None:
            return None
        if not torch.is_tensor(bias):
            raise TypeError(f"{name} must be a torch.Tensor or None, got {type(bias)}.")
        if bias.dtype == torch.bool:
            raise TypeError(f"{name} must be an additive float bias, not bool mask.")
        if bias.dim() == 2:
            bias = bias.unsqueeze(0).unsqueeze(0)
        elif bias.dim() == 3:
            bias = bias.unsqueeze(1)
        elif bias.dim() != 4:
            raise ValueError(
                f"{name} must have dim 2/3/4, got shape {tuple(bias.shape)}."
            )
        expected = (batch, self.heads, q_len, k_len)
        for idx, (dim, exp) in enumerate(zip(bias.shape, expected)):
            if dim not in (1, exp):
                raise ValueError(
                    f"{name} shape {tuple(bias.shape)} is not broadcastable to "
                    f"(B,H,Q,K)={expected} at dim {idx}."
                )
        bias = bias.to(device=device, dtype=dtype)
        return bias.expand(expected)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        name: str = "attn_bias",
    ) -> torch.Tensor:
        batch, q_len, dim = query.shape
        _, k_len, _ = key.shape
        q = self.q_proj(query).view(batch, q_len, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch, k_len, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch, k_len, self.heads, self.head_dim).transpose(1, 2)

        attn_bias = self._prepare_attn_bias(
            bias, batch, q_len, k_len, q.device, q.dtype, name
        )
        if key_padding_mask is not None:
            if key_padding_mask.dim() != 2 or key_padding_mask.shape != (batch, k_len):
                raise ValueError(
                    f"key_padding_mask must be (B,K)=({batch},{k_len}), got {tuple(key_padding_mask.shape)}."
                )
            padding_bias = torch.zeros((batch, 1, 1, k_len), dtype=q.dtype, device=q.device)
            padding_bias = padding_bias.masked_fill(key_padding_mask[:, None, None, :], -1e4)
            attn_bias = padding_bias if attn_bias is None else (attn_bias + padding_bias)

        # MOD: SDPA variant. Both "flash" and "normal" backends use torch SDPA
        # so bias handling stays identical and easy to modify.
        if self.backend in {"flash", "normal"}:
            context = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_bias,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )
        else:
            raise ValueError(f"Unsupported attention backend: {self.backend}")

        context = context.transpose(1, 2).contiguous().view(batch, q_len, dim)
        return self.out_proj(context)


# MOD: simple local SDPA FFN used only by the standalone simple adapter.
class _simpleFeedForward(nn.Module):
    def __init__(self, d: int, dropout: float, mult: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d * mult, d),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# MOD: simple adapter block kept independent from the deleted SDPA adapter.
# path here follows the simpler_flash Block order more closely:
# optional cross-attn -> self-attn -> FFN, with an explicit residual stream.
class _simpleTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float,
        cross_attn: bool = False,
        cross_dim: Optional[int] = None,
        backend: str = "flash",
        ffn_mult: int = 2,
    ):
        super().__init__()
        if cross_dim is not None and cross_dim != d_model:
            raise ValueError(
                f"_simpleTransformerBlock expects cross_dim == d_model ({d_model}), got {cross_dim}."
            )
        self.cross_attn_enabled = cross_attn
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model) if cross_attn else None
        self.norm_kv = nn.LayerNorm(d_model) if cross_attn else None
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout) if cross_attn else None
        self.self_attn = _AdapterAttention(d=d_model, heads=nhead, dropout=dropout, backend=backend)
        self.cross_attn = (
            _AdapterAttention(d=d_model, heads=nhead, dropout=dropout, backend=backend)
            if cross_attn
            else None
        )
        self.ffn = _simpleFeedForward(d=d_model, dropout=dropout, mult=ffn_mult)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        x_kv: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = self.dropout1(hidden_states) if residual is None else (residual + self.dropout1(hidden_states))
        hidden_states = self.norm1(residual)

        if self.cross_attn_enabled and x_kv is not None:
            kv = self.norm_kv(x_kv)
            hidden_states = self.cross_attn(
                query=hidden_states,
                key=kv,
                value=kv,
            )
            residual = residual + self.dropout3(hidden_states)
            hidden_states = self.norm3(residual)

        hidden_states = self.self_attn(
            query=hidden_states,
            key=hidden_states,
            value=hidden_states,
            bias=bias,
            key_padding_mask=key_padding_mask,
        )
        residual = residual + self.dropout2(hidden_states)
        hidden_states = self.ffn(self.norm2(residual))
        return hidden_states, residual


# MOD: local transformer used only by the standalone simple adapter.
class _simpleTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float,
        attn_dropout: float,
        nlayers: int = 1,
        cross_attn: bool = False,
        cross_dim: Optional[int] = None,
        attn_type: str = "flash",
        ffn_mult: int = 2,
    ):
        super().__init__()
        _ = attn_dropout
        self.cross_attn = cross_attn
        self.blocks = nn.ModuleList(
            [
                _simpleTransformerBlock(
                    d_model=d_model,
                    nhead=nhead,
                    dropout=dropout,
                    ffn_mult=ffn_mult,
                    cross_attn=cross_attn,
                    cross_dim=cross_dim,
                    backend=attn_type,
                )
                for _ in range(nlayers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        x_kv: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        bias_layer: Optional[list[int]] = None,
        mask_zeros: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        _ = kwargs
        if bias_layer is None:
            bias_layer = []
        hidden_states = x
        residual = None
        for i, block in enumerate(self.blocks):
            hidden_states, residual = block(
                hidden_states=hidden_states,
                residual=residual,
                x_kv=x_kv,
                bias=bias if i in bias_layer else None,
                key_padding_mask=mask_zeros,
            )
        residual = self.dropout(hidden_states) if residual is None else (residual + self.dropout(hidden_states))
        return self.norm(residual)


# MOD: additive adapter variant backed by _simpleTransformer. This leaves the
# original CellGeneAdapter untouched and gives you a second adapter whose cross
# blocks are closer to simpler_flash's "cross-attn -> self-attn -> MLP" order.
