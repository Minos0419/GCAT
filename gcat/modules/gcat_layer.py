"""Gene-module/cell interaction layers."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .attention import CellAttentionLayer, _simpleTransformer


class GeneModuleAttentionLayer(nn.Module):
    def __init__(
        self,
        d=1024,
        heads=4,
        dropout=0.1,
        transformer_backend="flash",
        use_attn_bias=False,
        gene_self_backend=None,
        cross_attn_backend=None,
        ffn_mult=2,
    ):
        super().__init__()
        self.heads = heads
        self.ln_g = nn.LayerNorm(d)
        self.ln_c = nn.LayerNorm(d)
        self.dp = nn.Dropout(dropout)
        self.use_attn_bias = use_attn_bias
        main_backend = transformer_backend
        gene_backend = gene_self_backend if gene_self_backend is not None else main_backend
        self.enable_cross_attn = cross_attn_backend is not None
        cross_backend = cross_attn_backend
        valid_backends = {"flash", "normal"}
        for backend_name, backend_value in [
            ("transformer_backend", main_backend),
            ("gene_self_backend", gene_backend),
        ]:
            if backend_value not in valid_backends:
                raise ValueError(
                    f"Unsupported {backend_name}='{backend_value}'. "
                    f"Supported backends: {sorted(valid_backends)}"
                )
        if cross_backend is not None and cross_backend not in valid_backends:
            raise ValueError(
                f"Unsupported cross_attn_backend='{cross_backend}'. "
                f"Supported backends: {sorted(valid_backends)}"
            )
        backend_map = {
            "flash": "flash",
            "normal": "normal",
        }
        self.gene_module_attn_layer = _simpleTransformer(
            d_model=d,
            nhead=heads,
            dropout=dropout,
            attn_dropout=dropout,
            nlayers=1,
            cross_attn=False,
            attn_type=backend_map[gene_backend],
            ffn_mult=ffn_mult,
        )
        if self.enable_cross_attn:
            self.gene_xattn_from_cell = _simpleTransformer(
                d_model=d,
                nhead=heads,
                dropout=dropout,
                attn_dropout=dropout,
                nlayers=1,
                cross_attn=True,
                cross_dim=d,
                attn_type=backend_map[cross_backend],
            )
            self.cell_xattn_from_gene = _simpleTransformer(
                d_model=d,
                nhead=heads,
                dropout=dropout,
                attn_dropout=dropout,
                nlayers=1,
                cross_attn=True,
                cross_dim=d,
                attn_type=backend_map[cross_backend],
            )
        else:
            self.gene_xattn_from_cell = None
            self.cell_xattn_from_gene = None

    def _build_gene_padding_mask(
        self,
        gene_tok: torch.Tensor,
        mask_dict: Optional[dict] = None,
        gene_padding_mask: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        batch, gene_len, _ = gene_tok.shape
        if gene_padding_mask is not None:
            if gene_padding_mask.dim() == 1:
                gene_padding_mask = gene_padding_mask.unsqueeze(0).expand(batch, gene_len)
            if gene_padding_mask.shape != (batch, gene_len):
                raise ValueError(
                    f"gene_padding_mask must be (B,G)=({batch},{gene_len}), got {tuple(gene_padding_mask.shape)}."
                )
            return gene_padding_mask.bool()

        if mask_dict is None or "input_mask" not in mask_dict:
            return None
        input_mask = mask_dict["input_mask"]
        if input_mask.shape != (batch, gene_len):
            raise ValueError(
                f"mask_dict['input_mask'] must be (B,G)=({batch},{gene_len}), got {tuple(input_mask.shape)}."
            )
        mask_type = torch.zeros_like(input_mask, dtype=torch.long)
        mask_type[input_mask.bool()] = 1

        if "gene_mask" in mask_dict:
            gene_mask = mask_dict["gene_mask"]
            if gene_mask.dim() == 1:
                gene_mask = gene_mask.unsqueeze(0).expand(batch, gene_len)
            if gene_mask.shape != (batch, gene_len):
                raise ValueError(
                    f"mask_dict['gene_mask'] must be broadcastable to (B,G)=({batch},{gene_len}), got {tuple(gene_mask.shape)}."
                )
            mask_type[gene_mask.logical_not()] = 2
        return mask_type == 2

    def _prepare_gene_self_bias(
        self,
        bias: Optional[torch.Tensor],
        batch_size: int,
        seqlen: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if bias is None:
            return None
        if not torch.is_tensor(bias):
            bias = torch.as_tensor(bias, device=device, dtype=dtype)
        else:
            bias = bias.to(device=device, dtype=dtype)

        if bias.dim() == 2:
            if tuple(bias.shape) == (seqlen, seqlen):
                bias = bias.unsqueeze(0)
            elif tuple(bias.shape) == (1, seqlen):
                bias = bias.unsqueeze(0)
            else:
                raise ValueError(
                    f"gene_self_bias 2D shape must be ({seqlen},{seqlen}) or (1,{seqlen}), got {tuple(bias.shape)}."
                )
        elif bias.dim() == 3:
            if bias.shape[0] not in (1, batch_size):
                raise ValueError(
                    f"gene_self_bias 3D batch dim must be 1 or {batch_size}, got {bias.shape[0]}."
                )
            if tuple(bias.shape[1:]) not in ((seqlen, seqlen), (1, seqlen)):
                raise ValueError(
                    f"gene_self_bias 3D trailing dims must be ({seqlen},{seqlen}) or (1,{seqlen}), got {tuple(bias.shape[1:])}."
                )
        elif bias.dim() == 4:
            if bias.shape[0] not in (1, batch_size):
                raise ValueError(
                    f"gene_self_bias 4D batch dim must be 1 or {batch_size}, got {bias.shape[0]}."
                )
            if bias.shape[1] not in (1, self.heads):
                raise ValueError(
                    f"gene_self_bias 4D head dim must be 1 or {self.heads}, got {bias.shape[1]}."
                )
            if bias.shape[-1] != seqlen or bias.shape[-2] not in (1, seqlen):
                raise ValueError(
                    f"gene_self_bias 4D last dims must be (1,{seqlen}) or ({seqlen},{seqlen}), got {tuple(bias.shape[-2:])}."
                )
            bias = bias[:, 0]
        else:
            raise ValueError(
                f"gene_self_bias must have dim 2/3/4, got shape {tuple(bias.shape)}."
            )

        if bias.shape[0] not in (1, batch_size):
            raise ValueError(
                f"gene_self_bias batch dim must be 1 or {batch_size}, got {bias.shape[0]}."
            )
        if bias.shape[-1] != seqlen or bias.shape[-2] not in (1, seqlen):
            raise ValueError(
                f"gene_self_bias last dims must be (1,{seqlen}) or ({seqlen},{seqlen}), got {tuple(bias.shape[-2:])}."
            )
        return bias.contiguous()

    def forward(
        self,
        gene_tok,
        cell_tok,
        mask_dict=None,
        gene_self_bias=None,
        gene2cell_bias=None,
        cell2gene_bias=None,
        gene_padding_mask=None,
        **kwargs,
    ):
        key_padding = self._build_gene_padding_mask(
            gene_tok=gene_tok,
            mask_dict=mask_dict,
            gene_padding_mask=gene_padding_mask,
        )
        if cell_tok.dim() == 2:
            cell_tok = cell_tok.unsqueeze(1)
        if cell_tok.dim() != 3 or cell_tok.shape[1] != 1:
            raise ValueError(
                f"cell_tok must be (B,D) or (B,1,D), got {tuple(cell_tok.shape)}."
            )

        x = self.ln_g(gene_tok)
        gene_bias = gene_self_bias if self.use_attn_bias else None
        if gene_bias is not None:
            gene_bias = self._prepare_gene_self_bias(
                gene_bias,
                batch_size=x.shape[0],
                seqlen=x.shape[1],
                device=x.device,
                dtype=x.dtype,
            )
            gene_delta = self.gene_module_attn_layer(
                x,
                bias=gene_bias,
                bias_layer=[0],
            ) - x
        else:
            gene_delta = self.gene_module_attn_layer(x) - x
        if key_padding is not None:
            gene_delta = gene_delta.masked_fill(key_padding.unsqueeze(-1), 0.0)
        gene_tok = gene_tok + self.dp(gene_delta)

        if self.enable_cross_attn:
            x = self.ln_g(gene_tok)
            kv = self.ln_c(cell_tok)
            gene_delta_2 = self.gene_xattn_from_cell(x, x_kv=kv) - x
            if key_padding is not None:
                gene_delta_2 = gene_delta_2.masked_fill(key_padding.unsqueeze(-1), 0.0)
            gene_tok = gene_tok + self.dp(gene_delta_2)

            x = self.ln_c(cell_tok)
            kv = self.ln_g(gene_tok)
            cell_delta = self.cell_xattn_from_gene(x, x_kv=kv) - x
            cell_tok = cell_tok + self.dp(cell_delta)

        return gene_tok, cell_tok.squeeze(1)



class GCATLayer(GeneModuleAttentionLayer):
    def __init__(
        self,
        d=1024,
        heads=4,
        dropout=0.1,
        transformer_backend="flash",
        use_attn_bias=False,
        gene_self_backend=None,
        cross_attn_backend=None,
        activation="gelu",
        norm="layernorm",
        cell_self_attn_backend="flash",
        cell_gate_mode="full",
        spatial_dist_metric="l2",
        ffn_mult=2,
    ):
        super().__init__(
            d=d,
            heads=heads,
            dropout=dropout,
            transformer_backend=transformer_backend,
            use_attn_bias=use_attn_bias,
            gene_self_backend=gene_self_backend,
            cross_attn_backend=cross_attn_backend,
            ffn_mult=ffn_mult,
        )
        self.cell_attn_layer = CellAttentionLayer(
            embed_dim=d,
            num_heads=heads,
            dropout=dropout,
            activation=activation,
            norm=norm,
            cell_self_attn_backend=cell_self_attn_backend,
            cell_gate_mode=cell_gate_mode,
            ffn_mult=ffn_mult,
        )
        self.cell_attn_layer.self_attn.spatial_dist_metric = spatial_dist_metric

    def _prepare_gene_self_bias_with_cls(
        self,
        gene_self_bias: Optional[torch.Tensor],
        batch_size: int,
        gene_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if gene_self_bias is None or not self.use_attn_bias:
            return None
        gene_bias = self._prepare_gene_self_bias(
            gene_self_bias,
            batch_size=batch_size,
            seqlen=gene_len,
            device=device,
            dtype=dtype,
        )
        if gene_bias is None:
            return None
        if gene_bias.shape[-2] == 1:
            cls_bias = gene_bias.new_zeros(gene_bias.shape[:-1] + (gene_len + 1,))
            cls_bias[..., 1:] = gene_bias
            return cls_bias.contiguous()

        cls_bias = gene_bias.new_zeros(gene_bias.shape[:-2] + (gene_len + 1, gene_len + 1))
        cls_bias[..., 1:, 1:] = gene_bias
        return cls_bias.contiguous()

    def forward(
        self,
        gene_tok,
        cell_tok,
        mask_dict=None,
        gene_self_bias=None,
        gene2cell_bias=None,
        cell2gene_bias=None,
        gene_padding_mask=None,
        pos=None,
        dist_bias=None,
        cell_attn_bias=None,
        return_pair_rep: bool = False,
        apply_ccc_bias: bool = False,
        disable_cell_transformer: bool = False,
        disable_gene_module_attention: bool = False,
        flow_scores=None,
        flow_gate_mask=None,
        flow_gate=None,
        **kwargs,
    ):
        _ = gene2cell_bias, cell2gene_bias, kwargs
        key_padding = self._build_gene_padding_mask(
            gene_tok=gene_tok,
            mask_dict=mask_dict,
            gene_padding_mask=gene_padding_mask,
        )
        if cell_tok.dim() == 2:
            cell_tok = cell_tok.unsqueeze(1)
        if cell_tok.dim() != 3 or cell_tok.shape[1] != 1:
            raise ValueError(
                f"cell_tok must be (B,D) or (B,1,D), got {tuple(cell_tok.shape)}."
            )

        combined_tok = torch.cat([cell_tok, gene_tok], dim=1)
        combined_norm = torch.cat([self.ln_c(cell_tok), self.ln_g(gene_tok)], dim=1)
        combined_padding = None
        if key_padding is not None:
            combined_padding = torch.cat(
                [torch.zeros_like(key_padding[:, :1], dtype=torch.bool), key_padding],
                dim=1,
            )
        cls_gene_bias = self._prepare_gene_self_bias_with_cls(
            gene_self_bias=gene_self_bias,
            batch_size=combined_norm.shape[0],
            gene_len=gene_tok.shape[1],
            device=combined_norm.device,
            dtype=combined_norm.dtype,
        )
        if disable_gene_module_attention:
            combined_delta = torch.zeros_like(combined_norm)
        elif cls_gene_bias is not None:
            combined_delta = self.gene_module_attn_layer(
                combined_norm,
                bias=cls_gene_bias,
                bias_layer=[0],
                mask_zeros=combined_padding,
            ) - combined_norm
        else:
            combined_delta = self.gene_module_attn_layer(
                combined_norm,
                mask_zeros=combined_padding,
            ) - combined_norm
        if key_padding is not None:
            gene_delta = combined_delta[:, 1:].masked_fill(key_padding.unsqueeze(-1), 0.0)
            combined_delta = torch.cat([combined_delta[:, :1], gene_delta], dim=1)
        combined_tok = combined_tok + self.dp(combined_delta)

        cls_tok = combined_tok[:, :1]
        gene_tok = combined_tok[:, 1:]
        if disable_cell_transformer:
            cell_out = cls_tok.squeeze(1)
            if return_pair_rep:
                return gene_tok, cell_out, None
            return gene_tok, cell_out
        cell_out = self.cell_attn_layer(
            cls_tok.squeeze(1),
            pos=pos,
            dist_bias=dist_bias,
            cell_attn_bias=cell_attn_bias,
            return_pair_rep=return_pair_rep,
            apply_ccc_bias=apply_ccc_bias,
            flow_scores=flow_scores,
            flow_gate_mask=flow_gate_mask,
            flow_gate=flow_gate,
        )
        if return_pair_rep:
            cell_out, pair_rep = cell_out
            return gene_tok, cell_out, pair_rep
        return gene_tok, cell_out
