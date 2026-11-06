"""Interleaved gene-module and cell encoder."""

from __future__ import annotations

import torch
from torch import nn

from .attention import _parse_layer_spec


class GCATEncoder(nn.Module):
    def __init__(
        self,
        layer_cls,
        num_hidden: int,
        nhead: int,
        num_layers: int,
        dropout: float,
        activation: str = "gelu",
        norm: str = "layernorm",
        transformer_backend: str = "flash",
        use_attn_bias: bool = False,
        gene_self_backend=None,
        cross_attn_backend=None,
        cell_self_attn_backend: str = "flash",
        cell_gate_mode: str = "full",
        spatial_dist_metric: str = "l2",
        ccc_bias_layers="none",
        flow_attn_layers="none",
        ffn_mult: int = 2,
    ):
        super().__init__()
        self.ccc_bias_layer_indices = [] if str(ccc_bias_layers).lower() in {"none", "off", "false", "0"} else _parse_layer_spec(ccc_bias_layers, num_layers)
        self.flow_attn_layer_indices = [] if str(flow_attn_layers).lower() in {"none", "off", "false", "0"} else _parse_layer_spec(flow_attn_layers, num_layers)
        self.layers = nn.ModuleList(
            [
                layer_cls(
                    d=num_hidden,
                    heads=nhead,
                    dropout=dropout,
                    transformer_backend=transformer_backend,
                    use_attn_bias=use_attn_bias,
                    gene_self_backend=gene_self_backend,
                    cross_attn_backend=cross_attn_backend,
                    activation=activation,
                    norm=norm,
                    cell_self_attn_backend=cell_self_attn_backend,
                    cell_gate_mode=cell_gate_mode,
                    spatial_dist_metric=spatial_dist_metric,
                    ffn_mult=ffn_mult,
                )
                for layer_idx in range(num_layers)
            ]
        )

    def forward(
        self,
        x_dict,
        gene_self_bias=None,
        gene2cell_bias=None,
        cell2gene_bias=None,
        gene_padding_mask=None,
        pos=None,
        dist_bias=None,
        cell_attn_bias=None,
        gene_after_cell1_residual=None,
        gene_after_cell_residuals=None,
        return_pair_rep: bool = False,
        disable_cell_transformer: bool = False,
        disable_gene_module_attention: bool = False,
        flow_scores=None,
        flow_gate_mask=None,
        flow_gate=None,
    ):
        gene_tok = x_dict["gene_emb"]
        cell_tok = x_dict["h"]
        last_pair_rep = None
        ccc_bias_layers = set(self.ccc_bias_layer_indices)
        flow_attn_layers = set(self.flow_attn_layer_indices)

        for i, layer in enumerate(self.layers):
            if i == 0:
                x_dict["base0"] = cell_tok.detach()
            layer_out = layer(
                gene_tok=gene_tok,
                cell_tok=cell_tok,
                mask_dict=x_dict,
                gene_self_bias=gene_self_bias,
                gene2cell_bias=gene2cell_bias,
                cell2gene_bias=cell2gene_bias,
                gene_padding_mask=gene_padding_mask,
                pos=pos,
                dist_bias=dist_bias,
                cell_attn_bias=cell_attn_bias if i in ccc_bias_layers else None,
                return_pair_rep=return_pair_rep,
                apply_ccc_bias=i in ccc_bias_layers,
                disable_cell_transformer=disable_cell_transformer,
                disable_gene_module_attention=disable_gene_module_attention,
                flow_scores=flow_scores if i in flow_attn_layers else None,
                flow_gate_mask=flow_gate_mask if i in flow_attn_layers else None,
                flow_gate=(
                    flow_gate[i]
                    if flow_gate is not None and i in flow_attn_layers
                    else None
                ),
            )
            if return_pair_rep:
                gene_tok, cell_tok, last_pair_rep = layer_out
            else:
                gene_tok, cell_tok = layer_out
            if i == 0:
                x_dict["base1"] = cell_tok.detach()
            residual = None
            if gene_after_cell_residuals is not None and i < len(gene_after_cell_residuals):
                residual = gene_after_cell_residuals[i]
            elif i == 0 and gene_after_cell1_residual is not None:
                residual = gene_after_cell1_residual
            if residual is not None:
                if residual.dim() != 3:
                    raise ValueError(
                        "bridge residual must have shape (B,M_prior,D), "
                        f"got {tuple(residual.shape)}."
                    )
                if residual.shape[0] != gene_tok.shape[0] or residual.shape[2] != gene_tok.shape[2]:
                    raise ValueError(
                        "bridge residual batch/hidden shape must match gene_tok: "
                        f"residual={tuple(residual.shape)}, gene_tok={tuple(gene_tok.shape)}."
                    )
                prior_count = residual.shape[1]
                if prior_count > gene_tok.shape[1]:
                    raise ValueError(
                        f"Residual prior count {prior_count} exceeds token count {gene_tok.shape[1]}."
                    )
                gene_tok = torch.cat(
                    [gene_tok[:, :prior_count, :] + residual, gene_tok[:, prior_count:, :]],
                    dim=1,
                )

        result = {"hidden": cell_tok, "gene_emb": gene_tok}
        if return_pair_rep:
            result["pair_rep"] = last_pair_rep
        return result
