"""Active GCAT model assembled from the focused modules in :mod:`gcat.modules`."""

from __future__ import annotations

import json
import os
from typing import Optional, Union

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from gcat.modules.assignment import entmax15_exact
from gcat.modules.attention import (
    _parse_layer_spec,
    _resolve_dist_bias,
    _resolve_pair_labels,
    _resolve_spatial_pos,
)
from gcat.modules.ccc import prepare_cell_ccc_bias
from gcat.modules.decoder import NegativeBinomialDecoder
from gcat.modules.embedding import GCATEmbedding
from gcat.modules.encoder import GCATEncoder
from gcat.modules.gcat_layer import GCATLayer
from gcat.modules.loss import (
    NegativeBinomialLoss,
    masked_gene_standardized_mse,
    regulon_activity_auxiliary_loss,
)
from gcat.modules.masking import MaskBuilder, _select_dropnode_rows, _select_dropnode_square
from gcat.modules.vae import VAELatentLayer

class GCATFormer(nn.Module):
    def __init__(
        self,
        gene_list,
        hidden_dim,
        num_layers,
        latent_dim,
        decoder_hidden_dim,
        decoder_layers,
        output_dim,
        batch_count=0,
        dataset_count=0,
        platform_count=0,
        modality_count=0,
        dropout=0.1,
        activation="gelu",
        normalization="layernorm",
        num_heads=8,
        mask_node_rate=0.5,
        mask_feature_rate=0.8,
        drop_node_rate=0.25,
        max_batch_size=2000,
        kl_threshold=0.5,
        beta_masking=False,
        st_block_gene_mask_rate=0.0,
        pair_loss_weight=0.1,
        pair_mask_token=-100.0,
        distance_metric="l2",
        gene_token_mode: str = "module",
        gene_module_num: Union[int, str] = 64,
        gene_module_init_path: Optional[str] = None,
        dynamic_gene_modules: bool = False,
        prior_strength: float = 3.0,
        off_prior_init_std: float = 0.02,
        assignment_temperature: float = 1.0,
        use_ccc_bias: bool = True,
        ccc_bias_layers: str = "last",
        ccc_bias_trainable: bool = True,
        ccc_bias_init_scale: float = 1.0,
        ccc_program_num: int = 0,
        lambda_reg: float = 0.05,
        reg_activity_std_floor: float = 0.1,
        lambda_masked_zmse: float = 0.0,
        flow_cell_attention: bool = False,
        flow_attn_layers: str = "last",
        flow_gate_hidden_dim: int = 64,
        flow_attn_mode: str = "gated",
        ffn_mult: int = 2,
        flow_gate_init: float = 0.0,
        disable_cell_transformer: bool = False,
        disable_gene_module_attention: bool = False,
        **_legacy_kwargs,
    ):
        # GCAT_v2: legacy experiment kwargs (lambda_out, flow_linear_bottleneck,
        # flow_stage5, grn_*, lambda_pull, niche_*, ccc_program_head_bias, ...) are
        # accepted but ignored; those branches were removed from this model.
        super().__init__()

        self.embedder = GCATEmbedding(gene_list, hidden_dim, dropout)
        self.gene_list = list(gene_list)
        self.gene_set = set(gene_list)
        self._gene_to_bias_idx = {gene: idx for idx, gene in enumerate(gene_list)}
        self.gene_token_mode = str(gene_token_mode).lower()
        if self.gene_token_mode not in {"module", "full"}:
            raise ValueError(
                "gene_token_mode must be 'module' or 'full', "
                f"got {self.gene_token_mode!r}."
            )
        init_matrix = None
        init_summary = None
        if gene_module_init_path:
            init_path_str = os.fspath(gene_module_init_path)
            init_matrix = np.load(init_path_str)
            summary_path = os.path.splitext(init_path_str)[0] + "_summary.json"
            if os.path.exists(summary_path):
                try:
                    with open(summary_path, "r", encoding="utf-8") as fp:
                        init_summary = json.load(fp)
                except Exception as exc:
                    print(f"Warning: failed to read gene module summary {summary_path}: {exc}", flush=True)
            if init_matrix.ndim != 2:
                raise ValueError(
                    f"gene_module_init_path must be a 2D gene x module matrix, got shape {tuple(init_matrix.shape)}."
                )
            expected_rows = len(gene_list)
            if init_matrix.shape[0] != expected_rows:
                raise ValueError(
                    f"gene_module_init_path first dim must be {expected_rows}, got {tuple(init_matrix.shape)}."
                )
        self.gene_module_init_summary = init_summary
        self.dynamic_gene_modules = bool(dynamic_gene_modules)
        self.prior_strength = float(prior_strength)
        self.off_prior_init_std = float(off_prior_init_std)
        self.assignment_temperature = float(assignment_temperature)
        self.disable_cell_transformer = bool(disable_cell_transformer)
        self.disable_gene_module_attention = bool(disable_gene_module_attention)
        if self.dynamic_gene_modules and init_matrix is None and self.gene_token_mode == "module":
            raise ValueError("dynamic_gene_modules requires gene_module_init_path in module mode.")
        if self.prior_strength < 0:
            raise ValueError("prior_strength must be >= 0.")
        if self.off_prior_init_std < 0:
            raise ValueError("off_prior_init_std must be >= 0.")
        if self.assignment_temperature <= 0:
            raise ValueError("assignment_temperature must be > 0.")
        gene_module_num_raw = gene_module_num
        if isinstance(gene_module_num_raw, str) and gene_module_num_raw.lower() == "auto":
            if init_matrix is None:
                if self.gene_token_mode == "full":
                    self.gene_module_num = 1
                else:
                    raise ValueError("gene_module_num='auto' requires gene_module_init_path when gene_token_mode='module'.")
            else:
                self.gene_module_num = int(init_matrix.shape[1])
        else:
            self.gene_module_num = int(gene_module_num_raw)
            if init_matrix is not None and init_matrix.shape[1] != self.gene_module_num:
                raise ValueError(
                    "gene_module_init_path second dim must match gene_module_num, "
                    f"or set gene_module_num='auto'. got init shape {tuple(init_matrix.shape)} "
                    f"and gene_module_num={self.gene_module_num}."
                )
        if self.gene_module_num <= 0:
            raise ValueError(f"gene_module_num must be > 0, got {self.gene_module_num}.")
        self.prior_gene_module_num = int(
            init_summary.get("prior_module_count", self.gene_module_num)
            if isinstance(init_summary, dict)
            else self.gene_module_num
        )
        self.de_novo_gene_module_num = int(
            init_summary.get("de_novo_module_count", 0)
            if isinstance(init_summary, dict)
            else 0
        )
        if self.prior_gene_module_num + self.de_novo_gene_module_num != self.gene_module_num:
            raise ValueError(
                "Module summary prior/de_novo counts must sum to gene_module_num: "
                f"{self.prior_gene_module_num}+{self.de_novo_gene_module_num}!={self.gene_module_num}."
            )
        self.gene_module_backend = "expression_first"
        self.gene_module_assignment = "entmax15"
        if init_matrix is not None:
            prior_slice = init_matrix[:, : self.prior_gene_module_num]
            uncovered_np = np.asarray(np.abs(prior_slice).sum(axis=1) <= 1e-8, dtype=np.bool_)
        else:
            uncovered_np = np.zeros(len(gene_list), dtype=np.bool_)
        self.pathway_module_num = self.gene_module_num
        self.total_gene_module_num = self.pathway_module_num
        prior_mask_np = (
            np.asarray(np.abs(init_matrix) > 1e-8, dtype=np.bool_)
            if init_matrix is not None
            else np.zeros((len(gene_list), self.pathway_module_num), dtype=np.bool_)
        )
        if self.de_novo_gene_module_num > 0:
            prior_mask_np[:, self.prior_gene_module_num:] = False
        self.register_buffer("gene_module_prior_mask", torch.as_tensor(prior_mask_np, dtype=torch.bool))
        self.register_buffer("gene_module_uncovered_mask", torch.as_tensor(uncovered_np, dtype=torch.bool))
        self.gene_module_uncovered_count = int(uncovered_np.sum())
        self.gene_module_uncovered_ratio = float(self.gene_module_uncovered_count / max(1, len(gene_list)))
        self.gene_module_proj = nn.Parameter(torch.empty(len(gene_list), self.total_gene_module_num))
        nn.init.xavier_uniform_(self.gene_module_proj)
        if init_matrix is not None:
            with torch.no_grad():
                init_tensor = torch.as_tensor(init_matrix, dtype=self.gene_module_proj.dtype)
                if self.dynamic_gene_modules:
                    nn.init.normal_(
                        self.gene_module_proj[:, :self.pathway_module_num],
                        mean=0.0,
                        std=self.off_prior_init_std,
                    )
                    self.gene_module_proj[:, :self.pathway_module_num].add_(
                        init_tensor * self.prior_strength
                    )
                else:
                    self.gene_module_proj[:, :self.pathway_module_num].copy_(init_tensor)
        self.gene_module_norm = nn.LayerNorm(hidden_dim)
        self.expression_module_emb = nn.Parameter(torch.empty(self.total_gene_module_num, hidden_dim))
        nn.init.xavier_uniform_(self.expression_module_emb)
        self.expression_module_scale = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.lambda_reg = float(lambda_reg)
        if self.lambda_reg < 0:
            raise ValueError("lambda_reg must be >= 0.")
        self.reg_activity_std_floor = float(reg_activity_std_floor)
        if self.reg_activity_std_floor <= 0:
            raise ValueError("reg_activity_std_floor must be > 0.")
        # Scalar readout used only by the optional activity auxiliary loss.
        # It consumes encoder-updated module tokens rather than the raw
        # masked expression score.  Do not instantiate it when disabled so
        # lambda_reg=0 preserves the original parameter/RNG initialization.
        self.regulon_activity_readout = (
            nn.Linear(hidden_dim, 1) if self.lambda_reg > 0 else None
        )
        self.lambda_masked_zmse = float(lambda_masked_zmse)
        if self.lambda_masked_zmse < 0:
            raise ValueError("lambda_masked_zmse must be >= 0.")
        self.use_ccc_bias = bool(use_ccc_bias)
        self.ccc_bias_layers = str(ccc_bias_layers)
        self.ccc_bias_trainable = bool(ccc_bias_trainable)
        self.ccc_program_num = int(ccc_program_num)
        self.num_heads = int(num_heads)
        self.flow_cell_attention = bool(flow_cell_attention)
        self.flow_attn_layers = str(flow_attn_layers)
        self.flow_gate_hidden_dim = int(flow_gate_hidden_dim)
        if self.flow_gate_hidden_dim <= 0:
            raise ValueError("flow_gate_hidden_dim must be > 0.")
        self.flow_gate_init = float(flow_gate_init)
        self.flow_attn_mode = str(flow_attn_mode).lower()
        if self.flow_attn_mode not in {"gated", "replace"}:
            raise ValueError(
                "flow_attn_mode must be 'gated' or 'replace', "
                f"got {self.flow_attn_mode!r}."
            )
        if self.flow_attn_mode == "replace" and not self.flow_cell_attention:
            raise ValueError("flow_attn_mode requires flow_cell_attention.")
        if self.flow_cell_attention:
            # Stage 4 replaces the scalar CCC bias inside the cell attention
            # with directed flow attention over the same LR edges.  A hybrid
            # is allowed: scalar CCC on non-flow layers (ccc_bias_layers must
            # not overlap flow_attn_layers), flow on its own layers.
            if self.use_ccc_bias:
                ccc_layers = set(
                    _parse_layer_spec(self.ccc_bias_layers, num_layers)
                )
                flow_layers = set(
                    _parse_layer_spec(self.flow_attn_layers, num_layers)
                )
                if ccc_layers & flow_layers:
                    raise ValueError(
                        "flow_cell_attention with scalar CCC requires disjoint "
                        f"layer sets: ccc={sorted(ccc_layers)} flow={sorted(flow_layers)}."
                    )
            if self.gene_token_mode != "module":
                raise ValueError(
                    "flow_cell_attention requires gene_token_mode='module' for the regulon state."
                )
            if self.ccc_program_num <= 0:
                raise ValueError("ccc_program_num must be > 0 for flow_cell_attention.")
        if self.ccc_bias_trainable:
            self.ccc_bias_scale = nn.Parameter(torch.tensor(ccc_bias_init_scale, dtype=torch.float32))
        else:
            self.register_buffer("ccc_bias_scale", torch.tensor(ccc_bias_init_scale, dtype=torch.float32))
        self.mask_model = MaskBuilder(
            mask_node_rate,
            mask_feature_rate,
            drop_node_rate,
            max_batch_size,
            # Preserve the historical positional-call behavior: beta_masking
            # occupied the unused edge_mask slot, so masking stayed uniform.
            edge_mask=beta_masking,
            mask_beta=False,
            st_block_gene_mask_rate=st_block_gene_mask_rate,
        )

        self.gcat_encoder = GCATEncoder(
            layer_cls=GCATLayer,
            num_hidden=hidden_dim,
            nhead=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            norm=normalization,
            transformer_backend="flash",
            use_attn_bias=False,
            gene_self_backend=None,
            cross_attn_backend=None,
            cell_self_attn_backend="flash",
            cell_gate_mode="full",
            spatial_dist_metric=distance_metric,
            ccc_bias_layers=self.ccc_bias_layers if self.use_ccc_bias else "none",
            flow_attn_layers=self.flow_attn_layers if self.flow_cell_attention else "none",
            ffn_mult=int(ffn_mult),
        )
        self.pre_latent_norm = nn.LayerNorm(hidden_dim)
        self.latent = VAELatentLayer(
            hidden_dim,
            latent_dim,
            loss_threshold=kl_threshold,
        )
        self.decoder = NegativeBinomialDecoder(
            latent_dim,
            decoder_hidden_dim,
            output_dim,
            decoder_layers,
            dropout,
            normalization,
            batch_count=batch_count,
            dataset_count=dataset_count,
            platform_count=platform_count,
            modality_count=modality_count,
        )
        self.negative_binomial_loss = NegativeBinomialLoss()

        self.pair_loss_weight = pair_loss_weight
        self.pair_mask_token = pair_mask_token
        self.distance_metric = distance_metric
        self.pair_head = nn.Sequential(
            nn.Linear(num_heads, num_heads),
            nn.ReLU(),
            nn.Linear(num_heads, 1),
        )
        self.flow_attn_gate_in = None
        self.flow_attn_gate_out = None
        self.flow_attn_program_weight = None
        self.flow_attn_gate = None
        self._flow_stage_state = None
        if self.flow_cell_attention:
            # The Stage-4 branch is appended after all baseline modules and RNG
            # is restored, so enabling it cannot perturb any existing GCAT
            # parameter.  The per-head gate starts at zero, which makes the
            # initial forward exactly the no-CCC-bias baseline.
            cpu_rng_state = torch.random.get_rng_state()

            def _regulon_gate_mlp() -> nn.Sequential:
                return nn.Sequential(
                    nn.Linear(self.total_gene_module_num, self.flow_gate_hidden_dim),
                    nn.GELU(),
                    nn.Linear(self.flow_gate_hidden_dim, self.ccc_program_num),
                )

            self.flow_attn_gate_in = _regulon_gate_mlp()
            self.flow_attn_gate_out = _regulon_gate_mlp()
            torch.random.set_rng_state(cpu_rng_state)
            # Deterministic 1/sqrt(P) init draws no RNG; per-head program
            # weights w[h, p] keep S_flow on the base-score scale.
            self.flow_attn_program_weight = nn.Parameter(
                torch.full((self.num_heads, self.ccc_program_num), self.ccc_program_num ** -0.5)
            )
            if self.flow_attn_mode == "gated":
                # Per-layer x per-head interpolation gates.  Default zero-init
                # keeps the initial forward exactly the no-CCC baseline; a
                # nonzero flow_gate_init starts the mixture near that value.
                self.flow_attn_gate = nn.Parameter(
                    torch.full((num_layers, self.num_heads), self.flow_gate_init)
                )
            else:
                # Direct replacement: the interpolation coefficient is fixed at
                # one, so S_mix = G*S_flow + (1-G)*S_base with no learnable gate.
                del self.flow_attn_gate
                self.register_buffer(
                    "flow_attn_gate", torch.ones(num_layers, self.num_heads)
                )
    def _prepare_cell_ccc_bias(
        self,
        ccc_bias,
        cell_tok: torch.Tensor,
        program_edge_index=None,
        program_score=None,
    ) -> Optional[torch.Tensor]:
        return prepare_cell_ccc_bias(
            ccc_bias,
            cell_tok,
            enabled=self.use_ccc_bias,
            bias_scale=self.ccc_bias_scale,
            program_edge_index=program_edge_index,
            program_score=program_score,
            program_projection=None,
        )

    @staticmethod
    def _normalize_flow_activity(activity: torch.Tensor) -> torch.Tensor:
        """log1p + per-program positive-entry P95 + clamp01.

        Each LR program gets its own scale: the 95th percentile is taken over
        the positive entries of that program's column only (linear
        interpolation, matching ``torch.quantile``), so programs with very
        different expression magnitudes become comparable.  Columns with no
        positive entry stay all-zero.
        """
        positive = torch.log1p(activity.clamp_min(0.0).float())
        positive_mask = positive > 0
        if not bool(positive_mask.any().item()):
            return torch.zeros_like(positive)
        num_cells = positive.shape[0]
        # Sort each column descending with -inf padding for non-positive
        # entries, then read the 95th percentile of the positives by position
        # from the top (0.05 * (k - 1) in descending order).
        filled = positive.masked_fill(~positive_mask, float("-inf"))
        sorted_desc = filled.sort(dim=0, descending=True).values
        k = positive_mask.sum(dim=0).float()
        pos = 0.05 * (k - 1.0)
        idx_lo = pos.floor().long().clamp_(min=0, max=num_cells - 1)
        idx_hi = pos.ceil().long().clamp_(min=0, max=num_cells - 1)
        frac = (pos - idx_lo.float()).clamp_(0.0, 1.0)
        lo = sorted_desc.gather(0, idx_lo.unsqueeze(0)).squeeze(0)
        hi = sorted_desc.gather(0, idx_hi.unsqueeze(0)).squeeze(0)
        # All-zero columns gather -inf twice: (hi - lo) is NaN there.  Map any
        # non-finite scale to the floor; those columns stay all-zero anyway.
        scale = torch.nan_to_num(
            lo + frac * (hi - lo), nan=1e-6, posinf=1e-6, neginf=1e-6
        ).clamp_min(1e-6)
        return (positive / scale.unsqueeze(0)).clamp(0.0, 1.0)

    def _flow_gated_activities(
        self,
        reg_state: torch.Tensor,
        inflow: torch.Tensor,
        outflow: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Regulon-gated LR activities: I~ = I * sigmoid(MLP_in(R)), O~ = O * sigmoid(MLP_out(R))."""
        mlp_dtype = self.flow_attn_gate_in[0].weight.dtype
        gate_in = torch.sigmoid(self.flow_attn_gate_in(reg_state.to(dtype=mlp_dtype))).to(
            dtype=torch.float32
        )
        gate_out = torch.sigmoid(self.flow_attn_gate_out(reg_state.to(dtype=mlp_dtype))).to(
            dtype=torch.float32
        )
        return inflow * gate_in, outflow * gate_out

    def _prepare_flow_attention(
        self,
        x_dict: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build Stage-4 program-level flow scores and the directed G mask.

        Inflow and outflow are intrinsic per-cell receptor-/ligand-side LR
        program activities (no sender information enters the receiver
        representation).  The regulon state gates both before the program-level
        compatibility score is computed.  All inputs are detached fixed priors
        matching the CCC cache semantics; only the gating MLPs, per-head
        program weights, and gate are trainable here.
        """
        if "gene_module_activity" not in x_dict:
            raise RuntimeError(
                "flow_cell_attention requires the module tokenizer regulon activity."
            )
        for key in ("ccc_inflow_activity", "ccc_outflow_activity"):
            if key not in x_dict:
                raise KeyError(f"{key} is required for flow_cell_attention.")
        cell_tok = x_dict["h"]
        num_cells = cell_tok.shape[0]
        device = cell_tok.device
        expected = (num_cells, self.ccc_program_num)
        inflow = torch.as_tensor(x_dict["ccc_inflow_activity"], device=device, dtype=torch.float32)
        if inflow.shape != expected:
            raise ValueError(
                f"ccc_inflow_activity must have shape {expected}, got {tuple(inflow.shape)}."
            )
        outflow = torch.as_tensor(x_dict["ccc_outflow_activity"], device=device, dtype=torch.float32)
        if outflow.shape != expected:
            raise ValueError(
                f"ccc_outflow_activity must have shape {expected}, got {tuple(outflow.shape)}."
            )
        reg_state = x_dict["gene_module_activity"].to(device=device, dtype=torch.float32).detach()
        if reg_state.shape != (num_cells, self.total_gene_module_num):
            raise ValueError(
                "gene_module_activity must have shape "
                f"({num_cells},{self.total_gene_module_num}), got {tuple(reg_state.shape)}."
            )
        inflow_base = self._normalize_flow_activity(inflow)
        outflow_base = self._normalize_flow_activity(outflow)
        inflow_tilde, outflow_tilde = self._flow_gated_activities(
            reg_state, inflow_base, outflow_base
        )
        # S_flow[h, j, i] = sum_p w[h, p] * I~[j, p] * O~[i, p]:
        # receiver/query j against sender/key i, one aligned program index.
        flow_scores = torch.einsum(
            "hp,jp,ip->hji",
            self.flow_attn_program_weight.float(),
            inflow_tilde,
            outflow_tilde,
        )
        # Guard the base path: with Gamma == 0 the mixing term is
        # 0 * (S_flow - S_base), so any non-finite flow value would still
        # poison the mixed scores.  Never let one escape.
        flow_scores = torch.nan_to_num(
            flow_scores, nan=0.0, posinf=0.0, neginf=0.0
        )
        edge_index = torch.as_tensor(
            x_dict["ccc_program_edge_index"], device=device, dtype=torch.long
        )
        # G[j, i] = 1 iff a directed LR edge sender i -> receiver j exists in
        # the current (post drop-node) sparse CCC program edge set.
        gate_mask = torch.zeros(
            (num_cells, num_cells), device=device, dtype=torch.float32
        )
        if edge_index.numel():
            gate_mask[edge_index[0], edge_index[1]] = 1.0
        gate_mask.fill_diagonal_(0.0)
        return flow_scores, gate_mask

    def _resolve_gene_pos(
        self,
        x_dict: dict,
        input_gene_list,
        gene_len: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        gene_pos = None
        for key in ("gene_pos", "gene_idx", "input_gene_idx", "input_gene_index"):
            if key in x_dict and x_dict[key] is not None:
                gene_pos = x_dict[key]
                break
        if gene_pos is None and input_gene_list is not None:
            gene_pos = input_gene_list
        if gene_pos is None:
            if gene_len == len(self._gene_to_bias_idx):
                return torch.arange(gene_len, device=device, dtype=torch.long)
            return None

        if torch.is_tensor(gene_pos):
            gene_pos = gene_pos.to(device=device)
        elif isinstance(gene_pos, np.ndarray):
            if np.issubdtype(gene_pos.dtype, np.integer):
                gene_pos = torch.as_tensor(gene_pos, device=device)
            else:
                gene_pos = gene_pos.tolist()
                mapped = [self._gene_to_bias_idx.get(g, -1) for g in gene_pos]
                if any(idx < 0 for idx in mapped):
                    missing = [g for g in gene_pos if g not in self._gene_to_bias_idx]
                    raise KeyError(
                        f"Found genes not present in model gene_list while building attn bias: {missing[:5]}"
                    )
                gene_pos = torch.as_tensor(mapped, device=device)
        elif isinstance(gene_pos, (list, tuple)):
            if len(gene_pos) == 0:
                return None
            first = gene_pos[0]
            if isinstance(first, (str, np.str_)):
                mapped = [self._gene_to_bias_idx.get(g, -1) for g in gene_pos]
                if any(idx < 0 for idx in mapped):
                    missing = [g for g in gene_pos if g not in self._gene_to_bias_idx]
                    raise KeyError(
                        f"Found genes not present in model gene_list while building attn bias: {missing[:5]}"
                    )
                gene_pos = torch.as_tensor(mapped, device=device)
            else:
                gene_pos = torch.as_tensor(gene_pos, device=device)
        else:
            return None

        if gene_pos.dim() == 1:
            if gene_pos.numel() != gene_len:
                raise ValueError(
                    f"gene_pos length must equal gene token length ({gene_len}), got {gene_pos.numel()}."
                )
        elif gene_pos.dim() == 2:
            if gene_pos.shape[1] != gene_len:
                raise ValueError(
                    f"gene_pos shape must end with gene token length ({gene_len}), got {tuple(gene_pos.shape)}."
                )
        else:
            raise ValueError(
                f"gene_pos must be 1D or 2D, got shape {tuple(gene_pos.shape)}."
            )
        return gene_pos.long()

    def _apply_module_assignment(self, logits: torch.Tensor) -> torch.Tensor:
        scores = (
            logits / self.assignment_temperature
            if self.dynamic_gene_modules
            else logits
        )
        return entmax15_exact(scores, dim=-1)

    def _gene_module_assignment_matrix(self, gene_pos: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        proj_table = self.gene_module_proj.to(device=gene_pos.device, dtype=dtype)
        if gene_pos.dim() == 1:
            logits = proj_table.index_select(0, gene_pos)
        else:
            logits = proj_table[gene_pos]

        return self._apply_module_assignment(logits)

    def summarize_gene_module_coverage(self, genes: list[str] | tuple[str, ...]) -> dict:
        genes = list(genes)
        if not genes:
            return {
                "gene_count": 0,
                "biological_module_count": int(self.prior_gene_module_num),
                "de_novo_module_count": int(self.de_novo_gene_module_num),
                "allocated_module_count": int(self.pathway_module_num),
                "covered_count": 0,
                "uncovered_count": 0,
                "uncovered_ratio": 0.0,
            }
        device = self.gene_module_proj.device
        gene_pos = self._resolve_gene_pos({}, genes, len(genes), device)
        with torch.no_grad():
            assignment = self._gene_module_assignment_matrix(gene_pos, dtype=torch.float32)
            biological = assignment[..., :self.prior_gene_module_num]
            bio_row_sum = biological.abs().sum(dim=-1)
            uncovered = bio_row_sum <= 1e-8
            uncovered_count = int(uncovered.sum().item())
            return {
                "gene_count": int(len(genes)),
                "biological_module_count": int(self.prior_gene_module_num),
                "de_novo_module_count": int(self.de_novo_gene_module_num),
                "allocated_module_count": int(self.pathway_module_num),
                "covered_count": int(len(genes) - uncovered_count),
                "uncovered_count": uncovered_count,
                "uncovered_ratio": float(uncovered_count / max(1, len(genes))),
            }

    def _expression_matrix_and_gene_pos(
        self,
        x_dict: dict,
        input_gene_list=None,
        device: Optional[torch.device] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = x_dict["masked_x_seq"] if "masked_x_seq" in x_dict else x_dict["x_seq"]
        if device is None:
            device = x.device
        if "dropout" in x_dict:
            indices = x._indices().t()
            values = x._values().float()
            values = torch.distributions.binomial.Binomial(values, x_dict["dropout"]).sample()
            x = torch.sparse.FloatTensor(indices.t(), values, x.shape).to(device)
        x = torch.log1p(x)
        x_dense = x.to_dense() if x.is_sparse else x
        gene_pos = self._resolve_gene_pos(
            x_dict=x_dict,
            input_gene_list=input_gene_list,
            gene_len=x_dense.shape[1],
            device=device,
        )
        if gene_pos is None:
            raise ValueError(
                "Unable to resolve gene indices for expression-first module tokenization."
            )
        if gene_pos.dim() == 1:
            # The decoder predicts the full config gene space; the loss selects
            # the genes present in the current batch.
            x_dict["input_gene_mask"] = gene_pos
        return x_dense, gene_pos

    def _embed_expression_first(
        self,
        x_dict: dict,
        input_gene_list=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_dense, gene_pos = self._expression_matrix_and_gene_pos(
            x_dict=x_dict,
            input_gene_list=input_gene_list,
        )
        emb_table = self.embedder.gene_embedding.embedding.to(device=x_dense.device)
        if gene_pos.dim() != 1:
            raise ValueError("expression_first currently expects a shared 1D gene_pos for the batch.")
        gene_emb_table = emb_table.index_select(0, gene_pos)
        cell_feat = x_dense.float() @ gene_emb_table.float()
        cell_feat = cell_feat.to(dtype=emb_table.dtype)

        cell_tok = self.embedder.cell_projection(cell_feat)

        assignment = self._gene_module_assignment_matrix(gene_pos, dtype=x_dense.dtype)
        if assignment.dim() != 2:
            raise ValueError("expression_first currently expects a 2D assignment matrix.")
        assignment = assignment.contiguous()
        assignment_denom = assignment.abs().sum(dim=0).clamp_min(1.0)
        module_expr = torch.matmul(x_dense.to(dtype=assignment.dtype), assignment)
        module_expr = module_expr / assignment_denom.unsqueeze(0)
        # Keep the scalar activity already produced for each module.  The
        # auxiliary objective consumes this value; no decoder/CCC path changes.
        x_dict["gene_module_activity"] = module_expr
        module_tok = self.expression_module_scale(module_expr.unsqueeze(-1).to(dtype=cell_tok.dtype))
        module_tok = module_tok + self.expression_module_emb.to(device=module_tok.device, dtype=module_tok.dtype).unsqueeze(0)
        module_tok = self.gene_module_norm(module_tok)
        module_padding_mask = torch.zeros(
            (x_dense.shape[0], self.total_gene_module_num),
            dtype=torch.bool,
            device=x_dense.device,
        )
        return module_tok, cell_tok, module_padding_mask, assignment

    def forward(self, x_dict, input_gene_list=None, pos=None, dist_bias=None, pair_labels=None):
        x_dict = self.mask_model.apply_mask(x_dict)
        drop_cell_idx = x_dict.get("_drop_node_cell_idx", None)
        drop_old_n = x_dict.get("_drop_node_original_n", None)
        if drop_cell_idx is not None and drop_old_n is not None:
            pos = _select_dropnode_rows(pos, drop_cell_idx, drop_old_n)
            dist_bias = _select_dropnode_square(dist_bias, drop_cell_idx, drop_old_n)
            pair_labels = _select_dropnode_square(pair_labels, drop_cell_idx, drop_old_n)
        module_assignment = None
        gene_after_cell1_residual = None
        gene_after_cell_residuals = None
        if self.gene_token_mode == "full":
            x_dict['gene_emb_raw'], x_dict['h'] = self.embedder(x_dict, input_gene_list)
            x_dict['gene_emb'] = x_dict['gene_emb_raw']
            x_dict['gene_module_emb'] = None
            module_padding_mask = None
            gene_self_bias = None
        else:
            module_emb, cell_emb, module_padding_mask, module_assignment = self._embed_expression_first(
                x_dict=x_dict,
                input_gene_list=input_gene_list,
            )
            x_dict["gene_emb_raw"] = module_emb
            x_dict["h"] = cell_emb
            x_dict["gene_module_emb"] = module_emb

        if self.gene_token_mode == "module":
            gene_self_bias = None
            x_dict['gene_emb'] = x_dict['gene_module_emb']

        pair_labels = _resolve_pair_labels(x_dict, pair_labels=pair_labels)
        cell_attn_bias = self._prepare_cell_ccc_bias(
            x_dict.get("ccc_bias", None),
            x_dict["h"],
            program_edge_index=x_dict.get("ccc_program_edge_index", None),
            program_score=x_dict.get("ccc_program_score", None),
        )

        pos = _resolve_spatial_pos(x_dict, pos=pos)
        dist_bias = _resolve_dist_bias(x_dict, dist_bias=dist_bias)
        flow_scores = None
        flow_gate_mask = None
        if self.flow_cell_attention:
            flow_scores, flow_gate_mask = self._prepare_flow_attention(x_dict)
        enc_out = self.gcat_encoder(
            x_dict,
            gene_self_bias=gene_self_bias,
            gene2cell_bias=None,
            cell2gene_bias=None,
            gene_padding_mask=module_padding_mask,
            pos=pos,
            dist_bias=dist_bias,
            cell_attn_bias=cell_attn_bias,
            gene_after_cell1_residual=gene_after_cell1_residual,
            gene_after_cell_residuals=gene_after_cell_residuals,
            return_pair_rep=True,
            disable_cell_transformer=self.disable_cell_transformer,
            disable_gene_module_attention=self.disable_gene_module_attention,
            flow_scores=flow_scores,
            flow_gate_mask=flow_gate_mask,
            flow_gate=self.flow_attn_gate if self.flow_cell_attention else None,
        )
        x_dict["gene_emb"] = enc_out["gene_emb"]
        if module_assignment is not None and self.regulon_activity_readout is not None:
            x_dict["gene_module_activity_pred"] = self.regulon_activity_readout(
                x_dict["gene_emb"]
            ).squeeze(-1)
        x_dict["h"] = enc_out["hidden"]
        pair_rep = enc_out.get("pair_rep", None)
        # Part of the S4 backbone, independent of the removed pull/GRN branches.
        x_dict["h"] = self.pre_latent_norm(x_dict["h"])
        x_dict["h"], latent_loss = self.latent(x_dict)
        out_dict = self.decoder(x_dict)
        nb_loss = self.negative_binomial_loss(out_dict, x_dict)
        loss = latent_loss + nb_loss
        out_dict["latent_loss"] = latent_loss.item() if torch.is_tensor(latent_loss) else latent_loss
        out_dict["target_loss"] = nb_loss.item()

        masked_zmse_raw = loss.new_tensor(0.0)
        block_mask = x_dict.get("st_block_gene_mask", None)
        if self.lambda_masked_zmse > 0 and block_mask is not None:
            masked_zmse_raw = masked_gene_standardized_mse(
                predicted_expression=out_dict["pred"],
                full_expression=x_dict["x_seq"],
                block_mask=block_mask,
                seed_mask=x_dict.get("seed_mask", None),
                source_mask=x_dict.get("source", None),
            )
            loss = loss + self.lambda_masked_zmse * masked_zmse_raw
        out_dict["masked_zmse_loss_raw"] = float(masked_zmse_raw.item())
        out_dict["masked_zmse_loss"] = float(
            (self.lambda_masked_zmse * masked_zmse_raw).item()
        )
        out_dict["lambda_masked_zmse"] = self.lambda_masked_zmse
        out_dict["st_block_gene_count"] = int(
            block_mask.any(dim=0).sum().item()
        ) if block_mask is not None else 0

        regulon_activity_raw = loss.new_tensor(0.0)
        if self.lambda_reg > 0 and module_assignment is not None and "gene_module_activity_pred" in x_dict:
            regulon_activity_raw = regulon_activity_auxiliary_loss(
                predicted_activity=x_dict["gene_module_activity_pred"],
                full_expression=x_dict["x_seq"],
                assignment=module_assignment,
                lambda_reg=self.lambda_reg,
                std_floor=self.reg_activity_std_floor,
            )
            loss = loss + self.lambda_reg * regulon_activity_raw
        out_dict["regulon_activity_loss_raw"] = regulon_activity_raw.item()
        out_dict["regulon_activity_loss"] = (self.lambda_reg * regulon_activity_raw).item()
        out_dict["lambda_reg"] = self.lambda_reg

        out_dict["flow_cell_attention"] = self.flow_cell_attention
        out_dict["flow_attn_layers"] = self.flow_attn_layers if self.flow_cell_attention else "none"
        out_dict["flow_attn_mode"] = self.flow_attn_mode
        if self.flow_cell_attention:
            selected_layers = list(self.gcat_encoder.flow_attn_layer_indices)
            if selected_layers:
                out_dict["flow_attn_gate_mean"] = float(
                    self.flow_attn_gate.detach().float()[selected_layers].mean().item()
                )
            else:
                out_dict["flow_attn_gate_mean"] = 0.0
            out_dict["flow_attn_edge_count"] = int(flow_gate_mask.sum().item())

        out_dict["gene_token_mode"] = self.gene_token_mode
        out_dict["gene_module_backend"] = self.gene_module_backend
        out_dict["gene_module_assignment"] = self.gene_module_assignment
        out_dict["gene_module_biological_num"] = int(self.pathway_module_num)
        out_dict["dynamic_gene_modules"] = bool(self.dynamic_gene_modules)
        out_dict["prior_strength"] = float(self.prior_strength)
        out_dict["off_prior_init_std"] = float(self.off_prior_init_std)
        out_dict["assignment_temperature"] = float(self.assignment_temperature)
        out_dict["gene_module_global_uncovered_count"] = int(self.gene_module_uncovered_count)
        out_dict["gene_module_global_uncovered_ratio"] = float(self.gene_module_uncovered_ratio)
        if module_assignment is not None:
            with torch.no_grad():
                flat_assignment = module_assignment.detach().float().reshape(-1, module_assignment.shape[-1])
                flat_bio = flat_assignment[:, :self.pathway_module_num]
                current_uncovered = flat_bio.abs().sum(dim=-1) <= 1e-8
                current_total = int(current_uncovered.numel())
                current_uncovered_count = int(current_uncovered.sum().item())
            out_dict["gene_module_active_gene_count"] = current_total
            out_dict["gene_module_uncovered_count"] = current_uncovered_count
            out_dict["gene_module_uncovered_ratio"] = float(current_uncovered_count / max(1, current_total))
            if self.dynamic_gene_modules:
                gene_pos = self._resolve_gene_pos(
                    x_dict=x_dict,
                    input_gene_list=input_gene_list,
                    gene_len=module_assignment.shape[-2],
                    device=module_assignment.device,
                )
                if gene_pos is not None and gene_pos.dim() == 1:
                    prior_mask = self.gene_module_prior_mask.index_select(0, gene_pos.long()).to(
                        device=flat_assignment.device,
                        dtype=flat_assignment.dtype,
                    )
                    prior_mass = (flat_assignment * prior_mask).sum()
                    total_mass = flat_assignment.sum().clamp_min(1e-8)
                    out_dict["dynamic_prior_assignment_fraction"] = float(
                        (prior_mass / total_mass).item()
                    )
                    module_mass = flat_assignment.sum(dim=0)
                    out_dict["dynamic_module_mass_min"] = float(module_mass.min().item())
                    out_dict["dynamic_module_mass_max"] = float(module_mass.max().item())
        else:
            out_dict["gene_module_active_gene_count"] = int(len(self.gene_list))
            out_dict["gene_module_uncovered_count"] = int(self.gene_module_uncovered_count)
            out_dict["gene_module_uncovered_ratio"] = float(self.gene_module_uncovered_ratio)
        source = x_dict.get("source")
        if torch.is_tensor(source):
            is_source_batch = bool(source.reshape(-1)[0].item()) if source.numel() else False
        elif isinstance(source, np.ndarray):
            is_source_batch = bool(source.reshape(-1)[0]) if source.size else False
        elif isinstance(source, (list, tuple)):
            is_source_batch = bool(source[0]) if source else False
        else:
            is_source_batch = bool(source) if source is not None else False

        pair_loss = None
        if (not is_source_batch) and pair_labels is not None and pair_rep is not None:
            pair_mask = pair_labels != self.pair_mask_token
            if torch.is_tensor(pair_mask) and bool(pair_mask.any().item()):
                pair_pred = self.pair_head(pair_rep[pair_mask]).squeeze(-1)
                pair_target = pair_labels[pair_mask].to(pair_pred.dtype)
                pair_loss = F.mse_loss(pair_pred, pair_target)
                pair_loss *= self.pair_loss_weight
                loss = loss + pair_loss
                out_dict["pair_loss"] = pair_loss.item()
                out_dict["pair_pred"] = pair_pred.detach()
            else:
                out_dict["pair_loss"] = 0.0

        if pair_loss is None:
            out_dict["pair_loss"] = out_dict.get("pair_loss", 0.0)

        return out_dict, loss
