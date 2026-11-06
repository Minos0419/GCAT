import warnings

warnings.filterwarnings("ignore")

import argparse
import json
import os
import pathlib
import random
import sys
import time
from contextlib import nullcontext
from typing import List, Optional, Union

import anndata as ad
import hdf5plugin  # noqa: F401
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
from tqdm import tqdm

from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gcat.data import build_dataloaders
from gcat.data import XDict
from gcat.evaluation.imputation import imputation_eval
from gcat.model import GCATFormer
from gcat.modules.common import set_seed
from torch.optim.lr_scheduler import ReduceLROnPlateau


def load_gcat_former():
    return GCATFormer


_GENE_SYMBOL_CACHE: dict[str, str] = {}


def _map_ensembl_to_symbol(ensembl_ids: List[str]) -> dict:
    missing_ids = [gene for gene in ensembl_ids if gene not in _GENE_SYMBOL_CACHE]
    if not missing_ids:
        return {gene: _GENE_SYMBOL_CACHE.get(gene, gene) for gene in ensembl_ids}
    try:
        import mygene

        mg = mygene.MyGeneInfo()
        mapping = {}
        for start in range(0, len(missing_ids), 500):
            batch = missing_ids[start:start + 500]
            results = mg.querymany(
                batch,
                scopes="ensembl.gene",
                fields="symbol",
                species="human",
                verbose=False,
            )
            for item in results:
                if "symbol" in item and not item.get("notfound", False):
                    mapping[item["query"]] = item["symbol"]
        _GENE_SYMBOL_CACHE.update({gene: mapping.get(gene, gene) for gene in missing_ids})
    except Exception:
        _GENE_SYMBOL_CACHE.update({gene: gene for gene in missing_ids})
    return {gene: _GENE_SYMBOL_CACHE.get(gene, gene) for gene in ensembl_ids}


def _specificity_scores(assignment: torch.Tensor) -> torch.Tensor:
    if assignment.shape[1] <= 1:
        return assignment
    row_sum = assignment.sum(dim=1, keepdim=True)
    other_mean = (row_sum - assignment) / (assignment.shape[1] - 1)
    return assignment - other_mean


def _write_module_top_genes(
    scores: torch.Tensor,
    raw_assignment: torch.Tensor,
    genes: List[str],
    symbol_map: dict,
    out_path: pathlib.Path,
    top_k: int,
    score_type: str,
) -> None:
    rows = []
    n_modules = scores.shape[1]
    top_k = min(top_k, scores.shape[0])
    for module_idx in range(n_modules):
        values, indices = torch.topk(scores[:, module_idx], k=top_k)
        for rank, gene_idx_tensor in enumerate(indices, start=1):
            gene_idx = int(gene_idx_tensor.item())
            gene = genes[gene_idx]
            rows.append(
                {
                    "module": module_idx,
                    "rank": rank,
                    "gene_ensembl": gene,
                    "gene_symbol": symbol_map.get(gene, gene),
                    "score": float(values[rank - 1].item()),
                    "raw_weight": float(raw_assignment[gene_idx, module_idx].item()),
                    "score_type": score_type,
                }
            )
    pd.DataFrame(rows).to_csv(out_path, index=False)


def _write_redundancy(top_csv: pathlib.Path) -> None:
    df = pd.read_csv(top_csv)
    gene_counts = (
        df.groupby("gene_ensembl")["module"]
        .nunique()
        .reset_index()
        .rename(columns={"module": "n_modules"})
    )
    symbol_map = dict(zip(df["gene_ensembl"], df["gene_symbol"]))
    gene_counts["gene_symbol"] = gene_counts["gene_ensembl"].map(symbol_map).fillna(gene_counts["gene_ensembl"])
    gene_counts.sort_values("n_modules", ascending=False).to_csv(
        top_csv.with_name(f"{top_csv.stem}_redundancy.csv"),
        index=False,
    )
    total = int(len(gene_counts))
    unique_to_one = int((gene_counts["n_modules"] == 1).sum())
    summary = {
        "top_file": top_csv.name,
        "total_unique": total,
        "unique_to_one": unique_to_one,
        "shared": total - unique_to_one,
    }
    with open(top_csv.with_name(f"{top_csv.stem}_redundancy_summary.json"), "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)


def export_gene_module_assignment(model, genes: List[str], out_dir: pathlib.Path, prefix: str) -> None:
    if not hasattr(model, "_gene_module_assignment_matrix"):
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    gene_pos = model._resolve_gene_pos({}, genes, len(genes), device)
    with torch.no_grad():
        assignment = model._gene_module_assignment_matrix(gene_pos, dtype=torch.float32)
        if assignment.dim() == 3:
            assignment = assignment[0]
        assignment = assignment.detach().cpu().float()
    symbol_map = _map_ensembl_to_symbol(genes)

    np.save(out_dir / f"{prefix}_gene_module_assignment.npy", assignment.numpy())
    (out_dir / f"{prefix}_gene_module_genes.txt").write_text(
        "\n".join(map(str, genes)) + "\n", encoding="utf-8"
    )
    rows = []
    topk = min(5, assignment.shape[1])
    values, indices = torch.topk(assignment.abs(), k=topk, dim=1)
    for gene_idx, gene in enumerate(genes):
        for rank in range(topk):
            module_idx = int(indices[gene_idx, rank].item())
            rows.append(
                {
                    "gene": gene,
                    "gene_symbol": symbol_map.get(gene, gene),
                    "module": module_idx,
                    "rank": rank + 1,
                    "weight": float(assignment[gene_idx, module_idx].item()),
                    "abs_weight": float(values[gene_idx, rank].item()),
                }
            )
    pd.DataFrame(rows).to_csv(out_dir / f"{prefix}_gene_module_assignment_top5_by_gene.csv", index=False)

    for top_k in (3, 5, 10):
        raw_path = out_dir / f"{prefix}_gene_module_top{top_k}.csv"
        spec_path = out_dir / f"{prefix}_gene_module_top{top_k}_specificity.csv"
        _write_module_top_genes(
            scores=assignment,
            raw_assignment=assignment,
            genes=genes,
            symbol_map=symbol_map,
            out_path=raw_path,
            top_k=top_k,
            score_type="raw_weight",
        )
        _write_module_top_genes(
            scores=_specificity_scores(assignment),
            raw_assignment=assignment,
            genes=genes,
            symbol_map=symbol_map,
            out_path=spec_path,
            top_k=top_k,
            score_type="specificity",
        )
        _write_redundancy(raw_path)
        _write_redundancy(spec_path)

    stats = {
        "n_genes": int(assignment.shape[0]),
        "n_modules": int(assignment.shape[1]),
        "nonzero_ratio": float((assignment.abs() > 1e-6).float().mean().item()),
        "avg_modules_per_gene": float((assignment.abs() > 1e-6).float().sum(dim=1).mean().item()),
    }
    with open(out_dir / f"{prefix}_gene_module_assignment_summary.json", "w", encoding="utf-8") as fp:
        json.dump(stats, fp, ensure_ascii=False, indent=2)


def export_gene_module_trajectory_snapshot(
    model,
    genes: List[str],
    out_dir: pathlib.Path,
    prefix: str,
    epoch: int,
    min_support: int,
    inactive_mass_fraction: float,
) -> dict:
    """Export lightweight current-state module usage without changing attention."""
    out_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    gene_pos = model._resolve_gene_pos({}, genes, len(genes), device)
    with torch.no_grad():
        assignment = model._gene_module_assignment_matrix(gene_pos, dtype=torch.float32)
        if assignment.dim() == 3:
            assignment = assignment[0]
        assignment = assignment.detach().cpu().float().numpy()
    magnitude = np.abs(assignment)
    positive = magnitude > 1e-6
    support = positive.sum(axis=0)
    mass = magnitude.sum(axis=0)
    argmax_count = np.bincount(magnitude.argmax(axis=1), minlength=assignment.shape[1])
    positive_mass = mass[mass > 1e-6]
    reference_mass = float(np.median(positive_mass)) if positive_mass.size else 0.0
    effective = (support >= int(min_support)) & (
        mass >= reference_mass * float(inactive_mass_fraction)
    )
    mass_effective_count = float(mass.sum() ** 2 / max(float(np.square(mass).sum()), 1e-12))
    stage = "initial" if epoch == 0 else f"epoch{epoch}"
    stem = f"{prefix}_{stage}"
    np.save(out_dir / f"{stem}_assignment.npy", assignment)
    gene_path = out_dir / f"{prefix}_genes.txt"
    if not gene_path.exists():
        gene_path.write_text("\n".join(map(str, genes)) + "\n", encoding="utf-8")
    pd.DataFrame(
        {
            "module": np.arange(assignment.shape[1]),
            "support_gene_count": support.astype(int),
            "argmax_gene_count": argmax_count.astype(int),
            "assignment_mass": mass,
            "effective": effective,
            "participates_in_attention": True,
        }
    ).to_csv(out_dir / f"{stem}_modules.tsv", sep="\t", index=False)
    summary = {
        "epoch": int(epoch),
        "stage": stage,
        "allocated_modules": int(assignment.shape[1]),
        "attention_participating_modules": int(assignment.shape[1]),
        "nonempty_modules": int((support > 0).sum()),
        "primary_modules": int((argmax_count > 0).sum()),
        "effective_modules": int(effective.sum()),
        "mass_effective_module_count": mass_effective_count,
        "min_support": int(min_support),
        "inactive_mass_fraction_of_median": float(inactive_mass_fraction),
        "module_mass_median": reference_mass,
        "gene_count": len(genes),
    }
    (out_dir / f"{stem}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query_data", type=pathlib.Path,
        default=pathlib.Path(os.environ.get("QUERY_DATA", "data/data1_st.h5ad")),
    )
    parser.add_argument(
        "--ref_data", type=pathlib.Path,
        default=pathlib.Path(os.environ.get("REF_DATA", "data/data1_sc.h5ad")),
    )
    parser.add_argument(
        "--mask_gene", type=pathlib.Path,
        default=pathlib.Path(os.environ.get("MASK_GENE", "data/data1_10fold_gene.npy")),
    )
    parser.add_argument(
        "--mask_fold_index",
        type=int,
        default=0,
        help="Index into the 10-fold target-gene array stored by --mask_gene.",
    )
    parser.add_argument("--random_seed", type=int, default=11)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic PyTorch/CUDA algorithms for reproducible comparisons.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--validation_split_mode",
        choices=["legacy", "batch_last_10pct"],
        default="legacy",
    )
    parser.add_argument("--lr", type=float, default=5e-4, help="AdamW learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="AdamW weight decay.")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Linear LR warmup epochs. 0 disables warmup.")
    parser.add_argument(
        "--config", type=pathlib.Path,
        default=pathlib.Path(os.environ.get("GCAT_CONFIG", "assets/config.json")),
    )
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--latent_dim", type=int, default=None)
    parser.add_argument("--decoder_hidden_dim", type=int, default=None)
    parser.add_argument("--decoder_layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--prefix", type=str, default="cp_multi_cls_GM_CCCbias")
    parser.add_argument("--out_dir", type=pathlib.Path,
                        default=pathlib.Path(os.environ.get("OUT_DIR", "results"))
    )
    parser.add_argument("--n_neighbors", type=int, default=4)
    parser.add_argument("--use_gp", action="store_true")
    parser.add_argument(
        "--coord_norm_mode",
        type=str,
        default="pc_l2",
        choices=["none", "pc_l1", "pc_l2", "minmax_01"],
    )
    parser.add_argument(
        "--dist_metric",
        type=str,
        default="l2",
        choices=["l1", "l2"],
    )
    parser.add_argument(
        "--no_dist_bias",
        action="store_true",
        help="Ablation: drop the spatial distance prior from cell attention "
        "(model receives neither dist_bias nor pos/coord, so no fallback either).",
    )
    parser.add_argument("--masa_noise_rate", type=float, default=0.0)
    parser.add_argument("--masa_noise_std", type=float, default=0.0)
    parser.add_argument("--masa_mask_rate", type=float, default=0.1)
    parser.add_argument("--drop_node_rate", type=float, default=0.25, help="Drop this fraction of sampled neighbor cells during training; center/seed cells are always kept.")
    parser.add_argument(
        "--balance_modalities",
        action="store_true",
        help="Oversample the shorter ST/scRNA loader so both provide equal optimizer steps per epoch.",
    )
    parser.add_argument(
        "--st_block_gene_mask_rate",
        type=float,
        default=0.0,
        help="Fraction of platform-observed genes hidden across every ST cell in a training batch.",
    )
    parser.add_argument("--pair_loss_weight", type=float, default=10.0)
    parser.add_argument(
        "--lambda_reg",
        type=float,
        default=0.05,
        help="Weight for the regulon/module activity auxiliary loss; 0 disables it.",
    )
    parser.add_argument(
        "--reg_activity_std_floor",
        type=float,
        default=0.1,
        help="Minimum batch reference std used only to standardize the activity target.",
    )
    parser.add_argument(
        "--lambda_masked_zmse",
        type=float,
        default=0.0,
        help="Weight for standardized MSE on block-masked genes in ST seed cells.",
    )
    parser.add_argument(
        "--lambda_out",
        type=float,
        choices=[0.0, 0.01, 0.05, 0.1],
        default=0.0,
        help="Weight for outgoing ligand-program activity loss; 0 disables it.",
    )
    parser.add_argument(
        "--lambda_pull",
        type=float,
        default=0.0,
        help=(
            "Weight for the positive-pair pull loss: mean 1-cos(mu_i, mu_j) "
            "over mutual-top-k positive pairs; 0 disables it. Cosine pull is "
            "O(1), so meaningful weights are O(1-10)."
        ),
    )
    parser.add_argument(
        "--pull_label_mode",
        choices=["state", "niche", "both"],
        default="both",
        help=(
            "Positive-pair source: state (PCA cosine on observed expression), "
            "niche (spatial-kNN niche representation cosine), or both "
            "(mean of the two pull terms)."
        ),
    )
    parser.add_argument(
        "--pull_topk",
        type=int,
        default=10,
        help="k for the mutual top-k positive-pair graphs.",
    )
    parser.add_argument(
        "--pull_niche_k",
        type=int,
        default=15,
        help="Spatial kNN size for the niche representation (from coordinates).",
    )
    parser.add_argument(
        "--pull_target",
        choices=["latent", "embedding"],
        default="latent",
        help=(
            "Where the pull loss acts: 'latent' (VAE encoder mean) or "
            "'embedding' (pre-VAE encoder cell embedding; the VAE then "
            "regularizes downstream as usual)."
        ),
    )
    parser.add_argument(
        "--st_aligned_sc_prior",
        action="store_true",
        help=(
            "ST-aligned SC communication prior: each ST cell adopts the LR "
            "program activity of its top-1 most-similar SC cell (PCA + cosine "
            "on shared genes); ST interaction topology stays spatial. Works "
            "with the scalar CCC bias and with S4 flow attention."
        ),
    )
    parser.add_argument(
        "--outgoing_lr_std_floor",
        type=float,
        default=0.1,
        help="Minimum batch target std for outgoing ligand-program standardization.",
    )
    parser.add_argument(
        "--flow_linear_bottleneck",
        action="store_true",
        help="Enable the FlowSig-inspired inflow -> module state -> outflow linear bottleneck.",
    )
    parser.add_argument(
        "--lambda_flow",
        type=float,
        choices=[0.0, 0.01, 0.05, 0.1],
        default=0.0,
        help="Weight for the Stage-1 flow bottleneck outflow loss.",
    )
    parser.add_argument(
        "--flow_structure",
        choices=["reg_out", "full"],
        default="full",
        help="Stage-1 structure: regulon state to outflow, or inflow residual through regulon state.",
    )
    parser.add_argument(
        "--flow_supervision",
        choices=["source", "all"],
        default="source",
        help="Supervise outflow on complete source/scRNA cells only or on all cells.",
    )
    parser.add_argument(
        "--flow_graph_mask_path",
        type=pathlib.Path,
        default=None,
        help="Stage-3 NPZ containing graph-constrained in_reg_mask and reg_out_mask.",
    )
    parser.add_argument(
        "--flow_oracle_st_truth",
        action="store_true",
        help="Leakage-only upper bound: use unmasked ST expression for flow scores/targets.",
    )
    parser.add_argument(
        "--flow_cell_attention",
        action="store_true",
        help=(
            "Enable Stage-4 flow-directed cell attention: receiver queries from "
            "inflow+regulon, sender keys from outflow+regulon, mixed into the "
            "existing cell attention on known directed LR edges. Replaces the "
            "scalar CCC bias."
        ),
    )
    parser.add_argument(
        "--flow_attn_layers",
        type=str,
        default="last",
        help="Encoder cell-attention layers receiving flow attention mixing: none/last/all/0,1/-1/1:3.",
    )
    parser.add_argument(
        "--flow_gate_hidden_dim",
        type=int,
        default=64,
        help="Hidden width of the regulon-gating MLPs in flow-directed cell attention.",
    )
    parser.add_argument(
        "--flow_gate_init",
        type=float,
        default=0.0,
        help="Initial value of the per-layer/per-head interpolation gate Gamma (0 = baseline-equivalent).",
    )
    parser.add_argument("--num_heads", type=int, default=None, help="Override attention head count.")
    parser.add_argument(
        "--ffn_mult",
        type=int,
        default=None,
        help="Override the transformer FFN expansion ratio (default 2, i.e. d -> 2d -> d).",
    )
    parser.add_argument(
        "--flow_attn_mode",
        choices=["gated", "replace"],
        default="gated",
        help=(
            "gated: per-layer/per-head zero-init residual interpolation "
            "S_base + Gamma*G*(S_flow - S_base); replace: fixed Gamma=1, i.e. "
            "G*S_flow + (1-G)*S_base on directed LR edges only."
        ),
    )
    parser.add_argument(
        "--flow_stage5",
        action="store_true",
        help=(
            "Minimal Stage 5: fold the layer-(L-1) flow-attention receiver "
            "message into an updated regulon state (zero-init beta) and "
            "recompute the final layer's flow scores. Requires last2 layers."
        ),
    )
    parser.add_argument(
        "--grn_decoder_residual",
        action="store_true",
        help=(
            "Signed GRN-guided NB decoder residual: a = Linear(z) predicts "
            "signed regulon activity, W_GRN = S*softplus(theta) on signed "
            "prior edges only, eta = eta_base + alpha * a W_GRN with a "
            "zero-init alpha (initial forward equals the baseline)."
        ),
    )
    parser.add_argument(
        "--grn_sign_path",
        type=pathlib.Path,
        default=None,
        help="NPZ from tools/build_signed_grn.py (module_idx/gene_idx/signs/gene_names).",
    )
    parser.add_argument(
        "--grn_frozen_baseline",
        action="store_true",
        help=(
            "Two-stage mode: load the scalar-CCC baseline checkpoint, freeze every "
            "non-GRN parameter, and train only the GRN residual branch."
        ),
    )
    parser.add_argument(
        "--grn_baseline_ckpt",
        type=pathlib.Path,
        default=None,
        help="Baseline checkpoint (.pt with model_state_dict) for two-stage mode.",
    )
    parser.add_argument(
        "--reg_loss_stop_below",
        type=float,
        default=0.0,
        help="Stop after the configured batch check if mean raw regulon activity loss is below this value; 0 disables.",
    )
    parser.add_argument(
        "--reg_loss_check_batches",
        type=int,
        default=10,
        help="Number of first-epoch train batches used by --reg_loss_stop_below.",
    )
    parser.add_argument("--pair_mask_token", type=float, default=-100.0)
    parser.add_argument(
        "--gene_token_mode",
        type=str,
        default="full",
        choices=["full", "module"],
        help="full uses original gene tokens for gene attention; module keeps the compressed gene-module tokens.",
    )
    parser.add_argument("--use_ccc_bias", action="store_true", default=True, help="Inject cached CCC bias into cell self-attention.")
    parser.add_argument("--no_ccc_bias", action="store_false", dest="use_ccc_bias", help="Disable CCC bias injection.")
    parser.add_argument(
        "--ccc_bias_layers",
        type=str,
        default="last",
        help="Encoder cell-attention layers that receive CCC bias: none/off/last/all/0,1/-1/1:3.",
    )
    parser.add_argument("--ccc_bias_trainable", action="store_true", help="Learn a single global scale for CCC bias.")
    parser.add_argument("--ccc_bias_init_scale", type=float, default=1.0, help="Initial global scale applied to normalized CCC bias.")
    parser.add_argument(
        "--ccc_program_head_bias",
        action="store_true",
        help=(
            "Add a zero-initialized shared Linear(LR programs, attention heads) on the "
            "existing ST/SC CCC edges. Disabled by default."
        ),
    )
    parser.add_argument(
        "--niche_regulon_bridge",
        action="store_true",
        help="Inject a fixed NicheNet LR-program-to-prior-regulon signal before the first GCAT block.",
    )
    parser.add_argument(
        "--niche_regulon_bridge_path",
        type=pathlib.Path,
        default=None,
        help="NPZ containing fixed C with shape (matched LR programs, prior regulons).",
    )
    parser.add_argument(
        "--niche_bridge_mode",
        type=str,
        choices=["dense", "top10"],
        default="dense",
        help="Use the row-normalized dense bridge or per-row top-10%% sparse bridge.",
    )
    parser.add_argument(
        "--niche_bridge_stage",
        type=str,
        choices=["off", "pre_encoder", "after_cell1", "multi_layer"],
        default="pre_encoder",
        help="Inject before the encoder, after cell 1, after cells 1-3, or disable the bridge.",
    )
    parser.add_argument(
        "--ccc_ensembl_interaction_cache",
        type=pathlib.Path,
        default=pathlib.Path(
            os.environ.get("CCC_ENSEMBL_INTERACTION_CACHE", "data/interactions_human__ensembl_lr.pt")
        ),
        help="Ensembl LR cache built from CCC/LR database.",
    )
    parser.add_argument(
        "--ccc_cache_dir",
        type=pathlib.Path,
        default=None,
        help="Directory for the two precomputed CCC sparse graphs. Defaults to out_dir/ccc_cache.",
    )
    parser.add_argument("--ccc_cache_name", type=str, default="fold0")
    parser.add_argument("--ccc_topk", type=int, default=64, help="Top-k CCC neighbors kept per cell in the cached sparse graph.")
    parser.add_argument("--ccc_block_size", type=int, default=512, help="Block size for CCC score construction.")
    parser.add_argument("--ccc_min_pairs", type=int, default=32, help="Minimum matched LR programs required to enable CCC bias.")
    parser.add_argument(
        "--ccc_st_candidate_mode",
        type=str,
        default="spatial",
        choices=["spatial", "all"],
        help="Candidate sender graph for ST CCC: spatial uses the existing spatial KNN; all keeps previous all-to-all scoring.",
    )
    parser.add_argument(
        "--ccc_sc_candidate_mode",
        type=str,
        default="expression",
        choices=["expression", "all", "none"],
        help="Candidate sender graph for SC CCC: expression uses expression PCA/KNN; all keeps previous all-to-all scoring; none disables SC CCC.",
    )
    parser.add_argument("--ccc_st_candidate_k", type=int, default=0, help="ST candidate K. 0 means reuse --n_neighbors.")
    parser.add_argument("--ccc_sc_candidate_k", type=int, default=0, help="SC candidate K. 0 means reuse --n_neighbors.")
    parser.add_argument("--ccc_sc_pca_n_components", type=int, default=10, help="PCA/SVD components for SC expression candidate KNN.")
    parser.add_argument("--gene_module_num", type=str, default="64", help="Module count or 'auto' when gene_module_init_path is provided.")
    parser.add_argument("--gene_module_export_stats", action="store_true")
    parser.add_argument(
        "--gene_module_track_every_epochs",
        type=int,
        default=0,
        help="Export lightweight module usage at initialization and every N epochs; 0 disables.",
    )
    parser.add_argument("--gene_module_track_min_support", type=int, default=5)
    parser.add_argument("--gene_module_track_inactive_mass_fraction", type=float, default=0.1)
    parser.add_argument("--gene_module_init_path", type=pathlib.Path, default=None)
    parser.add_argument(
        "--use_data_covariates",
        action="store_true",
        help="Use current ST/SC batch and platform IDs in decoder embeddings; disabled for the reproduced default.",
    )
    parser.add_argument(
        "--use_batch_covariate",
        action="store_true",
        help="Use only the prefixed ST/SC batch ID as a decoder conditional embedding.",
    )
    parser.add_argument(
        "--dynamic_gene_modules",
        action="store_true",
        help="Use one unified prior-initialized, fully learnable module pool. No module token is gated off.",
    )
    parser.add_argument("--prior_strength", type=float, default=3.0)
    parser.add_argument("--off_prior_init_std", type=float, default=0.02)
    parser.add_argument("--assignment_temperature", type=float, default=1.0)
    parser.add_argument(
        "--disable_cell_transformer",
        action="store_true",
        help="Ablation: bypass every cell-attention transformer while retaining the gene/module path.",
    )
    parser.add_argument(
        "--disable_gene_module_attention",
        action="store_true",
        help="Ablation: bypass the joint cell-token/gene-module attention while retaining cell attention.",
    )
    parser.add_argument(
        "--amp_mode",
        type=str,
        default="off",
        choices=["off", "fp16", "bf16", "full_fp16", "full_bf16"],
    )
    parser.add_argument("--save_speed_metrics", action="store_true")
    parser.add_argument("--save_model", action="store_true", help="Save final model checkpoint. Disabled by default for sweeps.")
    parser.add_argument("--save_pred", action="store_true", help="Save prediction matrix. Disabled by default for sweeps.")
    parser.add_argument("--save_per_gene_score", action="store_true", help="Save per-gene metric table. Disabled by default for sweeps.")
    parser.add_argument(
        "--skip_final_metrics",
        action="store_true",
        help="Save final predictions but skip metrics (useful when hold-out targets are unmeasured zero placeholders).",
    )
    parser.add_argument("--max_train_batches", type=int, default=0, help="Debug only: stop each epoch after N train batches when >0.")
    parser.add_argument("--max_eval_batches", type=int, default=0, help="Debug only: stop validation/prediction after N batches when >0.")
    parser.add_argument("--skip_final_eval", action="store_true", help="Debug only: skip final prediction and metric computation.")
    parser.add_argument(
        "--eval_every_epochs",
        type=int,
        default=10,
        help=(
            "Run full prediction/metric evaluation every N epochs using the best validation-loss "
            "checkpoint seen so far. 0 disables interval evaluation."
        ),
    )
    parser.add_argument("--save_interval_checkpoints", action="store_true")
    parser.add_argument("--log_batch_every", type=int, default=0, help="Debug only: print train batch timing every N batches when >0.")
    return parser.parse_args()


def save_prediction_outputs(pred_df: pd.DataFrame, out_dir: pathlib.Path, prefix: str) -> dict:
    pred_matrix = pred_df.to_numpy(dtype=np.float32, copy=True)
    npz_path = out_dir / f"{prefix}_pred.npz"
    np.savez_compressed(
        npz_path,
        pred=pred_matrix,
        cells=np.asarray([str(x) for x in pred_df.index.tolist()], dtype="U"),
        genes=np.asarray([str(x) for x in pred_df.columns.tolist()], dtype="U"),
    )
    return {"pred_npz": str(npz_path)}


def save_metric_outputs(result_all: pd.DataFrame, combined_score: dict, out_dir: pathlib.Path, prefix: str) -> dict:
    score_path = out_dir / f"{prefix}_score.tsv"
    pd.DataFrame.from_dict(
        combined_score,
        orient="index",
        columns=[prefix],
    ).to_csv(score_path, sep="\t", index=True, float_format="%.8g")
    return {
        "score_tsv": str(score_path),
    }


def save_per_gene_metric_outputs(result_all: pd.DataFrame, out_dir: pathlib.Path, prefix: str) -> dict:
    per_gene_path = out_dir / f"{prefix}_per_gene_score.tsv"
    result_all.to_csv(per_gene_path, sep="\t", float_format="%.6g")
    return {"per_gene_score_tsv": str(per_gene_path)}


def common_preprocess(adata, hvg, covariate_fields, ensembl_auto_conversion):
    _ = ensembl_auto_conversion
    if covariate_fields:
        for field in covariate_fields:
            assert field in ["batch", "dataset", "platform"], (
                'Currently does not support customized covariate other than "batch", "dataset" and "platform"'
            )
    adata = adata.copy()
    if hvg > 0 and hvg < adata.shape[1]:
        sc.pp.highly_variable_genes(adata, n_top_genes=hvg, subset=True, flavor="seurat_v3")
    return adata


def move_xdict_to_device(x_dict, device):
    for key, value in x_dict.items():
        if torch.is_tensor(value):
            x_dict[key] = value.to(device)
    return x_dict


def _flatten_lr_genes(side) -> list[str]:
    genes = []
    if isinstance(side, (str, bytes)):
        return [str(side)]
    if side is None:
        return genes
    for item in side:
        if isinstance(item, (list, tuple, set)):
            genes.extend(str(x) for x in item if x is not None and str(x) != "")
        elif item is not None and str(item) != "":
            genes.append(str(item))
    return sorted(set(genes))


def load_ensembl_lr_programs(path: pathlib.Path, var_names) -> list[tuple[list[str], list[str]]]:
    if path is None:
        return []
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ccc_ensembl_interaction_cache not found: {path}")
    raw = torch.load(path, map_location="cpu")
    var_set = set(map(str, var_names))
    programs = []
    for rec in raw:
        if not isinstance(rec, (list, tuple)) or len(rec) < 2:
            continue
        lig = [g for g in _flatten_lr_genes(rec[0]) if g in var_set]
        rcp = [g for g in _flatten_lr_genes(rec[1]) if g in var_set]
        if lig and rcp:
            programs.append((lig, rcp))
    seen = set()
    out = []
    for lig, rcp in programs:
        key = (tuple(lig), tuple(rcp))
        if key not in seen:
            seen.add(key)
            out.append((lig, rcp))
    return out


def _matrix_to_dense_float32(x):
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


def _program_expression(adata: ad.AnnData, programs: list[list[str]], block_size: int = 512) -> np.ndarray:
    gene_to_idx = {str(g): i for i, g in enumerate(adata.var_names)}
    out = np.zeros((adata.n_obs, len(programs)), dtype=np.float32)
    for start in range(0, len(programs), int(block_size)):
        end = min(len(programs), start + int(block_size))
        cols = []
        for genes in programs[start:end]:
            idx = [gene_to_idx[g] for g in genes if g in gene_to_idx]
            if not idx:
                cols.append(np.zeros(adata.n_obs, dtype=np.float32))
                continue
            x = _matrix_to_dense_float32(adata[:, idx].X)
            vals = np.expm1(np.log1p(np.clip(x, a_min=0.0, a_max=None)).mean(axis=1))
            cols.append(vals.astype(np.float32, copy=False))
        out[:, start:end] = np.stack(cols, axis=1)
    return out


def _compute_topk_ccc_sparse(ligand: np.ndarray, receptor: np.ndarray, topk: int, block_size: int):
    from scipy import sparse

    n_cells = int(ligand.shape[0])
    topk = int(max(1, min(topk, max(1, n_cells - 1))))
    rows = []
    cols = []
    vals = []
    ligand_t = ligand.T.astype(np.float32, copy=False)
    for start in range(0, n_cells, int(block_size)):
        end = min(n_cells, start + int(block_size))
        scores = receptor[start:end].astype(np.float32, copy=False) @ ligand_t
        local_rows = np.arange(start, end)
        scores[np.arange(end - start), local_rows] = 0.0
        take = min(topk, n_cells)
        idx = np.argpartition(scores, -take, axis=1)[:, -take:]
        row_idx = np.repeat(np.arange(end - start), take)
        col_idx = idx.reshape(-1)
        val = scores[row_idx, col_idx]
        keep = val > 0
        rows.extend((row_idx[keep] + start).tolist())
        cols.extend(col_idx[keep].tolist())
        vals.extend(val[keep].astype(np.float32).tolist())
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n_cells, n_cells), dtype=np.float32)


def _compute_candidate_ccc_sparse(
    ligand: np.ndarray,
    receptor: np.ndarray,
    candidate_graph,
    topk: int,
):
    from scipy import sparse

    candidate_graph = candidate_graph.tocsr()
    n_cells = int(ligand.shape[0])
    topk = int(max(1, min(topk, max(1, n_cells - 1))))
    rows = []
    cols = []
    vals = []
    for receiver in range(n_cells):
        start, end = candidate_graph.indptr[receiver], candidate_graph.indptr[receiver + 1]
        cand = candidate_graph.indices[start:end]
        if cand.size == 0:
            continue
        cand = cand[cand != receiver]
        if cand.size == 0:
            continue
        score = ligand[cand].astype(np.float32, copy=False) @ receptor[receiver].astype(np.float32, copy=False)
        keep = score > 0
        if not np.any(keep):
            continue
        cand = cand[keep]
        score = score[keep]
        take = min(topk, score.size)
        if take < score.size:
            chosen = np.argpartition(score, -take)[-take:]
            chosen = chosen[np.argsort(score[chosen])[::-1]]
        else:
            chosen = np.argsort(score)[::-1]
        rows.extend([receiver] * len(chosen))
        cols.extend(cand[chosen].astype(np.int64).tolist())
        vals.extend(score[chosen].astype(np.float32).tolist())
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n_cells, n_cells), dtype=np.float32)


def _st_spatial_candidate_graph(adata: ad.AnnData, k: int):
    from scipy import sparse
    from sklearn.neighbors import NearestNeighbors

    n_cells = int(adata.n_obs)
    k = int(max(1, min(k, max(1, n_cells - 1))))
    if "connectivities" in adata.obsp:
        adj = adata.obsp["connectivities"].tocsr()
        adj = adj.copy()
        adj.setdiag(0)
        adj.eliminate_zeros()
        if adj.nnz > 0:
            return adj
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(coords)
    neigh = nn.kneighbors(coords, return_distance=False)[:, 1:]
    rows = np.repeat(np.arange(n_cells), neigh.shape[1])
    cols = neigh.reshape(-1)
    data = np.ones_like(rows, dtype=np.float32)
    return sparse.csr_matrix((data, (rows, cols)), shape=(n_cells, n_cells), dtype=np.float32)


def _sc_expression_candidate_graph(adata: ad.AnnData, k: int, n_components: int):
    from scipy import sparse
    from sklearn.decomposition import PCA, TruncatedSVD
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    n_cells = int(adata.n_obs)
    k = int(max(1, min(k, max(1, n_cells - 1))))
    n_components = int(max(2, min(n_components, n_cells - 1, max(2, adata.n_vars - 1))))
    x = adata.X
    if sparse.issparse(x):
        x = x.astype(np.float32).copy()
        x.data = np.log1p(np.clip(x.data, a_min=0.0, a_max=None))
        emb = TruncatedSVD(n_components=n_components, random_state=11).fit_transform(x)
    else:
        x = np.log1p(np.clip(np.asarray(x, dtype=np.float32), a_min=0.0, a_max=None))
        emb = PCA(n_components=n_components, random_state=11).fit_transform(x)
    emb = StandardScaler().fit_transform(emb.astype(np.float32, copy=False))
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(emb)
    neigh = nn.kneighbors(emb, return_distance=False)[:, 1:]
    rows = np.repeat(np.arange(n_cells), neigh.shape[1])
    cols = neigh.reshape(-1)
    data = np.ones_like(rows, dtype=np.float32)
    return sparse.csr_matrix((data, (rows, cols)), shape=(n_cells, n_cells), dtype=np.float32)


def _build_single_ccc_graph(
    adata: ad.AnnData,
    lr_programs,
    cache_path: pathlib.Path,
    topk: int,
    block_size: int,
    candidate_graph=None,
    program_cache_path: Optional[pathlib.Path] = None,
):
    from scipy import sparse

    graph = None
    ligand = None
    receptor = None
    if cache_path.exists():
        print(f"CCC cache hit: {cache_path}", flush=True)
        graph = sparse.load_npz(cache_path).tocsr()
    else:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Building CCC cache: {cache_path}", flush=True)
        lig_programs = [lig for lig, _ in lr_programs]
        rec_programs = [rcp for _, rcp in lr_programs]
        ligand = _program_expression(adata, lig_programs, block_size=block_size)
        receptor = _program_expression(adata, rec_programs, block_size=block_size)
        if candidate_graph is None:
            graph = _compute_topk_ccc_sparse(ligand=ligand, receptor=receptor, topk=topk, block_size=block_size)
        else:
            graph = _compute_candidate_ccc_sparse(ligand=ligand, receptor=receptor, candidate_graph=candidate_graph, topk=topk)
        graph = graph.tocsr()
        tmp_path = cache_path.with_name(f"{cache_path.name}.tmp.{os.getpid()}.npz")
        sparse.save_npz(tmp_path, graph)
        os.replace(tmp_path, cache_path)

    program_scores = None
    if program_cache_path is not None:
        program_cache_path = pathlib.Path(program_cache_path)
        if program_cache_path.exists():
            with np.load(program_cache_path, allow_pickle=False) as cached:
                cached_scores = cached["scores"]
                cached_indptr = cached["indptr"]
                cached_indices = cached["indices"]
            cache_matches = (
                cached_scores.shape == (graph.nnz, len(lr_programs))
                and np.array_equal(cached_indptr, graph.indptr)
                and np.array_equal(cached_indices, graph.indices)
            )
            if cache_matches:
                program_scores = np.asarray(cached_scores, dtype=np.float32)
                print(f"CCC program cache hit: {program_cache_path}", flush=True)
            else:
                print(f"CCC program cache mismatch; rebuilding: {program_cache_path}", flush=True)
        if program_scores is None:
            if ligand is None or receptor is None:
                lig_programs = [lig for lig, _ in lr_programs]
                rec_programs = [rcp for _, rcp in lr_programs]
                ligand = _program_expression(adata, lig_programs, block_size=block_size)
                receptor = _program_expression(adata, rec_programs, block_size=block_size)
            receiver = np.repeat(np.arange(graph.shape[0], dtype=np.int64), np.diff(graph.indptr))
            sender = graph.indices.astype(np.int64, copy=False)
            # Rows are attention receivers/queries and columns are senders/keys.
            # This is evaluated only for the already-selected sparse CCC edges.
            program_scores = (
                receptor[receiver].astype(np.float32, copy=False)
                * ligand[sender].astype(np.float32, copy=False)
            )
            program_cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_program_path = program_cache_path.with_name(
                f"{program_cache_path.name}.tmp.{os.getpid()}.npz"
            )
            np.savez_compressed(
                tmp_program_path,
                scores=program_scores,
                indptr=graph.indptr,
                indices=graph.indices,
            )
            os.replace(tmp_program_path, program_cache_path)
            print(f"Built CCC program cache: {program_cache_path}", flush=True)
    return graph, program_scores


def _mutual_topk_edge_list(features: np.ndarray, k: int, chunk: int = 4096) -> np.ndarray:
    """Unique mutual top-k cosine pairs over row features, as an (E, 2) array.

    A pair (i, j) is kept only when j is among i's top-k cosine neighbors AND
    i is among j's top-k neighbors (self always excluded).  Similarities are
    computed in chunked torch matmuls so no (n, n) matrix is materialized.
    """
    import torch as _torch

    feats = _torch.nn.functional.normalize(
        _torch.as_tensor(np.asarray(features), dtype=_torch.float32), dim=-1
    )
    num_cells = feats.shape[0]
    take = int(max(1, min(k, num_cells - 1)))
    device = "cuda" if _torch.cuda.is_available() else "cpu"
    feats = feats.to(device)
    neighbor_parts = []
    for start in range(0, num_cells, int(chunk)):
        end = min(start + int(chunk), num_cells)
        block = feats[start:end] @ feats.t()
        rows = _torch.arange(end - start, device=device)
        block[rows, rows + start] = -2.0  # mask self
        neighbor_parts.append(block.topk(take, dim=-1).indices.cpu())
    neighbors = _torch.cat(neighbor_parts)  # (n, k)
    src = _torch.arange(num_cells).unsqueeze(1).expand_as(neighbors).reshape(-1)
    dst = neighbors.reshape(-1)
    lo = _torch.minimum(src, dst)
    hi = _torch.maximum(src, dst)
    keys = lo * num_cells + hi
    unique_keys, counts = _torch.unique(keys, return_counts=True)
    mutual = unique_keys[counts >= 2]
    edges = _torch.stack([mutual // num_cells, mutual % num_cells], dim=1)
    return edges.numpy().astype(np.int64)


def _match_st_to_sc_top1(
    query_data: ad.AnnData,
    ref_data: ad.AnnData,
    *,
    n_components: int = 50,
    chunk: int = 4096,
    return_similarity: bool = False,
):
    """Top-1 SC match for every ST cell (PCA + cosine on shared genes).

    Many ST cells may map to the same SC cell (no dedup).  ST features come
    from the holdout-zeroed expression, SC features from the complete
    reference, so no held-out ST truth is ever read.
    """
    from scipy import sparse as _sparse
    from sklearn.decomposition import TruncatedSVD

    if list(map(str, query_data.var_names)) != list(map(str, ref_data.var_names)):
        raise ValueError("ST-aligned matching requires identical var order in query/ref.")

    def _log1p_dense(adata):
        x = adata.X
        if _sparse.issparse(x):
            x = x.astype(np.float32).copy()
            x.data = np.log1p(np.clip(x.data, a_min=0.0, a_max=None))
            return np.asarray(x.todense(), dtype=np.float32)
        return np.log1p(np.clip(np.asarray(x, dtype=np.float32), a_min=0.0, a_max=None))

    st_x = _log1p_dense(query_data)
    sc_x = _log1p_dense(ref_data)
    n_comp = int(max(2, min(n_components, st_x.shape[1] - 1, (st_x.shape[0] + sc_x.shape[0]) - 1)))
    pca = TruncatedSVD(n_components=n_comp, random_state=11)
    joint = pca.fit_transform(np.vstack([st_x, sc_x])).astype(np.float32)
    st_emb = joint[: st_x.shape[0]]
    sc_emb = joint[st_x.shape[0] :]

    import torch as _torch

    st_t = _torch.nn.functional.normalize(_torch.as_tensor(st_emb), dim=-1)
    sc_t = _torch.nn.functional.normalize(_torch.as_tensor(sc_emb), dim=-1)
    device = "cuda" if _torch.cuda.is_available() else "cpu"
    st_t, sc_t = st_t.to(device), sc_t.to(device)
    match_parts = []
    sim_parts = []
    for start in range(0, st_t.shape[0], int(chunk)):
        block = st_t[start : start + int(chunk)] @ sc_t.t()
        best = block.max(dim=-1)
        match_parts.append(best.indices.cpu())
        sim_parts.append(best.values.cpu())
    match_idx = _torch.cat(match_parts).numpy().astype(np.int64)
    if return_similarity:
        return match_idx, _torch.cat(sim_parts).numpy().astype(np.float32)
    return match_idx


def build_ccc_cache_bundle(
    args: argparse.Namespace,
    query_data: ad.AnnData,
    ref_data: ad.AnnData,
    oracle_query_data: Optional[ad.AnnData] = None,
):
    from scipy import sparse
    from sklearn.decomposition import TruncatedSVD

    if not (
        bool(getattr(args, "use_ccc_bias", False))
        or bool(getattr(args, "flow_cell_attention", False))
        or float(getattr(args, "lambda_pull", 0.0)) > 0
    ):
        # Stage-4 flow attention still consumes the CCC/LR program cache
        # (edges, program scores, ligand-side activity) even though the
        # scalar bias itself is disabled; the pull-graph builder rides the
        # same bundle.
        return None
    lr_programs = load_ensembl_lr_programs(args.ccc_ensembl_interaction_cache, query_data.var_names)
    if len(lr_programs) < int(args.ccc_min_pairs):
        raise ValueError(
            f"Only {len(lr_programs)} matched LR programs for CCC bias; "
            f"need at least {int(args.ccc_min_pairs)}."
        )
    cache_dir = pathlib.Path(args.ccc_cache_dir) if args.ccc_cache_dir is not None else pathlib.Path(args.out_dir) / "ccc_cache"
    st_candidate_k = int(args.ccc_st_candidate_k) if int(args.ccc_st_candidate_k) > 0 else int(args.n_neighbors)
    sc_candidate_k = int(args.ccc_sc_candidate_k) if int(args.ccc_sc_candidate_k) > 0 else int(args.n_neighbors)
    st_candidate_graph = None
    sc_candidate_graph = None
    if args.ccc_st_candidate_mode == "spatial":
        st_candidate_graph = _st_spatial_candidate_graph(query_data, k=st_candidate_k)
    if args.ccc_sc_candidate_mode == "expression":
        sc_candidate_graph = _sc_expression_candidate_graph(
            ref_data,
            k=sc_candidate_k,
            n_components=int(args.ccc_sc_pca_n_components),
        )
    tag = str(args.ccc_cache_name)
    keep_program_scores = bool(
        getattr(args, "ccc_program_head_bias", False)
        or getattr(args, "niche_regulon_bridge", False)
        or getattr(args, "flow_cell_attention", False)
        or (
            getattr(args, "flow_linear_bottleneck", False)
            and getattr(args, "flow_structure", "full") == "full"
        )
    )
    st_cache, st_program_scores = _build_single_ccc_graph(
        query_data,
        lr_programs,
        cache_dir / f"{tag}_st.npz",
        topk=args.ccc_topk,
        block_size=args.ccc_block_size,
        candidate_graph=st_candidate_graph,
        program_cache_path=(cache_dir / f"{tag}_st_programs.npz") if keep_program_scores else None,
    )
    if oracle_query_data is not None and keep_program_scores:
        oracle_ligand = _program_expression(
            oracle_query_data,
            [lig for lig, _ in lr_programs],
            block_size=args.ccc_block_size,
        )
        oracle_receptor = _program_expression(
            oracle_query_data,
            [rcp for _, rcp in lr_programs],
            block_size=args.ccc_block_size,
        )
        receiver = np.repeat(
            np.arange(st_cache.shape[0], dtype=np.int64), np.diff(st_cache.indptr)
        )
        sender = st_cache.indices.astype(np.int64, copy=False)
        st_program_scores = oracle_receptor[receiver] * oracle_ligand[sender]
        print("ORACLE LEAKAGE: ST flow program scores use held-out expression.", flush=True)
    st_aligned_ligand = None
    st_aligned_receptor = None
    if bool(getattr(args, "st_aligned_sc_prior", False)):
        # ST-aligned SC communication prior: every ST cell adopts the LR
        # program activity of its top-1 most-similar SC cell.  Interaction
        # topology stays the ST spatial candidate graph; the SC block and the
        # model are untouched.
        sc_ligand = _program_expression(
            ref_data, [lig for lig, _ in lr_programs], block_size=args.ccc_block_size
        )
        sc_receptor = _program_expression(
            ref_data, [rcp for _, rcp in lr_programs], block_size=args.ccc_block_size
        )
        match_idx, match_cos = _match_st_to_sc_top1(
            query_data, ref_data, n_components=50, return_similarity=True
        )
        st_aligned_ligand = sc_ligand[match_idx]
        st_aligned_receptor = sc_receptor[match_idx]
        match_counts = np.bincount(match_idx, minlength=ref_data.n_obs)
        matched_sc = int((match_counts > 0).sum())
        top5_share = float(
            np.sort(match_counts)[-5:].sum() / max(1, match_counts.sum())
        )
        qc_path = cache_dir / f"{tag}_st_aligned_qc.npz"
        np.savez_compressed(
            qc_path,
            match_idx=match_idx,
            top1_cos=match_cos,
            match_counts=match_counts,
        )
        print(
            f"ST-aligned QC | top1 cos p5/50/95/max = "
            f"{float(np.percentile(match_cos, 5)):.4f}/"
            f"{float(np.percentile(match_cos, 50)):.4f}/"
            f"{float(np.percentile(match_cos, 95)):.4f}/"
            f"{float(match_cos.max()):.4f} | unique SC matched = "
            f"{matched_sc}/{ref_data.n_obs} | max count = "
            f"{int(match_counts.max())} | top-5 SC share = {top5_share:.3f} | "
            f"saved {qc_path}",
            flush=True,
        )
        aligned_cache = cache_dir / f"{tag}_st_aligned.npz"
        if aligned_cache.exists():
            st_cache = sparse.load_npz(aligned_cache).tocsr()
        else:
            st_cache = _compute_candidate_ccc_sparse(
                ligand=st_aligned_ligand,
                receptor=st_aligned_receptor,
                candidate_graph=st_candidate_graph,
                topk=args.ccc_topk,
            ).tocsr()
            aligned_cache.parent.mkdir(parents=True, exist_ok=True)
            tmp_aligned = aligned_cache.with_name(
                f"{aligned_cache.name}.tmp.{os.getpid()}.npz"
            )
            sparse.save_npz(tmp_aligned, st_cache)
            os.replace(tmp_aligned, aligned_cache)
        receiver = np.repeat(
            np.arange(st_cache.shape[0], dtype=np.int64), np.diff(st_cache.indptr)
        )
        sender = st_cache.indices.astype(np.int64, copy=False)
        if keep_program_scores:
            # Rows are receivers/queries, columns are senders/keys.
            st_program_scores = (
                st_aligned_receptor[receiver] * st_aligned_ligand[sender]
            )
        print(
            f"ST-aligned SC prior: matched {query_data.n_obs} ST cells to "
            f"{int(np.unique(match_idx).size)} unique SC cells (top-1 cosine, "
            f"PCA 50); rebuilt ST graph edges={st_cache.nnz}",
            flush=True,
        )
    if args.ccc_sc_candidate_mode == "none":
        sc_cache = sparse.csr_matrix((ref_data.n_obs, ref_data.n_obs), dtype=np.float32)
        sc_program_scores = np.zeros((0, len(lr_programs)), dtype=np.float32) if keep_program_scores else None
    else:
        sc_cache, sc_program_scores = _build_single_ccc_graph(
            ref_data,
            lr_programs,
            cache_dir / f"{tag}_sc.npz",
            topk=args.ccc_topk,
            block_size=args.ccc_block_size,
            candidate_graph=sc_candidate_graph,
            program_cache_path=(cache_dir / f"{tag}_sc_programs.npz") if keep_program_scores else None,
        )
    st_outgoing_lr = None
    sc_outgoing_lr = None
    if (
        float(getattr(args, "lambda_out", 0.0)) > 0
        or bool(getattr(args, "flow_cell_attention", False))
        or (
            bool(getattr(args, "flow_linear_bottleneck", False))
            and float(getattr(args, "lambda_flow", 0.0)) > 0
        )
    ):
        ligand_programs = [lig for lig, _ in lr_programs]
        # Reuse the exact ligand-side activity used by CCC construction:
        # expm1(mean(log1p(nonnegative expression))) over each program's ligands.
        st_outgoing_lr = _program_expression(
            oracle_query_data if oracle_query_data is not None else query_data,
            ligand_programs,
            block_size=args.ccc_block_size,
        )
        sc_outgoing_lr = _program_expression(
            ref_data,
            ligand_programs,
            block_size=args.ccc_block_size,
        )
    st_inflow_lr = None
    sc_inflow_lr = None
    if bool(getattr(args, "flow_cell_attention", False)):
        # Stage-4 intrinsic receiver inflow: receptor-side program activity
        # computed from the same holdout-zeroed expression as the CCC cache,
        # never aggregated from sender-mixed edge scores.
        receptor_programs = [rcp for _, rcp in lr_programs]
        st_inflow_lr = _program_expression(
            query_data,
            receptor_programs,
            block_size=args.ccc_block_size,
        )
        sc_inflow_lr = _program_expression(
            ref_data,
            receptor_programs,
            block_size=args.ccc_block_size,
        )
    if (
        st_aligned_ligand is not None
        and bool(getattr(args, "flow_cell_attention", False))
    ):
        # Flow attention consumes the aligned SC-derived LR state for ST rows;
        # the intrinsic computation above is bypassed for the ST block.
        st_outgoing_lr = st_aligned_ligand
        st_inflow_lr = st_aligned_receptor
    st_pull_state_edges = None
    st_pull_niche_edges = None
    if float(getattr(args, "lambda_pull", 0.0)) > 0:
        # Positive-pair graphs from mutual top-k cosine neighbors.
        # state: PCA of the holdout-safe log1p observed expression.
        # niche: uniform spatial-kNN mean of the same expression (self
        # excluded), i.e. the niche representation; no labels, no clustering.
        pull_mode = str(getattr(args, "pull_label_mode", "both")).lower()
        pull_topk = int(getattr(args, "pull_topk", 10))
        x = query_data.X
        if sparse.issparse(x):
            x_log = x.astype(np.float32).copy()
            x_log.data = np.log1p(np.clip(x_log.data, a_min=0.0, a_max=None))
        else:
            x_log = sparse.csr_matrix(
                np.log1p(np.clip(np.asarray(x, dtype=np.float32), a_min=0.0, a_max=None))
            )
        if pull_mode in ("state", "both"):
            n_comp = int(max(2, min(50, x_log.shape[1] - 1, x_log.shape[0] - 1)))
            pca = TruncatedSVD(n_components=n_comp, random_state=11)
            state_feats = pca.fit_transform(x_log).astype(np.float32)
            st_pull_state_edges = _mutual_topk_edge_list(state_feats, pull_topk)
            print(
                f"Pull graph state: st_cells={x_log.shape[0]}, pca={n_comp}, "
                f"topk={pull_topk}, mutual_pairs={st_pull_state_edges.shape[0]}",
                flush=True,
            )
        if pull_mode in ("niche", "both"):
            # A dedicated wider spatial kNN (from coordinates) defines the
            # niche: the model's k=4 obsp graph makes neighbor averages so
            # degenerate (massive cosine ties) that mutual top-k is empty.
            niche_k = int(getattr(args, "pull_niche_k", 15))
            coords = np.asarray(query_data.obsm["spatial"], dtype=np.float32)[:, :2]
            from sklearn.neighbors import NearestNeighbors

            nn = NearestNeighbors(n_neighbors=min(niche_k + 1, x_log.shape[0]), metric="euclidean")
            nn.fit(coords)
            neigh = nn.kneighbors(coords, return_distance=False)[:, 1:]
            niche_rows = np.repeat(np.arange(x_log.shape[0]), neigh.shape[1])
            x_log_dense = np.asarray(x_log.todense(), dtype=np.float32)
            niche_feats = (
                np.zeros_like(x_log_dense)
                .astype(np.float32)
            )
            np.add.at(niche_feats, niche_rows, x_log_dense[neigh.reshape(-1)])
            niche_feats /= float(neigh.shape[1])
            valid_rows = np.ones(x_log.shape[0], dtype=bool)
            edges = _mutual_topk_edge_list(niche_feats, pull_topk)
            st_pull_niche_edges = edges
            print(
                f"Pull graph niche: st_cells={x_log.shape[0]}, "
                f"niche_k={niche_k}, topk={pull_topk}, "
                f"mutual_pairs={st_pull_niche_edges.shape[0]}",
                flush=True,
            )
    print(
        f"CCC bias enabled: matched_lr={len(lr_programs)}, "
        f"st_edges={st_cache.nnz}, sc_edges={sc_cache.nnz}, "
        f"st_mode={args.ccc_st_candidate_mode}, sc_mode={args.ccc_sc_candidate_mode}, "
        f"layers={args.ccc_bias_layers}",
        flush=True,
    )
    return {
        "st": st_cache,
        "sc": sc_cache,
        "st_program_scores": st_program_scores,
        "sc_program_scores": sc_program_scores,
        "st_outgoing_lr": st_outgoing_lr,
        "sc_outgoing_lr": sc_outgoing_lr,
        "st_inflow_lr": st_inflow_lr,
        "sc_inflow_lr": sc_inflow_lr,
        "st_pull_state_edges": st_pull_state_edges,
        "st_pull_niche_edges": st_pull_niche_edges,
        "n_st": int(query_data.n_obs),
        "n_lr": int(len(lr_programs)),
        "st_candidate_mode": str(args.ccc_st_candidate_mode),
        "sc_candidate_mode": str(args.ccc_sc_candidate_mode),
        "st_candidate_k": int(st_candidate_k),
        "sc_candidate_k": int(sc_candidate_k),
    }


def attach_ccc_bias_to_batch(data_dict: dict, ccc_cache: Optional[dict]) -> dict:
    if ccc_cache is None:
        return data_dict
    n_id = data_dict.get("n_id", None)
    if n_id is None:
        return data_dict
    if torch.is_tensor(n_id):
        global_idx = n_id.detach().cpu().numpy().astype(np.int64)
    else:
        global_idx = np.asarray(n_id, dtype=np.int64)
    n = int(global_idx.size)
    local_bias = np.zeros((n, n), dtype=np.float32)
    n_st = int(ccc_cache["n_st"])
    st_mask = global_idx < n_st
    sc_mask = ~st_mask
    if bool(st_mask.any()):
        st_idx = global_idx[st_mask]
        local_bias[np.ix_(st_mask, st_mask)] = ccc_cache["st"][st_idx][:, st_idx].toarray()
    if bool(sc_mask.any()):
        sc_idx = global_idx[sc_mask] - n_st
        local_bias[np.ix_(sc_mask, sc_mask)] = ccc_cache["sc"][sc_idx][:, sc_idx].toarray()
    device = None
    for value in data_dict.values():
        if torch.is_tensor(value):
            device = value.device
            break
    bias_t = torch.from_numpy(local_bias)
    if device is not None:
        bias_t = bias_t.to(device)
    data_dict["ccc_bias"] = bias_t
    if ccc_cache.get("st_outgoing_lr", None) is not None:
        program_count = int(ccc_cache["n_lr"])
        outgoing_target = np.zeros((n, program_count), dtype=np.float32)
        if bool(st_mask.any()):
            outgoing_target[st_mask] = ccc_cache["st_outgoing_lr"][global_idx[st_mask]]
        if bool(sc_mask.any()):
            outgoing_target[sc_mask] = ccc_cache["sc_outgoing_lr"][global_idx[sc_mask] - n_st]
        outgoing_target_t = torch.from_numpy(outgoing_target)
        if device is not None:
            outgoing_target_t = outgoing_target_t.to(device)
        data_dict["outgoing_lr_target"] = outgoing_target_t
        # Stage-4 flow attention consumes the same holdout-zeroed ligand-side
        # activity as a fixed input feature under a separate key.
        data_dict["ccc_outflow_activity"] = outgoing_target_t.clone()
    if ccc_cache.get("st_inflow_lr", None) is not None:
        program_count = int(ccc_cache["n_lr"])
        inflow_activity = np.zeros((n, program_count), dtype=np.float32)
        if bool(st_mask.any()):
            inflow_activity[st_mask] = ccc_cache["st_inflow_lr"][global_idx[st_mask]]
        if bool(sc_mask.any()):
            inflow_activity[sc_mask] = ccc_cache["sc_inflow_lr"][global_idx[sc_mask] - n_st]
        inflow_t = torch.from_numpy(inflow_activity)
        if device is not None:
            inflow_t = inflow_t.to(device)
        data_dict["ccc_inflow_activity"] = inflow_t
    if (
        ccc_cache.get("st_pull_state_edges", None) is not None
        or ccc_cache.get("st_pull_niche_edges", None) is not None
    ):
        # Induce the global ST positive-pair graphs on the batch cells and
        # express them in batch-local indices; drop-node remaps them further.
        local_of_st = np.full(n_st, -1, dtype=np.int64)
        st_positions = np.flatnonzero(st_mask)
        local_of_st[global_idx[st_mask]] = st_positions
        for source in ("state", "niche"):
            edges = ccc_cache.get(f"st_pull_{source}_edges", None)
            if edges is None:
                continue
            if edges.size:
                li = local_of_st[edges[:, 0]]
                lj = local_of_st[edges[:, 1]]
                keep = (li >= 0) & (lj >= 0)
                local_edges = np.stack([li[keep], lj[keep]], axis=1)
            else:
                local_edges = np.zeros((0, 2), dtype=np.int64)
            edge_t = torch.as_tensor(local_edges.T, dtype=torch.long)
            if device is not None:
                edge_t = edge_t.to(device)
            data_dict[f"pull_edge_index_{source}"] = edge_t
    if ccc_cache.get("st_program_scores", None) is not None:
        edge_parts = []
        score_parts = []

        def _append_local_program_edges(mask, group_idx, graph, program_scores):
            local_pos = np.flatnonzero(mask).astype(np.int64, copy=False)
            if local_pos.size == 0 or graph.nnz == 0:
                return
            global_to_local = {int(g): int(l) for g, l in zip(group_idx, local_pos)}
            for receiver_global, receiver_local in zip(group_idx, local_pos):
                start, end = graph.indptr[int(receiver_global)], graph.indptr[int(receiver_global) + 1]
                for edge_offset in range(start, end):
                    sender_local = global_to_local.get(int(graph.indices[edge_offset]))
                    if sender_local is None:
                        continue
                    edge_parts.append((int(receiver_local), sender_local))
                    score_parts.append(program_scores[edge_offset])

        if bool(st_mask.any()):
            _append_local_program_edges(
                st_mask,
                global_idx[st_mask],
                ccc_cache["st"],
                ccc_cache["st_program_scores"],
            )
        if bool(sc_mask.any()):
            _append_local_program_edges(
                sc_mask,
                global_idx[sc_mask] - n_st,
                ccc_cache["sc"],
                ccc_cache["sc_program_scores"],
            )
        if edge_parts:
            program_edge_index = torch.as_tensor(
                np.asarray(edge_parts, dtype=np.int64).T,
                dtype=torch.long,
            )
            program_score = torch.as_tensor(
                np.asarray(score_parts, dtype=np.float32),
                dtype=torch.float32,
            )
        else:
            program_edge_index = torch.empty((2, 0), dtype=torch.long)
            program_score = torch.empty((0, int(ccc_cache["n_lr"])), dtype=torch.float32)
        if device is not None:
            program_edge_index = program_edge_index.to(device)
            program_score = program_score.to(device)
        data_dict["ccc_program_edge_index"] = program_edge_index
        data_dict["ccc_program_score"] = program_score
    return data_dict


def _valid_coord_mask_tensor(coord: torch.Tensor) -> torch.Tensor:
    # Normalized ST coordinates can be negative; treat only the (-1, -1)
    # sentinel and non-finite values as invalid.
    coord = coord[..., :2]
    sentinel = torch.full_like(coord, -1.0)
    return torch.isfinite(coord).all(dim=-1) & (~torch.isclose(coord, sentinel).all(dim=-1))


def build_pos_from_xdict(x_dict):
    coord = x_dict.get("coord", None)
    if coord is None:
        return None
    if not torch.is_tensor(coord):
        coord = torch.as_tensor(coord)
    if coord.numel() > 0 and bool(_valid_coord_mask_tensor(coord).any().item()):
        return coord
    return None


def normalize_spatial_coords(coords: np.ndarray, mode: str = "none") -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"Expected coords shape [N, 2+], got {coords.shape}")

    coords = coords[:, :2].copy()
    valid_mask = np.isfinite(coords).all(axis=1)
    if mode == "none" or not np.any(valid_mask):
        return coords

    valid_coords = coords[valid_mask]
    if mode == "pc_l1":
        # Center the full slice, then scale by the maximum L1 radius.
        centroid = valid_coords.mean(axis=0, keepdims=True)
        centered = valid_coords - centroid
        l1_radius = np.abs(centered).sum(axis=1).max()
        if l1_radius > 0:
            centered = centered / l1_radius
        coords[valid_mask] = centered
        return coords

    if mode == "pc_l2":
        # Same point-cloud style normalization, but scale by the maximum L2 radius.
        centroid = valid_coords.mean(axis=0, keepdims=True)
        centered = valid_coords - centroid
        l2_radius = np.linalg.norm(centered, axis=1).max()
        if l2_radius > 0:
            centered = centered / l2_radius
        coords[valid_mask] = centered
        return coords

    if mode == "minmax_01":
        coord_min = valid_coords.min(axis=0, keepdims=True)
        coord_range = valid_coords.max(axis=0, keepdims=True) - coord_min
        coord_range[coord_range == 0] = 1.0
        coords[valid_mask] = (valid_coords - coord_min) / coord_range
        return coords

    raise ValueError(f"Unsupported coord_norm_mode: {mode}")


def build_spatial_dist_from_pos(pos, dist_metric: str = "l2", eps: float = 1e-6):
    if pos is None:
        return None
    if not torch.is_tensor(pos):
        pos = torch.as_tensor(pos)
    pos = pos[..., :2]
    valid_nodes = _valid_coord_mask_tensor(pos)
    if pos.dim() != 2 or not bool(valid_nodes.any().item()):
        return None

    valid_pair_mask = valid_nodes.unsqueeze(1) & valid_nodes.unsqueeze(0)
    # Build distances with a switchable L1/L2 metric and keep the same median normalization.
    p = 1 if dist_metric == "l1" else 2
    spatial_dist = torch.cdist(pos, pos, p=p)
    positive_dist = spatial_dist[valid_pair_mask & (spatial_dist > 0)]
    scale = positive_dist.median() if positive_dist.numel() > 0 else spatial_dist.new_tensor(1.0)
    scale = torch.clamp(scale, min=eps)
    spatial_dist = spatial_dist / scale
    spatial_dist = spatial_dist.masked_fill(~valid_pair_mask, 0.0)
    spatial_dist.fill_diagonal_(0.0)
    return spatial_dist


def build_noisy_masa_targets(
    pos: Optional[torch.Tensor],
    dist_metric: str = "l2",
    noise_rate: float = 0.1,
    mask_rate: float = 0.0,
    noise_std: float = 0.1,
    pair_mask_token: float = -100.0,
):
    # Keep noisy targets and pair labels on the exact same L1/L2 scale as dist_bias.
    clean_dist = build_spatial_dist_from_pos(pos, dist_metric=dist_metric)
    if clean_dist is None:
        return None, None

    device = clean_dist.device
    dtype = clean_dist.dtype
    n_nodes = clean_dist.size(0)
    if n_nodes <= 1:
        return clean_dist, None

    if not torch.is_tensor(pos):
        pos = torch.as_tensor(pos, device=device, dtype=dtype)
    else:
        pos = pos.to(device=device, dtype=dtype)
    valid_nodes = _valid_coord_mask_tensor(pos)
    valid_idx = torch.nonzero(valid_nodes, as_tuple=False).view(-1)
    if valid_idx.numel() <= 1:
        return clean_dist, None

    noise_rate = float(max(0.0, min(1.0, noise_rate)))
    mask_rate = float(max(0.0, min(1.0, mask_rate)))
    noisy_dist = clean_dist.clone()

    perm = valid_idx[torch.randperm(valid_idx.numel(), device=device)]
    num_noise = min(valid_idx.numel(), int(round(valid_idx.numel() * noise_rate)))
    num_mask = min(valid_idx.numel() - num_noise, int(round(valid_idx.numel() * mask_rate)))

    noise_nodes = perm[:num_noise]
    mask_nodes = perm[num_noise : num_noise + num_mask]

    pair_target_mask = torch.zeros((n_nodes, n_nodes), dtype=torch.bool, device=device)
    valid_pairs = valid_nodes[:, None] & valid_nodes[None, :]

    if noise_nodes.numel() > 0:
        noise_node_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
        noise_node_mask[noise_nodes] = True
        noise_pair_mask = (noise_node_mask[:, None] | noise_node_mask[None, :]) & valid_pairs

        positive_dist = clean_dist[valid_pairs & (clean_dist > 0)]
        if positive_dist.numel() > 0:
            base_scale = positive_dist.median()
        else:
            base_scale = clean_dist.new_tensor(1.0)

        noise = torch.randn_like(clean_dist) * (base_scale * float(max(noise_std, 0.0)))
        noise = 0.5 * (noise + noise.transpose(0, 1))
        noise.fill_diagonal_(0.0)
        noisy_dist = torch.where(noise_pair_mask, torch.clamp(clean_dist + noise, min=0.0), noisy_dist)
        pair_target_mask |= noise_pair_mask

    if mask_nodes.numel() > 0:
        mask_node_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
        mask_node_mask[mask_nodes] = True
        mask_pair_mask = (mask_node_mask[:, None] | mask_node_mask[None, :]) & valid_pairs
        noisy_dist = noisy_dist.masked_fill(mask_pair_mask, 0.0)
        pair_target_mask |= mask_pair_mask

    noisy_dist = noisy_dist.masked_fill(~valid_pairs, 0.0)
    noisy_dist.fill_diagonal_(0.0)

    pair_target_mask &= valid_pairs
    pair_target_mask.fill_diagonal_(False)
    if not bool(pair_target_mask.any().item()):
        return clean_dist, None

    pair_labels = torch.full(
        clean_dist.shape,
        float(pair_mask_token),
        dtype=clean_dist.dtype,
        device=device,
    )
    pair_labels[pair_target_mask] = clean_dist[pair_target_mask]
    return noisy_dist, pair_labels


def build_train_data(query_data: ad.AnnData, ref_data: ad.AnnData) -> ad.AnnData:
    st = query_data.copy()
    sc_adata = ref_data[:, st.var_names].copy()
    train_data = ad.concat(
        [st, sc_adata],
        axis=0,
        join="outer",
        label="source",
        keys=None,
        fill_value=0,
        index_unique=None,
    )

    import scipy.sparse as sp

    a_st = st.obsp["connectivities"].tocsr()
    a_sc = sp.csr_matrix((sc_adata.n_obs, sc_adata.n_obs))
    a_block = sp.block_diag((a_st, a_sc), format="csr")
    train_data.obsp["connectivities"] = a_block
    train_data.obsp["spatial_connectivities"] = a_block
    train_data.obs["source"] = train_data.obs["source"].astype(str).map({"0": False, "1": True})
    return train_data


def resolve_amp_config(device: Union[str, torch.device], amp_mode: str) -> dict:
    device_str = str(device)
    device_type = "cuda" if device_str.startswith("cuda") and torch.cuda.is_available() else "cpu"
    amp_enabled = device_type == "cuda" and amp_mode in {"fp16", "bf16"}
    amp_dtype = None
    model_dtype = None
    if amp_mode == "fp16":
        amp_dtype = torch.float16
    elif amp_mode == "bf16":
        amp_dtype = torch.bfloat16
    elif amp_mode == "full_fp16":
        model_dtype = torch.float16
    elif amp_mode == "full_bf16":
        model_dtype = torch.bfloat16
    return {
        "device_type": device_type,
        "amp_enabled": amp_enabled,
        "amp_mode": amp_mode,
        "amp_dtype": amp_dtype,
        "model_dtype": model_dtype,
        "use_grad_scaler": device_type == "cuda" and amp_mode == "fp16",
    }


def amp_autocast_context(amp_cfg: dict):
    if not amp_cfg["amp_enabled"] or amp_cfg["amp_dtype"] is None:
        return nullcontext()
    return torch.autocast(device_type=amp_cfg["device_type"], dtype=amp_cfg["amp_dtype"])


def maybe_synchronize(device: Union[str, torch.device]) -> None:
    device_str = str(device)
    if device_str.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def inference(
    model,
    dataloader,
    split,
    device,
    dist_metric="l2",
    order_required=False,
    ccc_cache: Optional[dict] = None,
    max_batches: int = 0,
    no_dist_bias: bool = False,
):
    if order_required and split:
        warnings.warn("When cell order required to be preserved, dataset split will be ignored.")
    with torch.no_grad():
        model.eval()
        epoch_loss = []
        order_list = []
        pred = []
        for batch_idx, data_dict in enumerate(dataloader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            if not order_required and split and np.sum(data_dict["split"] == split) == 0:
                continue
            input_dict = move_xdict_to_device(data_dict, device)
            input_dict = attach_ccc_bias_to_batch(input_dict, ccc_cache)
            x_dict = XDict(input_dict)
            pos = build_pos_from_xdict(x_dict)
            # Pass the precomputed normalized spatial distance explicitly so inference matches the chosen metric.
            spatial_dist = build_spatial_dist_from_pos(pos, dist_metric=dist_metric)
            if no_dist_bias:
                pos = None
                spatial_dist = None
                x_dict.pop("coord", None)
            out_dict, loss = model(x_dict, data_dict["gene_list"], pos=pos, dist_bias=spatial_dist)
            if loss is not None:
                epoch_loss.append(loss.item())

            if order_required:
                seed_size = int(input_dict.get("seed_size", out_dict["pred"].shape[0]))
                pred.append(out_dict["pred"][:seed_size])
                order_list.append(input_dict["order_list"][:seed_size])
            else:
                pred.append(out_dict["pred"])

        pred = torch.cat(pred) if pred else torch.empty((0, 0), device=device)
        if order_required and order_list:
            order = torch.cat(order_list)
            order.scatter_(0, order.clone(), torch.arange(order.shape[0], device=order.device))
            pred = pred[order]

        loss_value = float("nan") if not epoch_loss else sum(epoch_loss) / len(epoch_loss)
        return {"pred": pred, "loss": loss_value}


def split_train_valid(adata: ad.AnnData, mode: str = "legacy") -> None:
    adata.obs["split"] = "train"
    if mode == "batch_last_10pct":
        batch_values = adata.obs["batch"].astype(str)
        for batch_name in pd.Index(batch_values).unique():
            batch_index = adata.obs.index[batch_values == str(batch_name)]
            valid_count = max(1, int(round(0.1 * len(batch_index))))
            valid_count = min(valid_count, max(len(batch_index) - 1, 1))
            adata.obs.loc[batch_index[-valid_count:], "split"] = "valid"
        return
    if len(adata.obs["batch"].unique()) == 1:
        valid_idx = adata.obs.index[-int(round(0.1 * adata.n_obs)) :]
        adata.obs.loc[valid_idx, "split"] = "valid"
    else:
        last_batch = adata.obs["batch"].iloc[-1]
        adata.obs.loc[adata.obs["batch"] == last_batch, "split"] = "valid"


def prepare_datasets(args: argparse.Namespace):
    query_data = ad.read_h5ad(args.query_data)
    query_data.obs_names = "st_" + query_data.obs_names.astype(str)
    query_data.obs["batch"] = "st_" + query_data.obs["batch"].astype(str)

    with open(args.config, "r", encoding="utf-8") as openfile:
        base_config = json.load(openfile)

    gene_mask = [gene in base_config["gene_list"] for gene in query_data.var_names]
    query_data = query_data[:, gene_mask].copy()

    ref_data = ad.read_h5ad(args.ref_data)
    ref_data.obs_names = "sc_" + ref_data.obs_names.astype(str)
    ref_data.obs["batch"] = "sc_" + ref_data.obs["batch"].astype(str)

    shared_genes = [gene for gene in query_data.var_names if gene in ref_data.var_names]
    query_data = query_data[:, shared_genes].copy()
    ref_data = ref_data[:, shared_genes].copy()
    oracle_query_data = query_data.copy() if args.flow_oracle_st_truth else None

    use_current_covariates = args.use_data_covariates or args.use_batch_covariate
    ref_data.obs["modality"] = "sc" if use_current_covariates else 3
    ref_data.obs["specie"] = 5
    ref_data.obs["assay"] = 12
    ref_data.obs["source"] = True

    query_data.obs["modality"] = "st" if use_current_covariates else 4
    query_data.obs["specie"] = 5
    query_data.obs["assay"] = 7
    query_data.obs["source"] = False

    set_seed(args.random_seed, deterministic=args.deterministic)
    gene_10fold = np.load(args.mask_gene, allow_pickle=True)
    if args.mask_fold_index < 0 or args.mask_fold_index >= len(gene_10fold):
        raise ValueError(
            f"mask_fold_index={args.mask_fold_index} is out of range for "
            f"{args.mask_gene} with {len(gene_10fold)} folds."
        )
    target_genes = gene_10fold[args.mask_fold_index]
    query_data.obsm["truth"] = query_data[:, target_genes].X.toarray()
    query_data[:, target_genes].X = 0
    if {"x_slide_mm", "y_slide_mm"}.issubset(query_data.obs.columns):
        raw_spatial = query_data.obs[["x_slide_mm", "y_slide_mm"]].to_numpy()
    else:
        raw_spatial = query_data.obs[["x_FOV_px", "y_FOV_px"]].to_numpy()
    query_data.obsm["spatial"] = normalize_spatial_coords(raw_spatial, mode=args.coord_norm_mode)

    sc.pp.neighbors(
        query_data,
        use_rep="spatial",
        n_neighbors=args.n_neighbors,
        method="umap",
    )

    split_train_valid(query_data, mode=args.validation_split_mode)
    split_train_valid(ref_data, mode=args.validation_split_mode)

    train_data = build_train_data(query_data, ref_data)
    if args.use_batch_covariate:
        args.covariate_fields = ("batch",)
    elif args.use_data_covariates:
        args.covariate_fields = ("batch", "platform", "modality")
    else:
        args.covariate_fields = ()
    if args.covariate_fields:
        args.covariate_categories = {
            field: sorted(train_data.obs[field].astype(str).unique().tolist())
            for field in args.covariate_fields
            if field in train_data.obs
        }
    else:
        args.covariate_fields = ()
        args.covariate_categories = {}
    ccc_cache = build_ccc_cache_bundle(
        args,
        query_data=query_data,
        ref_data=ref_data,
        oracle_query_data=oracle_query_data,
    )
    query_genes = [gene for gene in query_data.var_names if gene not in target_genes]
    return base_config, query_data, ref_data, train_data, target_genes, query_genes, ccc_cache


def build_loader_cfg(adata, args, observed_genes, order_required=False, train_split="train", val_split="valid"):
    return {
        "adata": adata,
        "split_field": "split",
        "train_split": train_split,
        "val_split": val_split,
        "batch_size": args.batch_size,
        "num_neighbors": args.n_neighbors,
        "max_nodes_per_batch": int(args.batch_size * 5),
        "spatial_adj_key": "spatial_connectivities",
        "spatial_k": args.n_neighbors,
        "sc_knn_k": args.n_neighbors,
        "sc_pca_n_components": 10,
        "sc_log1p": True,
        "num_workers": 0,
        "neighbor_agg": "mean",
        "observed_genes": observed_genes,
        "order_required": order_required,
        "balance_modalities": bool(getattr(args, "balance_modalities", False)),
        "covariate_fields": tuple(getattr(args, "covariate_fields", ())),
        "covariate_categories": getattr(args, "covariate_categories", {}),
    }


def predict(
    model,
    adata: ad.AnnData,
    args: argparse.Namespace,
    target_genes: List[str],
    observed_genes,
    inference_config: Optional[dict] = None,
    covariate_fields: List[str] = None,
    ensembl_auto_conversion: bool = True,
    device: Union[str, torch.device] = "cpu",
    ccc_cache: Optional[dict] = None,
):
    model.to(device)
    pipeline_config = {
        "max_eval_batch_size": 100000,
    }
    if inference_config:
        pipeline_config.update(inference_config)
    adata = common_preprocess(adata, 0, covariate_fields, ensembl_auto_conversion)
    print(f"After filtering, {adata.shape[1]} genes remain.")

    pred_loader_cfg = build_loader_cfg(
        adata=adata,
        args=args,
        observed_genes=observed_genes,
        order_required=True,
        train_split=["train", "valid"],
        val_split=[],
    )
    pred_loader = build_dataloaders(pred_loader_cfg, use_gp=bool(getattr(args, "use_gp", False)))["train"]
    pred = inference(
        model,
        pred_loader,
        None,
        device,
        dist_metric=args.dist_metric,
        order_required=True,
        ccc_cache=ccc_cache,
        # Final imputation metrics must cover every query cell. The training-time
        # validation cap is intentionally not reused here.
        max_batches=0,
        no_dist_bias=bool(args.no_dist_bias),
    )["pred"]

    if "target_genes" in pipeline_config:
        target_mask = torch.tensor(
            [adata.var.index.get_loc(gene) for gene in pipeline_config["target_genes"]],
            dtype=torch.long,
            device=pred.device,
        )
        pred = pred[:, target_mask]
    return pred


def load_baseline_and_freeze_for_grn(model, ckpt_path) -> dict:
    """Two-stage GRN residual: load baseline weights, freeze all non-GRN params.

    The baseline checkpoint was saved without the GRN branch, so the only
    tolerable missing keys are the GRN buffers/parameters; any unexpected key
    or any other missing key is a hard error.
    """
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint)
    incompatible = model.load_state_dict(state, strict=False)
    allowed_missing = all(
        name.startswith(("grn_", "_grn_")) for name in incompatible.missing_keys
    )
    if incompatible.unexpected_keys or not allowed_missing:
        raise ValueError(
            "Baseline checkpoint does not match the two-stage model: "
            f"unexpected={list(incompatible.unexpected_keys)[:5]}, "
            f"missing={list(incompatible.missing_keys)[:5]}"
        )
    frozen_count = 0
    trainable_scalars = 0
    for name, parameter in model.named_parameters():
        if name.startswith("grn_"):
            trainable_scalars += parameter.numel()
            continue
        parameter.requires_grad_(False)
        frozen_count += 1
    return {
        "frozen_param_tensors": frozen_count,
        "trainable_scalars": trainable_scalars,
        "ckpt_best_epoch": checkpoint.get("best_epoch", None) if isinstance(checkpoint, dict) else None,
    }


def grn_residual_final_summary(model, args: argparse.Namespace) -> dict:
    """Final alpha / edge-strength statistics for the GRN decoder residual."""
    if not bool(getattr(args, "grn_decoder_residual", False)):
        return {}
    with torch.no_grad():
        theta = model.grn_edge_strength.detach().float()
        strength = torch.nn.functional.softplus(theta)
    init_value = float(np.log(2.0))  # softplus(0); kept as a python float for device safety
    return {
        "grn_residual_alpha": float(model.grn_residual_alpha.detach().item()),
        "grn_edges": int(theta.numel()),
        "grn_edge_updated_frac": float((theta.abs() > 1e-8).float().mean().item()),
        "grn_edge_strength_mean": float(strength.mean().item()),
        "grn_edge_strength_init": init_value,
        "grn_edge_strength_rel_change": float(
            ((strength - init_value).norm() / (strength.numel() ** 0.5 * init_value)).item()
        ),
    }


def flow_attention_final_summary(model, args: argparse.Namespace) -> dict:
    """Final per-layer/per-head gate matrix and program-weight drift for Stage 4."""
    if not bool(getattr(args, "flow_cell_attention", False)):
        return {}
    summary = {"flow_attn_mode": str(getattr(args, "flow_attn_mode", "gated"))}
    layer_indices = list(getattr(model.gcat_encoder, "flow_attn_layer_indices", []))
    if getattr(args, "flow_attn_mode", "gated") == "gated":
        summary["flow_attn_gate_layers"] = {
            int(i): [
                round(float(v), 6)
                for v in model.flow_attn_gate.detach().float()[i].tolist()
            ]
            for i in layer_indices
        }
    with torch.no_grad():
        weight = model.flow_attn_program_weight.detach().float()
        init_value = float(weight.shape[1] ** -0.5)
        init_norm = init_value * float(weight.numel() ** 0.5)
        summary["flow_attn_program_weight_init"] = init_value
        summary["flow_attn_program_weight_rel_change"] = float(
            ((weight - init_value).norm() / max(init_norm, 1e-12)).item()
        )
        summary["flow_attn_program_weight_head_mean"] = [
            round(float(head_mean), 6) for head_mean in weight.mean(dim=1).tolist()
        ]
    if bool(getattr(args, "flow_stage5", False)):
        summary["flow_stage5_beta"] = float(model.flow_reg_update_beta.detach().item())
    return summary


def main(args: argparse.Namespace) -> None:
    args.out_dir = pathlib.Path(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not 0.0 <= args.st_block_gene_mask_rate <= 1.0:
        raise ValueError("st_block_gene_mask_rate must be in [0, 1].")
    if args.lambda_masked_zmse < 0:
        raise ValueError("lambda_masked_zmse must be >= 0.")
    if args.lambda_masked_zmse > 0 and args.st_block_gene_mask_rate <= 0:
        raise ValueError("lambda_masked_zmse > 0 requires st_block_gene_mask_rate > 0.")
    if args.niche_bridge_stage == "off":
        args.niche_regulon_bridge = False
        args.niche_regulon_bridge_path = None
    if args.niche_regulon_bridge:
        if not args.use_ccc_bias:
            raise ValueError("niche_regulon_bridge requires the existing scalar CCC pipeline.")
        if args.ccc_program_head_bias:
            raise ValueError(
                "niche_regulon_bridge and ccc_program_head_bias are mutually exclusive in this experiment."
            )
        if args.niche_regulon_bridge_path is None or not args.niche_regulon_bridge_path.is_file():
            raise FileNotFoundError(
                f"NicheNet bridge NPZ not found: {args.niche_regulon_bridge_path}"
            )
    if args.lambda_out > 0 and not args.use_ccc_bias:
        raise ValueError("lambda_out > 0 requires the existing scalar CCC/LR program pipeline.")
    if args.flow_cell_attention:
        if args.use_ccc_bias and args.ccc_bias_layers == "last" and args.flow_attn_layers in ("last",):
            raise ValueError(
                "flow_cell_attention with scalar CCC on the same last layer is "
                "ambiguous; pass --no_ccc_bias or move --ccc_bias_layers off the "
                "flow layers (e.g. --ccc_bias_layers 0,1,2)."
            )
        if args.ccc_program_head_bias or args.niche_regulon_bridge:
            raise ValueError(
                "flow_cell_attention requires Experiment B and the NicheNet bridge to be disabled."
            )
        if args.flow_linear_bottleneck:
            raise ValueError("flow_cell_attention and flow_linear_bottleneck are mutually exclusive.")
        if args.lambda_out > 0:
            raise ValueError(
                "flow_cell_attention and direct lambda_out supervision are mutually exclusive."
            )
        if args.flow_oracle_st_truth:
            raise ValueError(
                "flow_cell_attention must stay leakage-free; flow_oracle_st_truth is not allowed."
            )
        if args.flow_stage5 and args.flow_attn_mode != "gated":
            raise ValueError("flow_stage5 requires the gated interpolation mode.")
    elif args.flow_attn_mode != "gated":
        raise ValueError("flow_attn_mode requires --flow_cell_attention.")
    elif args.flow_stage5:
        raise ValueError("flow_stage5 requires --flow_cell_attention.")
    if args.grn_decoder_residual:
        if args.grn_sign_path is None or not args.grn_sign_path.is_file():
            raise FileNotFoundError(
                f"Signed GRN asset not found: {args.grn_sign_path}"
            )
        if args.flow_cell_attention:
            raise ValueError(
                "grn_decoder_residual targets the scalar-CCC baseline; disable flow attention."
            )
    if args.grn_frozen_baseline:
        if not args.grn_decoder_residual:
            raise ValueError("grn_frozen_baseline requires --grn_decoder_residual.")
        if args.grn_baseline_ckpt is None or not args.grn_baseline_ckpt.is_file():
            raise FileNotFoundError(
                f"Baseline checkpoint not found: {args.grn_baseline_ckpt}"
            )
    if args.flow_linear_bottleneck:
        if not args.use_ccc_bias:
            raise ValueError("flow_linear_bottleneck requires scalar CCC.")
        if args.ccc_program_head_bias or args.niche_regulon_bridge:
            raise ValueError(
                "flow_linear_bottleneck requires Experiment B and NicheNet bridge to be disabled."
            )
        if args.lambda_out > 0:
            raise ValueError("flow_linear_bottleneck and direct lambda_out supervision are mutually exclusive.")
        if args.flow_graph_mask_path is not None and not args.flow_graph_mask_path.is_file():
            raise FileNotFoundError(f"Flow graph mask NPZ not found: {args.flow_graph_mask_path}")
        if args.flow_oracle_st_truth and args.flow_supervision != "all":
            raise ValueError("flow_oracle_st_truth requires --flow_supervision all.")
    elif args.flow_graph_mask_path is not None:
        raise ValueError("flow_graph_mask_path requires --flow_linear_bottleneck.")
    elif args.flow_oracle_st_truth:
        raise ValueError("flow_oracle_st_truth requires --flow_linear_bottleneck.")
    run_start_time = time.perf_counter()
    amp_cfg = resolve_amp_config(args.device, args.amp_mode)
    maybe_synchronize(args.device)
    if amp_cfg["device_type"] == "cuda":
        torch.cuda.reset_peak_memory_stats(args.device)
    base_config, query_data, ref_data, train_data, target_genes, query_genes, ccc_cache = prepare_datasets(args)

    model_class = load_gcat_former()
    model_config = {
        "mask_node_rate": 0.95,
        "mask_feature_rate": 0.25,
        "st_block_gene_mask_rate": float(args.st_block_gene_mask_rate),
        "drop_node_rate": float(args.drop_node_rate),
        "max_batch_size": 70000,
        "pair_loss_weight": float(args.pair_loss_weight),
        "pair_mask_token": float(args.pair_mask_token),
        "lambda_reg": float(args.lambda_reg),
        "reg_activity_std_floor": float(args.reg_activity_std_floor),
        "lambda_masked_zmse": float(args.lambda_masked_zmse),
        "lambda_out": float(args.lambda_out),
        "outgoing_lr_std_floor": float(args.outgoing_lr_std_floor),
        "lambda_pull": float(args.lambda_pull),
        "pull_label_mode": str(args.pull_label_mode),
        "flow_linear_bottleneck": bool(args.flow_linear_bottleneck),
        "lambda_flow": float(args.lambda_flow),
        "flow_structure": str(args.flow_structure),
        "flow_supervision": str(args.flow_supervision),
        "flow_graph_mask_path": (
            str(args.flow_graph_mask_path) if args.flow_graph_mask_path is not None else None
        ),
        "flow_cell_attention": bool(args.flow_cell_attention),
        "flow_attn_layers": str(args.flow_attn_layers),
        "flow_gate_hidden_dim": int(args.flow_gate_hidden_dim),
        "flow_gate_init": float(args.flow_gate_init),
        "flow_attn_mode": str(args.flow_attn_mode),
        "flow_stage5": bool(args.flow_stage5),
        "grn_decoder_residual": bool(args.grn_decoder_residual),
        "grn_sign_path": (
            str(args.grn_sign_path) if args.grn_sign_path is not None else None
        ),
        "distance_metric": str(args.dist_metric),
        "gene_token_mode": str(args.gene_token_mode),
        "gene_module_num": str(args.gene_module_num),
        "gene_module_init_path": str(args.gene_module_init_path) if args.gene_module_init_path is not None else None,
        "dynamic_gene_modules": bool(args.dynamic_gene_modules),
        "prior_strength": float(args.prior_strength),
        "off_prior_init_std": float(args.off_prior_init_std),
        "assignment_temperature": float(args.assignment_temperature),
        "disable_cell_transformer": bool(args.disable_cell_transformer),
        "disable_gene_module_attention": bool(args.disable_gene_module_attention),
        "use_ccc_bias": bool(args.use_ccc_bias),
        "ccc_bias_layers": str(args.ccc_bias_layers),
        "ccc_bias_trainable": bool(args.ccc_bias_trainable),
        "ccc_bias_init_scale": float(args.ccc_bias_init_scale),
        "ccc_program_head_bias": bool(args.ccc_program_head_bias),
        "ccc_program_num": int(ccc_cache["n_lr"]) if (
            args.ccc_program_head_bias
            or args.niche_regulon_bridge
            or args.lambda_out > 0
            or args.flow_linear_bottleneck
            or args.flow_cell_attention
        ) else 0,
        "niche_regulon_bridge": bool(args.niche_regulon_bridge),
        "niche_regulon_bridge_path": (
            str(args.niche_regulon_bridge_path) if args.niche_regulon_bridge_path is not None else None
        ),
        "niche_bridge_mode": str(args.niche_bridge_mode),
        "niche_bridge_stage": str(args.niche_bridge_stage),
    }
    pipeline_config = {
        "lr": float(args.lr),
        "wd": float(args.weight_decay),
        "scheduler": "plat",
        "epochs": args.epoch,
        "patience": 5,
        "workers": 0,
        "warmup_epochs": max(int(args.warmup_epochs), 0),
    }

    model_kwargs = dict(base_config)
    for key in (
        "hidden_dim",
        "num_layers",
        "latent_dim",
        "decoder_hidden_dim",
        "decoder_layers",
        "dropout",
        "num_heads",
        "ffn_mult",
    ):
        value = getattr(args, key)
        if value is not None:
            model_kwargs[key] = value
    model_kwargs.update(model_config)
    if args.use_batch_covariate or args.use_data_covariates:
        for field, count_name in (
            ("batch", "batch_count"),
            ("platform", "platform_count"),
            ("modality", "modality_count"),
        ):
            model_kwargs[count_name] = len(getattr(args, "covariate_categories", {}).get(field, ()))
        model_kwargs["dataset_count"] = 0
    model = model_class(**model_kwargs)
    if args.grn_frozen_baseline:
        freeze_stats = load_baseline_and_freeze_for_grn(model, args.grn_baseline_ckpt)
        print(
            "Two-stage GRN residual | baseline ckpt: "
            f"{args.grn_baseline_ckpt} (ckpt best_epoch={freeze_stats['ckpt_best_epoch']}) | "
            f"frozen param tensors: {freeze_stats['frozen_param_tensors']} | "
            f"trainable GRN scalars: {freeze_stats['trainable_scalars']}",
            flush=True,
        )
    if str(args.gene_token_mode) == "module":
        def _fmt_module_summary(label: str, genes) -> str:
            if not hasattr(model, "summarize_gene_module_coverage"):
                return f"{label}: unavailable"
            stats = model.summarize_gene_module_coverage([str(g) for g in genes])
            return (
                f"{label}: genes={stats['gene_count']}, "
                f"prior={stats['biological_module_count']}, "
                f"de_novo={stats.get('de_novo_module_count', 0)}, "
                f"allocated={stats.get('allocated_module_count', stats['biological_module_count'])}, "
                f"covered={stats['covered_count']}, uncovered={stats['uncovered_count']} "
                f"({stats['uncovered_ratio']:.4f})"
            )

        print("Gene module coverage:", flush=True)
        print("  " + _fmt_module_summary("data_genes", query_data.var_names.tolist()), flush=True)
        print("  " + _fmt_module_summary("st_observed_genes", query_genes), flush=True)
        print("  " + _fmt_module_summary("target_genes", target_genes), flush=True)
    model = model.to(args.device)
    if amp_cfg["model_dtype"] is not None:
        model = model.to(dtype=amp_cfg["model_dtype"])

    observed_genes = {
        "st": query_genes,
        "sc": ref_data.var_names.tolist(),
    }
    dataloaders = build_dataloaders(
        build_loader_cfg(train_data, args, observed_genes, order_required=False),
        use_gp=bool(getattr(args, "use_gp", False)),
    )
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    optim = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=pipeline_config["lr"],
        weight_decay=pipeline_config["wd"],
    )
    grad_scaler = torch.amp.GradScaler("cuda", enabled=amp_cfg["use_grad_scaler"])
    scheduler = None
    if pipeline_config["scheduler"] == "plat":
        scheduler = ReduceLROnPlateau(optim, "min", patience=pipeline_config["patience"], factor=0.9)

    train_loss = []
    valid_loss = []
    epoch_records = []
    best_dict = None
    best_loss = float("inf")
    best_epoch = None
    epochs_no_improve = 0
    min_delta = 0.5
    module_trajectory: list[dict] = []
    module_trajectory_dir = args.out_dir / f"{args.prefix}_module_trajectory"
    module_trajectory_csv = module_trajectory_dir / f"{args.prefix}_module_trajectory.csv"
    if args.gene_module_track_every_epochs > 0 and args.gene_token_mode == "module":
        initial_module_summary = export_gene_module_trajectory_snapshot(
            model=model,
            genes=[str(gene) for gene in query_data.var_names],
            out_dir=module_trajectory_dir,
            prefix=args.prefix,
            epoch=0,
            min_support=args.gene_module_track_min_support,
            inactive_mass_fraction=args.gene_module_track_inactive_mass_fraction,
        )
        module_trajectory.append(initial_module_summary)
        pd.DataFrame(module_trajectory).to_csv(module_trajectory_csv, index=False)
        print(
            "Initial modules | "
            f"allocated={initial_module_summary['allocated_modules']} | "
            f"nonempty={initial_module_summary['nonempty_modules']} | "
            f"primary={initial_module_summary['primary_modules']} | "
            f"effective={initial_module_summary['effective_modules']} | "
            f"mass_effective={initial_module_summary['mass_effective_module_count']:.2f}",
            flush=True,
        )

    def run_full_evaluation(eval_prefix: str, eval_label: str):
        print(
            f"Running full evaluation ({eval_label}) with best_epoch={best_epoch}, best_valid_loss={best_loss:.6f}",
            flush=True,
        )
        pred_tensor = predict(
            model=model,
            adata=query_data,
            args=args,
            target_genes=target_genes,
            observed_genes=observed_genes,
            inference_config={"target_genes": target_genes},
            device=args.device,
            ccc_cache=ccc_cache,
        )

        pred_tensor = pred_tensor.to("cpu")
        pred_df = pd.DataFrame(pred_tensor.numpy(), index=query_data.obs_names, columns=target_genes)
        pred_outputs = save_prediction_outputs(pred_df, args.out_dir, eval_prefix) if args.save_pred else {}

        labels = torch.from_numpy(query_data.obsm["truth"]).float()
        log_pred = torch.log1p(pred_tensor)
        log_labels = torch.log1p(labels)
        score_res = imputation_eval(log_pred, log_labels)

        from gcat.evaluation import CalculateMetrics

        truth_df = pd.DataFrame(query_data.obsm["truth"], index=query_data.obs_names, columns=target_genes)
        metric_obj = CalculateMetrics(truth_df, pred_df)
        result_all = metric_obj.compute_all()

        metric_mean = result_all.mean(axis=1)
        metric_summary = {
            "PCC": metric_mean["PCC"],
            "SSIM": metric_mean["SSIM"],
            "RMSE": metric_mean["RMSE"],
            "JS": metric_mean["JS"],
        }
        combined_score = {**score_res, **metric_summary}
        eval_epoch_number = -1
        if eval_label.startswith("interval_epoch"):
            try:
                eval_epoch_number = int(eval_label.replace("interval_epoch", ""))
            except ValueError:
                eval_epoch_number = -1
        combined_score["eval_checkpoint_epoch"] = eval_epoch_number
        combined_score["best_epoch"] = best_epoch
        combined_score["best_valid_loss"] = best_loss
        metric_outputs = save_metric_outputs(result_all, combined_score, args.out_dir, eval_prefix)
        if args.save_per_gene_score:
            metric_outputs.update(save_per_gene_metric_outputs(result_all, args.out_dir, eval_prefix))

        maybe_synchronize(args.device)
        print(f"Evaluation metrics ({eval_label}):", flush=True)
        for key, value in combined_score.items():
            if isinstance(value, (int, float, np.floating)) and not isinstance(value, bool):
                print(f"  {key}: {float(value):.6f}", flush=True)
            else:
                print(f"  {key}: {value}", flush=True)
        return combined_score, pred_outputs, metric_outputs

    for epoch in tqdm(range(pipeline_config["epochs"])):
        epoch_index = epoch + 1
        model.train()
        epoch_loss = []
        epoch_start_time = time.perf_counter()
        train_start_time = time.perf_counter()
        attach_seconds = 0.0
        forward_backward_seconds = 0.0
        if epoch < pipeline_config["warmup_epochs"]:
            warm_lr = pipeline_config["lr"] * epoch_index / pipeline_config["warmup_epochs"]
            for param_group in optim.param_groups:
                param_group["lr"] = warm_lr

        train_batch_count = 0
        train_regulon_activity_raw = []
        train_regulon_activity_weighted = []
        train_masked_zmse_raw = []
        train_masked_zmse_weighted = []
        train_block_gene_count = []
        train_outgoing_lr_raw = []
        train_outgoing_lr_weighted = []
        train_flow_outgoing_raw = []
        train_flow_outgoing_weighted = []
        train_flow_attn_gate_mean = []
        train_flow_stage5_beta = []
        train_grn_alpha = []
        train_grn_residual_l2 = []
        train_grn_residual_pos_frac = []
        train_pull_raw = []
        train_pull_weighted = []
        train_pull_pairs = []
        for data_dict in train_loader:
            if np.sum(data_dict["split"] == "train") == 0:
                continue
            if args.max_train_batches > 0 and train_batch_count >= args.max_train_batches:
                break
            batch_start_time = time.perf_counter()
            attach_start_time = time.perf_counter()
            input_dict = move_xdict_to_device(data_dict, args.device)
            input_dict = attach_ccc_bias_to_batch(input_dict, ccc_cache)
            attach_seconds += time.perf_counter() - attach_start_time
            x_dict = XDict(input_dict)
            pos = build_pos_from_xdict(x_dict)
            dist_bias, pair_labels = build_noisy_masa_targets(
                pos=pos,
                dist_metric=args.dist_metric,
                noise_rate=args.masa_noise_rate,
                mask_rate=args.masa_mask_rate,
                noise_std=args.masa_noise_std,
                pair_mask_token=args.pair_mask_token,
            )
            optim.zero_grad()
            fb_start_time = time.perf_counter()
            # Ablation hook: with --no_dist_bias the model must not see any
            # spatial prior, so strip coord from the batch as well (otherwise
            # _resolve_spatial_pos would rebuild pos from x_dict["coord"]).
            model_pos = None if args.no_dist_bias else pos
            model_dist_bias = None if args.no_dist_bias else dist_bias
            if args.no_dist_bias:
                x_dict.pop("coord", None)
            with amp_autocast_context(amp_cfg):
                out_dict, loss = model(
                    x_dict,
                    data_dict["gene_list"],
                    pos=model_pos,
                    dist_bias=model_dist_bias,
                    pair_labels=pair_labels,
                )
            if loss is None:
                continue

            if grad_scaler.is_enabled():
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optim)
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                grad_scaler.step(optim)
                grad_scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optim.step()
            forward_backward_seconds += time.perf_counter() - fb_start_time
            epoch_loss.append(loss.detach().float().item())
            train_regulon_activity_raw.append(float(out_dict.get("regulon_activity_loss_raw", 0.0)))
            train_regulon_activity_weighted.append(float(out_dict.get("regulon_activity_loss", 0.0)))
            train_masked_zmse_raw.append(float(out_dict.get("masked_zmse_loss_raw", 0.0)))
            train_masked_zmse_weighted.append(float(out_dict.get("masked_zmse_loss", 0.0)))
            train_block_gene_count.append(float(out_dict.get("st_block_gene_count", 0)))
            train_outgoing_lr_raw.append(float(out_dict.get("outgoing_lr_loss_raw", 0.0)))
            train_outgoing_lr_weighted.append(float(out_dict.get("outgoing_lr_loss", 0.0)))
            train_flow_outgoing_raw.append(float(out_dict.get("flow_outgoing_loss_raw", 0.0)))
            train_flow_outgoing_weighted.append(float(out_dict.get("flow_outgoing_loss", 0.0)))
            train_flow_attn_gate_mean.append(float(out_dict.get("flow_attn_gate_mean", float("nan"))))
            train_flow_stage5_beta.append(float(out_dict.get("flow_stage5_beta", float("nan"))))
            train_grn_alpha.append(float(out_dict.get("grn_residual_alpha", float("nan"))))
            train_grn_residual_l2.append(float(out_dict.get("grn_residual_l2", float("nan"))))
            train_grn_residual_pos_frac.append(float(out_dict.get("grn_residual_pos_frac", float("nan"))))
            train_pull_raw.append(float(out_dict.get("pull_loss_raw", 0.0)))
            train_pull_weighted.append(float(out_dict.get("pull_loss", 0.0)))
            train_pull_pairs.append(float(
                out_dict.get("pull_pair_count_state", 0)
                + out_dict.get("pull_pair_count_niche", 0)
            ))
            train_batch_count += 1
            if (
                epoch == 0
                and args.reg_loss_stop_below > 0
                and train_batch_count == max(1, args.reg_loss_check_batches)
            ):
                checked_reg_loss = float(np.mean(train_regulon_activity_raw))
                print(
                    f"Regulon activity check | batches={train_batch_count} | "
                    f"raw_mean={checked_reg_loss:.6f} | stop_below={args.reg_loss_stop_below:.6f}",
                    flush=True,
                )
                if checked_reg_loss < args.reg_loss_stop_below:
                    raise RuntimeError(
                        "Stopping for regulon activity loss adjustment: "
                        f"raw mean {checked_reg_loss:.6f} < {args.reg_loss_stop_below:.6f}."
                    )
            if args.log_batch_every > 0 and train_batch_count % args.log_batch_every == 0:
                batch_seconds = time.perf_counter() - batch_start_time
                print(
                    f"Epoch {epoch} batch {train_batch_count} | "
                    f"batch_seconds={batch_seconds:.3f} | "
                    f"attach_seconds_total={attach_seconds:.2f} | "
                    f"fb_seconds_total={forward_backward_seconds:.2f} | "
                    f"reg_activity_raw={train_regulon_activity_raw[-1]:.6f} | "
                    f"reg_activity_weighted={train_regulon_activity_weighted[-1]:.6f} | "
                    f"masked_zmse={train_masked_zmse_weighted[-1]:.6f} | "
                    f"outgoing_lr_raw={train_outgoing_lr_raw[-1]:.6f} | "
                    f"outgoing_lr_weighted={train_outgoing_lr_weighted[-1]:.6f} | "
                    f"flow_out_raw={train_flow_outgoing_raw[-1]:.6f} | "
                    f"flow_out_weighted={train_flow_outgoing_weighted[-1]:.6f}",
                    flush=True,
                )

        train_epoch_loss = float("nan") if not epoch_loss else sum(epoch_loss) / len(epoch_loss)
        train_regulon_activity_raw_mean = (
            float("nan") if not train_regulon_activity_raw else float(np.mean(train_regulon_activity_raw))
        )
        train_regulon_activity_weighted_mean = (
            float("nan") if not train_regulon_activity_weighted else float(np.mean(train_regulon_activity_weighted))
        )
        train_masked_zmse_raw_mean = (
            float("nan") if not train_masked_zmse_raw else float(np.mean(train_masked_zmse_raw))
        )
        train_masked_zmse_weighted_mean = (
            float("nan") if not train_masked_zmse_weighted else float(np.mean(train_masked_zmse_weighted))
        )
        train_block_gene_count_mean = (
            float("nan") if not train_block_gene_count else float(np.mean(train_block_gene_count))
        )
        train_outgoing_lr_raw_mean = (
            float("nan") if not train_outgoing_lr_raw else float(np.mean(train_outgoing_lr_raw))
        )
        train_outgoing_lr_weighted_mean = (
            float("nan") if not train_outgoing_lr_weighted else float(np.mean(train_outgoing_lr_weighted))
        )
        train_flow_outgoing_raw_mean = (
            float("nan") if not train_flow_outgoing_raw else float(np.mean(train_flow_outgoing_raw))
        )
        train_flow_outgoing_weighted_mean = (
            float("nan") if not train_flow_outgoing_weighted else float(np.mean(train_flow_outgoing_weighted))
        )
        train_loss.append(train_epoch_loss)

        train_seconds = time.perf_counter() - train_start_time
        valid_start_time = time.perf_counter()
        result_dict = inference(
            model,
            val_loader,
            None,
            args.device,
            dist_metric=args.dist_metric,
            order_required=False,
            ccc_cache=ccc_cache,
            max_batches=int(args.max_eval_batches),
            no_dist_bias=bool(args.no_dist_bias),
        )
        valid_seconds = time.perf_counter() - valid_start_time
        current_valid_loss = result_dict["loss"]
        valid_loss.append(current_valid_loss)
        epoch_seconds = time.perf_counter() - epoch_start_time
        current_lr = float(optim.param_groups[0]["lr"])
        current_peak_memory_mb = None
        if amp_cfg["device_type"] == "cuda":
            current_peak_memory_mb = torch.cuda.max_memory_allocated(args.device) / (1024 ** 2)
        epoch_record = {
            "epoch": epoch_index,
            "train_loss": train_epoch_loss,
            "valid_loss": current_valid_loss,
            "train_regulon_activity_loss_raw": train_regulon_activity_raw_mean,
            "train_regulon_activity_loss": train_regulon_activity_weighted_mean,
            "lambda_reg": float(args.lambda_reg),
            "train_masked_zmse_loss_raw": train_masked_zmse_raw_mean,
            "train_masked_zmse_loss": train_masked_zmse_weighted_mean,
            "train_block_gene_count": train_block_gene_count_mean,
            "lambda_masked_zmse": float(args.lambda_masked_zmse),
            "train_outgoing_lr_loss_raw": train_outgoing_lr_raw_mean,
            "train_outgoing_lr_loss": train_outgoing_lr_weighted_mean,
            "lambda_out": float(args.lambda_out),
            "train_flow_outgoing_loss_raw": train_flow_outgoing_raw_mean,
            "train_flow_outgoing_loss": train_flow_outgoing_weighted_mean,
            "train_flow_attn_gate_mean": (
                float("nan") if not train_flow_attn_gate_mean else float(np.nanmean(train_flow_attn_gate_mean))
            ),
            "train_flow_stage5_beta": (
                float("nan") if not train_flow_stage5_beta else float(np.nanmean(train_flow_stage5_beta))
            ),
            "train_grn_alpha": (
                float("nan") if not train_grn_alpha else float(np.nanmean(train_grn_alpha))
            ),
            "train_grn_residual_l2": (
                float("nan") if not train_grn_residual_l2 else float(np.nanmean(train_grn_residual_l2))
            ),
            "train_grn_residual_pos_frac": (
                float("nan") if not train_grn_residual_pos_frac else float(np.nanmean(train_grn_residual_pos_frac))
            ),
            "train_pull_loss_raw": (
                float("nan") if not train_pull_raw else float(np.mean(train_pull_raw))
            ),
            "train_pull_loss": (
                float("nan") if not train_pull_weighted else float(np.mean(train_pull_weighted))
            ),
            "train_pull_pair_count": (
                float("nan") if not train_pull_pairs else float(np.mean(train_pull_pairs))
            ),
            "lambda_flow": float(args.lambda_flow),
            "lr": current_lr,
            "epoch_seconds": epoch_seconds,
            "train_seconds": train_seconds,
            "valid_seconds": valid_seconds,
            "train_batches": train_batch_count,
            "ccc_attach_seconds": attach_seconds,
            "forward_backward_seconds": forward_backward_seconds,
            "peak_memory_mb": current_peak_memory_mb,
        }
        epoch_records.append(epoch_record)
        if (
            args.gene_module_track_every_epochs > 0
            and args.gene_token_mode == "module"
            and epoch_index % args.gene_module_track_every_epochs == 0
        ):
            module_summary = export_gene_module_trajectory_snapshot(
                model=model,
                genes=[str(gene) for gene in query_data.var_names],
                out_dir=module_trajectory_dir,
                prefix=args.prefix,
                epoch=epoch_index,
                min_support=args.gene_module_track_min_support,
                inactive_mass_fraction=args.gene_module_track_inactive_mass_fraction,
            )
            module_trajectory.append(module_summary)
            pd.DataFrame(module_trajectory).to_csv(module_trajectory_csv, index=False)
            epoch_record.update(
                {
                    "module_allocated": module_summary["allocated_modules"],
                    "module_nonempty": module_summary["nonempty_modules"],
                    "module_primary": module_summary["primary_modules"],
                    "module_effective": module_summary["effective_modules"],
                    "module_mass_effective": module_summary["mass_effective_module_count"],
                }
            )
            print(
                f"Epoch {epoch_index} modules | "
                f"allocated={module_summary['allocated_modules']} | "
                f"nonempty={module_summary['nonempty_modules']} | "
                f"primary={module_summary['primary_modules']} | "
                f"effective={module_summary['effective_modules']} | "
                f"mass_effective={module_summary['mass_effective_module_count']:.2f}",
                flush=True,
            )
        print(
            f"Epoch {epoch} | Train loss: {train_epoch_loss:.4f} | Valid loss: {current_valid_loss:.4f} | "
            f"Seconds: {epoch_seconds:.2f} | train: {train_seconds:.2f} | valid: {valid_seconds:.2f} | "
            f"ccc_attach: {attach_seconds:.2f} | fwdbwd: {forward_backward_seconds:.2f} | "
            f"reg_raw: {train_regulon_activity_raw_mean:.6f} | "
            f"reg_weighted: {train_regulon_activity_weighted_mean:.6f} | "
            f"masked_zmse: {train_masked_zmse_weighted_mean:.6f} | "
            f"out_raw: {train_outgoing_lr_raw_mean:.6f} | "
            f"out_weighted: {train_outgoing_lr_weighted_mean:.6f} | "
            f"flow_raw: {train_flow_outgoing_raw_mean:.6f} | "
            f"flow_weighted: {train_flow_outgoing_weighted_mean:.6f} | "
            f"peak_mb: {current_peak_memory_mb if current_peak_memory_mb is not None else 'NA'}",
            flush=True,
        )
        
        pipeline_config['patience'] = 25
        if scheduler is not None and not np.isnan(current_valid_loss):
            scheduler.step(current_valid_loss)

        if not np.isnan(current_valid_loss) and current_valid_loss < best_loss - min_delta:
            best_loss = current_valid_loss
            best_epoch = epoch_index
            epochs_no_improve = 0
            best_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= pipeline_config["patience"]:
                print(
                    f"Early-stopping at epoch {epoch} "
                    f"(no improvement >= {min_delta} for {pipeline_config['patience']} epochs)"
                )
                break
        epoch_records[-1]["best_epoch"] = best_epoch
        epoch_records[-1]["best_valid_loss"] = best_loss

        if (
            int(args.eval_every_epochs) > 0
            and not args.skip_final_eval
            and epoch_index % int(args.eval_every_epochs) == 0
            and best_dict is not None
        ):
            python_rng_state = random.getstate()
            numpy_rng_state = np.random.get_state()
            torch_rng_state = torch.get_rng_state()
            cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            current_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            model.load_state_dict(best_dict)
            interval_label = f"interval_epoch{epoch_index}"
            interval_prefix = f"{args.prefix}_{interval_label}"
            run_full_evaluation(interval_prefix, interval_label)
            if args.save_interval_checkpoints:
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model_config": model_kwargs,
                        "args": vars(args),
                        "checkpoint_epoch": epoch_index,
                        "best_epoch": best_epoch,
                        "best_valid_loss": best_loss,
                    },
                    args.out_dir / f"{interval_prefix}_model.pt",
                )
            if args.gene_module_export_stats and args.gene_token_mode == "module":
                export_gene_module_assignment(
                    model=model,
                    genes=query_data.var_names.tolist(),
                    out_dir=args.out_dir / f"{interval_prefix}_module_analysis",
                    prefix=interval_prefix,
                )
            model.load_state_dict(current_dict)
            del current_dict
            random.setstate(python_rng_state)
            np.random.set_state(numpy_rng_state)
            torch.set_rng_state(torch_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)
            if amp_cfg["device_type"] == "cuda":
                torch.cuda.empty_cache()

    if best_dict is not None:
        model.load_state_dict(best_dict)

    model_path = None
    if args.save_model:
        model_path = args.out_dir / f"{args.prefix}_model.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": model_kwargs,
                "args": vars(args),
            },
            model_path,
        )

    if args.gene_module_export_stats and args.gene_token_mode == "module":
        export_gene_module_assignment(
            model=model,
            genes=query_data.var_names.tolist(),
            out_dir=args.out_dir / f"{args.prefix}_module_analysis",
            prefix=args.prefix,
        )

    if args.skip_final_eval:
        epoch_history_path = args.out_dir / f"{args.prefix}_epoch_history.csv"
        speed_path = args.out_dir / f"{args.prefix}_speed.json"
        pd.DataFrame(epoch_records).to_csv(epoch_history_path, index=False)
        if args.save_speed_metrics:
            with open(speed_path, "w", encoding="utf-8") as fp:
                json.dump(
                    {
                        "prefix": args.prefix,
                        "device": str(args.device),
                        "random_seed": int(args.random_seed),
                        "mask_fold_index": int(args.mask_fold_index),
                        "deterministic": bool(args.deterministic),
                        "batch_size": int(args.batch_size),
                        "amp_mode": amp_cfg["amp_mode"],
                        "model_dtype": None if amp_cfg["model_dtype"] is None else str(amp_cfg["model_dtype"]),
                        "epochs_completed": len(epoch_records),
                        "eval_every_epochs": int(args.eval_every_epochs),
                        "best_epoch": best_epoch,
                        "best_valid_loss": best_loss,
                        "drop_node_rate": float(args.drop_node_rate),
                        "pair_loss_weight": float(args.pair_loss_weight),
                        "model_pt": None if model_path is None else str(model_path),
                        "save_model": bool(args.save_model),
                        "save_pred": bool(args.save_pred),
                        "save_per_gene_score": bool(args.save_per_gene_score),
                        "skip_final_eval": True,
                        "lr": float(args.lr),
                        "weight_decay": float(args.weight_decay),
                        "dropout": float(model_kwargs["dropout"]),
                        "warmup_epochs": int(args.warmup_epochs),
                        "gene_token_mode": str(args.gene_token_mode),
                        "gene_module_backend": "expression_first",
                        "gene_module_assignment": "entmax15",
                        "use_ccc_bias": bool(args.use_ccc_bias),
                        "ccc_program_head_bias": bool(args.ccc_program_head_bias),
                        "niche_regulon_bridge": bool(args.niche_regulon_bridge),
                        "niche_regulon_bridge_path": (
                            str(args.niche_regulon_bridge_path) if args.niche_regulon_bridge_path else None
                        ),
                        "niche_bridge_mode": str(args.niche_bridge_mode),
                        "niche_bridge_stage": str(args.niche_bridge_stage),
                        "lambda_out": float(args.lambda_out),
                        "outgoing_lr_std_floor": float(args.outgoing_lr_std_floor),
                        "flow_linear_bottleneck": bool(args.flow_linear_bottleneck),
                        "lambda_flow": float(args.lambda_flow),
                        "flow_structure": str(args.flow_structure),
                        "flow_supervision": str(args.flow_supervision),
                        "flow_graph_mask_path": (
                            str(args.flow_graph_mask_path) if args.flow_graph_mask_path else None
                        ),
                        "flow_oracle_st_truth": bool(args.flow_oracle_st_truth),
                        "flow_cell_attention": bool(args.flow_cell_attention),
                        "flow_attn_layers": str(args.flow_attn_layers),
                        "flow_attn_params": int(sum(
                            parameter.numel()
                            for name, parameter in model.named_parameters()
                            if name.startswith("flow_attn_")
                        )) if args.flow_cell_attention else 0,
                        **flow_attention_final_summary(model, args),
                        **grn_residual_final_summary(model, args),
                        "niche_regulon_bridge_params": int(
                            model.prior_gene_module_num * int(model_kwargs["hidden_dim"])
                        ) if args.niche_regulon_bridge else 0,
                        "ccc_lr_programs": None if ccc_cache is None else int(ccc_cache.get("n_lr", 0)),
                        "ccc_program_head_bias_params": int(
                            (int(ccc_cache.get("n_lr", 0)) + 1) * int(model_kwargs["num_heads"])
                        ) if args.ccc_program_head_bias else 0,
                        "epoch_history_csv": str(epoch_history_path),
                    },
                    fp,
                    ensure_ascii=False,
                    indent=2,
                )
        print(
            "Skipped final evaluation. "
            f"Model saved: {str(model_path) if model_path is not None else 'disabled'}",
            flush=True,
        )
        return

    pred_tensor = predict(
        model=model,
        adata=query_data,
        args=args,
        target_genes=target_genes,
        observed_genes=observed_genes,
        inference_config={"target_genes": target_genes},
        device=args.device,
        ccc_cache=ccc_cache,
    )

    pred_tensor = pred_tensor.to("cpu")
    pred_df = pd.DataFrame(pred_tensor.numpy(), index=query_data.obs_names, columns=target_genes)
    pred_outputs = save_prediction_outputs(pred_df, args.out_dir, args.prefix) if args.save_pred else {}

    if args.skip_final_metrics:
        epoch_history_path = args.out_dir / f"{args.prefix}_epoch_history.csv"
        pd.DataFrame(epoch_records).to_csv(epoch_history_path, index=False)
        print(
            "Saved final predictions and skipped metrics for unmeasured zero-placeholder targets.",
            flush=True,
        )
        return

    labels = torch.from_numpy(query_data.obsm["truth"]).float()
    log_pred = torch.log1p(pred_tensor)
    log_labels = torch.log1p(labels)
    score_res = imputation_eval(log_pred, log_labels)

    from gcat.evaluation import CalculateMetrics

    truth_df = pd.DataFrame(query_data.obsm["truth"], index=query_data.obs_names, columns=target_genes)
    metric_obj = CalculateMetrics(truth_df, pred_df)
    result_all = metric_obj.compute_all()

    metric_mean = result_all.mean(axis=1)
    metric_summary = {
        "PCC": metric_mean["PCC"],
        "SSIM": metric_mean["SSIM"],
        "RMSE": metric_mean["RMSE"],
        "JS": metric_mean["JS"],
    }
    combined_score = {**score_res, **metric_summary}
    combined_score["eval_checkpoint_epoch"] = -1
    combined_score["best_epoch"] = best_epoch
    combined_score["best_valid_loss"] = best_loss
    metric_outputs = save_metric_outputs(result_all, combined_score, args.out_dir, args.prefix)
    if args.save_per_gene_score:
        metric_outputs.update(save_per_gene_metric_outputs(result_all, args.out_dir, args.prefix))

    maybe_synchronize(args.device)
    total_seconds = time.perf_counter() - run_start_time
    peak_memory_mb = None
    if amp_cfg["device_type"] == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(args.device) / (1024 ** 2)

    epoch_history_path = args.out_dir / f"{args.prefix}_epoch_history.csv"
    speed_path = args.out_dir / f"{args.prefix}_speed.json"

    pd.DataFrame(epoch_records).to_csv(epoch_history_path, index=False)
    speed_metrics = {
        "prefix": args.prefix,
        "device": str(args.device),
        "random_seed": int(args.random_seed),
        "mask_fold_index": int(args.mask_fold_index),
        "deterministic": bool(args.deterministic),
        "batch_size": int(args.batch_size),
        "amp_mode": amp_cfg["amp_mode"],
        "amp_enabled": amp_cfg["amp_enabled"],
        "model_dtype": None if amp_cfg["model_dtype"] is None else str(amp_cfg["model_dtype"]),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "dropout": float(model_kwargs["dropout"]),
        "warmup_epochs": int(args.warmup_epochs),
        "gene_token_mode": str(args.gene_token_mode),
        "gene_module_backend": "expression_first",
        "gene_module_assignment": "entmax15",
        "gene_module_num": str(args.gene_module_num),
        "gene_module_init_path": str(args.gene_module_init_path) if args.gene_module_init_path is not None else None,
        "dynamic_gene_modules": bool(args.dynamic_gene_modules),
        "prior_strength": float(args.prior_strength),
        "off_prior_init_std": float(args.off_prior_init_std),
        "assignment_temperature": float(args.assignment_temperature),
        "disable_cell_transformer": bool(args.disable_cell_transformer),
        "disable_gene_module_attention": bool(args.disable_gene_module_attention),
        "no_dist_bias": bool(args.no_dist_bias),
        "drop_node_rate": float(args.drop_node_rate),
        "pair_loss_weight": float(args.pair_loss_weight),
        "balance_modalities": bool(args.balance_modalities),
        "st_block_gene_mask_rate": float(args.st_block_gene_mask_rate),
        "lambda_masked_zmse": float(args.lambda_masked_zmse),
        "use_ccc_bias": bool(args.use_ccc_bias),
        "ccc_bias_layers": str(args.ccc_bias_layers),
        "ccc_bias_trainable": bool(args.ccc_bias_trainable),
        "ccc_bias_init_scale": float(args.ccc_bias_init_scale),
        "ccc_program_head_bias": bool(args.ccc_program_head_bias),
        "niche_regulon_bridge": bool(args.niche_regulon_bridge),
        "niche_regulon_bridge_path": (
            str(args.niche_regulon_bridge_path) if args.niche_regulon_bridge_path else None
        ),
        "niche_bridge_mode": str(args.niche_bridge_mode),
        "niche_bridge_stage": str(args.niche_bridge_stage),
        "lambda_pull": float(args.lambda_pull),
        "pull_label_mode": str(args.pull_label_mode),
        "pull_target": str(args.pull_target),
        "pull_topk": int(args.pull_topk),
        "pull_niche_k": int(args.pull_niche_k),
        "st_aligned_sc_prior": bool(getattr(args, "st_aligned_sc_prior", False)),
        "lambda_out": float(args.lambda_out),
        "outgoing_lr_std_floor": float(args.outgoing_lr_std_floor),
        "flow_linear_bottleneck": bool(args.flow_linear_bottleneck),
        "lambda_flow": float(args.lambda_flow),
        "flow_structure": str(args.flow_structure),
        "flow_supervision": str(args.flow_supervision),
        "flow_graph_mask_path": (
            str(args.flow_graph_mask_path) if args.flow_graph_mask_path else None
        ),
        "flow_oracle_st_truth": bool(args.flow_oracle_st_truth),
        "flow_cell_attention": bool(args.flow_cell_attention),
        "flow_attn_layers": str(args.flow_attn_layers),
        "flow_attn_params": int(sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if name.startswith("flow_attn_")
        )) if args.flow_cell_attention else 0,
        **flow_attention_final_summary(model, args),
        **grn_residual_final_summary(model, args),
        "grn_residual_params": int(sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if name.startswith("grn_")
        )) if args.grn_decoder_residual else 0,
        "niche_regulon_bridge_params": int(
            model.prior_gene_module_num * int(model_kwargs["hidden_dim"])
        ) if args.niche_regulon_bridge else 0,
        "ccc_program_head_bias_params": int(
            (int(ccc_cache.get("n_lr", 0)) + 1) * int(model_kwargs["num_heads"])
        ) if args.ccc_program_head_bias else 0,
        "ccc_topk": int(args.ccc_topk),
        "ccc_lr_programs": None if ccc_cache is None else int(ccc_cache.get("n_lr", 0)),
        "ccc_st_candidate_mode": str(args.ccc_st_candidate_mode),
        "ccc_sc_candidate_mode": str(args.ccc_sc_candidate_mode),
        "ccc_st_candidate_k": None if ccc_cache is None else int(ccc_cache.get("st_candidate_k", 0)),
        "ccc_sc_candidate_k": None if ccc_cache is None else int(ccc_cache.get("sc_candidate_k", 0)),
        "epochs_requested": int(args.epoch),
        "epochs_completed": len(epoch_records),
        "eval_every_epochs": int(args.eval_every_epochs),
        "best_epoch": best_epoch,
        "best_valid_loss": best_loss,
        "total_seconds": total_seconds,
        "mean_epoch_seconds": float(np.mean([row["epoch_seconds"] for row in epoch_records])) if epoch_records else None,
        "peak_memory_mb": peak_memory_mb,
        "model_pt": None if model_path is None else str(model_path),
        "save_model": bool(args.save_model),
        "save_pred": bool(args.save_pred),
        "save_per_gene_score": bool(args.save_per_gene_score),
        **pred_outputs,
        **metric_outputs,
        "epoch_history_csv": str(epoch_history_path),
    }
    if args.save_speed_metrics:
        with open(speed_path, "w", encoding="utf-8") as fp:
            json.dump(speed_metrics, fp, ensure_ascii=False, indent=2)

    print("Final metrics:")
    for key, value in combined_score.items():
        print(f"  {key}: {float(value):.6f}")
    if args.flow_cell_attention and args.flow_attn_mode == "gated":
        for i in model.gcat_encoder.flow_attn_layer_indices:
            gate_values = " ".join(
                f"{value:.4f}"
                for value in model.flow_attn_gate.detach().float()[i].tolist()
            )
            print(f"Flow attention gate | layer {i} | per-head: {gate_values}", flush=True)
        weight = model.flow_attn_program_weight.detach().float()
        print(
            f"Flow program weight | rel_change={flow_attention_final_summary(model, args)['flow_attn_program_weight_rel_change']:.4f} "
            f"| head_mean={[round(float(v), 4) for v in weight.mean(dim=1).tolist()]}",
            flush=True,
        )
    if args.flow_stage5:
        print(
            f"Flow Stage-5 regulon update | beta={float(model.flow_reg_update_beta.detach().item()):.6f}",
            flush=True,
        )
    if args.grn_decoder_residual:
        grn_stats = grn_residual_final_summary(model, args)
        print(
            "GRN decoder residual | "
            f"alpha={grn_stats['grn_residual_alpha']:.6f} | "
            f"edges={grn_stats['grn_edges']} | "
            f"updated_frac={grn_stats['grn_edge_updated_frac']:.4f} | "
            f"strength_mean={grn_stats['grn_edge_strength_mean']:.4f} "
            f"(init {grn_stats['grn_edge_strength_init']:.4f}) | "
            f"strength_rel_change={grn_stats['grn_edge_strength_rel_change']:.4f}",
            flush=True,
        )
    print("Speed metrics:")
    print(f"  amp_mode: {amp_cfg['amp_mode']}")
    print(f"  epochs_completed: {len(epoch_records)}")
    print(f"  total_seconds: {total_seconds:.2f}")
    if peak_memory_mb is not None:
        print(f"  peak_memory_mb: {peak_memory_mb:.2f}")


if __name__ == "__main__":
    main(parse_args())
