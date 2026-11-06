#!/usr/bin/env python
import argparse
import json
import pathlib
import sys
from typing import Any

import anndata as ad
import hdf5plugin  # noqa: F401
import numpy as np
import pandas as pd
import torch
from scipy import sparse

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge fold-level CV prediction/per-gene outputs and recompute total score.")
    parser.add_argument("--dataset", required=True, help="Dataset name, used for output filenames.")
    parser.add_argument("--run_dir", type=pathlib.Path, required=True, help="Dataset CV run directory containing fold* outputs.")
    parser.add_argument("--query_data", type=pathlib.Path, required=True, help="Original ST h5ad used as ground truth source.")
    parser.add_argument("--prefix", default=None, help="Merged output prefix. Defaults to <dataset>_cv.")
    parser.add_argument("--mask_gene", type=pathlib.Path, required=True,
                        help="Five-fold benchmark mask; enforce exact fold and gene coverage before merging.")
    parser.add_argument("--delete_fold_outputs", action="store_true", help="Delete fold pred/per_gene files after successful merge.")
    return parser.parse_args()


def read_score_tsv(path: pathlib.Path) -> dict[str, Any]:
    df = pd.read_csv(path, sep="\t", index_col=0)
    values = df.iloc[:, 0] if df.shape[1] else pd.Series(dtype=float)
    out = {"score_tsv": str(path), "fold": infer_fold(path)}
    for key, value in values.items():
        try:
            out[str(key)] = float(value)
        except Exception:
            out[str(key)] = value
    return out


def infer_fold(path: pathlib.Path) -> int:
    for part in path.parts:
        if part.startswith("fold"):
            try:
                return int(part[4:])
            except Exception:
                pass
    return -1


def load_pred_npz(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    obj = np.load(path, allow_pickle=False)
    return obj["pred"].astype(np.float32, copy=False), obj["cells"].astype(str), obj["genes"].astype(str)


def to_dense_float32(x) -> np.ndarray:
    if sparse.issparse(x):
        return x.toarray().astype(np.float32, copy=False)
    return np.asarray(x, dtype=np.float32)


def save_score(score: dict[str, float], path: pathlib.Path, prefix: str) -> None:
    pd.DataFrame.from_dict(score, orient="index", columns=[prefix]).to_csv(
        path, sep="\t", index=True, float_format="%.8g"
    )


def resolve_prediction_cells(pred_cells: list[str], adata: ad.AnnData) -> list[str]:
    missing_cells = [cell for cell in pred_cells if cell not in adata.obs_names]
    if not missing_cells:
        return pred_cells

    resolved = []
    for cell in pred_cells:
        if not cell.startswith("st_"):
            raise ValueError(
                f"{len(missing_cells)} prediction cells are absent from query_data; "
                f"first={missing_cells[0]}"
            )
        suffix = cell[3:]
        if suffix in adata.obs_names:
            resolved.append(str(suffix))
            continue
        try:
            idx = int(suffix)
        except ValueError as exc:
            raise ValueError(
                f"{len(missing_cells)} prediction cells are absent from query_data; "
                f"first={missing_cells[0]}. Tried stripping 'st_' to '{suffix}', "
                "but it is neither an obs_name nor a numeric row index."
            ) from exc
        if idx < 0 or idx >= adata.n_obs:
            raise ValueError(
                f"Prediction cell {cell} maps to index {idx}, but query_data has {adata.n_obs} cells."
            )
        resolved.append(str(adata.obs_names[idx]))
    return resolved


def compute_total_score(pred_df: pd.DataFrame, query_data_path: pathlib.Path) -> tuple[dict[str, float], pd.DataFrame]:
    from gcat.evaluation.imputation import imputation_eval
    from gcat.evaluation import CalculateMetrics

    adata = ad.read_h5ad(query_data_path)
    resolved_cells = resolve_prediction_cells([str(cell) for cell in pred_df.index.tolist()], adata)
    if resolved_cells != pred_df.index.tolist():
        pred_df = pred_df.copy()
        pred_df.index = resolved_cells
    missing_genes = [gene for gene in pred_df.columns if gene not in adata.var_names]
    if missing_genes:
        raise ValueError(f"{len(missing_genes)} prediction genes are absent from query_data; first={missing_genes[0]}")

    truth = to_dense_float32(adata[pred_df.index.tolist(), pred_df.columns.tolist()].X)
    pred = pred_df.to_numpy(dtype=np.float32, copy=False)
    log_pred = torch.log1p(torch.from_numpy(pred))
    log_truth = torch.log1p(torch.from_numpy(truth))
    score_res = imputation_eval(log_pred, log_truth)

    truth_df = pd.DataFrame(truth, index=pred_df.index, columns=pred_df.columns)
    result_all = CalculateMetrics(truth_df, pred_df).compute_all()
    metric_mean = result_all.mean(axis=1)
    combined_score = {
        **score_res,
        "PCC": float(metric_mean["PCC"]),
        "SSIM": float(metric_mean["SSIM"]),
        "RMSE": float(metric_mean["RMSE"]),
        "JS": float(metric_mean["JS"]),
    }
    return combined_score, result_all


def main() -> None:
    args = parse_args()
    prefix = args.prefix or f"{args.dataset}_cv"
    merged_dir = args.run_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    pred_paths = sorted(
        (
            path
            for path in args.run_dir.glob("fold*/*_pred.npz")
            if "_interval_epoch" not in path.name
        ),
        key=infer_fold,
    )
    per_gene_paths = sorted(
        (
            path
            for path in args.run_dir.glob("fold*/*_per_gene_score.tsv")
            if "_interval_epoch" not in path.name
        ),
        key=infer_fold,
    )
    score_paths = sorted(
        (
            path
            for path in args.run_dir.glob("fold*/*_score.tsv")
            if not path.name.endswith("_per_gene_score.tsv")
            and "_interval_epoch" not in path.name
        ),
        key=infer_fold,
    )
    if not pred_paths:
        raise FileNotFoundError(f"No fold prediction files found under {args.run_dir}")
    expected_folds = np.load(args.mask_gene, allow_pickle=True)
    if len(expected_folds) != 5:
        raise ValueError(f"Expected a five-fold benchmark mask, got {len(expected_folds)} folds")
    if [infer_fold(path) for path in pred_paths] != list(range(5)):
        raise ValueError("Merge requires exactly one prediction file for every fold 0..4")
    if len(per_gene_paths) != len(pred_paths):
        raise ValueError(f"Found {len(pred_paths)} pred files but {len(per_gene_paths)} per-gene files.")

    pred_blocks = []
    genes_all: list[str] = []
    fold_for_gene: list[int] = []
    cells_ref = None
    raw_cells_ref = None

    for pred_path in pred_paths:
        fold = infer_fold(pred_path)
        pred, cells, genes = load_pred_npz(pred_path)
        if not np.array_equal(genes, np.asarray(expected_folds[fold], dtype=str)):
            raise ValueError(f"Target genes do not match benchmark fold {fold}: {pred_path}")
        if pred.shape != (len(cells), len(genes)) or not np.isfinite(pred).all():
            raise ValueError(f"Invalid prediction shape/values: {pred_path}")
        if cells_ref is None:
            cells_ref = cells
            raw_cells_ref = cells.copy()
        elif not np.array_equal(cells_ref, cells):
            raise ValueError(f"Cell order mismatch in {pred_path}")
        if set(genes_all).intersection(set(genes.tolist())):
            dup = sorted(set(genes_all).intersection(set(genes.tolist())))[0]
            raise ValueError(f"Duplicate gene across folds: {dup}")
        pred_blocks.append(pred)
        genes_all.extend(genes.tolist())
        fold_for_gene.extend([fold] * len(genes))

    pred_merged = np.concatenate(pred_blocks, axis=1)
    cells_arr = np.asarray(cells_ref, dtype="U")
    genes_arr = np.asarray(genes_all, dtype="U")
    fold_for_gene_arr = np.asarray(fold_for_gene, dtype=np.int16)
    adata_for_cells = ad.read_h5ad(args.query_data)
    resolved_cells_arr = np.asarray(
        resolve_prediction_cells([str(cell) for cell in cells_arr.tolist()], adata_for_cells),
        dtype="U",
    )
    pred_path = merged_dir / f"{prefix}_pred.npz"
    np.savez_compressed(
        pred_path,
        pred=pred_merged,
        cells=resolved_cells_arr,
        raw_cells=np.asarray(raw_cells_ref, dtype="U"),
        genes=genes_arr,
        fold=fold_for_gene_arr,
    )

    pred_df = pd.DataFrame(pred_merged, index=cells_arr, columns=genes_arr)
    combined_score, recomputed_per_gene = compute_total_score(pred_df, args.query_data)
    score_path = merged_dir / f"{prefix}_score.tsv"
    save_score(combined_score, score_path, prefix)

    per_gene_npz_path = merged_dir / f"{prefix}_per_gene_score.npz"
    np.savez_compressed(
        per_gene_npz_path,
        values=recomputed_per_gene.to_numpy(dtype=np.float32, copy=True),
        metrics=np.asarray([str(x) for x in recomputed_per_gene.index.tolist()], dtype="U"),
        genes=np.asarray([str(x) for x in recomputed_per_gene.columns.tolist()], dtype="U"),
        fold=fold_for_gene_arr,
    )

    fold_scores = pd.DataFrame([read_score_tsv(path) for path in score_paths])
    fold_scores_path = merged_dir / f"{prefix}_fold_scores.tsv"
    fold_scores.to_csv(fold_scores_path, sep="\t", index=False)

    manifest = {
        "dataset": args.dataset,
        "run_dir": str(args.run_dir),
        "query_data": str(args.query_data),
        "n_folds": len(pred_paths),
        "n_cells": int(pred_merged.shape[0]),
        "n_genes": int(pred_merged.shape[1]),
        "mask_gene": str(args.mask_gene.resolve()),
        "pred_npz": str(pred_path),
        "per_gene_score_npz": str(per_gene_npz_path),
        "score_tsv": str(score_path),
        "fold_scores_tsv": str(fold_scores_path),
    }
    manifest_path = merged_dir / f"{prefix}_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2, ensure_ascii=False)

    if args.delete_fold_outputs:
        for path in pred_paths + per_gene_paths:
            path.unlink()

    print(f"Wrote: {pred_path}")
    print(f"Wrote: {per_gene_npz_path}")
    print(f"Wrote: {score_path}")
    print(f"Wrote: {fold_scores_path}")
    if args.delete_fold_outputs:
        print("Deleted fold-level pred/per_gene files; fold score TSV files were retained.")


if __name__ == "__main__":
    main()
