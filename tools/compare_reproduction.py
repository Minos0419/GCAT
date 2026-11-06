"""Compare benchmark-aligned fold predictions and scores against archived references."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METRICS = ["PCC", "SSIM", "RMSE", "JS"]


def one_file(folder, suffix):
    paths = [p for p in folder.glob(f"*{suffix}") if "_interval_epoch" not in p.name
             and (suffix != "_score.tsv" or not p.name.endswith("_per_gene_score.tsv"))]
    if len(paths) != 1:
        raise ValueError(f"Expected one {suffix} in {folder}, found {len(paths)}")
    return paths[0]


def compare_folder(dataset, level, reference, candidate):
    old_path, new_path = one_file(reference, "_pred.npz"), one_file(candidate, "_pred.npz")
    with np.load(old_path, allow_pickle=False) as old, np.load(new_path, allow_pickle=False) as new:
        if not np.array_equal(old["genes"], new["genes"]):
            raise ValueError(f"Different target genes: {old_path}, {new_path}")
        old_cells, new_cells = old["cells"].astype(str), new["cells"].astype(str)
        # Merged artifacts strip the training 'st_' prefix, fold artifacts do not.
        if not np.array_equal(old_cells, new_cells):
            old_cells = np.char.removeprefix(old_cells, "st_") if hasattr(np.char, "removeprefix") else np.array([x.removeprefix("st_") for x in old_cells])
            new_cells = np.array([x.removeprefix("st_") for x in new_cells])
            if not np.array_equal(old_cells, new_cells):
                raise ValueError(f"Different cell order: {old_path}, {new_path}")
        old_pred, new_pred = old["pred"], new["pred"]
        if old_pred.shape != new_pred.shape:
            raise ValueError(f"Different prediction shape: {old_path}, {new_path}")
        details = {"dataset": dataset, "level": level, "n_cells": int(new_pred.shape[0]),
                   "n_genes": int(new_pred.shape[1]), "pred_exact": bool(np.array_equal(old_pred, new_pred)),
                   "pred_max_abs_delta": float(np.max(np.abs(old_pred - new_pred))),
                   "pred_mean_abs_delta": float(np.mean(np.abs(old_pred - new_pred)))}
    old_scores = pd.read_csv(one_file(reference, "_score.tsv"), sep="\t", index_col=0).iloc[:, 0]
    new_scores = pd.read_csv(one_file(candidate, "_score.tsv"), sep="\t", index_col=0).iloc[:, 0]
    rows = [{**details, "metric": metric, "reference": float(old_scores[metric]),
             "reproduced": float(new_scores[metric]), "delta": float(new_scores[metric] - old_scores[metric])}
            for metric in METRICS]
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["AD", "data1", "dev_brain", "heart_pcw13"])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, default=ROOT.parent / "GCAT_final/results")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    spec = json.loads((ROOT / "configs/benchmarks.json").read_text())["datasets"][args.dataset]
    reference = args.reference_root / spec["reference"]
    rows = []
    for fold in range(5):
        candidate = args.run / f"fold{fold}"
        if any(candidate.glob("*_pred.npz")) and any(candidate.glob("*_score.tsv")):
            rows.extend(compare_folder(args.dataset, f"fold{fold}", reference / f"fold{fold}", candidate))
    if any((args.run / "merged").glob("*_pred.npz")) and any((args.run / "merged").glob("*_score.tsv")):
        rows.extend(compare_folder(args.dataset, "merged", reference / "merged", args.run / "merged"))
    if not rows:
        raise ValueError("No completed predictions/scores found")
    result = pd.DataFrame(rows)
    output = args.output or args.run / "reference_comparison.tsv"
    result.to_csv(output, sep="\t", index=False, float_format="%.10g")
    print(result[["dataset", "level", "metric", "reference", "reproduced", "delta", "pred_exact"]].to_string(index=False))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
