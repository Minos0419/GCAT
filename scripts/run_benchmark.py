"""Canonical S4 benchmark runner; explicit profiles, five-fold validation, provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "results/.matplotlib"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def preflight(dataset: str, data_root: Path, spec: dict) -> dict:
    import anndata as ad
    import hdf5plugin  # noqa: F401
    import numpy as np
    from scripts.train_v1 import load_ensembl_lr_programs

    config_path = ROOT / "assets/config.json"
    config = json.loads(config_path.read_text())
    mask = spec["mask"]
    mask_path = ROOT / mask[len("@package/"):] if mask.startswith("@package/") else data_root / mask
    paths = {
        "query_data": data_root / spec["query"], "ref_data": data_root / spec["ref"],
        "mask_gene": mask_path,
        "gene_module_init_path": ROOT / "assets/regulons" / spec["module"],
        "config": config_path,
        "ccc_ensembl_interaction_cache": ROOT / "assets/interactions_human__ensembl_lr.pt",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    q = ad.read_h5ad(paths["query_data"], backed="r")
    r = ad.read_h5ad(paths["ref_data"], backed="r")
    config_genes, ref_genes = set(config["gene_list"]), set(r.var_names)
    shared = [str(g) for g in q.var_names if g in config_genes and g in ref_genes]
    folds = np.load(mask_path, allow_pickle=True)
    sizes = [len(fold) for fold in folds]
    if sizes != spec["expected_fold_genes"]:
        raise ValueError(f"{dataset}: expected five-fold gene counts {spec['expected_fold_genes']}, got {sizes} ({mask_path})")
    union = [str(g) for fold in folds for g in fold]
    if len(union) != len(set(union)) or set(union) != set(shared):
        raise ValueError(f"{dataset}: masks must be disjoint and cover exactly all {len(shared)} shared genes")
    programs = load_ensembl_lr_programs(paths["ccc_ensembl_interaction_cache"], shared)
    if len(programs) != spec["expected_lr_programs"]:
        raise ValueError(f"{dataset}: expected {spec['expected_lr_programs']} LR programs, got {len(programs)}; check assets/data, do not lower the threshold")
    info = {"n_cells": int(q.n_obs), "n_ref_cells": int(r.n_obs),
            "n_genes": len(shared), "fold_genes": sizes, "n_lr_programs": len(programs)}
    q.file.close()
    r.file.close()
    return {"paths": {k: str(v.resolve()) for k, v in paths.items()}, "data": info,
            "input_sha256": {k: sha256(v) for k, v in paths.items()}}


def validate_predictions(run_dir: Path, prefix: str, mask_path: str, fold: int) -> None:
    import numpy as np
    with np.load(run_dir / f"{prefix}_pred.npz", allow_pickle=False) as pred:
        expected = np.load(mask_path, allow_pickle=True)[fold].astype(str)
        if not np.array_equal(pred["genes"], expected):
            raise ValueError(f"Prediction genes do not match benchmark fold {fold}: {run_dir}")
        if pred["pred"].shape != (len(pred["cells"]), len(expected)):
            raise ValueError(f"Prediction dimensions do not match cells/genes: {run_dir}")
        if not np.isfinite(pred["pred"]).all():
            raise ValueError(f"Non-finite predictions: {run_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["AD", "data1", "dev_brain", "heart_pcw13"])
    parser.add_argument("--data-root", type=Path, default=ROOT / "../../data")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cache-source", type=Path, help="Read-only legacy cache to import into this run (same benchmark masks required)")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip completed folds only after checking their provenance and predictions")
    parser.add_argument("--deterministic", action="store_true", help="Diagnostic protocol; differs from archived fast FP16 training")
    args = parser.parse_args()
    if len(args.folds) != len(set(args.folds)) or any(f not in range(5) for f in args.folds):
        parser.error("--folds must contain unique indices in 0..4")
    profiles = json.loads((ROOT / "configs/benchmarks.json").read_text())
    spec = profiles["datasets"][args.dataset]
    prepared = preflight(args.dataset, args.data_root.resolve(), spec)
    options = dict(profiles["common"])
    options.update({k: spec[k] for k in ["epoch", "batch_size", "ccc_min_pairs"]})
    options["dropout"] = spec.get("dropout", options["dropout"])
    options["device"] = args.device
    flags = ["--dynamic_gene_modules", "--flow_cell_attention", "--no_ccc_bias", "--ccc_bias_trainable",
             "--save_speed_metrics", "--save_per_gene_score", "--save_pred"]
    if spec["balance_modalities"]:
        flags.append("--balance_modalities")
    if args.deterministic:
        flags.append("--deterministic")
    source = {str(p.relative_to(ROOT)): sha256(p) for folder in ["gcat", "scripts"]
              for p in sorted((ROOT / folder).rglob("*.py"))}
    identity = {"dataset": args.dataset, **prepared, "options": options, "flags": flags,
                "source_sha256": source, "python": sys.executable}
    print(json.dumps({"dataset": args.dataset, **prepared["data"], "options": options,
                      "mask": prepared["paths"]["mask_gene"]}, indent=2), flush=True)
    if args.check_only:
        return
    import torch
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; run in the GPU-enabled host environment")
    out = args.out.resolve()
    manifest_path = out / "run_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if not args.resume or previous["identity"] != identity:
            raise ValueError(f"Existing run requires --resume and identical inputs/source/options: {out}")
        manifest = previous
    else:
        if out.exists() and any(out.iterdir()):
            raise ValueError(f"Output directory must be empty for a new run: {out}")
        out.mkdir(parents=True, exist_ok=True)
        manifest = {"identity": identity, "created_utc": timestamp(), "folds": {}, "environment": {
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else None,
            "python_version": sys.version,
            "env": {k: os.environ.get(k) for k in ["PYTHONHASHSEED", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES"]},
        }}
    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    save()
    cache_dir = out / "ccc_cache"
    for fold in args.folds:
        fold_dir = out / f"fold{fold}"
        prefix = f"gcat_{args.dataset}_fold{fold}_s4"
        prior = manifest["folds"].get(str(fold), {})
        if prior.get("status") == "complete":
            validate_predictions(fold_dir, prefix, prepared["paths"]["mask_gene"], fold)
            print(f"SKIP complete fold {fold}", flush=True)
            continue
        if fold_dir.exists() and any(fold_dir.iterdir()):
            raise ValueError(f"Incomplete fold directory: use a fresh run directory: {fold_dir}")
        fold_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(exist_ok=True)
        # Imported caches are copied, never mutated in the source directory.
        # The cache provenance guard inside train_v1 validates them against
        # freshly computed masked expression before any training begins.
        if args.cache_source:
            import shutil
            for suffix in ["st.npz", "sc.npz", "st_programs.npz", "sc_programs.npz"]:
                source_path = args.cache_source / f"fold{fold}_{suffix}"
                if not source_path.is_file():
                    raise FileNotFoundError(source_path)
                shutil.copy2(source_path, cache_dir / source_path.name)
        cmd = [sys.executable, str(ROOT / "scripts/train_v1.py")]
        for key, value in {**prepared["paths"], **options, "mask_fold_index": fold,
                           "ccc_cache_dir": str(cache_dir), "ccc_cache_name": f"fold{fold}",
                           "out_dir": str(fold_dir), "prefix": prefix}.items():
            cmd.extend([f"--{key}", str(value)])
        cmd.extend(flags)
        manifest["folds"][str(fold)] = {"status": "running", "started_utc": timestamp(), "command": cmd}
        save()
        print(shlex.join(cmd), flush=True)
        with (fold_dir / "train.log").open("x") as log:
            process = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            manifest["folds"][str(fold)]["pid"] = process.pid
            save()
            code = process.wait()
        manifest["folds"][str(fold)].update(status="failed" if code else "validating", exit_code=code, finished_utc=timestamp())
        save()
        if code:
            raise RuntimeError(f"fold {fold} exited {code}; see {fold_dir / 'train.log'}")
        validate_predictions(fold_dir, prefix, prepared["paths"]["mask_gene"], fold)
        manifest["folds"][str(fold)]["status"] = "complete"
        save()
        print(f"COMPLETE {args.dataset} fold {fold}", flush=True)
    if all(manifest["folds"].get(str(f), {}).get("status") == "complete" for f in range(5)):
        subprocess.run([sys.executable, str(ROOT / "tools/merge_cv_results.py"),
                        "--dataset", args.dataset, "--run_dir", str(out),
                        "--query_data", prepared["paths"]["query_data"],
                        "--mask_gene", prepared["paths"]["mask_gene"],
                        "--prefix", f"{args.dataset}_s4"], check=True, cwd=ROOT)
        manifest["merged_utc"] = timestamp()
        save()


if __name__ == "__main__":
    main()
