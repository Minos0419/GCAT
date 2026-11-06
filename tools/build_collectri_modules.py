#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Collectri TF-regulon modules with optional uncovered-gene de novo slots."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--adata", type=Path, action="append", default=[])
    parser.add_argument(
        "--edge_csv",
        type=Path,
        required=True,
        help="Collectri human edge list (TSV or CSV).",
    )
    parser.add_argument("--edge_source_column", type=str, default="from")
    parser.add_argument("--edge_target_column", type=str, default="to")
    parser.add_argument("--edge_weight_column", type=str, default=None)
    parser.add_argument(
        "--regulator_list",
        type=Path,
        default=None,
        help="Optional one-regulator-per-line allowlist. Use this for a curated TF list.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min_targets", type=int, default=5)
    parser.add_argument("--max_targets", type=int, default=0)
    parser.add_argument(
        "--max_modules",
        type=int,
        default=0,
        help="Optional compute cap after ranking by target count and prior mass. 0 keeps all valid TFs.",
    )
    parser.add_argument("--min_edge_weight", type=float, default=0.0)
    parser.add_argument(
        "--require_regulator_in_task",
        action="store_true",
        help="Keep a TF module only when the regulator itself is in the task gene panel.",
    )
    parser.add_argument("--de_novo_modules", type=int, default=0)
    parser.add_argument(
        "--de_novo_uncovered_only",
        action="store_true",
        help="Assign de novo slots only to task genes not covered by prior modules.",
    )
    parser.add_argument(
        "--de_novo_init_weight",
        type=float,
        default=0.995,
        help="Weak round-robin bootstrap weight for de novo modules.",
    )
    parser.add_argument("--include_regulator", action="store_true", default=True)
    parser.add_argument("--no_include_regulator", action="store_false", dest="include_regulator")
    return parser.parse_args()


def read_model_genes(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as fp:
        config = json.load(fp)
    genes = [str(gene) for gene in config["gene_list"]]
    if len(genes) != len(set(genes)):
        raise ValueError("config gene_list contains duplicate identifiers.")
    return genes


def task_gene_context(
    paths: list[Path], model_genes: list[str]
) -> tuple[set[str], dict[str, str]]:
    if not paths:
        genes = set(model_genes)
        return genes, {gene: gene for gene in genes}
    shared: set[str] | None = None
    aliases: dict[str, set[str]] = {}
    model_gene_set = set(model_genes)
    for path in paths:
        data = ad.read_h5ad(path, backed="r")
        current = set(map(str, data.var_names))
        shared = current if shared is None else shared.intersection(current)
        symbols = data.var["gene_name"].astype(str).tolist() if "gene_name" in data.var else []
        for idx, gene in enumerate(map(str, data.var_names)):
            if gene not in model_gene_set:
                continue
            aliases.setdefault(gene, set()).add(gene)
            if symbols:
                aliases.setdefault(symbols[idx], set()).add(gene)
        if getattr(data, "file", None) is not None:
            data.file.close()
    used_genes = model_gene_set.intersection(shared or set())
    alias_to_gene = {
        alias: sorted(genes.intersection(used_genes))[0]
        for alias, genes in aliases.items()
        if genes.intersection(used_genes)
    }
    return used_genes, alias_to_gene


def regulator_allowlist(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    values = {
        line.strip().split("\t")[0]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if not values:
        raise ValueError(f"Regulator allowlist is empty: {path}")
    return values


def candidates_from_edge_list(
    args: argparse.Namespace,
    used_genes: set[str],
    alias_to_gene: dict[str, str],
) -> list[dict]:
    # ``gcat_collectri`` is distributed as TSV while older inputs are CSV.
    # Let pandas infer the delimiter so both formats use the same path.
    read_kwargs = {"sep": None, "engine": "python"}
    columns = pd.read_csv(args.edge_csv, nrows=0, **read_kwargs).columns.tolist()
    required = [args.edge_source_column, args.edge_target_column]
    if args.edge_weight_column is not None:
        required.append(args.edge_weight_column)
    missing = [column for column in required if column not in columns]
    if missing:
        raise ValueError(f"Missing edge-list columns {missing}; available columns are {columns}.")
    edges = pd.read_csv(args.edge_csv, usecols=required, **read_kwargs)
    edges[args.edge_source_column] = edges[args.edge_source_column].astype(str)
    edges[args.edge_target_column] = edges[args.edge_target_column].astype(str)
    allowed = regulator_allowlist(args.regulator_list)
    if allowed is not None:
        edges = edges[edges[args.edge_source_column].isin(allowed)]
    edges["_target_gene"] = edges[args.edge_target_column].map(alias_to_gene)
    edges = edges[edges["_target_gene"].isin(used_genes)]
    if args.edge_weight_column is None:
        edges["_edge_weight"] = 1.0
    else:
        edges["_edge_weight"] = pd.to_numeric(
            edges[args.edge_weight_column], errors="coerce"
        ).abs()
        edges = edges[edges["_edge_weight"].notna()]
    candidates: list[dict] = []
    for regulator, frame in edges.groupby(args.edge_source_column, sort=False):
        targets = (
            frame.groupby("_target_gene", sort=False)["_edge_weight"].max().astype(float).to_dict()
        )
        regulator_gene = alias_to_gene.get(str(regulator))
        candidates.append(
            {
                "regulator": str(regulator),
                "regulator_gene": regulator_gene,
                "targets": targets,
            }
        )
    return candidates


def main() -> None:
    args = parse_args()
    if args.min_targets < 1:
        raise ValueError("--min_targets must be >= 1.")
    if args.max_targets < 0 or args.max_modules < 0 or args.de_novo_modules < 0:
        raise ValueError("--max_targets, --max_modules, and --de_novo_modules must be >= 0.")
    if args.de_novo_init_weight <= 0:
        raise ValueError("--de_novo_init_weight must be > 0.")

    model_genes = read_model_genes(args.config)
    model_pos = {gene: idx for idx, gene in enumerate(model_genes)}
    used_genes, alias_to_gene = task_gene_context(args.adata, model_genes)
    raw_candidates = candidates_from_edge_list(args, used_genes, alias_to_gene)

    candidates: list[dict] = []
    for candidate in raw_candidates:
        regulator = candidate["regulator"]
        regulator_gene = candidate["regulator_gene"]
        if args.require_regulator_in_task and regulator_gene is None:
            continue
        targets = dict(candidate["targets"])
        target_count = len(targets)
        if target_count < args.min_targets:
            continue
        if args.max_targets > 0 and target_count > args.max_targets:
            ranked = sorted(targets.items(), key=lambda item: (-item[1], item[0]))[: args.max_targets]
            targets = dict(ranked)
            target_count = len(targets)
        module_genes = dict(targets)
        if args.include_regulator and regulator_gene is not None:
            module_genes[regulator_gene] = max(
                module_genes.get(regulator_gene, 0.0), max(module_genes.values(), default=1.0)
            )
        candidates.append(
            {
                "regulator": regulator,
                "regulator_gene": regulator_gene,
                "targets": module_genes,
                "target_count": target_count,
                "module_gene_count": len(module_genes),
                "prior_mass": float(sum(module_genes.values())),
            }
        )

    candidates.sort(key=lambda item: (-item["target_count"], -item["prior_mass"], item["regulator"]))
    total_valid = len(candidates)
    if args.max_modules > 0:
        candidates = candidates[: args.max_modules]
    if not candidates:
        raise ValueError("No TF regulon passed the requested target filters.")

    prior_module_count = len(candidates)
    total_module_count = prior_module_count + args.de_novo_modules
    assignment = np.zeros((len(model_genes), total_module_count), dtype=np.float32)
    module_rows: list[dict] = []
    membership_rows: list[dict] = []
    for module_idx, candidate in enumerate(candidates):
        targets = candidate["targets"]
        max_weight = max(targets.values())
        for rank, (gene, raw_weight) in enumerate(
            sorted(targets.items(), key=lambda item: (-item[1], item[0])), start=1
        ):
            weight = float(raw_weight / max(max_weight, 1e-12))
            assignment[model_pos[gene], module_idx] = weight
            membership_rows.append(
                {
                    "module": module_idx,
                    "module_name": f"TF::{candidate['regulator']}::out",
                    "module_type": "tf_prior",
                    "regulator": candidate["regulator"],
                    "rank": rank,
                    "gene": gene,
                    "prior_weight": weight,
                }
            )
        module_rows.append(
            {
                "module": module_idx,
                "module_name": f"TF::{candidate['regulator']}::out",
                "module_type": "tf_prior",
                "regulator": candidate["regulator"],
                "target_count": candidate["target_count"],
                "module_gene_count": candidate["module_gene_count"],
                "prior_mass": candidate["prior_mass"],
            }
        )

    prior_covered_task = sum(
        1
        for gene in used_genes
        if assignment[model_pos[gene], :prior_module_count].sum() > 0
    )
    if args.de_novo_modules > 0:
        task_genes_in_model_order = [gene for gene in model_genes if gene in used_genes]
        if args.de_novo_uncovered_only:
            task_genes_in_model_order = [
                gene for gene in task_genes_in_model_order
                if assignment[model_pos[gene], :prior_module_count].sum() <= 0
            ]
        for offset in range(args.de_novo_modules):
            module_idx = prior_module_count + offset
            module_name = f"DE_NOVO::{offset}"
            assigned_genes = task_genes_in_model_order[offset:: args.de_novo_modules]
            for rank, gene in enumerate(assigned_genes, start=1):
                assignment[model_pos[gene], module_idx] = float(args.de_novo_init_weight)
                membership_rows.append(
                    {
                        "module": module_idx,
                        "module_name": module_name,
                        "module_type": "de_novo",
                        "regulator": "",
                        "rank": rank,
                        "gene": gene,
                        "prior_weight": float(args.de_novo_init_weight),
                    }
                )
            module_rows.append(
                {
                    "module": module_idx,
                    "module_name": module_name,
                    "module_type": "de_novo",
                    "regulator": "",
                    "target_count": 0,
                    "module_gene_count": len(assigned_genes),
                    "prior_mass": 0.0,
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, assignment)
    prefix = args.out.with_suffix("")
    pd.DataFrame(module_rows).to_csv(
        prefix.with_name(prefix.name + "_modules.tsv"), sep="\t", index=False
    )
    pd.DataFrame(membership_rows).to_csv(
        prefix.with_name(prefix.name + "_membership.tsv"), sep="\t", index=False
    )
    prefix.with_name(prefix.name + "_genes.txt").write_text(
        "\n".join(model_genes) + "\n", encoding="utf-8"
    )

    covered_task = sum(
        1 for gene in used_genes if assignment[model_pos[gene]].sum() > 0
    )
    summary = {
        "assignment": str(args.out),
        "shape": list(assignment.shape),
        "module_semantics": "observed-TF outgoing regulons followed by unified de novo slots",
        "model_gene_count": len(model_genes),
        "task_gene_count": len(used_genes),
        "covered_task_gene_count": int(covered_task),
        "covered_task_gene_ratio": float(covered_task / max(1, len(used_genes))),
        "prior_covered_task_gene_count": int(prior_covered_task),
        "prior_covered_task_gene_ratio": float(prior_covered_task / max(1, len(used_genes))),
        "valid_tf_modules_before_cap": total_valid,
        "prior_module_count": prior_module_count,
        "allocated_tf_modules": prior_module_count,
        "de_novo_module_count": int(args.de_novo_modules),
        "de_novo_uncovered_only": bool(args.de_novo_uncovered_only),
        "allocated_total_modules": total_module_count,
        "min_targets": args.min_targets,
        "max_targets": args.max_targets,
        "max_modules": args.max_modules,
        "include_regulator": bool(args.include_regulator),
        "require_regulator_in_task": bool(args.require_regulator_in_task),
        "de_novo_init_weight": float(args.de_novo_init_weight),
        "prior_source": "collectri_edge_list",
        "edge_csv": str(args.edge_csv),
        "regulator_list": None if args.regulator_list is None else str(args.regulator_list),
        "adata": [str(path) for path in args.adata],
    }
    prefix.with_name(prefix.name + "_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
