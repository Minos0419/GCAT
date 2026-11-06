"""Compare full-size S4 initialization across isolated source trees."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def worker(root: Path, dataset: str, training_check: bool = False) -> dict:
    sys.path.insert(0, str(root.resolve()))
    import torch
    from gcat.model import GCATFormer
    from gcat.modules.common import set_seed

    torch.set_num_threads(2)
    config = json.loads((root / "assets/config.json").read_text())
    module = next((root / "assets/regulons").glob(f"{dataset}_collectri*_denovo*.npy"))
    config.update(
        gene_token_mode="module", gene_module_num="auto",
        gene_module_init_path=str(module.resolve()), dynamic_gene_modules=True,
        prior_strength=3.0, off_prior_init_std=0.02, assignment_temperature=1.0,
        mask_node_rate=0.95, mask_feature_rate=0.25, drop_node_rate=0.25,
        max_batch_size=70000, pair_loss_weight=10.0, pair_mask_token=-100.0,
        lambda_reg=0.1, use_ccc_bias=False, ccc_bias_trainable=True,
        flow_cell_attention=True, flow_attn_layers="last", flow_attn_mode="gated",
        ccc_program_num={"AD": 62, "data1": 75, "dev_brain": 17, "heart_pcw13": 12}[dataset],
    )
    if dataset == "heart_pcw13":
        config["dropout"] = 0.0
    set_seed(11)
    model = GCATFormer(**config)
    def digest(tensor):
        return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    result = {
        "root": str(root.resolve()), "dataset": dataset,
        "rng_after_init": digest(torch.get_rng_state()),
        "state": {key: {"shape": list(value.shape), "sha256": digest(value)}
                  for key, value in model.state_dict().items()},
    }
    if training_check:
        count, genes = 6, config["gene_list"][:64]
        x = (torch.arange(count * len(genes)).reshape(count, -1) % 7).float()
        edge = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]])
        batch = {
            "x_seq": x, "gene_mask": torch.tensor([True] * 48 + [False] * 16),
            "source": torch.zeros(count, dtype=torch.bool),
            "seed_mask": torch.ones(count, dtype=torch.bool),
            "ccc_program_edge_index": edge,
            "ccc_program_score": torch.ones(edge.shape[1], config["ccc_program_num"]),
            "ccc_inflow_activity": torch.ones(count, config["ccc_program_num"]),
            "ccc_outflow_activity": torch.ones(count, config["ccc_program_num"]),
        }
        model.train()
        out, loss = model(batch, genes)
        loss.backward()
        result["training"] = {"loss": float(loss.detach()), "prediction": digest(out["pred"]),
                              "rng": digest(torch.get_rng_state()),
                              "gradients": {key: digest(value.grad) for key, value in model.named_parameters()
                                            if value.grad is not None}}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=["AD", "data1"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--training-check", action="store_true", help="Also compare a seeded CPU training forward/backward on identical synthetic inputs")
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.roots[0], args.datasets[0], args.training_check)))
        return
    report = {"comparisons": [], "runs": []}
    for dataset in args.datasets:
        runs = []
        for root in args.roots:
            proc = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--worker",
                 "--roots", str(root.resolve()), "--datasets", dataset]
                + (["--training-check"] if args.training_check else []),
                check=True, text=True, capture_output=True,
            )
            runs.append(json.loads(proc.stdout))
        report["runs"].extend(runs)
        reference = runs[0]
        for run in runs[1:]:
            differing = [key for key in reference["state"].keys() & run["state"].keys()
                         if reference["state"][key] != run["state"][key]]
            comparison = {
                "dataset": dataset, "reference": reference["root"], "candidate": run["root"],
                "differing_tensors": sorted(differing),
                "reference_only": sorted(reference["state"].keys() - run["state"].keys()),
                "candidate_only": sorted(run["state"].keys() - reference["state"].keys()),
                "rng_equal": reference["rng_after_init"] == run["rng_after_init"],
            }
            if args.training_check:
                comparison["training_equal"] = reference["training"] == run["training"]
            report["comparisons"].append(comparison)
            print(json.dumps(comparison), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    if any(c["differing_tensors"] or c["reference_only"] or c["candidate_only"]
           or not c["rng_equal"] or not c.get("training_equal", True) for c in report["comparisons"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
