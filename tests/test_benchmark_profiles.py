"""Keep the legacy single-fold launcher's defaults aligned with final profiles."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class BenchmarkProfileTests(unittest.TestCase):
    def test_single_fold_defaults_match_selected_profiles(self):
        profiles = json.loads((ROOT / "configs/benchmarks.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            shim = tmp / "args_only"
            shim.write_text(f"#!{sys.executable}\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n")
            shim.chmod(0o755)
            for dataset, spec in profiles["datasets"].items():
                with self.subTest(dataset=dataset):
                    env = {k: v for k, v in os.environ.items()
                           if not k.endswith("_OVERRIDE") and k not in ("GCAT_DATA_ROOT", "GCAT_CONFIG", "NICHE_BRIDGE_STAGE")}
                    env["PYTHON_BIN"] = str(shim)
                    result = subprocess.run(
                        ["bash", str(ROOT / "scripts/run_fold0.sh"), dataset, str(tmp / dataset / "fold0")],
                        check=True, env=env, capture_output=True, text=True)
                    args = json.loads(result.stdout.strip())
                    options, flags = {}, set()
                    index = 1  # train_v1.py path
                    while index < len(args):
                        name = args[index]
                        if index + 1 < len(args) and not args[index + 1].startswith("--"):
                            options[name[2:]] = args[index + 1]
                            index += 2
                        else:
                            flags.add(name)
                            index += 1
                    for key in ("epoch", "batch_size", "ccc_min_pairs"):
                        self.assertEqual(options[key], str(spec[key]))
                    self.assertEqual(float(options.get("dropout", profiles["common"]["dropout"])),
                                     spec.get("dropout", profiles["common"]["dropout"]))
                    self.assertEqual("--balance_modalities" in flags, spec["balance_modalities"])
                    self.assertIn("--flow_cell_attention", flags)
                    self.assertIn("--no_ccc_bias", flags)
                    mask = spec["mask"]
                    expected = ROOT / mask[len("@package/"):] if mask.startswith("@package/") else ROOT / "../../data" / mask
                    self.assertEqual(Path(options["mask_gene"]).resolve(), expected.resolve())
                    for key in ("random_seed", "lr", "warmup_epochs", "lambda_reg", "prior_strength",
                                "flow_attn_layers", "flow_attn_mode", "amp_mode"):
                        value = profiles["common"][key]
                        if isinstance(value, (int, float)):
                            self.assertEqual(float(options[key]), value)
                        else:
                            self.assertEqual(options[key], value)


if __name__ == "__main__":
    unittest.main()
