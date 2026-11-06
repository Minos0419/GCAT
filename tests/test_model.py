"""GCAT_v2 core tests: S4 flow attention, zero-gate baseline equivalence."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from gcat.model.core import GCATFormer


class GCATv2Tests(unittest.TestCase):
    def _module_path(self, root: Path) -> Path:
        path = root / "modules.npy"
        np.save(path, np.asarray(
            [[1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0]], dtype=np.float32))
        (root / "modules_summary.json").write_text(
            json.dumps({"prior_module_count": 2, "de_novo_module_count": 2}))
        return path

    def _model(self, module_path, **extra):
        kwargs = dict(
            gene_list=["g0", "g1", "g2", "g3"], hidden_dim=16, num_layers=2,
            latent_dim=8, decoder_hidden_dim=16, decoder_layers=2, output_dim=4,
            dropout=0.0, num_heads=2, mask_node_rate=0.0, mask_feature_rate=0.0,
            drop_node_rate=0.0, gene_token_mode="module", gene_module_num="auto",
            gene_module_init_path=str(module_path), use_ccc_bias=False, lambda_reg=0.1)
        kwargs.update(extra)
        return GCATFormer(**kwargs)

    @staticmethod
    def _batch():
        return {
            "x_seq": torch.tensor([[1., 2., 0., 3.], [0., 1., 4., 2.], [2., 0., 1., 1.]]),
            "gene_mask": torch.ones(4, dtype=torch.bool),
            "source": torch.ones(3, dtype=torch.bool),
            "ccc_program_edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            "ccc_program_score": torch.rand(2, 2),
            "ccc_inflow_activity": torch.rand(3, 2),
            "ccc_outflow_activity": torch.rand(3, 2),
        }

    def test_forward_backward_and_gate_gradient(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = self._model(self._module_path(Path(tmp)),
                                flow_cell_attention=True, flow_attn_layers="last",
                                ccc_program_num=2).eval()
            out, loss = model(self._batch())
            self.assertEqual(tuple(out["pred"].shape), (3, 4))
            self.assertTrue(out["flow_cell_attention"])
            loss.backward()
            self.assertGreater(model.flow_attn_gate.grad.abs().sum().item(), 0.0)
            self.assertIsNotNone(model.pre_latent_norm.weight.grad)
            self.assertGreater(model.pre_latent_norm.weight.grad.abs().sum().item(), 0.0)

    def test_zero_gate_equals_no_flow_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            mp = self._module_path(Path(tmp))
            torch.manual_seed(7)
            base = self._model(mp).eval()
            torch.manual_seed(7)
            flow = self._model(mp, flow_cell_attention=True, flow_attn_layers="last",
                               ccc_program_num=2).eval()
            bo, bl = base(self._batch())
            fo, fl = flow(self._batch())
            torch.testing.assert_close(fo["pred"], bo["pred"], rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(fl, bl, rtol=1e-5, atol=1e-6)

    def test_legacy_kwargs_accepted_and_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = self._model(
                self._module_path(Path(tmp)),
                flow_cell_attention=True, ccc_program_num=2,
                lambda_out=0.0, flow_linear_bottleneck=False, flow_stage5=False,
                grn_decoder_residual=False, lambda_pull=0.0,
                niche_regulon_bridge=False, ccc_program_head_bias=False)
            out, _ = model(self._batch())
            self.assertEqual(tuple(out["pred"].shape), (3, 4))


if __name__ == "__main__":
    unittest.main()
