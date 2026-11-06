"""Tests for modality balancing and task-aligned whole-gene masking."""

from __future__ import annotations

import copy
import random
import unittest

import torch

from gcat.data.loader import BalancedShuffledCombinedLoader
from gcat.modules.loss import masked_gene_standardized_mse
from gcat.modules.masking import MaskBuilder, drop_nodes


class _TinyLoader:
    def __init__(self, label: str, length: int):
        self.label = label
        self.length = length

    def __len__(self):
        return self.length

    def __iter__(self):
        for index in range(self.length):
            yield {"label": self.label, "index": index}


class BalancedLoaderTests(unittest.TestCase):
    def test_shorter_loader_is_cycled_to_equal_steps(self):
        random.seed(11)
        loader = BalancedShuffledCombinedLoader(
            [_TinyLoader("st", 3), _TinyLoader("sc", 1)]
        )
        batches = list(loader)
        self.assertEqual(len(loader), 6)
        self.assertEqual(sum(batch["label"] == "st" for batch in batches), 3)
        self.assertEqual(sum(batch["label"] == "sc" for batch in batches), 3)


class WholeGeneMaskTests(unittest.TestCase):
    def test_masks_same_observed_genes_across_st_only(self):
        torch.manual_seed(7)
        builder = MaskBuilder(
            mask_node_rate=0.0,
            mask_feature_rate=0.0,
            st_block_gene_mask_rate=0.5,
        ).train()
        batch = {
            "x_seq": torch.arange(1, 25, dtype=torch.float32).reshape(4, 6),
            "gene_mask": torch.tensor([True, True, False, True, True, False]),
            "source": [False, False, True, True],
        }
        original = batch["x_seq"].clone()
        result = builder.apply_mask(copy.deepcopy(batch))
        block = result["st_block_gene_mask"]

        # Four platform-observed genes at rate 0.5 -> two shared masked genes.
        self.assertEqual(int(block[0].sum()), 2)
        torch.testing.assert_close(block[0], block[1], rtol=0, atol=0)
        self.assertFalse(bool(block[2:].any()))
        self.assertFalse(bool(block[:, 2].any()))
        self.assertFalse(bool(block[:, 5].any()))
        dense_masked = result["masked_x_seq"].to_dense()
        self.assertTrue(bool((dense_masked[block] == 0).all()))
        torch.testing.assert_close(result["x_seq"], original, rtol=0, atol=0)


class DropNodeShapeRegressionTests(unittest.TestCase):
    def test_square_expression_matrix_keeps_gene_axis(self):
        torch.manual_seed(7)
        expression = torch.arange(1, 17, dtype=torch.float32).reshape(4, 4)
        ccc_bias = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        batch = {
            "x_seq": expression.to_sparse(),
            "x_seq_agg_neighbors": (expression + 100).to_sparse(),
            "ccc_bias": ccc_bias,
            "seed_mask": torch.tensor([True, True, False, False]),
            "gene_mask": torch.ones(4, dtype=torch.bool),
            "gene_list": ["g0", "g1", "g2", "g3"],
        }

        result = drop_nodes(copy.deepcopy(batch), drop_node_rate=0.5, max_batch_size=4)
        kept = result["_drop_node_cell_idx"]

        # Expression-like tensors are cell-by-gene even when accidentally square.
        self.assertEqual(tuple(result["x_seq"].shape), (3, 4))
        self.assertEqual(tuple(result["x_seq_agg_neighbors"].shape), (3, 4))
        torch.testing.assert_close(
            result["x_seq"].to_dense(), expression.index_select(0, kept), rtol=0, atol=0
        )
        # Explicit cell-by-cell fields still require both axes to be remapped.
        expected_ccc = ccc_bias.index_select(0, kept).index_select(1, kept)
        torch.testing.assert_close(result["ccc_bias"], expected_ccc, rtol=0, atol=0)
        self.assertEqual(result["gene_list"], batch["gene_list"])


class MaskedStandardizedLossTests(unittest.TestCase):
    def test_affine_prediction_has_zero_standardized_loss(self):
        truth = torch.tensor(
            [[1.0, 4.0], [2.0, 1.0], [4.0, 3.0], [8.0, 2.0]]
        )
        pred = truth * torch.tensor([3.0, 2.0]) + torch.tensor([5.0, 7.0])
        mask = torch.ones_like(truth, dtype=torch.bool)
        loss = masked_gene_standardized_mse(pred, truth, mask)
        self.assertLess(float(loss), 1e-10)

    def test_seed_and_st_filters_are_respected_and_gradient_flows(self):
        truth = torch.tensor([[1.0], [2.0], [4.0], [8.0]])
        pred = torch.tensor([[3.0], [1.0], [2.0], [99.0]], requires_grad=True)
        mask = torch.ones_like(truth, dtype=torch.bool)
        loss = masked_gene_standardized_mse(
            pred,
            truth,
            mask,
            seed_mask=torch.tensor([True, True, True, False]),
            source_mask=torch.tensor([False, False, False, True]),
        )
        self.assertGreater(float(loss), 1.0)
        loss.backward()
        self.assertGreater(float(pred.grad[:3].abs().sum()), 0.0)
        self.assertEqual(float(pred.grad[3:].abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
