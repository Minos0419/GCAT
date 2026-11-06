"""Prevent the wrong-mask cache reuse that invalidated earlier reproduction runs."""
import json
import tempfile
import unittest
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse

from scripts.train_v1 import _build_single_ccc_graph


class CacheProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.adata = ad.AnnData(sparse.csr_matrix(np.array([[0, 2], [3, 4], [5, 6]], dtype=np.float32)))
        self.adata.var_names = ["ligand", "receptor"]
        self.programs = [(["ligand"], ["receptor"])]
        self.candidates = sparse.csr_matrix(np.ones((3, 3), dtype=np.float32))

    def build(self, root, data=None):
        return _build_single_ccc_graph(
            self.adata if data is None else data, self.programs, root / "fold0_st.npz",
            2, 512, self.candidates, root / "fold0_st_programs.npz")

    def test_same_shape_different_mask_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.build(root)
            other = self.adata.copy()
            other.X[:, 1] = 0
            with self.assertRaisesRegex(ValueError, "input/mask mismatch"):
                self.build(root, other)

    def test_legacy_cache_verified_and_stale_legacy_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, _ = self.build(root)
            metadata = root / "fold0_st.provenance.json"
            metadata.unlink()
            graph, _ = self.build(root)
            np.testing.assert_array_equal(first.data, graph.data)
            self.assertEqual(json.loads(metadata.read_text())["version"], 1)
            metadata.unlink()
            other = self.adata.copy()
            other.X[:, 1] = 0
            with self.assertRaisesRegex(ValueError, "Legacy CCC cache does not match"):
                self.build(root, other)

    def test_corrupted_program_cache_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.build(root)
            path = root / "fold0_st_programs.npz"
            with np.load(path) as cached:
                values = {k: cached[k].copy() for k in cached.files}
            values["scores"] *= 2
            np.savez_compressed(path, **values)
            with self.assertRaisesRegex(ValueError, "program cache content changed"):
                self.build(root)


if __name__ == "__main__":
    unittest.main()
