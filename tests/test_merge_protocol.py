"""Do not label partial or mismasked folds as a completed five-fold benchmark."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tools.merge_cv_results import main


class MergeProtocolTests(unittest.TestCase):
    def test_partial_folds_cannot_be_merged_as_five_cv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mask = np.empty(5, dtype=object)
            for fold in range(5):
                mask[fold] = np.array([f"g{fold}"])
            mask_path = root / "five.npy"
            np.save(mask_path, mask)
            for fold in [0, 1]:
                folder = root / f"fold{fold}"
                folder.mkdir()
                (folder / "test_pred.npz").touch()
            with patch("sys.argv", ["merge_cv_results.py", "--dataset", "tiny", "--run_dir", str(root),
                                    "--query_data", str(root / "not_read.h5ad"), "--mask_gene", str(mask_path)]):
                with self.assertRaisesRegex(ValueError, "exactly one prediction file for every fold"):
                    main()


if __name__ == "__main__":
    unittest.main()
