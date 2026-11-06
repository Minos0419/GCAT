#!/usr/bin/env python
import argparse
import pathlib

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a temporary mask file whose fold 0 is the requested CV fold.")
    parser.add_argument("--mask", type=pathlib.Path, required=True, help="Original 5/10-fold gene mask .npy.")
    parser.add_argument("--fold", type=int, default=None, help="Fold index to move to position 0.")
    parser.add_argument("--out_dir", type=pathlib.Path, default=None, help="Output directory for temporary mask files.")
    parser.add_argument("--list_folds", action="store_true", help="Print available fold indices and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    folds = np.load(args.mask, allow_pickle=True)
    n_folds = int(len(folds))
    if args.list_folds:
        print(" ".join(str(i) for i in range(n_folds)))
        return
    if args.fold is None:
        raise ValueError("--fold is required unless --list_folds is set.")
    if args.fold < 0 or args.fold >= n_folds:
        raise ValueError(f"Requested fold {args.fold}, but {args.mask} has {n_folds} folds.")

    out_dir = args.out_dir or args.mask.parent / "cv_fold_masks"
    out_dir.mkdir(parents=True, exist_ok=True)
    order = [args.fold] + [i for i in range(n_folds) if i != args.fold]
    reordered = np.asarray([folds[i] for i in order], dtype=object)
    out_path = out_dir / f"{args.mask.stem}_fold{args.fold}_as0.npy"
    np.save(out_path, reordered, allow_pickle=True)
    print(out_path)


if __name__ == "__main__":
    main()
