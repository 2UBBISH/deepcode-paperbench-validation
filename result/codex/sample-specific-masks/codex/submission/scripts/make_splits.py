#!/usr/bin/env python3
"""Report (and cache) the train/test splits used for every dataset.

Table 6 of the paper fixes the number of training and testing images; this
script rebuilds them from the downloaded data and prints the realised sizes so
that they can be compared with the paper.

    python scripts/make_splits.py --data-root data --datasets cifar10 dtd
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smm.datasets import DATASET_SPECS, build_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--datasets", nargs="*", default=sorted(DATASET_SPECS))
    parser.add_argument("--split-dir", default=None)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    print(f"{'dataset':<12}{'train':>10}{'test':>10}   (paper: train/test)")
    for name in args.datasets:
        spec = DATASET_SPECS[name]
        try:
            train = build_dataset(name, "train", spec.resolution, root=args.data_root,
                                  download=not args.no_download, split_dir=args.split_dir)
            test = build_dataset(name, "test", spec.resolution, root=args.data_root,
                                 download=not args.no_download, split_dir=args.split_dir)
        except Exception as exc:  # pragma: no cover - data dependent
            print(f"{name:<12}    unavailable ({type(exc).__name__}: {exc})")
            continue
        print(f"{name:<12}{len(train):>10}{len(test):>10}   "
              f"({spec.train_size}/{spec.test_size})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
