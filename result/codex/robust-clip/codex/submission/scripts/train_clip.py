#!/usr/bin/env python
"""Train a robust (FARE / TeCoA) CLIP vision encoder on ImageNet (Sec. 3, App. B)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robust_clip.training.train import main  # noqa: E402

if __name__ == "__main__":
    main()
