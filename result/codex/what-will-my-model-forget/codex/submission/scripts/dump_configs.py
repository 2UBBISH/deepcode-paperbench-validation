#!/usr/bin/env python
"""Write the JSON configuration of every Table 1 / Table 3 column into configs/."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wwmf.config import ExperimentConfig  # noqa: E402
from wwmf.utils import ensure_dir  # noqa: E402

SETUPS = [
    ("bart0_large", "p3_test", "head"),
    ("bart0_large", "p3_test", "full_ft"),
    ("flan_t5_large", "mmlu", "head"),
    ("flan_t5_large", "mmlu", "lora"),
    ("flan_t5_large", "mmlu", "full_ft"),
    ("flan_t5_3b", "mmlu", "head"),
    ("flan_t5_3b", "mmlu", "lora"),
]


def main() -> None:
    ensure_dir("configs")
    for model, data, mode in SETUPS:
        cfg = ExperimentConfig(model=model, refinement_data=data, tuning_mode=mode)
        payload = cfg.to_dict()
        payload["hyperparameters"] = {
            "steps_single_error": cfg.steps_single(),
            "lr_single_error": cfg.lr_single(),
            "lr_sequential": cfg.lr_sequential(),
            "replay": cfg.replay_schedule(),
        }
        path = os.path.join("configs", f"{model}_{mode}.json")
        with open(path, "w", encoding="utf8") as fh:
            json.dump(payload, fh, indent=2)
        print("wrote", path)


if __name__ == "__main__":
    main()
