#!/usr/bin/env python
"""One-command reproduction of every main-body experiment of the paper.

Stages (each can be skipped with the corresponding flag):

1. cache the four datasets
2. CoT baselines (Table 2 first row)
3. BBOX-ADAPTER: 4 datasets x 3 positive-sample settings x 2 adapter sizes (Table 2)
4. plug-and-play on davinci-002 and Mixtral-8x7B (Table 3)
5. cost analysis for StrategyQA and GSM8K (Table 4)
6. MLM-vs-NCE ablation (Table 5)
7. scale analysis: beams and iterations (Figure 3)
8. VRAM + accuracy for Mixtral-8x7B (Table 6)
9. assemble the tables from the artifacts

Requires the environment variables described in the README (Azure OpenAI) and,
for stages 4/6/8, a GPU.  Every stage tolerates a missing dataset mirror and
writes its artifacts under ``--runs-root``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(HERE, "scripts")


def run(command: list, check: bool = True) -> int:
    print("\n$ " + " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=HERE)
    if check and completed.returncode != 0:
        raise SystemExit(f"Command failed: {' '.join(command)}")
    return completed.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=str, default="runs")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--datasets", type=str, default="strategyqa,gsm8k,truthfulqa,scienceqa")
    parser.add_argument("--settings", type=str, default="ground_truth,ai_feedback,combined")
    parser.add_argument("--sizes", type=str, default="0.1B,0.3B")
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument("--skip-cot", action="store_true")
    parser.add_argument("--skip-table2", action="store_true")
    parser.add_argument("--skip-plug-and-play", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--skip-scale", action="store_true")
    parser.add_argument("--skip-cost", action="store_true")
    parser.add_argument("--skip-vram", action="store_true")
    args = parser.parse_args()

    python = sys.executable
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    runs = args.runs_root

    if not args.skip_data:
        run([python, os.path.join(SCRIPTS, "prepare_data.py"),
             "--data-dir", args.data_dir,
             "--datasets", args.datasets,
             "--limit-train", str(args.limit_train or ""),
             "--limit-test", str(args.limit_test or "")], check=False)

    # ---------------------------------------------------------------- CoT
    if not args.skip_cot:
        for dataset in datasets:
            run([python, os.path.join(SCRIPTS, "run_cot_baseline.py"),
                 "--config", f"configs/{dataset}.yaml",
                 "--data-dir", args.data_dir,
                 "--output-dir", os.path.join(runs, "baselines", dataset)], check=False)

    # -------------------------------------------------------------- Table 2
    if not args.skip_table2:
        command = [
            python, os.path.join(HERE, "reproduce_main_results.py"),
            "--data-dir", args.data_dir,
            "--output-root", os.path.join(runs, "table2"),
            "--datasets", args.datasets,
            "--settings", args.settings,
            "--sizes", args.sizes,
        ]
        if args.limit_train:
            command += ["--limit-train", str(args.limit_train)]
        if args.limit_test:
            command += ["--limit-test", str(args.limit_test)]
        run(command, check=False)

    # ---------------------------------------------------- plug-and-play
    if not args.skip_plug_and_play:
        for dataset in datasets:
            plugger = os.path.join(runs, "table2", dataset, "combined_0.3B", "adapter_final")
            if not os.path.isdir(plugger):
                plugger = os.path.join(runs, "table2", dataset, "combined_0.1B", "adapter_final")
            if os.path.isdir(plugger):
                run([python, os.path.join(SCRIPTS, "run_plug_and_play.py"),
                     "--config", f"configs/{dataset}.yaml",
                     "--plugger-adapter", plugger,
                     "--data-dir", args.data_dir,
                     "--output-dir", os.path.join(runs, "plug_and_play", dataset)], check=False)

    # --------------------------------------------------------- ablation
    if not args.skip_ablation:
        for dataset in datasets:
            run([python, os.path.join(SCRIPTS, "ablation_loss.py"),
                 "--config", f"configs/{dataset}.yaml",
                 "--data-dir", args.data_dir,
                 "--output-dir", os.path.join(runs, "ablation", dataset)], check=False)

    # ---------------------------------------------------- scale analysis
    if not args.skip_scale:
        for dataset in datasets:
            run([python, os.path.join(SCRIPTS, "scale_analysis.py"),
                 "--config", f"configs/{dataset}.yaml",
                 "--run-dir", os.path.join(runs, "table2", dataset, "combined"),
                 "--data-dir", args.data_dir,
                 "--output-dir", os.path.join(runs, "scale", dataset)], check=False)

    # ------------------------------------------------------------- cost
    if not args.skip_cost:
        for dataset in ("strategyqa", "gsm8k"):
            adapter = os.path.join(runs, "table2", dataset, "ground_truth_0.1B", "adapter_final")
            if os.path.isdir(adapter):
                run([python, os.path.join(SCRIPTS, "cost_analysis.py"),
                     "--config", f"configs/{dataset}.yaml",
                     "--adapter-path", adapter,
                     "--training-results",
                     os.path.join(runs, "table2", dataset, "ground_truth_0.1B", "results.json"),
                     "--data-dir", args.data_dir,
                     "--output-dir", os.path.join(runs, "cost", dataset)], check=False)

    # ------------------------------------------------------------- VRAM
    if not args.skip_vram:
        run([python, os.path.join(SCRIPTS, "measure_vram.py"),
             "--config", "configs/mixtral_strategyqa.yaml",
             "--output-dir", os.path.join(runs, "vram")], check=False)

    # ----------------------------------------------------------- tables
    run([python, os.path.join(SCRIPTS, "make_tables.py"),
         "--runs-root", os.path.join(runs, "table2"),
         "--output-dir", os.path.join(runs, "tables"),
         "--datasets", args.datasets], check=False)
    print(f"\nAll artifacts are under {os.path.abspath(runs)}")


if __name__ == "__main__":
    main()
