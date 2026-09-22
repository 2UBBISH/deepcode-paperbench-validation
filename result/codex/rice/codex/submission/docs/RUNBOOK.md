# Runbook for the full reproduction

Everything below is meant to be executed on the machine that runs the
experiments (the paper used 8×A100).  The CPU-only grading environment only
runs `tests/`; see `docs/SMOKE_RESULTS.md`.

## 0. Setup

```bash
pip install -r requirements.txt
# optional applications
git clone https://github.com/cage-challenge/cage-challenge-2 && pip install -e cage-challenge-2
export RICE_CAGE_PATH=<path to a cage-2 scenario yaml>
python -c "import rice, torch, gymnasium, mujoco; print('ok')"
```

## 1. Pre-train the target agents (one per application and seed)

```bash
python scripts/pretrain_agents.py \
    --envs Hopper Walker2d Reacher HalfCheetah SelfishMining \
    --steps 300000 --seeds 0 1 2
```

`checkpoints/agents/<env>_seed<k>.pt` plus `results/pretrained_agents.json`
(the "No Refine" column of Table 1) are produced.

## 2. Mask networks (Algorithm 1 + StateMask baseline)

```bash
for env in Hopper Walker2d Reacher HalfCheetah SelfishMining; do
  python scripts/train_mask_network.py --env $env --method ours      --agent checkpoints/agents/${env}_seed0.pt
  python scripts/train_mask_network.py --env $env --method statemask --agent checkpoints/agents/${env}_seed0.pt
done
```

Both runs use the sample budget of Table 4
(`ENV_SPECS[env].mask_samples`) and write their wall-clock time, which is the
efficiency comparison of the paper (~16.8 % faster with our objective).

If the mask rate of a run collapses to 0 or 1 (the two degenerate corners of
`R + alpha * a^m`), pick `alpha` for that application first:

```bash
python scripts/calibrate_alpha.py --env <ENV> --agent checkpoints/agents/<ENV>_seed0.pt
```

and put the reported value into `EnvSpec.alpha` in `rice/envs/registry.py`
(`alpha = 0.01` is the paper's value and the default everywhere; the selfish
mining MDP of this repository uses 0.05 for the reason documented in
`docs/EXPERIMENTS.md`).

## 3. Experiments

| Experiment | Command | Paper output |
|---|---|---|
| I (fidelity + efficiency) | `python scripts/run_experiment1_fidelity.py --env <ENV>` | Figure 5, Table 4 |
| II + III (refining) | `python scripts/run_experiment2_refine.py --env <ENV> --seeds 0 1 2` | Table 1 |
| III (explanations only) | `python scripts/run_experiment3_explanations.py --env <ENV>` | Table 1 (right half) |
| II on sparse games | `python scripts/run_experiment2_refine.py --env SparseHopper --seeds 0 1 2` (same for `SparseHalfCheetah`) | Figure 2 |
| IV (SAC) | `python scripts/run_experiment4_sac.py --env Hopper --seeds 0 1 2` | Figure 3, Figure 6 |
| V (hyper-parameters) | `python scripts/run_experiment5_hyperparams.py --env <ENV> --seeds 0 1 2` | Figures 7, 8, 9 |

For a cluster, one job per (application, method, seed) can be launched with
`scripts/run_one.py`; `scripts/run_all.py` is the sequential version.

## 4. Aggregation and figures

```bash
python scripts/make_tables.py --results results      # reproduction vs paper
python scripts/plot_results.py --results results --out figures
```

## 5. Budget planning

With the reduced budgets of `scripts/smoke_pipeline.py` the whole pipeline runs
in a couple of minutes on CPU.  With the paper's budgets, one application
requires

* `3·10^5` steps × 3 seeds of pre-training,
* `3·10^5` samples of mask training per explanation method,
* `3·10^5` steps × 4 refining methods × 3 seeds (Table 1) — the dominating cost,
* `500` fidelity trajectories × 3 seeds × 4 values of `K`,

which is why the paper used 8 GPUs; `rice/envs/registry.py` exposes every budget
as a field of `EnvSpec` so they can be scaled down for a smaller machine.
