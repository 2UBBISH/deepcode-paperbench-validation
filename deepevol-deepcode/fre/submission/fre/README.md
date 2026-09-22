# Functional Reward Encodings (FRE)

Reference implementation of **"Zero-Shot Reinforcement Learning via Functional Reward
Encodings"**.

FRE is a permutation-invariant transformer variational auto-encoder over **reward
functions**. It encodes a small set of `(state, reward)` samples into a 128-d task latent
`z`, is trained with an information-bottleneck objective over a mixture of three random
unsupervised reward priors, and conditions an IQL policy so that **new tasks are solved
zero-shot** from ~32 reward-annotated samples (no policy training, no fine-tuning).

---

## 1. Method summary

| Component | What it does | Where |
|---|---|---|
| Reward embedding | Rescale scalar reward to `[0,1]`, floor into 32 bins, look up a learned 64-d embedding | `fre/models/reward_embedding.py` |
| FRE encoder | 4 pre-norm transformer blocks (4 heads, MLP 256, **no positional encodings**, no causal mask), mean-pool over tokens → Gaussian `p_θ(z∣Lᵉ)`, `z ∈ R¹²⁸` | `fre/models/fre_encoder.py` |
| Token layout | 128-d = linear projection of the raw state (64-d) ⊕ 64-d reward embedding | `fre/models/fre_encoder.py` |
| FRE decoder | Feed-forward MLP `[512,512,512]` on `[raw state, z]` → `q_θ(η(s^d)∣s^d, z)`; K′=8 decoder states, disjoint from the encoder states | `fre/models/fre_decoder.py` |
| Objective (Eq. 6) | maximize `E log q_θ` − β·KL → MSE reconstruction + `β·KL(p_θ(z∣Lᵉ) ‖ N(0,I))`, β = 0.01 | `fre/models/fre_model.py` |
| Reward priors | Uniform 1/3 mixture of (a) singleton goal-reaching, (b) random linear (0.9 sparse mask, XY dropped on AntMaze), (c) random MLP `(state_dim, 32, 1)` tanh, clipped to `[-1,1]` | `fre/priors/*` |
| Policy agent | IQL with `z` concatenated to the observation: `Q(s,a,z)`, `V(s,z)`, `π(a∣s,z)`, `[512,512,512]`; expectile 0.8, AWR temp 3.0, τ = 0.001, γ = 0.88, Adam 1e-4 | `fre/models/iql.py`, `fre/models/rl_networks.py` |
| Strided schedule | Phase 1: encoder+decoder only; Phase 2: **freeze the encoder**, train RL on frozen `z`. AntMaze 150k + 850k steps, ExORL/Kitchen 1M + 1M | `fre/run_fre.py` (Algorithm 1) |
| Evaluation | Encode K = 32 `(s, η(s))` samples of the test task (posterior mean) → `z` → rollout, no training; returns normalized to 0–100, 20 episodes × 5 seeds | `fre/evaluation/evaluate.py`, `fre/evaluation/metrics.py` |

Key paper-silent choices (exposed as config flags so paper-faithful defaults can be
swapped): posterior **mean** for the evaluation `z`, fixed β = 0.01 with no annealing,
unit-variance Gaussian decoder (MSE-equivalent), 4 pre-norm blocks, Polyak τ = 0.001,
geometric future-goal probability `p = 0.5`, per-reward-function min/max normalization
before binning, MLP prior weights scaled by `1/sqrt(average layer dim)`, AntMaze XY at
observation indices `(0, 1)` and XY velocity at `(15, 16)`.

---

## 2. Repository layout

```
fre/
  main.py                    # CLI: train / train-all / eval / reproduce / reproduce-tables / info
  run_fre.py                 # strided schedule (Algorithm 1), checkpoints, freezing assertions
  models/                    # reward_embedding, fre_encoder, fre_decoder, fre_model, iql, rl_networks
  priors/                    # goal_functions, linear_functions, mlp_functions, reward_prior (mixture)
  data/                      # dataset loaders + trajectory indexing, AntMaze/ExORL preprocessing
  envs/                      # antmaze_tasks, exorl_tasks, kitchen_tasks, reward_wrappers
  evaluation/                # evaluate.py (zero-shot rollouts), metrics.py (0–100 aggregation)
  baselines/                 # gc_iql, gc_bc, opal, fb_sf_runner
  configs/                   # antmaze.yaml, exorl.yaml, kitchen.yaml (Table 3 + per-domain settings)
  scripts/run_all.sh         # train + eval + tables for all domains/methods
```

---

## 3. Installation

```bash
python -m pip install -r fre/requirements.txt        # Python 3.9/3.10, CUDA GPU recommended
```

Two components need **source installs** (see the tail of `requirements.txt`):

1. **D4RL**, pinned to a **pre-June-2024 commit** (the newer releases changed the
   observation layout / dataset registration):
   ```bash
   pip install git+https://github.com/Farama-Foundation/d4rl@<pre-2024-06-commit>
   ```
   Needed for `antmaze-large-diverse-v2` and `kitchen-mixed-v0`.
2. **ExORL RND datasets** (walker/cheetah) — download the archives and pass the directory
   to the loader (`dataset_dir=...`).
3. **FB / SF baselines** — clone `facebookresearch/controllable_agent`; those methods are
   trained/evaluated with that repo (not re-implemented here), see §7.

Everything heavy (`d4rl`, `gym`, `dm_control`, `torch`) is imported lazily, so
`import fre` and `python -m fre.main info` work in a minimal environment.

---

## 4. Quick start

```bash
# 0) sanity check: resolved config + run dir + task sets (no torch/dataset needed)
python -m fre.main info --domain antmaze

# 1) train FRE on one domain (Phase 1 encoder 150k -> Phase 2 frozen encoder 850k)
python -m fre.main train --domain antmaze --config fre/configs/antmaze.yaml \
    --save runs/antmaze/checkpoint.pt --device cuda

# 2) zero-shot evaluation: encode 32 (s, eta(s)) samples -> z -> 20 episodes x 5 seeds
python -m fre.main eval --domain antmaze --config fre/configs/antmaze.yaml \
    --checkpoint runs/antmaze/checkpoint.pt --task-set all --results-dir results

# 3) aggregate every results JSON into Table 1 / Table 4
python -m fre.main reproduce-tables --results-dir results --table1 --table4
```

Or drive the whole sweep (train → eval → tables, all domains, all methods):

```bash
bash fre/scripts/run_all.sh                    # full reproduction
bash fre/scripts/run_all.sh --dry-run          # print the commands only
bash fre/scripts/run_all.sh --quick            # 2k steps/phase smoke test
bash fre/scripts/run_all.sh --eval-only        # skip training, evaluate + tables
bash fre/scripts/run_all.sh --domains antmaze --methods fre
```

`run_all.sh` forwards only the CLI flags the installed parser accepts (it probes
`--help`), so it stays compatible across revisions.

### Training script knobs

* `--domain` — `antmaze`, `exorl:walker`, `exorl:cheetah`, `exorl`, `kitchen`
* `--encoder-steps` / `--policy-steps` — override the per-domain strided schedule
* `--prior-preset` — `fre-all` (default) or an ablation subset (see §6)
* `--phase` — `encoder`, `policy`, or `both`
* `--eval-after` — run zero-shot evaluation immediately after training

---

## 5. Data and preprocessing

* **AntMaze** (`antmaze-large-diverse-v2`): X/Y positions (dims 0, 1) are discretized into
  **32 bins** for FRE/GC-IQL/GC-BC/OPAL; evaluation resets the ant to the maze center.
* **ExORL** (`walker`/`cheetah` RND): physics augmentation is applied to the **encoder
  stream only** — walker appends `horizontal_velocity()`, `torso_upright()`,
  `torso_height()` (+4 dims), cheetah appends `speed()` (+1 dim); the same features are
  used at evaluation. Observations are standardized per dimension by dataset std, and the
  appended dims are **excluded** from goal distance.
* **Kitchen** (`kitchen-mixed-v0`): raw 59-d observation, no discretization.

Trajectory indexing supports the HER goal distribution used by the goal prior: 0.2 current
state, 0.5 future state within the trajectory, 0.3 random dataset state (at least one
encoder sample is forced to be the goal itself).

---

## 6. Studies

**Scaling study (Table 4 / Figure 5)** — retrain with equal budget on reward-family
subsets and normalize so the best agent per task set scores 1.0:

```bash
for p in fre-all fre-goals fre-lin fre-mlp fre-lin-mlp fre-goal-mlp fre-goal-lin; do
  python -m fre.main train --domain antmaze --prior-preset "$p" \
      --save "runs/antmaze_$p/checkpoint.pt"
  python -m fre.main eval  --domain antmaze --checkpoint "runs/antmaze_$p/checkpoint.pt" \
      --results-dir "results/$p"
done
```
Expected ordering: `fre-all` best (≈47.3 on AntMaze). `relative_normalize` in
`fre/evaluation/metrics.py` implements the "best = 1.0" convention.

**Domain-knowledge study (Figure 6)** — `--prior-preset fre-hint` adds the task-superset
priors (unit `(x,y)` directional rewards on AntMaze; target-velocity rewards on ExORL)
with **no** architecture or algorithm change; this improves directional/velocity tasks.

**Ablation presets** live in `fre/priors/reward_prior.py` (`PRESETS`) and are listed in
each `configs/*.yaml` under `scaling_presets:`.

*Out of scope:* the Figure 3 / §5.1 qualitative AntMaze generalization analysis.

---

## 7. Baselines

| Method | Implementation |
|---|---|
| **GC-IQL** | In-house: IQL with the goal concatenated to the observation; HER goals (0.2 current / 0.5 future / 0.3 random); reward 0 at goal else −1; ground-truth goal at eval — `fre/baselines/gc_iql.py` |
| **GC-BC** | In-house: 3×512 MLP with LayerNorm before activations, Gaussian head with `log_std` clamped at −5.0, MLE loss, **geometric-only** hindsight goals — `fre/baselines/gc_bc.py` |
| **OPAL** | Re-implemented: encoder reuses FRE's transformer blocks; privileged eval = 10 unit-Gaussian skills, whole-episode rollout each, best kept — `fre/baselines/opal.py` |
| **FB / SF** | External: `fre/baselines/fb_sf_runner.py` drives `facebookresearch/controllable_agent` (RND data, ICM features for SF, custom reward functions injected into the envs, **5120** eval reward samples) |

```bash
python -m fre.main eval --method gc_iql --domain antmaze --checkpoint runs/antmaze_gc_iql.pt
python -m fre.main eval --method opal   --domain antmaze --checkpoint runs/antmaze_opal.pt
# FB / SF: plan and inspect the exact external commands first (dry run by default)
python -m fre.baselines.fb_sf_runner --method fb --domain antmaze --dry-run
```

---

## 8. Evaluation protocol and expected results

Zero-shot: K = 32 `(s, η(s))` samples → posterior-mean `z` → rollout **without any
training**. Episodes: AntMaze ≤ 2000 steps, ExORL ≤ 1000, Kitchen ≤ 280. Returns are
normalized to **0–100**, averaged over **20 episodes** and **5 seeds**, and reported with
the std across seeds. FRE uses only 32 reward samples at evaluation, versus 5120 for
FB/SF.

Targets from Table 1 of the paper (normalized return, mean ± std over seeds):

| Task set | Target |
|---|---|
| `ant-goal-reaching` | 48.8 ± 6.0 |
| `ant-directional` | 55.2 ± 8.0 |
| `ant-random-simplex` | 21.3 ± 4.0 |
| `ant-path-loop` | 67.2 |
| `ant-path-edges` | 60.0 |
| `ant-path-center` | 64.4 |
| **antmaze-all** | **52.8 ± 18.2** |
| `exorl-walker-goals` | 94 |
| `exorl-cheetah-goals` | 58 |
| `exorl-walker-velocity` | 34 |
| `exorl-cheetah-velocity` | 20 |
| **exorl-all** | **51.5 ± 6.3** |
| **kitchen** | **66 ± 3** |
| **all** | **57 ± 9** |

AntMaze task sets (Appendix C.1): 5 fixed goal-reaching goals (bottom `(28,0)`, left
`(0,15)`, top `(35,24)`, center `(12,24)`, right `(33,16)`; goal distance 2.0), 4
directional velocity dot-product tasks (`(-1,0)`, `(0,1)`, `(0,-1)`, `(1,0)`), 5 seeded
opensimplex tasks (baseline −1 plus height and preferred-velocity bonuses), and 3 path
corridor tasks (center / loop / edges). ExORL: cheetah-run `v ≥ 10`, run-backwards,
walk `v ≥ 1`, walk-backwards; walker velocity thresholds `0.1 / 1 / 4 / 8` with linear
decay and 0 reward for the opposite direction; 5 fixed goal states with Euclidean
threshold 0.1. Kitchen: 7 standard sparse subtasks.

Four result keys per task set are aggregated by `python -m fre.main reproduce-tables` into
`table1.json` / `table1.txt` and `table4.json` / `table4.txt`.

---

## 9. Correctness checks

Useful unit-level invariants (all hold for the shipped defaults):

* **Permutation invariance** — shuffling the K `(s, η(s))` pairs leaves `z` (near-)
  unchanged; the encoder has no positional encodings and no causal mask.
* **Shapes** — `z` is 128-d, the token is 128-d (64 state ⊕ 64 reward), reward bins are
  integers in `[0, 31]` (`r == 1.0` maps to bin 31, not 32).
* **Phase 1** — decoder MSE on held-out decoder states decreases and the KL stays finite
  with β = 0.01.
* **Freezing** — the encoder parameter hash (`parameter_hash` /
  `assert_encoder_frozen` in `fre/run_fre.py`) is identical at the start and end of
  Phase 2, keeping `η → z` stationary for TD learning.

---

## 10. Citation

```bibtex
@inproceedings{fre2023,
  title  = {Zero-Shot Reinforcement Learning via Functional Reward Encodings},
  year   = {2023}
}
```
