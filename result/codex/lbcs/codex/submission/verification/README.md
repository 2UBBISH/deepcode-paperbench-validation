# Verification log

Outputs of the reduced-budget runs that were executed while building this
reproduction (all on CPU, tiny `T` / 1-10 epochs / small `k`).  They are
committed because the *full* configurations of the paper need hours of compute
and are meant to be run in the grading environment (`bash run_all.sh`).

| file | produced by |
|---|---|
| `theorem2_convergence.{json,csv}` | `python -m experiments.exp_theorem2_convergence --restarts 5 --steps 800` |
| `quick_validation.{json,csv}` | `python -m experiments.exp_quick_validation --ks 200 --epsilons 0.2 0.4 --T 60 --epochs 5` (sparse local search + thresholds including the candidate) |
| `quick_validation_greedy_thresholds.json` | the same experiment with the earlier settings, kept to document the difference (see the README) |
| `table1_mnist_s.{json,md}` | `python -m experiments.exp_table1_mnist_s --dry-run --ks 50 --epsilons 0.2` |
| `fig1_summary.json`, `fig1_eq{3,4}_trace.csv`, `fig1_trivial_solutions.png` | `python -m experiments.exp_fig1_trivial` (reduced `T`) |
| `fig2_noise30.csv`, `fig2_robustness.png` | `python -m experiments.exp_fig2_robustness --dry-run --methods Uniform LBCS --ks 50` |
| `table2_table3_fmnist.csv` | `python -m experiments.exp_table2_table3 --dry-run --datasets fmnist --methods Uniform Moderate LBCS --skip-table3` |
| `table5_moderate_init.csv` | `python -m experiments.exp_table5_moderate_init --dry-run --ks 50` |
| `table6_cross_arch.csv` | `python -m experiments.exp_table6_cross_arch --dry-run --dataset fmnist --methods Uniform LBCS --targets wideresnet --ks 50` |

What the runs show:

* `quick_validation`: `f_1` drops from `1.93` (random initial mask) to `0.17`;
  the coreset size stays at the predefined level and the larger compromise
  gives the smaller coreset (`206` for `eps=0.2`, `202` for `eps=0.4`).
* `theorem2_convergence`: `f_2`-optimality was reached in **100 %** of the
  restarts while respecting `f_1 <= f_1*(1+eps)`; the "no-harm" form of
  Condition 1 holds on ~81 % of the actual mask updates.
* `fig1_summary` (reduced run: `k = 30`, `n = 200`, `T = 120`, outer Adam
  lr `5e-2`): with equation (3) the coreset stays large
  (`f_2 = 67`, `f_1 = 2.02`), while with equation (4) and `lambda = 0.5` it
  collapses (`f_2 = 18`) at a higher loss (`f_1 = 2.25`) - the phenomenon of
  Section 2.1.  With the paper's outer learning rate of 2.5 the same
  comparison gives `f_2 = 101 / 32`; both are stored in
  `fig1_eq{3,4}_trace.csv`-style runs of the script.
