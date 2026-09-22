# Results produced by the development runs

The development environment had no GPU and long experiments were not allowed, so
the artifacts here come from

* `smoke/` -- the output of the `--quick` variants of the four scripts
  (`run_gaussian_targets.py --quick`, `run_shash_targets.py --quick`,
  `run_posteriordb.py --quick`, `verify_theorem31.py --dim 8 --n-iters 60`),
  together with their logs.  These are end-to-end runs of the exact code paths
  used by the full experiments, with tiny budgets.
* `validation/` -- medium-sized runs performed with the same scripts and
  smaller-than-paper budgets, used to check that the trends of the paper's
  figures are reproduced:

  | File | Command | Result |
  | --- | --- | --- |
  | `gaussian_targets_summary_D16.json`, `gaussian_D16_*.png` | `run_gaussian_targets.py --dims 16 --n-runs 3 --max-grad-evals 4000` | BaM and GSM reach `KL ~ 1e-14` within 4000 gradient evaluations; ADVI/Fisher/Score are still at `KL = 4-65` |
  | `posteriordb_arK_*.png`, `posteriordb_gp_pois_regr_*.png`, `posteriordb_eight_schools_centered_*.png` | `run_posteriordb.py --batches 8 32 --n-runs 2 --max-grad-evals 5000` (`pdb2.log`, `pdb3.log`) | arK: BaM 0.046 vs ADVI 95-214 relative mean error; gp_pois_regr: BaM 0.44 vs ADVI 8.3; eight-schools: BaM 0.40/0.51 vs ADVI 3.0, GSM 0.15 (faster at `B=8`) and BaM with the larger relative SD error (1.03 vs 0.71) -- exactly the trends reported in Figures 5.3 and E.6 |
  | `shash_*.png`, `shash_medium_run.log` | `run_shash_targets.py --skews 0.2 1.8 --tails 0.1 --bam-batches 5 20 --n-runs 2 --max-grad-evals 2000` | skew 1.8: reverse KL `BaM 7.6 / ADVI 7.1 / Score 57 / GSM 8633` (GSM and Score diverge, BaM and ADVI similar) while the forward KL is larger for BaM -- the paper's Figure 5.2/E.4 claims; tail 0.1: all methods end at similar reverse KL values |
  | `vae_posterior_synthetic.log` | `train_vae_cifar10.py --synthetic --subset 256 --epochs 3 --c-hid 8 --latent-dim 32` followed by `run_vae_posterior.py --quick --batch-sizes 10 40` | the Section 5.3 pipeline (VAE pre-training, posterior inference, AVI baseline, reconstruction MSE, 3000-gradient-evaluation budget analysis) runs end to end and BaM reaches the lowest reconstruction MSE, improving with batch size (`0.3331` at `B = 10` vs `0.3314` at `B = 40`) -- the paper's "BaM becomes competitive as the batch size is increased" |
  | `vae_train_cifar_small.log`, `vae_posterior_cifar_small.log`, `vae_posterior_*_cifar_small.{png,json}` | `train_vae_cifar10.py --subset 2000 --epochs 3 --c-hid 8 --latent-dim 32` then `run_vae_posterior.py --batch-sizes 10 40 100 --n-iters 300` on a real CIFAR-10 test image | the same pipeline on **real CIFAR-10 data**: the negative ELBO decreases during training (4025 -> 3970) and, on the test image, BaM reaches the lowest reconstruction MSE (`0.15439 / 0.15417 / 0.15420` for `B = 10/40/100`) versus ADVI (`0.15473-0.15496`), GSM (`0.15503`) and the amortized encoder AVI (`0.15524`), i.e. the paper's Section 5.3 ordering.  (The full-scale experiment uses `c_hid = 64`, `latent_dim = 256`, 100 epochs and thousands of iterations, which needs a GPU.) |
  | `posteriordb_schedule_paper.log`, `posteriordb_schedule_sqrt.log`, `posteriordb_unconstrained_paper.log`, `posteriordb_unconstrained_sqrt.log` (+ the corresponding `posteriordb_*.png`) | `run_posteriordb.py --n-runs 3 --max-grad-evals 8000` with the four combinations of `--parameterization {constrained,unconstrained}` and `--bam-schedule {paper,sqrt}` | the full comparison behind the README's "known sensitivity" table for Section 5.2.  In every configuration where BaM runs stably it beats ADVI by an order of magnitude, and `--parameterization unconstrained --bam-schedule sqrt` reproduces the paper's ordering for `arK` (BaM `0.10/0.07` vs ADVI `39/167`) and `eight_schools_centered` (BaM `0.77/0.49` vs ADVI `3.0`), while `--parameterization constrained --gp-variable f_tilde --bam-schedule sqrt` is the only stable configuration for `gp_pois_regr` (BaM `5.0` vs ADVI `86.2`). |

The full-scale runs described in the top-level README regenerate these figures
with the paper's settings (10 runs, `D = 4..256`, up to `1e5` gradient
evaluations, CIFAR-10 VAE training).
