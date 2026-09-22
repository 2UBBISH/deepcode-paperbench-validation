# Experiment matrix: paper item -> command -> file

Every run is one cell of a table and writes
`runs/<dataset>/<backbone>_<dataset>_<method>_<mapping>_seed<k>.json`
(test accuracy, per-epoch history, label mapping, parameter counts).

| Paper item | Command | Notes |
|---|---|---|
| Table 1, ResNet-18, 11 datasets, 5 methods, 3 seeds | `bash scripts/run_all.sh resnet18` | 200 epochs each, `configs/resnet18.yaml` |
| Table 1, ResNet-50 | `bash scripts/run_all.sh resnet50` | `configs/resnet50.yaml` |
| Table 2, ViT-B/32 | `bash scripts/run_all.sh vitb32` | 384x384 input, 6-layer generator, `configs/vitb32.yaml` |
| Table 3, ablations | included in `scripts/run_all.sh resnet18` (`only_delta`, `only_mask`, `single_channel`) | ResNet-18, Ilm |
| Figure 4, patch size `2**l` | `bash scripts/run_all.sh patches` | `l in {0,1,2,3,4}` |
| Appendix D.1, SMM with Rlm/Flm | `bash scripts/run_all.sh mappings` | reuses the same training loop |
| Appendix D.2, learning curves | any run: `history` in the JSON (`eval_every`) | train loss/accuracy every epoch, test metrics every `eval_every` |
| Section 4 / Appendix B theory | `python scripts/verify_theory.py` | exact inclusion checks + empirical approximation error |
| Table 4 parameter budget | `python scripts/param_stats.py [--search]` | mask generator vs pattern vs backbone |
| Table 6 dataset sizes | `python scripts/make_splits.py` | realised split sizes vs the paper |
| Aggregate anything into tables | `python -m smm.aggregate --runs runs --table {1,2,3}` | mean +/- std over seeds |
| Fast trend sanity check | `bash scripts/mini_table1.sh` | small subset, reduced resolution, all 5 methods |
| Redraw learning curves | `python scripts/plot_curves.py --runs runs --dataset cifar10` | uses the `history` field of every run |
| Table 7 (ViT lr/decay search) | `bash scripts/lr_search_vit.sh` | `alpha in {0.1,0.01,0.001,1e-4} x gamma in {1,0.1}` on CIFAR-10 |
| Table 8 (UCF101, dataset-specific lr) | `python -m smm.main --config configs/vitb32.yaml --dataset ucf101 --method smm --lr-delta 0.01 --gamma-delta 0.1 --lr-mask 0.01 --gamma-mask 0.1` | unified setting (`0.001`, `1`) is the default |

## Per-dataset settings (Table 6 / Table 9)

| dataset | classes | resolution | train | test | batch | lr (ResNet) |
|---|---|---|---|---|---|---|
| cifar10 | 10 | 32 | 50,000 | 10,000 | 256 | 0.01 |
| cifar100 | 100 | 32 | 50,000 | 10,000 | 256 | 0.01 |
| svhn | 10 | 32 | 73,257 | 26,032 | 256 | 0.01 |
| gtsrb | 43 | 32 | 39,209 | 12,630 | 256 | 0.01 |
| flowers102 | 102 | 128 | 4,093 | 2,463 | 256 | 0.01 |
| dtd | 47 | 128 | 2,820 | 1,692 | 64 | 0.01 |
| ucf101 | 101 | 128 | 7,639 | 3,783 | 256 | 0.01 |
| food101 | 101 | 128 | 50,500 | 30,300 | 256 | 0.01 |
| sun397 | 397 | 128 | 15,888 | 19,850 | 256 | 0.01 |
| eurosat | 10 | 128 | 13,500 | 8,100 | 256 | 0.01 |
| oxfordpets | 37 | 128 | 2,944 | 3,669 | 64 | 0.01 |

ViT-B/32 uses the same batch sizes with `lr = 0.001` and no decay.

## Reading the results

The paper's trends that the code is expected to reproduce:

* SMM > every shared-mask baseline for both ResNets and for ViT-B/32, with the
  largest margins on SVHN, Flowers102, Food101, SUN397 and EuroSAT (i.e. target
  domains far from ImageNet).
* DTD with ResNet-18 is the known exception where `Pad` can win, because
  resizing-based methods (including SMM) disturb texture features.
* EuroSAT with ViT-B/32 is the second exception: the task is easy, resizing
  based methods over-fit and `Pad` is competitive.
* Ablation ordering (Table 3): `Only delta` < `Only f_mask` (on small/medium
  data) < `Single-channel f_mask` < SMM.
* Patch size (Figure 4): accuracy rises from `l = 0`, peaks around `l = 3`
  (patch size 8) and degrades for larger patches.
