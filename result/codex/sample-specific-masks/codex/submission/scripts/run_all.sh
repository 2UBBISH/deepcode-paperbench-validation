#!/usr/bin/env bash
# Reproduce the paper's tables. Every call writes a JSON result under runs/.
#
#   bash scripts/run_all.sh resnet18      # Table 1 (ResNet-18) + Table 3 ablations
#   bash scripts/run_all.sh resnet50      # Table 1 (ResNet-50)
#   bash scripts/run_all.sh vitb32        # Table 2
#   bash scripts/run_all.sh patches       # Figure 4 (patch size 2**l)
#   bash scripts/run_all.sh mappings      # Appendix D.1 (different f_out)
#
# Three seeds per cell are used, as in the paper ("Experiments are run with
# three seeds on a single A100 GPU and the averaged test accuracy is reported").
set -euo pipefail

BACKBONE="${1:-resnet18}"
DATA_ROOT="${DATA_ROOT:-data}"
OUT="${OUT:-runs}"
SEEDS="${SEEDS:-0 1 2}"

DATASETS=(cifar10 cifar100 svhn gtsrb flowers102 dtd ucf101 food101 sun397 eurosat oxfordpets)

case "$BACKBONE" in
  resnet18) CONFIG=configs/resnet18.yaml; METHODS=(pad narrow medium full smm) ;;
  resnet50) CONFIG=configs/resnet50.yaml; METHODS=(pad narrow medium full smm) ;;
  vitb32)   CONFIG=configs/vitb32.yaml;   METHODS=(pad narrow medium full smm) ;;
  patches)  CONFIG=configs/resnet18.yaml; METHODS=(smm) ;;
  mappings) CONFIG=configs/resnet18.yaml; METHODS=(smm) ;;
  *) echo "unknown target $BACKBONE"; exit 1 ;;
esac

run() {
  python -m smm.main --config "$CONFIG" --data-root "$DATA_ROOT" --output-dir "$OUT" "$@"
}

if [[ "$BACKBONE" == "patches" ]]; then
  for dataset in "${DATASETS[@]}"; do
    for l in 0 1 2 3 4; do
      for seed in $SEEDS; do
        run --dataset "$dataset" --method smm --num-pool-layers "$l" --seed "$seed" \
            --run-name "smm_patch$((2**l))_seed${seed}"
      done
    done
  done
  exit 0
fi

if [[ "$BACKBONE" == "mappings" ]]; then
  for dataset in cifar10 cifar100 svhn gtsrb flowers102 dtd ucf101 food101 sun397 eurosat oxfordpets; do
    for mapping in ilm flm rlm; do
      for seed in $SEEDS; do
        run --dataset "$dataset" --method smm --label-mapping "$mapping" --seed "$seed"
      done
    done
  done
  exit 0
fi

for dataset in "${DATASETS[@]}"; do
  for method in "${METHODS[@]}"; do
    for seed in $SEEDS; do
      run --dataset "$dataset" --method "$method" --seed "$seed"
    done
  done
done

# Table 3 ablations (ResNet-18, Ilm).
if [[ "$BACKBONE" == "resnet18" ]]; then
  for dataset in "${DATASETS[@]}"; do
    for method in only_delta only_mask single_channel; do
      for seed in $SEEDS; do
        run --dataset "$dataset" --method "$method" --seed "$seed"
      done
    done
  done
fi

echo "done. Aggregate with: python -m smm.aggregate --runs $OUT"
