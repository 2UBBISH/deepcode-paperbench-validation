#!/usr/bin/env bash
#
# prepare_data.sh -- download the SMM target-task datasets.
#
# SMM (ICML 2024) evaluates visual reprogramming on 11 target datasets:
#   CIFAR10, CIFAR100, SVHN, GTSRB, Flowers102, DTD, UCF101,
#   Food101, SUN397, EuroSAT, OxfordPets
# plus StanfordCars, used only for the Appendix D.4 failure-case table.
#
# All of them ship through ``torchvision.datasets``, so this script simply
# instantiates each one with ``download=True`` (see smm_vr/data/datasets.py).
#
# Usage:
#   bash smm_vr/scripts/prepare_data.sh [--data-root DIR] [--datasets "..."] \
#        [--skip-stanfordcars] [--num-workers N]
#
# Environment overrides (same names as the training scripts):
#   DATA_ROOT    destination directory (default: ./data, or $SMM_DATA_ROOT)
#   DATASETS     space separated dataset keys (default: all 11 main tasks)
#
set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
: "${SMM_DATA_ROOT:=}"
DATA_ROOT="${DATA_ROOT:-${SMM_DATA_ROOT:-${REPO_DIR}/data}}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PYTHON="${PYTHON:-python}"
DATASETS="${DATASETS:-cifar10 cifar100 svhn gtsrb flowers102 dtd ucf101 food101 sun397 eurosat oxfordpets}"
WITH_STANFORDCARS="${WITH_STANFORDCARS:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

usage() {
    cat <<'EOF'
prepare_data.sh -- download SMM target-task datasets via torchvision.

Options:
  --data-root DIR        destination directory (default: ./data)
  --datasets "A B ..."   dataset keys to fetch (default: all 11 main tasks)
  --skip-stanfordcars    do not download the StanfordCars failure-case set
  --num-workers N        torchvision/torch DataLoader workers (default: 4)
  -h, --help             show this message

Environment:
  DATA_ROOT / SMM_DATA_ROOT   destination directory
  DATASETS                    dataset keys to fetch
  WITH_STANFORDCARS=0|1       fetch StanfordCars (default: 1)
  EXTRA_ARGS                  extra flags forwarded to the Python helper
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-root)
            DATA_ROOT="$2"; shift 2 ;;
        --datasets)
            DATASETS="$2"; shift 2 ;;
        --skip-stanfordcars)
            WITH_STANFORDCARS=0; shift ;;
        --num-workers)
            NUM_WORKERS="$2"; shift 2 ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            echo "prepare_data.sh: unknown argument '$1'" >&2
            usage >&2
            exit 2 ;;
    esac
done

mkdir -p "${DATA_ROOT}"
LOG_DIR="${DATA_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/prepare_data_$(date +%Y%m%d_%H%M%S).log"

echo "==================================================================="
echo " SMM data preparation"
echo "-------------------------------------------------------------------"
echo " repository   : ${REPO_DIR}"
echo " data root    : ${DATA_ROOT}"
echo " datasets     : ${DATASETS}"
echo " stanfordcars : ${WITH_STANFORDCARS}"
echo " log file     : ${LOG_FILE}"
echo "==================================================================="

# ---------------------------------------------------------------------------
# Stage 1 -- parameter / environment sanity check
# ---------------------------------------------------------------------------
run_stage() {
    local stage_name="$1"; shift
    echo ""
    echo ">>> [${stage_name}]"
    "$@" 2>&1 | tee -a "${LOG_FILE}"
}

run_stage "environment check" "${PYTHON}" - <<'PY'
import sys

try:
    import torch
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit("torch is required: pip install -r smm_vr/requirements.txt") from exc

try:
    import torchvision
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit("torchvision is required") from exc

print(f"python      : {sys.version.split()[0]}")
print(f"torch       : {torch.__version__} (cuda={torch.cuda.is_available()})")
print(f"torchvision : {torchvision.__version__}")
PY

# ---------------------------------------------------------------------------
# Stage 2 -- download / verify every target dataset
# ---------------------------------------------------------------------------
export SMM_DATA_ROOT="${DATA_ROOT}"
export SMM_DATASETS="${DATASETS}"
export SMM_NUM_WORKERS="${NUM_WORKERS}"
export SMM_WITH_STANFORDCARS="${WITH_STANFORDCARS}"

run_stage "download datasets" "${PYTHON}" - <<'PY'
"""Materialise the SMM target datasets with torchvision.

Falls back to direct torchvision constructors when the project package has not
been imported yet (e.g. a bare checkout), so this script is usable standalone.
"""
import os
import sys

data_root = os.environ["SMM_DATA_ROOT"]
datasets = os.environ["SMM_DATASETS"].split()
num_workers = int(os.environ.get("SMM_NUM_WORKERS", "4"))
with_cars = os.environ.get("SMM_WITH_STANFORDCARS", "1") == "1"

repo_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

ok = True
try:
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    from smm_vr.data.datasets import build_datasets, get_dataset_spec  # type: ignore

    def fetch(name):
        train, test, spec = build_datasets(name, root=data_root, download=True)
        return spec, len(train), len(test)
except ImportError:  # pragma: no cover - standalone fallback
    import torchvision.datasets as tvd
    from torchvision import transforms

    _TF = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                              transforms.Normalize([0.485, 0.456, 0.406],
                                                   [0.229, 0.224, 0.225])])
    _NATIVE = {
        "cifar10": (tvd.CIFAR10, lambda root: dict(root=root, train=True, download=True, transform=_TF)),
        "cifar100": (tvd.CIFAR100, lambda root: dict(root=root, train=True, download=True, transform=_TF)),
        "svhn": (tvd.SVHN, lambda root: dict(root=root, split="train", download=True, transform=_TF)),
        "gtsrb": (tvd.GTSRB, lambda root: dict(root=root, split="train", download=True, transform=_TF)),
        "flowers102": (tvd.Flowers102, lambda root: dict(root=root, split="train", download=True, transform=_TF)),
        "dtd": (tvd.DTD, lambda root: dict(root=root, split="train", download=True, transform=_TF)),
        "food101": (tvd.Food101, lambda root: dict(root=root, split="train", download=True, transform=_TF)),
        "sun397": (tvd.SUN397, lambda root: dict(root=root, download=True, transform=_TF)),
        "eurosat": (tvd.EuroSAT, lambda root: dict(root=root, download=True, transform=_TF)),
        "oxfordpets": (tvd.OxfordIIITPet, lambda root: dict(root=root, split="trainval", download=True, transform=_TF)),
        "stanfordcars": (tvd.StanfordCars, lambda root: dict(root=root, split="train", download=True, transform=_TF)),
    }

    def fetch(name):
        cls, kwargs = _NATIVE[name]
        ds = cls(**kwargs(data_root))
        return None, len(ds), None

    if "ucf101" in datasets:
        # UCF101 zip often needs manual download; record the hint and continue.
        print("  ucf101: torchvision will attempt download; if it fails, place "
              "UCF101.rar in <data-root>/UCF101 and re-run.")

for name in datasets:
    try:
        spec, n_train, n_test = fetch(name)
        spec_name = getattr(spec, "name", name) if spec is not None else name
        print(f"  [ok]   {spec_name:<14} train={n_train:<8} test={n_test}")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"  [FAIL] {name:<14} {type(exc).__name__}: {exc}")

if with_cars and "stanfordcars" not in datasets:
    try:
        spec, n_train, n_test = fetch("stanfordcars")
        spec_name = getattr(spec, "name", "stanfordcars") if spec is not None else "stanfordcars"
        print(f"  [ok]   {spec_name:<14} train={n_train:<8} test={n_test}")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"  [FAIL] stanfordcars   {type(exc).__name__}: {exc}")

if not ok:
    raise SystemExit("one or more datasets failed to download")
PY

# ---------------------------------------------------------------------------
# Stage 3 -- report Table 6 metadata (sizes / classes) for reference
# ---------------------------------------------------------------------------
run_stage "dataset metadata (Table 6)" "${PYTHON}" - <<'PY'
import os
import sys

repo_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)

try:
    from smm_vr.data.dataset_stats import format_statistics_table, summarise_datasets

    print(format_statistics_table(summarise_datasets()))
except Exception as exc:  # noqa: BLE001
    print(f"(metadata report skipped: {exc})")
PY

echo ""
echo "==================================================================="
echo " Data preparation finished."
echo " Results and logs: ${DATA_ROOT}"
echo " Next: bash smm_vr/scripts/run_all_main.sh"
echo "==================================================================="
