#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Download the NLD-AA dataset (Human Monk expert games) used by the NetHack
# fine-tuning experiments of
#   "Fine-tuning Reinforcement Learning Models is Secretly a
#    Forgetting Mitigation Problem" (Wolczyk et al., 2024)
#
# The paper (Appendix B.1 / addendum "NLD-AA") uses ~8000 Human Monk games and
# 10000 mini-batches of size 128 to build the BC state buffer and to estimate
# the diagonal Fisher Information Matrix.  The data is distributed as 16 shards
# hosted at https://dl.fbaipublicfiles.com/nld/nld-aa/ and is consumed through
# the `nle.dataset` interface under the database name `nld-aa-v0`.
#
# Usage:
#   scripts/download_nld_aa.sh                     # download + unpack
#   scripts/download_nld_aa.sh --register          # ... and register in nle
#   scripts/download_nld_aa.sh --data-dir /data    # custom destination
#   scripts/download_nld_aa.sh --num-shards 16     # (default)
#   scripts/download_nld_aa.sh --force             # re-download everything
#
# Environment variables honoured (mirroring src/nethack/dataset.py):
#   NLD_AA_DATA_DIR   destination directory (default: <repo>/data/nld-aa)
#   NLD_AA_URL_BASE   mirror base URL
#   NLD_AA_NAME       dataset name registered with nle.dataset (nld-aa-v0)
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# --- defaults --------------------------------------------------------------
URL_BASE="${NLD_AA_URL_BASE:-https://dl.fbaipublicfiles.com/nld/nld-aa/}"
DATA_DIR="${NLD_AA_DATA_DIR:-${REPO_ROOT}/data/nld-aa}"
DATASET_NAME="${NLD_AA_NAME:-nld-aa-v0}"
NUM_SHARDS="${NLD_AA_NUM_SHARDS:-16}"
FORCE=0
REGISTER=0
UNPACK=1
KEEP_ARCHIVES=0
JOBS=1
PYTHON="${PYTHON:-python}"

# Candidate shard naming conventions seen in the NLD-AA release distributions.
SUFFIX_NOEXT=("aa" "ab" "ac" "ad" "ae" "af" "ag" "ah" \
              "ai" "aj" "ak" "al" "am" "an" "ao" "ap")
PATTERNS=("nld-aa.tar.%s" "nld-aa.%s.tar" "nld-aa.%s" \
          "nld_aa.tar.%s" "nld_aa.%s.tar" "nld-aa-shard%s.tar" \
          "nld-aa-v0.tar.%s")

usage() {
  sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  cat <<'EOF'

Options:
  --data-dir DIR      Destination directory (default: $NLD_AA_DATA_DIR or <repo>/data/nld-aa)
  --url-base URL      Mirror base URL (default: https://dl.fbaipublicfiles.com/nld/nld-aa/)
  --num-shards N      Number of shards to fetch (default: 16)
  --name NAME         Dataset name registered with nle.dataset (default: nld-aa-v0)
  --register          Register/build the sqlite database via src/nethack/dataset.py
  --no-unpack         Keep archives, do not extract
  --keep-archives     Do not delete archives after extraction
  --jobs N            Parallel downloads with curl/wget (default: 1)
  --force             Re-download shards that already exist
  -h, --help          Show this help
EOF
}

# --- argument parsing ------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir)      DATA_DIR="$2"; shift 2 ;;
    --url-base)      URL_BASE="$2"; shift 2 ;;
    --num-shards)    NUM_SHARDS="$2"; shift 2 ;;
    --name)          DATASET_NAME="$2"; shift 2 ;;
    --register)      REGISTER=1; shift ;;
    --no-unpack)     UNPACK=0; shift ;;
    --keep-archives) KEEP_ARCHIVES=1; shift ;;
    --jobs)          JOBS="$2"; shift 2 ;;
    --force)         FORCE=1; shift ;;
    -h|--help)       usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

info()  { printf '[download_nld_aa] %s\n' "$*" >&2; }
warn()  { printf '[download_nld_aa][warn] %s\n' "$*" >&2; }
die()   { printf '[download_nld_aa][error] %s\n' "$*" >&2; exit 1; }

command -v curl >/dev/null 2>&1 || command -v wget >/dev/null 2>&1 \
  || die "need either 'curl' or 'wget' to download the dataset"

mkdir -p "${DATA_DIR}"
info "destination : ${DATA_DIR}"
info "url base    : ${URL_BASE}"
info "shards      : ${NUM_SHARDS}"
info "dataset name: ${DATASET_NAME}"

# --- helpers ---------------------------------------------------------------
# Human-readable byte size (used only for logging).
human_bytes() {
  "${PYTHON}" - "$1" <<'PY' 2>/dev/null || echo "$1 bytes"
import sys
n = float(sys.argv[1])
for unit in ("B", "KB", "MB", "GB", "TB"):
    if n < 1024.0 or unit == "TB":
        print(f"{n:.1f}{unit}")
        break
    n /= 1024.0
PY
}

# Probe a URL with a HEAD request; returns 0 when the resource exists.
url_exists() {
  local url="$1"
  if command -v curl >/dev/null 2>&1; then
    curl -fsIL --max-time 30 -o /dev/null "${url}"
  else
    wget --spider -q --timeout=30 "${url}"
  fi
}

# Download a single file (follows redirects, resumes partial transfers).
fetch() {
  local url="$1" dest="$2"
  if [[ -s "${dest}" && "${FORCE}" -ne 1 ]]; then
    info "skip (exists): $(basename -- "${dest}") ($(human_bytes "$(stat -c%s "${dest}" 2>/dev/null || echo 0)"))"
    return 0
  fi
  info "downloading  : ${url}"
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 --retry-delay 5 -C - -o "${dest}.part" "${url}" \
      && mv -f "${dest}.part" "${dest}"
  else
    wget -c -q --show-progress -O "${dest}.part" "${url}" \
      && mv -f "${dest}.part" "${dest}"
  fi
}

# Find and download the first existing candidate name for a given shard index.
download_shard() {
  local index="$1"
  local suffix="${SUFFIX_NOEXT[${index}]}"
  local pattern candidate url dest downloaded
  downloaded=0
  for pattern in "${PATTERNS[@]}"; do
    if [[ "${pattern}" == *"%s"* ]]; then
      if [[ "$(grep -o '%s' <<<"${pattern}" | wc -l)" -ge 2 ]]; then
        candidate="$(printf "${pattern}" "${suffix}" "${index}")"
      else
        candidate="$(printf "${pattern}" "${suffix}")"
      fi
    else
      candidate="${pattern}"
    fi
    url="${URL_BASE%/}/${candidate}"
    dest="${DATA_DIR}/${candidate}"
    if [[ -s "${dest}" && "${FORCE}" -ne 1 ]]; then
      fetch "${url}" "${dest}"   # logs "skip (exists)"
      DOWNLOADED_FILES+=("${dest}")
      downloaded=1
      break
    fi
    if url_exists "${url}"; then
      fetch "${url}" "${dest}"
      DOWNLOADED_FILES+=("${dest}")
      downloaded=1
      break
    fi
  done
  if [[ "${downloaded}" -ne 1 ]]; then
    warn "shard ${index} (${suffix}): no candidate name found under ${URL_BASE}"
    return 1
  fi
  return 0
}

# Extract an archive in place (supports .tar, .tar.gz, .tgz, .zip).
unpack() {
  local archive="$1"
  local dir="$2"
  case "${archive}" in
    *.tar.gz|*.tgz) tar -xzf "${archive}" -C "${dir}" ;;
    *.tar)          tar -xf  "${archive}" -C "${dir}" ;;
    *.zip)          ( cd "${dir}" && unzip -o -q "${archive}" ) ;;
    *)              warn "unrecognised archive extension: ${archive}" ;;
  esac
}

# --- main ------------------------------------------------------------------
DOWNLOADED_FILES=()
MISSING=0

if [[ "${JOBS}" -gt 1 ]]; then
  for ((i = 0; i < NUM_SHARDS; i++)); do
    ( download_shard "${i}" || true ) &
    while [[ "$(jobs -rp | wc -l)" -ge "${JOBS}" ]]; do wait -n || true; done
  done
  wait || true
else
  for ((i = 0; i < NUM_SHARDS; i++)); do
    download_shard "${i}" || MISSING=$((MISSING + 1))
  done
fi

if [[ "${MISSING}" -gt 0 ]]; then
  warn "${MISSING}/${NUM_SHARDS} shards could not be located."
  warn "Check the mirror, or pre-place the shards inside ${DATA_DIR}."
fi

if [[ "${UNPACK}" -eq 1 ]]; then
  shopt -s nullglob
  archives=("${DATA_DIR}"/*.tar "${DATA_DIR}"/*.tar.gz "${DATA_DIR}"/*.tgz "${DATA_DIR}"/*.zip)
  shopt -u nullglob
  if [[ "${#archives[@]}" -eq 0 ]]; then
    info "no archives to unpack (shards may already be unpacked)"
  fi
  for archive in "${archives[@]}"; do
    info "unpacking    : $(basename -- "${archive}")"
    unpack "${archive}" "${DATA_DIR}" || warn "failed to unpack ${archive}"
    if [[ "${KEEP_ARCHIVES}" -ne 1 ]]; then
      rm -f -- "${archive}"
    fi
  done
fi

# --- registration with nle.dataset ----------------------------------------
if [[ "${REGISTER}" -eq 1 ]]; then
  info "registering '${DATASET_NAME}' with nle.dataset (building sqlite db)"
  (
    cd -- "${REPO_ROOT}"
    NLD_AA_DATA_DIR="${DATA_DIR}" "${PYTHON}" -m src.nethack.dataset \
      --name "${DATASET_NAME}" \
      --data-dir "${DATA_DIR}" \
      --build \
      --download 0 || true
  ) || warn "registration via src/nethack/dataset.py returned an error"
  info "verify with: ${PYTHON} -m src.nethack.dataset --name ${DATASET_NAME} --describe"
fi

# --- summary ---------------------------------------------------------------
total_files=$(find "${DATA_DIR}" -maxdepth 2 -type f 2>/dev/null | wc -l)
info "done: ${total_files} file(s) in ${DATA_DIR}"
cat <<EOF

Next steps
----------
1) Verify the dataset is visible to NLE:
     python -m src.nethack.dataset --name ${DATASET_NAME} --describe
2) Estimate the diagonal Fisher matrix (10000 batches, batch size 128):
     python src/nethack/train_nethack.py --config configs/nethack.yaml --compute-fisher
3) Fine-tune with a retention method (actor only):
     python src/nethack/train_nethack.py --config configs/nethack.yaml --method bc
EOF
