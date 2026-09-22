#!/usr/bin/env bash
#
# Download the NetHack Human Monk pre-trained checkpoint (π*).
#
# The paper ("Fine-tuning Reinforcement Learning Models is Secretly a Forgetting
# Mitigation Problem", Wołczyk et al., 2024) starts every NetHack fine-tuning run
# from the public 30M-step LSTM checkpoint released by Tuyls et al. (2023), which
# scores ~5K on the Human Monk role.  This script fetches that artifact from the
# Google Drive id listed in the reproduction plan
#
#     https://drive.google.com/uc?id=1tWxA92qkat7Uee8SKMNsj-BV1K9ENExl
#
# and stores it under <repo>/data/checkpoints by default, where
# `src/nethack/model.py :: load_lstm_checkpoint` / `build_model(checkpoint=...)`
# and `configs/nethack.yaml -> pretrained.checkpoint` expect to find it.
#
# Usage:
#   scripts/download_pretrained_ckpt.sh [options]
#
# Options:
#   --dest FILE        Destination file path (default: <repo>/data/checkpoints/nethack_challenge_30M.pt)
#   --dest-dir DIR     Directory for the checkpoint (default: <repo>/data/checkpoints)
#   --file-id ID       Google Drive file id (default: the paper's released checkpoint)
#   --url URL          Direct download URL (overrides --file-id)
#   --force            Re-download even if the destination already exists
#   --no-verify        Skip the torch-based sanity check (shape/parameter report)
#   --min-bytes N      Minimum plausible size in bytes (default: 100000000 = 100 MB)
#   -h, --help         Show this help
#
# Environment variables honoured:
#   NETHACK_CKPT_URL         direct URL override
#   NETHACK_CKPT_FILE_ID     Google Drive file id override
#   NETHACK_CKPT_PATH        destination file path override
#   NETHACK_CKPT_DIR         destination directory override
#   PYTHON                   python interpreter used for the torch sanity check
#
# Exit status:
#   0  checkpoint present (downloaded or already cached)
#   1  download/verification failure
#   2  usage error
#
set -euo pipefail

# --------------------------------------------------------------------------------------
# Defaults / paths
# --------------------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DRIVE_FILE_ID="${NETHACK_CKPT_FILE_ID:-1tWxA92qkat7Uee8SKMNsj-BV1K9ENExl}"
DIRECT_URL="${NETHACK_CKPT_URL:-}"
DEST_DIR="${NETHACK_CKPT_DIR:-${REPO_ROOT}/data/checkpoints}"
DEST_FILE="${NETHACK_CKPT_PATH:-}"
DEFAULT_NAME="nethack_challenge_30M.pt"

FORCE=0
VERIFY=1
MIN_BYTES="${NETHACK_CKPT_MIN_BYTES:-100000000}"
PYTHON_BIN="${PYTHON:-python3}"

DRIVE_UC="https://drive.google.com/uc"
DRIVE_CONFIRM="https://drive.usercontent.google.com/download"

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
usage() {
  sed -n '2,45p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

info()  { printf '[download_pretrained_ckpt] %s\n' "$*" >&2; }
warn()  { printf '[download_pretrained_ckpt] WARNING: %s\n' "$*" >&2; }
die()   { printf '[download_pretrained_ckpt] ERROR: %s\n' "$*" >&2; exit 1; }

have() { command -v "$1" >/dev/null 2>&1; }

human_bytes() {
  local bytes="${1:-0}"
  if have "${PYTHON_BIN}"; then
    "${PYTHON_BIN}" - "$bytes" <<'PY' 2>/dev/null || echo "${bytes} B"
import sys
try:
    n = float(sys.argv[1])
except Exception:
    print(sys.argv[1])
else:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0 or unit == "TiB":
            print(f"{n:.1f} {unit}")
            break
        n /= 1024.0
PY
  else
    echo "${bytes} B"
  fi
}

file_size() {
  local path="$1"
  if [ -f "$path" ]; then
    if have stat; then
      stat -c %s "$path" 2>/dev/null || stat -f %z "$path" 2>/dev/null || echo 0
    else
      wc -c <"$path" | tr -d ' '
    fi
  else
    echo 0
  fi
}

# Google Drive serves an HTML "virus scan" interstitial for large files; detect and
# normalise both the plain `uc?export=download&id=...` and `usercontent` variants.
drive_url() {
  local file_id="$1"
  printf '%s?id=%s&export=download&confirm=t' "${DRIVE_UC}" "${file_id}"
}

drive_fallback_url() {
  local file_id="$1"
  printf '%s?id=%s&export=download&confirm=t' "${DRIVE_CONFIRM}" "${file_id}"
}

# Download one URL to a path, resuming when possible.  Returns non-zero on failure.
fetch() {
  local url="$1" dest="$2"
  local tmp="${dest}.part"

  if have curl; then
    curl -fL --retry 5 --retry-delay 5 --retry-connrefused \
         -C - --connect-timeout 30 -o "${tmp}" "${url}"
  elif have wget; then
    wget -c --tries=5 --timeout=30 --show-progress -O "${tmp}" "${url}" \
      || wget -c --tries=5 --timeout=30 -O "${tmp}" "${url}"
  else
    die "neither curl nor wget is available"
  fi

  mv -f "${tmp}" "${dest}"
}

# Reject Google-Drive HTML interstitials / partial error pages.
looks_like_html() {
  local path="$1"
  if have file; then
    file -b --mime-type "$path" 2>/dev/null | grep -qi 'text/html' && return 0
  fi
  head -c 512 "$path" 2>/dev/null | grep -qiE '<html|googlevideo|drive.usercontent.google.com/download' && return 0
  return 1
}

verify_checkpoint() {
  local path="$1"
  if [ "${VERIFY}" != "1" ]; then
    return 0
  fi
  if ! have "${PYTHON_BIN}"; then
    warn "python not found; skipping checkpoint sanity check"
    return 0
  fi
  "${PYTHON_BIN}" - "$path" <<'PY' || return 1
import sys

path = sys.argv[1]
try:
    import torch  # noqa: F401
except Exception:
    print(f"[download_pretrained_ckpt] torch unavailable; skipping load of {path}", file=sys.stderr)
    raise SystemExit(0)

# NetHack checkpoints may be raw state dicts or wrapped dicts produced by
# src/nethack/model.py :: save_checkpoint(path, model, step=...).
obj = torch.load(path, map_location="cpu", weights_only=False)
state = obj
for key in ("model", "state_dict", "policy", "actor", "weights"):
    if isinstance(obj, dict) and key in obj and isinstance(obj[key], dict):
        state = obj[key]
        break

if not isinstance(state, dict) or not state:
    print(f"[download_pretrained_ckpt] {path}: not a state dict ({type(obj).__name__})", file=sys.stderr)
    raise SystemExit(1)

tensors = [v for v in state.values() if hasattr(v, "shape")]
num_params = 0
for tensor in tensors:
    n = 1
    for dim in tensor.shape:
        n *= int(dim)
    num_params += n

print(
    f"[download_pretrained_ckpt] {path}: {len(state)} entries, "
    f"{num_params/1e6:.1f}M parameters",
    file=sys.stderr,
)
have_lstm = any("lstm" in str(k).lower() for k in state)
have_head = any(("policy" in str(k).lower() or "baseline" in str(k).lower() or "value" in str(k).lower()) for k in state)
if have_lstm:
    print("[download_pretrained_ckpt] LSTM backbone found -> pi_* checkpoint", file=sys.stderr)
if not (have_lstm or have_head):
    print("[download_pretrained_ckpt] WARNING: no LSTM/policy/baseline keys recognised", file=sys.stderr)
PY
}

# --------------------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------------------
while [ "$#" -gt 0 ]; do
  case "$1" in
    --dest)       DEST_FILE="${2:?--dest needs a path}"; shift 2 ;;
    --dest-dir)   DEST_DIR="${2:?--dest-dir needs a path}"; shift 2 ;;
    --file-id)    DRIVE_FILE_ID="${2:?--file-id needs an id}"; shift 2 ;;
    --url)        DIRECT_URL="${2:?--url needs a url}"; shift 2 ;;
    --min-bytes)  MIN_BYTES="${2:?--min-bytes needs a number}"; shift 2 ;;
    --force)      FORCE=1; shift ;;
    --no-verify)  VERIFY=0; shift ;;
    -h|--help)    usage; exit 0 ;;
    *)            die "unknown argument: $1 (try --help)" ;;
  esac
done

if [ -z "${DEST_FILE}" ]; then
  DEST_FILE="${DEST_DIR}/${DEFAULT_NAME}"
fi

# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
mkdir -p "$(dirname "${DEST_FILE}")"

existing="$(file_size "${DEST_FILE}")"
if [ "${FORCE}" != "1" ] && [ "${existing}" -ge "${MIN_BYTES}" ] 2>/dev/null; then
  info "checkpoint already present: ${DEST_FILE} ($(human_bytes "${existing}"))"
  verify_checkpoint "${DEST_FILE}" || die "existing checkpoint at ${DEST_FILE} failed verification (use --force to re-download)"
  info "done"
  exit 0
fi

if [ "${existing}" -gt 0 ] && [ "${FORCE}" != "1" ]; then
  warn "existing file ${DEST_FILE} is smaller than ${MIN_BYTES} bytes ($(human_bytes "${existing}")); re-downloading"
fi

info "destination : ${DEST_FILE}"
info "source       : ${DIRECT_URL:-Google Drive id ${DRIVE_FILE_ID}}"

downloaded=0
if [ -n "${DIRECT_URL}" ]; then
  info "downloading from direct URL ..."
  if fetch "${DIRECT_URL}" "${DEST_FILE}"; then
    downloaded=1
  else
    warn "direct URL download failed"
  fi
else
  primary="$(drive_url "${DRIVE_FILE_ID}")"
  fallback="$(drive_fallback_url "${DRIVE_FILE_ID}")"
  info "downloading from Google Drive ..."
  if fetch "${primary}" "${DEST_FILE}"; then
    downloaded=1
  else
    warn "primary Drive endpoint failed; retrying alternate endpoint"
    if fetch "${fallback}" "${DEST_FILE}"; then
      downloaded=1
    fi
  fi
fi

if [ "${downloaded}" != "1" ]; then
  rm -f "${DEST_FILE}.part"
  die "could not download the pre-trained checkpoint. Check network access, or download
     https://drive.google.com/file/d/${DRIVE_FILE_ID}/view
  manually and re-run with:  $0 --dest ${DEST_FILE}"
fi

if looks_like_html "${DEST_FILE}"; then
  rm -f "${DEST_FILE}"
  die "the downloaded file is an HTML page rather than a checkpoint (Google Drive served
     a confirmation page). Download it manually from
     https://drive.google.com/file/d/${DRIVE_FILE_ID}/view
     and re-run with:  $0 --dest ${DEST_FILE}"
fi

size="$(file_size "${DEST_FILE}")"
info "downloaded   : ${DEST_FILE} ($(human_bytes "${size}"))"

if [ "${size}" -lt "${MIN_BYTES}" ] 2>/dev/null; then
  warn "downloaded file is smaller than the expected minimum ($(human_bytes "${MIN_BYTES}"))"
fi

verify_checkpoint "${DEST_FILE}" || die "downloaded checkpoint failed the torch sanity check"

info "checkpoint ready: ${DEST_FILE}"
info "use it with:  --checkpoint ${DEST_FILE}   (or configs/nethack.yaml -> pretrained.checkpoint)"
exit 0
