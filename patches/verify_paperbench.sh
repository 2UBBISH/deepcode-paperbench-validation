#!/usr/bin/env bash
# Prove that the vendored frontier-evals/ (PaperBench) equals upstream openai/frontier-evals at the pinned commit
# (patches/UPSTREAM_BASE.txt), sparse paths project/paperbench + project/common, plus exactly:
#   · patches/paperbench_local_changes.patch (5 files)
#   · the files added from paperbench_changes/ (single-paper splits, the judge-bias script)
#   · LFS assets hydrated: a data file that upstream stores as an LFS pointer must be byte-identical to that pointer,
#     or hash to the sha256 the pointer names
#   · .gitattributes renamed to .gitattributes.upstream (so the data is stored as plain bytes in this repository)
# Needs network (fetches upstream into a temp dir); nothing in the repository is modified.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FE_COMMIT="$(grep '^frontier-evals upstream' "$ROOT/patches/UPSTREAM_BASE.txt" | awk '{print $NF}')"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
echo "upstream openai/frontier-evals @ $FE_COMMIT → $TMP"
git init -q "$TMP/up"
git -C "$TMP/up" remote add origin https://github.com/openai/frontier-evals.git
git -C "$TMP/up" sparse-checkout init --cone
git -C "$TMP/up" sparse-checkout set project/paperbench project/common
git -C "$TMP/up" fetch -q --depth 1 origin "$FE_COMMIT"
GIT_LFS_SKIP_SMUDGE=1 git -C "$TMP/up" checkout -q FETCH_HEAD
git -C "$TMP/up" apply "$ROOT/patches/paperbench_local_changes.patch"
cp "$ROOT/paperbench_changes/experiments/splits/"*.txt "$TMP/up/project/paperbench/experiments/splits/"
cp "$ROOT/paperbench_changes/analyze_judge_eval_bias.py" "$TMP/up/project/paperbench/"
mv "$TMP/up/project/paperbench/.gitattributes" "$TMP/up/project/paperbench/.gitattributes.upstream"

fail=0
echo "— source trees (everything but data/):"
if diff -rq --exclude=.git --exclude=.venv --exclude=runs --exclude=__pycache__ --exclude='*.egg-info' --exclude=records \
     --exclude=data --exclude='.env*' --exclude=run_eval.log "$TMP/up" "$ROOT/frontier-evals"; then
  echo "  ✅ identical to upstream + patch + added files"
else fail=1; fi

echo "— data/ (LFS pointers may be hydrated):"
n_same=0; n_hyd=0
while IFS= read -r -d '' up; do
  rel="${up#$TMP/up/}"; ours="$ROOT/frontier-evals/$rel"
  [ -f "$ours" ] || { echo "  ❌ missing: $rel"; fail=1; continue; }
  if cmp -s "$up" "$ours"; then n_same=$((n_same+1)); continue; fi
  if head -c 40 "$up" | grep -q '^version https://git-lfs'; then
    oid="$(grep -o 'oid sha256:[0-9a-f]*' "$up" | cut -d: -f2)"
    if [ "$(shasum -a 256 "$ours" | cut -d' ' -f1)" = "$oid" ]; then n_hyd=$((n_hyd+1)); continue; fi
    echo "  ❌ $rel differs from upstream and does not hash to its LFS oid"; fail=1
  else
    echo "  ❌ $rel differs from upstream (not an LFS pointer there)"; fail=1
  fi
done < <(find "$TMP/up/project/paperbench/data" -type f -print0)
echo "  $n_same files identical, $n_hyd LFS assets hydrated and hash-verified"
[ "$fail" = 0 ] && echo "VERIFY_OK" || { echo "VERIFY_FAILED"; exit 1; }
