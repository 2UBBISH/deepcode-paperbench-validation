#!/usr/bin/env bash
# 校验 DeepCode/ 就是「上游 21ebc57f + patches/deepcode_local_changes.patch」，一个字节都没多改。
#   ① 15 个打过补丁的文件 sha256 与 patches/deepcode_patched.sha256 一致
#   ② 补丁可以反向干跑（说明 DeepCode/ − patch = 上游原样）
# 用法: bash patches/verify_deepcode.sh   （setup.sh 会自动调用；退出码非 0 = 有人改过 DeepCode/）
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UP="$(grep '^DeepCode upstream' "$ROOT/patches/UPSTREAM_BASE.txt" | awk '{print $NF}')"
cd "$ROOT/DeepCode"
if ! shasum -a 256 -c "$ROOT/patches/deepcode_patched.sha256" >/dev/null 2>&1; then
  echo "  ❌ DeepCode/ 里打过补丁的文件与 patches/deepcode_patched.sha256 不一致："
  shasum -a 256 -c "$ROOT/patches/deepcode_patched.sha256" 2>/dev/null | grep -v ': OK$' || true
  exit 1
fi
echo "  ✅ $(wc -l < "$ROOT/patches/deepcode_patched.sha256" | tr -d ' ') 个打过补丁的文件 sha256 一致"
if ! patch -p1 -R --dry-run --no-backup-if-mismatch < "$ROOT/patches/deepcode_local_changes.patch" >/dev/null 2>&1; then
  echo "  ❌ 补丁无法反向干跑：DeepCode/ 不等于上游 ${UP:0:8} + patch"; exit 1
fi
echo "  ✅ 补丁可反向干跑：DeepCode/ = HKUDS/DeepCode@${UP:0:8} + patches/deepcode_local_changes.patch"
