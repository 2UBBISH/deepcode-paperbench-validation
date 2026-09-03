#!/usr/bin/env bash
# 修复验证两轮串行包装(fre,四修复开关全开)。独立文件,避免 pkill 自匹配。
export PAPER=fre DEEPCODE_EXPECT_MODEL=DeepSeek-V4-Pro
export DEEPCODE_PLAN_COVERAGE_CHECK=1 DEEPCODE_ALLOW_PLAN_EXTENSION=1 DEEPCODE_POSTWRITE_COMPILE=1
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; export DEEPEVOL_ROOT="$R"
LED="$R/deepcode_test/fre/logs/fre_ledger_fix.txt"
for T in trial_fx1 trial_fx2; do
  SUB=$HOME/pb_submissions/fre/$T
  if [ -d "$SUB" ] && [ "$(find "$SUB" -type f | wc -l)" -ge 5 ]; then echo "[$(date +%H:%M:%S)] ⏭ $T 已有提交,跳过" >> "$LED"; continue; fi
  TRIAL=$T bash $R/deepcode_test/scripts/run_trial.sh > $R/deepcode_test/fre/logs/${T}_console.log 2>&1
  echo "[$(date +%H:%M:%S)] $T 退出码=$?" >> "$LED"; sleep 30
done
echo "[$(date +%H:%M:%S)] ========== 修复验证两轮结束 ==========" >> "$LED"
