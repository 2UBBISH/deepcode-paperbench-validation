#!/usr/bin/env bash
# ============================================================
# 一键环境搭建(clone 之后只需跑这一个脚本)
#
#   git clone <this repo> && cd <this repo> && bash setup.sh
#
# 做的事:
#   1. 检查 docker / uv / git / git-lfs / python3
#   2. 稀疏克隆 openai/frontier-evals(固定 commit),只取 paperbench + common,
#      打上 patches/paperbench_local_changes.patch,复制我们新增的 split 等文件,
#      拉取 fre / rice 两篇论文的 LFS 资产
#   3. uv sync:DeepCode(修改版,已随仓库提供)与 paperbench
#   4. 生成 ~/.deepcode/{deepcode_config.json,credentials.json} 与 paperbench/.env(模板,需填 key)
#   5. 设置 git insteadOf 防作弊封锁(复现时禁止克隆论文官方实现)
#   6. 建 ~/pb_submissions/{fre,rice}/ 判分提交池
#
# 幂等:重复执行只补缺,不覆盖你已填好的 key。
# ============================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FE_REPO="https://github.com/openai/frontier-evals.git"
FE_COMMIT="$(grep '^frontier-evals upstream' "$ROOT/patches/UPSTREAM_BASE.txt" | awk '{print $NF}')"
PB="$ROOT/frontier-evals/project/paperbench"

echo "==== [1/6] 依赖检查 ===="
need() { command -v "$1" >/dev/null 2>&1 || { echo "  ❌ 缺 $1 —— $2"; exit 1; }; echo "  ✅ $1"; }
need git      "https://git-scm.com"
need git-lfs  "https://git-lfs.com(判分数据用 LFS 存放)"
need uv       "curl -LsSf https://astral.sh/uv/install.sh | sh"
need python3  "3.11+"
need docker   "判分要用 Docker 起沙箱;安装后确保 'docker info' 能通"
docker info >/dev/null 2>&1 && echo "  ✅ docker daemon 在跑" || echo "  ⚠️ docker daemon 未运行(复现不需要,判分前再启动)"

echo "==== [2/6] frontier-evals(PaperBench)@ ${FE_COMMIT:0:10} ===="
if [ ! -d "$ROOT/frontier-evals/.git" ]; then
  git clone --filter=blob:none --no-checkout "$FE_REPO" "$ROOT/frontier-evals"
  git -C "$ROOT/frontier-evals" sparse-checkout init --cone
  git -C "$ROOT/frontier-evals" sparse-checkout set project/paperbench project/common
  git -C "$ROOT/frontier-evals" checkout "$FE_COMMIT"
  echo "  ✅ 稀疏克隆完成"
else
  echo "  ⏭ 已存在,跳过克隆"
fi
if ! git -C "$ROOT/frontier-evals" apply --check "$ROOT/patches/paperbench_local_changes.patch" >/dev/null 2>&1; then
  git -C "$ROOT/frontier-evals" apply --reverse --check "$ROOT/patches/paperbench_local_changes.patch" >/dev/null 2>&1 \
    && echo "  ⏭ patch 已打过" || { echo "  ❌ patch 无法应用(上游 commit 不符?)"; exit 1; }
else
  git -C "$ROOT/frontier-evals" apply "$ROOT/patches/paperbench_local_changes.patch"; echo "  ✅ 已打 paperbench patch(3 文件)"
fi
cp "$ROOT/paperbench_changes/experiments/splits/"*.txt "$PB/experiments/splits/"
cp "$ROOT/paperbench_changes/analyze_judge_eval_bias.py" "$PB/"
echo "  ✅ 已复制新增文件(fre/rice split、裁判偏差分析脚本)"
echo "  ⏳ 拉取 fre / rice 论文资产(LFS,约 56MB)…"
git -C "$ROOT/frontier-evals" lfs pull --include="project/paperbench/data/papers/fre/**,project/paperbench/data/papers/rice/**"
[ "$(wc -l < "$PB/data/papers/fre/paper.md")" -gt 5 ] && echo "  ✅ 论文资产已水合" || { echo "  ❌ paper.md 仍是 LFS 指针"; exit 1; }

echo "==== [3/6] Python 环境(uv sync)===="
( cd "$ROOT/DeepCode" && uv sync --python 3.11 >/dev/null && echo "  ✅ DeepCode .venv" )
( cd "$PB" && uv sync >/dev/null && echo "  ✅ paperbench .venv" )

echo "==== [4/6] 配置模板 ===="
mkdir -p "$HOME/.deepcode"
[ -f "$HOME/.deepcode/deepcode_config.json" ] && echo "  ⏭ ~/.deepcode/deepcode_config.json 已存在" \
  || { cp "$ROOT/config/deepcode_config.example.json" "$HOME/.deepcode/deepcode_config.json"; echo "  ✅ 写入 ~/.deepcode/deepcode_config.json(默认 provider=paratera, model=DeepSeek-V4-Pro)"; }
[ -f "$HOME/.deepcode/credentials.json" ] && echo "  ⏭ ~/.deepcode/credentials.json 已存在" \
  || { cp "$ROOT/config/credentials.example.json" "$HOME/.deepcode/credentials.json"; chmod 600 "$HOME/.deepcode/credentials.json"; echo "  ✏️  请填写 ~/.deepcode/credentials.json 里的 API key"; }
[ -f "$PB/.env" ] && echo "  ⏭ paperbench/.env 已存在" \
  || { cp "$ROOT/config/paperbench.env.example" "$PB/.env"; echo "  ✏️  请填写 $PB/.env 里的 OPENAI_API_KEY(裁判用)"; }

echo "==== [5/6] 防作弊 git 封锁(按各论文 blacklist.txt)===="
for p in fre rice; do
  i=0
  while read -r url; do
    [ -z "$url" ] && continue
    repo="${url#https://github.com/}"; i=$((i+1))
    lower="$(echo "$repo" | tr 'A-Z' 'a-z')"; upper="$(echo "$repo" | tr 'a-z' 'A-Z')"
    for v in "$repo" "$lower" "$upper"; do
      git config --global "url.https://blocked.invalid/$p-$i-$v.insteadOf" "https://github.com/$v"
    done
    git config --global "url.https://blocked.invalid/$p-$i-ssh.insteadOf" "git@github.com:$repo"
    echo "  🚫 $p: $url(含大小写与 ssh 变体)"
  done < <(grep -vE '^\s*(#|$)' "$PB/data/papers/$p/blacklist.txt")
done

echo "==== [6/6] 判分提交池 ===="
mkdir -p "$HOME/pb_submissions/fre" "$HOME/pb_submissions/rice" "$HOME/pb_submissions_archive"
echo "  ✅ ~/pb_submissions/{fre,rice}/"

echo
echo "全部就绪。下一步:"
echo "  1) 填 key:~/.deepcode/credentials.json 与 $PB/.env"
echo "  2) 免费自检:PREFLIGHT_ONLY=1 PAPER=fre bash deepcode_test/scripts/run_trial.sh"
echo "  3) 跑一轮复现:PAPER=fre TRIAL=trial1 nohup bash deepcode_test/scripts/run_trial.sh > run.log 2>&1 &"
echo "  4) 判分(先 DRY 看报价):PAPER=fre DRY=1 bash deepcode_test/scripts/run_grade.sh"
