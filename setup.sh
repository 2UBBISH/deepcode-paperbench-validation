#!/usr/bin/env bash
# ============================================================
# 一键环境搭建（clone 之后只需跑这一个脚本；幂等，重复执行只补缺）
#
#   git clone git@github.com:2UBBISH/deepcode-paperbench-validation.git && cd deepcode-paperbench-validation
#   bash setup.sh                       # 默认只水合 sapg 的论文资产；PAPERS="sapg bam" bash setup.sh 可多篇
#
# 做的事：
#   1. 检查 git / curl / uv / node / npm / patch / docker
#   2. 稀疏克隆 openai/frontier-evals（固定 commit），打 patches/paperbench_local_changes.patch，
#      复制我们新增的 split 等文件，按 $PAPERS 下载论文资产（LFS 直链）
#   3. 校验 DeepCode/ = 上游 21ebc57f + patches/deepcode_local_changes.patch，然后 uv venv + uv pip install -r requirements.txt（Python 3.12）
#   4. 生成 $DEEPCODE_HOME/deepcode_config.json（口径：DeepSeek-V4-Flash、思考关、无阶段覆盖）与 paperbench/.env 模板
#   5. 按各论文 blacklist.txt 设 git insteadOf 封锁（复现时禁止克隆论文官方实现）
#   6. 建 ~/pb_submissions/<paper>/ 判分提交池
#
# 环境变量：
#   PAPERS         要准备的论文 id，空格分隔（默认 sapg）
#   DEEPCODE_HOME  DeepCode 的配置目录（默认 <仓库>/.deepcode-home，与你机器上别的 DeepCode 完全隔离）
#   PYTHON_VERSION DeepCode venv 的 Python（默认 3.12；上游 ruff target 也是 3.12）
# key 不在这里填：跑 run_trial.sh 时用 ENV_FILE=<含 PARATERA_API_KEY=... 的文件>，或写 $DEEPCODE_HOME/credentials.json。
# ============================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPERS="${PAPERS:-sapg}"
DEEPCODE_HOME="${DEEPCODE_HOME:-$ROOT/.deepcode-home}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
FE_REPO="https://github.com/openai/frontier-evals.git"
FE_COMMIT="$(grep '^frontier-evals upstream' "$ROOT/patches/UPSTREAM_BASE.txt" | awk '{print $NF}')"
DC_COMMIT="$(grep '^DeepCode upstream' "$ROOT/patches/UPSTREAM_BASE.txt" | awk '{print $NF}')"
PB="$ROOT/frontier-evals/project/paperbench"

echo "==== [1/6] 依赖检查 ===="
need() { command -v "$1" >/dev/null 2>&1 || { echo "  ❌ 缺 $1 —— $2"; exit 1; }; echo "  ✅ $1 ($(command -v "$1"))"; }
need git   "https://git-scm.com"
need curl  "下载论文资产"
need uv    "curl -LsSf https://astral.sh/uv/install.sh | sh"
need node  "Node.js ≥ 18；filesystem MCP 是一个 node 服务器"
need npm   "随 Node 一起；setup 把 @modelcontextprotocol/server-filesystem 装到 <仓库>/.mcp-node"
need patch "GNU/BSD patch，校验 DeepCode/ 用"
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then echo "  ✅ docker daemon 在跑（判分要用）"; else echo "  ⚠️ docker 未运行（复现不需要，判分前再启动）"; fi

echo "==== [2/6] frontier-evals（PaperBench）@ ${FE_COMMIT:0:10} ===="
# 不能用 --filter=blob:none 部分克隆：git-lfs 在缺 blob 的仓库里逐个懒取指针，实测挂死 3 分钟以上。
# 改为「按 commit 浅取（depth 1）+ 稀疏检出」，只下载 paperbench 与 common 两个目录。
if [ ! -d "$ROOT/frontier-evals/.git" ]; then
  git init -q "$ROOT/frontier-evals"
  git -C "$ROOT/frontier-evals" remote add origin "$FE_REPO"
  git -C "$ROOT/frontier-evals" sparse-checkout init --cone
  git -C "$ROOT/frontier-evals" sparse-checkout set project/paperbench project/common
  git -C "$ROOT/frontier-evals" fetch -q --depth 1 origin "$FE_COMMIT"
  # checkout 时跳过 LFS smudge（否则把全部 20 余篇论文约 300MB 资产都拉下来），稍后只按需拉 $PAPERS
  GIT_LFS_SKIP_SMUDGE=1 git -C "$ROOT/frontier-evals" checkout -q FETCH_HEAD
  git -C "$ROOT/frontier-evals" lfs install --local --skip-smudge >/dev/null 2>&1 || true
  echo "  ✅ 稀疏检出完成（paperbench + common）"
else
  echo "  ⏭ 已存在，跳过克隆"
fi
if ! git -C "$ROOT/frontier-evals" apply --check "$ROOT/patches/paperbench_local_changes.patch" >/dev/null 2>&1; then
  git -C "$ROOT/frontier-evals" apply --reverse --check "$ROOT/patches/paperbench_local_changes.patch" >/dev/null 2>&1 \
    && echo "  ⏭ paperbench patch 已打过" || { echo "  ❌ paperbench patch 无法应用（上游 commit 不符？）"; exit 1; }
else
  git -C "$ROOT/frontier-evals" apply "$ROOT/patches/paperbench_local_changes.patch"; echo "  ✅ 已打 paperbench patch（5 文件）"
fi
cp "$ROOT/paperbench_changes/experiments/splits/"*.txt "$PB/experiments/splits/"
cp "$ROOT/paperbench_changes/analyze_judge_eval_bias.py" "$PB/"
echo "  ✅ 已复制新增文件（单篇 split、裁判偏差分析脚本）"
# 论文资产在上游是 LFS 对象。稀疏+浅克隆下 `git lfs pull --include` 实测拿不到对象（退出 0 但仍是指针），
# 改为直接从 GitHub 的 LFS 媒体直链按固定 commit 下载，只取 $PAPERS，不依赖 git-lfs 客户端。
for p in $PAPERS; do
  [ -d "$PB/data/papers/$p" ] || { echo "  ❌ PaperBench 没有论文 id '$p'（见 $PB/data/papers/）"; exit 1; }
  n=0
  while read -r f; do
    rel="${f#$ROOT/frontier-evals/}"
    curl -sfL --retry 3 -o "$f" "https://media.githubusercontent.com/media/openai/frontier-evals/$FE_COMMIT/$rel" \
      || { echo "  ❌ 下载失败: $rel"; exit 1; }
    n=$((n+1))
  done < <(grep -rl '^version https://git-lfs' "$PB/data/papers/$p" 2>/dev/null || true)
  [ "$(wc -l < "$PB/data/papers/$p/paper.md")" -gt 5 ] && echo "  ✅ $p 论文资产已水合（本次下载 $n 个文件）" || { echo "  ❌ $p/paper.md 仍是 LFS 指针"; exit 1; }
done

echo "==== [3/6] DeepCode = HKUDS/DeepCode@${DC_COMMIT:0:8} + patch；venv（Python $PYTHON_VERSION）+ requirements.txt ===="
bash "$ROOT/patches/verify_deepcode.sh"
# 上游 21ebc57f 的依赖在 requirements.txt（setup.py 读它；pyproject 没有 [project]，uv.lock 是空的），所以不用 uv sync。
# --only-binary cryptography：Intel macOS 上最新 cryptography 没有 wheel、从源码编译失败，让 uv 退到有 wheel 的版本（实测 48.0.1）
( cd "$ROOT/DeepCode" \
  && { [ -x .venv/bin/python ] || uv venv --python "$PYTHON_VERSION" .venv >/dev/null; } \
  && uv pip install --python .venv/bin/python -q --only-binary cryptography -r requirements.txt mcp-server-fetch \
  && echo "  ✅ DeepCode/.venv ($(.venv/bin/python --version)，requirements.txt + mcp-server-fetch 已装)" )
# 两个外部 MCP 服务器装成固定路径，而不是运行时 npx/uvx（npx 首次解析包要 20 秒以上、会撞 MCP 连接超时；
# uvx mcp-server-fetch 会重新编译 cryptography）。fetch 装进 DeepCode/.venv，filesystem 装到 <仓库>/.mcp-node。
# filesystem MCP 启动时校验允许目录存在，否则立刻退出（DeepCode 侧只看到 "Connection closed"）：先建好工作区
mkdir -p "$ROOT/DeepCode/deepcode_lab"
FS_JS="$ROOT/.mcp-node/node_modules/@modelcontextprotocol/server-filesystem/dist/index.js"
if [ ! -f "$FS_JS" ]; then
  npm install --silent --no-audit --no-fund --prefix "$ROOT/.mcp-node" @modelcontextprotocol/server-filesystem >/dev/null
fi
[ -f "$FS_JS" ] && echo "  ✅ filesystem MCP: $FS_JS" || { echo "  ❌ 装不上 @modelcontextprotocol/server-filesystem"; exit 1; }
[ -x "$ROOT/DeepCode/.venv/bin/mcp-server-fetch" ] && echo "  ✅ fetch MCP: DeepCode/.venv/bin/mcp-server-fetch" || { echo "  ❌ DeepCode/.venv/bin/mcp-server-fetch 不存在"; exit 1; }
( cd "$PB" && uv sync >/dev/null && echo "  ✅ paperbench .venv" )

echo "==== [4/6] 配置：DEEPCODE_HOME=$DEEPCODE_HOME ===="
mkdir -p "$DEEPCODE_HOME"; chmod 700 "$DEEPCODE_HOME"
if [ -f "$DEEPCODE_HOME/deepcode_config.json" ]; then
  echo "  ⏭ deepcode_config.json 已存在（不覆盖；要重生成先删掉它）"
else
  sed -e "s#__PY__#$ROOT/DeepCode/.venv/bin/python#g" \
      -e "s#__NODE__#$(command -v node)#g" \
      -e "s#__FS_JS__#$ROOT/.mcp-node/node_modules/@modelcontextprotocol/server-filesystem/dist/index.js#g" \
      -e "s#__FETCH__#$ROOT/DeepCode/.venv/bin/mcp-server-fetch#g" \
      -e "s#__WORKSPACE__#$ROOT/DeepCode/deepcode_lab#g" \
      "$ROOT/config/deepcode_config.template.json" > "$DEEPCODE_HOME/deepcode_config.json"
  chmod 600 "$DEEPCODE_HOME/deepcode_config.json"
  echo "  ✅ 写入 deepcode_config.json（DeepSeek-V4-Flash @ Paratera，compat.thinking=disabled，maxTokens 32768，7 个 MCP 服务器）"
fi
echo "  ✏️  key：跑 run_trial.sh 时传 ENV_FILE=<文件>（内容一行 PARATERA_API_KEY=...），"
echo "      或写 $DEEPCODE_HOME/credentials.json（模板 config/credentials.example.json，chmod 600）。两处都不进仓库。"
[ -f "$PB/.env" ] && echo "  ⏭ paperbench/.env 已存在" \
  || { cp "$ROOT/config/paperbench.env.example" "$PB/.env"; echo "  ✏️  判分才需要：填 $PB/.env 里的 OPENAI_API_KEY（裁判用）"; }

echo "==== [5/6] 防作弊 git 封锁（按各论文 blacklist.txt）===="
for p in $PAPERS; do
  i=0
  while read -r url; do
    [ -z "$url" ] && continue
    repo="${url#https://github.com/}"; repo="${repo%/}"; repo="${repo%.git}"; i=$((i+1))
    lower="$(echo "$repo" | tr 'A-Z' 'a-z')"; upper="$(echo "$repo" | tr 'a-z' 'A-Z')"
    for v in "$repo" "$lower" "$upper"; do
      git config --global "url.https://blocked.invalid/$p-$i-$v.insteadOf" "https://github.com/$v"
    done
    git config --global "url.https://blocked.invalid/$p-$i-ssh.insteadOf" "git@github.com:$repo"
    echo "  🚫 $p: $url（含大小写与 ssh 变体）"
  done < <(grep -vE '^\s*(#|$)' "$PB/data/papers/$p/blacklist.txt")
done
echo "  （git insteadOf 只挡 git 协议；HTTP 抓取由 run_trial.sh 注入的 DEEPCODE_URL_DENYLIST 在 MCP 层拦）"

echo "==== [6/6] 判分提交池 ===="
for p in $PAPERS; do mkdir -p "$HOME/pb_submissions/$p"; done
mkdir -p "$HOME/pb_submissions_archive"
echo "  ✅ ~/pb_submissions/{$(echo $PAPERS | tr ' ' ',')}/"

echo
echo "全部就绪。下一步："
echo "  1) 免费自检：PREFLIGHT_ONLY=1 PAPER=sapg ENV_FILE=~/my.env bash deepcode_test/scripts/run_trial.sh"
echo "  2) 跑一轮基线运行：PAPER=sapg TRIAL=trial1 ENV_FILE=~/my.env nohup bash deepcode_test/scripts/run_trial.sh > run.log 2>&1 &"
echo "  3) 判分（先 DRY 看报价）：PAPER=sapg DRY=1 bash deepcode_test/scripts/run_grade.sh"
