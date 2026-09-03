#!/usr/bin/env bash
# ============================================================
# DeepCode + PaperBench 本地评测环境 · 一键重建脚本(幂等)
# 用法: bash bootstrap.sh    (重复执行安全,已完成的步骤自动跳过)
# 产物: ./DeepCode  ./frontier-evals  两张 docker 镜像  论文数据
# 说明: 这是"图纸式可移植"的图纸本体;为什么这么装见 PAPERBENCH_RUNBOOK.md
# ============================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
step() { echo; echo "==== $* ===="; }

# 0) 基础工具:uv 与 git-lfs 装到 ~/.local/bin,不碰系统
export PATH="$HOME/.local/bin:$PATH"
step "0/6 基础工具"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
if ! command -v git-lfs >/dev/null; then
  tmp=$(mktemp -d)
  curl -LsS -o "$tmp/lfs.tgz" https://github.com/git-lfs/git-lfs/releases/download/v3.7.0/git-lfs-linux-amd64-v3.7.0.tar.gz
  tar xzf "$tmp/lfs.tgz" -C "$tmp"
  mkdir -p ~/.local/bin && cp "$tmp"/git-lfs-*/git-lfs ~/.local/bin/
  git lfs install --skip-repo
fi
echo "uv=$(uv --version) | $(git lfs version | cut -d' ' -f1)"

# 1) DeepCode 源码 + 虚拟环境
step "1/6 DeepCode"
[ -d "$ROOT/DeepCode" ] || git clone --depth 1 https://github.com/HKUDS/DeepCode.git "$ROOT/DeepCode"
cd "$ROOT/DeepCode"
[ -d .venv ] || uv venv --python 3.12
uv pip install -q -r requirements.txt --python .venv
.venv/bin/python deepcode.py --version

# 2) frontier-evals 稀疏克隆(只取 paperbench + common)
#    ⚠️ 故意不用 --filter=blob:none:极简克隆会让 git-lfs 枚举时
#    对仓库其他项目的指针对象发起几千次串行网络请求,直接卡死。
step "2/6 frontier-evals (稀疏,含 paperbench 依赖的 common 包)"
if [ ! -d "$ROOT/frontier-evals" ]; then
  git clone --depth 1 --sparse https://github.com/openai/frontier-evals.git "$ROOT/frontier-evals"
fi
cd "$ROOT/frontier-evals"
git sparse-checkout set project/paperbench project/common

# 3) paperbench Python 环境(uv.lock 锁定,含 4 个本仓库 editable 依赖)
step "3/6 paperbench 依赖"
cd "$ROOT/frontier-evals/project/paperbench"
uv sync -q
uv run python -m paperbench.nano.entrypoint --help >/dev/null && echo "入口 OK"

# 4) 论文数据(LFS 默认被 .lfsconfig 排除,必须显式水合)
step "4/6 论文数据(约 535MB)"
cd "$ROOT/frontier-evals"
if head -c 5 project/paperbench/data/papers/bam/paper.pdf 2>/dev/null | grep -q "%PDF"; then
  echo "已水合,跳过"
else
  git lfs fetch --include "project/paperbench/data/papers/**" --exclude ""
  git lfs checkout project/paperbench/data
fi

# 5) Docker 镜像(agent 环境 + 复跑环境)
step "5/6 Docker 镜像"
cd "$ROOT/frontier-evals/project/paperbench"
docker image inspect pb-env >/dev/null 2>&1 || \
  docker build --platform=linux/amd64 -t pb-env -f paperbench/Dockerfile.base .
docker image inspect pb-reproducer >/dev/null 2>&1 || \
  docker build --platform=linux/amd64 -t pb-reproducer -f paperbench/reproducer.Dockerfile .
# 上游 bug 修补:官方 Dockerfile 造的镜像只有 python3,而 Alcatraz 启动时
# 强制探测 `python` 和 `pip` 两个命令,缺了直接拒绝启动复跑容器。
if ! docker run --rm pb-reproducer:latest sh -c 'python --version && pip --version' >/dev/null 2>&1; then
  printf 'FROM pb-reproducer:latest\nRUN apt-get update && apt-get install -y python-is-python3 python3-pip && rm -rf /var/lib/apt/lists/*\n' | \
    docker build --platform=linux/amd64 -t pb-reproducer:latest -f - .
  echo "已补 python/pip 兼容层"
fi
docker images --format '  {{.Repository}}:{{.Tag}}  {{.Size}}' | grep '^  pb-'

# 6) 配置文件(只建骨架,key 需人工填)
step "6/6 配置文件"
[ -f .env ] || cp .env.example .env
[ -f paperbench/solvers/agent.env ] || cp paperbench/solvers/agent.env.example paperbench/solvers/agent.env
echo "待人工填写: $ROOT/frontier-evals/project/paperbench/.env (OPENAI_API_KEY 等)"

echo; echo "✅ 全部就绪。冒烟测试与跑分命令见 PAPERBENCH_RUNBOOK.md"
