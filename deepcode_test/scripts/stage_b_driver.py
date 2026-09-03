#!/usr/bin/env python3
"""阶段 B 驱动:直接调用 DeepCode 的论文复现流水线跑 rice。

为什么不用 `deepcode test rice`:该命令引用的 test_paper.py 在仓库里不存在
(上游帮助文本漂移),所以走官方管道的直接入口
execute_multi_agent_research_pipeline —— 这正是 CLI/桌面版底下用的同一函数。

用法: 在 DeepCode 目录下用它的 .venv 运行(通常由 run_trial.sh 调用)
  STAGE_B_INPUT=<paper.md> STAGE_B_SLUG=<paper> .venv/bin/python ../deepcode_test/scripts/stage_b_driver.py [--fast]
产出: deepcode_lab/tasks/<task>/generate_code/,路径写入 /tmp/stage_b_code_dir_<slug>.txt
"""

import asyncio
import glob
import logging
import os
import sys

FAST = "--fast" in sys.argv
# 断点续跑:喂任务目录内部的 paper.md,DeepCode 会复用该任务目录
# (跳过输入获取,且已有的合格 initial_plan.txt 不重新生成 = 省钱)。
#
# 无默认值是有意为之:曾经默认指向 rice,调用方忘了 export 就会静默复现错误
# 的论文,而分数看上去完全正常。宁可拒跑,不可跑错。
PAPER_MD = os.environ.get("STAGE_B_INPUT")
if not PAPER_MD:
    sys.exit(
        "STAGE_B_INPUT 未设置。显式指定要复现的论文,例如:\n"
        "  export STAGE_B_INPUT=.../data/papers/fre/paper.md"
    )
# 交接文件按论文分文件,避免跨论文读到上一篇的 stale 路径。
SLUG = os.environ.get("STAGE_B_SLUG", "default")
CODE_DIR_FILE = f"/tmp/stage_b_code_dir_{SLUG}.txt"
STATUS_FILE = f"/tmp/stage_b_status_{SLUG}.txt"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger("stage_b")


async def main() -> None:
    from workflows.agent_orchestration_engine import (
        execute_multi_agent_research_pipeline,
    )

    assert os.path.exists(PAPER_MD), f"论文不存在: {PAPER_MD}"
    print(f"输入论文: {PAPER_MD}")
    print(f"模式: {'快速(跳过参考/索引)' if FAST else '完整(含参考挖掘+索引)'}")

    result = await execute_multi_agent_research_pipeline(
        input_source=PAPER_MD,
        logger=logger,
        enable_indexing=not FAST,
    )

    print("流水线状态:", result.get("status"))
    print("摘要:", str(result.get("summary"))[:500])
    # 供点火脚本的判分闸读取(按论文分文件,防跨论文 stale)
    with open(STATUS_FILE, "w") as f:
        f.write(str(result.get("status")))
    paper_dir = result.get("paper_dir") or ""
    impl = result.get("implementation") or {}
    code_dir = impl.get("code_directory") or os.path.join(paper_dir, "generate_code")

    if not (code_dir and os.path.isdir(code_dir)):
        # 兜底只在**本轮任务目录内**找,绝不跨任务全局搜:旧版在整个
        # deepcode_lab 下按 mtime 取最新,一旦本轮早夭就会交出上一篇论文的
        # 产物,而下游只检查"路径是否在 tasks/ 下"、拦不住。
        if not paper_dir:
            raise RuntimeError(
                "流水线未返回 paper_dir,且实现阶段无产物目录;拒绝跨任务兜底"
            )
        cands = sorted(
            glob.glob(os.path.join(paper_dir, "**", "generate_code"), recursive=True),
            key=os.path.getmtime,
        )
        if not cands:
            raise RuntimeError(f"本轮任务目录内找不到 generate_code: {paper_dir}")
        code_dir = cands[-1]

    n_files = sum(len(fs) for _, _, fs in os.walk(code_dir))
    print(f"产物目录: {code_dir}({n_files} 个文件)")
    with open(CODE_DIR_FILE, "w") as f:
        f.write(os.path.abspath(code_dir))

    if result.get("status") not in ("completed", "completed_with_warnings"):
        print("⚠️ 流水线非完全成功(见上方状态);产物仍已定位,可继续判分对照。")


if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())
    asyncio.run(main())
