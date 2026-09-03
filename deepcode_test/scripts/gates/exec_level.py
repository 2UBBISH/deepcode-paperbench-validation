#!/usr/bin/env python3
"""执行级评分器 —— 给自优化循环用的确定性目标函数。

为什么存在:本仓库的 PaperBench 裁判分**不能**当目标函数用(σ≈0.1,真实组间效应
0.01~0.02;换裁判 serving 同一份代码 16% 叶级分歧)。详见 ../../docs/OPTIMIZER_NOTICE.md。
本脚本提供一组确定性、零成本、秒级的判据作为替代。

用法:
    python3 exec_level.py <提交目录> [<提交目录> ...]      # 人读表格
    python3 exec_level.py --json <提交目录>                # 机器读,自优化循环用

输出(--json):
    {"path": ..., "score": 4, "max": 5, "gates": {"compiles": true, ...},
     "details": {...}}

判据全部只看磁盘产物,不调用任何模型,不需要网络与 API key。
`score` 是通过的判据数,可直接作为 fitness;`gates` 逐项布尔值可用于定位。

⚠️ 这些是**结构性就绪度**判据,不是"跑通了"。真正的执行级验收(装依赖、import、
冒烟训练、指标 schema)需要容器,见 ../../docs/ARCHITECTURE_v0.2_OPTIMAL.md 的度量体 B。
本脚本是那之前的最小可用替代。
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import sys

GATE_DOC = {
    "compiles": "全部 .py 语法可解析(ast.parse)",
    "has_entrypoint": "存在 __main__ 入口",
    "declares_deps": "有 requirements.txt / setup.py / pyproject.toml / environment.yml",
    "has_run_script": "有 reproduce*.sh 或 run*.sh",
    "has_config": "有 .yaml/.yml 或 *config*.py",
}


def _py_files(root: str) -> list[str]:
    return [
        f
        for f in glob.glob(os.path.join(root, "**", "*.py"), recursive=True)
        if "/.git/" not in f and "/__pycache__/" not in f
    ]


def score_submission(root: str) -> dict:
    root = os.path.abspath(os.path.expanduser(root))
    pys = _py_files(root)

    bad_syntax = []
    for f in pys:
        try:
            ast.parse(open(f, encoding="utf-8", errors="replace").read())
        except Exception as exc:  # noqa: BLE001 - 任何解析失败都算不通过
            bad_syntax.append({"file": os.path.relpath(f, root), "error": str(exc)[:160]})

    def _read(f: str) -> str:
        return open(f, encoding="utf-8", errors="replace").read()

    entry = [os.path.relpath(f, root) for f in pys if "__main__" in _read(f)]
    dep_files = [
        x
        for x in ("requirements.txt", "setup.py", "pyproject.toml", "environment.yml")
        if os.path.exists(os.path.join(root, x))
    ]
    run_scripts = [
        os.path.relpath(f, root)
        for f in glob.glob(os.path.join(root, "**", "reproduce*.sh"), recursive=True)
        + glob.glob(os.path.join(root, "**", "run*.sh"), recursive=True)
    ]
    configs = [
        os.path.relpath(f, root)
        for f in glob.glob(os.path.join(root, "**", "*.yaml"), recursive=True)
        + glob.glob(os.path.join(root, "**", "*.yml"), recursive=True)
        + glob.glob(os.path.join(root, "**", "*config*.py"), recursive=True)
    ]

    gates = {
        "compiles": bool(pys) and not bad_syntax,
        "has_entrypoint": bool(entry),
        "declares_deps": bool(dep_files),
        "has_run_script": bool(run_scripts),
        "has_config": bool(configs),
    }
    return {
        "path": root,
        "score": sum(gates.values()),
        "max": len(gates),
        "gates": gates,
        "details": {
            "py_files": len(pys),
            "syntax_errors": bad_syntax,
            "entrypoints": entry[:5],
            "dep_files": dep_files,
            "run_scripts": run_scripts[:5],
            "configs": configs[:5],
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="提交目录")
    ap.add_argument("--json", action="store_true", help="输出 JSON(自优化循环用)")
    args = ap.parse_args()

    results = []
    for p in args.paths:
        if not os.path.isdir(os.path.expanduser(p)):
            print(f"❌ 不是目录: {p}", file=sys.stderr)
            return 2
        results.append(score_submission(p))

    if args.json:
        print(json.dumps(results if len(results) > 1 else results[0], ensure_ascii=False, indent=1))
        return 0

    names = [os.path.basename(r["path"]) for r in results]
    w = max(len(n) for n in names) + 1
    keys = list(GATE_DOC)
    print(f"{'提交':{w}s} " + " ".join(f"{k[:9]:>9s}" for k in keys) + "   得分")
    print("-" * (w + 10 * len(keys) + 8))
    for n, r in zip(names, results):
        cells = " ".join(f"{'✅' if r['gates'][k] else '❌':>9s}" for k in keys)
        print(f"{n:{w}s} {cells}   {r['score']}/{r['max']}")
    if any(r["details"]["syntax_errors"] for r in results):
        print("\n语法错误明细:")
        for n, r in zip(names, results):
            for e in r["details"]["syntax_errors"]:
                print(f"  {n}: {e['file']} — {e['error']}")
    print("\n判据说明:")
    for k, v in GATE_DOC.items():
        print(f"  {k:16s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
