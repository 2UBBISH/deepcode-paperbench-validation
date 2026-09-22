"""What the experiment actually did, in a shape the report and result JSON can carry.

"跑通了" 三个字只有在能回答下面几个问题时才算数：判据是什么（跑了哪条命令、
期望哪些测试），配环境的 Agent 敲了哪些命令、退出码是多少，裁决时 pytest 的
输出是什么，以及三点证伪有没有过。rsa 的 ``AgentOutcome`` 里这些都有，只是
散在 compile / pipeline / verdict 三层里，而 setup loop 每一轮的命令原文只在
升级卡的账本里，``RoundRecord`` 只留了个数。这里把它们收成一份 ``evidence``。
"""

from __future__ import annotations

import dataclasses
import logging
import threading
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

_LOG_TAIL_LIMIT = 6_000
_FAILURE_TAIL_LIMIT = 2_000
_ACTIONS_PER_ROUND = 60
_STATUS_LIMIT = 400


# --------------------------------------------------------------- actions --
# rsa runs in a worker thread (``asyncio.to_thread``); one experiment per
# thread, so the thread id is the natural key for "this run's rounds".
_ROUNDS: dict[int, list[dict[str, Any]]] = {}
_ROUNDS_LOCK = threading.Lock()


def begin_recording() -> None:
    with _ROUNDS_LOCK:
        _ROUNDS[threading.get_ident()] = []


def take_recorded_rounds() -> list[dict[str, Any]]:
    with _ROUNDS_LOCK:
        return _ROUNDS.pop(threading.get_ident(), [])


def record_round(actions: Any, container_id: str) -> None:
    """Called after every ``setup_loop.run_round``; a no-op outside a recording."""
    ident = threading.get_ident()
    with _ROUNDS_LOCK:
        rounds = _ROUNDS.get(ident)
        if rounds is None:
            return
        rounds.append(
            {
                "round_no": len(rounds) + 1,
                "container_id": str(container_id or ""),
                "actions": [str(a)[:300] for a in (getattr(actions, "actions", None) or [])][-_ACTIONS_PER_ROUND:],
                "stopped_voluntarily": bool(getattr(actions, "stopped_voluntarily", False)),
                "requested_help": bool(getattr(actions, "requested_help", False)),
                "error": str(getattr(actions, "error", "") or "")[:500],
            }
        )


def bind_round_recorder() -> None:
    """Wrap the vendored ``setup_loop.run_round`` so each round's commands are kept.

    Same pattern as the behaviour-summary rebind in run_flow: never let a
    diagnostic hook break the experiment, and say out loud when it is on.
    """
    try:
        from apps.v2.agent_engine.rsa import setup_loop
    except Exception:  # pragma: no cover - vendored module missing
        logger.warning("experiment evidence: rsa.setup_loop unavailable, rounds will not be recorded", exc_info=True)
        return
    if getattr(setup_loop, "_deepevol_round_recorder_bound", False):
        return
    original = setup_loop.run_round

    def run_round(*args: Any, **kwargs: Any):
        actions, cid = original(*args, **kwargs)
        try:
            record_round(actions, cid)
        except Exception:  # pragma: no cover - recording must never fail the loop
            logger.warning("experiment evidence: recording a setup round failed", exc_info=True)
        return actions, cid

    setup_loop.run_round = run_round
    setup_loop._deepevol_round_recorder_bound = True
    logger.warning("SetupX 每轮命令与退出码会记入实验证据")


# --------------------------------------------------------------- outcome --
def _str(value: Any, limit: int) -> str:
    return str(value or "")[-limit:]


def _plain(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool, dict)) or value is None:
        return value
    if dataclasses.is_dataclass(value):
        return _json_ready(dataclasses.asdict(value))
    return str(value)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(getattr(k, "value", k)): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _criterion(frozen: Any) -> dict[str, Any]:
    ladder = getattr(frozen, "ladder", None)
    rungs = getattr(ladder, "rungs", None) or {}
    approved = getattr(ladder, "approved_through", "")
    out: dict[str, Any] = {"approved_through": str(getattr(approved, "value", approved) or "")}
    listing = []
    items = rungs.items() if isinstance(rungs, dict) else []
    for name, rung in items:
        listing.append(
            {
                "rung": str(getattr(name, "value", name)),
                "command": str(getattr(rung, "command", "") or ""),
                "workdir": str(getattr(rung, "workdir", "") or ""),
                "timeout": getattr(rung, "timeout", None),
                "repeats": getattr(rung, "repeats", None),
                "expected": [str(t) for t in (getattr(rung, "expected", None) or [])][:_STATUS_LIMIT],
                "test_file_sha256": str(getattr(rung, "test_file_sha256", "") or ""),
                "artifacts": [str(getattr(a, "path", a)) for a in (getattr(rung, "artifacts", None) or [])][:50],
                "metrics": [_plain(m) for m in (getattr(rung, "metrics", None) or [])][:50],
            }
        )
    out["rungs"] = listing
    return out


def _verdict(verdict: Any) -> dict[str, Any] | None:
    if verdict is None:
        return None
    statuses = getattr(verdict, "statuses", None) or {}
    return {
        "verdict": str(getattr(verdict, "verdict", "") or ""),
        "rung": str(getattr(verdict, "rung", "") or ""),
        "authoritative": bool(getattr(verdict, "authoritative", True)),
        "expected_n": int(getattr(verdict, "expected_n", 0) or 0),
        "passed_expected": int(getattr(verdict, "passed_expected", 0) or 0),
        "pass_rate": float(getattr(verdict, "pass_rate", 0.0) or 0.0),
        "missing": [str(m) for m in (getattr(verdict, "missing", None) or [])][:_STATUS_LIMIT],
        "statuses": {str(k): str(v) for k, v in list(dict(statuses).items())[:_STATUS_LIMIT]},
        "status_counts": {str(k): int(v) for k, v in dict(getattr(verdict, "status_counts", None) or {}).items()},
        "exit_code": getattr(verdict, "exit_code", None),
        "parsed_any": bool(getattr(verdict, "parsed_any", False)),
        "duration_s": float(getattr(verdict, "duration_s", 0.0) or 0.0),
        "log_tail": _str(getattr(verdict, "log_tail", ""), _LOG_TAIL_LIMIT),
        "failure_tails": {str(k): _str(v, _FAILURE_TAIL_LIMIT) for k, v in list(dict(getattr(verdict, "failure_tails", None) or {}).items())[:20]},
        "reason": _str(getattr(verdict, "reason", ""), 1_000),
        "tampered": getattr(verdict, "tampered", None),
        "changed_files": [str(f) for f in (getattr(verdict, "changed_files", None) or [])][:50],
        "alarms": [str(a) for a in (getattr(verdict, "alarms", None) or [])][:20],
    }


def _falsification(compile_outcome: Any) -> list[dict[str, Any]]:
    reports = getattr(compile_outcome, "falsification", None) or {}
    out = []
    items = reports.items() if isinstance(reports, dict) else []
    for rung, report in items:
        out.append(
            {
                "rung": str(rung),
                "accepted": bool(getattr(report, "accepted", False)),
                "must_recompile": bool(getattr(report, "must_recompile", False)),
                "points": [
                    {
                        "point": getattr(p, "point", None),
                        "name": str(getattr(p, "name", "") or ""),
                        "verdict": str(getattr(p, "verdict", "") or ""),
                        "passed_expected": int(getattr(p, "passed_expected", 0) or 0),
                        "expected_n": int(getattr(p, "expected_n", 0) or 0),
                        "detail": _str(getattr(p, "detail", ""), 800),
                        "problems": [str(x) for x in (getattr(p, "problems", None) or [])][:10],
                    }
                    for p in (getattr(report, "points", None) or [])
                ],
                "reasons": [str(r) for r in (getattr(report, "reasons", None) or [])][:10],
                "disclosures": [str(d) for d in (getattr(report, "disclosures", None) or [])][:10],
            }
        )
    return out


def collect_evidence(outcome: Any, *, setup_rounds: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Flatten an rsa ``AgentOutcome`` (or anything shaped like it) into JSON."""
    pipeline = getattr(outcome, "pipeline", None)
    run = getattr(pipeline, "outcome", None)
    rungs = []
    for rung in getattr(run, "per_rung", None) or []:
        rungs.append(
            {
                "rung": str(getattr(rung, "rung", "") or ""),
                "terminal": str(getattr(getattr(rung, "terminal", None), "value", "") or getattr(rung, "terminal", "") or ""),
                "note": _str(getattr(rung, "note", ""), 500),
                "rounds": [
                    {
                        "round_no": getattr(r, "round_no", None),
                        "verdict": str(getattr(r, "verdict", "") or ""),
                        "passed_expected": getattr(r, "passed_expected", None),
                        "expected_n": getattr(r, "expected_n", None),
                        "actions_n": getattr(r, "actions_n", None),
                        "elapsed_s": getattr(r, "elapsed_s", None),
                        "loop_error": _str(getattr(r, "loop_error", ""), 300),
                    }
                    for r in (getattr(rung, "rounds", None) or [])
                ],
                "verdict": _verdict(getattr(rung, "verdict", None)),
            }
        )
    return {
        "agent_status": str(getattr(getattr(outcome, "status", None), "value", "") or getattr(outcome, "status", "") or ""),
        "error": _str(getattr(outcome, "error", ""), 1_000),
        "criterion": _criterion(getattr(outcome, "frozen", None)),
        "falsification": _falsification(getattr(outcome, "compile", None)),
        "terminal": str(getattr(pipeline, "terminal", "") or ""),
        "reached": str(getattr(pipeline, "reached", "") or getattr(run, "reached", "") or ""),
        "rungs": rungs,
        "setup_rounds": list(setup_rounds or []),
        "tokens": dict(getattr(pipeline, "tokens", None) or {}),
        "pipeline_attempts": len(getattr(outcome, "pipeline_attempts", None) or []),
    }


# ---------------------------------------------------------------- render --
def _code_block(text: str, limit: int) -> str:
    body = str(text or "").strip()
    if not body:
        return "（空）"
    if len(body) > limit:
        body = "…\n" + body[-limit:]
    return "```text\n" + body.replace("```", "'''") + "\n```"


def render_evidence(evidence: dict[str, Any]) -> str:
    """Markdown sections appended to the delivered report."""
    parts: list[str] = []
    criterion = evidence.get("criterion") or {}
    rungs_spec = criterion.get("rungs") or []
    if rungs_spec:
        lines = ["### 判据（冻结后不可改）"]
        for rung in rungs_spec:
            lines.append(f"- **{rung.get('rung')}**：`{rung.get('command')}`（目录 `{rung.get('workdir')}`，超时 {rung.get('timeout')}s）")
            expected = rung.get("expected") or []
            if expected:
                shown = ", ".join(f"`{t}`" for t in expected[:12])
                more = f" …共 {len(expected)} 项" if len(expected) > 12 else ""
                lines.append(f"  - 期望通过：{shown}{more}")
            if rung.get("artifacts"):
                lines.append(f"  - 期望产物：{', '.join('`%s`' % a for a in rung['artifacts'][:10])}")
        parts.append("\n".join(lines))

    falsification = evidence.get("falsification") or []
    if falsification:
        lines = ["### 判据自检（三点证伪）"]
        for report in falsification:
            mark = "✅ 通过" if report.get("accepted") else "❌ 未通过"
            lines.append(f"- {report.get('rung')}：{mark}")
            for p in report.get("points") or []:
                lines.append(f"  - 第 {p.get('point')} 点 {p.get('name')}：{p.get('verdict')}（{p.get('passed_expected')}/{p.get('expected_n')}）")
            for r in report.get("reasons") or []:
                lines.append(f"  - {r}")
        parts.append("\n".join(lines))

    setup_rounds = evidence.get("setup_rounds") or []
    if setup_rounds:
        lines = ["### 配环境的 Agent 做了什么"]
        for rnd in setup_rounds:
            head = f"**第 {rnd.get('round_no')} 轮**"
            if rnd.get("error"):
                head += f"（循环出错：{rnd['error']}）"
            lines.append(head)
            actions = rnd.get("actions") or []
            if not actions:
                lines.append("- （没有记录到动作）")
            for a in actions:
                lines.append(f"- `{a}`")
        parts.append("\n".join(lines))

    rungs = evidence.get("rungs") or []
    for rung in rungs:
        v = rung.get("verdict") or {}
        lines = [f"### 裁决 {rung.get('rung')}：{v.get('verdict') or rung.get('terminal') or '?'}"]
        if v:
            lines.append(
                f"- 期望 {v.get('expected_n')} 项，通过 {v.get('passed_expected')} 项"
                f"（通过率 {float(v.get('pass_rate') or 0):.0%}），目标命令退出码 {v.get('exit_code')}，裁决用时 {float(v.get('duration_s') or 0):.0f}s"
            )
            if v.get("status_counts"):
                lines.append("- 测试状态：" + ", ".join(f"{k}={n}" for k, n in v["status_counts"].items()))
            if v.get("missing"):
                lines.append("- 未通过：" + ", ".join(f"`{m}`" for m in v["missing"][:20]))
            if v.get("tampered"):
                lines.append(f"- ⚠️ 裁决发现仓库被改动：{', '.join(v.get('changed_files') or [])[:500]}")
            for alarm in v.get("alarms") or []:
                lines.append(f"- ⚠️ {alarm}")
            if v.get("reason"):
                lines.append(f"- 说明：{v['reason']}")
            lines.append("")
            lines.append("**目标命令输出（末尾）**")
            lines.append(_code_block(v.get("log_tail") or "", 4_000))
            for test_id, tail in (v.get("failure_tails") or {}).items():
                lines.append(f"**{test_id} 失败输出**")
                lines.append(_code_block(tail, 1_500))
        else:
            lines.append(f"- 没有裁决结果（{rung.get('note') or rung.get('terminal') or '未知'}）")
        parts.append("\n".join(lines))

    tokens = evidence.get("tokens") or {}
    if tokens:
        parts.append("### 配环境 Agent 用量\n- " + ", ".join(f"{k}={v}" for k, v in tokens.items()))
    return "\n\n".join(parts)


__all__ = [
    "begin_recording",
    "bind_round_recorder",
    "collect_evidence",
    "record_round",
    "render_evidence",
    "take_recorded_rounds",
]
