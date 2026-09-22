"""The ``plan_review`` phase: automatic approval, or the file-backed ``--ask`` loop.

Without ``--ask`` the engine's own gate runs with a callback that approves
(it still records ``plan_versions/initial_plan.v00.generated.txt`` and the
review history). With ``--ask`` the process cannot wait for a human, so
the same loop is driven one decision per invocation: the request goes to
``phases/04_plan_review.request.json`` and the phase is ``waiting``; the
operator writes ``phases/04_plan_review.decision.json``
(``{"action": "approve" | "modify" | "replace" | "cancel", "feedback":
..., "plan": ...}``) and runs ``step --phase plan_review`` again, which
consumes it. ``modify`` revises the plan through the engine and asks
again; at most three modify rounds. Versions, events and meta are written
with the engine's own helpers so the task directory looks the same either
way.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from apps.v2.agent_engine.paper2code.workflows.plan_review_runtime import (
    PlanReviewCancelled,
    _build_review_request,
    _decision_value,
    _normalise_action,
    _validation_error,
    append_plan_review_event,
    read_plan_file,
    revise_plan_with_feedback,
    run_plan_review_gate,
    save_plan_version,
    update_plan_review_meta,
    write_plan_file,
)
from apps.v2.agent_engine.paper2code.workflows.planning_runtime import (
    extract_yaml_candidate,
    utc_now_iso,
    validate_plan_text,
)

MAX_MODIFY_ROUNDS = 3
REQUEST_FILE = "04_plan_review.request.json"
DECISION_FILE = "04_plan_review.decision.json"
STATE_FILE = "04_plan_review.state.json"


async def _auto_approve(request: dict[str, Any]) -> dict[str, Any]:
    return {"action": "approve", "auto": True}


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    return data if isinstance(data, dict) else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


async def review_step(
    *,
    task_dir: Path,
    plan_path: Path,
    phases_dir: Path,
    ask: bool,
    logger: Any,
    max_rounds: int = MAX_MODIFY_ROUNDS,
) -> dict[str, Any]:
    """One invocation of the review. Returns the engine-shaped status dict; ``waiting`` when a decision is needed."""
    if not ask:
        result = await run_plan_review_gate(initial_plan_path=plan_path, paper_dir=task_dir, callback=_auto_approve, logger=logger)
        result["mode"] = "auto"
        return result

    state_path = phases_dir / STATE_FILE
    request_path = phases_dir / REQUEST_FILE
    decision_path = phases_dir / DECISION_FILE
    state = _read_json(state_path) or {"round": 0, "interactions": 0, "started": False, "last_error": None}
    plan = read_plan_file(plan_path)

    if not state["started"]:
        save_plan_version(task_dir, plan, version=0, label="generated")
        append_plan_review_event(task_dir, {"event": "review_started", "initial_plan_path": str(plan_path), "plan_chars": len(plan), "validation": validate_plan_text(plan), "mode": "ask"})
        update_plan_review_meta(task_dir, status="waiting_for_review", enabled=True, rounds=0, approved=False, initial_plan_path=str(plan_path))
        state["started"] = True
        _write_json(state_path, state)

    def _ask(last_error: str | None) -> dict[str, Any]:
        state["interactions"] += 1
        state["last_error"] = last_error
        request = _build_review_request(paper_dir=task_dir, initial_plan_path=plan_path, plan=plan, validation=validate_plan_text(plan), modification_round=state["round"], max_rounds=max_rounds, last_error=last_error)
        request["decision_file"] = str(decision_path)
        request["instructions"] = (
            f"Write {decision_path.name} next to this file with {{\"action\": \"approve\" | \"modify\" | \"replace\" | \"cancel\", "
            "\"feedback\": \"...\" (modify), \"plan\": \"...\" (replace)}} and run `step --phase plan_review` again."
        )
        request["requested_at"] = utc_now_iso()
        _write_json(request_path, request)
        append_plan_review_event(task_dir, {"event": "review_requested", "interaction": state["interactions"], "round": state["round"], "validation": request["data"]["plan_validation"], "mode": "ask"})
        _write_json(state_path, state)
        return {"status": "waiting", "mode": "ask", "request": str(request_path), "decision_file": str(decision_path), "round": state["round"], "interactions": state["interactions"], "last_error": last_error}

    decision = _read_json(decision_path)
    if decision is None:
        return _ask(state.get("last_error"))

    consumed = phases_dir / f"04_plan_review.decision.{state['interactions']:02d}.{int(time.time())}.json"
    decision_path.rename(consumed)
    action = _normalise_action(decision)
    append_plan_review_event(task_dir, {"event": "review_response", "interaction": state["interactions"], "round": state["round"], "action": action, "mode": "ask", "decision_file": str(consumed)})

    if action in {"approve", "skip"}:
        final_validation = validate_plan_text(plan)
        update_plan_review_meta(task_dir, status="approved", approved=True, auto_approved=False, rounds=state["round"], interactions=state["interactions"], final_plan_chars=len(plan), final_validation=final_validation, approved_at=utc_now_iso())
        append_plan_review_event(task_dir, {"event": "review_approved", "action": action, "round": state["round"], "validation": final_validation})
        if request_path.exists():
            request_path.unlink()
        _write_json(state_path, {**state, "done": True})
        return {"status": "approved", "mode": "ask", "action": action, "rounds": state["round"], "interactions": state["interactions"], "plan_validation": final_validation, "initial_plan_path": str(plan_path)}

    if action == "cancel":
        reason = str(_decision_value(decision, "reason", "feedback") or "cancelled at plan review")
        update_plan_review_meta(task_dir, status="cancelled", approved=False, rounds=state["round"], interactions=state["interactions"], cancel_reason=reason)
        append_plan_review_event(task_dir, {"event": "review_cancelled", "reason": reason, "round": state["round"]})
        _write_json(state_path, {**state, "done": True, "cancelled": True})
        raise PlanReviewCancelled(reason)

    if action == "replace":
        replacement = str(_decision_value(decision, "plan", "replacement_plan") or "").strip()
        if not replacement:
            return _ask("Replacement plan was empty")
        replacement_validation = validate_plan_text(replacement)
        if not replacement_validation.get("valid", False):
            append_plan_review_event(task_dir, {"event": "replacement_rejected", "round": state["round"], "validation": replacement_validation})
            return _ask(_validation_error(replacement_validation))
        state["round"] += 1
        plan = extract_yaml_candidate(replacement).strip()
        write_plan_file(plan_path, plan)
        save_plan_version(task_dir, plan, version=state["round"], label="manual")
        update_plan_review_meta(task_dir, status="modified", approved=False, rounds=state["round"], last_action="replace", last_validation=replacement_validation)
        return _ask(None)

    if action == "modify":
        feedback = str(_decision_value(decision, "feedback") or "").strip()
        if not feedback:
            return _ask("Modification feedback was empty")
        if state["round"] >= max_rounds:
            return _ask(f"Maximum modification rounds reached ({max_rounds}); approve, replace manually, or cancel.")
        update_plan_review_meta(task_dir, status="revision_running", approved=False, rounds=state["round"], last_feedback=feedback)
        try:
            revised = await revise_plan_with_feedback(plan, feedback, paper_dir=task_dir, logger=logger)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            append_plan_review_event(task_dir, {"event": "revision_failed", "round": state["round"], "error": error})
            return _ask(error)
        state["round"] += 1
        plan = revised
        write_plan_file(plan_path, plan)
        revised_validation = validate_plan_text(plan)
        save_plan_version(task_dir, plan, version=state["round"], label="ai")
        update_plan_review_meta(task_dir, status="modified", approved=False, rounds=state["round"], last_action="modify", last_feedback=feedback, last_validation=revised_validation)
        return _ask(None)

    return _ask(f"Unknown action {action!r}; use approve, modify, replace or cancel")


__all__ = ["DECISION_FILE", "MAX_MODIFY_ROUNDS", "REQUEST_FILE", "STATE_FILE", "review_step"]
