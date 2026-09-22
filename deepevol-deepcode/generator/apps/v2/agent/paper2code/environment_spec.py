"""Environment spec (环境规格): what the generated repository will need, read out of the blueprint.

One model call after planning (PLAN-3 item 2). The blueprint's ``environment_setup`` and
``implementation_components`` are free text; this turns them into a fixed JSON shape that the
compute step (tiers) and the environment step (recipe) consume. Rules the prompt enforces and
the normaliser re-checks: only what the blueprint states, unknown means ``null``, no guessed
versions or sizes. The planner prompt itself is untouched (the engine stays upstream-aligned).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

SPEC_VERSION = 1
MAX_TOKENS = 4096
SECTION_RE = re.compile(r"^  (?P<key>[a-z_]+):(?P<rest>.*?)(?=^  [a-z_]+:|\Z)", re.S | re.M)
SECTIONS = ("environment_setup", "implementation_components", "validation_approach")

PROMPT = """You are turning the free-text sections of a code reproduction blueprint into a fixed JSON
environment specification. Copy facts; do not invent. When the blueprint does not state something,
use null (or an empty list). Never guess versions, sizes or hardware the text does not mention.

Return exactly one JSON object with these keys and nothing else:
{
  "language": {"name": "python", "version": "<string or null>"},
  "system_packages": ["<apt/OS package names the blueprint names>"],
  "python_packages": [{"name": "<import/pip name>", "spec": "<version constraint or null>", "note": "<short, or null>"}],
  "gpu": {"required": true | false | "optional", "reason": "<short quote or paraphrase, or null>"},
  "cuda": "<version string or null>",
  "datasets": [{"name": "<name>", "source": "<url | 'generated' | 'provided' | null>", "size_gb": <number or null>, "required": true | false}],
  "external_tools": [{"name": "<tool/simulator/library outside pip>", "installable": true | false | null, "reason": "<why, or null>"}],
  "run_commands": ["<commands the blueprint gives for running or validating, verbatim>"],
  "notes": ["<anything about the environment that fits nowhere above>"]
}

Blueprint sections:
"""


def blueprint_sections(plan_text: str) -> dict[str, str]:
    """The three blueprint sections the spec is read from, as raw text (missing ones absent)."""
    found: dict[str, str] = {}
    for match in SECTION_RE.finditer(plan_text or ""):
        key = match.group("key")
        if key in SECTIONS:
            found[key] = match.group("rest").strip()
    return found


def build_prompt(plan_text: str) -> str:
    sections = blueprint_sections(plan_text)
    body = "\n\n".join(f"## {key}\n{sections[key]}" for key in SECTIONS if key in sections)
    if not body:
        body = "## (no recognised sections; the whole blueprint follows)\n" + (plan_text or "")[:12000]
    return PROMPT + body


def extract_json(text: str) -> dict[str, Any]:
    """The first JSON object in a reply (fenced or bare); raises ``ValueError`` when there is none."""
    body = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", body, re.S)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start = body.find("{")
        end = body.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("no JSON object in the reply")
        candidate = body[start : end + 1]
    data = json.loads(candidate)
    if not isinstance(data, dict):
        raise ValueError("the JSON is not an object")
    return data


def _str_or_none(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _number_or_none(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def normalize(data: dict[str, Any]) -> dict[str, Any]:
    """Coerce a model reply into the fixed shape; anything malformed becomes null or is dropped."""
    lang = data.get("language") if isinstance(data.get("language"), dict) else {}
    gpu = data.get("gpu") if isinstance(data.get("gpu"), dict) else {}
    gpu_required = gpu.get("required")
    if gpu_required not in (True, False, "optional"):
        gpu_required = None
    packages = []
    for item in data.get("python_packages") or []:
        if isinstance(item, str):
            item = {"name": item}
        if isinstance(item, dict) and _str_or_none(item.get("name")):
            packages.append({"name": item["name"].strip(), "spec": _str_or_none(item.get("spec")), "note": _str_or_none(item.get("note"))})
    datasets = []
    for item in data.get("datasets") or []:
        if isinstance(item, dict) and _str_or_none(item.get("name")):
            datasets.append({"name": item["name"].strip(), "source": _str_or_none(item.get("source")), "size_gb": _number_or_none(item.get("size_gb")), "required": bool(item.get("required", True))})
    tools = []
    for item in data.get("external_tools") or []:
        if isinstance(item, dict) and _str_or_none(item.get("name")):
            installable = item.get("installable")
            tools.append({"name": item["name"].strip(), "installable": installable if isinstance(installable, bool) else None, "reason": _str_or_none(item.get("reason"))})
    return {
        "version": SPEC_VERSION,
        "language": {"name": _str_or_none(lang.get("name")) or "python", "version": _str_or_none(lang.get("version"))},
        "system_packages": [s.strip() for s in data.get("system_packages") or [] if isinstance(s, str) and s.strip()],
        "python_packages": packages,
        "gpu": {"required": gpu_required, "reason": _str_or_none(gpu.get("reason"))},
        "cuda": _str_or_none(data.get("cuda")),
        "datasets": datasets,
        "external_tools": tools,
        "run_commands": [s.strip() for s in data.get("run_commands") or [] if isinstance(s, str) and s.strip()],
        "notes": [s.strip() for s in data.get("notes") or [] if isinstance(s, str) and s.strip()],
    }


async def extract_environment_spec(provider: Any, plan_text: str) -> dict[str, Any]:
    """One model call; returns ``{"status": "ok", "spec": …}`` or ``{"status": "failed", "error": …}``.

    A failed extraction never fails the plan: the compute and environment steps treat an absent
    spec explicitly (they do not fall back to guessing).
    """
    prompt = build_prompt(plan_text)
    response = await provider.chat_with_retry(
        [{"role": "user", "content": prompt}], max_tokens=MAX_TOKENS, temperature=0.0, retry_mode="standard"
    )
    usage = dict(response.usage or {})
    if response.finish_reason == "error":
        return {"status": "failed", "error": f"provider error: {response.content}", "usage": usage}
    if response.finish_reason == "length":
        return {"status": "failed", "error": f"reply truncated at {MAX_TOKENS} tokens", "usage": usage}
    try:
        spec = normalize(extract_json(response.content or ""))
    except (ValueError, TypeError) as exc:
        return {"status": "failed", "error": f"unparseable reply: {exc}", "usage": usage, "reply_head": (response.content or "")[:400]}
    return {"status": "ok", "spec": spec, "usage": usage}


def summary(spec: dict[str, Any]) -> dict[str, Any]:
    """The few fields worth echoing into a phase record."""
    return {
        "python": spec["language"].get("version"),
        "packages": len(spec["python_packages"]),
        "gpu_required": spec["gpu"]["required"],
        "datasets": [d["name"] for d in spec["datasets"]],
        "external_tools": [t["name"] for t in spec["external_tools"]],
        "run_commands": len(spec["run_commands"]),
    }


def write_spec(path: Path, record: dict[str, Any]) -> None:
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


__all__ = ["SPEC_VERSION", "blueprint_sections", "build_prompt", "extract_environment_spec", "extract_json", "normalize", "summary", "write_spec"]
