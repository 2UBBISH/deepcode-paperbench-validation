"""PLAN-3 item 2: the environment spec is read from the blueprint, never guessed."""

from __future__ import annotations

import asyncio

import pytest

from apps.v2.agent.paper2code.environment_spec import (
    blueprint_sections,
    build_prompt,
    extract_environment_spec,
    extract_json,
    normalize,
    summary,
)
from apps.v2.agent_engine.paper2code.seams.llm_runtime import LLMResponse

PLAN = """```yaml
complete_reproduction_plan:
  file_structure: |
    proj/
    └── main.py
  environment_setup: |
    - Language: Python 3.8+ (PyTorch).
    - Core deps: torch, numpy, isaacgym (GPU physics) or mujoco.
    - Hardware: GPU required. IsaacGym requires NVIDIA GPU.
  validation_approach: |
    - run python main.py --task ShadowHand
  implementation_strategy: |
    Networks first.
```
"""


def test_sections_are_cut_at_the_next_top_level_key() -> None:
    sections = blueprint_sections(PLAN)
    assert set(sections) == {"environment_setup", "validation_approach"}
    assert sections["environment_setup"].startswith("|")
    assert "Networks first" not in sections["validation_approach"]
    prompt = build_prompt(PLAN)
    assert "## environment_setup" in prompt
    assert "## validation_approach" in prompt
    assert "implementation_strategy" not in prompt.split("Blueprint sections:")[1]


def test_extract_json_accepts_fenced_and_bare_objects() -> None:
    assert extract_json('here\n```json\n{"a": 1}\n```')["a"] == 1
    assert extract_json('noise {"a": {"b": 2}} tail')["a"]["b"] == 2
    with pytest.raises(ValueError, match="no JSON object"):
        extract_json("no object here")
    with pytest.raises(ValueError, match="no JSON object"):
        extract_json("[1, 2]")  # an array is not a spec


def test_normalize_keeps_only_the_fixed_shape_and_nulls_the_unknown() -> None:
    spec = normalize({
        "language": {"name": "Python", "version": ""},
        "python_packages": ["torch", {"name": " numpy ", "spec": ">=1.21"}, {"spec": "x"}, 7],
        "gpu": {"required": "yes", "reason": "GPU required"},
        "cuda": 11.8,
        "datasets": [{"name": "D4RL", "source": "provided", "size_gb": "big"}],
        "external_tools": [{"name": "IsaacGym", "installable": "no", "reason": "NVIDIA licence"}],
        "run_commands": ["python main.py", ""],
        "notes": None,
    })
    assert spec["version"] == 1
    assert spec["language"] == {"name": "Python", "version": None}
    assert [p["name"] for p in spec["python_packages"]] == ["torch", "numpy"]
    assert spec["python_packages"][1]["spec"] == ">=1.21"
    assert spec["gpu"] == {"required": None, "reason": "GPU required"}  # "yes" is not a value we accept
    assert spec["cuda"] is None  # a number is not a version string
    assert spec["datasets"] == [{"name": "D4RL", "source": "provided", "size_gb": None, "required": True}]
    assert spec["external_tools"] == [{"name": "IsaacGym", "installable": None, "reason": "NVIDIA licence"}]
    assert spec["run_commands"] == ["python main.py"]
    assert summary(spec)["external_tools"] == ["IsaacGym"]


class _Provider:
    def __init__(self, response: LLMResponse) -> None:
        self.response = response
        self.prompts: list[str] = []

    async def chat_with_retry(self, messages, **kwargs):
        self.prompts.append(messages[-1]["content"])
        return self.response


def test_extract_records_failures_instead_of_raising() -> None:
    ok = asyncio.run(extract_environment_spec(_Provider(LLMResponse(content='{"gpu": {"required": true}}', usage={"reasoning_tokens": 0})), PLAN))
    assert ok["status"] == "ok"
    assert ok["spec"]["gpu"]["required"] is True
    truncated = asyncio.run(extract_environment_spec(_Provider(LLMResponse(content='{"gpu"', finish_reason="length")), PLAN))
    assert truncated["status"] == "failed"
    assert "truncated" in truncated["error"]
    error = asyncio.run(extract_environment_spec(_Provider(LLMResponse(content="boom", finish_reason="error")), PLAN))
    assert error["status"] == "failed"
    assert "provider error" in error["error"]
    garbage = asyncio.run(extract_environment_spec(_Provider(LLMResponse(content="I cannot help with that")), PLAN))
    assert garbage["status"] == "failed"
    assert "unparseable" in garbage["error"]
