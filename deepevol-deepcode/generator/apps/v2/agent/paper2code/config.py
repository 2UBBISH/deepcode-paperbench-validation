"""Run configuration, run-directory layout, kernel config, seam installation.

A run is a directory (PLAN.md §2). ``RunConfig`` is what ``run.json``
freezes; ``RunPaths`` names everything under the run directory;
``build_kernel_config`` derives the engine's ``KernelConfig`` from the run;
``install`` binds every seam and installs the ``PaperKernelRuntime`` for
the process. Env defaults follow the validation repo's ``run_trial.sh``
(PLAN.md §6) and never override a variable the operator already set.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from apps.v2.agent.paper2code.tools.registry import ToolContext
from apps.v2.agent_engine.paper2code.seams import harness
from apps.v2.agent_engine.paper2code.seams.config import (
    KernelConfig,
    KernelRuntime,
    set_runtime,
)
from apps.v2.agent_engine.paper2code.seams.llm_runtime import GenerationSettings, LLMProvider

#: the phase model. Owner's call (2026-09-17 evening, PLAN-3 §0 "模型（全部）"): every slot — phases, the figure
#: pass, the experiment agent behind the loopback — runs ``DeepSeek-V4-Flash-Vision-Exp`` until
#: ``DeepSeek-V4.1-Flash`` is granted (both keys answer ``team_model_access_denied`` for it today).
DEFAULT_MODEL = "DeepSeek-V4-Flash-Vision-Exp"
DEFAULT_PROVIDER_BASE_URL = "https://llmapi.paratera.com/v1"
THINKING_MODES = ("disabled", "enabled")
DEFAULT_PROVIDER_KEY_ENV = "PARATERA_API_KEY"
KERNEL_MAX_TOKENS = 32768
KERNEL_TEMPERATURE = 0.1

COMPUTE_KINDS = ("aliyun", "local")
#: CPU tiers by name; GPU tiers (``gpu-economy`` … ``gpu-speed``, PLAN-3 S6) and raw ``ecs.*`` types are accepted too
COMPUTE_TIERS = ("enough", "comfortable")
GPU_COMPUTE_TIERS = ("gpu-economy", "gpu-standard", "gpu-insurance", "gpu-speed")
#: intake's figure pass (``figures.py``): auto = only when the vision probe says the model takes images
FIGURE_MODES = ("auto", "on", "off")
#: the vision model the figure pass calls; a separate slot for input preparation, not a phase model
DEFAULT_FIGURES_MODEL = "DeepSeek-V4-Flash-Vision-Exp"
#: the model the experiment agent (RSA's compiler, SetupX) gets behind the loopback endpoint (step 10)
DEFAULT_EXPERIMENT_MODEL = "DeepSeek-V4-Flash-Vision-Exp"
#: the phase model's context window in tokens; the planner's segment budget derives from it (PLAN-3 item 7b).
#: Paratera's V4-Flash family advertises 1M.
DEFAULT_CONTEXT_WINDOW = 1_000_000

# PLAN.md §6 — values run_trial.sh injected; still overridable from the environment.
ENV_DEFAULTS: dict[str, str] = {
    "DEEPCODE_LLM_RETRY_MODE": "persistent",
    "DEEPCODE_CHAT_RETRY_DELAYS": "10,30,60,180,300",
    "DEEPCODE_PERSISTENT_MAX_DELAY": "900",
    "DEEPCODE_PERSISTENT_IDENTICAL_ERROR_LIMIT": "30",
    "DEEPCODE_OPENAI_REQUEST_TIMEOUT_S": "600",
    "DEEPCODE_PREFILTER_MAX_TOKENS": "32000",
    "DEEPCODE_ANALYSIS_MAX_TOKENS": "16000",
    "DEEPCODE_RELATIONSHIP_MAX_TOKENS": "16000",
    "DEEPCODE_CODE_ANALYZER_TIMEOUT_S": "600",
    "DEEPCODE_REFERENCE_MAX_TOKENS": "32768",
    "DEEPCODE_DOWNLOAD_MAX_TOKENS": "16384",
    "DEEPCODE_REFERENCE_MAX_ITERATIONS": "40",
    "DEEPCODE_DOWNLOAD_MAX_ITERATIONS": "12",
    "DEEPCODE_STALL_THRESHOLD": "7200",
    "DEEPCODE_MAX_WALL_SECONDS": "21600",
    # one implementation call's output budget (VENDOR 12; upstream 8192 truncates a 30 KB write_file and aborts).
    # With thinking on the reasoning shares this budget: fre-t17 spent 19.5k of 32768 thinking and cut a 52 KB
    # write_file mid-string ("invalid JSON arguments" → upstream aborts the whole implementation). 65536 is DeepSeek's cap.
    "DEEPCODE_IMPLEMENT_MAX_TOKENS": "65536",
    # VENDOR 13 (2026-09-19): the blueprint quotes formulas verbatim with Source pointers and the coding agent gets
    # read_paper(section=…); the baseline repository's upstream copy has neither, so this is a line-only switch
    "DEEPCODE_PAPER_FIDELITY": "1",
    # the planner's output budget (ADR 0003 plans are long; with thinking on the reasoning shares it — 09-20 batch: two
    # plans truncated at 32768); DeepSeek's output cap is 65536
    "DEEPCODE_PLANNING_MAX_TOKENS": "65536",
}


def apply_env_defaults(
    denylist: tuple[str, ...] = (), *, context_window: int | None = None, planning_fanout: bool = False
) -> dict[str, str]:
    """``setdefault`` every PLAN §6 knob; ``DEEPCODE_URL_DENYLIST`` is always the run's.

    The two PLAN-3 engine switches (VENDOR 11) are always the run's too: ``DEEPCODE_PLANNER_CONTEXT_WINDOW``
    from ``run.json.context_window`` (7b) and ``DEEPCODE_PLANNING_FANOUT`` from ``run.json.planning_fanout`` (7).
    """
    applied: dict[str, str] = {}
    for key, value in ENV_DEFAULTS.items():
        applied[key] = os.environ.setdefault(key, value)
    os.environ["DEEPCODE_URL_DENYLIST"] = ",".join(denylist)
    if context_window:
        os.environ["DEEPCODE_PLANNER_CONTEXT_WINDOW"] = str(int(context_window))
        applied["DEEPCODE_PLANNER_CONTEXT_WINDOW"] = os.environ["DEEPCODE_PLANNER_CONTEXT_WINDOW"]
    os.environ["DEEPCODE_PLANNING_FANOUT"] = "1" if planning_fanout else "0"
    applied["DEEPCODE_PLANNING_FANOUT"] = os.environ["DEEPCODE_PLANNING_FANOUT"]
    # git must never wait for a username/password on a missing or private repository
    applied["GIT_TERMINAL_PROMPT"] = os.environ.setdefault("GIT_TERMINAL_PROMPT", "0")
    applied["DEEPCODE_URL_DENYLIST"] = os.environ["DEEPCODE_URL_DENYLIST"]
    return applied


# ---------------------------------------------------------------------------
# run.json
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RunConfig:
    run_id: str
    paper_dir: str
    paper_sha256: str
    model: str = DEFAULT_MODEL
    provider_base_url: str = DEFAULT_PROVIDER_BASE_URL
    provider_key_env: str = DEFAULT_PROVIDER_KEY_ENV
    provider_stream: bool = False
    #: "disabled" (the Paratera batches up to 09-18) or "enabled" (the 09-19 deepseek-flash batch: DeepSeek's default,
    #: the desktop arms cannot turn it off); the provider sends thinking:{type:<this>} and guards accordingly
    thinking: str = "disabled"
    compute: str = "aliyun"
    compute_tier: str = "enough"
    run_hours: float = 2.0
    #: repair rounds after the experiment agent's round 0 (PLAN-3 item 5); 0 for a comparison run
    repair_rounds: int = 3
    figures: str = "auto"
    figures_model: str = DEFAULT_FIGURES_MODEL
    #: step 10: what RSA / SetupX call through the loopback (``llm_loopback.py``)
    experiment_model: str = DEFAULT_EXPERIMENT_MODEL
    #: tokens the phase model can take in one request; the engine's planner budget derives from it (7b)
    context_window: int = DEFAULT_CONTEXT_WINDOW
    #: the legacy planning fan-out (Concept + Algorithm analysis before the planner), PLAN-3 item 7; off = upstream
    planning_fanout: bool = False
    denylist: tuple[str, ...] = ()
    skip: tuple[str, ...] = ()
    ask: bool = False
    created_at: float = field(default_factory=time.time)
    engine_commit: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.compute not in COMPUTE_KINDS:
            raise ValueError(f"compute must be one of {COMPUTE_KINDS}, got {self.compute!r}")
        if self.compute_tier not in COMPUTE_TIERS and self.compute_tier not in GPU_COMPUTE_TIERS and not self.compute_tier.startswith("ecs."):
            raise ValueError(f"compute_tier must be one of {COMPUTE_TIERS}, {GPU_COMPUTE_TIERS} or an ecs.* type, got {self.compute_tier!r}")
        if self.run_hours <= 0:
            raise ValueError("run_hours must be positive")
        if int(self.repair_rounds) < 0:
            raise ValueError("repair_rounds must be >= 0")
        if self.figures not in FIGURE_MODES:
            raise ValueError(f"figures must be one of {FIGURE_MODES}, got {self.figures!r}")
        if not self.model.strip():
            raise ValueError("model is required")
        if self.thinking not in THINKING_MODES:
            raise ValueError(f"thinking must be one of {THINKING_MODES}, got {self.thinking!r}")
        if not str(self.experiment_model).strip():
            raise ValueError("experiment_model is required")
        if int(self.context_window) <= 0:
            raise ValueError("context_window must be positive")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["denylist"] = list(self.denylist)
        data["skip"] = list(self.skip)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunConfig":
        payload = dict(data)
        payload["denylist"] = tuple(payload.get("denylist") or ())
        payload["skip"] = tuple(payload.get("skip") or ())
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        unknown = {k: payload.pop(k) for k in list(payload) if k not in known}
        cfg = cls(**payload)
        if unknown:
            cfg.extra.update(unknown)
        return cfg

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "RunConfig":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# run directory
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RunPaths:
    root: Path

    @property
    def run_json(self) -> Path:
        return self.root / "run.json"

    @property
    def status_json(self) -> Path:
        return self.root / "status.json"

    @property
    def events_jsonl(self) -> Path:
        return self.root / "events.jsonl"

    @property
    def input_dir(self) -> Path:
        return self.root / "input"

    @property
    def paper_md(self) -> Path:
        return self.input_dir / "paper.md"

    @property
    def workspace(self) -> Path:
        return self.root / "workspace"

    @property
    def phases_dir(self) -> Path:
        return self.root / "phases"

    @property
    def llm_dir(self) -> Path:
        return self.root / "llm"

    @property
    def jobs_dir(self) -> Path:
        return self.root / "jobs"

    @property
    def lease_json(self) -> Path:
        return self.root / "lease.json"

    @property
    def environment_spec_json(self) -> Path:
        return self.root / "environment_spec.json"

    @property
    def canary_log(self) -> Path:
        return self.root / "canary.log"

    def ensure(self) -> "RunPaths":
        for directory in (self.root, self.input_dir, self.workspace, self.phases_dir, self.llm_dir, self.jobs_dir):
            directory.mkdir(parents=True, exist_ok=True)
        return self


class EventLog:
    """Append-only ``events.jsonl``; one JSON object per line, thread-safe."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def emit(self, kind: str, **fields: Any) -> dict[str, Any]:
        record = {"ts": time.time(), "kind": kind, **fields}
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        return record

    __call__ = emit


# ---------------------------------------------------------------------------
# kernel config + runtime
# ---------------------------------------------------------------------------


def build_kernel_config(run: RunConfig, paths: RunPaths) -> KernelConfig:
    """The engine's six config groups for this run.

    One model for every phase (``planning`` / ``implementation`` carry no
    override, which the preflight gate re-checks), ``max_tokens`` 32768, the
    run workspace as the engine's workspace root, no MCP server table
    (tools are in-process), permission mode ``full_auto``.
    """
    cfg = KernelConfig()
    cfg.workspace.root = str(paths.workspace)
    cfg.agents.defaults.model = run.model
    cfg.agents.defaults.max_tokens = KERNEL_MAX_TOKENS
    cfg.agents.defaults.temperature = KERNEL_TEMPERATURE
    cfg.agents.defaults.provider = "paratera"
    cfg.security.permission_mode = "full_auto"
    cfg.security.sandbox = False
    return cfg


class PaperKernelRuntime(KernelRuntime):
    """One provider instance for the whole run; ``tool_context`` for the Agent seam."""

    def __init__(self, config: KernelConfig, *, provider: LLMProvider, tool_context: ToolContext) -> None:
        super().__init__(config)
        self._provider = provider
        self.tool_context = tool_context

    @property
    def provider(self) -> LLMProvider:
        return self._provider

    def provider_for(
        self,
        *,
        provider_name: str | None = None,
        connection_id: str | None = None,
        phase: str = "default",
        model: str | None = None,
        execution_profile: Any | None = None,
    ) -> LLMProvider:
        settings = self.config.resolve_phase(phase)
        default_model = self._provider.get_default_model()
        wanted = (model or settings.model or default_model).strip()
        if wanted != default_model:
            raise ValueError(
                f"phase {phase!r} asked for model {wanted!r} but this run is pinned to {default_model!r}"
            )
        self._provider.generation = GenerationSettings(
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            reasoning_effort=settings.reasoning_effort,
        )
        return self._provider

    async def aclose(self) -> None:
        close = getattr(self._provider, "aclose", None)
        if close is not None:
            await close()


# ---------------------------------------------------------------------------
# harness seams
# ---------------------------------------------------------------------------


def build_permission_engine(
    security: Any,
    *,
    cwd: str | None = None,
    default_mode: harness.PermissionMode = harness.PermissionMode.FULL_AUTO,
) -> harness.PermissionEngine:
    """Unattended line: every tool call is allowed; the sandbox is the remote container."""
    return harness.FullAutoEngine()


def _approver_call(self: harness.TerminalApprover, tool_name: str, arguments: Any = None, reason: str | None = None) -> bool:
    raise RuntimeError(
        f"paper2code runs unattended; a TerminalApprover was asked about {tool_name!r} ({reason})"
    )


def bind_harness() -> None:
    harness.build_permission_engine = build_permission_engine  # type: ignore[assignment]
    harness.TerminalApprover.__call__ = _approver_call  # type: ignore[method-assign]
    workflow = sys.modules.get("apps.v2.agent_engine.paper2code.workflows.code_implementation_workflow")
    if workflow is not None:
        workflow.build_permission_engine = build_permission_engine  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------


def bind_seams() -> None:
    """Bind all ten seam methods (idempotent)."""
    from apps.v2.agent.paper2code import agent as agent_mod
    from apps.v2.agent.paper2code import runner as runner_mod

    agent_mod.bind()
    runner_mod.bind()
    bind_harness()


def install(
    run: RunConfig,
    paths: RunPaths,
    *,
    provider: LLMProvider | None = None,
    port: Any | None = None,
    events: EventLog | None = None,
) -> PaperKernelRuntime:
    """Env defaults, seams, provider, tool context, runtime — in that order."""
    run.validate()
    apply_env_defaults(run.denylist, context_window=run.context_window, planning_fanout=run.planning_fanout)
    bind_seams()
    if provider is None:
        from apps.v2.agent.paper2code.provider import ParateraProvider

        provider = ParateraProvider.from_run(run, paths, events=events)
    from apps.v2.agent_engine.paper2code.workflows.environment import TASKS_DIRNAME

    ctx = ToolContext(
        workspace=paths.workspace,
        denylist=tuple(run.denylist),
        port=port,
        code_base=paths.workspace / TASKS_DIRNAME / f"paper_{run.run_id}" / "code_base",
    )
    runtime = PaperKernelRuntime(build_kernel_config(run, paths), provider=provider, tool_context=ctx)
    set_runtime(runtime)
    return runtime


__all__ = [
    "COMPUTE_KINDS",
    "COMPUTE_TIERS",
    "DEFAULT_CONTEXT_WINDOW",
    "DEFAULT_EXPERIMENT_MODEL",
    "DEFAULT_FIGURES_MODEL",
    "DEFAULT_MODEL",
    "DEFAULT_PROVIDER_BASE_URL",
    "DEFAULT_PROVIDER_KEY_ENV",
    "ENV_DEFAULTS",
    "FIGURE_MODES",
    "GPU_COMPUTE_TIERS",
    "KERNEL_MAX_TOKENS",
    "EventLog",
    "PaperKernelRuntime",
    "RunConfig",
    "RunPaths",
    "apply_env_defaults",
    "bind_harness",
    "bind_seams",
    "build_kernel_config",
    "build_permission_engine",
    "install",
]
