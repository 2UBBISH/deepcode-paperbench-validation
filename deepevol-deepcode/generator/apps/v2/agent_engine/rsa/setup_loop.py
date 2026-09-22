"""Drive SetupX's ReAct loop, with the verdict taken away from it.

Zero edits to `setupx/src/`. Everything below is either a subclass or a rebinding
of a module-level name resolved at import time -- the mechanism
`harness/local_env.py:445` already uses for the environment manager.

What changes, and why each one is where it is:

* **The in-loop verifier stops judging.** `VerifierAgent` is replaced wholesale by
  `DeterministicVerifier`, so `VERIFY` runs the *frozen criterion* and reports
  what actually failed. The design document asks for the verifier to be demoted
  from verdicts to diagnosis; this goes further and removes the model from the
  verify path entirely, because once the criterion exists there is nothing left
  for a model to judge -- and the measured cost of the model doing it was one
  VERIFY consuming 95 calls and 690,738 prompt tokens.

  Note on the design document's own pointer: it names `verifier_agent.py:5-6` as
  the escape hatch to delete. Those two lines are the module docstring; they never
  reach a model. The behaviour comes from the attribution rules inside
  `SYSTEM_PROMPT` (:68-90). Deleting lines 5-6 would satisfy the instruction and
  change nothing at runtime.

* **The kickback goes into the system prompt.** `LLMEngine.SYSTEM_PROMPT_TEMPLATE`
  is re-rendered on every step (`llm_engine.py:460`), so a message put there
  cannot age out of the 10-entry history window. The two obvious alternatives are
  both wrong: `SETUPX_VERIFIER_HINT` reaches only the verifier sub-agent's first
  message, and `last_error` additionally switches on the XPU retrieval branch
  (`agent.py:217`), costing an extra call per step.

* **A fresh agent object per round.** `run()` closes its LLM client on the way out
  (`agent.py:374`), so a second `run()` on the same object raises. Building a new
  one is also the conversation reset the design document asks for, for free: only
  the ineffective-action ledger crosses a round boundary.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .adjudicator import Adjudicator, PASS, Verdict
from .bridge import DockerBridge
from .criterion import Criterion
from .freezer import FrozenCriterion
from .router import RoundActions

# Set by `bind()`; read by the verifier shim, which SetupX constructs itself and
# therefore cannot be handed arguments.
_CTX: "LoopContext | None" = None

# Installed by the pipeline so the Router has a live token count to enforce its
# budget against; None when the loop is driven directly.
_METER = None

# The SET_ENV values the most recent round established. `docker exec` starts a
# fresh environment, so the Adjudicator must replay these or it measures a
# different environment than the one the agent built -- 19 of the 29 repositories
# that used SET_ENV were misjudged before the harness started forwarding them.
_LAST_ENV: dict = {}


@dataclass
class LoopContext:
    frozen: FrozenCriterion
    workdir: str = "/workspace/repo"
    previews: list[Verdict] = field(default_factory=list)

    @property
    def last_preview(self) -> Verdict | None:
        return self.previews[-1] if self.previews else None


class DeterministicVerifier:
    """Stands in for `src.agent.VerifierAgent`, constructed as `(env, hint=...)`.

    Runs the frozen criterion as a *preview*: no working-tree reset, because the
    agent is still working and resetting mid-round would delete its uncommitted
    progress for no benefit. The authoritative verdict comes later, from the
    Router, with the reset. The gap between the two is itself a signal -- a
    preview that passes and an authoritative run that fails means the pass
    depended on something the reset removed.
    """

    def __init__(self, env, hint: str = ""):
        self._env = env
        self._hint = hint

    def verify(self):
        from .setupx_interop import import_setupx
        _, _, _, models = import_setupx()

        if _CTX is None:
            raise RuntimeError("rsa.setup_loop.bind() was not called")

        cid = getattr(self._env, "container_id", None)
        if not cid:
            return models.VerifyResult(
                success=False, test_framework="rsa", collect_count=0,
                command="(none)", exit_code=1, stdout="",
                stderr="no container to verify", messages=[])

        bridge = getattr(self._env, "backend", None)
        if bridge is None:
            bridge = DockerBridge(cid, workdir=_CTX.workdir)
        adj = Adjudicator(bridge, workdir=_CTX.workdir)
        v = adj.adjudicate(_CTX.frozen, agent_env=_agent_env(self._env), reset=False)
        _CTX.previews.append(v)

        return models.VerifyResult(
            success=v.verdict == PASS,
            test_framework="rsa-frozen-criterion",
            collect_count=v.expected_n,
            command=_CTX.frozen.criterion.command,
            exit_code=v.exit_code,
            stdout=_preview_report(v),
            stderr="",
            # Kept unconditionally. The stock loop stores the verifier transcript
            # only when verification passed (agent.py:767) and discards it on
            # failure (:783) -- which is the only path a diagnosis-only verifier
            # ever takes, so the diagnosis would never survive the round.
            messages=[{"role": "rsa", "content": _preview_report(v)}],
        )


def _agent_env(env) -> dict[str, str]:
    """Whatever the agent established with SET_ENV.

    `docker exec` starts a fresh environment, so a criterion run without these
    measures a different environment than the one the agent built -- 29 of 100
    repositories used SET_ENV and 19 of those were misjudged before the harness
    started forwarding them. Read straight off the manager rather than scraped out
    of a log, which is what `harness/run_real.py:75` has to do from outside.
    """
    return dict(getattr(env, "_env_vars", {}) or {})


def _preview_report(v: Verdict) -> str:
    if v.verdict == PASS:
        return (f"[FROZEN CRITERION] all {v.expected_n} assertions pass. "
                "The external ruler will confirm this after restoring the working tree.")
    lines = [
        f"[FROZEN CRITERION] {v.passed_expected}/{v.expected_n} assertions pass. "
        f"Still failing:",
    ]
    lines += [f"  {t}" for t in v.missing[:25]]
    tails = v.failure_tails
    for t in v.missing[:5]:
        name = t.split("::")[-1]
        if name in tails:
            lines += ["", f"--- {name} ---", tails[name][-700:]]
    if v.reason:
        lines += ["", f"note: {v.reason}"]
    return "\n".join(lines)


# --------------------------------------------------------------------------

# `{` and `}` in a kickback would be eaten by `SYSTEM_PROMPT_TEMPLATE.format()`
# at llm_engine.py:460 -- and error tails are full of dicts, f-strings and regex
# quantifiers. Escaping is not optional.
def _escape_braces(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}")


def bind(ctx: LoopContext) -> None:
    """Rebind SetupX's verifier and record the criterion the loop is working toward.

    Rebinding `src.agent.VerifierAgent` rather than `src.verifier_agent.VerifierAgent`
    is deliberate and is the whole subtlety of this technique: `agent.py:37` did
    `from .verifier_agent import VerifierAgent`, so that name was resolved at
    import time and patching the defining module alone would change nothing.
    """
    global _CTX
    _CTX = ctx
    agent_mod, _, _, _ = _modules()
    agent_mod.VerifierAgent = DeterministicVerifier


def _modules():
    from .setupx_interop import import_setupx
    return import_setupx()


def make_agent(repo_url: str, *, revision: str, max_steps: int,
               contract: str, kickback_text: str = "",
               container_id: str | None = None, is_final_round: bool = False):
    """One agent, for one round."""
    agent_mod, _, llm_mod, _ = _modules()

    class ResearchSetupAgent(agent_mod.SpeculativeSetupAgent):
        def _handle_finish(self, action):
            """FINISH means "I am done"; it does not mean "I passed".

            The stock gate (agent.py:805) only accepts FINISH after a passing
            VERIFY and otherwise coaches the model to "run VERIFY and let the
            Verifier record that judgement" -- advice that made sense when the
            verifier could be talked into a pass, and that is now simply a wasted
            step. Accepting the hand-off costs nothing, because the Router
            adjudicates afterwards either way.
            """
            self._state.completed = True
            self._state.final_message = action.message or "agent stopped voluntarily"
            self._agent_stopped_voluntarily = True
            self._state.add_to_history({
                "action": action.to_dict(),
                "result": {"exit_code": 0, "stderr": "",
                           "stdout": "[FINISH] handed to the external ruler"},
            })
            return True

        def _cleanup_snapshots_safely(self) -> None:
            # Snapshot images are the rollback stack. Dropping them at the end of
            # every round would remove the Router's ability to rebuild from the
            # initial checkout, which is its last move before escalating.
            if is_final_round:
                super()._cleanup_snapshots_safely()

    agent = ResearchSetupAgent(repo_url, max_steps=max_steps, revision=revision)
    agent._agent_stopped_voluntarily = False

    # Meter this round's client, if the pipeline installed a meter. Every agent's
    # LLM traffic funnels through this one object (llm_engine.py:133), and a fresh
    # agent per round means a fresh client per round -- so it is attached here
    # rather than once at startup.
    if _METER is not None:
        from .meter import attach as attach_meter
        attach_meter(agent._llm._client, _METER)

    if container_id:
        # Round 2+: keep the environment the previous round built. `attach` sets
        # the manager's container without creating or cloning; neutering
        # `create_container` stops `run()` (agent.py:165) from making a new one.
        agent._env.attach(container_id, repo_dir="/workspace/repo")
        agent._env.create_container = lambda *a, **k: container_id

    # Instance attribute shadows the class attribute, so one round's kickback
    # cannot leak into another run in the same process.
    block = _escape_braces(contract + ("\n\n" + kickback_text if kickback_text else ""))
    agent._llm.SYSTEM_PROMPT_TEMPLATE = (
        llm_mod.LLMEngine.SYSTEM_PROMPT_TEMPLATE
        + "\n\n# EXTERNAL GRADING CONTRACT AND FEEDBACK\n\n"
        + block
        + "\n\nThis block is authoritative and is re-stated every step. Your own "
          "sense that the environment is ready does not decide anything: an "
          "external, frozen criterion does.\n"
    )
    return agent


_SHELL = re.compile(r"^\[?(SHELL_COMMAND|SET_ENV|TRY_XPU_SUGGESTION|ROLLBACK_ENV)\]?", re.I)


def summarise_actions(history: list[dict], limit: int = 40) -> list[str]:
    """One readable line per action, for the ledger and the escalation card."""
    out: list[str] = []
    for h in history:
        a = h.get("action") or {}
        kind = str(a.get("action_type") or a.get("type") or "").upper()
        detail = (a.get("command") or a.get("message") or
                  (f"{a.get('key')}={a.get('value')}" if a.get("key") else "") or "")
        line = f"{kind}: {detail}".strip().rstrip(":")
        if line and line not in out:
            out.append(line[:300])
    return out[-limit:]


def run_round(repo_url: str, *, revision: str, max_steps: int, contract: str,
              kickback_text: str, container_id: str | None,
              is_final_round: bool = False) -> tuple[RoundActions, str]:
    """Run one pass of the setup loop. Returns (what it did, the container id)."""
    global _LAST_ENV
    agent = make_agent(repo_url, revision=revision, max_steps=max_steps,
                       contract=contract, kickback_text=kickback_text,
                       container_id=container_id, is_final_round=is_final_round)
    try:
        result = agent.run()
    except Exception as e:
        _LAST_ENV = _agent_env(agent._env)
        cid = getattr(agent._env, "container_id", None) or (container_id or "")
        return RoundActions(actions=summarise_actions(agent._state.history),
                            error=f"{type(e).__name__}: {e}"), cid

    _LAST_ENV = _agent_env(agent._env)
    actions = summarise_actions(result.history)
    requested = any("help" in (a.get("action") or {}).get("message", "").lower()
                    for a in result.history if isinstance(a, dict))
    return (RoundActions(actions=actions,
                         stopped_voluntarily=bool(getattr(agent, "_agent_stopped_voluntarily", False)),
                         requested_help=requested),
            result.container_id or container_id or "")
