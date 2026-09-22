"""Import and configure SetupX without editing a line of it.

The existing project doctrine is that the harness drives SetupX through seams
rather than a fork, so upstream pulls stay clean. The design document asks for two
behavioural changes inside the loop (a diagnosis-only verifier, and a kickback
that reaches the agent). Both turn out to be reachable the same way
`harness/local_env.py:445` already reaches the environment manager: rebind a
module-level name that was resolved at import time.

Every binding this package relies on, verified against the source rather than
assumed:

    src.agent.VerifierAgent            agent.py:37   constructed at :732
    LLMEngine.SYSTEM_PROMPT_TEMPLATE   llm_engine.py:178, re-rendered at :460
    SpeculativeSetupAgent._handle_*    agent.py:334 / :361, dispatched via self.

Configuration is the fiddly part and gets its own function, because two separate
traps live there:

* `src/config.py:17` loads `.env` then `.env.local` **at import time**, both with
  `override=True`. Setting `os.environ` before importing does nothing -- `.env`
  overwrites it. The only reliable lever is `.env.local`, which is loaded last.
* A crashed run leaves `.env.local` behind pointing at a dead endpoint. One is
  sitting in the tree right now aimed at `http://127.0.0.1:39907/v1`. Anything
  that imports SetupX without clearing it talks to nothing, silently.
"""

from __future__ import annotations

import os
import shutil
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

DEFAULT_BASE_IMAGE = "setupx-base:py310-proxy"


def setupx_root() -> Path:
    """Where SetupX lives. `RSA_SETUPX_ROOT` wins; otherwise the sibling checkout."""
    env = os.environ.get("RSA_SETUPX_ROOT")
    if env:
        p = Path(env).expanduser().resolve()
    else:
        # DeepEvol: the vendored SetupX checkout is a sibling package of rsa
        # (apps/v2/agent_engine/setupx), not a sibling of the rsa *repository*.
        p = Path(__file__).resolve().parent.parent / "setupx"
    if not (p / "src" / "agent.py").exists():
        raise RuntimeError(
            f"SetupX not found at {p}. Set RSA_SETUPX_ROOT to the checkout."
        )
    return p


@dataclass
class SetupXSession:
    root: Path
    backend: str
    model: str
    base_url: str
    cleared_stale_env_local: bool
    remote_backend: object | None = None


# Re-entrancy guard. Nesting matters in practice: the pipeline configures once for
# a whole run and then opens throwaway containers inside it, each of which wants
# the same configuration. Without this, the inner block's exit deletes the outer
# block's `.env.local` and everything after it silently reads a different config.
_ACTIVE: SetupXSession | None = None


@contextmanager
def setupx_configured(backend: str = "small", *, base_image: str = DEFAULT_BASE_IMAGE,
                      keep_container: bool = True, extra: dict[str, str] | None = None,
                      remote_backend: object | None = None):
    """Configure and import SetupX for the duration of the block.

    `backend` selects `.env.small` (gemma, free) or `.env.large` (deepseek, billed
    per token) per the project's CLAUDE.md. It is an explicit argument rather than
    something inherited from whatever `.env` happens to say, so a run's record
    always states which model it used.

    Re-entrant: a nested call with the same backend yields the active session and
    leaves teardown to the outermost block.
    """
    global _ACTIVE
    if _ACTIVE is not None:
        if _ACTIVE.backend != backend:
            raise RuntimeError(
                f"already configured for backend {_ACTIVE.backend!r}; a nested "
                f"switch to {backend!r} would change the model mid-run without "
                "the record saying so")
        if remote_backend is not None and _ACTIVE.remote_backend is not remote_backend:
            raise RuntimeError("nested SetupX configuration cannot switch remote backend")
        yield _ACTIVE
        return

    root = setupx_root()
    src = root / f".env.{backend}"
    if not src.exists():
        raise RuntimeError(f"no backend config at {src}")

    env_local = root / ".env.local"
    cleared = env_local.exists()
    backup = env_local.with_suffix(".local.rsa-backup") if cleared else None
    if cleared:
        shutil.move(str(env_local), str(backup))

    lines = [
        "# Written by rsa.setupx_interop; removed on exit.",
        src.read_text(encoding="utf-8"),
        "",
        "# XPU off: the user's brief, and experiment 07 measured it net-zero on this",
        "# workload. The gate is XPU_DISABLED, not XPU_ENABLED (config.py:112-113).",
        "XPU_DISABLED=true",
        f"DOCKER_BASE_IMAGE={base_image}",
    ]
    for k, v in (extra or {}).items():
        lines.append(f"{k}={v}")
    env_local.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    # Two steps, and skipping the first silently does nothing: `_load_env_files`
    # already ran at import, so clearing the memo alone re-reads a stale os.environ.
    import src.config as cfg  # noqa: E402
    cfg._load_env_files()
    cfg._config = None
    conf = cfg.get_config()

    if keep_container:
        # Without this SetupX destroys the container in its own teardown, and the
        # Adjudicator has nothing left to exec into (main.py:315).
        os.environ["OURSYS_KEEP_CONTAINER"] = "1"
    os.environ["XPU_DISABLED"] = "true"
    for key in ("dns", "XPU_DB_DNS", "XPU_VECTOR_ENABLED", "XPU_ENABLED"):
        os.environ.pop(key, None)

    session = SetupXSession(
        root=root, backend=backend,
        model=conf.openai.model if conf.openai else "?",
        base_url=conf.openai.base_url if conf.openai else "?",
        cleared_stale_env_local=cleared,
        remote_backend=remote_backend,
    )
    _ACTIVE = session
    original_env_manager = original_em_manager = None
    if remote_backend is not None:
        # SetupX's agent imported EnvironmentManager into its own module namespace
        # at import time. Rebinding only src.environment_manager would be inert;
        # bind the live name in src.agent, exactly as the verifier seam does.
        from .remote_backend import RemoteEnvironmentManager, bind_remote_backend
        bind_remote_backend(remote_backend)
        agent_mod, em_mod, _, _ = import_setupx()
        original_env_manager = getattr(agent_mod, "EnvironmentManager", None)
        original_em_manager = getattr(em_mod, "EnvironmentManager", None)
        agent_mod.EnvironmentManager = RemoteEnvironmentManager
        em_mod.EnvironmentManager = RemoteEnvironmentManager
    try:
        yield session
    finally:
        _ACTIVE = None
        if remote_backend is not None:
            from .remote_backend import bind_remote_backend
            bind_remote_backend(None)
            if original_env_manager is not None:
                agent_mod.EnvironmentManager = original_env_manager
            if original_em_manager is not None:
                em_mod.EnvironmentManager = original_em_manager
        env_local.unlink(missing_ok=True)
        if backup and backup.exists():
            # The stale file is deliberately NOT restored -- it is what poisons the
            # next run. Keep it aside for forensics instead.
            backup.replace(root / ".env.local.stale")


def import_setupx():
    """Return the SetupX modules this package binds against.

    Import is deferred to call time rather than module scope so that the
    deterministic core (freezer, adjudicator, linter) keeps working on a machine
    where SetupX is absent or its dependencies are not installed.
    """
    root = setupx_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import src.agent as agent_mod
    import src.environment_manager as em_mod
    import src.llm_engine as llm_mod
    import src.models as models_mod
    return agent_mod, em_mod, llm_mod, models_mod
