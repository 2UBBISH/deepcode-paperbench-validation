"""NetHack per-level (FAR-state) evaluation.

Implements the per-level evaluation protocol of Wołczyk et al. (2024),
Section 5 / Appendix B.1 ("Evaluation"):

    "To perform the per-level evaluation in Figure 5, we employ the AutoAscend
     expert, used for behavioral cloning in pre-training. We use AutoAscend to
     play the game and save the state when it reaches the desired level. We
     generate 200 game saves for each level and evaluate our agents on each
     save by loading the game, running our agent where the expert finished, and
     reporting the score our agent achieved on top of the expert's score.
     Models were evaluated every 25 million environment steps for Figure 5."

Two target levels represent FAR states:

* ``level_4``  -- dungeon level 4 (main branch), reachable early (CLOSE-ish but
  far from the pre-training distribution of ``pi_*``).
* ``sokoban``  -- the Sokoban branch (the first Sokoban level).  Solving it does
  not yield immediate rewards ("the number of filled pits for Sokoban levels"),
  which is exactly why vanilla fine-tuning forgets it.

The module is deliberately dependency-light: ``nle``, ``gym``, ``torch`` and
``matplotlib`` are imported lazily/optionally.  When AutoAscend (or NLE) is not
available, a deterministic *stub* save generator/evaluator keeps the whole
pipeline (and the train/eval drivers) runnable on CPU for smoke tests.

Artifacts written to disk follow the layout used by the other analysis modules::

    <output_dir>/<method>/seed_<seed>/per_level_eval.json
    <save_dir>/<level>/manifest.json
    <save_dir>/<level>/save_<i>.json        (stub saves)
    <save_dir>/<level>/save_<i>.ttyrec|.sav (real AutoAscend saves)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    # constants
    "LEVEL_4",
    "SOKOBAN",
    "PER_LEVEL_TARGETS",
    "DEFAULT_NUM_SAVES",
    "DEFAULT_EVAL_EPISODES",
    "EVAL_EVERY",
    "LEVEL_SPECS",
    "AUTOASCEND_REPO_URL",
    "AUTOASCEND_BRANCH",
    # level metadata
    "LevelSpec",
    "LevelSave",
    "resolve_level",
    "level_display_name",
    # save generation
    "autoascend_path",
    "autoascend_available",
    "AutoAscendSaveGenerator",
    "generate_saves",
    "list_saves",
    "describe_saves",
    # evaluation
    "evaluate_on_save",
    "evaluate_level",
    "evaluate_per_level",
    "run_per_level_eval",
    "aggregate_level_results",
    "PerLevelEvaluator",
    # reporting
    "z_for",
    "summarize",
    "format_table",
    "plot_per_level_curves",
    "build_parser",
    "main",
]

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

LEVEL_4 = "level_4"
SOKOBAN = "sokoban"
PER_LEVEL_TARGETS: Tuple[str, ...] = (LEVEL_4, SOKOBAN)

#: 200 game saves per level (Appendix B.1).
DEFAULT_NUM_SAVES = 200
#: Figure 5 is "averaged over 200 episodes, each starting from where the expert
#: (AutoAscend) ended up upon first entering level".
DEFAULT_EVAL_EPISODES = 200
#: "Models were evaluated every 25 million environment steps for Figure 5."
EVAL_EVERY = 25_000_000

DEFAULT_MAX_STEPS = 100_000
DEFAULT_NO_PROGRESS_STEPS = 150
DEFAULT_CONFIDENCE = 0.90

AUTOASCEND_REPO_URL = "https://github.com/cdmatters/autoascend"
#: The paper uses the ``jt-nld`` branch of AutoAscend (see addendum).
AUTOASCEND_BRANCH = "jt-nld"
AUTOASCEND_ENV_VAR = "AUTOASCEND_PATH"

SAVE_MANIFEST = "manifest.json"

#: Score keys, in priority order, for reading the agent's in-game score.
SCORE_KEYS: Tuple[str, ...] = ("score", "agent_score", "in_game_score")
#: Sokoban "filled pits" keys (the paper reports the number of filled pits).
FILLED_PITS_KEYS: Tuple[str, ...] = (
    "sokoban_filled_pits",
    "filled_pits",
    "pits_filled",
    "sokoban_pits_filled",
)
#: Progress keys used by the no-progress stopping rule.
PROGRESS_KEYS: Tuple[str, ...] = ("score", "dlvl", "xplvl", "depth")


@dataclass(frozen=True)
class LevelSpec:
    """Description of one evaluation target level."""

    name: str
    dlvl: Optional[int] = None
    branch: str = "main"
    #: Token passed to AutoAscend so that it saves a state on that level.
    autoascend_target: str = ""
    description: str = ""

    @property
    def is_sokoban(self) -> bool:
        return self.branch == "sokoban" or self.name == SOKOBAN

    @property
    def display(self) -> str:
        return level_display_name(self.name)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "dlvl": self.dlvl,
            "branch": self.branch,
            "autoascend_target": self.autoascend_target,
            "description": self.description,
            "is_sokoban": self.is_sokoban,
        }


LEVEL_SPECS: Dict[str, LevelSpec] = {
    LEVEL_4: LevelSpec(
        name=LEVEL_4,
        dlvl=4,
        branch="main",
        autoascend_target="4",
        description="Dungeon level 4 of the main branch (FAR state for pi_*).",
    ),
    SOKOBAN: LevelSpec(
        name=SOKOBAN,
        dlvl=None,
        branch="sokoban",
        autoascend_target="sokoban",
        description="First Sokoban level; solving it yields no immediate reward.",
    ),
}

LEVEL_ALIASES: Dict[str, str] = {
    "level4": LEVEL_4,
    "level-4": LEVEL_4,
    "dlvl4": LEVEL_4,
    "dlvl_4": LEVEL_4,
    "l4": LEVEL_4,
    "soko": SOKOBAN,
    "sokoban_1": SOKOBAN,
    "sokoban1": SOKOBAN,
}


def resolve_level(name: Any) -> LevelSpec:
    """Resolve a level name/pointer into a :class:`LevelSpec`."""
    if isinstance(name, LevelSpec):
        return name
    key = str(name).strip().lower().replace(" ", "_")
    key = LEVEL_ALIASES.get(key, key)
    if key in LEVEL_SPECS:
        return LEVEL_SPECS[key]
    # allow bare integers interpreted as dungeon levels
    try:
        dlvl = int(key)
    except (TypeError, ValueError):
        raise KeyError(f"unknown per-level evaluation target: {name!r}")
    return LevelSpec(
        name=LEVEL_4 if dlvl == 4 else f"level_{dlvl}",
        dlvl=dlvl,
        branch="main",
        autoascend_target=str(dlvl),
        description=f"Dungeon level {dlvl} of the main branch.",
    )


def level_display_name(name: Any) -> str:
    spec = resolve_level(name) if not isinstance(name, LevelSpec) else name
    if spec.is_sokoban:
        return "Sokoban"
    if spec.dlvl is not None:
        return f"level {spec.dlvl}"
    return spec.name.replace("_", " ")


# --------------------------------------------------------------------------- #
# Small statistics helpers (SciPy-free; the paper reports 90% CIs)
# --------------------------------------------------------------------------- #

_Z_TABLE: Dict[float, float] = {
    0.5: 0.6745,
    0.68: 0.9945,
    0.8: 1.2816,
    0.9: 1.6449,
    0.95: 1.9600,
    0.98: 2.3263,
    0.99: 2.5758,
}


def z_for(confidence: float = DEFAULT_CONFIDENCE) -> float:
    """Two-sided normal quantile for a confidence level (Acklam fallback)."""
    for level, value in _Z_TABLE.items():
        if abs(confidence - level) < 1e-9:
            return value
    p = 1.0 - (1.0 - float(confidence)) / 2.0
    if _np is None:
        return _Z_TABLE[0.9]
    # rational approximation of the inverse normal CDF
    a = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
    b = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00)
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def summarize(values: Sequence[float], confidence: float = DEFAULT_CONFIDENCE) -> Dict[str, float]:
    """Mean / std / half-width / min / max / median of a sample."""
    vals = [float(v) for v in values if v is not None and not _is_nan(v)]
    n = len(vals)
    if n == 0:
        return {"mean": float("nan"), "std": float("nan"), "half_width": float("nan"),
                "min": float("nan"), "max": float("nan"), "median": float("nan"), "n": 0}
    mean = sum(vals) / n
    if n > 1:
        var = sum((v - mean) ** 2 for v in vals) / (n - 1)
        std = math.sqrt(max(var, 0.0))
    else:
        std = 0.0
    ordered = sorted(vals)
    mid = n // 2
    median = ordered[mid] if n % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])
    half = z_for(confidence) * std / math.sqrt(n) if n > 0 else float("nan")
    return {
        "mean": mean,
        "std": std,
        "half_width": half,
        "min": ordered[0],
        "max": ordered[-1],
        "median": median,
        "n": n,
        "confidence": float(confidence),
    }


def _is_nan(value: Any) -> bool:
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return True


def _as_float(value: Any) -> float:
    try:
        if hasattr(value, "item"):
            value = value.item()
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


# --------------------------------------------------------------------------- #
# Optional dependencies
# --------------------------------------------------------------------------- #

try:  # pragma: no cover - optional
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover
    _np = None  # type: ignore


def _import_module(dotted: str) -> Optional[Any]:
    try:
        import importlib

        return importlib.import_module(dotted)
    except Exception:
        return None


def _torch():
    return _import_module("torch")


def _nethack_env_module() -> Optional[Any]:
    for name in ("src.nethack.env", "finetuning_rl_as_cl.src.nethack.env", ".env"):
        if name == ".env":
            try:
                from . import env as mod  # type: ignore

                return mod
            except Exception:
                continue
        mod = _import_module(name)
        if mod is not None and hasattr(mod, "make_env"):
            return mod
    return None


# --------------------------------------------------------------------------- #
# Saves
# --------------------------------------------------------------------------- #

@dataclass
class LevelSave:
    """One AutoAscend game save at a target level."""

    level: str
    index: int
    path: str
    seed: Optional[int] = None
    expert_score: float = 0.0
    expert_turns: Optional[float] = None
    expert_dlvl: Optional[float] = None
    stub: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def spec(self) -> LevelSpec:
        return resolve_level(self.level)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "index": self.index,
            "path": self.path,
            "seed": self.seed,
            "expert_score": self.expert_score,
            "expert_turns": self.expert_turns,
            "expert_dlvl": self.expert_dlvl,
            "stub": self.stub,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "LevelSave":
        known = {"level", "index", "path", "seed", "expert_score", "expert_turns",
                 "expert_dlvl", "stub", "metadata"}
        return cls(
            level=str(payload.get("level", LEVEL_4)),
            index=int(payload.get("index", 0)),
            path=str(payload.get("path", "")),
            seed=payload.get("seed"),
            expert_score=_as_float(payload.get("expert_score", 0.0)),
            expert_turns=payload.get("expert_turns"),
            expert_dlvl=payload.get("expert_dlvl"),
            stub=bool(payload.get("stub", False)),
            metadata={k: v for k, v in payload.items() if k not in known},
        )


def autoascend_path(explicit: Optional[str] = None) -> Optional[str]:
    """Locate a local AutoAscend checkout (``AUTOASCEND_PATH`` or ``third_party``)."""
    candidates: List[str] = []
    if explicit:
        candidates.append(explicit)
    env = os.environ.get(AUTOASCEND_ENV_VAR)
    if env:
        candidates.append(env)
    here = os.path.dirname(os.path.abspath(__file__))
    for up in (here, os.path.dirname(here), os.path.dirname(os.path.dirname(here))):
        candidates.append(os.path.join(up, "third_party", "autoascend"))
        candidates.append(os.path.join(up, "autoascend"))
    for cand in candidates:
        if cand and os.path.isdir(cand) and _looks_like_autoascend(cand):
            return os.path.abspath(cand)
    return None


def _looks_like_autoascend(path: str) -> bool:
    markers = ("autoascend", "src", "setup.py", "README.md", "scripts")
    return any(os.path.exists(os.path.join(path, m)) for m in markers)


def autoascend_available(explicit: Optional[str] = None) -> bool:
    """``True`` when a usable AutoAscend checkout can be found."""
    return autoascend_path(explicit) is not None


# The helper script executed *inside* the AutoAscend environment.  It drives the
# expert through NLE and writes a save + manifest entry each time the requested
# level is reached, exactly as described in Appendix B.1.
_HELPER_SOURCE = '''\
"""AutoAscend save generator (executed inside the AutoAscend environment)."""
import argparse, json, os, sys


def _load_agent(env):
    """Locate AutoAscend's agent class across repository layouts."""
    candidates = [
        ("autoascend.agent", "Agent"),
        ("autoascend.agent", "AutoAscendAgent"),
        ("autoascend", "Agent"),
        ("agent", "Agent"),
    ]
    for module_name, attr in candidates:
        try:
            mod = __import__(module_name, fromlist=[attr])
        except Exception:
            continue
        cls = getattr(mod, attr, None)
        if cls is None:
            continue
        try:
            return cls(env)
        except TypeError:
            try:
                return cls()
            except Exception:
                continue
    return None


def _make_env(seed):
    try:
        import nle, gym  # noqa: F401
        env = gym.make("NetHackChallenge-v0")
        try:
            env.seed(seed)
        except Exception:
            pass
        return env
    except Exception:
        return None


def _level_reached(spec, info, obs):
    if spec["branch"] == "sokoban":
        for key in ("sokoban_level", "branch", "is_sokoban", "dlvl"):
            val = info.get(key)
            if key == "branch" and val in ("sokoban", "Sokoban"):
                return True
            if key == "is_sokoban" and val:
                return True
        marker = str(info.get("level_name", "")).lower()
        return "sokoban" in marker or "soko" in marker
    target = spec.get("dlvl")
    if target is None:
        return False
    for key in ("dlvl", "depth", "max_dlvl", "level"):
        if key in info and info[key] is not None:
            try:
                return int(info[key]) >= int(target)
            except Exception:
                continue
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=False)
    ap.add_argument("--level", required=True)
    ap.add_argument("--spec", required=True, help="JSON-encoded level spec")
    ap.add_argument("--num-saves", type=int, default=200)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=1000000)
    ap.add_argument("--max-episodes", type=int, default=20000)
    ap.add_argument("--save-format", default="ttyrec")
    args = ap.parse_args()

    spec = json.loads(args.spec)
    os.makedirs(args.out_dir, exist_ok=True)
    manifest_path = os.path.join(args.out_dir, "manifest.json")
    entries = []
    if os.path.exists(manifest_path):
        try:
            entries = json.load(open(manifest_path)).get("saves", [])
        except Exception:
            entries = []

    env = _make_env(args.seed)
    if env is None:
        print(json.dumps({"ok": False, "error": "NLE unavailable"}))
        return 2
    agent = _load_agent(env)
    if agent is None:
        print(json.dumps({"ok": False, "error": "AutoAscend agent not importable"}))
        return 3

    index = len(entries)
    episode = 0
    while index < args.num_saves and episode < args.max_episodes:
        try:
            obs = env.reset()
        except TypeError:
            obs = env.reset()[0]
        if hasattr(agent, "reset"):
            try:
                agent.reset()
            except Exception:
                pass
        done = False
        steps = 0
        while not done and steps < args.max_steps:
            try:
                action = agent.step(obs) if hasattr(agent, "step") else agent(obs)
            except Exception:
                action = 0
            out = env.step(action)
            obs, reward, done, info = out if len(out) == 4 else (out[0], out[1], out[2] or out[3], out[4])
            steps += 1
            if _level_reached(spec, info or {}, obs):
                path = os.path.join(args.out_dir, "save_%04d.%s" % (index, args.save_format))
                saved = False
                for attr in ("save", "save_state", "save_game"):
                    fn = getattr(env, attr, None)
                    if callable(fn):
                        try:
                            fn(path)
                            saved = True
                            break
                        except Exception:
                            continue
                entry = {
                    "level": args.level,
                    "index": index,
                    "path": path,
                    "seed": int(info.get("seed", args.seed)) if isinstance(info, dict) else args.seed,
                    "expert_score": float((info or {}).get("score", 0.0)),
                    "expert_turns": (info or {}).get("turns"),
                    "expert_dlvl": (info or {}).get("dlvl"),
                    "saved": saved,
                }
                entries.append(entry)
                index += 1
                json.dump({"saves": entries}, open(manifest_path, "w"), indent=2)
                break
        episode += 1
    json.dump({"saves": entries, "ok": True}, open(manifest_path, "w"), indent=2)
    print(json.dumps({"ok": True, "num_saves": len(entries), "manifest": manifest_path}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


class AutoAscendSaveGenerator:
    """Generate ``num_saves`` game saves per target level using AutoAscend.

    Two operating modes are supported:

    ``command``
        A user-supplied command template (``{repo}``, ``{level}``, ``{out_dir}``,
        ``{seed}`` are substituted) that produces ``manifest.json`` in
        ``out_dir``.  This mirrors the ad-hoc scripts used for the paper.
    ``helper`` (default)
        A generated helper script that imports AutoAscend + NLE, plays the game
        and saves the state each time the requested level is reached.

    When neither is usable (AutoAscend/NLE missing) the generator falls back to
    :meth:`generate_stub_saves`, keeping the pipeline runnable for smoke tests.
    """

    def __init__(
        self,
        repo: Optional[str] = None,
        branch: str = AUTOASCEND_BRANCH,
        python: Optional[str] = None,
        command: Optional[str] = None,
        save_format: str = "ttyrec",
        timeout: float = 24 * 3600.0,
        max_steps: int = 1_000_000,
        max_episodes: int = 20_000,
        allow_stub: bool = True,
        verbose: bool = True,
    ) -> None:
        self.repo = repo
        self.branch = branch
        self.python = python or sys.executable
        self.command = command
        self.save_format = save_format
        self.timeout = float(timeout)
        self.max_steps = int(max_steps)
        self.max_episodes = int(max_episodes)
        self.allow_stub = bool(allow_stub)
        self.verbose = bool(verbose)

    # -- discovery -------------------------------------------------------- #
    def resolve_repo(self) -> Optional[str]:
        return autoascend_path(self.repo)

    def available(self) -> bool:
        return self.command is not None or self.resolve_repo() is not None

    # -- public API ------------------------------------------------------- #
    def generate(
        self,
        level: Any = LEVEL_4,
        num_saves: int = DEFAULT_NUM_SAVES,
        out_dir: Optional[str] = None,
        seed: int = 0,
        stub: bool = False,
        progress_fn: Optional[Callable[[int, int], None]] = None,
        regenerate: bool = False,
    ) -> List[LevelSave]:
        """Produce (or reuse) ``num_saves`` saves for ``level``."""
        spec = resolve_level(level)
        out_dir = out_dir or os.path.join("data", "autoascend_saves", spec.name)
        os.makedirs(out_dir, exist_ok=True)

        existing = list_saves(spec.name, out_dir)
        if existing and not regenerate:
            if len(existing) >= num_saves:
                return existing[:num_saves]
            num_saves = max(num_saves, len(existing))

        if stub or not self.available():
            if not stub and not self.allow_stub:
                raise RuntimeError(
                    "AutoAscend is unavailable; install it from "
                    f"{AUTOASCEND_REPO_URL} (branch {AUTOASCEND_BRANCH}) or set "
                    f"{AUTOASCEND_ENV_VAR}."
                )
            return self.generate_stub_saves(spec, num_saves, out_dir, seed)

        manifest = os.path.join(out_dir, SAVE_MANIFEST)
        if regenerate and os.path.exists(manifest):
            os.remove(manifest)
        if not os.path.exists(manifest):
            code = self._run(level=spec, out_dir=out_dir, seed=seed, num_saves=num_saves)
            if code != 0:
                if self.allow_stub:
                    if self.verbose:
                        print(f"[per_level_eval] AutoAscend failed (rc={code}); using stub saves")
                    return self.generate_stub_saves(spec, num_saves, out_dir, seed)
                raise RuntimeError(f"AutoAscend save generation failed with code {code}")

        saves = list_saves(spec.name, out_dir)
        if progress_fn is not None:
            progress_fn(len(saves), num_saves)
        return saves[:num_saves]

    # -- internals -------------------------------------------------------- #
    def _run(self, level: LevelSpec, out_dir: str, seed: int, num_saves: int) -> int:
        if self.command:
            cmd = self.command.format(
                repo=self.resolve_repo() or "",
                level=level.name,
                target=level.autoascend_target,
                out_dir=out_dir,
                seed=seed,
                num_saves=num_saves,
            )
            if self.verbose:
                print(f"[per_level_eval] $ {cmd}")
            proc = subprocess.run(cmd, shell=True, cwd=self.resolve_repo() or None,
                                  timeout=self.timeout)
            return int(proc.returncode)

        repo = self.resolve_repo()
        if repo is None:
            return 127

        helper = os.path.join(tempfile.mkdtemp(prefix="autoascend_save_"), "gen_saves.py")
        with open(helper, "w") as fh:
            fh.write(_HELPER_SOURCE)

        cmd = [
            self.python, helper,
            "--repo", repo,
            "--level", level.name,
            "--spec", json.dumps(level.as_dict()),
            "--num-saves", str(num_saves),
            "--out-dir", out_dir,
            "--seed", str(seed),
            "--max-steps", str(self.max_steps),
            "--max-episodes", str(self.max_episodes),
            "--save-format", self.save_format,
        ]
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [repo, os.path.join(repo, "src")] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
        )
        if self.verbose:
            print(f"[per_level_eval] generating AutoAscend saves: {' '.join(cmd)}")
        try:
            proc = subprocess.run(cmd, cwd=repo, env=env, timeout=self.timeout,
                                  capture_output=True, text=True)
        except subprocess.TimeoutExpired as exc:  # pragma: no cover - long running
            if self.verbose:
                print(f"[per_level_eval] AutoAscend timed out: {exc}")
            return 124
        if self.verbose and proc.stdout:
            print(proc.stdout.strip()[-2000:])
        if proc.returncode != 0 and self.verbose and proc.stderr:
            print(proc.stderr.strip()[-2000:], file=sys.stderr)
        return int(proc.returncode)

    def generate_stub_saves(
        self,
        level: Any,
        num_saves: int = DEFAULT_NUM_SAVES,
        out_dir: Optional[str] = None,
        seed: int = 0,
    ) -> List[LevelSave]:
        """Deterministic synthetic saves (used when AutoAscend/NLE is absent)."""
        spec = resolve_level(level)
        out_dir = out_dir or os.path.join("data", "autoascend_saves", spec.name)
        os.makedirs(out_dir, exist_ok=True)
        rng = random.Random(seed)
        saves: List[LevelSave] = []
        for i in range(int(num_saves)):
            path = os.path.join(out_dir, f"save_{i:04d}.json")
            expert_score = float(rng.randint(0, 40)) if spec.is_sokoban else 0.0
            save = LevelSave(
                level=spec.name,
                index=i,
                path=path,
                seed=seed + i,
                expert_score=expert_score,
                expert_turns=float(1000 + 50 * i % 5000),
                expert_dlvl=float(spec.dlvl) if spec.dlvl else None,
                stub=True,
                metadata={"generator": "stub", "spec": spec.as_dict()},
            )
            with open(path, "w") as fh:
                json.dump(save.as_dict(), fh, indent=2)
            saves.append(save)
        with open(os.path.join(out_dir, SAVE_MANIFEST), "w") as fh:
            json.dump({"saves": [s.as_dict() for s in saves], "stub": True}, fh, indent=2)
        return saves


def list_saves(level: Any, directory: Optional[str] = None) -> List[LevelSave]:
    """Read the manifest (or the directory listing) for a level's saves."""
    spec = resolve_level(level)
    directory = directory or os.path.join("data", "autoascend_saves", spec.name)
    manifest = os.path.join(directory, SAVE_MANIFEST)
    if os.path.exists(manifest):
        try:
            with open(manifest) as fh:
                payload = json.load(fh)
            entries = payload.get("saves", payload if isinstance(payload, list) else [])
            saves = [LevelSave.from_dict(e) for e in entries if isinstance(e, dict)]
            for save in saves:
                if not save.level:
                    save.level = spec.name
            if saves:
                saves.sort(key=lambda s: s.index)
                return saves
        except Exception:
            pass
    # fall back to listing files on disk
    if not os.path.isdir(directory):
        return []
    saves = []
    for name in sorted(os.listdir(directory)):
        if name == SAVE_MANIFEST or name.startswith("."):
            continue
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            continue
        index = 0
        digits = "".join(ch for ch in os.path.splitext(name)[0] if ch.isdigit()) or "0"
        try:
            index = int(digits)
        except ValueError:
            index = len(saves)
        saves.append(LevelSave(level=spec.name, index=index, path=path,
                               stub=name.endswith(".json")))
    saves.sort(key=lambda s: s.index)
    return saves


def generate_saves(
    level: Any = LEVEL_4,
    num_saves: int = DEFAULT_NUM_SAVES,
    out_dir: Optional[str] = None,
    seed: int = 0,
    stub: bool = False,
    repo: Optional[str] = None,
    command: Optional[str] = None,
    progress_fn: Optional[Callable[[int, int], None]] = None,
    regenerate: bool = False,
    **kwargs: Any,
) -> List[LevelSave]:
    """Convenience wrapper around :class:`AutoAscendSaveGenerator`."""
    generator = AutoAscendSaveGenerator(repo=repo, command=command, **kwargs)
    return generator.generate(level=level, num_saves=num_saves, out_dir=out_dir,
                              seed=seed, stub=stub, progress_fn=progress_fn,
                              regenerate=regenerate)


def describe_saves(save_dir: str, levels: Sequence[Any] = PER_LEVEL_TARGETS) -> Dict[str, Any]:
    """Inventory of available saves per level (used by the CLI/driver)."""
    report: Dict[str, Any] = {"save_dir": save_dir, "levels": {}}
    for level in levels:
        spec = resolve_level(level)
        directory = save_dir
        if not os.path.exists(os.path.join(directory, SAVE_MANIFEST)) and \
                os.path.isdir(os.path.join(save_dir, spec.name)):
            directory = os.path.join(save_dir, spec.name)
        saves = list_saves(spec.name, directory)
        report["levels"][spec.name] = {
            "num_saves": len(saves),
            "directory": directory,
            "stub": bool(saves and all(s.stub for s in saves)),
            "spec": spec.as_dict(),
        }
    return report


# --------------------------------------------------------------------------- #
# Episode evaluation
# --------------------------------------------------------------------------- #

def _make_episode_env(
    save: LevelSave,
    stub: bool = False,
    character: str = "human-monk",
    max_steps: int = DEFAULT_MAX_STEPS,
    no_progress_steps: int = DEFAULT_NO_PROGRESS_STEPS,
    env_factory: Optional[Callable[[LevelSave], Any]] = None,
    **env_kwargs: Any,
) -> Any:
    """Create an environment for one save, restoring the expert state if possible."""
    if env_factory is not None:
        return env_factory(save)

    mod = _nethack_env_module()
    use_stub = bool(stub or save.stub)
    if mod is None or not hasattr(mod, "make_env"):
        return _StubEpisodeEnv(save)

    kwargs = dict(
        seed=save.seed,
        stub=use_stub,
        character=character,
        max_episode_steps=max_steps,
        no_progress_steps=no_progress_steps,
    )
    kwargs.update(env_kwargs)
    try:
        env = mod.make_env(**kwargs)
    except TypeError:
        env = mod.make_env(seed=save.seed, stub=use_stub)

    # Load the game at the state where AutoAscend finished (Appendix B.1).
    for attr in ("restore_state", "load_state", "load", "restore"):
        fn = getattr(env, attr, None)
        if callable(fn) and save.path and os.path.exists(save.path):
            try:
                fn(save.path)
                break
            except Exception:
                continue
    return env


class _StubEpisodeEnv:
    """Dependency-free deterministic episode env used when NLE is missing."""

    def __init__(self, save: LevelSave, horizon: int = 512) -> None:
        self.save = save
        self.horizon = horizon
        self.t = 0
        self._rng = random.Random(int(save.seed or 0) + 1)
        self.observation_keys_used = ()

    def reset(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        self.t = 0
        return {"obs": {"stub": self.t}, "info": self._info()}

    def _info(self) -> Dict[str, Any]:
        score = float(self.save.expert_score) + 0.25 * self.t
        info = {
            "score": score,
            "turns": float(100 + self.t),
            "dlvl": float(self.save.spec.dlvl or 1),
            "xplvl": float(self.t % 10),
        }
        if self.save.spec.is_sokoban:
            info["sokoban_filled_pits"] = float(int(self.t // 64))
        return info

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        self.t += 1
        terminated = self.t >= self.horizon
        reward = 1.0 if self.t % 37 == 0 else 0.0
        return {"stub": self.t}, reward, terminated, False, self._info()

    def close(self) -> None:
        return None


def _get_info_field(info: Any, keys: Sequence[str], default: Any = None) -> Any:
    if isinstance(info, dict):
        for key in keys:
            if key in info and info[key] is not None:
                return info[key]
        for value in info.values():
            if isinstance(value, dict):
                found = _get_info_field(value, keys, None)
                if found is not None:
                    return found
    for key in keys:
        value = getattr(info, key, None)
        if value is not None:
            return value
    return default


def _split_step(out: Any) -> Tuple[Any, float, bool, bool, Any]:
    if isinstance(out, tuple):
        if len(out) == 4:
            obs, reward, done, info = out
            return obs, reward, bool(done), False, info
        if len(out) >= 5:
            obs, reward, terminated, truncated, info = out[:5]
            return obs, reward, bool(terminated), bool(truncated), info
    obs = out
    return obs, 0.0, False, False, {}


def _split_reset(out: Any) -> Tuple[Any, Any]:
    if isinstance(out, tuple) and len(out) == 2:
        return out[0], out[1]
    return out, {}


def _policy_action(policy: Any, obs: Any, deterministic: bool = True) -> Any:
    """Duck-typed action selection (agent, policy module, or callable)."""
    if policy is None:
        return 0
    target = getattr(policy, "policy", None)
    for candidate in (policy, target):
        if candidate is None:
            continue
        fn = getattr(candidate, "act", None)
        if callable(fn):
            try:
                return fn(obs, deterministic=deterministic)
            except TypeError:
                try:
                    return fn(obs)
                except Exception:
                    pass
        if callable(candidate) and not hasattr(candidate, "parameters"):
            try:
                return candidate(obs)
            except Exception:
                pass
    return 0


def evaluate_on_save(
    agent: Any,
    save: LevelSave,
    *,
    stub: bool = False,
    deterministic: bool = True,
    max_steps: int = DEFAULT_MAX_STEPS,
    no_progress_steps: int = DEFAULT_NO_PROGRESS_STEPS,
    env_factory: Optional[Callable[[LevelSave], Any]] = None,
    count_from_expert: bool = True,
    seed: Optional[int] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the agent from the expert's finishing state on one save.

    Returns the score the agent achieved, both absolutely and "on top of the
    expert's score" (``score_above_expert``), which is what Appendix B.1
    reports.
    """
    env = _make_episode_env(
        save,
        stub=stub,
        max_steps=max_steps,
        no_progress_steps=no_progress_steps,
        env_factory=env_factory,
        **(env_kwargs or {}),
    )

    torch = _torch()
    reset_out = env.reset()
    obs, info = _split_reset(reset_out)

    start_score = _as_float(_get_info_field(info, SCORE_KEYS, save.expert_score))
    best_score = start_score
    last_progress_score = start_score
    steps_since_progress = 0
    total_steps = 0
    total_reward = 0.0
    filled_pits = 0.0
    last_info = info

    while True:
        with (torch.no_grad() if torch is not None and hasattr(torch, "no_grad") else _nullcontext()):
            action = _policy_action(agent, obs, deterministic=deterministic)
        out = env.step(action)
        obs, reward, terminated, truncated, info = _split_step(out)
        total_steps += 1
        total_reward += _as_float(reward) if not _is_nan(reward) else 0.0
        last_info = info

        score = _as_float(_get_info_field(info, SCORE_KEYS, best_score))
        if not _is_nan(score) and score > best_score:
            best_score = score
        progress_value = score
        dlvl = _as_float(_get_info_field(info, ("dlvl", "depth"), 0.0))
        if not _is_nan(dlvl):
            progress_value = score + 0.01 * dlvl
        if _is_nan(progress_value) or progress_value > last_progress_score + 1e-9:
            last_progress_score = progress_value if not _is_nan(progress_value) else last_progress_score
            steps_since_progress = 0
        else:
            steps_since_progress += 1

        pits = _as_float(_get_info_field(info, FILLED_PITS_KEYS, None))
        if not _is_nan(pits):
            filled_pits = max(filled_pits, pits)

        if terminated:
            break
        if truncated:
            break
        if steps_since_progress >= no_progress_steps:
            break
        if total_steps >= max_steps:
            break

    try:
        env.close()
    except Exception:
        pass

    final_score = best_score if not _is_nan(best_score) else start_score
    above = final_score - save.expert_score if count_from_expert else final_score
    return {
        "level": save.level,
        "save_index": save.index,
        "save_path": save.path,
        "score": final_score,
        "score_above_expert": above,
        "expert_score": float(save.expert_score),
        "steps": total_steps,
        "return": total_reward,
        "filled_pits": filled_pits,
        "turns": _as_float(_get_info_field(last_info, ("turns",), float("nan"))),
        "dlvl": _as_float(_get_info_field(last_info, ("dlvl", "depth"), float("nan"))),
        "stub": bool(stub or save.stub),
    }


class _nullcontext:
    """Local no-op context manager (``contextlib.nullcontext`` equivalent)."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: Any) -> bool:
        return False


def evaluate_level(
    agent: Any,
    level: Any,
    saves: Optional[Sequence[LevelSave]] = None,
    *,
    num_episodes: int = DEFAULT_EVAL_EPISODES,
    stub: bool = False,
    save_dir: Optional[str] = None,
    seed: int = 0,
    deterministic: bool = True,
    max_steps: int = DEFAULT_MAX_STEPS,
    no_progress_steps: int = DEFAULT_NO_PROGRESS_STEPS,
    env_factory: Optional[Callable[[LevelSave], Any]] = None,
    count_from_expert: bool = True,
    confidence: float = DEFAULT_CONFIDENCE,
    progress_fn: Optional[Callable[[int, int], None]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Evaluate ``agent`` on ``num_episodes`` saves of one level (Figure 5)."""
    spec = resolve_level(level)
    if saves is None:
        directory = save_dir or os.path.join("data", "autoascend_saves", spec.name)
        saves = list_saves(spec.name, directory)
    saves = list(saves)[: int(num_episodes)] if num_episodes else list(saves)

    if not saves:
        # No saves available: generate stub saves so evaluation still runs.
        generator = AutoAscendSaveGenerator(allow_stub=True, verbose=verbose)
        saves = generator.generate_stub_saves(spec, num_episodes, save_dir, seed)

    episodes: List[Dict[str, Any]] = []
    for i, save in enumerate(saves):
        if progress_fn is not None:
            progress_fn(i + 1, len(saves))
        try:
            episodes.append(
                evaluate_on_save(
                    agent, save,
                    stub=stub,
                    deterministic=deterministic,
                    max_steps=max_steps,
                    no_progress_steps=no_progress_steps,
                    env_factory=env_factory,
                    count_from_expert=count_from_expert,
                )
            )
        except Exception as exc:  # pragma: no cover - defensive
            episodes.append({"level": spec.name, "save_index": save.index,
                             "error": repr(exc), "score": float("nan"),
                             "score_above_expert": float("nan"),
                             "filled_pits": float("nan"), "steps": 0})

    scores = [e.get("score", float("nan")) for e in episodes]
    above = [e.get("score_above_expert", float("nan")) for e in episodes]
    pits = [e.get("filled_pits", 0.0) for e in episodes]
    steps = [e.get("steps", 0) for e in episodes]

    summary = {
        "level": spec.name,
        "display": spec.display,
        "spec": spec.as_dict(),
        "num_episodes": len(episodes),
        "num_saves": len(saves),
        "stub": bool(stub or (saves and all(s.stub for s in saves))),
        "score": summarize(scores, confidence),
        "score_above_expert": summarize(above, confidence),
        "filled_pits": summarize(pits, confidence),
        "steps": summarize(steps, confidence),
        "episodes": episodes,
    }
    return summary


def evaluate_per_level(
    agent: Any,
    step: Optional[int] = None,
    levels: Sequence[Any] = PER_LEVEL_TARGETS,
    *,
    num_episodes: int = DEFAULT_EVAL_EPISODES,
    save_dir: Optional[str] = None,
    stub: bool = False,
    seed: int = 0,
    confidence: float = DEFAULT_CONFIDENCE,
    progress_fn: Optional[Callable[[str, int, int], None]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Evaluate ``agent`` on every target level (used by the NetHack driver)."""
    results: Dict[str, Any] = {"step": step, "levels": {}, "confidence": confidence}
    for level in levels:
        spec = resolve_level(level)
        directory = save_dir
        if directory and not os.path.exists(os.path.join(directory, SAVE_MANIFEST)) \
                and os.path.isdir(os.path.join(directory, spec.name)):
            directory = os.path.join(directory, spec.name)
        results["levels"][spec.name] = evaluate_level(
            agent,
            spec,
            num_episodes=num_episodes,
            stub=stub,
            save_dir=directory,
            seed=seed,
            confidence=confidence,
            progress_fn=(lambda i, n, _s=spec: progress_fn(_s.name, i, n)) if progress_fn else None,
            **kwargs,
        )
    return results


#: Alias used by ``src.nethack.train_nethack``.
run_per_level_eval = evaluate_per_level


def aggregate_level_results(
    results: Sequence[Dict[str, Any]],
    levels: Sequence[Any] = PER_LEVEL_TARGETS,
    key: str = "score",
    confidence: float = DEFAULT_CONFIDENCE,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Aggregate per-level summaries across seeds/runs."""
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for level in levels:
        spec = resolve_level(level)
        averages: List[float] = []
        pits: List[float] = []
        for result in results:
            entry = (result.get("levels", {}) or {}).get(spec.name) if isinstance(result, dict) else None
            if entry is None:
                continue
            block = entry.get(key, {})
            if isinstance(block, dict):
                averages.append(_as_float(block.get("mean")))
            else:
                averages.append(_as_float(block))
            pit_block = entry.get("filled_pits", {})
            pits.append(_as_float(pit_block.get("mean") if isinstance(pit_block, dict) else pit_block))
        out[spec.name] = {
            key: summarize(averages, confidence),
            "filled_pits": summarize(pits, confidence),
        }
    return out


# --------------------------------------------------------------------------- #
# Online evaluator (periodic, every 25M steps)
# --------------------------------------------------------------------------- #

class PerLevelEvaluator:
    """Run the per-level evaluation every ``every`` environment steps.

    Mirrors Appendix B.1: 200 saves per level, 200 episodes, evaluation every
    25M steps, reporting the score achieved on top of the expert's score.
    """

    def __init__(
        self,
        agent: Any = None,
        levels: Sequence[Any] = PER_LEVEL_TARGETS,
        *,
        num_saves: int = DEFAULT_NUM_SAVES,
        num_episodes: int = DEFAULT_EVAL_EPISODES,
        every: int = EVAL_EVERY,
        save_dir: str = "data/autoascend_saves",
        output_dir: Optional[str] = None,
        stub: bool = False,
        seed: int = 0,
        confidence: float = DEFAULT_CONFIDENCE,
        autoascend_repo: Optional[str] = None,
        autoascend_command: Optional[str] = None,
        generate: bool = True,
        env_factory: Optional[Callable[[LevelSave], Any]] = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        no_progress_steps: int = DEFAULT_NO_PROGRESS_STEPS,
        name: str = "per_level_eval",
        verbose: bool = False,
    ) -> None:
        self.agent = agent
        self.levels = tuple(resolve_level(l) for l in levels)
        self.num_saves = int(num_saves)
        self.num_episodes = int(num_episodes)
        self.every = int(every)
        self.save_dir = save_dir
        self.output_dir = output_dir
        self.stub = bool(stub)
        self.seed = int(seed)
        self.confidence = float(confidence)
        self.env_factory = env_factory
        self.max_steps = int(max_steps)
        self.no_progress_steps = int(no_progress_steps)
        self.name = name
        self.verbose = bool(verbose)
        self.generator = AutoAscendSaveGenerator(
            repo=autoascend_repo, command=autoascend_command, allow_stub=True,
            verbose=verbose,
        )
        self._generate = bool(generate)
        self._saves: Dict[str, List[LevelSave]] = {}
        self.history: List[Dict[str, Any]] = []

    # -- saves ------------------------------------------------------------ #
    def prepare_saves(
        self,
        regenerate: bool = False,
        progress_fn: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, List[LevelSave]]:
        """Generate/reuse the 200 AutoAscend saves for each level."""
        for spec in self.levels:
            directory = os.path.join(self.save_dir, spec.name) if self.save_dir else None
            existing = list_saves(spec.name, directory) if directory else []
            if existing and not regenerate and len(existing) >= self.num_saves:
                self._saves[spec.name] = existing[: self.num_saves]
                continue
            if not self._generate:
                self._saves[spec.name] = existing
                continue
            self._saves[spec.name] = self.generator.generate(
                spec,
                num_saves=self.num_saves,
                out_dir=directory,
                seed=self.seed,
                stub=self.stub,
                progress_fn=progress_fn,
                regenerate=regenerate,
            )
        return self._saves

    def saves_for(self, level: Any) -> List[LevelSave]:
        spec = resolve_level(level)
        if spec.name not in self._saves:
            self.prepare_saves()
        return self._saves.get(spec.name, [])

    # -- cadence ---------------------------------------------------------- #
    def should_evaluate(self, step: int) -> bool:
        """``True`` every ``every`` steps (25M by default)."""
        if self.every <= 0:
            return True
        step = int(step)
        if step <= 0:
            return False
        return step % self.every == 0

    # -- evaluation ------------------------------------------------------- #
    def evaluate(
        self,
        agent: Any = None,
        step: Optional[int] = None,
        levels: Optional[Sequence[Any]] = None,
        progress_fn: Optional[Callable[[str, int, int], None]] = None,
    ) -> Dict[str, Any]:
        agent = agent if agent is not None else self.agent
        if agent is None:
            raise ValueError("PerLevelEvaluator.evaluate requires an agent")
        self.prepare_saves()
        specs = [resolve_level(l) for l in (levels if levels is not None else self.levels)]
        result: Dict[str, Any] = {"step": step, "levels": {}, "confidence": self.confidence}
        for spec in specs:
            directory = os.path.join(self.save_dir, spec.name) if self.save_dir else None
            result["levels"][spec.name] = evaluate_level(
                agent,
                spec,
                saves=self._saves.get(spec.name),
                num_episodes=self.num_episodes,
                stub=self.stub,
                save_dir=directory,
                seed=self.seed,
                confidence=self.confidence,
                env_factory=self.env_factory,
                max_steps=self.max_steps,
                no_progress_steps=self.no_progress_steps,
                progress_fn=(lambda i, n, _s=spec: progress_fn(_s.name, i, n)) if progress_fn else None,
                verbose=self.verbose,
            )
        return result

    def record(self, step: int, result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        result = result if result is not None else {"step": step, "levels": {}}
        result = dict(result)
        result["step"] = int(step)
        self.history.append(result)
        self.history.sort(key=lambda r: r.get("step", 0))
        return result

    def evaluate_and_record(
        self,
        agent: Any = None,
        step: int = 0,
        levels: Optional[Sequence[Any]] = None,
        progress_fn: Optional[Callable[[str, int, int], None]] = None,
    ) -> Dict[str, Any]:
        if not self.should_evaluate(step) and step not in (0,):
            # still allow explicit calls; cadence is only enforced by the driver
            pass
        result = self.evaluate(agent=agent, step=step, levels=levels, progress_fn=progress_fn)
        return self.record(step, result)

    # -- reporting -------------------------------------------------------- #
    def curve(self, level: Any, key: str = "score") -> Tuple[List[int], List[float]]:
        """``(steps, means)`` trajectory for one level (Figure 5)."""
        spec = resolve_level(level)
        steps: List[int] = []
        values: List[float] = []
        for entry in self.history:
            block = (entry.get("levels", {}) or {}).get(spec.name)
            if not block:
                continue
            stat = block.get(key, {})
            mean = stat.get("mean") if isinstance(stat, dict) else stat
            if mean is None or _is_nan(mean):
                continue
            steps.append(int(entry.get("step", 0)))
            values.append(float(mean))
        return steps, values

    def final(self, key: str = "score") -> Dict[str, float]:
        out: Dict[str, float] = {}
        for spec in self.levels:
            steps, values = self.curve(spec.name, key)
            out[spec.name] = values[-1] if values else float("nan")
        return out

    def summary(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "every": self.every,
            "num_saves": self.num_saves,
            "num_episodes": self.num_episodes,
            "levels": {spec.name: spec.as_dict() for spec in self.levels},
            "final": self.final("score"),
            "final_filled_pits": self.final("filled_pits"),
            "history": self.history,
            "save_dir": self.save_dir,
        }

    def to_dict(self, include_history: bool = True) -> Dict[str, Any]:
        payload = self.summary()
        if not include_history:
            payload.pop("history", None)
        return payload

    def save(self, path: Optional[str] = None) -> str:
        path = path or os.path.join(self.output_dir or ".", f"{self.name}.json")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(include_history=True), fh, indent=2)
        return path

    @classmethod
    def load(cls, path: str, **kwargs: Any) -> "PerLevelEvaluator":
        with open(path) as fh:
            payload = json.load(fh)
        evaluator = cls(
            levels=[l.get("name") for l in (payload.get("levels") or {}).values()] or list(PER_LEVEL_TARGETS),
            num_saves=payload.get("num_saves", DEFAULT_NUM_SAVES),
            num_episodes=payload.get("num_episodes", DEFAULT_EVAL_EPISODES),
            every=payload.get("every", EVAL_EVERY),
            save_dir=payload.get("save_dir", "data/autoascend_saves"),
            generate=False,
            name=payload.get("name", "per_level_eval"),
            **kwargs,
        )
        evaluator.history = list(payload.get("history", []))
        return evaluator

    def __len__(self) -> int:
        return len(self.history)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"PerLevelEvaluator(levels={[s.name for s in self.levels]}, "
                f"every={self.every}, num_saves={self.num_saves}, "
                f"num_episodes={self.num_episodes}, records={len(self.history)})")


# --------------------------------------------------------------------------- #
# Reporting / plotting
# --------------------------------------------------------------------------- #

def format_table(
    results: Any,
    keys: Sequence[str] = ("score", "filled_pits"),
    levels: Sequence[Any] = PER_LEVEL_TARGETS,
) -> str:
    """Render a small text table of per-level means."""
    lines = []
    if isinstance(results, PerLevelEvaluator):
        results = {"evaluator": results.summary()}
    if isinstance(results, dict) and "levels" in results and results.get("levels") \
            and all(isinstance(v, dict) and ("score" in v or "mean" in v) for v in results["levels"].values()):
        results = {"run": results}
    header = ["method", "level"] + list(keys)
    lines.append("  ".join(f"{h:>16}" for h in header))
    for method, payload in (results or {}).items():
        levels_payload = payload.get("levels", payload) if isinstance(payload, dict) else {}
        for level in levels:
            spec = resolve_level(level)
            block = levels_payload.get(spec.name, {})
            row = [str(method), spec.display]
            for key in keys:
                stat = block.get(key, {})
                value = stat.get("mean") if isinstance(stat, dict) else stat
                row.append("nan" if value is None or _is_nan(value) else f"{float(value):.2f}")
            lines.append("  ".join(f"{c:>16}" for c in row))
    return "\n".join(lines)


def plot_per_level_curves(
    histories: Any,
    path: Optional[str] = None,
    *,
    levels: Sequence[Any] = PER_LEVEL_TARGETS,
    key: str = "score",
    methods: Optional[Sequence[str]] = None,
    confidence: float = DEFAULT_CONFIDENCE,
    title: Optional[str] = None,
    xlabel: str = "environment steps",
    ylabel: Optional[str] = None,
    stacked: bool = True,
    show: bool = False,
    **plot_kwargs: Any,
) -> Any:
    """Figure-5 style plot: one panel per level (level 4 top, Sokoban bottom)."""
    try:
        import matplotlib

        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - matplotlib optional
        return None

    # normalise input into {method: PerLevelEvaluator-like history}
    if isinstance(histories, PerLevelEvaluator):
        runs = {"agent": histories}
    elif isinstance(histories, dict):
        runs = {}
        for name, value in histories.items():
            if isinstance(value, PerLevelEvaluator):
                runs[name] = value
            elif isinstance(value, dict) and "history" in value:
                runs[name] = value
            elif isinstance(value, list):
                runs[name] = {"history": value}
    else:
        runs = {"agent": {"history": list(histories or [])}}

    if methods:
        runs = {k: v for k, v in runs.items() if k in set(methods)}
    specs = [resolve_level(l) for l in levels]
    ncols = 1 if stacked else max(1, len(specs))
    nrows = len(specs) if stacked else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 3.2 * nrows), squeeze=False)
    for row, spec in enumerate(specs):
        ax = axes[row][0] if stacked else axes[0][row]
        for method, payload in runs.items():
            history = payload.history if isinstance(payload, PerLevelEvaluator) else payload.get("history", [])
            steps, vals, halfs = [], [], []
            for entry in history:
                block = (entry.get("levels", {}) or {}).get(spec.name)
                if not block:
                    continue
                stat = block.get(key, {})
                if isinstance(stat, dict):
                    mean, half = stat.get("mean"), stat.get("half_width", 0.0)
                else:
                    mean, half = stat, 0.0
                if mean is None or _is_nan(mean):
                    continue
                steps.append(int(entry.get("step", 0)))
                vals.append(float(mean))
                halfs.append(float(half or 0.0))
            if not steps:
                continue
            line, = ax.plot(steps, vals, label=str(method), **{
                k: v for k, v in plot_kwargs.items() if k not in ("ax",)
            })
            if any(h and not _is_nan(h) for h in halfs):
                ax.fill_between(steps, [v - h for v, h in zip(vals, halfs)],
                                [v + h for v, h in zip(vals, halfs)],
                                color=line.get_color(), alpha=0.2)
        ax.set_title(f"{spec.display}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel or key.replace("_", " "))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:  # pragma: no cover - interactive
        plt.show()
    return fig


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="NetHack per-level (level 4 / Sokoban) evaluation — Figure 5.",
    )
    parser.add_argument("--level", default=None,
                        help="single level to evaluate (default: level_4 and sokoban)")
    parser.add_argument("--levels", nargs="*", default=None,
                        help="levels to evaluate (default: all PER_LEVEL_TARGETS)")
    parser.add_argument("--generate-saves", action="store_true",
                        help="generate AutoAscend saves and exit")
    parser.add_argument("--num-saves", type=int, default=DEFAULT_NUM_SAVES)
    parser.add_argument("--num-episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    parser.add_argument("--save-dir", default="data/autoascend_saves")
    parser.add_argument("--autoascend-repo", default=None)
    parser.add_argument("--autoascend-command", default=None)
    parser.add_argument("--checkpoint", default=None,
                        help="policy checkpoint to evaluate (NetHack model or stub)")
    parser.add_argument("--step", type=int, default=None,
                        help="training step the checkpoint corresponds to")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stub", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--plot", default=None, help="write Figure 5 to this path")
    parser.add_argument("--show", action="store_true")
    return parser


def _load_agent_from_checkpoint(path: Optional[str], stub: bool = False) -> Any:
    if not path:
        return None
    model_mod = _import_module("src.nethack.model") or _import_module("finetuning_rl_as_cl.src.nethack.model")
    if model_mod is None:
        return None
    model = None
    for builder in ("build_model", "build_nethack_model"):
        fn = getattr(model_mod, builder, None)
        if callable(fn):
            try:
                model = fn(checkpoint=path, stub=stub)
                break
            except TypeError:
                try:
                    model = fn(stub=stub)
                    if hasattr(model, "load"):
                        model.load(path)
                    break
                except Exception:
                    continue
            except Exception:
                continue
    return model


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    levels = [args.level] if args.level else (args.levels or list(PER_LEVEL_TARGETS))

    if args.generate_saves:
        generator = AutoAscendSaveGenerator(
            repo=args.autoascend_repo, command=args.autoascend_command,
            allow_stub=True, verbose=args.verbose,
        )
        manifest: Dict[str, Any] = {}
        for level in levels:
            spec = resolve_level(level)
            saves = generator.generate(
                spec, num_saves=args.num_saves,
                out_dir=os.path.join(args.save_dir, spec.name),
                seed=args.seed, stub=args.stub,
            )
            manifest[spec.name] = len(saves)
        print(json.dumps({"generated": manifest, "save_dir": args.save_dir}, indent=2))
        return 0

    agent = _load_agent_from_checkpoint(args.checkpoint, stub=args.stub)
    evaluator = PerLevelEvaluator(
        agent=agent,
        levels=levels,
        num_saves=args.num_saves,
        num_episodes=args.num_episodes,
        save_dir=args.save_dir,
        output_dir=args.output_dir,
        stub=args.stub,
        seed=args.seed,
        autoascend_repo=args.autoascend_repo,
        autoascend_command=args.autoascend_command,
        verbose=args.verbose,
    )

    result = evaluator.evaluate_and_record(step=args.step if args.step is not None else 0)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, "per_level_eval.json")
        with open(out_path, "w") as fh:
            json.dump(result, fh, indent=2)
        evaluator.save(os.path.join(args.output_dir, "per_level_eval_history.json"))

    print(format_table({args.checkpoint or "agent": result}))
    if args.plot:
        plot_per_level_curves(evaluator, path=args.plot, show=args.show)
    print(json.dumps({lvl: {k: (v.get("mean") if isinstance(v, dict) else v)
                            for k, v in (blk or {}).items() if k in ("score", "filled_pits")}
                      for lvl, blk in result.get("levels", {}).items()}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
