"""Return normalisation and seed-aggregation utilities for FRE evaluation.

Paper protocol (Section 5, Table 1 caption, addendum):

    "Results are normalized between 0 and 100."
    "Table 1 calculates uncertainty as the standard deviation over 5 seeds
     (with 20 rollouts each, averaged)."

Every evaluation suite in :mod:`fre.envs` returns *raw* episode returns together
with a per-task ``min_return`` / ``max_return`` pair derived from the reward
definition (the paper never states explicit constants, so each task computes its
own bounds).  This module converts those raw returns into the 0-100 scale used in
Table 1 and aggregates per-seed scores into ``mean +/- std`` rows.

The public surface is deliberately small and dependency-light (``numpy`` only)
so it can be imported by scripts, tests and reporting code without pulling in
torch / gym.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    # scaling
    "normalize_return",
    "normalize_returns",
    "normalize_with_bounds",
    "task_bounds",
    "SUCCESS_BONUS",
    # aggregation
    "aggregate_seeds",
    "aggregate_evaluations",
    "aggregate_task_scores",
    "aggregate_domain_scores",
    "aggregate_rows",
    "ResultAccumulator",
    # formatting / reporting
    "format_mean_std",
    "format_score",
    "compare_to_reference",
    "score_within_band",
    "TABLE1_FRE_TARGETS",
    "TABLE1_AGGREGATE_TARGETS",
    "TABLE4_FRE_TARGETS",
    "to_markdown_table",
    # helpers
    "mean_std",
    "coefficient_of_variation",
    "seeded_rng",
]

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

#: Lower/upper edge of the normalised return range used throughout the paper.
NORMALIZED_MIN = 0.0
NORMALIZED_MAX = 100.0

#: Small positive bonus added to the theoretical max when a task is terminal on
#: success (so that an optimal policy that never wastes a step maps exactly to
#: 100).  The paper does not specify this; it only guarantees the 0-100 range.
SUCCESS_BONUS = 0.0

#: Reference aggregate row of Table 1 for FRE (mean +/- std over 5 seeds).
TABLE1_FRE_TARGETS: Dict[str, Tuple[float, float]] = {
    "ant-goal-reaching": (48.8, 6.0),
    "ant-directional": (55.2, 8.0),
    "ant-random-simplex": (21.3, 4.0),
    "ant-path-loop": (67.2, 36.0),
    "ant-path-edges": (60.0, 17.0),
    "ant-path-center": (64.4, 38.0),
    "antmaze-all": (52.8, 18.2),
    "exorl-walker-goals": (94.0, 2.0),
    "exorl-cheetah-goals": (58.0, 8.0),
    "exorl-walker-velocity": (34.0, 13.0),
    "exorl-cheetah-velocity": (20.0, 2.0),
    "exorl-all": (51.5, 6.3),
    "kitchen": (66.0, 3.0),
    "all": (57.0, 9.0),
}

#: Domain-level aggregate rows used to build the final ``all`` row of Table 1.
TABLE1_AGGREGATE_TARGETS: Dict[str, Tuple[float, float]] = {
    "antmaze-all": (52.8, 18.2),
    "exorl-all": (51.5, 6.3),
    "kitchen": (66.0, 3.0),
    "all": (57.0, 9.0),
}

#: Table 4 (Appendix D) prior-mixture ablation totals on AntMaze.
TABLE4_FRE_TARGETS: Dict[str, Tuple[float, float]] = {
    "FRE-all": (47.3, 7.0),
    "FRE-goals": (26.1, 8.0),
    "FRE-lin": (31.6, 5.0),
    "FRE-mlp": (25.3, 8.0),
    "FRE-lin-mlp": (32.3, 5.0),
    "FRE-goal-mlp": (33.8, 15.0),
    "FRE-goal-lin": (46.9, 7.0),
}


# ---------------------------------------------------------------------------
# 0-100 return normalisation
# ---------------------------------------------------------------------------


def normalize_return(
    total_return: float,
    min_return: float = 0.0,
    max_return: Optional[float] = None,
    clip: bool = True,
    success: Optional[bool] = None,
) -> float:
    """Map a raw episode return onto the paper's 0-100 scale.

    ``score = 100 * (total_return - min_return) / (max_return - min_return)``

    Parameters
    ----------
    total_return:
        Undiscounted sum of rewards collected during the episode.
    min_return:
        Return of the worst possible policy (usually ``reward_off_success *
        max_episode_steps``).
    max_return:
        Return of an optimal policy.  When ``None`` the range degenerates and the
        function returns 0.0 for a degenerate task (guarded below).
    clip:
        Clamp the result into ``[0, 100]`` (the paper reports normalised returns
        strictly inside that range).  Clipping is also what makes early
        termination on success safe: a solved task cannot exceed 100.
    success:
        Optional terminal-success flag.  When the environment terminates on
        success the theoretical maximum may be unreachable in the allotted
        horizon; if ``success`` is given and true, the episode is counted as a
        solved task and the score is forced to the maximum.  The paper is silent
        here, so this stays opt-in (``None`` = ignore).
    """
    total_return = float(total_return)
    min_return = float(min_return)

    if success is True:
        return NORMALIZED_MAX

    if max_return is None:
        return 0.0

    max_return = float(max_return) + SUCCESS_BONUS
    span = max_return - min_return
    if abs(span) < 1e-12:
        return 0.0

    score = (total_return - min_return) / span * (NORMALIZED_MAX - NORMALIZED_MIN)
    score += NORMALIZED_MIN
    if not math.isfinite(score):
        return 0.0
    if clip:
        score = min(max(score, NORMALIZED_MIN), NORMALIZED_MAX)
    return float(score)


def normalize_returns(
    total_returns: Iterable[float],
    min_return: float = 0.0,
    max_return: Optional[float] = None,
    clip: bool = True,
) -> np.ndarray:
    """Vectorised :func:`normalize_return` over a sequence of returns."""
    values = np.asarray(list(total_returns), dtype=np.float64)
    if values.size == 0:
        return values
    if max_return is None:
        return np.zeros_like(values)
    span = (float(max_return) + SUCCESS_BONUS) - float(min_return)
    if abs(span) < 1e-12:
        return np.zeros_like(values)
    scores = (values - float(min_return)) / span * (NORMALIZED_MAX - NORMALIZED_MIN)
    scores = scores + NORMALIZED_MIN
    scores = np.where(np.isfinite(scores), scores, 0.0)
    if clip:
        scores = np.clip(scores, NORMALIZED_MIN, NORMALIZED_MAX)
    return scores


def normalize_with_bounds(
    total_returns: Iterable[float],
    bounds: Tuple[float, float],
    clip: bool = True,
) -> np.ndarray:
    """Convenience wrapper taking ``(min_return, max_return)`` as a tuple."""
    lo, hi = bounds
    return normalize_returns(total_returns, lo, hi, clip=clip)


def task_bounds(
    reward_off_success: float,
    max_episode_steps: int,
    reward_on_success: float = 0.0,
    success_steps: int = 1,
) -> Tuple[float, float]:
    """Theoretical ``(min_return, max_return)`` of a sparse goal-style task.

    The reward is ``reward_off_success`` for every timestep until the objective is
    reached and ``reward_on_success`` afterwards.  The worst policy collects
    ``reward_off_success * max_episode_steps``; the best reaches the objective in
    ``success_steps`` steps and then collects ``reward_on_success``.
    """
    lo = float(reward_off_success) * float(max_episode_steps)
    hi = float(reward_off_success) * float(max(0, success_steps - 1))
    hi += float(reward_on_success) * float(max_episode_steps - max(0, success_steps - 1))
    return lo, hi


# ---------------------------------------------------------------------------
# aggregation over rollouts / seeds
# ---------------------------------------------------------------------------


def mean_std(
    values: Iterable[float],
    ddof: int = 1,
) -> Tuple[float, float]:
    """Return ``(mean, std)`` with ``ddof=1`` (sample std) by default.

    Table 1 reports the standard deviation over the 5 *seeds*, i.e. the sample
    standard deviation of the seed-averaged scores.
    """
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=ddof))


def coefficient_of_variation(values: Iterable[float]) -> float:
    """``std / |mean|`` (guarded); useful to flag unstable seeds in logs."""
    mean, std = mean_std(values)
    if not math.isfinite(mean) or abs(mean) < 1e-12:
        return float("nan")
    return std / abs(mean)


def aggregate_seeds(
    per_seed_scores: Sequence[float],
    ddof: int = 1,
    keys: Optional[Mapping[str, str]] = None,
) -> Dict[str, float]:
    """Aggregate one task's scores across seeds into a reportable row.

    Returns ``{"mean", "std", "min", "max", "num_seeds", "sem"}`` plus the
    individual seed scores under ``seed_<i>`` (handy for CSV logs).
    """
    arr = np.asarray(list(per_seed_scores), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    out: Dict[str, float] = {}
    if arr.size == 0:
        out.update(mean=float("nan"), std=float("nan"), min=float("nan"), max=float("nan"))
        out["num_seeds"] = 0
        out["sem"] = float("nan")
        return out

    mean = float(arr.mean())
    std = float(arr.std(ddof=ddof)) if arr.size > 1 else 0.0
    out["mean"] = mean
    out["std"] = std
    out["min"] = float(arr.min())
    out["max"] = float(arr.max())
    out["num_seeds"] = int(arr.size)
    out["sem"] = std / math.sqrt(arr.size) if arr.size > 0 else float("nan")

    if keys:
        # allow callers to rename canonical keys (e.g. ''score'' instead of 'mean')
        renamed = {keys.get(k, k): v for k, v in out.items()}
        out = renamed

    for i, value in enumerate(np.asarray(list(per_seed_scores), dtype=np.float64)):
        out[f"seed_{i}"] = float(value) if np.isfinite(value) else float("nan")
    return out


def _iter_score_values(payload: Any) -> Iterable[float]:
    """Best-effort extraction of a single score from an eval result object."""
    if payload is None:
        return ()
    if isinstance(payload, (int, float, np.floating, np.integer)):
        return (float(payload),)
    if isinstance(payload, Mapping):
        for key in ("score", "mean", "normalized_return", "return", "value"):
            if key in payload and isinstance(
                payload[key], (int, float, np.floating, np.integer)
            ):
                return (float(payload[key]),)
        return ()
    if isinstance(payload, (list, tuple, np.ndarray)):
        try:
            return tuple(float(v) for v in payload if np.isfinite(float(v)))
        except (TypeError, ValueError):
            return ()
    return ()


def aggregate_evaluations(results: Sequence[Any], ddof: int = 1) -> Dict[str, float]:
    """Aggregate an evaluation result (or list of per-episode scores).

    Accepts anything the env wrappers return: a dict with ``score``/``mean``, a
    bare number, or an explicit list of episode scores.  When given a list of
    *per-seed* dicts, use :func:`aggregate_task_scores` instead.
    """
    if isinstance(results, Mapping):
        # already-aggregated dict: pass through the numeric fields we care about
        out: Dict[str, float] = {}
        for key in ("score", "mean", "score_std", "std", "num_episodes", "success_rate"):
            if key in results and isinstance(
                results[key], (int, float, np.floating, np.integer)
            ):
                out["mean" if key == "score" else key] = float(results[key])
        if "mean" in out and "std" not in out and "score_std" in out:
            out["std"] = out["score_std"]
        return out

    values = list(_iter_score_values(results))
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "num_seeds": 0}
    return aggregate_seeds(values, ddof=ddof)


def _result_to_score(result: Any) -> Optional[float]:
    """Extract a single scalar score from a suite/task result payload."""
    if result is None:
        return None
    if isinstance(result, (int, float, np.floating, np.integer)):
        return float(result)
    if isinstance(result, Mapping):
        for key in ("score", "mean", "normalized_return"):
            value = result.get(key)
            if isinstance(value, (int, float, np.floating, np.integer)):
                return float(value)
            if isinstance(value, Mapping):  # nested {"mean": ...}
                inner = value.get("mean")
                if isinstance(inner, (int, float, np.floating, np.integer)):
                    return float(inner)
        return None
    try:
        return float(result)  # e.g. numpy scalar / 0-d array
    except (TypeError, ValueError):
        return None


def _result_to_std(result: Any) -> Optional[float]:
    if isinstance(result, Mapping):
        for key in ("score_std", "std"):
            value = result.get(key)
            if isinstance(value, (int, float, np.floating, np.integer)):
                return float(value)
        if isinstance(result.get("score"), Mapping):
            inner = result["score"].get("std")
            if isinstance(inner, (int, float, np.floating, np.integer)):
                return float(inner)
    return None


def aggregate_task_scores(
    per_seed_results: Sequence[Mapping[str, Any]],
    ddof: int = 1,
) -> Dict[str, Dict[str, float]]:
    """Aggregate ``task_name -> mean +/- std`` from a list of per-seed results.

    ``per_seed_results`` is typically the output of
    :func:`fre.envs.antmaze_eval.evaluate_antmaze_suite` (or the ExORL / Kitchen
    equivalents) collected for each of the 5 seeds.
    """
    task_names: List[str] = []
    for seed_result in per_seed_results:
        if not isinstance(seed_result, Mapping):
            continue
        for name in seed_result:
            if name not in task_names:
                task_names.append(name)

    aggregated: Dict[str, Dict[str, float]] = {}
    for name in task_names:
        values: List[float] = []
        for seed_result in per_seed_results:
            if not isinstance(seed_result, Mapping):
                continue
            score = _result_to_score(seed_result.get(name))
            if score is not None and math.isfinite(score):
                values.append(score)
        if not values:
            continue
        row = aggregate_seeds(values, ddof=ddof)
        # keep the seed-mean std when the suite reported one (informational)
        reported = [
            _result_to_std(seed_result.get(name))
            for seed_result in per_seed_results
            if isinstance(seed_result, Mapping)
        ]
        reported = [r for r in reported if r is not None and math.isfinite(r)]
        if reported:
            row["mean_episode_std"] = float(np.mean(reported))
        aggregated[name] = row
    return aggregated


def aggregate_domain_scores(
    domain_rows: Mapping[str, Any],
    domains: Optional[Sequence[str]] = None,
    ddof: int = 1,
) -> Dict[str, float]:
    """Aggregate domain-level means into the paper's final ``all`` row.

    The paper's aggregate row is approximately the average of the domain rows
    (``(52.8 + 51.5 + 66) / 3 ~= 57``) with the spread over those domains as the
    uncertainty.  ``ddof=1`` (sample std) reproduces the reported ``+/- 9``.
    """
    if domains is None:
        domains = [k for k in domain_rows if k.lower() != "all"]

    means: List[float] = []
    for name in domains:
        payload = domain_rows.get(name)
        score = _result_to_score(payload)
        if score is not None and math.isfinite(score):
            means.append(score)

    if not means:
        return {"mean": float("nan"), "std": float("nan"), "num_domains": 0}
    row = aggregate_seeds(means, ddof=ddof)
    row["num_domains"] = len(means)
    row["domains"] = list(domains)  # type: ignore[assignment]
    return row


def aggregate_rows(
    per_seed_task_results: Sequence[Mapping[str, Any]],
    domains: Optional[Sequence[str]] = None,
    ddof: int = 1,
) -> Dict[str, Dict[str, float]]:
    """Aggregate tasks *and* the final ``all`` row across seeds.

    ``domains`` selects which aggregated rows feed the ``all`` row (default:
    any row whose name ends in ``-all`` plus ``kitchen``).
    """
    rows = aggregate_task_scores(per_seed_task_results, ddof=ddof)

    if domains is None:
        domains = [name for name in rows if name.endswith("-all")]
        if "kitchen" in rows:
            domains.append("kitchen")

    # average per-seed domain means (statistically cleaner than averaging stds)
    per_seed_all: List[float] = []
    for seed_result in per_seed_task_results:
        if not isinstance(seed_result, Mapping):
            continue
        seed_values = []
        for name in domains:
            score = _result_to_score(seed_result.get(name))
            if score is not None and math.isfinite(score):
                seed_values.append(score)
        if seed_values:
            per_seed_all.append(float(np.mean(seed_values)))

    if per_seed_all:
        rows["all"] = aggregate_seeds(per_seed_all, ddof=ddof)
        rows["all"]["num_domains"] = len(domains)
    return rows


# ---------------------------------------------------------------------------
# result accumulator
# ---------------------------------------------------------------------------


@dataclass
class ResultAccumulator:
    """Collect per-seed suite results and emit Table-1 style rows.

    Usage::

        acc = ResultAccumulator(name="FRE")
        for seed in range(5):
            acc.add_seed(seed, evaluate_antmaze_suite(...))
        table = acc.table()          # {task: {"mean": .., "std": ..}}
        print(acc.markdown())        # markdown table with paper targets
    """

    name: str = "method"
    ddof: int = 1
    auto_all_row: bool = True
    domains: Optional[Sequence[str]] = None
    _seed_results: List[Tuple[int, Mapping[str, Any]]] = field(default_factory=list, repr=False)

    # -- collection ---------------------------------------------------------
    def add_seed(self, seed: int, results: Mapping[str, Any], merge: bool = True) -> None:
        if results is None:
            return
        if not isinstance(results, Mapping):
            raise TypeError("results must be a mapping of task name -> score")
        existing: Dict[str, Any] = {}
        if merge:
            for other_seed, other in self._seed_results:
                if other_seed == seed:
                    existing = dict(other)
                    self._seed_results = [
                        (s, r) for (s, r) in self._seed_results if s != seed
                    ]
                    break
        existing.update(results)
        self._seed_results.append((seed, existing))

    @property
    def seeds(self) -> List[int]:
        return [seed for seed, _ in self._seed_results]

    @property
    def num_seeds(self) -> int:
        return len(self._seed_results)

    def clear(self) -> None:
        self._seed_results.clear()

    # -- aggregation --------------------------------------------------------
    def table(self, include_all: Optional[bool] = None) -> Dict[str, Dict[str, float]]:
        results = [r for _, r in self._seed_results]
        if not results:
            return {}
        rows = aggregate_task_scores(results, ddof=self.ddof)
        include_all = self.auto_all_row if include_all is None else include_all
        if include_all:
            rows = aggregate_rows(results, domains=self.domains, ddof=self.ddof)
        return rows

    def scores(self) -> Dict[str, float]:
        return {name: row["mean"] for name, row in self.table().items()}

    # -- reporting ----------------------------------------------------------
    def format_row(self, name: str, digits: int = 1) -> str:
        row = self.table().get(name)
        if not row:
            return f"{name}: n/a"
        return f"{name}: {format_mean_std(row['mean'], row['std'], digits=digits)}"

    def markdown(self, digits: int = 1, targets: Optional[Mapping[str, Any]] = None) -> str:
        rows = self.table()
        if targets is None:
            targets = TABLE1_FRE_TARGETS if self.name.lower().startswith("fre") else {}
        lines = [f"| Eval Task | {self.name} | Paper |", "| --- | --- | --- |"]
        for name, row in rows.items():
            observed = format_mean_std(row["mean"], row["std"], digits=digits)
            target = targets.get(name) if targets else None
            expected = format_mean_std(*target, digits=digits) if target else "-"
            lines.append(f"| {name} | {observed} | {expected} |")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "num_seeds": self.num_seeds,
            "seeds": self.seeds,
            "rows": self.table(),
        }


# ---------------------------------------------------------------------------
# formatting / comparison
# ---------------------------------------------------------------------------


def format_mean_std(mean: float, std: float, digits: int = 1) -> str:
    """``"52.8±18.2"`` style formatting used in Table 1 (rounded to ``digits``)."""
    if mean is None or (isinstance(mean, float) and not math.isfinite(mean)):
        return "n/a"
    if std is None or (isinstance(std, float) and not math.isfinite(std)):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f}±{std:.{digits}f}"


def format_score(score: float, digits: int = 1) -> str:
    if score is None or (isinstance(score, float) and not math.isfinite(score)):
        return "n/a"
    return f"{score:.{digits}f}"


def compare_to_reference(
    observed_mean: float,
    observed_std: float,
    reference_mean: float,
    reference_std: float,
    tolerance: float = 1.0,
    relative: bool = True,
) -> Dict[str, Any]:
    """Compare a reproduced row against a Table-1 reference.

    Success criterion from the plan: "Table 1 aggregates within reported std
    bands".  A row is accepted when the observed mean lies within
    ``reference_mean +/- tolerance * max(reference_std, observed_std)``.
    """
    band = tolerance * max(reference_std, observed_std if relative else 0.0)
    band = max(band, 1e-9)
    delta = observed_mean - reference_mean
    return {
        "observed": observed_mean,
        "reference": reference_mean,
        "delta": delta,
        "abs_delta": abs(delta),
        "band": band,
        "within_band": bool(abs(delta) <= band),
    }


def score_within_band(
    observed_mean: float,
    reference_mean: float,
    reference_std: float,
    tolerance: float = 1.0,
) -> bool:
    """Boolean convenience wrapper around :func:`compare_to_reference`."""
    return compare_to_reference(
        observed_mean, reference_std, reference_mean, reference_std, tolerance=tolerance
    )["within_band"]


def to_markdown_table(
    rows: Mapping[str, Mapping[str, Any]],
    columns: Sequence[str] = ("mean", "std"),
    digits: int = 1,
    title: Optional[str] = None,
) -> str:
    """Render ``{row_name: {col: value}}`` as a markdown table."""
    header = ["Eval Task"] + list(columns)
    lines = []
    if title:
        lines.append(f"### {title}")
        lines.append("")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for name, row in rows.items():
        cells = [name]
        for col in columns:
            value = row.get(col) if isinstance(row, Mapping) else None
            cells.append(format_score(value, digits=digits) if value is not None else "-")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def seeded_rng(seed: int) -> np.random.Generator:
    """Deterministic numpy generator (used for reproducible goal sampling)."""
    return np.random.default_rng(int(seed))


def denormalize_return(score: float, min_return: float, max_return: float) -> float:
    """Inverse of :func:`normalize_return` (useful for logging raw returns)."""
    return float(min_return) + float(score) / (NORMALIZED_MAX - NORMALIZED_MIN) * (
        float(max_return) - float(min_return)
    )
