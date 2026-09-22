"""Sampling-entropy analysis for Classifier-Free Guidance (Section 5.1).

The paper ("Stay on Topic with Classifier-Free Guidance", Section 5.1) states:

    We suspect that CFG, by focusing P(y | x) on the prompt, will reduce the
    entropy of the logit distribution. CFG entropy distribution is significantly
    lower across generation steps than vanilla prompting, with a mean of 4.7 vs
    5.49. This restricts the number of tokens in the top-p = 90% of the
    vocabulary distribution. We observe, in Section 5.3, that the top tokens
    re-order, showing that CFG is not simply having the same effect as
    temperature.

The addendum to the paper clarifies the exact definition used:

    In Section 5.1, the definition of entropy used is H(p) = -sum_k p_k log p_k
    and the implementation used was from
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html
    ...
    the mean entropy is averaged over each token, i.e. if H(p(x_i | x_<i)) is the
    entropy of the vocabulary distribution produced by the LLM then the
    "mean entropy" is (1/n) * sum_{i=1..n} H(p(x_i | x_<i)) over n completion
    tokens.

This module therefore provides:

* ``entropy`` / ``shannon_entropy``      -- H(p) = -sum_k p_k log p_k (scipy when
  available, numerically identical NumPy fallback otherwise).
* ``entropy_from_logits``                -- entropy of softmax(logits).
* ``top_p_token_count``                  -- #tokens in the top-p = 90% nucleus
  (the "fewer possible tokens" statement in Section 5.1).
* ``EntropyStats`` / ``mean_entropy``    -- per-token entropies plus their
  per-token average (1/n) * sum_i H(p(x_i | x_<i)).
* ``EntropyAnalyzer``                    -- drives a ``CFGModelWrapper`` for a
  prompt, records the *post-CFG* vocabulary distribution at every completion
  step, and aggregates the per-token entropies.
* ``compare_entropy_modes``              -- CFG (gamma = 1.5) vs vanilla P(y|x)
  vs unprompted P(x) vs instruction-tuned (e.g. Falcon-7b-Instruct), i.e. the
  comparison behind Figure 5 / Section 5.1.
* ``entropy_report`` / ``format_entropy_table`` -- headline numbers with the
  paper's expected values (4.7 CFG vs 5.49 vanilla) for easy verification.

Everything except the optional ``CFGModelWrapper`` path is NumPy-only (SciPy is
used when present), so the mathematics remains unit-testable on CPU-only
machines.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Optional dependencies
# --------------------------------------------------------------------------- #

try:  # pragma: no cover - scipy is usually available
    from scipy.stats import entropy as _scipy_entropy

    _HAS_SCIPY = True
except Exception:  # pragma: no cover - graceful NumPy fallback
    _scipy_entropy = None  # type: ignore[assignment]
    _HAS_SCIPY = False

try:  # pragma: no cover - torch is optional for the pure-math helpers
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False

# Softmax / log-softmax are re-used from the core CFG package (torch & numpy
# aware).  Import guarded so this module stays importable on its own.
try:  # pragma: no cover
    from ..cfg.logits import log_softmax as _log_softmax
    from ..cfg.logits import softmax as _softmax
except Exception:  # pragma: no cover
    try:
        from src.cfg.logits import log_softmax as _log_softmax  # type: ignore
        from src.cfg.logits import softmax as _softmax  # type: ignore
    except Exception:
        _log_softmax = None  # type: ignore[assignment]
        _softmax = None  # type: ignore[assignment]


# --------------------------------------------------------------------------- #
# Constants (analysis gamma / top-p come from the paper and the addendum)
# --------------------------------------------------------------------------- #

#: Guidance strength used for every Section 5 analysis (CFG column of Fig. 5).
ANALYSIS_GAMMA: float = 1.5

#: Nucleus mass referenced in Section 5.1 ("top-p = 90%").  Note the addendum:
#: the top-p mention is incidental -- entropy directly controls how many tokens
#: carry probability mass.
TOP_P: float = 0.9

#: Paper-reported means (Section 5.1) used as verification anchors.
CFG_ENTROPY_MEAN: float = 4.7
VANILLA_ENTROPY_MEAN: float = 5.49

#: Relative tolerance used by ``check_against_paper``.
ENTROPY_TOLERANCE: float = 0.15

#: Names of the four distributions compared in Section 5.1 / Appendix E.
ENTROPY_MODES: Tuple[str, ...] = ("cfg", "vanilla", "unprompted", "instruct")

#: gamma value that reproduces each mode when passed to the CFG combiner.
MODE_GAMMAS: Dict[str, float] = {
    "cfg": ANALYSIS_GAMMA,
    "vanilla": 1.0,
    "unprompted": 0.0,
    "instruct": 1.0,
}

#: Human-readable labels (used by the text table rendering).
MODE_LABELS: Dict[str, str] = {
    "cfg": "CFG (gamma=1.5)",
    "vanilla": "Vanilla P(y|x)",
    "unprompted": "Unprompted P(x)",
    "instruct": "Instruction-tuned",
}


# --------------------------------------------------------------------------- #
# Core entropy mathematics
# --------------------------------------------------------------------------- #


def _to_numpy(x: Any) -> np.ndarray:
    """Detach/convert a torch tensor to a float64 NumPy array."""
    if _HAS_TORCH and isinstance(x, torch.Tensor):  # pragma: no cover - torch path
        return x.detach().to("cpu").double().numpy()
    return np.asarray(x, dtype=np.float64)


def entropy(
    probs: Any,
    axis: int = -1,
    base: Optional[Union[float, str]] = None,
    normalize: bool = True,
) -> Any:
    """Shannon entropy ``H(p) = -sum_k p_k log p_k`` (Section 5.1 definition).

    Implemented with :func:`scipy.stats.entropy` when available (the exact
    implementation the paper cites) and an equivalent NumPy fallback otherwise.
    Mirrors scipy semantics: zero-probability entries contribute 0, the
    distribution is renormalised unless ``normalize=False``, and ``base=None``
    means nats (natural log).

    Parameters
    ----------
    probs:
        Probability vectors; supports any number of leading batch dimensions.
    axis:
        Axis along which the vocabulary distribution lives (default: last).
    base:
        Logarithm base; ``None``/``"e"`` -> nats, ``2`` -> bits, ``10`` -> dits.
    normalize:
        If True (default) renormalise ``probs`` to sum to one first.

    Returns
    -------
    float or np.ndarray
        A scalar for 1-D input, otherwise an array with ``axis`` removed.
        (Returns a ``torch.Tensor`` when a torch tensor with a single 1-D
        distribution is given and torch is installed.)
    """
    if base in ("e", None):
        log_base = None  # natural log (nats)
    elif base == "2":
        log_base = 2.0
    elif base == "10":
        log_base = 10.0
    else:
        log_base = float(base)

    was_torch = _HAS_TORCH and isinstance(probs, torch.Tensor)
    p = _to_numpy(probs)

    if p.ndim == 0:
        raise ValueError("entropy() expects at least a 1-D distribution")

    # Move the vocabulary axis last so we can use scipy on the flattened batch.
    if axis not in (-1, p.ndim - 1):
        p = np.moveaxis(p, axis, -1)

    flat = p.reshape(-1, p.shape[-1])
    out = np.empty(flat.shape[0], dtype=np.float64)

    if _HAS_SCIPY and log_base is None:
        # scipy.stats.entropy normalises internally (unless we already did) and
        # treats pk = 0 as a zero contribution -- exactly Eq. in Section 5.1.
        for i, row in enumerate(flat):
            out[i] = float(_scipy_entropy(row, base=None))
    else:
        for i, row in enumerate(flat):
            row = np.asarray(row, dtype=np.float64)
            if normalize:
                total = row.sum()
                if total > 0:
                    row = row / total
            with np.errstate(divide="ignore", invalid="ignore"):
                terms = np.where(row > 0, row * np.log(row), 0.0)
            h = float(-np.sum(terms))
            if log_base is not None:
                h = h / math.log(log_base)
            out[i] = h

    out = out.reshape(p.shape[:-1])
    if out.ndim == 0:
        value = float(out)
        return value
    if was_torch:  # pragma: no cover - convenience for torch callers
        return torch.as_tensor(out, dtype=torch.float64)
    return out


#: Alias matching the paper's ``H(p)`` notation.
shannon_entropy = entropy


def entropy_from_logits(
    logits: Any,
    axis: int = -1,
    base: Optional[Union[float, str]] = None,
    temperature: float = 1.0,
) -> Any:
    """Entropy of the softmax distribution induced by ``logits``.

    ``temperature`` is applied before the softmax (``temperature=1.0`` leaves
    the logits untouched).  This is the distribution Section 5.1 measures, with
    the addendum noting the entropy is computed *after* the CFG combination.
    """
    if temperature != 1.0:
        logits = logits / float(temperature)
    if _HAS_TORCH and isinstance(logits, torch.Tensor):
        p = torch.softmax(logits, dim=axis)
    else:
        arr = _to_numpy(logits)
        m = np.max(arr, axis=axis, keepdims=True)
        e = np.exp(arr - m)
        p = e / np.sum(e, axis=axis, keepdims=True)
    return entropy(p, axis=axis, base=base)


def entropy_from_logprobs(
    logprobs: Any,
    axis: int = -1,
    base: Optional[Union[float, str]] = None,
) -> Any:
    """Entropy computed directly from log-probabilities (numerically stable).

    ``H(p) = -sum_k exp(logp_k) * logp_k``; entries with ``logp = -inf`` are
    skipped.
    """
    lp = _to_numpy(logprobs)
    if base in ("e", None):
        log_base = None
    elif base == "2":
        log_base = 2.0
    elif base == "10":
        log_base = 10.0
    else:
        log_base = float(base)

    if axis not in (-1, lp.ndim - 1):
        lp = np.moveaxis(lp, axis, -1)
    flat = lp.reshape(-1, lp.shape[-1])
    out = np.empty(flat.shape[0], dtype=np.float64)
    for i, row in enumerate(flat):
        finite = np.isfinite(row)
        p = np.exp(row[finite])
        terms = p * row[finite]
        h = float(-np.sum(terms))
        if log_base is not None:
            h = h / math.log(log_base)
        out[i] = h
    out = out.reshape(lp.shape[:-1])
    return float(out) if out.ndim == 0 else out


def top_p_token_count(probs: Any, top_p: float = TOP_P, axis: int = -1) -> np.ndarray:
    """Number of tokens in the smallest top-``top_p`` nucleus of each distribution.

    Section 5.1: lower entropy "restricts the number of tokens in the top-p = 90%
    of the vocabulary distribution".  Implemented as the count of sorted-descending
    tokens needed to cover ``top_p`` of the probability mass (always >= 1).
    """
    p = _to_numpy(probs)
    if axis not in (-1, p.ndim - 1):
        p = np.moveaxis(p, axis, -1)
    flat = p.reshape(-1, p.shape[-1])
    counts = np.empty(flat.shape[0], dtype=np.int64)
    for i, row in enumerate(flat):
        row = np.asarray(row, dtype=np.float64)
        total = row.sum()
        if total <= 0:
            counts[i] = 0
            continue
        row = row / total
        order = np.argsort(-row, kind="mergesort")
        cumulative = np.cumsum(row[order])
        counts[i] = int(np.searchsorted(cumulative, top_p, side="left") + 1)
    counts = counts.reshape(p.shape[:-1])
    if counts.ndim == 0:
        return counts.reshape(1)
    return counts


def effective_vocab_size(probs: Any, axis: int = -1) -> Any:
    """Perplexity-style ``exp(H(p))`` -- average number of plausible tokens."""
    return np.exp(entropy(probs, axis=axis))


def mean_entropy(per_token_entropies: Any) -> float:
    """``(1/n) * sum_i H(p(x_i | x_<i))`` over completion tokens.

    This is exactly the "mean entropy" definition given in the paper's addendum.
    """
    arr = _to_numpy(per_token_entropies).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


# --------------------------------------------------------------------------- #
# Containers
# --------------------------------------------------------------------------- #


@dataclass
class EntropyStats:
    """Per-token entropy record for one (or many) completions.

    Attributes
    ----------
    per_token:
        Entropy of the vocabulary distribution at each completion step.
    n_tokens:
        Number of completion tokens ``n`` in the mean-entropy formula.
    top_p_counts:
        Optional number of tokens inside the top-p nucleus per step.
    sample_ids:
        Optional identifier per token (which P3 sample it came from) so that
        per-sample means can be recovered from a concatenated record.
    """

    per_token: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    n_tokens: int = 0
    top_p_counts: Optional[np.ndarray] = None
    sample_ids: Optional[np.ndarray] = None
    name: str = "entropy"

    # ---------------------------------------------------------------- helpers
    @property
    def mean(self) -> float:
        """Mean entropy averaged over each token (paper's Section 5.1 metric)."""
        return mean_entropy(self.per_token)

    @property
    def std(self) -> float:
        arr = _to_numpy(self.per_token).reshape(-1)
        return float(np.std(arr)) if arr.size else float("nan")

    @property
    def median(self) -> float:
        arr = _to_numpy(self.per_token).reshape(-1)
        return float(np.median(arr)) if arr.size else float("nan")

    @property
    def min(self) -> float:
        arr = _to_numpy(self.per_token).reshape(-1)
        return float(np.min(arr)) if arr.size else float("nan")

    @property
    def max(self) -> float:
        arr = _to_numpy(self.per_token).reshape(-1)
        return float(np.max(arr)) if arr.size else float("nan")

    @property
    def sem(self) -> float:
        arr = _to_numpy(self.per_token).reshape(-1)
        if arr.size < 2:
            return float("nan")
        return float(arr.std(ddof=1) / math.sqrt(arr.size))

    @property
    def mean_top_p_count(self) -> float:
        if self.top_p_counts is None or len(self.top_p_counts) == 0:
            return float("nan")
        return float(np.mean(self.top_p_counts))

    def per_sample_means(self) -> Dict[int, float]:
        """Mean entropy grouped by ``sample_ids`` (token-unweighted per sample)."""
        if self.sample_ids is None:
            return {0: self.mean}
        arr = _to_numpy(self.per_token).reshape(-1)
        ids = _to_numpy(self.sample_ids).reshape(-1)
        out: Dict[int, float] = {}
        for sid in np.unique(ids):
            out[int(sid)] = mean_entropy(arr[ids == sid])
        return out

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "n_tokens": int(self.n_tokens if self.n_tokens else len(np.reshape(self.per_token, -1))),
            "mean": self.mean,
            "std": self.std,
            "sem": self.sem,
            "median": self.median,
            "min": self.min,
            "max": self.max,
            "mean_top_p_count": self.mean_top_p_count,
            "per_token": _to_numpy(self.per_token).reshape(-1).tolist(),
        }

    def __len__(self) -> int:
        return len(np.reshape(_to_numpy(self.per_token), -1))


@dataclass
class EntropyComparison:
    """Result of comparing per-token entropies across the Section 5.1 modes."""

    mean: Dict[str, float] = field(default_factory=dict)
    stats: Dict[str, EntropyStats] = field(default_factory=dict)
    n: Dict[str, int] = field(default_factory=dict)
    mean_top_p_count: Dict[str, float] = field(default_factory=dict)
    gamma: float = ANALYSIS_GAMMA

    @property
    def delta(self) -> float:
        """``mean(vanilla) - mean(cfg)`` -- positive means CFG lowered entropy."""
        if "cfg" in self.mean and "vanilla" in self.mean:
            return float(self.mean["vanilla"] - self.mean["cfg"])
        return float("nan")

    @property
    def ratio(self) -> float:
        if "cfg" in self.mean and "vanilla" in self.mean and self.mean["cfg"]:
            return float(self.mean["vanilla"] / self.mean["cfg"])
        return float("nan")

    def check_against_paper(self, tolerance: float = ENTROPY_TOLERANCE) -> Dict[str, Any]:
        """Compare the measured means with the paper's anchors (4.7 vs 5.49)."""
        out: Dict[str, Any] = {}
        if "cfg" in self.mean:
            out["cfg"] = {
                "measured": self.mean["cfg"],
                "expected": CFG_ENTROPY_MEAN,
                "within_tolerance": abs(self.mean["cfg"] - CFG_ENTROPY_MEAN) <= tolerance,
            }
        if "vanilla" in self.mean:
            out["vanilla"] = {
                "measured": self.mean["vanilla"],
                "expected": VANILLA_ENTROPY_MEAN,
                "within_tolerance": abs(self.mean["vanilla"] - VANILLA_ENTROPY_MEAN) <= tolerance,
            }
        if "cfg" in self.mean and "vanilla" in self.mean:
            out["cfg_lower_than_vanilla"] = self.mean["cfg"] < self.mean["vanilla"]
        return out

    def as_dict(self) -> Dict[str, Any]:
        return {
            "gamma": self.gamma,
            "mean": dict(self.mean),
            "n": dict(self.n),
            "mean_top_p_count": dict(self.mean_top_p_count),
            "delta": self.delta,
            "ratio": self.ratio,
            "check": self.check_against_paper(),
        }


# --------------------------------------------------------------------------- #
# Aggregation helpers
# --------------------------------------------------------------------------- #


def sample_entropy_of_distributions(
    distributions: Sequence[Any],
    top_p: float = TOP_P,
    name: str = "entropy",
) -> EntropyStats:
    """Per-token entropy of a sequence of vocab distributions (one per step)."""
    entropies: List[float] = []
    counts: List[int] = []
    for dist in distributions:
        if dist is None:
            continue
        h = entropy(dist)
        entropies.append(float(h))
        counts.append(int(np.reshape(top_p_token_count(dist, top_p), -1)[0]))
    return EntropyStats(
        per_token=np.asarray(entropies, dtype=np.float64),
        n_tokens=len(entropies),
        top_p_counts=np.asarray(counts, dtype=np.int64) if counts else None,
        name=name,
    )


def concatenate_stats(stats: Iterable[EntropyStats]) -> EntropyStats:
    """Concatenate per-token records from several completions into one record."""
    toks: List[np.ndarray] = []
    counts: List[np.ndarray] = []
    ids: List[np.ndarray] = []
    for i, st in enumerate(stats):
        if st is None or len(st) == 0:
            continue
        arr = _to_numpy(st.per_token).reshape(-1)
        toks.append(arr)
        if st.top_p_counts is not None and len(st.top_p_counts) == arr.size:
            counts.append(_to_numpy(st.top_p_counts).reshape(-1))
        ids.append(np.full(arr.size, i, dtype=np.int64))
    if not toks:
        return EntropyStats()
    return EntropyStats(
        per_token=np.concatenate(toks),
        n_tokens=int(sum(t.size for t in toks)),
        top_p_counts=np.concatenate(counts) if len(counts) == len(toks) else None,
        sample_ids=np.concatenate(ids),
    )


def batch_mean_entropy(distributions_per_sample: Iterable[Sequence[Any]]) -> float:
    """Token-level mean entropy over many completions (addendum definition)."""
    all_stats = [
        sample_entropy_of_distributions(dists) for dists in distributions_per_sample
    ]
    merged = concatenate_stats(all_stats)
    return merged.mean


# --------------------------------------------------------------------------- #
# Model-driven analysis (Section 5.1 protocol)
# --------------------------------------------------------------------------- #


def _guided_distribution(
    logits_cond: Any,
    logits_uncond: Any,
    gamma: float,
    temperature: float = 1.0,
) -> np.ndarray:
    """Post-CFG softmax distribution ``p = softmax(uncond + gamma*(cond-uncond))``.

    CFG is applied to the *raw pre-softmax logits* (Section 2.2), before the
    temperature scaling / softmax, exactly as in ``src/cfg/logits.py``.
    """
    cond = _to_numpy(logits_cond).reshape(-1)
    uncond = _to_numpy(logits_uncond).reshape(-1)
    if gamma == 1.0:
        guided = cond
    elif gamma == 0.0:
        guided = uncond
    else:
        guided = uncond + gamma * (cond - uncond)
    if temperature != 1.0 and temperature > 0:
        guided = guided / float(temperature)
    m = np.max(guided)
    e = np.exp(guided - m)
    return e / np.sum(e)


class EntropyAnalyzer:
    """Measures the Section 5.1 sampling entropy of a CFG-guided language model.

    For each prompt the analyzer autoregressively decodes ``max_new_tokens``
    steps.  At every step it obtains both the conditional and the unconditional
    next-token logits from a :class:`~src.cfg.model_wrapper.CFGModelWrapper`
    (two forward passes over the same weights) and computes the entropy of the
    *post-CFG* vocabulary distribution.  The reported mean is the per-token
    average ``(1/n) * sum_i H(p(x_i | x_<i))``.

    Modes
    -----
    ``cfg``         raw CFG with ``gamma`` (default 1.5)       -- the CFG column
    ``vanilla``     ``gamma = 1`` (plain conditional P(y|x))
    ``unprompted``  ``gamma = 0`` (the unconditional P(x) pass)
    ``instruct``    optional second wrapper (e.g. Falcon-7b-Instruct) scored at
                    ``gamma = 1`` -- used by Section 5.2 / Appendix E.

    Sampling is greedy by default so that entropies are deterministic and
    directly comparable across modes.
    """

    def __init__(
        self,
        model_wrapper: Any = None,
        gamma: float = ANALYSIS_GAMMA,
        top_p: float = TOP_P,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        unconditional_mode: str = "empty_prefix",
        instruct_wrapper: Any = None,
        device: Optional[str] = None,
        seed: Optional[int] = 0,
        batch_size: int = 1,
    ) -> None:
        self.wrapper = model_wrapper
        self.instruct_wrapper = instruct_wrapper
        self.gamma = float(gamma)
        self.top_p = float(top_p)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.unconditional_mode = unconditional_mode
        self.device = device
        self.seed = seed
        self.batch_size = int(batch_size)

    # ------------------------------------------------------------------ utils
    def _set_mode(self, wrapper: Any, unconditional_mode: Optional[str] = None) -> None:
        mode = unconditional_mode or self.unconditional_mode
        try:
            wrapper.unconditional_mode = mode
        except Exception:  # pragma: no cover - wrapper without setter
            logger.debug("Could not set unconditional_mode on wrapper")

    def _next_pair(self, wrapper: Any, input_ids: Any, prompt_length: int) -> Tuple[Any, Any]:
        """Return ``(logits_cond, logits_uncond)`` for the next token."""
        try:
            dual = wrapper.dual_logits(input_ids, prompt_length=prompt_length, only_last=True)
            return dual.cond, dual.uncond
        except TypeError:
            # Older/alternative wrapper signature.
            dual = wrapper.dual_logits(input_ids, only_last=True)  # type: ignore[call-arg]
            return dual.cond, dual.uncond

    def _distribution_for_mode(
        self,
        logits_cond: Any,
        logits_uncond: Any,
        mode: str,
    ) -> Tuple[np.ndarray, float]:
        gamma = MODE_GAMMAS.get(mode, self.gamma)
        if mode == "cfg":
            gamma = self.gamma
        probs = _guided_distribution(logits_cond, logits_uncond, gamma, self.temperature)
        return probs, gamma

    # ------------------------------------------------------------- single pass
    def distributions_for_prompt(
        self,
        prompt: str,
        mode: str = "cfg",
        wrapper: Any = None,
        max_new_tokens: Optional[int] = None,
        do_sample: bool = False,
        seed: Optional[int] = None,
        negative_prompt: Optional[str] = None,
        prompt_length: Optional[int] = None,
    ) -> List[np.ndarray]:
        """Return the list of post-CFG vocabulary distributions along a completion."""
        w = wrapper if wrapper is not None else self.wrapper
        if w is None:
            raise ValueError(
                "EntropyAnalyzer needs a CFGModelWrapper (or a `wrapper=` argument)"
            )
        self._set_mode(w)
        n_new = self.max_new_tokens if max_new_tokens is None else int(max_new_tokens)

        input_ids = w.encode(prompt)
        if _HAS_TORCH and isinstance(input_ids, torch.Tensor):
            device = getattr(w, "device", None) or self.device or "cpu"
            input_ids = input_ids.to(device)
        p_len = int(prompt_length) if prompt_length else int(
            np.reshape(_to_numpy(input_ids).shape, -1)[-1]
        )

        neg_ids = None
        if negative_prompt:
            neg_ids = w.encode(negative_prompt)
            if _HAS_TORCH and isinstance(neg_ids, torch.Tensor):
                neg_ids = neg_ids.to(getattr(w, "device", "cpu"))

        rng = None
        if do_sample and _HAS_TORCH:
            rng = torch.Generator(device=getattr(w, "device", "cpu"))
            rng.manual_seed(int(seed if seed is not None else (self.seed or 0)))

        dists: List[np.ndarray] = []
        eos_id = getattr(w, "eos_token_id", None)
        for _ in range(n_new):
            if neg_ids is not None:
                dual = w.dual_logits(input_ids, prompt_length=p_len, only_last=True,
                                     negative_input_ids=neg_ids)
                lc, lu = dual.cond, dual.uncond
            else:
                lc, lu = self._next_pair(w, input_ids, p_len)

            probs, _ = self._distribution_for_mode(lc, lu, mode)
            dists.append(probs)

            # continue the sequence (greedy by default; deterministic analysis)
            if do_sample and _HAS_TORCH:
                tok = int(torch.multinomial(torch.as_tensor(probs), 1, generator=rng).item())
            else:
                tok = int(np.argmax(probs))
            if eos_id is not None and tok == int(eos_id):
                break
            if _HAS_TORCH and isinstance(input_ids, torch.Tensor):
                new_tok = torch.tensor([[tok]], dtype=input_ids.dtype, device=input_ids.device)
                input_ids = torch.cat([input_ids, new_tok], dim=1)
            else:  # pragma: no cover - numpy decode path
                input_ids = np.concatenate([np.asarray(input_ids), np.asarray([[tok]])], axis=1)
        return dists

    # ------------------------------------------------------------- aggregation
    def measure(
        self,
        prompts: Sequence[str],
        mode: str = "cfg",
        wrapper: Any = None,
        max_new_tokens: Optional[int] = None,
        do_sample: bool = False,
        seed: Optional[int] = None,
        negative_prompt: Optional[str] = None,
        progress: bool = False,
    ) -> EntropyStats:
        """Per-token entropy statistics over a set of prompts."""
        stats: List[EntropyStats] = []
        for i, prompt in enumerate(prompts):
            if progress:
                logger.info("[%s] %d/%d", mode, i + 1, len(prompts))
            dists = self.distributions_for_prompt(
                prompt,
                mode=mode,
                wrapper=wrapper,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                seed=seed,
                negative_prompt=negative_prompt,
            )
            stats.append(sample_entropy_of_distributions(dists, top_p=self.top_p, name=mode))
        merged = concatenate_stats(stats)
        merged.name = mode
        return merged

    def compare(
        self,
        prompts: Sequence[str],
        modes: Sequence[str] = ("cfg", "vanilla", "unprompted"),
        max_new_tokens: Optional[int] = None,
        do_sample: bool = False,
        seed: Optional[int] = None,
        progress: bool = False,
    ) -> EntropyComparison:
        """Compare the per-token entropies of CFG / vanilla / unprompted (and instruct)."""
        stats: Dict[str, EntropyStats] = {}
        for mode in modes:
            w = self.instruct_wrapper if mode == "instruct" else self.wrapper
            if w is None:
                logger.warning("Skipping mode '%s': no wrapper available", mode)
                continue
            stats[mode] = self.measure(
                prompts,
                mode=mode,
                wrapper=w,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                seed=seed,
                progress=progress,
            )

        if self.instruct_wrapper is not None and "instruct" not in stats:
            stats["instruct"] = self.measure(
                prompts,
                mode="instruct",
                wrapper=self.instruct_wrapper,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                seed=seed,
                progress=progress,
            )

        return EntropyComparison(
            mean={m: s.mean for m, s in stats.items()},
            stats=stats,
            n={m: len(s) for m, s in stats.items()},
            mean_top_p_count={m: s.mean_top_p_count for m, s in stats.items()},
            gamma=self.gamma,
        )


# --------------------------------------------------------------------------- #
# Reporting / tables
# --------------------------------------------------------------------------- #


def entropy_report(
    comparison: Union[EntropyComparison, Dict[str, Any]],
) -> Dict[str, Any]:
    """Summarise an :class:`EntropyComparison` (or plain ``{mode: mean}`` dict)."""
    if isinstance(comparison, EntropyComparison):
        report = comparison.as_dict()
    elif isinstance(comparison, dict) and "stats" in comparison:
        report = dict(comparison)
    else:  # plain mapping of mode -> mean
        means = {k: float(v) for k, v in dict(comparison).items()}
        report = {
            "gamma": ANALYSIS_GAMMA,
            "mean": means,
            "delta": float(means.get("vanilla", float("nan")) - means.get("cfg", float("nan"))),
        }
    report.setdefault("expected", {"cfg": CFG_ENTROPY_MEAN, "vanilla": VANILLA_ENTROPY_MEAN})
    return report


def format_entropy_table(
    comparison: Union[EntropyComparison, Dict[str, float]],
    modes: Sequence[str] = ENTROPY_MODES,
) -> str:
    """Render the Section 5.1 entropy comparison as plain text."""
    if isinstance(comparison, EntropyComparison):
        means = comparison.mean
        counts = comparison.mean_top_p_count
        ns = comparison.n
    else:
        means = {k: float(v) for k, v in comparison.items()}
        counts = {}
        ns = {}

    lines = [
        "Section 5.1 -- mean sampling entropy (nats) per completion token",
        f"{'mode':<24}{'mean H(p)':>12}{'n tokens':>12}{'mean |top-90%|':>18}",
        "-" * 66,
    ]
    for mode in modes:
        if mode not in means:
            continue
        lines.append(
            f"{MODE_LABELS.get(mode, mode):<24}"
            f"{means[mode]:>12.3f}"
            f"{ns.get(mode, 0):>12d}"
            f"{counts.get(mode, float('nan')):>18.1f}"
        )
    if "cfg" in means and "vanilla" in means:
        lines.append("-" * 66)
        lines.append(
            f"{'delta (vanilla - cfg)':<24}{means['vanilla'] - means['cfg']:>12.3f}"
        )
        lines.append(
            f"{'paper reference':<24}{'CFG 4.7':>12} / vanilla 5.49"
        )
    return "\n".join(lines)


def check_against_paper(
    comparison: Union[EntropyComparison, Dict[str, float]],
    tolerance: float = ENTROPY_TOLERANCE,
) -> Dict[str, Any]:
    """Convenience wrapper around :meth:`EntropyComparison.check_against_paper`."""
    if isinstance(comparison, EntropyComparison):
        return comparison.check_against_paper(tolerance=tolerance)
    means = {k: float(v) for k, v in dict(comparison).items()}
    return EntropyComparison(mean=means).check_against_paper(tolerance=tolerance)


def summarize_entropy(stats: EntropyStats) -> str:
    """One-line textual summary of an :class:`EntropyStats` record."""
    return (
        f"{stats.name}: mean H = {stats.mean:.3f} "
        f"(+-{stats.sem:.3f}) over {len(stats)} tokens, "
        f"min {stats.min:.3f}, max {stats.max:.3f}, "
        f"mean top-90% size {stats.mean_top_p_count:.1f}"
    )


# --------------------------------------------------------------------------- #
# Self-test / demo (no model required)
# --------------------------------------------------------------------------- #


def _demo() -> None:  # pragma: no cover - manual smoke test
    """Sanity-check the entropy math without loading a model."""
    rng = np.random.default_rng(0)
    # Uniform over 1000 tokens -> exactly log(1000) nats.
    uniform = np.ones(1000) / 1000.0
    assert abs(float(entropy(uniform)) - math.log(1000)) < 1e-9
    # One-hot -> zero entropy.
    onehot = np.zeros(1000)
    onehot[3] = 1.0
    assert abs(float(entropy(onehot))) < 1e-12
    # Top-90% of a uniform distribution covers all-but-one token.
    assert int(top_p_token_count(uniform)[0]) == 999
    # Sharper logits -> lower entropy.
    a = rng.normal(size=1000)
    b = rng.normal(size=1000) * 2.0
    assert float(entropy_from_logits(b)) < float(entropy_from_logits(a))
    print("entropy self-test OK", summarize_entropy(sample_entropy_of_distributions(
        [np.ones(1000) / 1000.0] * 5
    )))


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    _demo()
