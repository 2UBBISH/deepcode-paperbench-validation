"""PPLM-based toxic continuation generation for pairwise preference data.

Paper reference
---------------
Section 4.2 "Constructing Pairwise Toxic Data"::

    We build our pairwise toxicity dataset using PPLM (Dathathri et al., 2019).
    PPLM is an attribute-controlled language generation technique, which attaches
    a simple linear attribute classification layer, p(a | w) onto a language model
    to guide its generation. During generation, PPLM uses the attribute classifier
    to compute the gradients that increases the likelihood of the language model's
    output to contain the desired attribute a, and shifts the activations in such
    direction:

        p(y | a) \propto p(y) p(a | y)                                     (Eq. 1)

    To generate pairwise preference data, we use sentences from Wikitext-2 as
    prompts. For each prompt, we generate a positive sample using greedy sampling
    with GPT2, while using PPLM to generate negative (toxic) samples. We use our
    toxic probe W_Toxic as our attribute classifier to guide towards toxic outputs.
    We create 24,576 pairs of toxic and nontoxic continuations.

Appendix E, Table 9 (PPLM hyperparameters) -- reproduced verbatim in
:data:`PAPER_PPLM_HYPERPARAMETERS`::

    STEP SIZE     0.4
    TEMPERATURE   1
    TOP K         10
    NUM ITERATIONS 50
    WINDOW LENGTH 0
    HORIZON LENGTH 1
    DECAY         FALSE
    GAMMA         1
    GM SCALE      0.95
    KL SCALE      0.1

Design decisions that the paper leaves unspecified (documented so downstream
readers know what is a faithful choice and what is a default):

* The attribute classifier is the linear probe ``W_Toxic`` (``[d_model, 2]``)
  trained in :mod:`src.probe`.  PPLM's discriminator consumes the LM's hidden
  state; the probe was trained on the *last block output* residual stream
  (pre-``ln_f``), so PPLM here feeds the same representation (``pool="last"`` by
  default, with ``"mean"`` available) to keep the classifier in-distribution.
* Perturbations are applied to the past key/value tensors of every layer, with
  per-tensor gradient normalisation (PPLM's default ``loss_type=0`` scheme).
* ``num_iterations`` defaults to the paper's 50; it is expensive, so the value is
  overridable everywhere (``--num-iterations``) and ``quick`` mode uses 3.
* ``horizon_length`` defaults to 1 (one PPLM token per perturbation step).  For
  ``horizon_length > 1`` the remaining tokens of the horizon are sampled greedily
  from the perturbed distribution -- documented simplification.
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------------------
# Constants / paper hyperparameters
# --------------------------------------------------------------------------------------

#: Appendix E, Table 9 -- PPLM hyperparameters exactly as published.
PAPER_PPLM_HYPERPARAMETERS: Dict[str, Any] = {
    "step_size": 0.4,
    "temperature": 1.0,
    "top_k": 10,
    "num_iterations": 50,
    "window_length": 0,
    "horizon_length": 1,
    "decay": False,
    "gamma": 1.0,
    "gm_scale": 0.95,
    "kl_scale": 0.1,
}

#: Tokens generated per continuation (greedy GPT2 / PPLM), Section 3.3 defaults.
DEFAULT_MAX_NEW_TOKENS = 20
#: Number of preference pairs (Section 4.2).
N_PAIRS = 24_576
#: Fraction of pairs held out for validation (90:10 split).
VALID_RATIO = 0.1
#: Small constant to avoid log/div-by-zero in the GM score.
EPS = 1e-8
TOXIC_INDEX = 1

ARTIFACT_DIR = os.path.join("artifacts", "data")
DEFAULT_SHARD_DIR = os.path.join(ARTIFACT_DIR, "pairs_shards")
DEFAULT_PAIRS_PATH = os.path.join(ARTIFACT_DIR, "pairs.jsonl")


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------


@dataclass
class PPLMConfig:
    """PPLM generation configuration (Appendix E, Table 9 defaults).

    Fields default to the paper's published values; ``from_dict`` tolerates
    nested YAML layouts (``pplm:``/``train:``) and common alias names.
    """

    step_size: float = 0.4
    temperature: float = 1.0
    top_k: int = 10
    num_iterations: int = 50
    window_length: int = 0
    horizon_length: int = 1
    decay: bool = False
    gamma: float = 1.0
    gm_scale: float = 0.95
    kl_scale: float = 0.1

    # Generation / bookkeeping knobs (not in Table 9).
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    max_prompt_length: Optional[int] = None
    sample: bool = True
    seed: int = 0
    attribute_class: int = TOXIC_INDEX
    pool: str = "last"
    grad_norm: str = "unit"
    normalize_inside: bool = True
    device: Optional[str] = None
    quiet: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_size": self.step_size,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "num_iterations": self.num_iterations,
            "window_length": self.window_length,
            "horizon_length": self.horizon_length,
            "decay": self.decay,
            "gamma": self.gamma,
            "gm_scale": self.gm_scale,
            "kl_scale": self.kl_scale,
            "max_new_tokens": self.max_new_tokens,
            "max_prompt_length": self.max_prompt_length,
            "sample": self.sample,
            "seed": self.seed,
            "attribute_class": self.attribute_class,
            "pool": self.pool,
            "grad_norm": self.grad_norm,
            "normalize_inside": self.normalize_inside,
            "device": self.device,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]] = None, **overrides: Any) -> "PPLMConfig":
        data = _flatten_config(data or {})
        aliases = {
            "steps": "step_size",
            "stepsize": "step_size",
            "step": "step_size",
            "topk": "top_k",
            "top-k": "top_k",
            "numiterations": "num_iterations",
            "num-iterations": "num_iterations",
            "iterations": "num_iterations",
            "num_iter": "num_iterations",
            "windowlength": "window_length",
            "window": "window_length",
            "horizonlength": "horizon_length",
            "horizon": "horizon_length",
            "gmscale": "gm_scale",
            "gm-scale": "gm_scale",
            "klscale": "kl_scale",
            "kl-scale": "kl_scale",
            "max_new_tokens": "max_new_tokens",
            "maxnewtokens": "max_new_tokens",
            "new_tokens": "max_new_tokens",
        }
        kwargs: Dict[str, Any] = {}
        valid = set(cls.__dataclass_fields__.keys())  # type: ignore[attr-defined]
        for key, value in data.items():
            norm = str(key).strip().lower().replace(" ", "_")
            norm = aliases.get(norm, norm)
            if norm in valid and value is not None:
                kwargs[norm] = value
        for key, value in overrides.items():
            if norm := str(key).strip().lower().replace(" ", "_"):
                norm = aliases.get(norm, norm)
                if norm in valid and value is not None:
                    kwargs[norm] = value
        return cls(**kwargs)

    def quick(self) -> "PPLMConfig":
        """Return a fast smoke-test variant of this configuration."""
        clone = PPLMConfig.from_dict(self.to_dict())
        clone.num_iterations = min(int(self.num_iterations), 3)
        clone.max_new_tokens = min(int(self.max_new_tokens), 8)
        clone.top_k = min(int(self.top_k or 10), 10)
        return clone


def _flatten_config(data: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten nested YAML config sections (``pplm`` / ``train`` / ``generation``)."""
    if not isinstance(data, dict):
        return {}
    nested_keys = ("pplm", "train", "training", "generation", "generate", "hyperparameters")
    for key in nested_keys:
        section = data.get(key)
        if isinstance(section, dict):
            merged = dict(data)
            merged.pop(key)
            merged.update(section)
            return merged
    return dict(data)


def load_pplm_config(path: Optional[str] = None, **overrides: Any) -> PPLMConfig:
    """Load a :class:`PPLMConfig` from YAML (optional) plus overrides."""
    data: Dict[str, Any] = {}
    if path and os.path.exists(path):
        try:
            import yaml  # type: ignore

            with open(path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        except Exception:
            data = {}
    return PPLMConfig.from_dict(data, **overrides)


# --------------------------------------------------------------------------------------
# Low level helpers (HF cache handling / sampling)
# --------------------------------------------------------------------------------------


def _legacy_past(past: Any) -> List[Any]:
    """Return the past as a flat list ``[k_0, v_0, k_1, v_1, ...]`` of tensors.

    Supports both the legacy tuple-of-tuples layout and the modern ``Cache``
    objects exposed by recent ``transformers`` releases.
    """
    if past is None:
        return []
    if hasattr(past, "to_legacy_cache"):
        try:
            past = past.to_legacy_cache()
        except Exception:  # pragma: no cover - defensive
            pass
    flat: List[Any] = []
    for layer in past:
        if isinstance(layer, (tuple, list)):
            for tensor in layer:
                flat.append(tensor)
        else:  # already flat
            flat.append(layer)
    return flat


def _from_legacy(flat: Sequence[Any]) -> Tuple[Tuple[Any, Any], ...]:
    """Rebuild nested ``((k, v), ...)`` past structure from a flat list."""
    nested: List[Tuple[Any, Any]] = []
    for i in range(0, len(flat), 2):
        nested.append((flat[i], flat[i + 1]))
    return tuple(nested)


def _resolve_device(model: Any, device: Optional[str] = None):
    import torch

    if device is not None:
        return torch.device(device)
    try:
        return next(model.parameters()).device
    except Exception:  # pragma: no cover - defensive
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def top_k_filtering(logits: Any, top_k: Optional[int]) -> Any:
    """Zero out all logits except the top-k (PPLM's ``top_k_filtering``)."""
    import torch

    if top_k is None or top_k <= 0:
        return logits
    top_k = min(int(top_k), logits.size(-1))
    values, _ = torch.topk(logits, top_k, dim=-1)
    min_value = values[..., -1].unsqueeze(-1)
    return torch.where(logits < min_value, torch.full_like(logits, -1e10), logits)


def geometric_mean_score(
    perturbed_probs: Any,
    unperturbed_probs: Any,
    gm_scale: float = 0.95,
) -> Any:
    """Geometric-mean interpolation of perturbed/unperturbed next-token probs.

    PPLM combines the distributions as ``p_pert ** gm_scale * p_unpert ** (1 - gm_scale)``
    and re-normalises; this keeps the fluent unperturbed distribution in the loop.
    """
    import torch

    gm = (perturbed_probs ** gm_scale) * (unperturbed_probs ** (1.0 - gm_scale))
    denom = gm.sum(dim=-1, keepdim=True)
    return gm / torch.clamp(denom, min=EPS)


def _normalise_grads(grads: Sequence[Any], mode: str = "unit"):
    """Normalise PPLM gradients (per-tensor) so the step size is scale-free."""
    import torch

    if mode in ("none", None):
        return list(grads)
    out = []
    for grad in grads:
        if grad is None:
            out.append(None)
            continue
        if mode == "max":
            # PPLM's loss_type=0 scheme: divide by the max gradient norm.
            norm = grad.norm()
        else:  # "unit"
            norm = grad.norm()
        denom = torch.clamp(norm, min=1e-8)
        out.append(grad / denom)
    return out


def _kl_divergence(perturbed_logits: Any, unperturbed_logits: Any) -> Any:
    """Forward KL(unperturbed || perturbed) on the final position (PPLM's KL term)."""
    import torch
    import torch.nn.functional as F

    unpert_probs = F.softmax(unperturbed_logits.float(), dim=-1)
    perturb_log_probs = F.log_softmax(perturbed_logits.float(), dim=-1)
    return F.kl_div(perturb_log_probs, unpert_probs, reduction="batchmean")


# --------------------------------------------------------------------------------------
# The PPLM generator
# --------------------------------------------------------------------------------------


class PPLMGenerator:
    """PPLM toxic-continuation generator using ``W_Toxic`` as attribute classifier.

    Parameters
    ----------
    model, tokenizer:
        A causal LM (GPT2-medium) and its tokenizer.
    probe / classifier:
        The linear toxicity probe ``W_Toxic`` (``src.probe.ToxicityProbe``) used as
        PPLM's attribute classifier ``p(a|w)``.  If ``None`` the probe is loaded
        from ``probe_path`` (or trained on demand via ``src.probe.load_or_train_probe``).
    config:
        A :class:`PPLMConfig`; keyword overrides are accepted as well.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        probe: Any = None,
        config: Optional[PPLMConfig] = None,
        device: Optional[str] = None,
        probe_path: Optional[str] = None,
        auto_load_probe: bool = True,
        **overrides: Any,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = config or PPLMConfig.from_dict(overrides)
        if device is not None:
            self.cfg.device = device
        self.device = _resolve_device(model, self.cfg.device)
        self._probe = probe
        self._probe_path = probe_path
        self._auto_load_probe = auto_load_probe

    # -- probe handling ------------------------------------------------------------

    @property
    def probe(self) -> Any:
        if self._probe is None:
            self._probe = _load_probe(self._probe_path, auto=self._auto_load_probe)
        return self._probe

    @probe.setter
    def probe(self, value: Any) -> None:
        self._probe = value

    # -- encoding / forward --------------------------------------------------------

    def encode(self, text: str) -> Any:
        import torch

        enc = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"]
        if self.cfg.max_prompt_length:
            input_ids = input_ids[:, -int(self.cfg.max_prompt_length) :]
        return input_ids.to(self.device)

    def _forward(
        self,
        input_ids: Any,
        past: Any = None,
        use_cache: bool = True,
        want_hidden: bool = False,
    ) -> Any:
        """Forward pass; tolerates transformers versions with/without ``Cache`` API."""
        kwargs: Dict[str, Any] = {
            "input_ids": input_ids,
            "use_cache": use_cache,
            "output_hidden_states": want_hidden,
        }
        if past is not None:
            cache = past
            try:
                from transformers.cache_utils import DynamicCache  # type: ignore

                if not isinstance(past, DynamicCache):
                    cache = DynamicCache.from_legacy_cache(past) if hasattr(
                        DynamicCache, "from_legacy_cache"
                    ) else past
            except Exception:  # pragma: no cover - older/newer transformers layouts
                cache = past
            kwargs["past_key_values"] = cache
        return self.model(**kwargs)

    def _classifier_features(self, hidden_states: Sequence[Any] = None, hidden: Any = None, attention_mask: Any = None) -> Any:
        """Pool hidden states for the probe (``l-mid``/last block output semantics)."""
        import torch

        if hidden is None:
            hidden = hidden_states[-1]
        # PPLM perturbs only the final position; use the final token's block output.
        if self.cfg.pool == "mean" and hidden.size(1) > 1:
            if attention_mask is None:
                return hidden.mean(dim=1)
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            return (hidden * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-6)
        return hidden[:, -1, :]

    def attribute_loss(
        self,
        hidden: Any,
        perturbed_logits: Any = None,
        unperturbed_logits: Any = None,
    ) -> Any:
        """PPLM objective: ``-log p(a|y) + kl_scale * KL(p_pert || p_unpert)``."""
        import torch

        probe = self.probe
        features = self._classifier_features(hidden=hidden)
        probs = probe.toxic_probability(features) if hasattr(probe, "toxic_probability") else _probe_probs(probe, features)
        probs = torch.clamp(probs, min=EPS)
        if self.cfg.attribute_class == 0:
            loss = -torch.log(torch.clamp(1.0 - probs, min=EPS)).mean()
        else:
            loss = -torch.log(probs).mean()
        if perturbed_logits is not None and unperturbed_logits is not None and self.cfg.kl_scale > 0:
            loss = loss + float(self.cfg.kl_scale) * _kl_divergence(perturbed_logits, unperturbed_logits)
        return loss

    # -- PPLM perturbation ---------------------------------------------------------

    def perturb_past(
        self,
        past: Any,
        last_input_ids: Any,
        unperturbed_logits: Any,
        num_iterations: Optional[int] = None,
        return_info: bool = False,
    ) -> Any:
        """PPLM's ``perturb_past``: shift activations towards the toxic attribute.

        Returns ``(perturbed_past, perturbed_logits, loss_history)`` where
        ``perturbed_past`` is the accumulated (shifted) past used for decoding.
        """
        import torch

        cfg = self.cfg
        num_iterations = int(cfg.num_iterations if num_iterations is None else num_iterations)
        original = [p.detach() for p in _legacy_past(past)]
        if not original:
            return past, unperturbed_logits, []

        seq_len = int(original[0].shape[-2])
        window_length = int(cfg.window_length or 0)
        if window_length > 0 and seq_len > window_length:
            window = torch.zeros(seq_len, dtype=original[0].dtype, device=original[0].device)
            window[-window_length:] = 1.0
        else:
            window = torch.ones(seq_len, dtype=original[0].dtype, device=original[0].device)

        if cfg.decay and window_length > 0:
            decay_mask = torch.linspace(
                1.0 / (window_length + 1), 1.0, window_length, device=window.device, dtype=window.dtype
            )
            window = window.clone()
            window[-window_length:] = window[-window_length:] * decay_mask

        applied = [torch.zeros_like(p) for p in original]
        grad_accum = [torch.zeros_like(p) for p in original]
        loss_history: List[float] = []
        gamma = float(cfg.gamma)

        for _ in range(num_iterations):
            curr = [g.detach().clone().requires_grad_(True) for g in grad_accum]
            for acc, c in zip(applied, curr):
                acc.data = acc.data + gamma * c.data * window.to(acc.dtype)
            effective = [p + a for p, a in zip(original, applied)]

            out = self._forward(
                last_input_ids, past=_from_legacy(effective), use_cache=True, want_hidden=True
            )
            logits = out.logits
            hidden = out.hidden_states[-1] if getattr(out, "hidden_states", None) else None
            loss = self.attribute_loss(hidden, logits, unperturbed_logits)

            grads = torch.autograd.grad(loss, curr, retain_graph=False, allow_unused=True)
            grads = [
                torch.zeros_like(g) if gr is None else gr for g, gr in zip(grad_accum, grads)
            ]
            grads = _normalise_grads(grads, cfg.grad_norm)
            for acc, gr in zip(grad_accum, grads):
                acc.data = acc.data + gr.data
            loss_history.append(float(loss.detach().cpu()))

            del out, logits, hidden, effective

        perturbed = [p + a for p, a in zip(original, applied)]
        with torch.no_grad():
            out = self._forward(last_input_ids, past=_from_legacy(perturbed), use_cache=False, want_hidden=False)
            perturbed_logits = out.logits[:, -1, :]
        if return_info:
            return perturbed, perturbed_logits, loss_history
        return perturbed, perturbed_logits, loss_history

    # -- next-token sampling -------------------------------------------------------

    def next_token(
        self,
        input_ids: Any,
        seed: Optional[int] = None,
        num_iterations: Optional[int] = None,
        return_info: bool = False,
    ) -> Any:
        """Return the next token id under PPLM guidance (GM of perturbed/unperturbed)."""
        import torch

        cfg = self.cfg
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(seed))

        with torch.no_grad():
            out = self._forward(input_ids, use_cache=True, want_hidden=True)
            past = out.past_key_values
            unperturbed_logits = out.logits[:, -1, :]

        last_ids = input_ids[:, -1:]
        past_list, perturbed_logits, loss_history = self.perturb_past(
            past, last_ids, unperturbed_logits, num_iterations=num_iterations
        )

        temperature = float(cfg.temperature if cfg.temperature else 1.0)
        unperturbed_probs = torch.softmax(unperturbed_logits / temperature, dim=-1)
        perturbed_probs = torch.softmax(perturbed_logits / temperature, dim=-1)
        gm = geometric_mean_score(perturbed_probs, unperturbed_probs, float(cfg.gm_scale))
        gm = gm ** (1.0 / temperature)
        gm = gm / torch.clamp(gm.sum(dim=-1, keepdim=True), min=EPS)

        filtered = top_k_filtering(gm, cfg.top_k)
        if cfg.sample and float(cfg.temperature) > 0:
            probs = torch.softmax(filtered, dim=-1).squeeze(0)
            token = torch.multinomial(probs, num_samples=1, generator=generator)
        else:
            token = torch.argmax(filtered, dim=-1)

        token_id = int(token.reshape(-1)[0].item())
        if return_info:
            return token_id, {
                "loss_iterations": loss_history,
                "toxic_probability": None,
                "num_iterations": int(cfg.num_iterations if num_iterations is None else num_iterations),
            }
        return token_id

    # -- public generation API -----------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        seed: Optional[int] = None,
        num_iterations: Optional[int] = None,
        return_info: bool = False,
    ) -> Any:
        """Generate a toxic continuation for ``prompt`` (prompt text is stripped)."""
        cfg = self.cfg
        if max_new_tokens is None:
            max_new_tokens = int(cfg.max_new_tokens)
        max_new_tokens = max(0, int(max_new_tokens))
        if seed is None:
            seed = int(cfg.seed or 0)

        input_ids = self.encode(prompt)
        prompt_len = int(input_ids.shape[1])
        new_tokens: List[int] = []
        losses: List[List[float]] = []

        for step in range(max_new_tokens):
            horizon = max(1, int(cfg.horizon_length or 1))
            if horizon > 1 and new_tokens:
                # Beyond the first token of a horizon, sample greedily (documented simplification).
                with _no_grad_ctx():
                    out = self._forward(input_ids, use_cache=False)
                    token_id = int(out.logits[:, -1, :].argmax(dim=-1).item())
                info = {"loss_iterations": [], "horizon_greedy": True, "num_iterations": 0}
            else:
                token_id, info = self.next_token(
                    input_ids,
                    seed=seed * 1_000_003 + step,
                    num_iterations=num_iterations,
                    return_info=True,
                )
            new_tokens.append(token_id)
            losses.append(info.get("loss_iterations") or [])
            input_ids = _cat(input_ids, token_id, self.device)
            if _is_eos(self.tokenizer, token_id):
                break

        continuation = self.tokenizer.decode(new_tokens, skip_special_tokens=False) if new_tokens else ""
        if not return_info:
            return continuation
        return {
            "prompt": prompt,
            "continuation": continuation,
            "token_ids": new_tokens,
            "n_prompt_tokens": prompt_len,
            "loss_history": losses,
            "config": cfg.to_dict(),
        }


# --------------------------------------------------------------------------------------
# Helpers used above
# --------------------------------------------------------------------------------------


def _probe_probs(probe: Any, features: Any) -> Any:
    """Fallback probability extraction for generic linear attribute classifiers."""
    import torch

    if hasattr(probe, "probabilities"):
        return probe.probabilities(features)[..., TOXIC_INDEX]
    if hasattr(probe, "forward"):
        logits = probe.forward(features)
    elif callable(probe):
        logits = probe(features)
    else:  # pragma: no cover - defensive
        raise TypeError("Unsupported attribute classifier: %r" % (type(probe),))
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    return torch.softmax(logits.float(), dim=-1)[..., TOXIC_INDEX]


def _cat(input_ids: Any, token_id: int, device: Any) -> Any:
    import torch

    token = torch.tensor([[int(token_id)]], dtype=input_ids.dtype, device=getattr(input_ids, "device", device))
    return torch.cat([input_ids, token], dim=-1)


def _is_eos(tokenizer: Any, token_id: int) -> bool:
    eos = getattr(tokenizer, "eos_token_id", None)
    return eos is not None and int(token_id) == int(eos)


class _no_grad_ctx:
    def __enter__(self):
        import torch

        self._ctx = torch.no_grad()
        return self._ctx.__enter__()

    def __exit__(self, *exc: Any) -> None:
        return self._ctx.__exit__(*exc)


def _load_probe(probe_path: Optional[str], auto: bool = True) -> Any:
    """Load ``W_Toxic`` from disk (optionally training it if missing)."""
    try:
        from .probe import load_or_train_probe, probe_exists, default_probe_path
    except Exception:  # pragma: no cover - package-relative fallback
        try:  # type: ignore
            from src.probe import (  # noqa: F401
                load_or_train_probe,
                probe_exists,
                default_probe_path,
            )
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("src.probe is required for PPLM attribute guidance") from exc

    path = probe_path or default_probe_path()
    if probe_exists(path):
        return load_or_train_probe(path)
    if not auto:
        raise FileNotFoundError(f"Toxicity probe not found at {path!r}")
    return load_or_train_probe(path, force=False)


# --------------------------------------------------------------------------------------
# Greedy (non-toxic / preferred) generation
# --------------------------------------------------------------------------------------


def greedy_continuation(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    device: Optional[str] = None,
    max_prompt_length: Optional[int] = None,
    return_ids: bool = False,
) -> Any:
    """Greedy GPT2 continuation -- the *positive / non-toxic* sample (Section 4.2)."""
    import torch

    dev = _resolve_device(model, device)
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(dev)
    if max_prompt_length:
        input_ids = input_ids[:, -int(max_prompt_length) :]
    with torch.no_grad():
        out = model.generate(
            input_ids,
            do_sample=False,
            num_beams=1,
            max_new_tokens=int(max_new_tokens),
            pad_token_id=getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", 0),
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
        )
    new_ids = out[0, input_ids.shape[1] :].tolist()
    text = tokenizer.decode(new_ids, skip_special_tokens=False)
    return (text, new_ids) if return_ids else text


def greedy_continuations(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    batch_size: int = 16,
    device: Optional[str] = None,
    verbose: bool = False,
) -> List[str]:
    """Batched greedy continuations (reuses the canonical eval helper when available)."""
    try:
        from .eval.toxicity import generate_continuations as _gen

        generations, _ = _gen(
            model,
            tokenizer,
            list(prompts),
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            device=device,
            progress=verbose,
        )
        return generations
    except Exception:
        return [
            greedy_continuation(model, tokenizer, p, max_new_tokens=max_new_tokens, device=device)
            for p in prompts
        ]


# --------------------------------------------------------------------------------------
# Pair construction / dataset generation
# --------------------------------------------------------------------------------------


def build_pair(
    prompt: str,
    preferred: str,
    non_preferred: str,
    index: int = -1,
    prompt_toxicity: float = float("nan"),
    **meta: Any,
) -> Any:
    """Create a :class:`data.pairwise.PairExample` (falls back to a dict)."""
    try:
        from . import _pairwise as pw  # type: ignore  # pragma: no cover
    except Exception:
        pw = None
    if pw is None:
        try:
            from data import pairwise as pw  # type: ignore
        except Exception:  # pragma: no cover - defensive
            pw = None
    if pw is not None and hasattr(pw, "make_pair"):
        return pw.make_pair(
            prompt,
            preferred,
            non_preferred,
            index=index,
            source="pplm",
            prompt_toxicity=prompt_toxicity,
            meta=meta or {},
        )
    return {
        "prompt": prompt,
        "preferred": preferred,
        "non_preferred": non_preferred,
        "index": index,
        "source": "pplm",
        **meta,
    }


def generate_pair(
    prompt: str,
    model: Any,
    tokenizer: Any,
    generator: Optional[PPLMGenerator] = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    seed: int = 0,
    index: int = -1,
    num_iterations: Optional[int] = None,
    device: Optional[str] = None,
) -> Any:
    """Generate one ``(prompt, non-toxic greedy, toxic PPLM)`` preference pair."""
    preferred = greedy_continuation(
        model, tokenizer, prompt, max_new_tokens=max_new_tokens, device=device
    )
    if generator is None:
        generator = PPLMGenerator(model, tokenizer)
    non_preferred = generator.generate(
        prompt, max_new_tokens=max_new_tokens, seed=seed, num_iterations=num_iterations
    )
    return build_pair(prompt, preferred, non_preferred, index=index, seed=seed, max_new_tokens=max_new_tokens)


def generate_pairs(
    prompts: Sequence[str],
    model: Any,
    tokenizer: Any,
    generator: Optional[PPLMGenerator] = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    seed: int = 0,
    start_index: int = 0,
    num_iterations: Optional[int] = None,
    device: Optional[str] = None,
    callback: Optional[Callable[[int, Any], None]] = None,
    verbose: bool = False,
) -> Iterator[Any]:
    """Yield preference pairs for ``prompts`` (deterministic per-prompt seeds).

    ``start_index`` allows resuming an interrupted generation run: prompts before
    ``start_index`` are skipped so generation can be sharded across processes.
    """
    if generator is None:
        generator = PPLMGenerator(model, tokenizer, device=device)
    ordered = list(prompts)
    if start_index > 0:
        ordered = ordered[start_index:]
    for offset, prompt in enumerate(ordered):
        index = start_index + offset
        pair_seed = int(seed) + index
        pair = generate_pair(
            prompt,
            model,
            tokenizer,
            generator=generator,
            max_new_tokens=max_new_tokens,
            seed=pair_seed,
            index=index,
            num_iterations=num_iterations,
            device=device,
        )
        if callback is not None:
            callback(index, pair)
        elif verbose:
            print(f"[pplm] pair {index} generated", flush=True)
        yield pair


def generate_and_save(
    prompts: Sequence[str],
    model: Any,
    tokenizer: Any,
    generator: Optional[PPLMGenerator] = None,
    shard_dir: str = DEFAULT_SHARD_DIR,
    out_path: str = DEFAULT_PAIRS_PATH,
    shard_size: int = 512,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    seed: int = 0,
    num_iterations: Optional[int] = None,
    device: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Generate pairs with resumable sharding, then merge into one JSONL artifact.

    Generation is incremental: pairs are appended to numbered shards as they are
    produced so an interrupted run resumes from ``count_existing_pairs``.
    """
    from data import pairwise as pw  # lazy import; provides sharding helpers

    os.makedirs(shard_dir, exist_ok=True)
    existing = pw.count_existing_pairs(shard_dir)
    total = len(prompts)

    pairs: List[Any] = []
    written: List[str] = []
    for index, pair in enumerate(
        generate_pairs(
            prompts,
            model,
            tokenizer,
            generator=generator,
            max_new_tokens=max_new_tokens,
            seed=seed,
            start_index=existing,
            num_iterations=num_iterations,
            device=device,
        ),
        start=existing,
    ):
        pairs.append(pair)
        if len(pairs) >= shard_size:
            written.extend(pw.append_pairs(pairs, shard_dir=shard_dir, shard_size=shard_size))
            pairs = []
            if verbose:
                print(f"[pplm] {index + 1}/{total} pairs persisted", flush=True)
    if pairs:
        written.extend(pw.append_pairs(pairs, shard_dir=shard_dir, shard_size=shard_size))

    merged = pw.merge_shards(shard_dir, out_path=out_path)
    n_written = pw.count_existing_pairs(shard_dir)
    return {
        "n_prompts": total,
        "n_pairs": n_written,
        "shards": written,
        "shard_dir": shard_dir,
        "pairs_path": merged,
        "config": (generator.cfg.to_dict() if generator is not None else None),
    }


def pair_statistics(pairs: Sequence[Any]) -> Dict[str, Any]:
    """Lightweight descriptive stats for a generated pair set."""
    lengths_pos, lengths_neg, n_equal = [], [], 0
    for pair in pairs:
        pos = _get(pair, "preferred", "chosen")
        neg = _get(pair, "non_preferred", "rejected")
        lengths_pos.append(len(str(pos).split()))
        lengths_neg.append(len(str(neg).split()))
        if str(pos).strip() == str(neg).strip():
            n_equal += 1

    def _mean(values: Sequence[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    return {
        "n_pairs": len(pairs),
        "mean_preferred_words": _mean(lengths_pos),
        "mean_non_preferred_words": _mean(lengths_neg),
        "identical_pairs": n_equal,
    }


def _get(obj: Any, *keys: str) -> Any:
    for key in keys:
        if isinstance(obj, dict) and key in obj:
            return obj[key]
        if hasattr(obj, key):
            return getattr(obj, key)
    return ""


# --------------------------------------------------------------------------------------
# Smoke test
# --------------------------------------------------------------------------------------


def _main() -> int:  # pragma: no cover - manual smoke test
    import argparse

    parser = argparse.ArgumentParser(description="PPLM toxic-continuation smoke test")
    parser.add_argument("--prompt", default="The history of the city begins with")
    parser.add_argument("--model", default="openai-community/gpt2-medium")
    parser.add_argument("--max-new-tokens", type=int, default=5)
    parser.add_argument("--num-iterations", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    from src.model_utils import load_model

    model, tokenizer = load_model(args.model, device=args.device)
    cfg = PPLMConfig()
    if args.quick:
        cfg = cfg.quick()
    if args.num_iterations is not None:
        cfg.num_iterations = args.num_iterations
    generator = PPLMGenerator(model, tokenizer, config=cfg)
    out = generator.generate(
        args.prompt, max_new_tokens=args.max_new_tokens, return_info=True
    )
    print("PPLM continuation:", repr(out["continuation"]))
    print("greedy continuation:", repr(greedy_continuation(model, tokenizer, args.prompt, args.max_new_tokens)))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
