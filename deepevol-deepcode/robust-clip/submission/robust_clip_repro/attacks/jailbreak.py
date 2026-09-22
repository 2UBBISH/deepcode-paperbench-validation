"""Universal targeted visual jailbreak attack (Qi et al., 2023) for LLaVA-1.5 7B.

Source
------
Addendum (relevant clarifications, quoted):

    "The jailbreaking implementation is taken from The codebase for the paper
     from Qi et al. (2023)]
     (https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/blob/main/
      llava_llama_2_utils/visual_attacker.py).
     Note, they implement it for minigpt and LLaVA-LLaMA-2 (13B), so this needs
     to be adapted to work on LlaVA-1.5 (7B). For the attacks, 5000 iterations and
     alpha = 1/255 is used, as Qi et al. In the jailbreaking attacks, a single
     image ... is used for the attacks. ... The PGD in the attacks doesn't use
     momentum. The adversarial attacks are universal targeted attacks based on
     these harmful target strings [derogatory_corpus.csv]."

Implemented invariants (asserted in :meth:`JailbreakConfig.__post_init__`):

* exactly ``5000`` optimisation iterations,
* step size ``alpha = 1/255``,
* **no** momentum (``momentum == 0.0``),
* a single source image (``single_source_image=True``) -- the perturbation is
  universal, i.e. one perturbation shared by every harmful target string,
* targeted objective against the harmful strings read from
  ``derogatory_corpus.csv``.

Values the Addendum does *not* state
------------------------------------
``eps`` (l_inf budget), the loss function and the perturbation initialisation are
not stated by the Addendum; the plan instructs to copy those exactly from the
upstream ``visual_attacker.py``.  Because the upstream file cannot be fetched at
import time, the values used here are declared explicitly as ``UPSTREAM_*``
constants below and every one of them can be overridden through
:class:`JailbreakConfig`.  :meth:`JailbreakAttack.summary` reports which values
are mandated by the Addendum vs. inherited from upstream so callers can log them
instead of silently inventing paper values.

The perturbation is always stored with the precision policy of
``utils/precision.py`` (int16 for half precision, int32 for single precision)
and is projected onto the l_inf ball around the **raw, non-normalized** pixels.
"""

from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from ..utils.precision import (
    QUANT_SCALE,
    decode_perturbation,
    encode_perturbation,
    float_dtype_for_precision,
    int_dtype_for_precision,
    load_perturbation,
    store_perturbation,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Addendum-mandated configuration
# --------------------------------------------------------------------------- #
JAILBREAK_ITERATIONS: int = 5000
"""Addendum: "For the attacks, 5000 iterations and alpha = 1/255 is used"."""

JAILBREAK_ALPHA: float = 1.0 / 255.0
"""Addendum: "alpha = 1/255"."""

JAILBREAK_MOMENTUM: float = 0.0
"""Addendum: "The PGD in the attacks doesn't use momentum." (vs. 0.9 for PGD)."""

UNSPECIFIED = "UNSPECIFIED_BY_ADDENDUM"
"""Marker for hyper-parameters the Addendum does not state."""

# Upstream (Qi et al. visual_attacker.py) values -- NOT stated by the Addendum.
UPSTREAM_EPS: float = 1.0 / 255.0
"""l_inf budget of the upstream visual attacker (upstream default, not Addendum)."""

UPSTREAM_LOSS: str = "targeted_ce"
"""Upstream optimises the cross-entropy of the harmful target continuation."""

UPSTREAM_INIT: str = "uniform"
"""Upstream initialises the perturbation uniformly inside the l_inf ball."""

UPSTREAM_OPTIMIZER: str = "adam"
"""Upstream uses Adam with ``lr == alpha`` on the single perturbation tensor."""

UPSTREAM_SOURCE_IMAGE: str = "clean.jpeg"
"""Addendum: "In the jailbreaking attacks, a single image ... clean.jpeg is used"."""

DEROGATORY_CORPUS_URL: str = (
    "https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/"
    "blob/main/harmful_corpus/derogatory_corpus.csv"
)
CLEAN_IMAGE_URL: str = (
    "https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/"
    "blob/main/adversarial_images/clean.jpeg"
)
MANUAL_HARMFUL_INSTRUCTIONS_URL: str = (
    "https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/"
    "blob/main/harmful_corpus/manual_harmful_instructions.csv"
)

SUPPORTED_LOSSES: Tuple[str, ...] = ("targeted_ce", "targeted_nll", "mse")
SUPPORTED_OPTIMIZERS: Tuple[str, ...] = ("adam", "sgd", "sign_sgd")
SUPPORTED_INITS: Tuple[str, ...] = ("uniform", "zeros", "gaussian")

HEADER_TOKENS = {"text", "target", "targets", "string", "strings", "prompt"}


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class JailbreakConfig:
    """Configuration of the universal targeted visual jailbreak attack.

    The first block of fields is fixed by the Addendum; the second block is
    inherited from the upstream Qi et al. ``visual_attacker.py`` (the Addendum is
    silent about them) and is therefore fully overridable.
    """

    # ---- Addendum-mandated -------------------------------------------------
    iterations: int = JAILBREAK_ITERATIONS
    alpha: float = JAILBREAK_ALPHA
    momentum: float = JAILBREAK_MOMENTUM
    targeted: bool = True
    single_source_image: bool = True

    # ---- Upstream defaults (Addendum silent -> configurable) ---------------
    eps: float = UPSTREAM_EPS
    loss: str = UPSTREAM_LOSS
    init: str = UPSTREAM_INIT
    optimizer: str = UPSTREAM_OPTIMIZER
    random_start: bool = True
    micro_batch_targets: int = 1
    clamp: Optional[Tuple[float, float]] = (0.0, 1.0)

    # ---- Storage / bookkeeping --------------------------------------------
    precision: str = "single"
    quant_scale: float = QUANT_SCALE
    seed: Optional[int] = None
    device: Optional[Union[str, torch.device]] = None
    verbose: bool = True
    log_every: int = 500

    # ---- Provenance bookkeeping -------------------------------------------
    source_image: str = UPSTREAM_SOURCE_IMAGE
    target_corpus: str = "derogatory_corpus.csv"
    provenance: Dict[str, str] = field(
        default_factory=lambda: {
            "iterations": "Addendum: 5000 iterations",
            "alpha": "Addendum: 1/255",
            "momentum": "Addendum: no momentum (0.0)",
            "single_source_image": "Addendum: single image clean.jpeg",
            "targeted": "Addendum: universal targeted attack",
            "eps": UNSPECIFIED + " (upstream visual_attacker.py default used)",
            "loss": UNSPECIFIED + " (upstream visual_attacker.py default used)",
            "init": UNSPECIFIED + " (upstream visual_attacker.py default used)",
            "optimizer": UNSPECIFIED + " (upstream visual_attacker.py default used)",
            "clamp": UNSPECIFIED + " (upstream default [0, 1] pixel range)",
            "micro_batch_targets": UNSPECIFIED + " (default 1 target per step)",
        }
    )

    def __post_init__(self) -> None:
        # Addendum invariants -- fail loudly if a caller tampers with them.
        if int(self.iterations) != JAILBREAK_ITERATIONS:
            raise ValueError(
                f"Addendum mandates {JAILBREAK_ITERATIONS} iterations for the jailbreak "
                f"attack, got {self.iterations}."
            )
        if abs(float(self.alpha) - JAILBREAK_ALPHA) > 1e-12:
            raise ValueError(
                f"Addendum mandates alpha = 1/255 ({JAILBREAK_ALPHA}), got {self.alpha}."
            )
        if float(self.momentum) != 0.0:
            raise ValueError(
                "Addendum: the PGD in the jailbreak attacks doesn't use momentum, "
                f"got momentum={self.momentum}."
            )
        if not self.single_source_image:
            raise ValueError("Addendum: a single source image (clean.jpeg) is used.")
        if not self.targeted:
            raise ValueError("Addendum: the adversarial attacks are universal *targeted* attacks.")
        if self.loss not in SUPPORTED_LOSSES:
            raise ValueError(f"loss must be one of {SUPPORTED_LOSSES}, got {self.loss!r}.")
        if self.optimizer not in SUPPORTED_OPTIMIZERS:
            raise ValueError(
                f"optimizer must be one of {SUPPORTED_OPTIMIZERS}, got {self.optimizer!r}."
            )
        if self.init not in SUPPORTED_INITS:
            raise ValueError(f"init must be one of {SUPPORTED_INITS}, got {self.init!r}.")
        if self.clamp is not None and not isinstance(self.clamp, tuple):
            self.clamp = tuple(self.clamp)
        if self.alpha > self.eps + 1e-12:
            logger.warning(
                "alpha (%s) exceeds eps (%s); the first step saturates the l_inf ball.",
                self.alpha,
                self.eps,
            )

    @classmethod
    def from_dict(cls, cfg: Optional[Dict] = None) -> "JailbreakConfig":
        """Build a config from a YAML/dict; unknown keys are ignored + logged."""
        cfg = dict(cfg or {})
        cfg.pop("provenance", None)
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        unknown = set(cfg) - known
        if unknown:
            logger.warning("Ignoring unknown JailbreakConfig keys: %s", sorted(unknown))
        cfg = {k: v for k, v in cfg.items() if k in known}
        return cls(**cfg)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def load_target_strings(csv_path: Union[str, os.PathLike]) -> List[str]:
    """Read the harmful target strings from ``derogatory_corpus.csv``.

    The upstream corpus may or may not ship with a header; both layouts are
    handled: the first non-empty cell of every row is used.  A possible header
    row is dropped and duplicates are removed while preserving order.
    """
    csv_path = str(csv_path)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Harmful target corpus not found: {csv_path}. "
            f"Download it from {DEROGATORY_CORPUS_URL} (see data/jailbreak_assets.py)."
        )
    strings: List[str] = []
    with open(csv_path, "r", encoding="utf-8", errors="ignore", newline="") as handle:
        for row in csv.reader(handle):
            for cell in row:
                cell = (cell or "").strip()
                if cell:
                    strings.append(cell)
                    break
    if strings and strings[0].strip().lower() in HEADER_TOKENS:
        strings = strings[1:]
    seen, unique = set(), []
    for s in strings:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    if not unique:
        raise ValueError(f"No harmful target strings found in {csv_path}.")
    logger.info("Loaded %d harmful target strings from %s.", len(unique), csv_path)
    return unique


def tokenize_target_ids(
    tokenizer,
    target_text: str,
    add_special_tokens: bool = False,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """Token ids of a harmful target string (the attack's target sequence)."""
    if hasattr(tokenizer, "encode"):
        ids = tokenizer.encode(target_text, add_special_tokens=add_special_tokens)
    elif hasattr(tokenizer, "tokenizer"):  # HF processors expose `.tokenizer`
        ids = tokenizer.tokenizer.encode(target_text, add_special_tokens=add_special_tokens)
    else:  # pragma: no cover - defensive
        raise TypeError("tokenizer must expose an `encode` method (or a `.tokenizer`).")
    if torch.is_tensor(ids):
        ids = ids.flatten().tolist()
    ids = [int(i) for i in ids]
    if not ids:
        raise ValueError(f"Target string tokenises to an empty sequence: {target_text!r}.")
    return torch.tensor(ids, dtype=torch.long, device=device)


def build_targeted_loss_fn(
    logits_fn: Callable[[torch.Tensor, str], torch.Tensor],
    tokenizer,
    *,
    loss: str = UPSTREAM_LOSS,
    shift: int = 1,
    reduction: str = "mean",
) -> Callable[[torch.Tensor, str], torch.Tensor]:
    """Build the upstream targeted loss over raw pixel inputs.

    Parameters
    ----------
    logits_fn
        ``logits_fn(pixels, target_text) -> logits`` where ``pixels`` are raw,
        non-normalized images ``[1, 3, H, W]`` and ``logits`` has shape
        ``[1, T, V]`` covering the teacher-forced prompt plus the target
        continuation.  ``models/llava_openclip.py`` exposes such a function.
    tokenizer
        Tokenizer used to encode the harmful target string.
    loss
        ``"targeted_ce"`` / ``"targeted_nll"`` maximise the likelihood of the
        target continuation; ``"mse"`` mirrors the upstream embedding-matching
        variant applied to the logits of the target tokens.
    shift
        Teacher-forcing offset: with ``shift=1`` the logits at positions
        ``-(L+1) .. -2`` predict the ``L`` target tokens (standard next-token
        formulation); ``shift=0`` uses the logits at the target positions.

    Returns
    -------
    Callable[[torch.Tensor, str], torch.Tensor]
        A loss to be *minimised* (the attack is targeted).
    """
    if loss not in SUPPORTED_LOSSES:
        raise ValueError(f"loss must be one of {SUPPORTED_LOSSES}, got {loss!r}.")

    def loss_fn(pixels: torch.Tensor, target_text: str) -> torch.Tensor:
        logits = logits_fn(pixels, target_text)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        ids = tokenize_target_ids(
            tokenizer, target_text, device=logits.device if torch.is_tensor(logits) else None
        )
        n = ids.numel()
        start = logits.shape[1] - n - shift
        if start < 0:
            raise ValueError(
                "Target string is longer than the teacher-forced sequence "
                f"({n + shift} > {logits.shape[1]})."
            )
        sel = logits[:, start : start + n, :].float()
        if loss == "mse":
            one_hot = F.one_hot(ids, num_classes=sel.shape[-1]).to(sel.dtype)
            return F.mse_loss(sel, one_hot, reduction=reduction)
        return F.cross_entropy(
            sel.reshape(-1, sel.shape[-1]),
            ids.reshape(-1),
            reduction=reduction,
        )

    return loss_fn


# --------------------------------------------------------------------------- #
# The attack
# --------------------------------------------------------------------------- #
class JailbreakAttack:
    """Universal targeted l_inf attack on a VLM's visual input (Qi et al., 2023).

    The optimiser runs for exactly ``config.iterations == 5000`` steps with
    ``alpha = 1/255`` and **no momentum**.  Every step the perturbation is
    projected onto the l_inf ball around the *raw* (non-normalized) source image
    and re-encoded with the mandated precision policy, so the step/gradient
    bookkeeping matches the rest of the benchmark.
    """

    def __init__(self, config: Optional[JailbreakConfig] = None, **kwargs) -> None:
        self.config = config if config is not None else JailbreakConfig(**kwargs)
        self.eps = float(self.config.eps)
        self.alpha = float(self.config.alpha)
        self.iterations = int(self.config.iterations)
        self.momentum = float(self.config.momentum)  # == 0.0 by Addendum
        self.precision = self.config.precision
        self.quant_scale = float(self.config.quant_scale)
        self.float_dtype = float_dtype_for_precision(self.precision)
        self.int_dtype = int_dtype_for_precision(self.precision)
        self.device = torch.device(self.config.device) if self.config.device else None
        self._generator: Optional[torch.Generator] = None

        if self.config.verbose:
            logger.info(
                "JailbreakAttack: iterations=%d alpha=%.8f (1/255) momentum=%.1f "
                "precision=%s int_dtype=%s eps=%.8f [eps: %s]",
                self.iterations,
                self.alpha,
                self.momentum,
                self.precision,
                str(self.int_dtype).replace("torch.", ""),
                self.eps,
                self.config.provenance.get("eps", UNSPECIFIED),
            )

    # ------------------------------------------------------------------ utils
    @property
    def generator(self) -> torch.Generator:
        if self._generator is None:
            device = self.device if self.device is not None else torch.device("cpu")
            gen = torch.Generator(device=device)
            if self.config.seed is not None:
                gen.manual_seed(int(self.config.seed))
            self._generator = gen
        return self._generator

    def summary(self) -> Dict[str, object]:
        """Report the configuration together with its provenance."""
        return {
            "iterations": self.iterations,
            "alpha": self.alpha,
            "momentum": self.momentum,
            "eps": self.eps,
            "loss": self.config.loss,
            "init": self.config.init,
            "optimizer": self.config.optimizer,
            "precision": self.precision,
            "int_dtype": str(self.int_dtype),
            "quant_scale": self.quant_scale,
            "single_source_image": self.config.single_source_image,
            "source_image": self.config.source_image,
            "target_corpus": self.config.target_corpus,
            "objective": "universal targeted (harmful target strings)",
            "provenance": dict(self.config.provenance),
        }

    # ------------------------------------------------------- perturbation API
    def _encode(self, delta_f: torch.Tensor) -> torch.Tensor:
        return encode_perturbation(delta_f, self.precision, quant_scale=self.quant_scale)

    def _decode(self, codes: torch.Tensor) -> torch.Tensor:
        return decode_perturbation(
            codes, float_dtype=self.float_dtype, quant_scale=self.quant_scale
        )

    def initial_perturbation(
        self, x: torch.Tensor, *, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """Initialise the perturbation inside the l_inf ball around raw pixels.

        ``"uniform"`` is the upstream default; ``"zeros"`` gives a clean
        initialisation (used e.g. when re-running the attack) and ``"gaussian"``
        is provided for completeness.
        """
        x = x.detach().to(self.float_dtype)
        gen = generator if generator is not None else self.generator
        if self.config.init == "uniform" and self.config.random_start:
            delta = torch.empty_like(x).uniform_(-self.eps, self.eps, generator=gen)
        elif self.config.init == "gaussian" and self.config.random_start:
            delta = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=gen)
            delta = (delta * (self.eps / 3.0)).clamp(-self.eps, self.eps)
        else:
            delta = torch.zeros_like(x)
        return self._encode(delta)

    def project(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """Project the perturbation onto the l_inf ball around the RAW pixels.

        ``x`` must be the non-normalized source image: the Addendum states that
        the l_inf ball is computed around non-normalized inputs.
        """
        codes = delta if delta.dtype == self.int_dtype else self._encode(delta)
        delta_f = self._decode(codes).to(self.float_dtype)
        delta_f = delta_f.clamp(-self.eps, self.eps)
        if self.config.clamp is not None:
            lo, hi = self.config.clamp
            lo = torch.as_tensor(lo, dtype=delta_f.dtype, device=delta_f.device) - x
            hi = torch.as_tensor(hi, dtype=delta_f.dtype, device=delta_f.device) - x
            delta_f = torch.max(torch.min(delta_f, hi), lo)
        return self._encode(delta_f)

    def adversarial_examples(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """Raw-pixel adversarial image ``x + delta`` (optionally clamped)."""
        delta_f = self._decode(delta).to(dtype=x.dtype, device=x.device)
        adv = x + delta_f
        if self.config.clamp is not None:
            adv = adv.clamp(*self.config.clamp)
        return adv

    # ------------------------------------------------------------- loss utils
    def _targeted_loss(
        self,
        loss_fn: Callable[..., torch.Tensor],
        adv: torch.Tensor,
        target: str,
    ) -> torch.Tensor:
        value = loss_fn(adv, target)
        if not torch.is_tensor(value):
            value = torch.as_tensor(value, dtype=torch.float32, device=adv.device)
        return value

    # ------------------------------------------------------------ attack loop
    def perturb(
        self,
        x: torch.Tensor,
        loss_fn: Callable[..., torch.Tensor],
        targets: Sequence[str],
        *,
        generator: Optional[torch.Generator] = None,
        delta0: Optional[torch.Tensor] = None,
        return_delta: bool = True,
        callback: Optional[Callable[[int, torch.Tensor, torch.Tensor], None]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Run the 5000-step universal targeted attack.

        Parameters
        ----------
        x
            Raw, non-normalized source image ``[1, 3, H, W]`` (``clean.jpeg`` is
            the image mandated by the Addendum).
        loss_fn
            ``loss_fn(adv_pixels, target_string) -> scalar``; a targeted loss that
            is *minimised* (see :func:`build_targeted_loss_fn`).
        targets
            Harmful target strings from ``derogatory_corpus.csv``.  The attack is
            *universal*: a single perturbation is optimised across all of them.
        delta0
            Optional integer perturbation to warm-start from; defaults to the
            uniform random initialisation inside the l_inf ball.
        return_delta
            ``True`` returns the integer-coded perturbation, ``False`` the
            adversarial raw pixels.
        callback
            Optional ``callback(step, adv_pixels, loss)`` hook for logging.

        Returns
        -------
        torch.Tensor
            Integer perturbation (mandated precision dtype) or adversarial pixels.
        """
        if isinstance(targets, str):
            targets = [targets]
        targets = list(targets)
        if not targets:
            raise ValueError("At least one harmful target string is required.")

        x = x.detach().to(self.float_dtype)
        if self.device is not None:
            x = x.to(self.device)
        if x.dim() == 3:
            x = x.unsqueeze(0)

        gen = generator if generator is not None else self.generator
        delta = delta0 if delta0 is not None else self.initial_perturbation(x, generator=gen)
        delta = self.project(x, delta)

        # NO momentum for the jailbreak attack (Addendum).  Upstream optimises the
        # perturbation directly (Adam with lr == alpha); "sign_sgd" reproduces the
        # PGD-style elementwise-sign step with mu = 0.
        param = torch.nn.Parameter(self._decode(delta).detach().clone())
        opt: Optional[torch.optim.Optimizer] = None
        if self.config.optimizer == "adam":
            opt = torch.optim.Adam([param], lr=self.alpha)
        elif self.config.optimizer == "sgd":
            opt = torch.optim.SGD([param], lr=self.alpha, momentum=0.0)

        n_targets = len(targets)
        fixed_target = targets[0] if n_targets == 1 else None
        if self.config.micro_batch_targets > 1:
            fixed_batch = targets[: self.config.micro_batch_targets]

        last_loss = torch.tensor(float("nan"))
        for step in range(self.iterations):
            adv = param.detach() + x
            if self.config.clamp is not None:
                adv = adv.clamp(*self.config.clamp)
            adv.requires_grad_(True)

            if self.config.micro_batch_targets > 1:
                losses = [self._targeted_loss(loss_fn, adv, t) for t in fixed_batch]
                loss = torch.stack(losses).mean()
            else:
                if fixed_target is not None:
                    target = fixed_target
                else:
                    idx = int(torch.randint(n_targets, (1,), generator=gen).item())
                    target = targets[idx]
                loss = self._targeted_loss(loss_fn, adv, target)

            if opt is not None:
                opt.zero_grad(set_to_none=True)
                # Gradient flows to `param` through `adv = param + x`.
                loss.backward()
                with torch.no_grad():
                    if param.grad is not None:
                        param.grad[torch.isnan(param.grad)] = 0.0
                opt.step()
            else:
                grad = torch.autograd.grad(loss, adv, retain_graph=False)[0]
                flat = grad.reshape(grad.shape[0], -1)
                flat = flat / flat.norm(p=1, dim=1, keepdim=True).clamp_min(1e-12)
                with torch.no_grad():
                    param.add_(flat.reshape_as(grad).sign(), alpha=self.alpha)

            # Project onto the l_inf ball around the RAW, non-normalized pixels
            # and re-encode with the mandated integer dtype.
            with torch.no_grad():
                codes = self.project(x, self._encode(param.detach()))
                param.copy_(self._decode(codes).to(param.dtype))
            last_loss = loss.detach()

            if callback is not None and (step == 0 or (step + 1) % self.config.log_every == 0):
                callback(step, self.adversarial_examples(x, codes), last_loss)
            if (
                self.config.verbose
                and self.config.log_every
                and (step == 0 or (step + 1) % self.config.log_every == 0)
            ):
                logger.info(
                    "jailbreak step %d/%d loss=%.4f target=%r",
                    step + 1,
                    self.iterations,
                    float(last_loss),
                    (fixed_target if fixed_target is not None else targets[0])[:48],
                )

        with torch.no_grad():
            codes = self.project(x, self._encode(param.detach()))

        if return_delta:
            return codes
        return self.adversarial_examples(x.to(codes.device), codes)

    def attack_untargeted(self, *args, **kwargs):
        raise NotImplementedError(
            "The jailbreak attack is a universal *targeted* attack (Addendum)."
        )

    def attack_targeted(
        self,
        x: torch.Tensor,
        targets: Sequence[str],
        loss_fn: Callable[..., torch.Tensor],
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Alias of :meth:`perturb` with the targeted argument order."""
        return self.perturb(x, loss_fn, targets, **kwargs)

    # --------------------------------------------------------- serialisation
    def save(
        self,
        path: str,
        delta: torch.Tensor,
        metadata: Optional[Dict] = None,
        targets: Optional[Sequence[str]] = None,
    ) -> str:
        """Persist the integer perturbation together with its provenance."""
        meta = dict(self.summary())
        meta.update(metadata or {})
        codes, store_meta = store_perturbation(
            self._decode(delta), self.precision, quant_scale=self.quant_scale
        )
        payload = {
            "perturbation": codes.cpu(),
            "storage": store_meta,
            "config": meta,
            "targets": list(targets or []),
        }
        torch.save(payload, path)
        logger.info(
            "Saved jailbreak perturbation to %s (dtype=%s).", path, store_meta.get("int_dtype")
        )
        return path

    def load(self, path: str) -> Tuple[torch.Tensor, Dict]:
        """Reload a stored perturbation (honours the int16/int32 policy)."""
        payload = torch.load(path, map_location="cpu")
        delta = load_perturbation(payload["perturbation"], payload["storage"])
        return delta, payload.get("config", {})


def perturb_jailbreak_image(
    model,
    image: torch.Tensor,
    targets: Sequence[str],
    *,
    config: Optional[JailbreakConfig] = None,
    **kwargs,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Convenience wrapper: attack ``image`` given a LLaVA-like victim ``model``.

    ``model`` must expose ``targeted_logits(pixels, target_text)`` (see
    ``models/llava_openclip.py``) and a tokenizer via ``model.tokenizer``.
    """
    cfg = config if config is not None else JailbreakConfig(**kwargs)
    attack = JailbreakAttack(cfg)
    loss_fn = build_targeted_loss_fn(
        model.targeted_logits, model.tokenizer, loss=cfg.loss
    )
    return attack.perturb(image, loss_fn, targets, **kwargs.get("perturb_kwargs", {}))
