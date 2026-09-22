"""OPAL baseline (privileged 10-skill execution) for the FRE paper.

Paper references (verbatim, Addendum -- "Additional Details on OPAL")::

    - No manually designed rewards are used in OPAL.
    - For the OPAL encoder, the same transformer architecture is used as in FRE.
    - For the privileged execution evaluation described in the paper:
      - OPAL's task policy is not used
      - 10 random skills are sampled from a unit Gaussian,
      - for each skill z, the policy is conditioned on it and evaluated for the
        entire episode,
      - and the best performing rollout is taken.

Consequence for the evaluation harness: OPAL is *unsupervised* -- it never sees a
reward function.  For a novel task the agent samples ``num_skills = 10`` latent
skills from the unit Gaussian prior, each skill is rolled out for the whole
episode, and the shared rollout driver (``fre.eval.baselines.default_rollout_fn``
with ``best_of_skills=True``) keeps the best performing rollout.  Consequently
``OpalAgent.condition`` / ``OpalAgent.condition_many`` ignore the evaluation
task's reward function entirely and simply draw prior samples.

How OPAL is re-implemented inside this codebase (the paper only specifies the
encoder architecture and the privileged evaluation protocol, everything else is
marked "Source: not specified in the paper" in the comments below):

*   ``OPALEncoder`` -- the *same* permutation-invariant transformer architecture
    as FRE (pre-norm blocks, no positional encodings, no causal masking, mean
    pooling, two heads parametrising a diagonal Gaussian), except that no reward
    tokens exist (OPAL uses no manually designed rewards), so the token is a
    projection of the state alone.
*   ``OPALDynamicsDecoder`` -- a latent-conditioned skill/dynamics model
    ``q(s' | s, a, z)`` trained jointly with the encoder by maximizing the
    variational lower bound (reconstruction MSE minus ``beta`` times the KL to
    the unit Gaussian, the same objective family as the FRE Equation (6) bound).
    The latent therefore encodes *how* the data was generated (the behaviour
    mode / skill).
*   ``OPALPolicy`` -- a latent-conditioned Gaussian policy ``pi(a | s, z)``
    extracted with plain behaviour cloning (MLE) from the unlabeled offline
    dataset.  This is the "primitive" policy; the reward-maximizing task policy
    (``OPAL's task policy is not used``) is deliberately absent, which is exactly
    why the privileged-execution protocol with best-of-10-rollouts is required.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - torch is required at runtime, module stays importable
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    _HAS_TORCH = False


# ---------------------------------------------------------------------------
# tolerant imports of the shared baseline infrastructure / model zoo
# ---------------------------------------------------------------------------
FRE_CONTEXT_SAMPLES = 32
NUM_EVAL_EPISODES = 20
FB_SF_CONTEXT_SAMPLES = 5120


def _load_baseline_package() -> Dict[str, Any]:
    """Resolve the shared baseline helpers, with progressively dumber fallbacks."""
    candidates = [
        lambda: __import__("fre.eval.baselines", fromlist=["*"]),
        lambda: __import__("baselines", fromlist=["*"]),
    ]
    for loader in candidates:
        try:
            return {name: getattr(loader(), name) for name in ("BaselineAgent",)}
        except Exception:
            continue
    # last resort: add the package root to sys.path and retry
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        module = __import__("fre.eval.baselines", fromlist=["*"])
        return {name: getattr(module, name) for name in ("BaselineAgent",)}
    except Exception:
        return {}


_BASELINE_PKG = _load_baseline_package()


class _FallbackBaselineAgent:
    """Minimal stand-in used only when ``fre.eval.baselines`` cannot be imported."""

    name: str = "baseline"
    num_skills: int = 1

    def condition(self, task, context=None, rng=None):  # pragma: no cover
        raise NotImplementedError

    def condition_many(self, task, num_skills=1, context=None, rng=None):  # pragma: no cover
        return np.stack([self.condition(task, context=context, rng=rng) for _ in range(int(num_skills))], axis=0)

    def act(self, observation, conditioning, deterministic: bool = True):  # pragma: no cover
        raise NotImplementedError

    def describe(self) -> Dict[str, Any]:  # pragma: no cover
        return {"name": self.name}


BaselineAgent = _BASELINE_PKG.get("BaselineAgent", _FallbackBaselineAgent)
try:  # constants may already be defined by the shared package
    from fre.eval.baselines import FRE_CONTEXT_SAMPLES as _SHARED_FRE_CTX  # type: ignore
    from fre.eval.baselines import NUM_EVAL_EPISODES as _SHARED_EPISODES  # type: ignore

    FRE_CONTEXT_SAMPLES = int(_SHARED_FRE_CTX)
    NUM_EVAL_EPISODES = int(_SHARED_EPISODES)
except Exception:  # pragma: no cover - defaults above are per the paper
    pass

# The FRE transformer architecture (paper: "For the OPAL encoder, the same
# transformer architecture is used as in FRE").
_FRETransformerBlock = None
try:  # pragma: no cover - depends on the rest of the repo being importable
    from fre.models.encoder import TransformerBlock as _FRETransformerBlock  # type: ignore
except Exception:  # pragma: no cover
    _FRETransformerBlock = None


# ---------------------------------------------------------------------------
# constants (Table 3 hyper-parameters + Addendum OPAL details)
# ---------------------------------------------------------------------------
DEFAULT_NUM_SKILLS = 10  # "10 random skills are sampled from a unit Gaussian"
DEFAULT_LATENT_DIM = 128  # FRE appendix: the latent embedding z is 128-dimensional
DEFAULT_TOKEN_DIM = 128
DEFAULT_NUM_ENCODER_LAYERS = 4
DEFAULT_NUM_ATTENTION_HEADS = 4
DEFAULT_MLP_DIM = 256  # MLP block expands to 256 then back to 128
DEFAULT_HIDDEN_DIMS: Tuple[int, int, int] = (512, 512, 512)
DEFAULT_LOG_STD_MIN = -5.0
DEFAULT_LOG_STD_MAX = 2.0
DEFAULT_BETA = 0.01  # KL weight; FRE uses 0.01 (Source: not specified for OPAL)
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_BATCH_SIZE = 512
DEFAULT_NUM_CONTEXT_STATES = 32  # K context states, same as FRE's encoder set
DEFAULT_TARGET_STEPS = 850_000  # Table 3 policy training steps (AntMaze)
DEFAULT_MAX_GRAD_NORM = 10.0
DEFAULT_SKILL_PRIOR_STD = 1.0
DEFAULT_NUM_TRANSITION_SAMPLES = 8  # K' decoding states, same as FRE
DEFAULT_ACTIVATION = "relu"


# ---------------------------------------------------------------------------
# small array helpers
# ---------------------------------------------------------------------------
def as_2d(observations: Any) -> np.ndarray:
    """Coerce ``observations`` to a 2-D float32 array ``(N, dim)``."""
    if observations is None:
        raise ValueError("observations must not be None")
    if isinstance(observations, np.ndarray):
        array = np.asarray(observations, dtype=np.float32)
    elif _HAS_TORCH and isinstance(observations, torch.Tensor):
        array = observations.detach().cpu().numpy().astype(np.float32)
    else:
        array = np.asarray(observations, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    return array


def sample_unit_gaussian_skills(
    num_skills: int = DEFAULT_NUM_SKILLS,
    latent_dim: int = DEFAULT_LATENT_DIM,
    rng: Optional[np.random.Generator] = None,
    std: float = DEFAULT_SKILL_PRIOR_STD,
) -> np.ndarray:
    """Sample skills from the unit Gaussian prior: ``z ~ N(0, std^2 I)``.

    Paper (Addendum): "10 random skills are sampled from a unit Gaussian".
    """
    rng = rng if rng is not None else np.random.default_rng()
    num_skills = int(max(1, num_skills))
    return rng.normal(0.0, float(std), size=(num_skills, int(latent_dim))).astype(np.float32)


def skill_log_prob(skills: Any, std: float = DEFAULT_SKILL_PRIOR_STD) -> np.ndarray:
    """Log density of skills under the unit Gaussian prior (diagnostics only)."""
    z = as_2d(skills)
    std = float(max(std, 1e-8))
    normaliser = z.shape[-1] * math.log(std * math.sqrt(2.0 * math.pi))
    return (-0.5 * np.sum((z / std) ** 2, axis=-1) - normaliser).astype(np.float32)


def select_best_rollout(
    episode_scores: Sequence[float],
    episode_returns: Optional[Sequence[float]] = None,
) -> int:
    """Index of the best performing rollout ("the best performing rollout is taken").

    ``episode_scores`` are the (already normalized) scores; ``episode_returns`` is
    accepted so callers can tie-break on the raw return when scores are equal.
    """
    scores = np.asarray([float(s) for s in episode_scores], dtype=np.float64)
    if scores.size == 0:
        raise ValueError("select_best_rollout requires at least one rollout")
    if episode_returns is not None:
        returns = np.asarray([float(r) for r in episode_returns], dtype=np.float64)
        if returns.shape == scores.shape:
            return int(np.lexsort((returns, scores))[-1])
    return int(np.argmax(scores))


def _to_numpy(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    if _HAS_TORCH and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    try:
        return np.asarray(value)
    except Exception:  # pragma: no cover
        return None


def _field(batch: Any, name: str) -> Optional[np.ndarray]:
    """Fetch a field from a ``Batch``-like dataclass, dict or object."""
    if batch is None:
        return None
    if isinstance(batch, dict):
        return _to_numpy(batch.get(name))
    value = getattr(batch, name, None)
    if value is None and name == "next_observations":
        value = getattr(batch, "next_obs", None)
    return _to_numpy(value)


def _sample_states_from_buffer(
    buffer: Any,
    num_samples: int,
    rng: Optional[np.random.Generator] = None,
    encoder_input: bool = False,
) -> np.ndarray:
    """Uniform state sampling with signature-tolerant fallbacks."""
    attempts = (
        lambda: buffer.sample_states(num_samples, rng=rng, encoder_input=encoder_input),
        lambda: buffer.sample_states(num_samples, rng=rng),
        lambda: buffer.sample_states(num_samples, rng, encoder_input),
        lambda: buffer.sample_states(num_samples),
    )
    for attempt in attempts:
        try:
            states = attempt()
        except TypeError:
            continue
        except Exception:
            continue
        if states is not None:
            return as_2d(states)
    raise RuntimeError("replay buffer does not expose a usable sample_states method")


def _sample_transitions_from_buffer(
    buffer: Any,
    batch_size: int,
    rng: Optional[np.random.Generator] = None,
    with_encoder_inputs: bool = False,
) -> Any:
    attempts = (
        lambda: buffer.sample_transitions(batch_size, rng=rng, with_encoder_inputs=with_encoder_inputs),
        lambda: buffer.sample_transitions(batch_size, rng=rng),
        lambda: buffer.sample_transitions(batch_size),
    )
    for attempt in attempts:
        try:
            return attempt()
        except TypeError:
            continue
        except Exception:
            continue
    raise RuntimeError("replay buffer does not expose a usable sample_transitions method")


def _activation_module(name: str):
    if not _HAS_TORCH:  # pragma: no cover
        raise RuntimeError("torch is required to build OPAL networks")
    name = (name or "relu").lower()
    table = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
        "silu": nn.SiLU,
        "swish": nn.SiLU,
        "leaky_relu": nn.LeakyReLU,
    }
    return table.get(name, nn.ReLU)()


def _build_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    activation: str = DEFAULT_ACTIVATION,
    layer_norm: bool = False,
) -> Any:
    if not _HAS_TORCH:  # pragma: no cover
        raise RuntimeError("torch is required to build OPAL networks")
    layers: List[Any] = []
    last = int(input_dim)
    for width in hidden_dims:
        layers.append(nn.Linear(last, int(width)))
        if layer_norm:
            layers.append(nn.LayerNorm(int(width)))
        layers.append(_activation_module(activation))
        last = int(width)
    layers.append(nn.Linear(last, int(output_dim)))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# transformer (same architecture as FRE, without reward tokens)
# ---------------------------------------------------------------------------
if _HAS_TORCH:

    class OPALTransformerBlock(nn.Module):
        """Pre-norm transformer block: no positional encodings, no causal mask.

        Fallback identical in design to ``fre.models.encoder.TransformerBlock``
        (used when that module cannot be imported).
        """

        def __init__(
            self,
            token_dim: int = DEFAULT_TOKEN_DIM,
            num_heads: int = DEFAULT_NUM_ATTENTION_HEADS,
            mlp_dim: int = DEFAULT_MLP_DIM,
            dropout: float = 0.0,
            activation: str = DEFAULT_ACTIVATION,
        ) -> None:
            super().__init__()
            self.norm1 = nn.LayerNorm(int(token_dim))
            self.attn = nn.MultiheadAttention(
                embed_dim=int(token_dim), num_heads=int(num_heads), dropout=float(dropout), batch_first=True
            )
            self.norm2 = nn.LayerNorm(int(token_dim))
            self.mlp = nn.Sequential(
                nn.Linear(int(token_dim), int(mlp_dim)),
                _activation_module(activation),
                nn.Linear(int(mlp_dim), int(token_dim)),
            )
            self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()

        def forward(self, tokens: Any) -> Any:
            normed = self.norm1(tokens)
            attended, _ = self.attn(normed, normed, normed, need_weights=False)
            tokens = tokens + self.dropout(attended)
            tokens = tokens + self.dropout(self.mlp(self.norm2(tokens)))
            return tokens

    _TRANSFORMER_BLOCK_CLS = _FRETransformerBlock or OPALTransformerBlock
else:  # pragma: no cover
    _TRANSFORMER_BLOCK_CLS = None


if _HAS_TORCH:

    @dataclass
    class OPALEncoderOutput:
        """Encoder output container (mirrors ``fre.models.encoder.EncoderOutput``)."""

        z: Any
        mu: Any
        log_std: Any
        std: Any

        def kl_to_unit_gaussian(self) -> Any:
            """D_KL( N(mu, sigma) || N(0, I) ) per batch element (unit Gaussian prior)."""
            return 0.5 * torch.sum(
                self.mu.pow(2) + self.std.pow(2) - 2.0 * self.log_std - 1.0, dim=-1
            )

    class OPALEncoder(nn.Module):
        """Permutation-invariant transformer encoder over a set of states.

        The architecture is the one used by FRE (paper: "For the OPAL encoder, the
        same transformer architecture is used as in FRE") minus the learned reward
        embedding table, because "No manually designed rewards are used in OPAL".
        A state is projected to a ``token_dim`` token, the K tokens are processed by
        pre-norm transformer blocks without positional encodings or causal masking,
        the final representations are mean-pooled and mapped to ``mu``/``log_std``
        of a diagonal Gaussian over the latent skill ``z``.
        """

        def __init__(
            self,
            state_dim: int,
            latent_dim: int = DEFAULT_LATENT_DIM,
            token_dim: int = DEFAULT_TOKEN_DIM,
            num_layers: int = DEFAULT_NUM_ENCODER_LAYERS,
            num_heads: int = DEFAULT_NUM_ATTENTION_HEADS,
            mlp_dim: int = DEFAULT_MLP_DIM,
            dropout: float = 0.0,
            activation: str = DEFAULT_ACTIVATION,
            log_std_min: float = DEFAULT_LOG_STD_MIN,
            log_std_max: float = DEFAULT_LOG_STD_MAX,
        ) -> None:
            super().__init__()
            self.state_dim = int(state_dim)
            self.latent_dim = int(latent_dim)
            self.token_dim = int(token_dim)
            self.log_std_min = float(log_std_min)
            self.log_std_max = float(log_std_max)
            self.state_projection = nn.Linear(self.state_dim, self.token_dim)
            self.blocks = nn.ModuleList(
                [
                    _TRANSFORMER_BLOCK_CLS(
                        token_dim=self.token_dim,
                        num_heads=int(num_heads),
                        mlp_dim=int(mlp_dim),
                        dropout=float(dropout),
                        activation=activation,
                    )
                    for _ in range(int(num_layers))
                ]
            )
            self.final_norm = nn.LayerNorm(self.token_dim)
            self.mu_head = nn.Linear(self.token_dim, self.latent_dim)
            self.log_std_head = nn.Linear(self.token_dim, self.latent_dim)

        def make_tokens(self, states: Any) -> Any:
            states = states.float()
            if states.dim() == 2:
                states = states.unsqueeze(0)
            return self.state_projection(states)

        def forward(self, states: Any, sample: bool = True) -> OPALEncoderOutput:
            tokens = self.make_tokens(states)
            for block in self.blocks:
                tokens = block(tokens)
            pooled = self.final_norm(tokens).mean(dim=1)
            mu = self.mu_head(pooled)
            log_std = torch.clamp(self.log_std_head(pooled), self.log_std_min, self.log_std_max)
            std = torch.exp(log_std)
            if sample:
                eps = torch.randn_like(std)
                z = mu + std * eps
            else:
                z = mu
            return OPALEncoderOutput(z=z, mu=mu, log_std=log_std, std=std)

        def encode(self, states: Any, sample: bool = False) -> Any:
            return self.forward(states, sample=sample).z

    class OPALDynamicsDecoder(nn.Module):
        """Latent-conditioned skill decoder ``q(s' | s, a, z)``.

        Trained jointly with :class:`OPALEncoder` by maximizing the variational
        lower bound (reconstruction minus ``beta * KL``), i.e. the same objective
        family as the FRE information-bottleneck bound.  The latent conditioning
        is obtained by concatenating ``z`` to the ``(s, a)`` input.
        """

        def __init__(
            self,
            state_dim: int,
            act_dim: int,
            latent_dim: int = DEFAULT_LATENT_DIM,
            hidden_dims: Sequence[int] = DEFAULT_HIDDEN_DIMS,
            activation: str = DEFAULT_ACTIVATION,
            layer_norm: bool = False,
        ) -> None:
            super().__init__()
            self.state_dim = int(state_dim)
            self.act_dim = int(act_dim)
            self.latent_dim = int(latent_dim)
            self.net = _build_mlp(
                self.state_dim + self.act_dim + self.latent_dim,
                hidden_dims,
                self.state_dim,
                activation=activation,
                layer_norm=layer_norm,
            )

        def forward(self, states: Any, actions: Any, z: Any) -> Any:
            states = states.float()
            actions = actions.float()
            z = z.float()
            if z.dim() == 3:
                z = z.mean(dim=1)
            if z.shape[0] != states.shape[0]:
                z = z.expand(states.shape[0], -1)
            return self.net(torch.cat([states, actions, z], dim=-1))

        def reconstruction_loss(self, states: Any, actions: Any, z: Any, targets: Any) -> Any:
            prediction = self.forward(states, actions, z)
            return F.mse_loss(prediction, targets.float())

    @dataclass
    class OPALPolicyOutput:
        """Policy forward output (mean/action/log-prob of the squashed Gaussian)."""

        mean: Any
        action: Any
        log_prob: Any = None
        log_std: Any = None
        std: Any = None
        pre_tanh_mean: Any = None
        squashed: bool = True

    class OPALPolicy(nn.Module):
        """Latent-conditioned Gaussian policy ``pi(a | s, z)`` (behaviour cloning).

        Table 3 network structure: three hidden layers of size 512 with ReLU
        activations; the log standard deviation is clamped to ``[-5.0, 2.0]``
        (the lower bound ``-5.0`` is the value quoted in the addendum for GC-BC).
        Actions are tanh-squashed so the policy matches typical bounded MuJoCo
        action spaces.
        """

        def __init__(
            self,
            obs_dim: int,
            act_dim: int,
            latent_dim: int = DEFAULT_LATENT_DIM,
            hidden_dims: Sequence[int] = DEFAULT_HIDDEN_DIMS,
            activation: str = DEFAULT_ACTIVATION,
            layer_norm: bool = False,
            tanh_squash: bool = True,
            log_std_min: float = DEFAULT_LOG_STD_MIN,
            log_std_max: float = DEFAULT_LOG_STD_MAX,
        ) -> None:
            super().__init__()
            self.obs_dim = int(obs_dim)
            self.act_dim = int(act_dim)
            self.latent_dim = int(latent_dim)
            self.tanh_squash = bool(tanh_squash)
            self.log_std_min = float(log_std_min)
            self.log_std_max = float(log_std_max)
            self.trunk = _build_mlp(
                self.obs_dim + self.latent_dim,
                hidden_dims,
                int(hidden_dims[-1]) if hidden_dims else 256,
                activation=activation,
                layer_norm=layer_norm,
            )
            last = int(hidden_dims[-1]) if hidden_dims else 256
            self.mean_head = nn.Linear(last, self.act_dim)
            self.log_std_head = nn.Linear(last, self.act_dim)

        def distribution_params(self, observations: Any, z: Any) -> Tuple[Any, Any]:
            observations = observations.float()
            z = z.float()
            if z.dim() == 3:
                z = z.mean(dim=1)
            if z.shape[0] != observations.shape[0]:
                z = z.expand(observations.shape[0], -1)
            features = self.trunk(torch.cat([observations, z], dim=-1))
            mean = self.mean_head(features)
            log_std = torch.clamp(self.log_std_head(features), self.log_std_min, self.log_std_max)
            return mean, log_std

        def forward(
            self,
            observations: Any,
            z: Any,
            action: Any = None,
            deterministic: bool = False,
            sample: bool = True,
        ) -> OPALPolicyOutput:
            mean, log_std = self.distribution_params(observations, z)
            std = torch.exp(log_std)
            if deterministic or not sample:
                pre_tanh = mean
            else:
                pre_tanh = mean + std * torch.randn_like(std)
            if self.tanh_squash:
                act = torch.tanh(pre_tanh)
            else:
                act = pre_tanh
            log_prob = None
            if action is not None:
                log_prob = self.log_prob_from_params(mean, log_std, action)
            return OPALPolicyOutput(
                mean=mean,
                action=act,
                log_prob=log_prob,
                log_std=log_std,
                std=std,
                pre_tanh_mean=mean,
                squashed=self.tanh_squash,
            )

        def log_prob_from_params(self, mean: Any, log_std: Any, action: Any) -> Any:
            action = action.float()
            if self.tanh_squash:
                eps = 1e-6
                clipped = torch.clamp(action, -1.0 + eps, 1.0 - eps)
                pre_tanh = torch.atanh(clipped)
                log_det = torch.sum(torch.log(1.0 - clipped.pow(2) + eps), dim=-1)
            else:
                pre_tanh = action
                log_det = torch.zeros(action.shape[0], device=action.device)
            std = torch.exp(log_std)
            var = std.pow(2)
            log_scale = log_std
            log_prob = -0.5 * (
                ((pre_tanh - mean) ** 2) / var + 2.0 * log_scale + math.log(2.0 * math.pi)
            )
            return torch.sum(log_prob, dim=-1) - log_det

        def log_prob(self, observations: Any, z: Any, action: Any) -> Any:
            mean, log_std = self.distribution_params(observations, z)
            return self.log_prob_from_params(mean, log_std, action)

        def act_numpy(self, observation: Any, z: Any, deterministic: bool = True) -> np.ndarray:
            obs = as_2d(observation)
            skill = as_2d(z)
            with torch.no_grad():
                out = self.forward(
                    torch.as_tensor(obs, dtype=torch.float32),
                    torch.as_tensor(skill, dtype=torch.float32),
                    deterministic=bool(deterministic),
                    sample=not bool(deterministic),
                )
            return out.action.cpu().numpy()

else:  # pragma: no cover - torch-less environment
    OPALEncoderOutput = None  # type: ignore
    OPALEncoder = None  # type: ignore
    OPALDynamicsDecoder = None  # type: ignore
    OPALPolicyOutput = None  # type: ignore
    OPALPolicy = None  # type: ignore


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
@dataclass
class OPALConfig:
    """OPAL hyper-parameters.

    Values taken from the paper where available; every other value is the same
    as the corresponding FRE setting and/or is annotated as
    "Source: not specified in the paper".
    """

    latent_dim: int = DEFAULT_LATENT_DIM
    token_dim: int = DEFAULT_TOKEN_DIM
    num_encoder_layers: int = DEFAULT_NUM_ENCODER_LAYERS  # 4 blocks (FRE architecture)
    num_attention_heads: int = DEFAULT_NUM_ATTENTION_HEADS
    mlp_dim: int = DEFAULT_MLP_DIM
    dropout: float = 0.0
    hidden_dims: Tuple[int, int, int] = DEFAULT_HIDDEN_DIMS  # Table 3: [512, 512, 512]
    activation: str = DEFAULT_ACTIVATION
    layer_norm: bool = False
    tanh_squash: bool = True
    log_std_min: float = DEFAULT_LOG_STD_MIN
    log_std_max: float = DEFAULT_LOG_STD_MAX
    num_context_states: int = DEFAULT_NUM_CONTEXT_STATES  # K = 32
    num_decoder_states: int = DEFAULT_NUM_TRANSITION_SAMPLES  # K' = 8
    beta: float = DEFAULT_BETA  # KL weight (Source: not specified in the paper)
    learning_rate: float = DEFAULT_LEARNING_RATE
    batch_size: int = DEFAULT_BATCH_SIZE
    num_steps: int = DEFAULT_TARGET_STEPS
    max_grad_norm: float = DEFAULT_MAX_GRAD_NORM
    normalize_states: bool = False  # optional per-dimension std normalization
    num_skills: int = DEFAULT_NUM_SKILLS  # "10 random skills" (Addendum)
    skill_prior_std: float = DEFAULT_SKILL_PRIOR_STD  # unit Gaussian prior
    log_interval: int = 1000
    seed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        values = asdict(self)
        values["hidden_dims"] = list(self.hidden_dims)
        return values

    @classmethod
    def from_dict(cls, values: Optional[Dict[str, Any]] = None, **overrides: Any) -> "OPALConfig":
        config = cls()
        payload: Dict[str, Any] = dict(values or {})
        payload.update({k: v for k, v in overrides.items() if v is not None})
        for key, value in payload.items():
            if hasattr(config, key):
                if key == "hidden_dims" and value is not None:
                    value = tuple(int(v) for v in value)
                setattr(config, key, value)
        return config


OPAL_DEFAULTS: Dict[str, Any] = OPALConfig().to_dict()


# ---------------------------------------------------------------------------
# agent
# ---------------------------------------------------------------------------
class OpalAgent(BaselineAgent):  # type: ignore[misc]
    """OPAL agent evaluated with the privileged 10-skill protocol.

    The agent owns an unsupervised skill encoder + dynamics decoder (trained with
    the variational objective) and a latent-conditioned behaviour-cloning policy.
    No reward function is ever consumed: at evaluation time
    :meth:`condition_many` draws ``num_skills`` samples from the unit Gaussian
    prior and the shared rollout driver keeps the best performing rollout.
    """

    name = "OPAL-10"
    num_skills = DEFAULT_NUM_SKILLS
    is_goal_conditioned = False
    uses_latent = True
    uses_rewards = False  # "No manually designed rewards are used in OPAL."

    def __init__(
        self,
        replay_buffer: Any = None,
        obs_dim: Optional[int] = None,
        act_dim: Optional[int] = None,
        latent_dim: Optional[int] = None,
        config: Optional[Any] = None,
        device: Optional[str] = None,
        seed: Optional[int] = None,
        state_mean: Optional[np.ndarray] = None,
        state_std: Optional[np.ndarray] = None,
        **config_overrides: Any,
    ) -> None:
        if config is None:
            self.config = OPALConfig.from_dict(config_overrides)
        elif isinstance(config, OPALConfig):
            self.config = OPALConfig.from_dict(config.to_dict(), **config_overrides)
        elif isinstance(config, dict):
            self.config = OPALConfig.from_dict(config, **config_overrides)
        else:  # duck-typed config object
            self.config = OPALConfig.from_dict(vars(config), **config_overrides)

        self.replay_buffer = replay_buffer
        if obs_dim is None and replay_buffer is not None:
            obs_dim = getattr(replay_buffer, "obs_dim", None)
        if act_dim is None and replay_buffer is not None:
            act_dim = getattr(replay_buffer, "act_dim", None)
        self.obs_dim = int(obs_dim) if obs_dim is not None else None
        self.act_dim = int(act_dim) if act_dim is not None else None
        self.latent_dim = int(latent_dim or self.config.latent_dim)
        self.num_skills = int(self.config.num_skills)
        self.seed = int(seed if seed is not None else self.config.seed)
        self.rng = np.random.default_rng(self.seed)
        self.device = device or ("cuda" if (_HAS_TORCH and torch.cuda.is_available()) else "cpu")

        self.state_mean = None if state_mean is None else np.asarray(state_mean, dtype=np.float32)
        self.state_std = None if state_std is None else np.asarray(state_std, dtype=np.float32)

        self.encoder = None
        self.decoder = None
        self.policy = None
        self.optimizer = None
        self.train_step_count = 0
        self._last_train_stats: Dict[str, float] = {}
        if self.obs_dim is not None and self.act_dim is not None and _HAS_TORCH:
            self._build_networks()

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def _build_networks(self) -> None:
        if not _HAS_TORCH:  # pragma: no cover
            raise RuntimeError("torch is required to build the OPAL networks")
        self.encoder = OPALEncoder(
            state_dim=self.obs_dim,
            latent_dim=self.latent_dim,
            token_dim=self.config.token_dim,
            num_layers=self.config.num_encoder_layers,
            num_heads=self.config.num_attention_heads,
            mlp_dim=self.config.mlp_dim,
            dropout=self.config.dropout,
            activation=self.config.activation,
            log_std_min=self.config.log_std_min,
            log_std_max=self.config.log_std_max,
        ).to(self.device)
        self.decoder = OPALDynamicsDecoder(
            state_dim=self.obs_dim,
            act_dim=self.act_dim,
            latent_dim=self.latent_dim,
            hidden_dims=self.config.hidden_dims,
            activation=self.config.activation,
            layer_norm=self.config.layer_norm,
        ).to(self.device)
        self.policy = OPALPolicy(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            latent_dim=self.latent_dim,
            hidden_dims=self.config.hidden_dims,
            activation=self.config.activation,
            layer_norm=self.config.layer_norm,
            tanh_squash=self.config.tanh_squash,
            log_std_min=self.config.log_std_min,
            log_std_max=self.config.log_std_max,
        ).to(self.device)
        parameters = (
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.policy.parameters())
        )
        self.optimizer = torch.optim.Adam(parameters, lr=float(self.config.learning_rate))

    def _require_networks(self) -> None:
        if self.policy is None or self.encoder is None or self.decoder is None:
            if self.obs_dim is None or self.act_dim is None:
                raise ValueError(
                    "OpalAgent needs obs_dim/act_dim (or a replay buffer exposing "
                    "obs_dim/act_dim) to build its networks"
                )
            self._build_networks()

    # ------------------------------------------------------------------
    # tensor helpers
    # ------------------------------------------------------------------
    def _to_tensor(self, value: Any) -> Any:
        if _HAS_TORCH and isinstance(value, torch.Tensor):
            return value.to(self.device)
        return torch.as_tensor(as_2d(value), dtype=torch.float32, device=self.device)

    def _normalize_states(self, states: np.ndarray, inverse: bool = False) -> np.ndarray:
        if not self.config.normalize_states or self.state_std is None:
            return np.asarray(states, dtype=np.float32)
        std = np.where(np.abs(self.state_std) < 1e-6, 1.0, self.state_std)
        mean = self.state_mean if self.state_mean is not None else np.zeros_like(std)
        return ((states - mean) / std if not inverse else states * std + mean).astype(np.float32)

    def _clip_grads(self) -> float:
        if self.optimizer is None or not float(self.config.max_grad_norm) > 0:
            return 0.0
        norm = float(
            torch.nn.utils.clip_grad_norm_(self._parameters(), float(self.config.max_grad_norm))
        )
        return norm

    def _parameters(self) -> List[Any]:
        parameters: List[Any] = []
        for module in (self.encoder, self.decoder, self.policy):
            if module is not None:
                parameters.extend(list(module.parameters()))
        return parameters

    # ------------------------------------------------------------------
    # sampling helpers
    # ------------------------------------------------------------------
    def _sample_context_states(self, batch_size: int) -> np.ndarray:
        """Uniformly sample ``batch_size * K`` states as the encoder context set."""
        if self.replay_buffer is None:
            raise ValueError("OpalAgent requires a replay buffer for training")
        num_context = int(max(2, self.config.num_context_states))
        states = _sample_states_from_buffer(
            self.replay_buffer, int(batch_size) * num_context, rng=self.rng
        )
        states = self._normalize_states(states)
        if states.shape[0] < int(batch_size) * num_context:
            reps = int(np.ceil(int(batch_size) * num_context / max(1, states.shape[0])))
            states = np.tile(states, (reps, 1))[: int(batch_size) * num_context]
        return states.reshape(int(batch_size), num_context, states.shape[-1])

    def _sample_transition_arrays(self, batch_size: int) -> Dict[str, np.ndarray]:
        if self.replay_buffer is None:
            raise ValueError("OpalAgent requires a replay buffer for training")
        batch = _sample_transitions_from_buffer(
            self.replay_buffer, int(batch_size), rng=self.rng, with_encoder_inputs=False
        )
        observations = _field(batch, "observations")
        actions = _field(batch, "actions")
        next_observations = _field(batch, "next_observations")
        terminals = _field(batch, "terminals")
        if observations is None or actions is None or next_observations is None:
            raise RuntimeError("sampled transition batch is missing observations/actions/next_observations")
        if terminals is None:
            terminals = np.zeros(observations.shape[0], dtype=np.float32)
        return {
            "observations": self._normalize_states(as_2d(observations)),
            "actions": as_2d(actions),
            "next_observations": self._normalize_states(as_2d(next_observations)),
            "terminals": np.asarray(terminals, dtype=np.float32).reshape(-1),
        }

    # ------------------------------------------------------------------
    # losses / training
    # ------------------------------------------------------------------
    def encode(self, states: Any, sample: bool = False) -> np.ndarray:
        """Encode a set of states (or a batch of sets) into a latent skill."""
        self._require_networks()
        array = np.asarray(states, dtype=np.float32)
        batched = array.ndim == 3
        if not batched:
            array = array[None, :, :] if array.ndim == 2 else array.reshape(1, 1, -1)
        tensor = torch.as_tensor(array, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            z = self.encoder.encode(tensor, sample=sample)
        out = z.cpu().numpy()
        return out if batched else out[0]

    def vae_loss(self, batch_size: Optional[int] = None) -> Dict[str, float]:
        """Variational lower bound: reconstruction MSE + ``beta`` * KL(unit Gaussian)."""
        self._require_networks()
        batch_size = int(batch_size or self.config.batch_size)
        context = self._sample_context_states(batch_size)
        sampled_epoch = batch_size * int(max(1, self.config.num_decoder_states))
        transitions = {
            "observations": None,
            "actions": None,
            "next_observations": None,
        }
        arrays = self._sample_transition_arrays(sampled_epoch)
        context_tensor = torch.as_tensor(context, dtype=torch.float32, device=self.device)
        encoder_out = self.encoder(context_tensor, sample=True)
        # one shared skill per encoder context set, applied to K' transitions each
        z_expanded = encoder_out.z.unsqueeze(1).repeat(
            1, int(max(1, self.config.num_decoder_states)), 1
        ).reshape(sampled_epoch, -1)
        observations = torch.as_tensor(arrays["observations"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(arrays["actions"], dtype=torch.float32, device=self.device)
        next_observations = torch.as_tensor(
            arrays["next_observations"], dtype=torch.float32, device=self.device
        )
        recon = self.decoder.reconstruction_loss(observations, actions, z_expanded, next_observations)
        kl = encoder_out.kl_to_unit_gaussian().mean()
        loss = recon + float(self.config.beta) * kl
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self._clip_grads()
        self.optimizer.step()
        return {
            "vae_loss": float(loss.detach().cpu()),
            "reconstruction_loss": float(recon.detach().cpu()),
            "kl": float(kl.detach().cpu()),
        }

    def bc_loss(self, batch_size: Optional[int] = None) -> Dict[str, float]:
        """Behaviour cloning of the offline actions under a prior skill sample."""
        self._require_networks()
        batch_size = int(batch_size or self.config.batch_size)
        arrays = self._sample_transition_arrays(batch_size)
        skills = sample_unit_gaussian_skills(
            batch_size, self.latent_dim, rng=self.rng, std=self.config.skill_prior_std
        )
        observations = torch.as_tensor(arrays["observations"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(arrays["actions"], dtype=torch.float32, device=self.device)
        skills_tensor = torch.as_tensor(skills, dtype=torch.float32, device=self.device)
        log_prob = self.policy.log_prob(observations, skills_tensor, actions)
        loss = -log_prob.mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self._clip_grads()
        self.optimizer.step()
        return {
            "policy_loss": float(loss.detach().cpu()),
            "bc_log_prob": float(log_prob.mean().detach().cpu()),
        }

    def train_step(self, batch_size: Optional[int] = None) -> Dict[str, float]:
        """One joint update of the skill VAE and the latent-conditioned policy."""
        stats: Dict[str, float] = {}
        stats.update(self.vae_loss(batch_size=batch_size))
        stats.update(self.bc_loss(batch_size=batch_size))
        self.train_step_count += 1
        stats["step"] = float(self.train_step_count)
        self._last_train_stats = dict(stats)
        return stats

    def train(
        self,
        num_steps: Optional[int] = None,
        callback: Optional[Any] = None,
        log_interval: Optional[int] = None,
    ) -> List[Dict[str, float]]:
        """Train OPAL on the unlabeled offline dataset (no rewards are used)."""
        self._require_networks()
        total = int(num_steps or self.config.num_steps)
        interval = int(log_interval or self.config.log_interval)
        history: List[Dict[str, float]] = []
        for _ in range(total):
            stats = self.train_step()
            history.append(stats)
            if callback is not None:
                try:
                    callback(self.train_step_count, stats)
                except TypeError:
                    callback(stats)
            if interval > 0 and self.train_step_count % interval == 0:
                print(
                    "[OPAL] step {}/{} vae {:.4f} recon {:.4f} kl {:.4f} bc {:.4f}".format(
                        self.train_step_count,
                        total,
                        stats.get("vae_loss", float("nan")),
                        stats.get("reconstruction_loss", float("nan")),
                        stats.get("kl", float("nan")),
                        stats.get("policy_loss", float("nan")),
                    ),
                    flush=True,
                )
        return history

    fit = train  # convenience alias (matches GCBCAgent/GCIQLAgent)

    # ------------------------------------------------------------------
    # evaluation interface
    # ------------------------------------------------------------------
    def sample_skills(self, num_skills: Optional[int] = None, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Draw skills from the unit Gaussian prior (see the Addendum)."""
        rng = rng if rng is not None else self.rng
        return sample_unit_gaussian_skills(
            num_skills or self.num_skills,
            self.latent_dim,
            rng=rng,
            std=self.config.skill_prior_std,
        )

    def condition(self, task: Any = None, context: Any = None, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Return a single prior skill (reward functions are *not* used by OPAL)."""
        return self.sample_skills(1, rng=rng)

    def condition_many(
        self,
        task: Any = None,
        num_skills: Optional[int] = None,
        context: Any = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Return ``num_skills`` unit-Gaussian skills (privileged execution).

        Addendum: "10 random skills are sampled from a unit Gaussian, for each
        skill z, the policy is conditioned on it and evaluated for the entire
        episode, and the best performing rollout is taken" -- the best-of
        selection itself is performed by the shared rollout driver.
        """
        return self.sample_skills(num_skills or self.num_skills, rng=rng)

    @staticmethod
    def best_rollout_index(episode_scores: Sequence[float], episode_returns: Optional[Sequence[float]] = None) -> int:
        """Index of the best performing rollout over the sampled skills."""
        return select_best_rollout(episode_scores, episode_returns)

    def act(self, observation: Any, conditioning: Any, deterministic: bool = True) -> np.ndarray:
        """Action of the skill-conditioned policy for a single observation."""
        self._require_networks()
        skill = as_2d(conditioning)
        if skill.shape[0] > 1:
            skill = skill[:1]
        return self.policy.act_numpy(as_2d(observation)[:1], skill, deterministic=deterministic)[0]

    def act_batch(self, observations: Any, conditioning: Any, deterministic: bool = True) -> np.ndarray:
        """Actions for a batch of observations under one conditioning skill."""
        self._require_networks()
        skill = as_2d(conditioning)
        if skill.shape[0] > 1:
            skill = skill[:1]
        return self.policy.act_numpy(as_2d(observations), skill, deterministic=deterministic)

    # ------------------------------------------------------------------
    # persistence / introspection
    # ------------------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        if not _HAS_TORCH:
            return {}
        payload: Dict[str, Any] = {"config": self.config.to_dict(), "train_step_count": self.train_step_count}
        if self.encoder is not None:
            payload["encoder"] = self.encoder.state_dict()
        if self.decoder is not None:
            payload["decoder"] = self.decoder.state_dict()
        if self.policy is not None:
            payload["policy"] = self.policy.state_dict()
        return payload

    def load_state_dict(self, state: Dict[str, Any], load_optimizer: bool = True) -> None:
        self._require_networks()
        if "config" in state:
            self.config = OPALConfig.from_dict(state["config"])
        for name in ("encoder", "decoder", "policy"):
            module = getattr(self, name, None)
            if module is not None and name in state:
                module.load_state_dict(state[name])
        self.train_step_count = int(state.get("train_step_count", self.train_step_count))
        if load_optimizer and self.optimizer is not None and "optimizer" in state:
            try:
                self.optimizer.load_state_dict(state["optimizer"])
            except Exception:  # pragma: no cover
                pass

    def save(self, path: str) -> str:
        if not _HAS_TORCH:  # pragma: no cover
            raise RuntimeError("torch is required to save OPAL checkpoints")
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload = self.state_dict()
        if self.optimizer is not None:
            payload["optimizer"] = self.optimizer.state_dict()
        torch.save(payload, path)
        return path

    def load(self, path: str, load_optimizer: bool = True) -> Dict[str, Any]:
        if not _HAS_TORCH:  # pragma: no cover
            raise RuntimeError("torch is required to load OPAL checkpoints")
        payload = torch.load(path, map_location=self.device)
        self.load_state_dict(payload, load_optimizer=load_optimizer)
        return payload

    def describe(self) -> Dict[str, Any]:
        parameters = 0
        for module in (self.encoder, self.decoder, self.policy):
            if module is not None:
                parameters += sum(p.numel() for p in module.parameters())
        return {
            "name": self.name,
            "num_skills": self.num_skills,
            "uses_rewards": self.uses_rewards,
            "evaluation": "privileged: 10 unit-Gaussian skills, best rollout taken",
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "latent_dim": self.latent_dim,
            "num_parameters": parameters,
            "device": self.device,
            "config": self.config.to_dict(),
            "train_step_count": self.train_step_count,
            "last_train_stats": dict(self._last_train_stats),
        }


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------
def make_opal_agent(
    replay_buffer: Any = None,
    obs_dim: Optional[int] = None,
    act_dim: Optional[int] = None,
    latent_dim: Optional[int] = None,
    config: Optional[Any] = None,
    device: Optional[str] = None,
    seed: int = 0,
    **config_overrides: Any,
) -> OpalAgent:
    """Factory mirroring ``make_gc_bc_agent`` / ``make_gc_iql_agent``."""
    return OpalAgent(
        replay_buffer=replay_buffer,
        obs_dim=obs_dim,
        act_dim=act_dim,
        latent_dim=latent_dim,
        config=config,
        device=device,
        seed=seed,
        **config_overrides,
    )


__all__ = [
    "OpalAgent",
    "OPALConfig",
    "OPALEncoder",
    "OPALEncoderOutput",
    "OPALDynamicsDecoder",
    "OPALPolicy",
    "OPALPolicyOutput",
    "OPALTransformerBlock",
    "OPAL_DEFAULTS",
    "make_opal_agent",
    "sample_unit_gaussian_skills",
    "skill_log_prob",
    "select_best_rollout",
    "as_2d",
    "DEFAULT_NUM_SKILLS",
    "DEFAULT_LATENT_DIM",
    "DEFAULT_SKILL_PRIOR_STD",
    "DEFAULT_TARGET_STEPS",
    "DEFAULT_HIDDEN_DIMS",
    "DEFAULT_BETA",
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_BATCH_SIZE",
]


# ---------------------------------------------------------------------------
# self-check (no replay buffer / no MuJoCo needed)
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    skills = sample_unit_gaussian_skills(10, 8, rng=np.random.default_rng(0))
    print("skills", skills.shape, "mean~0", float(np.mean(skills)).__round__(3))
    assert skills.shape == (10, 8)
    assert select_best_rollout([0.0, 12.0, 3.0]) == 1
    assert select_best_rollout([1.0, 1.0, 1.0], [5.0, 9.0, 2.0]) == 1
    if _HAS_TORCH:
        agent = OpalAgent(obs_dim=8, act_dim=3, seed=0, num_context_states=4)
        context = np.random.default_rng(1).normal(size=(2, 4, 8)).astype(np.float32)
        z = agent.encode(context, sample=False)
        print("encoded z shape", z.shape)
        assert z.shape == (2, 128)
        out = agent.condition_many(None, num_skills=10)
        print("skill batch", out.shape)
        assert out.shape == (10, 128)
        action = agent.act(np.zeros(8, dtype=np.float32), out)
        print("action", np.asarray(action).round(3), "num_skills", agent.num_skills)
        assert np.asarray(action).shape == (3,)
        print(agent.describe()["evaluation"])
    print("opal self-check OK")
