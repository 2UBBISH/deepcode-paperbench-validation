"""FRE core: Functional Reward Encodings.

This package contains the central model components of FRE
(*Zero-Shot Reinforcement Learning via Functional Reward Encodings*):

* :mod:`fre.fre.reward_embeddings` -- reward discretisation + 128-d token builder
* :mod:`fre.fre.encoder`           -- permutation-invariant transformer posterior q_theta(z | context)
* :mod:`fre.fre.decoder`           -- reward decoder q_theta(eta(s) | s, z)
* :mod:`fre.fre.vae_loss`          -- info-bottleneck objective Eq.(6): MSE + beta * KL
* :mod:`fre.fre.latent_policy`     -- z-conditioned IQL networks Q(s, a, z), V(s, z), pi(a | s, z)

Everything is re-exported here for convenience so that downstream code can simply do::

    from fre.fre import FREEncoder, FREDecoder, FREModel, VAELoss, LatentPolicyBundle

All submodule imports are defensive: a failure in one submodule (e.g. a missing
optional dependency) will not prevent the others from being importable.
"""

from __future__ import annotations

from typing import List as _List

__all__: _List[str] = []


def _try_import(module: str, names: _List[str]) -> None:
    """Best-effort re-export of ``names`` from ``fre.fre.<module>``.

    Import errors are swallowed so that a partially-installed environment can
    still import the remaining components.
    """
    try:  # pragma: no cover - import shim
        mod = __import__(f"{__name__}.{module}", fromlist=names)
    except Exception:  # pragma: no cover
        try:
            mod = __import__(module, fromlist=names)
        except Exception:
            return
    for name in names:
        if hasattr(mod, name):
            globals()[name] = getattr(mod, name)
            if name not in __all__:
                __all__.append(name)


# --- reward embeddings / token construction ---------------------------------
_try_import(
    "reward_embeddings",
    [
        "NUM_REWARD_BINS",
        "REWARD_RESCALE",
        "STATE_EMB_DIM",
        "REWARD_EMB_DIM",
        "TOKEN_DIM",
        "REWARD_MIN",
        "REWARD_MAX",
        "discretize_reward",
        "reward_to_one_hot",
        "StateProjection",
        "RewardEmbedding",
        "RewardEncoderToken",
        "build_encoder_tokenizer",
        "check_reward_range",
    ],
)

# --- encoder ----------------------------------------------------------------
_try_import(
    "encoder",
    [
        "FREEncoder",
        "TransformerBlock",
        "MultiheadSelfAttention",
    ],
)

# --- decoder ----------------------------------------------------------------
_try_import(
    "decoder",
    [
        "FREDecoder",
        "FREModel",
    ],
)

# --- VAE / info-bottleneck loss ---------------------------------------------
_try_import(
    "vae_loss",
    [
        "VAELoss",
        "LossOutput",
        "DEFAULT_BETA",
        "kl_divergence",
        "gaussian_nll",
        "reward_prediction_loss",
        "info_bottleneck_loss",
        "fre_phase1_loss",
    ],
)

# --- z-conditioned policy networks ------------------------------------------
_try_import(
    "latent_policy",
    [
        "LatentMLP",
        "LatentQNetwork",
        "LatentVNetwork",
        "LatentGaussianPolicy",
        "LatentPolicyBundle",
        "concat_obs_z",
        "sample_latent_z",
        "DEFAULT_HIDDEN_SIZES",
        "DEFAULT_LATENT_DIM",
        "LOG_STD_MIN",
        "LOG_STD_MAX",
    ],
)

# --- convenience aliases -----------------------------------------------------
#: Latent dimension used throughout FRE (paper: z is 128-dimensional).
LATENT_DIM = globals().get("DEFAULT_LATENT_DIM", 128)

#: Number of (state, reward) context pairs the encoder consumes at train/eval.
CONTEXT_SIZE = 32

#: Number of held-out decoder states used in the Phase-1 objective.
DECODER_SIZE = 8

__all__ += ["LATENT_DIM", "CONTEXT_SIZE", "DECODER_SIZE"]


def build_fre_model(state_dim: int, latent_dim: int = 128, **kwargs):
    """Factory returning a :class:`FREModel` with the paper's default sizes.

    Parameters
    ----------
    state_dim:
        Dimensionality of the raw observation used by encoder and decoder.
    latent_dim:
        Dimensionality of the task embedding ``z`` (paper default ``128``).
    **kwargs:
        Forwarded to :class:`FREEncoder` (``num_blocks``, ``num_heads``,
        ``mlp_dim``, ``dropout``).
    """
    encoder = FREEncoder(state_dim=state_dim, latent_dim=latent_dim, **kwargs)
    decoder = FREDecoder(state_dim=state_dim, latent_dim=latent_dim)
    return FREModel(encoder=encoder, decoder=decoder)


def __getattr__(name: str):  # pragma: no cover - module attribute fallback
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
