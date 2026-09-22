"""Model components for DPMs-ANT.

This subpackage collects the pieces needed to run the ANT transfer-learning
procedure of *Adapting Pretrained Diffusion Models for Few-Shot Image
Generation*:

* :mod:`dpm_ant.models.unet_loader` -- frozen DDPM (guided-diffusion) 256x256
  U-Net backbone with adaptor insertion into the residual "shift" blocks.
* :mod:`dpm_ant.models.ldm_loader` -- frozen LDM (CompVis) latent U-Net plus
  frozen :class:`AutoencoderKL`, again with shift-module adaptor insertion.
* :mod:`dpm_ant.models.adaptor` -- zero-initialized adaptor modules
  ``psi^l(x) = f(x W_down) W_up`` (Section 4.3 / 5.2), so the adapted network is
  numerically identical to the pretrained one before training.
* :mod:`dpm_ant.models.classifier` -- binary source/target classifier
  ``p_phi(y | x_t)`` used for the similarity-guided objective (Eq. 5 / Eq. 8).

Imports are performed lazily inside :func:`__getattr__` so that using the
subpackage does not force the (heavy) torch model definitions to be imported
until they are actually requested -- and so that a partial checkout (e.g. only
the toy experiment) still imports cleanly.
"""

from __future__ import annotations

from typing import Dict, List

__all__ = [
    # unet_loader (DDPM backbone)
    "UNetModel",
    "ResBlock",
    "AttentionBlock",
    "TimestepEmbedSequential",
    "FrozenDDPMUNet",
    "UNET_256_CONFIG",
    "build_unet_model",
    "load_guided_diffusion_unet",
    "insert_adaptors",
    "attach_adaptors_via_hooks",
    # ldm_loader (LDM backbone + autoencoder)
    "FrozenLDMUNet",
    "FrozenLDMAutoencoder",
    "LDM_256_CONFIG",
    "LDM_AUTOCODER_CONFIG",
    "load_ldm_unet",
    "load_autoencoder",
    "insert_adaptors_ldm",
    # adaptor
    "Adaptor",
    "AdaptorConfig",
    "build_adaptor",
    "build_adaptor_factory",
    "resolve_adaptor_config",
    # classifier
    "EncoderUNetModel",
    "AttentionPool2d",
    "PretrainedClassifier",
    "build_classifier",
    "load_pretrained_classifier",
    "classifier_logit_grad",
    "ENCODER_IMAGE_256_CONFIG",
    "ENCODER_IMAGE_64_CONFIG",
    # generic helpers
    "count_parameters",
    "adaptor_parameters",
]

#: Public name -> defining submodule (relative import path, module attribute).
_EXPORTS: Dict[str, str] = {
    # --- dpm_ant.models.unet_loader ------------------------------------
    "UNetModel": "unet_loader",
    "ResBlock": "unet_loader",
    "AttentionBlock": "unet_loader",
    "TimestepEmbedSequential": "unet_loader",
    "FrozenDDPMUNet": "unet_loader",
    "UNET_256_CONFIG": "unet_loader",
    "build_unet_model": "unet_loader",
    "load_guided_diffusion_unet": "unet_loader",
    "insert_adaptors": "unet_loader",
    "attach_adaptors_via_hooks": "unet_loader",
    # --- dpm_ant.models.ldm_loader -------------------------------------
    "FrozenLDMUNet": "ldm_loader",
    "FrozenLDMAutoencoder": "ldm_loader",
    "LDM_256_CONFIG": "ldm_loader",
    "LDM_AUTOCODER_CONFIG": "ldm_loader",
    "load_ldm_unet": "ldm_loader",
    "load_autoencoder": "ldm_loader",
    "insert_adaptors_ldm": "ldm_loader",
    # --- dpm_ant.models.adaptor ----------------------------------------
    "Adaptor": "adaptor",
    "AdaptorConfig": "adaptor",
    "build_adaptor": "adaptor",
    "build_adaptor_factory": "adaptor",
    "resolve_adaptor_config": "adaptor",
    # --- dpm_ant.models.classifier -------------------------------------
    "EncoderUNetModel": "classifier",
    "AttentionPool2d": "classifier",
    "PretrainedClassifier": "classifier",
    "build_classifier": "classifier",
    "load_pretrained_classifier": "classifier",
    "classifier_logit_grad": "classifier",
    "ENCODER_IMAGE_256_CONFIG": "classifier",
    "ENCODER_IMAGE_64_CONFIG": "classifier",
    # --- generic helpers (defined in both loaders / adaptor) -----------
    "count_parameters": "adaptor",
    "adaptor_parameters": "adaptor",
}


def __getattr__(name: str):
    """Lazily resolve the subpackage's public attributes (PEP 562)."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(f".{module_name}", __name__)
    try:
        value = getattr(module, name)
    except AttributeError as exc:  # pragma: no cover - defensive
        raise AttributeError(
            f"{module.__name__!r} does not define {name!r}; "
            "the model subpackage may be incomplete."
        ) from exc
    globals()[name] = value  # cache for subsequent lookups
    return value


def __dir__() -> List[str]:
    return sorted(set(list(globals().keys()) + __all__))
