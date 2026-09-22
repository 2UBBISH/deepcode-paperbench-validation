"""Data layer for the Robust CLIP reproduction.

Provides dataset loaders and schema normalization for every benchmark used by
the evaluation harnesses:

* :mod:`robust_clip_repro.data.imagenet` -- HuggingFace ``imagenet-1k`` with
  ``trust_remote_code=True`` (Addendum requirement), deterministic CLIP-style
  preprocessing and a **raw, non-normalized** pixel access path so PGD/APGD can
  project l_inf/l_2 balls around un-normalized inputs.
* :mod:`robust_clip_repro.data.benchmarks` -- TextVQA / POPE / SQA-I loaders
  mirroring the LLaVA harnesses, plus the Addendum hooks: top-5 most frequent
  ground truths, arg-min ground-truth selection, per-sample scoring and the
  "skip the ``Word`` attack on TextVQA" rule.
* :mod:`robust_clip_repro.data.coco_captioning` -- COCO / Flickr30k image and
  reference-caption loader (raw pixel tensors preserved for attack projection).
* :mod:`robust_clip_repro.data.jailbreak_assets` -- the upstream Qi et al.
  (2023) assets: a single ``clean.jpeg`` source image, the harmful target
  corpus ``derogatory_corpus.csv`` and the evaluation prompt corpus
  ``manual_harmful_instructions.csv``.

Every module keeps heavy dependencies (``torch``, ``datasets``, ``PIL``)
lazily imported so that schema/normalization helpers stay offline-testable.

This package is intentionally import-light: nothing is eagerly imported here,
so ``import robust_clip_repro.data`` never triggers a dataset download or a
torch import.  Use the :func:`__getattr__` lazy namespace (PEP 562) to reach
the individual loaders.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

__all__: List[str] = [
    "imagenet",
    "benchmarks",
    "coco_captioning",
    "jailbreak_assets",
    "available_datasets",
    "preload",
    "load_dataset_by_name",
]

# Submodules bundled with this package.
_SUBMODULES = ("imagenet", "benchmarks", "coco_captioning", "jailbreak_assets")

# Public names re-exported lazily: name -> (submodule, attribute).
_EXPORTS: Dict[str, Any] = {
    # ---------------------------------------------------------------- ImageNet
    "ImageNetSample": ("imagenet", "ImageNetSample"),
    "load_imagenet": ("imagenet", "load_imagenet"),
    "load_imagenet_dataset": ("imagenet", "load_imagenet_dataset"),
    "load_hf_split": ("imagenet", "load_hf_split"),
    "image_to_pixels": ("imagenet", "image_to_pixels"),
    "normalize_pixels": ("imagenet", "normalize_pixels"),
    "denormalize_pixels": ("imagenet", "denormalize_pixels"),
    "normalize_imagenet_sample": ("imagenet", "normalize_imagenet_sample"),
    "iter_batches": ("imagenet", "iter_batches"),
    "synthetic_imagenet_samples": ("imagenet", "synthetic_samples"),
    "CLIP_MEAN": ("imagenet", "CLIP_MEAN"),
    "CLIP_STD": ("imagenet", "CLIP_STD"),
    "HF_DATASET_ID": ("imagenet", "HF_DATASET_ID"),
    # -------------------------------------------------------------- benchmarks
    "VQASample": ("benchmarks", "VQASample"),
    "load_benchmark": ("benchmarks", "load_benchmark"),
    "load_benchmarks": ("benchmarks", "load_benchmarks"),
    "load_textvqa": ("benchmarks", "load_textvqa"),
    "load_pope": ("benchmarks", "load_pope"),
    "load_sqa_i": ("benchmarks", "load_sqa_i"),
    "normalize_dataset_name": ("benchmarks", "normalize_dataset_name"),
    "normalize_sample": ("benchmarks", "normalize_sample"),
    "top_k_ground_truths": ("benchmarks", "top_k_ground_truths"),
    "top_ground_truth": ("benchmarks", "top_ground_truth"),
    "ground_truth_counts": ("benchmarks", "ground_truth_counts"),
    "select_lowest_scoring_ground_truth": (
        "benchmarks",
        "select_lowest_scoring_ground_truth",
    ),
    "should_skip_word_attack": ("benchmarks", "should_skip_word_attack"),
    "sample_to_pixels": ("benchmarks", "sample_to_pixels"),
    "make_score_fn": ("benchmarks", "make_score_fn"),
    "vqa_accuracy": ("benchmarks", "vqa_accuracy"),
    "TEXT_VQA": ("benchmarks", "TEXT_VQA"),
    "POPE": ("benchmarks", "POPE"),
    "SQA_I": ("benchmarks", "SQA_I"),
    "DATASET_NAMES": ("benchmarks", "DATASET_NAMES"),
    "TOP_K_GROUND_TRUTHS": ("benchmarks", "TOP_K_GROUND_TRUTHS"),
    # ---------------------------------------------------------- captioning data
    "load_captioning_dataset": ("coco_captioning", "load_captioning_dataset"),
    "load_coco": ("coco_captioning", "load_coco"),
    "load_flickr30k": ("coco_captioning", "load_flickr30k"),
    "normalize_captioning_sample": (
        "coco_captioning",
        "normalize_captioning_sample",
    ),
    "extract_captions": ("coco_captioning", "extract_captions"),
    "synthetic_samples": ("coco_captioning", "synthetic_samples"),
    "COCO": ("coco_captioning", "COCO"),
    "FLICKR30K": ("coco_captioning", "FLICKR30K"),
    "DEFAULT_MAX_GROUND_TRUTHS": ("coco_captioning", "DEFAULT_MAX_GROUND_TRUTHS"),
    # ----------------------------------------------------------- jailbreak data
    "JailbreakAssets": ("jailbreak_assets", "JailbreakAssets"),
    "load_jailbreak_assets": ("jailbreak_assets", "load_jailbreak_assets"),
    "load_clean_image": ("jailbreak_assets", "load_clean_image"),
    "load_source_image": ("jailbreak_assets", "load_source_image"),
    "load_derogatory_corpus": ("jailbreak_assets", "load_derogatory_corpus"),
    "load_target_strings": ("jailbreak_assets", "load_target_strings"),
    "load_manual_harmful_instructions": (
        "jailbreak_assets",
        "load_manual_harmful_instructions",
    ),
    "load_eval_prompts": ("jailbreak_assets", "load_eval_prompts"),
    "ensure_all_assets": ("jailbreak_assets", "ensure_all_assets"),
    "resolve_assets_dir": ("jailbreak_assets", "resolve_assets_dir"),
    "AssetUnavailableError": ("jailbreak_assets", "AssetUnavailableError"),
    "CLEAN_IMAGE_URL": ("jailbreak_assets", "CLEAN_IMAGE_URL"),
    "DEROGATORY_CORPUS_URL": ("jailbreak_assets", "DEROGATORY_CORPUS_URL"),
    "MANUAL_HARMFUL_INSTRUCTIONS_URL": (
        "jailbreak_assets",
        "MANUAL_HARMFUL_INSTRUCTIONS_URL",
    ),
}

# ``synthetic_samples`` is exported under a disambiguated alias above; keep the
# bare name pointing at the captioning implementation for backwards
# compatibility with :mod:`eval_captioning`.
_EXPORTS["synthetic_samples"] = ("coco_captioning", "synthetic_samples")


def available_datasets() -> List[str]:
    """Return the bundled dataset submodule names."""
    return list(_SUBMODULES)


def list_exports() -> Dict[str, str]:
    """Map every lazy public name to its ``"submodule.attribute"`` source."""
    return {
        name: f"{mod}.{attr}"
        for name, (mod, attr) in sorted(_EXPORTS.items())
    }


def preload(modules: Optional[Any] = None) -> Dict[str, bool]:
    """Eagerly import the given submodules (or all of them).

    Never raises for a missing *optional* dependency: failures are reported as
    ``False`` so smoke tests can detect absent packages (e.g. ``datasets``)
    without aborting the run.

    Returns
    -------
    dict
        ``submodule -> success`` mapping.
    """
    import importlib

    if modules is None:
        names = list(_SUBMODULES)
    elif isinstance(modules, str):
        names = [modules]
    else:
        names = list(modules)

    results: Dict[str, bool] = {}
    for name in names:
        full = f"{__name__}.{name}"
        try:
            importlib.import_module(full)
            results[name] = True
        except Exception:  # pragma: no cover - optional dependency missing
            results[name] = False
    return results


def load_dataset_by_name(dataset_name: str, **kwargs: Any) -> Any:
    """Dispatch to the right loader using a canonical dataset name.

    ``ImageNet``/``imagenet`` -> :func:`imagenet.load_imagenet_dataset`,
    ``TextVQA``/``POPE``/``SQA-I`` -> :func:`benchmarks.load_benchmark`,
    ``COCO``/``Flickr30k`` -> :func:`coco_captioning.load_captioning_dataset`.
    """
    from . import benchmarks as _bm
    from . import coco_captioning as _cap
    from . import imagenet as _im

    key = str(dataset_name).strip().lower().replace("-", "").replace("_", "")
    if key in {"imagenet", "imagenet1k", "ILSVRC".lower()}:
        return _im.load_imagenet_dataset(**kwargs)
    if key in {"coco", "coco2014", "captioning", "flickr30k", "flickr"}:
        return _cap.load_captioning_dataset(dataset_name=dataset_name, **kwargs)
    # Fall back to the VQA benchmark loaders (TextVQA / POPE / SQA-I).
    return _bm.load_benchmark(dataset_name, **kwargs)


def __getattr__(name: str) -> Any:
    """Lazily resolve public names and submodules (PEP 562)."""
    import importlib

    if name in _SUBMODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module

    if name in _EXPORTS:
        submodule, attribute = _EXPORTS[name]
        module = importlib.import_module(f"{__name__}.{submodule}")
        value = getattr(module, attribute)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    """Expose lazily available names for introspection."""
    return sorted(set(list(globals().keys()) + __all__ + list(_EXPORTS)))
