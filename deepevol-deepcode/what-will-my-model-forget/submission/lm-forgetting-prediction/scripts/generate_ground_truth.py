#!/usr/bin/env python
"""Generate ground-truth forgetting labels ``z_ij`` and logit streams.

This CLI drives the machinery implemented in :mod:`src.forgetting.ground_truth`:

  * load the base PTLM ``f_0`` (BART0_L / FLAN-T5_L / FLAN-T5_3B / FLAN-T5_small),
  * build a model-refinement engine ``Head`` / ``LoRA`` / ``Full FT`` (Sec. 2, Sec. 4.1),
  * sample online error examples ``(x_i, y_i)`` from ``D_R^Train``,
  * refine ``f_0`` into ``f_i`` and evaluate it on the upstream pool ``D_PT_hat``,
  * record ``z_ij = 1[f_i(x_j) != y_j]``  (Sec. 2 definition -- NOT the Appendix F
    ``1[f_0(x_i) != f_i(x_i)]`` typo form),
  * persist the four logit streams ``f_0(x_i)``, ``f_i(x_i)``, ``f_0(x_j)``, ``f_i(x_j)``
    as top-k (``k = 100``) caches for cheap forecaster training / inference,
  * optionally estimate and cache the frequency prior ``b_j`` (Sec. 3.3),
  * optionally brute-force verify a subset of the labels.

Artifacts written under ``<artifact_root>/ground_truth/``:

    online.jsonl           one record per refined online example (f0/fi predictions)
    pairs.jsonl            one record per (i, j) pair: z_ij + cached logit streams
    pairs_train.jsonl      pair records whose online example is in D_R^Train
    pairs_test.jsonl       pair records whose online example is in D_R^Test
    online_train.jsonl     online records of the train split
    online_test.jsonl      online records of the test split
    meta.json              run metadata (model key, tuning, counts, prevalence, ...)
    frequency_prior.json   per-upstream frequency prior b_j (unless --no-prior)

Usage
-----
    python scripts/generate_ground_truth.py --model-key BART0_L --tuning full_ft \
        --max-online 200 --max-upstream 3600

    python scripts/generate_ground_truth.py --self-test     # offline smoke test
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import random
import sys
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger("generate_ground_truth")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

DEFAULT_CONFIG_PATH = os.path.join(_REPO_ROOT, "config", "config.yaml")
DEFAULT_GT_DIRNAME = "ground_truth"
DEFAULT_TRAIN_RATIO = 0.6


# --------------------------------------------------------------------------------------
# config / io helpers
# --------------------------------------------------------------------------------------
def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load the project YAML config; return ``{}`` on failure."""
    path = path or DEFAULT_CONFIG_PATH
    try:
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        if isinstance(cfg, dict):
            return cfg
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("could not load config %s: %s", path, exc)
    return {}


def cfg_get(cfg: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    """Nested dictionary lookup with a default."""
    cur: Any = cfg
    for key in keys:
        if isinstance(cur, Mapping) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read a JSONL file, skipping malformed lines."""
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("skipping malformed JSONL line in %s", path)
    return out


def write_json(path: str, payload: Any) -> str:
    """Write ``payload`` as JSON, creating parent directories."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return path


def write_jsonl(records: Sequence[Any], path: str) -> str:
    """Write a sequence of mappings / records as JSONL."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            if hasattr(rec, "to_dict"):
                rec = rec.to_dict()
            fh.write(json.dumps(rec, default=str) + "\n")
    return path


# --------------------------------------------------------------------------------------
# path resolution
# --------------------------------------------------------------------------------------
def artifact_root(config: Mapping[str, Any], model_key: str, tuning: Optional[str] = None) -> str:
    """``<output_dir>/<model_key>[/<tuning>]`` artifact root."""
    out = cfg_get(config, "output_dir", default="artifacts") or "artifacts"
    parts = [out, model_key]
    if tuning and tuning not in ("none", "None", ""):
        parts.append(tuning)
    return os.path.join(*parts)


def dataset_paths(
    config: Mapping[str, Any], model_key: str, tuning: Optional[str] = None
) -> Dict[str, str]:
    """Standard dataset artifact paths for one experiment."""
    root = artifact_root(config, model_key, tuning)
    return {
        "root": root,
        "d_pt": os.path.join(root, "d_pt.jsonl"),
        "d_pt_hat": os.path.join(root, "d_pt_hat.jsonl"),
        "d_r": os.path.join(root, "d_r.jsonl"),
        "d_r_train": os.path.join(root, "d_r_train.jsonl"),
        "d_r_test": os.path.join(root, "d_r_test.jsonl"),
        "gt_dir": os.path.join(root, DEFAULT_GT_DIRNAME),
        "cache_dir": cfg_get(config, "cache_dir", default=os.path.join("artifacts", "caches")),
    }


def _first_existing(paths: Sequence[str]) -> Optional[str]:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def resolve_upstream_examples(
    config: Mapping[str, Any], model_key: str, tuning: Optional[str] = None,
    upstream_file: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """Locate ``D_PT_hat`` (falling back to ``D_PT``)."""
    paths = dataset_paths(config, model_key, tuning)
    path = _first_existing([upstream_file, paths["d_pt_hat"], paths["d_pt"]])
    if not path:
        raise FileNotFoundError(
            "no upstream pool found; expected one of %s (run scripts/build_datasets.py)"
            % [upstream_file, paths["d_pt_hat"], paths["d_pt"]]
        )
    if os.path.basename(path).startswith("d_pt.jsonl"):
        logger.warning("D_PT_hat not found, falling back to unfiltered D_PT (%s)", path)
    return load_jsonl(path), path


def resolve_online_examples(
    config: Mapping[str, Any], model_key: str, tuning: Optional[str] = None,
    online_file: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """Locate ``D_R^Train`` (the online error pool used to generate labels)."""
    paths = dataset_paths(config, model_key, tuning)
    path = _first_existing([online_file, paths["d_r_train"], paths["d_r"]])
    if not path:
        raise FileNotFoundError(
            "no online error pool found; expected one of %s (run scripts/build_datasets.py)"
            % [online_file, paths["d_r_train"], paths["d_r"]]
        )
    return load_jsonl(path), path


# --------------------------------------------------------------------------------------
# model / engine construction
# --------------------------------------------------------------------------------------
def build_base_lm(model_key: str, config: Mapping[str, Any], device: str, dtype: str) -> Any:
    """Load the base PTLM ``f_0``."""
    from src.modeling.base_lm import load_base_lm  # local import (heavy)

    return load_base_lm(
        model_key=model_key,
        device=device,
        dtype=dtype,
        cache_dir=cfg_get(config, "cache_dir", default=None),
        max_input_len=int(cfg_get(config, "data", "max_input_len", default=512) or 512),
        max_output_len=int(cfg_get(config, "data", "max_output_len", default=64) or 64),
    )


def build_refine_fn(model_key: str, config: Mapping[str, Any], mode: str, base_lm: Any,
                    sequential: bool = False) -> Callable[..., Any]:
    """Build a refinement callable ``(example) -> f_i`` driven by ``RefinementEngine``.

    The engine API is probed defensively (``__call__`` / ``refine`` / ``refine_on_example``
    / ``fit_on_example``) so the driver keeps working if the engine grows aliases.
    """
    from src.modeling.refinement import build_refinement_engine  # local import (heavy)

    engine = build_refinement_engine(
        base_lm, model_key=model_key, mode=mode, config=config, sequential=sequential
    )

    candidate_names = ("__call__", "refine", "refine_on_example", "fit_on_example")
    methods = [(n, getattr(engine, n, None)) for n in candidate_names]
    methods = [(n, m) for n, m in methods if callable(m)]
    if not methods:
        raise AttributeError("refinement engine exposes no callable refinement entry point")
    logger.info("refinement engine methods: %s", [n for n, _ in methods])

    def refine_fn(*args: Any, **kwargs: Any) -> Any:
        example = _as_example(args, kwargs)
        last_exc: Optional[Exception] = None
        for name, method in methods:
            try:
                out = _call_with_example(method, example)
            except TypeError as exc:  # signature mismatch -> try the next alias
                last_exc = exc
                continue
            model = _extract_model(out)
            if model is not None:
                return model
            last_exc = RuntimeError("method %r returned no model (got %r)" % (name, type(out)))
        raise RuntimeError("all refinement entry points failed: %s" % (last_exc,))

    setattr(refine_fn, "engine", engine)
    return refine_fn


def _as_example(args: Sequence[Any], kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize the many ways ground-truth generation may call the refine hook."""
    if len(args) >= 1 and isinstance(args[0], Mapping):
        example = dict(args[0])
        example.update(kwargs or {})
        return example
    if len(args) >= 2:
        example = {"input": args[0], "target": args[1]}
        if len(args) >= 3 and isinstance(args[2], Mapping):
            example.update(args[2])
        example.update(kwargs or {})
        return example
    if kwargs:
        return dict(kwargs)
    raise TypeError("refine_fn called without an identifiable example")


def _call_with_example(fn: Callable[..., Any], example: Mapping[str, Any]) -> Any:
    """Call ``fn`` with the example in whichever form its signature accepts."""
    try:
        sig = inspect.signature(fn)
        params = [
            p for p in sig.parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
        ]
        names = {p.name for p in params}
        if len(params) <= 1:
            return fn(example)
        if "input" in names or "inputs" in names or "x" in names:
            return fn(example.get("input", example.get("inputs", "")), example.get("target", ""))
        return fn(example.get("input", ""), example.get("target", ""))
    except (TypeError, ValueError):
        pass
    try:
        return fn(example)
    except TypeError:
        return fn(example.get("input", ""), example.get("target", ""))


def _extract_model(out: Any) -> Optional[Any]:
    """Pull the refined model out of a refinement return value."""
    if out is None:
        return None
    if isinstance(out, tuple):
        for item in out:
            if hasattr(item, "predict") or hasattr(item, "generate"):
                return item
        return None
    if isinstance(out, Mapping):
        for key in ("model", "base_lm", "lm", "f_i", "refined"):
            if key in out:
                return out[key]
        return None
    return out


# --------------------------------------------------------------------------------------
# pair splitting
# --------------------------------------------------------------------------------------
def _rec_get(rec: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(rec, Mapping) and name in rec:
            return rec[name]
        if hasattr(rec, name):
            return getattr(rec, name)
    return default


def split_pairs_by_online(
    online_records: Sequence[Any],
    pair_records: Sequence[Any],
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    seed: int = 42,
) -> Dict[str, List[Any]]:
    """Split pair records 60/40 on the *online* example index (D_R^Train / D_R^Test)."""
    indices = sorted({int(_rec_get(o, "index", "i", default=k)) for k, o in enumerate(online_records)})
    rng = random.Random(seed)
    shuffled = list(indices)
    rng.shuffle(shuffled)
    n_train = int(round(len(shuffled) * float(train_ratio)))
    n_train = max(1, min(len(shuffled) - 1, n_train)) if len(shuffled) > 1 else len(shuffled)
    train_ids = set(shuffled[:n_train])
    test_ids = set(shuffled[n_train:])

    train_pairs, test_pairs = [], []
    for rec in pair_records:
        i = int(_rec_get(rec, "i", "online_index", default=-1))
        (train_pairs if i in train_ids else test_pairs).append(rec)

    train_online = [o for k, o in enumerate(online_records)
                    if int(_rec_get(o, "index", "i", default=k)) in train_ids]
    test_online = [o for k, o in enumerate(online_records)
                   if int(_rec_get(o, "index", "i", default=k)) in test_ids]
    return {
        "pairs_train": train_pairs,
        "pairs_test": test_pairs,
        "online_train": train_online,
        "online_test": test_online,
    }


# --------------------------------------------------------------------------------------
# prior estimation
# --------------------------------------------------------------------------------------
def estimate_prior(gt_dir: str, config: Mapping[str, Any], n_upstream: Optional[int] = None) -> Any:
    """Estimate and persist b_j from the freshly written ``pairs.jsonl``."""
    from src.forgetting.frequency_prior import (
        estimate_prior_from_ground_truth_dir,
        save_prior,
    )

    prior = estimate_prior_from_ground_truth_dir(gt_dir, n_upstream=n_upstream)
    save_prior(prior, os.path.join(gt_dir, "frequency_prior.json"))
    return prior


# --------------------------------------------------------------------------------------
# main driver
# --------------------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    from src.data.dataset_builders import (  # local import: pure-python helper
        DEFAULT_BART0_R_TASKS,
        DEFAULT_ID_TASKS,
        DEFAULT_OOD_TASKS,
    )  # noqa: F401  (keeps import-time validation close to the CLI)

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    p.add_argument("--model-key", default="BART0_L",
                   choices=["BART0_L", "FLAN-T5_L", "FLAN-T5_3B", "FLAN-T5_small"])
    p.add_argument("--tuning", "--mode", dest="tuning", default="full_ft",
                   choices=["head", "lora", "full_ft", "none"])
    p.add_argument("--sequential", action="store_true",
                   help="refine sequentially (drift across online examples)")
    p.add_argument("--online-file", default=None, help="override D_R^Train artifact path")
    p.add_argument("--upstream-file", default=None, help="override D_PT_hat artifact path")
    p.add_argument("--out-dir", default=None, help="override ground-truth output directory")
    p.add_argument("--max-online", type=int, default=200, help="number of online errors to refine")
    p.add_argument("--max-upstream", type=int, default=None, help="limit D_PT_hat size")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--topk", type=int, default=100)
    p.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default=None)
    p.add_argument("--no-prior", action="store_true", help="skip frequency-prior estimation")
    p.add_argument("--verify", type=int, default=0,
                   help="brute-force verify this many pair labels against f_i")
    p.add_argument("--self-test", action="store_true", help="offline smoke test with dummy models")
    return p.parse_args(argv)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    """Execute ground-truth generation and return the result manifest."""
    from src.forgetting import ground_truth as gt

    config = load_config(args.config)
    device = args.device or cfg_get(config, "device", default="cuda")
    dtype = args.dtype or cfg_get(config, "dtype", default="float32")
    seed = int(args.seed if args.seed is not None else cfg_get(config, "seed", default=42))

    tuning = None if args.tuning in ("none", None) else args.tuning
    if tuning is None and args.model_key in ("FLAN-T5_L", "FLAN-T5_3B"):
        logger.warning("refinement 'none' is unusual for %s; using 'full_ft'", args.model_key)
        tuning = "full_ft"

    online_examples, online_path = resolve_online_examples(
        config, args.model_key, tuning, args.online_file
    )
    upstream_examples, upstream_path = resolve_upstream_examples(
        config, args.model_key, tuning, args.upstream_file
    )
    logger.info("online pool %s (%d examples)", online_path, len(online_examples))
    logger.info("upstream pool %s (%d examples)", upstream_path, len(upstream_examples))

    online_indices = gt.sample_online_subset(online_examples, n=args.max_online, seed=seed)
    upstream_indices = gt.sample_upstream_subset(
        upstream_examples, n=args.max_upstream, seed=seed
    )
    logger.info("sampled %d online / %d upstream examples", len(online_indices), len(upstream_indices))

    f0 = build_base_lm(args.model_key, config, device, dtype)
    refine_fn = build_refine_fn(
        args.model_key, config, tuning or "full_ft", f0, sequential=bool(args.sequential)
    )

    online_records, pair_records = gt.generate_forgetting_pairs(
        f0,
        refine_fn,
        online_examples,
        upstream_examples,
        online_indices=online_indices,
        upstream_indices=upstream_indices,
        batch_size=int(args.batch_size),
        topk=int(args.topk),
        collect_logits=True,
        progress=True,
    )

    out_dir = args.out_dir or dataset_paths(config, args.model_key, tuning)["gt_dir"]
    os.makedirs(out_dir, exist_ok=True)

    summary = gt.summarize_records(online_records, pair_records)
    meta = {
        "model_key": args.model_key,
        "tuning": tuning or "none",
        "sequential": bool(args.sequential),
        "online_path": online_path,
        "upstream_path": upstream_path,
        "n_online": len(online_records),
        "n_upstream": len({int(_rec_get(r, "j", default=-1)) for r in pair_records}),
        "n_pairs": len(pair_records),
        "topk": int(args.topk),
        "batch_size": int(args.batch_size),
        "seed": seed,
        "summary": summary,
        # Sec. 2 label definition; Appendix F's z_ij uses x_i (flagged as a typo there).
        "label_definition": "z_ij = 1[f_i(x_j) != y_j] (Sec. 2)",
    }
    paths = gt.save_ground_truth_jsonl(online_records, pair_records, out_dir, meta=meta)

    split = split_pairs_by_online(
        online_records, pair_records, train_ratio=float(args.train_ratio), seed=seed
    )
    paths["pairs_train"] = write_jsonl(split["pairs_train"], os.path.join(out_dir, "pairs_train.jsonl"))
    paths["pairs_test"] = write_jsonl(split["pairs_test"], os.path.join(out_dir, "pairs_test.jsonl"))
    paths["online_train"] = write_jsonl(split["online_train"], os.path.join(out_dir, "online_train.jsonl"))
    paths["online_test"] = write_jsonl(split["online_test"], os.path.join(out_dir, "online_test.jsonl"))

    if not args.no_prior:
        try:
            prior = estimate_prior(
                out_dir, config, n_upstream=meta["n_upstream"] or None
            )
            paths["frequency_prior"] = os.path.join(out_dir, "frequency_prior.json")
            meta["prior_positive_ratio"] = getattr(prior, "positive_ratio", lambda: None)()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("frequency prior estimation failed: %s", exc)

    if args.verify and args.verify > 0:
        try:
            verify_result = gt.verify_labels_bruteforce(
                f0, upstream_examples, pair_records[: int(args.verify)], batch_size=int(args.batch_size)
            )
            meta["verify"] = verify_result
            logger.info("brute-force verification: %s", verify_result)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("brute-force verification failed: %s", exc)

    write_json(os.path.join(out_dir, "run_meta.json"), meta)
    logger.info("ground truth written to %s", out_dir)
    logger.info("summary: %s", summary)
    return {"out_dir": out_dir, "paths": paths, "meta": meta, "summary": summary}


# --------------------------------------------------------------------------------------
# offline self-test
# --------------------------------------------------------------------------------------
def _self_test() -> int:
    """Offline smoke test: dummy LM + dummy refinement, no downloads."""
    import torch

    from src.forgetting import ground_truth as gt

    vocab_size, dim, max_len = 16, 8, 6

    class _DummyLM:
        def __init__(self, flip: float = 0.0, seed: int = 0):
            self.flip = float(flip)
            self.rng = random.Random(seed)

        def predict(self, inputs, **kwargs):
            return ["correct"] * len(inputs)

        def predict_batch(self, inputs, **kwargs):
            return self.predict(inputs, **kwargs)

        def generate(self, inputs, **kwargs):
            return self.predict(inputs, **kwargs)

        def token_logits(self, inputs, targets, **kwargs):
            n = len(inputs)
            logits = torch.randn(n, max_len, vocab_size)
            target_ids = torch.zeros(n, max_len, dtype=torch.long)
            return {"logits": logits, "target_ids": target_ids}

    def dummy_refine(example):
        # every 4th online example "forgets" some upstream examples
        flip = 0.5 if int(example.get("id", 0) or 0) % 4 == 0 else 0.0
        return _DummyLM(flip=flip, seed=1)

    online = [{"id": i, "input": "q%d" % i, "target": "correct", "task": "toy"} for i in range(8)]
    upstream = [{"id": j, "input": "u%d" % j, "target": "correct", "task": "toy"} for j in range(12)]

    online_records, pair_records = gt.generate_forgetting_pairs(
        _DummyLM(), dummy_refine, online, upstream,
        online_indices=list(range(4)), upstream_indices=list(range(6)),
        batch_size=2, topk=5, progress=False,
    )
    assert online_records, "no online records produced"
    assert pair_records, "no pair records produced"
    summary = gt.summarize_records(online_records, pair_records)
    split = split_pairs_by_online(online_records, pair_records, seed=42)
    assert len(split["pairs_train"]) + len(split["pairs_test"]) == len(pair_records)

    tmp = os.path.join(_REPO_ROOT, "artifacts", "_self_test_gt")
    paths = gt.save_ground_truth_jsonl(online_records, pair_records, tmp, meta={"self_test": True})
    reloaded = gt.load_ground_truth_jsonl(paths["pairs"]) if "pairs" in paths else None
    write_jsonl(split["pairs_train"], os.path.join(tmp, "pairs_train.jsonl"))
    logger.info("self-test OK: %d online, %d pairs, summary=%s, reloaded=%s",
                len(online_records), len(pair_records), summary,
                None if reloaded is None else len(reloaded))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = parse_args(argv)
    if args.self_test:
        return _self_test()
    result = run(args)
    print(json.dumps({"out_dir": result["out_dir"], "summary": result["summary"]}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
