"""Section 4.2 -- constructing the pairwise toxicity dataset.

For every prompt (a sentence from Wikitext-2) the paper generates

* a **positive** (non-toxic) continuation with greedy sampling from GPT2, and
* a **negative** (toxic) continuation with PPLM, using the toxicity probe
  ``W_toxic`` as the attribute classifier,

for a total of **24,576** preference pairs.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch

from .generation import generate_continuations
from .pplm import AttributeClassifier, PPLMConfig, PPLMGenerator
from .utils import load_tokenizer, save_json, set_seed


@dataclass
class PairConfig:
    n_pairs: int = 24576
    n_tokens: int = 20                 # length of both continuations
    prompt_tokens: int = 0             # 0 == use the whole Wikitext sentence
    greedy_batch_size: int = 32
    probe_threshold: float = 0.5       # keep pairs whose negative is classified toxic
    filter_with_probe: bool = True
    toxicity_model: Optional[str] = None   # e.g. "unitary/unbiased-toxic-roberta"
    toxicity_threshold: float = 0.5
    dedupe: bool = True
    seed: int = 0
    wikitext_split: str = "train"
    shard_size: int = 256
    pplm_num_iterations: int = 50   # Table 9 (lower it for smoke tests)


def _load_prompts(cfg: PairConfig, cache_dir: Optional[str] = None) -> List[str]:
    from .data.wikitext import wikitext_sentences

    prompts = wikitext_sentences(split=cfg.wikitext_split, min_chars=50, cache_dir=cache_dir)
    rng = random.Random(cfg.seed)
    rng.shuffle(prompts)
    return prompts


def build_pair_dataset(model: torch.nn.Module, probe, cfg: Optional[PairConfig] = None,
                       out_path: str = "artifacts/pairs/pairs.jsonl",
                       cache_dir: Optional[str] = None,
                       device: Optional[str] = None,
                       tokenizer=None,
                       toxicity_scorer=None,
                       resume: bool = True,
                       progress_every: int = 50) -> Dict:
    """Generate the 24,576 toxic/non-toxic preference pairs of Section 4.2."""
    cfg = cfg or PairConfig()
    set_seed(cfg.seed)
    device = device or str(next(model.parameters()).device)
    tokenizer = tokenizer or load_tokenizer()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prompts = _load_prompts(cfg, cache_dir=cache_dir)
    start_idx = 0
    if resume and out_path.exists():
        with open(out_path) as f:
            start_idx = sum(1 for _ in f)
        if start_idx >= cfg.n_pairs:
            return {"n_pairs": start_idx, "path": str(out_path), "resumed": True}

    pplm_cfg = PPLMConfig(min_length=cfg.n_tokens, max_length=cfg.n_tokens, top_k=10,
                          num_iterations=cfg.pplm_num_iterations)
    classifier = AttributeClassifier(probe, tokenizer=tokenizer, device=device)
    generator = PPLMGenerator(model, classifier, cfg=pplm_cfg, device=device, tokenizer=tokenizer)

    n_written = start_idx
    cursor = start_idx  # every prompt yields at most one pair
    mode = "a" if start_idx else "w"
    with open(out_path, mode) as f:
        while n_written < cfg.n_pairs and cursor < len(prompts):
            chunk = prompts[cursor: cursor + cfg.shard_size]
            cursor += len(chunk)
            # 1) greedy (non-toxic) continuations, batched
            positives = generate_continuations(
                model, tokenizer, chunk, max_new_tokens=cfg.n_tokens,
                batch_size=cfg.greedy_batch_size, device=device, do_sample=False)
            # 2) PPLM (toxic) continuations, one prompt at a time
            for prompt, positive in zip(chunk, positives):
                if n_written >= cfg.n_pairs:
                    break
                negative = generator.generate(prompt, seed=cfg.seed + n_written)
                if not negative.strip():
                    continue
                if cfg.filter_with_probe:
                    with torch.no_grad():
                        ids = tokenizer(prompt + negative, return_tensors="pt",
                                        truncation=True, max_length=256).input_ids.to(device)
                        hidden = model(input_ids=ids, output_hidden_states=True).hidden_states[-1]
                        mean_hidden = hidden.mean(dim=1)
                        p_toxic = float(classifier.prob(mean_hidden)[0])
                    if p_toxic < cfg.probe_threshold:
                        continue
                if cfg.toxicity_model is not None and toxicity_scorer is not None:
                    score = toxicity_scorer.score([negative])[0]
                    if score < cfg.toxicity_threshold:
                        continue
                record = {
                    "prompt": prompt,
                    "chosen": positive,
                    "rejected": negative,
                    "index": n_written,
                }
                f.write(json.dumps(record) + "\n")
                n_written += 1
                if n_written % progress_every == 0:
                    f.flush()
                    print(f"[pairs] {n_written}/{cfg.n_pairs}")
    stats = {"n_pairs": n_written, "path": str(out_path), "config": cfg.__dict__}
    save_json(stats, out_path.with_suffix(".stats.json"))
    return stats


def load_pairs(path: str | os.PathLike) -> List[Dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
