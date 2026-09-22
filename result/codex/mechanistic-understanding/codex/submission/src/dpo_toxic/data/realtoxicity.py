"""RealToxicityPrompts (Gehman et al., 2020).

Section 3.3 evaluates interventions on the "challenge" subset of
RealToxicityPrompts: 1,199 prompts that are flagged as ``challenging`` in the
official release (prompts that are highly likely to elicit toxic continuations;
the flag is what the RealToxicityPrompts paper uses to define its challenge
set).  Section 4/5 additionally uses the full set of prompts for the logit-lens
and residual-stream analyses.
"""

from __future__ import annotations

from typing import Dict, List, Optional

RT_HF_ID = "allenai/real-toxicity-prompts"
CHALLENGE_THRESHOLD = 0.5
N_CHALLENGE_PROMPTS = 1199


def _prompt_field(example: Dict) -> Dict:
    p = example.get("prompt")
    if isinstance(p, dict):
        return p
    # Some mirrors flatten the prompt fields.
    return {"text": example.get("prompt.text", example.get("text", "")),
            "toxicity": example.get("prompt.toxicity", example.get("toxicity", 0.0))}


def load_realtoxicity_prompts(hf_id: str = RT_HF_ID, cache_dir: Optional[str] = None,
                              split: str = "train") -> List[Dict]:
    """Load all prompts as a list of ``{"text": str, "toxicity": float, ...}``."""
    from datasets import load_dataset

    ds = load_dataset(hf_id, cache_dir=cache_dir)
    if "challenge" in ds:
        ds_split = ds["challenge"]
    else:
        ds_split = ds[split] if split in ds else ds[list(ds.keys())[0]]
    out = []
    for ex in ds_split:
        p = _prompt_field(ex)
        tox = p.get("toxicity")
        out.append({
            "text": p.get("text", ""),
            "toxicity": float(tox) if tox is not None else 0.0,
            "challenging": bool(ex.get("challenging", False)),
            "continuation": (ex.get("continuation") or {}).get("text") if isinstance(ex.get("continuation"), dict) else None,
        })
    return out


def load_realtoxicity_challenge(hf_id: str = RT_HF_ID, cache_dir: Optional[str] = None,
                                threshold: float = CHALLENGE_THRESHOLD,
                                expected: int = N_CHALLENGE_PROMPTS) -> List[str]:
    """The 1,199 challenge prompts of RealToxicityPrompts.

    Preference order: the ``challenging`` flag of the official release (exactly
    1,199 prompts), then an explicit ``challenge`` split if a mirror provides
    one, then a fallback that keeps prompts whose own toxicity exceeds
    ``threshold``.
    """
    prompts = load_realtoxicity_prompts(hf_id=hf_id, cache_dir=cache_dir)
    challenge = [p["text"] for p in prompts if p.get("challenging")]
    if len(challenge) >= 100:
        return challenge
    challenge = [p["text"] for p in prompts if p["toxicity"] is not None and p["toxicity"] >= threshold]
    if len(challenge) < 100:  # mirror without toxicity scores -> fall back to everything
        challenge = [p["text"] for p in prompts]
    return challenge
