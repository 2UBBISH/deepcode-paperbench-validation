"""Wikitext-2 (Merity et al., 2016).

Wikitext-2 is used twice in the paper: as the source of prompts for the
preference-pair construction (Section 4.2) and as the perplexity benchmark
(Section 3.3).
"""

from __future__ import annotations

from typing import List, Optional

WIKITEXT_HF_ID = "Salesforce/wikitext"
WIKITEXT_CONFIG = "wikitext-2-raw-v1"


def load_wikitext2(hf_id: str = WIKITEXT_HF_ID, config: str = WIKITEXT_CONFIG,
                   cache_dir: Optional[str] = None):
    from datasets import load_dataset

    return load_dataset(hf_id, config, cache_dir=cache_dir)


def wikitext_sentences(split: str = "test", min_chars: int = 200,
                       hf_id: str = WIKITEXT_HF_ID, config: str = WIKITEXT_CONFIG,
                       cache_dir: Optional[str] = None) -> List[str]:
    """Non-trivial lines of a Wikitext split (used as sentence prompts)."""
    ds = load_wikitext2(hf_id=hf_id, config=config, cache_dir=cache_dir)[split]
    out = []
    for text in ds["text"]:
        t = text.strip()
        # Wikitext-2 raw contains headings (``= Title =``) and empty lines.
        if len(t) >= min_chars and not (t.startswith("=") and t.endswith("=")):
            out.append(t)
    return out
