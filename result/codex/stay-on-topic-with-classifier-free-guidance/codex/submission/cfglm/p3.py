"""Sampling the dataset used by Section 5.

The paper runs its interpretability analysis on

    "a sample dataset of 32,902 datapoints from P3 (Sanh et al., 2021)"

and the addendum clarifies how the sample is drawn:

    "~50 samples are randomly sampled from each of the 660 datasets in P3.
     Some datasets have less than 50 samples in which case you just take the
     entire dataset.  When sampling, the authors filtered out samples that
     had an input length longer than 200 tokens ..."

The HuggingFace ``bigscience/P3`` repository exposes the P3 prompts as
individual configurations (one per dataset/prompt-template pair); the state
of the hub changes over time, so this module enumerates whatever
configurations are available, samples up to ``n_per_dataset`` documents from
each (whole-config when the config is smaller), and filters on input length.
``P3Sampler.total`` reports how many datapoints were actually collected,
which can be compared against the paper's 32,902.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence


def list_p3_configs(dataset_path: str = "bigscience/P3") -> List[str]:
    """Enumerate the P3 configurations available on the Hub."""
    from datasets import get_dataset_config_names

    return sorted(get_dataset_config_names(dataset_path))


def pretty_name(config: str) -> str:
    """Turn a P3 config name into the label style used in the paper's tables.

    e.g. ``super_glue_wsc.fixed_p_is_are_r_score_eval`` becomes
    ``super glue wsc.fixed p is are r score eval``.
    """
    return config.replace("_", " ")


@dataclass
class P3Example:
    """One sampled P3 datapoint."""

    config: str
    index: int
    input_text: str
    target_text: str
    answer_choices: Optional[list] = None

    @property
    def dataset_label(self) -> str:
        return pretty_name(self.config)


@dataclass
class P3Sampler:
    """Draw the Section 5 sample from P3."""

    dataset_path: str = "bigscience/P3"
    n_per_dataset: int = 50
    max_input_tokens: int = 200
    seed: int = 42
    split: str = "train"
    configs: Optional[Sequence[str]] = None
    examples: List[P3Example] = field(default_factory=list)
    n_skipped_long: int = 0

    @property
    def total(self) -> int:
        return len(self.examples)

    def sample(self, limit_datasets: Optional[int] = None) -> List[P3Example]:
        from datasets import load_dataset
        from transformers import AutoTokenizer

        rng = random.Random(self.seed)
        configs = list(self.configs) if self.configs else list_p3_configs(self.dataset_path)
        if limit_datasets is not None:
            configs = configs[:limit_datasets]
        # A whitespace-ish tokenizer proxies the paper's "input length > 200
        # tokens" filter without downloading a model; a GPT-2 tokenizer is
        # used when available.
        tokenizer = None
        try:  # pragma: no cover - depends on the environment
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
        except Exception:
            tokenizer = None

        examples: List[P3Example] = []
        for config in configs:
            try:
                dataset = load_dataset(self.dataset_path, config, split=self.split)
            except Exception as exc:  # pragma: no cover
                print(f"[p3] skipping {config}: {exc}")
                continue
            n = min(len(dataset), self.n_per_dataset)
            indices = rng.sample(range(len(dataset)), n) if len(dataset) > n else list(range(len(dataset)))
            for idx in indices:
                row = dataset[idx]
                input_text = str(row.get("inputs_pretokenized", ""))
                target_text = str(row.get("targets_pretokenized", ""))
                if tokenizer is not None:
                    n_tokens = len(tokenizer(input_text).input_ids)
                else:  # pragma: no cover
                    n_tokens = len(input_text.split())
                if n_tokens > self.max_input_tokens:
                    self.n_skipped_long += 1
                    continue
                choices = row.get("answer_choices")
                examples.append(
                    P3Example(
                        config=config,
                        index=int(idx),
                        input_text=input_text,
                        target_text=target_text,
                        answer_choices=list(choices) if choices is not None else None,
                    )
                )
        self.examples = examples
        return examples

    def per_dataset_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for example in self.examples:
            counts[example.dataset_label] = counts.get(example.dataset_label, 0) + 1
        return counts
