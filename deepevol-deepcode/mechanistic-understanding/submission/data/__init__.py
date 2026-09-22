"""Data loaders for the DPO/toxicity mechanistic-interpretability reproduction.

This package groups the four dataset components used by the reproduction:

* :mod:`data.jigsaw`        -- Jigsaw toxic-comment data for training the ``W_Toxic`` probe (§3.1).
* :mod:`data.wikitext`      -- Wikitext-2 prompts (PPLM/greedy pair generation) and the PPL corpus.
* :mod:`data.realtoxicity`  -- RealToxicityPrompts challenge subset (1,199 prompts) and the 295
                               ``sh*t``-eliciting prompts used for the logit-lens figure.
* :mod:`data.pairwise`      -- preference-pair container for the 24,576 PPLM-toxic / greedy-non-toxic
                               pairs and their 90:10 split (used by DPO training).

All heavy third-party imports (``datasets``, ``torch``, ``transformers``) are performed lazily
inside the loader functions, so importing this package never touches the network or requires a GPU.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------------------------
# Jigsaw (toxicity probe training data)
# ---------------------------------------------------------------------------------------------
from .jigsaw import (  # noqa: F401
    JIGSAW_HF_NAME,
    N_TOTAL_COMMENTS,
    JigsawData,
    JigsawSplit,
    binarise_labels,
    iter_batches,
    load_jigsaw,
    load_jigsaw_raw,
    stratified_split,
    tokenize_comments,
)

# ---------------------------------------------------------------------------------------------
# Wikitext-2 (prompts for pair generation / F1, and the perplexity corpus)
# ---------------------------------------------------------------------------------------------
from .wikitext import (  # noqa: F401
    N_F1_SENTENCES,
    N_PAIRS,
    WIKITEXT2_CONFIG,
    WIKITEXT2_HF_NAME,
    PPLCorpus,
    iter_sentences,
    load_ppl_corpus,
    load_wikitext2_ppl_text,
    load_wikitext2_raw,
    load_wikitext2_sentences,
    ppl_windows,
    prompt_pool,
    split_sentences,
    wiki_sentences_for_f1,
)

# ---------------------------------------------------------------------------------------------
# RealToxicityPrompts (evaluation prompts)
# ---------------------------------------------------------------------------------------------
from .realtoxicity import (  # noqa: F401
    N_CHALLENGE_PROMPTS,
    N_SHIT_PROMPTS,
    RTP_HF_NAME,
    TARGET_TOKEN,
    RTPPrompt,
    challenge_prompts,
    is_target_token,
    load_prompts,
    load_rtp_from_jsonl,
    load_rtp_raw,
    load_target_token_prompts,
    next_token_ids,
    normalise_token,
    prompt_texts,
    sample_prompts,
    save_prompts,
    select_target_token_prompts,
    toxicity_stats,
)

# ---------------------------------------------------------------------------------------------
# Pairwise preference data (DPO training data)
# ---------------------------------------------------------------------------------------------
from .pairwise import (  # noqa: F401
    DEFAULT_PAIRS_PATH,
    DEFAULT_SHARD_DIR,
    SHARD_SIZE,
    VALID_RATIO,
    PairExample,
    PairSplit,
    PairwiseDataset,
    append_pairs,
    build_dataset,
    count_existing_pairs,
    deduplicate,
    existing_shards,
    load_pairwise_dataset,
    load_pairs,
    load_shards,
    make_pair,
    merge_shards,
    save_pairs,
    shard_path,
    to_hf_dataset,
)

__all__ = [
    # jigsaw
    "JIGSAW_HF_NAME",
    "N_TOTAL_COMMENTS",
    "JigsawData",
    "JigsawSplit",
    "binarise_labels",
    "iter_batches",
    "load_jigsaw",
    "load_jigsaw_raw",
    "stratified_split",
    "tokenize_comments",
    # wikitext
    "N_F1_SENTENCES",
    "N_PAIRS",
    "WIKITEXT2_CONFIG",
    "WIKITEXT2_HF_NAME",
    "PPLCorpus",
    "iter_sentences",
    "load_ppl_corpus",
    "load_wikitext2_ppl_text",
    "load_wikitext2_raw",
    "load_wikitext2_sentences",
    "ppl_windows",
    "prompt_pool",
    "split_sentences",
    "wiki_sentences_for_f1",
    # realtoxicity
    "N_CHALLENGE_PROMPTS",
    "N_SHIT_PROMPTS",
    "RTP_HF_NAME",
    "TARGET_TOKEN",
    "RTPPrompt",
    "challenge_prompts",
    "is_target_token",
    "load_prompts",
    "load_rtp_from_jsonl",
    "load_rtp_raw",
    "load_target_token_prompts",
    "next_token_ids",
    "normalise_token",
    "prompt_texts",
    "sample_prompts",
    "save_prompts",
    "select_target_token_prompts",
    "toxicity_stats",
    # pairwise
    "DEFAULT_PAIRS_PATH",
    "DEFAULT_SHARD_DIR",
    "SHARD_SIZE",
    "VALID_RATIO",
    "PairExample",
    "PairSplit",
    "PairwiseDataset",
    "append_pairs",
    "build_dataset",
    "count_existing_pairs",
    "deduplicate",
    "existing_shards",
    "load_pairwise_dataset",
    "load_pairs",
    "load_shards",
    "make_pair",
    "merge_shards",
    "save_pairs",
    "shard_path",
    "to_hf_dataset",
]
