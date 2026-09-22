"""Reproduction of "A Mechanistic Understanding of Alignment Algorithms:
A Case Study on DPO and Toxicity" (Lee et al., ICML 2024).

The package is organised along the sections of the paper:

* :mod:`dpo_toxic.probe`          -- Section 3.1, the toxicity probe ``W_toxic``
* :mod:`dpo_toxic.toxic_vectors`  -- Section 3.1, toxic MLP value/key vectors + SVD
* :mod:`dpo_toxic.vocab_projection` -- Section 3.2, projecting vectors onto the vocabulary
* :mod:`dpo_toxic.interventions`  -- Section 3.3, subtracting toxic vectors during a forward pass
* :mod:`dpo_toxic.pplm`           -- Section 4.2, PPLM toxic generation
* :mod:`dpo_toxic.pairs`          -- Section 4.2, building the 24,576 preference pairs
* :mod:`dpo_toxic.dpo`            -- Section 4.1, the DPO objective + trainer
* :mod:`dpo_toxic.analysis`       -- Section 5/6, post-DPO analyses and un-alignment
* :mod:`dpo_toxic.evaluation`     -- toxicity / perplexity / F1 metrics
"""

__version__ = "1.0.0"

MODEL_NAME = "gpt2-medium"
REPO_ROOT_ARTIFACTS = "artifacts"
