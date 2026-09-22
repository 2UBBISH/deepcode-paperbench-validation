"""Task registry.

Every task list in this module is copied verbatim from the paper or from
paper/addendum.md.

* UPSTREAM_TASKS   -- the 36 tasks of D_PT (Sec. 4.1 / addendum).
* P3_TEST_TASKS    -- the tasks used as D_R for BART0 (Sec. 4.1 / addendum).
* P3_TEST_ID_TASKS / P3_TEST_OOD_TASKS -- the in-/out-of-domain split used for
  Table 2 (Appendix B, but the experiment itself is in the main text, Sec. 5.1).
* MMLU_SUBJECTS    -- the 57 subjects of the MMLU *validation* split that is used
  as D_R for FLAN-T5 (Sec. 4.1 / addendum).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

#: 36 upstream pretraining tasks, i.e. the intersection of the T0-Train tasks
#: (Table 5 of the T0 paper) with the tasks shipped by the BART0/ReCross data dump.
UPSTREAM_TASKS: List[str] = [
    "glue-mrpc",
    "glue-qqp",
    "paws_x-en",
    "kilt_tasks-hotpotqa",
    "wiki_qa",
    "adversarial_qa-dbert",
    "adversarial_qa-dbidaf",
    "adversarial_qa-droberta",
    "duorc-SelfRC",
    "duorc-ParaphraseRC",
    "ropes",
    "quoref",
    "cos_e-v1.11",
    "cosmos_qa",
    "dream",
    "qasc",
    "quail",
    "quartz",
    "sciq",
    "social_i_qa",
    "wiki_hop-original",
    "wiqa",
    "amazon_polarity",
    "app_reviews",
    "imdb",
    "rotten_tomatoes",
    "yelp_review_full",
    "common_gen",
    "wiki_bio",
    "cnn_dailymail-3.0.0",
    "gigaword",
    "multi_news",
    "samsum",
    "xsum",
    "ag_news",
    "dbpedia_14",
]

#: D_R for the BART0 experiments: test split of P3 (Sec. 4.1 / addendum).
P3_TEST_TASKS: List[str] = [
    "super_glue-wsc.fixed",
    "winogrande-winogrande_xl",
    "super_glue-cb",
    "super_glue-rte",
    "anli",
    "super_glue-copa",
    "hellaswag",
    "super_glue-wic",
]

#: Table 2 / Appendix B split of P3-Test.
P3_TEST_ID_TASKS: List[str] = [
    "super_glue-cb",
    "super_glue-rte",
    "super_glue-wsc.fixed",
    "super_glue-copa",
    "super_glue-wic",
]
P3_TEST_OOD_TASKS: List[str] = [
    "storycloze",
    "hellaswag",
    "anli",
    "winogrande-winogrande_xl",
]

REFINEMENT_TASKS_BY_MODEL: Dict[str, str] = {
    # "For BART0, we use tasks from the test split of the P3 dataset.
    #  For FLAN-T5, we use MMLU, since the P3 dataset (including the test split)
    #  is involved in pretraining the model." (Sec. 4.1)
    "bart0_large": "p3_test",
    "flan_t5_large": "mmlu",
    "flan_t5_3b": "mmlu",
}

#: The 57 subjects of the original MMLU *validation* release
#: (https://people.eecs.berkeley.edu/~hendrycks/data.tar), used as D_R for FLAN-T5.
MMLU_SUBJECTS: List[str] = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]


# --------------------------------------------------------------------------------------
# Mapping of T0 task names to the template configs of bigscience/P3
# --------------------------------------------------------------------------------------
# T0 names use <dataset>-<template>; the P3 hub dataset names every template
# <dataset>_<template id>.  The mapping is therefore a prefix match on
# <dataset>_<template> after replacing "-" with "_", with a few exceptions.
_P3_PREFIX_OVERRIDES: Dict[str, List[str]] = {
    # Taken from the ReCross/BART0 data dump; the released bigscience/P3 snapshot
    # only ships the paws_labeled_final_* templates.
    "paws_x-en": ["paws_x_en_"],
    # T0's storycloze comes from the T0 data dump; not present in bigscience/P3.
    "storycloze": ["storycloze_"],
    # super_glue-wsc.fixed -> super_glue_wsc.fixed_* (note the literal dot).
    "super_glue-wsc.fixed": ["super_glue_wsc.fixed_"],
}


def p3_config_prefixes(task: str) -> List[str]:
    """Return the P3 template config name prefixes that belong to task."""
    if task in _P3_PREFIX_OVERRIDES:
        return list(_P3_PREFIX_OVERRIDES[task])
    if "-" not in task:
        return [task + "_"]
    dataset, template = task.split("-", 1)
    return [f"{dataset}_{template}_"]


def find_p3_configs(task: str, available_configs: List[str]) -> List[str]:
    """Return all template configs of bigscience/P3 that belong to task.

    The addendum clarifies that *all* variants of a task are used (e.g. every
    anli_* template).
    """
    prefixes: Tuple[str, ...] = tuple(p3_config_prefixes(task))
    return sorted(c for c in available_configs if c.startswith(prefixes))
