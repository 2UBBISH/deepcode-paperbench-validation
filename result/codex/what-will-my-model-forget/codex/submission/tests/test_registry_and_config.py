from wwmf.config import (
    EXAMPLES_PER_UPSTREAM_TASK,
    LORA_CONFIG,
    REPLAY_SCHEDULE,
    STEPS_SINGLE_ERROR,
    ExperimentConfig,
)
from wwmf.data.registry import (
    MMLU_SUBJECTS,
    P3_TEST_ID_TASKS,
    P3_TEST_OOD_TASKS,
    P3_TEST_TASKS,
    UPSTREAM_TASKS,
    find_p3_configs,
    p3_config_prefixes,
)


def test_upstream_task_list_matches_the_paper():
    assert len(UPSTREAM_TASKS) == 36
    assert len(set(UPSTREAM_TASKS)) == 36
    for task in ["glue-mrpc", "cnn_dailymail-3.0.0", "wiki_hop-original", "ag_news", "dbpedia_14"]:
        assert task in UPSTREAM_TASKS


def test_refinement_task_lists():
    assert len(P3_TEST_TASKS) == 8
    assert set(P3_TEST_ID_TASKS).isdisjoint(P3_TEST_OOD_TASKS)
    # Appendix B additionally lists `storycloze` in the out-of-domain split,
    # which is not among the 8 tasks used to collect D_R (addendum).
    assert set(P3_TEST_ID_TASKS) | (set(P3_TEST_OOD_TASKS) - {"storycloze"}) == set(P3_TEST_TASKS)
    assert len(MMLU_SUBJECTS) == 57
    assert "public_relations" in MMLU_SUBJECTS  # the subject of Figure 1


def test_p3_config_mapping():
    available = [
        "glue_mrpc_equivalent",
        "glue_mrpc_paraphrase",
        "glue_qqp_duplicate",
        "super_glue_wsc.fixed_GPT_3_Style",
        "super_glue_cb_MNLI_crowdsource",
        "anli_GPT_3_style_r1",
        "anli_GPT_3_style_r2",
    ]
    assert find_p3_configs("glue-mrpc", available) == ["glue_mrpc_equivalent", "glue_mrpc_paraphrase"]
    assert find_p3_configs("glue-qqp", available) == ["glue_qqp_duplicate"]
    # all variants of a task are used (addendum)
    assert len(find_p3_configs("anli", available)) == 2
    assert p3_config_prefixes("super_glue-wsc.fixed") == ["super_glue_wsc.fixed_"]
    assert p3_config_prefixes("cos_e-v1.11") == ["cos_e_v1.11_"]


def test_hyperparameters_from_section_4_1():
    assert STEPS_SINGLE_ERROR == {"head": 100, "lora": 30, "full_ft": 30}
    assert LORA_CONFIG["r"] == 16 and LORA_CONFIG["lora_alpha"] == 32
    assert LORA_CONFIG["lora_dropout"] == 0.1 and LORA_CONFIG["target_modules"] == ["q", "v"]
    assert REPLAY_SCHEDULE["flan_t5_3b"] == {"batch_size": 4, "every_n_steps": 5}
    assert REPLAY_SCHEDULE["bart0_large"] == {"batch_size": 8, "every_n_steps": 10}
    assert EXAMPLES_PER_UPSTREAM_TASK == 100

    cfg = ExperimentConfig(model="bart0_large", tuning_mode="full_ft")
    assert cfg.lr_single() == 1e-5
    assert cfg.lr_sequential() == 1e-6
    assert cfg.steps_single() == 30
    cfg = ExperimentConfig(model="flan_t5_large", tuning_mode="lora")
    assert cfg.lr_single() == 1e-4
    assert cfg.lr_sequential() == 1e-5
