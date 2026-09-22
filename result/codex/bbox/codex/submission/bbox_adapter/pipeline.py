"""Wiring of the components into the experiments of Section 4."""

from __future__ import annotations

import os
from typing import Optional, Sequence

from .adapter.energy import EnergyAdapter
from .adapter.mlm_adapter import MLMAdapter
from .adapter.trainer import AdapterTrainer
from .config import RunConfig, adapter_backbone
from .data.loaders import QAExample, load_train_test
from .data.prompts import TRUTHFULQA_INSTRUCTION
from .eval.evaluator import Evaluator
from .inference.adaptive_inference import AdaptedInference
from .llm import build_llm
from .online.online_adaptation import OnlineAdaptation
from .online.feedback import build_feedback
from .utils.seed import set_seed


def pick_device(device: Optional[str] = None) -> str:
    if device:
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def build_adapter(config: RunConfig, dataset: Optional[str] = None,
                  adapter_size: Optional[str] = None, device: Optional[str] = None):
    """Instantiate ``g_theta`` with the backbone reported in Appendix H.2."""

    dataset = (dataset or config.dataset).lower()
    size = adapter_size or config.adapter.adapter_size
    model_name = config.adapter.model_name
    if config.adapter.model_name in {"auto", "", None}:
        model_name = adapter_backbone(dataset, size)
    device = pick_device(device)
    if config.adapter.loss_type == "mlm":
        return MLMAdapter.from_pretrained(model_name, max_length=config.adapter.max_length,
                                          device=device)
    return EnergyAdapter.from_pretrained(
        model_name,
        max_length=config.adapter.max_length,
        pooling=config.adapter.pooling,
        dropout=config.adapter.dropout,
        device=device,
        freeze_encoder_layers=config.adapter.freeze_encoder_layers,
    )


def build_trainer(adapter, config: RunConfig, device: Optional[str] = None) -> AdapterTrainer:
    return AdapterTrainer(adapter, config=config.adapter, device=device)


def build_inference(llm, adapter, dataset: str, config: RunConfig, mode: Optional[str] = None):
    inference_cfg = config.inference
    return AdaptedInference(
        llm=llm,
        adapter=adapter,
        dataset=dataset,
        beam_size=inference_cfg.beam_size,
        num_samples_per_beam=inference_cfg.num_samples_per_beam,
        max_sentence_steps=inference_cfg.max_sentence_steps,
        max_new_tokens=inference_cfg.max_new_tokens,
        max_solution_tokens=inference_cfg.max_solution_tokens,
        temperature=inference_cfg.temperature,
        top_p=inference_cfg.top_p,
        mode=mode or inference_cfg.mode,
        num_single_step_candidates=inference_cfg.num_single_step_candidates,
        # Appendix J gives the instructions (including the TruthfulQA safe
        # behaviour instruction) as part of the prompt text, so no separate
        # system message is used.
        system_prompt=None,
    )


def build_evaluator(dataset: str, inference, config: RunConfig, judge=None,
                    output_dir: Optional[str] = None, logger=None) -> Evaluator:
    truthfulqa_judge = None
    if dataset == "truthfulqa" and judge is not None:
        from .data.truthfulqa_judge import TruthfulQAJudge

        truthfulqa_judge = TruthfulQAJudge(judge)
    return Evaluator(
        dataset=dataset,
        inference=inference,
        judge=judge,
        output_dir=output_dir,
        logger=logger,
        truthfulqa_judge=truthfulqa_judge,
    )


def build_judge_llm(config: RunConfig, cache_dir: Optional[str] = None):
    """The gpt-4 client used for AI feedback (Appendix G) and for TruthfulQA."""

    from .config import LLMConfig
    from .llm import build_llm as _build

    judge_cfg = LLMConfig(
        name="gpt-4",
        provider=config.llm.provider,
        deployment=os.environ.get("AZURE_OPENAI_GPT4_DEPLOYMENT", "gpt-4"),
    )
    return _build(judge_cfg, cache_dir=cache_dir)


def load_data(config: RunConfig, data_dir: Optional[str] = None,
              max_train: Optional[int] = None, max_test: Optional[int] = None,
              dataset: Optional[str] = None):
    dataset = (dataset or config.dataset).lower()
    return load_train_test(
        dataset,
        data_dir=data_dir,
        seed=config.seed,
        cache_dir=os.path.join(config.output_dir, "hf_cache"),
        max_train=max_train,
        max_test=max_test,
    )


def run_online_adaptation(config: RunConfig, dataset: Optional[str] = None,
                          adapter_size: Optional[str] = None,
                          data_dir: Optional[str] = None,
                          max_train: Optional[int] = None,
                          max_test: Optional[int] = None,
                          device: Optional[str] = None,
                          logger=print):
    """Full pipeline: load data, build the adapter, run Algorithm 1, evaluate."""

    dataset = (dataset or config.dataset).lower()
    set_seed(config.seed)
    os.makedirs(config.output_dir, exist_ok=True)
    config.save(os.path.join(config.output_dir, "run_config.json"))

    train_examples, test_examples = load_data(config, data_dir, max_train, max_test, dataset)
    logger(f"[data] {dataset}: {len(train_examples)} train / {len(test_examples)} test")

    llm = build_llm(config.llm, cache_dir=os.path.join(config.output_dir, "llm_cache"))
    adapter = build_adapter(config, dataset, adapter_size, device=device)
    trainer = build_trainer(adapter, config, device=device)
    inference = build_inference(llm, adapter, dataset, config)

    feedback = None
    judge_llm = build_judge_llm(config, cache_dir=os.path.join(config.output_dir, "llm_cache")) \
        if (dataset == "truthfulqa" or config.online.positive_source in {"ai_feedback", "combined"}) \
        else None
    if config.online.positive_source in {"ai_feedback", "combined"}:
        feedback = build_feedback(config.online.positive_source, dataset, judge_llm)

    online = OnlineAdaptation(
        llm=llm,
        adapter=adapter,
        dataset=dataset,
        inference=inference,
        trainer=trainer,
        config=config.online,
        feedback=feedback,
        output_dir=config.output_dir,
        logger=logger,
    )

    evaluator = build_evaluator(dataset, inference, config, judge=judge_llm,
                                output_dir=os.path.join(config.output_dir, "eval"),
                                logger=logger)
    # The intermediate dev reports of the online loop (Figure 3b) are enabled by
    # ``online.dev_eval_samples``; the final report is computed after the loop.
    history = online.run(
        train_examples,
        dev_examples=test_examples if config.online.dev_eval_samples > 0 else None,
        evaluator=evaluator if config.online.dev_eval_samples > 0 else None,
    )

    report = evaluator.evaluate(test_examples)
    # TruthfulQA is reported with the "True + Info" metric of Table 2.
    headline = report.extra.get("true_info", report.accuracy) if dataset == "truthfulqa" else report.accuracy
    logger(f"[eval] {dataset} accuracy: {headline:.2f}%")
    adapter.save(os.path.join(config.output_dir, "adapter_final"))
    return {
        "history": [stats.as_dict() for stats in history],
        "test_accuracy": headline,
        "test_accuracy_exact_match": report.accuracy,
        "metrics": report.as_dict(),
        "usage": llm.usage.as_dict(),
        "judge_usage": judge_llm.usage.as_dict() if judge_llm is not None else None,
        "training_cost_usd": (
            (llm.usage.cost_or_none(config.llm.name) or 0.0)
            + ((judge_llm.usage.cost_or_none("gpt-4") or 0.0) if judge_llm is not None else 0.0)
        ),
    }
