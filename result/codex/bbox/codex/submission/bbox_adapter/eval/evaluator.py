"""Evaluation harness shared by every experiment."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from ..data.answer_extraction import extract_answer
from ..data.loaders import QAExample
from ..data.metrics import EvalReport, correctness
from ..data.truthfulqa_judge import TruthfulQAJudge


@dataclass
class Prediction:
    key: str
    question: str
    gold: str
    text: str
    prediction: Optional[str]
    correct: bool
    meta: Dict[str, object] = field(default_factory=dict)


class Evaluator:
    """Runs a model over a split and computes the metrics of Section 4.2."""

    def __init__(
        self,
        dataset: str,
        inference=None,
        judge=None,
        output_dir: Optional[str] = None,
        logger=None,
        truthfulqa_judge: Optional[TruthfulQAJudge] = None,
    ) -> None:
        self.dataset = dataset
        self.inference = inference
        self.judge = judge
        self.output_dir = output_dir
        self.logger = logger
        self.truthfulqa_judge = truthfulqa_judge

    def _log(self, message: str) -> None:
        if self.logger:
            self.logger(message)
        else:
            print(message, flush=True)

    # ------------------------------------------------------------------ core
    def predict(self, examples: Sequence[QAExample], limit: Optional[int] = None) -> List[Prediction]:
        if self.inference is None:
            raise ValueError("Evaluator.predict requires an inference engine")
        subset = list(examples)[:limit] if limit else list(examples)
        predictions: List[Prediction] = []
        started = time.time()
        for index, example in enumerate(subset):
            text = self.inference.answer(example.question, example.choices)
            predicted = extract_answer(self.dataset, text, example.num_choices)
            predictions.append(
                Prediction(
                    key=example.key,
                    question=example.question,
                    gold=example.answer,
                    text=text,
                    prediction=predicted,
                    correct=correctness(self.dataset, text, example.answer, example.num_choices),
                )
            )
            if (index + 1) % 25 == 0:
                self._log(f"[eval] {index + 1}/{len(subset)} examples ({time.time() - started:.0f}s)")
        return predictions

    def evaluate(self, examples: Sequence[QAExample], limit: Optional[int] = None) -> EvalReport:
        predictions = self.predict(examples, limit)
        report = EvalReport(num_examples=len(predictions))
        report.predictions = [p.text for p in predictions]
        report.gold = [p.gold for p in predictions]
        report.per_example = [p.correct for p in predictions]
        report.num_correct = sum(report.per_example)

        if self.dataset == "truthfulqa":
            judge = self.truthfulqa_judge
            if judge is None and self.judge is not None:
                judge = TruthfulQAJudge(self.judge)
            if judge is not None:
                judgements = judge.judge_batch(
                    [p.question for p in predictions], [p.text for p in predictions]
                )
                report.extra["true_info"] = TruthfulQAJudge.true_info_rate(judgements)
                report.extra["informative"] = 100.0 * sum(
                    1 for j in judgements if j.get("informative")
                ) / max(1, len(judgements))
                report.extra["truthful"] = 100.0 * sum(
                    1 for j in judgements if j.get("truthful")
                ) / max(1, len(judgements))

        if self.output_dir:
            self.save_predictions(predictions)
        return report

    # -------------------------------------------------------------------- io
    def save_predictions(self, predictions: Sequence[Prediction]) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, "predictions.jsonl")
        with open(path, "w", encoding="utf-8") as handle:
            for prediction in predictions:
                handle.write(json.dumps(prediction.__dict__) + "\n")


def evaluate_texts(dataset: str, predictions: Sequence[str], gold: Sequence[str],
                   num_choices: Optional[Sequence[int]] = None) -> EvalReport:
    """Evaluate pre-computed generations (used by the baseline scripts)."""

    report = EvalReport(num_examples=len(predictions))
    for index, (prediction, answer) in enumerate(zip(predictions, gold)):
        choices = num_choices[index] if num_choices is not None else None
        correct = correctness(dataset, prediction, answer, choices)
        report.predictions.append(prediction)
        report.gold.append(answer)
        report.per_example.append(correct)
        report.num_correct += int(correct)
    return report
