"""Algorithm 1: BBOX-ADAPTER online adaptation.

For each iteration ``t``:

1. sample ``M`` candidates ``{y_hat_i,m}`` from the adapted inference
   ``p_{theta_t}`` (Eq. 4),
2. update the positive samples ``y_{i+}^{(t)}`` (Eq. 5) and the negative
   samples ``y_{i-}^{(t)}`` (Eq. 6) using ground-truth / human / AI feedback,
3. update the adapter parameters with Eq. (7):
   ``theta_{t+1} = theta_t - eta * grad_theta l(theta_t)``.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from ..data.loaders import QAExample
from ..data.prompts import build_generator_prompt
from .bank import SampleBank, SampleEntry
from .feedback import GroundTruthFeedback, build_feedback


@dataclass
class IterationStats:
    iteration: int
    loss: float = float("nan")
    positive_energy: float = float("nan")
    negative_energy: float = float("nan")
    pairwise_accuracy: float = float("nan")
    bank_positives_per_question: float = 0.0
    bank_negatives_per_question: float = 0.0
    seconds: float = 0.0
    dev_accuracy: Optional[float] = None

    def as_dict(self) -> Dict[str, object]:
        return self.__dict__.copy()


class OnlineAdaptation:
    def __init__(
        self,
        llm,
        adapter,
        dataset: str,
        inference,
        trainer,
        config,
        feedback=None,
        output_dir: Optional[str] = None,
        logger=None,
    ) -> None:
        self.llm = llm
        self.adapter = adapter
        self.dataset = dataset
        self.inference = inference
        self.trainer = trainer
        self.config = config  # OnlineConfig
        self.feedback = feedback or build_feedback(config.positive_source, dataset)
        self.output_dir = output_dir
        self.logger = logger
        self.history: List[IterationStats] = []
        self.bank: Optional[SampleBank] = None

    # ------------------------------------------------------------------ utils
    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger(message)
        else:
            print(message, flush=True)

    def _build_bank(self, examples: Sequence[QAExample]) -> SampleBank:
        bank = SampleBank(
            dataset=self.dataset,
            positive_source=self.config.positive_source,
            outcome_supervision=self.config.outcome_supervision,
        )
        for example in examples:
            bank.add(
                SampleEntry(
                    key=example.key,
                    question=example.question,
                    gold_answer=example.answer,
                    gold_solution=example.solution,
                    choices=example.choices,
                )
            )
        return bank

    def _sample_initial_candidates(self, examples: Sequence[QAExample]) -> Dict[str, List[str]]:
        """Initialization: ``K`` responses of the un-adapted black-box LLM."""

        prompts = [build_generator_prompt(self.dataset, ex.question, ex.choices) for ex in examples]
        k = self.config.init_candidates_per_question
        results = self.llm.generate(
            prompts,
            n=k,
            temperature=1.0,
            top_p=1.0,
            max_new_tokens=self.inference.max_solution_tokens,
        )
        return {
            example.key: list(result.texts)
            for example, result in zip(examples, results)
        }

    # ------------------------------------------------------------------- run
    def run(self, train_examples: Sequence[QAExample],
            dev_examples: Optional[Sequence[QAExample]] = None,
            evaluator=None) -> List[IterationStats]:
        cfg = self.config
        self._log(
            f"[online] initialising positive/negative sets for {len(train_examples)} "
            f"questions with {cfg.init_candidates_per_question} un-adapted samples"
        )
        bank = self._build_bank(train_examples)
        initial = self._sample_initial_candidates(train_examples)
        bank.initialize(initial, selector=self.feedback)
        self.bank = bank
        self._log(f"[online] bank statistics after init: {bank.statistics()}")

        for iteration in range(cfg.num_iterations):
            started = time.time()
            # (1) sample M candidates from the adapted inference p_theta_t.
            for entry in bank:
                candidates = self.inference.sample_candidates(
                    entry.question,
                    entry.choices,
                    num_candidates=cfg.num_candidates_per_question,
                )
                # (2) update positives (Eq. 5) and negatives (Eq. 6).
                bank.update(entry, candidates, selector=self.feedback)

            statistics = bank.statistics()
            self._log(
                f"[online] iteration {iteration}: sampled candidates for "
                f"{statistics['num_questions']} questions "
                f"(+{statistics['positives_per_question']:.2f}/-"
                f"{statistics['negatives_per_question']:.2f} per question)"
            )

            # (3) update the adapter parameters with Eq. (3) and Eq. (7).
            training_examples = bank.to_training_examples()
            log = self.trainer.fit(
                training_examples,
                num_steps=self.trainer.config.num_train_steps,
                output_dir=os.path.join(self.output_dir, f"iteration_{iteration}")
                if self.output_dir else None,
            )
            stats = IterationStats(
                iteration=iteration,
                loss=log.loss[-1] if log.loss else float("nan"),
                positive_energy=log.positive_energy[-1] if log.positive_energy else float("nan"),
                negative_energy=log.negative_energy[-1] if log.negative_energy else float("nan"),
                pairwise_accuracy=(log.pairwise_accuracy[-1] if log.pairwise_accuracy else float("nan")),
                bank_positives_per_question=statistics["positives_per_question"],
                bank_negatives_per_question=statistics["negatives_per_question"],
            )

            if self.output_dir:
                checkpoint = os.path.join(self.output_dir, f"adapter_iter{iteration}")
                self.adapter.save(checkpoint)
                self._write_bank(os.path.join(self.output_dir, f"bank_iter{iteration}.jsonl"))

            if dev_examples and evaluator is not None and cfg.dev_eval_samples > 0:
                subset = list(dev_examples)[: cfg.dev_eval_samples]
                report = evaluator.evaluate(subset)
                stats.dev_accuracy = report.accuracy
                self._log(f"[online] iteration {iteration}: dev accuracy = {report.accuracy:.2f}%")

            stats.seconds = time.time() - started
            self.history.append(stats)
            self._log(f"[online] iteration {iteration} finished in {stats.seconds:.1f}s: {stats.as_dict()}")
            self._save_history()
        return self.history

    # ------------------------------------------------------------------ io
    def _save_history(self) -> None:
        if not self.output_dir:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, "online_history.json"), "w", encoding="utf-8") as handle:
            json.dump([stats.as_dict() for stats in self.history], handle, indent=2)

    def _write_bank(self, path: str) -> None:
        if self.bank is None:
            return
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            for entry in self.bank:
                handle.write(
                    json.dumps(
                        {
                            "key": entry.key,
                            "question": entry.question,
                            "gold_answer": entry.gold_answer,
                            "positive": entry.positive,
                            "extra_positives": entry.extra_positives,
                            "negatives": entry.negatives,
                        }
                    )
                    + "\n"
                )
