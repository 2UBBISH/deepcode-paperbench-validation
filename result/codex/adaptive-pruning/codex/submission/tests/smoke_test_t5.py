"""End-to-end smoke test of the T5 / encoder-decoder path.

Exercises the parts of the code that only appear for encoder-decoder LMs:
the text-to-text GLUE formulation, gated / non-gated FFN handling, the
encoder+decoder hidden-state distillation, the T5 physical pruning (with the
relative-attention-bias head-uniformity constraint) and generation-based
evaluation.

    python tests/smoke_test_t5.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings("ignore")

from datasets import Dataset, DatasetDict  # noqa: E402

from apt.trainer import APTConfig, APTTrainer  # noqa: E402
from baselines.common import measure_inference  # noqa: E402
from data.tasks import build_task  # noqa: E402


def build_dataset():
    texts = ["a great movie", "terrible film", "i loved it", "boring and dull"] * 16
    labels = [1, 0, 1, 0] * 16
    ds = Dataset.from_dict({"sentence": texts, "label": labels})
    split = ds.train_test_split(test_size=0.25, seed=0)
    return DatasetDict({"train": split["train"], "validation": split["test"]})


def build_toy_t5():
    from transformers import T5Config, T5ForConditionalGeneration

    conf = T5Config(
        vocab_size=32128,
        d_model=64,
        d_ff=128,
        d_kv=16,
        num_layers=2,
        num_decoder_layers=2,
        num_heads=4,
        decoder_start_token_id=0,
        pad_token_id=0,
        eos_token_id=1,
    )
    return T5ForConditionalGeneration(conf)


def main() -> int:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    dataset = build_dataset()
    task = build_task("sst2", dataset, tokenizer, text_to_text=True)
    train_features = task.tokenize("train")
    eval_features = task.tokenize("validation")

    with tempfile.TemporaryDirectory() as tmp:
        cfg = APTConfig(
            model_name="toy-t5",
            task="sst2",
            output_dir=tmp,
            device="cpu",
            target_sparsity=0.6,
            epochs=2,
            distill_epochs=1,
            batch_size=8,
            learning_rate=1e-3,
            adjust_interval=3,
            eval_interval=3,
            eval_batch_size=8,
            target_rank=16,
            seed=0,
        )
        t0 = time.time()
        trainer = APTTrainer(build_toy_model(), tokenizer, task, cfg, train_features, eval_features)
        result = trainer.train()
        elapsed = time.time() - t0

    print(f"achieved sparsity : {result['achieved_sparsity']:.3f}")
    print(f"final metrics     : {result['final_metrics']}")
    print(f"parameters        : {result['n_parameters_dense']} -> {result['n_parameters_pruned']}")
    print(f"wall time         : {elapsed:.1f}s")
    assert result["achieved_sparsity"] >= cfg.target_sparsity - 1e-6
    assert result["n_parameters_pruned"] < result["n_parameters_dense"]
    return 0


def build_toy_model():
    return build_toy_t5()


if __name__ == "__main__":
    raise SystemExit(main())
