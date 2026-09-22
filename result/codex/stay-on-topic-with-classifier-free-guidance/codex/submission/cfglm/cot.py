"""Chain-of-Thought prompting with CFG (Section 3.2).

The paper evaluates GSM8K (Cobbe et al., 2021) and AQuA (Ling et al., 2017)
with WizardLM-30B and Guanaco-65B, following the few-shot Chain-of-Thought
prompts of Wang et al. (2023), and reports two quantities as a function of
the guidance strength:

* **accuracy** -- whether the final answer is correct, and
* **% valid** -- whether the reasoning chain terminates in a parsable answer
  of the form ``The answer is ...``.

The paper's Figure 2 shows that for small ``gamma`` CFG *increases* the share
of valid, parsable chains, but that for ``gamma > 1.5`` the reasoning chains
degrade and accuracy drops.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch


GSM8K_8SHOT = """Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: We start with 15 trees. Later we have 21 trees. The difference must be the number of trees they planted. So, they must have planted 21 - 15 = 6 trees. The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are 3 cars in the parking lot already. 2 more arrive. Now there are 3 + 2 = 5 cars. The answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Leah had 32 chocolates and Leah's sister had 42. That means there were originally 32 + 42 = 74 chocolates. 35 have been eaten. So in total they still have 74 - 35 = 39 chocolates. The answer is 39.

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason had 20 lollipops. Since he only has 12 now, he must have given the rest to Denny. The number of lollipops he has given to Denny must have been 20 - 12 = 8 lollipops. The answer is 8.

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: He has 5 toys. He got 2 from mom, so after that he has 5 + 2 = 7 toys. Then he got 2 more from dad, so in total he has 7 + 2 = 9 toys. The answer is 9.

Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: There are 4 days from monday to thursday. 5 computers were added each day. That means 4 * 5 = 20 computers were added in total. We had 9 computers in the beginning, so now there are 9 + 20 = 29 computers. The answer is 29.

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: Michael initially had 58 balls. He lost 23 on Tuesday, so after that he has 58 - 23 = 35 balls. On Wednesday he lost 2 more so now he has 35 - 2 = 33 balls. The answer is 33.

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: She bought 5 bagels for $3 each. This means she spent 5 * $3 = $15. She originally had $23. Now she has $23 - $15 = $8. The answer is $8.

Q: {question}
A:"""


AQUA_4SHOT = """Q: John found that the average of 15 numbers is 40. If 10 is added to each number then the mean of the numbers is?
Answer Choices: (a) 50 (b) 45 (c) 65 (d) 78 (e) 64
A: If 10 is added to each number, then the mean of the numbers also increases by 10. So the new mean would be 50. The answer is (a).

Q: If a / b = 3/4 and 8a + 5b = 22,then find the value of a.
Answer Choices: (a) 1/2 (b) 3/2 (c) 5/2 (d) 4/2 (e) 7/2
A: If a / b = 3/4, then substituting the value of a = 3/2 and b = 4/2 we get 8*(3/2) + 5*(4/2) = 8*1.5 + 5*2 = 12 + 10 = 22. So a = 3/2. The answer is (b).

Q: A person is traveling at 20 km/hr and reached his destiny in 2.5 hr then find the distance?
Answer Choices: (a) 53 km (b) 55 km (c) 52 km (d) 60 km (e) 50 km
A: The distance that the person traveled would have been 20 km/hr * 2.5 hrs = 50 km. The answer is (e).

Q: How many keystrokes are needed to type the numbers from 1 to 500?
Answer Choices: (a) 1156 (b) 1392 (c) 1480 (d) 1562 (e) 1788
A: There are 9 one-digit numbers from 1 to 9. There are 90 two-digit numbers from 10 to 99. There are 401 three-digit numbers from 100 to 500. 9 + 90(2) + 401(3) = 9 + 180 + 1203 = 1392. The answer is (b).

Q: {question}
Answer Choices: {choices}
A:"""


ANSWER_RE = re.compile(r"the answer is\s*(?:\(?([a-e])\)?|(-?[\d,\.]+))", re.IGNORECASE)
NUMBER_RE = re.compile(r"-?[\d,]*\.?\d+")


def build_gsm8k_prompt(question: str) -> str:
    return GSM8K_8SHOT.format(question=question.strip())


def build_aqua_prompt(question: str, options: Sequence[str]) -> str:
    labels = "abcdefghij"[: len(options)]
    choices = " ".join(f"({label}) {text}" for label, text in zip(labels, options))
    return AQUA_4SHOT.format(question=question.strip(), choices=choices)


def parse_answer(text: str) -> Optional[str]:
    """Extract the answer from a chain-of-thought completion.

    Returns a lowercased choice letter for AQuA-style answers or a
    comma-stripped number for GSM8K-style answers.  ``None`` means the chain
    did not terminate in a parsable answer and therefore counts as *invalid*.
    """
    matches = list(ANSWER_RE.finditer(text))
    if not matches:
        return None
    letter, number = matches[-1].groups()
    if letter:
        return letter.lower()
    if number:
        return number.replace(",", "").rstrip(".")
    return None


def is_valid(text: str) -> bool:
    """Whether the completion ends with a parsable ``The answer is ...``."""
    return parse_answer(text) is not None


def _to_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def answers_match(prediction: Optional[str], gold: str, dataset: str) -> bool:
    """Compare a parsed prediction against the gold answer."""
    if prediction is None:
        return False
    if dataset == "aqua":
        return prediction.strip().lower() == gold.strip().lower()
    pred_value = _to_float(prediction)
    gold_match = NUMBER_RE.search(gold.replace(",", ""))
    gold_value = _to_float(gold_match.group(0)) if gold_match else None
    if pred_value is None or gold_value is None:
        return prediction.strip() == gold.strip()
    return abs(pred_value - gold_value) < 1e-4


@dataclass
class CoTResult:
    n: int
    accuracy: float
    valid_fraction: float
    n_valid: int
    n_correct: int


@torch.no_grad()
def evaluate_cot(
    model,
    tokenizer,
    dataset: str,
    gamma: float,
    limit: Optional[int] = None,
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 0.0,
    uncond_prefix_tokens: int = 1,
    seed: int = 0,
) -> CoTResult:
    """Run GSM8K or AQuA with CFG and report accuracy / % valid.

    ``dataset`` is ``"gsm8k"`` or ``"aqua"``.
    """
    from .generation import cfg_generate

    questions, golds = load_cot_examples(dataset, limit=limit)
    n_valid = n_correct = 0
    for i, (prompt, gold) in enumerate(zip(questions, golds)):
        outputs = cfg_generate(
            model,
            tokenizer,
            prompt,
            gamma=gamma,
            uncond_prefix_tokens=uncond_prefix_tokens,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=max(temperature, 1e-5),
            stop_sequences=("\n\nQ:",),
            seed=seed + i,
        )
        text = outputs[0]
        n_valid += int(is_valid(text))
        n_correct += int(answers_match(parse_answer(text), gold, dataset))
    n = len(questions)
    return CoTResult(
        n=n,
        accuracy=n_correct / n if n else float("nan"),
        valid_fraction=n_valid / n if n else float("nan"),
        n_valid=n_valid,
        n_correct=n_correct,
    )


def load_cot_examples(dataset: str, limit: Optional[int] = None) -> Tuple[List[str], List[str]]:
    """Load GSM8K / AQuA and build their few-shot CoT prompts."""
    from datasets import load_dataset

    prompts: List[str] = []
    golds: List[str] = []
    if dataset == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        for row in ds:
            answer = row["answer"].split("####")[-1].strip()
            prompts.append(build_gsm8k_prompt(row["question"]))
            golds.append(answer)
    elif dataset == "aqua":
        ds = load_dataset("nguyen-brat/aqua", split="test")
        for row in ds:
            options = row.get("options") or [
                row.get(f"option_{l}") for l in "abcdefghij" if row.get(f"option_{l}")
            ]
            prompts.append(build_aqua_prompt(row["question"], options))
            golds.append(str(row["correct"]).strip().lower())
    else:
        raise ValueError("dataset must be 'gsm8k' or 'aqua'")
    if limit is not None:
        prompts, golds = prompts[:limit], golds[:limit]
    return prompts, golds


@torch.no_grad()
def evaluate_cot_self_consistency(
    model,
    tokenizer,
    dataset: str,
    gamma: float,
    n_paths: int = 5,
    limit: Optional[int] = None,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    uncond_prefix_tokens: int = 1,
    seed: int = 0,
) -> CoTResult:
    """CoT + self-consistency + CFG (contribution 3 of the paper).

    Samples ``n_paths`` reasoning chains per question at ``temperature``,
    takes the majority answer among the *valid* chains, and reports accuracy
    together with the fraction of questions that produced at least one valid
    chain.  The paper reports that CFG stacks with CoT and self-consistency
    and yields further improvements on difficult tasks.
    """
    from collections import Counter

    from .generation import cfg_generate

    prompts, golds = load_cot_examples(dataset, limit=limit)
    n_valid_questions = n_correct = 0
    for i, (prompt, gold) in enumerate(zip(prompts, golds)):
        outputs = cfg_generate(
            model,
            tokenizer,
            prompt,
            gamma=gamma,
            uncond_prefix_tokens=uncond_prefix_tokens,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            num_return_sequences=n_paths,
            stop_sequences=("\n\nQ:",),
            seed=seed + i,
        )
        parsed = [parse_answer(text) for text in outputs]
        valid = [answer for answer in parsed if answer is not None]
        if not valid:
            continue
        n_valid_questions += 1
        majority = Counter(valid).most_common(1)[0][0]
        n_correct += int(answers_match(majority, gold, dataset))
    n = len(prompts)
    return CoTResult(
        n=n,
        accuracy=n_correct / n if n else float("nan"),
        valid_fraction=n_valid_questions / n if n else float("nan"),
        n_valid=n_valid_questions,
        n_correct=n_correct,
    )
