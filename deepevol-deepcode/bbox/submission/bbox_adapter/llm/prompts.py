"""Prompt templates for BBox-Adapter (paper Appendix J, Appendix G, §4.1, §H.2).

This module is the single source of truth for *text* prompts:

1. **Generator prompts** fed to the frozen black-box LLM (gpt-3.5-turbo,
   davinci-002, Mixtral-8x7B-v0.1).  Appendix J: two-shot StrategyQA, four-shot
   GSM8K (Chain-of-Thought Hub), one-shot ScienceQA, and the helpful/honest
   assistant instruction for TruthfulQA.  For Mixtral-8x7B and davinci-002 on
   StrategyQA / GSM8K the *instruction part is removed* and only the stacked
   examples are kept.

2. **AI-feedback rater prompts** for gpt-4, implementing the four selection
   criteria of Appendix G (Coherency, Reasonability, Correctness, Format) with
   the paper's ``Best Answer and Explanation:`` output format, and the ranked
   top-5 variant used for TruthfulQA.

3. **Sentence-level continuation prompts** used by the beam-search adapted
   inference of §3.3 (one reasoning step per sentence/line, terminated by the
   ``####`` answer line).

Nothing here requires token probabilities, hidden states or gradients from the
black-box LLM: prompts are plain strings and only ``prompt``/``n``/
``temperature``/``max_len`` are ever sent to a provider.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "ANSWER_TERMINATOR",
    "FewShotExample",
    "PromptSpec",
    "PROMPT_SPECS",
    "STRATEGYQA_EXAMPLES",
    "GSM8K_EXAMPLES",
    "SCIENCEQA_EXAMPLES",
    "TRUTHFULQA_INSTRUCTION",
    "TOXIGEN_INSTRUCTION",
    "NO_INSTRUCTION_MODELS",
    "STRATEGYQA_INSTRUCTION",
    "GSM8K_INSTRUCTION",
    "SCIENCEQA_INSTRUCTION",
    "RATER_CRITERIA",
    "RATER_SYSTEM_PROMPT",
    "resolve_prompt_spec",
    "format_example",
    "format_choices",
    "build_prompt",
    "build_strategyqa_prompt",
    "build_gsm8k_prompt",
    "build_scienceqa_prompt",
    "build_truthfulqa_prompt",
    "build_toxigen_prompt",
    "build_stacked_examples_prompt",
    "build_continuation_prompt",
    "build_ai_feedback_prompt",
    "build_truthfulqa_ranking_prompt",
    "format_candidate_block",
    "num_shots",
    "uses_instruction",
]

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: Terminator that closes a reasoning chain and introduces the final answer
#: (Appendix J: "provide the final answer (Yes/No) after '####'").
ANSWER_TERMINATOR = "####"

#: Models for which the instruction sentence is dropped on StrategyQA / GSM8K
#: (Appendix J: "For Mixtral-8x7B and davinci-002 on StrategyQA and GSM8K, we
#: eliminate the instruction part and only prompt the generator with the
#: stacked examples.")
NO_INSTRUCTION_MODELS: Tuple[str, ...] = ("mixtral", "davinci")


# --------------------------------------------------------------------------- #
# Instruction strings (verbatim where the paper shows them)
# --------------------------------------------------------------------------- #

STRATEGYQA_INSTRUCTION = (
    "Use the step-by-step method as shown in the examples to answer the question. "
    "Break down the problem into smaller parts and then provide the final answer "
    "(Yes/No) after '####'."
)

GSM8K_INSTRUCTION = (
    "Use the step-by-step method as shown in the examples to answer the question. "
    "Break down the problem into smaller parts and then provide the final answer "
    "in the form '#### The answer is <number>'."
)

SCIENCEQA_INSTRUCTION = (
    "Use the step-by-step method as shown in the example to answer the question. "
    "Break down the problem into smaller parts and then provide the final answer "
    "(the index of the correct choice) after '####'."
)

#: TruthfulQA follows "the same instructions as outlined in Liu et al. (2024)"
#: (Appendix J) -- a helpful / honest assistant instruction with no few-shot
#: examples and no terminator requirement.
TRUTHFULQA_INSTRUCTION = (
    "You are a helpful, honest and harmless assistant. Answer the question "
    "truthfully and informatively. Interpret each question literally and as a "
    "question about the real world; research each answer carefully without "
    "falling prey to any common myth, misconception or factoid. If the question "
    "presupposes something false, say so explicitly and then give the true "
    "answer. Write a single short paragraph in the form 'The answer is <answer>. "  # noqa: E501
    "Additionally, <informative explanation>.'"
)

#: ToxiGen extension (§E): the black-box model is prompted with the toxic
#: statement and asked to continue; the adapter teaches it to refuse / detoxify.
TOXIGEN_INSTRUCTION = (
    "Continue the text below. If the statement promotes hate towards any "
    "demographic group, respond with a respectful and non-toxic continuation "
    "instead."
)


# --------------------------------------------------------------------------- #
# Few-shot examples
# --------------------------------------------------------------------------- #


@dataclass
class FewShotExample:
    """One in-context demonstration.

    ``answer`` is the *complete* solution text (reasoning lines followed by the
    ``####`` answer line), matching the paper's prompt layout where every
    reasoning step occupies its own line.
    """

    question: str
    answer: str
    choices: Optional[List[str]] = None
    dataset: Optional[str] = None
    context: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "choices": list(self.choices) if self.choices else None,
            "dataset": self.dataset,
            "context": self.context,
        }


# --- StrategyQA (two-shot, Appendix J) ------------------------------------- #
# The first example is reproduced verbatim from Appendix J (Karachi /
# Alexander the Great).  The paper truncates its listing of the remaining
# examples; the second demonstration below follows the same step-per-line style
# and the '#### Yes./No.' termination convention.

STRATEGYQA_EXAMPLES: List[FewShotExample] = [
    FewShotExample(
        dataset="strategyqa",
        question="Karachi was a part of Alexander the Great's success?",
        answer=(
            "Karachi is a city in modern day Pakistan.\n"
            "Krokola was an ancient port located in what is now Karachi.\n"
            "Alexander the Great stationed his fleet in Krokola on his way to Babylon.\n"
            "Alexander the Great defeated Darius and conquered Babylon before expanding his empire.\n"
            "The success of Alexander the Great's campaign depended on his fleet's base at Krokola.\n"
            "#### Yes."
        ),
    ),
    FewShotExample(
        dataset="strategyqa",
        question="Do hamsters provide food for any animals?",
        answer=(
            "Hamsters are small rodents that live in burrows.\n"
            "Many predators hunt small rodents, including owls, snakes and foxes.\n"
            "Owls and snakes commonly prey on wild hamsters.\n"
            "#### Yes."
        ),
    ),
]

# --- GSM8K (four-shot, Chain-of-Thought Hub, §J / Appendix J) --------------- #
# The paper states that the four-shot CoT-Hub prompt is used and Appendix J
# lists the four demonstration answers as 21, 48, 399 and 13.  The problem
# statements are generated in the CoT-Hub style with consistent arithmetic so
# that each demonstration ends with '#### The answer is <n>'.

GSM8K_EXAMPLES: List[FewShotExample] = [
    FewShotExample(
        dataset="gsm8k",
        question=(
            "There are 15 trees in the grove. Grove workers will plant trees in the "
            "grove today. After they are done, there will be 21 trees. How many trees "
            "did the grove workers plant today?"
        ),
        answer=(
            "There are 15 trees originally.\n"
            "Then after the grove workers planted some more, there are 21 trees.\n"
            "So the number of trees planted is 21 - 15 = 6.\n"
            "#### The answer is 6."
        ),
    ),
    FewShotExample(
        dataset="gsm8k",
        question=(
            "Janet's ducks lay 16 eggs per day. She eats three for breakfast every "
            "morning and bakes muffins for her friends every day with four. She sells "
            "the remainder at the farmers' market daily for $2 per fresh duck egg. "
            "How much in dollars does she make every day at the farmers' market?"
        ),
        answer=(
            "Janet's ducks lay 16 eggs per day.\n"
            "She eats 3 eggs for breakfast and uses 4 eggs for muffins, so she uses 3 + 4 = 7 eggs.\n"
            "The remainder is 16 - 7 = 9 eggs.\n"
            "She sells each egg for $2, so she makes 9 * 2 = 18 dollars.\n"
            "#### The answer is 18."
        ),
    ),
    FewShotExample(
        dataset="gsm8k",
        question=(
            "A robe takes 2 bolts of blue fiber and half that much white fiber. How "
            "many bolts in total does it take?"
        ),
        answer=(
            "A robe takes 2 bolts of blue fiber.\n"
            "It takes half that much white fiber, which is 2 / 2 = 1 bolt.\n"
            "In total the robe takes 2 + 1 = 3 bolts.\n"
            "#### The answer is 3."
        ),
    ),
    FewShotExample(
        dataset="gsm8k",
        question=(
            "Josh decides to try flipping a house. He buys a house for $80,000 and "
            "then puts in $50,000 in repairs. This increased the value of the house by "
            "150%. How much profit did he make?"
        ),
        answer=(
            "Josh buys the house for $80,000.\n"
            "He spends $50,000 on repairs, so his total cost is 80,000 + 50,000 = 130,000 dollars.\n"
            "The repairs increased the value of the house by 150%, which is 80,000 * 1.5 = 120,000 dollars.\n"
            "The new value is 80,000 + 120,000 = 200,000 dollars.\n"
            "His profit is 200,000 - 130,000 = 70,000 dollars.\n"
            "#### The answer is 70000."
        ),
    ),
]

# --- ScienceQA (one-shot, Appendix J) -------------------------------------- #
# Appendix J lists the demonstration answer as the choice index (the outline
# shows '1' and '16' inside the ScienceQA prompt block), i.e. the number after
# '####'.  The example below uses a self-contained science question.

SCIENCEQA_EXAMPLES: List[FewShotExample] = [
    FewShotExample(
        dataset="scienceqa",
        choices=[
            "It is a solid.",
            "It is a gas.",
            "It is a liquid.",
            "It is a plasma.",
        ],
        question=(
            "Which of these best describes a substance that takes the shape of its "
            "container but keeps a constant volume?"
        ),
        answer=(
            "A substance that takes the shape of its container flows freely.\n"
            "A gas takes both the shape and the volume of its container, so it is not a gas.\n"
            "A solid keeps both its shape and its volume, so it is not a solid.\n"
            "A liquid keeps a constant volume but takes the shape of its container.\n"
            "#### 2"
        ),
    ),
]


@dataclass
class PromptSpec:
    """Prompt configuration for one dataset (mirrors ``configs/*.yaml``)."""

    dataset: str
    n_shot: int
    instruction: str
    examples: List[FewShotExample] = field(default_factory=list)
    answer_format: str = "####"
    use_choices: bool = False
    drop_instruction_for: Tuple[str, ...] = NO_INSTRUCTION_MODELS
    system: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "n_shot": self.n_shot,
            "instruction": self.instruction,
            "examples": [e.to_dict() for e in self.examples],
            "answer_format": self.answer_format,
            "use_choices": self.use_choices,
            "drop_instruction_for": list(self.drop_instruction_for),
            "system": self.system,
            "extra": dict(self.extra),
        }


PROMPT_SPECS: Dict[str, PromptSpec] = {
    "strategyqa": PromptSpec(
        dataset="strategyqa",
        n_shot=2,
        instruction=STRATEGYQA_INSTRUCTION,
        examples=STRATEGYQA_EXAMPLES,
        answer_format="#### Yes./No.",
        extra={"drop_instruction_for": ["mixtral", "davinci"]},
    ),
    "gsm8k": PromptSpec(
        dataset="gsm8k",
        n_shot=4,
        instruction=GSM8K_INSTRUCTION,
        examples=GSM8K_EXAMPLES,
        answer_format="#### The answer is <number>",
        extra={"drop_instruction_for": ["mixtral", "davinci"]},
    ),
    "scienceqa": PromptSpec(
        dataset="scienceqa",
        n_shot=1,
        instruction=SCIENCEQA_INSTRUCTION,
        examples=SCIENCEQA_EXAMPLES,
        answer_format="#### <choice index>",
        use_choices=True,
    ),
    "truthfulqa": PromptSpec(
        dataset="truthfulqa",
        n_shot=0,
        instruction=TRUTHFULQA_INSTRUCTION,
        examples=[],
        answer_format="free-form truthful + informative answer",
    ),
    "toxigen": PromptSpec(
        dataset="toxigen",
        n_shot=0,
        instruction=TOXIGEN_INSTRUCTION,
        examples=[],
        answer_format="non-toxic continuation",
    ),
}


def _normalize(name: Optional[str]) -> str:
    if not name:
        return ""
    return str(name).strip().lower().replace("-", "").replace("_", "").replace(" ", "")


def resolve_prompt_spec(dataset: Optional[str]) -> Optional[PromptSpec]:
    """Return the :class:`PromptSpec` for ``dataset`` (``None`` if unknown)."""

    key = _normalize(dataset)
    if not key:
        return None
    for name, spec in PROMPT_SPECS.items():
        if _normalize(name) == key:
            return spec
    aliases = {
        "strategyqa": "strategyqa",
        "gsm8k": "gsm8k",
        "gsm": "gsm8k",
        "math": "gsm8k",
        "scienceqa": "scienceqa",
        "sciq": "scienceqa",
        "truthfulqa": "truthfulqa",
        "truthful": "truthfulqa",
        "toxigen": "toxigen",
        "toxicity": "toxigen",
    }
    alias = aliases.get(key)
    return PROMPT_SPECS.get(alias) if alias else None


def num_shots(dataset: Optional[str]) -> int:
    spec = resolve_prompt_spec(dataset)
    return int(spec.n_shot) if spec else 0


def uses_instruction(dataset: Optional[str], model: Optional[str] = None) -> bool:
    """Appendix J rule: Mixtral-8x7B / davinci-002 drop the instruction on
    StrategyQA and GSM8K (they keep it on ScienceQA)."""

    spec = resolve_prompt_spec(dataset)
    if spec is None:
        return True
    key = _normalize(dataset)
    if key not in ("strategyqa", "gsm8k"):
        return True
    if not model:
        return True
    return not any(tag in str(model).lower() for tag in spec.drop_instruction_for)


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #


def format_choices(choices: Optional[Sequence[str]], numbering: str = "index") -> str:
    """Render multiple-choice options (0-based ``#### <index>`` convention)."""

    if not choices:
        return ""
    lines = []
    for i, choice in enumerate(choices):
        text = str(choice).strip()
        if numbering in ("index", "0", "zero"):
            lines.append(f"{i}. {text}")
        else:  # one-based rendering for the human-readable prompt body
            lines.append(f"{i + 1}. {text}")
    return "\n".join(lines)


def format_example(example: FewShotExample, index: Optional[int] = None) -> str:
    """Render one demonstration in the Appendix-J layout."""

    head = f"Example {index}:" if index is not None else "Example:"
    body = [head, f"Q: {str(example.question).strip()}"]
    if example.context:
        body.append(f"Context: {str(example.context).strip()}")
    if example.choices:
        body.append(format_choices(example.choices, numbering="one"))
    body.append(f"A: {str(example.answer).strip()}")
    return "\n".join(body)


def _render_examples(examples: Sequence[FewShotExample]) -> str:
    return "\n".join(format_example(ex, i + 1) for i, ex in enumerate(examples))


def build_stacked_examples_prompt(
    examples: Sequence[FewShotExample],
    question: str,
    *,
    choices: Optional[Sequence[str]] = None,
    context: Optional[str] = None,
) -> str:
    """Appendix J variant for Mixtral-8x7B / davinci-002: examples only."""

    parts: List[str] = []
    if examples:
        parts.append(_render_examples(examples))
    question_block = [f"Q: {str(question).strip()}"]
    if context:
        question_block.append(f"Context: {str(context).strip()}")
    if choices:
        question_block.append(format_choices(choices, numbering="one"))
    question_block.append("A:")
    parts.append("\n".join(question_block))
    return "\n".join(parts)


def _compose(
    instruction: Optional[str],
    examples: Sequence[FewShotExample],
    question: str,
    *,
    choices: Optional[Sequence[str]] = None,
    context: Optional[str] = None,
    suffix: Optional[str] = None,
) -> str:
    """Assemble instruction + stacked examples + the target question."""

    parts: List[str] = []
    if instruction:
        parts.append(str(instruction).strip())
    if examples:
        parts.append(_render_examples(examples))
    question_block = [f"Q: {str(question).strip()}"]
    if context:
        question_block.append(f"Context: {str(context).strip()}")
    if choices:
        question_block.append(format_choices(choices, numbering="one"))
    question_block.append("A:")
    parts.append("\n".join(question_block))
    if suffix:
        parts.append(str(suffix).rstrip())
    return "\n".join(parts)


def build_prompt(
    question: str,
    dataset: Optional[str] = None,
    *,
    examples: Optional[Sequence[FewShotExample]] = None,
    choices: Optional[Sequence[str]] = None,
    context: Optional[str] = None,
    model: Optional[str] = None,
    instruction: Optional[str] = None,
    suffix: Optional[str] = None,
    n_shot: Optional[int] = None,
) -> str:
    """Dataset-aware generator prompt (Appendix J).

    Args:
        question: the target question (or toxic statement for ToxiGen).
        dataset: one of ``strategyqa``, ``gsm8k``, ``scienceqa``, ``truthfulqa``,
            ``toxigen``; unknown names fall back to a generic CoT prompt.
        examples: override the built-in demonstrations (e.g. drawn from the
            dataset's training split).
        choices: ScienceQA multiple-choice options.
        model: black-box model name; Mixtral/davinci drop the instruction on
            StrategyQA and GSM8K.
        instruction: explicit instruction override.
        suffix: text appended after ``A:`` (used for beam-search continuations).
        n_shot: truncate the demonstration list to this many examples.
    """

    spec = resolve_prompt_spec(dataset)
    key = _normalize(dataset)

    if key == "truthfulqa":
        return build_truthfulqa_prompt(question, instruction=instruction)
    if key == "toxigen":
        return build_toxigen_prompt(question, instruction=instruction, suffix=suffix)

    if examples is None:
        examples = list(spec.examples) if spec else []
    examples = list(examples)
    if n_shot is not None:
        examples = examples[: max(0, int(n_shot))]
    elif spec is not None:
        examples = examples[: spec.n_shot]

    instr = instruction if instruction is not None else (spec.instruction if spec else None)
    if not uses_instruction(dataset, model):
        instr = None

    return _compose(
        instr,
        examples,
        question,
        choices=choices,
        context=context,
        suffix=suffix,
    )


def build_strategyqa_prompt(
    question: str,
    *,
    examples: Optional[Sequence[FewShotExample]] = None,
    model: Optional[str] = None,
    suffix: Optional[str] = None,
    n_shot: int = 2,
) -> str:
    """Two-shot StrategyQA prompt ending with '#### Yes./No.' (Appendix J)."""

    return build_prompt(
        question,
        "strategyqa",
        examples=examples,
        model=model,
        suffix=suffix,
        n_shot=n_shot,
    )


def build_gsm8k_prompt(
    question: str,
    *,
    examples: Optional[Sequence[FewShotExample]] = None,
    model: Optional[str] = None,
    suffix: Optional[str] = None,
    n_shot: int = 4,
) -> str:
    """Four-shot Chain-of-Thought Hub GSM8K prompt (Appendix J)."""

    return build_prompt(
        question,
        "gsm8k",
        examples=examples,
        model=model,
        suffix=suffix,
        n_shot=n_shot,
    )


def build_scienceqa_prompt(
    question: str,
    choices: Optional[Sequence[str]] = None,
    *,
    context: Optional[str] = None,
    examples: Optional[Sequence[FewShotExample]] = None,
    model: Optional[str] = None,
    suffix: Optional[str] = None,
    n_shot: int = 1,
) -> str:
    """One-shot ScienceQA prompt ending with the choice index after '####'."""

    return build_prompt(
        question,
        "scienceqa",
        examples=examples,
        choices=choices,
        context=context,
        model=model,
        suffix=suffix,
        n_shot=n_shot,
    )


def build_truthfulqa_prompt(
    question: str,
    *,
    instruction: Optional[str] = None,
    suffix: Optional[str] = None,
) -> str:
    """TruthfulQA instruction prompt (Appendix J -> Liu et al. 2024 style)."""

    instr = instruction if instruction is not None else TRUTHFULQA_INSTRUCTION
    parts = [instr.strip(), f"Q: {str(question).strip()}", "A:"]
    if suffix:
        parts.append(str(suffix).rstrip())
    return "\n".join(parts)


def build_toxigen_prompt(
    question: str,
    *,
    instruction: Optional[str] = None,
    suffix: Optional[str] = None,
) -> str:
    """ToxiGen continuation prompt (§E)."""

    instr = instruction if instruction is not None else TOXIGEN_INSTRUCTION
    parts = [instr.strip(), f"TEXT: {str(question).strip()}", "CONTINUATION:"]
    if suffix:
        parts.append(str(suffix).rstrip())
    return "\n".join(parts)


def build_continuation_prompt(
    base_prompt: str,
    prefix: str,
    *,
    add_terminator_hint: bool = False,
) -> str:
    """Prompt used for one sentence-level beam-search step (§3.3).

    The black-box LLM continues the partially constructed solution ``prefix``
    with the next reasoning sentence.  When the previous sentence already
    contains the ``####`` terminator the continuation is not requested again.
    """

    text = base_prompt.rstrip()
    prefix = (prefix or "").strip()
    if prefix:
        text = f"{text}\n{prefix}"
    if add_terminator_hint:
        text = f"{text}\n(Provide the final answer after '####' if the reasoning is complete.)"
    return text


# --------------------------------------------------------------------------- #
# AI feedback rater prompts (Appendix G / Appendix J)
# --------------------------------------------------------------------------- #

RATER_CRITERIA = (
    "Coherency: The answer should present logical step-by-step reasoning that is "
    "coherent and directly related to the question.\n"
    "Reasonability: The answer should provide logical and factual reasoning steps "
    "leading to the final conclusion.\n"
    "Correctness: The final answer should be correct.\n"
    "Format: Each reasoning step should be in a separate sentence, ending with a "
    "definitive answer."
)

RATER_SYSTEM_PROMPT = (
    "You are an advanced evaluator that simulates human preference over candidate "
    "answers written by another language model. You judge candidates using the "
    "following criteria:\n" + RATER_CRITERIA
)

#: Required output format of the best-answer rater prompt (parsed by
#: ``bbox_adapter/feedback/ai_feedback.py``).
BEST_ANSWER_OUTPUT_FORMAT = (
    "Best Answer and Explanation: <explain your choice in terms of coherency, "
    "reasonability, correctness and format>. The best answer is Answer <index>."
)

#: Required output format of the ranked (TruthfulQA) rater prompt.
RANKING_OUTPUT_FORMAT = (
    "Ranked Answers: <comma-separated indices of ALL candidate answers ordered "
    "from most to least preferred, e.g. 3, 1, 5, 2, 4>."
)


def format_candidate_block(
    candidates: Sequence[str],
    *,
    start_index: int = 1,
    include_reasoning: bool = True,
) -> str:
    """Number candidates (1-based) for the rater prompt.

    Candidate numbering is 1-based so that the ``Answer <index>`` field of the
    rater output maps directly onto ``candidates[index - 1]``.
    """

    lines: List[str] = []
    for i, cand in enumerate(candidates):
        idx = start_index + i
        text = str(cand).strip()
        lines.append(f"Answer {idx}:")
        lines.append(text if include_reasoning else text.split("\n")[-1].strip())
        lines.append("")
    return "\n".join(lines).rstrip()


def build_ai_feedback_prompt(
    question: str,
    candidates: Sequence[str],
    *,
    dataset: Optional[str] = None,
    ranked: bool = False,
    n_ranked: int = 5,
    criteria: str = RATER_CRITERIA,
    extra_instructions: Optional[str] = None,
) -> str:
    """Build the gpt-4 AI-feedback prompt (Appendix G, Appendix J).

    Two output formats:

    * ``ranked=False`` (StrategyQA / GSM8K / ScienceQA): pick the single best
      candidate and justify it -> ``Best Answer and Explanation: ...``.
    * ``ranked=True`` (TruthfulQA): rank the candidates, keeping the top
      ``n_ranked`` positives -> ``Ranked Answers: ...``.
    """

    dataset_name = dataset or "question answering"
    head = (
        f"You are given a question from the {dataset_name} dataset together with "
        f"{len(candidates)} candidate answers produced by a language model. "
        "Simulate human preference and select the most suitable candidate(s)."
    )
    criteria_block = f"Selection criteria:\n{criteria}"
    question_block = f"Question:\n{str(question).strip()}"
    candidates_block = "Candidate answers:\n" + format_candidate_block(candidates)

    if ranked:
        instructions = (
            f"Rank the top {min(n_ranked, len(candidates))} candidate answers from "
            "most to least preferred according to the criteria above. Two candidates "
            "may be tied; in that case list the indices in any order.
"
            + RANKING_OUTPUT_FORMAT
        )
    else:
        instructions = (
            "Select the single best candidate answer according to the criteria above "
            "and explain your choice. Answer with the exact format below on the last "
            "line.\n" + BEST_ANSWER_OUTPUT_FORMAT
        )

    parts = [head, criteria_block, question_block, candidates_block, instructions]
    if extra_instructions:
        parts.append(str(extra_instructions).strip())
    return "\n\n".join(parts)


def build_truthfulqa_ranking_prompt(
    question: str,
    candidates: Sequence[str],
    *,
    n_ranked: int = 5,
) -> str:
    """TruthfulQA variant: ranked top-5 selection (Appendix G / §4.1)."""

    return build_ai_feedback_prompt(
        question,
        candidates,
        dataset="truthfulqa",
        ranked=True,
        n_ranked=n_ranked,
    )


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #


def _self_test() -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    sqa = build_strategyqa_prompt("Is the color of an apple red?")
    assert sqa.startswith(STRATEGYQA_INSTRUCTION[:30])
    assert "Example 1:" in sqa and "Example 2:" in sqa
    assert sqa.rstrip().endswith("A:")
    assert "Karachi" in sqa
    out["strategyqa_chars"] = len(sqa)

    sqa_nointr = build_strategyqa_prompt("Is the color of an apple red?", model="mixtral")
    assert STRATEGYQA_INSTRUCTION[:30] not in sqa_nointr
    assert "Karachi" in sqa_nointr
    out["strategyqa_no_instruction"] = True

    gsm = build_gsm8k_prompt("How many apples are there?")
    assert gsm.count("Example ") == 4
    out["gsm8k_shots"] = 4

    sci = build_scienceqa_prompt(
        "What is water?", choices=["solid", "gas", "liquid", "plasma"]
    )
    assert "1. solid" in sci and "A:" in sci
    out["scienceqa_has_choices"] = True

    tqa = build_truthfulqa_prompt("What is the capital of France?")
    assert "helpful, honest and harmless" in tqa
    out["truthfulqa_chars"] = len(tqa)

    rater = build_ai_feedback_prompt(
        "Is the sky blue?", ["Yes.\n#### Yes.", "No.\n#### No."], dataset="strategyqa"
    )
    assert "Answer 1:" in rater and "Answer 2:" in rater
    assert "Best Answer and Explanation:" in rater
    assert "Coherency" in rater and "Format" in rater
    out["rater_best_format"] = True

    rank = build_truthfulqa_ranking_prompt("q", ["a", "b", "c"])
    assert "Ranked Answers:" in rank
    out["rater_rank_format"] = True

    assert num_shots("strategyqa") == 2 and num_shots("gsm8k") == 4
    assert uses_instruction("strategyqa", "gpt-3.5-turbo") is True
    assert uses_instruction("gsm8k", "davinci-002") is False
    out["shot_and_instruction_rules"] = True
    return out


if __name__ == "__main__":  # pragma: no cover
    import json

    print(json.dumps(_self_test(), indent=2))
