"""Few-shot prompts, system prompts and shared sweep constants for CFG-LM.

This module centralises every *prompt* used across the reproduction so the CoT
scripts (Figure 2 / Figure 17), the zero-shot sweep (Table 5) and the (out of
scope to run, but kept for completeness) system-prompt experiments of Section
3.4 all share one code path.

Contents
--------
* ``GSM8K_8SHOT`` / ``build_gsm8k_prompt`` : the 8-shot chain-of-thought prompt
  of Wang et al. (2023) "Self-Consistency Improves Chain of Thought Reasoning in
  Language Models".  Section 3.2 states: *"We follow (Wang et al, 2023)'s
  few-shot prompt"*.  Chains terminate in a ``#### <number>`` answer, and the
  paper's CoT evaluation asks whether each generation yields a *valid, parsable*
  answer, so the prompt format matters for the parser in
  :mod:`src.eval.cot_eval`.
* ``AQUA_4SHOT`` / ``build_aqua_prompt`` : the standard AQuA few-shot
  arithmetic prompt (Ling et al. 2017), with answers formatted
  ``The answer is (x).`` so that ``AQUA_ANSWER_MARKER = "The answer is"``
  (used by :mod:`src.eval.cot_eval`) extracts the final choice letter.
* ``DEFAULT_SYSTEM_PROMPT`` and ``EDITED_SYSTEM_PROMPTS`` : the Section 3.4
  system prompts.  CFG with negative prompting (Eq. 5) sets the negative prompt
  ``c_bar`` to the *default* system prompt and ``c`` to an *edited* one, e.g.
  "... decide which and write an appropriate response" ->
  "... decide which and write a sad response".  Section 3.4 generates
  ``n_c = 25`` system prompts and ``n_p = 46`` user prompts, and samples
  ``1740`` random ``(system, user)`` combinations, choosing the guidance
  strength randomly from ``{1, 2, 3, 4, 5, 6}``.
* Shared sweep constants so Table 5, Figures 2/11-17 and Tables 2/7/8/9 all use
  identical grids: ``CFG_GAMMAS``, ``HUMANEVAL_TEMPERATURES``,
  ``NEGATIVE_PROMPT_GAMMAS``, ``ANALYSIS_GAMMA``, and the unconditional-prompt
  conventions ``UNCONDITIONAL_MODE_ZERO_SHOT`` / ``UNCONDITIONAL_MODE_DEFAULT``.
  The HumanEval generation budget is 512 tokens (Section 3.3.1).

The module is deliberately dependency-free (stdlib only) so plotting and
evaluation code can import it without pulling in torch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    # GSM8K
    "GSM8K_EXEMPLARS",
    "GSM8K_8SHOT",
    "GSM8K_PROMPT",
    "GSM8K_ANSWER_MARKER",
    "GSM8K_ANSWER_FORMAT",
    "build_gsm8k_prompt",
    # AQuA
    "AQUA_EXEMPLARS",
    "AQUA_4SHOT",
    "AQUA_PROMPT",
    "AQUA_ANSWER_MARKER",
    "build_aqua_prompt",
    # HumanEval / CoT budgets
    "HUMANEVAL_INSTRUCTION",
    "HUMANEVAL_MAX_NEW_TOKENS",
    "COT_MAX_NEW_TOKENS",
    # Section 3.4 system prompts
    "DEFAULT_SYSTEM_PROMPT",
    "EDITED_SYSTEM_PROMPTS",
    "SYSTEM_PROMPTS",
    "USER_PROMPTS",
    "N_SYSTEM_PROMPTS",
    "N_USER_PROMPTS",
    "N_PROMPT_PAIRS",
    "NEGATIVE_PROMPT_GAMMAS",
    "build_chat_prompt",
    "negative_prompt_pair",
    # shared sweep constants
    "CFG_GAMMAS",
    "HUMANEVAL_TEMPERATURES",
    "ANALYSIS_GAMMA",
    "UNCONDITIONAL_MODE_DEFAULT",
    "UNCONDITIONAL_MODE_ZERO_SHOT",
    "UNCONDITIONAL_MODES",
    "ZERO_SHOT_MAX_LENGTH",
    "HARNESS_DEFAULT_TEMPERATURE",
    "HARNESS_DEFAULT_TOP_P",
    # helpers
    "PromptExample",
    "PROMPT_REGISTRY",
    "COT_PROMPTS",
    "get_prompt",
    "build_cot_prompt",
]


# ---------------------------------------------------------------------------
# Answer markers / formats (kept in sync with src/eval/cot_eval.py)
# ---------------------------------------------------------------------------
GSM8K_ANSWER_MARKER = "####"
GSM8K_ANSWER_FORMAT = "#### {answer}"
AQUA_ANSWER_MARKER = "The answer is"

#: Generation budgets.  HumanEval uses 512 new tokens (Section 3.3.1); the CoT
#: tasks stop when the chain reaches its answer marker.
HUMANEVAL_MAX_NEW_TOKENS = 512
COT_MAX_NEW_TOKENS = 512


# ---------------------------------------------------------------------------
# GSM8K -- Self-Consistency 8-shot prompt (Wang et al. 2023)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PromptExample:
    """A single (question, chain-of-thought) few-shot exemplar."""

    question: str
    answer: str

    def render(self, template: str = "Q: {q}\nA: {a}\n\n") -> str:
        return template.format(q=self.question.strip(), a=self.answer.strip())


GSM8K_EXEMPLARS: Tuple[PromptExample, ...] = (
    PromptExample(
        "There are 15 trees in the grove. Grove workers will plant trees in the "
        "grove today. After they are done, there will be 21 trees. How many trees "
        "did the grove workers plant today?",
        "We start with 15 trees. Later we have 21 trees. The difference must be "
        "the number of trees they planted. So, they must have planted "
        "21 - 15 = 6 trees. The answer is 6.",
    ),
    PromptExample(
        "If there are 3 cars in the parking lot and 2 more cars arrive, how many "
        "cars are in the parking lot?",
        "There are 3 cars in the parking lot already. 2 more arrive. Now there "
        "are 3 + 2 = 5 cars. The answer is 5.",
    ),
    PromptExample(
        "Leah had 32 chocolates and her sister had 42. If they ate 35, how many "
        "pieces do they have left in total?",
        "Leah had 32 chocolates and Leah's sister had 42. That means there were "
        "originally 32 + 42 = 74 chocolates. 35 have been eaten. So in total they "
        "have now 74 - 35 = 39 chocolates. The answer is 39.",
    ),
    PromptExample(
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 "
        "lollipops. How many lollipops did Jason give to Denny?",
        "Jason had 20 lollipops. Since he only has 12 now, he must have given the "
        "rest to Denny. The number of lollipops he has given to Denny must have "
        "been 20 - 12 = 8 lollipops. The answer is 8.",
    ),
    PromptExample(
        "Shawn has five toys. For Christmas, he got two toys each from his mom "
        "and dad. How many toys does he have now?",
        "He has 5 toys. He got 2 from mom, so after that he has 5 + 2 = 7 toys. "
        "Then he got 2 more from dad, so in total he has 7 + 2 = 9 toys. The "
        "answer is 9.",
    ),
    PromptExample(
        "There were nine computers in the server room. Five more computers were "
        "installed each day, from monday to thursday. How many computers are now "
        "in the server room?",
        "There are 4 days from monday to thursday. 5 computers were added each "
        "day. That means in total 4 * 5 = 20 computers were added. There were 9 "
        "computers in the beginning, so now there are 9 + 20 = 29 computers. The "
        "answer is 29.",
    ),
    PromptExample(
        "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On "
        "wednesday, he lost 2 more. How many golf balls did he have at the end of "
        "wednesday?",
        "Michael initially had 58 balls. He lost 23 on Tuesday, so after that he "
        "has 58 - 23 = 35 balls. On Wednesday he lost 2 more so now he has "
        "35 - 2 = 33 balls. The answer is 33.",
    ),
    PromptExample(
        "Olivia has $23. She bought five bagels for $3 each. How much money does "
        "she have left?",
        "She bought 5 bagels for $3 each. This means she spent 5 * $3 = $15 on "
        "the bagels. She was initially $23 and spent $15, so she has now "
        "$23 - $15 = $8. The answer is 8.",
    ),
)

_GSM8K_TEMPLATE = "Q: {q}\nA: {a}\n\n"


def _render_exemplars(exemplars, template: str) -> str:
    return "".join(
        template.format(q=e.question.strip(), a=e.answer.strip()) for e in exemplars
    )


#: The full 8-shot GSM8K prompt prefix.  The test question is appended by
#: :func:`build_gsm8k_prompt` as ``"Q: {question}\nA:"``.
GSM8K_8SHOT = _render_exemplars(GSM8K_EXEMPLARS, _GSM8K_TEMPLATE)
GSM8K_PROMPT = GSM8K_8SHOT


def build_gsm8k_prompt(question: str, with_answer_prefix: bool = True) -> str:
    """Prepend the Self-Consistency 8-shot prefix to ``question``.

    Parameters
    ----------
    question:
        The GSM8K test question (without a ``#### ...`` answer line).
    with_answer_prefix:
        Append the ``Q: ...\\nA:`` opener so the model continues directly with
        its chain of thought (the paper's CoT setting).
    """

    prompt = GSM8K_8SHOT + "Q: " + question.strip() + "\n"
    if with_answer_prefix:
        prompt += "A:"
    return prompt


# ---------------------------------------------------------------------------
# AQuA -- standard 4-shot arithmetic prompt (Figure 17)
# ---------------------------------------------------------------------------
AQUA_EXEMPLARS: Tuple[PromptExample, ...] = (
    PromptExample(
        "John found that the average of 15 numbers is 40. If 10 is added to each "
        "number then the mean of the numbers is\nAnswer Choices: (a) 50 (b) 45 "
        "(c) 65 (d) 78 (e) 64",
        "If 10 is added to each number, then the mean of the numbers also "
        "increases by 10. So the new mean would be 50. The answer is (a).",
    ),
    PromptExample(
        "A person is traveling at 20 km/hr and reached his destiny in 2.5 hr "
        "then find the distance?\nAnswer Choices: (a) 53 km (b) 55 km (c) 52 km "
        "(d) 60 km (e) 50 km",
        "Distance = speed * time = 20 km/hr * 2.5 hr = 50 km. "
        "The answer is (e).",
    ),
    PromptExample(
        "How many keystrokes are needed to type the numbers from 1 to 500?\n"
        "Answer Choices: (a) 1156 (b) 1392 (c) 1480 (d) 1562 (e) 1788",
        "There are 9 one-digit numbers from 1 to 9. There are 90 two-digit "
        "numbers from 10 to 99. There are 401 three-digit numbers from 100 to "
        "500. 9 + 90 + 401 = 500 numbers. 9*1 + 90*2 + 401*3 = 9 + 180 + 1203 = "
        "1392 keystrokes. The answer is (b).",
    ),
    PromptExample(
        "If a / b = 3/4 and 8a + 5b = 22, then find the value of a.\n"
        "Answer Choices: (a) 1/2 (b) 3/2 (c) 5/2 (d) 4/2 (e) 7/2",
        "a/b = 3/4 implies a = 3k and b = 4k for some k. Then "
        "8a + 5b = 8(3k) + 5(4k) = 24k + 20k = 44k = 22, so k = 0.5 = 1/2. "
        "Therefore a = 3k = 3/2. The answer is (b).",
    ),
)

AQUA_4SHOT = _render_exemplars(AQUA_EXEMPLARS, _GSM8K_TEMPLATE)
AQUA_PROMPT = AQUA_4SHOT


def build_aqua_prompt(question: str, with_answer_prefix: bool = True) -> str:
    """Prepend the 4-shot AQuA prefix to a question(+choices) string."""

    prompt = AQUA_4SHOT + "Q: " + question.strip() + "\n"
    if with_answer_prefix:
        prompt += "A:"
    return prompt


# ---------------------------------------------------------------------------
# HumanEval / code generation
# ---------------------------------------------------------------------------
HUMANEVAL_INSTRUCTION = "Complete the following Python function.\n\n"


# ---------------------------------------------------------------------------
# Section 3.4 -- system prompts for negative prompting (Eq. 5)
# ---------------------------------------------------------------------------
#: The default system prompt of GPT4All-J (Section 3.4).  This is the *negative*
#: prompt ``c_bar`` in Equation 5.
DEFAULT_SYSTEM_PROMPT = (
    "The prompt below is a question to answer, a task to complete, or a "
    "conversation to respond to; decide which and write an appropriate response."
)

#: Edited system prompts (the conditional prompt ``c``).  Section 3.4 generates
#: ``n_c = 25`` such prompts; a representative subset is included here, starting
#: with the exact example quoted in the paper ("write a sad response").
EDITED_SYSTEM_PROMPTS: Dict[str, str] = {
    "sad": (
        "The prompt below is a question to answer, a task to complete, or a "
        "conversation to respond to; decide which and write a sad response."
    ),
    "happy": (
        "The prompt below is a question to answer, a task to complete, or a "
        "conversation to respond to; decide which and write a happy response."
    ),
    "formal": (
        "The prompt below is a question to answer, a task to complete, or a "
        "conversation to respond to; decide which and write a formal response."
    ),
    "casual": (
        "The prompt below is a question to answer, a task to complete, or a "
        "conversation to respond to; decide which and write a casual response."
    ),
    "concise": (
        "The prompt below is a question to answer, a task to complete, or a "
        "conversation to respond to; decide which and write a concise response."
    ),
    "detailed": (
        "The prompt below is a question to answer, a task to complete, or a "
        "conversation to respond to; decide which and write a detailed response."
    ),
    "poetic": (
        "The prompt below is a question to answer, a task to complete, or a "
        "conversation to respond to; decide which and write a poetic response."
    ),
    "sarcastic": (
        "The prompt below is a question to answer, a task to complete, or a "
        "conversation to respond to; decide which and write a sarcastic response."
    ),
    "expert": (
        "The prompt below is a question to answer, a task to complete, or a "
        "conversation to respond to; decide which and write an expert response."
    ),
    "child": (
        "The prompt below is a question to answer, a task to complete, or a "
        "conversation to respond to; decide which and write a response that a "
        "child would understand."
    ),
}

#: All system prompts keyed by name ("default" plus every edit).
SYSTEM_PROMPTS: Dict[str, str] = {"default": DEFAULT_SYSTEM_PROMPT}
SYSTEM_PROMPTS.update(EDITED_SYSTEM_PROMPTS)

#: Section 3.4 protocol: n_c = 25 system prompts, n_p = 46 user prompts,
#: 1740 randomly sampled (system, user) pairs.
N_SYSTEM_PROMPTS = 25
N_USER_PROMPTS = 46
N_PROMPT_PAIRS = 1740

#: Section 3.4 randomly chooses the guidance strength in {1, 2, 3, 4, 5, 6}
#: for the two chatbot completions (one vanilla, one CFG).
NEGATIVE_PROMPT_GAMMAS: Tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)

#: User prompts (the user query ``p``); a representative subset of the 46
#: prompts of the paper's Appendix G.
USER_PROMPTS: Tuple[str, ...] = (
    "What is the meaning of life?",
    "Tell me a joke.",
    "Write a haiku about the ocean.",
    "Explain quantum computing to a five-year-old.",
    "What is the capital of France?",
    "Give me three tips for sleeping better.",
    "Summarise the plot of Romeo and Juliet in two sentences.",
    "How do I bake sourdough bread?",
    "What are the benefits of exercise?",
    "Describe a sunset in one paragraph.",
    "How do I learn to play the guitar?",
    "What should I cook for dinner tonight?",
    "Why is the sky blue?",
    "Write a short story about a robot.",
    "What is the best way to study for an exam?",
)


def build_chat_prompt(
    user_prompt: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    template: str = "{system}\n\n{prompt}",
) -> str:
    """Assemble a two-part chatbot prompt (system prompt + user prompt)."""

    return template.format(system=system_prompt.strip(), prompt=user_prompt.strip())


def negative_prompt_pair(
    edited: str,
    default: str = DEFAULT_SYSTEM_PROMPT,
) -> Tuple[str, str]:
    """Return ``(cond, negative)`` prompts for Equation 5.

    ``cond`` is the edited system prompt ``c``; ``negative`` is the default
    system prompt ``c_bar`` (Section 3.4).  ``edited`` may be a key of
    :data:`EDITED_SYSTEM_PROMPTS` or the full prompt text.
    """

    if edited in EDITED_SYSTEM_PROMPTS:
        edited = EDITED_SYSTEM_PROMPTS[edited]
    return edited, default


# ---------------------------------------------------------------------------
# Shared sweep constants (mirrored in configs/default.yaml)
# ---------------------------------------------------------------------------
#: Guidance strengths swept for Table 5 (zero-shot), Tables 2/7/8/9 (HumanEval)
#: and Figures 2/11-17.  gamma=1.0 is the vanilla baseline; gamma=0.0 would be
#: fully unconditioned.
CFG_GAMMAS: Tuple[float, ...] = (1.0, 1.1, 1.25, 1.5, 1.75, 2.0)

#: Temperatures used for HumanEval (Section 3.3.1).
HUMANEVAL_TEMPERATURES: Tuple[float, ...] = (0.2, 0.6, 0.8)

#: Guidance strength used for the Section 5 analysis (entropy / overlap / PPL).
ANALYSIS_GAMMA: float = 1.5

#: Unconditional-prompt conventions (Section 2.2 / Section 3.1).
UNCONDITIONAL_MODE_DEFAULT = "empty_prefix"
UNCONDITIONAL_MODE_ZERO_SHOT = "last_prompt_token"
UNCONDITIONAL_MODES: Tuple[str, ...] = (
    UNCONDITIONAL_MODE_DEFAULT,
    UNCONDITIONAL_MODE_ZERO_SHOT,
)

#: Zero-shot (LM Evaluation Harness) defaults: greedy scoring, no nucleus
#: truncation, 2048-token context (matching the harness shim in
#: :mod:`src.eval.harness_cfg`).
HARNESS_DEFAULT_TEMPERATURE = 0.0
HARNESS_DEFAULT_TOP_P = 1.0
ZERO_SHOT_MAX_LENGTH = 2048


# ---------------------------------------------------------------------------
# Registry / convenience helpers
# ---------------------------------------------------------------------------
PROMPT_REGISTRY: Dict[str, str] = {
    "gsm8k_8shot": GSM8K_8SHOT,
    "aqua_4shot": AQUA_4SHOT,
    "default_system": DEFAULT_SYSTEM_PROMPT,
}

for _name, _text in EDITED_SYSTEM_PROMPTS.items():
    PROMPT_REGISTRY[f"system_{_name}"] = _text

#: Mapping of task name -> few-shot prefix / builder for the CoT scripts.
COT_PROMPTS: Dict[str, Dict[str, object]] = {
    "gsm8k": {
        "prefix": GSM8K_8SHOT,
        "builder": build_gsm8k_prompt,
        "answer_marker": GSM8K_ANSWER_MARKER,
    },
    "aqua": {
        "prefix": AQUA_4SHOT,
        "builder": build_aqua_prompt,
        "answer_marker": AQUA_ANSWER_MARKER,
    },
}


def get_prompt(name: str) -> str:
    """Look up a prompt by name (raises ``KeyError`` listing what is available)."""

    if name not in PROMPT_REGISTRY:
        raise KeyError(
            f"Unknown prompt '{name}'. Available: {sorted(PROMPT_REGISTRY)}"
        )
    return PROMPT_REGISTRY[name]


def build_cot_prompt(
    question: str,
    task: str = "gsm8k",
    with_answer_prefix: bool = True,
) -> str:
    """Build the few-shot CoT prompt for ``task`` (``gsm8k`` or ``aqua``).

    The prompts follow Wang et al. (2023) for GSM8K, as specified in Section 3.2.
    """

    key = task.lower()
    if key in ("gsm8k", "gsm-8k", "math"):
        return build_gsm8k_prompt(question, with_answer_prefix=with_answer_prefix)
    if key in ("aqua", "aqua-rat", "aqua_rat"):
        return build_aqua_prompt(question, with_answer_prefix=with_answer_prefix)
    raise ValueError(f"Unknown CoT task '{task}'; expected 'gsm8k' or 'aqua'.")
