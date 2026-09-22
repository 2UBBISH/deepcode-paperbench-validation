"""System/user prompts of the assistant experiment (Section 3.4, Appendix G).

Section 3.4 (negative prompting for assistants, including the human
preference study) is **out of scope** for this reproduction -- the addendum
excludes it because it requires a human evaluation.  The prompts are included
anyway because the *mechanism* (Equation 5, negative prompting) is part of
the CFG framework implemented in ``cfglm/generation.py``: the negative prompt
is the default system prompt and the positive prompt is one of the edited
system prompts below.

    cfg_generate(model, tokenizer, user_prompt,
                 gamma=3.0,
                 negative_prompt=DEFAULT_SYSTEM_PROMPT,
                 ...)
"""

from __future__ import annotations

from typing import List

DEFAULT_SYSTEM_PROMPT = (
    "The prompt below is a question to answer, a task to complete, or a "
    "conversation to respond to; decide which and write an appropriate response."
)

SYSTEM_PROMPT_SUFFIXES: List[str] = [
    "... write a rap response.",
    "... write an appropriate response as an expert of the field.",
    "... write an appropriate response as a PhD thesis.",
    "... write an appropriate response as a mathematical proof.",
    "... write an appropriate response as an epic poem.",
    "... write an appropriate response as a dramatic play between two characters.",
    "... write an inappropriate response.",
    "... write an appropriate response as a Freudian analysis.",
    "... write a scientific paper responding to it.",
    "... write an appropriate response using metaphors.",
    "... write an appropriate response using deep emotional language.",
    "... write an appropriate extremely thorough response.",
    (
        "The prompt below is a question to answer, a task to complete, or a "
        "conversation to respond to from a 5 years old; decide which and write "
        "an appropriate response."
    ),
    "... write an appropriate response in three parts.",
    "... write an appropriate response as a Python program.",
    "... write an appropriate response as a JSON datastructure.",
    "... write an appropriate response as a list.",
    (
        "... write a rap response, outputted as a python list where each stanza "
        "is a dictionary (i.e. [{'stanza': ''}, {'stanza': ''},...])."
    ),
    "... write an appropriate an enthusiastic response to it.",
    "... write a saddening response to it.",
    "... write a love letter responding to it.",
    "... write an irritating response to it.",
    "... write a seductive response to it.",
]

USER_PROMPTS: List[str] = [
    "Why is The Matrix a great movie?",
    "Why did the chicken cross the road?",
    "What is the meaning of life?",
    "What is the answer to life, the universe, and everything?",
    "What is the best way to cook a steak?",
    "How do you make a pizza?",
    "What is the best way to make a pizza?",
    "Why is the sky blue?",
    "Who is the best basketball player of all time?",
    "What are trans fats?",
    "What are transformers?",
    "What are neural networks?",
    "What is the best way to learn a language?",
    "Who is Optimus Prime?",
    "Write a haiku about the meaning of life.",
    "Write the python code to print the first 100 prime numbers.",
    "Give me a recipe for a delicious meal.",
    "How to implement authentication with Flask?",
    "What is the easiest python library to bootstrap a web app?",
    "I am in France and I want to be polite, give me some advice.",
    "Is Yann LeCun the father of deep learning?",
    "Is Yann LeCun the father of convolutional neural networks?",
    "Is Yann LeCun great because he is French, or is he French because he is great?",
    "Is Yann LeCun great because he is French, or despite being French?",
    "Explain the algorithm AlphaZero in few sentences.",
    "I want to learn how to play chess, what is the best way to start?",
    "How are metal vocalists able to scream for so long?",
    "What is the best way to learn how to sing?",
    "What is the best way to learn how to play the guitar?",
    "Give me compelling ideas for a startup.",
    "Give me compelling ideas for a D&D campaign in a medfan version of Italy.",
    "Give me compelling ideas for a D&D campaign in a medfan version of Greece.",
    "Give me compelling ideas for a D&D campaign in a medfan version of France.",
    "Write the lyrics of a death metal song about chickens.",
    "Write the lyrics of a death metal song about AI research.",
    "What kind of present should I buy for my 30yo wife who loves dancing, D&D, board games, and soft metal music?",
    "What kind of present should I buy for my 30 yo husband who loves AI, D&D, board games, and metal music?",
    "Are nerds trendy?",
    "What is a taxonomy?",
    "What are the main differences between driving in France and in the US?",
    "Who are artists that are similar to Gojira?",
    "Who are artists that are famous in the US but not abroad?",
    "Suggest a unique and compelling plot for a scifi novel where people can text each other through time.",
    "Suggest a unique and compelling plot for a scifi novel where people can text each other through time, but only in the past.",
    "What was the Cambridge Analytica scandal?",
    "Tell me about the band Halocene.",
]


def build_prompt(system_suffix: str, user_prompt: str, prefix: str = DEFAULT_SYSTEM_PROMPT) -> str:
    """Combine a system instruction and a user prompt as in Table 1."""
    return f"Instruction: {prefix}{system_suffix}\nPrompt: {user_prompt}\n"
