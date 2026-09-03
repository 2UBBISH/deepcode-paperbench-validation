"""
FRE Evaluation Package

Provides zero-shot evaluation of trained FRE agents on downstream tasks.
"""

from fre.evaluation.evaluate import (
    evaluate_agent,
    evaluate_all_tasks,
    run_evaluation,
    make_env,
)

__all__ = [
    "evaluate_agent",
    "evaluate_all_tasks",
    "run_evaluation",
    "make_env",
]