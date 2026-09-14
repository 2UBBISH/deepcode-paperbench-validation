'''Standard-library I/O for the evaluator job directory.

The runtime protocol freezes three evaluator paths relative to the workspace
root: job/trial.json and job/evaluation_input.json are inputs, and
job/evaluation_output.json is the evaluator output. These helpers operate on
those exact paths and create the job directory when writing.
'''

from __future__ import annotations

import json
import os
from typing import Any, Dict

from .contract import METRIC_SPECS as _METRIC_SPECS  # noqa: F401  (kept as the blueprint dependency surface)


_TRIAL_PATH = 'job/trial.json'
_EVALUATION_INPUT_PATH = 'job/evaluation_input.json'
_EVALUATION_OUTPUT_PATH = 'job/evaluation_output.json'


def read_trial() -> Dict[str, Any]:
    '''Read job/trial.json and return its parsed JSON object.'''
    with open(_TRIAL_PATH, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def read_evaluation_input() -> object:
    '''Read job/evaluation_input.json and return its parsed JSON object.'''
    with open(_EVALUATION_INPUT_PATH, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def write_evaluation_output(payload: Dict[str, Any]) -> None:
    '''Write the evaluator output object to job/evaluation_output.json.'''
    output_dir = os.path.dirname(_EVALUATION_OUTPUT_PATH)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(_EVALUATION_OUTPUT_PATH, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, allow_nan=False)
