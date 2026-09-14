'''Evaluator entrypoint for a single reproduction trial.

The evaluator reads the frozen job/trial.json and job/evaluation_input.json,
maps the result object through the shared evaluator contract, and writes
job/evaluation_output.json in the evaluator-output schema. Malformed result
data is reported as invalid per metric, never as a measured zero.
'''

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .io import read_evaluation_input, read_trial, write_evaluation_output
from .metrics import evaluate_metrics

_SCHEMA_VERSION = 'reproduction-evaluator-output.v1'


def _partition_from_trial(trial: Dict[str, Any]) -> Optional[str]:
    '''Return the input_key partition from a nested or flattened trial object.'''
    inner = trial.get('trial') if isinstance(trial.get('trial'), dict) else trial
    if isinstance(inner, dict):
        value = inner.get('input_key')
        return value if isinstance(value, str) else None
    return None


def _has_trial_identifier(trial: object) -> bool:
    '''Return True when trial is a dict carrying a trial_sha256 string.'''
    return isinstance(trial, dict) and isinstance(trial.get('trial_sha256'), str)


def main() -> None:
    '''Run the evaluator for the trial described by job/trial.json.'''
    try:
        trial = read_trial()
    except Exception:
        raise SystemExit(1)

    if not _has_trial_identifier(trial):
        raise SystemExit(1)

    if _partition_from_trial(trial) is None:
        raise SystemExit(1)

    try:
        data = read_evaluation_input()
    except Exception:
        # A missing or unparseable evaluation input cannot yield measured values.
        # Report every metric of the known partition as invalid.
        data = None

    values: List[Dict[str, Any]] = evaluate_metrics(trial, data)
    if not values:
        raise SystemExit(1)

    payload = {
        'schema_version': _SCHEMA_VERSION,
        'trial_sha256': trial['trial_sha256'],
        'values': values,
    }
    write_evaluation_output(payload)


if __name__ == '__main__':
    main()
