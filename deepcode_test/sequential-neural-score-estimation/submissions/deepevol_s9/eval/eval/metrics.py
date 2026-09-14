'''Derive frozen evaluator metrics from a trial and a result object.

Each metric is reported exactly once for the trial partition, in the order
declared by METRIC_SPECS. A metric whose required result key is missing or
malformed is reported as invalid rather than as a measured zero.
'''

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .contract import METRIC_SPECS, validate_contract


def _trial_partition(trial: Dict[str, Any]) -> Optional[str]:
    '''Return the trial partition from either nested or flattened trial data.'''
    inner = trial.get('trial') if isinstance(trial.get('trial'), dict) else trial
    if isinstance(inner, dict):
        value = inner.get('input_key')
        return value if isinstance(value, str) else None
    return None


def evaluate_metrics(trial: Dict[str, Any], data: object) -> List[Dict[str, Any]]:
    '''Return status/value entries for the trial partition frozen metrics.

    Parameters
    ----------
    trial:
        Parsed contents of job/trial.json.
    data:
        Parsed contents of job/evaluation_input.json.

    Returns
    -------
    list of entries, one per metric for the trial partition, each with keys
    metric, unit, status and value. Status is measured when the corresponding
    result key is valid and invalid otherwise.
    '''
    partition = _trial_partition(trial)
    valid = validate_contract(data)

    entries: List[Dict[str, Any]] = []
    for spec in METRIC_SPECS:
        if spec['partition'] != partition:
            continue

        metric = spec['metric']
        if metric in valid:
            status = 'measured'
            value = valid[metric]
        else:
            status = 'invalid'
            value = None

        entries.append(
            {
                'metric': metric,
                'unit': spec['unit'],
                'status': status,
                'value': value,
            }
        )

    return entries
