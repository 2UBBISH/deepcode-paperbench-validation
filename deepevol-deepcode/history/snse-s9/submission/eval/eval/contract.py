'''Result contract and frozen metric schedule for the evaluator.

The implementation and the evaluator exchange one JSON object with four
top-level numeric keys: c2st_score, valid_summary_stats_percent,
empirical_expected_coverage and posterior_predictive_visual_agreement. This
module owns the evaluator copy of that contract and the metric schedule per
trial partition. It is intentionally pure standard library so malformed inputs
can be parsed without third-party imports.
'''

from __future__ import annotations

import math
from typing import Any, Dict, List


METRIC_SPECS: List[Dict[str, Any]] = [
    {
        'partition': 'benchmarks',
        'metric': 'c2st_score',
        'unit': 'score [0.5, 1]',
        'policy_metric': 'C2ST score',
        'result_expression': 'c2st_score',
    },
    {
        'partition': 'pyloric_network',
        'metric': 'valid_summary_stats_percent',
        'unit': 'percent',
        'policy_metric': 'Percentage of valid summary statistics',
        'result_expression': 'valid_summary_stats_percent',
    },
    {
        'partition': 'pyloric_network',
        'metric': 'empirical_expected_coverage',
        'unit': 'coverage probability',
        'policy_metric': 'Empirical expected coverage',
        'result_expression': 'empirical_expected_coverage',
    },
    {
        'partition': 'pyloric_network',
        'metric': 'posterior_predictive_visual_agreement',
        'unit': 'qualitative',
        'policy_metric': 'Posterior predictive visual agreement',
        'result_expression': 'posterior_predictive_visual_agreement',
    },
]


_CONTRACT_KEYS = tuple(spec['metric'] for spec in METRIC_SPECS)
_VISUAL_AGREEMENT_VALUES = {0.0, 1.0}


def validate_contract(data: object) -> Dict[str, float]:
    '''Parse a result object and return only valid numeric contract values.

    Only finite int/float values for the four documented keys are returned.
    Booleans, strings, nested values, missing keys and non-finite numbers are
    all treated as invalid for the corresponding metric. The visual-agreement
    key additionally has to be exactly 0.0 or 1.0.
    '''
    if not isinstance(data, dict):
        return {}

    valid: Dict[str, float] = {}
    for key in _CONTRACT_KEYS:
        if key not in data:
            continue
        value = data[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        number = float(value)
        if not math.isfinite(number):
            continue
        if key == 'posterior_predictive_visual_agreement' and number not in _VISUAL_AGREEMENT_VALUES:
            continue
        valid[key] = number
    return valid
