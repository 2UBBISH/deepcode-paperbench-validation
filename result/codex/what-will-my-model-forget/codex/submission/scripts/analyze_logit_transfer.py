#!/usr/bin/env python
"""Figure 2(a): inspect the transfer of logit changes between two examples.

The paper illustrates the analysis with one online learning example (about public
relations) and one upstream example (paraphrase detection).  Any pair of examples
can be passed on the command line; the script reports the largest logit changes of
both examples, the top-2 candidate tokens of the upstream example before/after the
update and whether its prediction flipped (and renders Figure 2a-style bars).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _common import config_from_args, base_parser  # noqa: E402

from wwmf.analysis.logit_transfer import logit_change_report, plot_logit_changes  # noqa: E402
from wwmf.data.types import Example  # noqa: E402
from wwmf.models.tuning import fix_single_error, prepare_model  # noqa: E402
from wwmf.utils import ensure_dir, write_json  # noqa: E402


def main() -> None:
    parser = base_parser(__doc__ or "")
    parser.add_argument("--online-input", required=True)
    parser.add_argument("--online-target", required=True)
    parser.add_argument("--upstream-input", required=True)
    parser.add_argument("--upstream-target", required=True)
    parser.add_argument("--position", type=int, default=0)
    args = parser.parse_args()

    cfg = config_from_args(args)
    model = prepare_model(cfg)
    online = Example(args.online_input, args.online_target, task="online")
    upstream = Example(args.upstream_input, args.upstream_target, task="upstream")
    updated = model.clone()
    fix_single_error(updated, online, steps=cfg.steps_single(), lr=cfg.lr_single(), mode=cfg.tuning_mode)
    report = logit_change_report(model, updated, online, upstream, position=args.position)
    out_dir = ensure_dir(os.path.join(cfg.output_root, "figure2a"))
    write_json(os.path.join(out_dir, "logit_transfer.json"), report)
    plot_logit_changes(report, os.path.join(out_dir, "logit_transfer.png"))
    print(json.dumps(report, indent=2)[:4000])


if __name__ == "__main__":
    main()
