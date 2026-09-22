#!/usr/bin/env bash
# Figure 2 (couplings vs conditioning on 3-mode GMMs) and the Section 3.3 /
# Proposition 3.1 transport-cost study.  Both run on CPU in a few minutes.
set -euo pipefail
python experiments/gmm_coupling.py --steps 4000 --out results
python experiments/transport_cost.py --out results
