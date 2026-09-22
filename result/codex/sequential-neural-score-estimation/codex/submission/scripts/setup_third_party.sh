#!/usr/bin/env bash
# Fetch the external resources required for parts of the reproduction.
#
# The addendum of the reproduction task specifies that
#   * the TSNPE baseline (Section 5.2) and the neuroscience problem (Section 5.3)
#     should use https://github.com/mackelab/tsnpe_neurips, and
#   * the pyloric network simulator comes from https://github.com/mackelab/pyloric.
#
# Nothing here is committed to the repository (see .gitignore); run this script
# before using `baselines/tsnpe_adapter.py` or `experiments/run_pyloric.py`.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY="${HERE}/third_party"
mkdir -p "${THIRD_PARTY}"

if [ ! -d "${THIRD_PARTY}/tsnpe_neurips" ]; then
  git clone --depth 1 https://github.com/mackelab/tsnpe_neurips.git "${THIRD_PARTY}/tsnpe_neurips"
else
  echo "third_party/tsnpe_neurips already present"
fi

if [ ! -d "${THIRD_PARTY}/pyloric" ]; then
  git clone --depth 1 https://github.com/mackelab/pyloric.git "${THIRD_PARTY}/pyloric"
else
  echo "third_party/pyloric already present"
fi

cat <<'EOF'

Next steps
----------
1) Install the modified sbi fork that ships with the TSNPE repository (it
   provides `sbi.utils.support_posterior.PosteriorSupport`, which implements the
   truncated proposal):

       pip install -e third_party/tsnpe_neurips/sbi

2) Install the pyloric simulator (requires NEURON):

       pip install -e third_party/pyloric

3) Verify the TSNPE adapter can find the checkout:

       python -c "from baselines.tsnpe_adapter import ensure_on_path; print(ensure_on_path())"
EOF
