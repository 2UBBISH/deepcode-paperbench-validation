"""Download the three posteriordb targets used in Section 5.2 and store compact copies.

The paper uses three targets from posteriordb (Magnusson et al., 2022):

    arK                    D = 7   (nearly Gaussian)
    gp_pois_regr           D = 13  (non-Gaussian, GP Poisson regression)
    eight_schools_centered D = 10  (non-Gaussian hierarchical model)

For each of them we download

* the data (``posterior_database/data/data/<name>.json.zip``), and
* the reference posterior draws obtained with Hamiltonian Monte Carlo
  (``posterior_database/reference_posteriors/draws/draws/<posterior>.json.zip``),

and write them into ``data/posteriordb/`` as small ``.npz``/``.json`` files.  Those
committed files are what the experiment scripts use, so that reproducing the
paper does not require network access.  Running this script again only
refreshes the cached copies.

``eight_schools_centered`` has no draws of its own in posteriordb, but the
non-centered version is an exact reparameterization of the same posterior and
its draws contain the transformed parameter ``theta`` (i.e. ``theta_j = mu +
tau * theta_tilde_j``), so the reference summaries of the centered model can be
computed directly from them.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import zipfile
from typing import Dict, List

import numpy as np

RAW = "https://raw.githubusercontent.com/stan-dev/posteriordb/master/posterior_database"

MODELS = {
    "arK": {
        "posterior": "arK-arK",
        "data": "arK",
        "params": ["alpha"] + [f"beta[{k}]" for k in range(1, 6)] + ["sigma"],
        "dimensions": 7,
    },
    "gp_pois_regr": {
        "posterior": "gp_pois_regr-gp_pois_regr",
        "data": "gp_pois_regr",
        "params": ["rho", "alpha"] + [f"f[{k}]" for k in range(1, 12)],
        # posteriordb stores the *transformed* parameter f = L(rho, alpha) f_tilde,
        # while the model's parameters (and hence the paper's D = 13 target) are
        # (rho, alpha, f_tilde).  The draws are mapped back below.
        "back_transform": "gp_f_to_f_tilde",
        "dimensions": 13,
    },
    "eight_schools_centered": {
        "posterior": "eight_schools-eight_schools_noncentered",  # same posterior, see docstring
        "data": "eight_schools",
        "params": [f"theta[{j}]" for j in range(1, 9)] + ["mu", "tau"],
        "dimensions": 10,
    },
}


def _fetch_zip_json(url: str):
    import urllib.request

    with urllib.request.urlopen(url, timeout=120) as fh:
        blob = fh.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as f:
            return json.load(f)


def _gp_f_to_f_tilde(draws: np.ndarray, data: dict) -> np.ndarray:
    """Map reference draws of ``(rho, alpha, f)`` to ``(rho, alpha, f_tilde)``.

    ``f = L(rho, alpha) f_tilde`` with ``L`` the Cholesky factor of
    ``exp_quad_cov(x, alpha, rho) + 1e-10 I``, so ``f_tilde = L^{-1} f``.
    """
    x = np.asarray(data["x"], dtype=np.float64)
    d2 = (x[:, None] - x[None, :]) ** 2
    out = draws.copy()
    for i in range(draws.shape[0]):
        rho, alpha = draws[i, 0], draws[i, 1]
        cov = alpha**2 * np.exp(-d2 / (2.0 * rho**2)) + 1e-10 * np.eye(x.shape[0])
        L = np.linalg.cholesky(cov)
        out[i, 2:] = np.linalg.solve(L, draws[i, 2:])
    return out


BACK_TRANSFORMS = {"gp_f_to_f_tilde": _gp_f_to_f_tilde}


def _ref_stats(name: str, params, dim: int, draws: np.ndarray, data_url: str, draws_url: str) -> dict:
    return {
        "name": name,
        "params": params,
        "dim": dim,
        "n_draws": int(draws.shape[0]),
        "mean": draws.mean(axis=0).tolist(),
        "sd": draws.std(axis=0, ddof=1).tolist(),
        "source": {"data": data_url, "draws": draws_url},
    }


def download(outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    for name, spec in MODELS.items():
        data_url = f"{RAW}/data/data/{spec['data']}.json.zip"
        draws_url = f"{RAW}/reference_posteriors/draws/draws/{spec['posterior']}.json.zip"
        data = _fetch_zip_json(data_url)
        draws_raw = _fetch_zip_json(draws_url)

        # draws_raw is a list of chains; each chain is a dict param -> list of draws
        chains: List[np.ndarray] = []
        for chain in draws_raw:
            cols = [np.asarray(chain[p], dtype=np.float64) for p in spec["params"]]
            chains.append(np.stack(cols, axis=1))
        draws = np.concatenate(chains, axis=0)
        back = spec.get("back_transform")
        if back is not None:
            # keep both parameterizations: the transformed parameter f (as stored
            # by posteriordb, and as used by the Section 5.2 experiment) and the
            # model parameter f_tilde = L(rho, alpha)^{-1} f
            draws_alt = BACK_TRANSFORMS[back](draws, data).astype(np.float32)
            np.savez_compressed(os.path.join(outdir, f"{name}_ftilde_draws.npz"), draws=draws_alt)
            alt_params = ["rho", "alpha"] + [f"f_tilde[{k}]" for k in range(1, draws.shape[1] - 1)]
            with open(os.path.join(outdir, f"{name}_ftilde_ref_stats.json"), "w") as fh:
                json.dump(_ref_stats(f"{name}_ftilde", alt_params, spec["dimensions"], draws_alt,
                                     data_url, draws_url), fh, indent=2)
        draws = draws.astype(np.float32)

        with open(os.path.join(outdir, f"{name}_data.json"), "w") as fh:
            json.dump(data, fh)
        np.savez_compressed(os.path.join(outdir, f"{name}_draws.npz"), draws=draws)
        with open(os.path.join(outdir, f"{name}_ref_stats.json"), "w") as fh:
            json.dump(_ref_stats(name, spec["params"], spec["dimensions"], draws, data_url, draws_url),
                      fh, indent=2)
        print(f"{name}: dim={spec['dimensions']} draws={draws.shape} -> {outdir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                     "data", "posteriordb"))
    args = ap.parse_args()
    download(args.outdir)
