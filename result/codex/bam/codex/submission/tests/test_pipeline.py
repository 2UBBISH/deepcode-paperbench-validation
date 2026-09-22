"""End-to-end checks of the experiment plumbing (runner + metrics + plotting)."""

import os
import sys
import tempfile
import unittest

import jax
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from bam.deep_generative import DeepGenerativePosterior  # noqa: E402
from bam.posterior_models import load_posteriordb_target  # noqa: E402
from bam.runner import (  # noqa: E402
    bam_spec,
    evaluate_metric,
    gradient_spec,
    gsm_spec,
    plot_curves,
    run_method,
    run_repeats,
    summarise_curves,
)
from bam.targets import random_gaussian_target  # noqa: E402
from bam.vae import VAEConfig, init_params  # noqa: E402


class TestPipeline(unittest.TestCase):
    def test_gaussian_metrics_and_method_dispatch(self):
        D = 4
        target = random_gaussian_target(D, seed=0)
        key = jax.random.PRNGKey(0)
        mu0, Sigma0 = np.zeros(D), np.eye(D)
        specs = [bam_spec(5, 40, lam_value=5 * D), gradient_spec("ADVI", 2, 40, 0.01),
                 gsm_spec(2, 40)]
        finals = {}
        for spec in specs:
            out = run_method(target, spec, key, mu0, Sigma0)
            self.assertEqual(out["grad_evals"][-1], spec.batch_size * spec.n_iters)
            for metric in ("forward_kl", "reverse_kl", "score_div"):
                vals = evaluate_metric(target, out, metric, jax.random.PRNGKey(1), n_samples=2000,
                                       stride=10)
                self.assertTrue(np.isfinite(vals).any(), msg=f"{spec.name}/{metric} all non-finite")
            finals[spec.name] = evaluate_metric(target, out, "forward_kl", key, n_samples=2000)[-1]
        self.assertLess(finals["BaM"], finals["ADVI"])
        self.assertLess(finals["BaM"], 1e-2)

    def test_run_repeats_and_summarise(self):
        D = 3
        spec = bam_spec(2, 20, lam_value=2 * D)
        res = run_repeats(lambda seed: random_gaussian_target(D, seed=seed), spec, "forward_kl",
                          n_runs=2, stride=5)
        self.assertEqual(res["values"].shape[0], 2)
        s = summarise_curves(res)
        self.assertEqual(s["x"].shape, s["y"].shape)
        self.assertTrue(np.all(s["y"] <= s["y"][0] + 1e-6))

    def test_posterior_metrics(self):
        target = load_posteriordb_target("eight_schools_centered")
        spec = bam_spec(4, 20, lam_value=4 * target.dim, decay=True)
        out = run_method(target, spec, jax.random.PRNGKey(0), np.zeros(target.dim), np.eye(target.dim))
        for metric in ("rel_mean", "rel_sd"):
            vals = evaluate_metric(target, out, metric, jax.random.PRNGKey(1))
            self.assertTrue(np.isfinite(vals).all())
            self.assertLess(vals[-1], vals[0])

    def test_deep_generative_mse_metric(self):
        config = VAEConfig(latent_dim=4, c_hid=2)
        params = init_params(jax.random.PRNGKey(0), config)
        x_obs = np.zeros((32, 32, 3))
        target = DeepGenerativePosterior(params["dec"], config, x_obs)
        spec = bam_spec(3, 5, lam_value=10.0)
        out = run_method(target, spec, jax.random.PRNGKey(0), np.zeros(4), np.eye(4))
        vals = evaluate_metric(target, out, "mse", jax.random.PRNGKey(1))
        self.assertTrue(np.isfinite(vals).all())

    def test_bam_covariance_stays_positive_definite_on_ill_conditioned_posterior(self):
        """Regression test: on the (very ill-conditioned) GP Poisson regression
        posterior the covariance must remain positive definite and the run must
        not fail, even though that target is hostile to the paper's
        initialization (see the README section 9)."""
        target = load_posteriordb_target("gp_pois_regr")
        for B in (8, 32):
            spec = bam_spec(B, 150, lam_value=float(B * target.dim), decay=True)
            out = run_method(target, spec, jax.random.PRNGKey(B), np.zeros(target.dim), np.eye(target.dim))
            for Sigma in out["Sigma"]:
                self.assertTrue(np.isfinite(Sigma).all())
                self.assertGreater(np.linalg.eigvalsh(Sigma).min(), 0.0)
                np.linalg.cholesky(Sigma)  # must not raise

    def test_plot_curves_writes_a_file(self):
        curves = [{"label": "a", "x": np.array([1.0, 10.0, 100.0]), "y": np.array([1.0, 0.1, 0.01])},
                  {"label": "b", "x": np.array([1.0, 10.0, 100.0]), "y": np.array([2.0, 1.0, 0.5]),
                   "linestyle": "--"}]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "plot.png")
            plot_curves(curves, "x", "y", "title", path)
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 1000)


if __name__ == "__main__":
    unittest.main()
