"""Baselines of the paper: NPE, NLE and NRE implemented with the ``sbi`` library.

Appendix A2.1: "For implementing Neural Posterior Estimation (NPE), Neural Ratio
Estimation (NRE), and Neural Likelihood Estimation (NLE), we utilize the sbi
library, adopting default parameters but opting for a more expressive neural
spline flow for NPE and NLE.  Each method was trained using the provided training
loop with a batch size of 1000 and an Adam optimizer.  Training ceased upon
convergence, as indicated by early stopping based on validation loss."
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch


def get_prior(task):
    """Return a ``torch`` prior distribution for a task (for the sbi baselines)."""
    if hasattr(task, "prior_distribution"):
        return task.prior_distribution()
    # Fallback: a box prior spanning the empirical range of the prior samples.
    samples = task.prior_sample(20000, np.random.default_rng(0))
    low = torch.as_tensor(samples.min(axis=0) - 1e-6, dtype=torch.float32)
    high = torch.as_tensor(samples.max(axis=0) + 1e-6, dtype=torch.float32)
    from sbi.utils import BoxUniform
    return BoxUniform(low=low, high=high)


class SBIBaseline:
    """Wrapper around the ``sbi`` inference classes (NPE / NLE / NRE)."""

    def __init__(self, task, method: str = "npe", device: str = "cpu",
                 nsf_hidden: int = 50, nsf_transforms: int = 5,
                 mcmc_method: str = "slice_np_vectorized",
                 show_progress_bars: bool = False):
        self.task = task
        self.method = method.lower()
        self.device = device
        self.nsf_hidden = nsf_hidden
        self.nsf_transforms = nsf_transforms
        self.mcmc_method = mcmc_method
        self.prior = get_prior(task)
        self.show_progress_bars = show_progress_bars
        self.inference = None
        self.posterior = None
        self.likelihood = None
        self.density_estimator = None

    # ------------------------------------------------------------------ build
    def _make_inference(self):
        from sbi.inference import NLE, NPE, NRE
        from sbi.neural_nets import classifier_nn, likelihood_nn, posterior_nn

        if self.method == "npe":
            density_estimator = posterior_nn(
                model="nsf", hidden_features=self.nsf_hidden,
                num_transforms=self.nsf_transforms,
                z_score_theta="independent", z_score_x="independent")
            return NPE(prior=self.prior, density_estimator=density_estimator,
                       show_progress_bars=self.show_progress_bars), \
                "density_estimator"
        if self.method == "nle":
            density_estimator = likelihood_nn(
                model="nsf", hidden_features=self.nsf_hidden,
                num_transforms=self.nsf_transforms,
                z_score_theta="independent", z_score_x="independent")
            return NLE(prior=self.prior, density_estimator=density_estimator,
                       show_progress_bars=self.show_progress_bars), \
                "density_estimator"
        if self.method == "nre":
            classifier = classifier_nn(model="resnet", z_score_theta="independent",
                                       z_score_x="independent")
            return NRE(prior=self.prior, classifier=classifier,
                       show_progress_bars=self.show_progress_bars), "classifier"
        raise ValueError(f"unknown sbi baseline '{self.method}'")

    # ------------------------------------------------------------------ train
    def train(self, theta: np.ndarray, x: np.ndarray,
              training_batch_size: int = 1000, learning_rate: float = 5e-4,
              stop_after_epochs: int = 20, max_num_epochs: int = 1000,
              validation_fraction: float = 0.1, show_progress_bar: bool = False,
              **kwargs):
        """Train the density estimator with the (default) sbi training loop.

        Appendix A2.1: batch size 1000, Adam and early stopping on the validation
        loss.  Keyword arguments that the installed version of ``sbi`` does not
        support are dropped, so the wrapper works with several sbi versions.
        """
        import inspect
        inference, kind = self._make_inference()
        theta_t = torch.as_tensor(np.asarray(theta), dtype=torch.float32)
        x_t = torch.as_tensor(np.asarray(x), dtype=torch.float32)
        inference = inference.append_simulations(theta_t, x_t)
        train_kwargs = dict(
            training_batch_size=training_batch_size,
            learning_rate=learning_rate,
            stop_after_epochs=stop_after_epochs,
            max_num_epochs=max_num_epochs,
            validation_fraction=validation_fraction,
            **kwargs)
        supported = inspect.signature(inference.train).parameters
        train_kwargs = {k: v for k, v in train_kwargs.items() if k in supported}
        self.density_estimator = inference.train(**train_kwargs)
        if self.method == "npe":
            self.posterior = inference.build_posterior(self.density_estimator)
        elif self.method == "nle":
            # The legacy sbi API exposed a likelihood object; in recent versions
            # the trained conditional density estimator *is* the likelihood
            # estimator (``sbi.neural_nets.estimators``) and can be sampled with
            # ``density_estimator.sample(shape, condition=theta)``.
            if hasattr(inference, "build_likelihood"):
                self.likelihood = inference.build_likelihood(
                    self.density_estimator)
            else:
                self.likelihood = None
            self.posterior = inference.build_posterior(
                self.density_estimator, sample_with="mcmc",
                mcmc_method=self.mcmc_method)
        else:  # nre
            self.posterior = inference.build_posterior(
                self.density_estimator, sample_with="mcmc",
                mcmc_method=self.mcmc_method)
        self.inference = inference
        return self

    # ----------------------------------------------------------------- sample
    def sample_posterior(self, x_obs: np.ndarray, num_samples: int = 5000,
                         **kwargs) -> np.ndarray:
        if self.posterior is None:
            raise RuntimeError("the baseline has to be trained first")
        x_obs = torch.as_tensor(np.atleast_2d(x_obs), dtype=torch.float32)
        samples = self.posterior.sample((num_samples,), x=x_obs, **kwargs)
        return np.asarray(samples.detach().cpu().numpy()).reshape(
            x_obs.shape[0], num_samples, -1)

    def sample_likelihood(self, theta: np.ndarray, num_samples: int = 5000,
                          **kwargs) -> np.ndarray:
        if self.density_estimator is None:
            raise RuntimeError("the (NLE) baseline has to be trained first")
        theta = torch.as_tensor(np.atleast_2d(theta), dtype=torch.float32)
        if self.likelihood is not None:          # legacy sbi API
            samples = self.likelihood.sample((num_samples,), condition=theta,
                                             **kwargs)
        else:
            # in recent versions of sbi the trained conditional density
            # estimator is itself the likelihood estimator
            samples = self.density_estimator.sample((num_samples,),
                                                    condition=theta, **kwargs)
        samples = np.asarray(samples.detach().cpu().numpy())
        if samples.shape[0] == num_samples and samples.ndim == 3:
            samples = np.transpose(samples, (1, 0, 2))
        return samples.reshape(theta.shape[0], num_samples, -1)
