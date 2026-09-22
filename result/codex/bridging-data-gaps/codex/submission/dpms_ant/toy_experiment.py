"""Visualisation on toy data (Section 5.1 of the paper, Figure 2).

Setup (Section 5.1):

    "we train a diffusion model to generate 2-dimensional toy data with two
    Gaussian noise distributions.  The means of the Gaussian noise
    distributions for the source and target are (1,1) and (-1,-1), and their
    variances are denoted by I.  We train a simple neural network with source
    domain samples and then transfer this pre-trained model to target samples."

Reproduced here:

* **Figure 2(a)** -- the gradient direction of the *output layer* in the first
  iteration for four settings, all with the same noise and timestep ``t``: the
  reference gradient computed on 10,000 samples (cyan), the 10-shot baseline
  (blue), similarity-guided training only (red) and the full method (orange).
  For the 10-shot settings the 10 target samples are repeated 1000 times so
  that the batch matches the 10,000-sample reference.  The "worse-case" noises
  of Eq. (7) are drawn as red points: their cloud changes from a circle to an
  ellipse.  ``gradient_direction_study`` returns the angle error of every
  setting w.r.t. the reference direction.
* **Figure 2(b)/(c)** -- heat maps of (timestep, sampled value) for the
  generated samples, for the baseline and for our method; the cyan/yellow lines
  are the sampling trajectories of the original DDPM and of our method.
* the convergence comparison (Section 5.1: "our method can learn the
  distribution more quickly than the baseline method") measured with the energy
  distance between the generated and the target distributions.

Everything here runs on CPU in a couple of minutes.  The dimension of the toy
data is configurable (``ToyConfig.data_dim``); the paper's setup is 2-D.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .adversarial_noise import AdversarialNoiseConfig, select_adversarial_noise
from .classifier import ToyDomainClassifier, train_toy_classifier
from .guidance import classifier_guidance, similarity_guided_loss
from .schedules import DiffusionSchedule
from .utils import Logger, ensure_dir, progbar


@dataclass
class ToyConfig:
    """Configuration of the Section 5.1 experiment.

    The paper reports means ``(1,1)`` / ``(-1,-1)``, variance ``I``, a "simple
    neural network" and 10-shot transfer.  The remaining knobs (network width,
    number of source iterations, learning rates, the timestep at which the
    gradient study is run and the guidance strength) are not specified in the
    paper and are set here to values that reproduce its qualitative findings.
    """

    num_timesteps: int = 1000
    data_dim: int = 2
    mean_offset: float = 1.0
    num_target_samples: int = 10
    hidden: int = 128
    depth: int = 3
    source_iterations: int = 10000
    source_batch_size: int = 256
    source_lr: float = 1e-3
    transfer_iterations: int = 300
    transfer_lr: float = 1e-4
    gamma: float = 5.0
    # In 2-D the per-element classifier gradient is orders of magnitude larger
    # than for a 256x256 image, so the guidance strength is expressed relative
    # to the magnitude of the noise instead of using the image value of gamma
    # (the paper does not report hyper-parameters for the toy study).
    guidance_rms: float = 0.5
    # Figure 2(a) analyses "the first iteration" at "the same noise and
    # timestep t"; the paper does not state t.  0.2 of the trajectory is used,
    # which is where the paper's ordering (our gradient closest to the
    # reference) is observed for the 2-D setup.
    gradient_timestep_fraction: float = 0.2
    adversary: AdversarialNoiseConfig = field(
        default_factory=lambda: AdversarialNoiseConfig(
            num_steps=10, step_size=0.2, normalization="global"
        )
    )
    classifier_iterations: int = 300
    classifier_batch_size: int = 64
    classifier_lr: float = 1e-4
    reference_batch_size: int = 10000
    repeats: int = 1000
    num_samples_for_heatmap: int = 20000
    eval_every: int = 10
    seed: int = 0


def mean_vector(value: float, dim: int) -> torch.Tensor:
    return torch.full((dim,), float(value))


def source_mean(config: ToyConfig) -> torch.Tensor:
    return mean_vector(config.mean_offset, config.data_dim)


def target_mean(config: ToyConfig) -> torch.Tensor:
    return mean_vector(-config.mean_offset, config.data_dim)


def projection_direction(config: ToyConfig) -> torch.Tensor:
    """Unit vector along the source->target direction (used for 1-D readouts)."""
    return mean_vector(1.0, config.data_dim) / math.sqrt(config.data_dim)


class ToyDenoiser(nn.Module):
    """Small MLP ``eps_theta(x_t, t)`` (the paper's "simple neural network").

    The final linear layer is exposed as ``output_layer`` because Figure 2(a)
    analyses "the output layer gradient direction".
    """

    def __init__(self, hidden: int = 128, depth: int = 3, time_dim: int = 16, data_dim: int = 2):
        super().__init__()
        self.time_dim = time_dim
        self.data_dim = data_dim
        layers: List[nn.Module] = []
        in_features = data_dim + time_dim
        for _ in range(depth):
            layers += [nn.Linear(in_features, hidden), nn.SiLU()]
            in_features = hidden
        self.trunk = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden, data_dim)

    def timestep_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.time_dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, dtype=torch.float32, device=timesteps.device)
            / half
        )
        args = timesteps.float().reshape(-1, 1) * freqs.reshape(1, -1)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.time_dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        embedding = self.timestep_embedding(timesteps).to(x.dtype)
        return self.output_layer(self.trunk(torch.cat([x, embedding], dim=-1)))


# ---------------------------------------------------------------------- #
# training
# ---------------------------------------------------------------------- #
def train_source_model(
    schedule: DiffusionSchedule,
    config: ToyConfig,
    device: str = "cpu",
    logger: Optional[Logger] = None,
) -> ToyDenoiser:
    """Pre-train the toy diffusion model on the source distribution."""
    torch.manual_seed(config.seed)
    model = ToyDenoiser(config.hidden, config.depth, data_dim=config.data_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.source_lr)
    generator = torch.Generator(device=device).manual_seed(config.seed)
    mean = source_mean(config).to(device)
    model.train()
    for step in progbar(range(config.source_iterations), desc="toy source"):
        x0 = mean + torch.randn(
            config.source_batch_size, config.data_dim, device=device, generator=generator
        )
        t = torch.randint(
            1,
            schedule.num_timesteps,
            (config.source_batch_size,),
            device=device,
            generator=generator,
        ).long()
        noise = torch.randn(x0.shape, device=device, generator=generator)
        loss = similarity_guided_loss(model(schedule.q_sample(x0, t, noise), t), noise)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if logger is not None and step % max(1, config.source_iterations // 5) == 0:
            logger.log(f"[toy] source pre-training step {step:6d} loss {float(loss):.4f}")
    model.eval()
    return model


def transfer_toy_model(
    model: ToyDenoiser,
    schedule: DiffusionSchedule,
    target_samples: torch.Tensor,
    classifier: Optional[ToyDomainClassifier],
    config: ToyConfig,
    use_similarity_guidance: bool = True,
    use_adversarial_noise: bool = True,
    device: str = "cpu",
    logger: Optional[Logger] = None,
) -> Tuple[ToyDenoiser, Dict[str, List[float]]]:
    """Algorithm 1 on the toy task (a 10-shot transfer run)."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.transfer_lr)
    generator = torch.Generator(device=device).manual_seed(config.seed + 7)
    target = target_samples.to(device)
    history: Dict[str, List[float]] = {"loss": [], "value": [], "distance": [], "step": []}
    projection = projection_direction(config).to(device)

    for step in progbar(range(config.transfer_iterations + 1), desc="toy transfer"):
        index = torch.randint(0, target.shape[0], (config.num_target_samples,), generator=generator)
        x0 = target[index]
        t = torch.randint(
            1, schedule.num_timesteps, (x0.shape[0],), device=device, generator=generator
        ).long()
        noise = torch.randn(x0.shape, device=device, generator=generator)
        if use_adversarial_noise:
            eps_star = select_adversarial_noise(
                model, schedule, x0, t, config=config.adversary, initial_noise=noise
            )
        else:
            eps_star = noise
        x_t = schedule.q_sample(x0, t, eps_star)
        guidance = None
        if use_similarity_guidance and classifier is not None:
            guidance = classifier_guidance(
                classifier,
                schedule,
                x_t,
                t,
                target_class=1,
                gamma=config.gamma,
                target_rms=config.guidance_rms,
            )
        loss = similarity_guided_loss(model(x_t, t), eps_star, guidance=guidance)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history["loss"].append(float(loss.detach()))
        history["step"].append(step)

        if step % config.eval_every == 0 or step == config.transfer_iterations:
            generated = sample(model, schedule, 256, device=device, seed=config.seed)
            history["value"].append(float((generated @ projection).mean()))
            history["distance"].append(energy_distance(generated, target.detach()))
            if logger is not None:
                logger.log(
                    f"[toy] transfer step {step:4d} loss {float(loss):.4f} "
                    f"value {history['value'][-1]:+.3f} distance {history['distance'][-1]:.4f}"
                )
    return model, history


# ---------------------------------------------------------------------- #
# Figure 2(a): gradient direction of the output layer
# ---------------------------------------------------------------------- #
def _flatten_gradient(grads, parameters) -> torch.Tensor:
    pieces = []
    for grad, parameter in zip(grads, parameters):
        pieces.append(grad.reshape(-1) if grad is not None else torch.zeros_like(parameter).reshape(-1))
    return torch.cat(pieces).detach()


def output_layer_gradient(
    model: ToyDenoiser,
    schedule: DiffusionSchedule,
    x_start: torch.Tensor,
    timesteps: torch.Tensor,
    noise: torch.Tensor,
    classifier: Optional[ToyDomainClassifier] = None,
    gamma: float = 5.0,
    target_rms: Optional[float] = None,
    use_similarity_guidance: bool = False,
    use_adversarial_noise: bool = False,
    adversary: Optional[AdversarialNoiseConfig] = None,
    device: str = "cpu",
) -> torch.Tensor:
    """Gradient of Eq. (5) / (1) w.r.t. ``model.output_layer`` parameters."""
    model = model.to(device).eval()
    x_start = x_start.to(device)
    timesteps = timesteps.to(device)
    noise = noise.to(device)
    if use_adversarial_noise:
        noise = select_adversarial_noise(
            model, schedule, x_start, timesteps, config=adversary, initial_noise=noise
        )
    x_t = schedule.q_sample(x_start, timesteps, noise)
    guidance = None
    if use_similarity_guidance and classifier is not None:
        guidance = classifier_guidance(
            classifier,
            schedule,
            x_t,
            timesteps,
            target_class=1,
            gamma=gamma,
            target_rms=target_rms,
        )
    eps_pred = model(x_t, timesteps)
    loss = similarity_guided_loss(eps_pred, noise, guidance=guidance, reduction="sum")
    parameters = list(model.output_layer.parameters())
    grads = torch.autograd.grad(loss, parameters, allow_unused=True)
    return _flatten_gradient(grads, parameters)


def epsilon_space_gradient(
    model: ToyDenoiser,
    schedule: DiffusionSchedule,
    x_start: torch.Tensor,
    timesteps: torch.Tensor,
    noise: torch.Tensor,
    classifier: Optional[ToyDomainClassifier] = None,
    gamma: float = 5.0,
    target_rms: Optional[float] = None,
    use_similarity_guidance: bool = True,
    device: str = "cpu",
) -> torch.Tensor:
    """Mean gradient of the loss w.r.t. the noise ``eps`` (the direction the
    adversarial-noise ascent of Eq. (7) follows)."""
    model = model.to(device).eval()
    x_start = x_start.to(device)
    timesteps = timesteps.to(device)
    noise = noise.to(device).clone().requires_grad_(True)
    x_t = schedule.q_sample(x_start, timesteps, noise)
    guidance = None
    if use_similarity_guidance and classifier is not None:
        guidance = classifier_guidance(
            classifier,
            schedule,
            x_t,
            timesteps,
            target_class=1,
            gamma=gamma,
            target_rms=target_rms,
        )
    loss = similarity_guided_loss(model(x_t, timesteps), noise, guidance=guidance, reduction="sum")
    (grad,) = torch.autograd.grad(loss, noise)
    return grad.mean(dim=0).detach()


def gradient_direction_study(
    model: ToyDenoiser,
    schedule: DiffusionSchedule,
    classifier: ToyDomainClassifier,
    config: ToyConfig,
    device: str = "cpu",
) -> Dict[str, object]:
    """Reproduce Figure 2(a): reference vs baseline / SG / ANT gradient directions."""
    torch.manual_seed(config.seed + 11)
    dim = config.data_dim
    target = target_mean(config)
    reference_samples = target + torch.randn(config.reference_batch_size, dim)
    shot_samples = target + torch.randn(config.num_target_samples, dim)
    repeated = shot_samples.repeat(config.repeats, 1)

    # "with the same noise and timestep t" for every setting
    timestep_value = max(1, int(config.gradient_timestep_fraction * schedule.num_timesteps))
    timestep = torch.full((config.reference_batch_size,), timestep_value, dtype=torch.long)
    noise = torch.randn(config.reference_batch_size, dim)

    directions: Dict[str, torch.Tensor] = {}
    directions["reference"] = output_layer_gradient(
        model, schedule, reference_samples, timestep, noise, device=device
    )
    directions["baseline"] = output_layer_gradient(
        model, schedule, repeated, timestep, noise, device=device
    )
    directions["sg"] = output_layer_gradient(
        model,
        schedule,
        repeated,
        timestep,
        noise,
        classifier=classifier,
        gamma=config.gamma,
        target_rms=config.guidance_rms,
        use_similarity_guidance=True,
        device=device,
    )
    directions["ant"] = output_layer_gradient(
        model,
        schedule,
        repeated,
        timestep,
        noise,
        classifier=classifier,
        gamma=config.gamma,
        target_rms=config.guidance_rms,
        use_similarity_guidance=True,
        use_adversarial_noise=True,
        adversary=config.adversary,
        device=device,
    )

    reference = directions["reference"]
    angles: Dict[str, float] = {}
    cosines: Dict[str, float] = {}
    for name, vector in directions.items():
        cosine = torch.nn.functional.cosine_similarity(
            vector.reshape(1, -1), reference.reshape(1, -1)
        ).item()
        cosines[name] = float(cosine)
        angles[name] = float(math.degrees(math.acos(max(-1.0, min(1.0, cosine)))))

    # the "worse-case" noise cloud of Figure 2(a)
    torch.manual_seed(config.seed + 12)
    cloud_start = target + torch.randn(2000, dim)
    cloud_t = torch.full((2000,), timestep_value, dtype=torch.long)
    cloud_noise = torch.randn(2000, dim)
    gaussian_cloud = torch.randn(2000, dim)
    worse_case = select_adversarial_noise(
        model,
        schedule,
        cloud_start,
        cloud_t,
        config=AdversarialNoiseConfig(
            num_steps=config.adversary.num_steps,
            step_size=config.adversary.step_size,
            normalization=config.adversary.normalization,
        ),
        initial_noise=cloud_noise,
    )
    cloud = worse_case.detach().cpu().numpy()
    covariance = np.cov(cloud.T).reshape(dim, dim)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    principal_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    eps_direction = epsilon_space_gradient(
        model,
        schedule,
        repeated,
        timestep,
        noise,
        classifier=classifier,
        gamma=config.gamma,
        target_rms=config.guidance_rms,
        use_similarity_guidance=True,
        device=device,
    ).numpy()
    eps_direction = eps_direction / max(np.linalg.norm(eps_direction), 1e-12)

    return {
        "timestep": timestep_value,
        "directions": {key: value.detach().cpu().numpy() for key, value in directions.items()},
        "angle_error_deg": angles,
        "cosine_similarity": cosines,
        "worse_case_noise": cloud,
        "gaussian_noise": gaussian_cloud.numpy(),
        "noise_covariance_eigenvalues": eigenvalues.tolist(),
        "noise_anisotropy": float(eigenvalues.max() / max(eigenvalues.min(), 1e-12)),
        "noise_principal_axis": principal_axis.tolist(),
        "gradient_axis_alignment": float(abs(np.dot(principal_axis, eps_direction))),
    }


# ---------------------------------------------------------------------- #
# sampling / Figure 2(b),(c)
# ---------------------------------------------------------------------- #
def _reverse_step(
    schedule: DiffusionSchedule,
    x_t: torch.Tensor,
    timesteps: torch.Tensor,
    eps: torch.Tensor,
    clip_denoised: bool = False,
) -> torch.Tensor:
    """One DDPM reverse step.

    ``clip_denoised`` is False here: the toy data are Gaussians centred at
    ``(+-1, ..., +-1)`` and are *not* confined to ``[-1, 1]``, so the usual
    image clipping would bias every result towards the origin.
    """
    from .sampling import ddpm_step

    return ddpm_step(schedule, x_t, timesteps, eps, eta=1.0, clip_denoised=clip_denoised)


def sample(
    model: ToyDenoiser,
    schedule: DiffusionSchedule,
    num_samples: int,
    device: str = "cpu",
    seed: int = 0,
) -> torch.Tensor:
    """Generate samples with the DDPM reverse process (Section 3)."""
    model = model.to(device).eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(num_samples, model.data_dim, device=device, generator=generator)
    with torch.no_grad():
        for t_value in range(schedule.num_timesteps - 1, -1, -1):
            t = torch.full((num_samples,), t_value, device=device, dtype=torch.long)
            eps = model(x, t)
            if t_value == 0:
                x = schedule.predict_x0_from_eps(x, t, eps)
            else:
                x = _reverse_step(schedule, x, t, eps)
    return x


def sample_mean_value(
    model: ToyDenoiser,
    schedule: DiffusionSchedule,
    num_samples: int,
    config: ToyConfig,
    device: str = "cpu",
    seed: int = 0,
) -> float:
    """Mean generated value along the source->target direction."""
    x = sample(model, schedule, num_samples, device=device, seed=seed)
    projection = projection_direction(config).to(device)
    return float((x @ projection).mean())


def sample_trajectories(
    model: ToyDenoiser,
    schedule: DiffusionSchedule,
    config: ToyConfig,
    num_samples: int,
    device: str = "cpu",
    seed: int = 0,
    time_bins: int = 50,
    value_bins: int = 60,
) -> Dict[str, object]:
    """Reverse-process histogram over (timestep, generated value).

    The "value" is the projection of each sample on the source->target
    direction, i.e. the "sampled values produced by the generative model" of
    Figures 2(b)/(c).
    """
    model = model.to(device).eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    dim = config.data_dim
    x = torch.randn(num_samples, dim, device=device, generator=generator)
    single_x = x[:1].clone()
    projection = projection_direction(config).to(device)
    limit = math.sqrt(dim) + 2.5
    value_range = (-limit, limit)
    steps = list(range(schedule.num_timesteps - 1, -1, -1))
    stride = max(1, schedule.num_timesteps // time_bins)
    histogram = np.zeros((time_bins, value_bins), dtype=np.float64)
    single: List[float] = []

    with torch.no_grad():
        for index, t_value in enumerate(progbar(steps, desc="toy sampling")):
            t = torch.full((num_samples,), t_value, device=device, dtype=torch.long)
            eps = model(x, t)
            if index == len(steps) - 1:
                x = schedule.predict_x0_from_eps(x, t, eps)
                break
            if t_value % stride == 0:
                values = (x @ projection).detach().cpu().numpy()
                counts, _ = np.histogram(values, bins=value_bins, range=value_range)
                histogram[min(time_bins - 1, t_value // stride)] += counts
            x = _reverse_step(schedule, x, t, eps)
            t_single = torch.full((1,), t_value, device=device, dtype=torch.long)
            single_x = _reverse_step(schedule, single_x, t_single, model(single_x, t_single))
            single.append(float(single_x @ projection))

    values = (x @ projection).detach().cpu()
    return {
        "histogram": histogram,
        "single_trajectory": np.asarray(single),
        "final_values": values.numpy(),
        "final_mean": float(values.mean()),
        "final_std": float(values.std()),
        "value_range": value_range,
        "time_bins": time_bins,
        "value_bins": value_bins,
    }


def energy_distance(a: torch.Tensor, b: torch.Tensor, max_samples: int = 1024) -> float:
    """Energy distance between two sample sets (0 = identical distributions).

    A distribution-level metric: unlike the mean it also penalises the
    diversity collapse caused by overfitting the 10 target samples.
    """
    a = a.detach().float()
    b = b.detach().float()
    if a.shape[0] > max_samples:
        a = a[:max_samples]
    if b.shape[0] > max_samples:
        b = b[:max_samples]
    d_ab = torch.cdist(a, b).mean()
    d_aa = torch.cdist(a, a).mean()
    d_bb = torch.cdist(b, b).mean()
    return float(2 * d_ab - d_aa - d_bb)


def gap_closed(value: float, config: ToyConfig) -> float:
    """Fraction of the source->target gap covered by the generated distribution."""
    source = config.mean_offset * math.sqrt(config.data_dim)
    target = -source
    return float((source - value) / (source - target))


# ---------------------------------------------------------------------- #
# plots
# ---------------------------------------------------------------------- #
def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def plot_gradient_directions(study: Dict[str, object], path: str) -> Optional[str]:
    plt = _pyplot()
    if plt is None:
        return None
    colors = {"reference": "cyan", "baseline": "blue", "sg": "red", "ant": "orange"}
    labels = {
        "reference": "10,000 samples (reference)",
        "baseline": "baseline (10-shot)",
        "sg": "DPMs-ANT w/o AN (10-shot)",
        "ant": "DPMs-ANT (10-shot)",
    }
    figure, axes = plt.subplots(figsize=(5.5, 5.5), dpi=150)
    cloud = study["worse_case_noise"]
    if cloud.shape[1] >= 2:
        axes.scatter(
            cloud[:, 0], cloud[:, 1], s=2, c="red", alpha=0.25, label="'worse-case' noise"
        )
    for name in ("reference", "baseline", "sg", "ant"):
        vector = study["directions"][name][:2]
        vector = vector / max(np.linalg.norm(vector), 1e-12)
        axes.arrow(
            0,
            0,
            float(vector[0]),
            float(vector[1]),
            head_width=0.05,
            length_includes_head=True,
            color=colors[name],
            label=labels[name],
        )
    axes.set_aspect("equal")
    axes.set_xlim(-1.6, 1.6)
    axes.set_ylim(-1.6, 1.6)
    axes.legend(loc="upper right", fontsize=7)
    axes.set_title(
        "Figure 2(a): output-layer gradient directions (t = %d)" % study["timestep"]
    )
    figure.tight_layout()
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    figure.savefig(path)
    plt.close(figure)
    return path


def plot_heatmap(
    result: Dict[str, object],
    trajectory: np.ndarray,
    reference: Optional[np.ndarray],
    path: str,
    title: str,
) -> Optional[str]:
    plt = _pyplot()
    if plt is None:
        return None
    histogram = result["histogram"]
    value_range = result["value_range"]
    figure, axes = plt.subplots(figsize=(6, 4), dpi=150)
    axes.imshow(
        histogram.T,
        origin="lower",
        aspect="auto",
        extent=(0, result["time_bins"], value_range[0], value_range[1]),
        cmap="Blues",
    )
    if reference is not None:
        axes.plot(
            np.linspace(0, result["time_bins"], len(reference)),
            reference,
            color="cyan",
            lw=1.5,
            label="original DDPM",
        )
    axes.plot(
        np.linspace(0, result["time_bins"], len(trajectory)),
        trajectory,
        color="gold",
        lw=1.5,
        label="our method",
    )
    axes.set_xlabel("time-step of the diffusion process")
    axes.set_ylabel("sampled values")
    axes.set_title(title)
    axes.legend(loc="lower right", fontsize=8)
    figure.tight_layout()
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    figure.savefig(path)
    plt.close(figure)
    return path


def plot_convergence(curves: Dict[str, np.ndarray], config: ToyConfig, path: str) -> Optional[str]:
    plt = _pyplot()
    if plt is None:
        return None
    scale = config.mean_offset * math.sqrt(config.data_dim)
    figure, axes = plt.subplots(figsize=(5.5, 3.5), dpi=150)
    for name, curve in curves.items():
        axes.plot(curve, label=name)
    axes.axhline(-scale, ls="--", c="k", lw=1, label="target mean")
    axes.axhline(scale, ls=":", c="k", lw=1, label="source mean")
    axes.set_xlabel(f"transfer iteration (x{config.eval_every})")
    axes.set_ylabel("generated sample value")
    axes.set_title("Convergence of the generated distribution")
    axes.legend(fontsize=8)
    figure.tight_layout()
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    figure.savefig(path)
    plt.close(figure)
    return path


def plot_distances(curves: Dict[str, np.ndarray], source_distance: float, path: str) -> Optional[str]:
    """Energy distance to the target samples over transfer iterations."""
    plt = _pyplot()
    if plt is None:
        return None
    figure, axes = plt.subplots(figsize=(5.5, 3.5), dpi=150)
    for name, curve in curves.items():
        axes.plot(curve, label=name)
    axes.axhline(source_distance, ls=":", c="k", lw=1, label="source model")
    axes.set_xlabel("transfer iteration (x10)")
    axes.set_ylabel("energy distance to the target")
    axes.set_title("How quickly the generated distribution reaches the target")
    axes.legend(fontsize=8)
    figure.tight_layout()
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    figure.savefig(path)
    plt.close(figure)
    return path


# ---------------------------------------------------------------------- #
# orchestration
# ---------------------------------------------------------------------- #
def run_toy_experiment(
    config: Optional[ToyConfig] = None,
    device: str = "cpu",
    output_dir: str = "outputs/toy",
    logger: Optional[Logger] = None,
    quick: bool = False,
) -> Dict[str, object]:
    """Full Section 5.1 reproduction (all three panels of Figure 2)."""
    config = config or ToyConfig()
    if quick:
        config.source_iterations = max(300, config.source_iterations // 10)
        config.transfer_iterations = min(100, config.transfer_iterations)
        config.reference_batch_size = min(2000, config.reference_batch_size)
        config.repeats = max(1, config.reference_batch_size // config.num_target_samples)
        config.num_samples_for_heatmap = min(2000, config.num_samples_for_heatmap)
    schedule = DiffusionSchedule(num_timesteps=config.num_timesteps).to(device)
    ensure_dir(output_dir)

    if logger:
        logger.log("[toy] pre-training the source model")
    source_model = train_source_model(schedule, config, device=device, logger=logger)

    torch.manual_seed(config.seed + 3)
    source_samples = source_mean(config) + torch.randn(2000, config.data_dim)
    target_samples = target_mean(config) + torch.randn(config.num_target_samples, config.data_dim)
    classifier = ToyDomainClassifier(
        hidden=config.hidden, depth=config.depth, data_dim=config.data_dim
    ).to(device)
    classifier = train_toy_classifier(
        classifier,
        schedule,
        source_samples,
        target_samples,
        batch_size=config.classifier_batch_size,
        iterations=config.classifier_iterations,
        lr=config.classifier_lr,
        device=device,
        seed=config.seed,
    )

    if logger:
        logger.log("[toy] gradient-direction study (Figure 2a)")
    study = gradient_direction_study(source_model, schedule, classifier, config, device=device)
    plot_gradient_directions(study, os.path.join(output_dir, "figure2a_gradient_directions.png"))
    if logger:
        logger.log(
            f"[toy] gradient angle errors (deg, lower is better): {study['angle_error_deg']}"
        )
        logger.log(
            f"[toy] worse-case noise anisotropy {study['noise_anisotropy']:.3f}; "
            f"|cos(principal axis, gradient direction)| = {study['gradient_axis_alignment']:.3f}"
        )

    methods = {
        "baseline": dict(use_similarity_guidance=False, use_adversarial_noise=False),
        "sg": dict(use_similarity_guidance=True, use_adversarial_noise=False),
        "ant": dict(use_similarity_guidance=True, use_adversarial_noise=True),
    }
    transferred: Dict[str, ToyDenoiser] = {}
    curves: Dict[str, np.ndarray] = {}
    histories: Dict[str, Dict[str, List[float]]] = {}
    for name, flags in methods.items():
        if logger:
            logger.log(f"[toy] transfer run: {name}")
        torch.manual_seed(config.seed)
        model = ToyDenoiser(config.hidden, config.depth, data_dim=config.data_dim)
        model.load_state_dict(source_model.state_dict())
        model, history = transfer_toy_model(
            model, schedule, target_samples, classifier, config, device=device, logger=logger, **flags
        )
        transferred[name] = model
        histories[name] = history
        curves[name] = np.asarray(history["value"])

    source_value = sample_mean_value(source_model, schedule, 512, config, device=device, seed=0)
    curves["source (no transfer)"] = np.full_like(curves["baseline"], source_value)
    plot_convergence(curves, config, os.path.join(output_dir, "convergence.png"))

    distance_curves: Dict[str, np.ndarray] = {
        name: np.asarray(history["distance"]) for name, history in histories.items()
    }
    source_pool = sample(source_model, schedule, 1024, device=device, seed=config.seed)
    source_distance = energy_distance(source_pool, target_samples)
    plot_distances(
        distance_curves, source_distance, os.path.join(output_dir, "convergence_distance.png")
    )

    if logger:
        logger.log("[toy] sampling heat maps (Figures 2b/2c)")
    baseline_result = sample_trajectories(
        transferred["baseline"], schedule, config, config.num_samples_for_heatmap, device=device, seed=0
    )
    ant_result = sample_trajectories(
        transferred["ant"], schedule, config, config.num_samples_for_heatmap, device=device, seed=0
    )
    plot_heatmap(
        baseline_result,
        baseline_result["single_trajectory"],
        None,
        os.path.join(output_dir, "figure2b_baseline_heatmap.png"),
        "Figure 2(b): baseline",
    )
    plot_heatmap(
        ant_result,
        ant_result["single_trajectory"],
        baseline_result["single_trajectory"],
        os.path.join(output_dir, "figure2c_ant_heatmap.png"),
        "Figure 2(c): our method",
    )

    summary: Dict[str, object] = {
        "data_dim": config.data_dim,
        "gradient_timestep": study["timestep"],
        "angle_error_deg": study["angle_error_deg"],
        "cosine_similarity": study["cosine_similarity"],
        "noise_anisotropy": study["noise_anisotropy"],
        "gradient_axis_alignment": study["gradient_axis_alignment"],
        "baseline_final_mean": baseline_result["final_mean"],
        "ant_final_mean": ant_result["final_mean"],
        "baseline_final_std": baseline_result["final_std"],
        "ant_final_std": ant_result["final_std"],
        "source_projection": config.mean_offset * math.sqrt(config.data_dim),
        "target_projection": -config.mean_offset * math.sqrt(config.data_dim),
        "baseline_gap_closed": gap_closed(baseline_result["final_mean"], config),
        "ant_gap_closed": gap_closed(ant_result["final_mean"], config),
        "convergence_curves": {name: curve.tolist() for name, curve in curves.items()},
        "distance_curves": {name: curve.tolist() for name, curve in distance_curves.items()},
        "source_energy_distance": source_distance,
        "transfer_loss": {name: history["loss"] for name, history in histories.items()},
        "figures": sorted(
            [
                os.path.join(output_dir, "figure2a_gradient_directions.png"),
                os.path.join(output_dir, "figure2b_baseline_heatmap.png"),
                os.path.join(output_dir, "figure2c_ant_heatmap.png"),
                os.path.join(output_dir, "convergence.png"),
                os.path.join(output_dir, "convergence_distance.png"),
            ]
        ),
    }
    return summary
