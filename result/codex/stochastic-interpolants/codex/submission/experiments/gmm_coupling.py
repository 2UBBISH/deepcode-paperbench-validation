"""Figure 2 of the paper: couplings versus conditioning on two 3-mode GMMs.

The transport problem is a 2-D Gaussian mixture model with three modes
mapping onto another 3-mode Gaussian mixture model.  We train the three
velocity models of Figure 2:

1. ``coupled``      data-dependent coupling rho(x_0, x_1) = rho_1(x_1) rho_0(x_0 | x_1)
                    with x_0 = A x_1 + sigma zeta and an *unconditional* velocity;
                    all samples follow simple trajectories and no auxiliary
                    modes form in the intermediate density rho_t.
2. ``conditioned``  independent coupling inside each class and a velocity
                    conditioned on the class label, b_t(x, xi); the transport
                    factorises into three separate probability flows X_t^xi.
3. ``independent``  independent coupling rho_0(x_0) rho_1(x_1) with an
                    unconditional velocity; the base marginal is the same as in
                    the coupled case, which isolates the effect of the coupling.

For every setting we report

* the analytic transport-cost bound int_0^1 E|I_dot_t|^2 dt (Proposition 3.1),
* the empirical transport cost E|X_{t=1}(x_0) - x_0|^2 of the learned ODE,
* the number of modes of the intermediate marginal rho_{t=1/2}, estimated as
  the number of local maxima of a kernel density estimate.  The uncoupled,
  unconditional transport develops auxiliary modes (up to 3 x 3 = 9 of them),
  while the coupled one keeps three.

Usage:  python experiments/gmm_coupling.py --steps 6000 --out results
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from si_couplings.interpolants import build_interpolant  # noqa: E402
from si_couplings.losses import flatten_sum, velocity_loss  # noqa: E402
from si_couplings.models.toy import MLPVelocity  # noqa: E402
from si_couplings.solvers import odeint  # noqa: E402
from si_couplings.utils import set_seed  # noqa: E402


# ---------------------------------------------------------------------------
# the two 3-mode Gaussian mixtures
# ---------------------------------------------------------------------------
class GaussianMixture:
    def __init__(self, centres: Tensor, std: float):
        self.centres = centres
        self.std = float(std)
        self.n_modes = centres.shape[0]

    def sample(self, n: int, generator=None) -> tuple[Tensor, Tensor]:
        idx = torch.randint(0, self.n_modes, (n,), generator=generator)
        mean = self.centres[idx]
        x = mean + self.std * torch.randn(n, 2, generator=generator)
        return x, idx

    @staticmethod
    def triangle(radius: float, std: float, angle_offset: float = 0.0) -> "GaussianMixture":
        angles = torch.tensor(
            [angle_offset + k * 2 * math.pi / 3 for k in range(3)], dtype=torch.float32
        )
        centres = torch.stack([radius * torch.cos(angles), radius * torch.sin(angles)], dim=-1)
        return GaussianMixture(centres, std)


def rotation(theta: float) -> Tensor:
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[c, -s], [s, c]])


# ---------------------------------------------------------------------------
def make_dataset(n: int, target: GaussianMixture, A: Tensor, sigma_base: float, mode: str, generator=None):
    """Return (x_0, x_1, labels) for one of the three couplings."""
    x1, labels = target.sample(n, generator=generator)
    if mode == "coupled":
        x0 = x1 @ A.T + sigma_base * torch.randn(n, 2, generator=generator)
    elif mode == "conditioned":
        # within-class independent coupling: the base is a Gaussian centred on
        # the image of the target mode, but drawn independently of x_1
        centres = target.centres @ A.T
        x0 = centres[labels] + sigma_base * torch.randn(n, 2, generator=generator)
    elif mode == "independent":
        # same base marginal as the coupled case, but x_0 _||_ x_1
        perm = torch.randperm(n, generator=generator)
        x1_other = x1[perm]
        x0 = x1_other @ A.T + sigma_base * torch.randn(n, 2, generator=generator)
    else:
        raise ValueError(mode)
    return x0, x1, labels


def train_velocity(
    mode: str,
    target: GaussianMixture,
    A: Tensor,
    sigma_base: float,
    *,
    steps: int = 6000,
    batch: int = 512,
    lr: float = 2e-3,
    schedule_name: str = "linear_zero_gamma",
    seed: int = 0,
) -> MLPVelocity:
    set_seed(seed)
    schedule = build_interpolant(schedule_name)
    conditional = mode == "conditioned"
    model = MLPVelocity(dim=2, hidden=128, depth=3, time_dim=32,
                        num_classes=target.n_modes if conditional else 0)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(steps // 3, 1), gamma=0.5)
    g = torch.Generator().manual_seed(seed)
    for step in range(steps):
        x0, x1, labels = make_dataset(batch, target, A, sigma_base, mode, generator=g)
        from si_couplings.couplings import CoupledBatch

        b = CoupledBatch(x0=x0, x1=x1, labels=labels if conditional else None)
        loss, _ = velocity_loss(model, schedule, b, mse_form=True)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    model.eval()
    return model


# ---------------------------------------------------------------------------
def analytic_transport_bound(mode: str, target: GaussianMixture, A: Tensor, sigma_base: float,
                             n: int = 100_000, schedule_name: str = "linear_zero_gamma",
                             seed: int = 1) -> float:
    """Monte-Carlo estimate of int_0^1 E|I_dot_t|^2 dt for the true interpolant."""
    set_seed(seed)
    schedule = build_interpolant(schedule_name)
    g = torch.Generator().manual_seed(seed)
    x0, x1, _ = make_dataset(n, target, A, sigma_base, mode, generator=g)
    ts = torch.linspace(0.0, 1.0, 51)
    vals = []
    for t in ts:
        tb = t.expand(n)
        i_dot = schedule.interpolate_velocity(tb, x0, x1)
        vals.append(flatten_sum(i_dot**2).mean())
    return float(torch.trapz(torch.stack(vals), ts))


@torch.no_grad()
def empirical_transport_cost(model, mode: str, target, A, sigma_base, n: int = 20_000,
                             steps: int = 200, seed: int = 2) -> float:
    set_seed(seed)
    g = torch.Generator().manual_seed(seed)
    x0, _, labels = make_dataset(n, target, A, sigma_base, mode, generator=g)

    def vf(x: Tensor, t: Tensor) -> Tensor:
        return model(x, t, None, labels if mode == "conditioned" else None)

    x1_hat = odeint(vf, x0, method="dopri5", atol=1e-6, rtol=1e-6)
    return float(((x1_hat - x0) ** 2).sum(dim=-1).mean())


@torch.no_grad()
def count_intermediate_modes(interpolant_samples: np.ndarray, grid: int = 128, rel_threshold: float = 0.25) -> int:
    """Number of local maxima of a KDE of the intermediate marginal."""
    from scipy.ndimage import maximum_filter
    from scipy.stats import gaussian_kde

    kde = gaussian_kde(interpolant_samples.T)
    lo = interpolant_samples.min(axis=0) - 0.2
    hi = interpolant_samples.max(axis=0) + 0.2
    xs = np.linspace(lo[0], hi[0], grid)
    ys = np.linspace(lo[1], hi[1], grid)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    Z = kde(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
    peaks = (Z == maximum_filter(Z, size=9)) & (Z > rel_threshold * Z.max())
    # merge peaks that are closer than the KDE bandwidth (5 grid cells)
    coords = np.argwhere(peaks)
    kept: list[np.ndarray] = []
    for c in coords:
        if all(np.linalg.norm(c - k) > 5 for k in kept):
            kept.append(c)
    return len(kept)


def intermediate_samples(mode: str, target, A, sigma_base, n: int = 20_000, t: float = 0.5,
                         seed: int = 3) -> np.ndarray:
    set_seed(seed)
    g = torch.Generator().manual_seed(seed)
    x0, x1, _ = make_dataset(n, target, A, sigma_base, mode, generator=g)
    schedule = build_interpolant("linear_zero_gamma")
    tb = torch.full((n,), t)
    return schedule.interpolate(tb, x0, x1).numpy()


# ---------------------------------------------------------------------------
def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--n-modes-sample", type=int, default=20_000)
    parser.add_argument("--out", default="results")
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    target = GaussianMixture.triangle(radius=1.0, std=0.12)
    A = rotation(math.pi / 3) * 1.2  # a rotation+scale between base and target
    sigma_base = 0.15

    metrics = {}
    models = {}
    for mode in ["coupled", "conditioned", "independent"]:
        model = train_velocity(mode, target, A, sigma_base, steps=args.steps, seed=0)
        models[mode] = model
        bound = analytic_transport_bound(mode, target, A, sigma_base)
        cost = empirical_transport_cost(model, mode, target, A, sigma_base)
        samples = intermediate_samples(mode, target, A, sigma_base, n=args.n_modes_sample)
        n_modes = count_intermediate_modes(samples)
        metrics[mode] = {
            "transport_cost_bound": bound,
            "empirical_transport_cost": cost,
            "intermediate_modes_at_t_half": n_modes,
        }
        print(f"{mode:12s} bound={bound:8.4f}  cost={cost:8.4f}  modes(t=1/2)={n_modes}")

    (out / "gmm_metrics.json").write_text(json.dumps(metrics, indent=2))

    if not args.no_figure:
        make_figure(models, target, A, sigma_base, out / "gmm_figure2.png")


def make_figure(models, target, A, sigma_base, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["coupled", "conditioned", "independent"]
    titles = [
        r"data-dependent coupling $\rho_1(x_1)\rho_0(x_0|x_1)$" "\n" "unconditional $b_t(x)$",
        r"conditioned $b_t(x,\xi)$ on each mode" "\n" "(independent within class)",
        r"independent coupling $\rho_0(x_0)\rho_1(x_1)$" "\n" "unconditional $b_t(x)$",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for j, (name, title) in enumerate(zip(names, titles)):
        model = models[name]
        with torch.no_grad():
            x0, x1, labels = make_dataset(400, target, A, sigma_base, name,
                                          generator=torch.Generator().manual_seed(7))

            # forward-Euler sampling of the flow for plotting
            traj = [x0]
            x = x0
            for t in torch.linspace(0.0, 1.0, 21)[:-1]:
                tb = torch.full((x.shape[0],), float(t))
                x = x + (1 / 20) * model(x, tb, None, labels if name == "conditioned" else None)
                traj.append(x)
            traj = torch.stack(traj)
        ax = axes[0, j]
        for i in range(0, 400, 4):
            ax.plot(traj[:, i, 0], traj[:, i, 1], alpha=0.25, lw=0.7)
        ax.scatter(traj[0, :, 0], traj[0, :, 1], s=2, c="tab:blue", label=r"$x_0\sim\rho_0$")
        ax.scatter(traj[-1, :, 0], traj[-1, :, 1], s=2, c="tab:red", label=r"$X_{t=1}$")
        ax.scatter(target.centres[:, 0], target.centres[:, 1], marker="x", c="k", label=r"$\rho_1$ modes")
        ax.set_title(title)
        ax.legend(fontsize=7, loc="upper right")
        ax.set_aspect("equal")

        # intermediate marginal: true interpolant vs the model flow
        ax = axes[1, j]
        for t_plot, colour in [(0.0, "tab:blue"), (0.5, "tab:green"), (1.0, "tab:red")]:
            tb = torch.full((x0.shape[0],), t_plot)
            pts = build_interpolant("linear_zero_gamma").interpolate(tb, x0, x1)
            ax.scatter(pts[:, 0], pts[:, 1], s=1, alpha=0.15, c=colour,
                       label=rf"$\rho_t$, $t={t_plot}$")
        ax.scatter(target.centres[:, 0], target.centres[:, 1], marker="x", c="k")
        ax.set_title("intermediate marginal $\\rho_t$")
        ax.set_aspect("equal")
        ax.legend(fontsize=7, loc="upper right", markerscale=4)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
