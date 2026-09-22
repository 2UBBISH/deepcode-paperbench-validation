"""Section 3.3 / Proposition 3.1: couplings reduce the transport cost.

The script produces three quantitative checks of the theoretical claims.

A. Equation (21).  For the data-decorruption coupling x_0 = x_1 + sigma zeta
   with C = sigma^2 Id, alpha_t = 1 - t, beta_t = t and gamma_t = 0, the
   paper states

       E|I_dot_t|^2 = d sigma^2                     (coupled)
       E|I_dot_t|^2 = 2 E|x_1|^2 + d sigma^2        (independent,
                                                     same base marginal)

   which we verify by Monte Carlo for several dimensions.

B. Proposition 3.1.  E_{x_0}[|X_{t=1}(x_0) - x_0|^2] <= int_0^1 E|I_dot_t|^2 dt.
   We evaluate both sides with the *exact* velocity field of a linear-Gaussian
   coupling (where the probability-flow ODE is an affine ODE and can be
   integrated very accurately) and compare coupled and independent pairings.

C. The same comparison with a learned velocity network in 2-D.

Usage:  python experiments/transport_cost.py --out results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from si_couplings.couplings import CoupledBatch, DataDecorruptionCoupling, IndependentCoupling  # noqa: E402
from si_couplings.interpolants import GaussianCoupling, build_interpolant  # noqa: E402
from si_couplings.losses import flatten_sum, transport_cost_upper_bound, velocity_loss  # noqa: E402
from si_couplings.models.toy import MLPVelocity  # noqa: E402
from si_couplings.solvers import odeint  # noqa: E402
from si_couplings.utils import set_seed  # noqa: E402


# ---------------------------------------------------------------------------
def part_a(dims=(2, 8, 64, 256), sigma: float = 0.5, n: int = 50_000) -> dict:
    """Monte-Carlo verification of equation (21)."""
    results = {}
    for d in dims:
        x1 = torch.randn(n, d)
        coupled = DataDecorruptionCoupling(sigma=sigma)
        indep = IndependentCoupling(base_scale=(1 + sigma**2) ** 0.5)
        b_c = coupled.sample(x1)
        b_i = indep.sample(x1)
        sch = build_interpolant("linear_zero_gamma")
        bound_c = float(transport_cost_upper_bound(sch, b_c))
        bound_i = float(transport_cost_upper_bound(sch, b_i))
        results[d] = {
            "coupled_mc": bound_c,
            "coupled_theory": d * sigma**2,
            "independent_mc": bound_i,
            "independent_theory": float(2 * (x1**2).sum(-1).mean() + d * sigma**2),
        }
        print(
            f"d={d:4d}  coupled MC={bound_c:10.3f} theory={d * sigma ** 2:10.3f}"
            f"   independent MC={bound_i:10.3f} theory={results[d]['independent_theory']:10.3f}"
        )
    return results


# ---------------------------------------------------------------------------
def _gaussian_coupling_moments(d: int, sigma: float, coupled: bool) -> GaussianCoupling:
    eye = torch.eye(d)
    if coupled:  # x_0 = x_1 + sigma zeta
        return GaussianCoupling(
            mu_0=torch.zeros(d), mu_1=torch.zeros(d),
            C_00=(1 + sigma**2) * eye, C_11=eye, C_01=eye,
        )
    # same base marginal, but x_0 _||_ x_1
    return GaussianCoupling(
        mu_0=torch.zeros(d), mu_1=torch.zeros(d),
        C_00=(1 + sigma**2) * eye, C_11=eye, C_01=torch.zeros(d, d),
    )


def part_b(dims=(2, 8), sigma: float = 0.5, n: int = 20_000, steps: int = 800) -> dict:
    """Proposition 3.1 with the exact (affine) velocity of a Gaussian coupling.

    The probability-flow ODE of a jointly Gaussian coupling is affine in x_0,
    so the transport cost E|X_{t=1}(x_0) - x_0|^2 can be evaluated to high
    accuracy and compared with the bound int_0^1 E|I_dot_t|^2 dt.
    """
    results = {}
    sch = build_interpolant("linear_zero_gamma")
    for d in dims:
        for name, coupled in [("coupled", True), ("independent", False)]:
            gc = _gaussian_coupling_moments(d, sigma, coupled)
            x1 = torch.randn(n, d)
            if coupled:
                x0 = x1 + sigma * torch.randn(n, d)
            else:
                x0 = torch.randn(n, d) * (1 + sigma**2) ** 0.5

            def vf(x, t, gc=gc):
                t_scalar = float(t.reshape(-1)[0]) if torch.is_tensor(t) else float(t)
                return gc.optimal_velocity(x, 1 - t_scalar, t_scalar, -1.0, 1.0)

            x1_hat = odeint(vf, x0, method="euler", steps=steps)
            cost = float(((x1_hat - x0) ** 2).sum(-1).mean())
            bound = float(transport_cost_upper_bound(sch, CoupledBatch(x0=x0, x1=x1)))
            generated_var = float(x1_hat.var(dim=0).mean())
            results[f"d{d}_{name}"] = {
                "transport_cost": cost,
                "bound": bound,
                "inequality_holds": bool(cost <= bound + 1e-6),
                "generated_variance": generated_var,
            }
            print(f"d={d:3d} {name:12s} cost={cost:9.4f}  bound={bound:9.4f}"
                  f"  var(X_1)={generated_var:6.4f}  (holds: {cost <= bound + 1e-6})")
    return results


# ---------------------------------------------------------------------------
def part_c(steps: int = 3000, d: int = 2, sigma: float = 0.5, n: int = 20_000) -> dict:
    """Learned velocity in 2-D: empirical cost versus the Proposition-3.1 bound."""
    set_seed(0)
    results = {}
    sch = build_interpolant("linear_zero_gamma")
    for name in ["coupled", "independent"]:
        coupling = DataDecorruptionCoupling(sigma=sigma) if name == "coupled" else IndependentCoupling(
            base_scale=(1 + sigma**2) ** 0.5
        )
        model = MLPVelocity(dim=d, hidden=128, depth=3, time_dim=32)
        opt = torch.optim.Adam(model.parameters(), lr=2e-3)
        for _ in range(steps):
            x1 = torch.randn(512, d)
            batch = coupling.sample(x1)
            loss, _ = velocity_loss(model, sch, batch, mse_form=True)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            x1 = torch.randn(n, d)
            batch = coupling.sample(x1)
            bound = float(transport_cost_upper_bound(sch, batch))
            x1_hat = odeint(lambda x, t: model(x, t), batch.x0, method="dopri5", atol=1e-6, rtol=1e-6)
            cost = float(((x1_hat - batch.x0) ** 2).sum(-1).mean())
            # closeness of the generated samples to the target density
            mean_shift = float(x1_hat.mean(0).abs().max())
            generated_var = float(x1_hat.var(dim=0).mean())
        results[name] = {
            "transport_cost": cost,
            "bound": bound,
            "abs_mean_generated": mean_shift,
            "generated_variance": generated_var,
        }
        print(f"learned {name:12s} cost={cost:8.4f}  bound={bound:8.4f}"
              f"  |mean|={mean_shift:.4f}  var={generated_var:.4f}")
    return results


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    dims = (2, 8, 64) if args.quick else (2, 8, 64, 256)
    results = {
        "eq21": part_a(dims=dims),
        "prop31_exact_velocity": part_b(),
        "prop31_learned_velocity": part_c(steps=1000 if args.quick else 3000),
    }
    (out / "transport_cost.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {out / 'transport_cost.json'}")


if __name__ == "__main__":
    main()
