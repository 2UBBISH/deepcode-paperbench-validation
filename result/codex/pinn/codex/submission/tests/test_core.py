"""Unit tests / smoke tests.

Run with ``python -m pytest tests -q`` (or ``python tests/test_core.py``).
The tests are deliberately tiny: they check correctness of the pieces that are
easy to get wrong (PDE residuals vs. the analytical solutions, L2RE, the
Hessian-vector product, and the unrolled L-BFGS preconditioner) without running
any expensive experiment.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pinn.data import build_dataset  # noqa: E402
from pinn.hessian.hvp import HessianOperator, loss_hessian_matrix  # noqa: E402
from pinn.hessian.lbfgs_precond import LBFGSHistory  # noqa: E402
from pinn.losses import component_losses, pinn_loss  # noqa: E402
from pinn.metrics import l2_relative_error  # noqa: E402
from pinn.models import build_model  # noqa: E402
from pinn.optim.lbfgs import LBFGSOptimizer  # noqa: E402
from pinn.optim.nncg import NNCG, NNCGConfig  # noqa: E402
from pinn.optim.nncg import nystrom_pcg, randomized_nystrom_approximation  # noqa: E402
from pinn.hessian.spectral import slq_lanczos, spectral_density  # noqa: E402
from pinn.optim.objective import Objective  # noqa: E402
from pinn.optim.trainers import TrainConfig, train  # noqa: E402
from pinn.problems import build_problem  # noqa: E402


def _tiny_dataset(problem, n_res=64, seed=0):
    return build_dataset(problem, n_res=n_res, seed=seed)


def test_residuals_vanish_on_the_analytical_solution():
    """The exact solution must satisfy the PDE and its conditions."""
    for name in ("convection", "reaction", "wave"):
        problem = build_problem(name)
        ds = _tiny_dataset(problem)

        def exact_net(X, problem=problem):
            return problem.exact(X)

        r = problem.residual(exact_net, ds.X_res).abs().max().item()
        # a wrong sign or a wrong coefficient would give O(1) residues; the
        # tolerance only has to absorb float32 round-off (the wave residual
        # contains second derivatives with factors of order 10^3)
        tol = 1e-3
        assert r < tol, (name, r)
        for term in problem.ic_terms():
            v = term.fn(exact_net, ds.X_ic[term.name]).abs().max().item()
            assert v < tol, (name, term.name, v)
        for term in problem.bc_terms():
            v = term.fn(exact_net, ds.X_bc[term.name]).abs().max().item()
            # float32 round-off of 2*pi is amplified by the periodic wrap-around
            assert v < tol, (name, term.name, v)


def test_evaluation_points_and_l2re():
    for name in ("convection", "reaction", "wave"):
        problem = build_problem(name)
        X = problem.evaluation_points()
        # 255 x 100 interior grid + 257 initial points + 101 per boundary
        n_ic = sum(t.n_points for t in problem.ic_terms())
        n_bc = sum(t.n_points for t in problem.bc_terms())
        assert X.shape[0] <= 255 * 100 + n_ic + n_bc
        assert X.shape[0] >= 255 * 100
        exact_net = lambda X, p=problem: p.exact(X)  # noqa: E731
        assert l2_relative_error(exact_net, problem) < 1e-12


def test_hessian_matches_finite_differences():
    torch.manual_seed(0)
    problem = build_problem("reaction")
    ds = _tiny_dataset(problem, n_res=32)
    net = build_model(width=8, n_layers=2).double()
    obj = Objective(problem, net, ds)
    H = loss_hessian_matrix(obj.loss, obj.params)
    assert torch.allclose(H, H.T, atol=1e-6)
    # finite differences of the gradient in a random direction
    v = torch.randn(obj.n_params, dtype=torch.float64)
    v = v / torch.linalg.norm(v)
    w0 = obj.flat_params().detach()
    eps = 1e-5
    obj.set_flat_params(w0 + eps * v)
    g1 = obj.grad()
    obj.set_flat_params(w0 - eps * v)
    g2 = obj.grad()
    obj.set_flat_params(w0)
    fd = (g1 - g2) / (2 * eps)
    assert torch.allclose(fd, H @ v, rtol=1e-4, atol=1e-6)


def test_lbfgs_preconditioner_factorisation():
    """Htilde Htilde^T must equal the usual two-loop L-BFGS matrix H_k."""
    torch.manual_seed(0)
    p = 12
    m = 5
    # curvature pairs with y^T s > 0, as produced by a real L-BFGS run
    s = [torch.randn(p, dtype=torch.float64) for _ in range(m)]
    y = [2.5 * s[i] + 0.1 * torch.randn(p, dtype=torch.float64) for i in range(m)]
    rho = [1.0 / float(torch.dot(y[i], s[i])) for i in range(m)]
    hist = LBFGSHistory(s=s[::-1], y=y[::-1], rho=rho[::-1], gamma=0.7)
    pre = __import__("pinn.hessian.lbfgs_precond", fromlist=["LBFGSPreconditioner"]).LBFGSPreconditioner(hist)
    u = torch.randn(p, dtype=torch.float64)
    two_loop = hist.two_loop(u)
    factored = pre.apply_H(u)
    assert torch.allclose(two_loop, factored, rtol=1e-8, atol=1e-8)
    # with H = I the non-zero eigenvalues of M = Htilde^T H Htilde must equal
    # the non-zero eigenvalues of H_k H = H_k
    Id = lambda x: x  # noqa: E731
    M = torch.stack([pre.matvec(torch.eye(p + m, dtype=torch.float64)[:, i], Id) for i in range(p + m)], dim=1)
    Hk = torch.stack([pre.apply_H(torch.eye(p, dtype=torch.float64)[:, i]) for i in range(p)], dim=1)
    ev_M = torch.linalg.eigvalsh(0.5 * (M + M.T))
    ev_Hk = torch.linalg.eigvalsh(0.5 * (Hk + Hk.T))
    # M has m additional (numerically) zero eigenvalues
    assert torch.allclose(torch.sort(ev_M).values[m:], ev_Hk, atol=1e-8)


def test_lbfgs_history_from_torch_state():
    torch.manual_seed(0)
    problem = build_problem("reaction")
    ds = _tiny_dataset(problem, n_res=32)
    net = build_model(width=8, n_layers=2)
    obj = Objective(problem, net, ds)
    opt = LBFGSOptimizer(obj, history_size=5)
    for _ in range(6):
        opt.step()
    hist = opt.history()
    assert len(hist) <= 5
    u = torch.randn(obj.n_params, dtype=torch.float64)
    from pinn.hessian.lbfgs_precond import LBFGSPreconditioner

    assert torch.allclose(
        hist.two_loop(u), LBFGSPreconditioner(hist).apply_H(u), rtol=1e-6, atol=1e-6
    )


def test_component_losses_sum_to_total():
    problem = build_problem("wave")
    ds = _tiny_dataset(problem, n_res=64)
    net = build_model(width=8, n_layers=2)
    total = float(pinn_loss(problem, net, ds))
    parts = component_losses(problem, net, ds)
    assert abs(total - sum(parts.values())) <= 1e-6 * max(abs(total), 1e-12)


def test_training_smoke():
    problem = build_problem("convection")
    ds = _tiny_dataset(problem, n_res=64)
    net = build_model(width=8, n_layers=2, seed=0)
    cfg = TrainConfig(optimizer="adam_lbfgs", lr=1e-2, switch_iter=3, iters=6, log_every=1)
    res = train(problem, net, ds, cfg, seed=0)
    assert res.adam_steps == 3 and res.lbfgs_steps == 3
    assert math.isfinite(res.final_loss)


def test_nncg_smoke():
    problem = build_problem("reaction")
    ds = _tiny_dataset(problem, n_res=32)
    net = build_model(width=8, n_layers=2, seed=0)
    obj = Objective(problem, net, ds)
    cfg = NNCGConfig(iters=3, sketch_size=4, preconditioner_frequency=2, cg_max_iter=20, log_every=1)
    res = NNCG(obj, cfg, problem=problem).run()
    assert math.isfinite(res.final_loss)
    assert res.final_loss <= res.trace["loss"][0] * 1.5


def test_slq_recovers_the_extreme_eigenvalues():
    """SLQ must place mass at the true eigenvalues (Figures 3 and 7)."""
    torch.manual_seed(0)
    n = 256
    B = torch.randn(n, n, dtype=torch.float64)
    A = B @ B.T / n + 5.0 * torch.eye(n, dtype=torch.float64)  # well conditioned
    A = 0.5 * (A + A.T)
    true = torch.linalg.eigvalsh(A)
    v0 = torch.randn(n, dtype=torch.float64)
    nodes, weights, _ = slq_lanczos(lambda v: A @ v, n, n_iter=80, v0=v0)
    # the total quadrature weight is one
    assert abs(float(np.sum(weights)) - 1.0) < 1e-8
    # the extreme Ritz values approximate the extreme eigenvalues
    assert abs(float(nodes.max()) - float(true.max())) < 1e-6 * float(true.max())
    # the *smallest* Ritz value converges more slowly (the spectrum is clustered)
    assert abs(float(nodes.min()) - float(true.min())) < 1e-3 * float(true.max())
    # and the estimated density integrates to one
    grids, density, _ = spectral_density(lambda v: A @ v, n, n_runs=2, n_iter=80, seed=0)
    integral = float(np.trapz(density, grids))
    assert 0.5 < integral < 1.5, integral


def test_randomized_nystrom_approximation():
    """Algorithm 5 must return the top-``s`` eigenpairs of a decaying spectrum."""
    torch.manual_seed(0)
    n, s = 400, 40
    Q, _ = torch.linalg.qr(torch.randn(n, n, dtype=torch.float64))
    true_evals = torch.tensor([(j + 1) ** (-2.0) for j in range(n)], dtype=torch.float64)
    A = Q @ torch.diag(true_evals) @ Q.T
    A = 0.5 * (A + A.T)
    U, Lam = randomized_nystrom_approximation(
        lambda v: A @ v, n, s, generator=torch.Generator().manual_seed(1)
    )
    assert U.shape == (n, s) and Lam.shape == (s,)
    assert torch.all(Lam >= 0)
    approx_evals = torch.sort(Lam, descending=True).values
    target = true_evals[:s]
    rel_err = float(torch.linalg.norm(approx_evals[:10] - target[:10]) / torch.linalg.norm(target[:10]))
    assert rel_err < 0.1, rel_err
    # the Nystroem approximation of A must be accurate for a fast-decaying spectrum
    approx = U @ torch.diag(Lam) @ U.T
    err = float(torch.linalg.norm(A - approx) / torch.linalg.norm(A))
    assert err < 0.05, err


def test_nystrom_pcg_solves_the_damped_newton_system():
    """Algorithm 6 must reproduce the dense solution of ``(A + mu I) x = b``."""
    torch.manual_seed(0)
    n, s = 256, 32
    B = torch.randn(n, n, dtype=torch.float64)
    A = B @ B.T / n + torch.eye(n, dtype=torch.float64)
    A = 0.5 * (A + A.T)
    b = torch.randn(n, dtype=torch.float64)
    mu = 1e-2
    U, Lam = randomized_nystrom_approximation(
        lambda v: A @ v, n, s, generator=torch.Generator().manual_seed(0)
    )
    x, n_cg = nystrom_pcg(
        lambda v: A @ v,
        b,
        torch.zeros(n, dtype=torch.float64),
        U,
        Lam,
        s,
        mu,
        eps=1e-12,
        max_iter=500,
    )
    x_ref = torch.linalg.solve(A + mu * torch.eye(n, dtype=torch.float64), b)
    err = float(torch.linalg.norm(x - x_ref) / torch.linalg.norm(x_ref))
    assert err < 1e-6, (err, n_cg)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
