# Theory ↔ code map

This note records how every theoretical statement of the paper is realised in
the code and how it is checked numerically.

## Definition 3.1 (stochastic interpolant with coupling)

    I_t = alpha_t x_0 + beta_t x_1 + gamma_t z,   t in [0, 1]

with `alpha_0 = beta_1 = 1`, `alpha_1 = beta_0 = gamma_0 = gamma_1 = 0` and
`alpha_t^2 + beta_t^2 + gamma_t^2 > 0`.

* Implementation: `si_couplings/interpolants.py`
  (`LinearInterpolant`, `VPInterpolant`, `InterpolantSchedule.interpolate`,
  `InterpolantSchedule.interpolate_velocity`).
* Schedules: `"linear"` (`alpha_t = 1-t`, `beta_t = t`,
  `gamma_t = sqrt(2t(1-t))`), `"linear_zero_gamma"` (`gamma_t = 0`, used with
  every coupling whose randomness is carried by the coupling itself),
  `"linear_reversed"` (the literal pair printed in Section 4.1) and `"vp"`.
* Tests: `tests/test_interpolants.py` checks the boundary conditions, the
  positivity condition, the shapes for vector- and image-valued data and the
  fact that `"linear_reversed"` is the time reversal of `"linear"`.

## Theorem 3.1 / A.1 (transport equation, velocity and score objectives)

    b_t(x, xi) = E[I_dot_t | I_t = x, xi],   g_t(x, xi) = E[z | I_t = x, xi]
    d/dt rho_t + div(b_t rho_t) = 0
    grad log rho_t(x | xi) = -gamma_t^{-1} g_t(x, xi)
    L_b(b_hat) = int E[ |b_hat_t(I_t, xi)|^2 - 2 I_dot_t . b_hat_t(I_t, xi) ] dt
    L_g(g_hat) = int E[ |g_hat_t(I_t, xi)|^2 - 2 z . g_hat_t(I_t, xi) ] dt

* Implementation: `si_couplings/losses.py` (`velocity_loss`, `score_loss`,
  `velocity_objective`, `score_objective`).  Both the exact objective of the
  paper and the equivalent MSE form `|b_hat - I_dot|^2` (which differs by the
  constant `E|I_dot|^2`) are available; the latter is the default because it
  has better numerical behaviour.
* The optimal velocity for a jointly Gaussian coupling is available in closed
  form in `si_couplings/interpolants.py` (`GaussianCoupling`), which is used
  as a reference in the tests and in `experiments/transport_cost.py`.
* Tests: `tests/test_theory.py`
  * `test_score_identity_matches_closed_form_gaussian` verifies both
    ingredients of the score identity numerically: the marginal of `I_t` is
    the predicted Gaussian, and a kernel-regression estimate of
    `E[z | I_t ≈ x]` matches `gamma_t Sigma_t^{-1} (x - mu_t)`.
  * `test_velocity_objective_minimised_by_conditional_expectation` shows that
    the least-squares minimiser of `L_b` over affine models equals the
    closed-form `E[I_dot_t | I_t = x]`.
  * `test_velocity_loss_is_mse_up_to_constant` checks the algebraic identity.
* Conditioning on `xi` (Remark 3.1 and Appendix A): every coupling exposes the
  conditioning signal it wants the model to see (`cond` in
  `CoupledBatch`), and the velocity model appends it to the channels of `x_t`
  (`si_couplings/models/velocity.py`).  Class labels are passed to the class
  embedding of the U-Net.

### Effective noise scale for `x_0 = m(x_1) + sigma zeta`

Section 3.2 notes that when the base is built as `x_0 = m(x_1) + sigma zeta`
one may take `gamma_t = 0`, and that the score is nevertheless available
because of the `sigma zeta` factor.  Writing

    I_t = alpha_t m(x_1) + beta_t x_1 + (alpha_t sigma) zeta,

the general theory applies with the *effective* coefficient
`gamma_tilde_t = alpha_t sigma`, so that
`grad log rho_t = -gamma_tilde_t^{-1} E[zeta | I_t = x]`.  This is implemented
as `InterpolantSchedule.effective_gamma` and tested in
`tests/test_interpolants.py::test_effective_gamma_of_coupled_interpolant`.

## Corollary 3.1 / A.1 (probability flow and diffusions with coupling)

    ODE:          X_dot_t = b_t(X_t, xi)
    forward SDE:  dX_t = [b_t - eps_t gamma_t^{-1} g_t] dt + sqrt(2 eps_t) dW_t
    backward SDE: dX_t = [b_t + eps_t gamma_t^{-1} g_t] dt + sqrt(2 eps_t) dW_t

* Implementation: `si_couplings/solvers.py`
  (`odeint` with `dopri5` (default, matching the paper), `euler`
  (= Algorithm 2), `heun` and an optional `torchdiffeq` backend), plus
  `forward_sde_sample` / `backward_sde_sample` for the two SDEs.
* The dopri5 implementation is a standard FSAL Dormand-Prince 5(4) pair with
  the same error control as `torchdiffeq`; `tests/test_solvers.py` checks it
  against the analytic solution of `x_dot = -x`.

## Proposition 3.1 (control of the transport cost)

    E_{x_0 ~ rho_0}[|X_{t=1}(x_0) - x_0|^2] <= int_0^1 E|I_dot_t|^2 dt

* Implementation: `si_couplings/losses.py::transport_cost_upper_bound`
  computes the right-hand side by Monte Carlo.
* Verification: `experiments/transport_cost.py` (part B) integrates the
  probability-flow ODE with the *exact* affine velocity of a Gaussian coupling
  and checks the inequality in several dimensions; `tests/test_theory.py::
  test_transport_cost_bound_of_proposition_31` does the same in 2-D and also
  checks that the flow recovers the target marginal.

## Section 3.3 (reducing transport costs via coupling)

For the data-decorruption coupling `m(x_1) = x_1`, `C = sigma^2 Id`,
`alpha_t = 1-t`, `beta_t = t`, `gamma_t = 0`:

    E|I_dot_t|^2 = d sigma^2                     (coupled)
    E|I_dot_t|^2 = 2 E|x_1|^2 + d sigma^2        (independent)

because `I_dot_t = x_1 - x_0 = -sigma zeta`.  `experiments/transport_cost.py`
(part A) verifies both expressions by Monte Carlo for `d = 2, 8, 64, 256` and
`tests/test_couplings.py::test_data_decorruption_coupling_statistics` checks
`E|x_0 - x_1|^2 = d sigma^2`.

## Section 4 (numerical experiments)

* In-painting (Section 4.1): `si_couplings/couplings.py::InpaintingCoupling`
  (`x_0 = xi o x_1 + (1 - xi) o zeta`, 64 tiles, `p = 0.3`), the output
  masking that enforces `b_t = 0` on the observed pixels
  (`si_couplings/models/velocity.py`, `VelocityModel.masks_output`), sampling
  in `si_couplings/sample.py::sample_inpainting`, FID in
  `si_couplings/fid.py`, figures in `si_couplings/visualize.py`.
* Super-resolution (Section 4.2): `SuperResolutionCoupling`
  (`x_0 = U(D(x_1)) + sigma zeta`, `xi = U(D(x_1))` appended to the channels
  of `x_t`), `si_couplings/data/superres.py` for the down/up-sampling
  operators and `sample_super_resolution` for generation.
* Architecture and optimisation (Appendix B): `si_couplings/models/unet.py`
  (DDPM U-Net with `dim=256`, `dim_mults=(1,1,2,3,4)`,
  `resnet_block_groups=8`, learned sinusoidal time conditioning of dimension
  32, attention head dimension 64 with 4 heads, class embeddings) and
  `si_couplings/train.py` (Adam at `2e-4`, StepLR with `gamma=0.99` every
  1000 steps, no weight decay, gradient-norm clipping at 10,000, 200,000
  steps of batch size 32 as stated in the addendum).
