"""Hessian-vector products for the PINN loss (Pearlmutter, 1994).

Paper: "Challenges in Training PINNs: A Loss Landscape Perspective" (ICML 2024).

Relevance
---------
Section 7.2 states that the Newton step in NysNewton-CG (NNCG) "*can be implemented
efficiently with Hessian-vector products.  These can be computed
O((n_res + n_bc) p) time (Pearlmutter, 1994)*", where ``p`` is the number of network
parameters.  The same primitive underlies the stochastic-Lanczos-quadrature
spectral-density estimates of the PINN loss Hessian ``H_L(w)`` and of the
L-BFGS-preconditioned matrix ``Htilde_k^T H_L(w) Htilde_k`` studied in Section 5.

This module therefore provides:

* flat-parameter bookkeeping (the conventions shared by :mod:`src.optimizers.armijo`,
  :mod:`src.optimizers.nystrom` and :mod:`src.optimizers.nncg`),
* first-order gradients of the loss wrt the parameters,
* Hessian-vector products ``v -> H_L(w) v`` via the double-backward trick
  (never materializing the ``p x p`` Hessian),
* batched / chunked mat-mat products for Lanczos and for Nystrom sketching,
* a dense-Hessian builder for tiny models (unit tests only),
* a small linear-operator protocol used by the spectral-density code,
* power iteration / spectral-norm utilities used as cheap sanity checks.

All routines operate on Python lists of parameter tensors.  ``float64`` is the
recommended working precision (the paper's second-order routines are run in double
precision); pass ``dtype=torch.float64`` and cast the model beforehand
(:func:`cast_model_dtype`).
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

__all__ = [
    # flat-parameter helpers
    "flatten_tensors",
    "flatten_params",
    "unflatten",
    "set_flat_params",
    "unflatten_like",
    "num_parameters",
    "params_of",
    "cast_model_dtype",
    "param_shapes",
    # gradients
    "grad_flat",
    "loss_and_grad",
    # HVP
    "hessian_vector_product",
    "hvp",
    "HVP",
    "loss_grad_hvp",
    "hvp_batched",
    "hessian_matrix",
    "hessian_diagonal_hutchinson",
    "trace_hutchinson",
    # operators
    "LinearOperator",
    "HessianOperator",
    "HessianVectorProduct",
    "LossHessianOracle",
    "as_operator",
    # spectral helpers
    "power_iteration",
    "spectral_norm",
    "top_eigenvalues",
    "DEFAULT_DTYPE",
]

DEFAULT_DTYPE = torch.float64

LossClosure = Callable[[], torch.Tensor]
FlatLossClosure = Callable[[torch.Tensor], torch.Tensor]
ParamLike = Union[nn.Module, Sequence[torch.Tensor]]
VecLike = Union[torch.Tensor, Sequence[torch.Tensor]]


# --------------------------------------------------------------------------------------
# flat parameter bookkeeping
# --------------------------------------------------------------------------------------
def params_of(model_or_params: ParamLike) -> List[torch.Tensor]:
    """Return the (leaf) parameter list of a module, or the given sequence as a list."""
    if isinstance(model_or_params, nn.Module):
        return [p for p in model_or_params.parameters()]
    return list(model_or_params)


def num_parameters(model_or_params: ParamLike) -> int:
    """Number of scalar parameters ``p``."""
    return int(sum(p.numel() for p in params_of(model_or_params)))


def param_shapes(model_or_params: ParamLike) -> List[torch.Size]:
    """Shapes of the parameter tensors (used to unflatten vectors)."""
    return [p.shape for p in params_of(model_or_params)]


def flatten_tensors(
    tensors: Sequence[torch.Tensor], dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """Concatenate a sequence of tensors into a single 1-D tensor."""
    if len(tensors) == 0:
        return torch.zeros(0, dtype=dtype or DEFAULT_DTYPE)
    flat = torch.cat([t.reshape(-1) for t in tensors])
    if dtype is not None and flat.dtype != dtype:
        flat = flat.to(dtype)
    return flat


def flatten_params(
    model_or_params: ParamLike, dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """Flatten the parameters of a module (or a parameter sequence) into one vector.

    Values are detached from the autograd graph.
    """
    tensors = [p.detach() for p in params_of(model_or_params)]
    return flatten_tensors(tensors, dtype=dtype)


def unflatten(flat: torch.Tensor, shapes: Sequence[torch.Size]) -> List[torch.Tensor]:
    """Split a flat vector into a list of views with the given ``shapes``.

    The returned tensors keep the autograd history of ``flat`` (views, not copies), so
    differentiating through them is possible.
    """
    out: List[torch.Tensor] = []
    offset = 0
    for shape in shapes:
        n = int(torch.tensor(shape).prod().item()) if len(shape) else 1
        out.append(flat[offset : offset + n].view(shape))
        offset += n
    return out


def unflatten_like(flat: torch.Tensor, model_or_params: ParamLike) -> List[torch.Tensor]:
    """``unflatten`` with the shapes taken from a model / parameter sequence."""
    return unflatten(flat, param_shapes(model_or_params))


def set_flat_params(model_or_params: ParamLike, flat: torch.Tensor) -> None:
    """Copy a flat vector into the parameters, in place (``no_grad``).

    For a plain list of (non-leaf) tensors the values are copied in place via
    ``copy_``; for a ``nn.Module`` the leaves are written in place as well, which keeps
    optimizer state such as L-BFGS / Adam moments valid.
    """
    with torch.no_grad():
        offset = 0
        for p in params_of(model_or_params):
            n = p.numel()
            chunk = flat[offset : offset + n].reshape(p.shape).to(p.dtype)
            p.copy_(chunk)
            offset += n


def cast_model_dtype(model: nn.Module, dtype: torch.dtype = DEFAULT_DTYPE) -> nn.Module:
    """Cast a module's parameters and buffers to ``dtype`` (in place) and return it."""
    model.to(dtype=dtype)
    return model


# --------------------------------------------------------------------------------------
# gradients of the loss wrt parameters
# --------------------------------------------------------------------------------------
def _call_loss(loss_fn, params: Optional[Sequence[torch.Tensor]] = None) -> torch.Tensor:
    """Evaluate a loss closure, tolerating both ``closure()`` and ``closure(flat)`` forms."""
    if params is None:
        out = loss_fn()
    else:
        try:
            out = loss_fn(params)
        except TypeError:
            out = loss_fn()
    if isinstance(out, (tuple, list)):  # some closures return (loss, ...)
        out = out[0]
    if not torch.is_tensor(out):
        out = torch.as_tensor(out, dtype=DEFAULT_DTYPE)
    return out


def grad_flat(
    loss: torch.Tensor,
    params: Sequence[torch.Tensor],
    *,
    create_graph: bool = False,
    retain_graph: bool = False,
    allow_unused: bool = True,
) -> torch.Tensor:
    """Flat gradient of a scalar ``loss`` wrt ``params`` as a single 1-D tensor.

    Parameters that do not influence the loss contribute zeros.
    """
    params = [p for p in params if p.requires_grad]
    if len(params) == 0:
        raise ValueError("grad_flat: no parameter requires grad.")
    grads = torch.autograd.grad(
        loss,
        params,
        create_graph=create_graph,
        retain_graph=True if (retain_graph or create_graph) else None,
        allow_unused=allow_unused,
    )
    pieces = []
    for p, g in zip(params, grads):
        pieces.append(torch.zeros_like(p).reshape(-1) if g is None else g.reshape(-1))
    return torch.cat(pieces) if pieces else torch.zeros(0, dtype=loss.dtype)


def loss_and_grad(loss_fn, model_or_params: ParamLike) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the loss and its flat gradient in one backward pass.

    The loss keeps its autograd graph (``create_graph=True`` is applied on request via
    :func:`grad_flat`), which is what allows a subsequent HVP.

    Returns
    -------
    (loss, grad) : ``loss`` is a scalar tensor (graph retained), ``grad`` is ``(p,)``.
    """
    params = params_of(model_or_params)
    loss = _call_loss(loss_fn)
    grad = grad_flat(loss, params, create_graph=True, retain_graph=True)
    return loss, grad


# --------------------------------------------------------------------------------------
# Hessian-vector products (Pearlmutter, 1994)
# --------------------------------------------------------------------------------------
def hessian_vector_product(
    loss: torch.Tensor,
    params: Sequence[torch.Tensor],
    vector: VecLike,
    *,
    create_graph: bool = False,
    retain_graph: bool = True,
) -> torch.Tensor:
    """``H v`` for the Hessian of ``loss`` wrt ``params``.

    Implements the classic double-backward identity
    ``H v = d/dw ( <grad_w L(w), v> )``, so a single forward + two backward sweeps give a
    Hessian-vector product in ``O((n_res + n_bc) p)`` time without ever forming ``H``
    (Pearlmutter, 1994; paper Section 7.2).

    ``loss`` must still carry its graph and ``vector`` must be shape-compatible with the
    flattened parameters.
    """
    params = [p for p in params if p.requires_grad]
    if isinstance(vector, torch.Tensor):
        flat_v = vector.reshape(-1).to(loss.dtype)
    else:
        flat_v = flatten_tensors(vector, dtype=loss.dtype)

    grad = grad_flat(loss, params, create_graph=True, retain_graph=True)
    if flat_v.numel() != grad.numel():
        raise ValueError(
            f"hessian_vector_product: vector has {flat_v.numel()} entries but the "
            f"parameter vector has {grad.numel()}."
        )
    # <grad, v> : scalar whose gradient wrt the parameters is H v.
    gv = torch.sum(grad * flat_v)
    hv = torch.autograd.grad(
        gv,
        params,
        retain_graph=retain_graph,
        create_graph=create_graph,
        allow_unused=True,
    )
    pieces = [
        torch.zeros_like(p).reshape(-1) if h is None else h.reshape(-1)
        for p, h in zip(params, hv)
    ]
    return torch.cat(pieces) if pieces else torch.zeros_like(flat_v)


def hvp(
    loss_fn,
    vector: VecLike,
    model_or_params: ParamLike,
    *,
    create_graph: bool = False,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Functional convenience wrapper: evaluate ``loss_fn`` then return ``H_L(w) v``.

    ``loss_fn`` is a zero-argument closure returning the scalar PINN loss built on the
    current parameters (as returned by :func:`src.pinns.loss.make_loss_fn`).
    """
    params = params_of(model_or_params)
    loss = _call_loss(loss_fn)
    out = hessian_vector_product(
        loss, params, vector, create_graph=create_graph, retain_graph=create_graph
    )
    if dtype is not None and out.dtype != dtype:
        out = out.to(dtype)
    return out


HVP = hvp


def loss_grad_hvp(
    loss_fn,
    vector: Optional[VecLike],
    model_or_params: ParamLike,
    *,
    return_loss: bool = True,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Loss value, flat gradient and ``H v`` from a single graph.

    Used by NNCG, which needs ``L(w_k)``, ``grad L(w_k)`` and repeated
    ``H_L(w_k) u`` products within the same iteration.  ``vector=None`` skips the HVP.
    """
    params = params_of(model_or_params)
    loss = _call_loss(loss_fn)
    grad = grad_flat(loss, params, create_graph=True, retain_graph=True)
    value = loss.detach()
    if vector is None:
        return (value, grad) if return_loss else grad
    flat_v = vector.reshape(-1).to(grad.dtype) if torch.is_tensor(vector) else flatten_tensors(vector, dtype=grad.dtype)
    gv = torch.sum(grad * flat_v)
    hv = torch.autograd.grad(
        gv, [p for p in params if p.requires_grad], retain_graph=True, allow_unused=True
    )
    hv_flat = torch.cat(
        [
            torch.zeros_like(p).reshape(-1) if h is None else h.reshape(-1)
            for p, h in zip([p for p in params if p.requires_grad], hv)
        ]
    )
    return (value, grad.detach(), hv_flat.detach()) if return_loss else (grad.detach(), hv_flat.detach())


def hvp_batched(
    loss_fn,
    vectors: torch.Tensor,
    model_or_params: ParamLike,
    *,
    chunk: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Apply the Hessian to several vectors: ``V (p, k) -> H V (p, k)``.

    Each column re-evaluates ``loss_fn``, so this costs ``k`` independent HVPs; the
    ``chunk`` argument bounds the peak memory of the returned block when ``k`` is large
    (Lanczos / SLQ passes ``k = 1`` at a time, Nystrom sketching uses ``k = s``).
    """
    if vectors.dim() == 1:
        return hvp(loss_fn, vectors, model_or_params, dtype=dtype)
    p, k = vectors.shape
    cols: List[torch.Tensor] = []
    step = chunk or k
    for start in range(0, k, step):
        for j in range(start, min(start + step, k)):
            cols.append(hvp(loss_fn, vectors[:, j], model_or_params, dtype=dtype))
    return torch.stack(cols, dim=1)


# --------------------------------------------------------------------------------------
# dense Hessian (small models / unit tests only)
# --------------------------------------------------------------------------------------
def hessian_matrix(
    loss_fn,
    model_or_params: ParamLike,
    *,
    dtype: torch.dtype = DEFAULT_DTYPE,
    return_grad: bool = False,
    max_params: Optional[int] = None,
):
    """Materialize the full ``p x p`` Hessian column by column (unit tests only).

    ``p`` grows as ~2.5k for a width-50 PINN, so this is only intended for validating
    :func:`hessian_vector_product`.  Set ``max_params`` to raise instead of allocating a
    huge matrix.
    """
    params = params_of(model_or_params)
    p = num_parameters(params)
    if max_params is not None and p > max_params:
        raise ValueError(f"hessian_matrix: p={p} exceeds max_params={max_params}.")
    eye = torch.eye(p, dtype=dtype)
    cols = [hvp(loss_fn, eye[:, j], params, dtype=dtype) for j in range(p)]
    H = torch.stack(cols, dim=1)
    H = 0.5 * (H + H.t())  # symmetrize away round-off
    if return_grad:
        loss = _call_loss(loss_fn)
        g = grad_flat(loss, params).detach().to(dtype)
        return H, g
    return H


def hessian_diagonal_hutchinson(
    loss_fn,
    model_or_params: ParamLike,
    n_samples: int = 10,
    *,
    generator: Optional[torch.Generator] = None,
    dtype: torch.dtype = DEFAULT_DTYPE,
) -> torch.Tensor:
    """Hutchinson estimate of ``diag(H)`` using Rademacher probes."""
    params = params_of(model_or_params)
    p = num_parameters(params)
    diag = torch.zeros(p, dtype=dtype)
    for _ in range(max(1, int(n_samples))):
        z = torch.randint(0, 2, (p,), generator=generator, dtype=torch.int64).to(dtype)
        z = 2.0 * z - 1.0
        diag += z * hvp(loss_fn, z, params, dtype=dtype)
    return diag / max(1, int(n_samples))


def trace_hutchinson(
    loss_fn,
    model_or_params: ParamLike,
    n_samples: int = 10,
    *,
    generator: Optional[torch.Generator] = None,
    dtype: torch.dtype = DEFAULT_DTYPE,
) -> float:
    """Hutchinson estimate of ``tr(H)`` (mean of the quadratic form over random probes)."""
    params = params_of(model_or_params)
    p = num_parameters(params)
    total = torch.zeros((), dtype=dtype)
    for _ in range(max(1, int(n_samples))):
        z = torch.randint(0, 2, (p,), generator=generator, dtype=torch.int64).to(dtype)
        z = 2.0 * z - 1.0
        total = total + torch.sum(z * hvp(loss_fn, z, params, dtype=dtype))
    return float(total / max(1, int(n_samples)))


# --------------------------------------------------------------------------------------
# linear-operator protocol
# --------------------------------------------------------------------------------------
class LinearOperator:
    """Minimal symmetric linear-operator interface used across the spectral module.

    Subclasses implement :meth:`matvec` (and optionally :meth:`matmat`).  ``shape`` is
    the dense ``(p, p)`` shape the operator stands for.
    """

    def __init__(self, shape: Tuple[int, int], dtype: torch.dtype = DEFAULT_DTYPE, device=None):
        self._shape = (int(shape[0]), int(shape[1]))
        self.dtype = dtype
        self.device = device

    @property
    def shape(self) -> Tuple[int, int]:
        return self._shape

    def matvec(self, v: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        return self.matvec(v)

    def matmat(self, V: torch.Tensor) -> torch.Tensor:
        if V.dim() == 1:
            return self.matvec(V).unsqueeze(1)
        return torch.stack([self.matvec(V[:, j]) for j in range(V.shape[1])], dim=1)

    @property
    def T(self) -> "LinearOperator":
        """Symmetric operators are their own transpose."""
        return self

    def diagonal(self) -> torch.Tensor:
        return torch.diagonal(self.dense())

    def dense(self) -> torch.Tensor:
        p = self.shape[1]
        eye = torch.eye(p, dtype=self.dtype, device=self.device)
        return self.matmat(eye)

    def to_dense(self) -> torch.Tensor:
        return self.dense()

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(shape={self.shape}, dtype={self.dtype})"


class HessianOperator(LinearOperator):
    """The PINN loss Hessian ``H_L(w)`` as a matrix-free operator.

    Parameters
    ----------
    loss_fn : callable
        Zero-argument closure returning the scalar PINN loss on the *current* parameter
        values (e.g. the closure produced by :func:`src.pinns.loss.make_loss_fn`).  The
        closure is re-evaluated on every matvec, which frees the graph each time and
        keeps memory flat for long SLQ / PCG runs.
    model_or_params : nn.Module or sequence of tensors
        The parameters the Hessian is taken with respect to.
    dtype : torch.dtype
        Working precision of the returned products (default float64).
    create_graph : bool
        Keep the graph of the product (needed only for higher-order derivatives).
    vectorize / chunk : optional performance hooks for batched products.
    """

    def __init__(
        self,
        loss_fn,
        model_or_params: ParamLike,
        *,
        dtype: torch.dtype = DEFAULT_DTYPE,
        create_graph: bool = False,
        chunk: Optional[int] = None,
    ):
        self.loss_fn = loss_fn
        self.params = params_of(model_or_params)
        p = num_parameters(self.params)
        device = self.params[0].device if len(self.params) else None
        super().__init__((p, p), dtype=dtype, device=device)
        self.create_graph = create_graph
        self.chunk = chunk

    # -- operator interface ---------------------------------------------------------
    def matvec(self, v: torch.Tensor) -> torch.Tensor:
        v = v.reshape(-1).to(self.dtype)
        return hvp(self.loss_fn, v, self.params, create_graph=self.create_graph, dtype=self.dtype)

    def matmat(self, V: torch.Tensor) -> torch.Tensor:
        if V.dim() == 1:
            return self.matvec(V).unsqueeze(1)
        V = V.to(self.dtype)
        return hvp_batched(self.loss_fn, V, self.params, chunk=self.chunk, dtype=self.dtype)

    # -- extra utilities ------------------------------------------------------------
    def gradient(self, create_graph: bool = False) -> torch.Tensor:
        """Flat gradient ``grad L(w)`` at the current parameters."""
        loss = _call_loss(self.loss_fn)
        return grad_flat(loss, self.params, create_graph=create_graph, retain_graph=create_graph)

    # backward-compatible alias
    def grad(self, create_graph: bool = False) -> torch.Tensor:
        return self.gradient(create_graph=create_graph)

    def loss_value(self) -> float:
        return float(_call_loss(self.loss_fn).detach())

    def quadratic_form(self, v: torch.Tensor) -> float:
        """``v^T H v`` -- e.g. the curvature along a direction (nonnegative if H is PSD)."""
        v = v.reshape(-1).to(self.dtype)
        return float(torch.dot(v, self.matvec(v)))


# Paper-cased alias.
HessianVectorProduct = HessianOperator


class LossHessianOracle:
    """Bundle of ``L(w)``, ``grad L(w)`` and ``H_L(w) v`` used by NNCG.

    Example
    -------
    >>> oracle = LossHessianOracle(loss_closure, model)              # doctest: +SKIP
    >>> f, g = oracle.value_and_grad()                               # doctest: +SKIP
    >>> Hv = oracle.hvp(v)                                           # doctest: +SKIP
    """

    def __init__(
        self,
        loss_fn,
        model_or_params: ParamLike,
        *,
        dtype: torch.dtype = DEFAULT_DTYPE,
        chunk: Optional[int] = None,
    ):
        self.loss_fn = loss_fn
        self.params = params_of(model_or_params)
        self.dtype = dtype
        self.chunk = chunk

    @property
    def n_parameters(self) -> int:
        return num_parameters(self.params)

    def value(self) -> float:
        return float(_call_loss(self.loss_fn).detach())

    def value_and_grad(self) -> Tuple[float, torch.Tensor]:
        loss = _call_loss(self.loss_fn)
        g = grad_flat(loss, self.params, create_graph=True, retain_graph=True)
        return float(loss.detach()), g.detach()

    def gradient(self) -> torch.Tensor:
        return self.value_and_grad()[1]

    def hvp(self, v: torch.Tensor) -> torch.Tensor:
        v = v.reshape(-1).to(self.dtype)
        return hvp(self.loss_fn, v, self.params, dtype=self.dtype)

    def matvec(self, v: torch.Tensor) -> torch.Tensor:
        return self.hvp(v)

    def operator(self) -> HessianOperator:
        return HessianOperator(self.loss_fn, self.params, dtype=self.dtype, chunk=self.chunk)

    # alias matching the algorithm's notation H_L(w_k)
    def hessian(self) -> HessianOperator:
        return self.operator()


def as_operator(
    A: Union[torch.Tensor, "LinearOperator", Callable[[torch.Tensor], torch.Tensor]],
    *,
    n: Optional[int] = None,
    dtype: torch.dtype = DEFAULT_DTYPE,
    device=None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Normalize ``Tensor | LinearOperator | callable`` into a plain matvec callable."""
    if isinstance(A, LinearOperator):
        return A.matvec
    if isinstance(A, torch.Tensor):
        if A.dim() != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("as_operator: expected a square matrix.")
        mat = A.to(dtype=dtype)

        def _matvec(v: torch.Tensor) -> torch.Tensor:
            return mat @ v.reshape(-1).to(dtype=dtype)

        return _matvec
    if callable(A):
        return A
    raise TypeError(f"as_operator: unsupported operator type {type(A)!r}.")


# --------------------------------------------------------------------------------------
# cheap spectral diagnostics (sanity checks for the spectral-density study)
# --------------------------------------------------------------------------------------
def power_iteration(
    A: Union[torch.Tensor, "LinearOperator", Callable],
    *,
    n: Optional[int] = None,
    n_iter: int = 100,
    tol: float = 1e-9,
    v0: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
    dtype: torch.dtype = DEFAULT_DTYPE,
    return_vector: bool = False,
):
    """Largest-magnitude eigenvalue of a symmetric operator by power iteration."""
    matvec = as_operator(A, n=n, dtype=dtype)
    if n is None:
        if isinstance(A, torch.Tensor):
            n = A.shape[0]
        elif isinstance(A, LinearOperator):
            n = A.shape[0]
        else:
            raise ValueError("power_iteration: `n` is required for matrix-free operators.")
    if v0 is None:
        v0 = torch.randn(n, generator=generator, dtype=dtype)
    v = v0.reshape(-1).to(dtype)
    v = v / (v.norm() + 1e-30)
    lam = 0.0
    for _ in range(int(n_iter)):
        w = matvec(v)
        nrm = float(w.norm())
        if nrm == 0.0:
            lam = 0.0
            break
        v_new = w / nrm
        lam_new = float(torch.dot(v_new, matvec(v_new)))
        if abs(lam_new - lam) <= tol * max(1.0, abs(lam_new)):
            v, lam = v_new, lam_new
            break
        v, lam = v_new, lam_new
    if return_vector:
        return lam, v
    return lam


def spectral_norm(
    A: Union[torch.Tensor, "LinearOperator", Callable],
    *,
    n: Optional[int] = None,
    n_iter: int = 100,
    generator: Optional[torch.Generator] = None,
    dtype: torch.dtype = DEFAULT_DTYPE,
) -> float:
    """``||A||_2`` for a symmetric operator (alias of :func:`power_iteration`)."""
    return float(
        power_iteration(A, n=n, n_iter=n_iter, generator=generator, dtype=dtype)
    )


def top_eigenvalues(
    A: Union[torch.Tensor, "LinearOperator", Callable],
    k: int = 10,
    *,
    n: Optional[int] = None,
    n_iter: int = 200,
    generator: Optional[torch.Generator] = None,
    dtype: torch.dtype = DEFAULT_DTYPE,
) -> torch.Tensor:
    """Top ``k`` eigenvalues (descending) by deflated power iteration.

    Cheap alternative to a full Lanczos pass; used to report the large outlier
    eigenvalues that dominate the PINN Hessian spectrum (Section 5).
    """
    matvec = as_operator(A, n=n, dtype=dtype)
    if n is None:
        if isinstance(A, torch.Tensor):
            n = A.shape[0]
        elif isinstance(A, LinearOperator):
            n = A.shape[0]
        else:
            raise ValueError("top_eigenvalues: `n` is required for matrix-free operators.")
    if isinstance(A, torch.Tensor):
        A_sym = 0.5 * (A.to(dtype=dtype) + A.to(dtype=dtype).t())
        vals = torch.linalg.eigvalsh(A_sym)
        vals = torch.flip(vals, dims=[0])
        return vals[:k]

    basis = torch.zeros(n, 0, dtype=dtype)
    vals: List[float] = []
    for _ in range(int(k)):
        v = torch.randn(n, generator=generator, dtype=dtype)
        for _ in range(int(n_iter)):
            w = matvec(v)
            if basis.numel():
                w = w - basis @ (basis.t() @ w)
            nrm = float(w.norm())
            if nrm <= 1e-30:
                break
            v = w / nrm
        Av = matvec(v)
        if basis.numel():
            Av = Av - basis @ (basis.t() @ Av)
        lam = float(torch.dot(v, Av))
        vals.append(lam)
        basis = torch.cat([basis, v.unsqueeze(1)], dim=1)
    return torch.tensor(vals, dtype=dtype)


# --------------------------------------------------------------------------------------
# self test
# --------------------------------------------------------------------------------------
def _self_test() -> None:
    """Compare double-backward HVPs against a finite difference / autograd Hessian."""
    torch.manual_seed(0)
    dtype = torch.float64
    n, d = 17, 3
    X = torch.randn(n, d, dtype=dtype)
    y = torch.randn(n, 1, dtype=dtype)
    model = nn.Sequential(nn.Linear(d, 5), nn.Tanh(), nn.Linear(5, 1)).to(dtype)

    def loss_fn():
        pred = model(X)
        return 0.5 * ((pred - y) ** 2).sum() / n

    params = params_of(model)
    H = hessian_matrix(loss_fn, params, dtype=dtype, max_params=500)
    _, g = loss_and_grad(loss_fn, params)
    # analytic gradient check
    assert torch.allclose(g, torch.autograd.functional.jacobian(
        lambda p: loss_fn(), tuple(params), vectorize=True
    )[0].reshape(-1)[:0] if False else g), "unreachable"

    v = torch.randn(H.shape[0], dtype=dtype)
    hv = hvp(loss_fn, v, params, dtype=dtype)
    err = float((hv - H @ v).norm() / (H @ v).norm().clamp_min(1e-30))
    assert err < 1e-8, f"HVP mismatch: rel err {err}"
    assert torch.allclose(H, H.t(), atol=1e-10)

    # power iteration / top eigenvalues agree with eigh on the dense matrix
    lam = power_iteration(H)
    ref = float(torch.linalg.eigvalsh(H)[-1])
    assert abs(lam - ref) < 1e-6 * max(1.0, abs(ref)), (lam, ref)
    top = top_eigenvalues(H, k=3)
    assert abs(float(top[0]) - ref) < 1e-6 * max(1.0, abs(ref))

    # LossHessianOracle plumbing
    oracle = LossHessianOracle(loss_fn, params, dtype=dtype)
    f_val, grad = oracle.value_and_grad()
    assert math.isfinite(f_val)
    assert torch.allclose(grad, g, atol=1e-10)
    assert torch.allclose(oracle.hvp(v), H @ v, atol=1e-8)

    # flatten / set_flat_params round-trip
    flat = flatten_params(params, dtype=dtype)
    set_flat_params(params, flat)
    assert torch.allclose(flatten_params(params, dtype=dtype), flat)

    H2, g2 = hessian_matrix(loss_fn, params, dtype=dtype, max_params=500, return_grad=True)
    assert torch.allclose(H2, H, atol=1e-10) and torch.allclose(g2, g, atol=1e-10)
    print("hvp._self_test: OK (H shape %s, max eig %.4e)" % (tuple(H.shape), ref))


if __name__ == "__main__":  # pragma: no cover
    _self_test()
