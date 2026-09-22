"""Distributed / parallelism helpers (Lightning Fabric glue).

This module wraps `lightning.fabric` so that the training and sampling entry
points compiled by the reproduction plan (``train.py``, ``sample.py``,
``evaluate.py``) can run either on a single device or across several
GPUs/processes without changing the rest of the code base.

The paper (Appendix B / Addendum) trains pixel-space ImageNet 256/512 models
with a batch size of 32 over 200,000 gradient steps using PyTorch + Lightning
Fabric for parallelism.  Nothing here is paper-specific: it is engineering glue
that keeps the training loop agnostic to the launch mechanism
(``python train.py`` vs. ``torchrun --nproc_per_node=N train.py``).

Public API
----------
- :class:`FabricLike`      : minimal protocol describing the subset of the
  Fabric API this repo relies on (useful for typing / dependency injection).
- :func:`setup`            : create (and cache) a ``Fabric``/``FabricLike``
  object from CLI/config style arguments.
- :func:`is_distributed`   : whether more than one process participates.
- :func:`get_world_size`   : number of participating processes (default 1).
- :func:`get_rank`         : global rank of the current process (default 0).
- :func:`is_main_process`  : ``get_rank() == 0`` (with a Fabric-aware variant).
- :func:`barrier`          : synchronize all processes (no-op if not torch.distributed).
- :func:`all_reduce_mean`  : mean-reduce a tensor across processes.
- :func:`gather_tensor`    : concatenate tensors across processes (rank 0 gets all).
- :func:`broadcast_tensor` : broadcast a tensor from ``src`` to all processes.
- :func:`distributed_loader`: wrap a data loader for distributed sampling.
- :func:`rank_zero_print`  : print only from the main process.
- :func:`get_fabric`       : return the cached fabric (creating it lazily).

All functions degrade gracefully to single-process semantics when
``lightning.fabric`` (or ``torch.distributed``) is unavailable, which keeps the
CPU smoke tests / toy unit tests runnable in minimal environments.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Protocol, Sequence, runtime_checkable

import torch

__all__ = [
    "FabricLike",
    "setup",
    "get_fabric",
    "is_distributed",
    "get_world_size",
    "get_rank",
    "is_main_process",
    "barrier",
    "all_reduce_mean",
    "gather_tensor",
    "broadcast_tensor",
    "distributed_loader",
    "rank_zero_print",
    "rank_zero_log",
    "unwrap_model",
    "is_available",
    "SHARED_FABRIC",
]

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Optional Lightning Fabric import (kept lazy / guarded).
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - exercised implicitly via import path
    from lightning.fabric import Fabric  # type: ignore

    FABRIC_AVAILABLE = True
except Exception:  # ImportError or any broken transitive import
    try:  # Older layouts re-exported Fabric from the top-level package
        from lightning import Fabric  # type: ignore

        FABRIC_AVAILABLE = True
    except Exception:
        Fabric = None  # type: ignore
        FABRIC_AVAILABLE = False


# Cache so that repeated calls (e.g. setup() in train.py and again in eval)
# do not spin up multiple Fabric instances.
SHARED_FABRIC: Optional[Any] = None


# --------------------------------------------------------------------------- #
# Protocol describing the Fabric surface used by the repo.
# --------------------------------------------------------------------------- #
@runtime_checkable
class FabricLike(Protocol):  # pragma: no cover - protocol declaration only
    """Subset of the ``lightning.fabric.Fabric`` API used by this repository.

    Methods are optional from the caller's perspective; helper functions in
    this module probe for their presence with ``getattr`` so that a torch-free
    stub (e.g. a simple namespace in tests) still works.
    """

    device: Any
    global_rank: int
    world_size: int
    is_global_zero: bool

    def launch(self, *args: Any, **kwargs: Any) -> Any: ...

    def setup(self, *args: Any, **kwargs: Any) -> Any: ...

    def setup_module(self, module: torch.nn.Module, *a: Any, **kw: Any) -> torch.nn.Module: ...

    def setup_optimizers(self, *args: Any, **kwargs: Any) -> Any: ...

    def setup_dataloaders(self, *args: Any, **kwargs: Any) -> Any: ...

    def backward(self, *args: Any, **kwargs: Any) -> Any: ...

    def all_reduce(self, *args: Any, **kwargs: Any) -> Any: ...

    def all_gather(self, *args: Any, **kwargs: Any) -> Any: ...

    def barrier(self, *args: Any, **kwargs: Any) -> Any: ...

    def broadcast(self, *args: Any, **kwargs: Any) -> Any: ...


# --------------------------------------------------------------------------- #
# Availability / introspection helpers.
# --------------------------------------------------------------------------- #
def is_available() -> bool:
    """Return ``True`` when ``lightning.fabric`` could be imported."""

    return bool(FABRIC_AVAILABLE)


def _env_world_size() -> int:
    """Best-effort world size from the standard torchrun env vars."""

    for key in ("WORLD_SIZE", "LOCAL_WORLD_SIZE", "SLURM_NTASKS"):
        val = os.environ.get(key)
        if val is not None:
            try:
                return max(int(val), 1)
            except (TypeError, ValueError):
                continue
    return 1


def _env_rank() -> int:
    """Best-effort global rank from the standard torchrun env vars."""

    for key in ("RANK", "SLURM_PROCID", "GLOBAL_RANK"):
        val = os.environ.get(key)
        if val is not None:
            try:
                return max(int(val), 0)
            except (TypeError, ValueError):
                continue
    return 0


def _dist_ready() -> bool:
    """Whether ``torch.distributed`` is initialized and usable."""

    try:
        return bool(torch.distributed.is_available() and torch.distributed.is_initialized())
    except Exception:  # pragma: no cover - defensive
        return False


# --------------------------------------------------------------------------- #
# Setup / access.
# --------------------------------------------------------------------------- #
def setup(
    accelerator: str = "auto",
    devices: Any = "auto",
    num_nodes: int = 1,
    precision: str = "32-true",
    strategy: str = "auto",
    *,
    seed: Optional[int] = None,
    cache: bool = True,
    **fabric_kwargs: Any,
) -> Any:
    """Create and launch a ``Fabric`` object (or a single-process fallback).

    Parameters mirror the subset of ``lightning.fabric.Fabric.__init__`` used
    by the reproduction plan.  When Lightning Fabric is not installed a tiny
    :class:`_SingleProcessFabric` shim is returned so that the training loop
    keeps working on CPU.

    Parameters
    ----------
    accelerator:
        ``"auto"``, ``"cpu"``, ``"gpu"``, ...
    devices:
        ``"auto"``, an int, a list of device indices, ...
    precision:
        Fabric/PL precision string (the paper does not specify one; the default
        ``"32-true"`` matches "no mixed precision").
    seed:
        Optional seed forwarded to Fabric (``fabric.seed_everything``).
    cache:
        Reuse (and store) the module level :data:`SHARED_FABRIC`.
    """

    global SHARED_FABRIC

    if cache and SHARED_FABRIC is not None:
        return SHARED_FABRIC

    fabric: Any
    if FABRIC_AVAILABLE and Fabric is not None:
        try:
            fabric = Fabric(
                accelerator=accelerator,
                devices=devices,
                num_nodes=num_nodes,
                precision=precision,
                strategy=strategy,
                **fabric_kwargs,
            )
            fabric.launch()
            if seed is not None and hasattr(fabric, "seed_everything"):
                fabric.seed_everything(seed)
        except Exception as exc:  # pragma: no cover - runtime/device failures
            logger.warning(
                "Fabric setup failed (%s); falling back to single-process mode.", exc
            )
            fabric = _SingleProcessFabric(seed=seed)
    else:
        logger.info("lightning.fabric unavailable; using single-process fallback.")
        fabric = _SingleProcessFabric(seed=seed)

    if cache:
        SHARED_FABRIC = fabric
    return fabric


def get_fabric(**kwargs: Any) -> Any:
    """Return the cached fabric, creating it lazily if needed."""

    global SHARED_FABRIC
    if SHARED_FABRIC is None:
        return setup(**kwargs)
    return SHARED_FABRIC


def set_fabric(fabric: Any) -> Any:
    """Inject an externally constructed fabric into the module cache."""

    global SHARED_FABRIC
    SHARED_FABRIC = fabric
    return fabric


# --------------------------------------------------------------------------- #
# Rank / world size queries.
# --------------------------------------------------------------------------- #
def get_world_size(fabric: Optional[Any] = None) -> int:
    """Number of distributed processes (1 when not distributed)."""

    if fabric is not None:
        ws = getattr(fabric, "world_size", None)
        if ws is not None:
            try:
                return max(int(ws), 1)
            except (TypeError, ValueError):
                pass

    if _dist_ready():
        try:
            return max(int(torch.distributed.get_world_size()), 1)
        except Exception:  # pragma: no cover - defensive
            pass
    return _env_world_size()


def get_rank(fabric: Optional[Any] = None) -> int:
    """Global rank of the current process (0 when not distributed)."""

    if fabric is not None:
        for attr in ("global_rank", "rank"):
            rank = getattr(fabric, attr, None)
            if rank is not None:
                try:
                    return max(int(rank), 0)
                except (TypeError, ValueError):
                    pass

    if _dist_ready():
        try:
            return max(int(torch.distributed.get_rank()), 0)
        except Exception:  # pragma: no cover - defensive
            pass
    return _env_rank()


def is_distributed(fabric: Optional[Any] = None) -> bool:
    """Whether more than one process participates in the run."""

    return get_world_size(fabric) > 1


def is_main_process(fabric: Optional[Any] = None) -> bool:
    """Whether the current process is the main (rank 0) process."""

    if fabric is not None:
        flag = getattr(fabric, "is_global_zero", None)
        if isinstance(flag, bool):
            return flag
        flag = getattr(fabric, "is_global_zero", None)
        if flag is not None:
            try:
                return bool(flag)
            except Exception:  # pragma: no cover - defensive
                pass
    return get_rank(fabric) == 0


# --------------------------------------------------------------------------- #
# Collectives.
# --------------------------------------------------------------------------- #
def barrier(fabric: Optional[Any] = None) -> None:
    """Synchronize all processes; no-op in single-process mode."""

    if fabric is not None and hasattr(fabric, "barrier"):
        try:
            fabric.barrier()
            return
        except Exception:  # pragma: no cover - fall through to torch.dist
            pass

    if _dist_ready():
        try:
            torch.distributed.barrier()
        except Exception:  # pragma: no cover - defensive
            pass


def all_reduce_mean(
    tensor: torch.Tensor,
    fabric: Optional[Any] = None,
    *,
    average: bool = True,
) -> torch.Tensor:
    """Mean-reduce ``tensor`` across processes.

    Returns ``tensor`` unchanged when not distributed so callers can always
    chain ``all_reduce_mean(loss).item()`` in logging code.
    """

    if get_world_size(fabric) <= 1:
        return tensor

    if fabric is not None and hasattr(fabric, "all_reduce"):
        try:
            out = fabric.all_reduce(tensor)
            if average:
                out = out / get_world_size(fabric)
            return out
        except Exception:  # pragma: no cover - fall through
            pass

    if _dist_ready():
        try:
            reduced = tensor.clone()
            torch.distributed.all_reduce(reduced, op=torch.distributed.ReduceOp.SUM)
            if average:
                reduced = reduced / get_world_size(fabric)
            return reduced
        except Exception:  # pragma: no cover - defensive
            return tensor

    return tensor


def gather_tensor(
    tensor: torch.Tensor,
    fabric: Optional[Any] = None,
    *,
    concat: bool = True,
    dim: int = 0,
) -> Any:
    """Gather tensors from all processes.

    With ``concat=True`` (default) returns a single tensor concatenated along
    ``dim`` on every process.  With ``concat=False`` returns a list of tensors
    (Fabric semantics).  Falls back to the local tensor when not distributed.
    """

    if get_world_size(fabric) <= 1:
        return tensor if concat else [tensor]

    if fabric is not None and hasattr(fabric, "all_gather"):
        try:
            gathered = fabric.all_gather(tensor)
            if concat:
                return gathered
            if isinstance(gathered, torch.Tensor):
                return list(torch.unbind(gathered, dim=dim))
            return list(gathered)
        except Exception:  # pragma: no cover - fall through
            pass

    if _dist_ready():
        try:
            world = get_world_size(fabric)
            buffer = [torch.zeros_like(tensor) for _ in range(world)]
            torch.distributed.all_gather(buffer, tensor.contiguous())
            if concat:
                return torch.cat(buffer, dim=dim)
            return buffer
        except Exception:  # pragma: no cover - defensive
            return tensor if concat else [tensor]

    return tensor if concat else [tensor]


def broadcast_tensor(
    tensor: torch.Tensor,
    src: int = 0,
    fabric: Optional[Any] = None,
) -> torch.Tensor:
    """Broadcast ``tensor`` from process ``src`` to all processes."""

    if get_world_size(fabric) <= 1:
        return tensor

    if fabric is not None and hasattr(fabric, "broadcast"):
        try:
            return fabric.broadcast(tensor, src=src)
        except TypeError:
            try:
                return fabric.broadcast(tensor)
            except Exception:  # pragma: no cover - fall through
                pass
        except Exception:  # pragma: no cover - fall through
            pass

    if _dist_ready():
        try:
            torch.distributed.broadcast(tensor, src=src)
        except Exception:  # pragma: no cover - defensive
            pass
    return tensor


# --------------------------------------------------------------------------- #
# DataLoader / logging helpers.
# --------------------------------------------------------------------------- #
def distributed_loader(
    dataloader: Any,
    fabric: Optional[Any] = None,
    *,
    use_distributed_sampler: bool = True,
    **kwargs: Any,
) -> Any:
    """Wrap/tile a DataLoader for distributed sampling.

    Uses ``fabric.setup_dataloaders`` when available; otherwise falls back to a
    :class:`torch.utils.data.distributed.DistributedSampler` when
    ``torch.distributed`` is initialized.
    """

    world = get_world_size(fabric)
    if world <= 1:
        return dataloader

    if fabric is not None and hasattr(fabric, "setup_dataloaders"):
        try:
            return fabric.setup_dataloaders(
                dataloader,
                use_distributed_sampler=use_distributed_sampler,
                **kwargs,
            )
        except TypeError:
            try:
                return fabric.setup_dataloaders(dataloader, **kwargs)
            except Exception:  # pragma: no cover - fall through
                pass
        except Exception:  # pragma: no cover - fall through
            pass

    if _dist_ready():
        try:
            from torch.utils.data import DataLoader
            from torch.utils.data.distributed import DistributedSampler

            dataset = getattr(dataloader, "dataset", None)
            if dataset is not None:
                sampler = DistributedSampler(
                    dataset,
                    num_replicas=world,
                    rank=get_rank(fabric),
                    shuffle=bool(getattr(dataloader, "shuffle", False)),
                    drop_last=bool(getattr(dataloader, "drop_last", False)),
                )
                # Recreate the loader with the distributed sampler.
                return DataLoader(
                    dataset,
                    batch_size=getattr(dataloader, "batch_size", None) or 1,
                    sampler=sampler,
                    num_workers=getattr(dataloader, "num_workers", 0),
                    collate_fn=getattr(dataloader, "collate_fn", None),
                    pin_memory=getattr(dataloader, "pin_memory", False),
                    drop_last=bool(getattr(dataloader, "drop_last", False)),
                )
        except Exception:  # pragma: no cover - defensive
            pass

    return dataloader


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Strip ``DistributedDataParallel``/``Fabric`` wrappers from a module."""

    for attr in ("module", "_orig_mod"):
        inner = getattr(model, attr, None)
        if isinstance(inner, torch.nn.Module):
            model = inner
    return model


def rank_zero_print(*args: Any, fabric: Optional[Any] = None, **kwargs: Any) -> None:
    """``print`` only on the main process."""

    if is_main_process(fabric):
        print(*args, **kwargs)


def rank_zero_log(
    message: str,
    *,
    level: int = logging.INFO,
    fabric: Optional[Any] = None,
    logger_: Optional[logging.Logger] = None,
) -> None:
    """Log ``message`` only on the main process."""

    if is_main_process(fabric):
        (logger_ or logger).log(level, message)


# --------------------------------------------------------------------------- #
# Single-process fallback shim.
# --------------------------------------------------------------------------- #
class _SingleProcessFabric:  # pragma: no cover - trivial shim
    """Very small stand-in for ``lightning.fabric.Fabric`` on one device.

    Only the members used by the rest of this repository are implemented so the
    training loop can run unchanged on CPU or a single GPU when Lightning
    Fabric is not installed.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_rank = 0
    world_size = 1
    is_global_zero = True
    precision = "32-true"

    def __init__(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.seed_everything(seed)

    # -- launcher / setup ---------------------------------------------------
    def launch(self, *args: Any, **kwargs: Any) -> "_SingleProcessFabric":
        return self

    def __enter__(self) -> "_SingleProcessFabric":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def setup(self, *args: Any, **kwargs: Any) -> Any:
        if len(args) == 1:
            return args[0]
        return args or None

    def setup_module(self, module: torch.nn.Module, *a: Any, **kw: Any) -> torch.nn.Module:
        return module.to(self.device)

    def setup_optimizers(self, *args: Any, **kwargs: Any) -> Any:
        if len(args) == 1:
            return args[0]
        return args

    def setup_dataloaders(self, *args: Any, **kwargs: Any) -> Any:
        if len(args) == 1:
            return args[0]
        return args

    # -- runtime ------------------------------------------------------------
    def backward(self, tensor: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        tensor.backward(*args, **kwargs)
        return tensor

    def clip_gradients(
        self,
        module: torch.nn.Module,
        max_norm: Optional[float] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if max_norm is None:
            return
        torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm)

    # -- collectives (no-ops on one process) --------------------------------
    def all_reduce(self, tensor: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        return tensor

    def all_gather(self, tensor: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        return tensor

    def broadcast(self, tensor: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        return tensor

    def barrier(self, *args: Any, **kwargs: Any) -> None:
        return None

    # -- misc ---------------------------------------------------------------
    def seed_everything(self, seed: int) -> None:
        import random

        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def to_device(self, obj: Any) -> Any:
        if isinstance(obj, torch.Tensor):
            return obj.to(self.device)
        if isinstance(obj, torch.nn.Module):
            return obj.to(self.device)
        if isinstance(obj, (list, tuple)):
            return type(obj)(self.to_device(o) for o in obj)
        if isinstance(obj, dict):
            return {k: self.to_device(v) for k, v in obj.items()}
        return obj

    def save(self, *args: Any, **kwargs: Any) -> None:
        torch.save(*args, **kwargs)

    def log(self, *args: Any, **kwargs: Any) -> None:
        return None


# --------------------------------------------------------------------------- #
# Self-test.
# --------------------------------------------------------------------------- #
def _self_test() -> None:
    """Lightweight sanity checks for the single-process semantics."""

    fab = setup(accelerator="cpu", devices=1, cache=False)
    assert get_world_size(fab) == 1, get_world_size(fab)
    assert get_rank(fab) == 0, get_rank(fab)
    assert not is_distributed(fab)
    assert is_main_process(fab)

    t = torch.tensor([1.0, 2.0])
    assert torch.allclose(all_reduce_mean(t, fab), t)
    gathered = gather_tensor(t, fab)
    assert torch.allclose(gathered, t)
    assert torch.allclose(broadcast_tensor(t, 0, fab), t)
    barrier(fab)

    class _M(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.l = torch.nn.Linear(2, 2)

    m = _M()
    assert unwrap_model(m) is m
    rank_zero_print("[distributed] self-test OK", fabric=fab)
    print("si.utils.distributed self-test passed")


if __name__ == "__main__":  # pragma: no cover
    _self_test()
