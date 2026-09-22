"""Offline data utilities for FRE.

This subpackage contains the *unlabeled* offline trajectory buffers that FRE
consumes, together with the three benchmark dataset loaders used in the paper.

* :mod:`fre.data.replay` -- generic offline trajectory buffer.  It provides the
  uniform state sampling used by Algorithm 1 to draw the ``K`` encoding states
  :math:`\\{s^{e}_k\\} \\sim \\mathcal{D}` and the ``K'`` decoding states
  :math:`\\{s^{d}_k\\} \\sim \\mathcal{D}`, and the ``(s, a)`` batch sampler used
  by the IQL phase to train :math:`\\pi(a|s,z), Q(s,a,z), V(s,z)`.
* :mod:`fre.data.antmaze_dataset` -- ``antmaze-large-diverse-v2`` loader with the
  discretized X/Y preprocessing (32 bins) shared by FRE/GC-IQL/GC-BC/OPAL, and
  the maze-center start state used at evaluation time.
* :mod:`fre.data.exorl_dataset` -- ExORL (RND) walker/cheetah loader.  The
  physics augmentation (``horizontal_velocity``/``torso_upright``/
  ``torso_height`` for walker, ``speed`` for cheetah) is appended *only* for the
  encoder network, and every state dimension is normalized by its offline
  standard deviation.
* :mod:`fre.data.kitchen_dataset` -- D4RL Kitchen loader exposing the seven
  standard sparse subtasks as evaluation reward functions.

Paper bindings: Algorithm 1 ("Sample K states for encoder :math:`\\{s_k^e\\} \\sim
\\mathcal{D}`", "Sample K' states for decoder :math:`\\{s_k^d\\} \\sim
\\mathcal{D}`", "Train :math:`\\pi(a|s,z), Q(s,a,z), V(s,z)`") and Section 5.2 /
Appendix C (evaluation protocol and dataset descriptions).
"""

from fre.data.replay import (
    Batch,
    ReplayBuffer,
    Trajectory,
    UniformBatchSampler,
    make_replay_buffer,
)
from fre.data.antmaze_dataset import (
    ANTMAZE_DATASET_NAME,
    AntMazeDataset,
    discretize_xy,
    load_antmaze_dataset,
)
from fre.data.exorl_dataset import (
    EXORL_DATASET_NAMES,
    ExORLDataset,
    append_physics_features,
    load_exorl_dataset,
    normalize_observations,
)
from fre.data.kitchen_dataset import (
    KITCHEN_TASKS,
    KitchenDataset,
    load_kitchen_dataset,
)

__all__ = [
    # replay buffer / batch sampling
    "Batch",
    "ReplayBuffer",
    "Trajectory",
    "UniformBatchSampler",
    "make_replay_buffer",
    # AntMaze
    "ANTMAZE_DATASET_NAME",
    "AntMazeDataset",
    "discretize_xy",
    "load_antmaze_dataset",
    # ExORL
    "EXORL_DATASET_NAMES",
    "ExORLDataset",
    "append_physics_features",
    "load_exorl_dataset",
    "normalize_observations",
    # Kitchen
    "KITCHEN_TASKS",
    "KitchenDataset",
    "load_kitchen_dataset",
]
