"""Training / evaluation configuration objects.

Defaults follow Appendix A of the paper:

===========================  ==========================
Batch Size                   512
Encoder Training Steps       150,000 (1M for ExORL/Kitchen)
Policy Training Steps        850,000 (1M for ExORL/Kitchen)
Reward Pairs to Encode       32
Reward Pairs to Decode       8
Ratio of Goal-Reaching       0.33
Ratio of Linear              0.33
Ratio of Random MLP          0.33
Number of Reward Embeddings  32
Optimizer                    Adam
Learning Rate                0.0001
RL Network Layers            [512, 512, 512]
Decoder Network Layers       [512, 512, 512]
Encoder MLP dim              256 (4 transformer layers, 4 heads, 128-d residual)
Beta (KL weight)             0.01
Target Update Rate           0.001
Discount Factor              0.88
AWR Temperature              3.0
IQL Expectile                0.8
===========================  ==========================
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class TrainConfig:
    """Full configuration for unsupervised FRE pre-training."""

    # -- domain -------------------------------------------------------------------
    domain: str = "antmaze"
    env_name: str = "antmaze-large-diverse-v2"
    exorl_algo: str = "rnd"
    exorl_data_dir: Optional[str] = None

    # -- prior reward distribution (Sections 4.2, 5.3, 5.4) -----------------------
    prior_name: str = "FRE-all"
    prior_ratios: Optional[Dict[str, float]] = None
    use_hint_priors: bool = False
    hint_ratio: float = 0.5
    goal_threshold: float = 2.0
    goal_dims: Optional[List[int]] = None
    linear_excluded_dims: Optional[List[int]] = None

    # -- FRE architecture ---------------------------------------------------------
    z_dim: int = 128
    state_embed_dim: int = 64
    reward_embed_dim: int = 64
    d_model: int = 128
    encoder_layers: int = 4
    encoder_heads: int = 4
    encoder_mlp_dim: int = 256
    num_reward_bins: int = 32
    decoder_hidden_dims: Tuple[int, ...] = (512, 512, 512)
    beta: float = 0.01
    num_encode_pairs: int = 32
    num_decode_pairs: int = 8

    # -- RL (IQL) -----------------------------------------------------------------
    rl_hidden_dims: Tuple[int, ...] = (512, 512, 512)
    discount: float = 0.88
    expectile: float = 0.8
    awr_temperature: float = 3.0
    target_update_rate: float = 0.001

    # -- optimisation -------------------------------------------------------------
    batch_size: int = 512
    learning_rate: float = 1e-4
    encoder_steps: int = 150_000
    policy_steps: int = 850_000
    log_interval: int = 1000
    eval_interval: int = 50_000
    checkpoint_interval: int = 50_000

    # -- misc ---------------------------------------------------------------------
    seed: int = 0
    device: str = "cuda"
    output_dir: str = "runs"
    run_name: Optional[str] = None
    # Discretise the (x, y) coordinates into bins (AntMaze preprocessing).
    discretize_xy: bool = True
    num_xy_bins: int = 32

    def __post_init__(self) -> None:
        # ExORL goal-reaching uses the normalised observation distance with a
        # threshold of 0.1 (addendum); the training-step schedule is applied by
        # ``fre.experiment.default_config`` so that explicit overrides win.
        if self.domain in ("walker", "cheetah"):
            self.goal_threshold = 0.1 if self.goal_threshold == 2.0 else self.goal_threshold

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @property
    def total_steps(self) -> int:
        return self.encoder_steps + self.policy_steps


@dataclass
class EvalConfig:
    """Evaluation configuration (Section 5.2)."""

    num_episodes: int = 20          # "mean over twenty evaluation episodes"
    num_seeds: int = 5              # "each agent is trained using five random seeds"
    num_encode_pairs: int = 32      # FRE uses 32 (state, reward) samples at test time
    baseline_samples: int = 5120    # FB / SF use 5120 reward samples
    opal_num_skills: int = 10       # privileged OPAL evaluation
    use_opensimplex: bool = True
    batch_size: int = 512
    output_dir: str = "eval"
