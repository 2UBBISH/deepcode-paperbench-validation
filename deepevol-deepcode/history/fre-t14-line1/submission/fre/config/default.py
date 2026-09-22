"""Default hyperparameters for FRE (Functional Reward Encoding).

Values come from Table 3 (Appendix A) of "Zero-Shot Reinforcement Learning via
Functional Reward Encodings" (ICML 2024) plus the addendum clarifications.

Every hyperparameter that the paper leaves unspecified is annotated with a
comment describing the default we chose.
"""


class Config:
    # ------------------------------------------------------------------ #
    # Optimisation (Table 3)
    # ------------------------------------------------------------------ #
    batch_size = 512
    learning_rate = 0.0001
    optimizer = "adam"  # Table 3: Adam
    adam_betas = (0.9, 0.999)  # torch default (paper does not specify)
    grad_clip_norm = 10.0  # conservative default; paper does not specify

    # ------------------------------------------------------------------ #
    # Training schedule (Table 3, Algorithm 1)
    # ------------------------------------------------------------------ #
    # "Encoder Training Steps: 150,000 (1M for ExORL/Kitchen)"
    encoder_train_steps = 150_000
    # "Policy Training Steps: 850,000 (1M for ExORL/Kitchen)"
    policy_train_steps = 850_000
    log_freq = 1000

    # ------------------------------------------------------------------ #
    # FRE encoder / decoder (Table 3 + addendum "Additional Details on the
    # FRE architecture")
    # ------------------------------------------------------------------ #
    # "Reward Pairs to Encode: 32"
    num_encoder_samples = 32          # K
    # "Reward Pairs to Decode: 8"
    num_decoder_samples = 8           # K'
    # "Number of Reward Embeddings: 32"
    num_reward_bins = 32
    # Reward is rescaled to [0, 1], multiplied by 32 and floored -> bins.
    reward_bin_scale = 32.0
    # Appendix lists 128 but the addendum corrects this to 64 + 64 = 128.
    state_embedding_dim = 64
    reward_embedding_dim = 64
    # "The latent embedding (z) is 128-dimensional"
    latent_dim = 128
    # "the residual/attention activations are all 128-dimensional"
    transformer_width = 128
    # "the MLP block expands to 256, then back to 128"
    transformer_mlp_dim = 256
    # "Encoder Layers = [256, 256, 256, 256]" -> four transformer blocks, each
    # with a 128 -> 256 -> 128 MLP block.
    num_encoder_blocks = 4
    # "Encoder Attention Heads: 4"
    num_attention_heads = 4
    attention_dropout = 0.0  # paper does not use dropout
    # Whether to add learned positional encodings: the paper explicitly does NOT
    # (inputs are treated as an unordered set).
    use_positional_encoding = False
    use_causal_mask = False

    # "Decoder Network Layers: [512, 512, 512]"
    decoder_layers = (512, 512, 512)
    decoder_activation = "gelu"  # paper does not state; GELU (plan default)

    # "beta KL Weight: 0.01"  -- compression term of Equation 6
    beta_kl = 0.01
    # Clamp on the encoder log-std for numerical stability (paper does not
    # specify; -5.0 matches the GC-BC log-std clamp convention).
    log_std_min = -5.0
    log_std_max = 2.0

    # ------------------------------------------------------------------ #
    # Prior reward distribution (Table 3 + Appendix B)
    # ------------------------------------------------------------------ #
    # "Ratio of Goal-Reaching Rewards / Linear Rewards / Random MLP Rewards"
    prior_ratios = {"goal": 0.33, "linear": 0.33, "mlp": 0.33}
    # Hindsight experience relabelling distribution for goal sampling
    # (Appendix B / Park et al. 2023): current / future / random.
    her_current_prob = 0.2
    her_future_prob = 0.5
    her_random_prob = 0.3
    # Goal is considered reached when the (normalised) distance is below this.
    prior_goal_threshold = 0.5
    # Random linear functions: uniform vector in [-1, 1], each dimension zeroed
    # with probability 0.9 ("A random binary mask is applied with a 0.9 chance
    # to zero the vector at that dimension").
    linear_mask_zero_prob = 0.9
    # On AntMaze the XY coordinates are removed from random linear functions
    # because of the scale instability they introduce (Appendix B).
    linear_exclude_dims = {"antmaze": (0, 1)}
    # Random MLP functions: network of size (state_dim, 32, 1), normal init
    # scaled by the average layer dimension, tanh between layers, output clipped
    # to [-1, 1].
    mlp_hidden_dim = 32
    mlp_output_clip = 1.0

    # ------------------------------------------------------------------ #
    # IQL (Table 3)
    # ------------------------------------------------------------------ #
    iql_temperature = 3.0        # AWR temperature
    iql_expectile = 0.8
    discount = 0.88
    target_update_rate = 0.001   # Polyak coefficient
    rl_hidden_layers = (512, 512, 512)
    rl_activation = "relu"       # IQL convention; paper does not specify

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #
    num_eval_seeds = 5
    num_eval_episodes = 20
    # "Online evaluation is performed with a maximum length of 2000 steps"
    antmaze_max_episode_steps = 2000
    # "... with a maximum length of 1000 steps"
    exorl_max_episode_steps = 1000
    kitchen_max_episode_steps = 1000
    # "the X and Y coordinates are discretized into 32 bins"
    antmaze_xy_bins = 32
    # FB / SF require 5120 reward samples at evaluation, FRE uses 32.
    fre_eval_samples = 32
    fb_sf_eval_samples = 5120
    # OPAL privileged evaluation: 10 skills sampled from N(0, I).
    opal_num_skills = 10

    # ------------------------------------------------------------------ #
    # Domain / bookkeeping
    # ------------------------------------------------------------------ #
    domain = "antmaze"  # one of {antmaze, exorl-walker, exorl-cheetah, kitchen}
    seed = 0
    device = "cuda"

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if not hasattr(type(self), key):
                raise KeyError(f"Unknown config key: {key}")
            setattr(self, key, value)

    # ------------------------------------------------------------------ #
    # Convenience helpers
    # ------------------------------------------------------------------ #
    @property
    def prior_mixture(self):
        """Return (names, probabilities) for the prior reward mixture."""
        names = list(self.prior_ratios.keys())
        probs = [self.prior_ratios[n] for n in names]
        total = sum(probs)
        return names, [p / total for p in probs]

    def encoder_steps(self):
        """1M encoder steps for ExORL/Kitchen, 150k otherwise (Table 3)."""
        if self.domain.startswith("exorl") or self.domain == "kitchen":
            return 1_000_000
        return self.encoder_train_steps

    def policy_steps(self):
        """1M policy steps for ExORL/Kitchen, 850k otherwise (Table 3)."""
        if self.domain.startswith("exorl") or self.domain == "kitchen":
            return 1_000_000
        return self.policy_train_steps

    def to_dict(self):
        out = {}
        for key in dir(type(self)):
            if key.startswith("_"):
                continue
            value = getattr(self, key)
            if callable(value) or isinstance(value, property):
                continue
            out[key] = value
        return out
