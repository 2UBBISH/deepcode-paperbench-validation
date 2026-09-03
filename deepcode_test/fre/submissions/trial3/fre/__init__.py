"""
Functional Reward Encodings (FRE) for Zero-Shot Offline Reinforcement Learning.

FRE is a transformer-based variational auto-encoder that learns latent representations
of arbitrary reward functions from state-reward samples, enabling pre-training of a
generalist RL agent on random unsupervised rewards and zero-shot transfer to novel
downstream tasks.
"""

__version__ = "0.1.0"