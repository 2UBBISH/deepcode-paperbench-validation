"""
RICE: Breaking Through Training Bottlenecks of RL with Explanation

Core components:
1. Mask Network (Algorithm 1) - improved StateMask explanation method
2. Refining Method (Algorithm 2) - mixed initial distribution + RND exploration
3. Fidelity Score computation
4. Baseline methods (PPO fine-tuning, JSRL, StateMask-R)

Paper: RICE: Breaking Through the Training Bottlenecks of Reinforcement
       Learning with Explanation (Cheng et al., ICML 2024)
"""

__version__ = "0.1.0"