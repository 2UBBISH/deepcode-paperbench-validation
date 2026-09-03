"""
RICE Custom Environments Package

This package contains custom environment implementations for the RICE framework,
including:
- Selfish Mining (blockchain mining MDP)
- CAGE Challenge 2 (cybersecurity defense)
- Autonomous Driving (MetaDrive)
- Malware Mutation (MalConv)

These environments are used in Experiment IV and V of the RICE paper.
"""

__all__ = [
    "selfish_mining_env",
    "cage_env",
    "auto_driving_env",
    "malware_env",
]