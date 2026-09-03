"""Setup script for the RICE package."""
from setuptools import find_packages, setup

setup(
    name="rice",
    version="0.1.0",
    description="Refining Reinforcement Learning with Explanation (RICE)",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "gymnasium>=0.29.0",
        "stable-baselines3>=2.0.0",
        "matplotlib>=3.7.0",
    ],
)
