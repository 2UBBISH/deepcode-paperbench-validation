#!/usr/bin/env python3
"""
Setup script for RICE: Refining via Critical State Explanation.

RICE is a method to refine reinforcement learning agents by identifying
critical states via a mask network (explanation) and using them to create
a mixed initial state distribution with exploration bonus, improving
performance and overcoming training bottlenecks.

Reference:
    RICE: Refining via Critical State Explanation
"""

import os
from setuptools import setup, find_packages

# Read the contents of README.md if it exists
readme_path = os.path.join(os.path.dirname(__file__), "README.md")
long_description = ""
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()

# Core dependencies
install_requires = [
    # Deep Learning
    "torch>=1.13.0",
    # RL Framework
    "stable-baselines3>=2.0.0",
    "gymnasium>=0.26.0",
    # MuJoCo (optional, for standard benchmarks)
    # "mujoco>=2.3.0",
    # Utilities
    "numpy>=1.21.0",
    "scipy>=1.7.0",
    "matplotlib>=3.5.0",
    "seaborn>=0.12.0",
    "pandas>=1.3.0",
    "pyyaml>=6.0",
    "tqdm>=4.64.0",
    # Additional environments (optional)
    # "metadrive>=0.4.0",       # Autonomous driving
    # "tianshou>=0.5.0",        # Malware mutation
]

# Extra dependencies for specific environments
extras_require = {
    "mujoco": [
        "mujoco>=2.3.0",
    ],
    "metadrive": [
        "metadrive>=0.4.0",
    ],
    "malware": [
        "tianshou>=0.5.0",
    ],
    "all": [
        "mujoco>=2.3.0",
        "metadrive>=0.4.0",
        "tianshou>=0.5.0",
    ],
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "black>=23.0.0",
        "isort>=5.12.0",
        "flake8>=6.0.0",
    ],
}

setup(
    name="rice-refining",
    version="0.1.0",
    author="RICE Authors",
    description="RICE: Refining via Critical State Explanation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rice-refining/rice",
    packages=find_packages(include=["rice", "rice.*", "baselines", "baselines.*", "envs", "envs.*", "experiments", "experiments.*", "config", "config.*"]),
    include_package_data=True,
    package_data={
        "config": ["*.yaml", "env_specific/*.yaml"],
    },
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "rice-train-agent=experiments.train_agent:main",
            "rice-train-mask=experiments.train_mask:main",
            "rice-refine=experiments.refine:main",
            "rice-evaluate=experiments.evaluate:main",
            "rice-statemask=baselines.statemask:main",
            "rice-jsrl=baselines.jsrl:main",
            "rice-sil=baselines.sil:main",
            "rice-random-explanation=baselines.random_explanation:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Artificial Life",
    ],
    keywords="reinforcement-learning, explanation, critical-states, refining, rl, ppo, mask-network",
)