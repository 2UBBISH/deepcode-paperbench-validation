#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RICE: Refining via Critical State Explanation

A method to refine reinforcement learning agents by training a mask network
to identify critical states (explanation), then using those states to construct
a mixed initial state distribution for further training with an exploration
bonus (RND), yielding improved policy performance.
"""

import os
from setuptools import setup, find_packages

# Read the README for long description
readme_path = os.path.join(os.path.dirname(__file__), "README.md")
long_description = ""
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()

# Read requirements
requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
install_requires = []
if os.path.exists(requirements_path):
    with open(requirements_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if line and not line.startswith("#"):
                # Skip optional/commented-out dependencies
                if not line.startswith("# "):
                    install_requires.append(line)

# Core dependencies that are always required
core_requires = [
    "numpy>=1.21.0",
    "scipy>=1.7.0",
    "torch>=1.10.0,<2.0.0",
    "gym>=0.21.0",
    "pyyaml>=6.0",
    "tqdm>=4.62.0",
    "matplotlib>=3.5.0",
    "seaborn>=0.11.0",
]

# Optional dependencies for specific domains
extras_require = {
    "mujoco": [
        "mujoco>=2.1.0",
        "stable-baselines3>=1.7.0",
    ],
    "metadrive": [
        "metadrive>=0.3.0",
        "stable-baselines3>=1.7.0",
    ],
    "malware": [
        "tianshou>=0.5.0",
        "lief>=0.12.0",
        "ember>=0.1.0",
    ],
    "cage2": [
        # CybORG requires separate installation from:
        # https://github.com/cage-challenge/cage-challenge-2
    ],
    "selfish_mining": [
        "stable-baselines3>=1.7.0",
    ],
    "baselines": [
        "stable-baselines3>=1.7.0",
    ],
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "black>=22.0.0",
        "isort>=5.10.0",
        "flake8>=5.0.0",
        "pre-commit>=2.20.0",
    ],
    "all": [
        "stable-baselines3>=1.7.0",
        "mujoco>=2.1.0",
        "metadrive>=0.3.0",
        "tianshou>=0.5.0",
    ],
}

setup(
    # Package metadata
    name="rice-rl",
    version="0.1.0",
    author="RICE Implementation",
    author_email="rice@example.com",
    description="RICE: Refining via Critical State Explanation for Reinforcement Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/rice",
    license="MIT",
    # Package structure
    packages=find_packages(
        include=[
            "rice",
            "rice.*",
            "baselines",
            "baselines.*",
            "experiments",
            "experiments.*",
            "configs",
            "configs.*",
        ]
    ),
    # Dependencies
    python_requires=">=3.8,<3.10",
    install_requires=core_requires,
    extras_require=extras_require,
    # Entry points for experiment scripts
    entry_points={
        "console_scripts": [
            # MuJoCo experiments
            "rice-mujoco-train-target=experiments.mujoco.train_target:main",
            "rice-mujoco-train-mask=experiments.mujoco.train_mask:main",
            "rice-mujoco-refine=experiments.mujoco.refine:main",
            "rice-mujoco-eval=experiments.mujoco.eval:main",
            # Autonomous Driving experiments
            "rice-ad-train-target=experiments.autonomous_driving.train_target:main",
            "rice-ad-train-mask=experiments.autonomous_driving.train_mask:main",
            "rice-ad-refine=experiments.autonomous_driving.refine:main",
            "rice-ad-eval=experiments.autonomous_driving.eval:main",
            # Selfish Mining experiments
            "rice-sm-train-target=experiments.selfish_mining.train_target:main",
            "rice-sm-train-mask=experiments.selfish_mining.train_mask:main",
            "rice-sm-refine=experiments.selfish_mining.refine:main",
            "rice-sm-eval=experiments.selfish_mining.eval:main",
            # CAGE2 experiments
            "rice-cage2-train-target=experiments.cage2.train_target:main",
            "rice-cage2-train-mask=experiments.cage2.train_mask:main",
            "rice-cage2-refine=experiments.cage2.refine:main",
            "rice-cage2-eval=experiments.cage2.eval:main",
            # Malware experiments
            "rice-malware-train-target=experiments.malware.train_target:main",
            "rice-malware-train-mask=experiments.malware.train_mask:main",
            "rice-malware-refine=experiments.malware.refine:main",
            "rice-malware-eval=experiments.malware.eval:main",
            # Baselines
            "rice-baseline-statemask=baselines.statemask:main",
            "rice-baseline-jsrl=baselines.jsrl:main",
            "rice-baseline-sil=baselines.sil:main",
            "rice-baseline-random=baselines.random_explanation:main",
        ],
    },
    # Package data
    include_package_data=True,
    package_data={
        "configs": ["*.yaml", "env_specific/*.yaml"],
    },
    # Classifiers
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Artificial Life",
    ],
    # Additional options
    zip_safe=False,
    platforms=["any"],
)