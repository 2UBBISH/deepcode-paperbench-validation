#!/usr/bin/env python
"""
Setup script for the Functional Reward Encodings (FRE) project.

FRE is a general unsupervised method for zero-shot offline RL that learns
a functional representation of arbitrary reward functions via a transformer-based
variational auto-encoder, enabling a single pre-trained agent to solve novel
downstream tasks given only a few reward-annotated state samples.

Reference:
    "Functional Reward Encodings (FRE) for Zero-Shot Offline Reinforcement Learning"
"""

import os
from setuptools import setup, find_packages


# ---------------------------------------------------------------------------
# Helper: read a file relative to this directory
# ---------------------------------------------------------------------------
def _read(filename: str) -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, filename), "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Parse requirements.txt into a list of dependency strings
# ---------------------------------------------------------------------------
def _parse_requirements(filename: str = "requirements.txt") -> list:
    """Return a list of package requirement strings from requirements.txt,
    ignoring comments, blank lines, and '-r' / '--index-url' lines."""
    reqs = []
    try:
        content = _read(filename)
    except FileNotFoundError:
        return reqs

    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-r") or line.startswith("--index-url"):
            continue
        # Remove inline comments
        if "#" in line:
            line = line[: line.index("#")].strip()
        reqs.append(line)
    return reqs


# ---------------------------------------------------------------------------
# Long description from README.md (if available)
# ---------------------------------------------------------------------------
try:
    long_description = _read("README.md")
except FileNotFoundError:
    long_description = (
        "Functional Reward Encodings (FRE) for Zero-Shot Offline "
        "Reinforcement Learning – a transformer-based VAE that learns "
        "functional representations of arbitrary reward functions."
    )


# ---------------------------------------------------------------------------
# Package setup
# ---------------------------------------------------------------------------
setup(
    # -- Package metadata ----------------------------------------------------
    name="fre",
    version="1.0.0",
    description="Functional Reward Encodings (FRE) for Zero-Shot Offline RL",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="FRE Authors",
    author_email="",
    url="https://github.com/example/fre",  # placeholder
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",

    # -- Package discovery ---------------------------------------------------
    packages=find_packages(
        include=[
            "configs",
            "data",
            "models",
            "rewards",
            "training",
            "evaluation",
            "utils",
            "scripts",
        ]
    ),
    include_package_data=True,
    package_data={
        "configs": ["*.yaml", "*.yml"],
    },

    # -- Dependencies --------------------------------------------------------
    install_requires=_parse_requirements("requirements.txt"),

    # -- Entry points (console scripts) --------------------------------------
    entry_points={
        "console_scripts": [
            "fre-train=scripts.train:main",
            "fre-evaluate=scripts.evaluate:main",
            "fre-demo=scripts.demo:main",
        ],
    },

    # -- Extras --------------------------------------------------------------
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=22.0",
            "isort>=5.10",
            "flake8>=4.0",
            "pre-commit>=2.20",
        ],
        "exorl": [
            "exorl",
            "dm_control",
        ],
        "wandb": [
            "wandb>=0.13.0",
        ],
    },
)