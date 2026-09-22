"""Setup script for the RICE project.

RICE: A Refining Scheme for Reinforcement Learning with Explanation
(Cheng et al., ICML 2024, PMLR 235).

Install in development mode with::

    pip install -e .

The heavy RL dependencies (torch, stable-baselines3, gym, mujoco, ...) are
declared in ``requirements.txt``; this script keeps the *core* install light so
that the package can be imported (and the CLI introspected) on machines where
the optional simulators are unavailable.
"""

from __future__ import annotations

import os
import re

from setuptools import find_packages, setup

HERE = os.path.abspath(os.path.dirname(__file__))


# --------------------------------------------------------------------------- #
# Metadata helpers
# --------------------------------------------------------------------------- #
def _read(*parts: str) -> str:
    """Read a text file relative to the project root (empty string if missing)."""
    path = os.path.join(HERE, *parts)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()
    except OSError:
        return ""


def _version() -> str:
    """Extract ``__version__`` from ``rice/__init__.py``."""
    init = _read("rice", "__init__.py")
    match = re.search(r"""__version__\s*=\s*["']([^"']+)["']""", init)
    return match.group(1) if match else "0.1.0"


def _long_description() -> str:
    for candidate in ("README.md", "README.rst", "README.txt"):
        text = _read(candidate)
        if text:
            return text
    return (
        "RICE: A Refining Scheme for Reinforcement Learning with Explanation.\n\n"
        "Two-stage policy refinement: (1) train a mask network with vanilla PPO "
        "plus a blinding bonus to obtain step-level state importance and critical "
        "states, (2) refine the frozen pre-trained policy with PPO on a mixed "
        "initial state distribution plus a Random Network Distillation intrinsic "
        "reward."
    )


# Core runtime requirements (everything else lives in requirements.txt).
INSTALL_REQUIRES = [
    "numpy>=1.21",
    "PyYAML>=5.4",
]

# Optional dependency groups (``pip install -e ".[rl]"`` etc.).
EXTRAS_REQUIRE = {
    "rl": [
        "torch>=1.13",
        "stable-baselines3>=1.8,<2.4",
        "gym>=0.21",
    ],
    "mujoco": [
        "mujoco>=2.3",
        "mujoco-py>=2.1",
        "gym>=0.21",
        "stable-baselines3>=1.8,<2.4",
    ],
    "plot": [
        "matplotlib>=3.5",
        "pandas>=1.3",
    ],
    "apps": [
        "di-drive",
        "metadrive-simulator",
    ],
    "baselines": [
        "captum>=0.5",
    ],
    "dev": [
        "pytest>=7.0",
        "black>=22.0",
        "flake8>=4.0",
    ],
}
EXTRAS_REQUIRE["all"] = sorted(
    {item for group in EXTRAS_REQUIRE.values() for item in group}
)


setup(
    name="rice",
    version=_version(),
    description=(
        "RICE: A Refining Scheme for Reinforcement Learning with Explanation "
        "(ICML 2024)"
    ),
    long_description=_long_description(),
    long_description_content_type="text/markdown",
    author="RICE reproduction",
    license="MIT",
    python_requires=">=3.8",
    url="https://github.com/chengzelei/RICE",
    project_urls={
        "Paper": "https://proceedings.mlr.press/v235/",
        "Original code": "https://github.com/chengzelei/RICE",
        "StateMask baseline": "https://github.com/nuwuxian/RL-state_mask",
        "JSRL baseline": "https://github.com/steventango/jumpstart-rl",
    },
    packages=find_packages(
        include=["rice", "rice.*", "experiments", "experiments.*", "scripts*", "tools*"]
    ),
    py_modules=["main"],
    include_package_data=True,
    package_data={
        "rice": ["../configs/*.yaml"],
    },
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    entry_points={
        "console_scripts": [
            "rice=main:main",
            "rice-train-target=scripts.train_target:main",
            "rice-train-mask=scripts.train_mask:main",
            "rice-fidelity=scripts.run_fidelity:main",
            "rice-refine=scripts.run_refine:main",
            "rice-ablation=scripts.run_ablation:main",
            "rice-plot=scripts.plot_results:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords=[
        "reinforcement-learning",
        "explainable-rl",
        "state-mask",
        "random-network-distillation",
        "ppo",
        "policy-refinement",
    ],
    zip_safe=False,
)
