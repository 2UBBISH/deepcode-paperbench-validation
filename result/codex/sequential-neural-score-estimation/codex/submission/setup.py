from setuptools import find_packages, setup

setup(
    name="snpse",
    version="0.1.0",
    description=(
        "Reproduction of 'Sequential Neural Score Estimation: Likelihood-Free Inference "
        "with Conditional Score Based Diffusion Models' (Sharrock, Simons, Liu & Beaumont, "
        "ICML 2024)"
    ),
    packages=find_packages(include=["snpse", "snpse.*", "experiments", "baselines"]),
    python_requires=">=3.8",
    install_requires=["torch>=2.0", "numpy>=1.21"],
)
