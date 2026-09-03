from setuptools import setup, find_packages

setup(
    name="rice",
    version="0.1.0",
    description="RICE: Breaking Through Training Bottlenecks of RL with Explanation",
    author="RICE Reproduction",
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21",
        "torch>=2.0",
    ],
)