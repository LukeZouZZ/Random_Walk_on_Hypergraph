from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hypergraph-molecule-screening",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Molecular screening via MCMC random walk on hypergraphs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/hypergraph-molecule-screening",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "rdkit>=2022.03.1",
        "pyyaml>=6.0",
        "matplotlib>=3.4.0",
        "tqdm>=4.62.0",
    ],
    entry_points={
        "console_scripts": [
            "hgms-screen=scripts.run_screening:main",
        ],
    },
)
