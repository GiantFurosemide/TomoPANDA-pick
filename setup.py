#!/usr/bin/env python3
"""
Setup script for TomoPANDA-pick
A deep learning framework for 3D particle picking in cryo-electron tomography (cryoET) data.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="tomopanda-pick",
    version="0.1.0",
    author="TomoPANDA Team",
    author_email="contact@tomopanda.com",
    description="A deep learning framework for 3D particle picking in cryo-electron tomography (cryoET) data",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/TomoPANDA-pick",
    packages=find_packages(exclude=["tests", "notebooks", "docs", "example"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.1.0",
            "pytest-cov>=3.0.0",
            "pytest-mock>=3.8.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "isort>=5.10.0",
            "pre-commit>=2.20.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
            "ipykernel>=6.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "tomopanda-train=scripts.train:main",
            # Uncomment when other scripts are created:
            # "tomopanda-eval=scripts.evaluate:main",
            # "tomopanda-predict=scripts.predict:main",
            # "tomopanda-prepare=scripts.data_preparation:main",
            # "tomopanda-benchmark=scripts.benchmark:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": [
            "config/*.yaml",
            "config/model_configs/*.yaml",
        ],
    },
    zip_safe=False,
)
