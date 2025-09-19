#!/usr/bin/env python3
"""
Setup script for PyCoeus - Python bindings for Coeus tensor library

This setup script uses maturin to build and install the Rust extension.
"""

import os
import sys
from pathlib import Path

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

# Read version from Cargo.toml
def get_version():
    cargo_toml = Path(__file__).parent / "Cargo.toml"
    if cargo_toml.exists():
        with open(cargo_toml, 'r') as f:
            for line in f:
                if line.startswith('version'):
                    # Extract version from workspace or package
                    version = line.split('=')[1].strip().strip('"')
                    return version
    return "0.1.0"

def get_long_description():
    readme = Path(__file__).parent / "README.md"
    if readme.exists():
        with open(readme, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

def main():
    # Check if maturin is available
    try:
        import maturin
    except ImportError:
        print("Error: maturin is required to build PyCoeus.")
        print("Install it with: pip install maturin")
        sys.exit(1)

    # Use maturin to build
    os.system("maturin develop")

    # Setup Python package
    setup(
        name="pycoeus",
        version=get_version(),
        description="PyTorch-compatible tensor library in Rust with Python bindings",
        long_description=get_long_description(),
        long_description_content_type="text/markdown",
        author="Coeus Team",
        author_email="team@coeus.dev",
        url="https://github.com/your-org/coeus",
        packages=["pycoeus"],
        package_dir={"pycoeus": "python"},
        python_requires=">=3.8",
        install_requires=[
            "numpy>=1.21.0",
            "tiktoken>=0.5.0",
            "torch>=1.9.0",
            "requests>=2.25.0",
        ],
        extras_require={
            "dev": [
                "pytest>=6.0",
                "pytest-benchmark>=4.0.0",
                "black>=21.0",
                "isort>=5.0",
                "mypy>=0.900",
                "psutil>=5.8.0",
            ],
            "docs": [
                "sphinx>=4.0",
                "sphinx-rtd-theme>=1.0",
            ],
        },
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "Programming Language :: Rust",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Software Development :: Libraries :: Python Modules",
        ],
        keywords="tensor machine-learning autograd pytorch rust gpt2",
        zip_safe=False,
    )

if __name__ == "__main__":
    main()
