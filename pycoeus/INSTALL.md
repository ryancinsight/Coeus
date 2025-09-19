# PyCoeus Installation Guide

This guide explains how to build and install PyCoeus from source.

## Prerequisites

- **Rust**: Install from [rustup.rs](https://rustup.rs/)
- **Python**: 3.8 or later
- **Maturin**: Python package for building Rust extensions

## Quick Installation

### Option 1: Using the build script (Recommended)

```bash
# Navigate to the pycoeus directory
cd pycoeus

# Install in development mode (editable install)
python build.py --install

# Or build wheel for distribution
python build.py
pip install target/wheels/pycoeus-*.whl
```

### Option 2: Using maturin directly

```bash
# Navigate to the pycoeus directory
cd pycoeus

# Install maturin if not already installed
pip install maturin

# Development install (editable)
maturin develop --release

# Or build wheel
maturin build --release
pip install target/wheels/pycoeus-*.whl
```

### Option 3: Using pip (when available)

```bash
# This will be available once published to PyPI
pip install pycoeus
```

## Development Setup

For development, use an editable install:

```bash
cd pycoeus

# Install in development mode
maturin develop

# Or with optimizations
maturin develop --release
```

## Verification

Test your installation:

```python
import pycoeus as pc
import numpy as np

# Create a tensor
x = pc.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
print(f"Tensor shape: {x.shape()}")

# Create a simple model
linear = pc.nn.Linear(2, 1)
output = linear.forward(x)
print(f"Output shape: {output.shape()}")

# Test loss function
target = pc.tensor([[1.0], [2.0]])
loss_fn = pc.nn.MSELoss()
loss = loss_fn.forward(output, target)
print(f"Loss: {loss.data()}")

print("✅ PyCoeus installation successful!")
```

## Build Options

### Debug vs Release

```bash
# Debug build (faster compilation, slower runtime)
maturin develop

# Release build (slower compilation, faster runtime)
maturin develop --release
```

### Features

```bash
# Build with GPU support (if available)
maturin develop --release --features gpu

# Build with MKL support (if available)
maturin develop --release --features mkl
```

## Troubleshooting

### Common Issues

1. **Rust not found**
   ```bash
   # Install Rust
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source ~/.cargo/env
   ```

2. **Maturin not found**
   ```bash
   pip install maturin
   ```

3. **Build fails with linking errors**
   - Make sure you have a C compiler installed
   - On Windows: Install Visual Studio Build Tools
   - On macOS: Install Xcode Command Line Tools
   - On Linux: Install build-essential

4. **Import errors**
   ```bash
   # Make sure you're in the right environment
   which python
   pip list | grep pycoeus
   ```

### Clean Build

If you encounter issues, try a clean build:

```bash
python build.py --clean
python build.py --install
```

## Performance Tips

1. **Always use release builds for production**:
   ```bash
   maturin develop --release
   ```

2. **Enable CPU optimizations**:
   ```bash
   export RUSTFLAGS="-C target-cpu=native"
   maturin develop --release
   ```

3. **Use GPU acceleration** (if available):
   ```bash
   maturin develop --release --features gpu
   ```

## Uninstallation

```bash
pip uninstall pycoeus
```

## Getting Help

- Check the [GitHub Issues](https://github.com/coeus-ai/coeus/issues)
- Read the [Documentation](https://coeus.readthedocs.io)
- Join our [Discord Community](https://discord.gg/coeus)