#!/usr/bin/env python3
"""PyCoeus Build and Install Script

This script handles the reliable building and installation of the PyCoeus
Python bindings. It ensures a clean build by clearing cargo caches when needed.

Usage:
    python scripts/build.py              # Normal build
    python scripts/build.py --clean      # Full clean build
    python scripts/build.py --check      # Check build without installing
    python scripts/build.py --test       # Build and run tests
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


def get_project_root():
    """Get the pycoeus project root directory."""
    script_dir = Path(__file__).parent
    return script_dir.parent


def run_command(cmd, cwd=None, check=True):
    """Run a command and return the result."""
    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=False, text=True)
    if check and result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    return result


def clean_cargo_cache(project_root):
    """Clean cargo cache for pycoeus to force full rebuild."""
    print("\n=== Cleaning cargo cache for pycoeus ===")
    workspace_root = project_root.parent
    
    # Clean specific crates to force rebuild
    crates_to_clean = ["pycoeus", "nn"]
    for crate in crates_to_clean:
        run_command(["cargo", "clean", "-p", crate], cwd=workspace_root, check=False)
    
    # Also remove any cached .pyd files
    pyd_files = list(project_root.glob("python/coeus/*.pyd"))
    for pyd in pyd_files:
        print(f"Removing cached {pyd}")
        pyd.unlink()


def check_venv(project_root):
    """Check if virtual environment exists and is activated."""
    venv_path = project_root / "venv"
    if not venv_path.exists():
        print("Virtual environment not found. Creating...")
        run_command([sys.executable, "-m", "venv", str(venv_path)], cwd=project_root)
    
    # Check if we're in the venv
    if sys.prefix == sys.base_prefix:
        print("\nWARNING: Virtual environment not activated!")
        print(f"Run: .\\venv\\Scripts\\Activate.ps1 (Windows) or source venv/bin/activate (Unix)")
        return False
    return True


def install_dependencies(project_root):
    """Install build dependencies."""
    print("\n=== Installing dependencies ===")
    run_command([sys.executable, "-m", "pip", "install", "-U", "pip", "maturin", "numpy", "pytest"])


def build_pycoeus(project_root, release=True):
    """Build pycoeus using maturin."""
    print("\n=== Building PyCoeus with maturin ===")
    cmd = ["maturin", "develop"]
    if release:
        cmd.append("--release")
    run_command(cmd, cwd=project_root)


def verify_installation():
    """Verify the installation works correctly."""
    print("\n=== Verifying installation ===")
    
    # Test basic import
    try:
        import coeus
        print(f"✓ coeus module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import coeus: {e}")
        return False
    
    # Test nn submodule
    try:
        import coeus.nn
        print(f"✓ coeus.nn module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import coeus.nn: {e}")
        return False
    
    # Test tensor creation
    try:
        import numpy as np
        arr = np.random.randn(2, 3).astype(np.float32)
        t = coeus.tensor(arr)
        print(f"✓ Tensor creation works: shape={t.shape}")
    except Exception as e:
        print(f"✗ Tensor creation failed: {e}")
        return False
    
    # Test nn modules
    nn_modules = [
        "Linear", "Conv2D", "BatchNorm2d", "LayerNorm", "Dropout",
        "RNN", "LSTM", "GRU", "ReLU", "GELU", "SiLU",
        "MaxPool1d", "MaxPool2d", "AvgPool1d", "AvgPool2d"
    ]
    missing = []
    for mod in nn_modules:
        if not hasattr(coeus.nn, mod):
            missing.append(mod)
    
    if missing:
        print(f"✗ Missing nn modules: {missing}")
        return False
    else:
        print(f"✓ All {len(nn_modules)} nn modules available")
    
    return True


def run_tests(project_root):
    """Run the parity test suite."""
    print("\n=== Running parity tests ===")
    result = run_command(
        [sys.executable, "-m", "pytest", "tests/", "-v"], 
        cwd=project_root,
        check=False
    )
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Build and install PyCoeus")
    parser.add_argument("--clean", action="store_true", help="Perform a clean build")
    parser.add_argument("--check", action="store_true", help="Check build without installing")
    parser.add_argument("--test", action="store_true", help="Run tests after building")
    parser.add_argument("--debug", action="store_true", help="Build in debug mode")
    args = parser.parse_args()
    
    project_root = get_project_root()
    print(f"Project root: {project_root}")
    
    # Check venv
    if not check_venv(project_root):
        print("\nPlease activate the virtual environment and try again.")
        sys.exit(1)
    
    # Install dependencies
    install_dependencies(project_root)
    
    # Clean if requested
    if args.clean:
        clean_cargo_cache(project_root)
    
    # Build
    if not args.check:
        build_pycoeus(project_root, release=not args.debug)
        
        # Verify
        if not verify_installation():
            print("\n✗ Installation verification failed!")
            sys.exit(1)
        
        print("\n✓ PyCoeus built and installed successfully!")
    
    # Run tests if requested
    if args.test:
        if run_tests(project_root):
            print("\n✓ All tests passed!")
        else:
            print("\n✗ Some tests failed!")
            sys.exit(1)


if __name__ == "__main__":
    main()
