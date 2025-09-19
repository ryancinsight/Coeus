#!/usr/bin/env python3
"""
Build script for PyCoeus

This script builds the PyCoeus Python package using maturin.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, cwd=None):
    """Run a command and handle errors."""
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd, 
            cwd=cwd, 
            check=True, 
            capture_output=True, 
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        sys.exit(1)

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import maturin
        print("✅ Maturin found")
    except ImportError:
        print("❌ Maturin not found. Installing...")
        run_command([sys.executable, "-m", "pip", "install", "maturin"])
    
    try:
        import numpy
        print(f"✅ NumPy found: {numpy.__version__}")
    except ImportError:
        print("❌ NumPy not found. Installing...")
        run_command([sys.executable, "-m", "pip", "install", "numpy"])

def build_package(release=True, install=False):
    """Build the PyCoeus package."""
    print("🚀 Building PyCoeus package...")
    
    # Change to pycoeus directory
    pycoeus_dir = Path(__file__).parent
    os.chdir(pycoeus_dir)
    
    # Build command
    cmd = ["maturin", "build"]
    if release:
        cmd.append("--release")
    if install:
        cmd = ["maturin", "develop"]
        if release:
            cmd.append("--release")
    
    run_command(cmd)
    
    if not install:
        print("✅ Build completed! Wheel files are in target/wheels/")
        print("To install, run: pip install target/wheels/pycoeus-*.whl")
    else:
        print("✅ Package built and installed in development mode!")

def clean():
    """Clean build artifacts."""
    print("🧹 Cleaning build artifacts...")
    pycoeus_dir = Path(__file__).parent
    
    # Remove target directory
    target_dir = pycoeus_dir / "target"
    if target_dir.exists():
        import shutil
        shutil.rmtree(target_dir)
        print("Removed target/")
    
    # Remove Python cache
    for cache_dir in pycoeus_dir.rglob("__pycache__"):
        import shutil
        shutil.rmtree(cache_dir)
        print(f"Removed {cache_dir}")
    
    print("✅ Clean completed!")

def main():
    """Main build script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build PyCoeus package")
    parser.add_argument("--debug", action="store_true", help="Build in debug mode")
    parser.add_argument("--install", action="store_true", help="Install after building")
    parser.add_argument("--clean", action="store_true", help="Clean build artifacts")
    
    args = parser.parse_args()
    
    if args.clean:
        clean()
        return
    
    check_dependencies()
    build_package(release=not args.debug, install=args.install)

if __name__ == "__main__":
    main()