#!/bin/bash

# Fix CpuBackend generic parameters throughout the codebase
# This script adds <T> to CpuBackend where it's missing

echo "Searching for CpuBackend instances that need fixing..."

# Find all .rs files and process them
# First pass: add <T> after CpuBackend
find . -name "*.rs" -type f -print0 | xargs -0 sed -i 's/\bCpuBackend\([^(><]\)/CpuBackend<T>\1/g'

# Second pass: fix struct field and parameter types
find . -name "*.rs" -type f -print0 | xargs -0 sed -i 's/CpuBackend, DenseStorage<T>, T/CpuBackend<T>, DenseStorage<T>, T/g'

# Third pass: fix nested generic types
find . -name "*.rs" -type f -print0 | xargs -0 sed -i 's/CpuBackend<[^>]*>, DenseStorage<[^>]*>/CpuBackend<T>, DenseStorage<T>/g'

echo "Fix complete. Run tests to verify."
