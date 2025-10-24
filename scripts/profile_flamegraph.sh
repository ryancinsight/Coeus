#!/bin/bash

# Performance Profiling Script using Flamegraph
# This script generates flamegraphs for hot path identification

set -e

echo "🔥 Coeus Performance Flamegraph Profiling"
echo "========================================="

# Check if cargo-flamegraph is installed
if ! command -v cargo-flamegraph &> /dev/null; then
    echo "Installing cargo-flamegraph..."
    cargo install cargo-flamegraph
fi

# Create output directory
mkdir -p flamegraphs

echo "Generating flamegraph for NN benchmarks..."
cargo flamegraph --bin neural_networks --root --output flamegraphs/nn_benchmarks.svg

echo "Generating flamegraph for autograd operations..."
cargo flamegraph --bin autograd_demo --root --output flamegraphs/autograd_demo.svg

echo "Generating flamegraph for tensor operations..."
cargo flamegraph --bench neural_networks --bench bench_linear_forward::small_784_128 --root --output flamegraphs/linear_forward.svg

echo ""
echo "Flamegraphs generated in flamegraphs/ directory:"
echo "- nn_benchmarks.svg: Neural network operations"
echo "- autograd_demo.svg: Autograd computation graph"
echo "- linear_forward.svg: Linear layer forward pass"
echo ""
echo "Open these files in a web browser to analyze performance bottlenecks."