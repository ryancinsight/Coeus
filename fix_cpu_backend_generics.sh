#!/bin/bash
# Fix CpuBackend generics systematically across NN crate

echo "Fixing CpuBackend generics in nn/src/functional.rs..."
sed -i 's/CpuBackend,/CpuBackend<T>,/g' nn/src/functional.rs

echo "Fixing CpuBackend generics in nn/src/transformer.rs..."
sed -i 's/CpuBackend,/CpuBackend<T>,/g' nn/src/transformer.rs

echo "Fixing CpuBackend generics in nn/src/upsample.rs..."
sed -i 's/CpuBackend,/CpuBackend<T>,/g' nn/src/upsample.rs

echo "Fixing CpuBackend generics in nn/src/init.rs..."
sed -i 's/CpuBackend,/CpuBackend<T>,/g' nn/src/init.rs

echo "Fixing CpuBackend generics in nn/src/functional_conv.rs..."
sed -i 's/CpuBackend,/CpuBackend<T>,/g' nn/src/functional_conv.rs

echo "Fixing CpuBackend generics in nn/src/loss/*.rs..."
sed -i 's/CpuBackend,/CpuBackend<T>,/g' nn/src/loss/*.rs

echo "CpuBackend generics fix complete"