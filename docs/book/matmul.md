# Matrix Multiplication

Matrix multiplication is one of the most performance-critical ops in Coeus.
It dispatches through Hephaestus `DenseProductOps` to vendor BLAS or
native GPU kernels.

## Basic GEMM

```rust,ignore
let c = coeus::ops::matmul(&a, &b)?;   // [M, K] x [K, N] -> [M, N]
```

With autograd:

```rust,ignore
let c: Var<f32> = coeus::autograd::matmul(&a_var, &b_var)?;
c.backward();
// a_var.grad() = dL/dA = dL/dC @ B^T
// b_var.grad() = dL/dB = A^T @ dL/dC
```

## Batched GEMM

```rust,ignore
let c = coeus::ops::bmm(&a, &b)?;  // [B, M, K] x [B, K, N] -> [B, M, N]
```

## `Transpose<T, B>`

`Transpose` is a lazy zero-copy view that swaps the last two axes.
Matmul accepts it directly to avoid materializing a transposed copy:

```rust,ignore
let c = coeus::ops::matmul(&a, &b.T())?;  // A @ B^T
```

## Backend Dispatch

- **CPU (`MoiraiBackend`)**: uses Moirai's parallel tile loop
- **CUDA**: cuBLAS `sgemm`/`dgemm`
- **wgpu**: WGSL tile-based GEMM kernel
- **ROCm**: rocBLAS `sgemm`
- **Metal**: MPSMatrixMultiplication

The dispatch path is selected by the `B: ComputeBackend` type parameter
at compile time.
