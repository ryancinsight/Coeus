# Tensor Backend

Coeus supports multiple compute backends through the `ComputeBackend` trait.
The backend is a type parameter on `Tensor<T, B>`, so switching backends is
a compile-time decision with no runtime overhead.

## Built-In Backends

| Backend type | Description |
|-------------|-------------|
| `MoiraiBackend` | Default CPU backend via Moirai parallel pool |
| `CudaBackend` | NVIDIA CUDA via `hephaestus-cuda` |
| `WgpuBackend` | Cross-platform GPU via `hephaestus-wgpu` |
| `RocmBackend` | AMD ROCm via `hephaestus-rocm` |
| `MetalBackend` | Apple Metal via `hephaestus-metal` |

## Selecting a Backend

```rust,ignore
use coeus::Tensor;
use coeus::backend::CudaBackend;

// Default CPU tensor
let a: Tensor<f32> = Tensor::zeros([256, 256])?;

// CUDA tensor (same API)
let b: Tensor<f32, CudaBackend> = Tensor::zeros([256, 256])?;
```

## `ComputeBackend` Trait

The trait abstracts over device allocation, data transfer, and op dispatch:

```rust,ignore
pub trait ComputeBackend {
    type Error: std::error::Error;
    type DeviceBuffer<T: Scalar>: StorageMut<T>;
    fn allocate<T: Scalar>(&self, len: usize)
        -> Result<Self::DeviceBuffer<T>, Self::Error>;
    fn copy_to_device<T: Scalar>(&self, host: &[T], device: &mut Self::DeviceBuffer<T>)
        -> Result<(), Self::Error>;
    fn copy_to_host<T: Scalar>(&self, device: &Self::DeviceBuffer<T>, host: &mut [T])
        -> Result<(), Self::Error>;
}
```

Allocation and transfer errors remain typed by the selected backend and
propagate through tensor constructors and operations.

Ops are dispatched through Hephaestus op traits, so the same autograd
graph works on any backend.

## COW Semantics

`Tensor` storage uses copy-on-write. Views (slices, transposes, reshapes)
share the underlying buffer without copying. A write to a shared buffer
triggers a copy automatically.
