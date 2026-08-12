# Position in the Stack

## What Coeus Owns

Coeus is the Atlas tensor engine and machine learning layer. It owns:

- **Tensor algebra** — N-dimensional tensors, COW views, strided layouts
- **Automatic differentiation** — dynamic reverse-mode autograd graph
- **Neural network layers** — complete layer library (conv, attention, RNN, normalization, etc.)
- **Optimizers** — SGD, Adam, AdamW, AdaGrad, RmsProp
- **Model checkpointing** — `StateDict` / `StateArchive` via Consus

Coeus does **not** own GPU kernel implementations (Hephaestus), the runtime
scheduler (Moirai), storage I/O (Consus), or signal transforms (Apollo).

## Where Coeus Sits

`	ext
hephaestus (GPU op contracts)  +  moirai (parallel pool)
  |                                   |
  v                                   v
coeus-tensor / coeus-autograd / coeus-nn
  |
  v (consumed by)
kwavers (physics-informed NNs)  apollo (learned spectra)  application models
`

## Consumers

| Consumer | How Coeus is used |
|----------|------------------|
| `kwavers` | Physics-informed neural network models for acoustic simulation |
| `apollo` | Learned spectrum representations and spectral NNs |
| Application code | Model training, inference, and fine-tuning |

## Hephaestus Integration

Every compute-intensive op dispatches through a Hephaestus op trait.
The backend is selected by the `B: ComputeBackend` type parameter:
`CudaBackend` routes to `hephaestus-cuda`, `WgpuBackend` to
`hephaestus-wgpu`, etc.

## Consus Integration

`StateDict::save(path)` serializes model parameters to any Consus-supported
format (HDF5, Zarr, NPZ). `StateDict::load(path)` restores them.

## Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `cuda` | no | CUDA backend via hephaestus-cuda |
| `wgpu` | no | wgpu backend via hephaestus-wgpu |
| `rocm` | no | ROCm backend via hephaestus-rocm |
| `metal` | no | Metal backend via hephaestus-metal |
| `python` | no | PyO3 Python bindings (coeus-python) |
