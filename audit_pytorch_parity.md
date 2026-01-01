# Coeus Pytorch API Parity Audit

This audit evaluates the current Coeus workspace structure against the PyTorch 2.x API to identify missing foundational components.

## Current Workspace Coverage

| Coeus Crate | PyTorch Counterpart | Status |
| :--- | :--- | :--- |
| `tensor` | `torch.Tensor` | Core operations implemented |
| `autograd` | `torch.autograd` | Engine and custom functions implemented |
| `nn` | `torch.nn` | Extensive coverage (layers, functional, loss) |
| `optim` | `torch.optim` | Core optimizers implemented |
| `fft` | `torch.fft` | Recently added |
| `distributed` | `torch.distributed` | DDP and ProcessGroups implemented |
| `backend` | `torch.cuda` / `torch.mps` | Hardware abstraction via wgpu |
| `dtype` | `torch.dtype` | Basic type support |
| `storage` | `torch.UntypedStorage` | Dense and Sparse formats |
| `utils` | `torch.utils.data` | DataLoader and Dataset basic support |
| `hub` | `torch.hub` | Model management |
| `jit` | `torch.jit` | Foundational implementation |
| `profiling` | `torch.profiler` | Basic timing and metrics |

## Identified Gaps (Recommended New Crates)

To achieve full PyTorch API parity, the following crates are recommended for addition:

### 1. `coeus-linalg` (`torch.linalg`)
Currently, higher-level linear algebra is scattered or missing.
- **Goal**: Implement matrix decompositions, solvers, and norms.
- **Key Ops**: `inv`, `solve`, `det`, `eig`, `svd`, `cholesky`, `qr`, `norm`.

### 2. `coeus-signal` (`torch.signal`)
Advanced signal processing logic beyond basic FFT.
- **Goal**: Windows, filtering, and time-frequency analysis.
- **Key Ops**: `windows` (Hann, Hamming), `stft`, `istft`, `filters`.

### 3. `coeus-special` (`torch.special`)
Advanced mathematical functions required for scientific computing and complex ML architectures.
- **Goal**: High-precision implementations of niche functions.
- **Key Ops**: `bessel`, `gamma`, `beta`, `polygamma`, `digamma`, `ndtr`.

### 4. `coeus-distributions` (`torch.distributions`)
A dedicated API for probability distributions and sampling.
- **Goal**: Unified trait-based API for discrete and continuous distributions.
- **Key Classes**: `Normal`, `Bernoulli`, `Categorical`, `MultivariateNormal`, `KLDivergence`.

### 5. `coeus-sparse` (`torch.sparse`)
Promotion of sparse operations to a first-class citizen.
- **Goal**: Optimized sparse-sparse and sparse-dense matrix operations and coordinate conversions.
- **Refactoring**: Move high-level kernels (e.g., `matmul_sparse`, `matvec_mul`) and optimizer ops currently in the `storage` crate to this dedicated crate to separate memory layout from numerical logic.
- **Focus**: CSR, CSC, COO formats with backend-specific kernels.

### 6. `coeus-vision` (`torchvision`)
Dedicated vision library for image processing.
- **Goal**: High-level image operations and augmentation logic.
- **Focus**: `transforms` (Resize, Crop, ColorJitter), `io` (JPEG/PNG decoding), standard model architectures.

### 7. `coeus-data` (`torch.utils.data` expansions)
Enhanced data handling for multi-modal and large-scale datasets.
- **Goal**: Support for `iterable` datasets, streaming, and complex formats like WebDataset/TFRecord.

## Summary Recommendation

The most critical missing foundational piece is **`coeus-linalg`**, followed by **`coeus-signal`** given the project's focus on audio/tokenization. **`coeus-distributions`** is also essential for probabilistic modeling (e.g., VAEs, RL).

## Implementation Updates

- `coeus-signal`: Enabled `stft` module and implemented `stft_1d`/`istft_1d` with reflect padding, centered window placement, overlap-add reconstruction, optional normalization, and onesided spectrum handling.
- `coeus-signal`: Hardened window generation APIs to return `Result` (no error masking) and added unit tests for window coefficients.
- `coeus-fft`: Removed unused/unimplemented generic wrapper API and added a CPU FFT inverse roundtrip unit test.
