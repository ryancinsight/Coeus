# Coeus Backend

Compute device abstractions for the Coeus deep learning framework.

## Overview

This crate provides backend trait abstractions for executing tensor operations on different compute substrates (CPU, GPU, TPU, NPU). The architecture enables zero-cost abstraction through static dispatch while supporting adaptive backend selection for optimal performance across heterogeneous hardware.

## Features

- **Backend Trait**: Zero-cost device abstraction via static dispatch
- **CpuBackend**: Native CPU execution with SIMD-ready operations
- **GpuBackend**: Cross-platform GPU acceleration via wgpu (Vulkan/Metal/DX12/WebGPU)
- **TpuBackend**: Tensor Processing Unit support (placeholder for XLA integration)
- **NpuBackend**: Neural Processing Unit support (placeholder for hardware integration)
- **Adaptive Selection**: Workload-aware backend selection with performance learning
- **Distributed Coordination**: Multi-GPU backend coordination for distributed training
- **Memory Integration**: RL-based memory allocation across heterogeneous hardware
- **Sparse Operations**: GPU-accelerated sparse matrix operations
- **Thread-Safe**: All backends are `Send + Sync`

## Quick Start

### Basic Usage

```rust
use backend::{Backend, CpuBackend};
use dtype::float::Float32;

// Create CPU backend
let backend = CpuBackend::<Float32>::new();
assert_eq!(backend.device_name(), "cpu");
assert!(backend.supports("arithmetic"));

// Perform operations
let a = DenseStorage::from_vec(vec![1.0, 2.0, 3.0], &[3])?;
let b = DenseStorage::from_vec(vec![4.0, 5.0, 6.0], &[3])?;
let result = backend.add_dense(&a, &b)?;
```

### GPU Backend

```rust
use backend::GpuBackend;
use dtype::float::Float32;

// Create GPU backend (async initialization)
let backend = GpuBackend::<Float32>::new().await?;

// GPU operations use compute shaders
let result = backend.matmul_dense(&a, &b)?;
```

### Adaptive Backend Selection

```rust
use backend::{BackendSelector, WorkloadCharacteristics, OperationType};

// Create backend selector
let selector = BackendSelector::new();

// Define workload characteristics
let workload = WorkloadCharacteristics {
    total_elements: 1_000_000,
    access_pattern: MemoryAccessPattern::Dense,
    compute_intensity: 50.0,
    data_locality: DataLocality::High,
    operation_type: OperationType::MatrixMultiplication,
};

// Select optimal backend
let backend_type = selector.select_backend(&workload);
println!("Selected backend: {}", backend_type);
```

## Architecture

### Backend Trait Hierarchy

```
Backend<T: DataType>
├── Associated Types
│   ├── Data: DataType
│   └── Device: DeviceInfo
├── Device Queries
│   ├── device() -> &Device
│   ├── device_name() -> &str
│   └── supports(operation) -> bool
└── Operations
    ├── Arithmetic: add_dense, mul_dense, sub_dense
    ├── Matrix: matmul_dense, spmm_csr, spmv_csr
    ├── Activation: relu_dense, exp_dense, log_dense, sin_dense, cos_dense
    ├── Reduction: sum_dense, mean_dense, max_dense, min_dense
    ├── Sparse: coo_matmul_sparse, coo_add_sparse, coo_mul_sparse
    ├── Quantization: quantize, dequantize, quantized_matmul
    └── Convolution: conv2d_dense
```

### Backend Implementations

```
Backend Implementations
├── CpuBackend<T>           ✅ Fully Implemented
│   ├── Native CPU execution
│   ├── Element-wise operations
│   ├── Matrix multiplication (naive O(n³))
│   ├── Sparse operations (CSR, COO)
│   └── SIMD-ready (future optimization)
│
├── GpuBackend<T>           ✅ Fully Implemented
│   ├── wgpu-based cross-platform GPU
│   ├── Vulkan/Metal/DX12/WebGPU support
│   ├── Compute shader pipelines
│   ├── Async initialization
│   └── Sparse GPU operations (separate module)
│
├── TpuBackend<T>           🚧 Placeholder
│   ├── XLA compilation stubs
│   ├── Cloud TPU v4/v5 support (future)
│   └── Falls back to CPU
│
└── NpuBackend<T>           🚧 Placeholder
    ├── Neural engine support (future)
    ├── Edge TPU support (future)
    └── Falls back to CPU
```

### Module Structure

```
backend/src/
├── lib.rs                    # Backend trait + adaptive selection
├── cpu.rs                    # CPU backend implementation
├── gpu.rs                    # GPU backend implementation (wgpu)
├── npu.rs                    # NPU backend (placeholder)
├── tpu.rs                    # TPU backend (placeholder)
├── device.rs                 # Device enumeration and info traits
├── distributed.rs            # Distributed backend coordination
├── memory_integration.rs     # RL-based memory allocation
├── sparse_gpu.rs             # GPU sparse matrix operations
└── shaders/                  # WGSL compute shaders
    ├── element_wise.wgsl
    ├── binary_ops.wgsl
    ├── matmul.wgsl
    ├── fft.wgsl
    ├── clip_attention.wgsl
    └── clip_loss.wgsl
```

## Backend Selection Strategy

The `BackendSelector` uses workload characteristics to choose the optimal backend:

### Selection Criteria

1. **Operation Type**:
   - Element-wise: CPU for small, GPU for large
   - Matrix multiplication: GPU/TPU for large matrices
   - Convolution: GPU/NPU preferred
   - Sparse operations: GPU with sparse kernels

2. **Workload Size**:
   - Small (<10K elements): CPU preferred (lower overhead)
   - Medium (10K-100K): CPU or GPU depending on operation
   - Large (>100K): GPU/TPU preferred

3. **Memory Access Pattern**:
   - Dense: All backends supported
   - Sparse: GPU with sparse kernels or CPU
   - Strided: CPU or GPU with optimized kernels

4. **Performance Learning**:
   - Records actual execution times
   - Adjusts scores based on historical performance
   - Time-weighted decay for older records

### Example Selection Logic

```rust
// Element-wise operations
if workload.total_elements < 10_000 {
    BackendType::Cpu  // Low overhead for small ops
} else if workload.total_elements > 100_000 {
    BackendType::Gpu  // Parallel processing for large ops
}

// Matrix multiplication
if workload.total_elements > 1_000_000 {
    BackendType::Gpu  // GPU excels at large matmul
} else {
    BackendType::Cpu  // CPU competitive for smaller matrices
}

// Convolution
BackendType::Gpu  // GPU always preferred for convolutions
```

## Distributed Backend Coordination

For multi-GPU training, the `DistributedBackendCoordinator` manages backend selection across processes:

```rust
use backend::distributed::{DistributedBackendCoordinator, DistributedWorkloadCharacteristics};

// Create coordinator
let coordinator = DistributedBackendCoordinator::new(process_group);
coordinator.initialize().await?;

// Coordinate backend selection across GPUs
let decision = coordinator.coordinate_backend_selection(&workload).await?;

// Each process gets optimal backend assignment
let my_backend = decision.process_backends[&my_rank];
```

### Features

- **Fault Tolerance**: Handles backend failures gracefully
- **Memory-Aware**: Considers memory constraints across processes
- **Communication Optimization**: Minimizes cross-backend transfers
- **Performance Tracking**: Learns from distributed execution patterns

## Memory Integration

The RL-based memory allocation system optimizes memory usage across heterogeneous hardware:

```rust
use backend::memory_integration::{MemoryManager, MemoryAllocationRLAgent};

// Create memory manager with RL agent
let mut memory_manager = MemoryManager::new();

// Get optimal allocation action
let action = memory_manager.rl_agent
    .get_optimal_allocation_action(&pool)
    .await;

// Execute allocation
match action {
    MemoryAllocationAction::AllocateToBackend { backend_type, numa_node, size } => {
        // Allocate to specific backend/NUMA node
    }
    MemoryAllocationAction::TransferBetweenBackends { source, dest, size } => {
        // Transfer memory between backends
    }
    _ => {}
}
```

### Features

- **RL-Based Learning**: Q-learning for optimal allocation policies
- **NUMA-Aware**: Considers NUMA topology for allocation
- **Heterogeneous Pools**: Manages memory across CPU/GPU/TPU/NPU
- **Transfer Optimization**: Learns optimal transfer patterns
- **>90% Utilization Target**: Optimizes for high memory efficiency

## GPU Sparse Operations

Specialized GPU kernels for sparse matrix operations:

```rust
use backend::sparse_gpu::{GpuSparseBackend, ActivationType};

// Create sparse GPU backend
let sparse_backend = GpuSparseBackend::new().await?;

// Sparse-dense matrix multiplication
sparse_backend.spmm_gpu(
    csr_data, csr_indices, csr_indptr,
    dense_matrix, output,
    rows, cols
)?;

// Sparse matrix transpose
sparse_backend.sparse_transpose_gpu(
    input_data, input_indices, input_indptr,
    output_data, output_indices, output_indptr,
    rows, cols
)?;

// Gradient accumulation
sparse_backend.gradient_accumulate_gpu(
    grad_values, row_indices, col_indices,
    accumulated_grads, matrix_cols
)?;
```

## Device Information

Query device capabilities at runtime:

```rust
use backend::{Device, DeviceInfo};

// Get device info
let device_info = backend.device_info();

println!("Device: {}", device_info.name());
println!("Available: {}", device_info.is_available());
println!("Memory: {} GB", device_info.memory_gb());
println!("Compute Units: {}", device_info.compute_units());

// Check specific device type
match backend.device() {
    Device::Cpu => println!("Running on CPU"),
    Device::Gpu { name, vendor, backend, .. } => {
        println!("Running on GPU: {} ({})", name, backend);
    }
    Device::Tpu { name, generation, cores, .. } => {
        println!("Running on TPU: {} {}", name, generation);
    }
    Device::Npu { name, manufacturer, .. } => {
        println!("Running on NPU: {} by {}", name, manufacturer);
    }
}
```

## Performance Considerations

### Zero-Cost Abstraction

The Backend trait uses static dispatch for zero runtime overhead:

```rust
// Monomorphized at compile time
fn compute<B: Backend>(backend: &B, data: &[B::Data]) -> Result<B::Data> {
    // No virtual dispatch - direct function call
    backend.sum_dense(data)
}
```

### Backend-Specific Optimizations

- **CPU**: Direct memory access, cache-friendly algorithms
- **GPU**: Parallel compute shaders, coalesced memory access
- **TPU**: XLA compilation, systolic array optimization (future)
- **NPU**: Neural engine acceleration (future)

### Adaptive Selection Overhead

Backend selection adds minimal overhead:
- Selection: ~1-10 microseconds
- Caching: Amortizes cost over multiple operations
- Learning: Asynchronous, doesn't block execution

## Testing

```bash
# Run all backend tests
cargo test --package backend

# Run specific backend tests
cargo test --package backend --test cpu
cargo test --package backend --test gpu

# Run with GPU features
cargo test --package backend --features gpu

# Run distributed tests
cargo test --package backend --test distributed
```

## Feature Flags

```toml
[features]
default = ["std", "cpu", "gpu"]
std = []
cpu = []
gpu = ["wgpu", "pollster", "futures-intrusive", "bytemuck", "tokio", "futures"]
npu = ["std"]
tpu = ["std"]
serde = ["dep:serde"]
```

## Examples

See `examples/` directory for complete examples:
- `gpu_basic.rs` - Basic GPU operations
- `gpu_mnist_training.rs` - MNIST training on GPU
- `distributed_training.rs` - Multi-GPU distributed training
- `memory_benchmarking_suite.rs` - Memory allocation benchmarks

## Contributing

When adding new backend implementations:

1. Implement the `Backend` trait
2. Add device-specific optimizations
3. Provide fallback to CPU for unsupported operations
4. Add comprehensive tests
5. Update this documentation

## License

See workspace LICENSE file.

## References

- [ADR-003: Backend Architecture](../docs/adr/dynamic-graph-architecture.md)
- [ADR-034: GPU Backend Implementation](../docs/adr/034-gpu-backend-implementation.md)
- [ADR-035: Storage-GPU Production Readiness](../docs/adr/035-storage-gpu-production-readiness.md)

