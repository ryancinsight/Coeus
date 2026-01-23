# Coeus Quantization

Quantization algorithms and operations for the Coeus deep learning framework.

## Overview

This crate provides comprehensive quantization functionality extracted from the nn and dtype crates to ensure proper domain separation. It implements various quantization algorithms, calibration methods, and fake quantization techniques for both training and inference optimization.

## Features

- **Quantization Algorithms**: Symmetric, asymmetric, and dynamic quantization
- **Calibration Methods**: Entropy, percentile, and MSE-based calibration
- **Fake Quantization**: Training-time quantization simulation
- **Multiple Precisions**: 4-bit, 8-bit, and 16-bit quantization support
- **Hardware Optimization**: Optimized kernels for quantized operations
- **Domain Separation**: Clean separation from nn and dtype crates
- **B<S<T>> Architecture**: Generic over Backend, Storage, and DataType

## Architecture Overview

The quantization crate is organized into distinct functional areas:

```
┌─────────────────────────────────────────────────────────────┐
│                    Quantization Algorithms                   │
│  - Symmetric quantization (zero-point = 0)                  │
│  - Asymmetric quantization (arbitrary zero-point)           │
│  - Dynamic quantization (runtime calibration)               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Calibration Methods                       │
│  - Entropy calibration (KL divergence minimization)         │
│  - Percentile calibration (outlier handling)                │
│  - MSE calibration (mean squared error minimization)        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Fake Quantization                         │
│  - Linear layer fake quantization                           │
│  - Convolution layer fake quantization                      │
│  - Training-time quantization simulation                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Quantization Kernels                      │
│  - Quantize/dequantize operations                           │
│  - Quantized arithmetic operations                          │
│  - Hardware-optimized implementations                       │
└─────────────────────────────────────────────────────────────┘
```

## File Structure

The quantization crate uses a hierarchical structure for clear organization:

### Algorithms (`src/algorithms/`)

Core quantization algorithms:

```
quantization/src/algorithms/
├── symmetric.rs            - Symmetric quantization (zero_point = 0)
├── asymmetric.rs           - Asymmetric quantization (arbitrary zero_point)
├── dynamic.rs              - Dynamic quantization (runtime calibration)
├── core.rs                 - Common quantization utilities
└── mod.rs                  - Module exports
```

### Calibration (`src/calibration/`)

Calibration methods for determining quantization parameters:

```
quantization/src/calibration/
├── entropy.rs              - KL divergence-based calibration
├── percentile.rs           - Percentile-based calibration
├── mse.rs                  - MSE-based calibration
├── histogram.rs            - Histogram collection utilities
└── mod.rs                  - Module exports
```

### Fake Quantization (`src/fake_quantize/`)

Training-time quantization simulation:

```
quantization/src/fake_quantize/
├── linear.rs               - Fake quantization for linear layers
├── conv.rs                 - Fake quantization for convolution layers
├── activation.rs           - Fake quantization for activations
├── core.rs                 - Common fake quantization utilities
└── mod.rs                  - Module exports
```

### Quantization Types (`src/types/`)

Quantized data types and storage formats:

```
quantization/src/types/
├── qint4.rs                - 4-bit quantized integer type
├── qint8.rs                - 8-bit quantized integer type
├── qint16.rs               - 16-bit quantized integer type
├── scale_zero_point.rs     - Scale and zero-point parameters
└── mod.rs                  - Module exports
```

### Kernels (`src/kernels/`)

Optimized quantization operations:

```
quantization/src/kernels/
├── quantize.rs             - Quantization kernels
├── dequantize.rs           - Dequantization kernels
├── quantized_ops.rs        - Quantized arithmetic operations
├── simd.rs                 - SIMD-optimized implementations
└── mod.rs                  - Module exports
```

### Core Infrastructure (`src/`)

```
quantization/src/
├── lib.rs                  - Public API and module declarations
├── error.rs                - Error types for quantization operations
├── config.rs               - Quantization configuration
└── utils.rs                - Utility functions
```

## Usage Examples

### Symmetric Quantization

```rust
use coeus_quantization::algorithms::SymmetricQuantizer;
use coeus_tensor::Tensor;
use coeus_backend::CpuBackend;
use coeus_storage::DenseStorage;
use coeus_dtype::float::Float32;

type CpuTensor = Tensor<CpuBackend, DenseStorage<Float32>, Float32>;

// Create quantizer for 8-bit symmetric quantization
let quantizer = SymmetricQuantizer::<Float32>::new(8)?;

// Calibrate on sample data
let calibration_data = CpuTensor::from_vec(
    vec![1.0, 2.0, 3.0, -1.0, -2.0, -3.0], 
    &[6]
)?;
let scale = quantizer.calibrate(&calibration_data)?;

// Quantize tensor
let input = CpuTensor::from_vec(vec![1.5, -2.3, 0.8], &[3])?;
let quantized = quantizer.quantize(&input, scale)?;

// Dequantize back to float
let dequantized = quantizer.dequantize(&quantized, scale)?;
```

### Asymmetric Quantization

```rust
use coeus_quantization::algorithms::AsymmetricQuantizer;

// Create quantizer for 8-bit asymmetric quantization
let quantizer = AsymmetricQuantizer::<Float32>::new(8)?;

// Calibrate to get scale and zero_point
let (scale, zero_point) = quantizer.calibrate(&calibration_data)?;

// Quantize with asymmetric parameters
let quantized = quantizer.quantize(&input, scale, zero_point)?;
let dequantized = quantizer.dequantize(&quantized, scale, zero_point)?;
```

### Dynamic Quantization

```rust
use coeus_quantization::algorithms::DynamicQuantizer;

// Create dynamic quantizer (calibrates at runtime)
let quantizer = DynamicQuantizer::<Float32>::new(8)?;

// Quantize with automatic calibration
let quantized = quantizer.quantize_dynamic(&input)?;
let dequantized = quantizer.dequantize_dynamic(&quantized)?;
```

### Calibration Methods

```rust
use coeus_quantization::calibration::{EntropyCalibrator, PercentileCalibrator, MSECalibrator};

// Entropy-based calibration (KL divergence)
let entropy_calibrator = EntropyCalibrator::new(2048)?; // 2048 histogram bins
let scale = entropy_calibrator.calibrate(&calibration_data, 8)?;

// Percentile-based calibration (handles outliers)
let percentile_calibrator = PercentileCalibrator::new(99.9)?; // 99.9th percentile
let scale = percentile_calibrator.calibrate(&calibration_data, 8)?;

// MSE-based calibration (minimizes reconstruction error)
let mse_calibrator = MSECalibrator::new();
let scale = mse_calibrator.calibrate(&calibration_data, 8)?;
```

### Fake Quantization for Training

```rust
use coeus_quantization::fake_quantize::{FakeQuantizeLinear, FakeQuantizeConfig};

// Configure fake quantization
let config = FakeQuantizeConfig {
    bits: 8,
    symmetric: true,
    per_channel: false,
    observer_type: ObserverType::MinMax,
};

// Create fake quantization layer
let fake_quant = FakeQuantizeLinear::new(config)?;

// Use in training (simulates quantization without actual precision loss)
let weight = CpuTensor::randn(&[10, 5])?;
let fake_quantized_weight = fake_quant.forward(&weight)?;

// Gradients flow through fake quantization
fake_quantized_weight.backward()?;
```

### Quantized Operations

```rust
use coeus_quantization::kernels::{quantized_add, quantized_mul, quantized_matmul};
use coeus_quantization::types::{QInt8, ScaleZeroPoint};

// Quantized arithmetic operations
let a_quantized = QInt8::from_tensor(&a, scale_a, zero_point_a)?;
let b_quantized = QInt8::from_tensor(&b, scale_b, zero_point_b)?;

// Quantized addition
let result_params = ScaleZeroPoint::for_add(scale_a, zero_point_a, scale_b, zero_point_b);
let sum_quantized = quantized_add(&a_quantized, &b_quantized, result_params)?;

// Quantized matrix multiplication
let result_params = ScaleZeroPoint::for_matmul(scale_a, zero_point_a, scale_b, zero_point_b);
let matmul_quantized = quantized_matmul(&a_quantized, &b_quantized, result_params)?;
```

## Quantization Types

### QInt4 (4-bit Quantization)

```rust
use coeus_quantization::types::QInt4;

// 4-bit quantization (2 values per byte)
let qint4_tensor = QInt4::from_tensor(&float_tensor, scale, zero_point)?;
let reconstructed = qint4_tensor.to_tensor(scale, zero_point)?;

// Memory usage: 8x reduction compared to Float32
println!("Memory savings: {}x", std::mem::size_of::<f32>() / std::mem::size_of::<u8>() * 2);
```

### QInt8 (8-bit Quantization)

```rust
use coeus_quantization::types::QInt8;

// 8-bit quantization (1 value per byte)
let qint8_tensor = QInt8::from_tensor(&float_tensor, scale, zero_point)?;
let reconstructed = qint8_tensor.to_tensor(scale, zero_point)?;

// Memory usage: 4x reduction compared to Float32
println!("Memory savings: {}x", std::mem::size_of::<f32>() / std::mem::size_of::<u8>());
```

### QInt16 (16-bit Quantization)

```rust
use coeus_quantization::types::QInt16;

// 16-bit quantization (2 bytes per value)
let qint16_tensor = QInt16::from_tensor(&float_tensor, scale, zero_point)?;
let reconstructed = qint16_tensor.to_tensor(scale, zero_point)?;

// Memory usage: 2x reduction compared to Float32
println!("Memory savings: {}x", std::mem::size_of::<f32>() / std::mem::size_of::<u16>());
```

## Calibration Strategies

### Entropy Calibration

Minimizes KL divergence between original and quantized distributions:

```rust
use coeus_quantization::calibration::EntropyCalibrator;

let calibrator = EntropyCalibrator::new(2048)?; // histogram bins
let scale = calibrator.calibrate(&data, 8)?; // 8-bit quantization

// Best for: Preserving distribution shape, general-purpose quantization
```

### Percentile Calibration

Uses percentile clipping to handle outliers:

```rust
use coeus_quantization::calibration::PercentileCalibrator;

let calibrator = PercentileCalibrator::new(99.9)?; // 99.9th percentile
let scale = calibrator.calibrate(&data, 8)?;

// Best for: Data with outliers, robust quantization
```

### MSE Calibration

Minimizes mean squared error between original and quantized values:

```rust
use coeus_quantization::calibration::MSECalibrator;

let calibrator = MSECalibrator::new();
let scale = calibrator.calibrate(&data, 8)?;

// Best for: Minimizing reconstruction error, accuracy-critical applications
```

## Performance Optimizations

### SIMD Acceleration

```rust
use coeus_quantization::kernels::simd::{quantize_simd, dequantize_simd};

// SIMD-accelerated quantization (when available)
let quantized = quantize_simd(&float_data, scale, zero_point)?;
let dequantized = dequantize_simd(&quantized, scale, zero_point)?;
```

### Hardware-Specific Kernels

The quantization crate provides optimized kernels for different hardware:

- **CPU**: SIMD instructions (AVX2, NEON)
- **GPU**: CUDA/OpenCL kernels for parallel quantization
- **NPU**: Specialized quantization units
- **Mobile**: ARM NEON optimizations

## Integration with NN Layers

### Quantized Linear Layer

```rust
use coeus_nn::modules::Linear;
use coeus_quantization::fake_quantize::FakeQuantizeLinear;

// Create quantized linear layer
let linear = Linear::new(784, 128, true)?;
let fake_quant = FakeQuantizeLinear::new(FakeQuantizeConfig::default())?;

// Forward pass with fake quantization
let quantized_weight = fake_quant.forward(linear.weight())?;
let output = linear.forward_with_weight(&input, &quantized_weight)?;
```

### Post-Training Quantization

```rust
use coeus_quantization::algorithms::AsymmetricQuantizer;

// Load pre-trained model
let mut model = load_pretrained_model()?;

// Calibrate on representative data
let calibration_dataset = load_calibration_data()?;
let quantizer = AsymmetricQuantizer::new(8)?;

// Quantize model weights
for param in model.parameters_mut() {
    let (scale, zero_point) = quantizer.calibrate(param.data(), 8)?;
    let quantized = quantizer.quantize(param.data(), scale, zero_point)?;
    param.set_data(quantized.to_tensor(scale, zero_point)?);
}
```

## Testing

```bash
# Run all quantization tests
cargo test --package quantization

# Run specific test categories
cargo test --package quantization --test algorithms
cargo test --package quantization --test calibration
cargo test --package quantization --test fake_quantize

# Run benchmarks
cargo bench --package quantization
```

**Test Coverage**: Comprehensive test suite covering all quantization methods and precision levels

## Benchmarks

Performance comparison of quantization methods:

```bash
# Run quantization benchmarks
cargo bench --package quantization

# Example results (on modern CPU):
# Symmetric 8-bit:    ~2.5 GB/s quantization throughput
# Asymmetric 8-bit:   ~2.2 GB/s quantization throughput
# Dynamic 8-bit:      ~1.8 GB/s quantization throughput
# SIMD acceleration:  ~3.5x speedup on AVX2
```

## Memory Usage

Quantization provides significant memory savings:

| Precision | Memory Usage | Compression Ratio |
|-----------|--------------|-------------------|
| Float32   | 4 bytes      | 1x (baseline)     |
| QInt16    | 2 bytes      | 2x                |
| QInt8     | 1 byte       | 4x                |
| QInt4     | 0.5 bytes    | 8x                |

## Error Handling

The quantization crate uses comprehensive error handling:

```rust
use coeus_quantization::error::QuantizationError;

match quantizer.quantize(&input, scale, zero_point) {
    Ok(quantized) => println!("Quantization successful"),
    Err(QuantizationError::InvalidScale(scale)) => {
        eprintln!("Invalid scale parameter: {}", scale);
    }
    Err(QuantizationError::UnsupportedPrecision(bits)) => {
        eprintln!("Unsupported precision: {} bits", bits);
    }
    Err(e) => eprintln!("Quantization error: {:?}", e),
}
```

## Contributing

When adding new quantization functionality:

1. **Algorithms**: Add to `src/algorithms/`
2. **Calibration**: Add to `src/calibration/`
3. **Fake Quantization**: Add to `src/fake_quantize/`
4. **Types**: Add to `src/types/`
5. **Kernels**: Add to `src/kernels/`
6. **Tests**: Add comprehensive tests for all new functionality
7. **Benchmarks**: Add performance benchmarks
8. **Documentation**: Update this README

### Guidelines

- **Domain Separation**: Keep quantization logic within this crate
- **Generic Architecture**: Maintain B<S<T>> generics where applicable
- **Error Handling**: Use Result types for all fallible operations
- **Performance**: Provide SIMD-optimized implementations
- **Testing**: Include both unit and integration tests

## See Also

- [Coeus NN](../nn/) - Neural network layers and operations
- [Coeus Tensor](../tensor/) - Tensor operations and automatic differentiation
- [Coeus Storage](../storage/) - Memory storage abstractions
- [Coeus Backend](../backend/) - Compute backend implementations
- [Coeus DType](../dtype/) - Data type abstractions

## License

See workspace LICENSE file.