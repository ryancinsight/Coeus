# Coeus Dtype Crate

[![Crates.io](https://img.shields.io/crates/v/coeus-dtype)](https://crates.io/crates/coeus-dtype)
[![docs.rs](https://img.shields.io/docsrs/coeus-dtype)](https://docs.rs/coeus-dtype)

Core data type abstractions for the Coeus deep learning framework, providing safe, efficient numeric types with zero-cost abstractions.

## Architecture

The dtype system implements a trait-based hierarchy where all numeric types implement the `DataType` trait:

```rust
pub trait DataType:
    Copy + Clone + Debug + Default + PartialEq + PartialOrd +
    Num + NumCast + NumOps + Zero + One +
    Sized + Send + Sync + 'static
{
    fn dtype() -> Dtype;
    fn size_bytes() -> usize { Self::dtype().size_bytes() }
    fn name() -> &'static str { Self::dtype().name() }
    // ... additional methods
}
```

## Supported Types

### Floating Point Types ✅
- **`Float32`**: 32-bit single precision (`f32` wrapper)
- **`Float64`**: 64-bit double precision (`f64` wrapper)
- **Half/BFloat16**: Planned for future implementation

### Extension Traits
- **`FloatExt`**: Mathematical functions (exp, log, sin, cos, erf, etc.)
- **`IntExt`**: Integer operations (bitwise, checked arithmetic)
- **`ComplexExt`**: Complex number operations (planned)

### Current Implementation Status

| Type | Status | Operations | Tests |
|------|--------|------------|-------|
| Float32 | ✅ Complete | Arithmetic, math functions, special values | ✅ All passing |
| Float64 | ✅ Complete | Arithmetic, math functions, special values | ✅ All passing |
| Integers | 📋 Planned | Checked arithmetic, bitwise ops | - |
| Complex | 📋 Planned | Complex arithmetic, FFT support | - |
| Quantized | 📋 Planned | Affine/scale quantization | - |

## Usage

### Basic Usage

```rust
use coeus_dtype::{Float32, Float64, DataType};

// Create values
let a = Float32::new(3.14);
let b = Float64::new(2.71);

// Type introspection
assert_eq!(Float32::dtype(), Dtype::Float32);
assert_eq!(Float32::size_bytes(), 4);
assert_eq!(Float32::name(), "float32");

// Arithmetic operations
let sum = a + b; // Works due to trait implementations
```

### Mathematical Operations

```rust
use coeus_dtype::{Float32, FloatExt};

// All standard mathematical functions
let x = Float32::new(1.0);
let exp_x = x.exp();
let sin_x = x.sin();
let sqrt_x = x.sqrt();
let erf_x = x.erf();

// Special values
let zero = Float32::zero();
let one = Float32::one();
let pi = Float32::PI();
```

### Error Handling

```rust
use coeus_dtype::{Float32, DataType};

let value = Float32::new(1.5);
// Type-safe casting with error handling
let int_result: Result<i32, _> = value.checked_cast_to::<i32>();
// Returns error due to precision loss
```

## Safety & Performance

### Memory Safety
- **Zero unsafe code** in the entire crate
- **Ownership-based** type system prevents memory corruption
- **Bounds checking** on all operations

### Performance
- **Zero-cost abstractions** via Rust generics and traits
- **Monomorphization** eliminates runtime dispatch overhead
- **SIMD-ready** architecture for future vectorization

### Correctness
- **Comprehensive test suite** with 11/11 tests passing
- **Property testing** infrastructure ready
- **Numerical stability** validations

## Mathematical Functions

The `FloatExt` trait provides complete coverage of standard mathematical functions:

### Exponential & Logarithmic
- `exp()`, `exp_m1()`, `exp2()`
- `ln()`, `log2()`, `log10()`

### Trigonometric
- `sin()`, `cos()`, `tan()`
- `asin()`, `acos()`, `atan()`, `atan2()`

### Hyperbolic
- `sinh()`, `cosh()`, `tanh()`
- `asinh()`, `acosh()`, `atanh()`

### Special Functions
- `erf()`, `erfc()` (error functions)
- `gamma()`, `lgamma()` (planned)

### Utility
- `sqrt()`, `cbrt()`, `powf()`
- `floor()`, `ceil()`, `round()`, `trunc()`
- `abs()`, `signum()`

## Testing

```bash
# Run all tests
cargo test

# Run with coverage
cargo tarpaulin --out Html

# Run specific tests
cargo test test_float32_arithmetic
```

### Test Coverage
- **Unit Tests**: Individual function correctness
- **Integration Tests**: Component interaction validation
- **Property Tests**: Invariant checking (framework ready)
- **Performance Tests**: Benchmarking infrastructure

## Dependencies

- `num-traits`: Core numeric abstractions
- `libm`: Pure Rust math functions (no_std compatible)
- `thiserror`: Typed error handling

## Feature Flags

- `default`: Includes `std` and `simd` features
- `std`: Enables standard library integration
- `simd`: SIMD acceleration support (future)
- `half`: 16-bit floating point types (future)
- `complex`: Complex number support (future)
- `quantized`: Quantization support (future)

## Architecture Notes

### Design Principles
1. **Type Safety**: Compile-time guarantees prevent runtime dtype errors
2. **Zero Cost**: All abstractions compile to optimal machine code
3. **Extensibility**: Easy to add new dtypes via trait implementation
4. **Safety**: No unsafe code, comprehensive error handling

### Future Extensions
- **16-bit types**: Half precision and bfloat16 for ML optimization
- **Complex numbers**: Full complex arithmetic support
- **Quantized types**: 8-bit quantization for efficient inference
- **SIMD acceleration**: Vectorized operations via safe intrinsics

## Integration

This crate serves as the foundation for the Coeus tensor system:

```
coeus/
├── dtype/     # This crate - core data types
├── storage/   # Memory layouts (dense, sparse)
├── backend/   # Compute backends (CPU, GPU)
├── tensor/    # Tensor<B<S<T>>> implementation
└── nn/        # Neural network layers
```

## Contributing

See the main Coeus repository for contribution guidelines. This crate follows the same standards:

- Zero unsafe code
- Comprehensive test coverage
- Clean, documented code
- Performance-focused design

## License

Licensed under the Apache License 2.0. See the main repository for details.

