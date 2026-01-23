# Coeus Dtype Crate

[![Crates.io](https://img.shields.io/crates/v/coeus-dtype)](https://crates.io/crates/coeus-dtype)
[![docs.rs](https://img.shields.io/docsrs/coeus-dtype)](https://docs.rs/coeus-dtype)

Core data type abstractions for the Coeus deep learning framework, providing safe, efficient numeric types with zero-cost abstractions.

## Architecture

### Design Principles

The dtype crate follows strict architectural principles:

1. **Single Source of Truth (SSOT)**: Each dtype operation is defined exactly once
2. **Separation of Concerns (SoC)**: Clear separation between traits, implementations, and utilities
3. **Single Responsibility Principle (SRP)**: Each module has one clear purpose
4. **Zero-cost abstractions**: All abstractions compile to optimal machine code
5. **Memory safety**: No unsafe code, comprehensive error handling

### Module Organization

```
dtype/src/
├── lib.rs           # Type enumeration and module exports (SSOT for Dtype enum)
├── traits.rs        # Core trait definitions (DataType, FloatExt, IntExt, ComplexExt)
├── float.rs         # Floating-point type implementations
├── int.rs           # Integer type implementations (macro-based to eliminate duplication)
├── complex.rs       # Complex number type implementations
├── quantized.rs     # Quantized type implementations
├── promotion.rs     # Type promotion rules (SSOT for promotion logic)
├── quantization.rs  # Quantization utilities (feature-gated)
└── error.rs         # Error type definitions (SSOT for error handling)
```

### Trait Hierarchy

The dtype system implements a trait-based hierarchy where all numeric types implement the `DataType` trait:

```rust
pub trait DataType:
    Copy + Clone + Debug + Default + PartialEq +
    Num + NumCast + NumOps + Zero + One +
    Sized + Send + Sync + 'static
{
    fn dtype() -> Dtype;
    fn size_bytes() -> usize { Self::dtype().size_bytes() }
    fn name() -> &'static str { Self::dtype().name() }
    fn cast_to<T: DataType>(self) -> Option<T>;
    fn checked_cast_to<T: DataType>(self) -> Result<T, DtypeError>;
    // ... additional methods
}
```

**Extension Traits** provide type-specific operations:
- **`FloatExt`**: Mathematical functions for floating-point types (erf, erfc, etc.)
- **`IntExt`**: Integer-specific operations (checked arithmetic, bitwise ops)
- **`ComplexExt`**: Complex number operations (conjugate, norm, arg)

## Type Promotion Strategy

The dtype crate implements **PyTorch/NumPy-compatible** type promotion for mixed-dtype operations.

### Promotion Hierarchy

```
bool < int8 < int16 < int32 < int64
     < uint8 < uint16 < uint32 < uint64
     < float16 < float32 < float64
     < complex32 < complex64
```

### Promotion Rules

1. **Identity**: Same type promotes to itself
2. **Float dominance**: Float + Integer → Float (larger float type)
3. **Complex dominance**: Complex + Any → Complex (with promoted real type)
4. **Size promotion**: Within same category, promote to larger size
5. **Mixed signedness**: Signed + Unsigned → Larger signed type
   - **Special case**: `i64 + u64 → Float64` (u64::MAX > i64::MAX)

### Examples

```rust
use dtype::{Dtype, promotion::promote};

// Same type
assert_eq!(promote(Dtype::Float32, Dtype::Float32), Dtype::Float32);

// Float dominance
assert_eq!(promote(Dtype::Float32, Dtype::Int32), Dtype::Float32);
assert_eq!(promote(Dtype::Int32, Dtype::Float64), Dtype::Float64);

// Size promotion
assert_eq!(promote(Dtype::Float32, Dtype::Float64), Dtype::Float64);
assert_eq!(promote(Dtype::Int8, Dtype::Int32), Dtype::Int32);

// Mixed signedness
assert_eq!(promote(Dtype::Int8, Dtype::UInt8), Dtype::Int16);
assert_eq!(promote(Dtype::Int32, Dtype::UInt32), Dtype::Int64);

// Critical edge case
assert_eq!(promote(Dtype::Int64, Dtype::UInt64), Dtype::Float64);
```

### Cast Safety

The `can_cast()` function determines if a cast is safe (no precision loss):

```rust
use dtype::{Dtype, promotion::can_cast};

// Safe casts (widening)
assert!(can_cast(Dtype::Int8, Dtype::Int32));
assert!(can_cast(Dtype::Float32, Dtype::Float64));
assert!(can_cast(Dtype::Int16, Dtype::Float32));

// Unsafe casts (narrowing or precision loss)
assert!(!can_cast(Dtype::Int32, Dtype::Int8));
assert!(!can_cast(Dtype::Float64, Dtype::Float32));
assert!(!can_cast(Dtype::Float32, Dtype::Int32)); // Fractional part loss
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
use dtype::{Float32, Float64, DataType, Dtype};

// Create values
let a = Float32::new(3.14);
let b = Float64::new(2.71);

// Type introspection
assert_eq!(Float32::dtype(), Dtype::Float32);
assert_eq!(Float32::size_bytes(), 4);
assert_eq!(Float32::name(), "float32");
assert!(Float32::is_floating_point());

// Arithmetic operations
let sum = a + Float32::new(1.0);
let product = a * Float32::new(2.0);
```

### Type Promotion

```rust
use dtype::{Dtype, promotion::promote};

// Automatic promotion for mixed-type operations
let result_type = promote(Dtype::Float32, Dtype::Int32);
assert_eq!(result_type, Dtype::Float32);

// Check cast safety
use dtype::promotion::can_cast;
assert!(can_cast(Dtype::Int8, Dtype::Int32));  // Safe widening
assert!(!can_cast(Dtype::Int32, Dtype::Int8)); // Unsafe narrowing
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
use dtype::{Float32, DataType, DtypeError};

let value = Float32::new(1.5);

// Type-safe casting with error handling
let int_result: Result<i32, DtypeError> = value.checked_cast_to::<i32>();
// Returns Err(DtypeError::CastError) due to precision loss

// Safe casting returns Option
let safe_cast = value.cast_to::<f64>();
assert!(safe_cast.is_some());
```

### Integer Operations

```rust
use dtype::{Int32, IntExt};

let a = Int32::new(100);
let b = Int32::new(50);

// Checked arithmetic (returns None on overflow)
let sum = a.checked_add(b);
assert_eq!(sum, Some(Int32::new(150)));

// Bitwise operations
let and_result = a.bitand(b);
let or_result = a.bitor(b);

// Bit manipulation
let leading = a.leading_zeros();
let ones = a.count_ones();
```

### Complex Numbers

```rust
#[cfg(feature = "complex")]
use dtype::{Complex32, Complex64};
use dtype::traits::ComplexExt;

let c = Complex32::new(3.0, 4.0);

// Complex operations
assert_eq!(c.re(), 3.0);
assert_eq!(c.im(), 4.0);
assert_eq!(c.norm(), 5.0);  // sqrt(3^2 + 4^2)

let conj = c.conj();
assert_eq!(conj.im(), -4.0);
```

## Safety & Performance

### Memory Safety
- **Zero unsafe code** in the entire crate
- **Ownership-based** type system prevents memory corruption
- **Bounds checking** on all operations
- **Typed errors** with descriptive messages

### Performance
- **Zero-cost abstractions** via Rust generics and traits
- **Monomorphization** eliminates runtime dispatch overhead
- **Transparent wrappers** (`#[repr(transparent)]`) for zero overhead
- **SIMD-ready** architecture for future vectorization

### Correctness
- **Comprehensive test suite** with 100% passing tests
- **Property testing** infrastructure ready
- **Numerical stability** validations
- **Edge case coverage** (overflow, division by zero, NaN handling)

### Type Safety
- **Compile-time guarantees** prevent runtime dtype errors
- **Explicit conversions** via `cast_to()` and `checked_cast_to()`
- **Type promotion** follows industry standards (PyTorch/NumPy)
- **No implicit conversions** that could lose precision

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
1. **Single Source of Truth (SSOT)**: Each operation defined exactly once
2. **Separation of Concerns (SoC)**: Traits, implementations, and utilities are separate
3. **Single Responsibility Principle (SRP)**: Each module has one clear purpose
4. **Type Safety**: Compile-time guarantees prevent runtime dtype errors
5. **Zero Cost**: All abstractions compile to optimal machine code
6. **Extensibility**: Easy to add new dtypes via trait implementation
7. **Safety**: No unsafe code, comprehensive error handling

### Implementation Patterns

#### Macro-Based Integer Implementation
Integer types use a macro to eliminate duplication while maintaining clarity:
```rust
impl_int_dtype!(Int32, i32, Dtype::Int32, signed);
impl_int_dtype!(UInt32, u32, Dtype::UInt32, unsigned);
```
This ensures consistent implementation across all integer types.

#### Feature-Gated Types
Complex and quantized types are feature-gated for flexibility:
```rust
#[cfg(feature = "complex")]
pub use num_complex::{Complex32, Complex64};
```

#### Transparent Wrappers
All wrapper types use `#[repr(transparent)]` for zero overhead:
```rust
#[repr(transparent)]
pub struct Float32(pub f32);
```

### Type Promotion Implementation

Type promotion follows a clear hierarchy defined in `promotion.rs`:
1. Check for identity (same type)
2. Apply dominance rules (complex > float > int)
3. Apply size rules (larger wins within category)
4. Handle edge cases (i64/u64 → Float64)

See `ARCHITECTURE_AUDIT.md` for detailed architectural analysis.

### Future Extensions
- **16-bit types**: Half precision and bfloat16 for ML optimization
- **Complex numbers**: Full complex arithmetic support (already implemented)
- **Quantized types**: 8-bit quantization for efficient inference (already implemented)
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

