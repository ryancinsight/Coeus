# Dtype Usage Guide

**Version:** 0.1.0  
**Last Updated:** January 14, 2026

## Table of Contents

1. [Introduction](#introduction)
2. [Basic Usage](#basic-usage)
3. [Type Promotion](#type-promotion)
4. [Working with Floats](#working-with-floats)
5. [Working with Integers](#working-with-integers)
6. [Working with Complex Numbers](#working-with-complex-numbers)
7. [Error Handling](#error-handling)
8. [Advanced Patterns](#advanced-patterns)
9. [Performance Tips](#performance-tips)
10. [Common Pitfalls](#common-pitfalls)

## Introduction

The dtype crate provides a comprehensive set of numeric types for the Coeus deep learning framework. All types implement the `DataType` trait, providing a uniform interface for tensor operations.

### Key Features

- **Type safety**: Compile-time guarantees prevent runtime errors
- **Zero-cost abstractions**: No runtime overhead
- **PyTorch/NumPy compatibility**: Familiar type promotion rules
- **Comprehensive operations**: Math functions, checked arithmetic, bitwise ops
- **Memory safety**: Zero unsafe code

## Basic Usage

### Creating Values

```rust
use dtype::{Float32, Float64, Int32, UInt8};

// Floating-point types
let f32_val = Float32::new(3.14);
let f64_val = Float64::new(2.71828);

// Integer types
let i32_val = Int32::new(42);
let u8_val = UInt8::new(255);

// Access inner values
assert_eq!(f32_val.get(), 3.14);
assert_eq!(i32_val.get(), 42);
```

### Type Introspection

```rust
use dtype::{Float32, Dtype, DataType};

// Get dtype enum
assert_eq!(Float32::dtype(), Dtype::Float32);

// Get size in bytes
assert_eq!(Float32::size_bytes(), 4);

// Get type name
assert_eq!(Float32::name(), "float32");

// Check type category
assert!(Float32::is_floating_point());
assert!(!Float32::is_integer());
assert!(!Float32::is_complex());
```

### Arithmetic Operations

```rust
use dtype::Float32;

let a = Float32::new(10.0);
let b = Float32::new(3.0);

// Basic arithmetic
let sum = a + b;        // 13.0
let diff = a - b;       // 7.0
let product = a * b;    // 30.0
let quotient = a / b;   // 3.333...
let remainder = a % b;  // 1.0

// Negation
let neg = -a;           // -10.0

// Compound assignment
let mut x = Float32::new(5.0);
x += Float32::new(3.0); // x = 8.0
x *= Float32::new(2.0); // x = 16.0
```

## Type Promotion

### Understanding Promotion

Type promotion determines the result type when operating on different dtypes:

```rust
use dtype::{Dtype, promotion::promote};

// Float dominance
assert_eq!(promote(Dtype::Float32, Dtype::Int32), Dtype::Float32);

// Size promotion
assert_eq!(promote(Dtype::Int8, Dtype::Int32), Dtype::Int32);
assert_eq!(promote(Dtype::Float32, Dtype::Float64), Dtype::Float64);

// Complex dominance
assert_eq!(promote(Dtype::Complex32, Dtype::Float64), Dtype::Complex32);
```

### Promotion Hierarchy

```
Lowest  → Highest Priority
─────────────────────────────
int8    → int16 → int32 → int64
uint8   → uint16 → uint32 → uint64
float16 → float32 → float64
complex32 → complex64
```

### Mixed Signedness

```rust
use dtype::{Dtype, promotion::promote};

// Mixed signed/unsigned promotes to larger signed
assert_eq!(promote(Dtype::Int8, Dtype::UInt8), Dtype::Int16);
assert_eq!(promote(Dtype::Int16, Dtype::UInt16), Dtype::Int32);
assert_eq!(promote(Dtype::Int32, Dtype::UInt32), Dtype::Int64);

// Critical edge case: i64/u64 cannot safely promote to i64
// (u64::MAX > i64::MAX), so promotes to Float64
assert_eq!(promote(Dtype::Int64, Dtype::UInt64), Dtype::Float64);
```

### Checking Cast Safety

```rust
use dtype::{Dtype, promotion::can_cast};

// Safe casts (widening)
assert!(can_cast(Dtype::Int8, Dtype::Int32));
assert!(can_cast(Dtype::Float32, Dtype::Float64));
assert!(can_cast(Dtype::Int16, Dtype::Float32));

// Unsafe casts (narrowing or precision loss)
assert!(!can_cast(Dtype::Int32, Dtype::Int8));
assert!(!can_cast(Dtype::Float64, Dtype::Float32));
assert!(!can_cast(Dtype::Float32, Dtype::Int32)); // Loses fractional part
```

## Working with Floats

### Mathematical Functions

```rust
use dtype::{Float32, FloatExt};
use num_traits::Float;

let x = Float32::new(1.0);

// Exponential and logarithmic
let exp_x = x.exp();        // e^x
let ln_x = x.ln();          // natural log
let log10_x = x.log10();    // base-10 log

// Trigonometric
let sin_x = x.sin();
let cos_x = x.cos();
let tan_x = x.tan();

// Inverse trigonometric
let asin_x = x.asin();
let acos_x = x.acos();
let atan_x = x.atan();

// Hyperbolic
let sinh_x = x.sinh();
let cosh_x = x.cosh();
let tanh_x = x.tanh();

// Special functions (FloatExt trait)
let erf_x = x.erf();        // Error function
let erfc_x = x.erfc();      // Complementary error function

// Power and roots
let sqrt_x = x.sqrt();
let cbrt_x = x.cbrt();
let pow_x = x.powf(Float32::new(2.0));
```

### Special Values

```rust
use dtype::Float32;
use num_traits::Float;

// Create special values
let nan = Float32::nan();
let inf = Float32::infinity();
let neg_inf = Float32::neg_infinity();
let neg_zero = Float32::neg_zero();

// Check for special values
let x = Float32::new(1.0);
assert!(!x.is_nan());
assert!(!x.is_infinite());
assert!(x.is_finite());
assert!(x.is_normal());

// Classification
use core::num::FpCategory;
match x.classify() {
    FpCategory::Normal => println!("Normal number"),
    FpCategory::Nan => println!("NaN"),
    FpCategory::Infinite => println!("Infinity"),
    FpCategory::Zero => println!("Zero"),
    FpCategory::Subnormal => println!("Subnormal"),
}
```

### Rounding and Truncation

```rust
use dtype::Float32;
use num_traits::Float;

let x = Float32::new(3.7);

let floor = x.floor();      // 3.0
let ceil = x.ceil();        // 4.0
let round = x.round();      // 4.0
let trunc = x.trunc();      // 3.0
let fract = x.fract();      // 0.7
```

## Working with Integers

### Checked Arithmetic

```rust
use dtype::{Int32, IntExt};

let a = Int32::new(100);
let b = Int32::new(50);

// Checked operations return Option (None on overflow)
let sum = a.checked_add(b);
assert_eq!(sum, Some(Int32::new(150)));

let diff = a.checked_sub(b);
assert_eq!(diff, Some(Int32::new(50)));

let product = a.checked_mul(b);
assert_eq!(product, Some(Int32::new(5000)));

let quotient = a.checked_div(b);
assert_eq!(quotient, Some(Int32::new(2)));

// Overflow example
let max = Int32::max_value();
let overflow = max.checked_add(Int32::new(1));
assert_eq!(overflow, None); // Overflow detected
```

### Bitwise Operations

```rust
use dtype::{Int32, IntExt};

let a = Int32::new(0b1100);
let b = Int32::new(0b1010);

// Bitwise operations
let and = a.bitand(b);      // 0b1000
let or = a.bitor(b);        // 0b1110
let xor = a.bitxor(b);      // 0b0110
let not = a.bitnot();       // Bitwise NOT

// Bit shifts (checked)
let shl = a.checked_shl(2); // Left shift by 2
let shr = a.checked_shr(1); // Right shift by 1
```

### Bit Manipulation

```rust
use dtype::{Int32, IntExt};

let x = Int32::new(0b00101100);

// Count operations
let leading = x.leading_zeros();    // Count leading zeros
let trailing = x.trailing_zeros();  // Count trailing zeros
let ones = x.count_ones();          // Count set bits
let zeros = x.count_zeros();        // Count unset bits

println!("Leading zeros: {}", leading);
println!("Trailing zeros: {}", trailing);
println!("Set bits: {}", ones);
```

### Signed vs Unsigned

```rust
use dtype::{Int32, UInt32};
use num_traits::Signed;

let signed = Int32::new(-42);
let unsigned = UInt32::new(42);

// Signed-specific operations
assert!(signed.is_negative());
assert!(!signed.is_positive());

let abs = signed.abs();
assert_eq!(abs, Int32::new(42));

let signum = signed.signum();
assert_eq!(signum, Int32::new(-1));

// Unsigned types don't have negative values
assert!(unsigned.is_positive());
```

## Working with Complex Numbers

### Basic Complex Operations

```rust
#[cfg(feature = "complex")]
use dtype::{Complex32, Complex64};
use dtype::traits::ComplexExt;

let c1 = Complex32::new(3.0, 4.0);  // 3 + 4i
let c2 = Complex32::new(1.0, 2.0);  // 1 + 2i

// Access components
assert_eq!(c1.re(), 3.0);
assert_eq!(c1.im(), 4.0);

// Arithmetic
let sum = c1 + c2;          // (4 + 6i)
let diff = c1 - c2;         // (2 + 2i)
let product = c1 * c2;      // (-5 + 10i)
let quotient = c1 / c2;     // Complex division
```

### Complex Properties

```rust
#[cfg(feature = "complex")]
use dtype::{Complex32};
use dtype::traits::ComplexExt;

let c = Complex32::new(3.0, 4.0);

// Conjugate
let conj = c.conj();
assert_eq!(conj.re(), 3.0);
assert_eq!(conj.im(), -4.0);

// Magnitude (norm)
let norm = c.norm();
assert_eq!(norm, 5.0);  // sqrt(3^2 + 4^2)

// Magnitude squared
let norm_sqr = c.norm_sqr();
assert_eq!(norm_sqr, 25.0);  // 3^2 + 4^2

// Argument (phase angle)
let arg = c.arg();
// arg ≈ 0.927 radians (≈ 53.13 degrees)
```

## Error Handling

### Type Casting Errors

```rust
use dtype::{Float32, Int32, DataType, DtypeError};

let float_val = Float32::new(3.14);

// Checked cast returns Result
let result: Result<Int32, DtypeError> = float_val.checked_cast_to();
match result {
    Ok(int_val) => println!("Cast succeeded: {}", int_val.get()),
    Err(DtypeError::CastError { from, to, value }) => {
        println!("Cannot cast {} from {} to {}", value, from, to);
    }
    Err(e) => println!("Other error: {}", e),
}

// Safe cast returns Option
let option_result: Option<Int32> = float_val.cast_to();
if let Some(int_val) = option_result {
    println!("Cast succeeded: {}", int_val.get());
} else {
    println!("Cast failed");
}
```

### Overflow Errors

```rust
use dtype::{Int32, IntExt, DtypeError};

let max = Int32::max_value();

// Checked operations return None on overflow
match max.checked_add(Int32::new(1)) {
    Some(result) => println!("Result: {}", result.get()),
    None => println!("Overflow detected!"),
}
```

### Division by Zero

```rust
use dtype::{Int32, IntExt};

let a = Int32::new(10);
let zero = Int32::new(0);

// Checked division returns None on division by zero
match a.checked_div(zero) {
    Some(result) => println!("Result: {}", result.get()),
    None => println!("Division by zero!"),
}
```

## Advanced Patterns

### Generic Functions

```rust
use dtype::{DataType, Dtype};
use num_traits::Float;

// Generic function over any DataType
fn print_type_info<T: DataType>() {
    println!("Type: {}", T::name());
    println!("Size: {} bytes", T::size_bytes());
    println!("Is float: {}", T::is_floating_point());
    println!("Is integer: {}", T::is_integer());
}

// Generic function over floating-point types
fn compute_sigmoid<T>(x: T) -> T
where
    T: DataType + Float,
{
    T::one() / (T::one() + (-x).exp())
}
```

### Type Conversion Patterns

```rust
use dtype::{Float32, Float64, Int32, DataType};
use num_traits::NumCast;

// Convert between types using NumCast
let f32_val = Float32::new(3.14);
let f64_val: Float64 = NumCast::from(f32_val).unwrap();

// Convert from primitives
let from_prim: Float32 = NumCast::from(42i32).unwrap();

// Convert to primitives
use num_traits::ToPrimitive;
let to_prim: f32 = f32_val.to_f32().unwrap();
```

### Working with Dtype Enum

```rust
use dtype::{Dtype, Float32, Int32, DataType};

// Runtime type dispatch
fn process_value(dtype: Dtype, value: f64) {
    match dtype {
        Dtype::Float32 => {
            let val = Float32::new(value as f32);
            println!("Processing as Float32: {}", val.get());
        }
        Dtype::Int32 => {
            let val = Int32::new(value as i32);
            println!("Processing as Int32: {}", val.get());
        }
        _ => println!("Unsupported dtype: {}", dtype),
    }
}

// Type introspection
fn describe_dtype(dtype: Dtype) {
    println!("Name: {}", dtype.name());
    println!("Size: {} bytes", dtype.size_bytes());
    println!("Is float: {}", dtype.is_floating_point());
    println!("Is integer: {}", dtype.is_integer());
    println!("Is complex: {}", dtype.is_complex());
    println!("Is quantized: {}", dtype.is_quantized());
}
```

## Performance Tips

### 1. Use Appropriate Types

```rust
// Good: Use smallest type that fits your data
use dtype::Float32;  // For most ML applications
use dtype::Int8;     // For small integer ranges

// Avoid: Using larger types than necessary
// Float64 uses 2x memory and may be slower
```

### 2. Avoid Unnecessary Conversions

```rust
use dtype::Float32;

// Good: Work in same type
let a = Float32::new(1.0);
let b = Float32::new(2.0);
let result = a + b;

// Avoid: Unnecessary conversions
// let result = a.get() + b.get(); // Converts to f32, then back
```

### 3. Use Checked Operations Judiciously

```rust
use dtype::{Int32, IntExt};

// In hot loops, consider wrapping arithmetic if overflow is impossible
let a = Int32::new(10);
let b = Int32::new(20);
let sum = a + b;  // Wrapping add (faster)

// Use checked operations when overflow is possible
let max = Int32::max_value();
if let Some(result) = max.checked_add(Int32::new(1)) {
    // Handle result
}
```

### 4. Leverage Monomorphization

```rust
use dtype::{DataType, Float32};

// Generic functions are monomorphized at compile time
fn compute<T: DataType>(x: T, y: T) -> T
where
    T: core::ops::Add<Output = T>,
{
    x + y  // Zero-cost abstraction
}

// Each call site gets optimized version
let f32_result = compute(Float32::new(1.0), Float32::new(2.0));
```

## Common Pitfalls

### 1. Precision Loss in Conversions

```rust
use dtype::{Float32, Int32, DataType};

// Pitfall: Precision loss when converting float to int
let float_val = Float32::new(3.14);
let int_val: Result<Int32, _> = float_val.checked_cast_to();
// This will fail because fractional part is lost

// Solution: Explicitly round before converting
use num_traits::Float;
let rounded = float_val.round();
// Then convert if needed
```

### 2. Overflow in Integer Operations

```rust
use dtype::{Int32, IntExt};

// Pitfall: Wrapping arithmetic can silently overflow
let max = Int32::max_value();
let overflow = max + Int32::new(1);  // Wraps to Int32::MIN

// Solution: Use checked operations
let safe_result = max.checked_add(Int32::new(1));
assert_eq!(safe_result, None);  // Overflow detected
```

### 3. Mixed Signedness Promotion

```rust
use dtype::{Dtype, promotion::promote};

// Pitfall: Unexpected promotion to larger type
let result = promote(Dtype::Int32, Dtype::UInt32);
assert_eq!(result, Dtype::Int64);  // Not Int32 or UInt32!

// Solution: Be aware of promotion rules
// Mixed signed/unsigned promotes to larger signed type
```

### 4. NaN Comparisons

```rust
use dtype::Float32;
use num_traits::Float;

// Pitfall: NaN is not equal to anything, including itself
let nan = Float32::nan();
assert!(nan != nan);  // True!

// Solution: Use is_nan() to check for NaN
assert!(nan.is_nan());
```

### 5. Division by Zero

```rust
use dtype::{Float32, Int32, IntExt};
use num_traits::Float;

// Pitfall: Float division by zero returns infinity (no error)
let f = Float32::new(1.0) / Float32::new(0.0);
assert!(f.is_infinite());

// Pitfall: Integer division by zero panics
// let i = Int32::new(1) / Int32::new(0);  // PANIC!

// Solution: Use checked operations for integers
let safe_div = Int32::new(1).checked_div(Int32::new(0));
assert_eq!(safe_div, None);
```

## Best Practices

1. **Use checked operations** when overflow is possible
2. **Prefer smaller types** when appropriate (Float32 over Float64)
3. **Be explicit about conversions** using `cast_to()` or `checked_cast_to()`
4. **Understand promotion rules** when mixing types
5. **Handle NaN and infinity** explicitly in floating-point code
6. **Use type introspection** (`dtype()`, `is_floating_point()`) for runtime dispatch
7. **Leverage generics** for type-agnostic algorithms
8. **Test edge cases** (overflow, underflow, NaN, infinity, zero)

## Further Reading

- [Architecture Audit](ARCHITECTURE_AUDIT.md) - Detailed architectural analysis
- [README](README.md) - Quick start and overview
- [Rust num-traits documentation](https://docs.rs/num-traits/) - Underlying trait library
- [PyTorch dtype documentation](https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype) - Comparison with PyTorch

---

**Questions or Issues?** See the main Coeus repository for contribution guidelines.
