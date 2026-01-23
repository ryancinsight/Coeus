# Coeus Architecture Enhancement Migration Guide

This guide helps you migrate your code to the enhanced Coeus architecture with quantization crate extraction, dense crate creation, storage simplification, and hierarchical file structure improvements.

## Overview of Changes

The architectural enhancement introduces several major changes:

1. **Quantization Crate Extraction**: Quantization logic moved from `nn` and `dtype` to dedicated `quantization` crate
2. **Dense Crate Creation**: Dense operations extracted from `tensor` to dedicated `dense` crate  
3. **Storage Simplification**: Storage limited to basic operations, complex operations moved to higher layers
4. **Hierarchical File Structure**: Deep vertical organization for better parity tracking
5. **Domain Separation**: Stricter boundaries between crate responsibilities

## Breaking Changes Summary

### Import Changes

| Old Import | New Import | Notes |
|------------|------------|-------|
| `use coeus_nn::quantization::*` | `use coeus_quantization::*` | Quantization moved to dedicated crate |
| `use coeus_dtype::quantized::*` | `use coeus_quantization::types::*` | Quantized types moved |
| `use coeus_tensor::dense::*` | `use coeus_dense::*` | Dense operations extracted |
| `use coeus_storage::complex_ops::*` | `use coeus_tensor::ops::*` | Complex operations moved up |

### Crate Dependencies

Update your `Cargo.toml` dependencies:

```toml
[dependencies]
# Add new crates
coeus-quantization = { path = "../quantization" }
coeus-dense = { path = "../dense" }

# Existing crates (no changes needed)
coeus-nn = { path = "../nn" }
coeus-tensor = { path = "../tensor" }
coeus-storage = { path = "../storage" }
coeus-backend = { path = "../backend" }
coeus-dtype = { path = "../dtype" }
```

## Migration Steps

### Step 1: Update Quantization Usage

#### Before (Old Architecture)

```rust
// Old: Quantization in nn crate
use coeus_nn::quantization::{SymmetricQuantizer, FakeQuantize};
use coeus_dtype::quantized::{QInt8, QuantizedStorage};

let quantizer = SymmetricQuantizer::new(8)?;
let fake_quant = FakeQuantize::new(8, true)?;
let qint8_tensor = QInt8::from_tensor(&tensor, scale, zero_point)?;
```

#### After (New Architecture)

```rust
// New: Dedicated quantization crate
use coeus_quantization::algorithms::SymmetricQuantizer;
use coeus_quantization::fake_quantize::FakeQuantizeLinear;
use coeus_quantization::types::QInt8;

let quantizer = SymmetricQuantizer::new(8)?;
let fake_quant = FakeQuantizeLinear::new(config)?;
let qint8_tensor = QInt8::from_tensor(&tensor, scale, zero_point)?;
```

#### Migration Script

```bash
# Update imports automatically
find . -name "*.rs" -exec sed -i 's/coeus_nn::quantization/coeus_quantization::algorithms/g' {} \;
find . -name "*.rs" -exec sed -i 's/coeus_dtype::quantized/coeus_quantization::types/g' {} \;
```

### Step 2: Update Dense Operations Usage

#### Before (Old Architecture)

```rust
// Old: Dense operations in tensor crate
use coeus_tensor::dense::{dense_add, dense_matmul};
use coeus_tensor::DenseTensor;

let result = dense_add(&a, &b)?;
let matmul_result = dense_matmul(&a, &b)?;
```

#### After (New Architecture)

```rust
// New: Dedicated dense crate
use coeus_dense::ops::elementwise::add;
use coeus_dense::ops::linear_algebra::matmul;
use coeus_tensor::Tensor;
use coeus_storage::DenseStorage;

let result = add(a.storage(), b.storage())?;
let matmul_result = matmul(a.storage(), b.storage())?;
```

#### Migration Script

```bash
# Update dense operation imports
find . -name "*.rs" -exec sed -i 's/coeus_tensor::dense::/coeus_dense::ops::/g' {} \;
find . -name "*.rs" -exec sed -i 's/DenseTensor/Tensor<B, DenseStorage<T>, T>/g' {} \;
```

### Step 3: Update Storage Usage

#### Before (Old Architecture)

```rust
// Old: Complex operations in storage
use coeus_storage::{DenseStorage, linear_transform, convolution};

let storage = DenseStorage::zeros(&[10, 10])?;
let transformed = linear_transform(&storage, &weight, &bias)?;
let conv_result = convolution(&storage, &kernel, stride, padding)?;
```

#### After (New Architecture)

```rust
// New: Basic operations only in storage
use coeus_storage::{DenseStorage, StorageFromVec};
use coeus_dense::ops::linear_algebra::matmul;
use coeus_nn::functional::ops::convolution::conv2d;

let storage = DenseStorage::zeros(&[10, 10])?;
// Use tensor/nn operations for complex operations
let transformed = matmul(&storage, &weight)?; // + bias separately
let conv_result = conv2d(&tensor, &kernel, stride, padding)?;
```

### Step 4: Update File Structure References

#### Before (Old Architecture)

```rust
// Old: Monolithic files
use coeus_nn::functional::activation; // Single large file
use coeus_backend::cpu; // Single large file
```

#### After (New Architecture)

```rust
// New: Hierarchical structure
use coeus_nn::functional::ops::activation::relu;
use coeus_nn::functional::ops::activation::gelu;
use coeus_backend::cpu::arithmetic::add;
use coeus_backend::cpu::linear_algebra::matmul;
```

## Detailed Migration Examples

### Example 1: Quantized Neural Network

#### Before

```rust
use coeus_nn::{Linear, quantization::FakeQuantize};
use coeus_dtype::quantized::QInt8;

struct QuantizedModel {
    linear: Linear<CpuBackend, DenseStorage<f32>, f32>,
    fake_quant: FakeQuantize,
}

impl QuantizedModel {
    fn new() -> Result<Self> {
        Ok(Self {
            linear: Linear::new(784, 10, true)?,
            fake_quant: FakeQuantize::new(8, true)?,
        })
    }
    
    fn forward(&self, input: &Tensor<CpuBackend, DenseStorage<f32>, f32>) -> Result<Tensor<CpuBackend, DenseStorage<f32>, f32>> {
        let quantized_weight = self.fake_quant.forward(self.linear.weight())?;
        self.linear.forward_with_weight(input, &quantized_weight)
    }
}
```

#### After

```rust
use coeus_nn::modules::Linear;
use coeus_quantization::fake_quantize::{FakeQuantizeLinear, FakeQuantizeConfig};

struct QuantizedModel {
    linear: Linear<CpuBackend, DenseStorage<f32>, f32>,
    fake_quant: FakeQuantizeLinear,
}

impl QuantizedModel {
    fn new() -> Result<Self> {
        let config = FakeQuantizeConfig {
            bits: 8,
            symmetric: true,
            per_channel: false,
            observer_type: ObserverType::MinMax,
        };
        
        Ok(Self {
            linear: Linear::new(784, 10, true)?,
            fake_quant: FakeQuantizeLinear::new(config)?,
        })
    }
    
    fn forward(&self, input: &Tensor<CpuBackend, DenseStorage<f32>, f32>) -> Result<Tensor<CpuBackend, DenseStorage<f32>, f32>> {
        let quantized_weight = self.fake_quant.forward(self.linear.weight())?;
        self.linear.forward_with_weight(input, &quantized_weight)
    }
}
```

### Example 2: Dense Tensor Operations

#### Before

```rust
use coeus_tensor::{Tensor, dense::*};

fn dense_computation(a: &Tensor<CpuBackend, DenseStorage<f32>, f32>, 
                    b: &Tensor<CpuBackend, DenseStorage<f32>, f32>) -> Result<Tensor<CpuBackend, DenseStorage<f32>, f32>> {
    let sum = dense_add(a.storage(), b.storage())?;
    let product = dense_mul(a.storage(), b.storage())?;
    let result = dense_matmul(&sum, &product)?;
    Ok(Tensor::from_storage(result, a.backend().clone()))
}
```

#### After

```rust
use coeus_tensor::Tensor;
use coeus_dense::ops::{elementwise, linear_algebra};

fn dense_computation(a: &Tensor<CpuBackend, DenseStorage<f32>, f32>, 
                    b: &Tensor<CpuBackend, DenseStorage<f32>, f32>) -> Result<Tensor<CpuBackend, DenseStorage<f32>, f32>> {
    let sum = elementwise::add(a.storage(), b.storage())?;
    let product = elementwise::mul(a.storage(), b.storage())?;
    let result = linear_algebra::matmul(&sum, &product)?;
    Ok(Tensor::from_storage(result, a.backend().clone()))
}
```

### Example 3: Storage Operations

#### Before

```rust
use coeus_storage::{DenseStorage, complex_operations::*};

fn complex_storage_ops(storage: &DenseStorage<f32>) -> Result<DenseStorage<f32>> {
    let normalized = batch_normalize(storage, &mean, &var, &gamma, &beta)?;
    let activated = apply_relu(&normalized)?;
    let pooled = max_pool_2d(&activated, kernel_size, stride)?;
    Ok(pooled)
}
```

#### After

```rust
use coeus_storage::DenseStorage;
use coeus_nn::functional::ops::{normalization, activation, pooling};

fn complex_storage_ops(tensor: &Tensor<CpuBackend, DenseStorage<f32>, f32>) -> Result<Tensor<CpuBackend, DenseStorage<f32>, f32>> {
    let normalized = normalization::batch_norm(tensor, &mean, &var, &gamma, &beta)?;
    let activated = activation::relu(&normalized)?;
    let pooled = pooling::max_pool_2d(&activated, kernel_size, stride)?;
    Ok(pooled)
}
```

## Common Migration Issues and Solutions

### Issue 1: Import Resolution Errors

**Problem**: `use coeus_nn::quantization::*` not found

**Solution**: Update to `use coeus_quantization::*` and add dependency

```toml
# Add to Cargo.toml
coeus-quantization = { path = "../quantization" }
```

### Issue 2: Type Mismatch Errors

**Problem**: `QInt8` type not found in `coeus_dtype`

**Solution**: Import from new location

```rust
// Old
use coeus_dtype::quantized::QInt8;

// New  
use coeus_quantization::types::QInt8;
```

### Issue 3: Missing Dense Operations

**Problem**: `dense_add` function not found in `coeus_tensor`

**Solution**: Use new dense crate

```rust
// Old
use coeus_tensor::dense::dense_add;

// New
use coeus_dense::ops::elementwise::add;
```

### Issue 4: Storage Complex Operations

**Problem**: Complex operations no longer available in storage

**Solution**: Use appropriate higher-level crate

```rust
// Old
use coeus_storage::convolution;

// New
use coeus_nn::functional::ops::convolution::conv2d;
```

## Testing Your Migration

### 1. Compilation Test

```bash
# Test that your code compiles
cargo check --workspace

# Test specific crates
cargo check --package your-crate
```

### 2. Unit Tests

```bash
# Run tests to ensure functionality is preserved
cargo test --workspace

# Run specific tests
cargo test --package your-crate
```

### 3. Integration Tests

```bash
# Test end-to-end functionality
cargo test --test integration_tests
```

### 4. Performance Validation

```bash
# Ensure performance is maintained
cargo bench --workspace
```

## Automated Migration Tools

### Migration Script

Create a migration script to automate common changes:

```bash
#!/bin/bash
# migrate_coeus.sh

echo "Starting Coeus architecture migration..."

# Update quantization imports
find . -name "*.rs" -exec sed -i 's/coeus_nn::quantization::/coeus_quantization::/g' {} \;
find . -name "*.rs" -exec sed -i 's/coeus_dtype::quantized::/coeus_quantization::types::/g' {} \;

# Update dense imports  
find . -name "*.rs" -exec sed -i 's/coeus_tensor::dense::/coeus_dense::ops::/g' {} \;

# Update storage imports
find . -name "*.rs" -exec sed -i 's/coeus_storage::complex_operations::/coeus_nn::functional::ops::/g' {} \;

echo "Migration complete. Please review changes and test thoroughly."
```

### Cargo.toml Update Script

```bash
#!/bin/bash
# update_cargo_toml.sh

# Add new dependencies to Cargo.toml files
find . -name "Cargo.toml" -exec sed -i '/coeus-nn = /a coeus-quantization = { path = "../quantization" }\ncoeus-dense = { path = "../dense" }' {} \;
```

## Validation Checklist

After migration, verify:

- [ ] All imports resolve correctly
- [ ] Code compiles without errors
- [ ] All tests pass
- [ ] Performance benchmarks show no regression
- [ ] Documentation builds successfully
- [ ] Examples run correctly

## Rollback Plan

If migration issues occur:

1. **Git Rollback**: Use version control to revert changes
2. **Incremental Migration**: Migrate one module at a time
3. **Compatibility Layer**: Create temporary compatibility wrappers
4. **Gradual Transition**: Keep old and new APIs side by side temporarily

### Compatibility Wrapper Example

```rust
// Temporary compatibility for gradual migration
pub mod compat {
    pub use coeus_quantization::algorithms::SymmetricQuantizer;
    pub use coeus_quantization::types::QInt8;
    pub use coeus_dense::ops::elementwise::add as dense_add;
    
    // Deprecated warnings
    #[deprecated(note = "Use coeus_quantization::algorithms::SymmetricQuantizer")]
    pub type OldQuantizer = SymmetricQuantizer;
}
```

## Support and Resources

### Documentation

- [Architecture Documentation](docs/ARCHITECTURE_INDEX.md)
- [Quantization Crate README](quantization/README.md)
- [Dense Crate README](dense/README.md)
- [Storage Crate README](storage/README.md)

### Getting Help

1. **Check Examples**: Look at updated examples in each crate
2. **Read Tests**: Test files show correct usage patterns
3. **Review Documentation**: Each crate has comprehensive documentation
4. **Ask Questions**: Create issues for migration problems

### Migration Timeline

**Recommended migration approach:**

1. **Week 1**: Update dependencies and imports
2. **Week 2**: Migrate quantization usage
3. **Week 3**: Migrate dense operations
4. **Week 4**: Update storage usage and test thoroughly

## Conclusion

The architectural enhancement provides better domain separation, clearer responsibilities, and improved maintainability. While migration requires some effort, the new structure offers:

- **Better Organization**: Clear separation of concerns
- **Improved Maintainability**: Easier to find and modify functionality
- **Enhanced Testing**: Better test coverage through domain separation
- **Future Extensibility**: Easier to add new functionality

Follow this guide step by step, test thoroughly, and don't hesitate to ask for help if you encounter issues during migration.