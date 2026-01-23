# Developer Guide: Hierarchical Structure Navigation

This guide helps developers navigate and work with the Coeus hierarchical file structure.

## Overview

Coeus uses a deep vertical hierarchical file structure that enables:
- **Script-based parity tracking** across backends
- **Clear domain separation** between different concerns
- **Consistent organization** across all crates
- **Easy navigation** to find specific functionality

## File Organization Principles

### 1. Single Responsibility Per File
Each file contains exactly one operation or closely related functionality:

```
✅ Good: backend/src/cpu/arithmetic/add.rs        (only addition)
✅ Good: nn/src/functional/ops/activation/relu.rs (only ReLU)
❌ Bad:  backend/src/cpu/math_ops.rs              (multiple operations)
```

### 2. Hierarchical Categories
Operations are organized in a hierarchy from general to specific:

```
crate/src/
├── domain/           # Broad domain (cpu, gpu, dense, sparse)
│   ├── category/     # Operation category (arithmetic, activation)
│   │   ├── op1.rs    # Specific operation
│   │   ├── op2.rs    # Another specific operation
│   │   └── mod.rs    # Module exports
│   └── mod.rs
└── lib.rs
```

### 3. Parallel Structures
Related crates maintain parallel structures for easy comparison:

```
backend/src/cpu/arithmetic/add.rs
backend/src/gpu/arithmetic/add.rs    # Same structure
backend/src/tpu/arithmetic/add.rs    # Same structure
backend/src/npu/arithmetic/add.rs    # Same structure
```

## Navigation Guide

### Finding Operations

#### By Domain
1. **Neural Network Operations**: `nn/src/functional/ops/{category}/{operation}.rs`
2. **Dense Operations**: `dense/src/ops/{category}/{operation}.rs`
3. **Sparse Operations**: `sparse/src/formats/{format}/{category}/{operation}.rs`
4. **Backend Operations**: `backend/src/{device}/{category}/{operation}.rs`
5. **Storage Operations**: `storage/src/{format}/{category}/{operation}.rs`

#### By Category
Common categories across crates:
- **arithmetic**: add, sub, mul, div, pow
- **linear_algebra**: matmul, transpose, inverse, svd
- **activation**: relu, sigmoid, tanh, gelu, softmax
- **reduction**: sum, mean, max, min, argmax, argmin
- **convolution**: conv1d, conv2d, conv3d
- **pooling**: maxpool, avgpool, adaptivepool

#### By Operation Type
- **Element-wise**: Look in `arithmetic/` or `elementwise/` directories
- **Matrix operations**: Look in `linear_algebra/` directories
- **Neural network**: Look in `nn/src/functional/ops/`
- **Hardware-specific**: Look in `backend/src/{device}/`

### Quick Navigation Commands

```bash
# Find all implementations of an operation
find . -name "relu.rs" -type f

# Find all operations in a category
ls backend/src/cpu/arithmetic/

# Find missing implementations
diff <(ls backend/src/cpu/arithmetic/) <(ls backend/src/gpu/arithmetic/)

# Search for operation usage
rg "pub fn relu" --type rust
```

## Directory Structure Reference

### Backend Crate
```
backend/src/
├── cpu/
│   ├── arithmetic/
│   │   ├── add.rs              # Element-wise addition
│   │   ├── sub.rs              # Element-wise subtraction
│   │   ├── mul.rs              # Element-wise multiplication
│   │   ├── div.rs              # Element-wise division
│   │   └── mod.rs
│   ├── linear_algebra/
│   │   ├── matmul.rs           # Matrix multiplication
│   │   ├── transpose.rs        # Matrix transpose
│   │   ├── decomposition.rs    # Matrix decomposition
│   │   └── mod.rs
│   ├── activation/
│   │   ├── relu.rs             # ReLU activation
│   │   ├── sigmoid.rs          # Sigmoid activation
│   │   ├── tanh.rs             # Tanh activation
│   │   └── mod.rs
│   ├── reduction/
│   │   ├── sum.rs              # Sum reduction
│   │   ├── mean.rs             # Mean reduction
│   │   ├── max.rs              # Max reduction
│   │   └── mod.rs
│   └── mod.rs
├── gpu/                        # Identical structure to cpu/
├── tpu/                        # Identical structure to cpu/
├── npu/                        # Identical structure to cpu/
└── lib.rs
```

### NN Crate
```
nn/src/
├── functional/
│   └── ops/
│       ├── activation/
│       │   ├── relu.rs         # ReLU function
│       │   ├── gelu.rs         # GELU function
│       │   ├── softmax.rs      # Softmax function
│       │   └── mod.rs
│       ├── loss/
│       │   ├── mse.rs          # Mean Squared Error
│       │   ├── cross_entropy.rs # Cross-entropy loss
│       │   └── mod.rs
│       ├── convolution/
│       │   ├── conv1d.rs       # 1D convolution
│       │   ├── conv2d.rs       # 2D convolution
│       │   ├── conv3d.rs       # 3D convolution
│       │   └── mod.rs
│       └── mod.rs
├── modules/                    # Parallel structure to functional/ops/
│   ├── activation/
│   ├── loss/
│   ├── convolution/
│   └── mod.rs
└── lib.rs
```

### Storage Crate
```
storage/src/
├── dense/
│   ├── arithmetic/
│   │   ├── add.rs              # Dense addition
│   │   ├── sub.rs              # Dense subtraction
│   │   └── mod.rs
│   ├── layout/
│   │   ├── reshape.rs          # Reshape operations
│   │   ├── transpose.rs        # Transpose operations
│   │   └── mod.rs
│   └── mod.rs
├── sparse/
│   ├── csr/
│   │   ├── arithmetic/
│   │   ├── conversion/
│   │   └── mod.rs
│   ├── csc/                    # Parallel structure to csr/
│   ├── coo/                    # Parallel structure to csr/
│   └── mod.rs
└── lib.rs
```

### Sparse Crate
```
sparse/src/
├── formats/
│   ├── csr/
│   │   ├── arithmetic/
│   │   │   ├── add.rs          # CSR addition
│   │   │   ├── mul.rs          # CSR multiplication
│   │   │   └── mod.rs
│   │   ├── conversion/
│   │   │   ├── to_csc.rs       # CSR to CSC conversion
│   │   │   ├── to_dense.rs     # CSR to dense conversion
│   │   │   └── mod.rs
│   │   └── mod.rs
│   ├── csc/                    # Parallel structure to csr/
│   ├── coo/                    # Parallel structure to csr/
│   └── mod.rs
└── lib.rs
```

## Working with the Structure

### Adding a New Operation

1. **Identify the correct domain**: Where does this operation belong?
   - Neural network operation → `nn/src/functional/ops/`
   - Dense algorithm → `dense/src/ops/`
   - Sparse algorithm → `sparse/src/formats/`
   - Hardware primitive → `backend/src/{device}/`

2. **Find the correct category**: What type of operation is it?
   - Mathematical function → `arithmetic/` or `math/`
   - Activation function → `activation/`
   - Matrix operation → `linear_algebra/`
   - Reduction operation → `reduction/`

3. **Create the file**: Follow the naming convention
   ```bash
   # Example: Adding a new activation function
   touch nn/src/functional/ops/activation/swish.rs
   ```

4. **Implement across backends** (if applicable):
   ```bash
   # For backend operations, implement in all backends
   touch backend/src/cpu/activation/swish.rs
   touch backend/src/gpu/activation/swish.rs
   touch backend/src/tpu/activation/swish.rs
   touch backend/src/npu/activation/swish.rs
   ```

5. **Update module exports**:
   ```rust
   // In mod.rs files
   pub mod swish;
   pub use swish::*;
   ```

### Finding Related Operations

Use the hierarchical structure to find related operations:

```bash
# Find all activation functions
ls nn/src/functional/ops/activation/

# Find all arithmetic operations for CPU
ls backend/src/cpu/arithmetic/

# Find all CSR operations
ls sparse/src/formats/csr/arithmetic/
```

### Checking Implementation Status

```bash
# Check if operation exists across all backends
for backend in cpu gpu tpu npu; do
  if [ -f "backend/src/$backend/arithmetic/add.rs" ]; then
    echo "$backend: ✅ add.rs exists"
  else
    echo "$backend: ❌ add.rs missing"
  fi
done
```

## Common Patterns

### Module Organization Pattern

Each directory follows this pattern:

```rust
// category/mod.rs
pub mod operation1;
pub mod operation2;
pub mod operation3;

pub use operation1::*;
pub use operation2::*;
pub use operation3::*;
```

### Operation File Pattern

Each operation file follows this pattern:

```rust
// category/operation.rs
use crate::error::Result;
use coeus_dtype::DataType;

/// Brief description of the operation
pub fn operation_name<T: DataType>(
    input: &[T],
    // other parameters
) -> Result<Vec<T>> {
    // Implementation
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_operation_basic() {
        // Test implementation
    }
}
```

### Backend Implementation Pattern

Backend implementations maintain identical APIs:

```rust
// backend/src/cpu/category/operation.rs
pub fn operation_primitive<T: DataType>(
    input: &[T],
    output: &mut [T],
) -> Result<()> {
    // CPU-specific implementation
}

// backend/src/gpu/category/operation.rs
pub fn operation_primitive<T: DataType>(
    input: &[T],
    output: &mut [T],
) -> Result<()> {
    // GPU-specific implementation
}
```

## Troubleshooting

### Can't Find an Operation?

1. **Check the category**: Is it in the right category directory?
2. **Check the domain**: Is it in the right crate?
3. **Check naming**: Are you using the correct operation name?
4. **Use search**: `rg "operation_name" --type rust`

### Operation Exists But Can't Import?

1. **Check mod.rs**: Is the operation exported in mod.rs?
2. **Check lib.rs**: Is the module declared in lib.rs?
3. **Check visibility**: Is the function marked `pub`?

### Missing Implementation?

1. **Check all backends**: Some backends might not have the implementation
2. **Check parity scripts**: Run parity checking scripts
3. **Check status**: Look for `unimplemented!()` macros

## Scripts and Tools

### Parity Checking Scripts

```bash
# Check backend parity
python scripts/check_backend_parity.py

# Generate parity report
python scripts/generate_parity_report.py

# Check for missing implementations
bash scripts/check_missing_implementations.sh
```

### Navigation Helpers

```bash
# Find operation across all crates
function find_op() {
  find . -name "$1.rs" -type f | grep -v target
}

# List operations in category
function list_ops() {
  ls */src/*/$1/ 2>/dev/null || ls */src/*/*/$1/ 2>/dev/null
}

# Check implementation status
function check_impl() {
  for backend in cpu gpu tpu npu; do
    if [ -f "backend/src/$backend/$1/$2.rs" ]; then
      echo "$backend: ✅"
    else
      echo "$backend: ❌"
    fi
  done
}
```

## Best Practices

### File Organization
- **One operation per file**: Keep files focused and small
- **Consistent naming**: Use the same name across all backends
- **Clear categories**: Put operations in logical categories
- **Parallel structures**: Maintain identical structures across related domains

### Documentation
- **Document each operation**: Include purpose and usage
- **Document categories**: Explain what belongs in each category
- **Update navigation**: Keep navigation guides up to date
- **Provide examples**: Show how to use operations

### Testing
- **Test each operation**: Every operation should have tests
- **Test across backends**: Ensure consistent behavior
- **Test integration**: Test how operations work together
- **Test edge cases**: Handle boundary conditions

## Getting Help

If you're having trouble navigating the structure:

1. **Check this guide**: Look for similar patterns
2. **Look at examples**: Find similar operations and follow their pattern
3. **Use search tools**: grep, ripgrep, find are your friends
4. **Ask for help**: Create an issue or ask in discussions

## Contributing

When contributing to the hierarchical structure:

1. **Follow the patterns**: Use existing patterns as templates
2. **Maintain parity**: Implement across all relevant backends
3. **Update documentation**: Keep guides and docs up to date
4. **Run checks**: Use parity checking scripts
5. **Test thoroughly**: Ensure your changes don't break existing functionality

---

This guide should help you navigate and work effectively with the Coeus hierarchical file structure. The key is understanding the patterns and using the tools available to find and organize code effectively.