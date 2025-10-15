# ADR: Dynamic Graph Architecture for PyTorch Compatibility

## Status
**APPROVED** - Committed to PyTorch-compatible dynamic graph architecture. Abandoning operation-based approach in favor of automatic graph construction.

## Context

Coeus aims for 100% PyTorch API compatibility while providing Rust's memory safety and performance benefits. The current autograd implementation uses an operation-based approach that stores full tensor data in each operation, which:

- **Memory Intensive**: Each operation contains complete input/output tensors
- **Not PyTorch Compatible**: Missing dynamic graph construction during forward pass
- **Performance Issues**: Excessive memory allocation and copying
- **Maintenance Burden**: Complex to extend and debug

PyTorch uses a dynamic computation graph where:
- Each tensor operation creates a new `Function` object with `grad_fn`
- Graph construction happens implicitly during forward pass
- `grad_fn` points to parent operations in reverse topological order
- Backward pass traverses `grad_fn` chain with efficient gradient accumulation

## Current Implementation Analysis

### Operation-Based Approach (Current)
```rust
pub enum Operation<T: DataType> {
    Conv2D {
        input: Variable<T>,      // Full tensor data stored here
        weight: Variable<T>,     // Full tensor data stored here
        bias: Option<Variable<T>>, // Full tensor data stored here
        stride_h: usize,
        // ... parameters
    },
    // ...
}
```

**Problems:**
- 100MB Conv2D operation stores 3 full tensors (~300MB total)
- Circular references: Variable → Operation → Variable
- No dynamic graph construction
- Memory usage scales with computation depth

### Node-Based Approach (Target)
```rust
pub struct Function {
    pub grad_fn: Option<Arc<Function>>,
    pub next_functions: Vec<Arc<Function>>,
    pub requires_grad: bool,
}

impl Function {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor>;
}
```

## Decision

**Adopt PyTorch-compatible dynamic graph architecture with the following changes:**

### 1. Replace Operation Enum with Function Trait
```rust
pub trait Function: Send + Sync {
    fn backward(&self, grad_output: &dyn Any) -> Vec<Box<dyn Any>>;
    fn name(&self) -> &'static str;
}

pub struct Conv2DFunction {
    input: TensorRef,    // Lightweight reference, not full data
    weight: TensorRef,
    bias: Option<TensorRef>,
    stride_h: usize,
    stride_w: usize,
    padding_h: usize,
    padding_w: usize,
}

impl Function for Conv2DFunction {
    fn backward(&self, grad_output: &dyn Any) -> Vec<Box<dyn Any>> {
        // Compute gradients using stored references
        // Return lightweight gradient functions
    }

    fn name(&self) -> &'static str { "Conv2DBackward" }
}
```

### 2. Tensor-Centric Graph Construction
```rust
impl<B, S, T> Tensor<B, S, T> {
    pub fn grad_fn(&self) -> Option<&Arc<dyn Function>> {
        self.grad_fn.as_ref()
    }

    pub fn set_grad_fn(&mut self, grad_fn: Arc<dyn Function>) {
        self.grad_fn = Some(grad_fn);
    }
}
```

### 3. Implicit Graph Building
```rust
// Instead of manual graph construction:
let result = input.conv2d(&weight, &bias, stride, padding);

// Result automatically gets grad_fn:
assert_eq!(result.grad_fn().unwrap().name(), "Conv2DBackward");
```

### 4. Backward Pass via grad_fn Chain
```rust
impl<B, S, T> AutoGradTensor<B, S, T> {
    pub fn backward(&self) {
        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();

        queue.push_back(self.tensor.grad_fn().unwrap().clone());

        while let Some(func) = queue.pop_front() {
            if visited.insert(func.name()) {
                let grad_inputs = func.backward(&self.grad);

                // Accumulate gradients to inputs
                for (input_tensor, grad) in func.inputs().iter().zip(grad_inputs) {
                    input_tensor.accumulate_grad(grad);
                }

                // Add parent functions to queue
                for parent in func.next_functions() {
                    queue.push_back(parent.clone());
                }
            }
        }
    }
}
```

## Implementation Plan

### Phase 1: Core Function Trait (Sprint MS-6)
- Define `Function` trait with `backward()` and `name()` methods
- Implement basic arithmetic operations (Add, Mul, etc.)
- Update tensor to store `grad_fn: Option<Arc<dyn Function>>`

### Phase 2: Neural Network Functions (Sprint MS-7)
- Implement `Conv2DFunction`, `LinearFunction`, etc.
- Replace operation-based Conv2D with function-based implementation
- Update all NN modules to use new Function trait

### Phase 3: Graph Management (Sprint MS-8)
- Implement topological sorting for gradient computation
- Add gradient accumulation and retention
- Optimize memory usage with buffer reuse

### Phase 4: PyTorch API Compatibility (Sprint MS-9)
- Implement `requires_grad_()`, `backward()`, `grad` properties
- Add higher-order derivatives support
- Complete API parity testing

## Performance Impact

### Memory Efficiency
- **Before**: O(n) memory for n operations (stores full tensors)
- **After**: O(1) memory per operation (stores lightweight references)

### Computation Efficiency
- **Before**: Gradient computation requires full tensor data lookup
- **After**: Direct reference access with optimized memory layout

### API Compatibility
- **Before**: Manual graph construction required
- **After**: Automatic graph building matches PyTorch exactly

## Risks and Mitigations

### Risk: Breaking Change
**Mitigation**: Implement alongside existing system, migrate incrementally

### Risk: Performance Regression
**Mitigation**: Benchmark each phase, optimize hot paths with SIMD

### Risk: API Incompatibility
**Mitigation**: Comprehensive testing against PyTorch reference implementations

## Alternatives Considered

### 1. Keep Operation-Based Approach
**Rejected**: Cannot achieve PyTorch API compatibility

### 2. Use Macros for Code Generation
**Rejected**: Increases complexity without solving fundamental issues

### 3. Hybrid Approach (Operations + Functions)
**Rejected**: Adds complexity without clear benefits

## References

- [PyTorch Autograd Documentation](https://pytorch.org/docs/stable/autograd.html)
- [PyTorch Internals: Autograd](https://pytorch.org/blog/overview-of-pytorch-autograd-engine/)
- [JAX Autograd Implementation](https://jax.readthedocs.io/en/latest/autodiff.html)

## Success Metrics

- ✅ Zero memory overhead vs PyTorch for graph construction
- ✅ 100% PyTorch API compatibility for autograd operations
- ✅ <1% performance regression on gradient computation
- ✅ All existing tests pass with new implementation
