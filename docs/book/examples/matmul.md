# Example: Matrix Multiplication

**Crate**: `coeus-ops`
**Source**: `crates/coeus-ops/examples/book_matmul.rs`

Multiply a `2×3` matrix by a `3×2` matrix, verify `A × I = A`, and confirm
identical results on both `SequentialBackend` and `MoiraiBackend`.

## Source

```rust
{{#include ../../../crates/coeus-ops/examples/book_matmul.rs}}
```

## Output

```text
A × B (2×3 · 3×2) = [58.0, 64.0, 139.0, 154.0]
A × I = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
same result on MoiraiBackend: [58.0, 64.0, 139.0, 154.0]
all matmul assertions passed
```

## What to notice

- `Tensor::from_slice([2, 3], &data)` is shorthand for `from_slice_on` with
  `B::default()` as the backend.

- `matmul(&a, &b, &backend)` is generic over any `BackendOps<T> + Default`.
  The `SequentialBackend` runs the kernel synchronously; `MoiraiBackend` may
  dispatch to idle worker threads.

- The identity property `A × I = A` holds exactly for integers stored as
  `f32`; the assertion uses a 1e-6 tolerance for floating-point safety.
