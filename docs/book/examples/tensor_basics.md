# Example: Tensor Basics

**Crate**: `coeus-ops`
**Source**: `crates/coeus-ops/examples/book_tensor_basics.rs`

Construct 1-D and 2-D tensors with `MoiraiBackend`, compute `sum` and `mean`.

## Source

```rust
{{#include ../../../crates/coeus-ops/examples/book_tensor_basics.rs}}
```

## Output

```text
a = [1.0, 2.0, 3.0, 4.0, 5.0]
sum  = 15
mean = 3
3×2 matrix sum = 21
all tensor-basics assertions passed
```

## What to notice

- `MoiraiBackend` is a zero-sized unit struct; constructing it allocates nothing.
  It routes CPU operations through the moirai work-stealing scheduler.

- `Tensor::from_slice_on(shape, data, &backend)` takes a `Vec<usize>` shape and
  a `&[T]` slice.  The backend owns the storage type; switching to a GPU backend
  would copy `data` to device memory.

- `coeus_ops::sum(&tensor, &backend)` returns `Result<T, _>` because an empty
  tensor has no defined sum.  The `.expect("sum")` pattern is idiomatic.
