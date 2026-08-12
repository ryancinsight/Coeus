# Layouts and Views

Coeus tensors support non-contiguous views through a stride-based layout
system, enabling zero-copy transpositions, slices, and broadcasts.

## Shape and Strides

A tensor's **shape** is the per-axis element count. Its **strides** are the
per-axis step sizes (in elements) to reach the next element along that axis.

For a row-major [M, N] tensor: `strides = [N, 1]`.
For a column-major [M, N] tensor: `strides = [1, M]`.

## Zero-Copy Views

```rust,ignore
let a = Tensor::<f32>::zeros([8, 8, 3]);

let row  = a.slice(0..1, .., ..);     // shape [1, 8, 3]
let t    = a.transpose();             // shape [3, 8, 8]; no copy
let view = a.reshape([64, 3])?;       // shape [64, 3]; no copy if contiguous
```

All views share the underlying buffer. Writes to a shared view trigger COW.

## `permute`

`permute(axes)` rearranges axes in the specified order:

```rust,ignore
let b = a.permute([2, 0, 1]);  // [8, 8, 3] -> [3, 8, 8]
```

## `contiguous`

`contiguous()` returns a new tensor with standard row-major strides,
copying data if the current layout is non-contiguous:

```rust,ignore
let c = b.contiguous();
```

## `StridedView` in Hephaestus

When passing views to Hephaestus op dispatch, `StridedView` carries
the offset, shape, and strides so the GPU kernel can index correctly
without materializing a contiguous copy.
