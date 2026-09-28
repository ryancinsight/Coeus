# Linear interpolation scalar migration

Linear interpolation now uses the image and coordinate scalar type throughout
forward evaluation and reverse mode. `f32`, `f64`, `F16`, and `Bf16` are
supported where the selected backend implements the required operations.

Calls that rely on type inference need no source change. Calls with an
explicit turbofish append the scalar type parameter after the existing
dimension, backend, and boundary-policy parameters:

```rust,ignore
// Before
linear_interpolation::<2, _, _>(&image, &grid, Replicate)?;

// After
linear_interpolation::<2, _, _, _>(&image, &grid, Replicate)?;
```

The same change applies to `coeus_ops::linear_interpolation_backward` and
`coeus_autograd::linear_interpolation`. The scalar remains inferable from the
input tensors, so `_` is sufficient when a caller does not need to name it.

Gradient and autograd node type parameters preserve the previous `f32`
annotation through a trailing default:

```rust,ignore
let gradients: InterpolationGradients<MyBackend> = /* ... */;
let low_precision: InterpolationGradients<MyBackend, Bf16> = /* ... */;
```

Neighbour indices are now clamped against the integer tensor extent. This
preserves distinct adjacent voxels when a low-precision scalar cannot
represent every integer in the extent.
