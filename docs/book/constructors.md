# Constructors

Coeus tensors are created through a set of factory functions that mirror
NumPy and PyTorch conventions.

## Constant Tensors

```rust,ignore
let zeros  = Tensor::<f32>::zeros([256, 256]);    // all zeros
let ones   = Tensor::<f32>::ones([64, 64, 3]);    // all ones
let filled = Tensor::<f32>::full([8, 8], 3.14);   // constant fill
```

## Identity and Diagonal

```rust,ignore
let eye = Tensor::<f32>::eye(128);                // identity matrix
let diag = Tensor::diag(&values_vec);             // diagonal from 1D tensor
```

## Range Tensors

```rust,ignore
let lin = Tensor::<f32>::linspace(0.0, 1.0, 101); // 101 evenly spaced values
let rng = Tensor::<f32>::arange(0.0, 10.0, 0.5);  // [0.0, 0.5, 1.0, ..., 9.5]
```

## From Existing Data

```rust,ignore
let from_slice = Tensor::from_slice([4, 4], &data)?;  // validates shape
let from_fn   = Tensor::from_fn([8, 8], |i, j| (i + j) as f32);
```

## Random Tensors

```rust,ignore
let uniform = Tensor::<f32>::rand([256, 256]);           // U[0, 1)
let normal  = Tensor::<f32>::randn([256, 256]);          // N(0, 1)
let init    = xavier_uniform([in_features, out_features], 1.0);
```

Random tensors are generated using the active Tyche sampling backend.

## Shape and Data Type

All constructors accept a shape as `impl Into<Shape>` (slices, arrays, or
vecs). The scalar type `T: Scalar` covers `f32`, `f64`, `f16`,
`bf16`, and integer types.
