# Constructors

Coeus tensors are created through a set of factory functions that mirror
NumPy and PyTorch conventions.

## Constant Tensors

```rust,ignore
let zeros  = Tensor::<f32>::zeros([256, 256])?;    // all zeros
let ones   = Tensor::<f32>::ones([64, 64, 3])?;    // all ones
let filled = Tensor::<f32>::full([8, 8], 3.14)?;   // constant fill
```

## Identity and Diagonal

```rust,ignore
let eye = Tensor::<f32>::eye(128)?;                  // identity matrix
let diag = coeus_ops::diag(&values, 0, &backend)?;   // diagonal from 1D tensor
```

## Range Tensors

```rust,ignore
let lin = Tensor::<f32>::linspace(0.0, 1.0, 101)?; // 101 evenly spaced values
let rng = Tensor::<f32>::arange(10)?;              // [0.0, 1.0, ..., 9.0]
```

## From Existing Data

```rust,ignore
let from_slice = Tensor::from_slice([4, 4], &data)?;  // validates shape
let from_fn = Tensor::from_fn([8, 8], |index| (index[0] + index[1]) as f32)?;
```

## Random Tensors

```rust,ignore
let mut weights = Var::new(Tensor::<f32>::zeros([256, 256])?, true)?;
coeus_nn::init::uniform(&mut weights, 0.0, 1.0)?;  // U[0, 1)
coeus_nn::init::normal(&mut weights, 0.0, 1.0)?;   // N(0, 1)
coeus_nn::init::xavier_uniform(&mut weights, 256, 256)?;
```

Random tensors are generated using the active Tyche sampling backend.

## Shape and Data Type

All constructors accept a shape as `impl Into<Shape>` (slices, arrays, or
vecs). The scalar type `T: Scalar` covers `f32`, `f64`, `f16`,
`bf16`, and integer types.
