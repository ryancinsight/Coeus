# Elementwise and Reduction Ops

## Elementwise Operations

Elementwise ops apply a function independently to each element.
In autograd, they register a backward node that computes the gradient.

```rust,ignore
let c = &a + &b;        // AddOp
let d = &a * &b;        // MulOp
let e = a.relu();       // ReluOp
let f = a.gelu();       // GeluOp
let g = a.sigmoid();    // SigmoidOp
```

All arithmetic operators (`+`, `-`, `*`, `/`) are overloaded on
`Var` and call the appropriate autograd op.

## Reduction Operations

Reductions collapse one or more axes:

```rust,ignore
let total = a.sum();              // scalar sum over all elements
let row_sums = a.sum_axis(1)?;   // sum along axis 1 -> shape [M]
let mean     = a.mean_axis(0)?;  // mean along axis 0 -> shape [N]
let max_val  = a.max_axis(2)?;   // max along axis 2
let std_dev  = a.std_dev_axis(1, 0)?;  // std dev (biased or unbiased)
```

Reduction ops support backward passes that broadcast the upstream gradient
back to the original shape.

## Scan Operations

`scan_sum` and `scan_product` compute prefix sums/products:

```rust,ignore
let cumsum = a.scan_sum(0)?;   // cumulative sum along axis 0
```

## Norm Operations

```rust,ignore
let l2 = a.norm2();               // Euclidean norm (scalar)
let frobenius = a.norm_frobenius(); // Frobenius norm
```
