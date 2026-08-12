# Convolution

Coeus provides 1D, 2D, and 3D convolution with full forward and backward passes,
dispatching through Hephaestus `ConvolutionOps`.

## Forward Pass

```rust,ignore
use coeus::autograd::{conv2d, Var};

// weight: [out_channels, in_channels/groups, kH, kW]
// bias:   [out_channels] or None
let output = conv2d(
    &input,   // [N, C_in, H, W]
    &weight,
    bias.as_ref(),
    stride,   // [sH, sW]
    padding,  // [pH, pW]
    dilation, // [dH, dW]
    groups,
)?;
```

Equivalent functions exist for `conv1d` and `conv3d`.

## Transposed Convolution

```rust,ignore
let output = conv_transpose2d(&input, &weight, bias.as_ref(), stride, padding, output_padding, groups, dilation)?;
```

## Plan and Dispatch

Hephaestus plans the convolution dispatch once per unique configuration:

```rust,ignore
let plan = device.plan_convolution_forward(input_shape, weight_shape, stride, padding, dilation, groups)?;
```

The plan is reused across batches and training iterations.

## Backward Pass

Autograd computes three gradients:
- `d_input`: gradient with respect to the input tensor
- `d_weight`: gradient with respect to the convolution kernel
- `d_bias`: gradient with respect to the bias vector (if present)

All three use the corresponding `hephaestus_core::ConvolutionPlan`
dispatch path.
