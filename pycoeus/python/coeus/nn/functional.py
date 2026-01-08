from .._coeus import (
    relu, sigmoid, tanh, gelu, silu, leaky_relu, elu,
    mse_loss, cross_entropy, nll_loss, bce_with_logits_loss, softmax,
    max_pool2d, avg_pool2d,
    conv1d, conv2d, conv_transpose2d, conv3d,
    matmul, bmm, addmm,
    dropout, layer_norm, linear,
    reshape, view, flatten, squeeze, unsqueeze, transpose, permute,
)

# Aliases for PyTorch compatibility
cross_entropy_loss = cross_entropy
