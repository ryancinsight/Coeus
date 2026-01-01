from .._coeus import (
    relu, sigmoid, tanh, gelu, silu, leaky_relu, elu,
    mse_loss, cross_entropy, softmax, max_pool2d, avg_pool2d,
    dropout, layer_norm, bce_with_logits_loss,
)

# Aliases for PyTorch compatibility
cross_entropy_loss = cross_entropy
