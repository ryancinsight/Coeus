from .._coeus import (
    relu, sigmoid, tanh, gelu, silu, leaky_relu, elu, softmax, log_softmax,
    mse_loss, cross_entropy, nll_loss, l1_loss, smooth_l1_loss, binary_cross_entropy, bce_with_logits_loss,
    max_pool2d, avg_pool2d,
    conv1d, conv2d, conv3d, conv_transpose1d, conv_transpose2d, conv_transpose3d,
    matmul, bmm, addmm,
    dropout, layer_norm, batch_norm, linear, bilinear,
    pairwise_distance, cosine_similarity,
    reshape, view, flatten, squeeze, unsqueeze, transpose, permute,
    cat, stack,
)

# Aliases for PyTorch compatibility
cross_entropy_loss = cross_entropy
binary_cross_entropy_with_logits = bce_with_logits_loss
