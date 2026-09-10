// ── Attention and Transformer architecture tests ──
//
// Validates:
//   1. SDPA forward shape
//   2. Causal mask correctness (upper-triangle → zero attention)
//   3. Gradient flow: q.grad, k.grad, v.grad non-None and finite
//   4. MHA output shape
//   5. TransformerEncoderLayer output shape and non-zero gradients
//   6. SinusoidalEncoding shape and non-zero values

#[cfg(test)]
mod encoder;
#[cfg(test)]
mod mha_mask;
#[cfg(test)]
mod tests;
