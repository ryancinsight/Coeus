//! SwiGLU gated feed-forward unit.
//!
//! `SwiGLU(x) = silu(W_inner · x) ⊙ (W_outer · x)` — the Swish/SiLU-gated linear
//! unit used in modern transformer feed-forward blocks (PaLM, LLaMA). Two
//! parallel linear projections of the same input share the input tensor; the
//! inner projection is SiLU-gated and multiplied element-wise by the outer
//! projection. Burn parity: `burn::nn::SwiGlu`.

use crate::activation::silu;
use crate::linear::Linear;
use crate::module::Module;
use coeus_autograd::Var;
use coeus_core::{Float, MoiraiBackend};

/// SwiGLU gated linear unit projecting `d_input -> d_output`.
///
/// Composed of two [`Linear`] layers over the same input: the inner projection
/// is SiLU-gated, the outer is the gate's value path, combined by an
/// element-wise product. Generic over any [`Float`] scalar and backend.
#[derive(Clone)]
pub struct SwiGlu<T: Float, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend> {
    /// Inner (SiLU-gated) projection, weight `[d_output × d_input]`.
    pub linear_inner: Linear<T, B>,
    /// Outer (value) projection, weight `[d_output × d_input]`.
    pub linear_outer: Linear<T, B>,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> SwiGlu<T, B> {
    /// Create a SwiGLU unit projecting `d_input -> d_output`, with optional bias
    /// on both linear layers.
    pub fn new(d_input: usize, d_output: usize, bias: bool) -> Self {
        Self {
            linear_inner: Linear::new(d_input, d_output, bias),
            linear_outer: Linear::new(d_input, d_output, bias),
        }
    }
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for SwiGlu<T, B> {
    fn parameters(&self) -> Vec<Var<T, B>> {
        let mut params = self.linear_inner.parameters();
        params.extend(self.linear_outer.parameters());
        params
    }

    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
        let gated = silu(&self.linear_inner.forward(input));
        let outer = self.linear_outer.forward(input);
        coeus_autograd::mul(&gated, &outer)
    }
}
