// ── GRUCell ──

use crate::linear::Linear;
use crate::module::Module;
use coeus_autograd::Var;
use coeus_core::{Float, MoiraiBackend};
use coeus_tensor::Tensor;

/// Single-timestep Gated Recurrent Unit cell.
///
/// Gate computation:
/// ```text
/// gates_r = sigmoid(x @ W_ih_r.T + h @ W_hh_r.T + b_r)   — reset gate
/// gates_z = sigmoid(x @ W_ih_z.T + h @ W_hh_z.T + b_z)   — update gate
/// n       = tanh(x @ W_ih_n.T + r ⊙ (h @ W_hh_n.T) + b_n) — new gate
///
/// h_new = (1 - z) ⊙ n + z ⊙ h
/// ```
///
/// Fused implementation uses a single `[3*H]` projection for efficiency.
#[derive(Clone)]
pub struct GRUCell<T: Float, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend> {
    pub w_ih: Linear<T, B>,
    pub w_hh: Linear<T, B>,
    pub input_size: usize,
    pub hidden_size: usize,
}

impl<T: Float + coeus_leto::RandomScalar, B: coeus_ops::BackendOps<T> + Default> GRUCell<T, B> {
    /// Create with Xavier-initialized weights and zero biases.
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let w_ih = Linear::new(input_size, 3 * hidden_size, true);
        let w_hh = Linear::new(hidden_size, 3 * hidden_size, true);
        Self { w_ih, w_hh, input_size, hidden_size }
    }

    /// Forward step.
    ///
    /// - `x`: `[batch, input_size]`
    /// - `h`: `[batch, hidden_size]`
    ///
    /// Returns `h_new` of shape `[batch, hidden_size]`.
    pub fn step(&self, x: &Var<T, B>, h: &Var<T, B>) -> Var<T, B> {
        let hs = self.hidden_size;
        let ih = self.w_ih.forward(x);
        let hh = self.w_hh.forward(h);

        let slice_ih = |start: usize, end: usize| -> Var<T, B> {
            let batch = ih.tensor.shape()[0];
            coeus_autograd::slice(&ih, &[(0, batch), (start, end)])
        };
        let slice_hh = |start: usize, end: usize| -> Var<T, B> {
            let batch = hh.tensor.shape()[0];
            coeus_autograd::slice(&hh, &[(0, batch), (start, end)])
        };

        let r = coeus_autograd::sigmoid(
            &coeus_autograd::add(&slice_ih(0, hs), &slice_hh(0, hs))
        );
        let z = coeus_autograd::sigmoid(
            &coeus_autograd::add(&slice_ih(hs, 2 * hs), &slice_hh(hs, 2 * hs))
        );
        // n = tanh(ih_n + r * hh_n)
        let n = coeus_autograd::tanh(
            &coeus_autograd::add(
                &slice_ih(2 * hs, 3 * hs),
                &coeus_autograd::mul(&r, &slice_hh(2 * hs, 3 * hs)),
            )
        );

        // h_new = (1 - z) * n + z * h
        let backend = B::default();
        let ones = Var::new(
            coeus_tensor::Tensor::ones_on(z.tensor.shape_cloned(), &backend),
            false,
        );
        let one_minus_z = coeus_autograd::sub(&ones, &z);
        coeus_autograd::add(
            &coeus_autograd::mul(&one_minus_z, &n),
            &coeus_autograd::mul(&z, h),
        )
    }
}

impl<T: Float + coeus_leto::RandomScalar, B: coeus_ops::BackendOps<T> + Default> Module<T, B>
    for GRUCell<T, B>
{
    fn parameters(&self) -> Vec<Var<T, B>> {
        let mut p = self.w_ih.parameters();
        p.extend(self.w_hh.parameters());
        p
    }

    fn forward(&self, x: &Var<T, B>) -> Var<T, B> {
        let batch = x.tensor.shape()[0];
        let backend = B::default();
        let h = Var::new(Tensor::zeros_on([batch, self.hidden_size], &backend), false);
        self.step(x, &h)
    }
}
