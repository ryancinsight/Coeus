// ── LSTMCell ──

use crate::linear::Linear;
use crate::module::Module;
use coeus_autograd::Var;
use coeus_core::{Float, MoiraiBackend};
use coeus_tensor::Tensor;

/// Single-timestep Long Short-Term Memory cell.
///
/// Gate computation:
/// ```text
/// gates = x @ W_ih.T + b_ih + h @ W_hh.T + b_hh   // [batch, 4*H]
/// i = sigmoid(gates[:, 0:H])        — input gate
/// f = sigmoid(gates[:, H:2H])       — forget gate
/// g = tanh(gates[:, 2H:3H])         — cell gate
/// o = sigmoid(gates[:, 3H:4H])      — output gate
///
/// c_new = f ⊙ c + i ⊙ g
/// h_new = o ⊙ tanh(c_new)
/// ```
#[derive(Clone)]
pub struct LSTMCell<T: Float, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend> {
    pub w_ih: Linear<T, B>,
    pub w_hh: Linear<T, B>,
    pub input_size: usize,
    pub hidden_size: usize,
}

impl<T: Float + coeus_leto::RandomScalar, B: coeus_ops::BackendOps<T> + Default> LSTMCell<T, B> {
    /// Create with Xavier-initialized weights and zero biases.
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let w_ih = Linear::new(input_size, 4 * hidden_size, true);
        let w_hh = Linear::new(hidden_size, 4 * hidden_size, true);
        Self {
            w_ih,
            w_hh,
            input_size,
            hidden_size,
        }
    }

    /// Forward step.
    ///
    /// - `x`: `[batch, input_size]`
    /// - `h`: `[batch, hidden_size]`
    /// - `c`: `[batch, hidden_size]`
    ///
    /// Returns `(h_new, c_new)`, both `[batch, hidden_size]`.
    pub fn step(&self, x: &Var<T, B>, h: &Var<T, B>, c: &Var<T, B>) -> (Var<T, B>, Var<T, B>) {
        let hs = self.hidden_size;
        let gates = coeus_autograd::add(&self.w_ih.forward(x), &self.w_hh.forward(h));

        let slice = |start: usize, end: usize| -> Var<T, B> {
            let batch = gates.tensor.shape()[0];
            coeus_autograd::slice(&gates, &[(0, batch), (start, end)])
        };

        let i_g = coeus_autograd::sigmoid(&slice(0, hs));
        let f_g = coeus_autograd::sigmoid(&slice(hs, 2 * hs));
        let g_g = coeus_autograd::tanh(&slice(2 * hs, 3 * hs));
        let o_g = coeus_autograd::sigmoid(&slice(3 * hs, 4 * hs));

        let c_new = coeus_autograd::add(
            &coeus_autograd::mul(&f_g, c),
            &coeus_autograd::mul(&i_g, &g_g),
        );
        let h_new = coeus_autograd::mul(&o_g, &coeus_autograd::tanh(&c_new));
        (h_new, c_new)
    }
}

impl<T: Float + coeus_leto::RandomScalar, B: coeus_ops::BackendOps<T> + Default> Module<T, B>
    for LSTMCell<T, B>
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
        let c = Var::new(Tensor::zeros_on([batch, self.hidden_size], &backend), false);
        self.step(x, &h, &c).0
    }
}
