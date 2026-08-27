// ── GRUCell ──

use crate::linear::Linear;
use crate::module::{prefixed_parameters, Module, ModuleError};
use crate::rnn::validation;
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
    /// Input-to-hidden projection: `[input_size, 3*hidden_size]`.
    pub w_ih: Linear<T, B>,
    /// Hidden-to-hidden projection: `[hidden_size, 3*hidden_size]`.
    pub w_hh: Linear<T, B>,
    /// Number of input features per timestep.
    pub input_size: usize,
    /// Number of hidden features.
    pub hidden_size: usize,
}

impl<T: Float + coeus_leto::RandomScalar, B: coeus_ops::BackendOps<T> + Default> GRUCell<T, B> {
    /// Create with Kaiming-initialized weights and zero biases.
    ///
    /// # Errors
    ///
    /// Returns [`crate::init::InitializationError`] when a size is zero or the
    /// backend's draw fails.
    pub fn new(
        input_size: usize,
        hidden_size: usize,
    ) -> Result<Self, crate::init::InitializationError<B::Error>>
    where
        B: coeus_ops::RandomInitOps<T>,
    {
        let w_ih = Linear::new(input_size, 3 * hidden_size, true)?;
        let w_hh = Linear::new(hidden_size, 3 * hidden_size, true)?;
        Ok(Self {
            w_ih,
            w_hh,
            input_size,
            hidden_size,
        })
    }

    /// Forward step.
    ///
    /// - `x`: `[batch, input_size]`
    /// - `h`: `[batch, hidden_size]`
    ///
    /// Returns `h_new` of shape `[batch, hidden_size]`.
    pub fn step(&self, x: &Var<T, B>, h: &Var<T, B>) -> Result<Var<T, B>, ModuleError<B::Error>> {
        let batch = validation::cell_input(x.tensor.shape(), self.input_size, "GRUCell")?;
        validation::state(
            h.tensor.shape(),
            batch,
            self.hidden_size,
            "GRUCell",
            "hidden state",
        )?;
        self.step_validated(x, h, batch)
    }

    fn step_validated(
        &self,
        x: &Var<T, B>,
        h: &Var<T, B>,
        batch: usize,
    ) -> Result<Var<T, B>, ModuleError<B::Error>> {
        let hs = self.hidden_size;
        let ih = self.w_ih.forward(x)?;
        let hh = self.w_hh.forward(h)?;

        let slice_ih = |start: usize, end: usize| -> Var<T, B> {
            coeus_autograd::slice(&ih, &[(0, batch), (start, end)])
        };
        let slice_hh = |start: usize, end: usize| -> Var<T, B> {
            coeus_autograd::slice(&hh, &[(0, batch), (start, end)])
        };

        let r = coeus_autograd::sigmoid(&coeus_autograd::add(&slice_ih(0, hs), &slice_hh(0, hs)));
        let z = coeus_autograd::sigmoid(&coeus_autograd::add(
            &slice_ih(hs, 2 * hs),
            &slice_hh(hs, 2 * hs),
        ));
        // n = tanh(ih_n + r * hh_n)
        let n = coeus_autograd::tanh(&coeus_autograd::add(
            &slice_ih(2 * hs, 3 * hs),
            &coeus_autograd::mul(&r, &slice_hh(2 * hs, 3 * hs)),
        ));

        // h_new = (1 - z) * n + z * h
        let backend = B::default();
        let ones = Var::new(
            coeus_tensor::Tensor::ones_on(z.tensor.shape_cloned(), &backend),
            false,
        );
        let one_minus_z = coeus_autograd::sub(&ones, &z);
        Ok(coeus_autograd::add(
            &coeus_autograd::mul(&one_minus_z, &n),
            &coeus_autograd::mul(&z, h),
        ))
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

    fn named_parameters(&self) -> Vec<coeus_autograd::Parameter<T, B>> {
        let mut parameters = prefixed_parameters("input", &self.w_ih);
        parameters.extend(prefixed_parameters("hidden", &self.w_hh));
        parameters
    }

    fn forward(&self, x: &Var<T, B>) -> Result<Var<T, B>, ModuleError<B::Error>> {
        let batch = validation::cell_input(x.tensor.shape(), self.input_size, "GRUCell")?;
        let backend = B::default();
        let h = Var::new(Tensor::zeros_on([batch, self.hidden_size], &backend), false);
        self.step_validated(x, &h, batch)
    }
}

// ── Gru (sequence-level) ──

/// Sequence-level GRU module: unrolls [`GRUCell`] across the time axis.
///
/// Input layout: `[batch, seq_len, input_size]`.
/// Output layout: `[batch, seq_len, hidden_size]`.
/// Final hidden state: `h_n` of shape `[batch, hidden_size]`.
///
/// Initial hidden state defaults to zeros; use [`Gru::forward_seq`]
/// for full `(output, h_n)` when the final state is needed, or
/// `Module::forward` when only the output sequence is needed.
///
/// # Example
/// ```text
/// zeros input ⟹ all gate pre-activations = 0
///   r = z = σ(0) = 0.5,  n = tanh(0 + 0.5·0) = 0
///   h_new = (1−z)·n + z·h = 0
/// ∴ output is all zeros for any sequence length.
/// ```
#[derive(Clone)]
pub struct Gru<T: Float, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend> {
    cell: GRUCell<T, B>,
    /// Number of input features per timestep.
    pub input_size: usize,
    /// Number of hidden features.
    pub hidden_size: usize,
}

impl<T: Float + coeus_leto::RandomScalar, B: coeus_ops::BackendOps<T> + Default> Gru<T, B>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    /// Create with Kaiming-initialized weights and zero biases.
    ///
    /// # Errors
    ///
    /// Returns [`crate::init::InitializationError`] when a size is zero or the
    /// backend's draw fails.
    pub fn new(
        input_size: usize,
        hidden_size: usize,
    ) -> Result<Self, crate::init::InitializationError<B::Error>>
    where
        B: coeus_ops::RandomInitOps<T>,
    {
        Ok(Self {
            cell: GRUCell::new(input_size, hidden_size)?,
            input_size,
            hidden_size,
        })
    }

    /// Unroll over the sequence dimension with zero initial hidden state.
    ///
    /// Returns `(output, h_n)`:
    /// - `output`: `[batch, seq_len, hidden_size]` — all hidden states stacked.
    /// - `h_n`: `[batch, hidden_size]` — final hidden state.
    pub fn forward_seq(
        &self,
        x: &Var<T, B>,
    ) -> Result<(Var<T, B>, Var<T, B>), ModuleError<B::Error>> {
        let (batch, seq_len) =
            validation::sequence_input(x.tensor.shape(), self.input_size, "Gru")?;
        let backend = B::default();

        let mut h = Var::new(Tensor::zeros_on([batch, self.hidden_size], &backend), false);

        let mut outputs: Vec<Var<T, B>> = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let x_t_3d = coeus_autograd::slice(x, &[(0, batch), (t, t + 1), (0, self.input_size)]);
            let x_t = coeus_autograd::reshape(&x_t_3d, vec![batch, self.input_size]);
            let h_new = self.cell.step_validated(&x_t, &h, batch)?;
            outputs.push(coeus_autograd::reshape(
                &h_new,
                vec![batch, 1, self.hidden_size],
            ));
            h = h_new;
        }

        let refs: Vec<&Var<T, B>> = outputs.iter().collect();
        let output = coeus_autograd::cat(&refs, 1);
        Ok((output, h))
    }
}

impl<T: Float + coeus_leto::RandomScalar, B: coeus_ops::BackendOps<T> + Default> Module<T, B>
    for Gru<T, B>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    fn parameters(&self) -> Vec<Var<T, B>> {
        self.cell.parameters()
    }

    fn named_parameters(&self) -> Vec<coeus_autograd::Parameter<T, B>> {
        prefixed_parameters("cell", &self.cell)
    }

    /// Returns `output` of shape `[batch, seq_len, hidden_size]`.
    fn forward(&self, x: &Var<T, B>) -> Result<Var<T, B>, ModuleError<B::Error>> {
        Ok(self.forward_seq(x)?.0)
    }
}
