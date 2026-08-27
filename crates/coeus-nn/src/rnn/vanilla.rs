// ── Vanilla (Elman) RNN ──

use crate::linear::Linear;
use crate::module::{prefixed_parameters, Module, ModuleError};
use crate::rnn::validation;
use coeus_autograd::Var;
use coeus_core::{Float, MoiraiBackend};
use coeus_tensor::Tensor;

/// Pointwise nonlinearity applied to a vanilla RNN cell's pre-activation.
/// Mirrors PyTorch `RNNCell(nonlinearity=...)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RnnNonlinearity {
    /// `tanh` (PyTorch default).
    Tanh,
    /// `relu`.
    Relu,
}

/// Single-timestep Elman (vanilla) RNN cell.
///
/// Step computation (PyTorch `RNNCell`):
/// ```text
/// h_new = f(x @ W_ih.T + b_ih + h @ W_hh.T + b_hh)
/// ```
/// where `f` is [`RnnNonlinearity`] (`tanh` or `relu`).
#[derive(Clone)]
pub struct RNNCell<T: Float, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend> {
    /// Input-to-hidden projection: `[input_size, hidden_size]`.
    pub w_ih: Linear<T, B>,
    /// Hidden-to-hidden projection: `[hidden_size, hidden_size]`.
    pub w_hh: Linear<T, B>,
    /// Pointwise nonlinearity.
    pub nonlinearity: RnnNonlinearity,
    /// Number of input features per timestep.
    pub input_size: usize,
    /// Number of hidden features.
    pub hidden_size: usize,
}

impl<T: Float + coeus_leto::RandomScalar, B: coeus_ops::BackendOps<T> + Default> RNNCell<T, B> {
    /// Create with Kaiming-initialized weights, zero biases, and the given nonlinearity.
    ///
    /// # Errors
    ///
    /// Returns [`crate::init::InitializationError`] when a size is zero or the
    /// backend's draw fails.
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        nonlinearity: RnnNonlinearity,
    ) -> Result<Self, crate::init::InitializationError<B::Error>>
    where
        B: coeus_ops::RandomInitOps<T>,
    {
        Ok(Self {
            w_ih: Linear::new(input_size, hidden_size, true)?,
            w_hh: Linear::new(hidden_size, hidden_size, true)?,
            nonlinearity,
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
        let batch = validation::cell_input(x.tensor.shape(), self.input_size, "RNNCell")?;
        validation::state(
            h.tensor.shape(),
            batch,
            self.hidden_size,
            "RNNCell",
            "hidden state",
        )?;
        self.step_validated(x, h)
    }

    fn step_validated(
        &self,
        x: &Var<T, B>,
        h: &Var<T, B>,
    ) -> Result<Var<T, B>, ModuleError<B::Error>> {
        let input = self.w_ih.forward(x)?;
        let hidden = self.w_hh.forward(h)?;
        let pre = coeus_autograd::add(&input, &hidden);
        Ok(match self.nonlinearity {
            RnnNonlinearity::Tanh => coeus_autograd::tanh(&pre),
            RnnNonlinearity::Relu => coeus_autograd::relu(&pre),
        })
    }
}

impl<T: Float + coeus_leto::RandomScalar, B: coeus_ops::BackendOps<T> + Default> Module<T, B>
    for RNNCell<T, B>
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
        let batch = validation::cell_input(x.tensor.shape(), self.input_size, "RNNCell")?;
        let backend = B::default();
        let h = Var::new(Tensor::zeros_on([batch, self.hidden_size], &backend), false);
        self.step_validated(x, &h)
    }
}

// ── Rnn (sequence-level) ──

/// Sequence-level vanilla RNN: unrolls [`RNNCell`] across the time axis.
///
/// Input layout: `[batch, seq_len, input_size]`.
/// Output layout: `[batch, seq_len, hidden_size]`.
/// Final hidden state: `h_n` of shape `[batch, hidden_size]`.
#[derive(Clone)]
pub struct Rnn<T: Float, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend> {
    cell: RNNCell<T, B>,
    /// Number of input features per timestep.
    pub input_size: usize,
    /// Number of hidden features.
    pub hidden_size: usize,
}

impl<T: Float + coeus_leto::RandomScalar, B: coeus_ops::BackendOps<T> + Default> Rnn<T, B>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    /// Create with Kaiming-initialized weights, zero biases, and the given nonlinearity.
    ///
    /// # Errors
    ///
    /// Returns [`crate::init::InitializationError`] when a size is zero or the
    /// backend's draw fails.
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        nonlinearity: RnnNonlinearity,
    ) -> Result<Self, crate::init::InitializationError<B::Error>>
    where
        B: coeus_ops::RandomInitOps<T>,
    {
        Ok(Self {
            cell: RNNCell::new(input_size, hidden_size, nonlinearity)?,
            input_size,
            hidden_size,
        })
    }

    /// Unroll over the sequence dimension with a zero initial hidden state.
    ///
    /// Returns `(output, h_n)`:
    /// - `output`: `[batch, seq_len, hidden_size]` — all hidden states stacked.
    /// - `h_n`: `[batch, hidden_size]` — final hidden state.
    pub fn forward_seq(
        &self,
        x: &Var<T, B>,
    ) -> Result<(Var<T, B>, Var<T, B>), ModuleError<B::Error>> {
        let (batch, seq_len) =
            validation::sequence_input(x.tensor.shape(), self.input_size, "Rnn")?;
        let backend = B::default();

        let mut h = Var::new(Tensor::zeros_on([batch, self.hidden_size], &backend), false);
        let mut outputs: Vec<Var<T, B>> = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let x_t_3d = coeus_autograd::slice(x, &[(0, batch), (t, t + 1), (0, self.input_size)]);
            let x_t = coeus_autograd::reshape(&x_t_3d, vec![batch, self.input_size]);
            let h_new = self.cell.step_validated(&x_t, &h)?;
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
    for Rnn<T, B>
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
