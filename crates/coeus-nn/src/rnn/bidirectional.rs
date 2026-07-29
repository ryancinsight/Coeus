// ── Bidirectional recurrent wrapper ──

use crate::module::{prefixed_parameters, Module, ModuleError};
use crate::rnn::validation;
use coeus_autograd::Var;
use coeus_core::Scalar;

/// Bidirectional wrapper over any sequence [`Module`] (e.g. [`Rnn`](super::Rnn),
/// [`Gru`](super::Gru), [`Lstm`](super::Lstm)).
///
/// Runs `forward_module` over the input and `backward_module` over the
/// time-reversed input, then concatenates the two `[batch, seq, hidden]`
/// outputs along the hidden axis to `[batch, seq, 2*hidden]` — matching PyTorch
/// `bidirectional=True`. Generic over the cell type, so no per-cell code: the
/// reversal and concatenation reuse the tracked `flip`/`cat` autograd ops.
///
/// The two sub-modules carry independent parameters (PyTorch likewise learns
/// separate forward/backward weights).
#[derive(Clone)]
pub struct Bidirectional<M> {
    /// Module applied to the sequence in forward (left-to-right) order.
    pub forward_module: M,
    /// Module applied to the time-reversed sequence.
    pub backward_module: M,
}

impl<M> Bidirectional<M> {
    /// Wrap a forward and a (time-reversed) backward sequence module.
    pub fn new(forward_module: M, backward_module: M) -> Self {
        Self {
            forward_module,
            backward_module,
        }
    }
}

impl<T, B, M> Module<T, B> for Bidirectional<M>
where
    T: Scalar,
    B: coeus_ops::BackendOps<T> + Default,
    M: Module<T, B>,
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    fn parameters(&self) -> Vec<Var<T, B>> {
        let mut p = self.forward_module.parameters();
        p.extend(self.backward_module.parameters());
        p
    }

    fn named_parameters(&self) -> Vec<coeus_autograd::Parameter<T, B>> {
        let mut parameters = prefixed_parameters("forward", &self.forward_module);
        parameters.extend(prefixed_parameters("backward", &self.backward_module));
        parameters
    }

    /// `x`: `[batch, seq_len, input_size]` → `[batch, seq_len, 2*hidden_size]`.
    fn forward(&self, x: &Var<T, B>) -> Result<Var<T, B>, ModuleError<B::Error>> {
        let (batch, sequence) = validation::sequence_layout(x.tensor.shape(), "Bidirectional")?;
        let fwd = self.forward_module.forward(x)?;
        validation::child_sequence_output(
            fwd.tensor.shape(),
            batch,
            sequence,
            "Bidirectional",
            "forward output",
        )?;
        // Reverse along the time axis, run the backward module, then restore order.
        let rev_x = coeus_autograd::flip(x, 1);
        let bwd_rev = self.backward_module.forward(&rev_x)?;
        validation::child_sequence_output(
            bwd_rev.tensor.shape(),
            batch,
            sequence,
            "Bidirectional",
            "backward output",
        )?;
        validation::matching_child_outputs(fwd.tensor.shape(), bwd_rev.tensor.shape())?;
        let bwd = coeus_autograd::flip(&bwd_rev, 1);
        Ok(coeus_autograd::cat(&[&fwd, &bwd], 2))
    }
}
