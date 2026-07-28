//! Differentiable selective scan — the Mamba/S6 linear state-space recurrence.

use crate::{grad_buffer::GradBuffer, node::BackwardNode, var::Var};
use coeus_core::{Backend, CpuAddressableStorage, CpuAddressableStorageMut};
use coeus_tensor::Tensor;
use std::sync::Arc;

// ── Selective scan (linear first-order recurrence), Mamba/S6 core ──
//
// Convention (pinned; the recurrence primitive, C/D projection is composed by
// the caller from existing element-wise / reduction ops):
//   * `a_bar` (the discretized state transition Ā_t) and `u` (the input
//     contribution B̄_t·x_t, already discretized by the caller) share ONE
//     identical shape `[batch, length, ...]`.
//   * The recurrence runs along axis 1 (the `length`/time axis), matching the
//     reference `parallel_scan` in ritk-model. Every trailing axis (state, or
//     `(inner, state)`, or any rank ≥ 2 tail) is an independent element-wise
//     recurrence channel.
//   * Recurrence (0-indexed along the length axis, `h_{-1} = 0`):
//       h_0 = u_0
//       h_t = a_bar_t ⊙ h_{t-1} + u_t   for t = 1..length-1
//     The output `h` has the same shape as the inputs and is the h-sequence
//     (NOT the C-projected y): `y_t = (C_t ⊙ h_t).sum(state)` composes outside
//     this node from `mul`/`sum_axis`.
//   * `a_bar_0` does not affect any output (h_0 = u_0 is independent of it), so
//     ∂L/∂a_bar_0 = 0 by construction — this is exact, not a truncation.
//
// This is a *linear* recurrence, so the reverse pass is the well-known
// reverse-time associative scan. Writing g_t = ∂L/∂h_t (upstream grad on h_t
// plus the recurrence feedback), iterating t from length-1 down to 0:
//       g_t          = grad_out_t + a_bar_{t+1} ⊙ g_{t+1}   (2nd term absent at t = L-1)
//       grad_u_t     = g_t
//       grad_a_bar_t = g_t ⊙ h_{t-1}                        (0 at t = 0)
// with the feedback carry a_bar_t ⊙ g_t propagated to step t-1.

/// Flatten a `[batch, length, ...]` shape into `(outer, length, inner)`.
///
/// `outer = shape[0]`, `length = shape[1]`, `inner = ∏ shape[2..]`; the
/// element at `(o, t, i)` lives at contiguous offset `(o * length + t) * inner + i`.
#[inline]
fn scan_dims(shape: &[usize]) -> (usize, usize, usize) {
    let outer = shape[0];
    let length = shape[1];
    let inner: usize = shape[2..].iter().product();
    (outer, length, inner)
}

/// Sequential forward recurrence `h_t = a_bar_t ⊙ h_{t-1} + u_t`, `h_0 = u_0`.
fn selective_scan_forward<B>(
    a_bar: &Tensor<f32, B>,
    u: &Tensor<f32, B>,
) -> Result<Tensor<f32, B>, B::Error>
where
    B: Backend + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    let a_bar = a_bar.to_contiguous()?;
    let u = u.to_contiguous()?;
    let a = a_bar.as_slice();
    let u = u.as_slice();
    let (outer, length, inner) = scan_dims(a_bar.shape());

    let mut h = vec![0.0f32; a.len()];
    for o in 0..outer {
        for i in 0..inner {
            let base = o * length * inner + i;
            // t = 0: h_0 = u_0 (h_{-1} = 0).
            h[base] = u[base];
            for t in 1..length {
                let idx = base + t * inner;
                let prev = idx - inner;
                h[idx] = a[idx] * h[prev] + u[idx];
            }
        }
    }

    Tensor::from_slice_on(a_bar.shape().to_vec(), &h, &B::default())
}

/// Reverse pass: gradients for `(a_bar, u)` from the saved transition and the
/// saved forward output `h`.
fn selective_scan_backward<B>(
    a_bar: &Tensor<f32, B>,
    h: &Tensor<f32, B>,
    grad_output: &Tensor<f32, B>,
) -> Result<(Tensor<f32, B>, Tensor<f32, B>), B::Error>
where
    B: Backend + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    let a_bar = a_bar.to_contiguous()?;
    let h = h.to_contiguous()?;
    let grad_output = grad_output.to_contiguous()?;
    let a = a_bar.as_slice();
    let h = h.as_slice();
    let go = grad_output.as_slice();
    let (outer, length, inner) = scan_dims(a_bar.shape());

    let mut grad_a = vec![0.0f32; a.len()];
    let mut grad_u = vec![0.0f32; a.len()];
    for o in 0..outer {
        for i in 0..inner {
            let base = o * length * inner + i;
            // carry = a_bar_{t+1} ⊙ g_{t+1}, the recurrence feedback into g_t.
            let mut carry = 0.0f32;
            for t in (0..length).rev() {
                let idx = base + t * inner;
                let g_t = go[idx] + carry;
                grad_u[idx] = g_t;
                if t >= 1 {
                    // ∂h_t/∂a_bar_t = h_{t-1}; a_bar_0 has no effect (h_0 = u_0).
                    grad_a[idx] = g_t * h[idx - inner];
                }
                // ∂h_t/∂h_{t-1} = a_bar_t; feed g_t back to step t-1.
                carry = a[idx] * g_t;
            }
        }
    }

    let backend = B::default();
    Ok((
        Tensor::from_slice_on(a_bar.shape().to_vec(), &grad_a, &backend)?,
        Tensor::from_slice_on(a_bar.shape().to_vec(), &grad_u, &backend)?,
    ))
}

/// Reverse-mode node for [`selective_scan`].
struct SelectiveScanNode<B>
where
    B: Backend + coeus_ops::BackendOps<f32> + Default,
{
    output_grad: Arc<GradBuffer<f32, B>>,
    inputs: Vec<Var<f32, B>>,
    a_bar: Tensor<f32, B>,
    /// Saved forward output `h` — the reverse pass needs `h_{t-1}` for `grad_a_bar_t`.
    h: Tensor<f32, B>,
}

impl<B> BackwardNode<f32, B> for SelectiveScanNode<B>
where
    B: Backend + coeus_ops::BackendOps<f32> + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    fn op_name(&self) -> &'static str {
        "selective_scan"
    }

    fn output_grad(&self) -> &Arc<GradBuffer<f32, B>> {
        &self.output_grad
    }

    fn inputs(&self) -> &[Var<f32, B>] {
        &self.inputs
    }

    fn backward(
        &self,
        grad_out: &Tensor<f32, B>,
        input_grads: &[Option<Arc<GradBuffer<f32, B>>>],
    ) -> Result<(), B::Error> {
        let (grad_a, grad_u) = selective_scan_backward(&self.a_bar, &self.h, grad_out)?;
        let backend = B::default();
        if let Some(Some(gradient)) = input_grads.first() {
            coeus_ops::add_assign(gradient.write(), &grad_a, &backend)?;
        }
        if let Some(Some(gradient)) = input_grads.get(1) {
            coeus_ops::add_assign(gradient.write(), &grad_u, &backend)?;
        }

        Ok(())
    }
}

/// Differentiable selective scan: the Mamba/S6 linear state-space recurrence.
///
/// Computes the h-sequence of the discrete-time linear recurrence
/// `h_t = a_bar_t ⊙ h_{t-1} + u_t` (with `h_0 = u_0`, `h_{-1} = 0`) along the
/// length axis, tracking gradients for BOTH the discretized state transition
/// `a_bar` and the input contribution `u`. Because the recurrence is *linear*,
/// the backward is the exact reverse-time associative scan (no truncation).
///
/// # Convention (pinned)
/// - `a_bar` (Ā_t) and `u` (B̄_t·x_t) share one identical shape
///   `[batch, length, ...]`; the recurrence runs along axis 1 (the length/time
///   axis, matching ritk-model's `parallel_scan`). Every trailing axis is an
///   independent element-wise channel.
/// - `output`: same shape as the inputs — the h-sequence, NOT the C-projected
///   output. The Mamba output `y_t = (C_t ⊙ h_t).sum(state)` composes outside
///   this op from `mul`/`sum_axis`, and the discretization
///   `a_bar = exp(Δ·A)`, `u = (Δ·B) ⊙ x` composes from `exp`/`mul`.
/// - `a_bar_0` does not influence any output, so `∂L/∂a_bar_0 = 0` exactly.
///
/// # Precision
/// Concrete `f32`, matching the interpolation subsystem. All arithmetic and
/// accumulation run in `f32` (no widen/narrow): the op is honestly
/// single-precision rather than a generic body casting to a fixed type.
///
/// # Panics
/// If `a_bar` and `u` differ in shape, or either has rank < 2.
#[must_use]
pub fn selective_scan<B>(a_bar: &Var<f32, B>, u: &Var<f32, B>) -> Result<Var<f32, B>, B::Error>
where
    B: Backend + coeus_ops::BackendOps<f32> + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    assert_eq!(
        a_bar.tensor.shape(),
        u.tensor.shape(),
        "selective_scan: a_bar shape {:?} must equal u shape {:?}",
        a_bar.tensor.shape(),
        u.tensor.shape()
    );
    assert!(
        a_bar.tensor.ndim() >= 2,
        "selective_scan: inputs must be rank ≥ 2 ([batch, length, ...]), got {:?}",
        a_bar.tensor.shape()
    );

    let output = selective_scan_forward(&a_bar.tensor, &u.tensor)?;
    let requires_grad =
        crate::grad_mode::should_track_var(a_bar) || crate::grad_mode::should_track_var(u);
    if !requires_grad {
        return Ok(Var {
            tensor: output,
            grad: None,
            creator: None,
        });
    }

    let backend = B::default();
    let output_grad = Arc::new(GradBuffer::new(Tensor::zeros_on(output.shape(), &backend)?));
    let node = SelectiveScanNode::<B> {
        output_grad: output_grad.clone(),
        inputs: vec![a_bar.clone(), u.clone()],
        a_bar: a_bar.tensor.clone(),
        h: output.clone(),
    };
    Ok(Var {
        tensor: output,
        grad: Some(output_grad),
        creator: Some(Arc::new(node)),
    })
}
