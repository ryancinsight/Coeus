// ── Tracked diag / diagonal ──
//
// diag: backward of diag(v) is diagonal(grad_out).
// diagonal: backward of diagonal(M) is diag(grad_out).

use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::Scalar;
use coeus_tensor::Tensor;
use std::sync::Arc;

// ── DiagNode ─────────────────────────────────────────────────────────────────

pub struct DiagNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    pub output_grad: Arc<GradBuffer<T, B>>,
    pub inputs: Vec<Var<T, B>>,
    pub k: isize,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for DiagNode<T, B>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    fn op_name(&self) -> &'static str {
        "diag"
    }
    fn output_grad(&self) -> &Arc<GradBuffer<T, B>> {
        &self.output_grad
    }
    fn inputs(&self) -> &[Var<T, B>] {
        &self.inputs
    }
    fn backward(
        &self,
        grad_out: &Tensor<T, B>,
        input_grads: &[Option<Arc<GradBuffer<T, B>>>],
    ) -> Result<(), B::Error> {
        let backend = B::default();
        if let Some(Some(ref g)) = input_grads.first() {
            // backward of diag(v, k) is diagonal(grad_out, k).
            let gi = coeus_ops::diagonal(grad_out, self.k, &backend)?;
            let gl = g.write();
            coeus_ops::add_assign(gl, &gi, &backend)?;
        }

        Ok(())
    }
}

/// Tracked `diag(v, k)` — create diagonal matrix from 1-D vector.
#[must_use]
#[inline]
pub fn diag<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    v: &Var<T, B>,
    k: isize,
) -> Result<Var<T, B>, B::Error>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    let backend = B::default();
    let out_tensor = coeus_ops::diag(&v.tensor, k, &backend)?;

    let requires_grad = crate::grad_mode::should_track_var(v);
    let grad = if requires_grad {
        Some(Arc::new(GradBuffer::new(Tensor::zeros_on(
            out_tensor.shape_cloned(),
            &backend,
        )?)))
    } else {
        None
    };
    let creator = if let Some(ref output_grad) = grad {
        let node = DiagNode {
            output_grad: output_grad.clone(),
            inputs: vec![v.clone()],
            k,
        };
        Some(Arc::new(node) as Arc<dyn BackwardNode<T, B>>)
    } else {
        None
    };
    Ok(Var {
        tensor: out_tensor,
        grad,
        creator,
    })
}

// ── DiagonalNode ─────────────────────────────────────────────────────────────

pub struct DiagonalNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    pub output_grad: Arc<GradBuffer<T, B>>,
    pub inputs: Vec<Var<T, B>>,
    pub k: isize,
    /// Shape of the input matrix (for zero-extended diag backward).
    pub input_shape: Vec<usize>,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for DiagonalNode<T, B>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    fn op_name(&self) -> &'static str {
        "diagonal"
    }
    fn output_grad(&self) -> &Arc<GradBuffer<T, B>> {
        &self.output_grad
    }
    fn inputs(&self) -> &[Var<T, B>] {
        &self.inputs
    }
    fn backward(
        &self,
        grad_out: &Tensor<T, B>,
        input_grads: &[Option<Arc<GradBuffer<T, B>>>],
    ) -> Result<(), B::Error> {
        let backend = B::default();
        if let Some(Some(ref g)) = input_grads.first() {
            // backward of diagonal(M, k) is a matrix with grad_out on diagonal k.
            // We build this as diag(grad_out, k) then zero-pad to match input_shape.
            let gi_diag = coeus_ops::diag(grad_out, self.k, &backend)?;
            // gi_diag may be smaller than input_shape; zero-pad if necessary.
            let [rows, cols] = [self.input_shape[0], self.input_shape[1]];
            let (gi_rows, gi_cols) = (gi_diag.shape()[0], gi_diag.shape()[1]);
            let gi: Tensor<T, B> = if gi_rows == rows && gi_cols == cols {
                gi_diag
            } else {
                // Embed gi_diag into zeros of input_shape.
                let gi_cont = gi_diag.to_contiguous()?;
                let gi_s = gi_cont.as_slice();
                let mut data = vec![T::zero(); rows * cols];
                for r in 0..gi_rows.min(rows) {
                    for c in 0..gi_cols.min(cols) {
                        data[r * cols + c] = gi_s[r * gi_cols + c];
                    }
                }
                Tensor::from_slice_on(vec![rows, cols], &data, &backend)?
            };
            let gl = g.write();
            coeus_ops::add_assign(gl, &gi, &backend)?;
        }

        Ok(())
    }
}

/// Tracked `diagonal(M, k)` — extract diagonal from 2-D matrix.
#[must_use]
#[inline]
pub fn diagonal<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    m: &Var<T, B>,
    k: isize,
) -> Result<Var<T, B>, B::Error>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    let backend = B::default();
    let out_tensor = coeus_ops::diagonal(&m.tensor, k, &backend)?;

    let requires_grad = crate::grad_mode::should_track_var(m);
    let grad = if requires_grad {
        Some(Arc::new(GradBuffer::new(Tensor::zeros_on(
            out_tensor.shape_cloned(),
            &backend,
        )?)))
    } else {
        None
    };
    let creator = if let Some(ref output_grad) = grad {
        let node = DiagonalNode {
            output_grad: output_grad.clone(),
            inputs: vec![m.clone()],
            k,
            input_shape: m.tensor.shape().to_vec(),
        };
        Some(Arc::new(node) as Arc<dyn BackwardNode<T, B>>)
    } else {
        None
    };
    Ok(Var {
        tensor: out_tensor,
        grad,
        creator,
    })
}
