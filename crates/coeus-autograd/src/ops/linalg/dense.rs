//! Dense linear-algebra autograd: batched matmul and 2-D transpose.

use crate::grad_buffer::GradBuffer;
use crate::ops::activation::{unary_op, UnaryAutogradOp};
use crate::ops::arithmetic::{binary_op, BinaryAutogradOp};
use crate::var::Var;
use coeus_core::{Scalar, Shape};
use coeus_tensor::Tensor;
use std::sync::Arc;

/// ZST tag for Matrix Multiplication autograd.
pub struct MatmulOp;

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> BinaryAutogradOp<T, B> for MatmulOp {
    const OP_NAME: &'static str = "matmul";

    #[inline(always)]
    fn forward(a: &Tensor<T, B>, b: &Tensor<T, B>, backend: &B) -> Result<Tensor<T, B>, B::Error> {
        // The batched matmul kernels derive their layouts from shape alone and
        // do not honor view strides (see `swap_last_two`), so a non-contiguous
        // input (e.g. a `transpose`/`permute` view fed straight into `matmul`,
        // as attention's `Q Kᵀ` does) would be read with wrong strides and
        // produce a wrong product. Materialize such inputs contiguous first.
        let a_owned;
        let a = if a.is_contiguous() {
            a
        } else {
            a_owned = a.to_contiguous_on(backend)?;
            &a_owned
        };
        let b_owned;
        let b = if b.is_contiguous() {
            b
        } else {
            b_owned = b.to_contiguous_on(backend)?;
            &b_owned
        };
        coeus_ops::matmul(a, b, backend)
    }

    #[inline(always)]
    fn backward(
        grad_out: &Tensor<T, B>,
        a: &Tensor<T, B>,
        b: &Tensor<T, B>,
        _a_shape: &Shape,
        _b_shape: &Shape,
        input_grads: &[Option<Arc<GradBuffer<T, B>>>],
        backend: &B,
    ) -> Result<(), B::Error> {
        // Batched B ([…,k,n], bmm): per-batch transposes of the last two axes.
        // The permuted view is materialized contiguous because the batched
        // matmul kernels derive strides from shape (they do not honor view
        // strides, unlike the 2-D fast path).
        if b.ndim() > 2 {
            assert_eq!(
                a.ndim(),
                b.ndim(),
                "matmul backward: batched B requires equally-batched A \
                 (broadcast batching is not differentiable yet)"
            );
            // ∂/∂A: grad_C @ B^T — [batch,m,n] × [batch,n,k] → [batch,m,k]
            if let Some(Some(ref g)) = input_grads.first() {
                let b_t = swap_last_two(b, backend)?;
                let gl = g.write();
                coeus_ops::matmul_accumulate(grad_out, &b_t, gl, backend)?;
            }
            // ∂/∂B: A^T @ grad_C — [batch,k,m] × [batch,m,n] → [batch,k,n]
            if let Some(Some(ref g)) = input_grads.get(1) {
                let a_t = swap_last_two(a, backend)?;
                let gl = g.write();
                coeus_ops::matmul_accumulate(&a_t, grad_out, gl, backend)?;
            }
            return Ok(());
        }

        // ∂/∂A: grad_C @ B^T — grad_C may be batched ([batch,m,n] × [n,k] → [batch,m,k])
        if let Some(Some(ref g)) = input_grads.first() {
            let b_t = b.t(); // B is 2-D on this path; b.t() is a stride view.
            let gl = g.write();
            coeus_ops::matmul_accumulate(grad_out, &b_t, gl, backend)?;
        }
        // ∂/∂B: A^T @ grad_C
        // When A is batched ([…,m,k]), flatten to [batch*m, k] to perform 2D matmul.
        if let Some(Some(ref g)) = input_grads.get(1) {
            let a_ndim = a.ndim();
            let (a_flat, go_flat) = if a_ndim > 2 {
                let a_shape = a.shape();
                let m = a_shape[a_ndim - 2];
                let k = a_shape[a_ndim - 1];
                let batch: usize = a_shape[..a_ndim - 2].iter().product();
                let go_shape = grad_out.shape();
                let n = go_shape[grad_out.ndim() - 1];
                (
                    a.reshape([batch * m, k].as_slice()),
                    grad_out.reshape([batch * m, n].as_slice()),
                )
            } else {
                (a.clone(), grad_out.clone())
            };
            let a_flat_t = a_flat.t();
            let gl = g.write();
            coeus_ops::matmul_accumulate(&a_flat_t, &go_flat, gl, backend)?;
        }
        Ok(())
    }
}

/// Materialized transpose of the last two axes of a (batched) tensor.
///
/// Contiguous copy rather than a stride view: the batched matmul kernels
/// derive their layouts from shape alone, so a strided view would be read
/// with wrong strides.
fn swap_last_two<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    t: &Tensor<T, B>,
    backend: &B,
) -> Result<Tensor<T, B>, B::Error> {
    let nd = t.ndim();
    let mut dims: Vec<usize> = (0..nd).collect();
    dims.swap(nd - 2, nd - 1);
    t.permute(&dims).to_contiguous_on(backend)
}

/// ZST tag for 2-D Transpose autograd.
pub struct Transpose2dOp;

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> UnaryAutogradOp<T, B> for Transpose2dOp {
    const OP_NAME: &'static str = "transpose_2d";

    #[inline(always)]
    fn forward(x: &Tensor<T, B>, _backend: &B) -> Result<Tensor<T, B>, B::Error> {
        Ok(x.t())
    }

    #[inline(always)]
    fn backward(
        grad_out: &Tensor<T, B>,
        _x: &Tensor<T, B>,
        _y: &Tensor<T, B>,
        _backend: &B,
    ) -> Result<Tensor<T, B>, B::Error> {
        Ok(grad_out.t())
    }
}

/// Tracked matrix multiplication.
///
/// # Examples
///
/// For `C = A @ B` with `A = [[1, 2]]`, `B = [[3], [4]]`, the scalar output
/// `C[0,0] = 11` seeds with 1, giving `dA = B^T = [[3, 4]]` and
/// `dB = A^T = [[1], [2]]`.
///
/// ```
/// use coeus_autograd::Var;
/// use coeus_core::MoiraiBackend;
/// use coeus_tensor::Tensor;
///
/// let a = Var::<f32, MoiraiBackend>::new(
///     Tensor::from_slice([1, 2], &[1.0, 2.0]).expect("invariant: example shape matches data"),
///     true,
/// ).expect("invariant: example gradient buffer allocation succeeds");
/// let b = Var::<f32, MoiraiBackend>::new(
///     Tensor::from_slice([2, 1], &[3.0, 4.0]).expect("invariant: example shape matches data"),
///     true,
/// ).expect("invariant: example gradient buffer allocation succeeds");
/// let c = coeus_autograd::matmul(&a, &b).expect("invariant: example matrix shapes are compatible");
/// assert!((c.tensor.as_slice()[0] - 11.0).abs() < 1e-5); // 1*3 + 2*4
/// c.backward().expect("invariant: valid autograd fixture completes backward"); // scalar output, seed = 1
/// let ga = a.grad().expect("invariant: backward populates the tracked leaf gradient");
/// assert!((ga.as_slice()[0] - 3.0).abs() < 1e-5); // dA = B^T
/// assert!((ga.as_slice()[1] - 4.0).abs() < 1e-5);
/// let gb = b.grad().expect("invariant: backward populates the tracked leaf gradient");
/// assert!((gb.as_slice()[0] - 1.0).abs() < 1e-5); // dB = A^T
/// assert!((gb.as_slice()[1] - 2.0).abs() < 1e-5);
/// ```
#[inline]
pub fn matmul<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
    b: &Var<T, B>,
) -> Result<Var<T, B>, B::Error> {
    binary_op::<T, B, MatmulOp>(a, b)
}

/// Tracked 2-D Transpose.
#[inline]
pub fn transpose_2d<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
) -> Result<Var<T, B>, B::Error> {
    unary_op::<T, B, Transpose2dOp>(a)
}
