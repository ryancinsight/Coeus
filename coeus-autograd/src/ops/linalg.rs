use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::ops::activation::{unary_op, UnaryAutogradOp};
use crate::ops::arithmetic::{binary_op, BinaryAutogradOp};
use crate::var::Var;
use coeus_core::{CpuAddressableStorage, CpuAddressableStorageMut, Scalar, Shape};
use coeus_sparse::CsrTensor;
use coeus_tensor::Tensor;
use std::sync::Arc;

/// ZST tag for Matrix Multiplication autograd.
pub struct MatmulOp;

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> BinaryAutogradOp<T, B> for MatmulOp {
    const OP_NAME: &'static str = "matmul";

    #[inline(always)]
    fn forward(a: &Tensor<T, B>, b: &Tensor<T, B>, backend: &B) -> Tensor<T, B> {
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
    ) {
        // ∂/∂A: grad_C @ B^T — grad_C may be batched ([batch,m,n] × [n,k] → [batch,m,k])
        if let Some(Some(ref g)) = input_grads.get(0) {
            let b_t = b.t(); // B is always 2D (weight matrix); b.t() ✓
            let gl = g.write();
            coeus_ops::matmul_accumulate(grad_out, &b_t, gl, backend);
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
            coeus_ops::matmul_accumulate(&a_flat_t, &go_flat, gl, backend);
        }
    }
}

/// ZST tag for 2-D Transpose autograd.
pub struct Transpose2dOp;

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> UnaryAutogradOp<T, B> for Transpose2dOp {
    const OP_NAME: &'static str = "transpose_2d";

    #[inline(always)]
    fn forward(x: &Tensor<T, B>, _backend: &B) -> Tensor<T, B> {
        x.t()
    }

    #[inline(always)]
    fn backward(
        grad_out: &Tensor<T, B>,
        _x: &Tensor<T, B>,
        _y: &Tensor<T, B>,
        _backend: &B,
    ) -> Tensor<T, B> {
        grad_out.t()
    }
}

/// Tracked matrix multiplication.
#[must_use]
#[inline]
pub fn matmul<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
    b: &Var<T, B>,
) -> Var<T, B> {
    binary_op::<T, B, MatmulOp>(a, b)
}

/// Tracked 2-D Transpose.
#[must_use]
#[inline]
pub fn transpose_2d<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(a: &Var<T, B>) -> Var<T, B> {
    unary_op::<T, B, Transpose2dOp>(a)
}

pub struct SparseMatMulNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    pub output_grad: Arc<GradBuffer<T, B>>,
    pub inputs: Vec<Var<T, B>>,
    pub a_values_tensor: Tensor<T, B>,
    pub a_col_indices: Tensor<i64, B>,
    pub a_row_offsets: Tensor<i64, B>,
    pub a_shape: Shape,
    pub b_tensor: Tensor<T, B>,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + coeus_core::Backend + Default> BackwardNode<T, B>
    for SparseMatMulNode<T, B>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
    B::DeviceBuffer<i64>: CpuAddressableStorage<i64>,
{
    #[inline]
    fn op_name(&self) -> &'static str {
        "sparse_matmul"
    }

    #[inline]
    fn output_grad(&self) -> &Arc<GradBuffer<T, B>> {
        &self.output_grad
    }

    #[inline]
    fn inputs(&self) -> &[Var<T, B>] {
        &self.inputs
    }

    #[inline]
    fn backward(&self, grad_out: &Tensor<T, B>, input_grads: &[Option<Arc<GradBuffer<T, B>>>]) {
        let backend = B::default();
        // ∂/∂A_values
        if let Some(Some(ref g)) = input_grads.get(0) {
            let grad_a_vals = coeus_ops::spmm_backward_values(
                &self.a_col_indices,
                &self.a_row_offsets,
                &self.a_shape,
                &self.b_tensor,
                grad_out,
                &backend,
            );
            let gl = g.write();
            coeus_ops::add_assign(gl, &grad_a_vals, &backend);
        }
        // ∂/∂B
        if let Some(Some(ref g)) = input_grads.get(1) {
            let grad_b = coeus_ops::spmm_backward_dense(
                &self.a_values_tensor,
                &self.a_col_indices,
                &self.a_row_offsets,
                &self.a_shape,
                grad_out,
                &backend,
            );
            let gl = g.write();
            coeus_ops::add_assign(gl, &grad_b, &backend);
        }
    }
}

pub fn sparse_matmul<T: Scalar, B: coeus_ops::BackendOps<T> + coeus_core::Backend + Default>(
    a_values: &Var<T, B>,
    a_col_indices: &Tensor<i64, B>,
    a_row_offsets: &Tensor<i64, B>,
    a_shape: Shape,
    b: &Var<T, B>,
) -> Var<T, B>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
    B::DeviceBuffer<i64>: CpuAddressableStorage<i64>,
{
    let backend = B::default();
    let csr = CsrTensor::new(
        a_shape.clone(),
        a_values.tensor.clone(),
        a_col_indices.clone(),
        a_row_offsets.clone(),
    );
    let out_tensor = coeus_ops::spmm(&csr, &b.tensor, &backend);

    let requires_grad = a_values.grad.is_some() || b.grad.is_some();
    let grad = if requires_grad {
        Some(Arc::new(GradBuffer::new(Tensor::zeros_on(
            out_tensor.shape_cloned(),
            &backend,
        ))))
    } else {
        None
    };

    let creator = if requires_grad {
        let output_grad = grad.as_ref().unwrap().clone();
        let inputs = vec![a_values.clone(), b.clone()];

        let node = SparseMatMulNode {
            output_grad,
            inputs,
            a_values_tensor: a_values.tensor.clone(),
            a_col_indices: a_col_indices.clone(),
            a_row_offsets: a_row_offsets.clone(),
            a_shape,
            b_tensor: b.tensor.clone(),
        };
        Some(Arc::new(node) as Arc<dyn BackwardNode<T, B>>)
    } else {
        None
    };

    Var {
        tensor: out_tensor,
        grad,
        creator,
    }
}
