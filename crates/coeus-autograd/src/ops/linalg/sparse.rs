//! Sparse linear-algebra autograd: CSR and COO matrix-multiply against a dense operand.

use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::{CpuAddressableStorage, CpuAddressableStorageMut, Scalar, Shape};
use coeus_sparse::CsrTensor;
use coeus_tensor::Tensor;
use std::sync::Arc;

/// Autograd node for sparse CSR matrix-multiply (A_sparse × B_dense).
pub struct SparseMatMulNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    /// Accumulated gradient buffer for the output of this node.
    pub output_grad: Arc<GradBuffer<T, B>>,
    /// Input variables tracked for backward propagation.
    pub inputs: Vec<Var<T, B>>,
    /// Non-zero values of the sparse CSR matrix A.
    pub a_values_tensor: Tensor<T, B>,
    /// Column indices of the sparse CSR matrix A.
    pub a_col_indices: Tensor<i64, B>,
    /// Row offsets of the sparse CSR matrix A.
    pub a_row_offsets: Tensor<i64, B>,
    /// Dense shape of the sparse matrix A.
    pub a_shape: Shape,
    /// Saved dense matrix B for backward computation.
    pub b_tensor: Tensor<T, B>,
}

/// Autograd node for sparse COO matrix-multiply (A_sparse_coo × B_dense).
pub struct SparseCooMatMulNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    /// Accumulated gradient buffer for the output of this node.
    pub output_grad: Arc<GradBuffer<T, B>>,
    /// Input variables tracked for backward propagation.
    pub inputs: Vec<Var<T, B>>,
    /// Non-zero values of the COO matrix converted to CSR format.
    pub csr_values_tensor: Tensor<T, B>,
    /// Column indices of the CSR representation of the COO matrix.
    pub csr_col_indices: Tensor<i64, B>,
    /// Row offsets of the CSR representation of the COO matrix.
    pub csr_row_offsets: Tensor<i64, B>,
    /// Permutation mapping sorted CSR order back to original COO order.
    pub sorted_to_orig: Tensor<i64, B>,
    /// Dense shape of the sparse matrix A.
    pub a_shape: Shape,
    /// Saved dense matrix B for backward computation.
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
    fn backward(
        &self,
        grad_out: &Tensor<T, B>,
        input_grads: &[Option<Arc<GradBuffer<T, B>>>],
    ) -> Result<(), B::Error> {
        let backend = B::default();
        // ∂/∂A_values
        if let Some(Some(ref g)) = input_grads.first() {
            let grad_a_vals = coeus_ops::spmm_backward_values(
                &self.a_col_indices,
                &self.a_row_offsets,
                &self.a_shape,
                &self.b_tensor,
                grad_out,
                &backend,
            )?;
            let gl = g.write();
            coeus_ops::add_assign(gl, &grad_a_vals, &backend)?;
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
            )?;
            let gl = g.write();
            coeus_ops::add_assign(gl, &grad_b, &backend)?;
        }
        Ok(())
    }
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + coeus_core::Backend + Default> BackwardNode<T, B>
    for SparseCooMatMulNode<T, B>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
    B::DeviceBuffer<i64>: CpuAddressableStorage<i64> + CpuAddressableStorageMut<i64>,
{
    #[inline]
    fn op_name(&self) -> &'static str {
        "sparse_matmul_coo"
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
    fn backward(
        &self,
        grad_out: &Tensor<T, B>,
        input_grads: &[Option<Arc<GradBuffer<T, B>>>],
    ) -> Result<(), B::Error> {
        let backend = B::default();
        // ∂/∂A_values (COO values order)
        if let Some(Some(ref g)) = input_grads.first() {
            let grad_sorted = coeus_ops::spmm_backward_values(
                &self.csr_col_indices,
                &self.csr_row_offsets,
                &self.a_shape,
                &self.b_tensor,
                grad_out,
                &backend,
            )?;

            let nnz = self.sorted_to_orig.numel();
            let sorted_to_orig = self.sorted_to_orig.as_slice();
            let grad_sorted_slice = grad_sorted.as_slice();
            let mut grad_coo = Tensor::<T, B>::zeros_on([nnz], &backend)?;
            let grad_coo_slice = grad_coo.as_mut_slice()?;
            for i in 0..nnz {
                let orig = sorted_to_orig[i] as usize;
                grad_coo_slice[orig] += grad_sorted_slice[i];
            }
            let gl = g.write();
            coeus_ops::add_assign(gl, &grad_coo, &backend)?;
        }
        // ∂/∂B
        if let Some(Some(ref g)) = input_grads.get(1) {
            let grad_b = coeus_ops::spmm_backward_dense(
                &self.csr_values_tensor,
                &self.csr_col_indices,
                &self.csr_row_offsets,
                &self.a_shape,
                grad_out,
                &backend,
            )?;
            let gl = g.write();
            coeus_ops::add_assign(gl, &grad_b, &backend)?;
        }
        Ok(())
    }
}

/// CSR values, column indices and row offsets, plus the permutation carrying
/// each CSR slot back to the COO entry it came from -- the backward pass needs
/// that last one to scatter gradients to the original positions.
type CsrPartsWithPermutation<T, B> = (Tensor<T, B>, Tensor<i64, B>, Tensor<i64, B>, Tensor<i64, B>);

fn coo_to_csr_parts_with_permutation<T: Scalar, B: coeus_ops::BackendOps<T> + coeus_core::Backend>(
    a_values: &Tensor<T, B>,
    a_indices: &Tensor<i64, B>,
    a_shape: &Shape,
    backend: &B,
) -> Result<CsrPartsWithPermutation<T, B>, B::Error>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
    B::DeviceBuffer<i64>: CpuAddressableStorage<i64> + CpuAddressableStorageMut<i64>,
{
    assert_eq!(
        a_shape.len(),
        2,
        "COO sparse_matmul requires 2D sparse shape"
    );
    assert_eq!(a_indices.ndim(), 2, "COO indices must be rank-2 [2, nnz]");
    assert_eq!(
        a_indices.shape()[0],
        2,
        "COO indices first dimension must be 2 (row,col)"
    );
    let nnz = a_values.numel();
    assert_eq!(
        a_indices.shape()[1],
        nnz,
        "COO indices nnz must match values length"
    );

    let idx_slice = a_indices.as_slice();
    let val_slice = a_values.as_slice();
    let rows = a_shape[0];
    let cols = a_shape[1];

    let mut triples = Vec::with_capacity(nnz);
    for i in 0..nnz {
        let row = idx_slice[i];
        let col = idx_slice[nnz + i];
        assert!(
            row >= 0 && (row as usize) < rows,
            "COO row index out of bounds: row={row}, rows={rows}"
        );
        assert!(
            col >= 0 && (col as usize) < cols,
            "COO column index out of bounds: col={col}, cols={cols}"
        );
        let r = row as usize;
        let c = col as usize;
        triples.push((r, c, i, val_slice[i]));
    }
    triples.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

    let mut csr_values = Tensor::<T, B>::zeros_on([nnz], backend)?;
    let mut csr_col_indices = Tensor::<i64, B>::zeros_on([nnz], backend)?;
    let mut csr_row_offsets = Tensor::<i64, B>::zeros_on([rows + 1], backend)?;
    let mut sorted_to_orig = Tensor::<i64, B>::zeros_on([nnz], backend)?;

    let val_mut = csr_values.as_mut_slice()?;
    let col_mut = csr_col_indices.as_mut_slice()?;
    let row_mut = csr_row_offsets.as_mut_slice()?;
    let map_mut = sorted_to_orig.as_mut_slice()?;

    let mut current_row = 0usize;
    row_mut[0] = 0;
    for (i, &(r, c, orig, v)) in triples.iter().enumerate() {
        val_mut[i] = v;
        col_mut[i] = c as i64;
        map_mut[i] = orig as i64;
        while current_row < r {
            current_row += 1;
            row_mut[current_row] = i as i64;
        }
    }
    while current_row < rows {
        current_row += 1;
        row_mut[current_row] = nnz as i64;
    }

    Ok((csr_values, csr_col_indices, csr_row_offsets, sorted_to_orig))
}

/// Multiplies a CSR sparse matrix by a dense tracked matrix.
///
/// The sparse matrix is represented by tracked nonzero values plus CSR column
/// indices, row offsets, and shape. Backward propagation accumulates gradients
/// for the sparse values and dense right-hand matrix.
pub fn sparse_matmul<T: Scalar, B: coeus_ops::BackendOps<T> + coeus_core::Backend + Default>(
    a_values: &Var<T, B>,
    a_col_indices: &Tensor<i64, B>,
    a_row_offsets: &Tensor<i64, B>,
    a_shape: Shape,
    b: &Var<T, B>,
) -> Result<Var<T, B>, B::Error>
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
    let out_tensor = coeus_ops::spmm(&csr, &b.tensor, &backend)?;

    let requires_grad =
        crate::grad_mode::should_track_var(a_values) || crate::grad_mode::should_track_var(b);
    Var::from_tracked_op(out_tensor, requires_grad, &backend, |output_grad| {
        SparseMatMulNode {
            output_grad,
            inputs: vec![a_values.clone(), b.clone()],
            a_values_tensor: a_values.tensor.clone(),
            a_col_indices: a_col_indices.clone(),
            a_row_offsets: a_row_offsets.clone(),
            a_shape,
            b_tensor: b.tensor.clone(),
        }
    })
}

/// Multiplies a COO sparse matrix by a dense tracked matrix.
///
/// The COO coordinates are converted to CSR once for the forward pass while a
/// permutation map preserves gradients for the original COO value ordering.
pub fn sparse_matmul_coo<T: Scalar, B: coeus_ops::BackendOps<T> + coeus_core::Backend + Default>(
    a_values: &Var<T, B>,
    a_indices: &Tensor<i64, B>,
    a_shape: Shape,
    b: &Var<T, B>,
) -> Result<Var<T, B>, B::Error>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
    B::DeviceBuffer<i64>: CpuAddressableStorage<i64> + CpuAddressableStorageMut<i64>,
{
    let backend = B::default();
    let (csr_values, csr_col_indices, csr_row_offsets, sorted_to_orig) =
        coo_to_csr_parts_with_permutation(&a_values.tensor, a_indices, &a_shape, &backend)?;
    let csr = CsrTensor::new(
        a_shape.clone(),
        csr_values.clone(),
        csr_col_indices.clone(),
        csr_row_offsets.clone(),
    );
    let out_tensor = coeus_ops::spmm(&csr, &b.tensor, &backend)?;

    let requires_grad =
        crate::grad_mode::should_track_var(a_values) || crate::grad_mode::should_track_var(b);
    Var::from_tracked_op(out_tensor, requires_grad, &backend, |output_grad| {
        SparseCooMatMulNode {
            output_grad,
            inputs: vec![a_values.clone(), b.clone()],
            csr_values_tensor: csr_values,
            csr_col_indices,
            csr_row_offsets,
            sorted_to_orig,
            a_shape,
            b_tensor: b.tensor.clone(),
        }
    })
}
