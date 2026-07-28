use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::ops::activation::{UnaryAutogradOp, unary_op};
use crate::ops::arithmetic::{BinaryAutogradOp, binary_op};
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
            if let Some(Some(ref g)) = input_grads.get(0) {
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
        if let Some(Some(ref g)) = input_grads.get(0) {
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
///     Tensor::from_slice([1, 2], &[1.0, 2.0]).expect("construct tensor"),
///     true,
/// ).expect("construct variable");
/// let b = Var::<f32, MoiraiBackend>::new(
///     Tensor::from_slice([2, 1], &[3.0, 4.0]).expect("construct tensor"),
///     true,
/// ).expect("construct variable");
/// let c = coeus_autograd::matmul(&a, &b).expect("multiply matrices");
/// assert!((c.tensor.as_slice()[0] - 11.0).abs() < 1e-5); // 1*3 + 2*4
/// c.backward().expect("backward propagation"); // scalar output, seed = 1
/// let ga = a.grad().unwrap();
/// assert!((ga.as_slice()[0] - 3.0).abs() < 1e-5); // dA = B^T
/// assert!((ga.as_slice()[1] - 4.0).abs() < 1e-5);
/// let gb = b.grad().unwrap();
/// assert!((gb.as_slice()[0] - 1.0).abs() < 1e-5); // dB = A^T
/// assert!((gb.as_slice()[1] - 2.0).abs() < 1e-5);
/// ```
#[must_use]
#[inline]
pub fn matmul<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
    b: &Var<T, B>,
) -> Result<Var<T, B>, B::Error> {
    binary_op::<T, B, MatmulOp>(a, b)
}

/// Tracked 2-D Transpose.
#[must_use]
#[inline]
pub fn transpose_2d<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
) -> Result<Var<T, B>, B::Error> {
    unary_op::<T, B, Transpose2dOp>(a)
}

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
        if let Some(Some(ref g)) = input_grads.get(0) {
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

fn coo_to_csr_parts_with_permutation<T: Scalar, B: coeus_ops::BackendOps<T> + coeus_core::Backend>(
    a_values: &Tensor<T, B>,
    a_indices: &Tensor<i64, B>,
    a_shape: &Shape,
    backend: &B,
    ) -> Result<(Tensor<T, B>, Tensor<i64, B>, Tensor<i64, B>, Tensor<i64, B>), B::Error>
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
    let grad = if requires_grad {
        Some(Arc::new(GradBuffer::new(Tensor::zeros_on(
            out_tensor.shape_cloned(),
            &backend,
        )?)))
    } else {
        None
    };

    let creator = if let Some(ref output_grad) = grad {
        let inputs = vec![a_values.clone(), b.clone()];

        let node = SparseMatMulNode {
            output_grad: output_grad.clone(),
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

    Ok(Var {
        tensor: out_tensor,
        grad,
        creator,
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
    let grad = if requires_grad {
        Some(Arc::new(GradBuffer::new(Tensor::zeros_on(
            out_tensor.shape_cloned(),
            &backend,
        )?)))
    } else {
        None
    };

    let creator = if let Some(ref output_grad) = grad {
        let inputs = vec![a_values.clone(), b.clone()];

        let node = SparseCooMatMulNode {
            output_grad: output_grad.clone(),
            inputs,
            csr_values_tensor: csr_values,
            csr_col_indices,
            csr_row_offsets,
            sorted_to_orig,
            a_shape,
            b_tensor: b.tensor.clone(),
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
