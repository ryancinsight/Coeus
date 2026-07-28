//! Differential parity for sparse matrix operations.
//!
//! Functions exercised:
//!   `spmv`                - sparse CSR x dense vector -> dense vector
//!   `spmm`                - sparse CSR x dense matrix -> dense matrix
//!   `spmm_backward_dense` - gradient w.r.t. dense matrix B in CSR @ B
//!   `spmm_backward_values` - gradient w.r.t. sparse values in CSR @ B
//!
//! Sparse matrix used throughout:
//!
//!   A = [[2, 0, 1],      CSR: values     = [2, 1, 3, 1, 4]
//!        [0, 3, 0],           col_indices = [0, 2, 1, 0, 2]
//!        [1, 0, 4]]           row_offsets = [0, 2, 3, 5]
//!
//! All reference values are integer-valued (exact in f64) so assertions use
//! `assert_eq!` without an epsilon band.  SequentialBackend and MoiraiBackend
//! receive identical inputs and must return identical outputs.

use coeus_core::{
    CpuAddressableStorage, CpuAddressableStorageMut, MoiraiBackend, SequentialBackend,
};
use coeus_sparse::CsrTensor;
use coeus_tensor::Tensor;

fn make_csr<B>(backend: &B) -> CsrTensor<f64, B>
where
    B: coeus_core::Backend + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorageMut<f64>,
    B::DeviceBuffer<i64>: CpuAddressableStorage<i64> + CpuAddressableStorageMut<i64>,
{
    // A = [[2,0,1],[0,3,0],[1,0,4]], shape [3,3], nnz=5
    let values = Tensor::<f64, B>::from_slice_on(vec![5], &[2.0, 1.0, 3.0, 1.0, 4.0], backend).expect("construct tensor");
    let col_indices = Tensor::<i64, B>::from_slice_on(vec![5], &[0i64, 2, 1, 0, 2], backend).expect("construct tensor");
    let row_offsets = Tensor::<i64, B>::from_slice_on(vec![4], &[0i64, 2, 3, 5], backend).expect("construct tensor");
    CsrTensor::new(vec![3, 3].into(), values, col_indices, row_offsets)
}

fn t<B>(shape: &[usize], vals: &[f64], backend: &B) -> Tensor<f64, B>
where
    B: coeus_core::ComputeBackend,
    B::DeviceBuffer<f64>: CpuAddressableStorageMut<f64>,
{
    Tensor::from_slice_on(shape.to_vec(), vals, backend).expect("construct tensor")
}

// SPMV

fn check_spmv<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + coeus_core::Backend + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
    B::DeviceBuffer<i64>: CpuAddressableStorage<i64> + CpuAddressableStorageMut<i64>,
{
    // A x x where x = [1, 2, 3]:
    // y[0] = 2*1 + 1*3 = 5
    // y[1] = 3*2 = 6
    // y[2] = 1*1 + 4*3 = 13
    let a = make_csr(backend);
    let x = t(&[3], &[1.0, 2.0, 3.0], backend);
    let y = coeus_ops::spmv(&a, &x, backend).expect("run operation");
    assert_eq!(y.shape(), &[3], "spmv shape");
    assert_eq!(y.as_slice(), &[5.0_f64, 6.0, 13.0], "spmv");

    // Identity sparse matrix x x = x (nnz = 3, diagonal only).
    // I = [[1,0,0],[0,1,0],[0,0,1]]
    let i_vals = Tensor::<f64, B>::from_slice_on(vec![3], &[1.0, 1.0, 1.0], backend).expect("construct tensor");
    let i_cols = Tensor::<i64, B>::from_slice_on(vec![3], &[0i64, 1, 2], backend).expect("construct tensor");
    let i_rows = Tensor::<i64, B>::from_slice_on(vec![4], &[0i64, 1, 2, 3], backend).expect("construct tensor");
    let eye = CsrTensor::new(vec![3, 3].into(), i_vals, i_cols, i_rows);
    let yi = coeus_ops::spmv(&eye, &x, backend).expect("run operation");
    assert_eq!(yi.as_slice(), x.as_slice(), "spmv identity");
}

// SPMM

fn check_spmm<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + coeus_core::Backend + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
    B::DeviceBuffer<i64>: CpuAddressableStorage<i64> + CpuAddressableStorageMut<i64>,
{
    // A x B where B = [[1,0],[0,1],[1,0]] shape [3,2]:
    // C[0,0]=2*1+1*1=3  C[0,1]=2*0+1*0=0
    // C[1,0]=3*0=0       C[1,1]=3*1=3
    // C[2,0]=1*1+4*1=5   C[2,1]=1*0+4*0=0
    // -> C = [[3,0],[0,3],[5,0]]
    let a = make_csr(backend);
    let b = t(&[3, 2], &[1.0, 0.0, 0.0, 1.0, 1.0, 0.0], backend);
    let c = coeus_ops::spmm(&a, &b, backend).expect("run operation");
    assert_eq!(c.shape(), &[3, 2], "spmm shape");
    assert_eq!(c.as_slice(), &[3.0_f64, 0.0, 0.0, 3.0, 5.0, 0.0], "spmm");
}

// SPMM_BACKWARD_VALUES

fn check_spmm_backward_values<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + coeus_core::Backend + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
    B::DeviceBuffer<i64>: CpuAddressableStorage<i64> + CpuAddressableStorageMut<i64>,
{
    // grad_values[i] = sum_j grad_out[row(i), j] * B[col(i), j]
    // A has 5 non-zeros; B shape [3,2]; grad_out shape [3,2]
    // B = [[1,0],[0,1],[1,0]], grad_out = [[1,0],[0,1],[0,0]]
    // nz0 (r=0,c=0): sum_j=1*1+0*0=1
    // nz1 (r=0,c=2): sum_j=1*1+0*0=1
    // nz2 (r=1,c=1): sum_j=0*0+1*1=1
    // nz3 (r=2,c=0): sum_j=0*1+0*0=0
    // nz4 (r=2,c=2): sum_j=0*1+0*0=0
    let col_indices = Tensor::<i64, B>::from_slice_on(vec![5], &[0i64, 2, 1, 0, 2], backend).expect("construct tensor");
    let row_offsets = Tensor::<i64, B>::from_slice_on(vec![4], &[0i64, 2, 3, 5], backend).expect("construct tensor");
    let b = t(&[3, 2], &[1.0, 0.0, 0.0, 1.0, 1.0, 0.0], backend);
    let grad_out = t(&[3, 2], &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0], backend);
    let gv = coeus_ops::spmm_backward_values(
        &col_indices,
        &row_offsets,
        &[3, 3],
        &b,
        &grad_out,
        backend,
    ).expect("run operation");
    assert_eq!(gv.shape(), &[5], "spmm_backward_values shape");
    assert_eq!(
        gv.as_slice(),
        &[1.0_f64, 1.0, 1.0, 0.0, 0.0],
        "spmm_backward_values"
    );
}

// SPMM_BACKWARD_DENSE

fn check_spmm_backward_dense<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + coeus_core::Backend + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
    B::DeviceBuffer<i64>: CpuAddressableStorage<i64> + CpuAddressableStorageMut<i64>,
{
    // grad_B[k,j] = sum_r A[r,k] * grad_out[r,j]
    // With grad_out = [[1,0],[0,1],[0,0]]:
    // grad_B[0,0]=2*1+1*0=2  grad_B[0,1]=2*0+1*0=0
    // grad_B[1,0]=3*0=0       grad_B[1,1]=3*1=3
    // grad_B[2,0]=1*1+4*0=1   grad_B[2,1]=1*0+4*0=0
    // -> [[2,0],[0,3],[1,0]]
    let values = t(&[5], &[2.0, 1.0, 3.0, 1.0, 4.0], backend);
    let col_indices = Tensor::<i64, B>::from_slice_on(vec![5], &[0i64, 2, 1, 0, 2], backend).expect("construct tensor");
    let row_offsets = Tensor::<i64, B>::from_slice_on(vec![4], &[0i64, 2, 3, 5], backend).expect("construct tensor");
    let grad_out = t(&[3, 2], &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0], backend);
    let gb = coeus_ops::spmm_backward_dense(
        &values,
        &col_indices,
        &row_offsets,
        &[3, 3],
        &grad_out,
        backend,
    ).expect("run operation");
    assert_eq!(gb.shape(), &[3, 2], "spmm_backward_dense shape");
    assert_eq!(
        gb.as_slice(),
        &[2.0_f64, 0.0, 0.0, 3.0, 1.0, 0.0],
        "spmm_backward_dense"
    );
}

// wrappers

fn check_all<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + coeus_core::Backend + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
    B::DeviceBuffer<i64>: CpuAddressableStorage<i64> + CpuAddressableStorageMut<i64>,
{
    check_spmv(backend);
    check_spmm(backend);
    check_spmm_backward_values(backend);
    check_spmm_backward_dense(backend);
}

#[test]
fn sequential_sparse_ops_match_reference() {
    let backend = SequentialBackend;
    check_all(&backend);
}

#[test]
fn moirai_sparse_ops_match_reference() {
    let backend = MoiraiBackend;
    check_all(&backend);
}
