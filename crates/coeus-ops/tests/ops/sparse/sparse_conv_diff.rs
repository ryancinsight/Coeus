//! Differential parity for sparse format conversion operations.
//!
//! Functions exercised:
//!   `dense_to_coo`  - dense tensor -> COO sparse format
//!   `coo_to_dense`  - COO -> dense tensor
//!   `dense_to_csr`  - dense tensor -> CSR sparse format
//!   `csr_to_dense`  - CSR -> dense tensor
//!   `coo_to_csr`    - COO -> CSR format conversion
//!
//! Reference matrix throughout:
//!   A = [[2, 0, 1],     nnz=5: (0,0)=2, (0,2)=1, (1,1)=3, (2,0)=1, (2,2)=4
//!        [0, 3, 0],
//!        [1, 0, 4]]
//!
//! Roundtrip invariant: dense -> sparse -> dense = identity.
//! All assertions use `assert_eq!` (exact integer values).
//! SequentialBackend and MoiraiBackend must return identical results.

use coeus_core::{
    CpuAddressableStorage, CpuAddressableStorageMut, MoiraiBackend, SequentialBackend,
};
use coeus_sparse::{CooTensor, CsrTensor};
use coeus_tensor::Tensor;

// Dense form of test matrix A [3,3].
const A_DENSE: [f64; 9] = [2.0, 0.0, 1.0, 0.0, 3.0, 0.0, 1.0, 0.0, 4.0];

fn dense_a<B>(backend: &B) -> Tensor<f64, B>
where
    B: coeus_core::Backend,
    B::DeviceBuffer<f64>: CpuAddressableStorageMut<f64>,
{
    Tensor::from_slice_on(vec![3, 3], &A_DENSE, backend).expect("construct tensor")
}

fn make_coo<B>(backend: &B) -> CooTensor<f64, B>
where
    B: coeus_core::Backend,
    B::DeviceBuffer<f64>: CpuAddressableStorageMut<f64>,
    B::DeviceBuffer<i64>: CpuAddressableStorageMut<i64>,
{
    // Non-zeros in row-major order: (0,0)=2, (0,2)=1, (1,1)=3, (2,0)=1, (2,2)=4
    // indices shape [rank=2, nnz=5]:
    //   indices[0..5] (row coords) = [0,0,1,2,2]
    //   indices[5..10] (col coords) = [0,2,1,0,2]
    let indices =
        Tensor::<i64, B>::from_slice_on(vec![2, 5], &[0, 0, 1, 2, 2, 0, 2, 1, 0, 2], backend).expect("construct tensor");
    let values = Tensor::<f64, B>::from_slice_on(vec![5], &[2.0, 1.0, 3.0, 1.0, 4.0], backend).expect("construct tensor");
    CooTensor::new(vec![3, 3].into(), indices, values)
}

fn make_csr<B>(backend: &B) -> CsrTensor<f64, B>
where
    B: coeus_core::Backend,
    B::DeviceBuffer<f64>: CpuAddressableStorageMut<f64>,
    B::DeviceBuffer<i64>: CpuAddressableStorageMut<i64>,
{
    let values = Tensor::<f64, B>::from_slice_on(vec![5], &[2.0, 1.0, 3.0, 1.0, 4.0], backend).expect("construct tensor");
    let col_indices = Tensor::<i64, B>::from_slice_on(vec![5], &[0i64, 2, 1, 0, 2], backend).expect("construct tensor");
    let row_offsets = Tensor::<i64, B>::from_slice_on(vec![4], &[0i64, 2, 3, 5], backend).expect("construct tensor");
    CsrTensor::new(vec![3, 3].into(), values, col_indices, row_offsets)
}

// DENSE_TO_COO + COO_TO_DENSE roundtrip

fn check_dense_coo_roundtrip<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + coeus_core::Backend + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
    B::DeviceBuffer<i64>: CpuAddressableStorage<i64> + CpuAddressableStorageMut<i64>,
{
    let d = dense_a(backend);
    let coo = coeus_ops::dense_to_coo(&d, backend).expect("run operation");

    // COO should have 5 non-zeros.
    assert_eq!(coo.nnz(), 5, "dense_to_coo nnz");
    assert_eq!(&**coo.shape(), d.shape(), "dense_to_coo shape");

    // Roundtrip: COO -> dense.
    let recovered = coeus_ops::coo_to_dense(&coo, backend).expect("run operation");
    assert_eq!(recovered.shape(), &[3, 3], "coo_to_dense shape");
    assert_eq!(recovered.as_slice(), &A_DENSE, "coo_to_dense roundtrip");

    // All-zero matrix: COO should have 0 nnz.
    let zeros = Tensor::<f64, B>::zeros_on([2, 3], backend).expect("construct tensor");
    let coo_z = coeus_ops::dense_to_coo(&zeros, backend).expect("run operation");
    assert_eq!(coo_z.nnz(), 0, "dense_to_coo all-zero nnz");
}

// COO_TO_DENSE (from a manually-constructed COO)

fn check_coo_to_dense<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + coeus_core::Backend + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
    B::DeviceBuffer<i64>: CpuAddressableStorage<i64> + CpuAddressableStorageMut<i64>,
{
    let coo = make_coo(backend);
    let dense = coeus_ops::coo_to_dense(&coo, backend).expect("run operation");
    assert_eq!(dense.shape(), &[3, 3], "coo_to_dense shape");
    assert_eq!(dense.as_slice(), &A_DENSE, "coo_to_dense values");
}

// DENSE_TO_CSR + CSR_TO_DENSE roundtrip

fn check_dense_csr_roundtrip<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + coeus_core::Backend + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
    B::DeviceBuffer<i64>: CpuAddressableStorage<i64> + CpuAddressableStorageMut<i64>,
{
    let d = dense_a(backend);
    let csr = coeus_ops::dense_to_csr(&d, backend).expect("run operation");

    // Recovered dense must equal original.
    let recovered = coeus_ops::csr_to_dense(&csr, backend).expect("run operation");
    assert_eq!(
        recovered.shape(),
        &[3, 3],
        "dense_to_csr->csr_to_dense shape"
    );
    assert_eq!(
        recovered.as_slice(),
        &A_DENSE,
        "dense_to_csr->csr_to_dense roundtrip"
    );

    // Identity matrix: each row has exactly 1 non-zero.
    let eye_data = [1.0_f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let eye = Tensor::<f64, B>::from_slice_on(vec![3, 3], &eye_data, backend).expect("construct tensor");
    let csr_eye = coeus_ops::dense_to_csr(&eye, backend).expect("run operation");
    let rec_eye = coeus_ops::csr_to_dense(&csr_eye, backend).expect("run operation");
    assert_eq!(rec_eye.as_slice(), &eye_data, "dense_to_csr identity");
}

// CSR_TO_DENSE (from manually-constructed CSR)

fn check_csr_to_dense<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + coeus_core::Backend + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
    B::DeviceBuffer<i64>: CpuAddressableStorage<i64> + CpuAddressableStorageMut<i64>,
{
    let csr = make_csr(backend);
    let dense = coeus_ops::csr_to_dense(&csr, backend).expect("run operation");
    assert_eq!(dense.shape(), &[3, 3], "csr_to_dense shape");
    assert_eq!(dense.as_slice(), &A_DENSE, "csr_to_dense values");
}

// COO_TO_CSR then CSR_TO_DENSE

fn check_coo_to_csr<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + coeus_core::Backend + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
    B::DeviceBuffer<i64>: CpuAddressableStorage<i64> + CpuAddressableStorageMut<i64>,
{
    let coo = make_coo(backend);
    let csr = coeus_ops::coo_to_csr(&coo, backend).expect("run operation");

    // After COO->CSR, recover the dense matrix to verify correctness.
    let dense = coeus_ops::csr_to_dense(&csr, backend).expect("run operation");
    assert_eq!(dense.shape(), &[3, 3], "coo_to_csr dense shape");
    assert_eq!(dense.as_slice(), &A_DENSE, "coo_to_csr then csr_to_dense");
}

// wrappers

fn check_all<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + coeus_core::Backend + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
    B::DeviceBuffer<i64>: CpuAddressableStorage<i64> + CpuAddressableStorageMut<i64>,
{
    check_dense_coo_roundtrip(backend);
    check_coo_to_dense(backend);
    check_dense_csr_roundtrip(backend);
    check_csr_to_dense(backend);
    check_coo_to_csr(backend);
}

#[test]
fn sequential_sparse_conversions_match_reference() {
    let backend = SequentialBackend;
    check_all(&backend);
}

#[test]
fn moirai_sparse_conversions_match_reference() {
    let backend = MoiraiBackend;
    check_all(&backend);
}
