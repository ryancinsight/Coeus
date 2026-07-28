//! Differential parity for miscellaneous reduction and shape operations.
//!
//! Functions exercised:
//!   `amax`         - global maximum scalar
//!   `amin`         - global minimum scalar
//!   `dot`          - 1-D dot product
//!   `cumprod`      - cumulative product along a dimension
//!   `broadcast_to` - repeat along singleton dimensions
//!   `chunk`        - split tensor into N slices along a dimension
//!   `diag`         - construct diagonal matrix from 1-D vector
//!   `diagonal`     - extract main (or k-th) diagonal from 2-D matrix
//!
//! All reference values are integer-valued (exact in f64) so assertions use
//! `assert_eq!` without an epsilon band.  SequentialBackend and MoiraiBackend
//! receive identical inputs and must return identical results.

use coeus_core::{
    CpuAddressableStorage, CpuAddressableStorageMut, MoiraiBackend, SequentialBackend,
};
use coeus_tensor::Tensor;

fn t<B>(shape: &[usize], vals: &[f64], backend: &B) -> Tensor<f64, B>
where
    B: coeus_core::ComputeBackend,
    B::DeviceBuffer<f64>: CpuAddressableStorageMut<f64>,
{
    Tensor::from_slice_on(shape.to_vec(), vals, backend).expect("construct tensor")
}

// AMAX / AMIN

fn check_amax_amin<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    // 1-D: amax([1,5,3,2,4]) = 5, amin = 1
    let v = t(&[5], &[1.0, 5.0, 3.0, 2.0, 4.0], backend);
    assert_eq!(
        coeus_ops::amax(&v, backend).expect("valid amax"),
        5.0_f64,
        "amax 1-D"
    );
    assert_eq!(
        coeus_ops::amin(&v, backend).expect("valid amin"),
        1.0_f64,
        "amin 1-D"
    );

    // 2-D: amax([[3,1],[2,4]]) = 4, amin = 1
    let m = t(&[2, 2], &[3.0, 1.0, 2.0, 4.0], backend);
    assert_eq!(
        coeus_ops::amax(&m, backend).expect("valid amax"),
        4.0_f64,
        "amax 2-D"
    );
    assert_eq!(
        coeus_ops::amin(&m, backend).expect("valid amin"),
        1.0_f64,
        "amin 2-D"
    );

    // Single element: amax/amin both equal that element.
    let s = t(&[1], &[7.0], backend);
    assert_eq!(
        coeus_ops::amax(&s, backend).expect("valid amax"),
        7.0_f64,
        "amax scalar"
    );
    assert_eq!(
        coeus_ops::amin(&s, backend).expect("valid amin"),
        7.0_f64,
        "amin scalar"
    );
}

// DOT

fn check_dot<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    // [1,2,3] dot [4,5,6] = 4+10+18 = 32
    let a = t(&[3], &[1.0, 2.0, 3.0], backend);
    let b = t(&[3], &[4.0, 5.0, 6.0], backend);
    assert_eq!(coeus_ops::dot(&a, &b).expect("run operation"), 32.0_f64, "dot [1,2,3] dot [4,5,6]");

    // Orthogonal vectors: dot = 0.
    let x = t(&[2], &[1.0, 0.0], backend);
    let y = t(&[2], &[0.0, 1.0], backend);
    assert_eq!(coeus_ops::dot(&x, &y).expect("run operation"), 0.0_f64, "dot orthogonal");

    // Self dot: [3,4] dot [3,4] = 9+16 = 25
    let u = t(&[2], &[3.0, 4.0], backend);
    assert_eq!(coeus_ops::dot(&u, &u).expect("run operation"), 25.0_f64, "dot self");
}

// CUMPROD

fn check_cumprod<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    // 1-D [1,2,3,4], dim=0: [1, 2, 6, 24]
    let v = t(&[4], &[1.0, 2.0, 3.0, 4.0], backend);
    let cp = coeus_ops::cumprod(&v, 0, backend).expect("run operation");
    assert_eq!(cp.shape(), &[4], "cumprod 1-D shape");
    assert_eq!(cp.as_slice(), &[1.0_f64, 2.0, 6.0, 24.0], "cumprod 1-D");

    // 2-D [[1,2],[3,4]], dim=0 (column-wise):
    // col0: [1, 1*3] = [1, 3]; col1: [2, 2*4] = [2, 8]
    // -> [[1,2],[3,8]]
    let m = t(&[2, 2], &[1.0, 2.0, 3.0, 4.0], backend);
    let cp_dim0 = coeus_ops::cumprod(&m, 0, backend).expect("run operation");
    assert_eq!(cp_dim0.shape(), &[2, 2], "cumprod dim=0 shape");
    assert_eq!(
        cp_dim0.as_slice(),
        &[1.0_f64, 2.0, 3.0, 8.0],
        "cumprod dim=0"
    );

    // 2-D [[1,2],[3,4]], dim=1 (row-wise):
    // row0: [1, 1*2] = [1, 2]; row1: [3, 3*4] = [3, 12]
    // -> [[1,2],[3,12]]
    let cp_dim1 = coeus_ops::cumprod(&m, 1, backend).expect("run operation");
    assert_eq!(cp_dim1.shape(), &[2, 2], "cumprod dim=1 shape");
    assert_eq!(
        cp_dim1.as_slice(),
        &[1.0_f64, 2.0, 3.0, 12.0],
        "cumprod dim=1"
    );

    let suffix = coeus_ops::suffix_prod(&m, 1, backend).expect("run operation");
    assert_eq!(
        suffix.as_slice(),
        &[2.0_f64, 2.0, 12.0, 4.0],
        "suffix_prod dim=1"
    );
}

// BROADCAST_TO

fn check_broadcast_to<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    // [1,2,3] shape [1,3] -> [2,3]: rows repeated twice.
    // [[1,2,3],[1,2,3]]
    let v = t(&[1, 3], &[1.0, 2.0, 3.0], backend);
    let out = coeus_ops::broadcast_to(&v, &[2, 3], backend).expect("run operation");
    assert_eq!(out.shape(), &[2, 3], "broadcast_to [1,3]->[2,3] shape");
    assert_eq!(
        out.as_slice(),
        &[1.0_f64, 2.0, 3.0, 1.0, 2.0, 3.0],
        "broadcast_to [1,3]->[2,3]"
    );

    // [[1],[2]] shape [2,1] -> [2,3]: columns repeated.
    // [[1,1,1],[2,2,2]]
    let col = t(&[2, 1], &[1.0, 2.0], backend);
    let out2 = coeus_ops::broadcast_to(&col, &[2, 3], backend).expect("run operation");
    assert_eq!(out2.shape(), &[2, 3], "broadcast_to [2,1]->[2,3] shape");
    assert_eq!(
        out2.as_slice(),
        &[1.0_f64, 1.0, 1.0, 2.0, 2.0, 2.0],
        "broadcast_to [2,1]->[2,3]"
    );

    // Identity broadcast: same shape -> unchanged.
    let m = t(&[2, 2], &[1.0, 2.0, 3.0, 4.0], backend);
    let id = coeus_ops::broadcast_to(&m, &[2, 2], backend).expect("run operation");
    assert_eq!(id.as_slice(), m.as_slice(), "broadcast_to identity");
}

// CHUNK

fn check_chunk<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    // 1-D [1..5], chunks=2, dim=0: ceil(5/2)=3 -> [[1,2,3],[4,5]]
    let v = t(&[5], &[1.0, 2.0, 3.0, 4.0, 5.0], backend);
    let parts = coeus_ops::chunk(&v, 2, 0).expect("run operation");
    assert_eq!(parts.len(), 2, "chunk 5->2 count");
    assert_eq!(parts[0].shape(), &[3], "chunk part0 shape");
    assert_eq!(parts[0].as_slice(), &[1.0_f64, 2.0, 3.0], "chunk part0");
    assert_eq!(parts[1].shape(), &[2], "chunk part1 shape");
    assert_eq!(parts[1].as_slice(), &[4.0_f64, 5.0], "chunk part1");

    // 2-D [[1,2,3,4]] shape [1,4], chunks=2, dim=1: -> [[1,2]] and [[3,4]]
    let m = t(&[1, 4], &[1.0, 2.0, 3.0, 4.0], backend);
    let parts2 = coeus_ops::chunk(&m, 2, 1).expect("run operation");
    assert_eq!(parts2.len(), 2, "chunk 2d count");
    assert_eq!(parts2[0].shape(), &[1, 2], "chunk 2d part0 shape");
    assert_eq!(parts2[0].as_slice(), &[1.0_f64, 2.0], "chunk 2d part0");
    assert_eq!(parts2[1].shape(), &[1, 2], "chunk 2d part1 shape");
    assert_eq!(parts2[1].as_slice(), &[3.0_f64, 4.0], "chunk 2d part1");

    // Exact divisibility: [1,2,3,4,5,6], chunks=3, dim=0 -> three slices of 2.
    let v6 = t(&[6], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], backend);
    let parts3 = coeus_ops::chunk(&v6, 3, 0).expect("run operation");
    assert_eq!(parts3.len(), 3, "chunk 6/3 count");
    assert_eq!(parts3[0].as_slice(), &[1.0_f64, 2.0], "chunk 6/3 part0");
    assert_eq!(parts3[1].as_slice(), &[3.0_f64, 4.0], "chunk 6/3 part1");
    assert_eq!(parts3[2].as_slice(), &[5.0_f64, 6.0], "chunk 6/3 part2");
}

// DIAG / DIAGONAL

fn check_diag_diagonal<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    // diag([1,2,3], k=0) -> [[1,0,0],[0,2,0],[0,0,3]] shape [3,3]
    let v = t(&[3], &[1.0, 2.0, 3.0], backend);
    let d = coeus_ops::diag(&v, 0, backend).expect("run operation");
    assert_eq!(d.shape(), &[3, 3], "diag k=0 shape");
    assert_eq!(
        d.as_slice(),
        &[1.0_f64, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0],
        "diag k=0"
    );

    // diag([1,2], k=1) -> super-diagonal, shape [3,3]:
    // [[0,1,0],[0,0,2],[0,0,0]]
    let v2 = t(&[2], &[1.0, 2.0], backend);
    let d1 = coeus_ops::diag(&v2, 1, backend).expect("run operation");
    assert_eq!(d1.shape(), &[3, 3], "diag k=1 shape");
    assert_eq!(
        d1.as_slice(),
        &[0.0_f64, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0],
        "diag k=1"
    );

    // diag([1,2], k=-1) -> sub-diagonal, shape [3,3]:
    // [[0,0,0],[1,0,0],[0,2,0]]
    let d_neg = coeus_ops::diag(&v2, -1, backend).expect("run operation");
    assert_eq!(d_neg.shape(), &[3, 3], "diag k=-1 shape");
    assert_eq!(
        d_neg.as_slice(),
        &[0.0_f64, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0],
        "diag k=-1"
    );

    // diagonal([[1,2,3],[4,5,6],[7,8,9]], k=0) -> [1,5,9]
    let m = t(
        &[3, 3],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        backend,
    );
    let diag_main = coeus_ops::diagonal(&m, 0, backend).expect("run operation");
    assert_eq!(diag_main.shape(), &[3], "diagonal k=0 shape");
    assert_eq!(diag_main.as_slice(), &[1.0_f64, 5.0, 9.0], "diagonal k=0");

    // diagonal of non-square [[1,2,3],[4,5,6]], shape [2,3]:
    // k=0 -> min(2,3)=2 elements: [1,5]
    let rect = t(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], backend);
    let diag_rect = coeus_ops::diagonal(&rect, 0, backend).expect("run operation");
    assert_eq!(diag_rect.shape(), &[2], "diagonal rect shape");
    assert_eq!(diag_rect.as_slice(), &[1.0_f64, 5.0], "diagonal rect");
}

// wrappers

fn check_all<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    check_amax_amin(backend);
    check_dot(backend);
    check_cumprod(backend);
    check_broadcast_to(backend);
    check_chunk(backend);
    check_diag_diagonal(backend);
}

#[test]
fn sequential_misc_ops_match_reference() {
    let backend = SequentialBackend;
    check_all(&backend);
}

#[test]
fn moirai_misc_ops_match_reference() {
    let backend = MoiraiBackend;
    check_all(&backend);
}
