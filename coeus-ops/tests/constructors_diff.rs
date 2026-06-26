//! Differential parity for constructor and selection utilities.
//!
//! Functions exercised:
//!   `linspace`   - n evenly-spaced values (inclusive)
//!   `logspace`   - n logarithmically-spaced values (base^start ... base^end)
//!   `geomspace`  - n geometrically-spaced values
//!   `meshgrid`   - coordinate grids from 1-D tensors
//!   `nonzero`    - row-major indices of non-zero elements
//!   `where_cond` - element-wise conditional select (zero tests previously)
//!
//! Reference values are IEEE-exact (integer-valued or exact powers of 10) so
//! that `assert_eq!` holds without an epsilon band.  `where_cond` is a pure
//! element-wise selection so its output is bitwise identical to the selected
//! branch values.
//!
//! SequentialBackend and MoiraiBackend receive identical inputs and must
//! return identical outputs.  Divergence indicates a backend dispatch bug.

use coeus_core::{
    CpuAddressableStorage, CpuAddressableStorageMut, MoiraiBackend, SequentialBackend,
};
use coeus_tensor::Tensor;

// helper

fn t<B>(shape: &[usize], vals: &[f64], backend: &B) -> Tensor<f64, B>
where
    B: coeus_core::ComputeBackend,
    B::DeviceBuffer<f64>: CpuAddressableStorageMut<f64>,
{
    Tensor::from_slice_on(shape.to_vec(), vals, backend)
}

// LINSPACE

fn check_linspace<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    // linspace(0, 4, 5): step=1.0, integer sequence, bitwise-exact.
    let v = coeus_ops::linspace(0.0_f64, 4.0_f64, 5, backend);
    assert_eq!(v.shape(), &[5], "linspace shape");
    assert_eq!(
        v.as_slice(),
        &[0.0_f64, 1.0, 2.0, 3.0, 4.0],
        "linspace 0..4"
    );

    // linspace(1, 1, 1): single point equals start.
    let s = coeus_ops::linspace(1.0_f64, 1.0_f64, 1, backend);
    assert_eq!(s.as_slice(), &[1.0_f64], "linspace n=1");

    // linspace(0, 10, 11): step=1, 0 through 10.
    let w = coeus_ops::linspace(0.0_f64, 10.0_f64, 11, backend);
    assert_eq!(w.shape(), &[11], "linspace 11 shape");
    let expected_w: Vec<f64> = (0..=10).map(|i| i as f64).collect();
    assert_eq!(w.as_slice(), expected_w.as_slice(), "linspace 0..10");
}

// LOGSPACE

fn check_logspace<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    // logspace(0, 3, 4, base=10): exponents [0,1,2,3] -> [1, 10, 100, 1000].
    // All values are exact powers of 10 representable without rounding in f64.
    let v = coeus_ops::logspace(0.0_f64, 3.0_f64, 4, 10.0_f64, backend);
    assert_eq!(v.shape(), &[4], "logspace shape");
    assert_eq!(
        v.as_slice(),
        &[1.0_f64, 10.0, 100.0, 1000.0],
        "logspace base-10"
    );

    // logspace(0, 3, 4, base=2): exponents [0,1,2,3] -> [1, 2, 4, 8]. Exact.
    let v2 = coeus_ops::logspace(0.0_f64, 3.0_f64, 4, 2.0_f64, backend);
    assert_eq!(v2.as_slice(), &[1.0_f64, 2.0, 4.0, 8.0], "logspace base-2");
}

// GEOMSPACE

fn check_geomspace<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    // geomspace(1, 8, 4): ratio = 2^(1/(4-1)) = 2^(1/3) but 8 = 2^3, so ratio=2
    // -> [1.0, 2.0, 4.0, 8.0], all exact integers.
    let v = coeus_ops::geomspace(1.0_f64, 8.0_f64, 4, backend);
    assert_eq!(v.shape(), &[4], "geomspace shape");
    assert_eq!(
        v.as_slice(),
        &[1.0_f64, 2.0, 4.0, 8.0],
        "geomspace 1..8 n=4"
    );

    // geomspace(1, 1, 3): constant sequence (ratio=1) -> [1, 1, 1].
    let c = coeus_ops::geomspace(1.0_f64, 1.0_f64, 3, backend);
    assert_eq!(c.as_slice(), &[1.0_f64, 1.0, 1.0], "geomspace constant");
}

// MESHGRID

fn check_meshgrid<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    // x=[1,2] (len 2), y=[3,4,5] (len 3), indexing="ij" -> two [2,3] grids.
    //   grid0[i,j] = x[i]: [[1,1,1],[2,2,2]]
    //   grid1[i,j] = y[j]: [[3,4,5],[3,4,5]]
    let x = t(&[2], &[1.0, 2.0], backend);
    let y = t(&[3], &[3.0, 4.0, 5.0], backend);
    let grids = coeus_ops::meshgrid(&[&x, &y], "ij", backend);
    assert_eq!(grids.len(), 2, "meshgrid output count");

    assert_eq!(grids[0].shape(), &[2, 3], "meshgrid grid0 shape");
    assert_eq!(
        grids[0].as_slice(),
        &[1.0_f64, 1.0, 1.0, 2.0, 2.0, 2.0],
        "meshgrid grid0"
    );
    assert_eq!(grids[1].shape(), &[2, 3], "meshgrid grid1 shape");
    assert_eq!(
        grids[1].as_slice(),
        &[3.0_f64, 4.0, 5.0, 3.0, 4.0, 5.0],
        "meshgrid grid1"
    );

    // "xy" indexing: first arg varies along dim-1, second along dim-0.
    //   grid0[i,j] = x[j]: [[1,2],[1,2],[1,2]] shape [3,2]
    //   grid1[i,j] = y[i]: [[3,3],[4,4],[5,5]] shape [3,2]
    let grids_xy = coeus_ops::meshgrid(&[&x, &y], "xy", backend);
    assert_eq!(grids_xy[0].shape(), &[3, 2], "meshgrid xy grid0 shape");
    assert_eq!(
        grids_xy[0].as_slice(),
        &[1.0_f64, 2.0, 1.0, 2.0, 1.0, 2.0],
        "meshgrid xy grid0"
    );
    assert_eq!(grids_xy[1].shape(), &[3, 2], "meshgrid xy grid1 shape");
    assert_eq!(
        grids_xy[1].as_slice(),
        &[3.0_f64, 3.0, 4.0, 4.0, 5.0, 5.0],
        "meshgrid xy grid1"
    );
}

// NONZERO

fn check_nonzero<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    // [[0,1],[2,0]]: non-zero at (0,1) and (1,0) -> [[0,1],[1,0]] shape [2,2].
    let m = t(&[2, 2], &[0.0, 1.0, 2.0, 0.0], backend);
    let nz = coeus_ops::nonzero(&m, backend);
    assert_eq!(nz.shape(), &[2, 2], "nonzero 2x2 shape");
    assert_eq!(
        nz.as_slice(),
        &[0.0_f64, 1.0, 1.0, 0.0],
        "nonzero 2x2 indices"
    );

    // 1-D [0, 5, 0, 3]: non-zero at positions 1 and 3 -> [[1],[3]] shape [2,1].
    let v = t(&[4], &[0.0, 5.0, 0.0, 3.0], backend);
    let nzv = coeus_ops::nonzero(&v, backend);
    assert_eq!(nzv.shape(), &[2, 1], "nonzero 1-D shape");
    assert_eq!(nzv.as_slice(), &[1.0_f64, 3.0], "nonzero 1-D indices");

    // All-zero tensor -> empty result shape [0, 2].
    let z = Tensor::<f64, B>::from_slice_on(vec![2, 2], &[0.0; 4], backend);
    let nzz = coeus_ops::nonzero(&z, backend);
    assert_eq!(nzz.shape(), &[0, 2], "nonzero all-zero shape");
    assert!(nzz.as_slice().is_empty(), "nonzero all-zero empty");
}

// WHERE_COND

fn check_where_cond<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    // cond=[1,0,1,0], on_true=[10,20,30,40], on_false=[1,2,3,4]
    // out = [10, 2, 30, 4]
    let cond = t(&[4], &[1.0, 0.0, 1.0, 0.0], backend);
    let on_true = t(&[4], &[10.0, 20.0, 30.0, 40.0], backend);
    let on_false = t(&[4], &[1.0, 2.0, 3.0, 4.0], backend);
    let out = coeus_ops::where_cond(&cond, &on_true, &on_false, backend);
    assert_eq!(out.shape(), &[4], "where_cond shape");
    assert_eq!(
        out.as_slice(),
        &[10.0_f64, 2.0, 30.0, 4.0],
        "where_cond values"
    );

    // Negative cond treated as non-zero (true).
    // cond=[-1,0], on_true=[100,200], on_false=[0,0] -> [100, 0]
    let cond_neg = t(&[2], &[-1.0, 0.0], backend);
    let ot2 = t(&[2], &[100.0, 200.0], backend);
    let of2 = t(&[2], &[0.0, 0.0], backend);
    let out2 = coeus_ops::where_cond(&cond_neg, &ot2, &of2, backend);
    assert_eq!(
        out2.as_slice(),
        &[100.0_f64, 0.0],
        "where_cond negative cond"
    );

    // All-true: output equals on_true.
    let all_true = t(&[3], &[1.0, 2.0, 3.0], backend);
    let vt = t(&[3], &[7.0, 8.0, 9.0], backend);
    let vf = t(&[3], &[0.0, 0.0, 0.0], backend);
    let all_t = coeus_ops::where_cond(&all_true, &vt, &vf, backend);
    assert_eq!(all_t.as_slice(), vt.as_slice(), "where_cond all-true");

    // 2-D shape: [[1,0],[0,1]] selects from [[10,20],[30,40]] vs [[1,2],[3,4]].
    // out = [[10,2],[3,40]]
    let c2 = t(&[2, 2], &[1.0, 0.0, 0.0, 1.0], backend);
    let t2 = t(&[2, 2], &[10.0, 20.0, 30.0, 40.0], backend);
    let f2 = t(&[2, 2], &[1.0, 2.0, 3.0, 4.0], backend);
    let out2d = coeus_ops::where_cond(&c2, &t2, &f2, backend);
    assert_eq!(out2d.shape(), &[2, 2], "where_cond 2-D shape");
    assert_eq!(
        out2d.as_slice(),
        &[10.0_f64, 2.0, 3.0, 40.0],
        "where_cond 2-D"
    );
}

// wrappers

fn check_all<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    check_linspace(backend);
    check_logspace(backend);
    check_geomspace(backend);
    check_meshgrid(backend);
    check_nonzero(backend);
    check_where_cond(backend);
}

#[test]
fn sequential_constructors_match_reference() {
    let backend = SequentialBackend;
    check_all(&backend);
}

#[test]
fn moirai_constructors_match_reference() {
    let backend = MoiraiBackend;
    check_all(&backend);
}
