//! Differential parity for statistical and Lp-norm reductions.
//!
//! Functions exercised:
//!   `var`          - population / sample variance (scalar)
//!   `var_axis`     - per-axis population / sample variance (tensor)
//!   `std_dev`      - population / sample standard deviation (scalar)
//!   `std_dev_axis` - per-axis standard deviation (tensor)
//!   `norm_p`       - Lp-norm over all elements (scalar)
//!   `norm_p_axis`  - per-axis Lp-norm (tensor)
//!
//! All tolerances derive from f64 machine epsilon scaled by a conservative upper
//! bound on the operation count and libm rounding in these fixed small cases. Cases
//! where the exact result is an IEEE-representable value (e.g. sqrt(4.0)=2.0,
//! sqrt(25.0)=5.0) use `assert_eq!` directly.
//!
//! SequentialBackend and MoiraiBackend are verified against the same reference.
//! Divergence between backends indicates a dispatch bug.

use coeus_core::{
    CpuAddressableStorage, CpuAddressableStorageMut, MoiraiBackend, SequentialBackend,
};
use coeus_tensor::Tensor;

const VAR_STD_EPS: f64 = f64::EPSILON * 1_000_000.0;
const NORM_EPS: f64 = f64::EPSILON * 100_000.0;

// constructors

fn t<B>(shape: &[usize], vals: &[f64], backend: &B) -> Tensor<f64, B>
where
    B: coeus_core::ComputeBackend,
    B::DeviceBuffer<f64>: CpuAddressableStorageMut<f64>,
{
    Tensor::from_slice_on(shape.to_vec(), vals, backend)
}

// assertion helper

fn assert_close(got: &[f64], expected: &[f64], eps: f64, context: &str) {
    assert_eq!(got.len(), expected.len(), "{context}: length mismatch");
    for (i, (&g, &e)) in got.iter().zip(expected).enumerate() {
        assert!(
            (g - e).abs() <= eps,
            "{context}[{i}]: got {g:.10}, expected {e:.10}, eps={eps}"
        );
    }
}

// VAR / STD_DEV (global)

fn check_var_std<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    // x = [2,4,4,4,5,5,7,9], N=8
    // mean = 40/8 = 5.0
    // sum((x_i - mean)^2) = 9+1+1+1+0+0+4+16 = 32
    // pop var  (unbiased=false) = 32/8 = 4.0  (exact)
    // sample var (unbiased=true) = 32/7 ~= 4.571428
    // pop std  = sqrt(4.0) = 2.0  (exact)
    // sample std = sqrt(32/7)
    let x = t(&[8], &[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0], backend);

    let pop_var = coeus_ops::var(&x, false, backend);
    assert_eq!(pop_var, 4.0_f64, "var pop: got {pop_var}");

    let sample_var = coeus_ops::var(&x, true, backend);
    let expected_sample_var = 32.0_f64 / 7.0;
    assert!(
        (sample_var - expected_sample_var).abs() <= VAR_STD_EPS,
        "var sample: got {sample_var:.12}, expected {expected_sample_var:.12}"
    );

    let pop_std = coeus_ops::std_dev(&x, false, backend);
    assert_eq!(pop_std, 2.0_f64, "std_dev pop: got {pop_std}");

    let sample_std = coeus_ops::std_dev(&x, true, backend);
    let expected_sample_std = (32.0_f64 / 7.0_f64).sqrt();
    assert!(
        (sample_std - expected_sample_std).abs() <= VAR_STD_EPS,
        "std_dev sample: got {sample_std:.12}, expected {expected_sample_std:.12}"
    );

    // Constant tensor: all zeros -> var=0, std=0
    let zeros = t(&[4], &[3.0, 3.0, 3.0, 3.0], backend);
    assert_eq!(
        coeus_ops::var(&zeros, false, backend),
        0.0_f64,
        "var const=0"
    );
    assert_eq!(
        coeus_ops::std_dev(&zeros, false, backend),
        0.0_f64,
        "std const=0"
    );
}

// VAR_AXIS / STD_DEV_AXIS

fn check_var_axis<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    // M = [[0, 4], [4, 4]], shape [2, 2]
    //
    // axis=0 (column-wise), N=2 per slice:
    //   col0=[0,4]: mean=2, sum((x-mean)^2)=4+4=8, pop_var=4.0, pop_std=2.0
    //   col1=[4,4]: mean=4, sum((x-mean)^2)=0,     pop_var=0.0, pop_std=0.0
    //   output shape [1,2]
    //
    // axis=1 (row-wise), N=2 per slice:
    //   row0=[0,4]: mean=2, sum((x-mean)^2)=8, pop_var=4.0, pop_std=2.0
    //   row1=[4,4]: mean=4, sum((x-mean)^2)=0, pop_var=0.0, pop_std=0.0
    //   output shape [2,1]
    let m = t(&[2, 2], &[0.0, 4.0, 4.0, 4.0], backend);

    let v0 = coeus_ops::var_axis(&m, 0, false, backend);
    assert_eq!(v0.shape(), &[1, 2], "var_axis=0 shape");
    assert_close(v0.as_slice(), &[4.0, 0.0], VAR_STD_EPS, "var_axis=0 pop");

    let v1 = coeus_ops::var_axis(&m, 1, false, backend);
    assert_eq!(v1.shape(), &[2, 1], "var_axis=1 shape");
    assert_close(v1.as_slice(), &[4.0, 0.0], VAR_STD_EPS, "var_axis=1 pop");

    let sd0 = coeus_ops::std_dev_axis(&m, 0, false, backend);
    assert_eq!(sd0.shape(), &[1, 2], "std_dev_axis=0 shape");
    assert_close(sd0.as_slice(), &[2.0, 0.0], VAR_STD_EPS, "std_dev_axis=0 pop");

    let sd1 = coeus_ops::std_dev_axis(&m, 1, false, backend);
    assert_eq!(sd1.shape(), &[2, 1], "std_dev_axis=1 shape");
    assert_close(sd1.as_slice(), &[2.0, 0.0], VAR_STD_EPS, "std_dev_axis=1 pop");

    // sample (Bessel correction, N-1=1): same squared-deviation sum, divide by 1 instead of 2
    // col0 unbiased_var = 8/1 = 8.0, col1 = 0.0
    let vs0 = coeus_ops::var_axis(&m, 0, true, backend);
    assert_close(vs0.as_slice(), &[8.0, 0.0], VAR_STD_EPS, "var_axis=0 sample");

    // rank-3 shape [2,3,2], axis=2 (inner-most, N=2 per slice)
    // rows: [1,3],[5,7],[2,4],[6,8],[3,5],[7,9]
    // Each pair [a,b]: mean=(a+b)/2,
    // sum((x-mean)^2)=((a-b)/2)^2+((b-a)/2)^2=(b-a)^2/2
    // [1,3]: var=1.0; [5,7]: var=1.0; [2,4]: var=1.0; [6,8]: var=1.0; [3,5]: var=1.0; [7,9]: var=1.0
    let r = t(
        &[2, 3, 2],
        &[1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0, 3.0, 5.0, 7.0, 9.0],
        backend,
    );
    let vr = coeus_ops::var_axis(&r, 2, false, backend);
    assert_eq!(vr.shape(), &[2, 3, 1], "var_axis rank-3 shape");
    assert_close(
        vr.as_slice(),
        &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        VAR_STD_EPS,
        "var_axis rank-3 all-1",
    );
}

// NORM_P (global)

fn check_norm_p<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    // L1: [3,-4,12] -> |3|+|-4|+|12| = 19.0
    let v1 = t(&[3], &[3.0, -4.0, 12.0], backend);
    assert_eq!(
        coeus_ops::norm_p(&v1, 1.0_f64, backend),
        19.0_f64,
        "norm_p L1"
    );

    // L2: [3,4] -> sqrt(9+16) = 5.0  (3-4-5 Pythagorean triple)
    let v2 = t(&[2], &[3.0, 4.0], backend);
    assert_eq!(
        coeus_ops::norm_p(&v2, 2.0_f64, backend),
        5.0_f64,
        "norm_p L2"
    );

    // L3: [1,1,1,1,1,1,1,1] (8 ones) -> 8^(1/3) ~= 2.0
    let v3 = t(&[8], &[1.0; 8], backend);
    assert!(
        (coeus_ops::norm_p(&v3, 3.0_f64, backend) - 2.0).abs() <= NORM_EPS,
        "norm_p L3 ones"
    );

    // L1 of 2-D matrix (flattened): [[1,2],[3,4]] -> 1+2+3+4=10.0
    let m = t(&[2, 2], &[1.0, 2.0, 3.0, 4.0], backend);
    assert_eq!(
        coeus_ops::norm_p(&m, 1.0_f64, backend),
        10.0_f64,
        "norm_p L1 matrix"
    );
}

// NORM_P_AXIS

fn check_norm_p_axis<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    // M = [[3,4],[0,5]], shape [2,2]
    //
    // p=2, axis=1 (row-wise L2):
    //   row0=[3,4] -> sqrt(9+16) = 5.0
    //   row1=[0,5] -> sqrt(0+25) = 5.0
    //   output shape [2,1]: [5.0, 5.0]
    //
    // p=1, axis=0 (col-wise L1):
    //   col0=[3,0] -> 3.0
    //   col1=[4,5] -> 9.0
    //   output shape [1,2]: [3.0, 9.0]
    let m = t(&[2, 2], &[3.0, 4.0, 0.0, 5.0], backend);

    let r2_1 = coeus_ops::norm_p_axis(&m, 2.0_f64, 1, backend);
    assert_eq!(r2_1.shape(), &[2, 1], "norm_p_axis p=2 axis=1 shape");
    assert_close(
        r2_1.as_slice(),
        &[5.0, 5.0],
        NORM_EPS,
        "norm_p_axis p=2 axis=1",
    );

    let r1_0 = coeus_ops::norm_p_axis(&m, 1.0_f64, 0, backend);
    assert_eq!(r1_0.shape(), &[1, 2], "norm_p_axis p=1 axis=0 shape");
    assert_close(
        r1_0.as_slice(),
        &[3.0, 9.0],
        NORM_EPS,
        "norm_p_axis p=1 axis=0",
    );

    // Rank-3: shape [2,2,3], axis=2 (innermost L2)
    // Slice [i,j,*]: six slices; each is [1,2,2] or [0,0,1] etc.
    //   [1,2,2]: L2 = sqrt(1+4+4) = 3.0
    //   [0,0,1]: L2 = 1.0
    //   [3,4,0]: L2 = 5.0
    //   [0,0,5]: L2 = 5.0
    //   (repeat pattern for second batch)
    let r3 = t(
        &[2, 2, 3],
        &[1.0, 2.0, 2.0, 0.0, 0.0, 1.0, 3.0, 4.0, 0.0, 0.0, 0.0, 5.0],
        backend,
    );
    let rr = coeus_ops::norm_p_axis(&r3, 2.0_f64, 2, backend);
    assert_eq!(rr.shape(), &[2, 2, 1], "norm_p_axis rank-3 shape");
    assert_close(
        rr.as_slice(),
        &[3.0, 1.0, 5.0, 5.0],
        NORM_EPS,
        "norm_p_axis rank-3",
    );
}

// generic check (wraps all sub-checks)

fn check_all<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    check_var_std(backend);
    check_var_axis(backend);
    check_norm_p(backend);
    check_norm_p_axis(backend);
}

// test wrappers

#[test]
fn sequential_stats_match_reference() {
    let backend = SequentialBackend;
    check_all(&backend);
}

#[test]
fn moirai_stats_match_reference() {
    let backend = MoiraiBackend;
    check_all(&backend);
}
