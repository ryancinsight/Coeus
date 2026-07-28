//! Differential parity for `GlobalAvgPool1d`, `GlobalAvgPool3d`, `GlobalMaxPool3d`.
//!
//! (`GlobalAvgPool2d` and `GlobalMaxPool2d` are already covered in nn_new_ops_tests.rs.)
//!
//! Analytical oracles (all exact in IEEE-754 f64):
//!   GlobalAvgPool1d([1,2,3,4]) = mean = 2.5    (10/4, exact binary fraction)
//!   GlobalAvgPool3d([1..8])    = mean = 4.5    (36/8, exact binary fraction)
//!   GlobalMaxPool3d([1..8])    = max  = 8.0    (max of all elements)
//!
//! SequentialBackend and MoiraiBackend must produce bitwise-identical results.

use coeus_autograd::Var;
use coeus_core::{
    CpuAddressableStorage, CpuAddressableStorageMut, MoiraiBackend, SequentialBackend,
};
use coeus_nn::{GlobalAvgPool1d, GlobalAvgPool3d, GlobalMaxPool3d, Module};
use coeus_ops::BackendOps;
use coeus_tensor::Tensor;

fn v<B: BackendOps<f64> + Default>(shape: &[usize], vals: &[f64], backend: &B) -> Var<f64, B>
where
    B::DeviceBuffer<f64>: CpuAddressableStorageMut<f64>,
{
    Var::new(Tensor::from_slice_on(shape.to_vec(), vals, backend).expect("construct tensor"), false).expect("construct variable")
}

fn check_global_pools<B: BackendOps<f64> + Default>(backend: &B)
where
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    // GlobalAvgPool1d: [N=1,C=1,L=4] → [1,1,1]
    // mean([1,2,3,4]) = 10/4 = 2.5 (exact — 2.5 is representable in binary)
    let pool1d = GlobalAvgPool1d::<f64, B>::new();
    let inp1 = v(&[1, 1, 4], &[1.0, 2.0, 3.0, 4.0], backend);
    let out1 = Module::<f64, B>::forward(&pool1d, &inp1).expect("run forward");
    assert_eq!(
        out1.tensor.shape(),
        &[1, 1, 1],
        "GlobalAvgPool1d output shape"
    );
    assert_eq!(
        out1.tensor.as_slice(),
        &[2.5_f64],
        "GlobalAvgPool1d([1,2,3,4]) = 2.5"
    );

    // Uniform input: mean of all-same values = that value (exact).
    let inp1b = v(&[1, 2, 3], &[3.0_f64; 6], backend);
    let out1b = Module::<f64, B>::forward(&pool1d, &inp1b).expect("run forward");
    assert_eq!(
        out1b.tensor.shape(),
        &[1, 2, 1],
        "GlobalAvgPool1d uniform shape"
    );
    assert_eq!(
        out1b.tensor.as_slice(),
        &[3.0_f64, 3.0],
        "GlobalAvgPool1d uniform = constant"
    );

    // GlobalAvgPool3d: [N=1,C=1,D=2,H=2,W=2] → [1,1,1,1,1]
    // Values 1..8 row-major. Mean = (1+2+…+8)/8 = 36/8 = 4.5 (exact).
    let pool3d_avg = GlobalAvgPool3d::<f64, B>::new();
    let data3: Vec<f64> = (1u64..=8).map(|i| i as f64).collect();
    let inp3 = v(&[1, 1, 2, 2, 2], &data3, backend);
    let out3 = Module::<f64, B>::forward(&pool3d_avg, &inp3).expect("run forward");
    assert_eq!(
        out3.tensor.shape(),
        &[1, 1, 1, 1, 1],
        "GlobalAvgPool3d output shape"
    );
    assert_eq!(
        out3.tensor.as_slice(),
        &[4.5_f64],
        "GlobalAvgPool3d([1..8]) = 4.5"
    );

    // GlobalMaxPool3d: [N=1,C=1,D=2,H=2,W=2] → [1,1,1,1,1]
    // max([1..8]) = 8.0 exactly.
    let pool3d_max = GlobalMaxPool3d::<f64, B>::new();
    let out3m = Module::<f64, B>::forward(&pool3d_max, &inp3).expect("run forward");
    assert_eq!(
        out3m.tensor.shape(),
        &[1, 1, 1, 1, 1],
        "GlobalMaxPool3d output shape"
    );
    assert_eq!(
        out3m.tensor.as_slice(),
        &[8.0_f64],
        "GlobalMaxPool3d([1..8]) = 8.0"
    );
}

#[test]
fn sequential_global_pool_match_reference() {
    check_global_pools(&SequentialBackend);
}

#[test]
fn moirai_global_pool_match_reference() {
    check_global_pools(&MoiraiBackend);
}
