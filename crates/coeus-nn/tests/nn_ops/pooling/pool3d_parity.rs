//! Differential parity for `AvgPool3d` and `MaxPool3d`.
//!
//! The existing nn_3d_tests.rs covers these with MoiraiBackend only (via the
//! `Conv3d::<f64>` default type). This file adds SequentialBackend coverage.
//!
//! Analytical oracles (kernel=2, stride=2 on [1,1,2,2,2] with values 1..8):
//!   AvgPool3d → mean([1..8]) = 36/8 = 4.5  (exact binary fraction)
//!   MaxPool3d → max([1..8])  = 8.0          (exact)
//!   AvgPool3d kernel=1 → identity (each output element = corresponding input)
//!
//! SequentialBackend and MoiraiBackend must produce bitwise-identical results.

use coeus_autograd::Var;
use coeus_core::{
    CpuAddressableStorage, CpuAddressableStorageMut, MoiraiBackend, SequentialBackend,
};
use coeus_nn::{AvgPool3d, MaxPool3d, Module};
use coeus_ops::BackendOps;
use coeus_tensor::Tensor;

fn v<B: BackendOps<f64> + Default>(shape: &[usize], vals: &[f64], backend: &B) -> Var<f64, B>
where
    B::DeviceBuffer<f64>: CpuAddressableStorageMut<f64>,
{
    Var::new(Tensor::from_slice_on(shape.to_vec(), vals, backend).expect("construct tensor"), false).expect("construct variable")
}

fn check_pool3d<B: BackendOps<f64> + Default>(backend: &B)
where
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    let data: Vec<f64> = (1u64..=8).map(|i| i as f64).collect();

    // AvgPool3d kernel=2, stride=2: averages the entire 2×2×2 volume.
    // mean([1..8]) = 36/8 = 4.5 exactly.
    let avg = AvgPool3d::<f64, B>::with_params(2, 2, 0, 1);
    let inp = v(&[1, 1, 2, 2, 2], &data, backend);
    let out_avg = Module::<f64, B>::forward(&avg, &inp).expect("run forward");
    assert_eq!(out_avg.tensor.shape(), &[1, 1, 1, 1, 1], "AvgPool3d shape");
    assert_eq!(
        out_avg.tensor.as_slice(),
        &[4.5_f64],
        "AvgPool3d kernel=2 stride=2 = mean([1..8]) = 4.5"
    );

    // MaxPool3d kernel=2, stride=2: max of the 2×2×2 block = 8.0.
    let maxp = MaxPool3d::<f64, B>::with_params(2, 2, 0, 1);
    let out_max = Module::<f64, B>::forward(&maxp, &inp).expect("run forward");
    assert_eq!(out_max.tensor.shape(), &[1, 1, 1, 1, 1], "MaxPool3d shape");
    assert_eq!(
        out_max.tensor.as_slice(),
        &[8.0_f64],
        "MaxPool3d kernel=2 stride=2 = max([1..8]) = 8.0"
    );

    // AvgPool3d kernel=1, stride=1: identity (each element passed through unchanged).
    let avg1 = AvgPool3d::<f64, B>::new(1);
    let out_id = Module::<f64, B>::forward(&avg1, &inp).expect("run forward");
    assert_eq!(
        out_id.tensor.shape(),
        &[1, 1, 2, 2, 2],
        "AvgPool3d K=1 shape"
    );
    assert_eq!(
        out_id.tensor.as_slice(),
        inp.tensor.as_slice(),
        "AvgPool3d kernel=1 = identity"
    );
}

#[test]
fn sequential_pool3d_match_reference() {
    check_pool3d(&SequentialBackend);
}

#[test]
fn moirai_pool3d_match_reference() {
    check_pool3d(&MoiraiBackend);
}
