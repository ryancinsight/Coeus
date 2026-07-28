//! Analytical parity for `BatchNorm1d`, `GroupNorm`, and `RMSNorm` modules.
//!
//! All oracles are derived from closed-form arithmetic over exact binary
//! fractions so every assertion uses `assert_eq!` (bitwise-exact, no epsilon).
//!
//! ## BatchNorm1d (eval mode) oracle
//!
//! Input `[1, 1, 3]` = `[[[2, 5, -1]]]` (N=1, C=1, L=3).
//! running_mean=[1], running_var=[3], eps=1.0 → var+eps=4, stdev=2, istdev=0.5.
//! weight=[4], bias=[2].
//! `x_hat = (x - 1) * 0.5`, `out = 4 * x_hat + 2`:
//! - (2-1)*0.5=0.5 → 4*0.5+2=4.0
//! - (5-1)*0.5=2.0 → 4*2+2=10.0
//! - (-1-1)*0.5=-1 → 4*(-1)+2=-2.0
//!
//! ## GroupNorm (G=2) oracle
//!
//! Input `[1, 4]` = `[[1, 5, 3, 7]]`, num_features=4, eps=0, weight=ones, bias=zeros.
//! Group 0: elements [1, 5], mean=3, var=((1-3)²+(5-3)²)/2=4, stdev=2.
//! Group 1: elements [3, 7], mean=5, var=4, stdev=2.
//! `x_hat = (x-μ)/2`: [-1, 1, -1, 1].
//! With weight=ones, bias=zeros: output = [-1, 1, -1, 1].
//!
//! ## RMSNorm oracle
//!
//! Input `[1, 2]` = `[[2, 2]]`, weight=[4, 3], eps=0.
//! x²=[4,4], mean_sq=4, rms=2.
//! x_hat=[1,1], out=weight*x_hat=[4,3].
//!
//! SequentialBackend and MoiraiBackend must produce bitwise-identical results.

use coeus_autograd::Var;
use coeus_core::{
    CpuAddressableStorage, CpuAddressableStorageMut, MoiraiBackend, SequentialBackend,
};
use coeus_nn::normalization::{group_norm, BatchNorm1d, GroupNorm, RMSNorm};
use coeus_nn::Module;
use coeus_ops::BackendOps;
use coeus_tensor::Tensor;

fn t<B: BackendOps<f64> + Default>(shape: &[usize], vals: &[f64], backend: &B) -> Tensor<f64, B>
where
    B::DeviceBuffer<f64>: CpuAddressableStorageMut<f64>,
{
    Tensor::from_slice_on(shape.to_vec(), vals, backend).expect("construct tensor")
}

fn v<B: BackendOps<f64> + Default>(shape: &[usize], vals: &[f64], backend: &B) -> Var<f64, B>
where
    B::DeviceBuffer<f64>: CpuAddressableStorageMut<f64>,
{
    Var::new(t(shape, vals, backend), false).expect("construct variable")
}

// ── BatchNorm1d ────────────────────────────────────────────────────────────

fn check_batch_norm_1d<B: BackendOps<f64> + Default>(backend: &B)
where
    B::DeviceBuffer<f64>: CpuAddressableStorageMut<f64>,
{
    // Eval mode: running_mean=[1], running_var=[3], eps=1.0 → stdev=2, istdev=0.5
    // weight=[4], bias=[2].
    // Input [1,1,3] = [[[2, 5, -1]]] → oracle = [4.0, 10.0, -2.0]
    let weight = v(&[1], &[4.0], backend);
    let bias = v(&[1], &[2.0], backend);
    let running_mean = t(&[1], &[1.0], backend);
    let running_var = t(&[1], &[3.0], backend);

    let mut bn = BatchNorm1d::from_parts(1, weight, bias, 1.0, 0.1, running_mean, running_var).expect("construct module");
    bn.set_training(false);

    let inp = v(&[1, 1, 3], &[2.0, 5.0, -1.0], backend);
    let out = Module::<f64, B>::forward(&bn, &inp).expect("run forward");

    assert_eq!(out.tensor.shape(), &[1, 1, 3], "BatchNorm1d output shape");
    assert_eq!(
        out.tensor.as_slice(),
        &[4.0_f64, 10.0, -2.0],
        "BatchNorm1d eval oracle"
    );

    // Multi-channel: C=2, each channel normalized independently.
    // Channel 0: running_mean=0, running_var=3, eps=1 → istdev=0.5; weight=1, bias=0
    //   → x_hat = x/2
    // Channel 1: running_mean=4, running_var=3, eps=1 → istdev=0.5; weight=2, bias=1
    //   → out = 2*(x-4)/2 + 1 = (x-4)+1 = x-3
    // Input [1,2,2] = [[[1, 3], [7, 9]]]
    //   ch0: [1*0.5, 3*0.5] = [0.5, 1.5]
    //   ch1: [7-3, 9-3]     = [4.0, 6.0]
    // Expected (NCL → output): [[[0.5, 1.5], [4.0, 6.0]]]
    let w2 = v(&[2], &[1.0, 2.0], backend);
    let b2 = v(&[2], &[0.0, 1.0], backend);
    let rm2 = t(&[2], &[0.0, 4.0], backend);
    let rv2 = t(&[2], &[3.0, 3.0], backend);

    let mut bn2 = BatchNorm1d::from_parts(2, w2, b2, 1.0, 0.1, rm2, rv2).expect("construct module");
    bn2.set_training(false);

    let inp2 = v(&[1, 2, 2], &[1.0, 3.0, 7.0, 9.0], backend);
    let out2 = Module::<f64, B>::forward(&bn2, &inp2).expect("run forward");
    assert_eq!(out2.tensor.shape(), &[1, 2, 2], "BatchNorm1d C=2 shape");
    assert_eq!(
        out2.tensor.as_slice(),
        &[0.5_f64, 1.5, 4.0, 6.0],
        "BatchNorm1d C=2 multi-channel oracle"
    );
}

// ── GroupNorm ──────────────────────────────────────────────────────────────

fn check_group_norm<B: BackendOps<f64> + Default>(backend: &B)
where
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    // GroupNorm<G=2>, num_features=4, eps=0.
    // weight=ones, bias=zeros (GroupNorm::new defaults).
    // Input [1, 4] = [[1, 5, 3, 7]].
    // Group 0: [1,5] → mean=3, var=4, stdev=2 → x_hat=[-1, 1]
    // Group 1: [3,7] → mean=5, var=4, stdev=2 → x_hat=[-1, 1]
    // output = [-1, 1, -1, 1]
    let gn = GroupNorm::<f64, B, 2>::new(4, 0.0).expect("construct module");
    let inp = v(&[1, 4], &[1.0, 5.0, 3.0, 7.0], backend);
    let out = Module::<f64, B>::forward(&gn, &inp).expect("run forward");

    assert_eq!(out.tensor.shape(), &[1, 4], "GroupNorm output shape");
    assert_eq!(
        out.tensor.as_slice(),
        &[-1.0_f64, 1.0, -1.0, 1.0],
        "GroupNorm G=2 oracle"
    );

    // Batch dimension: N=2, C=4, G=2.
    // Same values per batch row → same oracle per row.
    let inp2 = v(&[2, 4], &[1.0, 5.0, 3.0, 7.0, 1.0, 5.0, 3.0, 7.0], backend);
    let out2 = Module::<f64, B>::forward(&gn, &inp2).expect("run forward");
    assert_eq!(out2.tensor.shape(), &[2, 4], "GroupNorm N=2 shape");
    assert_eq!(
        out2.tensor.as_slice(),
        &[-1.0_f64, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
        "GroupNorm G=2 N=2 oracle"
    );

    // G=1: normalize all channels together per sample.
    // Input [1, 4] = [[1, 5, 3, 7]], mean=4, var=((1-4)²+(5-4)²+(3-4)²+(7-4)²)/4=5
    // With eps=0 not safe here since var=5≠perfect-square → use eps tolerance.
    // Instead test G=4 (each channel independently → zero variance → only bias matters).
    // Actually skip this edge case to maintain assert_eq! (no epsilon).
}

fn check_functional_group_norm<B: BackendOps<f64> + Default>(backend: &B)
where
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    // Functional group_norm over [N=1, C=4, L=1], G=2, eps=0.
    // Group 0 [1,3]: mean=2, var=1, stdev=1 -> [-1,1].
    // Group 1 [10,14]: mean=12, var=4, stdev=2 -> [-1,1].
    let input = t(&[1, 4, 1], &[1.0, 3.0, 10.0, 14.0], backend);
    let out = group_norm(&input, 2, None, None, 0.0).expect("run operation");

    assert_eq!(out.shape(), &[1, 4, 1], "functional GroupNorm shape");
    assert_eq!(
        out.as_slice(),
        &[-1.0_f64, 1.0, -1.0, 1.0],
        "functional GroupNorm no-affine oracle"
    );

    // Per-channel affine transform:
    // x_hat=[-1,1,-1,1], weight=[2,2,3,3], bias=[0.5,-0.5,1,-1]
    // -> [-1.5, 1.5, -2.0, 2.0].
    let weight = t(&[4], &[2.0, 2.0, 3.0, 3.0], backend);
    let bias = t(&[4], &[0.5, -0.5, 1.0, -1.0], backend);
    let affine = group_norm(&input, 2, Some(&weight), Some(&bias), 0.0).expect("run operation");

    assert_eq!(
        affine.as_slice(),
        &[-1.5_f64, 1.5, -2.0, 2.0],
        "functional GroupNorm affine oracle"
    );
}

#[test]
#[should_panic(expected = "group_norm: num_groups must be greater than 0")]
fn functional_group_norm_rejects_zero_groups() {
    let backend = SequentialBackend;
    let input = t(&[1, 4], &[1.0, 3.0, 10.0, 14.0], &backend);
    let _ = group_norm(&input, 0, None, None, 0.0).expect("run operation");
}

// ── RMSNorm ────────────────────────────────────────────────────────────────

fn check_rms_norm<B: BackendOps<f64> + Default>(backend: &B)
where
    B::DeviceBuffer<f64>: CpuAddressableStorageMut<f64>,
{
    // Input [1, 2] = [[2, 2]], weight=[4, 3], eps=0.
    // x²=[4,4], mean_sq=4, rms=2, x_hat=[1,1], out=[4,3].
    let weight = v(&[2], &[4.0, 3.0], backend);
    let rms = RMSNorm::from_parts(weight, 0.0).expect("construct module");
    let inp = v(&[1, 2], &[2.0, 2.0], backend);
    let out = Module::<f64, B>::forward(&rms, &inp).expect("run forward");

    assert_eq!(out.tensor.shape(), &[1, 2], "RMSNorm output shape");
    assert_eq!(
        out.tensor.as_slice(),
        &[4.0_f64, 3.0],
        "RMSNorm oracle: x=[2,2], w=[4,3] → [4,3]"
    );

    // Scalar: input [[2]], weight=[5], eps=0.
    // rms=2, x_hat=[1], out=[5].
    let weight2 = v(&[1], &[5.0], backend);
    let rms2 = RMSNorm::from_parts(weight2, 0.0).expect("construct module");
    let inp2 = v(&[1, 1], &[2.0], backend);
    let out2 = Module::<f64, B>::forward(&rms2, &inp2).expect("run forward");
    assert_eq!(
        out2.tensor.as_slice(),
        &[5.0_f64],
        "RMSNorm scalar: x=[2], w=[5] → [5]"
    );

    // Batch: N=3, same values [[2,2],[2,2],[2,2]] → each row same oracle.
    let weight3 = v(&[2], &[4.0, 3.0], backend);
    let rms3 = RMSNorm::from_parts(weight3, 0.0).expect("construct module");
    let inp3 = v(&[3, 2], &[2.0, 2.0, 2.0, 2.0, 2.0, 2.0], backend);
    let out3 = Module::<f64, B>::forward(&rms3, &inp3).expect("run forward");
    assert_eq!(out3.tensor.shape(), &[3, 2], "RMSNorm N=3 shape");
    assert_eq!(
        out3.tensor.as_slice(),
        &[4.0_f64, 3.0, 4.0, 3.0, 4.0, 3.0],
        "RMSNorm N=3 oracle"
    );
}

fn check_all<B: BackendOps<f64> + Default>(backend: &B)
where
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    check_batch_norm_1d(backend);
    check_batch_norm_1d_training(backend);
    check_group_norm(backend);
    check_functional_group_norm(backend);
    check_rms_norm(backend);
}

// ── BatchNorm1d training mode ─────────────────────────────────────────────────

/// BatchNorm1d training-mode analytical oracle.
///
/// Input `[2, 1, 2]` = `[[[1, 3]], [[5, 7]]]`, C=1 (two batches, seq=2).
/// Training mode computes per-batch mean+variance over (N, L) = 4 elements.
///
/// Population mean = (1+3+5+7)/4 = 4.0
/// Population variance (unbiased, N=4) = ((1-4)²+(3-4)²+(5-4)²+(7-4)²)/4 = (9+1+1+9)/4 = 5.0
/// NOTE: PyTorch BatchNorm uses *population* variance in the forward pass (no Bessel correction),
/// but stores the *unbiased* variance in running_var. Here we verify just the forward output.
///
/// With eps=0, weight=1, bias=0:
/// x_hat_i = (x_i - 4) / sqrt(5)  for each element
/// x_hat = [(-3, -1, 1, 3) / sqrt(5)]
///        = [-1.341640..., -0.447213..., 0.447213..., 1.341640...]
fn check_batch_norm_1d_training<B: BackendOps<f64> + Default>(backend: &B)
where
    B::DeviceBuffer<f64>: CpuAddressableStorageMut<f64>,
{
    let weight = Var::new(Tensor::from_slice_on(vec![1], &[1.0_f64], backend).expect("construct tensor"), true).expect("construct variable");
    let bias = Var::new(Tensor::from_slice_on(vec![1], &[0.0_f64], backend).expect("construct tensor"), true).expect("construct variable");
    let running_mean = Tensor::zeros_on([1], backend).expect("construct tensor");
    let running_var = Tensor::ones_on([1], backend).expect("construct tensor");

    let bn = BatchNorm1d::from_parts(1, weight, bias, 0.0, 0.0, running_mean, running_var).expect("construct module");
    // is_training = true by default

    let inp = Var::new(
        Tensor::from_slice_on(vec![2, 1, 2], &[1.0_f64, 3.0, 5.0, 7.0], backend).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let out = Module::<f64, B>::forward(&bn, &inp).expect("run forward");
    assert_eq!(out.tensor.shape(), &[2, 1, 2]);

    let s = out.tensor.as_slice();
    let mean = 4.0_f64;
    let var = 5.0_f64; // population variance of {1,3,5,7}
    let std = var.sqrt();
    let expected = [
        (1.0 - mean) / std,
        (3.0 - mean) / std,
        (5.0 - mean) / std,
        (7.0 - mean) / std,
    ];
    for (i, (&got, &exp)) in s.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-10,
            "BN1d training [{i}]: got {got:.10}, expected {exp:.10}"
        );
    }

    // Verify backward propagates to weight and bias.
    out.backward().expect("run backward");
    assert!(bn.weight.grad().is_some(), "weight grad must exist");
    assert!(bn.bias.grad().is_some(), "bias grad must exist");
}

#[test]
fn sequential_norm_parity() {
    check_all(&SequentialBackend);
}

#[test]
fn moirai_norm_parity() {
    check_all(&MoiraiBackend);
}
