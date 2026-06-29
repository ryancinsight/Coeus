//! Differential parity for loss functions:
//!   `mse_loss`, `nll_loss`, `huber_loss`, `l1_loss`,
//!   `binary_cross_entropy`, `kl_divergence`, `margin_ranking_loss`,
//!   `cosine_embedding_loss`.
//!
//! (`cross_entropy_loss` is already covered by `nn_parity::test_cross_entropy_loss_parity`.)
//!
//! Analytical oracles (all derivable in closed form):
//!   mse_loss(x, x) = 0                          (exact — identical tensors)
//!   nll_loss([[-1,-2,-3]], [0]) = 1.0            (exact — pick index 0, val=-(-1)=1)
//!   huber_loss([2.0],[0.0], δ=1.0) = 1.5        (exact — δ*(|err|-0.5δ) = 1*(2-0.5))
//!   kl_divergence(log([0.25,0.75]), [0.25,0.75]) = 0 (exact — P == Q)
//!   margin_ranking_loss([2,0,1,2], [1,1,1,1], [1,-1,1,-1], m=0.5) = 0.5
//!   cosine_embedding_loss([1,0],[1,0], y=1) = 0  (exact — cos=1, loss=max(0,0)=0)
//!
//! binary_cross_entropy(pred=0.5, target=0) = -log(1-0.5) = log(2) (1-ULP eps)
//!
//! SequentialBackend and MoiraiBackend must produce bitwise-identical results.

use coeus_autograd::Var;
use coeus_core::{
    CpuAddressableStorage, CpuAddressableStorageMut, MoiraiBackend, SequentialBackend,
};
use coeus_nn::{
    binary_cross_entropy, cosine_embedding_loss, huber_loss, kl_divergence, l1_loss,
    margin_ranking_loss, mse_loss, nll_loss,
};
use coeus_ops::BackendOps;
use coeus_tensor::Tensor;

fn v<B: BackendOps<f64> + Default>(shape: &[usize], vals: &[f64], backend: &B) -> Var<f64, B>
where
    B::DeviceBuffer<f64>: CpuAddressableStorageMut<f64>,
{
    Var::new(Tensor::from_slice_on(shape.to_vec(), vals, backend), false)
}

fn check_losses<B: BackendOps<f64> + Default>(backend: &B)
where
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    // MSE identical tensors → 0 exactly.
    let pred = v(&[3], &[1.0, 2.0, 3.0], backend);
    let tgt = v(&[3], &[1.0, 2.0, 3.0], backend);
    let mse = mse_loss(&pred, &tgt);
    assert_eq!(mse.tensor.as_slice(), &[0.0_f64], "mse_loss(x,x)=0");

    // NLL: log_probs=[[-1,-2,-3]], targets=[0] → -(-1) = 1.0 exactly.
    let lp = v(&[1, 3], &[-1.0, -2.0, -3.0], backend);
    let nll = nll_loss(&lp, &[0]);
    assert_eq!(
        nll.tensor.as_slice(),
        &[1.0_f64],
        "nll_loss([[-1,-2,-3]], [0]) = 1.0"
    );

    // Huber: pred=[2.0], target=[0.0], delta=1.0 → 1*(2-0.5) = 1.5 exactly.
    // |err|=2 > delta=1 → linear branch: delta*(|err|-0.5*delta) = 1*(2-0.5) = 1.5
    let hp = v(&[1], &[2.0], backend);
    let ht = v(&[1], &[0.0], backend);
    let h = huber_loss(&hp, &ht, 1.0_f64);
    assert_eq!(
        h.tensor.as_slice(),
        &[1.5_f64],
        "huber_loss delta=1 |err|=2 → 1.5"
    );

    // Huber zero error → 0 exactly.
    let hzp = v(&[2], &[1.0, 2.0], backend);
    let hzt = v(&[2], &[1.0, 2.0], backend);
    let hz = huber_loss(&hzp, &hzt, 1.0_f64);
    assert_eq!(hz.tensor.as_slice(), &[0.0_f64], "huber_loss(x,x)=0");

    // L1: pred-target=[3,-1,0.5,0] over shape [2,2] → mean = 4.5/4 = 1.125.
    let l1p = v(&[2, 2], &[3.0, -1.0, 0.5, 4.0], backend);
    let l1t = v(&[2, 2], &[0.0, 0.0, 0.0, 4.0], backend);
    let l1 = l1_loss(&l1p, &l1t);
    assert_eq!(
        l1.tensor.as_slice(),
        &[1.125_f64],
        "l1_loss mean abs error over all elements"
    );

    // L1 zero error → 0 exactly.
    let l1zp = v(&[2], &[1.0, 2.0], backend);
    let l1zt = v(&[2], &[1.0, 2.0], backend);
    let l1z = l1_loss(&l1zp, &l1zt);
    assert_eq!(l1z.tensor.as_slice(), &[0.0_f64], "l1_loss(x,x)=0");

    // BCE: pred=[0.5], target=[0.0], eps=0.
    // loss = -0*log(0.5) - 1*log(1-0.5) = -log(0.5) = log(2).
    let bp = v(&[1], &[0.5], backend);
    let bt = v(&[1], &[0.0], backend);
    let bce = binary_cross_entropy(&bp, &bt, 0.0_f64);
    let bce_expected = std::f64::consts::LN_2;
    let bce_val = bce.tensor.as_slice()[0];
    assert!(
        (bce_val - bce_expected).abs() <= 2.0 * f64::EPSILON * bce_expected,
        "BCE pred=0.5 target=0 → ln(2): got {bce_val:.17}, expected {bce_expected:.17}"
    );

    // KL divergence is zero when target probabilities match input log-probabilities.
    let probs = [0.25_f64, 0.75_f64];
    let log_probs = [probs[0].ln(), probs[1].ln()];
    let kl_input = v(&[2], &log_probs, backend);
    let kl_target = v(&[2], &probs, backend);
    let kl = kl_divergence(&kl_input, &kl_target);
    assert!(
        kl.tensor.as_slice()[0].abs() <= 2.0 * f64::EPSILON,
        "kl_divergence(P || P) = 0"
    );

    // Margin ranking:
    // samples 0/1 are inactive, sample 2 has hinge 0.5, sample 3 has hinge 1.5.
    // mean = (0 + 0 + 0.5 + 1.5) / 4 = 0.5.
    let mr_i1 = v(&[4], &[2.0, 0.0, 1.0, 2.0], backend);
    let mr_i2 = v(&[4], &[1.0, 1.0, 1.0, 1.0], backend);
    let mr = margin_ranking_loss(&mr_i1, &mr_i2, &[1.0, -1.0, 1.0, -1.0], 0.5);
    assert_eq!(
        mr.tensor.as_slice(),
        &[0.5_f64],
        "margin_ranking_loss mixed active/inactive hinges"
    );

    // Cosine embedding loss: identical unit vectors, y=1 → loss=0 exactly.
    let x1 = v(&[1, 2], &[1.0, 0.0], backend);
    let x2 = v(&[1, 2], &[1.0, 0.0], backend);
    let cel = cosine_embedding_loss(&x1, &x2, &[1.0_f64], 0.5_f64);
    assert_eq!(
        cel.tensor.as_slice(),
        &[0.0_f64],
        "cosine_embedding_loss identical y=1 → 0"
    );

    // Cosine embedding loss: opposite unit vectors, y=-1, margin=0 →
    // loss = max(0, cos - margin) = max(0, -1 - 0) = 0 exactly.
    let x3 = v(&[1, 2], &[1.0, 0.0], backend);
    let x4 = v(&[1, 2], &[-1.0, 0.0], backend);
    let cel2 = cosine_embedding_loss(&x3, &x4, &[-1.0_f64], 0.0_f64);
    assert_eq!(
        cel2.tensor.as_slice(),
        &[0.0_f64],
        "cosine_embedding_loss opposite y=-1 margin=0 → 0"
    );
}

#[test]
fn sequential_loss_match_reference() {
    check_losses(&SequentialBackend);
}

#[test]
fn moirai_loss_match_reference() {
    check_losses(&MoiraiBackend);
}
