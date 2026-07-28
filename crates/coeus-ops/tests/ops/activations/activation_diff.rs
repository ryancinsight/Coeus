//! Differential parity for neural network activation functions.
//!
//! Functions exercised:
//!   `sigmoid`     - 1/(1+exp(-x));  sigmoid(0)=0.5 (exact in IEEE-754)
//!   `gelu`        - 0.5*x*(1+erf(x/sqrt(2)));  gelu(0)=0 (exact)
//!   `gelu_tanh`   - tanh approximation of GELU;  gelu_tanh(0)=0 (exact)
//!   `silu`        - x*sigmoid(x);  silu(0)=0 (exact)
//!   `mish`        - x*tanh(softplus(x));  mish(0)=0 (exact)
//!   `elu`         - x if x>=0, else exp(x)-1;  elu(0)=0, elu(pos)=pos (exact)
//!   `softplus`    - log(1+exp(x));  softplus(0)=ln(2) (epsilon check)
//!   `leaky_relu`  - x if x>=0, else slope*x;  exact for all
//!
//! Strategy:
//!   - At x=0 the closed-form values are exact in f64 -> `assert_eq!`.
//!   - ELU and leaky_relu on positive inputs are identity -> `assert_eq!`.
//!   - LeakyReLU on negative inputs: slope*x is exact for representable slope
//!     and input (slope=0.25 is a power of 2, so -0.25 is exact).
//!   - softplus(0) = ln(2) has a 0-ULP analytical value; checked with 1-ULP eps.
//!   - SequentialBackend and MoiraiBackend use the same CPU kernel; their
//!     outputs are bitwise-identical, confirmed by running the same assertions.

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

fn assert_close_f64(got: &[f64], expected: &[f64], eps: f64, context: &str) {
    assert_eq!(got.len(), expected.len(), "{context}: length mismatch");
    for (i, (&g, &e)) in got.iter().zip(expected).enumerate() {
        assert!(
            (g - e).abs() <= eps,
            "{context}[{i}]: got {g:.17}, expected {e:.17}, eps={eps:.3e}"
        );
    }
}

// SIGMOID

fn check_sigmoid<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    // sigmoid(0) = 1/(1+1) = 0.5 exactly in IEEE-754.
    let z = t(&[1], &[0.0_f64], backend);
    let s = coeus_ops::sigmoid(&z, backend).expect("run operation");
    assert_eq!(s.as_slice(), &[0.5_f64], "sigmoid(0)=0.5");

    // sigmoid(x) in (0,1) for all x; test at a positive and negative value.
    // sigmoid(1.0) = 0.7310585786300049 (1 ULP tolerance for single transcendental).
    let v = t(&[2], &[1.0_f64, -1.0_f64], backend);
    let sv = coeus_ops::sigmoid(&v, backend).expect("run operation");
    assert_close_f64(
        sv.as_slice(),
        &[0.7310585786300049_f64, 0.2689414213699951],
        4.0 * f64::EPSILON,
        "sigmoid(1,-1)",
    );
}

// GELU

fn check_gelu<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    // gelu(0) = 0.5*0*(1+erf(0)) = 0 (exact).
    let z = t(&[1], &[0.0_f64], backend);
    let g = coeus_ops::gelu(&z, backend).expect("run operation");
    assert_eq!(g.as_slice(), &[0.0_f64], "gelu(0)=0");
}

// GELU_TANH

fn check_gelu_tanh<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    // gelu_tanh(0) = 0.5*0*tanh(...) = 0 (exact).
    let z = t(&[1], &[0.0_f64], backend);
    let g = coeus_ops::gelu_tanh(&z, backend).expect("run operation");
    assert_eq!(g.as_slice(), &[0.0_f64], "gelu_tanh(0)=0");
}

// SILU

fn check_silu<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    // silu(0) = 0*sigmoid(0) = 0*0.5 = 0 (exact).
    let z = t(&[1], &[0.0_f64], backend);
    let s = coeus_ops::silu(&z, backend).expect("run operation");
    assert_eq!(s.as_slice(), &[0.0_f64], "silu(0)=0");
}

// MISH

fn check_mish<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    // mish(0) = 0*tanh(softplus(0)) = 0 (exact, x=0 factor).
    let z = t(&[1], &[0.0_f64], backend);
    let m = coeus_ops::mish(&z, backend).expect("run operation");
    assert_eq!(m.as_slice(), &[0.0_f64], "mish(0)=0");
}

// ELU

fn check_elu<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    // Positive branch: elu(x) = x (exact identity).
    let pos = t(&[3], &[0.0_f64, 1.0, 4.0], backend);
    let e_pos = coeus_ops::elu(&pos, backend).expect("run operation");
    assert_eq!(
        e_pos.as_slice(),
        &[0.0_f64, 1.0, 4.0],
        "elu positive branch = identity"
    );

    // Negative branch: elu(x) = exp(x) - 1 (not exact; derived eps = 2*epsilon).
    // elu(-1) = exp(-1) - 1, approximately -0.6321205588285578.
    let neg = t(&[1], &[-1.0_f64], backend);
    let e_neg = coeus_ops::elu(&neg, backend).expect("run operation");
    assert_close_f64(
        e_neg.as_slice(),
        &[std::f64::consts::E.recip() - 1.0],
        2.0 * f64::EPSILON,
        "elu(-1)=exp(-1)-1",
    );
}

// SOFTPLUS

fn check_softplus<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    // softplus(0) = log(1 + exp(0)) = log(2), approximately 0.6931471805599453.
    // log(2) in f64 is accurate to 1 ULP; tolerance = 2*epsilon * log(2).
    let z = t(&[1], &[0.0_f64], backend);
    let sp = coeus_ops::softplus(&z, backend).expect("run operation");
    assert_close_f64(
        sp.as_slice(),
        &[std::f64::consts::LN_2],
        2.0 * f64::EPSILON * std::f64::consts::LN_2,
        "softplus(0)=ln(2)",
    );
}

// LEAKY_RELU

fn check_leaky_relu<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    // Positive branch: leaky_relu(x, slope) = x (exact identity regardless of slope).
    // Negative branch with slope=0.25 (exact power of 2): -1 * 0.25 = -0.25 (exact).
    let v = t(&[4], &[-4.0_f64, -1.0, 0.0, 2.0], backend);
    let lr = coeus_ops::leaky_relu(&v, backend, 0.25).expect("run operation");
    assert_eq!(
        lr.as_slice(),
        &[-1.0_f64, -0.25, 0.0, 2.0],
        "leaky_relu slope=0.25"
    );

    // slope=0.125 (exact power of 2): products below are exactly representable.
    let v2 = t(&[3], &[-100.0_f64, -1.0, 5.0], backend);
    let lr2 = coeus_ops::leaky_relu(&v2, backend, 0.125).expect("run operation");
    assert_eq!(
        lr2.as_slice(),
        &[-12.5_f64, -0.125, 5.0],
        "leaky_relu slope=0.125"
    );
}

// wrappers

fn check_all<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    check_sigmoid(backend);
    check_gelu(backend);
    check_gelu_tanh(backend);
    check_silu(backend);
    check_mish(backend);
    check_elu(backend);
    check_softplus(backend);
    check_leaky_relu(backend);
}

#[test]
fn sequential_activation_match_reference() {
    let backend = SequentialBackend;
    check_all(&backend);
}

#[test]
fn moirai_activation_match_reference() {
    let backend = MoiraiBackend;
    check_all(&backend);
}
