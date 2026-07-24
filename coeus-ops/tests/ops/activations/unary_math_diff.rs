//! Differential parity for element-wise unary math operations.
//!
//! Functions exercised:
//!   `abs`, `neg`, `sign`, `recip`, `sqrt`  - algebraic scalar ops
//!   `floor`, `ceil`, `round`, `trunc`      - rounding ops
//!   `exp`, `log`                           - exponential/logarithm (Float)
//!   `sin`, `cos`, `tanh`                   - transcendental (Float)
//!
//! Inputs are chosen so outputs are IEEE-exact, enabling `assert_eq!`:
//!   - sqrt(4.0)=2.0, recip(4.0)=0.25 (exact power of 2)
//!   - exp(0.0)=1.0, log(1.0)=0.0 (exact identities)
//!   - sin(0.0)=0.0, cos(0.0)=1.0, tanh(0.0)=0.0 (exact identities)
//!   - floor/ceil/round/trunc on integer-valued floats -> same integer
//!
//! SequentialBackend and MoiraiBackend must return bitwise-identical results.

use coeus_core::{
    CpuAddressableStorage, CpuAddressableStorageMut, MoiraiBackend, SequentialBackend,
};
use coeus_tensor::Tensor;

fn t<B>(shape: &[usize], vals: &[f64], backend: &B) -> Tensor<f64, B>
where
    B: coeus_core::ComputeBackend,
    B::DeviceBuffer<f64>: CpuAddressableStorageMut<f64>,
{
    Tensor::from_slice_on(shape.to_vec(), vals, backend)
}

// ABS / NEG / SIGN

fn check_abs_neg_sign<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    let v = t(&[4], &[-3.0, -1.0, 0.0, 2.0], backend);

    // abs: |-3|=3, |-1|=1, |0|=0, |2|=2
    let a = coeus_ops::abs(&v, backend).expect("valid abs test input");
    assert_eq!(a.as_slice(), &[3.0_f64, 1.0, 0.0, 2.0], "abs");

    // neg: 3, 1, 0, -2
    let n = coeus_ops::neg(&v, backend).expect("valid negation test input");
    assert_eq!(n.as_slice(), &[3.0_f64, 1.0, 0.0, -2.0], "neg");

    // sign: -1, -1, 0, 1
    let s = coeus_ops::sign(&v, backend).expect("valid sign test input");
    assert_eq!(s.as_slice(), &[-1.0_f64, -1.0, 0.0, 1.0], "sign");
}

// RECIP / SQRT

fn check_recip_sqrt<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    // recip: 1/1=1.0, 1/2=0.5, 1/4=0.25 - all exact powers of 2.
    let r = t(&[3], &[1.0, 2.0, 4.0], backend);
    let rec = coeus_ops::recip(&r, backend).expect("valid reciprocal test input");
    assert_eq!(rec.as_slice(), &[1.0_f64, 0.5, 0.25], "recip");

    // sqrt: sqrt(1)=1, sqrt(4)=2, sqrt(9)=3, sqrt(16)=4 - exact.
    let s = t(&[4], &[1.0, 4.0, 9.0, 16.0], backend);
    let sq = coeus_ops::sqrt(&s, backend).expect("valid square-root test input");
    assert_eq!(sq.as_slice(), &[1.0_f64, 2.0, 3.0, 4.0], "sqrt");
}

// FLOOR / CEIL / ROUND / TRUNC

fn check_rounding<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    // All inputs are exact integers -> floor/ceil/round/trunc all identity.
    let v = t(&[4], &[-3.0, -1.0, 0.0, 4.0], backend);

    let fl = coeus_ops::floor(&v, backend).expect("valid floor test input");
    assert_eq!(fl.as_slice(), v.as_slice(), "floor of integers");

    let ce = coeus_ops::ceil(&v, backend).expect("valid ceiling test input");
    assert_eq!(ce.as_slice(), v.as_slice(), "ceil of integers");

    let ro = coeus_ops::round(&v, backend).expect("valid round test input");
    assert_eq!(ro.as_slice(), v.as_slice(), "round of integers");

    // Half-way ties round to even (IEEE-754 roundTiesToEven, torch.round):
    // -2.5 -> -2, -1.5 -> -2, -0.5 -> -0, 0.5 -> 0, 1.5 -> 2, 2.5 -> 2.
    let ties = t(&[6], &[-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], backend);
    let ro_ties = coeus_ops::round(&ties, backend).expect("valid tie-rounding test input");
    assert_eq!(
        ro_ties.as_slice(),
        &[-2.0, -2.0, -0.0, 0.0, 2.0, 2.0],
        "round ties-to-even"
    );

    let tr = coeus_ops::trunc(&v, backend).expect("valid truncation test input");
    assert_eq!(tr.as_slice(), v.as_slice(), "trunc of integers");

    // Fractional inputs: floor rounds down, ceil rounds up, trunc truncates toward 0.
    // 1.5: floor=1, ceil=2, trunc=1
    // -1.5: floor=-2, ceil=-1, trunc=-1
    let frac = t(&[2], &[1.5, -1.5], backend);
    let fl2 = coeus_ops::floor(&frac, backend).expect("valid fractional floor test input");
    assert_eq!(fl2.as_slice(), &[1.0_f64, -2.0], "floor fractions");

    let ce2 = coeus_ops::ceil(&frac, backend).expect("valid fractional ceiling test input");
    assert_eq!(ce2.as_slice(), &[2.0_f64, -1.0], "ceil fractions");

    let tr2 = coeus_ops::trunc(&frac, backend).expect("valid fractional truncation test input");
    assert_eq!(tr2.as_slice(), &[1.0_f64, -1.0], "trunc fractions");
}

// EXP / LOG (Float ops)

fn check_exp_log<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    // exp(0.0) = 1.0 (exact IEEE-754 identity)
    let z = t(&[1], &[0.0], backend);
    let ez = coeus_ops::exp(&z, backend).expect("valid exponential test input");
    assert_eq!(ez.as_slice(), &[1.0_f64], "exp(0)=1");

    // log(1.0) = 0.0 (exact IEEE-754 identity)
    let o = t(&[1], &[1.0], backend);
    let lo = coeus_ops::log(&o, backend).expect("valid logarithm test input");
    assert_eq!(lo.as_slice(), &[0.0_f64], "log(1)=0");
}

// SIN / COS / TANH (Float ops, exact identity points)

fn check_transcendental<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    // sin(0.0) = 0.0 (exact)
    let z = t(&[1], &[0.0], backend);
    let s = coeus_ops::sin(&z, backend).expect("valid sine test input");
    assert_eq!(s.as_slice(), &[0.0_f64], "sin(0)=0");

    // cos(0.0) = 1.0 (exact)
    let c = coeus_ops::cos(&z, backend).expect("valid cosine test input");
    assert_eq!(c.as_slice(), &[1.0_f64], "cos(0)=1");

    // tanh(0.0) = 0.0 (exact)
    let t0 = coeus_ops::tanh(&z, backend).expect("valid hyperbolic tangent test input");
    assert_eq!(t0.as_slice(), &[0.0_f64], "tanh(0)=0");
}

// wrappers

fn check_all<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    check_abs_neg_sign(backend);
    check_recip_sqrt(backend);
    check_rounding(backend);
    check_exp_log(backend);
    check_transcendental(backend);
}

#[test]
fn sequential_unary_math_match_reference() {
    let backend = SequentialBackend;
    check_all(&backend);
}

#[test]
fn moirai_unary_math_match_reference() {
    let backend = MoiraiBackend;
    check_all(&backend);
}
