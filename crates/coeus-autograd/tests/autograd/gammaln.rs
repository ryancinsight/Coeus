//! Provider-chain acceptance for the gamma-log derivative.
use coeus_autograd::{add, broadcast_to, lgamma, lgamma_forward, permute, slice, Var};
use coeus_core::{Float, SequentialBackend};
use coeus_tensor::Tensor;
use eunomia::{FloatElement, NumericElement};

#[path = "gammaln/bounds.rs"]
mod bounds;

fn close(actual: f64, expected: f64, bound: f64) {
    let _assert = tracing::info_span!("assert", actual, expected).entered();
    assert!(
        (actual - expected).abs() <= bound * (1.0 + expected.abs()),
        "actual={actual:.17e} expected={expected:.17e}"
    );
}

fn trace() -> tracing::subscriber::DefaultGuard {
    tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_test_writer()
            .with_span_events(tracing_subscriber::fmt::format::FmtSpan::FULL)
            .finish(),
    )
}

fn weighted<T: Float + FloatElement + leto_ops::RealScalar>(
    bound: f64,
    storage_unit: f64,
    evaluation_unit: f64,
    absolute: [f64; 4],
) {
    let arrange = tracing::info_span!("arrange", scalar = core::any::type_name::<T>()).entered();
    let data = [0.5, 1.0, 2.0, -0.5];
    let weights = [2.0, -3.0, 0.25, -2.0];
    let input: Vec<T> = data.into_iter().map(T::from_f64).collect();
    let seed: Vec<T> = weights.into_iter().map(T::from_f64).collect();
    let x = Var::new(
        Tensor::<T, SequentialBackend>::from_slice([4], &input).unwrap(),
        true,
    )
    .unwrap();
    drop(arrange);
    let _act = tracing::info_span!("act").entered();
    let y = lgamma(&x).unwrap();
    y.backward_with_seed(Tensor::from_slice([4], &seed).unwrap())
        .unwrap();
    let gamma = 0.577_215_664_901_532_9;
    let half = -gamma - 2.0 * core::f64::consts::LN_2;
    let gradients = [half, -gamma, 1.0 - gamma, half + 2.0];
    for ((actual, derivative), weight) in x
        .grad()
        .unwrap()
        .as_slice()
        .iter()
        .zip(gradients)
        .zip(weights)
    {
        close(NumericElement::to_f64(*actual), derivative * weight, bound);
    }
    y.backward_with_seed(Tensor::from_slice([4], &seed).unwrap())
        .unwrap();
    for ((actual, derivative), weight) in x
        .grad()
        .unwrap()
        .as_slice()
        .iter()
        .zip(gradients)
        .zip(weights)
    {
        close(
            NumericElement::to_f64(*actual),
            2.0 * derivative * weight,
            2.0 * bound,
        );
    }
    // Independent primary acceptance: scalar enclosure, at most one result
    // narrowing and one multiplication, two-pass accumulation and reference
    // anchor rounding. Every positive budget operation rounds outward.
    let up = |value: f64| value.next_up();
    let factor = up(up(1.0 + storage_unit) * up(1.0 + storage_unit));
    let scalar_relative = up(2.0 * evaluation_unit / (1.0 - evaluation_unit));
    let relative = up(up(up(factor * up(1.0 + scalar_relative)) - 1.0) / (1.0 - f64::EPSILON));
    let values = x.grad().unwrap();
    for i in 0..4 {
        let expected = 2.0 * gradients[i] * weights[i];
        let scalar_error = up(up(2.0 * weights[i].abs()) * absolute[i]);
        let anchor_error = up(8.0 * f64::EPSILON * weights[i].abs() * (1.0 + gradients[i].abs()));
        let limit =
            up(up(up(factor * scalar_error) + up(relative * expected.abs())) + anchor_error);
        assert!((NumericElement::to_f64(values.as_slice()[i]) - expected).abs() <= limit);
    }
    // The new operation preserves the established forward path exactly.
    let forward = lgamma_forward(&x).unwrap();
    assert_eq!(y.tensor.as_slice(), forward.as_slice());
}

#[test]
fn gammaln_native_and_reduced_cpu_gradients_use_nonunit_seed() {
    let _trace = trace();
    weighted::<f64>(
        128.0 * f64::EPSILON / 2.0,
        f64::EPSILON / 2.0,
        f64::EPSILON / 2.0,
        bounds::DOUBLE,
    );
    weighted::<f32>(
        128.0 * f64::from(f32::EPSILON) / 2.0,
        f64::from(f32::EPSILON) / 2.0,
        f64::from(f32::EPSILON) / 2.0,
        bounds::SINGLE,
    );
    // Narrowed derivative plus one target-format multiplication: <=2u relative
    // in addition to the provider's single-precision error envelope.
    weighted::<eunomia::F16>(
        2.0 * 2.0_f64.powi(-11) + 128.0 * f64::from(f32::EPSILON) / 2.0,
        2.0_f64.powi(-11),
        f64::from(f32::EPSILON) / 2.0,
        bounds::SINGLE,
    );
    weighted::<eunomia::Bf16>(
        2.0 * 2.0_f64.powi(-8) + 128.0 * f64::from(f32::EPSILON) / 2.0,
        2.0_f64.powi(-8),
        f64::from(f32::EPSILON) / 2.0,
        bounds::SINGLE,
    );
}

#[test]
fn gammaln_shared_accumulation_and_transposed_layout() {
    let _trace = trace();
    let arrange = tracing::info_span!("arrange").entered();
    let x = Var::new(
        Tensor::<f64, SequentialBackend>::from_slice([2, 2], &[0.5, 1.0, 2.0, -0.5]).unwrap(),
        true,
    )
    .unwrap();
    let transposed = permute(&x, &[1, 0]).unwrap();
    drop(arrange);
    let _act = tracing::info_span!("act").entered();
    let a = lgamma(&transposed).unwrap();
    let b = lgamma(&transposed).unwrap();
    let y = add(&a, &b).unwrap();
    y.backward_with_seed(Tensor::from_slice([2, 2], &[2.0, 3.0, -4.0, 0.5]).unwrap())
        .unwrap();
    let gamma = 0.577_215_664_901_532_9;
    let half = -gamma - 2.0 * core::f64::consts::LN_2;
    for (actual, expected) in x.grad().unwrap().as_slice().iter().zip([
        4.0 * half,
        8.0 * gamma,
        6.0 * (1.0 - gamma),
        half + 2.0,
    ]) {
        close(*actual, expected, 256.0 * f64::EPSILON / 2.0);
    }
}

#[test]
fn gammaln_sliced_offset_and_broadcast_gradients() {
    let _trace = trace();
    let arrange = tracing::info_span!("arrange").entered();
    let x = Var::new(
        Tensor::<f64, SequentialBackend>::from_slice([2, 3], &[9.0, 0.5, 1.0, 9.0, 2.0, -0.5])
            .unwrap(),
        true,
    )
    .unwrap();
    let selected = slice(&x, &[(0, 2), (1, 3)]).unwrap();
    drop(arrange);
    let _act = tracing::info_span!("act").entered();
    lgamma(&selected).unwrap().backward().unwrap();
    let gamma = 0.577_215_664_901_532_9;
    let half = -gamma - 2.0 * core::f64::consts::LN_2;
    for (actual, expected) in
        x.grad()
            .unwrap()
            .as_slice()
            .iter()
            .zip([0.0, half, -gamma, 0.0, 1.0 - gamma, half + 2.0])
    {
        close(*actual, expected, 128.0 * f64::EPSILON / 2.0);
    }
    let x = Var::new(
        Tensor::<f64, SequentialBackend>::from_slice([1, 2], &[0.5, 2.0]).unwrap(),
        true,
    )
    .unwrap();
    let expanded = broadcast_to(&x, [3, 2]).unwrap();
    lgamma(&expanded).unwrap().backward().unwrap();
    for (actual, expected) in x
        .grad()
        .unwrap()
        .as_slice()
        .iter()
        .zip([3.0 * half, 3.0 * (1.0 - gamma)])
    {
        close(*actual, expected, 256.0 * f64::EPSILON / 2.0);
    }
}

#[test]
fn gammaln_scalar_and_empty_layouts() {
    let _trace = trace();
    let arrange = tracing::info_span!("arrange").entered();
    let scalar = Var::new(
        Tensor::<f64, SequentialBackend>::from_slice([], &[0.5]).unwrap(),
        true,
    )
    .unwrap();
    drop(arrange);
    let _act = tracing::info_span!("act").entered();
    let y = lgamma(&scalar).unwrap();
    y.backward_with_seed(Tensor::from_slice([], &[-2.5]).unwrap())
        .unwrap();
    close(
        scalar.grad().unwrap().as_slice()[0],
        -2.5 * (-0.577_215_664_901_532_9 - 2.0 * core::f64::consts::LN_2),
        128.0 * f64::EPSILON,
    );
    assert_eq!(y.tensor.shape(), &[]);
    let empty = Var::new(
        Tensor::<f64, SequentialBackend>::from_slice([0, 3], &[]).unwrap(),
        true,
    )
    .unwrap();
    let y = lgamma(&empty).unwrap();
    y.backward().unwrap();
    assert_eq!(y.tensor.shape(), &[0, 3]);
    assert!(empty.grad().unwrap().as_slice().is_empty());
}
