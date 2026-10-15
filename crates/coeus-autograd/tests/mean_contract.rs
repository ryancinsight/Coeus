//! Analytical mean contracts: native float values and the 2^-16 half boundary.

use coeus_autograd::{mean, scalar_mul, Var};
use coeus_core::{FloatElement, Scalar, SequentialBackend};
use coeus_tensor::Tensor;
use eunomia::{Bf16, F16};

type B = SequentialBackend;

fn ordinary_case<T: Scalar + FloatElement + leto_ops::Scalar>()
where
    B: coeus_ops::BackendOps<T>,
{
    // Sum = 3, mean = 3/4. With seed 2, each derivative is 2/4.
    let values = [-3.0, 0.5, 1.5, 4.0].map(<T as FloatElement>::from_f64);
    for tracked in [false, true] {
        let tensor = Tensor::<T, B>::from_slice([2, 2], &values).expect("fixture allocation");
        let input = Var::new(tensor, tracked).expect("fixture variable");
        let output = mean(&input).expect("float mean");
        assert_eq!(output.tensor.shape(), &[1]);
        assert_eq!(
            output.tensor.as_slice(),
            &[<T as FloatElement>::from_f64(0.75)]
        );
        let loss = scalar_mul(&output, <T as FloatElement>::from_f64(2.0)).expect("weighted mean");
        loss.backward().expect("mean backward");
        if tracked {
            assert_eq!(
                input.grad().expect("tracked gradient").as_slice(),
                &[<T as FloatElement>::from_f64(0.5); 4]
            );
        } else {
            assert!(input.grad().is_none());
        }
    }
}

#[test]
fn ordinary_float_forward_and_backward() {
    ordinary_case::<f32>();
    ordinary_case::<f64>();
    ordinary_case::<F16>();
    ordinary_case::<Bf16>();
}

#[test]
fn half_mean_preserves_representable_reciprocal_for_large_count() {
    // A one-hot vector sums exactly to 1 in binary16. Its mean is exactly
    // 2^-16 = 256 * 2^-24, bits 0x0100, even though 65536 overflows binary16.
    let mut values = vec![F16::ZERO; 65_536];
    values[32_768] = F16::ONE;
    let input = Var::new(
        Tensor::<F16, B>::from_slice([values.len()], &values).expect("fixture allocation"),
        true,
    )
    .expect("fixture variable");
    let output = mean(&input).expect("half mean");
    assert_eq!(output.tensor.as_slice()[0].to_bits(), 0x0100);
    output.backward().expect("half mean backward");
    let gradient = input.grad().expect("tracked gradient");
    assert_eq!(gradient.shape(), &[65_536]);
    assert!(gradient
        .as_slice()
        .iter()
        .all(|value| value.to_bits() == 0x0100));
}

#[test]
fn mean_respects_strided_logical_elements() {
    // A transpose changes physical traversal but retains sum 10 / count 4.
    let tensor = Tensor::<f64, B>::from_slice([2, 2], &[1.0, 2.0, 3.0, 4.0])
        .expect("fixture allocation")
        .permute(&[1, 0]);
    let input = Var::new(tensor, true).expect("fixture variable");
    let output = mean(&input).expect("strided mean");
    assert_eq!(output.tensor.as_slice(), &[2.5]);
    output.backward().expect("strided mean backward");
    assert_eq!(
        input.grad().expect("tracked gradient").as_slice(),
        &[0.25; 4]
    );
}

#[test]
fn empty_float_mean_is_nan() {
    let input = Var::new(
        Tensor::<f64, B>::from_slice([0], &[]).expect("empty fixture allocation"),
        false,
    )
    .expect("fixture variable");
    let output = mean(&input).expect("empty float mean");
    assert!(output.tensor.as_slice()[0].is_nan());
}
