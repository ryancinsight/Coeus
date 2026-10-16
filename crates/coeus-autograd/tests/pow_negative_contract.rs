//! Closed-form negative integer power values and seeded derivatives.

use coeus_autograd::{pow, Var};
use coeus_core::{Float, FloatElement, SequentialBackend};
use coeus_tensor::Tensor;
use eunomia::{Bf16, F16};

fn negative_integer_contract<T: Float + FloatElement + std::ops::Neg<Output = T>>()
where
    SequentialBackend: coeus_ops::BackendOps<T>,
{
    // Logical inputs [-2, -1/2, 1/2, 2] and seed [2, -1, 1/2, -2].
    // Every value below is a representable dyadic in all four formats.
    // d(x^n)/dx = n / x^(1-n), including the sign of negative n.
    let cases = [
        (-1.0, [-0.5, -2.0, 2.0, 0.5], [-0.5, 4.0, -2.0, 0.5]),
        (-2.0, [0.25, 4.0, 4.0, 0.25], [0.5, -16.0, -8.0, 0.5]),
        (
            -3.0,
            [-0.125, -8.0, 8.0, 0.125],
            [-0.375, 48.0, -24.0, 0.375],
        ),
    ];
    for (exponent, values, gradients) in cases {
        for strided in [false, true] {
            for tracked in [false, true] {
                let storage = if strided {
                    [-2.0, 0.5, -0.5, 2.0]
                } else {
                    [-2.0, -0.5, 0.5, 2.0]
                }
                .map(<T as FloatElement>::from_f64);
                let tensor = Tensor::<T, SequentialBackend>::from_slice([2, 2], &storage)
                    .expect("fixture tensor");
                let tensor = if strided {
                    tensor.permute(&[1, 0])
                } else {
                    tensor
                };
                let input = Var::new(tensor, tracked).expect("fixture variable");
                let output = pow(&input, exponent).expect("negative integer power");
                assert_eq!(output.tensor.shape(), &[2, 2]);
                assert_eq!(
                    output.tensor.as_slice(),
                    &values.map(<T as FloatElement>::from_f64)
                );
                let seed = [2.0, -1.0, 0.5, -2.0].map(<T as FloatElement>::from_f64);
                output
                    .backward_with_seed(Tensor::from_slice([2, 2], &seed).expect("fixture seed"))
                    .expect("seeded power backward");
                if tracked {
                    let gradient = input.grad().expect("tracked input gradient");
                    assert_eq!(gradient.shape(), &[2, 2]);
                    assert_eq!(
                        gradient.as_slice(),
                        &gradients.map(<T as FloatElement>::from_f64),
                        "exponent {exponent}, strided {strided}"
                    );
                } else {
                    assert!(input.grad().is_none());
                }
            }
        }
    }
}

#[test]
fn negative_integer_power_native_single() {
    negative_integer_contract::<f32>();
}

#[test]
fn negative_integer_power_native_double() {
    negative_integer_contract::<f64>();
}

#[test]
fn negative_integer_power_half() {
    negative_integer_contract::<F16>();
}

#[test]
fn negative_integer_power_brain_half() {
    negative_integer_contract::<Bf16>();
}
