//! Property-based tests for activation function mathematical properties
//!
//! Feature: coeus-architecture-enhancement
//! Property 1: Single Source of Truth for Operations
//! Validates: Requirements 1.2, 1.4
//!
//! This module tests that activation functions maintain their mathematical properties
//! across all valid inputs using property-based testing.

use proptest::prelude::*;

use backend::CpuBackend;
use dtype::float::Float32;
use nn::functional::ops::activations::*;
use storage::DenseStorage;
use tensor::Tensor;

type TestTensor = Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>;

#[test]
fn prop_relu_output_non_negative() {
    proptest!(|(values in prop::collection::vec(-1000.0..1000.0f32, 1..100))| {
        let float_data: Vec<Float32> = values.iter().map(|&x| Float32::new(x)).collect();
        let shape = vec![float_data.len()];
        let tensor = TestTensor::from_vec(float_data, &shape).unwrap();

        let output = relu(&tensor).unwrap();

        for &val in output.as_slice() {
            prop_assert!(
                val.get() >= 0.0,
                "ReLU output {} is negative, violates non-negativity property",
                val.get()
            );
        }
    });
}

#[test]
fn prop_relu_correctness() {
    proptest!(|(values in prop::collection::vec(-1000.0..1000.0f32, 1..100))| {
        let float_data: Vec<Float32> = values.iter().map(|&x| Float32::new(x)).collect();
        let shape = vec![float_data.len()];
        let tensor = TestTensor::from_vec(float_data.clone(), &shape).unwrap();

        let output = relu(&tensor).unwrap();

        for (&input, &output_val) in float_data.iter().zip(output.as_slice()) {
            if input.get() > 0.0 {
                prop_assert_eq!(
                    output_val.get(),
                    input.get(),
                    "ReLU should preserve positive value {} but got {}",
                    input.get(),
                    output_val.get()
                );
            } else {
                prop_assert_eq!(
                    output_val.get(),
                    0.0,
                    "ReLU should zero negative/zero value {} but got {}",
                    input.get(),
                    output_val.get()
                );
            }
        }
    });
}

#[test]
fn prop_sigmoid_output_range() {
    proptest!(|(values in prop::collection::vec(-100.0..100.0f32, 1..100))| {
        let float_data: Vec<Float32> = values.into_iter().map(Float32::new).collect();
        let shape = vec![float_data.len()];
        let tensor = TestTensor::from_vec(float_data, &shape).unwrap();

        let output = sigmoid(&tensor).unwrap();

        for &val in output.as_slice() {
            let v = val.get();
            prop_assert!(
                v >= -1e-6 && v <= 1.0 + 1e-6,
                "Sigmoid output {} is not in [0, 1] within tolerance, violates range property",
                v
            );
        }
    });
}

#[test]
fn prop_sigmoid_monotonic() {
    proptest!(|(x1 in -50.0..50.0f32, x2 in -50.0..50.0f32)| {
        prop_assume!(x1 < x2);

        let tensor1 = TestTensor::from_vec(vec![Float32::new(x1)], &[1]).unwrap();
        let tensor2 = TestTensor::from_vec(vec![Float32::new(x2)], &[1]).unwrap();

        let output1 = sigmoid(&tensor1).unwrap();
        let output2 = sigmoid(&tensor2).unwrap();

        // Property: sigmoid is monotonically increasing
        prop_assert!(
            output1.as_slice()[0].get() <= output2.as_slice()[0].get(),
            "Sigmoid should be monotonic: sigmoid({}) = {} should be <= sigmoid({}) = {}",
            x1,
            output1.as_slice()[0].get(),
            x2,
            output2.as_slice()[0].get()
        );
    });
}

#[test]
fn prop_tanh_output_range() {
    proptest!(|(values in prop::collection::vec(-100.0..100.0f32, 1..100))| {
        let float_data: Vec<Float32> = values.into_iter().map(Float32::new).collect();
        let shape = vec![float_data.len()];
        let tensor = TestTensor::from_vec(float_data, &shape).unwrap();

        let output = tanh(&tensor).unwrap();

        for &val in output.as_slice() {
            let v = val.get();
            prop_assert!(
                v >= -1.0 - 1e-6 && v <= 1.0 + 1e-6,
                "Tanh output {} is not in [-1, 1] within tolerance, violates range property",
                v
            );
        }
    });
}

#[test]
fn prop_tanh_odd_function() {
    proptest!(|(x in -50.0..50.0f32)| {
        let tensor_pos = TestTensor::from_vec(vec![Float32::new(x)], &[1]).unwrap();
        let tensor_neg = TestTensor::from_vec(vec![Float32::new(-x)], &[1]).unwrap();

        let output_pos = tanh(&tensor_pos).unwrap();
        let output_neg = tanh(&tensor_neg).unwrap();

        // Property: tanh is an odd function
        let pos_val = output_pos.as_slice()[0].get();
        let neg_val = output_neg.as_slice()[0].get();

        prop_assert!(
            (pos_val + neg_val).abs() < 1e-6,
            "Tanh should be odd: tanh({}) = {} should equal -tanh({}) = {} (got {})",
            x,
            pos_val,
            -x,
            -pos_val,
            neg_val
        );
    });
}

#[test]
fn prop_tanh_monotonic() {
    proptest!(|(x1 in -50.0..50.0f32, x2 in -50.0..50.0f32)| {
        prop_assume!(x1 < x2);

        let tensor1 = TestTensor::from_vec(vec![Float32::new(x1)], &[1]).unwrap();
        let tensor2 = TestTensor::from_vec(vec![Float32::new(x2)], &[1]).unwrap();

        let output1 = tanh(&tensor1).unwrap();
        let output2 = tanh(&tensor2).unwrap();

        // Property: tanh is monotonically increasing
        prop_assert!(
            output1.as_slice()[0].get() <= output2.as_slice()[0].get(),
            "Tanh should be monotonic: tanh({}) = {} should be <= tanh({}) = {}",
            x1,
            output1.as_slice()[0].get(),
            x2,
            output2.as_slice()[0].get()
        );
    });
}

#[test]
fn prop_gelu_large_positive() {
    proptest!(|(x in 3.0..100.0f32)| {
        let tensor = TestTensor::from_vec(vec![Float32::new(x)], &[1]).unwrap();
        let output = gelu(&tensor).unwrap();
        let output_val = output.as_slice()[0].get();

        let ratio = output_val / x;
        prop_assert!(
            ratio > 0.95 && ratio < 1.05,
            "GELU({}) = {} should be approximately equal to {} (ratio: {})",
            x,
            output_val,
            x,
            ratio
        );
    });
}

#[test]
fn prop_gelu_large_negative() {
    proptest!(|(x in -100.0..-3.0f32)| {
        let tensor = TestTensor::from_vec(vec![Float32::new(x)], &[1]).unwrap();
        let output = gelu(&tensor).unwrap();
        let output_val = output.as_slice()[0].get();

        prop_assert!(
            output_val.abs() < 0.1,
            "GELU({}) = {} should be approximately zero",
            x,
            output_val
        );
    });
}

#[test]
fn prop_silu_zero() {
    let tensor = TestTensor::from_vec(vec![Float32::new(0.0)], &[1]).unwrap();
    let output = silu(&tensor).unwrap();
    let output_val = output.as_slice()[0].get();

    assert!(
        output_val.abs() < 1e-6,
        "SiLU(0) should be 0, got {}",
        output_val
    );
}

#[test]
fn prop_leaky_relu_positive() {
    proptest!(|(x in 0.0..100.0f32, slope in 0.0..1.0f32)| {
        let tensor = TestTensor::from_vec(vec![Float32::new(x)], &[1]).unwrap();
        let output = leaky_relu(&tensor, Float32::new(slope)).unwrap();
        let output_val = output.as_slice()[0].get();

        prop_assert!(
            (output_val - x).abs() < 1e-6,
            "LeakyReLU({}) should equal {} but got {}",
            x,
            x,
            output_val
        );
    });
}

#[test]
fn prop_leaky_relu_negative() {
    proptest!(|(x in -100.0..0.0f32, slope in 0.0..1.0f32)| {
        let tensor = TestTensor::from_vec(vec![Float32::new(x)], &[1]).unwrap();
        let output = leaky_relu(&tensor, Float32::new(slope)).unwrap();
        let output_val = output.as_slice()[0].get();

        let expected = slope * x;
        prop_assert!(
            (output_val - expected).abs() < 1e-5,
            "LeakyReLU({}) with slope {} should equal {} but got {}",
            x,
            slope,
            expected,
            output_val
        );
    });
}

#[test]
fn prop_elu_positive() {
    proptest!(|(x in 0.0..100.0f32, alpha in 0.1..2.0f32)| {
        let tensor = TestTensor::from_vec(vec![Float32::new(x)], &[1]).unwrap();
        let output = elu(&tensor, Float32::new(alpha)).unwrap();
        let output_val = output.as_slice()[0].get();

        prop_assert!(
            (output_val - x).abs() < 1e-6,
            "ELU({}) should equal {} but got {}",
            x,
            x,
            output_val
        );
    });
}

#[test]
fn prop_elu_negative_bounded() {
    proptest!(|(x in -100.0..0.0f32, alpha in 0.1..2.0f32)| {
        let tensor = TestTensor::from_vec(vec![Float32::new(x)], &[1]).unwrap();
        let output = elu(&tensor, Float32::new(alpha)).unwrap();
        let output_val = output.as_slice()[0].get();

        prop_assert!(
            output_val >= -alpha - 1e-6,
            "ELU({}) = {} should be >= -alpha ({}) within tolerance",
            x,
            output_val,
            -alpha
        );
    });
}

#[test]
fn prop_softmax_sums_to_one() {
    proptest!(|(values in prop::collection::vec(-10.0..10.0f32, 2..50))| {
        let float_data: Vec<Float32> = values.into_iter().map(Float32::new).collect();
        let shape = vec![float_data.len()];
        let tensor = TestTensor::from_vec(float_data, &shape).unwrap();

        let output = softmax(&tensor).unwrap();

        let sum: f32 = output.as_slice().iter().map(|x| x.get()).sum();
        prop_assert!(
            (sum - 1.0).abs() < 1e-5,
            "Softmax outputs should sum to 1, got {}",
            sum
        );
    });
}

#[test]
fn prop_softmax_all_positive() {
    proptest!(|(values in prop::collection::vec(-10.0..10.0f32, 2..50))| {
        let float_data: Vec<Float32> = values.into_iter().map(Float32::new).collect();
        let shape = vec![float_data.len()];
        let tensor = TestTensor::from_vec(float_data, &shape).unwrap();

        let output = softmax(&tensor).unwrap();

        for &val in output.as_slice() {
            prop_assert!(
                val.get() > 0.0,
                "Softmax output {} should be positive",
                val.get()
            );
        }
    });
}

#[test]
fn prop_log_softmax_all_negative() {
    proptest!(|(values in prop::collection::vec(-10.0..10.0f32, 2..50))| {
        let float_data: Vec<Float32> = values.into_iter().map(Float32::new).collect();
        let shape = vec![float_data.len()];
        let tensor = TestTensor::from_vec(float_data, &shape).unwrap();

        let output = log_softmax(&tensor, 0).unwrap();

        for &val in output.as_slice() {
            prop_assert!(
                val.get() <= 1e-6,
                "Log softmax output {} should be <= 0 within tolerance",
                val.get()
            );
        }
    });
}

#[test]
fn prop_log_softmax_exp_equals_softmax() {
    proptest!(|(values in prop::collection::vec(-10.0..10.0f32, 2..20))| {
        let float_data: Vec<Float32> = values.into_iter().map(Float32::new).collect();
        let shape = vec![float_data.len()];
        let tensor = TestTensor::from_vec(float_data, &shape).unwrap();

        let softmax_output = softmax(&tensor).unwrap();
        let log_softmax_output = log_softmax(&tensor, 0).unwrap();

        for (soft_val, log_soft_val) in softmax_output.as_slice().iter().zip(log_softmax_output.as_slice()) {
            let exp_log_soft = log_soft_val.get().exp();
            prop_assert!(
                (exp_log_soft - soft_val.get()).abs() < 1e-5,
                "exp(log_softmax) = {} should equal softmax = {}",
                exp_log_soft,
                soft_val.get()
            );
        }
    });
}
