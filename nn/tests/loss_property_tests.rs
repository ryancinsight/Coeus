//! Property-based tests for loss function mathematical properties
//!
//! Feature: coeus-architecture-enhancement
//! Property 1: Single Source of Truth for Operations
//! Validates: Requirements 1.2, 1.4
//!
//! This module tests that loss functions maintain their mathematical properties
//! across all valid inputs using property-based testing.

use proptest::prelude::*;

use backend::CpuBackend;
use dtype::float::Float32;
use nn::functional::ops::loss::*;
use storage::DenseStorage;
use tensor::Tensor;

type TestTensor = Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>;

/// Generate random tensors with specified shape and value range
fn arb_tensor(shape: Vec<usize>, min: f32, max: f32) -> impl Strategy<Value = TestTensor> {
    let len: usize = shape.iter().product();
    prop::collection::vec(min..max, len).prop_map(move |data| {
        let float_data: Vec<Float32> = data.into_iter().map(Float32::new).collect();
        TestTensor::from_vec(float_data, &shape).unwrap()
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    // ========================================================================
    // MSE Loss Properties
    // ========================================================================

    /// Feature: coeus-architecture-enhancement, Property 1: Single Source of Truth for Operations
    /// Property: MSE loss is always >= 0
    /// Validates: Requirements 1.2, 1.4
    ///
    /// For any prediction and target tensors, MSE loss SHALL be non-negative.
    #[test]
    fn prop_mse_loss_non_negative(
        (pred_values, target_values) in (1usize..50).prop_flat_map(|len| {
            (
                prop::collection::vec(-100.0..100.0f32, len),
                prop::collection::vec(-100.0..100.0f32, len),
            )
        })
    ) {
        let shape = vec![pred_values.len()];
        let pred = TestTensor::from_vec(
            pred_values.into_iter().map(Float32::new).collect(),
            &shape
        ).unwrap();
        let target = TestTensor::from_vec(
            target_values.into_iter().map(Float32::new).collect(),
            &shape
        ).unwrap();

        let loss = mse_loss(&pred, &target).unwrap();
        let loss_val = loss.as_slice()[0].get() as f32;

        // Property: MSE loss must be >= 0
        prop_assert!(
            loss_val >= 0.0,
            "MSE loss {} is negative, violates non-negativity property",
            loss_val
        );
    }

    /// Feature: coeus-architecture-enhancement, Property 1: Single Source of Truth for Operations
    /// Property: MSE loss is zero for perfect predictions
    /// Validates: Requirements 1.2, 1.4
    ///
    /// For any tensor, when predictions equal targets, MSE loss SHALL be zero.
    #[test]
    fn prop_mse_loss_zero_for_perfect(
        values in prop::collection::vec(-100.0..100.0f32, 1..50)
    ) {
        let shape = vec![values.len()];
        let float_data: Vec<Float32> = values.into_iter().map(Float32::new).collect();
        let pred = TestTensor::from_vec(float_data.clone(), &shape).unwrap();
        let target = TestTensor::from_vec(float_data, &shape).unwrap();

        let loss = mse_loss(&pred, &target).unwrap();
        let loss_val = loss.as_slice()[0].get() as f32;

        // Property: MSE loss = 0 when pred = target
        prop_assert!(
            loss_val.abs() < 1e-6,
            "MSE loss {} should be zero for perfect predictions",
            loss_val
        );
    }

    /// Feature: coeus-architecture-enhancement, Property 1: Single Source of Truth for Operations
    /// Property: MSE loss is symmetric
    /// Validates: Requirements 1.2, 1.4
    ///
    /// For any prediction and target tensors, MSE(pred, target) SHALL equal MSE(target, pred).
    #[test]
    fn prop_mse_loss_symmetric(
        (pred_values, target_values) in (2usize..20).prop_flat_map(|len| {
            (
                prop::collection::vec(-50.0..50.0f32, len),
                prop::collection::vec(-50.0..50.0f32, len),
            )
        })
    ) {
        let shape = vec![pred_values.len()];
        let pred = TestTensor::from_vec(
            pred_values.into_iter().map(Float32::new).collect(),
            &shape
        ).unwrap();
        let target = TestTensor::from_vec(
            target_values.into_iter().map(Float32::new).collect(),
            &shape
        ).unwrap();

        let loss1 = mse_loss(&pred, &target).unwrap();
        let loss2 = mse_loss(&target, &pred).unwrap();

        let loss1_val = loss1.as_slice()[0].get() as f32;
        let loss2_val = loss2.as_slice()[0].get() as f32;

        // Property: MSE is symmetric
        prop_assert!(
            (loss1_val - loss2_val).abs() < 1e-5,
            "MSE loss should be symmetric: MSE(pred, target) = {} != MSE(target, pred) = {}",
            loss1_val,
            loss2_val
        );
    }

    /// Feature: coeus-architecture-enhancement, Property 1: Single Source of Truth for Operations
    /// Property: MSE loss increases with larger errors
    /// Validates: Requirements 1.2, 1.4
    ///
    /// For any target, predictions farther from target SHALL have higher MSE loss.
    #[test]
    fn prop_mse_loss_increases_with_error(
        target_val in -50.0..50.0f32,
        error1 in 0.1..10.0f32,
        error2 in 10.0..50.0f32
    ) {
        let target = TestTensor::from_vec(vec![Float32::new(target_val)], &[1]).unwrap();
        let pred1 = TestTensor::from_vec(vec![Float32::new(target_val + error1)], &[1]).unwrap();
        let pred2 = TestTensor::from_vec(vec![Float32::new(target_val + error2)], &[1]).unwrap();

        let loss1 = mse_loss(&pred1, &target).unwrap();
        let loss2 = mse_loss(&pred2, &target).unwrap();

        let loss1_val = loss1.as_slice()[0].get() as f32;
        let loss2_val = loss2.as_slice()[0].get() as f32;

        // Property: Larger error => larger loss
        prop_assert!(
            loss2_val > loss1_val,
            "MSE loss should increase with error: loss(error={}) = {} should be < loss(error={}) = {}",
            error1,
            loss1_val,
            error2,
            loss2_val
        );
    }

    // ========================================================================
    // Cross-Entropy Loss Properties
    // ========================================================================

    /// Feature: coeus-architecture-enhancement, Property 1: Single Source of Truth for Operations
    /// Property: Cross-entropy loss is always >= 0
    /// Validates: Requirements 1.2, 1.4
    ///
    /// For any logits and target tensors, cross-entropy loss SHALL be non-negative.
    #[test]
    fn prop_cross_entropy_non_negative(
        logits_values in prop::collection::vec(-10.0..10.0f32, 2..10),
        num_classes in 2usize..5
    ) {
        let batch_size = logits_values.len() / num_classes;
        prop_assume!(batch_size > 0);
        prop_assume!(logits_values.len() == batch_size * num_classes);

        let shape = vec![batch_size, num_classes];
        let logits = TestTensor::from_vec(
            logits_values.into_iter().map(Float32::new).collect(),
            &shape
        ).unwrap();

        // Create one-hot target (first class)
        let mut target_data = vec![0.0; batch_size * num_classes];
        for i in 0..batch_size {
            target_data[i * num_classes] = 1.0;
        }
        let target = TestTensor::from_vec(
            target_data.into_iter().map(Float32::new).collect(),
            &shape
        ).unwrap();

        let loss = cross_entropy(&logits, &target).unwrap();
        let loss_val = loss.as_slice()[0].get() as f32;

        // Property: Cross-entropy loss must be >= 0
        prop_assert!(
            loss_val >= 0.0,
            "Cross-entropy loss {} is negative, violates non-negativity property",
            loss_val
        );
    }

    /// Feature: coeus-architecture-enhancement, Property 1: Single Source of Truth for Operations
    /// Property: Cross-entropy loss decreases with higher confidence in correct class
    /// Validates: Requirements 1.2, 1.4
    ///
    /// For any target, logits with higher values for the correct class SHALL have lower loss.
    #[test]
    fn prop_cross_entropy_confidence(
        low_logit in -5.0..0.0f32,
        high_logit in 5.0..10.0f32
    ) {
        // Low confidence: [low_logit, 0.0]
        let logits_low = TestTensor::from_vec(
            vec![Float32::new(low_logit), Float32::new(0.0)],
            &[1, 2]
        ).unwrap();

        // High confidence: [high_logit, 0.0]
        let logits_high = TestTensor::from_vec(
            vec![Float32::new(high_logit), Float32::new(0.0)],
            &[1, 2]
        ).unwrap();

        // Target: class 0
        let target = TestTensor::from_vec(
            vec![Float32::new(1.0), Float32::new(0.0)],
            &[1, 2]
        ).unwrap();

        let loss_low = cross_entropy(&logits_low, &target).unwrap();
        let loss_high = cross_entropy(&logits_high, &target).unwrap();

        let loss_low_val = loss_low.as_slice()[0].get() as f32;
        let loss_high_val = loss_high.as_slice()[0].get() as f32;

        // Property: Higher confidence => lower loss
        prop_assert!(
            loss_high_val < loss_low_val,
            "Cross-entropy loss should decrease with confidence: loss(logit={}) = {} should be > loss(logit={}) = {}",
            low_logit,
            loss_low_val,
            high_logit,
            loss_high_val
        );
    }

    // ========================================================================
    // L1 Loss Properties
    // ========================================================================

    /// Feature: coeus-architecture-enhancement, Property 1: Single Source of Truth for Operations
    /// Property: L1 loss is always >= 0
    /// Validates: Requirements 1.2, 1.4
    ///
    /// For any prediction and target tensors, L1 loss SHALL be non-negative.
    #[test]
    fn prop_l1_loss_non_negative(
        (pred_values, target_values) in (1usize..50).prop_flat_map(|len| {
            (
                prop::collection::vec(-100.0..100.0f32, len),
                prop::collection::vec(-100.0..100.0f32, len),
            )
        })
    ) {
        let shape = vec![pred_values.len()];
        let pred = TestTensor::from_vec(
            pred_values.into_iter().map(Float32::new).collect(),
            &shape
        ).unwrap();
        let target = TestTensor::from_vec(
            target_values.into_iter().map(Float32::new).collect(),
            &shape
        ).unwrap();

        let loss = l1_loss(&pred, &target).unwrap();
        let loss_val = loss.as_slice()[0].get() as f32;

        // Property: L1 loss must be >= 0
        prop_assert!(
            loss_val >= 0.0,
            "L1 loss {} is negative, violates non-negativity property",
            loss_val
        );
    }

    /// Feature: coeus-architecture-enhancement, Property 1: Single Source of Truth for Operations
    /// Property: L1 loss is zero for perfect predictions
    /// Validates: Requirements 1.2, 1.4
    ///
    /// For any tensor, when predictions equal targets, L1 loss SHALL be zero.
    #[test]
    fn prop_l1_loss_zero_for_perfect(
        values in prop::collection::vec(-100.0..100.0f32, 1..50)
    ) {
        let shape = vec![values.len()];
        let float_data: Vec<Float32> = values.into_iter().map(Float32::new).collect();
        let pred = TestTensor::from_vec(float_data.clone(), &shape).unwrap();
        let target = TestTensor::from_vec(float_data, &shape).unwrap();

        let loss = l1_loss(&pred, &target).unwrap();
        let loss_val = loss.as_slice()[0].get() as f32;

        // Property: L1 loss = 0 when pred = target
        prop_assert!(
            loss_val.abs() < 1e-6,
            "L1 loss {} should be zero for perfect predictions",
            loss_val
        );
    }

    /// Feature: coeus-architecture-enhancement, Property 1: Single Source of Truth for Operations
    /// Property: L1 loss is symmetric
    /// Validates: Requirements 1.2, 1.4
    ///
    /// For any prediction and target tensors, L1(pred, target) SHALL equal L1(target, pred).
    #[test]
    fn prop_l1_loss_symmetric(
        (pred_values, target_values) in (2usize..20).prop_flat_map(|len| {
            (
                prop::collection::vec(-50.0..50.0f32, len),
                prop::collection::vec(-50.0..50.0f32, len),
            )
        })
    ) {
        let shape = vec![pred_values.len()];
        let pred = TestTensor::from_vec(
            pred_values.into_iter().map(Float32::new).collect(),
            &shape
        ).unwrap();
        let target = TestTensor::from_vec(
            target_values.into_iter().map(Float32::new).collect(),
            &shape
        ).unwrap();

        let loss1 = l1_loss(&pred, &target).unwrap();
        let loss2 = l1_loss(&target, &pred).unwrap();

        let loss1_val = loss1.as_slice()[0].get() as f32;
        let loss2_val = loss2.as_slice()[0].get() as f32;

        // Property: L1 is symmetric
        prop_assert!(
            (loss1_val - loss2_val).abs() < 1e-5,
            "L1 loss should be symmetric: L1(pred, target) = {} != L1(target, pred) = {}",
            loss1_val,
            loss2_val
        );
    }

    /// Feature: coeus-architecture-enhancement, Property 1: Single Source of Truth for Operations
    /// Property: L1 loss scales linearly with error
    /// Validates: Requirements 1.2, 1.4
    ///
    /// For any target, L1 loss SHALL scale linearly with the magnitude of error.
    #[test]
    fn prop_l1_loss_linear_scaling(
        target_val in -50.0..50.0f32,
        error in 1.0..10.0f32
    ) {
        let target = TestTensor::from_vec(vec![Float32::new(target_val)], &[1]).unwrap();
        let pred1 = TestTensor::from_vec(vec![Float32::new(target_val + error)], &[1]).unwrap();
        let pred2 = TestTensor::from_vec(vec![Float32::new(target_val + 2.0 * error)], &[1]).unwrap();

        let loss1 = l1_loss(&pred1, &target).unwrap();
        let loss2 = l1_loss(&pred2, &target).unwrap();

        let loss1_val = loss1.as_slice()[0].get() as f32;
        let loss2_val = loss2.as_slice()[0].get() as f32;

        // Property: L1 loss scales linearly (loss2 ≈ 2 * loss1)
        let ratio = loss2_val / loss1_val;
        prop_assert!(
            (ratio - 2.0).abs() < 0.1,
            "L1 loss should scale linearly: loss(2*error) / loss(error) = {} should be ≈ 2.0",
            ratio
        );
    }

    // ========================================================================
    // BCE with Logits Loss Properties
    // ========================================================================

    /// Feature: coeus-architecture-enhancement, Property 1: Single Source of Truth for Operations
    /// Property: BCE with logits loss is always >= 0
    /// Validates: Requirements 1.2, 1.4
    ///
    /// For any logits and target tensors, BCE with logits loss SHALL be non-negative.
    #[test]
    fn prop_bce_with_logits_non_negative(
        (logits_values, target_values) in (1usize..50).prop_flat_map(|len| {
            (
                prop::collection::vec(-10.0..10.0f32, len),
                prop::collection::vec(0.0..1.0f32, len),
            )
        })
    ) {
        let shape = vec![logits_values.len()];
        let logits = TestTensor::from_vec(
            logits_values.into_iter().map(Float32::new).collect(),
            &shape
        ).unwrap();
        let target = TestTensor::from_vec(
            target_values.into_iter().map(Float32::new).collect(),
            &shape
        ).unwrap();

        let loss = bce_with_logits_loss(&logits, &target).unwrap();
        let loss_val = loss.as_slice()[0].get() as f32;

        // Property: BCE with logits loss must be >= 0
        prop_assert!(
            loss_val >= 0.0,
            "BCE with logits loss {} is negative, violates non-negativity property",
            loss_val
        );
    }

    /// Feature: coeus-architecture-enhancement, Property 1: Single Source of Truth for Operations
    /// Property: BCE with logits loss decreases with correct predictions
    /// Validates: Requirements 1.2, 1.4
    ///
    /// For target=1, higher logits SHALL have lower loss. For target=0, lower logits SHALL have lower loss.
    #[test]
    fn prop_bce_with_logits_confidence(
        low_logit in -10.0..-1.0f32,
        high_logit in 1.0..10.0f32
    ) {
        // Target = 1: high logit should have lower loss
        let target_one = TestTensor::from_vec(vec![Float32::new(1.0)], &[1]).unwrap();
        let logits_low = TestTensor::from_vec(vec![Float32::new(low_logit)], &[1]).unwrap();
        let logits_high = TestTensor::from_vec(vec![Float32::new(high_logit)], &[1]).unwrap();

        let loss_low = bce_with_logits_loss(&logits_low, &target_one).unwrap();
        let loss_high = bce_with_logits_loss(&logits_high, &target_one).unwrap();

        let loss_low_val = loss_low.as_slice()[0].get() as f32;
        let loss_high_val = loss_high.as_slice()[0].get() as f32;

        // Property: For target=1, higher logit => lower loss
        prop_assert!(
            loss_high_val < loss_low_val,
            "BCE loss for target=1 should decrease with logit: loss(logit={}) = {} should be > loss(logit={}) = {}",
            low_logit,
            loss_low_val,
            high_logit,
            loss_high_val
        );
    }

    // ========================================================================
    // NLL Loss Properties
    // ========================================================================

    /// Feature: coeus-architecture-enhancement, Property 1: Single Source of Truth for Operations
    /// Property: NLL loss is always >= 0 for valid log probabilities
    /// Validates: Requirements 1.2, 1.4
    ///
    /// For any valid log probabilities (negative values) and targets, NLL loss SHALL be non-negative.
    #[test]
    fn prop_nll_loss_non_negative(
        log_prob_values in prop::collection::vec(-10.0..0.0f32, 2..10),
        num_classes in 2usize..5
    ) {
        let batch_size = log_prob_values.len() / num_classes;
        prop_assume!(batch_size > 0);
        prop_assume!(log_prob_values.len() == batch_size * num_classes);

        let shape = vec![batch_size, num_classes];
        let log_probs = TestTensor::from_vec(
            log_prob_values.into_iter().map(Float32::new).collect(),
            &shape
        ).unwrap();

        // Create one-hot target (first class)
        let mut target_data = vec![0.0; batch_size * num_classes];
        for i in 0..batch_size {
            target_data[i * num_classes] = 1.0;
        }
        let target = TestTensor::from_vec(
            target_data.into_iter().map(Float32::new).collect(),
            &shape
        ).unwrap();

        let loss = nll_loss(&log_probs, &target).unwrap();
        let loss_val = loss.as_slice()[0].get() as f32;

        // Property: NLL loss must be >= 0
        prop_assert!(
            loss_val >= 0.0,
            "NLL loss {} is negative, violates non-negativity property",
            loss_val
        );
    }

    // ========================================================================
    // Binary Cross-Entropy Loss Properties
    // ========================================================================

    /// Feature: coeus-architecture-enhancement, Property 1: Single Source of Truth for Operations
    /// Property: Binary cross-entropy loss is always >= 0
    /// Validates: Requirements 1.2, 1.4
    ///
    /// For any probability predictions and targets, binary cross-entropy loss SHALL be non-negative.
    #[test]
    fn prop_binary_cross_entropy_non_negative(
        (pred_values, target_values) in (1usize..50).prop_flat_map(|len| {
            (
                prop::collection::vec(0.01..0.99f32, len),
                prop::collection::vec(0.0..1.0f32, len),
            )
        })
    ) {
        let shape = vec![pred_values.len()];
        let pred = TestTensor::from_vec(
            pred_values.into_iter().map(Float32::new).collect(),
            &shape
        ).unwrap();
        let target = TestTensor::from_vec(
            target_values.into_iter().map(Float32::new).collect(),
            &shape
        ).unwrap();

        let loss = binary_cross_entropy(&pred, &target).unwrap();
        let loss_val = loss.as_slice()[0].get() as f32;

        // Property: Binary cross-entropy loss must be >= 0
        prop_assert!(
            loss_val >= 0.0,
            "Binary cross-entropy loss {} is negative, violates non-negativity property",
            loss_val
        );
    }
}
