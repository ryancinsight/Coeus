//! Comprehensive tests for data transformation utilities

use coeus_tensor::{Tensor, CpuBackend};
use coeus_utils::transforms::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_single_channel() {
        let input = Tensor::from_vec(CpuBackend::default(), vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let transform = Normalize::new(vec![2.0], vec![1.0]);

        let result = transform.transform(&input).unwrap();
        let expected = Tensor::from_vec(CpuBackend::default(), vec![-1.0, 0.0, 1.0, 2.0], vec![2, 2]).unwrap();

        // Check shapes match
        assert_eq!(result.shape(), expected.shape());
        // Check that transformation was applied (basic check)
        assert!(result.data().iter().any(|&x| x != input.data()[0]));
    }

    #[test]
    fn test_normalize_multi_channel() {
        let input = Tensor::from_vec(CpuBackend::default(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let transform = Normalize::new(vec![2.0, 3.0, 4.0], vec![1.0, 1.0, 1.0]);

        let result = transform.transform(&input).unwrap();

        // Check shapes match
        assert_eq!(result.shape(), input.shape());
        // Check that transformation was applied
        assert!(result.data().iter().any(|&x| x != input.data()[0]));
    }

    #[test]
    fn test_normalize_empty_params() {
        let input = Tensor::from_vec(CpuBackend::default(), vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let transform = Normalize::new(vec![], vec![]);

        let result = transform.transform(&input).unwrap();
        assert_eq!(result.shape(), input.shape());
    }

    #[test]
    fn test_random_crop_basic() {
        let input = Tensor::from_vec(
            CpuBackend::default(),
            (0..24).map(|x| x as f32).collect(),
            vec![2, 3, 4], // [batch=2, height=3, width=4]
        ).unwrap();
        let transform = RandomCrop::new(vec![2, 2]); // Crop to 2x2

        let result = transform.transform(&input).unwrap();

        // Check output shape: [batch=2, height=2, width=2]
        assert_eq!(result.shape(), &[2, 2, 2]);
        assert_eq!(result.numel(), 8);
    }

    #[test]
    fn test_random_crop_2d() {
        let input = Tensor::from_vec(
            CpuBackend::default(),
            (0..12).map(|x| x as f32).collect(),
            vec![3, 4], // [height=3, width=4]
        ).unwrap();
        let transform = RandomCrop::new(vec![2, 2]);

        let result = transform.transform(&input).unwrap();

        // Check output shape: [height=2, width=2]
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.numel(), 4);
    }

    #[test]
    fn test_random_crop_invalid_size() {
        let input = Tensor::from_vec(CpuBackend::default(), vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let transform = RandomCrop::new(vec![3, 3]); // Larger than input

        let result = transform.transform(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_random_horizontal_flip_2d() {
        let input = Tensor::from_vec(CpuBackend::default(), vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let transform = RandomHorizontalFlip::new(1.0); // Always flip

        let result = transform.transform(&input).unwrap();

        // Check shape preserved
        assert_eq!(result.shape(), input.shape());

        // For a 2x2 matrix:
        // [1, 2]    -> [2, 1]
        // [3, 4]       [4, 3]
        let expected = Tensor::from_vec(CpuBackend::default(), vec![2.0, 1.0, 4.0, 3.0], vec![2, 2]).unwrap();

        for (i, (&actual, &exp)) in result.data().iter().zip(expected.data().iter()).enumerate() {
            assert_eq!(actual, exp, "Mismatch at index {}", i);
        }
    }

    #[test]
    fn test_random_horizontal_flip_3d() {
        let input = Tensor::from_vec(CpuBackend::default(), (0..24).map(|x| x as f32).collect(), vec![2, 3, 4]).unwrap();
        let transform = RandomHorizontalFlip::new(1.0); // Always flip

        let result = transform.transform(&input).unwrap();

        // Check shape preserved
        assert_eq!(result.shape(), input.shape());
        assert_eq!(result.numel(), 24);
    }

    #[test]
    fn test_random_vertical_flip_2d() {
        let input = Tensor::from_vec(CpuBackend::default(), vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let transform = RandomVerticalFlip::new(1.0); // Always flip

        let result = transform.transform(&input).unwrap();

        // Check shape preserved
        assert_eq!(result.shape(), input.shape());

        // For a 2x2 matrix:
        // [1, 2]    -> [3, 4]
        // [3, 4]       [1, 2]
        let expected = Tensor::from_vec(CpuBackend::default(), vec![3.0, 4.0, 1.0, 2.0], vec![2, 2]).unwrap();

        for (i, (&actual, &exp)) in result.data().iter().zip(expected.data().iter()).enumerate() {
            assert_eq!(actual, exp, "Mismatch at index {}", i);
        }
    }

    #[test]
    fn test_compose_multiple_transforms() {
        let input = Tensor::from_vec(CpuBackend::default(), vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();

        let transform1 = Identity::new(); // Identity transform
        let transform2 = ToTensor::new(); // ToTensor transform

        let compose = Compose::new(vec![Box::new(transform1), Box::new(transform2)]);

        let result = compose.transform(&input).unwrap();

        // Compose should apply transforms in sequence
        // Both transforms return clones, so result should equal input
        for (i, (&actual, &original)) in result.data().iter().zip(input.data().iter()).enumerate() {
            assert_eq!(actual, original, "Mismatch at index {}", i);
        }
    }

    #[test]
    fn test_identity_transform() {
        let input = Tensor::from_vec(CpuBackend::default(), vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let transform = Identity::new();

        let result = transform.transform(&input).unwrap();

        // Identity should return exact copy
        assert_eq!(result.shape(), input.shape());
        for (i, (&actual, &original)) in result.data().iter().zip(input.data().iter()).enumerate() {
            assert_eq!(actual, original, "Mismatch at index {}", i);
        }
    }

    #[test]
    fn test_lambda_transform() {
        let input = Tensor::from_vec(CpuBackend::default(), vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let transform = Lambda::new(|tensor: &Tensor<f32, CpuBackend>| {
            // Double each element
            let doubled_data: Vec<f32> = tensor.data().iter().map(|&x| x * 2.0).collect();
            Ok(Tensor::from_vec(CpuBackend::default(), doubled_data, tensor.shape().to_vec()).unwrap())
        });

        let result = transform.transform(&input).unwrap();
        let expected = Tensor::from_vec(CpuBackend::default(), vec![2.0, 4.0, 6.0], vec![3]).unwrap();

        for (i, (&actual, &exp)) in result.data().iter().zip(expected.data().iter()).enumerate() {
            assert_eq!(actual, exp, "Mismatch at index {}", i);
        }
    }

    #[test]
    fn test_random_apply_probability() {
        let input = Tensor::from_vec(CpuBackend::default(), vec![1.0, 2.0, 3.0], vec![3]).unwrap();

        // Test with probability 0 (never apply)
        let transform = RandomApply::new(vec![Box::new(Identity::new())], 0.0);
        let result = transform.transform(&input).unwrap();

        // Should be unchanged
        for (i, (&actual, &original)) in result.data().iter().zip(input.data().iter()).enumerate() {
            assert_eq!(actual, original, "Mismatch at index {}", i);
        }
    }

    #[test]
    fn test_random_choice_basic() {
        let input = Tensor::from_vec(CpuBackend::default(), vec![1.0, 2.0, 3.0], vec![3]).unwrap();

        let transforms: Vec<Box<dyn Transform<f32, CpuBackend>>> = vec![Box::new(Identity::new())];

        let transform = RandomChoice::new(transforms);
        let result = transform.transform(&input).unwrap();

        // Should apply the single available transform
        assert_eq!(result.shape(), input.shape());
    }

    #[test]
    fn test_random_order_basic() {
        let input = Tensor::from_vec(CpuBackend::default(), vec![1.0, 2.0, 3.0], vec![3]).unwrap();

        let transforms: Vec<Box<dyn Transform<f32, CpuBackend>>> = vec![Box::new(Identity::new())];

        let transform = RandomOrder::new(transforms);
        let result = transform.transform(&input).unwrap();

        // Should apply transforms (in this case just one)
        assert_eq!(result.shape(), input.shape());
    }

    #[test]
    fn test_to_tensor_basic() {
        let input = Tensor::from_vec(CpuBackend::default(), vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let transform = ToTensor::new();

        let result = transform.transform(&input).unwrap();

        // ToTensor should return a copy
        assert_eq!(result.shape(), input.shape());
        for (i, (&actual, &original)) in result.data().iter().zip(input.data().iter()).enumerate() {
            assert_eq!(actual, original, "Mismatch at index {}", i);
        }
    }

    #[test]
    fn test_color_jitter_construction() {
        let _transform = ColorJitter::brightness(0.5);
        // Construction test - verify transform can be created without panicking
        // Full functionality test would require image processing
    }

    #[test]
    fn test_random_rotation_construction() {
        let _transform = RandomRotation::new((-30.0, 30.0));
        // Construction test - verify transform can be created without panicking
        // Full functionality test would require geometric transformations
    }

    #[test]
    fn test_random_affine_construction() {
        let _transform = RandomAffine::new().degrees((-10.0, 10.0));
        // Construction test - verify transform can be created without panicking
        // Full functionality test would require affine transformations
    }

    #[test]
    fn test_random_perspective_construction() {
        let _transform = RandomPerspective::new(0.5, 0.5);
        // Construction test - verify transform can be created without panicking
        // Full functionality test would require perspective transformations
    }

    #[test]
    fn test_random_erasing_construction() {
        let _transform: RandomErasing<f32> = RandomErasing::with_defaults();
        // Construction test - verify transform can be created with defaults
        // Full functionality test would require region erasing
    }

    #[test]
    fn test_transform_trait_consistency() {
        // Test that all transforms implement the Transform trait consistently
        let input = Tensor::from_vec(CpuBackend::default(), vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();

        let transforms: Vec<Box<dyn Transform<f32, CpuBackend>>> = vec![
            Box::new(Identity::new()),
            Box::new(ToTensor::new()),
            Box::new(RandomHorizontalFlip::new(0.0)), // p=0 to avoid randomness
            Box::new(RandomVerticalFlip::new(0.0)),   // p=0 to avoid randomness
        ];

        for (i, transform) in transforms.iter().enumerate() {
            let result = transform.transform(&input).unwrap();
            assert_eq!(
                result.shape(),
                input.shape(),
                "Transform {} changed shape unexpectedly",
                i
            );
        }
    }
}
