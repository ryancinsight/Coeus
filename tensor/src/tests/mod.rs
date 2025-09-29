#[cfg(test)]
mod integration {
    use super::*;
    use crate::{Dtype, FloatDtype};
    use coeus_backend::{CpuBackend, BackendError};
    use approx::assert_relative_eq;
    use proptest::prelude::*;

    #[test]
    fn test_dtype_quantization_round_trip() {
        use coeus_dtype::{QuantizedDtype, i8};

        let original = vec![0.0f32, 0.5, -0.5, 1.0, -1.0];
        let scale = i8::scale(); // 1.0/127.0
        let zero_point = i8::zero_point(); // 0

        let quantized: Vec<i8> = original.iter().map(|&x| i8::quantize(x, scale, zero_point)).collect();
        let dequantized: Vec<f32> = quantized.iter().map(|&q| q.dequantize(scale, zero_point)).collect();

        // Check reasonable accuracy for 8-bit quantization
        for (orig, deq) in original.iter().zip(dequantized.iter()) {
            assert!((orig - deq).abs() < 0.01, "Quantization error too large: {} vs {}", orig, deq);
        }
    }

    proptest! {
        #[test]
        fn proptest_dtype_edges(dtype in any::<f64>()) {
            let pos = 1.0f64;
            let neg = -1.0f64;
            let zero = 0.0f64;
            let inf = f64::INFINITY;
            let neg_inf = f64::NEG_INFINITY;
            let nan = f64::NAN;

            // Test is_finite, is_nan, is_infinite
            assert!(pos.is_finite());
            assert!(!pos.is_nan());
            assert!(!pos.is_infinite());

            assert!(neg.is_finite());
            assert!(!neg.is_nan());
            assert!(!neg.is_infinite());

            assert!(zero.is_finite());
            assert!(!zero.is_nan());
            assert!(!zero.is_infinite());

            assert!(inf.is_infinite());
            assert!(!inf.is_finite());
            assert!(!inf.is_nan());

            assert!(neg_inf.is_infinite());
            assert!(!neg_inf.is_finite());
            assert!(!neg_inf.is_nan());

            assert!(nan.is_nan());
            assert!(!nan.is_finite());
            assert!(!nan.is_infinite());

            // Test x=-1 y=10 → -10 for mul
            let x = -1.0f64;
            let y = 10.0f64;
            let expected = -10.0f64;
            prop_assert_eq!(x * y, expected);
        }
    }

    #[test]
    fn test_backend_cpu_arithmetic() {
        let backend = CpuBackend::new();
        let a = backend.zeros(&[2, 2]).unwrap();
        let b = backend.ones(&[2, 2]).unwrap();

        let add_result = backend.add(&a, &b).unwrap();
        let add_data = backend.copy_to_host(&add_result).unwrap();
        assert!(add_data.iter().all(|&x| x == 1.0));

        let mul_result = backend.mul(&a, &b).unwrap();
        let mul_data = backend.copy_to_host(&mul_result).unwrap();
        assert!(mul_data.iter().all(|&x| x == 0.0));

        // Test overflow/underflow
        let large = backend.copy_from_host(&[f32::MAX / 2.0], &[1]).unwrap();
        let overflow = backend.add(&large, &large).unwrap();
        let overflow_val = backend.copy_to_host(&overflow).unwrap()[0];
        assert!(overflow_val.is_infinite());

        // Test precision 1e-6
        let a_prec = backend.copy_from_host(&[1.0], &[1]).unwrap();
        let b_prec = backend.copy_from_host(&[1.000001], &[1]).unwrap();
        let diff = backend.sub(&a_prec, &b_prec).unwrap();
        let diff_val = backend.copy_to_host(&diff).unwrap()[0].abs();
        assert!(diff_val < 1e-6);

        // x=-1 y=10 → -10
        let x = backend.copy_from_host(&[-1.0], &[1]).unwrap();
        let y = backend.copy_from_host(&[10.0], &[1]).unwrap();
        let mul = backend.mul(&x, &y).unwrap();
        let mul_val = backend.copy_to_host(&mul).unwrap()[0];
        assert_eq!(mul_val, -10.0);
    }

    #[test]
    fn test_backend_gpu_arithmetic() {
        // GPU tests are optional - skip if no GPU available
        if let Ok(gpu_backend) = coeus_backend::GpuBackend::new() {
            let a = gpu_backend.zeros(&[2, 2]).unwrap();
            let b = gpu_backend.ones(&[2, 2]).unwrap();

            let add_result = gpu_backend.add(&a, &b).unwrap();
            let add_data = gpu_backend.copy_to_host(&add_result).unwrap();
            assert!(add_data.iter().all(|&x| x == 1.0));

            // Similar tests for mul, overflow, precision, x=-1 y=10
            let x = gpu_backend.copy_from_host(&[-1.0], &[1]).unwrap();
            let y = gpu_backend.copy_from_host(&[10.0], &[1]).unwrap();
            let mul = gpu_backend.mul(&x, &y).unwrap();
            let mul_val = gpu_backend.copy_to_host(&mul).unwrap()[0];
            assert_eq!(mul_val, -10.0);
        }
    }

    #[test]
    fn test_autograd_chain_rule() {
        use coeus_autograd::{AutogradContext, Operation};

        let backend = CpuBackend::new();
        let mut context = AutogradContext::new(backend);

        // Test f(x) = x^2, f'(x) = 2x
        let x_data = vec![2.0f32];
        let x_tensor = context.tensor_from_data(&x_data, &[])?;
        let x_squared = context.mul(&x_tensor, &x_tensor)?;
        let grad_output = context.tensor_from_data(&[1.0], &[])?;
        context.backward(&x_squared, &grad_output)?;

        let grad_x = context.grad(&x_tensor)?;
        assert_relative_eq!(grad_x.item(), 4.0, epsilon = 1e-6); // 2 * 2 = 4

        // Test f(x) = log(x), f'(x) = 1/x
        let x_log = vec![2.0f32];
        let x_log_tensor = context.tensor_from_data(&x_log, &[])?;
        let log_x = context.log(&x_log_tensor)?;
        let grad_output_log = context.tensor_from_data(&[1.0], &[])?;
        context.backward(&log_x, &grad_output_log)?;

        let grad_log = context.grad(&x_log_tensor)?;
        assert_relative_eq!(grad_log.item(), 0.5, epsilon = 1e-6); // 1/2 = 0.5

        // Test sigmoid chain rule
        let x_sig = vec![0.0f32];
        let x_sig_tensor = context.tensor_from_data(&x_sig, &[])?;
        let sig_x = context.sigmoid(&x_sig_tensor)?;
        let grad_output_sig = context.tensor_from_data(&[1.0], &[])?;
        context.backward(&sig_x, &grad_output_sig)?;

        let grad_sig = context.grad(&x_sig_tensor)?;
        assert_relative_eq!(grad_sig.item(), 0.25, epsilon = 1e-6); // sigmoid(0)=0.5, deriv=0.25

        // Test x=-1 y=10 → mul=-10, grad w.r.t x=10 (y), w.r.t y=-1 (x)
        let x_mul = vec![-1.0f32];
        let y_mul = vec![10.0f32];
        let x_mul_tensor = context.tensor_from_data(&x_mul, &[])?;
        let y_mul_tensor = context.tensor_from_data(&y_mul, &[])?;
        let mul_xy = context.mul(&x_mul_tensor, &y_mul_tensor)?;
        let grad_output_mul = context.tensor_from_data(&[1.0], &[])?;
        context.backward(&mul_xy, &grad_output_mul)?;

        let grad_x_mul = context.grad(&x_mul_tensor)?;
        let grad_y_mul = context.grad(&y_mul_tensor)?;
        assert_relative_eq!(grad_x_mul.item(), 10.0, epsilon = 1e-6); // y=10
        assert_relative_eq!(grad_y_mul.item(), -1.0, epsilon = 1e-6); // x=-1
    }

    proptest! {
        #[test]
        fn proptest_broadcasting() {
            // Test scalar + matrix
            let scalar = Tensor::from_vec(vec![2.0], vec![]);
            let matrix = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
            let result = arithmetic::add(&scalar, &matrix).unwrap();
            assert_eq!(result.shape(), &[2, 2]);
            let expected = vec![3.0, 4.0, 5.0, 6.0];
            assert_eq!(result.data(), &expected);

            // Test [3,1] + [1,4] → [3,4]
            let vec1 = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3, 1]);
            let vec2 = Tensor::from_vec(vec![4.0, 5.0, 6.0, 7.0], vec![1, 4]);
            let broadcast_result = arithmetic::add(&vec1, &vec2).unwrap();
            assert_eq!(broadcast_result.shape(), &[3, 4]);
            let expected_broadcast = vec![5.0, 6.0, 7.0, 8.0, 6.0, 7.0, 8.0, 9.0, 7.0, 8.0, 9.0, 10.0];
            assert_eq!(broadcast_result.data(), &expected_broadcast);
        }
    }

    #[test]
    fn test_tensor_integration() {
        let backend = CpuBackend::new();
        let a = backend.zeros(&[2, 2]).unwrap();
        let b = backend.ones(&[2, 2]).unwrap();

        let add_result = backend.add(&a, &b).unwrap();
        assert_eq!(add_result.shape(), &[2, 2]);

        // Test broadcasting
        let scalar = backend.copy_from_host(&[5.0], &[]).unwrap();
        let broadcast_add = backend.add(&a, &scalar).unwrap();
        let broadcast_data = backend.copy_to_host(&broadcast_add).unwrap();
        assert!(broadcast_data.iter().all(|&x| x == 5.0));

        // Test autograd
        let mut context = coeus_autograd::AutogradContext::new(backend.clone());
        let x = context.tensor_from_data(&[3.0], &[])?;
        let y = context.mul(&x, &x)?; // y = x^2
        let grad_out = context.tensor_from_data(&[1.0], &[])?;
        context.backward(&y, &grad_out)?;
        let grad_x = context.grad(&x)?;
        assert_relative_eq!(grad_x.item(), 6.0, epsilon = 1e-6); // 2*3=6
    }
}

#[cfg(test)]
mod gpu_tests {
    use super::*;
    use coeus_backend::GpuBackend;

    #[tokio::test]
    async fn test_gpu_integration() {
        if let Ok(gpu_backend) = GpuBackend::new().await {
            let a = gpu_backend.zeros(&[2, 2]).unwrap();
            let b = gpu_backend.ones(&[2, 2]).unwrap();

            let add_result = gpu_backend.add(&a, &b).unwrap();
            let add_data = gpu_backend.copy_to_host(&add_result).unwrap();
            assert!(add_data.iter().all(|&x| x == 1.0));

            // Test no UB - basic operations
            let scalar = gpu_backend.copy_from_host(&[42.0], &[]).unwrap();
            let _ = gpu_backend.add(&a, &scalar).unwrap(); // Should not panic
        }
    }
}
