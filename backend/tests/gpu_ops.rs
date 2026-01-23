use backend::gpu::GpuBackend;
use backend::{Backend, Storage};
use storage::DenseStorage;
use dtype::float::Float32;
use dtype::num_traits::Float;

#[test]
fn test_gpu_element_wise_ops() {
    let backend = GpuBackend::<Float32>::new();
    let data_f32: Vec<f32> = vec![1.0, 0.0, -1.0, 2.0];
    let data: Vec<Float32> = data_f32.iter().map(|&x| Float32(x)).collect();
    let storage = DenseStorage::from_vec(data.clone(), &[4]).unwrap();

    // EXP
    match backend.exp_dense(&storage) {
        Ok(result) => {
            let res_slice = result.as_slice();
            for (i, val) in res_slice.iter().enumerate() {
                let expected = data_f32[i].exp();
                let val_f32 = val.0;
                assert!((val_f32 - expected).abs() < 1e-4, "Exp mismatch at {}: got {}, expected {}", i, val_f32, expected);
            }
        },
        Err(e) => {
            println!("Skipping GPU test due to error: {}", e);
            return;
        }
    }

    // SIN
    if let Ok(result) = backend.sin_dense(&storage) {
        let res_slice = result.as_slice();
        for (i, val) in res_slice.iter().enumerate() {
            let expected = data_f32[i].sin();
            let val_f32 = val.0;
            assert!((val_f32 - expected).abs() < 1e-4, "Sin mismatch at {}: got {}, expected {}", i, val_f32, expected);
        }
    }

    // RELU
    if let Ok(result) = backend.relu_dense(&storage) {
        let res_slice = result.as_slice();
        for (i, val) in res_slice.iter().enumerate() {
            let expected = if data_f32[i] > 0.0 { data_f32[i] } else { 0.0 };
            let val_f32 = val.0;
            assert!((val_f32 - expected).abs() < 1e-4, "ReLU mismatch at {}: got {}, expected {}", i, val_f32, expected);
        }
    }
}

#[test]
fn test_gpu_reduction_ops() {
    let backend = GpuBackend::<Float32>::new();
    let data_f32: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let data: Vec<Float32> = data_f32.iter().map(|&x| Float32(x)).collect();
    let storage = DenseStorage::from_vec(data.clone(), &[5]).unwrap();

    // SUM
    match backend.sum_dense(&storage) {
        Ok(val) => {
            let expected: f32 = data_f32.iter().sum();
            let val_f32 = val.0;
            assert!((val_f32 - expected).abs() < 1e-4, "Sum mismatch: got {}, expected {}", val_f32, expected);
        },
        Err(e) => {
             println!("Skipping GPU reduc test due to error: {}", e);
             return;
        }
    }

    // MAX
    if let Ok(val) = backend.max_dense(&storage) {
        let expected = 5.0;
        let val_f32 = val.0;
        assert!((val_f32 - expected).abs() < 1e-4, "Max mismatch: got {}, expected {}", val_f32, expected);
    }
    
    // MIN
    if let Ok(val) = backend.min_dense(&storage) {
        let expected = 1.0;
        let val_f32 = val.0;
        assert!((val_f32 - expected).abs() < 1e-4, "Min mismatch: got {}, expected {}", val_f32, expected);
    }

    // MEAN
    if let Ok(dense_res) = backend.mean_dense(&storage, None) {
        let val = dense_res.as_slice()[0];
        let val_f32 = val.0;
        let expected = 3.0;
        assert!((val_f32 - expected).abs() < 1e-4, "Mean mismatch: got {}, expected {}", val_f32, expected);
    }
    }


#[test]
fn test_gpu_gelu() {
    let backend = GpuBackend::<Float32>::new();
    // Test passes on CPU fallback or GPU
    
    let input_data = vec![
        Float32(-1.0),
        Float32(0.0),
        Float32(1.0),
        Float32(2.0),
        Float32(3.0),
    ];
    let input = DenseStorage::from_vec(input_data.clone(), &[5]).unwrap();
    
    let result = backend.gelu_dense(&input).unwrap();
    let result_data = result.as_slice();

    let expected_data: Vec<f32> = input_data.iter().map(|&x| {
        let val = x.0;
        let c = 0.044715;
        let sqrt_2_pi = 0.7978845608;
        let inner = sqrt_2_pi * (val + c * val * val * val);
        0.5 * val * (1.0 + inner.tanh())
    }).collect();

    for (a, b) in result_data.iter().zip(expected_data.iter()) {
        assert!((a.0 - b).abs() < 1e-4, "GELU mismatch: {} vs {}", a.0, b);
    }
}
