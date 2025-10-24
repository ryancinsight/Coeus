//! GRU unit tests.
//!
//! Comprehensive tests for GRU layer functionality.

#[cfg(test)]
mod gru_tests {
    use super::*;
    use crate::module::Module;
    use crate::rnn::GRU;
    use coeus_backend::CpuBackend;
    use coeus_dtype::float::Float32;
    use coeus_storage::DenseStorage;
    use coeus_tensor::Tensor;

    #[test]
    fn test_gru_creation() {
        let gru = GRU::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            10, 20, 1, true, false, false,
        )
        .unwrap();
        assert_eq!(gru.input_size, 10);
        assert_eq!(gru.hidden_size, 20);
        assert_eq!(gru.num_layers, 1);
        assert!(gru.bias);
        assert!(!gru.batch_first);
        assert!(!gru.bidirectional);
    }

    #[test]
    fn test_gru_bidirectional() {
        let gru = GRU::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            10, 20, 2, true, false, true,
        )
        .unwrap();
        assert!(gru.bidirectional);
        assert_eq!(gru.num_layers, 2);
        // Bidirectional GRU has 2x parameters per layer
        assert_eq!(gru.weight_ih.len(), 4); // 2 layers * 2 directions
    }

    #[test]
    fn test_gru_forward_shape() {
        let gru = GRU::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            10, 20, 1, true, false, false,
        )
        .unwrap();
        let input =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[5, 3, 10])
                .unwrap();
        let output = gru.forward(&input).unwrap();
        // GRU outputs (seq_len, batch_size, hidden_size) = (5, 3, 20)
        assert_eq!(output.shape().dims(), &[5, 3, 20]);
    }

    #[test]
    fn test_gru_parameters() {
        let gru = GRU::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            10, 20, 1, true, false, false,
        )
        .unwrap();
        let params = gru.parameters();
        // 1 layer, 1 direction: weight_ih, weight_hh, bias_ih, bias_hh
        assert_eq!(params.len(), 4);
    }

    #[test]
    fn test_gru_no_bias() {
        let gru = GRU::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            10, 20, 1, false, false, false,
        )
        .unwrap();
        let params = gru.parameters();
        // 1 layer, 1 direction, no bias: weight_ih, weight_hh only
        assert_eq!(params.len(), 2);
    }

    #[test]
    fn test_gru_multilayer() {
        let gru = GRU::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            10, 20, 3, true, false, false,
        )
        .unwrap();
        assert_eq!(gru.num_layers, 3);
        let params = gru.parameters();
        // 3 layers, 1 direction: 4 params per layer
        assert_eq!(params.len(), 12);
    }

    #[test]
    fn test_gru_batch_first() {
        let gru = GRU::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            10, 20, 1, true, true, false,
        )
        .unwrap();
        assert!(gru.batch_first);
    }

    #[test]
    #[should_panic(expected = "input_size must be > 0")]
    fn test_gru_invalid_input_size() {
        GRU::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            0, 20, 1, true, false, false,
        )
        .unwrap();
    }

    #[test]
    #[should_panic(expected = "hidden_size must be > 0")]
    fn test_gru_invalid_hidden_size() {
        GRU::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            10, 0, 1, true, false, false,
        )
        .unwrap();
    }

    #[test]
    #[should_panic(expected = "num_layers must be > 0")]
    fn test_gru_invalid_num_layers() {
        GRU::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            10, 20, 0, true, false, false,
        )
        .unwrap();
    }

    #[test]
    fn test_gru_multilayer_state_propagation() {
        // Test that multi-layer GRUs properly propagate hidden states between layers
        let gru = GRU::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            10, 20, 2, true, false, false,
        )
        .unwrap();

        // Create input: (seq_len=3, batch_size=2, input_size=10) with some non-zero values
        let input_data: Vec<Float32> = vec![
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1,
            7.1, 8.1, 9.1, 10.1, 1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2, 9.2, 10.2, 1.3, 2.3, 3.3,
            4.3, 5.3, 6.3, 7.3, 8.3, 9.3, 10.3, 1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4, 9.4, 10.4,
            1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5,
        ]
        .into_iter()
        .map(Float32::new)
        .collect();
        let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            input_data,
            &[3, 2, 10],
        )
        .unwrap();

        let output = gru.forward(&input).unwrap();

        // Verify output shape: (seq_len, batch_size, hidden_size)
        assert_eq!(output.shape().dims(), &[3, 2, 20]);

        // Verify that output is not all zeros (meaning computation occurred)
        let output_data = output.as_slice();
        let has_non_zero = output_data.iter().any(|&x| x != Float32::new(0.0));
        assert!(has_non_zero, "GRU output should not be all zeros");
    }
}
