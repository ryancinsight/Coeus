#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use coeus_backend::{Backend, BackendData, BackendKind, select_backend};

    proptest! {
        #[test]
        fn prop_attention_zero_q(n_heads in 1..=4usize, seq_len in 1..=8usize) {
            let backend = select_backend(BackendKind::Cpu).unwrap();
            let q = BackendData::cpu(vec![0.0f32; n_heads * seq_len * 64], vec![n_heads, seq_len, 64]); // Zero Q
            let k = BackendData::cpu(vec![1.0f32; n_heads * seq_len * 64], vec![n_heads, seq_len, 64]);
            let v = BackendData::cpu(vec![1.0f32; n_heads * seq_len * 64], vec![n_heads, seq_len, 64]);
            let output = backend.attention(&q, &k, &v).unwrap();
            prop_assert!(output.data().iter().all(|&x| x == 0.0)); // Zero Q → zero attention
        }

        #[test]
        fn prop_pooling_stride(values in proptest::collection::vec(0.0..=10.0f32, 1..=64), kernel in 1..=4usize, stride in 1..=2usize) {
            let backend = select_backend(BackendKind::Cpu).unwrap();
            let input = BackendData::cpu(values.clone(), vec![values.len()]);
            let output = backend.pooling(&input, vec![kernel], vec![stride], "max").unwrap();
            // Check downsampling: output.len() ≈ input.len() / (kernel*stride)
            prop_assert!(output.len() as f32 <= input.len() as f32 / (kernel * stride) as f32 + 1.0);
        }
    }
}
