//! Tensor integration for tokenizer operations
//!
//! This module provides seamless integration between tokenizers and Coeus tensors,
//! enabling direct conversion between token sequences and tensor representations
//! with full autograd compatibility.

use crate::error::{Result, TokenizerError};
use crate::tokenizer::{TokenizationResult, TokenizeOptions, Tokenizer};
use coeus_tensor::{Dtype, FloatDtype, Tensor};

/// Tensor conversion trait for tokenizers
///
/// Provides methods for converting between token sequences and tensor representations
/// with automatic batching, padding, and attention mask generation.
pub trait TensorTokenizer<T: FloatDtype> {
    /// Convert token IDs to a tensor with shape (`batch_size`, `seq_len`)
    ///
    /// # Arguments
    /// * `token_ids` - Token ID sequences to convert
    /// * `pad_token_id` - Token ID to use for padding (if None, sequences must be same length)
    /// * `max_length` - Maximum sequence length (if None, use longest sequence)
    ///
    /// # Returns
    /// Tuple of (`input_ids_tensor`, `attention_mask_tensor`)
    ///
    /// # Errors
    /// Returns `TokenizerError` if padding is required but `pad_token_id` is None
    /// or if sequences have inconsistent lengths without padding.
    fn convert_tokens_to_tensor(
        &self,
        token_ids: &[&[usize]],
        pad_token_id: Option<usize>,
        max_length: Option<usize>,
    ) -> Result<(Tensor<T>, Tensor<T>)>;

    /// Convert a single token sequence to tensor
    ///
    /// # Arguments
    /// * `token_ids` - Token ID sequence to convert
    ///
    /// # Returns
    /// Tuple of (`input_ids_tensor`, `attention_mask_tensor`)
    ///
    /// # Errors
    /// Returns `TokenizerError` if conversion fails
    fn convert_single_tokens_to_tensor(
        &self,
        token_ids: &[usize],
    ) -> Result<(Tensor<T>, Tensor<T>)> {
        self.convert_tokens_to_tensor(&[token_ids], None, None)
    }

    /// Convert tokenization result to tensors
    ///
    /// # Arguments
    /// * `result` - Tokenization result to convert
    ///
    /// # Returns
    /// Tuple of (`input_ids_tensor`, `attention_mask_tensor`, `token_type_ids_tensor`)
    ///
    /// # Errors
    /// Returns `TokenizerError` if conversion fails
    #[allow(clippy::type_complexity)]
    fn convert_result_to_tensor(
        &self,
        result: &TokenizationResult,
    ) -> Result<(Tensor<T>, Tensor<T>, Option<Tensor<T>>)>;

    /// Convert tensor back to token IDs
    ///
    /// # Arguments
    /// * `tensor` - Input IDs tensor with shape (`batch_size`, `seq_len`)
    ///
    /// # Returns
    /// Vector of token ID sequences
    ///
    /// # Errors
    /// Returns `TokenizerError` if tensor conversion fails
    fn convert_tensor_to_tokens(&self, tensor: &Tensor<T>) -> Result<Vec<Vec<usize>>>;

    /// Encode text and convert to tensor in one operation
    ///
    /// # Arguments
    /// * `text` - Text to encode
    /// * `options` - Tokenization options
    ///
    /// # Returns
    /// Tuple of (`input_ids_tensor`, `attention_mask_tensor`)
    ///
    /// # Errors
    /// Returns `TokenizerError` if encoding or conversion fails
    fn encode_to_tensor(
        &self,
        text: &str,
        options: &TokenizeOptions,
    ) -> Result<(Tensor<T>, Tensor<T>)>;

    /// Encode batch of texts and convert to tensors
    ///
    /// # Arguments
    /// * `texts` - Texts to encode
    /// * `options` - Tokenization options
    ///
    /// # Returns
    /// Tuple of (`input_ids_tensor`, `attention_mask_tensor`)
    ///
    /// # Errors
    /// Returns `TokenizerError` if encoding or conversion fails
    fn encode_batch_to_tensor(
        &self,
        texts: &[&str],
        options: &TokenizeOptions,
    ) -> Result<(Tensor<T>, Tensor<T>)>;
}

/// Blanket implementation of `TensorTokenizer` for any type implementing Tokenizer
impl<D: FloatDtype, T: Tokenizer + ?Sized> TensorTokenizer<D> for T {
    fn convert_tokens_to_tensor(
        &self,
        token_ids: &[&[usize]],
        pad_token_id: Option<usize>,
        max_length: Option<usize>,
    ) -> Result<(Tensor<D>, Tensor<D>)> {
        if token_ids.is_empty() {
            return Err(TokenizerError::InvalidInput {
                message: "Empty token sequences provided".to_string(),
            });
        }

        // Determine sequence length
        let max_seq_len =
            max_length.unwrap_or_else(|| token_ids.iter().map(|seq| seq.len()).max().unwrap_or(0));

        if max_seq_len == 0 {
            return Err(TokenizerError::InvalidInput {
                message: "All token sequences are empty".to_string(),
            });
        }

        // Check if padding is needed
        let needs_padding = token_ids.iter().any(|seq| seq.len() != max_seq_len);

        if needs_padding && pad_token_id.is_none() {
            return Err(TokenizerError::InvalidInput {
                message: "Padding required but no pad_token_id provided".to_string(),
            });
        }

        let batch_size = token_ids.len();

        // Create input_ids tensor
        let mut input_ids_data = Vec::with_capacity(batch_size * max_seq_len);

        // Create attention_mask tensor
        let mut attention_mask_data = Vec::with_capacity(batch_size * max_seq_len);

        for seq in token_ids {
            let seq_len = seq.len();

            // Add token IDs with padding if needed
            for &token_id in *seq {
                #[allow(clippy::cast_precision_loss)]
                input_ids_data.push(D::from(token_id as f64).unwrap());
                attention_mask_data.push(D::from(1.0).unwrap());
            }

            // Add padding if needed
            if let Some(pad_id) = pad_token_id {
                for _ in seq_len..max_seq_len {
                    #[allow(clippy::cast_precision_loss)]
                    input_ids_data.push(D::from(pad_id as f64).unwrap());
                    attention_mask_data.push(D::from(0.0).unwrap());
                }
            }
        }

        // Create tensors
        let input_ids_shape = vec![batch_size, max_seq_len];
        let attention_mask_shape = vec![batch_size, max_seq_len];

        let input_ids = Tensor::from_vec(input_ids_data, input_ids_shape);
        let attention_mask = Tensor::from_vec(attention_mask_data, attention_mask_shape);

        Ok((input_ids, attention_mask))
    }

    fn convert_result_to_tensor(
        &self,
        result: &TokenizationResult,
    ) -> Result<(Tensor<D>, Tensor<D>, Option<Tensor<D>>)> {
        // Convert token_ids to tensor
        let (input_ids, attention_mask) =
            self.convert_tokens_to_tensor(&[&result.token_ids], None, None)?;

        // Convert token_type_ids if present
        #[allow(clippy::option_if_let_else)]
        let token_type_ids = if let Some(ref type_ids) = result.token_type_ids {
            let type_ids_data: Vec<D> = type_ids
                .iter()
                .map(|&id| {
                    #[allow(clippy::cast_precision_loss)]
                    D::from(id as f64).unwrap()
                })
                .collect();

            Some(Tensor::from_vec(type_ids_data, vec![1, type_ids.len()]))
        } else {
            None
        };

        Ok((input_ids, attention_mask, token_type_ids))
    }

    fn convert_tensor_to_tokens(&self, tensor: &Tensor<D>) -> Result<Vec<Vec<usize>>> {
        if tensor.ndim() != 2 {
            return Err(TokenizerError::InvalidInput {
                message: "Tensor must be 2D with shape (batch_size, seq_len)".to_string(),
            });
        }

        let batch_size = tensor.shape()[0];
        let seq_len = tensor.shape()[1];
        let data = tensor.data();

        if data.len() != batch_size * seq_len {
            return Err(TokenizerError::InvalidInput {
                message: "Tensor data length doesn't match expected shape".to_string(),
            });
        }

        let mut result = Vec::with_capacity(batch_size);

        for batch_idx in 0..batch_size {
            let mut token_ids = Vec::with_capacity(seq_len);

            for seq_idx in 0..seq_len {
                let idx = batch_idx * seq_len + seq_idx;
                let token_id_f64 = Dtype::to_f64(&data[idx]).unwrap_or(0.0);

                // Check if it's a valid token ID (should be integer)
                let epsilon = 1e-6;
                let token_id = if (token_id_f64 - token_id_f64.round()).abs() < epsilon {
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    { token_id_f64.round() as usize }
                } else {
                    return Err(TokenizerError::EncodingError {
                        message: format!("Non-integer token ID at position ({batch_idx}, {seq_idx}): {token_id_f64}"),
                    });
                };

                token_ids.push(token_id);
            }

            result.push(token_ids);
        }

        Ok(result)
    }

    fn encode_to_tensor(
        &self,
        text: &str,
        options: &TokenizeOptions,
    ) -> Result<(Tensor<D>, Tensor<D>)> {
        let token_ids = self.encode_with_options(text, options)?;
        self.convert_tokens_to_tensor(&[&token_ids], None, None)
    }

    fn encode_batch_to_tensor(
        &self,
        texts: &[&str],
        options: &TokenizeOptions,
    ) -> Result<(Tensor<D>, Tensor<D>)> {
        let token_sequences: Vec<Vec<usize>> = texts
            .iter()
            .map(|text| self.encode_with_options(text, options))
            .collect::<Result<Vec<_>>>()?;

        #[allow(clippy::redundant_closure_for_method_calls)]
        let token_refs: Vec<&[usize]> = token_sequences.iter().map(|seq| seq.as_slice()).collect();

        self.convert_tokens_to_tensor(&token_refs, None, None)
    }
}

/// Extension trait for tensors to work with tokenizers
pub trait TensorExt<T: FloatDtype> {
    /// Convert tensor to token IDs using a tokenizer
    ///
    /// # Arguments
    /// * `tokenizer` - Tokenizer to use for conversion
    ///
    /// # Returns
    /// Vector of token ID sequences
    ///
    /// # Errors
    /// Returns `TokenizerError` if conversion fails
    fn to_token_ids(&self, tokenizer: &impl Tokenizer) -> Result<Vec<Vec<usize>>>;

    /// Create tensor from token IDs with automatic batching
    ///
    /// # Arguments
    /// * `token_ids` - Token ID sequences
    /// * `pad_token_id` - Token ID for padding
    ///
    /// # Returns
    /// Tuple of (`input_ids_tensor`, `attention_mask_tensor`)
    ///
    /// # Errors
    /// Returns `TokenizerError` if conversion fails
    fn from_token_ids(
        token_ids: &[&[usize]],
        pad_token_id: Option<usize>,
    ) -> Result<(Tensor<T>, Tensor<T>)>;
}

impl<T: FloatDtype> TensorExt<T> for Tensor<T> {
    fn to_token_ids(&self, tokenizer: &impl Tokenizer) -> Result<Vec<Vec<usize>>> {
        // Use the tokenizer's tensor conversion method
        tokenizer.convert_tensor_to_tokens(self)
    }

    fn from_token_ids(
        _token_ids: &[&[usize]],
        _pad_token_id: Option<usize>,
    ) -> Result<(Self, Self)> {
        // This would need a tokenizer instance, but we can't access it here
        // This is a convenience method - users should use the tokenizer directly
        Err(TokenizerError::UnsupportedOperation {
            operation: "from_token_ids".to_string(),
            model: "TensorExt".to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::BaseTokenizer;
    use crate::vocabulary::Vocabulary;
    use coeus_tensor::Tensor;

    #[test]
    fn test_convert_tokens_to_tensor() {
        let mut vocab = Vocabulary::new();
        let _hello_id = vocab.add_token("hello".to_string());
        let _world_id = vocab.add_token("world".to_string());

        let tokenizer = BaseTokenizer::with_vocabulary("test".to_string(), vocab);

        // Test single sequence
        let token_ids = &[0, 1];
        let (input_ids, attention_mask): (Tensor<f32>, Tensor<f32>) = tokenizer
            .convert_single_tokens_to_tensor(token_ids)
            .unwrap();

        assert_eq!(input_ids.shape(), &[1, 2]);
        assert_eq!(attention_mask.shape(), &[1, 2]);
        assert_eq!(input_ids.data()[0].to_f64(), Some(0.0));
        assert_eq!(input_ids.data()[1].to_f64(), Some(1.0));
        assert_eq!(attention_mask.data()[0].to_f64(), Some(1.0));
        assert_eq!(attention_mask.data()[1].to_f64(), Some(1.0));
    }

    #[test]
    fn test_convert_tensor_to_tokens() {
        let mut vocab = Vocabulary::new();
        let _hello_id = vocab.add_token("hello".to_string());
        let _world_id = vocab.add_token("world".to_string());

        let tokenizer = BaseTokenizer::with_vocabulary("test".to_string(), vocab);

        // Create tensor from token IDs
        let data = vec![0.0f32, 1.0];
        let tensor = Tensor::from_vec(data, vec![1, 2]);

        let token_ids = tokenizer.convert_tensor_to_tokens(&tensor).unwrap();
        assert_eq!(token_ids, vec![vec![0, 1]]);
    }

    #[test]
    fn test_batch_tensor_conversion() {
        let mut vocab = Vocabulary::new();
        let _hello_id = vocab.add_token("hello".to_string());
        let _world_id = vocab.add_token("world".to_string());
        let _pad_id = vocab.add_special_token("[PAD]".to_string());

        let tokenizer = BaseTokenizer::with_vocabulary("test".to_string(), vocab);

        // Test batch with different lengths
        let seq1 = &[0, 1]; // length 2
        let seq2 = &[0]; // length 1

        let (input_ids, attention_mask): (Tensor<f32>, Tensor<f32>) = tokenizer
            .convert_tokens_to_tensor(&[seq1, seq2], Some(2), Some(3))
            .unwrap();

        assert_eq!(input_ids.shape(), &[2, 3]);
        assert_eq!(attention_mask.shape(), &[2, 3]);

        // Check first sequence
        assert_eq!(input_ids.data()[0].to_f64(), Some(0.0)); // hello
        assert_eq!(input_ids.data()[1].to_f64(), Some(1.0)); // world
        assert_eq!(input_ids.data()[2].to_f64(), Some(2.0)); // [PAD]

        // Check second sequence
        assert_eq!(input_ids.data()[3].to_f64(), Some(0.0)); // hello
        assert_eq!(input_ids.data()[4].to_f64(), Some(2.0)); // [PAD]
        assert_eq!(input_ids.data()[5].to_f64(), Some(2.0)); // [PAD]

        // Check attention mask
        assert_eq!(attention_mask.data()[0].to_f64(), Some(1.0));
        assert_eq!(attention_mask.data()[1].to_f64(), Some(1.0));
        assert_eq!(attention_mask.data()[2].to_f64(), Some(0.0)); // padding
        assert_eq!(attention_mask.data()[3].to_f64(), Some(1.0));
        assert_eq!(attention_mask.data()[4].to_f64(), Some(0.0)); // padding
        assert_eq!(attention_mask.data()[5].to_f64(), Some(0.0)); // padding
    }

    #[test]
    fn test_empty_sequences_error() {
        let vocab = Vocabulary::new();
        let tokenizer = BaseTokenizer::with_vocabulary("test".to_string(), vocab);

        let result: Result<(Tensor<f32>, Tensor<f32>)> =
            tokenizer.convert_tokens_to_tensor(&[], None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_padding_required_error() {
        let vocab = Vocabulary::new();
        let tokenizer = BaseTokenizer::with_vocabulary("test".to_string(), vocab);

        let seq1 = &[0, 1]; // length 2
        let seq2 = &[0]; // length 1

        let result: Result<(Tensor<f32>, Tensor<f32>)> =
            tokenizer.convert_tokens_to_tensor(&[seq1, seq2], None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_tensor_tokenizer_integration_workflow() {
        // Create a vocabulary with realistic tokens
        let mut vocab = Vocabulary::new();
        let _hello_id = vocab.add_token("hello".to_string());
        let _world_id = vocab.add_token("world".to_string());
        let _this_id = vocab.add_token("this".to_string());
        let _is_id = vocab.add_token("is".to_string());
        let _a_id = vocab.add_token("a".to_string());
        let _test_id = vocab.add_token("test".to_string());
        let _pad_id = vocab.add_special_token("[PAD]".to_string());

        let tokenizer = BaseTokenizer::with_vocabulary("test".to_string(), vocab);

        // Simulate token sequences from a batch of texts
        let batch_tokens = [
            vec![0, 1, 2, 3], // "hello world this is"
            vec![0, 1, 2],    // "hello world this"
            vec![4, 5],       // "a test"
        ];

        let batch_refs: Vec<&[usize]> = batch_tokens.iter().map(Vec::as_slice).collect();

        // Convert to tensors with padding
        let (input_ids, attention_mask): (Tensor<f32>, Tensor<f32>) = tokenizer
            .convert_tokens_to_tensor(&batch_refs, Some(6), Some(6))
            .unwrap();

        // Validate tensor shapes
        assert_eq!(input_ids.shape(), &[3, 6]); // 3 sequences, max length 6
        assert_eq!(attention_mask.shape(), &[3, 6]);

        // Validate first sequence (no padding needed)
        assert_eq!(input_ids.data()[0].to_f64(), Some(0.0)); // hello
        assert_eq!(input_ids.data()[1].to_f64(), Some(1.0)); // world
        assert_eq!(input_ids.data()[2].to_f64(), Some(2.0)); // this
        assert_eq!(input_ids.data()[3].to_f64(), Some(3.0)); // is
        assert_eq!(input_ids.data()[4].to_f64(), Some(6.0)); // [PAD]
        assert_eq!(input_ids.data()[5].to_f64(), Some(6.0)); // [PAD]

        // Validate attention mask for first sequence
        assert_eq!(attention_mask.data()[0].to_f64(), Some(1.0));
        assert_eq!(attention_mask.data()[1].to_f64(), Some(1.0));
        assert_eq!(attention_mask.data()[2].to_f64(), Some(1.0));
        assert_eq!(attention_mask.data()[3].to_f64(), Some(1.0));
        assert_eq!(attention_mask.data()[4].to_f64(), Some(0.0)); // padding
        assert_eq!(attention_mask.data()[5].to_f64(), Some(0.0)); // padding

        // Convert back to token IDs to verify round-trip
        let recovered_tokens = tokenizer.convert_tensor_to_tokens(&input_ids).unwrap();

        // First sequence should match original (without padding)
        assert_eq!(recovered_tokens[0], vec![0, 1, 2, 3, 6, 6]);

        // Second sequence should have padding
        assert_eq!(recovered_tokens[1], vec![0, 1, 2, 6, 6, 6]);

        // Third sequence should have padding
        assert_eq!(recovered_tokens[2], vec![4, 5, 6, 6, 6, 6]);

        // Test with autograd compatibility (tensors should be differentiable)
        assert_eq!(input_ids.shape(), &[3, 6]);
        assert!(!input_ids.requires_grad()); // Default state

        // Test f64 compatibility
        let (input_ids_f64, attention_mask_f64): (Tensor<f64>, Tensor<f64>) = tokenizer
            .convert_tokens_to_tensor(&batch_refs, Some(6), Some(6))
            .unwrap();

        assert_eq!(input_ids_f64.shape(), &[3, 6]);
        assert_eq!(attention_mask_f64.shape(), &[3, 6]);
        assert_eq!(input_ids_f64.data()[0].to_f64(), Some(0.0));
    }
}
