use coeus_tokenizer::{
    bpe::BpeTokenizer as RustBpeTokenizer, encoding::Encoding as RustEncoding, special_tokens,
};
use pyo3::prelude::*;
use pyo3::{pyclass, pymethods, PyResult};

/// Encoding wrapper for tiktoken-compatible API
#[pyclass]
pub struct Encoding {
    encoding: RustEncoding,
}

#[pymethods]
impl Encoding {
    #[new]
    fn new(model_name: &str) -> PyResult<Self> {
        let encoding = RustEncoding::new(model_name).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to create encoding: {}",
                e
            ))
        })?;

        Ok(Encoding { encoding })
    }

    #[staticmethod]
    fn for_encoding_only(model_name: &str) -> PyResult<Self> {
        let encoding = RustEncoding::for_encoding_only(model_name).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to create encoding: {}",
                e
            ))
        })?;

        Ok(Encoding { encoding })
    }

    #[staticmethod]
    fn available_models() -> Vec<String> {
        RustEncoding::available_models()
    }

    fn encode(&self, text: &str) -> PyResult<Vec<usize>> {
        self.encoding.encode(text).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Encoding failed: {}", e))
        })
    }

    fn encode_with_special_tokens(&self, text: &str) -> PyResult<Vec<usize>> {
        self.encoding.encode_with_special_tokens(text).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Encoding failed: {}", e))
        })
    }

    #[pyo3(signature = (text, _options=None))]
    fn encode_with_options(&self, text: &str, _options: Option<PyObject>) -> PyResult<Vec<usize>> {
        // For now, ignore options and use default encoding
        self.encode(text)
    }

    fn encode_batch(&self, texts: Vec<String>) -> PyResult<Vec<Vec<usize>>> {
        let str_texts: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        self.encoding.encode_batch(&str_texts).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Batch encoding failed: {}",
                e
            ))
        })
    }

    fn encode_batch_with_special_tokens(&self, texts: Vec<String>) -> PyResult<Vec<Vec<usize>>> {
        let str_texts: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        self.encoding
            .encode_batch_with_special_tokens(&str_texts)
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Batch encoding with special tokens failed: {}",
                    e
                ))
            })
    }

    fn decode(&self, tokens: Vec<usize>) -> PyResult<String> {
        self.encoding.decode(&tokens).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Decoding failed: {}", e))
        })
    }

    fn decode_batch(&self, batches: Vec<Vec<usize>>) -> PyResult<Vec<String>> {
        let mut results = Vec::new();
        for tokens in batches {
            let text = self.encoding.decode(&tokens).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Batch decoding failed: {}",
                    e
                ))
            })?;
            results.push(text);
        }
        Ok(results)
    }

    fn model_name(&self) -> &str {
        self.encoding.model_name()
    }

    fn vocab_size(&self) -> usize {
        self.encoding.vocab_size()
    }
}

/// BPE Tokenizer
#[pyclass]
pub struct BpeTokenizer {
    tokenizer: RustBpeTokenizer,
}

#[pymethods]
impl BpeTokenizer {
    #[new]
    fn new() -> PyResult<Self> {
        let tokenizer = RustBpeTokenizer::new("bpe".to_string());
        Ok(BpeTokenizer { tokenizer })
    }

    fn train(&mut self, text: &str, vocab_size: usize) -> PyResult<()> {
        let corpus = vec![text.to_string()];
        self.tokenizer
            .train(&corpus, vocab_size, None)
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Training failed: {}", e))
            })
    }

    fn encode(&self, text: &str) -> PyResult<Vec<usize>> {
        self.tokenizer
            .encode_bpe(text)
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Encoding failed: {}", e))
            })
            .map(|tokens| {
                // Convert token strings to IDs
                let mut ids = Vec::new();
                for token in tokens {
                    if let Some(id) = self.tokenizer.vocabulary().get_token_id(&token) {
                        ids.push(id);
                    } else {
                        // Handle unknown token - for now, use 0
                        ids.push(0);
                    }
                }
                ids
            })
    }

    fn decode(&self, tokens: Vec<usize>) -> PyResult<String> {
        // Convert token IDs to token strings
        let mut token_strings = Vec::new();
        for &token_id in &tokens {
            if let Some(token) = self.tokenizer.vocabulary().get_token(token_id) {
                token_strings.push(token.to_string());
            } else {
                // Handle unknown token ID - for now, use empty string
                token_strings.push("".to_string());
            }
        }

        self.tokenizer.decode_bpe(&token_strings).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Decoding failed: {}", e))
        })
    }

    fn vocab_size(&self) -> usize {
        self.tokenizer.vocabulary().size()
    }
}

/// GPT-2 Tokenizer using BPE algorithm
#[pyclass]
pub struct GPT2Tokenizer {
    bpe_tokenizer: BpeTokenizer,
}

#[pymethods]
impl GPT2Tokenizer {
    #[new]
    fn new() -> PyResult<Self> {
        // Create a BPE tokenizer trained on basic English text
        // In production, this would load pre-trained GPT-2 vocabulary and merges
        let mut bpe_tokenizer = BpeTokenizer::new()?;

        // Train on basic English corpus for demonstration
        // TODO: Load actual GPT-2 vocabulary and merges file
        let training_text = "Hello world. This is a test. The quick brown fox jumps over the lazy dog. Machine learning is fascinating. Natural language processing with transformers.";

        bpe_tokenizer
            .train(training_text, 1000) // Smaller vocab for demo
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Failed to train BPE tokenizer: {}",
                    e
                ))
            })?;

        // Add GPT-2 special tokens
        bpe_tokenizer.tokenizer.add_special_tokens(&[
            special_tokens::END_OF_TEXT.to_string(),
            special_tokens::START_OF_SEQUENCE.to_string(),
        ]);

        Ok(GPT2Tokenizer { bpe_tokenizer })
    }

    fn encode(&self, text: &str) -> PyResult<Vec<usize>> {
        self.bpe_tokenizer.encode(text)
    }

    fn decode(&self, tokens: Vec<usize>) -> PyResult<String> {
        self.bpe_tokenizer.decode(tokens)
    }

    fn encode_with_special_tokens(&self, text: &str) -> PyResult<Vec<usize>> {
        // For GPT-2, we typically add end-of-text token at the end
        let mut tokens = self.bpe_tokenizer.encode(text)?;
        if let Some(eot_id) = self
            .bpe_tokenizer
            .tokenizer
            .vocabulary()
            .get_special_token_id(special_tokens::END_OF_TEXT)
        {
            tokens.push(eot_id);
        }
        Ok(tokens)
    }

    fn decode_with_special_tokens(&self, tokens: Vec<usize>) -> PyResult<String> {
        // Decode and skip special tokens
        let mut token_strings = Vec::new();
        for &token_id in &tokens {
            if let Some(token) = self
                .bpe_tokenizer
                .tokenizer
                .vocabulary()
                .get_token(token_id)
            {
                // Skip special tokens in output
                if !self
                    .bpe_tokenizer
                    .tokenizer
                    .vocabulary()
                    .is_special_token(token)
                {
                    token_strings.push(token.to_string());
                }
            }
        }

        self.bpe_tokenizer
            .tokenizer
            .decode_bpe(&token_strings)
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Decoding failed: {}", e))
            })
    }

    fn vocab_size(&self) -> usize {
        self.bpe_tokenizer.vocab_size()
    }

    /// Get tokenizer name
    fn name(&self) -> &str {
        "gpt2"
    }
}

/// CLIP Tokenizer
#[pyclass]
pub struct CLIPTokenizer {
    encoding: RustEncoding,
}

#[pymethods]
impl CLIPTokenizer {
    #[new]
    fn new() -> PyResult<Self> {
        let encoding = RustEncoding::new("clip").map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to create CLIP tokenizer: {}",
                e
            ))
        })?;

        Ok(CLIPTokenizer { encoding })
    }

    fn encode(&self, text: &str) -> PyResult<Vec<usize>> {
        self.encoding.encode(text).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Encoding failed: {}", e))
        })
    }

    fn decode(&self, tokens: Vec<usize>) -> PyResult<String> {
        self.encoding.decode(&tokens).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Decoding failed: {}", e))
        })
    }

    fn vocab_size(&self) -> usize {
        self.encoding.vocab_size()
    }
}

/// BERT Tokenizer
#[pyclass]
pub struct BERTTokenizer {
    encoding: RustEncoding,
}

#[pymethods]
impl BERTTokenizer {
    #[new]
    fn new() -> PyResult<Self> {
        let encoding = RustEncoding::new("bert").map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to create BERT tokenizer: {}",
                e
            ))
        })?;

        Ok(BERTTokenizer { encoding })
    }

    fn encode(&self, text: &str) -> PyResult<Vec<usize>> {
        self.encoding.encode(text).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Encoding failed: {}", e))
        })
    }

    fn decode(&self, tokens: Vec<usize>) -> PyResult<String> {
        self.encoding.decode(&tokens).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Decoding failed: {}", e))
        })
    }

    fn vocab_size(&self) -> usize {
        self.encoding.vocab_size()
    }
}
