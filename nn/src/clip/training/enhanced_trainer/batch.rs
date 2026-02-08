use backend::Backend;
use dtype::DataType;
use storage::Storage;
use tensor::Tensor;

/// Enhanced CLIP training batch
#[derive(Debug)]
pub struct EnhancedClipBatch {
    /// Image pixel values: [batch_size, height, width, 3]
    pub images: Vec<f32>,
    /// Tokenized text sequences: [batch_size, seq_len]
    pub text_tokens: Vec<u32>,
    /// Attention masks for text: [batch_size, seq_len]
    pub text_masks: Vec<u32>,
    /// Batch size
    pub batch_size: usize,
    /// Sequence length
    pub seq_length: usize,
}

/// Batch with gradient accumulation support
#[derive(Debug)]
pub struct GradientAccumulationBatch<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T>,
    T: DataType,
{
    /// Accumulated loss tensor
    pub accumulated_loss: Tensor<B, S, T>,
    /// Number of batches accumulated
    pub accumulation_count: usize,
    /// Current batch data (for external reference)
    pub batch_data: EnhancedClipBatch,
}
