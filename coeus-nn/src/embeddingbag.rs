use crate::module::Module;
use coeus_autograd::{cat, embedding, max_axis, mean_axis, slice, sum_axis, Var};
use coeus_core::{CpuAddressableStorage, CpuAddressableStorageMut, Float, MoiraiBackend, Scalar};
use coeus_tensor::Tensor;

/// Aggregation mode for EmbeddingBag.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmbeddingBagMode {
    /// Sum of embedding vectors in each bag.
    Sum,
    /// Mean of embedding vectors in each bag.
    Mean,
    /// Element-wise maximum over embedding vectors in each bag.
    Max,
}

impl EmbeddingBagMode {
    /// Parse from a string ("sum", "mean", "max").
    pub fn parse(s: &str) -> Option<Self> {
        s.parse().ok()
    }
}

impl core::str::FromStr for EmbeddingBagMode {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "sum" => Ok(Self::Sum),
            "mean" => Ok(Self::Mean),
            "max" => Ok(Self::Max),
            _ => Err("mode must be one of: sum, mean, max"),
        }
    }
}

/// EmbeddingBag layer: aggregates embedding rows by bag using sum, mean, or max.
#[derive(Clone)]
pub struct EmbeddingBag<T: Scalar, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend> {
    /// Weight matrix: `[num_embeddings, embedding_dim]`.
    pub weight: Var<T, B>,
    /// Number of rows in the weight matrix.
    pub num_embeddings: usize,
    /// Embedding vector dimension.
    pub embedding_dim: usize,
    /// Aggregation mode (sum / mean / max).
    pub mode: EmbeddingBagMode,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> EmbeddingBag<T, B>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    /// Create an EmbeddingBag with unit weight matrix.
    pub fn new(num_embeddings: usize, embedding_dim: usize, mode: EmbeddingBagMode) -> Self {
        let backend = B::default();
        let w_tensor = Tensor::ones_on([num_embeddings, embedding_dim], &backend);
        let weight = Var::new(w_tensor, true);
        Self {
            weight,
            num_embeddings,
            embedding_dim,
            mode,
        }
    }

    fn bag_starts(indices_len: usize, offsets: Option<&[usize]>) -> Vec<usize> {
        match offsets {
            Some(offs) => {
                assert!(
                    !offs.is_empty(),
                    "EmbeddingBag::forward_with_offsets: offsets must be non-empty when provided"
                );
                assert!(
                    offs.windows(2).all(|w| w[0] <= w[1]),
                    "EmbeddingBag::forward_with_offsets: offsets must be non-decreasing"
                );
                assert!(
                    offs.iter().all(|&o| o <= indices_len),
                    "EmbeddingBag::forward_with_offsets: offset out of bounds for indices length"
                );
                offs.to_vec()
            }
            None => vec![0],
        }
    }

    fn reduce_one_bag(&self, embeddings: &Var<T, B>, start: usize, end: usize) -> Var<T, B> {
        let backend = B::default();
        let d = self.embedding_dim;

        if start == end {
            return Var::new(Tensor::zeros_on([1, d], &backend), false);
        }

        let bag = slice(embeddings, &[(start, end), (0, d)]);
        match self.mode {
            EmbeddingBagMode::Sum => sum_axis(&bag, 0),
            EmbeddingBagMode::Mean => mean_axis(&bag, 0),
            EmbeddingBagMode::Max => max_axis(&bag, 0),
        }
    }

    /// Forward pass with explicit bag offsets.
    ///
    /// `indices` is a flattened list of token ids and `offsets` are bag start
    /// positions into that list. If `offsets` is `None`, `indices` is treated as one bag.
    ///
    /// Returns a tensor of shape `[num_bags, embedding_dim]`.
    pub fn forward_with_offsets(&self, indices: &[usize], offsets: Option<&[usize]>) -> Var<T, B> {
        for &idx in indices {
            assert!(
                idx < self.num_embeddings,
                "EmbeddingBag::forward_with_offsets: index {idx} out of bounds [0, {})",
                self.num_embeddings
            );
        }

        let backend = B::default();
        let starts = Self::bag_starts(indices.len(), offsets);
        let idx_data: Vec<i64> = indices.iter().map(|&x| x as i64).collect();
        let idx_tensor = Tensor::from_slice_on([indices.len()], &idx_data, &backend);
        let embeddings = embedding(&self.weight, &idx_tensor);

        let rows: Vec<Var<T, B>> = starts
            .iter()
            .enumerate()
            .map(|(bag, &start)| {
                let end = starts.get(bag + 1).copied().unwrap_or(indices.len());
                self.reduce_one_bag(&embeddings, start, end)
            })
            .collect();

        let row_refs: Vec<&Var<T, B>> = rows.iter().collect();
        cat(&row_refs, 0)
    }
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for EmbeddingBag<T, B>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![self.weight.clone()]
    }

    /// Forward pass treating `input` as flat index tensor for a single bag.
    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
        let indices: Vec<usize> = input
            .tensor
            .as_slice()
            .iter()
            .map(|&v| T::to_f64(v) as usize)
            .collect();
        self.forward_with_offsets(&indices, None)
    }
}
