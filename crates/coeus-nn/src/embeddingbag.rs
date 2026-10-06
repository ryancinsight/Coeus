use crate::module::{Module, ModuleError};
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

    fn bag_starts(
        indices_len: usize,
        offsets: Option<&[usize]>,
    ) -> Result<Vec<usize>, ModuleError<B::Error>> {
        match offsets {
            Some(offs) => {
                if offs.is_empty()
                    || !offs.windows(2).all(|window| window[0] <= window[1])
                    || !offs.iter().all(|&offset| offset <= indices_len)
                {
                    return Err(ModuleError::ShapeMismatch {
                        module: "EmbeddingBag",
                        parameter: "offsets must be non-empty, ordered, and within indices",
                        expected: vec![indices_len],
                        actual: offs.to_vec(),
                    });
                }
                Ok(offs.to_vec())
            }
            None => Ok(vec![0]),
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
    ///
    /// # Errors
    ///
    /// Returns [`ModuleError::ShapeMismatch`] when an index is outside the
    /// vocabulary or offsets are empty, unordered, or outside `indices`.
    pub fn forward_with_offsets(
        &self,
        indices: &[usize],
        offsets: Option<&[usize]>,
    ) -> Result<Var<T, B>, ModuleError<B::Error>> {
        for (position, &index) in indices.iter().enumerate() {
            if index >= self.num_embeddings || i64::try_from(index).is_err() {
                return Err(ModuleError::ShapeMismatch {
                    module: "EmbeddingBag",
                    parameter: "indices must be within the embedding vocabulary",
                    expected: vec![self.num_embeddings],
                    actual: vec![position],
                });
            }
        }

        let backend = B::default();
        let starts = Self::bag_starts(indices.len(), offsets)?;
        let idx_data: Vec<i64> = indices
            .iter()
            .map(|&index| {
                i64::try_from(index)
                    .expect("invariant: embedding index range was validated before conversion")
            })
            .collect();
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
        Ok(cat(&row_refs, 0))
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
    fn forward(&self, input: &Var<T, B>) -> Result<Var<T, B>, ModuleError<B::Error>> {
        let mut indices = Vec::with_capacity(input.tensor.numel());
        for (position, &index) in input.tensor.as_slice().iter().enumerate() {
            let value = <T as Scalar>::to_f64(index);
            if !value.is_finite()
                || value < 0.0
                || value.trunc() != value
                || value >= self.num_embeddings as f64
                || value > i64::MAX as f64
            {
                return Err(ModuleError::ShapeMismatch {
                    module: "EmbeddingBag",
                    parameter: "indices must be finite integers within the embedding vocabulary",
                    expected: vec![self.num_embeddings],
                    actual: vec![position],
                });
            }
            // SAFETY CONTRACT: the finite, integral, non-negative vocabulary
            // bounds above make this float-to-index conversion lossless.
            indices.push(value as usize);
        }
        self.forward_with_offsets(&indices, None)
    }
}
