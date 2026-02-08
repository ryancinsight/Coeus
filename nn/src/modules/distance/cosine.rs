use crate::core::error::Result;
use crate::Module;
use crate::core::parameter::Parameter;
use tensor::Tensor;
use backend::Backend;
use storage::Storage;
use dtype::DataType;
use dtype::traits::FloatExt;

/// Cosine Similarity module.
#[derive(Debug, Clone)]
pub struct CosineSimilarity {
    pub dim: usize,
    pub eps: f64,
}

impl Default for CosineSimilarity {
    fn default() -> Self {
        Self { dim: 1, eps: 1e-8 }
    }
}

impl CosineSimilarity {
    pub fn new(dim: usize, eps: f64) -> Self {
        Self { dim, eps }
    }
}

impl<B, S, T> Module<B, S, T> for CosineSimilarity
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + 'static,
    S: Storage<T> + storage::StorageToDense<T> + storage::StorageFromVec<T> + Clone + Send + Sync + 'static + tensor::ops::dispatch::TensorStorageOps<T>,
    T: DataType + FloatExt + num_traits::FromPrimitive + Copy + Send + Sync + 'static,
{
    type Input = (Tensor<B, S, T>, Tensor<B, S, T>);
    type Output = Tensor<B, S, T>;

    fn forward(&self, input: &(Tensor<B, S, T>, Tensor<B, S, T>)) -> Result<Tensor<B, S, T>> {
        crate::ops::distance::cosine_similarity(&input.0, &input.1, self.dim, self.eps)
    }

    fn parameters(&self) -> Vec<Parameter<B, S, T>> {
        vec![]
    }

    fn zero_grad(&mut self) {}

    fn train(&mut self, _mode: bool) {}

    fn name(&self) -> &str {
        "CosineSimilarity"
    }

    fn clone_box(&self) -> Box<dyn Module<B, S, T, Input = Self::Input, Output = Self::Output>> {
        Box::new(self.clone())
    }
}
