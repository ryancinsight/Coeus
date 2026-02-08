use crate::core::error::Result;
use crate::Module;
use crate::core::parameter::Parameter;
use tensor::Tensor;
use backend::Backend;
use storage::Storage;
use dtype::DataType;
use dtype::traits::FloatExt;

/// Pairwise Distance module.
#[derive(Debug, Clone)]
pub struct PairwiseDistance {
    pub p: f64,
    pub eps: f64,
    pub keepdim: bool,
}

impl Default for PairwiseDistance {
    fn default() -> Self {
        Self { p: 2.0, eps: 1e-6, keepdim: false }
    }
}

impl PairwiseDistance {
    pub fn new(p: f64, eps: f64, keepdim: bool) -> Self {
        Self { p, eps, keepdim }
    }
}

impl<B, S, T> Module<B, S, T> for PairwiseDistance
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + 'static,
    S: Storage<T> + storage::StorageToDense<T> + storage::StorageFromVec<T> + Clone + Send + Sync + 'static + tensor::ops::dispatch::TensorStorageOps<T>,
    T: DataType + FloatExt + num_traits::FromPrimitive + Copy + Send + Sync + 'static,
{
    type Input = (Tensor<B, S, T>, Tensor<B, S, T>);
    type Output = Tensor<B, S, T>;

    fn forward(&self, input: &(Tensor<B, S, T>, Tensor<B, S, T>)) -> Result<Tensor<B, S, T>> {
        crate::ops::distance::pairwise_distance(&input.0, &input.1, self.p, self.eps, self.keepdim)
    }

    fn parameters(&self) -> Vec<Parameter<B, S, T>> {
        vec![]
    }

    fn zero_grad(&mut self) {}

    fn train(&mut self, _mode: bool) {}

    fn name(&self) -> &str {
        "PairwiseDistance"
    }

    fn clone_box(&self) -> Box<dyn Module<B, S, T, Input = Self::Input, Output = Self::Output>> {
        Box::new(self.clone())
    }
}
