use crate::core::error::Result;
use crate::Module;
use crate::core::parameter::Parameter;
use tensor::Tensor;
use backend::Backend;
use storage::Storage;
use dtype::DataType;
use dtype::traits::FloatExt;

/// Kullback-Leibler divergence loss module.
#[derive(Debug, Default, Clone)]
pub struct KLDivLoss;

impl KLDivLoss {
    pub fn new() -> Self {
        Self
    }
}

impl<B, S, T> Module<B, S, T> for KLDivLoss
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + 'static,
    S: Storage<T> + storage::StorageToDense<T> + storage::StorageFromVec<T> + Clone + Send + Sync + 'static + tensor::ops::dispatch::TensorStorageOps<T>,
    T: DataType + FloatExt + num_traits::FromPrimitive + Copy + Send + Sync + 'static,
{
    type Input = (Tensor<B, S, T>, Tensor<B, S, T>);
    type Output = Tensor<B, S, T>;

    fn forward(&self, input: &(Tensor<B, S, T>, Tensor<B, S, T>)) -> Result<Tensor<B, S, T>> {
        crate::ops::loss::kl_div_loss(&input.0, &input.1)
    }

    fn parameters(&self) -> Vec<Parameter<B, S, T>> {
        vec![]
    }

    fn zero_grad(&mut self) {}

    fn train(&mut self, _mode: bool) {}

    fn name(&self) -> &str {
        "KLDivLoss"
    }

    fn clone_box(&self) -> Box<dyn Module<B, S, T, Input = Self::Input, Output = Self::Output>> {
        Box::new(self.clone())
    }
}
