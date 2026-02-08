use crate::core::error::Result;
use crate::Module;
use tensor::Tensor;
use backend::Backend;
use storage::Storage;
use dtype::DataType;
use dtype::traits::FloatExt;

/// Smooth L1 Loss module (also known as Huber loss).
#[derive(Debug, Clone)]
pub struct SmoothL1Loss {
    pub beta: f64,
}

impl Default for SmoothL1Loss {
    fn default() -> Self {
        Self { beta: 1.0 }
    }
}

impl SmoothL1Loss {
    pub fn new(beta: f64) -> Self {
        Self { beta }
    }
}

impl<B, S, T> Module<B, S, T> for SmoothL1Loss
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + 'static,
    S: Storage<T> + storage::StorageToDense<T> + storage::StorageFromVec<T> + Clone + Send + Sync + 'static + tensor::ops::dispatch::TensorStorageOps<T>,
    T: DataType + FloatExt + num_traits::FromPrimitive + Copy + Send + Sync + 'static,
{
    type Input = (Tensor<B, S, T>, Tensor<B, S, T>);
    type Output = Tensor<B, S, T>;

    fn forward(&self, input: &(Tensor<B, S, T>, Tensor<B, S, T>)) -> Result<Tensor<B, S, T>> {
        crate::ops::loss::smooth_l1_loss(&input.0, &input.1, self.beta)
    }

    fn parameters(&self) -> Vec<crate::core::parameter::Parameter<B, S, T>> {
        vec![]
    }

    fn zero_grad(&mut self) {}

    fn train(&mut self, _mode: bool) {}

    fn name(&self) -> &str {
        "SmoothL1Loss"
    }

    fn clone_box(&self) -> Box<dyn Module<B, S, T, Input = Self::Input, Output = Self::Output>> {
        Box::new(self.clone())
    }
}
