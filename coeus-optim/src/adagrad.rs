use coeus_core::{Float, MoiraiBackend};
use coeus_autograd::Var;
use coeus_tensor::Tensor;
use crate::traits::Optimizer;

/// AdaGrad optimizer.
pub struct AdaGrad<T: Float, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend> {
    /// Parameters to optimize.
    pub params: Vec<Var<T, B>>,
    /// Learning rate.
    pub lr: T,
    /// Small term to avoid division by zero.
    pub eps: T,
    /// Sum of squares of historical gradients.
    history: Vec<Tensor<T, B>>,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> AdaGrad<T, B> {
    /// Create AdaGrad optimizer.
    pub fn new(params: Vec<Var<T, B>>, lr: T, eps: T) -> Self {
        let backend = B::default();
        let history = params.iter()
            .map(|p| Tensor::zeros_on(p.tensor.shape(), &backend))
            .collect();
        Self {
            params,
            lr,
            eps,
            history,
        }
    }
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Optimizer<T, B> for AdaGrad<T, B> {
    fn step(&mut self) {
        let backend = B::default();

        for (i, param) in self.params.iter_mut().enumerate() {
            if let Some(ref g) = param.grad {
                let grad_tensor = g.lock().unwrap();
                let history_tensor = &mut self.history[i];

                let (param_storage, param_layout) = param.tensor.storage_mut_and_layout();
                let (history_storage, history_layout) = history_tensor.storage_mut_and_layout();

                backend.adagrad_step(
                    param_storage,
                    param_layout,
                    grad_tensor.storage(),
                    grad_tensor.layout(),
                    history_storage,
                    history_layout,
                    self.lr,
                    self.eps,
                );
            }
        }
    }

    fn zero_grad(&mut self) {
        for p in &self.params {
            p.zero_grad();
        }
    }

    fn set_lr(&mut self, lr: T) {
        self.lr = lr;
    }

    fn clip_grad_norm(&mut self, max_norm: T) -> T
    where
        B::DeviceBuffer<T>: coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
    {
        crate::clip::clip_grad_norm(&self.params, max_norm)
    }
}
