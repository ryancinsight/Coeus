use coeus_core::{Float, MoiraiBackend};
use coeus_autograd::Var;
use coeus_tensor::Tensor;
use crate::traits::Optimizer;

/// SGD with optional momentum.
pub struct SGD<T: Float, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend> {
    /// Parameters to optimize.
    pub params: Vec<Var<T, B>>,
    /// Learning rate.
    pub lr: T,
    /// Momentum coefficient (0 = no momentum).
    pub momentum: T,
    /// Velocity buffers (for momentum).
    velocity: Vec<Tensor<T, B>>,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> SGD<T, B> {
    /// Create SGD optimizer.
    pub fn new(params: Vec<Var<T, B>>, lr: T, momentum: T) -> Self {
        let backend = B::default();
        let velocity = params.iter()
            .map(|p| Tensor::zeros_on(p.tensor.shape(), &backend))
            .collect();
        Self {
            params,
            lr,
            momentum,
            velocity,
        }
    }
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Optimizer<T, B> for SGD<T, B> {
    fn step(&mut self) {
        let backend = B::default();

        for (i, param) in self.params.iter_mut().enumerate() {
            if let Some(ref g) = param.grad {
                let grad_tensor = g.lock().unwrap();
                let velocity_tensor = &mut self.velocity[i];

                let (param_storage, param_layout) = param.tensor.storage_mut_and_layout();
                let (velocity_storage, velocity_layout) = velocity_tensor.storage_mut_and_layout();

                backend.sgd_step(
                    param_storage,
                    param_layout,
                    grad_tensor.storage(),
                    grad_tensor.layout(),
                    velocity_storage,
                    velocity_layout,
                    self.lr,
                    self.momentum,
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
