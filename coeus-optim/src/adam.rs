use crate::traits::Optimizer;
use coeus_autograd::Var;
use coeus_core::{Float, MoiraiBackend};
use coeus_tensor::Tensor;

/// Adam optimizer.
pub struct Adam<T: Float, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend> {
    /// List of tracked parameters.
    pub params: Vec<Var<T, B>>,
    /// Learning rate.
    pub lr: T,
    /// Coefficient for first moment (momentum).
    pub beta1: T,
    /// Coefficient for second moment (uncentered variance).
    pub beta2: T,
    /// Small constant for numerical stability.
    pub eps: T,
    /// Current time step (for bias correction).
    pub t: usize,
    /// First moment vectors (m) matching the parameters shape.
    pub m: Vec<Tensor<T, B>>,
    /// Second moment vectors (v) matching the parameters shape.
    pub v: Vec<Tensor<T, B>>,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Adam<T, B> {
    /// Create a new Adam optimizer.
    pub fn new(params: Vec<Var<T, B>>, lr: T, beta1: T, beta2: T, eps: T) -> Self {
        let backend = B::default();
        let mut m = Vec::with_capacity(params.len());
        let mut v = Vec::with_capacity(params.len());
        for param in &params {
            m.push(Tensor::zeros_on(param.tensor.shape(), &backend));
            v.push(Tensor::zeros_on(param.tensor.shape(), &backend));
        }

        Self {
            params,
            lr,
            beta1,
            beta2,
            eps,
            t: 0,
            m,
            v,
        }
    }
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Optimizer<T, B> for Adam<T, B> {
    fn step(&mut self) {
        self.t += 1;
        let backend = B::default();

        for (i, param) in self.params.iter_mut().enumerate() {
            if let Some(ref g) = param.grad {
                let grad_tensor = g.read();
                let m_tensor = &mut self.m[i];
                let v_tensor = &mut self.v[i];

                let (param_storage, param_layout) = param.tensor.storage_mut_and_layout();
                let (m_storage, m_layout) = m_tensor.storage_mut_and_layout();
                let (v_storage, v_layout) = v_tensor.storage_mut_and_layout();

                backend.adam_step(
                    param_storage,
                    param_layout,
                    grad_tensor.storage(),
                    grad_tensor.layout(),
                    m_storage,
                    m_layout,
                    v_storage,
                    v_layout,
                    self.lr,
                    self.beta1,
                    self.beta2,
                    self.eps,
                    self.t,
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
        B::DeviceBuffer<T>:
            coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
    {
        crate::clip::clip_grad_norm(&self.params, max_norm)
    }
}
