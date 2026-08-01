use crate::backend::{CudaBackend, CudaStorage};
use crate::CudaBackendError;
use coeus_core::Layout;
use coeus_hephaestus::{StatefulUpdateBackend, StatefulUpdateProvider};
use hephaestus_core::{ComputeDevice, HephaestusError};
use hephaestus_cuda::{CudaDevice, CudaStatefulUpdateOps};

impl StatefulUpdateProvider for CudaBackend {
    type Operations = CudaStatefulUpdateOps;
}

impl StatefulUpdateBackend for CudaBackend {
    type Provider = Self;

    fn stateful_update_buffer(
        storage: &Self::DeviceBuffer<f32>,
    ) -> &<CudaDevice as ComputeDevice>::Buffer<f32> {
        storage.buffer.as_ref()
    }

    fn stateful_update_error(operation: &'static str, source: HephaestusError) -> Self::Error {
        CudaBackendError::dispatch(operation, source)
    }
}

impl coeus_ops::OptimizerOps<f32> for CudaBackend {
    fn sgd_step(
        &self,
        p: &mut CudaStorage<f32>,
        pl: &Layout,
        g: &CudaStorage<f32>,
        gl: &Layout,
        s: &mut CudaStorage<f32>,
        sl: &Layout,
        lr: f32,
        momentum: f32,
    ) -> Result<(), Self::Error> {
        self.dispatch_sgd_step(p, pl, g, gl, s, sl, lr, momentum)
    }

    fn adam_step(
        &self,
        p: &mut CudaStorage<f32>,
        pl: &Layout,
        g: &CudaStorage<f32>,
        gl: &Layout,
        first: &mut CudaStorage<f32>,
        fl: &Layout,
        second: &mut CudaStorage<f32>,
        sl: &Layout,
        lr: f32,
        b1: f32,
        b2: f32,
        eps: f32,
        step: usize,
    ) -> Result<(), Self::Error> {
        self.dispatch_adam_step(p, pl, g, gl, first, fl, second, sl, lr, b1, b2, eps, step)
    }

    fn rmsprop_step(
        &self,
        p: &mut CudaStorage<f32>,
        pl: &Layout,
        g: &CudaStorage<f32>,
        gl: &Layout,
        s: &mut CudaStorage<f32>,
        sl: &Layout,
        lr: f32,
        alpha: f32,
        eps: f32,
    ) -> Result<(), Self::Error> {
        self.dispatch_rmsprop_step(p, pl, g, gl, s, sl, lr, alpha, eps)
    }

    fn adamw_step(
        &self,
        p: &mut CudaStorage<f32>,
        pl: &Layout,
        g: &CudaStorage<f32>,
        gl: &Layout,
        first: &mut CudaStorage<f32>,
        fl: &Layout,
        second: &mut CudaStorage<f32>,
        sl: &Layout,
        lr: f32,
        b1: f32,
        b2: f32,
        eps: f32,
        decay: f32,
        step: usize,
    ) -> Result<(), Self::Error> {
        self.dispatch_adamw_step(
            p, pl, g, gl, first, fl, second, sl, lr, b1, b2, eps, decay, step,
        )
    }

    fn adagrad_step(
        &self,
        p: &mut CudaStorage<f32>,
        pl: &Layout,
        g: &CudaStorage<f32>,
        gl: &Layout,
        s: &mut CudaStorage<f32>,
        sl: &Layout,
        lr: f32,
        eps: f32,
    ) -> Result<(), Self::Error> {
        self.dispatch_adagrad_step(p, pl, g, gl, s, sl, lr, eps)
    }
}
