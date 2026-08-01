use super::{RocmBackend, RocmProvider};
use coeus_core::Layout;
use coeus_hephaestus::{HephaestusBackendError, HephaestusStorage, StatefulUpdateBackend};
use coeus_ops::{OptimizerOps, OptimizerStepValidation};
use hephaestus_core::{ComputeDevice, HephaestusError};

type Storage = HephaestusStorage<RocmProvider, f32>;

impl StatefulUpdateBackend for RocmBackend {
    type Provider = RocmProvider;

    fn stateful_update_buffer(
        storage: &Self::DeviceBuffer<f32>,
    ) -> &<hephaestus_rocm::RocmDevice as ComputeDevice>::Buffer<f32> {
        storage.buffer()
    }

    fn stateful_update_error(operation: &'static str, source: HephaestusError) -> Self::Error {
        HephaestusBackendError::Device { operation, source }
    }
}

impl OptimizerOps<f32> for RocmBackend {
    fn validate_optimizer_step(
        &self,
        validation: OptimizerStepValidation<'_, f32, Self>,
    ) -> Result<(), Self::Error> {
        StatefulUpdateBackend::validate_optimizer_step(self, validation)
    }

    fn sgd_step(
        &self,
        p: &mut Storage,
        pl: &Layout,
        g: &Storage,
        gl: &Layout,
        s: &mut Storage,
        sl: &Layout,
        lr: f32,
        momentum: f32,
    ) -> Result<(), Self::Error> {
        self.dispatch_sgd_step(p, pl, g, gl, s, sl, lr, momentum)
    }

    fn adam_step(
        &self,
        p: &mut Storage,
        pl: &Layout,
        g: &Storage,
        gl: &Layout,
        a: &mut Storage,
        al: &Layout,
        b: &mut Storage,
        bl: &Layout,
        lr: f32,
        b1: f32,
        b2: f32,
        eps: f32,
        step: usize,
    ) -> Result<(), Self::Error> {
        self.dispatch_adam_step(p, pl, g, gl, a, al, b, bl, lr, b1, b2, eps, step)
    }

    fn rmsprop_step(
        &self,
        p: &mut Storage,
        pl: &Layout,
        g: &Storage,
        gl: &Layout,
        s: &mut Storage,
        sl: &Layout,
        lr: f32,
        alpha: f32,
        eps: f32,
    ) -> Result<(), Self::Error> {
        self.dispatch_rmsprop_step(p, pl, g, gl, s, sl, lr, alpha, eps)
    }

    fn adamw_step(
        &self,
        p: &mut Storage,
        pl: &Layout,
        g: &Storage,
        gl: &Layout,
        a: &mut Storage,
        al: &Layout,
        b: &mut Storage,
        bl: &Layout,
        lr: f32,
        b1: f32,
        b2: f32,
        eps: f32,
        decay: f32,
        step: usize,
    ) -> Result<(), Self::Error> {
        self.dispatch_adamw_step(p, pl, g, gl, a, al, b, bl, lr, b1, b2, eps, decay, step)
    }

    fn adagrad_step(
        &self,
        p: &mut Storage,
        pl: &Layout,
        g: &Storage,
        gl: &Layout,
        s: &mut Storage,
        sl: &Layout,
        lr: f32,
        eps: f32,
    ) -> Result<(), Self::Error> {
        self.dispatch_adagrad_step(p, pl, g, gl, s, sl, lr, eps)
    }
}
