use crate::backend::{WgpuBackend, WgpuBackendError};
use coeus_core::Layout;
use coeus_hephaestus::{StatefulUpdateBackend, StatefulUpdateProvider};
use hephaestus_core::{ComputeDevice, HephaestusError};
use hephaestus_wgpu::{WgpuDevice, WgpuStatefulUpdateOps};

impl StatefulUpdateProvider for WgpuBackend {
    type Operations = WgpuStatefulUpdateOps;
}

impl StatefulUpdateBackend for WgpuBackend {
    type Provider = Self;

    fn stateful_update_buffer<T: coeus_core::Scalar>(
        storage: &Self::DeviceBuffer<T>,
    ) -> &<WgpuDevice as ComputeDevice>::Buffer<T> {
        storage.buffer()
    }

    fn stateful_update_error(operation: &'static str, source: HephaestusError) -> Self::Error {
        WgpuBackendError::dispatch(operation, source)
    }
}

impl coeus_ops::OptimizerOps<f32> for WgpuBackend {
    fn validate_optimizer_step(
        &self,
        validation: coeus_ops::OptimizerStepValidation<'_, f32, Self>,
    ) -> Result<(), Self::Error> {
        StatefulUpdateBackend::validate_optimizer_step(self, validation)
    }

    fn sgd_step(
        &self,
        p: &mut coeus_hephaestus::HephaestusStorage<crate::WgpuBackend, f32>,
        pl: &Layout,
        g: &coeus_hephaestus::HephaestusStorage<crate::WgpuBackend, f32>,
        gl: &Layout,
        s: &mut coeus_hephaestus::HephaestusStorage<crate::WgpuBackend, f32>,
        sl: &Layout,
        lr: f32,
        momentum: f32,
    ) -> Result<(), Self::Error> {
        self.dispatch_sgd_step(p, pl, g, gl, s, sl, lr, momentum)
    }

    fn adam_step(
        &self,
        p: &mut coeus_hephaestus::HephaestusStorage<crate::WgpuBackend, f32>,
        pl: &Layout,
        g: &coeus_hephaestus::HephaestusStorage<crate::WgpuBackend, f32>,
        gl: &Layout,
        first: &mut coeus_hephaestus::HephaestusStorage<crate::WgpuBackend, f32>,
        fl: &Layout,
        second: &mut coeus_hephaestus::HephaestusStorage<crate::WgpuBackend, f32>,
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
        p: &mut coeus_hephaestus::HephaestusStorage<crate::WgpuBackend, f32>,
        pl: &Layout,
        g: &coeus_hephaestus::HephaestusStorage<crate::WgpuBackend, f32>,
        gl: &Layout,
        s: &mut coeus_hephaestus::HephaestusStorage<crate::WgpuBackend, f32>,
        sl: &Layout,
        lr: f32,
        alpha: f32,
        eps: f32,
    ) -> Result<(), Self::Error> {
        self.dispatch_rmsprop_step(p, pl, g, gl, s, sl, lr, alpha, eps)
    }

    fn adamw_step(
        &self,
        p: &mut coeus_hephaestus::HephaestusStorage<crate::WgpuBackend, f32>,
        pl: &Layout,
        g: &coeus_hephaestus::HephaestusStorage<crate::WgpuBackend, f32>,
        gl: &Layout,
        first: &mut coeus_hephaestus::HephaestusStorage<crate::WgpuBackend, f32>,
        fl: &Layout,
        second: &mut coeus_hephaestus::HephaestusStorage<crate::WgpuBackend, f32>,
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
        p: &mut coeus_hephaestus::HephaestusStorage<crate::WgpuBackend, f32>,
        pl: &Layout,
        g: &coeus_hephaestus::HephaestusStorage<crate::WgpuBackend, f32>,
        gl: &Layout,
        s: &mut coeus_hephaestus::HephaestusStorage<crate::WgpuBackend, f32>,
        sl: &Layout,
        lr: f32,
        eps: f32,
    ) -> Result<(), Self::Error> {
        self.dispatch_adagrad_step(p, pl, g, gl, s, sl, lr, eps)
    }
}

impl coeus_ops::OptimizerOps<f64> for WgpuBackend {
    fn validate_optimizer_step(
        &self,
        validation: coeus_ops::OptimizerStepValidation<'_, f64, Self>,
    ) -> Result<(), Self::Error> {
        StatefulUpdateBackend::validate_optimizer_step_f64(self, validation)
    }

    fn sgd_step(
        &self,
        p: &mut coeus_hephaestus::HephaestusStorage<crate::WgpuBackend, f64>,
        pl: &Layout,
        g: &coeus_hephaestus::HephaestusStorage<crate::WgpuBackend, f64>,
        gl: &Layout,
        s: &mut coeus_hephaestus::HephaestusStorage<crate::WgpuBackend, f64>,
        sl: &Layout,
        lr: f64,
        momentum: f64,
    ) -> Result<(), Self::Error> {
        self.dispatch_sgd_step_f64(p, pl, g, gl, s, sl, lr, momentum)
    }

    fn adam_step(
        &self,
        p: &mut coeus_hephaestus::HephaestusStorage<crate::WgpuBackend, f64>,
        pl: &Layout,
        g: &coeus_hephaestus::HephaestusStorage<crate::WgpuBackend, f64>,
        gl: &Layout,
        first: &mut coeus_hephaestus::HephaestusStorage<crate::WgpuBackend, f64>,
        fl: &Layout,
        second: &mut coeus_hephaestus::HephaestusStorage<crate::WgpuBackend, f64>,
        sl: &Layout,
        lr: f64,
        b1: f64,
        b2: f64,
        eps: f64,
        step: usize,
    ) -> Result<(), Self::Error> {
        self.dispatch_adam_step_f64(p, pl, g, gl, first, fl, second, sl, lr, b1, b2, eps, step)
    }

    fn rmsprop_step(
        &self,
        p: &mut coeus_hephaestus::HephaestusStorage<crate::WgpuBackend, f64>,
        pl: &Layout,
        g: &coeus_hephaestus::HephaestusStorage<crate::WgpuBackend, f64>,
        gl: &Layout,
        s: &mut coeus_hephaestus::HephaestusStorage<crate::WgpuBackend, f64>,
        sl: &Layout,
        lr: f64,
        alpha: f64,
        eps: f64,
    ) -> Result<(), Self::Error> {
        self.dispatch_rmsprop_step_f64(p, pl, g, gl, s, sl, lr, alpha, eps)
    }

    fn adamw_step(
        &self,
        p: &mut coeus_hephaestus::HephaestusStorage<crate::WgpuBackend, f64>,
        pl: &Layout,
        g: &coeus_hephaestus::HephaestusStorage<crate::WgpuBackend, f64>,
        gl: &Layout,
        first: &mut coeus_hephaestus::HephaestusStorage<crate::WgpuBackend, f64>,
        fl: &Layout,
        second: &mut coeus_hephaestus::HephaestusStorage<crate::WgpuBackend, f64>,
        sl: &Layout,
        lr: f64,
        b1: f64,
        b2: f64,
        eps: f64,
        decay: f64,
        step: usize,
    ) -> Result<(), Self::Error> {
        self.dispatch_adamw_step_f64(
            p, pl, g, gl, first, fl, second, sl, lr, b1, b2, eps, decay, step,
        )
    }

    fn adagrad_step(
        &self,
        p: &mut coeus_hephaestus::HephaestusStorage<crate::WgpuBackend, f64>,
        pl: &Layout,
        g: &coeus_hephaestus::HephaestusStorage<crate::WgpuBackend, f64>,
        gl: &Layout,
        s: &mut coeus_hephaestus::HephaestusStorage<crate::WgpuBackend, f64>,
        sl: &Layout,
        lr: f64,
        eps: f64,
    ) -> Result<(), Self::Error> {
        self.dispatch_adagrad_step_f64(p, pl, g, gl, s, sl, lr, eps)
    }
}
