use super::{StatefulUpdateBackend, StatefulUpdateProvider};
use crate::HephaestusBackend;
use coeus_core::Layout;

impl<P> coeus_ops::OptimizerOps<f32> for HephaestusBackend<P>
where
    P: StatefulUpdateProvider,
{
    fn sgd_step(
        &self,
        p: &mut Self::DeviceBuffer<f32>,
        pl: &Layout,
        g: &Self::DeviceBuffer<f32>,
        gl: &Layout,
        s: &mut Self::DeviceBuffer<f32>,
        sl: &Layout,
        lr: f32,
        momentum: f32,
    ) -> Result<(), Self::Error> {
        self.dispatch_sgd_step(p, pl, g, gl, s, sl, lr, momentum)
    }

    fn adam_step(
        &self,
        p: &mut Self::DeviceBuffer<f32>,
        pl: &Layout,
        g: &Self::DeviceBuffer<f32>,
        gl: &Layout,
        first: &mut Self::DeviceBuffer<f32>,
        fl: &Layout,
        second: &mut Self::DeviceBuffer<f32>,
        sl: &Layout,
        lr: f32,
        beta_one: f32,
        beta_two: f32,
        epsilon: f32,
        step: usize,
    ) -> Result<(), Self::Error> {
        self.dispatch_adam_step(
            p, pl, g, gl, first, fl, second, sl, lr, beta_one, beta_two, epsilon, step,
        )
    }

    fn rmsprop_step(
        &self,
        p: &mut Self::DeviceBuffer<f32>,
        pl: &Layout,
        g: &Self::DeviceBuffer<f32>,
        gl: &Layout,
        state: &mut Self::DeviceBuffer<f32>,
        sl: &Layout,
        lr: f32,
        alpha: f32,
        epsilon: f32,
    ) -> Result<(), Self::Error> {
        self.dispatch_rmsprop_step(p, pl, g, gl, state, sl, lr, alpha, epsilon)
    }

    fn adamw_step(
        &self,
        p: &mut Self::DeviceBuffer<f32>,
        pl: &Layout,
        g: &Self::DeviceBuffer<f32>,
        gl: &Layout,
        first: &mut Self::DeviceBuffer<f32>,
        fl: &Layout,
        second: &mut Self::DeviceBuffer<f32>,
        sl: &Layout,
        lr: f32,
        beta_one: f32,
        beta_two: f32,
        epsilon: f32,
        weight_decay: f32,
        step: usize,
    ) -> Result<(), Self::Error> {
        self.dispatch_adamw_step(
            p,
            pl,
            g,
            gl,
            first,
            fl,
            second,
            sl,
            lr,
            beta_one,
            beta_two,
            epsilon,
            weight_decay,
            step,
        )
    }

    fn adagrad_step(
        &self,
        p: &mut Self::DeviceBuffer<f32>,
        pl: &Layout,
        g: &Self::DeviceBuffer<f32>,
        gl: &Layout,
        state: &mut Self::DeviceBuffer<f32>,
        sl: &Layout,
        lr: f32,
        epsilon: f32,
    ) -> Result<(), Self::Error> {
        self.dispatch_adagrad_step(p, pl, g, gl, state, sl, lr, epsilon)
    }
}

impl<P> StatefulUpdateBackend for HephaestusBackend<P>
where
    P: StatefulUpdateProvider,
{
    type Provider = P;

    fn stateful_update_buffer(
        storage: &Self::DeviceBuffer<f32>,
    ) -> &<P::Device as hephaestus_core::ComputeDevice>::Buffer<f32> {
        storage.buffer()
    }

    fn stateful_update_error(
        operation: &'static str,
        source: hephaestus_core::HephaestusError,
    ) -> Self::Error {
        crate::HephaestusBackendError::device(operation, source)
    }
}
