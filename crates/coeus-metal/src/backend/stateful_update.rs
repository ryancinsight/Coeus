use super::{MetalBackend, MetalProvider};
use coeus_core::Layout;
use coeus_hephaestus::{HephaestusBackend, HephaestusStorage};
use coeus_ops::OptimizerOps;

type Storage = HephaestusStorage<MetalProvider, f32>;

impl OptimizerOps<f32> for MetalBackend {
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
        HephaestusBackend::<MetalProvider>::new().sgd_step(p, pl, g, gl, s, sl, lr, momentum)
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
        HephaestusBackend::<MetalProvider>::new()
            .adam_step(p, pl, g, gl, a, al, b, bl, lr, b1, b2, eps, step)
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
        HephaestusBackend::<MetalProvider>::new().rmsprop_step(p, pl, g, gl, s, sl, lr, alpha, eps)
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
        HephaestusBackend::<MetalProvider>::new()
            .adamw_step(p, pl, g, gl, a, al, b, bl, lr, b1, b2, eps, decay, step)
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
        HephaestusBackend::<MetalProvider>::new().adagrad_step(p, pl, g, gl, s, sl, lr, eps)
    }
}
