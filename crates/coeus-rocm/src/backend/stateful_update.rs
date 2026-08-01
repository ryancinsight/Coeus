use super::{RocmBackend, RocmProvider};
use coeus_core::Layout;
use coeus_hephaestus::{HephaestusBackend, HephaestusStorage};
use coeus_ops::OptimizerOps;

type Storage = HephaestusStorage<RocmProvider, f32>;

impl OptimizerOps<f32> for RocmBackend {
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
        HephaestusBackend::<RocmProvider>::new().sgd_step(p, pl, g, gl, s, sl, lr, momentum)
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
        HephaestusBackend::<RocmProvider>::new()
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
        HephaestusBackend::<RocmProvider>::new().rmsprop_step(p, pl, g, gl, s, sl, lr, alpha, eps)
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
        HephaestusBackend::<RocmProvider>::new()
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
        HephaestusBackend::<RocmProvider>::new().adagrad_step(p, pl, g, gl, s, sl, lr, eps)
    }
}
