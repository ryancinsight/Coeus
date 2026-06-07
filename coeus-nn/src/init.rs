// ── Weight initialization ──

use std::cell::RefCell;
use coeus_core::{Float, MoiraiBackend};
use coeus_autograd::Var;
use coeus_tensor::Tensor;

/// A simple, fast, deterministic pseudo-random number generator (Xorshift64).
pub struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    /// Create a new generator with a seed. Seed must be non-zero.
    #[inline]
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1337 } else { seed },
        }
    }

    /// Draw next u64 value.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Draw a float in [0.0, 1.0).
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() as f64) / (u64::MAX as f64)
    }

    /// Draw a float from a normal distribution N(mean, std_dev).
    #[inline]
    pub fn next_normal(&mut self, mean: f64, std_dev: f64) -> f64 {
        let u1 = self.next_f64().max(1e-15); // Avoid ln(0)
        let u2 = self.next_f64();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        mean + std_dev * z
    }
}

/// Initialize weights with values from a uniform distribution U(a, b).
pub fn uniform_with_seed<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    weight: &mut Var<T, B>,
    a: f64,
    b: f64,
    seed: u64,
) {
    let rng = RefCell::new(Xorshift64::new(seed));
    let shape = weight.tensor.shape_cloned();
    let cpu_backend = MoiraiBackend::new();
    let new_tensor_cpu = Tensor::<T, MoiraiBackend>::from_fn_on(shape, &cpu_backend, |_| {
        let val = rng.borrow_mut().next_f64() * (b - a) + a;
        T::from_f64(val)
    });
    let new_tensor = new_tensor_cpu.to_backend_on(&cpu_backend, &B::default());
    weight.tensor = new_tensor;
}

/// Initialize weights with values from a uniform distribution U(a, b) using default seed.
pub fn uniform<T: Float, B: coeus_ops::BackendOps<T> + Default>(weight: &mut Var<T, B>, a: f64, b: f64) {
    uniform_with_seed(weight, a, b, 42);
}

/// Initialize weights with values from a normal distribution N(mean, std_dev).
pub fn normal_with_seed<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    weight: &mut Var<T, B>,
    mean: f64,
    std_dev: f64,
    seed: u64,
) {
    let rng = RefCell::new(Xorshift64::new(seed));
    let shape = weight.tensor.shape_cloned();
    let cpu_backend = MoiraiBackend::new();
    let new_tensor_cpu = Tensor::<T, MoiraiBackend>::from_fn_on(shape, &cpu_backend, |_| {
        let val = rng.borrow_mut().next_normal(mean, std_dev);
        T::from_f64(val)
    });
    let new_tensor = new_tensor_cpu.to_backend_on(&cpu_backend, &B::default());
    weight.tensor = new_tensor;
}

/// Initialize weights with values from a normal distribution N(mean, std_dev) using default seed.
pub fn normal<T: Float, B: coeus_ops::BackendOps<T> + Default>(weight: &mut Var<T, B>, mean: f64, std_dev: f64) {
    normal_with_seed(weight, mean, std_dev, 42);
}

/// Initialize weights with a constant value.
pub fn constant<T: Float, B: coeus_ops::BackendOps<T> + Default>(weight: &mut Var<T, B>, val: f64) {
    let shape = weight.tensor.shape_cloned();
    weight.tensor = Tensor::full_on(shape, T::from_f64(val), &B::default());
}

/// Initialize weights with zeros.
pub fn zeros<T: Float, B: coeus_ops::BackendOps<T> + Default>(weight: &mut Var<T, B>) {
    let shape = weight.tensor.shape_cloned();
    weight.tensor = Tensor::zeros_on(shape, &B::default());
}

/// Initialize weights with ones.
pub fn ones<T: Float, B: coeus_ops::BackendOps<T> + Default>(weight: &mut Var<T, B>) {
    let shape = weight.tensor.shape_cloned();
    weight.tensor = Tensor::ones_on(shape, &B::default());
}

/// Xavier (Glorot) uniform initialization with custom seed.
pub fn xavier_uniform_with_seed<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    weight: &mut Var<T, B>,
    fan_in: usize,
    fan_out: usize,
    seed: u64,
) {
    let limit = (6.0f64 / (fan_in + fan_out) as f64).sqrt();
    uniform_with_seed(weight, -limit, limit, seed);
}

/// Xavier (Glorot) uniform initialization.
pub fn xavier_uniform<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    weight: &mut Var<T, B>,
    fan_in: usize,
    fan_out: usize,
) {
    xavier_uniform_with_seed(weight, fan_in, fan_out, 42);
}

/// Xavier (Glorot) normal initialization with custom seed.
pub fn xavier_normal_with_seed<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    weight: &mut Var<T, B>,
    fan_in: usize,
    fan_out: usize,
    seed: u64,
) {
    let std_dev = (2.0f64 / (fan_in + fan_out) as f64).sqrt();
    normal_with_seed(weight, 0.0, std_dev, seed);
}

/// Xavier (Glorot) normal initialization.
pub fn xavier_normal<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    weight: &mut Var<T, B>,
    fan_in: usize,
    fan_out: usize,
) {
    xavier_normal_with_seed(weight, fan_in, fan_out, 42);
}

/// Kaiming (He) uniform initialization with custom seed.
pub fn kaiming_uniform_with_seed<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    weight: &mut Var<T, B>,
    fan_in: usize,
    seed: u64,
) {
    let limit = (6.0f64 / fan_in as f64).sqrt();
    uniform_with_seed(weight, -limit, limit, seed);
}

/// Kaiming (He) uniform initialization.
pub fn kaiming_uniform<T: Float, B: coeus_ops::BackendOps<T> + Default>(weight: &mut Var<T, B>, fan_in: usize) {
    kaiming_uniform_with_seed(weight, fan_in, 42);
}

/// Kaiming (He) normal initialization with custom seed.
pub fn kaiming_normal_with_seed<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    weight: &mut Var<T, B>,
    fan_in: usize,
    seed: u64,
) {
    let std_dev = (2.0f64 / fan_in as f64).sqrt();
    normal_with_seed(weight, 0.0, std_dev, seed);
}

/// Kaiming (He) normal initialization.
pub fn kaiming_normal<T: Float, B: coeus_ops::BackendOps<T> + Default>(weight: &mut Var<T, B>, fan_in: usize) {
    kaiming_normal_with_seed(weight, fan_in, 42);
}
