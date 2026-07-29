// ── Regularization modules (G-041) ──
//
// AlphaDropout, FeatureAlphaDropout, GaussianNoise, LocalResponseNorm.
// All are parameter-free regularization layers matching PyTorch's API surface.

use crate::module::{Module, ModuleError};
use coeus_autograd::Var;
use coeus_core::Float;
use coeus_tensor::Tensor;

// ── AlphaDropout ──────────────────────────────────────────────────────────────

/// Alpha-dropout: a dropout variant designed for SELU activation networks.
///
/// During training, replaces dropped elements with the negative saturation value
/// `-alpha * lambda` and rescales the remaining elements so that the expected
/// value is preserved. `alpha` = 1.6732632423543772026 (SELU ELU parameter)
/// and `lambda` = 1.0507009873554804934 (SELU scale parameter).
///
/// In eval mode (or when `p == 0`) the layer is an identity.
#[derive(Clone, Debug)]
pub struct AlphaDropout {
    /// Dropout probability (elements zeroed per forward call).
    pub p: f64,
    /// Whether the layer is in training mode.
    pub is_training: bool,
    /// RNG seed for Bernoulli sampling.
    pub seed: u64,
}

impl AlphaDropout {
    /// SELU ELU-parameter (α).
    const SELU_ALPHA: f64 = 1.673_263_242_354_377_2;
    /// SELU scale-parameter (λ).
    const SELU_LAMBDA: f64 = 1.050_700_987_355_480_5;

    /// Create a new AlphaDropout layer.
    pub fn new(p: f64) -> Self {
        assert!(p >= 0.0 && p < 1.0, "p must be in [0, 1)");
        Self {
            p,
            is_training: true,
            seed: 42,
        }
    }

    /// Set training/eval mode.
    pub fn set_training(&mut self, mode: bool) {
        self.is_training = mode;
    }
}

fn alpha_dropout_with_mask<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    p: f64,
    seed: u64,
    feature_wise: bool,
) -> Result<Var<T, B>, ModuleError<B::Error>> {
    let shape = input.tensor.shape_cloned();
    if feature_wise && shape.len() < 2 {
        return Err(ModuleError::InvalidRank {
            module: "FeatureAlphaDropout",
            expected: "at least 2",
            actual: shape.len(),
        });
    }

    let numel = shape.iter().product::<usize>();
    let spatial = shape.get(2..).map_or(1, |dims| dims.iter().product());
    let channels = shape.get(1).copied().unwrap_or(1);
    let mut rng = coeus_autograd::ops::nn::dropout::Xorshift64::new(seed);
    let mut keep = Vec::with_capacity(numel);

    if feature_wise {
        for _batch in 0..shape[0] {
            for _channel in 0..channels {
                let kept = rng.next_f64() >= p;
                keep.extend(std::iter::repeat_n(
                    if kept { T::one() } else { T::zero() },
                    spatial,
                ));
            }
        }
    } else {
        keep.extend((0..numel).map(|_| {
            if rng.next_f64() >= p {
                T::one()
            } else {
                T::zero()
            }
        }));
    }

    let alpha_prime = -(AlphaDropout::SELU_ALPHA * AlphaDropout::SELU_LAMBDA);
    let keep_probability = 1.0 - p;
    let scale = 1.0 / (keep_probability * (1.0 + p * alpha_prime * alpha_prime)).sqrt();
    let shift = -scale * p * alpha_prime;
    let saturation = T::from_f64(alpha_prime);
    let backend = B::default();
    let keep_tensor = Tensor::from_slice_on(shape.clone(), &keep, &backend);
    let saturation_data = keep
        .iter()
        .map(|&value| {
            if value == T::zero() {
                saturation
            } else {
                T::zero()
            }
        })
        .collect::<Vec<_>>();
    let keep_var = Var::new(keep_tensor, false);
    let saturation_var = Var::new(
        Tensor::from_slice_on(shape, &saturation_data, &backend),
        false,
    );
    let selected = coeus_autograd::add(&coeus_autograd::mul(input, &keep_var), &saturation_var);
    Ok(coeus_autograd::scalar_add(
        &coeus_autograd::scalar_mul(&selected, T::from_f64(scale)),
        T::from_f64(shift),
    ))
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for AlphaDropout {
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    fn train(&mut self, mode: bool) {
        self.set_training(mode);
    }

    /// Alpha-dropout forward pass.
    ///
    /// In training mode each element is independently dropped with probability `p`.
    /// Dropped elements are replaced with the SELU saturation value
    /// `α' = -λ·α ≈ -1.7581`.  A single affine shift `(a, b)` is then applied
    /// element-wise so that the output has the same mean and variance as the
    /// input: `a = 1/sqrt(1 - p*(1-α'^2*(1-p^2)))`, `b = -a*(p*α' + 0)`.
    fn forward(&self, input: &Var<T, B>) -> Result<Var<T, B>, ModuleError<B::Error>> {
        if !self.is_training || self.p == 0.0 {
            return Ok(input.clone());
        }
        alpha_dropout_with_mask(input, self.p, self.seed, false)
    }
}

// ── FeatureAlphaDropout ───────────────────────────────────────────────────────

/// Feature alpha-dropout: channel-wise alpha-dropout for 2D+ tensors.
///
/// Same as `AlphaDropout` but drops entire feature maps (channels) rather
/// than individual elements, matching `torch.nn.FeatureAlphaDropout`.
#[derive(Clone, Debug)]
pub struct FeatureAlphaDropout {
    /// Dropout probability per feature map.
    pub p: f64,
    /// Training mode flag.
    pub is_training: bool,
    /// RNG seed.
    pub seed: u64,
}

impl FeatureAlphaDropout {
    /// Create a new FeatureAlphaDropout layer.
    pub fn new(p: f64) -> Self {
        assert!(p >= 0.0 && p < 1.0, "p must be in [0, 1)");
        Self {
            p,
            is_training: true,
            seed: 42,
        }
    }

    /// Set training/eval mode.
    pub fn set_training(&mut self, mode: bool) {
        self.is_training = mode;
    }
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for FeatureAlphaDropout {
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    fn train(&mut self, mode: bool) {
        self.set_training(mode);
    }

    /// Feature alpha-dropout forward: same as AlphaDropout but applied channel-wise.
    fn forward(&self, input: &Var<T, B>) -> Result<Var<T, B>, ModuleError<B::Error>> {
        if !self.is_training || self.p == 0.0 {
            return Ok(input.clone());
        }
        alpha_dropout_with_mask(input, self.p, self.seed, true)
    }
}

// ── GaussianNoise ─────────────────────────────────────────────────────────────

/// Gaussian noise regularisation layer.
///
/// During training, adds i.i.d. Gaussian noise `N(0, std^2)` element-wise.
/// In eval mode the layer is an identity (no noise added).
#[derive(Clone, Debug)]
pub struct GaussianNoise {
    /// Standard deviation of the injected Gaussian noise.
    pub std: f64,
    /// Training mode flag.
    pub is_training: bool,
}

impl GaussianNoise {
    /// Create a GaussianNoise layer with given standard deviation.
    pub fn new(std: f64) -> Self {
        assert!(std >= 0.0, "std must be non-negative");
        Self {
            std,
            is_training: true,
        }
    }

    /// Set training/eval mode.
    pub fn set_training(&mut self, mode: bool) {
        self.is_training = mode;
    }
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for GaussianNoise {
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    fn train(&mut self, mode: bool) {
        self.set_training(mode);
    }

    /// Add Gaussian noise during training; identity during evaluation.
    fn forward(&self, input: &Var<T, B>) -> Result<Var<T, B>, ModuleError<B::Error>> {
        if !self.is_training || self.std == 0.0 {
            return Ok(input.clone());
        }
        let backend = B::default();
        let shape = input.tensor.shape_cloned();
        let numel = shape.iter().product::<usize>();

        // Box-Muller transform: two uniform samples → one normal sample.
        let mut rng = coeus_autograd::ops::nn::dropout::Xorshift64::new(42);
        let mut noise = vec![T::zero(); numel];
        let std = self.std;
        let mut i = 0;
        while i < numel {
            let u1 = rng.next_f64().max(1e-10);
            let u2 = rng.next_f64();
            let r = (-2.0 * u1.ln()).sqrt() * std;
            noise[i] = T::from_f64(r * (2.0 * std::f64::consts::PI * u2).cos());
            if i + 1 < numel {
                noise[i + 1] = T::from_f64(r * (2.0 * std::f64::consts::PI * u2).sin());
            }
            i += 2;
        }

        let noise_tensor = Tensor::from_slice_on(shape, &noise, &backend);
        let noise_var = Var::new(noise_tensor, false);
        Ok(coeus_autograd::add(input, &noise_var))
    }
}

// ── LocalResponseNorm ─────────────────────────────────────────────────────────

/// Local Response Normalization (LRN).
///
/// Normalises within a local neighbourhood across channels:
/// `b[n,c,h,w] = a[n,c,h,w] / (k + alpha/size * sum(a[n,j,h,w]^2))^beta`
/// where `j` runs over `[max(0,c-size/2), min(C,c+size/2+1))`.
///
/// Matches `torch.nn.LocalResponseNorm(size, alpha, beta, k)`.
#[derive(Clone, Debug)]
pub struct LocalResponseNorm {
    /// Number of adjacent channels to normalise across.
    pub size: usize,
    /// Scaling factor in the denominator.
    pub alpha: f64,
    /// Exponent.
    pub beta: f64,
    /// Additive constant for numerical stability.
    pub k: f64,
}

impl LocalResponseNorm {
    /// Create a LocalResponseNorm layer with PyTorch defaults.
    ///
    /// `torch.nn.LocalResponseNorm(size)` uses `alpha=0.0001`, `beta=0.75`, `k=1.0`.
    pub fn new(size: usize) -> Self {
        Self {
            size,
            alpha: 0.0001,
            beta: 0.75,
            k: 1.0,
        }
    }

    /// Create with full hyperparameter control.
    pub fn with_params(size: usize, alpha: f64, beta: f64, k: f64) -> Self {
        assert!(size >= 1, "size must be >= 1");
        Self {
            size,
            alpha,
            beta,
            k,
        }
    }
}

impl<T: Float + std::ops::Neg<Output = T>, B: coeus_ops::BackendOps<T> + Default> Module<T, B>
    for LocalResponseNorm
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    /// LRN forward pass — differentiable.
    ///
    /// Supports 2D (`[N, C]`), 3D (`[N, C, L]`), and 4D (`[N, C, H, W]`) inputs.
    /// Built entirely from autograd ops so gradients flow to the input: the
    /// cross-channel windowed sum-of-squares is a constant band-matrix product
    /// (differentiable through the squared activations), and the `^beta`
    /// denominator uses the differentiable `pow` (the base `k + .. >= k > 0`,
    /// so the `exp(beta*ln(.))` it computes is well defined).
    fn forward(&self, input: &Var<T, B>) -> Result<Var<T, B>, ModuleError<B::Error>> {
        let shape = input.tensor.shape_cloned();
        if !(2..=4).contains(&shape.len()) {
            return Err(ModuleError::InvalidRank {
                module: "LocalResponseNorm",
                expected: "2 to 4",
                actual: shape.len(),
            });
        }
        let n = shape[0];
        let c = shape[1];
        let spatial: usize = shape[2..].iter().product::<usize>().max(1);
        let half = self.size / 2;
        let backend = B::default();

        // View as [N, C, spatial] (channel axis = dim 1); square it.
        let x3 = coeus_autograd::reshape(input, [n, c, spatial]);
        let sq = coeus_autograd::mul(&x3, &x3);

        // Constant band matrix M [C, C], M[i, j] = 1 iff |i - j| <= half. Then
        // `M @ sq` over the channel axis is each channel's squared response
        // summed across its size-neighbourhood (with boundary clamping implicit
        // in the band), differentiable through `sq`.
        let mut m_data = vec![T::zero(); c * c];
        for i in 0..c {
            let lo = i.saturating_sub(half);
            let hi = (i + half + 1).min(c);
            for cell in m_data[i * c + lo..i * c + hi].iter_mut() {
                *cell = T::one();
            }
        }
        let m = Var::new(Tensor::from_slice_on([c, c], &m_data, &backend), false);

        // windowed = M @ sq:  [N,C,S] -> [C,N*S] -> M@ -> [C,N*S] -> [N,C,S].
        let sq_cns = coeus_autograd::permute(&sq, &[1, 0, 2]);
        let sq_2d = coeus_autograd::reshape(&sq_cns, [c, n * spatial]);
        let win_2d = coeus_autograd::matmul(&m, &sq_2d);
        let win_cns = coeus_autograd::reshape(&win_2d, [c, n, spatial]);
        let windowed = coeus_autograd::permute(&win_cns, &[1, 0, 2]);

        // denom = (k + (alpha / size) * windowed)^beta;  y = x / denom.
        let scaled =
            coeus_autograd::scalar_mul(&windowed, T::from_f64(self.alpha / self.size as f64));
        let denom = coeus_autograd::pow(
            &coeus_autograd::scalar_add(&scaled, T::from_f64(self.k)),
            self.beta,
        );
        let y3 = coeus_autograd::div(&x3, &denom);
        Ok(coeus_autograd::reshape(&y3, shape))
    }
}
