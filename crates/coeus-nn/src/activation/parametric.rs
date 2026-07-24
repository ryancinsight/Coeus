use crate::module::Module;
use coeus_autograd::Var;
use coeus_core::Float;

/// Functional ELU activation.
#[inline]
pub fn elu<T: Float, B: coeus_ops::BackendOps<T> + Default>(input: &Var<T, B>) -> Var<T, B> {
    coeus_autograd::elu(input)
}

/// ELU activation module (alpha = 1.0).
#[derive(Clone, Debug, Default)]
pub struct ELU;

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for ELU {
    #[inline]
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    #[inline]
    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
        elu(input)
    }
}

/// Functional GELU tanh approximation.
#[inline]
pub fn gelu_tanh<T: Float, B: coeus_ops::BackendOps<T> + Default>(input: &Var<T, B>) -> Var<T, B> {
    coeus_autograd::gelu_tanh(input)
}

/// GELU tanh approximation module.
#[derive(Clone, Debug, Default)]
pub struct GeLUTanh;

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for GeLUTanh {
    #[inline]
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    #[inline]
    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
        gelu_tanh(input)
    }
}

/// Functional Gated Linear Unit along `dim` (`torch.nn.functional.glu`).
///
/// Splits `input` into two equal halves `[a, b]` along `dim` and gates the first
/// by the sigmoid of the second: `a * sigmoid(b)`. Differentiable — composed from
/// tracked `slice`, `sigmoid`, and `mul`, so gradients flow to `input`.
///
/// # Panics
/// If `dim` is out of range or the extent along `dim` is odd.
pub fn glu<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    dim: usize,
) -> Var<T, B> {
    let shape = input.tensor.shape();
    let ndim = shape.len();
    assert!(dim < ndim, "glu: dim {dim} out of range for rank {ndim}");
    let axis = shape[dim];
    assert!(
        axis.is_multiple_of(2),
        "glu: dim {dim} must have even extent, got {axis}"
    );
    let half = axis / 2;
    let mut first: Vec<(usize, usize)> = shape.iter().map(|&extent| (0, extent)).collect();
    let mut second = first.clone();
    first[dim] = (0, half);
    second[dim] = (half, axis);
    let a = coeus_autograd::slice(input, &first);
    let b = coeus_autograd::slice(input, &second);
    coeus_autograd::mul(&a, &coeus_autograd::sigmoid(&b))
}

/// Gated Linear Unit module gating along a fixed `dim` (see [`glu`]).
///
/// Parameter-free; the split dimension is captured at construction so `GLU` can
/// participate in a [`Sequential`](crate::Sequential) stack like other activations.
#[derive(Clone, Copy, Debug)]
pub struct GLU {
    dim: usize,
}

impl GLU {
    /// Create a GLU module that gates along `dim`.
    #[inline]
    #[must_use]
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for GLU {
    #[inline]
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    #[inline]
    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
        glu(input, self.dim)
    }
}

/// Functional LeakyReLU activation.
#[inline]
pub fn leaky_relu<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    negative_slope: f64,
) -> Var<T, B> {
    coeus_autograd::leaky_relu(input, negative_slope)
}

/// LeakyReLU activation module.
#[derive(Clone, Debug)]
pub struct LeakyReLU {
    /// Slope for negative inputs.
    pub negative_slope: f64,
}

impl LeakyReLU {
    /// Create a LeakyReLU module.
    pub fn new(negative_slope: f64) -> Self {
        Self { negative_slope }
    }
}

impl Default for LeakyReLU {
    fn default() -> Self {
        Self {
            negative_slope: 0.01,
        }
    }
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for LeakyReLU {
    #[inline]
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    #[inline]
    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
        leaky_relu(input, self.negative_slope)
    }
}

/// Public descriptor for the functional Hardtanh parameter set.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HardtanhOp {
    /// Lower clamp bound.
    pub min_val: f64,
    /// Upper clamp bound.
    pub max_val: f64,
}

impl HardtanhOp {
    /// Construct a Hardtanh descriptor from explicit bounds.
    #[must_use]
    pub const fn from_bounds(min_val: f64, max_val: f64) -> Self {
        Self { min_val, max_val }
    }
}

/// Functional Hardtanh activation.
#[inline]
pub fn hardtanh<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    min_val: f64,
    max_val: f64,
) -> Var<T, B> {
    coeus_autograd::hardtanh(input, min_val, max_val)
}

/// Hardtanh activation module (default range [-1, 1], matching PyTorch's
/// `torch.nn.Hardtanh`).
#[derive(Clone, Debug)]
pub struct Hardtanh {
    /// Lower clamp bound.
    pub min_val: f64,
    /// Upper clamp bound.
    pub max_val: f64,
}

impl Hardtanh {
    /// Create a Hardtanh module with custom bounds.
    pub fn new(min_val: f64, max_val: f64) -> Self {
        Self { min_val, max_val }
    }
}

impl Default for Hardtanh {
    fn default() -> Self {
        Self {
            min_val: -1.0,
            max_val: 1.0,
        }
    }
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for Hardtanh {
    #[inline]
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    #[inline]
    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
        hardtanh(input, self.min_val, self.max_val)
    }
}

/// Public descriptor for the functional Hardshrink parameter set.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HardshrinkOp(pub f64);

impl HardshrinkOp {
    /// Construct a Hardshrink descriptor from lambda.
    #[must_use]
    pub const fn from_lambda(lambda: f64) -> Self {
        Self(lambda)
    }
}

/// Functional Hardshrink activation.
#[inline]
pub fn hardshrink<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    lambda: f64,
) -> Var<T, B> {
    coeus_autograd::hardshrink(input, lambda)
}

/// Hardshrink activation module (default λ = 0.5, matching PyTorch's
/// `torch.nn.Hardshrink`).
#[derive(Clone, Debug)]
pub struct Hardshrink {
    /// Threshold λ below which the activation is replaced with 0.
    pub lambda: f64,
}

impl Hardshrink {
    /// Create a Hardshrink module with custom λ.
    pub fn new(lambda: f64) -> Self {
        Self { lambda }
    }
}

impl Default for Hardshrink {
    fn default() -> Self {
        Self { lambda: 0.5 }
    }
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for Hardshrink {
    #[inline]
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    #[inline]
    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
        hardshrink(input, self.lambda)
    }
}

/// Public descriptor for the functional Softshrink parameter set.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SoftshrinkOp(pub f64);

impl SoftshrinkOp {
    /// Construct a Softshrink descriptor from lambda.
    #[must_use]
    pub const fn from_lambda(lambda: f64) -> Self {
        Self(lambda)
    }
}

/// Functional Softshrink activation.
#[inline]
pub fn softshrink<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    lambda: f64,
) -> Var<T, B> {
    coeus_autograd::softshrink(input, lambda)
}

/// Softshrink activation module (default λ = 0.5).
#[derive(Clone, Debug)]
pub struct Softshrink {
    /// Threshold λ below which the activation maps to 0.
    pub lambda: f64,
}

impl Softshrink {
    /// Create a Softshrink module with custom λ.
    pub fn new(lambda: f64) -> Self {
        Self { lambda }
    }
}

impl Default for Softshrink {
    fn default() -> Self {
        Self { lambda: 0.5 }
    }
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for Softshrink {
    #[inline]
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    #[inline]
    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
        softshrink(input, self.lambda)
    }
}

/// Public descriptor for the functional Threshold operation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ThresholdNode;

impl ThresholdNode {
    /// Operation name used by the autograd node.
    #[must_use]
    pub const fn op_name() -> &'static str {
        "threshold"
    }
}

/// Functional Threshold activation.
#[inline]
pub fn threshold<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    thresh: f64,
    value: f64,
) -> Var<T, B> {
    coeus_autograd::threshold(input, thresh, value)
}

/// Threshold activation module (default threshold=0, value=0).
#[derive(Clone, Debug)]
pub struct Threshold {
    /// Lower-bound threshold.
    pub threshold: f64,
    /// Replacement value below threshold.
    pub value: f64,
}

impl Threshold {
    /// Create a Threshold module with custom `threshold` and `value`.
    pub fn new(threshold: f64, value: f64) -> Self {
        Self { threshold, value }
    }
}

impl Default for Threshold {
    fn default() -> Self {
        Self {
            threshold: 0.0,
            value: 0.0,
        }
    }
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for Threshold {
    #[inline]
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    #[inline]
    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
        threshold(input, self.threshold, self.value)
    }
}

/// Public descriptor for the functional Celu parameter set.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CeluOp(pub f64);

impl CeluOp {
    /// Construct a Celu descriptor from alpha.
    #[must_use]
    pub const fn from_alpha(alpha: f64) -> Self {
        Self(alpha)
    }
}

/// Functional Celu activation.
#[inline]
pub fn celu<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    alpha: f64,
) -> Var<T, B> {
    coeus_autograd::celu(input, alpha)
}

/// Celu activation module (default α = 1.0, matching PyTorch).
#[derive(Clone, Debug)]
pub struct Celu {
    /// Continuity-preserving α constant.
    pub alpha: f64,
}

impl Celu {
    /// Create a Celu module with custom α.
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }
}

impl Default for Celu {
    fn default() -> Self {
        Self { alpha: 1.0 }
    }
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for Celu {
    #[inline]
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![]
    }

    #[inline]
    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
        celu(input, self.alpha)
    }
}

/// Functional PReLU activation with a learnable per-channel (or
/// shared-scalar) weight — see [`coeus_autograd::prelu`] for the composition
/// and gradient derivation.
#[inline]
pub fn prelu<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    weight: &Var<T, B>,
) -> Var<T, B>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    coeus_autograd::prelu(input, weight)
}

/// PReLU activation module with a learnable weight (PyTorch/Burn semantics:
/// `num_parameters = 1` for one shared slope, or the channel count for a
/// per-channel slope broadcasting against dim 1 of the input).
#[derive(Clone)]
pub struct PReLU<T: Float, B: coeus_ops::BackendOps<T> + Default = coeus_core::MoiraiBackend> {
    /// Learnable slope(s); shape `[1]` (shared) or `[num_parameters]`
    /// (per-channel).
    pub weight: Var<T, B>,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> PReLU<T, B> {
    /// Create a PReLU module with `num_parameters` learnable slopes (`1` for
    /// a shared scalar, or the channel count for per-channel slopes), each
    /// initialized to `init` (PyTorch/Burn default: `0.25`).
    pub fn new(num_parameters: usize, init: f64) -> Self {
        let backend = B::default();
        let weight = Var::new(
            coeus_tensor::Tensor::full_on([num_parameters], T::from_f64(init), &backend),
            true,
        );
        Self { weight }
    }
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Default for PReLU<T, B> {
    fn default() -> Self {
        Self::new(1, 0.25)
    }
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for PReLU<T, B>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    #[inline]
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![self.weight.clone()]
    }

    #[inline]
    fn load_parameters(&mut self, params: &[Var<T, B>]) {
        self.weight = params[0].clone();
    }

    #[inline]
    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
        prelu(input, &self.weight)
    }
}
