// ── Bilinear ──
//
// out = x1 @ W @ x2.T + b  for feature interaction.
// W shape: [out_features, in1_features, in2_features]

use crate::module::Module;
use coeus_autograd::Var;
use coeus_core::{Float, MoiraiBackend};
use coeus_tensor::Tensor;

/// Functional bilinear interaction.
///
/// Computes:
/// `out[n, k] = Σ_{i,j} x1[n,i] * weight[k,i,j] * x2[n,j] + bias[k]`.
///
/// # Shapes
/// - `x1`: `[batch, in1_features]`
/// - `x2`: `[batch, in2_features]`
/// - `weight`: `[out_features, in1_features, in2_features]`
/// - `bias` (optional): `[out_features]`
/// - Output: `[batch, out_features]`
///
/// # Panics
/// Panics if input shapes are incompatible.
pub fn bilinear<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    x1: &Var<T, B>,
    x2: &Var<T, B>,
    weight: &Var<T, B>,
    bias: Option<&Var<T, B>>,
) -> Var<T, B>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    let x1_shape = x1.tensor.shape();
    let x2_shape = x2.tensor.shape();
    let w_shape = weight.tensor.shape();
    assert_eq!(
        x1_shape.len(),
        2,
        "bilinear: x1 must be rank-2 [batch, in1]"
    );
    assert_eq!(
        x2_shape.len(),
        2,
        "bilinear: x2 must be rank-2 [batch, in2]"
    );
    assert_eq!(
        w_shape.len(),
        3,
        "bilinear: weight must be rank-3 [out, in1, in2]"
    );
    let batch = x1_shape[0];
    let in1 = x1_shape[1];
    let in2 = x2_shape[1];
    let out_features = w_shape[0];
    assert_eq!(x2_shape[0], batch, "bilinear: x1/x2 batch mismatch");
    assert_eq!(
        w_shape[1], in1,
        "bilinear: weight in1 dimension must match x1 feature dimension"
    );
    assert_eq!(
        w_shape[2], in2,
        "bilinear: weight in2 dimension must match x2 feature dimension"
    );
    if let Some(bias) = bias {
        assert_eq!(
            bias.tensor.shape(),
            &[out_features],
            "bilinear: bias must have shape [out_features]"
        );
    }

    let mut out_rows: Vec<Var<T, B>> = Vec::with_capacity(out_features);
    for k in 0..out_features {
        let w_k_var = coeus_autograd::slice(weight, &[(k, k + 1), (0, in1), (0, in2)]);
        let w_k = coeus_autograd::reshape(&w_k_var, vec![in1, in2]);
        let x1w = coeus_autograd::matmul(x1, &w_k);
        let prod = coeus_autograd::mul(&x1w, x2);
        let row = coeus_autograd::sum_axis(&prod, 1);
        out_rows.push(row);
    }
    let refs: Vec<&Var<T, B>> = out_rows.iter().collect();
    let out = coeus_autograd::cat(&refs, 1);
    if let Some(b) = bias {
        coeus_autograd::add(&out, b)
    } else {
        out
    }
}

/// Bilinear layer: `out[n] = Σ_{ij} x1[n,i] * W[k,i,j] * x2[n,j] + b[k]`
///
/// Equivalent to `torch.nn.Bilinear(in1_features, in2_features, out_features)`.
///
/// Shapes:
/// - `x1`: `[batch, in1_features]`
/// - `x2`: `[batch, in2_features]`
/// - `W`:  `[out_features, in1_features, in2_features]`
/// - `b`:  `[out_features]`
/// - output: `[batch, out_features]`
#[derive(Clone)]
pub struct Bilinear<T: Float, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend> {
    /// Weight tensor: `[out_features, in1_features, in2_features]`.
    pub weight: Var<T, B>,
    /// Bias vector: `[out_features]`.
    pub bias: Option<Var<T, B>>,
    /// Number of input features for the first operand.
    pub in1_features: usize,
    /// Number of input features for the second operand.
    pub in2_features: usize,
    /// Number of output features.
    pub out_features: usize,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Bilinear<T, B> {
    /// Create with Xavier-initialized weight and zero bias.
    pub fn new(in1_features: usize, in2_features: usize, out_features: usize, bias: bool) -> Self
    where
        T: coeus_leto::RandomScalar,
    {
        let _backend = B::default();
        let fan_in = in1_features * in2_features;
        let mut w = Var::new(
            Tensor::zeros_on([out_features, in1_features, in2_features], &_backend),
            true,
        );
        crate::init::xavier_uniform(&mut w, fan_in, out_features);
        let b = if bias {
            Some(Var::new(Tensor::zeros_on([out_features], &_backend), true))
        } else {
            None
        };
        Self {
            weight: w,
            bias: b,
            in1_features,
            in2_features,
            out_features,
        }
    }

    /// Forward pass.
    ///
    /// Computed as: for each output feature k:
    ///   `out[n,k] = x1[n,:] @ W[k,:,:] @ x2[n,:].T`
    /// then optionally adds `b[k]`.
    pub fn bilinear_forward(&self, x1: &Var<T, B>, x2: &Var<T, B>) -> Var<T, B>
    where
        B::DeviceBuffer<T>:
            coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
    {
        bilinear(x1, x2, &self.weight, self.bias.as_ref())
    }
}

impl<T: Float + coeus_leto::RandomScalar, B: coeus_ops::BackendOps<T> + Default> Module<T, B>
    for Bilinear<T, B>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    fn parameters(&self) -> Vec<Var<T, B>> {
        let mut p = vec![self.weight.clone()];
        if let Some(ref b) = self.bias {
            p.push(b.clone());
        }
        p
    }

    fn forward(&self, x1: &Var<T, B>) -> Var<T, B> {
        self.bilinear_forward(x1, x1)
    }
}
