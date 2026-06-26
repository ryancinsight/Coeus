// ── Bilinear ──
//
// out = x1 @ W @ x2.T + b  for feature interaction.
// W shape: [out_features, in1_features, in2_features]

use crate::module::Module;
use coeus_autograd::Var;
use coeus_core::{Float, MoiraiBackend};
use coeus_tensor::Tensor;

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
    pub in1_features: usize,
    pub in2_features: usize,
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
            Some(Var::new(
                Tensor::zeros_on([out_features], &_backend),
                true,
            ))
        } else {
            None
        };
        Self { weight: w, bias: b, in1_features, in2_features, out_features }
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
        // For each output feature k, compute:
        //   w_k = weight[k, :, :] — shape [in1, in2]
        //   x1_w = x1 @ w_k       — shape [batch, in2]
        //   row_k = sum(x1_w * x2, dim=1) — shape [batch]
        // Stack all rows into [batch, out_features]
        let mut out_rows: Vec<Var<T, B>> = Vec::with_capacity(self.out_features);
        for k in 0..self.out_features {
            // Extract w_k: [in1, in2]
            let w_k_var = coeus_autograd::slice(
                &self.weight,
                &[(k, k + 1), (0, self.in1_features), (0, self.in2_features)],
            );
            let w_k = coeus_autograd::reshape(&w_k_var, vec![self.in1_features, self.in2_features]);
            // x1 @ w_k: [batch, in2]
            let x1w = coeus_autograd::matmul(x1, &w_k);
            // element-wise mul with x2, then sum over in2 → [batch, 1]
            let prod = coeus_autograd::mul(&x1w, x2);
            let row = coeus_autograd::sum_axis(&prod, 1); // [batch, 1]
            out_rows.push(row);
        }
        // cat along dim=1: [batch, out_features]
        let refs: Vec<&Var<T, B>> = out_rows.iter().collect();
        let out = coeus_autograd::cat(&refs, 1);
        if let Some(ref b) = self.bias {
            coeus_autograd::add(&out, b)
        } else {
            out
        }
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
