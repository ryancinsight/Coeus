// ── Group Normalization ──
//
// GroupNorm<T, B, const G: usize> normalizes inputs by splitting channels into G groups
// and normalizing each group independently. G is encoded as a const generic,
// enabling compile-time loop unrolling hints and ensuring type-level correctness.
//
// Unlike BatchNorm, GroupNorm has no running statistics and is identical in
// train and eval modes — correct for variable-batch or single-sample inference.

use crate::module::Module;
use coeus_autograd::Var;
use coeus_core::{Float, MoiraiBackend};
use coeus_tensor::Tensor;
use std::cell::RefCell;

#[derive(Clone)]
struct GroupNormCache<T: Float, B: coeus_ops::BackendOps<T> + Default> {
    group_size: usize,
    ln_weight: Var<T, B>,
    ln_bias: Var<T, B>,
    eps_t: Tensor<T, B>,
    d_const: Tensor<T, B>,
    ones_cache: RefCell<Option<(usize, Tensor<T, B>)>>,
}

/// Group Normalization layer.
///
/// # Type parameters
/// - `G` — number of groups (const generic). `num_features % G == 0` is asserted at construction.
///
/// # Shape
/// Input: `[N, C, *]` where `C = num_features`.
/// Output: same shape as input.
///
/// Normalizes over `[C/G, *]` dimensions per group, independently for each sample.
#[derive(Clone)]
pub struct GroupNorm<
    T: Float,
    B: coeus_ops::BackendOps<T> + Default = MoiraiBackend,
    const G: usize = 1,
> {
    /// Trainable scale (gamma): shape `[num_features]`.
    pub weight: Var<T, B>,
    /// Trainable shift (beta): shape `[num_features]`.
    pub bias: Var<T, B>,
    pub num_features: usize,
    pub eps: f64,
    cache: RefCell<Option<GroupNormCache<T, B>>>,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default, const G: usize> GroupNorm<T, B, G> {
    /// Create a new GroupNorm layer.
    ///
    /// # Panics
    /// Panics if `num_features % G != 0`.
    pub fn new(num_features: usize, eps: f64) -> Self {
        assert!(
            G > 0 && num_features.is_multiple_of(G),
            "GroupNorm: num_features ({num_features}) must be divisible by G ({G})"
        );
        let backend = B::default();
        let weight = Var::new(Tensor::ones_on([num_features], &backend), true);
        let bias = Var::new(Tensor::zeros_on([num_features], &backend), true);
        Self {
            weight,
            bias,
            num_features,
            eps,
            cache: RefCell::new(None),
        }
    }

    fn get_cache(&self, group_size: usize) -> std::cell::RefMut<'_, Option<GroupNormCache<T, B>>> {
        let mut cache = self.cache.borrow_mut();
        let need_recreate = match &*cache {
            Some(c) => c.group_size != group_size,
            None => true,
        };
        if need_recreate {
            let backend = B::default();
            let ln_weight = Var::new(Tensor::ones_on([group_size], &backend), false);
            let ln_bias = Var::new(Tensor::zeros_on([group_size], &backend), false);
            let eps_t = Tensor::full_on([1], T::from_f64(self.eps), &backend);
            let d_const = Tensor::full_on([1], T::from_f64(group_size as f64), &backend);
            *cache = Some(GroupNormCache {
                group_size,
                ln_weight,
                ln_bias,
                eps_t,
                d_const,
                ones_cache: RefCell::new(None),
            });
        }
        cache
    }
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default, const G: usize> Module<T, B>
    for GroupNorm<T, B, G>
{
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![self.weight.clone(), self.bias.clone()]
    }

    /// Forward pass.
    ///
    /// Input shape: `[N, C]` or `[N, C, L]` or `[N, C, H, W]`.
    /// The implementation flattens to `[N*G, C/G * spatial]`, applies LayerNorm over the last dim,
    /// then reshapes back and applies per-channel weight/bias.
    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
        let shape = input.tensor.shape_cloned();
        assert!(shape.len() >= 2, "GroupNorm: input must be at least 2D");
        let n = shape[0];
        let c = shape[1];
        assert_eq!(
            c, self.num_features,
            "GroupNorm: channel dimension mismatch"
        );
        let c_per_g = c / G;

        // Compute total spatial elements (everything after batch and channel dims)
        let spatial: usize = if shape.len() > 2 {
            shape[2..].iter().product()
        } else {
            1
        };
        let group_size = c_per_g * spatial;

        // Flatten input to [N*G, group_size] via tracked reshape
        let flat = coeus_autograd::reshape(input, [n * G, group_size]);

        // Get cache
        let cache_borrow = self.get_cache(group_size);
        let cache = cache_borrow.as_ref().unwrap();

        let backend = B::default();

        // ── Mean over last dimension ──
        let mean_t = coeus_ops::mean_axis(&flat.tensor, 1, &backend); // [N*G, 1]

        // ── Centered: x - mu ──
        let xmu = coeus_ops::sub(&flat.tensor, &mean_t, &backend); // [N*G, group_size]

        // ── Variance ──
        let xmu_sq = coeus_ops::mul(&xmu, &xmu, &backend);
        let mut stdev = coeus_ops::mean_axis(&xmu_sq, 1, &backend); // [N*G, 1]

        // ── 1/sqrt(var + eps) ──
        coeus_ops::add_assign(&mut stdev, &cache.eps_t, &backend);
        coeus_ops::sqrt_assign(&mut stdev, &backend);

        let ones = {
            let mut o_cache = cache.ones_cache.borrow_mut();
            let n_groups = n * G;
            if let Some((cached_n, ref cached_ones)) = *o_cache {
                if cached_n == n_groups {
                    cached_ones.clone()
                } else {
                    let ones = Tensor::ones_on([n_groups, 1], &backend);
                    *o_cache = Some((n_groups, ones.clone()));
                    ones
                }
            } else {
                let ones = Tensor::ones_on([n_groups, 1], &backend);
                *o_cache = Some((n_groups, ones.clone()));
                ones
            }
        };
        let mut istdev = ones;
        coeus_ops::div_assign(&mut istdev, &stdev, &backend); // [N*G, 1]

        // ── Normalize ──
        let x_hat = coeus_ops::mul(&xmu, &istdev, &backend); // [N*G, group_size]

        // ── Scale and bias ──
        let w_reshaped = cache.ln_weight.tensor.reshape([1, group_size]);
        let b_reshaped = cache.ln_bias.tensor.reshape([1, group_size]);
        let mut out_tensor = coeus_ops::mul(&x_hat, &w_reshaped, &backend);
        coeus_ops::add_assign(&mut out_tensor, &b_reshaped, &backend);

        let normed_flat = coeus_autograd::layernorm(
            &flat,
            &cache.ln_weight,
            &cache.ln_bias,
            out_tensor,
            x_hat,
            istdev,
            cache.d_const.clone(),
        );

        // Reshape back to original shape via tracked reshape
        let normed = coeus_autograd::reshape(&normed_flat, shape.clone());

        // Apply per-channel affine transform:
        // weight/bias are [C]; reshape to [1, C, 1, ...] and broadcast-multiply
        let mut broadcast_shape = vec![1usize; shape.len()];
        broadcast_shape[1] = c;
        let w_reshaped = coeus_autograd::reshape(&self.weight, broadcast_shape.as_slice());
        let b_reshaped = coeus_autograd::reshape(&self.bias, broadcast_shape.as_slice());

        let scaled = coeus_autograd::mul(&normed, &w_reshaped);
        coeus_autograd::add(&scaled, &b_reshaped)
    }
}
