mod integer_exponent;

use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::{CpuAddressableStorage, CpuAddressableStorageMut, Float, FloatElement, Scalar};
use coeus_tensor::Tensor;
use integer_exponent::IntegerExponent;
use std::ops::Neg;
use std::sync::Arc;

// Parametric: stores `exp` as `u64` (bit-cast of f64) to stay `Sync`.
// Cannot use generic `UnaryNode<T, B, Op>` as `Op` would need to carry `exp`.
// Follows the same bespoke pattern as `LeakyReluNode`.

/// Autograd node for element-wise power: `y = x ^ exp`.
pub struct PowNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    /// Accumulated gradient buffer for the output of this node.
    pub output_grad: Arc<GradBuffer<T, B>>,
    /// Input variables tracked for backward propagation.
    pub inputs: Vec<Var<T, B>>,
    /// Input tensor snapshot for backward.
    pub input_tensor: Tensor<T, B>,
    /// Exponent stored as bit-cast `u64` for `Sync`-safe storage.
    pub exp_bits: u64,
}

impl<T: Float + FloatElement + Neg<Output = T>, B: coeus_ops::BackendOps<T> + Default>
    BackwardNode<T, B> for PowNode<T, B>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    #[inline]
    fn op_name(&self) -> &'static str {
        "pow"
    }
    #[inline]
    fn output_grad(&self) -> &Arc<GradBuffer<T, B>> {
        &self.output_grad
    }
    #[inline]
    fn inputs(&self) -> &[Var<T, B>] {
        &self.inputs
    }

    fn backward(
        &self,
        grad_out: &Tensor<T, B>,
        input_grads: &[Option<Arc<GradBuffer<T, B>>>],
    ) -> Result<(), B::Error> {
        let backend = B::default();
        let Some(Some(ref g)) = input_grads.first() else {
            return Ok(());
        };

        let exp = f64::from_bits(self.exp_bits);
        let exp_t = <T as FloatElement>::from_f64(exp);
        let n = self.input_tensor.numel();
        if n == 0 {
            return Ok(());
        }

        // Host-side copy of upstream gradient.
        let temp_grad;
        let grad_cont = if grad_out.is_contiguous() && grad_out.layout().offset() == 0 {
            grad_out
        } else {
            temp_grad = grad_out.to_contiguous_on(&backend)?;
            &temp_grad
        };
        let mut grad_host = vec![T::ZERO; n];
        backend.copy_to_host(grad_cont.storage(), &mut grad_host)?;

        // Host-side copy of forward input.
        let input_contig =
            if self.input_tensor.is_contiguous() && self.input_tensor.layout().offset() == 0 {
                self.input_tensor.reshape([n])
            } else {
                self.input_tensor.to_contiguous_on(&backend)?.reshape([n])
            };
        let mut x_host = vec![T::ZERO; n];
        backend.copy_to_host(input_contig.storage(), &mut x_host)?;

        // Decide whether `exp` is integer-valued in T.  When it is, the
        // PyTorch `Tensor.pow(scalar)` contract is sign-preserving integer power:
        //   (-x)^k = (-1)^k * x^k  (odd k negates, even k preserves).
        //   For k = 0 the result is 1 and d/dx = 0.
        // When `exp` is fractional, PyTorch follows IEEE: pow(x, p) = NaN
        // for x < 0; backward `n · x^(n-1)` mirrors that (NaN where x < 0).
        let integer = IntegerExponent::from_value(exp_t);

        let mut grad_in_host = vec![T::ZERO; n];
        let one = T::ONE;

        match integer {
            Some(integer) if integer.is_zero() => {
                // x^0 = 1 → d/dx = 0; grad contribution is zero everywhere.
            }
            Some(integer) if exp_t > T::ZERO => {
                // d/dx x^k = k · x^(k-1).  Sign-preserving integer power:
                //   x^(k-1) = sign(x)^(k-1) · |x|^(k-1).
                // When k is odd, k-1 is even → sign^(k-1) = +1 (always non-negative).
                // When k is even, k-1 is odd  → sign^(k-1) = sign(x) (local grad may
                // change sign with x).
                let coef = exp_t;
                let k_m1 = integer.predecessor();
                let k_m1_odd = k_m1.is_odd();
                for i in 0..n {
                    let x = x_host[i];
                    if x == T::ZERO {
                        // k = 1: d/dx x = 1; else x = 0 → grad = 0 (k > 1).
                        if exp_t == T::ONE {
                            grad_in_host[i] = grad_host[i] * coef;
                        }
                        continue;
                    }
                    let abs_x = <T as Float>::abs(x);
                    let abs_pow_m1 = k_m1.pow_magnitude(abs_x);
                    let local = if k_m1_odd {
                        // k-1 odd → sign(x) factor: x^(k-1) carries sign(x).
                        let sgn = if x < T::ZERO { -one } else { one };
                        coef * sgn * abs_pow_m1
                    } else {
                        // k-1 even → x^(k-1) is always non-negative.
                        coef * abs_pow_m1
                    };
                    grad_in_host[i] = grad_host[i] * local;
                }
            }
            Some(integer) => {
                // d/dx x^(-k) = -k · x^(-k-1) = -k / x^(k+1) (sign-preserving).
                // Denominator sign follows (k+1) parity on x.
                // exp_t already carries the negative derivative coefficient -k.
                let coef = exp_t;
                let exp_total = integer.successor();
                let exp_total_odd = exp_total.is_odd();
                for i in 0..n {
                    let x = x_host[i];
                    if x == T::ZERO {
                        // 1/0 = inf or NaN; PyTorch yields inf.  Leave as zero
                        // since `grad_host[i] * NaN` would taint the accumulator.
                        continue;
                    }
                    let abs_x = <T as Float>::abs(x);
                    let denom_abs = exp_total.pow_magnitude(abs_x);
                    let denom = if exp_total_odd {
                        let sgn = if x < T::ZERO { -one } else { one };
                        sgn * denom_abs
                    } else {
                        denom_abs
                    };
                    let local = coef / denom;
                    grad_in_host[i] = grad_host[i] * local;
                }
            }
            None => {
                // Fractional exponent path: compose with ln/exp so shape / device
                // multiplication is preserved via the backend dispatch.  For x < 0,
                // `ln(x)` propagates NaN (matches PyTorch IEEE).
                let exp_m1 = exp - 1.0;
                let ln_x = coeus_ops::log(&self.input_tensor, &backend)?;
                let scaled = {
                    let scale = Tensor::full_on(
                        ln_x.shape(),
                        <T as FloatElement>::from_f64(exp_m1),
                        &backend,
                    )?;
                    coeus_ops::mul(&ln_x, &scale, &backend)?
                };
                let x_pow_n_m1 = coeus_ops::exp(&scaled, &backend)?;
                let n_tensor = Tensor::full_on(x_pow_n_m1.shape(), exp_t, &backend)?;
                let local_grad = coeus_ops::mul(&n_tensor, &x_pow_n_m1, &backend)?;
                let grad_in = coeus_ops::mul(grad_out, &local_grad, &backend)?;
                let lock = g.write();
                coeus_ops::add_assign(lock, &grad_in, &backend)?;
                return Ok(());
            }
        }
        let grad_t =
            Tensor::from_slice_on(self.input_tensor.shape().to_vec(), &grad_in_host, &backend)?;
        let lock = g.write();
        coeus_ops::add_assign(lock, &grad_t, &backend)?;
        Ok(())
    }
}

/// Tracked element-wise power: `y = x ^ exp`.
///
/// `exp` is an `f64` scalar applied uniformly to all elements.
/// Backward: `d/dx x^n = n · x^(n−1)`.
///
/// # Parity contract
///
/// Matches `torch.Tensor.pow(scalar)` semantics:
/// - When the converted exponent is finite and integer-valued in `T`,
///   the forward is a sign-preserving integer power `(-x)^k = (-1)^k · x^k`,
///   with `x = 0` mapping to `1` when `k = 0`, to `0` when `k > 0`, and to
///   `+inf` when `k < 0` (delegated to PyTorch's IEEE-compliant convention).
/// - Otherwise forward composes `exp(n · ln(x))` (NaN for `x < 0`, IEEE).
///
/// # Precision
/// Forward and backward execute in the native precision of `T` without widening.
/// The exponent is converted once via `<T as FloatElement>::from_f64(exp)`; the integer-exponent
/// branch uses repeated multiplication in `T` (no `powf` fallback). Integer
/// sign, magnitude and parity all follow that converted value, without an
/// `i32`/`i64` range restriction. Backward retains the exact integer exponent
/// offset even when adding or subtracting one would round in `T`.
#[inline]
pub fn pow<T: Float + FloatElement + Neg<Output = T>, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
    exp: f64,
) -> Result<Var<T, B>, B::Error>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let backend = B::default();
    let exp_t = <T as FloatElement>::from_f64(exp);

    let n = a.tensor.numel();
    let integer = if n > 0 {
        IntegerExponent::from_value(exp_t)
    } else {
        None
    };
    let out_tensor = if let Some(integer) = integer {
        // Host-fold sign-preserving integer power.
        let input_contig = if a.tensor.is_contiguous() && a.tensor.layout().offset() == 0 {
            a.tensor.reshape([n])
        } else {
            a.tensor.to_contiguous_on(&backend)?.reshape([n])
        };
        let mut x_host = vec![T::ZERO; n];
        backend.copy_to_host(input_contig.storage(), &mut x_host)?;

        let one = T::ONE;
        let mut out_host = vec![T::ZERO; n];
        if integer.is_zero() {
            out_host.fill(one);
        } else if exp_t > T::ZERO {
            for i in 0..n {
                let x = x_host[i];
                if x == T::ZERO {
                    // x^0 = 1 already handled above; x^k for k > 0 is 0.
                    out_host[i] = T::ZERO;
                    continue;
                }
                let abs_x = <T as Float>::abs(x);
                let abs_pow = integer.pow_magnitude(abs_x);
                out_host[i] = if integer.is_odd() {
                    let sgn = if x < T::ZERO { -one } else { one };
                    sgn * abs_pow
                } else {
                    abs_pow
                };
            }
        } else {
            // Negative exponent: x^(-k) = 1 / x^k with sign preservation.
            for i in 0..n {
                let x = x_host[i];
                if x == T::ZERO {
                    // 1/0 → +inf per PyTorch IEEE; emit +inf to mirror that
                    // (avoids NaN from sign*0 division).
                    out_host[i] = <T as Float>::INFINITY;
                    continue;
                }
                let abs_x = <T as Float>::abs(x);
                let denom_abs = integer.pow_magnitude(abs_x);
                let denom = if integer.is_odd() {
                    let sgn = if x < T::ZERO { -one } else { one };
                    sgn * denom_abs
                } else {
                    denom_abs
                };
                out_host[i] = one / denom;
            }
        }
        Tensor::from_slice(a.tensor.shape().to_vec(), &out_host)
    } else {
        // Fractional exponent path: x^n = exp(n · ln(x)).  NaN for x ≤ 0
        // (IEEE) is preserved by `ln(x)` propagation through the backend.
        let ln_x = coeus_ops::log(&a.tensor, &backend)?;
        let n_tensor = Tensor::full_on(ln_x.shape(), exp_t, &backend)?;
        let scaled = coeus_ops::mul(&n_tensor, &ln_x, &backend)?;
        coeus_ops::exp(&scaled, &backend)
    }?;

    let requires_grad = crate::grad_mode::should_track_var(a);
    let grad = if requires_grad {
        Some(Arc::new(GradBuffer::new(Tensor::zeros_on(
            out_tensor.shape_cloned(),
            &backend,
        )?)))
    } else {
        None
    };
    let creator = if requires_grad {
        let output_grad = grad.as_ref().expect("invariant: requires_grad gates both the Some(grad) construction above and this read").clone();
        let node = PowNode {
            output_grad,
            inputs: vec![a.clone()],
            input_tensor: a.tensor.clone(),
            exp_bits: f64::to_bits(exp),
        };
        Some(Arc::new(node) as Arc<dyn BackwardNode<T, B>>)
    } else {
        None
    };
    Ok(Var {
        tensor: out_tensor,
        grad,
        creator,
    })
}
