// ── Tracked cumprod ──
//
// Backward: grad_x[i] = Σ_{j≥i} grad_out[j] · ∏_{k≤j, k≠i} x_k, evaluated
// exactly (including at zeros, matching PyTorch) in O(n) per line via a
// first-zero decomposition. With z1/z2 the first/second zero positions
// (or n when absent):
//
//   i < z1 : grad_i = (Σ_{i≤j<z1} grad_out[j] · out[j]) / x_i
//            — the standard suffix-sum form; out[j≥z1] = 0 truncates the sum.
//   i = z1 : grad_i = Σ_{z1≤j<z2} grad_out[j] · prefix · ∏_{z1<k≤j} x_k,
//            prefix = ∏_{k<z1} x_k — removing x_{z1} from the product leaves
//            it non-zero until the second zero enters at j = z2.
//   i > z1 : grad_i = 0 — every out[j] with j ≥ i keeps the x_{z1} = 0 factor.
//
// A zero-free line (z1 = n) reduces to the classic
// suffix_sum(grad_out · out) / x_i over the whole line.

use crate::grad_buffer::GradBuffer;
use crate::node::BackwardNode;
use crate::var::Var;
use coeus_core::Scalar;
use coeus_tensor::Tensor;
use std::sync::Arc;

pub struct CumprodNode<T: Scalar, B: coeus_ops::BackendOps<T> + Default>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    pub output_grad: Arc<GradBuffer<T, B>>,
    pub inputs: Vec<Var<T, B>>,
    /// Forward output (cumprod values).
    pub output: Tensor<T, B>,
    /// Input tensor saved for backward.
    pub input_saved: Tensor<T, B>,
    pub dim: usize,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> BackwardNode<T, B> for CumprodNode<T, B>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    fn op_name(&self) -> &'static str {
        "cumprod"
    }
    fn output_grad(&self) -> &Arc<GradBuffer<T, B>> {
        &self.output_grad
    }
    fn inputs(&self) -> &[Var<T, B>] {
        &self.inputs
    }

    fn backward(&self, grad_out: &Tensor<T, B>, input_grads: &[Option<Arc<GradBuffer<T, B>>>]) {
        let backend = B::default();
        if let Some(Some(ref g)) = input_grads.first() {
            // grad_in[i] = suffix_sum( grad_out * out )[i] / x[i]
            // For numerical stability we use the exact element-wise computation
            // over a contiguous slice along dim.
            let shape = grad_out.shape();
            let ndim = shape.len();
            let dim = self.dim;
            let n = shape[dim];

            let go_cont = grad_out.to_contiguous();
            let out_cont = self.output.to_contiguous();
            let in_cont = self.input_saved.to_contiguous();

            let go_s = go_cont.as_slice();
            let out_s = out_cont.as_slice();
            let in_s = in_cont.as_slice();

            let numel: usize = shape.iter().product();
            let mut strides = vec![1usize; ndim];
            for d in (0..ndim - 1).rev() {
                strides[d] = strides[d + 1] * shape[d + 1];
            }
            let dim_stride = strides[dim];
            let outer = numel / (n * dim_stride);
            let inner = dim_stride;

            let mut gi_data = vec![T::zero(); numel];

            for outer_idx in 0..outer {
                for inner_idx in 0..inner {
                    let base = outer_idx * n * inner + inner_idx;
                    let at = |i: usize| base + i * inner;

                    // Locate the first (z1) and second (z2) zeros on this
                    // line; `n` means "none". Also accumulate the product of
                    // the elements before z1 (the prefix the zero position's
                    // gradient factors through).
                    let mut z1 = n;
                    let mut z2 = n;
                    let mut prefix = T::one();
                    for i in 0..n {
                        let xi = in_s[at(i)];
                        if xi == T::zero() {
                            if z1 == n {
                                z1 = i;
                            } else {
                                z2 = i;
                                break;
                            }
                        } else if z1 == n {
                            prefix *= xi;
                        }
                    }

                    // Positions before the first zero: the standard
                    // suffix-sum formula, truncated at z1 — out[j] for
                    // j ≥ z1 is 0, and d out[j]/dx_i for j ≥ z1 (i < z1)
                    // also vanishes only through out, so the truncation is
                    // exactly the j < z1 restriction.
                    let mut suffix: T = T::zero();
                    for i in (0..z1).rev() {
                        let flat = at(i);
                        suffix += go_s[flat] * out_s[flat];
                        gi_data[flat] = suffix / in_s[flat];
                    }

                    // The first zero position: d out[j]/dx_{z1}
                    // = ∏_{k≤j, k≠z1} x_k = prefix · ∏_{z1<k≤j} x_k, which is
                    // non-zero until the second zero enters the product, so
                    // the sum runs over z1 ≤ j < z2. Positions after z1 keep
                    // gradient 0: every out[j] they influence (j ≥ i > z1)
                    // retains the x_{z1} = 0 factor.
                    if z1 < n {
                        let mut acc = prefix; // ∏_{k≤j, k≠z1} x_k at j = z1
                        let mut grad_z: T = T::zero();
                        for j in z1..z2 {
                            if j > z1 {
                                acc *= in_s[at(j)];
                            }
                            grad_z += go_s[at(j)] * acc;
                        }
                        gi_data[at(z1)] = grad_z;
                    }
                }
            }

            let gi_increment = Tensor::from_slice(shape.to_vec(), &gi_data);
            let gl = g.write();
            coeus_ops::add_assign(gl, &gi_increment, &backend);
        }
    }
}

/// Tracked cumulative product along `dim`.
#[must_use]
#[inline]
pub fn cumprod<T: Scalar, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    dim: usize,
) -> Var<T, B>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    let backend = B::default();
    let out_tensor = coeus_ops::cumprod(&input.tensor, dim, &backend);

    let requires_grad = crate::grad_mode::should_track_var(input);
    let grad = if requires_grad {
        Some(Arc::new(GradBuffer::new(Tensor::zeros_on(
            out_tensor.shape_cloned(),
            &backend,
        ))))
    } else {
        None
    };
    let creator = if requires_grad {
        let node = CumprodNode {
            output_grad: grad.as_ref().unwrap().clone(),
            inputs: vec![input.clone()],
            output: out_tensor.clone(),
            input_saved: input.tensor.clone(),
            dim,
        };
        Some(Arc::new(node) as Arc<dyn BackwardNode<T, B>>)
    } else {
        None
    };
    Var {
        tensor: out_tensor,
        grad,
        creator,
    }
}
