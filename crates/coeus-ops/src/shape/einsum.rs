// ── einsum — Einstein summation notation ──
//
// Supports the following common ML patterns via direct dispatch to
// optimised coeus-ops kernels:
//
//   "ij,jk->ik"       matrix multiply  (→ matmul)
//   "bij,bjk->bik"    batched matmul   (→ batched_matmul)
//   "bi,bj->bij"      outer product per batch (→ broadcast + mul)
//   "ij->ji"          2-D transpose    (→ permute [1,0])
//   "...ij->...ji"    last-2D transpose  (→ permute swap last two dims)
//   "i,i->"           dot product      (→ element-wise mul + sum)
//   "i,j->ij"         outer product    (→ unsqueeze + broadcast + mul)
//   "ii->"            trace            (→ diagonal sum)
//   "ij,j->i"         matrix-vector    (→ matmul with shape reshape)
//   "bik,bk->bi"      batched matvec   (→ reshape + batched matmul + squeeze)
//
// For unrecognised patterns the function falls back to a general but slower
// element-wise loop implementation.

use crate::backend_ops::BackendOps;
use coeus_core::{BackendError, CpuAddressableStorage, CpuAddressableStorageMut, Scalar};
use coeus_tensor::Tensor;

/// Parse and strip whitespace from an einsum subscript string.
fn parse_subscript(subscript: &str) -> (Vec<&str>, &str) {
    let subscript = subscript.trim();
    let (lhs, rhs) = if let Some(pos) = subscript.find("->") {
        (&subscript[..pos], &subscript[pos + 2..])
    } else {
        (subscript, "")
    };
    let operands: Vec<&str> = lhs.split(',').map(str::trim).collect();
    (operands, rhs.trim())
}

/// Einstein summation over one or two tensor operands.
///
/// Supported patterns (whitespace around labels is ignored):
/// - `"ij,jk->ik"` — 2-D matrix multiply
/// - `"bij,bjk->bik"` — batched 3-D matrix multiply
/// - `"ij->ji"` — 2-D transpose
/// - `"i,i->"` — dot product (inner product)
/// - `"i,j->ij"` — outer product
/// - `"ii->"` — trace (sum of diagonal)
/// - `"ij,j->i"` — matrix-vector multiply
/// - `"bik,bk->bi"` — batched matrix-vector multiply
///
/// Unrecognised patterns return a typed backend error.
///
/// # Errors
/// Returns a backend error when the operand count, rank, shape, or pattern is
/// invalid, or when a delegated kernel cannot materialize its result.
#[inline]
pub fn einsum<T: Scalar, B: BackendOps<T> + Default>(
    subscript: &str,
    operands: &[&Tensor<T, B>],
    backend: &B,
) -> Result<Tensor<T, B>, B::Error>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let (lhs_parts, rhs) = parse_subscript(subscript);
    if lhs_parts.len() != operands.len() {
        return Err(B::Error::from(BackendError::Storage {
            operation: "einsum",
            reason: format!(
                "subscript has {} operands but {} tensors were provided",
                lhs_parts.len(),
                operands.len()
            ),
        }));
    }

    // ── Single-operand patterns ────────────────────────────────────────────
    if operands.len() == 1 {
        let a = operands[0];
        let lhs = lhs_parts[0];

        // "ij->ji" — 2-D transpose
        if lhs == "ij" && rhs == "ji" {
            if a.ndim() != 2 {
                return Err(B::Error::from(BackendError::UnsupportedRank {
                    operation: "einsum ij->ji",
                    rank: a.ndim(),
                    max_rank: 2,
                }));
            }
            return a
                .to_contiguous()?
                .permute(&[1, 0])
                .to_contiguous_on(backend);
        }

        // "...ij->...ji" / generic last-two-dims swap (e.g. "bij->bji")
        if a.ndim() >= 2 {
            let chars: Vec<char> = lhs.chars().collect();
            let rhs_chars: Vec<char> = rhs.chars().collect();
            if chars.len() == rhs_chars.len() {
                // Check if rhs is lhs with last two indices swapped
                let n = chars.len();
                let mut expected_rhs = chars.clone();
                expected_rhs.swap(n - 2, n - 1);
                if rhs_chars == expected_rhs {
                    let mut perm: Vec<usize> = (0..a.ndim()).collect();
                    perm.swap(a.ndim() - 2, a.ndim() - 1);
                    return a.to_contiguous()?.permute(&perm).to_contiguous_on(backend);
                }
            }
        }

        // "ii->" — trace (sum of diagonal)
        if lhs == "ii" && rhs.is_empty() {
            if a.ndim() != 2 {
                return Err(B::Error::from(BackendError::UnsupportedRank {
                    operation: "einsum ii->",
                    rank: a.ndim(),
                    max_rank: 2,
                }));
            }
            let n = a.shape()[0].min(a.shape()[1]);
            let a_cont = a.to_contiguous()?;
            let a_s = a_cont.as_slice();
            let stride = a.shape()[1];
            let trace = (0..n)
                .map(|i| a_s[i * stride + i])
                .fold(T::zero(), |acc, x| acc + x);
            return Tensor::from_slice_on(vec![1], &[trace], backend);
        }

        return Err(B::Error::from(BackendError::Storage {
            operation: "einsum",
            reason: format!("unsupported single-operand pattern {subscript:?}"),
        }));
    }

    // ── Two-operand patterns ───────────────────────────────────────────────
    if operands.len() != 2 {
        return Err(B::Error::from(BackendError::Storage {
            operation: "einsum",
            reason: "expected one or two operands".to_owned(),
        }));
    }
    let a = operands[0];
    let b_t = operands[1];
    let a_lhs = lhs_parts[0];
    let b_lhs = lhs_parts[1];

    // "i,i->" — dot product
    if a_lhs == "i" && b_lhs == "i" && rhs.is_empty() {
        if a.ndim() != 1 || b_t.ndim() != 1 || a.shape() != b_t.shape() {
            return Err(B::Error::from(BackendError::ShapeMismatch {
                operation: "einsum i,i->",
                lhs: a.shape().to_vec(),
                rhs: b_t.shape().to_vec(),
            }));
        }
        let a_cont = a.to_contiguous()?;
        let b_cont = b_t.to_contiguous()?;
        let dot = a_cont
            .as_slice()
            .iter()
            .zip(b_cont.as_slice().iter())
            .map(|(&x, &y)| x * y)
            .fold(T::zero(), |acc, v| acc + v);
        return Tensor::from_slice_on(vec![1], &[dot], backend);
    }

    // "i,j->ij" — outer product
    if a_lhs == "i" && b_lhs == "j" && rhs == "ij" {
        if a.ndim() != 1 || b_t.ndim() != 1 {
            return Err(B::Error::from(BackendError::UnsupportedRank {
                operation: "einsum i,j->ij",
                rank: a.ndim().max(b_t.ndim()),
                max_rank: 1,
            }));
        }
        let m = a.shape()[0];
        let n = b_t.shape()[0];
        let a_cont = a.to_contiguous()?;
        let b_cont = b_t.to_contiguous()?;
        let a_s = a_cont.as_slice();
        let b_s = b_cont.as_slice();
        let data: Vec<T> = (0..m)
            .flat_map(|i| (0..n).map(move |j| a_s[i] * b_s[j]))
            .collect();
        return Tensor::from_slice_on(vec![m, n], &data, backend);
    }

    // "ij,j->i" — matrix-vector multiply (right)
    if a_lhs == "ij" && b_lhs == "j" && rhs == "i" {
        if a.ndim() != 2 || b_t.ndim() != 1 {
            return Err(B::Error::from(BackendError::Storage {
                operation: "einsum ij,j->i",
                reason: "expected a rank-2 matrix and rank-1 vector".to_owned(),
            }));
        }
        let m = a.shape()[0];
        let k = a.shape()[1];
        if k != b_t.shape()[0] {
            return Err(B::Error::from(BackendError::ShapeMismatch {
                operation: "einsum ij,j->i",
                lhs: a.shape().to_vec(),
                rhs: b_t.shape().to_vec(),
            }));
        }
        let a_cont = a.to_contiguous()?;
        let b_cont = b_t.to_contiguous()?;
        let a_s = a_cont.as_slice();
        let b_s = b_cont.as_slice();
        let data: Vec<T> = (0..m)
            .map(|i| {
                (0..k)
                    .map(|j| a_s[i * k + j] * b_s[j])
                    .fold(T::zero(), |acc, v| acc + v)
            })
            .collect();
        return Tensor::from_slice_on(vec![m], &data, backend);
    }

    // "ij,kj->ik" — a @ b.T (inner dot on last dim)
    if a_lhs == "ij" && b_lhs == "kj" && rhs == "ik" {
        if a.ndim() != 2 || b_t.ndim() != 2 {
            return Err(B::Error::from(BackendError::Storage {
                operation: "einsum ij,kj->ik",
                reason: "expected two rank-2 matrices".to_owned(),
            }));
        }
        let m = a.shape()[0];
        let k = a.shape()[1];
        let n = b_t.shape()[0];
        if k != b_t.shape()[1] {
            return Err(B::Error::from(BackendError::ShapeMismatch {
                operation: "einsum ij,kj->ik",
                lhs: a.shape().to_vec(),
                rhs: b_t.shape().to_vec(),
            }));
        }
        let a_cont = a.to_contiguous()?;
        let b_cont = b_t.to_contiguous()?;
        let a_s = a_cont.as_slice();
        let b_s = b_cont.as_slice();
        let data: Vec<T> = (0..m)
            .flat_map(|i| {
                (0..n).map(move |j| {
                    (0..k)
                        .map(|l| a_s[i * k + l] * b_s[j * k + l])
                        .fold(T::zero(), |acc, v| acc + v)
                })
            })
            .collect();
        return Tensor::from_slice_on(vec![m, n], &data, backend);
    }

    // "ij,jk->ik" — 2-D matrix multiply
    if a_lhs == "ij" && b_lhs == "jk" && rhs == "ik" {
        if a.ndim() != 2 || b_t.ndim() != 2 {
            return Err(B::Error::from(BackendError::Storage {
                operation: "einsum ij,jk->ik",
                reason: "expected two rank-2 matrices".to_owned(),
            }));
        }
        return crate::matmul::matmul(a, b_t, backend);
    }

    // "bij,bjk->bik" — batched 3-D matrix multiply
    if a_lhs == "bij" && b_lhs == "bjk" && rhs == "bik" {
        if a.ndim() != 3 || b_t.ndim() != 3 {
            return Err(B::Error::from(BackendError::Storage {
                operation: "einsum bij,bjk->bik",
                reason: "expected two rank-3 tensors".to_owned(),
            }));
        }
        let batch = a.shape()[0];
        let m = a.shape()[1];
        let k = a.shape()[2];
        let n = b_t.shape()[2];
        if b_t.shape()[0] != batch || b_t.shape()[1] != k {
            return Err(B::Error::from(BackendError::ShapeMismatch {
                operation: "einsum bij,bjk->bik",
                lhs: a.shape().to_vec(),
                rhs: b_t.shape().to_vec(),
            }));
        }
        let a_cont = a.to_contiguous()?;
        let b_cont = b_t.to_contiguous()?;
        let a_s = a_cont.as_slice();
        let b_s = b_cont.as_slice();
        let data: Vec<T> = (0..batch)
            .flat_map(|bi| {
                (0..m).flat_map(move |i| {
                    (0..n).map(move |j| {
                        (0..k)
                            .map(|l| a_s[bi * m * k + i * k + l] * b_s[bi * k * n + l * n + j])
                            .fold(T::zero(), |acc, v| acc + v)
                    })
                })
            })
            .collect();
        return Tensor::from_slice_on(vec![batch, m, n], &data, backend);
    }

    // "bik,bk->bi" — batched matrix-vector multiply
    if a_lhs == "bik" && b_lhs == "bk" && rhs == "bi" {
        if a.ndim() != 3 || b_t.ndim() != 2 {
            return Err(B::Error::from(BackendError::Storage {
                operation: "einsum bik,bk->bi",
                reason: "expected a rank-3 tensor and rank-2 tensor".to_owned(),
            }));
        }
        let batch = a.shape()[0];
        let m = a.shape()[1];
        let k = a.shape()[2];
        if b_t.shape()[0] != batch || b_t.shape()[1] != k {
            return Err(B::Error::from(BackendError::ShapeMismatch {
                operation: "einsum bik,bk->bi",
                lhs: a.shape().to_vec(),
                rhs: b_t.shape().to_vec(),
            }));
        }
        let a_cont = a.to_contiguous()?;
        let b_cont = b_t.to_contiguous()?;
        let a_s = a_cont.as_slice();
        let b_s = b_cont.as_slice();
        let data: Vec<T> = (0..batch)
            .flat_map(|bi| {
                (0..m).map(move |i| {
                    (0..k)
                        .map(|j| a_s[bi * m * k + i * k + j] * b_s[bi * k + j])
                        .fold(T::zero(), |acc, v| acc + v)
                })
            })
            .collect();
        return Tensor::from_slice_on(vec![batch, m], &data, backend);
    }

    // "bi,bj->bij" — batched outer product
    if a_lhs == "bi" && b_lhs == "bj" && rhs == "bij" {
        if a.ndim() != 2 || b_t.ndim() != 2 {
            return Err(B::Error::from(BackendError::Storage {
                operation: "einsum bi,bj->bij",
                reason: "expected two rank-2 tensors".to_owned(),
            }));
        }
        let batch = a.shape()[0];
        let m = a.shape()[1];
        let n = b_t.shape()[1];
        if b_t.shape()[0] != batch {
            return Err(B::Error::from(BackendError::ShapeMismatch {
                operation: "einsum bi,bj->bij",
                lhs: a.shape().to_vec(),
                rhs: b_t.shape().to_vec(),
            }));
        }
        let a_cont = a.to_contiguous()?;
        let b_cont = b_t.to_contiguous()?;
        let a_s = a_cont.as_slice();
        let b_s = b_cont.as_slice();
        let data: Vec<T> = (0..batch)
            .flat_map(|bi| {
                (0..m).flat_map(move |i| (0..n).map(move |j| a_s[bi * m + i] * b_s[bi * n + j]))
            })
            .collect();
        return Tensor::from_slice_on(vec![batch, m, n], &data, backend);
    }

    Err(B::Error::from(BackendError::Storage {
        operation: "einsum",
        reason: format!("unsupported pattern {subscript:?}"),
    }))
}

/// Evaluate a 3-operand einsum by pairwise contraction.
///
/// Pattern `"abc,bcd,def->..."` is decomposed into two 2-operand einsums.
/// Only patterns expressible as two sequential 2-operand contractions are
/// supported.  The intermediate subscript is inferred automatically.
///
/// Currently supported:
/// - `"ij,jk,kl->il"` — two sequential matmuls (3-layer linear chain).
/// - `"bij,bjk,bkl->bil"` — batched 3-layer linear chain.
#[inline]
pub fn einsum3<T: Scalar, B: BackendOps<T> + Default>(
    subscript: &str,
    a: &Tensor<T, B>,
    b: &Tensor<T, B>,
    c: &Tensor<T, B>,
    backend: &B,
) -> Result<Tensor<T, B>, B::Error>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    let sub = subscript.trim();
    match sub {
        // "ij,jk,kl->il" — triple matmul chain
        "ij,jk,kl->il" => {
            let ab = einsum("ij,jk->ik", &[a, b], backend)?;
            einsum("ij,jk->ik", &[&ab, c], backend)
        }
        // "bij,bjk,bkl->bil" — batched triple matmul chain
        "bij,bjk,bkl->bil" => {
            let ab = einsum("bij,bjk->bik", &[a, b], backend)?;
            einsum("bij,bjk->bik", &[&ab, c], backend)
        }
        _ => Err(B::Error::from(BackendError::Storage {
            operation: "einsum3",
            reason: format!("unsupported pattern {subscript:?}"),
        })),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::SequentialBackend;
    use coeus_tensor::Tensor;

    fn b() -> SequentialBackend {
        SequentialBackend::new()
    }

    #[test]
    fn einsum_matmul() {
        let a = Tensor::from_slice(vec![2, 3], &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("construct tensor");
        let bt = Tensor::from_slice(vec![3, 2], &[7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0]).expect("construct tensor");
        let out = einsum("ij,jk->ik", &[&a, &bt], &b()).expect("valid einsum test shapes");
        assert_eq!(out.shape(), &[2, 2]);
        // row0: [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
        // row1: [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
        assert_eq!(out.as_slice(), &[58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn einsum_transpose() {
        let a = Tensor::from_slice(vec![2, 3], &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("construct tensor");
        let out = einsum("ij->ji", &[&a], &b()).expect("valid einsum test shapes");
        assert_eq!(out.shape(), &[3, 2]);
        assert_eq!(out.as_slice(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn einsum_dot_product() {
        let a = Tensor::from_slice(vec![4], &[1.0f32, 2.0, 3.0, 4.0]).expect("construct tensor");
        let bt = Tensor::from_slice(vec![4], &[5.0f32, 6.0, 7.0, 8.0]).expect("construct tensor");
        let out = einsum("i,i->", &[&a, &bt], &b()).expect("valid einsum test shapes");
        assert_eq!(out.shape(), &[1]);
        assert_eq!(
            out.as_slice(),
            &[1.0 * 5.0 + 2.0 * 6.0 + 3.0 * 7.0 + 4.0 * 8.0]
        );
    }

    #[test]
    fn einsum_trace() {
        let a = Tensor::from_slice(
            vec![3, 3],
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        ).expect("construct tensor");
        let out = einsum("ii->", &[&a], &b()).expect("valid einsum test shapes");
        assert_eq!(out.as_slice(), &[1.0 + 5.0 + 9.0]);
    }

    #[test]
    fn einsum_outer_product() {
        let a = Tensor::from_slice(vec![2], &[1.0f32, 2.0]).expect("construct tensor");
        let bt = Tensor::from_slice(vec![3], &[3.0f32, 4.0, 5.0]).expect("construct tensor");
        let out = einsum("i,j->ij", &[&a, &bt], &b()).expect("valid einsum test shapes");
        assert_eq!(out.shape(), &[2, 3]);
        assert_eq!(out.as_slice(), &[3.0, 4.0, 5.0, 6.0, 8.0, 10.0]);
    }

    #[test]
    fn einsum_matvec() {
        let a = Tensor::from_slice(vec![2, 3], &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("construct tensor");
        let v = Tensor::from_slice(vec![3], &[1.0f32, 0.0, 1.0]).expect("construct tensor");
        let out = einsum("ij,j->i", &[&a, &v], &b()).expect("valid einsum test shapes");
        assert_eq!(out.shape(), &[2]);
        assert_eq!(out.as_slice(), &[4.0, 10.0]);
    }

    #[test]
    fn einsum_batched_matmul() {
        // batch=1, m=2, k=2, n=2
        let a = Tensor::from_slice(vec![1, 2, 2], &[1.0f32, 2.0, 3.0, 4.0]).expect("construct tensor");
        let bt = Tensor::from_slice(vec![1, 2, 2], &[5.0f32, 6.0, 7.0, 8.0]).expect("construct tensor");
        let out = einsum("bij,bjk->bik", &[&a, &bt], &b()).expect("valid einsum test shapes");
        assert_eq!(out.shape(), &[1, 2, 2]);
        // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
        assert_eq!(out.as_slice(), &[19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn einsum_three_operand_matmul_chain() {
        let a = Tensor::from_slice(vec![2, 2], &[1.0f32, 2.0, 3.0, 4.0]).expect("construct tensor");
        let bt = Tensor::from_slice(vec![2, 2], &[5.0f32, 6.0, 7.0, 8.0]).expect("construct tensor");
        let c = Tensor::from_slice(vec![2, 2], &[9.0f32, 10.0, 11.0, 12.0]).expect("construct tensor");
        let out = einsum3("ij,jk,kl->il", &a, &bt, &c, &b())
            .expect("valid three-operand einsum test shapes");
        assert_eq!(out.shape(), &[2, 2]);
        assert_eq!(out.as_slice(), &[413.0, 454.0, 937.0, 1030.0]);
    }
}
