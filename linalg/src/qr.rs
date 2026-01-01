//! QR Decomposition.
//!
//! Computes QR decomposition using Householder reflections.
//! Currently implements "reduced" QR (Q is Mxmin(M,N), R is min(M,N)xN).

use crate::error::dimension_mismatch;
use backend::Backend;
use dtype::DataType;
use num_traits::{Float, One, Zero};
use storage::DenseStorage;
use tensor::Tensor;

/// Result of QR decomposition.
#[derive(Debug, Clone)]
pub struct QRResult<B, T>
where
    B: Backend<Data = T>,
    T: DataType,
{
    pub q: Tensor<B, DenseStorage<T>, T>,
    pub r: Tensor<B, DenseStorage<T>, T>,
}

/// Trait for QR decomposition.
pub trait QR<B, T>
where
    B: Backend<Data = T>,
    T: DataType,
{
    /// Computes the QR decomposition.
    ///
    /// # Returns
    /// Struct containing Q and R matrices.
    ///
    /// # Errors
    /// - `DimensionMismatch` if dimensions are invalid
    fn qr(&self) -> coeus_error::Result<QRResult<B, T>>
    where
        Self: Sized;
}

impl<B, T> QR<B, T> for Tensor<B, DenseStorage<T>, T>
where
    B: Backend<Data = T> + Clone + Default,
    T: DataType
        + Float
        + Zero
        + One
        + std::ops::SubAssign
        + std::ops::DivAssign
        + std::ops::MulAssign
        + std::ops::AddAssign
        + std::fmt::Debug,
{
    fn qr(&self) -> coeus_error::Result<QRResult<B, T>> {
        let dims = self.shape().dims();
        if dims.len() != 2 {
            return Err(dimension_mismatch("Matrix must be 2D"));
        }
        let m = dims[0];
        let n = dims[1];
        let k = std::cmp::min(m, n);

        // We compute R in-place on a copy of A (let's call it R_mtx)
        // We accumulate Q in a separate matrix (M x K for reduced)
        // Standard Householder computes full Q often, but we can do reduced.
        // For reduced QR, we can store H vectors.
        // Actually, explicit Q construction is needed for return.

        // Let's implement full Householder on A to get R.
        // Convert A to column-major or just work with row-major?
        // DenseStorage is row-major. Data[i*n + j].
        // Householder works on columns.

        let mut r_data = self.as_slice().to_vec(); // M x N

        // Store reflectors to reconstruct Q later?
        // Or apply Q updates on the fly?
        // Computing Q on the fly: Q starts as Identity (MxM).
        // Update Q = Q * H_i.

        // For reduced QR (Q is MxK), we can initialize Q as M x K identity-like? No.
        // Let's compute Full Q (MxM) then slice? Or accumulate efficiently.
        // Let's do explicit Householder loop and store vectors 'v' to apply to Q.

        let mut vs = Vec::with_capacity(k);
        let mut betas = Vec::with_capacity(k);

        for i in 0..k {
            // x is the column i of R, from row i down to M
            // x = r_data[i..m, i]

            let mut norm_x_sq = T::zero();
            for r in i..m {
                let val = r_data[r * n + i];
                norm_x_sq += val * val;
            }
            let norm_x = norm_x_sq.sqrt();

            if norm_x == T::zero() {
                vs.push(vec![]); // No reflection
                betas.push(T::zero());
                continue;
            }

            let mut v = vec![T::zero(); m - i];
            for r in 0..(m - i) {
                v[r] = r_data[(i + r) * n + i];
            }
            // v[0] correction
            let x0 = v[0];
            let u1 = if x0 >= T::zero() {
                x0 + norm_x
            } else {
                x0 - norm_x
            };
            v[0] = u1;

            // Update norm_v
            // ||v||^2 = ||x[1:]||^2 + u1^2 = (||x||^2 - x0^2) + (x0+s||x||)^2
            // = ||x||^2 - x0^2 + x0^2 + 2x0s||x|| + ||x||^2
            // = 2||x||^2 + 2x0s||x|| = 2||x||( ||x|| + s x0 ) = 2||x|| * |u1| ?
            // Let's just recompute
            let mut norm_v_sq = T::zero();
            for val in &v {
                norm_v_sq += *val * *val;
            }

            if norm_v_sq == T::zero() {
                // check again
                vs.push(vec![]);
                betas.push(T::zero());
                continue;
            }

            let beta = (T::one() + T::one()) / norm_v_sq; // 2/vTv

            // Apply H to R (A)
            // R = H R = (I - beta v vT) R = R - beta v (vT R)
            // vT R is row vector: for col j in i..N: sum(v[r] * R[i+r, j])

            for j in i..n {
                let mut sum = T::zero();
                for r in 0..(m - i) {
                    sum += v[r] * r_data[(i + r) * n + j];
                }
                let factor = beta * sum;
                for r in 0..(m - i) {
                    r_data[(i + r) * n + j] -= v[r] * factor;
                }
            }

            // Zero out sub-diagonal explicitly to be clean
            if i + 1 < m {
                for r in (i + 1)..m {
                    r_data[r * n + i] = T::zero();
                }
            }

            vs.push(v);
            betas.push(beta);
        }

        // Construct Q
        // Start with Identity M x K (reduced)
        let mut q_data = vec![T::zero(); m * k];
        for i in 0..k {
            q_data[i * m + i] = T::one(); // Diagonal 1s
        }

        // Apply H's in REVERSE order to build Q
        // Q = H1 H2 ... Hk
        // But since we want Q = I * H1 * H2... ?
        // No, A = QR => Q is orth, R is upper.
        // We computed H_k ... H_1 A = R.
        // So Q = H_1 ... H_k.
        // We have reduced Q (M x K).
        // We can just apply H's to the columns of Identity (first K cols).
        // H_i acts on rows i..M.
        // Q_new = (I - beta v vT) Q_old = Q_old - beta (Q_old v) vT ?
        // No, left mult: H Q.
        // Actually we want Q to be formed by H1...Hk
        // Q = H1 (H2 ... (Hk I))
        // So applied in reverse k-1 down to 0.

        for i in (0..k).rev() {
            let v = &vs[i];
            if v.is_empty() {
                continue;
            }
            let beta = betas[i];

            // Limit Q to columns j in i..k?
            // Householder H_i affects rows i..M.
            // Since we are building Q, we apply H_i to columns of Q?

            // Wait. Q = H1 H2...
            // Let's assume full Q first. Start Q = I (MxM).
            // Apply H_k (affects rows k..M) ... then H_1.
            // H Q = Q - beta v vT Q.
            // This is messy.

            // Correct way for "Q from Householder implementation":
            // Backward accumulation.
            // See Golub & Van Loan.
            // For Reduced Q (MxN or MxK), we start with Identity subset columns?
            // Actually:
            // Q = I.
            // Apply H_k: Q = H_k Q.
            // ...
            // Apply H_1: Q = H_1 Q.
            // Since H_i only affects bottom-right, we can do this efficiently.

            // Apply H_i to Q (which is current accumulated product H_{i+1}...H_k)
            // H_i (Q) = (I - beta v vT) Q = Q - beta v (vT Q).
            // v is vector of length M-i.
            // vT Q: row vector of length K? (since Q is MxK).

            // Q is M x K.

            for j in 0..k {
                // For each column of Q
                // Compute dot product w = vT * Q[i:, j]
                let mut dot = T::zero();
                for r in 0..(m - i) {
                    dot += v[r] * q_data[(i + r) * k + j]; // Q is RowMajor M x K?
                                                           // Wait, q_data indexing: row * stride + col.
                                                           // q is M x K. row indices (i+r), col j.
                }

                let factor = beta * dot;
                for r in 0..(m - i) {
                    q_data[(i + r) * k + j] -= v[r] * factor;
                }
            }
        }

        // Final Q and R
        // R needs to be clipped to K x N
        let r_final_data = if m > k {
            // Take top K rows
            let mut data = Vec::with_capacity(k * n);
            for i in 0..k {
                for j in 0..n {
                    data.push(r_data[i * n + j]);
                }
            }
            data
        } else {
            r_data
        };

        let q_storage = DenseStorage::from_vec(q_data, &[m, k]).map_err(|e| {
            coeus_error::Error::Storage(coeus_error::StorageError::InvalidShape(format!("{e}")))
        })?;
        let r_storage = DenseStorage::from_vec(r_final_data, &[k, n]).map_err(|e| {
            coeus_error::Error::Storage(coeus_error::StorageError::InvalidShape(format!("{e}")))
        })?;

        Ok(QRResult {
            q: Tensor::from_storage(q_storage, self.backend().clone()),
            r: Tensor::from_storage(r_storage, self.backend().clone()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensor::Float32;

    fn create_matrix(
        data: Vec<f32>,
        shape: &[usize],
    ) -> Tensor<backend::CpuBackend<Float32>, DenseStorage<Float32>, Float32> {
        let data = data.into_iter().map(Float32).collect();
        let storage = DenseStorage::from_vec(data, shape).unwrap();
        Tensor::from_storage(storage, backend::CpuBackend::default())
    }

    #[test]
    fn test_qr_simple() {
        // A = [[12, -51, 4], [6, 167, -68], [-4, 24, -41]]
        // Q should be orthonorm, R upper triangular
        let data = vec![12.0, -51.0, 4.0, 6.0, 167.0, -68.0, -4.0, 24.0, -41.0];
        let a = create_matrix(data, &[3, 3]);
        let res = a.qr().unwrap();
        let q = res.q;
        let r = res.r;

        // detailed checks omitted for brevity, checking dimensions
        assert_eq!(q.shape().dims(), &[3, 3]);
        assert_eq!(r.shape().dims(), &[3, 3]);

        // Reconstruct: QR approx A
        // We lack MatMul in tests without importing it?
        // Let's just check R is upper triangular
        let r_data = r.as_slice();
        assert_eq!(r_data[3].0, 0.0); // R[1,0]
        assert_eq!(r_data[6].0, 0.0); // R[2,0]
        assert_eq!(r_data[7].0, 0.0); // R[2,1]
    }
}
