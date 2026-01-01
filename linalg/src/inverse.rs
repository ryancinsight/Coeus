use crate::error::{dimension_mismatch, not_square, singular_matrix};
use backend::Backend;
use dtype::DataType;
use num_traits::{Float, One, Zero};
use storage::DenseStorage;
use tensor::Tensor;

pub trait Inverse<B, T> {
    fn inv(&self) -> coeus_error::Result<Self>
    where
        Self: Sized;
}

impl<B, T> Inverse<B, T> for Tensor<B, DenseStorage<T>, T>
where
    B: Backend<Data = T> + Clone + Default,
    T: DataType
        + Float
        + Zero
        + One
        + std::ops::SubAssign
        + std::ops::DivAssign
        + std::ops::MulAssign
        + std::fmt::Debug,
{
    fn inv(&self) -> coeus_error::Result<Self> {
        let dims = self.shape().dims();
        if dims.len() != 2 {
            return Err(dimension_mismatch("Matrix is not 2D"));
        }
        let n = dims[0];
        if n != dims[1] {
            return Err(not_square(n, dims[1]));
        }

        // Clone data for Gaussian elimination (Augmented matrix [A|I])
        let mut data = self.as_slice().to_vec();
        // Create Identity matrix data
        let mut inv_data = vec![T::zero(); n * n];
        for i in 0..n {
            inv_data[i * n + i] = T::one();
        }

        // Gaussian Elimination
        for i in 0..n {
            // Find pivot
            let mut pivot_idx = i;
            let mut max_val = data[i * n + i].abs();

            for k in (i + 1)..n {
                let val = data[k * n + i].abs();
                if val > max_val {
                    max_val = val;
                    pivot_idx = k;
                }
            }

            if max_val == T::zero() {
                return Err(singular_matrix());
            }

            // Swap rows if needed
            if pivot_idx != i {
                for j in 0..n {
                    data.swap(i * n + j, pivot_idx * n + j);
                    inv_data.swap(i * n + j, pivot_idx * n + j);
                }
            }

            // Scale pivot row
            let pivot = data[i * n + i];
            for j in 0..n {
                data[i * n + j] /= pivot;
                inv_data[i * n + j] /= pivot;
            }

            // Eliminate other rows
            for k in 0..n {
                if k != i {
                    let factor = data[k * n + i];
                    for j in 0..n {
                        let d_val = data[i * n + j] * factor; // Read before write
                        data[k * n + j] -= d_val;
                        let i_val = inv_data[i * n + j] * factor;
                        inv_data[k * n + j] -= i_val;
                    }
                }
            }
        }

        let storage = DenseStorage::from_vec(inv_data, &[n, n]).map_err(|e| {
            coeus_error::Error::Storage(coeus_error::StorageError::InvalidShape(format!("{e}")))
        })?;

        Ok(Tensor::from_storage(storage, self.backend().clone()))
    }
}
