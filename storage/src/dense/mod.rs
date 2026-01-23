//! Dense storage module
//!
//! Provides contiguous memory storage for tensors with row-major layout.
//!
//! ## Submodule Structure
//! - `core.rs` - Struct definition and accessors
//! - `creation.rs` - Constructors (from_vec, zeros, ones, full)
//! - `conversion.rs` - Format conversions (to_csr, to_csc, to_coo)
//! - `traits.rs` - Trait implementations (Storage, AsAny)

mod conversion;
mod core;
mod creation;
mod traits;

pub use self::core::DenseStorage;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Storage;
    use alloc::vec;
    use dtype::float::{Float32, Float64};
    use dtype::int::Int32;

    #[test]
    fn test_from_vec_correct_shape() {
        let data = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)];
        let storage = DenseStorage::from_vec(data, &[3]).unwrap();
        assert_eq!(storage.len(), 3);
        assert_eq!(storage.shape().dims(), &[3]);
    }

    #[test]
    fn test_from_vec_shape_mismatch() {
        let data = vec![Float32::new(1.0), Float32::new(2.0)];
        let result = DenseStorage::from_vec(data, &[3]);
        assert!(result.is_err());
    }

    #[test]
    fn test_zeros() {
        let storage = DenseStorage::<Float64>::zeros(&[2, 3]).unwrap();
        assert_eq!(storage.len(), 6);
        assert!(storage.as_slice().iter().all(num_traits::Zero::is_zero));
    }

    #[test]
    fn test_ones() {
        let storage = DenseStorage::<Float32>::ones(&[3]).unwrap();
        assert_eq!(storage.len(), 3);
        assert!(storage.as_slice().iter().all(num_traits::One::is_one));
    }

    #[test]
    fn test_is_contiguous() {
        let storage = DenseStorage::<Int32>::zeros(&[2, 3]).unwrap();
        assert!(storage.is_contiguous());
    }

    #[test]
    fn test_strides_row_major() {
        let storage = DenseStorage::<Float32>::zeros(&[2, 3, 4]).unwrap();
        assert_eq!(storage.strides(), &[12, 4, 1]);
    }
}
