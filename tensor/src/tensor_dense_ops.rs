
    #[error("Serialization error: {0}")]
    SerdeError(#[from] serde_json::Error),

    #[error("Bincode error: {0}")]
    BincodeError(Box<bincode::ErrorKind>),

    #[error("Error: {0}")]
    StringError(String),
}

impl From<Box<bincode::ErrorKind>> for TensorError {
    fn from(err: Box<bincode::ErrorKind>) -> Self {
        TensorError::BincodeError(err)
    }
}

impl From<String> for TensorError {
    fn from(err: String) -> Self {
        TensorError::StringError(err)
    }
}


/// Memory layout for tensors
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Layout {
    /// Row-major (C-style) layout
    #[default]
    Contiguous,
    /// Column-major (Fortran-style) layout
    Fortran,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let t = Tensor::from_vec(CpuBackend::new(), vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        assert_eq!(t.shape(), &[3]);
        assert_eq!(t.numel(), 3);
    }

    #[test]
    fn test_tensor_addition() {
        let a = Tensor::from_vec(CpuBackend::default(), vec![1.0, 2.0], vec![2]).unwrap();
        let b = Tensor::from_vec(CpuBackend::default(), vec![3.0, 4.0], vec![2]).unwrap();
        // Note: Addition operator not yet implemented, this is a placeholder test
        // let c = &a + &b;
        // assert_eq!(c.unwrap().data(), &[4.0, 6.0]);
    }

    #[test]
    fn test_requires_grad() {
        let mut t = Tensor::from_vec(CpuBackend::new(), vec![1.0, 2.0], vec![2]).unwrap();
        t.set_requires_grad(true);

        assert!(t.requires_grad());
        assert!(t.grad().is_none());
    }
}

impl From<coeus_storage::StorageError> for TensorError {
    fn from(err: coeus_storage::StorageError) -> Self {
        TensorError::StorageError(err.to_string())
    }
}

#[cfg(test)]
include!("tests/autograd_tests.rs");

#[cfg(test)]
include!("tests/property_tests.rs");

#[cfg(test)]
include!("tests/autograd/numerical_gradient_tests.rs");


// Test integration removed - async_view method not implemented
// Tests are included via include! macros above

/// Const generics/Cow full (example in Tensor impl if not, but for ops in submods)
impl<T: Dtype, B: Backend<Data = T> + Clone + Send + Sync + Default, S: TensorStorage<T> + Clone + Send + Sync> Tensor<T, B, S> {
    pub fn view_cow(&self) -> Cow<'_, [T]> {
        Cow::Borrowed(self.data())
    }

    pub fn from_maybe_uninit(shape: Vec<usize>) -> Self where T: Default {
        let numel = shape.iter().product();
        let mut data = vec![MaybeUninit::uninit(); numel];
        // Init with Default or zero
        for i in 0..numel {
            data[i].write(T::default());
        }
        let data_init: Vec<T> = unsafe { std::mem::transmute(data) };
        let backend = B::default();
        Tensor::from_vec(backend, data_init, shape).unwrap()
    }
}

