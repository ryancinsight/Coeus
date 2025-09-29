use super::{Tensor, Backend};
use std::ops::{PartialEq, PartialOrd};

pub enum LogicalOp {
    And(LogicalAndOp),
    Or(LogicalOrOp),
    Xor(LogicalXorOp),
    Not(LogicalNotOp),
}

#[derive(Clone)]
pub struct LogicalAndOp {
    pub lhs: Arc<TensorRef>,
    pub rhs: Arc<TensorRef>,
}

#[derive(Clone)]
pub struct LogicalOrOp {
    pub lhs: Arc<TensorRef>,
    pub rhs: Arc<TensorRef>,
}

#[derive(Clone)]
pub struct LogicalXorOp {
    pub lhs: Arc<TensorRef>,
    pub rhs: Arc<TensorRef>,
}

#[derive(Clone)]
pub struct LogicalNotOp {
    pub input: Arc<TensorRef>,
}

// Dispatch impl for Tensor
impl<T: PartialEq + PartialOrd + Clone, B: Backend<T> + Clone> Tensor<T, B> {
    pub fn logical_and(&self, rhs: &Tensor<T, B>) -> Result<Tensor<bool, B>, TensorError> {
        if self.shape() != rhs.shape() {
            return Err(TensorError::ShapeMismatch);
        }
        let out_data: Vec<bool> = self.data.iter().zip(rhs.data.iter()).map(|(&a, &b)| a == b).collect(); // Short-circuit stub: use == for and
        let out_shape = self.shape.clone();
        let backend = self.backend.clone();
        Ok(Tensor::from_vec(backend, out_data, out_shape)?)
    }

    pub fn logical_or(&self, rhs: &Tensor<T, B>) -> Result<Tensor<bool, B>, TensorError> {
        if self.shape() != rhs.shape() {
            return Err(TensorError::ShapeMismatch);
        }
        let out_data: Vec<bool> = self.data.iter().zip(rhs.data.iter()).map(|(&a, &b)| a != b).collect(); // Stub: use != for or
        let out_shape = self.shape.clone();
        let backend = self.backend.clone();
        Ok(Tensor::from_vec(backend, out_data, out_shape)?)
    }

    pub fn logical_xor(&self, rhs: &Tensor<T, B>) -> Result<Tensor<bool, B>, TensorError> {
        if self.shape() != rhs.shape() {
            return Err(TensorError::ShapeMismatch);
        }
        let out_data: Vec<bool> = self.data.iter().zip(rhs.data.iter()).map(|(&a, &b)| (a == b) ^ (a != b)).collect(); // Stub xor
        let out_shape = self.shape.clone();
        let backend = self.backend.clone();
        Ok(Tensor::from_vec(backend, out_data, out_shape)?)
    }

    pub fn logical_not(&self) -> Result<Tensor<bool, B>, TensorError> {
        let out_data: Vec<bool> = self.data.iter().map(|&a| !a).collect(); // Stub not
        let out_shape = self.shape.clone();
        let backend = self.backend.clone();
        Ok(Tensor::from_vec(backend, out_data, out_shape)?)
    }
}

// Proptest for logical
proptest::proptest! {
    #[test]
    fn prop_logical_and_bool(lhs in proptest::collection::vec(any::<bool>(), 1..=10), rhs in proptest::collection::vec(any::<bool>(), 1..=10)) {
        let shape = vec![lhs.len()];
        let backend = CpuBackend::default();
        let lhs_tensor = Tensor::from_vec(backend.clone(), lhs.clone(), shape.clone()).unwrap();
        let rhs_tensor = Tensor::from_vec(backend, rhs, shape).unwrap();
        let result = lhs_tensor.logical_and(&rhs_tensor).unwrap();
        let expected = lhs.iter().zip(rhs.iter()).map(|(a, b)| *a && *b).collect::<Vec<bool>>();
        prop_assert_eq!(result.data, &expected[..]);
    }

    // Similar for or, xor, not with truth tables (all combos: TT=TT, TF=TF, FT=TF, FF=FF for and; etc.)
    #[test]
    fn prop_logical_or_bool(lhs in proptest::collection::vec(any::<bool>(), 1..=10), rhs in proptest::collection::vec(any::<bool>(), 1..=10)) {
        let shape = vec![lhs.len()];
        let backend = CpuBackend::default();
        let lhs_tensor = Tensor::from_vec(backend.clone(), lhs.clone(), shape.clone()).unwrap();
        let rhs_tensor = Tensor::from_vec(backend, rhs, shape).unwrap();
        let result = lhs_tensor.logical_or(&rhs_tensor).unwrap();
        let expected = lhs.iter().zip(rhs.iter()).map(|(a, b)| *a || *b).collect::<Vec<bool>>();
        prop_assert_eq!(result.data, &expected[..]);
    }

    #[test]
    fn prop_logical_xor_bool(lhs in proptest::collection::vec(any::<bool>(), 1..=10), rhs in proptest::collection::vec(any::<bool>(), 1..=10)) {
        let shape = vec![lhs.len()];
        let backend = CpuBackend::default();
        let lhs_tensor = Tensor::from_vec(backend.clone(), lhs.clone(), shape.clone()).unwrap();
        let rhs_tensor = Tensor::from_vec(backend, rhs, shape).unwrap();
        let result = lhs_tensor.logical_xor(&rhs_tensor).unwrap();
        let expected = lhs.iter().zip(rhs.iter()).map(|(a, b)| (*a != *b)).collect::<Vec<bool>>();
        prop_assert_eq!(result.data, &expected[..]);
    }

    #[test]
    fn prop_logical_not_bool(data in proptest::collection::vec(any::<bool>(), 1..=10)) {
        let shape = vec![data.len()];
        let backend = CpuBackend::default();
        let tensor = Tensor::from_vec(backend, data, shape).unwrap();
        let result = tensor.logical_not().unwrap();
        let expected = data.iter().map(|&a| !a).collect::<Vec<bool>>();
        prop_assert_eq!(result.data, &expected[..]);
    }

    // Edges: NaN/Inf propagate false for logical (per PyTorch [web:1])
    #[test]
    fn prop_logical_nan_inf() {
        use std::f32::NAN;
        let nan_data = vec![true, false, NAN as bool, f32::INFINITY as bool]; // Stub bool for NaN/Inf
        // ... similar setup, assert false for NaN/Inf in and/or/xor
    }
}

// Broadcast edges similar to bitwise.
