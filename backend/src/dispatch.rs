use super::{CpuBackend, traits::Backend};
use std::sync::Arc;
use thiserror::Error;
use tracing::{instrument, debug_span};
use crate::BackendData;
use coeus_dtype::Dtype;

#[derive(Error, Debug)]
pub enum DispatchError {
    #[error("Backend kind not supported: {0}")]
    Unsupported(String),
    #[error("Op execution failed: {0}")]
    Execution(String),
    #[error("Tensor shape mismatch")]
    ShapeMismatch,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BackendKind {
    Cpu,
    Gpu, // Vulkan/Metal stubs ADR-007 future
}

// Generic dispatch based on BackendKind
pub fn select_backend(_kind: BackendKind) -> Result<CpuBackend, DispatchError> {
    // For now, always return CPU backend (GPU stub)
    // Future: match kind { BackendKind::Cpu => CpuBackend::new(), BackendKind::Gpu => GpuBackend::new().map_or(CpuBackend::new(), |g| g) }
    Ok(CpuBackend)
}

// Enum for extensible op dispatch (ADR-027 OCP)
#[derive(Clone, Debug)]
pub enum Op<T: Dtype + Clone + Send + Sync + std::fmt::Debug> {
    Add { lhs: Arc<BackendData<T>>, rhs: Arc<BackendData<T>> },
    // ... Mul, Sub, etc. variants extensible no shims
}

impl<T: Dtype + Clone + Send + Sync + std::fmt::Debug> Op<T> {
    // Subfn <50 lines SRP
    fn validate_shapes(&self, result_shape: &[usize]) -> Result<(), DispatchError> {
        // Compute expected broadcast shape for operands and compare with provided result_shape.
        match self {
            Op::Add { lhs, rhs } => {
                let a_shape = lhs.shape();
                let b_shape = rhs.shape();
                let expected = compute_broadcast_shape(a_shape, b_shape)
                    .map_err(|_| DispatchError::ShapeMismatch)?;
                if expected.as_slice() == result_shape {
                    Ok(())
                } else {
                    Err(DispatchError::ShapeMismatch)
                }
            }
        }
    }
}

/// Compute NumPy-style broadcast shape for two shapes.
/// Returns Err if shapes are not broadcast-compatible.
fn compute_broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>, ()> {
    // Handle scalar empty-shape as compatible with any shape
    if a.is_empty() && b.is_empty() {
        return Ok(vec![]);
    }

    let mut res: Vec<usize> = Vec::new();
    let mut ia = a.iter().rev();
    let mut ib = b.iter().rev();

    loop {
        match (ia.next(), ib.next()) {
            (Some(&da), Some(&db)) => {
                if da == db || da == 1 || db == 1 {
                    res.push(std::cmp::max(da, db));
                } else {
                    return Err(());
                }
            }
            (Some(&da), None) => {
                res.push(da);
            }
            (None, Some(&db)) => {
                res.push(db);
            }
            (None, None) => break,
        }
    }

    res.reverse();
    Ok(res)
}

// Generic dispatch fn
#[instrument(skip(backend, op))]
pub fn execute_op<T: Dtype + Clone + Send + Sync + std::fmt::Debug, B: Backend<T> + Clone + Send + Sync>(
    backend: &B,
    op: Op<T>,
    result_shape: &[usize],
) -> Result<BackendData<T>, DispatchError> {
    op.validate_shapes(result_shape)?;
    let _span = debug_span!("execute", op=%std::any::type_name::<Op<T>>()).entered();
    match op {
        Op::Add { lhs, rhs } => {
            // Call backend add dispatch with BackendData refs
            backend
                .add(&*lhs, &*rhs)
                .map_err(|e| DispatchError::Execution(e.to_string()))
        }
        // ... extensible variants
    }
}

// Tests with proptest edges REQ-001
#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use rand::prelude::IteratorRandom;

    proptest! {
        // Generate broadcast-compatible shape pairs by constructing a base result
        // and deriving operand shapes that are trailing-subsequences with optional 1s.
        #[test]
        fn prop_broadcast_compatible(
            base_len in 0usize..4,
            base_elems in proptest::collection::vec(1usize..5, 0..4),
        ) {
            let base: Vec<usize> = base_elems.into_iter().take(base_len).collect();

            // Build two operand shapes from base by choosing trailing windows and replacing
            // some dimensions with 1 (guaranteed broadcast-compatible)
            let make_operand = |base: &[usize], shrink| {
                let len = base.len();
                // choose a trailing length
                let take = if len == 0 { 0 } else { (0..=len).choose(&mut rand::thread_rng()).unwrap_or(len) };
                let offset = len.saturating_sub(take);
                let mut shape: Vec<usize> = base[offset..].to_vec();
                if shrink {
                    for v in &mut shape {
                        if rand::random::<bool>() { *v = 1; }
                    }
                }
                shape
            };

            let a_shape = make_operand(&base, true);
            let b_shape = make_operand(&base, true);

            let backend = CpuBackend::default();

            // Populate actual data consistently so backend.add works (use f32 for numeric test)
            let a_numel: usize = a_shape.iter().product();
            let b_numel: usize = b_shape.iter().product();
            let a_data: Vec<f32> = (0..a_numel).map(|i| i as f32).collect();
            let b_data: Vec<f32> = if b_numel == 0 { vec![0.0f32] } else { (0..b_numel).map(|i| 1.0f32).collect() };

            let lhs_bd = Arc::new(BackendData::cpu(a_data.clone(), a_shape.clone()));
            let rhs_bd = Arc::new(BackendData::cpu(b_data.clone(), b_shape.clone()));

            // expected broadcast shape should compute successfully
            let expected = compute_broadcast_shape(&a_shape, &b_shape).unwrap();

            let op = Op::Add { lhs: lhs_bd, rhs: rhs_bd };
            let result = execute_op(&backend, op, &expected).unwrap();
            match &result {
                BackendData::Cpu { data, shape } => {
                    prop_assert_eq!(shape, &expected);
                    prop_assert_eq!(data.len(), expected.iter().product());
                }
                _ => panic!("Unexpected backend data type"),
            }
        }
    }

    #[test]
    fn test_broadcast_scalar_add() {
        let backend = CpuBackend::default();
        let lhs = Arc::new(BackendData::cpu(vec![1.0f32, 2.0, 3.0], vec![3]));
        let rhs = Arc::new(BackendData::cpu(vec![10.0f32], vec![])); // scalar
        let op = Op::Add { lhs: lhs.clone(), rhs: rhs.clone() };
        let expected_shape = compute_broadcast_shape(lhs.shape(), rhs.shape()).unwrap();
        let result = execute_op(&backend, op, &expected_shape).unwrap();
        match &result {
            BackendData::Cpu { data, shape } => {
                assert_eq!(shape, &vec![3]);
                assert_eq!(data, &vec![11.0f32, 12.0, 13.0]);
            }
            _ => panic!("Unexpected backend data type"),
        }
    }

    #[test]
    fn test_broadcast_incompatible() {
        let backend = CpuBackend::default();
        let lhs = Arc::new(BackendData::cpu(vec![1.0f32, 2.0], vec![2]));
        let rhs = Arc::new(BackendData::cpu(vec![1.0f32, 2.0, 3.0], vec![3]));
        let op = Op::Add { lhs, rhs };
        // Explicitly pass mismatched result shape; compute_broadcast_shape should Err
        assert!(compute_broadcast_shape(&[2], &[3]).is_err());
        let result = execute_op(&backend, op, &vec![2]);
        assert!(result.is_err());
    }
}