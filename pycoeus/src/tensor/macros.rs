

#[macro_export]
macro_rules! dispatch_tensor {
    ($tensor:expr, $inner:ident => $expr:expr) => {
        match &$tensor.inner {
            TensorWrapper::CpuDenseF32($inner) => $expr,
            TensorWrapper::CpuDenseF64($inner) => $expr,
            #[cfg(feature = "gpu")]
            TensorWrapper::GpuDenseF32($inner) => $expr,
            TensorWrapper::CpuSparseF32($inner) => $expr,
            TensorWrapper::CpuSparseF64($inner) => $expr,
            TensorWrapper::CpuDenseI64($inner) => $expr,
            TensorWrapper::CpuDenseC32($inner) => $expr,
            TensorWrapper::CpuStridedF32($inner) => $expr,
            TensorWrapper::CpuStridedF64($inner) => $expr,
            TensorWrapper::CpuStridedI64($inner) => $expr,
            #[cfg(feature = "gpu")]
            TensorWrapper::GpuStridedF32($inner) => $expr,
            TensorWrapper::CpuStridedC32($inner) => $expr,
        }
    };
}

#[macro_export]
macro_rules! dispatch_dense_tensor {
    ($tensor:expr, $inner:ident => $expr:expr) => {
        match &$tensor.inner {
            TensorWrapper::CpuDenseF32($inner) => $expr,
            TensorWrapper::CpuDenseF64($inner) => $expr,
            #[cfg(feature = "gpu")]
            TensorWrapper::GpuDenseF32($inner) => $expr,
            TensorWrapper::CpuDenseI64($inner) => $expr,
            TensorWrapper::CpuDenseC32($inner) => $expr,
            _ => Err(to_py_err("Operation requires dense storage")),
        }
    };
}

#[macro_export]
macro_rules! dispatch_ord_tensor {
    ($tensor:expr, $inner:ident => $expr:expr) => {
        match &$tensor.inner {
            TensorWrapper::CpuDenseF32($inner) => $expr,
            TensorWrapper::CpuDenseF64($inner) => $expr,
            #[cfg(feature = "gpu")]
            TensorWrapper::GpuDenseF32($inner) => $expr,
            TensorWrapper::CpuSparseF32($inner) => $expr,
            TensorWrapper::CpuSparseF64($inner) => $expr,
            TensorWrapper::CpuDenseI64($inner) => $expr,
            TensorWrapper::CpuStridedF32($inner) => $expr,
            TensorWrapper::CpuStridedF64($inner) => $expr,
            TensorWrapper::CpuStridedI64($inner) => $expr,
            #[cfg(feature = "gpu")]
            TensorWrapper::GpuStridedF32($inner) => $expr,
            _ => {
                Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                    "Operation not implemented for complex/unsupported tensors (no PartialOrd)"
                ))
            }
        }
    };
}

#[macro_export]
macro_rules! dispatch_float_tensor {
    ($tensor:expr, $inner:ident => $expr:expr) => {
        match &$tensor.inner {
            TensorWrapper::CpuDenseF32($inner) => $expr,
            TensorWrapper::CpuDenseF64($inner) => $expr,
            #[cfg(feature = "gpu")]
            TensorWrapper::GpuDenseF32($inner) => $expr,
            TensorWrapper::CpuSparseF32($inner) => $expr,
            TensorWrapper::CpuSparseF64($inner) => $expr,
            TensorWrapper::CpuStridedF32($inner) => $expr,
            TensorWrapper::CpuStridedF64($inner) => $expr,
            #[cfg(feature = "gpu")]
            TensorWrapper::GpuStridedF32($inner) => $expr,
            _ => {
                Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                    "Operation not implemented for integer/complex tensors"
                ))
            }
        }
    };
}

#[macro_export]
macro_rules! dispatch_float_dense_tensor {
    ($tensor:expr, $inner:ident => $expr:expr) => {
        match &$tensor.inner {
            TensorWrapper::CpuDenseF32($inner) => $expr,
            TensorWrapper::CpuDenseF64($inner) => $expr,
            #[cfg(feature = "gpu")]
            TensorWrapper::GpuDenseF32($inner) => $expr,
            _ => {
                Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                    "Operation requires dense float storage"
                ))
            }
        }
    };
}

#[macro_export]
macro_rules! dispatch_binary {
    ($lhs:expr, $rhs:expr, $a:ident, $b:ident => $expr:expr) => {
        {
            use crate::tensor::wrapper::WrapTensor;
            match (&$lhs.inner, &$rhs.inner) {
                (TensorWrapper::CpuDenseF32($a), TensorWrapper::CpuDenseF32($b)) => $expr,
                (TensorWrapper::CpuDenseF32($a), TensorWrapper::CpuStridedF32($b)) => $expr,
                (TensorWrapper::CpuStridedF32($a), TensorWrapper::CpuDenseF32($b)) => $expr,
                (TensorWrapper::CpuStridedF32($a), TensorWrapper::CpuStridedF32($b)) => $expr,

                (TensorWrapper::CpuDenseF64($a), TensorWrapper::CpuDenseF64($b)) => $expr,
                (TensorWrapper::CpuDenseF64($a), TensorWrapper::CpuStridedF64($b)) => $expr,
                (TensorWrapper::CpuStridedF64($a), TensorWrapper::CpuDenseF64($b)) => $expr,
                (TensorWrapper::CpuStridedF64($a), TensorWrapper::CpuStridedF64($b)) => $expr,

                (TensorWrapper::CpuDenseI64($a), TensorWrapper::CpuDenseI64($b)) => $expr,
                (TensorWrapper::CpuDenseI64($a), TensorWrapper::CpuStridedI64($b)) => $expr,
                (TensorWrapper::CpuStridedI64($a), TensorWrapper::CpuDenseI64($b)) => $expr,
                (TensorWrapper::CpuStridedI64($a), TensorWrapper::CpuStridedI64($b)) => $expr,

                (TensorWrapper::CpuSparseF32($a), TensorWrapper::CpuSparseF32($b)) => $expr,
                (TensorWrapper::CpuSparseF64($a), TensorWrapper::CpuSparseF64($b)) => $expr,
                _ => Err(to_py_err("Unsupported binary operation or mismatched storage formats")),
            }
        }
    };
}


#[macro_export]
macro_rules! dispatch_unary {
    ($self:expr, $inner:ident => $expr:expr) => {
        match &$self.inner {
            TensorWrapper::CpuDenseF32($inner) => $expr,
            TensorWrapper::CpuDenseF64($inner) => $expr,
            #[cfg(feature = "gpu")]
            TensorWrapper::GpuDenseF32($inner) => $expr,
            TensorWrapper::CpuStridedF32($inner) => $expr,
            TensorWrapper::CpuStridedF64($inner) => $expr,
            _ => Err(to_py_err("Unsupported unary operation")),
        }
    };
}
