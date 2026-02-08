use crate::DataType;
use std::fmt::{self, Debug};
use ::storage; // Import storage crate directly
use crate::BackendError; // Import BackendError from crate root

type Result<T> = std::result::Result<T, BackendError>;

/// Backend types available for selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BackendType {
    /// CPU backend
    Cpu,
    /// GPU backend
    Gpu,
    /// Tensor Processing Unit
    Tpu,
    /// Neural Processing Unit
    Npu,
}

impl core::fmt::Display for BackendType {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            BackendType::Cpu => write!(f, "CPU"),
            BackendType::Gpu => write!(f, "GPU"),
            BackendType::Tpu => write!(f, "TPU"),
            BackendType::Npu => write!(f, "NPU"),
        }
    }
}

/// Device information trait
pub trait DeviceInfo: fmt::Debug + Send + Sync {
    fn name(&self) -> &str;
    fn memory_total(&self) -> Option<usize>;
    fn memory_available(&self) -> Option<usize>;
    fn compute_capability(&self) -> Option<String>;
}

pub trait Backend: Debug + Clone + Send + Sync + Default + 'static {
    type Data: DataType;
    type Device: DeviceInfo + Send + Sync;
    
    fn device(&self) -> &Self::Device;
    fn supports(&self, operation: &str) -> bool;
    fn device_name(&self) -> &str;
    fn device_info(&self) -> Box<dyn DeviceInfo>;

    /// Add dense storage element-wise
    fn add_dense(
        &self,
        lhs: &storage::DenseStorage<Self::Data>,
        rhs: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>>;

    /// Add strided storage element-wise
    fn add_strided(
        &self,
        lhs: &storage::StridedStorage<Self::Data>,
        rhs: &storage::StridedStorage<Self::Data>,
    ) -> Result<storage::StridedStorage<Self::Data>>;

    /// Element-wise negation for CSR sparse storage
    fn neg_csr(
        &self,
        _input: &storage::CsrStorage<Self::Data>,
    ) -> Result<storage::CsrStorage<Self::Data>>
    where
        Self::Data: core::ops::Neg<Output = Self::Data> {
        Err(crate::BackendError::UnsupportedOperation {
            operation: "neg_csr".to_string(),
            backend: self.device_name().to_string(),
        })
    }

    /// Element-wise negation for strided storage
    fn neg_strided(
        &self,
        _input: &storage::StridedStorage<Self::Data>,
    ) -> Result<storage::StridedStorage<Self::Data>>
    where
        Self::Data: core::ops::Neg<Output = Self::Data> {
        Err(crate::BackendError::UnsupportedOperation {
            operation: "neg_strided".to_string(),
            backend: self.device_name().to_string(),
        })
    }

    /// Multiply dense storage element-wise
    fn mul_dense(
        &self,
        lhs: &storage::DenseStorage<Self::Data>,
        rhs: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>>;

    /// Multiply strided storage element-wise
    fn mul_strided(
        &self,
        lhs: &storage::StridedStorage<Self::Data>,
        rhs: &storage::StridedStorage<Self::Data>,
    ) -> Result<storage::StridedStorage<Self::Data>>;

    /// Matrix multiplication for dense storage
    fn matmul_dense(
        &self,
        lhs: &storage::DenseStorage<Self::Data>,
        rhs: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>>;

    /// Add matrix multiplication with scaling (beta * input + alpha * (mat1 @ mat2))
    fn addmm_dense(
        &self,
        _input: &storage::DenseStorage<Self::Data>,
        _mat1: &storage::DenseStorage<Self::Data>,
        _mat2: &storage::DenseStorage<Self::Data>,
        _beta: Self::Data,
        _alpha: Self::Data,
    ) -> Result<storage::DenseStorage<Self::Data>> {
         Err(crate::BackendError::UnsupportedOperation {
            operation: "addmm_dense".to_string(),
            backend: self.device_name().to_string(),
        })
    }

    /// Add matrix-vector multiplication with scaling (beta * input + alpha * (mat @ vec))
    fn addmv_dense(
        &self,
        _input: &storage::DenseStorage<Self::Data>,
        _mat: &storage::DenseStorage<Self::Data>,
        _vec: &storage::DenseStorage<Self::Data>,
        _beta: Self::Data,
        _alpha: Self::Data,
    ) -> Result<storage::DenseStorage<Self::Data>> {
         Err(crate::BackendError::UnsupportedOperation {
            operation: "addmv_dense".to_string(),
            backend: self.device_name().to_string(),
        })
    }

    /// Apply ReLU activation to dense storage
    fn relu_dense(
        &self,
        input: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>>
    where
        Self::Data: PartialOrd + Default;

    /// Apply sigmoid activation to dense storage
    fn sigmoid_dense(
        &self,
        input: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>>
    where
        Self::Data: dtype::num_traits::Float;

    /// Apply ReLU activation to strided storage
    fn relu_strided(
        &self,
        _input: &storage::StridedStorage<Self::Data>,
    ) -> Result<storage::StridedStorage<Self::Data>>
    where
        Self::Data: PartialOrd + Default {
        Err(crate::BackendError::UnsupportedOperation {
            operation: "relu_strided".to_string(),
            backend: self.device_name().to_string(),
        })
    }

    /// Apply sigmoid activation to strided storage
    fn sigmoid_strided(
        &self,
        _input: &storage::StridedStorage<Self::Data>,
    ) -> Result<storage::StridedStorage<Self::Data>>
    where
        Self::Data: dtype::num_traits::Float {
        Err(crate::BackendError::UnsupportedOperation {
            operation: "sigmoid_strided".to_string(),
            backend: self.device_name().to_string(),
        })
    }

    /// Sum all elements in dense storage
    fn sum_dense(&self, input: &storage::DenseStorage<Self::Data>) -> Result<Self::Data>;

    /// Find maximum value in dense storage
    fn max_dense(&self, input: &storage::DenseStorage<Self::Data>) -> Result<Self::Data>
    where
        Self::Data: PartialOrd;

    /// Find minimum value in dense storage
    fn min_dense(&self, input: &storage::DenseStorage<Self::Data>) -> Result<Self::Data>
    where
        Self::Data: PartialOrd;

    /// Sum all elements in strided storage
    fn sum_strided(
        &self,
        _input: &storage::StridedStorage<Self::Data>,
    ) -> Result<Self::Data> {
        Err(crate::BackendError::UnsupportedOperation {
            operation: "sum_strided".to_string(),
            backend: self.device_name().to_string(),
        })
    }
    
    /// Find maximum value in strided storage
    fn max_strided(
        &self,
        _input: &storage::StridedStorage<Self::Data>,
    ) -> Result<Self::Data>
    where
        Self::Data: PartialOrd {
        Err(crate::BackendError::UnsupportedOperation {
            operation: "max_strided".to_string(),
            backend: self.device_name().to_string(),
        })
    }

    /// Find minimum value in strided storage
    fn min_strided(
        &self,
        _input: &storage::StridedStorage<Self::Data>,
    ) -> Result<Self::Data>
    where
        Self::Data: PartialOrd {
        Err(crate::BackendError::UnsupportedOperation {
            operation: "min_strided".to_string(),
            backend: self.device_name().to_string(),
        })
    }

    /// Find index of maximum value in dense storage
    fn argmax_dense(&self, input: &storage::DenseStorage<Self::Data>) -> Result<usize>
    where
        Self::Data: PartialOrd;

    /// Find index of minimum value in dense storage
    fn argmin_dense(&self, input: &storage::DenseStorage<Self::Data>) -> Result<usize>
    where
        Self::Data: PartialOrd;

    /// Subtract dense storages element-wise
    fn sub_dense(
        &self,
        lhs: &storage::DenseStorage<Self::Data>,
        rhs: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>>;

    /// Subtract strided storages element-wise
    fn sub_strided(
        &self,
        lhs: &storage::StridedStorage<Self::Data>,
        rhs: &storage::StridedStorage<Self::Data>,
    ) -> Result<storage::StridedStorage<Self::Data>>;

    /// Apply exponential function element-wise
    fn exp_dense(
        &self,
        input: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>>
    where
        Self::Data: dtype::num_traits::Float;

    /// Apply natural logarithm element-wise
    fn log_dense(
        &self,
        input: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>>
    where
        Self::Data: dtype::num_traits::Float;

    /// Apply sine function element-wise
    fn sin_dense(
        &self,
        input: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>>
    where
        Self::Data: dtype::num_traits::Float;

    /// Apply cosine function element-wise
    fn cos_dense(
        &self,
        input: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>>
    where
        Self::Data: dtype::num_traits::Float;

    /// Apply tangent function element-wise
    fn tan_dense(
        &self,
        input: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>>
    where
        Self::Data: dtype::num_traits::Float;

    /// Apply arc sine function element-wise
    fn asin_dense(
        &self,
        input: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>>
    where
        Self::Data: dtype::num_traits::Float;

    /// Apply arc cosine function element-wise
    fn acos_dense(
        &self,
        input: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>>
    where
        Self::Data: dtype::num_traits::Float;

    /// Apply arc tangent function element-wise
    fn atan_dense(
        &self,
        input: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>>
    where
        Self::Data: dtype::num_traits::Float;

    /// Apply hyperbolic sine function element-wise
    fn sinh_dense(
        &self,
        input: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>>
    where
        Self::Data: dtype::num_traits::Float;

    /// Apply hyperbolic cosine function element-wise
    fn cosh_dense(
        &self,
        input: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>>
    where
        Self::Data: dtype::num_traits::Float;

    /// Apply hyperbolic tangent function element-wise
    fn tanh_dense(
        &self,
        input: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>>
    where
        Self::Data: dtype::num_traits::Float;

    /// Apply tanh activation to strided storage
    fn tanh_strided(
        &self,
        _input: &storage::StridedStorage<Self::Data>,
    ) -> Result<storage::StridedStorage<Self::Data>>
    where
        Self::Data: dtype::num_traits::Float {
        Err(crate::BackendError::UnsupportedOperation {
            operation: "tanh_strided".to_string(),
            backend: self.device_name().to_string(),
        })
    }

    /// Apply GELU activation to strided storage
    fn gelu_strided(
        &self,
        _input: &storage::StridedStorage<Self::Data>,
    ) -> Result<storage::StridedStorage<Self::Data>>
    where
        Self::Data: dtype::num_traits::Float {
        Err(crate::BackendError::UnsupportedOperation {
            operation: "gelu_strided".to_string(),
            backend: self.device_name().to_string(),
        })
    }

    /// Apply GELU activation element-wise
    fn gelu_dense(
        &self,
        input: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>>
    where
        Self::Data: dtype::num_traits::Float;

    /// Apply square root function element-wise
    fn sqrt_dense(
        &self,
        input: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>>
    where
        Self::Data: dtype::num_traits::Float;

    /// Apply absolute value function element-wise
    fn abs_dense(
        &self,
        input: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>>
    where
        Self::Data: dtype::num_traits::Signed;

    /// Apply floor function element-wise
    fn floor_dense(
        &self,
        input: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>>
    where
        Self::Data: dtype::num_traits::Float;

    /// Apply exponential function to strided storage
    fn exp_strided(
        &self,
        _input: &storage::StridedStorage<Self::Data>,
    ) -> Result<storage::StridedStorage<Self::Data>>
    where
        Self::Data: dtype::num_traits::Float {
        Err(crate::BackendError::UnsupportedOperation {
            operation: "exp_strided".to_string(),
            backend: self.device_name().to_string(),
        })
    }

    /// Apply natural logarithm to strided storage
    fn log_strided(
        &self,
        _input: &storage::StridedStorage<Self::Data>,
    ) -> Result<storage::StridedStorage<Self::Data>>
    where
        Self::Data: dtype::num_traits::Float {
        Err(crate::BackendError::UnsupportedOperation {
            operation: "log_strided".to_string(),
            backend: self.device_name().to_string(),
        })
    }
    
    /// Apply sqrt function to strided storage
    fn sqrt_strided(
        &self,
        _input: &storage::StridedStorage<Self::Data>,
    ) -> Result<storage::StridedStorage<Self::Data>>
    where
        Self::Data: dtype::num_traits::Float {
        Err(crate::BackendError::UnsupportedOperation {
            operation: "sqrt_strided".to_string(),
            backend: self.device_name().to_string(),
        })
    }
    
    /// Apply abs function to strided storage
    fn abs_strided(
        &self,
        _input: &storage::StridedStorage<Self::Data>,
    ) -> Result<storage::StridedStorage<Self::Data>>
    where
        Self::Data: dtype::num_traits::Signed {
        Err(crate::BackendError::UnsupportedOperation {
            operation: "abs_strided".to_string(),
            backend: self.device_name().to_string(),
        })
    }

    /// Apply ceil function element-wise
    fn ceil_dense(
        &self,
        input: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>>
    where
        Self::Data: dtype::num_traits::Float;

    /// Apply round function element-wise
    fn round_dense(
        &self,
        input: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>>
    where
        Self::Data: dtype::num_traits::Float;

    /// Cholesky decomposition: A = L L^T (for symmetric positive-definite A)
    fn cholesky_dense(
        &self,
        input: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>>
    where
        Self::Data: dtype::num_traits::Float;

    /// QR decomposition: A = QR (Q orthogonal, R upper triangular)
    fn qr_dense(
        &self,
        input: &storage::DenseStorage<Self::Data>,
    ) -> Result<(storage::DenseStorage<Self::Data>, storage::DenseStorage<Self::Data>)>
    where
        Self::Data: dtype::num_traits::Float;

    /// SVD decomposition: A = U S V^T
    fn svd_dense(
        &self,
        input: &storage::DenseStorage<Self::Data>,
    ) -> Result<(
        storage::DenseStorage<Self::Data>,
        storage::DenseStorage<Self::Data>,
        storage::DenseStorage<Self::Data>,
    )>
    where
        Self::Data: dtype::num_traits::Float;

    /// Select values from the input tensor using the given indices.
    fn take_dense(
        &self,
        input: &storage::DenseStorage<Self::Data>,
        indices: &storage::DenseStorage<dtype::int::Int64>,
    ) -> Result<storage::DenseStorage<Self::Data>>;

    /// Place values into the input tensor at the given indices.
    fn put_dense(
        &self,
        input: &mut storage::DenseStorage<Self::Data>,
        indices: &storage::DenseStorage<dtype::int::Int64>,
        values: &storage::DenseStorage<Self::Data>,
        accumulate: bool,
    ) -> Result<()>;

    /// Apply 2D convolution
    fn conv2d_dense(
        &self,
        input: &storage::DenseStorage<Self::Data>,
        weight: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>>;

    /// Compute mean along specified axes (dense)
    fn mean_dense(
        &self,
        input: &storage::DenseStorage<Self::Data>,
        axes: Option<&[usize]>,
    ) -> Result<storage::DenseStorage<Self::Data>>;

    /// Sparse matrix-matrix multiplication (CSR format)
    fn spmm_csr(
        &self,
        data: &[Self::Data],
        indices: &[usize],
        indptr: &[usize],
        other: &storage::DenseStorage<Self::Data>,
        num_rows: usize,
        num_cols: usize,
    ) -> Result<Vec<Self::Data>>;

    /// Sparse matrix-vector multiplication (CSR format)
    fn spmv_csr(
        &self,
        data: &[Self::Data],
        indices: &[usize],
        indptr: &[usize],
        vector: &[Self::Data],
        num_rows: usize,
        num_cols: usize,
    ) -> Result<Vec<Self::Data>>;

    /// Coordinate format sparse matrix multiplication (matrix-sparse)
    fn coo_matmul_sparse(
        &self,
        lhs_data: &[Self::Data],
        lhs_row: &[usize],
        lhs_col: &[usize],
        rhs_data: &[Self::Data],
        rhs_row: &[usize],
        rhs_col: &[usize],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<storage::CsrStorage<Self::Data>>;

    /// Coordinate format sparse matrix multiplication (sparse-dense)
    fn coo_matmul_dense(
        &self,
        lhs_data: &[Self::Data],
        lhs_row: &[usize],
        lhs_col: &[usize],
        rhs: &storage::DenseStorage<Self::Data>,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<storage::DenseStorage<Self::Data>>;

    /// Coordinate format sparse addition
    fn coo_add_sparse(
        &self,
        lhs_data: &[Self::Data],
        lhs_row: &[usize],
        lhs_col: &[usize],
        rhs_data: &[Self::Data],
        rhs_row: &[usize],
        rhs_col: &[usize],
        m: usize,
        n: usize,
    ) -> Result<storage::CsrStorage<Self::Data>>;

    /// Coordinate format sparse multiplication
    fn coo_mul_sparse(
        &self,
        lhs_data: &[Self::Data],
        lhs_row: &[usize],
        lhs_col: &[usize],
        rhs_data: &[Self::Data],
        rhs_row: &[usize],
        rhs_col: &[usize],
        m: usize,
        n: usize,
    ) -> Result<storage::CsrStorage<Self::Data>>;

    /// Coordinate format sparse subtraction
    fn coo_sub_sparse(
        &self,
        lhs_data: &[Self::Data],
        lhs_row: &[usize],
        lhs_col: &[usize],
        rhs_data: &[Self::Data],
        rhs_row: &[usize],
        rhs_col: &[usize],
        m: usize,
        n: usize,
    ) -> Result<storage::CsrStorage<Self::Data>> {
        Err(crate::BackendError::UnsupportedOperation {
            operation: "coo_sub_sparse".to_string(),
            backend: self.device_name().to_string(),
        })
    }

    /// Quantization operation
    fn quantize(
        &self,
        input: &storage::DenseStorage<Self::Data>,
        levels: usize,
    ) -> Result<storage::DenseStorage<Self::Data>>
    where
        Self::Data: PartialOrd;

    /// Compute CLIP InfoNCE loss for contrastive learning
    fn clip_info_nce_loss(
        &self,
        image_embeddings: &storage::DenseStorage<Self::Data>,
        text_embeddings: &storage::DenseStorage<Self::Data>,
        temperature: f32,
    ) -> Result<Self::Data>;

    /// Compute CLIP attention mechanism
    /// Compute CLIP attention mechanism
    fn clip_attention(
        &self,
        queries: &storage::DenseStorage<Self::Data>,
        keys: &storage::DenseStorage<Self::Data>,
        values: &storage::DenseStorage<Self::Data>,
        num_heads: usize,
    ) -> Result<storage::DenseStorage<Self::Data>>;

    // ================== Comparison Operations ==================

    /// Element-wise equality (dense)
    fn eq_dense(
        &self,
        lhs: &storage::DenseStorage<Self::Data>,
        rhs: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>>;

    /// Element-wise equality (strided)
    fn eq_strided(
        &self,
        lhs: &storage::StridedStorage<Self::Data>,
        rhs: &storage::StridedStorage<Self::Data>,
    ) -> Result<storage::StridedStorage<Self::Data>>;

    /// Element-wise inequality (dense)
    fn ne_dense(
        &self,
        lhs: &storage::DenseStorage<Self::Data>,
        rhs: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>>;

    /// Element-wise inequality (strided)
    fn ne_strided(
        &self,
        lhs: &storage::StridedStorage<Self::Data>,
        rhs: &storage::StridedStorage<Self::Data>,
    ) -> Result<storage::StridedStorage<Self::Data>>;

    /// Element-wise greater than (dense)
    fn gt_dense(
        &self,
        lhs: &storage::DenseStorage<Self::Data>,
        rhs: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>>
    where
        Self::Data: PartialOrd;

    /// Element-wise greater than (strided)
    fn gt_strided(
        &self,
        lhs: &storage::StridedStorage<Self::Data>,
        rhs: &storage::StridedStorage<Self::Data>,
    ) -> Result<storage::StridedStorage<Self::Data>>
    where
        Self::Data: PartialOrd;

    /// Element-wise greater or equal (dense)
    fn ge_dense(
        &self,
        lhs: &storage::DenseStorage<Self::Data>,
        rhs: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>>
    where
        Self::Data: PartialOrd;

    /// Element-wise greater or equal (strided)
    fn ge_strided(
        &self,
        lhs: &storage::StridedStorage<Self::Data>,
        rhs: &storage::StridedStorage<Self::Data>,
    ) -> Result<storage::StridedStorage<Self::Data>>
    where
        Self::Data: PartialOrd;

    /// Element-wise less than (dense)
    fn lt_dense(
        &self,
        lhs: &storage::DenseStorage<Self::Data>,
        rhs: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>>
    where
        Self::Data: PartialOrd;

    /// Element-wise less than (strided)
    fn lt_strided(
        &self,
        lhs: &storage::StridedStorage<Self::Data>,
        rhs: &storage::StridedStorage<Self::Data>,
    ) -> Result<storage::StridedStorage<Self::Data>>
    where
        Self::Data: PartialOrd;

    /// Element-wise less or equal (dense)
    fn le_dense(
        &self,
        lhs: &storage::DenseStorage<Self::Data>,
        rhs: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>>
    where
        Self::Data: PartialOrd;

    /// Element-wise less or equal (strided)
    fn le_strided(
        &self,
        lhs: &storage::StridedStorage<Self::Data>,
        rhs: &storage::StridedStorage<Self::Data>,
    ) -> Result<storage::StridedStorage<Self::Data>>
    where
        Self::Data: PartialOrd;

    /// Element-wise addition for CSR storage
    fn add_csr(
        &self,
        lhs: &storage::CsrStorage<Self::Data>,
        rhs: &storage::CsrStorage<Self::Data>,
    ) -> Result<storage::CsrStorage<Self::Data>>;

    /// Element-wise multiplication for CSR storage
    fn mul_csr(
        &self,
        lhs: &storage::CsrStorage<Self::Data>,
        rhs: &storage::CsrStorage<Self::Data>,
    ) -> Result<storage::CsrStorage<Self::Data>>;

    /// Element-wise subtraction for CSR storage
    fn sub_csr(
        &self,
        lhs: &storage::CsrStorage<Self::Data>,
        rhs: &storage::CsrStorage<Self::Data>,
    ) -> Result<storage::CsrStorage<Self::Data>>;

    /// Apply ReLU to CSR storage (sparsity-preserving)
    fn relu_csr(
        &self,
        _input: &storage::CsrStorage<Self::Data>,
    ) -> Result<storage::CsrStorage<Self::Data>>
    where
        Self::Data: PartialOrd + Default {
        Err(crate::BackendError::UnsupportedOperation {
            operation: "relu_csr".to_string(),
            backend: self.device_name().to_string(),
        })
    }

    /// Apply tanh to CSR storage (sparsity-preserving)
    fn tanh_csr(
        &self,
        _input: &storage::CsrStorage<Self::Data>,
    ) -> Result<storage::CsrStorage<Self::Data>>
    where
        Self::Data: dtype::num_traits::Float {
        Err(crate::BackendError::UnsupportedOperation {
            operation: "tanh_csr".to_string(),
            backend: self.device_name().to_string(),
        })
    }

    /// Apply abs to CSR storage (sparsity-preserving)
    fn abs_csr(
        &self,
        _input: &storage::CsrStorage<Self::Data>,
    ) -> Result<storage::CsrStorage<Self::Data>>
    where
        Self::Data: dtype::num_traits::Signed {
        Err(crate::BackendError::UnsupportedOperation {
            operation: "abs_csr".to_string(),
            backend: self.device_name().to_string(),
        })
    }

    /// Matrix multiplication for CSR storage
    fn matmul_csr(
        &self,
        lhs: &storage::CsrStorage<Self::Data>,
        rhs: &storage::CsrStorage<Self::Data>,
    ) -> Result<storage::CsrStorage<Self::Data>> {
        Err(crate::BackendError::UnsupportedOperation {
            operation: "matmul_csr".to_string(),
            backend: self.device_name().to_string(),
        })
    }

    /// Add matrix multiplication with scaling for CSR storage
    fn addmm_csr(
        &self,
        _input: &storage::CsrStorage<Self::Data>,
        _mat1: &storage::CsrStorage<Self::Data>,
        _mat2: &storage::CsrStorage<Self::Data>,
        _beta: Self::Data,
        _alpha: Self::Data,
    ) -> Result<storage::CsrStorage<Self::Data>> {
         Err(crate::BackendError::UnsupportedOperation {
            operation: "addmm_csr".to_string(),
            backend: self.device_name().to_string(),
        })
    }

    /// Add matrix-vector multiplication with scaling for CSR storage
    fn addmv_csr(
        &self,
        _input: &storage::CsrStorage<Self::Data>,
        _mat: &storage::CsrStorage<Self::Data>,
        _vec: &storage::CsrStorage<Self::Data>,
        _beta: Self::Data,
        _alpha: Self::Data,
    ) -> Result<storage::CsrStorage<Self::Data>> {
         Err(crate::BackendError::UnsupportedOperation {
            operation: "addmv_csr".to_string(),
            backend: self.device_name().to_string(),
        })
    }

    /// Divide strided storages element-wise
    fn div_strided(
        &self,
        lhs: &storage::StridedStorage<Self::Data>,
        rhs: &storage::StridedStorage<Self::Data>,
    ) -> Result<storage::StridedStorage<Self::Data>>;

    // ================== Mixed-Storage Operations ==================

    /// Add dense and CSR storage
    fn add_dense_csr(
        &self,
        _lhs: &storage::DenseStorage<Self::Data>,
        _rhs: &storage::CsrStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>> {
        Err(crate::BackendError::UnsupportedOperation {
            operation: "add_dense_csr".to_string(),
            backend: self.device_name().to_string(),
        })
    }

    /// Add CSR and dense storage
    fn add_csr_dense(
        &self,
        _lhs: &storage::CsrStorage<Self::Data>,
        _rhs: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>> {
        Err(crate::BackendError::UnsupportedOperation {
            operation: "add_csr_dense".to_string(),
            backend: self.device_name().to_string(),
        })
    }

    /// Multiply dense and CSR storage
    fn mul_dense_csr(
        &self,
        _lhs: &storage::DenseStorage<Self::Data>,
        _rhs: &storage::CsrStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>> {
        Err(crate::BackendError::UnsupportedOperation {
            operation: "mul_dense_csr".to_string(),
            backend: self.device_name().to_string(),
        })
    }

    /// Multiply CSR and dense storage
    fn mul_csr_dense(
        &self,
        _lhs: &storage::CsrStorage<Self::Data>,
        _rhs: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>> {
        Err(crate::BackendError::UnsupportedOperation {
            operation: "mul_csr_dense".to_string(),
            backend: self.device_name().to_string(),
        })
    }

    /// Sparse-dense matrix multiplication (CSR @ Dense)
    fn matmul_csr_dense(
        &self,
        _lhs: &storage::CsrStorage<Self::Data>,
        _rhs: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>> {
        Err(crate::BackendError::UnsupportedOperation {
            operation: "matmul_csr_dense".to_string(),
            backend: self.device_name().to_string(),
        })
    }

    /// Dense-sparse matrix multiplication (Dense @ CSR)
    fn matmul_dense_csr(
        &self,
        _lhs: &storage::DenseStorage<Self::Data>,
        _rhs: &storage::CsrStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>> {
        Err(crate::BackendError::UnsupportedOperation {
            operation: "matmul_dense_csr".to_string(),
            backend: self.device_name().to_string(),
        })
    }

    // ================== Status Checks ==================

    fn isnan_dense(&self, _input: &storage::DenseStorage<Self::Data>) -> Result<storage::DenseStorage<Self::Data>>
    where Self::Data: dtype::num_traits::Float + dtype::num_traits::One + dtype::num_traits::Zero {
        Err(BackendError::UnsupportedOperation { operation: "isnan_dense".to_string(), backend: self.device_name().to_string() })
    }

    fn isinf_dense(&self, _input: &storage::DenseStorage<Self::Data>) -> Result<storage::DenseStorage<Self::Data>>
    where Self::Data: dtype::num_traits::Float + dtype::num_traits::One + dtype::num_traits::Zero {
        Err(BackendError::UnsupportedOperation { operation: "isinf_dense".to_string(), backend: self.device_name().to_string() })
    }

    fn isfinite_dense(&self, _input: &storage::DenseStorage<Self::Data>) -> Result<storage::DenseStorage<Self::Data>>
    where Self::Data: dtype::num_traits::Float + dtype::num_traits::One + dtype::num_traits::Zero {
        Err(BackendError::UnsupportedOperation { operation: "isfinite_dense".to_string(), backend: self.device_name().to_string() })
    }

    // ================== Logical Operations ==================

    fn logical_and_dense(
        &self,
        _lhs: &storage::DenseStorage<Self::Data>,
        _rhs: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>>
    where Self::Data: dtype::num_traits::One + dtype::num_traits::Zero {
        Err(BackendError::UnsupportedOperation { operation: "logical_and_dense".to_string(), backend: self.device_name().to_string() })
    }

    fn logical_or_dense(
        &self,
        _lhs: &storage::DenseStorage<Self::Data>,
        _rhs: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>>
    where Self::Data: dtype::num_traits::One + dtype::num_traits::Zero {
        Err(BackendError::UnsupportedOperation { operation: "logical_or_dense".to_string(), backend: self.device_name().to_string() })
    }

    fn logical_xor_dense(
        &self,
        _lhs: &storage::DenseStorage<Self::Data>,
        _rhs: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>>
    where Self::Data: dtype::num_traits::One + dtype::num_traits::Zero {
        Err(BackendError::UnsupportedOperation { operation: "logical_xor_dense".to_string(), backend: self.device_name().to_string() })
    }

    fn logical_not_dense(
        &self,
        _input: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>>
    where Self::Data: dtype::num_traits::One + dtype::num_traits::Zero {
        Err(BackendError::UnsupportedOperation { operation: "logical_not_dense".to_string(), backend: self.device_name().to_string() })
    }

    // ================== Math Parity ==================

    fn log1p_dense(&self, _input: &storage::DenseStorage<Self::Data>) -> Result<storage::DenseStorage<Self::Data>>
    where Self::Data: dtype::num_traits::Float {
        Err(BackendError::UnsupportedOperation { operation: "log1p_dense".to_string(), backend: self.device_name().to_string() })
    }

    fn expm1_dense(&self, _input: &storage::DenseStorage<Self::Data>) -> Result<storage::DenseStorage<Self::Data>>
    where Self::Data: dtype::num_traits::Float {
        Err(BackendError::UnsupportedOperation { operation: "expm1_dense".to_string(), backend: self.device_name().to_string() })
    }

    fn reciprocal_dense(&self, _input: &storage::DenseStorage<Self::Data>) -> Result<storage::DenseStorage<Self::Data>>
    where Self::Data: dtype::num_traits::Float {
        Err(BackendError::UnsupportedOperation { operation: "reciprocal_dense".to_string(), backend: self.device_name().to_string() })
    }

    fn atan2_dense(
        &self,
        _y: &storage::DenseStorage<Self::Data>,
        _x: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>>
    where Self::Data: dtype::num_traits::Float {
        Err(BackendError::UnsupportedOperation { operation: "atan2_dense".to_string(), backend: self.device_name().to_string() })
    }

    fn rsqrt_dense(&self, _input: &storage::DenseStorage<Self::Data>) -> Result<storage::DenseStorage<Self::Data>>
    where Self::Data: dtype::num_traits::Float {
        Err(BackendError::UnsupportedOperation { operation: "rsqrt_dense".to_string(), backend: self.device_name().to_string() })
    }

    fn erf_dense(&self, _input: &storage::DenseStorage<Self::Data>) -> Result<storage::DenseStorage<Self::Data>>
    where Self::Data: dtype::num_traits::Float {
        Err(BackendError::UnsupportedOperation { operation: "erf_dense".to_string(), backend: self.device_name().to_string() })
    }

    fn erfc_dense(&self, _input: &storage::DenseStorage<Self::Data>) -> Result<storage::DenseStorage<Self::Data>>
    where Self::Data: dtype::num_traits::Float {
        Err(BackendError::UnsupportedOperation { operation: "erfc_dense".to_string(), backend: self.device_name().to_string() })
    }

    fn erfinv_dense(&self, _input: &storage::DenseStorage<Self::Data>) -> Result<storage::DenseStorage<Self::Data>>
    where Self::Data: dtype::num_traits::Float {
        Err(BackendError::UnsupportedOperation { operation: "erfinv_dense".to_string(), backend: self.device_name().to_string() })
    }
}

/// Stub device for testing
#[derive(Debug, Clone, PartialEq)]
pub struct StubDevice;

impl DeviceInfo for StubDevice {
    fn name(&self) -> &str { "stub_device" }
    fn memory_total(&self) -> Option<usize> { Some(1024 * 1024 * 1024) }
    fn memory_available(&self) -> Option<usize> { Some(1024 * 1024 * 1024) }
    fn compute_capability(&self) -> Option<String> { None }
}
