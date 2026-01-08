//! Core pooling layer structures and common functionality.
//!
//! This module contains the fundamental structures and traits shared
//! across all pooling layer implementations.

use std::fmt;

/// Common trait for pooling operations
pub trait PoolingOps {
    /// Get the pooling kernel size
    fn kernel_size(&self) -> &[usize];

    /// Get the pooling stride
    fn stride(&self) -> &[usize];

    /// Get the pooling padding
    fn padding(&self) -> &[usize];

    /// Check if the pooling is adaptive
    fn is_adaptive(&self) -> bool {
        false
    }
}

/// Pooling mode enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolingMode {
    /// Maximum pooling
    Max,
    /// Average pooling
    Average,
}

/// Pooling dimensionality
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolingDim {
    /// 1D pooling
    Dim1,
    /// 2D pooling
    Dim2,
}

/// Common pooling configuration
#[derive(Debug, Clone)]
pub struct PoolingConfig {
    /// Kernel size for each dimension
    pub kernel_size: Vec<usize>,
    /// Stride for each dimension (if None, defaults to kernel_size)
    pub stride: Option<Vec<usize>>,
    /// Padding for each dimension
    pub padding: Vec<usize>,
    /// Pooling mode
    pub mode: PoolingMode,
    /// Dimensionality
    pub dim: PoolingDim,
    /// Whether this is adaptive pooling
    pub adaptive: bool,
}

impl PoolingConfig {
    /// Create a new pooling configuration
    pub fn new(
        kernel_size: Vec<usize>,
        stride: Option<Vec<usize>>,
        padding: Vec<usize>,
        mode: PoolingMode,
        dim: PoolingDim,
        adaptive: bool,
    ) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
            mode,
            dim,
            adaptive,
        }
    }

    /// Get effective stride (defaults to kernel_size if not specified)
    pub fn effective_stride(&self) -> &[usize] {
        self.stride.as_ref().map_or(&self.kernel_size, |s| s)
    }

    /// Validate configuration for given dimensionality
    pub fn validate(&self) -> Result<(), String> {
        let expected_dims = match self.dim {
            PoolingDim::Dim1 => 1,
            PoolingDim::Dim2 => 2,
        };

        if self.kernel_size.len() != expected_dims {
            return Err(format!(
                "Expected {} kernel_size dimensions for {:?}, got {}",
                expected_dims,
                self.dim,
                self.kernel_size.len()
            ));
        }

        if let Some(ref stride) = self.stride {
            if stride.len() != expected_dims {
                return Err(format!(
                    "Expected {} stride dimensions for {:?}, got {}",
                    expected_dims,
                    self.dim,
                    stride.len()
                ));
            }
        }

        if self.padding.len() != expected_dims {
            return Err(format!(
                "Expected {} padding dimensions for {:?}, got {}",
                expected_dims,
                self.dim,
                self.padding.len()
            ));
        }

        Ok(())
    }
}

impl fmt::Display for PoolingConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:?}{}Pool(kernel_size={:?}, stride={:?}, padding={:?})",
            self.mode,
            match self.dim {
                PoolingDim::Dim1 => "1d",
                PoolingDim::Dim2 => "2d",
            },
            self.kernel_size,
            self.effective_stride(),
            self.padding
        )
    }
}
