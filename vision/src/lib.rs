mod error;
pub mod io;
pub mod transforms;

pub use error::{Error, Result};

use backend::CpuBackend;
use dtype::float::Float32;
use storage::DenseStorage;
use tensor::Tensor;

pub type ImageTensor = Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>;
