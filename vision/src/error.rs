#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid input: {0}")]
    InvalidInput(String),

    #[error("unsupported image color type: {0}")]
    UnsupportedColorType(String),

    #[error(transparent)]
    Image(#[from] image::ImageError),

    #[error(transparent)]
    Tensor(#[from] tensor::TensorError),

    #[error(transparent)]
    Transform(#[from] utils::TransformError),
}

pub type Result<T> = std::result::Result<T, Error>;
