use thiserror::Error;

#[derive(Error, Debug)]
pub enum NNError {
    #[error("Invalid input: {message}")]
    InvalidInput { message: String },
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Network error: {message}")]
    Network { message: String },
    
    #[error("Computation error: {message}")]
    Computation { message: String },
    
    #[error("Unknown error: {message}")]
    Unknown { message: String },
}

pub type Result<T> = std::result::Result<T, NNError>;
