//! Generic Coeus-to-Hephaestus convolution dispatch.

mod dispatch;
mod implementation;
mod provider;

pub use dispatch::{
    regular_backward, regular_forward, transposed_backward, transposed_forward,
    Backward as ConvolutionBackwardDispatch, Forward as ConvolutionForwardDispatch,
};
pub use provider::{ConvolutionBackend, ConvolutionProvider};
