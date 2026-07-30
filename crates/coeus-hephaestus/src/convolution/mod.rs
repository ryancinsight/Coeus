//! Generic Coeus-to-Hephaestus convolution dispatch.

mod dispatch;
mod implementation;
mod provider;

pub use dispatch::{
    Backward as ConvolutionBackwardDispatch, Forward as ConvolutionForwardDispatch,
    regular_backward, regular_forward, transposed_backward, transposed_forward,
};
pub use provider::{ConvolutionBackend, ConvolutionProvider};
