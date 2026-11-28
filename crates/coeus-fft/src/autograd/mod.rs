//! Reverse-mode autograd nodes for the Apollo-backed FFTs.
//!
//! One leaf per rank plus the energy node, mirroring the `fft`/`fft2d`/
//! `fft3d` kernel split: `nodes1d`, `nodes2d`, `nodes3d`, `nodesrfftn`,
//! `energy`.

mod energy;
mod nodes1d;
mod nodes2d;
mod nodes3d;
mod nodesrfftn;

pub use energy::fft_energy;
pub use nodes1d::{fft_1d_var, ifft_1d_var, Fft1DNode, Ifft1DNode};
pub use nodes2d::{fft_2d_var, ifft_2d_var, Fft2DNode, Ifft2DNode};
pub use nodes3d::{fft_3d_var, ifft_3d_var, Fft3DNode, Ifft3DNode};
pub use nodesrfftn::{
    irfft_2d_var, irfft_3d_var, rfft_2d_var, rfft_3d_var, Irfft2DNode, Irfft3DNode, Rfft2DNode,
    Rfft3DNode,
};
