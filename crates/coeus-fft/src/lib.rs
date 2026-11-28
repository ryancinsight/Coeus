//! Differentiable 1-D, 2-D and 3-D FFT for Coeus, routed through the Atlas-owned Apollo FFT library.
//!
//! Apollo owns the FFT itself (core slice/array transforms); this crate is Coeus's
//! **autograd for Apollo's FFT** — it wraps `apollo-fft`'s core transforms
//! into tensor-level `fft_1d`/`ifft_1d`, `fft_2d`/`ifft_2d`,
//! `fft_3d`/`ifft_3d`, `rfft_1d`/`irfft_1d` and `rfft_2d`/`irfft_2d`/
//! `rfft_3d`/`irfft_3d` plus the reverse-mode
//! nodes, building on the [`coeus_autograd`] engine ([`coeus_autograd::Var`],
//! [`coeus_autograd::BackwardNode`], [`coeus_autograd::GradBuffer]). No dependency on `rustfft`.
//!
//! # Numerical contract
//! [`fft_1d`] computes the unnormalized forward DFT; [`ifft_1d`] the `1/N`-normalized
//! inverse, so `ifft_1d(fft_1d(x)) == x` up to floating-point rounding.
//! [`fft_2d`] computes the unnormalized 2-D DFT; [`ifft_2d`] the `1/(nx·ny)`-normalized
//! inverse. [`fft_3d`] computes the unnormalized 3-D DFT; [`ifft_3d`] the
//! `1/(nx·ny·nz)`-normalized inverse. [`rfft_1d`] returns the `n/2 + 1`
//! independent bins of a real spectrum; [`irfft_1d`] inverts them.
//! [`rfft_2d`]/[`rfft_3d`] return the `[nx, ny/2 + 1]`/`[nx, ny, nz/2 + 1]`
//! independent bins; [`irfft_2d`]/[`irfft_3d`] invert them.
//! FFT/IFFT form an adjoint pair in every rank, giving the gradient
//! rules encoded in the backward nodes.

// ── Coeus FFT ──
// Apollo-backed FFT autograd for Coeus tensors.
#![deny(missing_docs)]
#![forbid(unsafe_code)]

mod autograd;
mod fft;
mod fft2d;
mod fft3d;
mod ffthalf;
mod rfftn;

pub use autograd::{
    fft_1d_var, fft_2d_var, fft_3d_var, fft_energy, ifft_1d_var, ifft_2d_var, ifft_3d_var,
    irfft_2d_var, irfft_3d_var, rfft_2d_var, rfft_3d_var, Fft1DNode, Fft2DNode, Fft3DNode,
    Ifft1DNode, Ifft2DNode, Ifft3DNode, Irfft2DNode, Irfft3DNode, Rfft2DNode, Rfft3DNode,
};
pub use fft::{fft_1d, ifft_1d, FftScalar};
pub use fft2d::{fft_2d, ifft_2d, Fft2DScalar};
pub use fft3d::{fft_3d, ifft_3d, Fft3DScalar};
pub use ffthalf::{
    irfft_1d, irfft_1d_var, rfft_1d, rfft_1d_var, FftHalfScalar, Irfft1DNode, Rfft1DNode,
};
pub use rfftn::{irfft_2d, irfft_3d, rfft_2d, rfft_3d, FftHalfNdScalar};
