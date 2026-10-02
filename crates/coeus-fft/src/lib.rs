//! Differentiable 1-D FFT for Coeus, routed through the Atlas-owned Apollo FFT library.
//!
//! Apollo owns the FFT itself (core slice/array transforms); this crate is Coeus's
//! **autograd for Apollo's FFT** — it wraps `apollo-fft`'s core `fft_1d_slice`
//! into tensor-level `fft_1d`/`ifft_1d` and the reverse-mode nodes [`Fft1DNode`],
//! [`Ifft1DNode`], and [`fft_energy`], building on the [`coeus_autograd`] engine
//! ([`Var`], [`BackwardNode`], [`GradBuffer`]). No dependency on `rustfft`.
//!
//! # Numerical contract
//! [`fft_1d`] computes the unnormalized forward DFT; [`ifft_1d`] the `1/N`-normalized
//! inverse, so `ifft_1d(fft_1d(x)) == x` up to floating-point rounding. FFT/IFFT form
//! an adjoint pair, giving the gradient rules encoded in the backward nodes.

// ── Coeus FFT ──
// Apollo-backed FFT autograd for Coeus tensors.
#![deny(missing_docs)]
#![forbid(unsafe_code)]

mod autograd;
mod fft;

pub use autograd::{fft_1d_var, fft_energy, ifft_1d_var, Fft1DNode, Ifft1DNode};
pub use fft::{fft_1d, ifft_1d, FftScalar};
