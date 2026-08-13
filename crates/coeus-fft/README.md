# coeus-fft

FFT integration for [Coeus](../../README.md) through
[Apollo](https://github.com/ryancinsight/apollo).

A thin integration layer. Coeus owns no FFT math: every transform is computed
by `apollo-fft`, which is a hard, non-optional dependency of this crate.

## What is here

- `fft_1d` / `ifft_1d` — tensor-level wrappers over Apollo's
  `fft_1d_slice_typed`.
- `fft_1d_var` / `ifft_1d_var` — the differentiable forms, with the
  `Fft1DNode` and `Ifft1DNode` autograd nodes.
- `fft_energy`.

## Documentation

API docs: <https://docs.rs/coeus-fft>

## License

Licensed under either of [Apache License, Version 2.0](../../LICENSE-APACHE) or
[MIT license](../../LICENSE-MIT) at your option.
