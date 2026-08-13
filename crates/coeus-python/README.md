# coeus-python

PyO3 bindings exposing the [Coeus](../../README.md) stack to Python.

The extension module is built as `pycoeus`. This crate is a binding surface:
it converts types, maps Rust errors onto Python exceptions, and releases the
GIL around compute. All numerics live in the Rust crates it wraps.

It is marked `publish = false` and distributed as a wheel, not on crates.io.

## What is exposed

`PyTensor` and complex tensors, neural-network layers (Linear, convolution,
normalization, pooling), optimizers and LR schedulers, losses, activations,
initializers, distributed helpers, state dicts, and no-grad contexts.

Mnemosyne is registered as the global allocator by default, via the
`mnemosyne-global` feature.

## Releases

Wheels for CPython 3.9-3.13 on Linux, Windows, and macOS build from GitHub
Releases tagged `coeus-python-v<version>` and publish to PyPI through OIDC
Trusted Publishing.

## License

Licensed under either of [Apache License, Version 2.0](../../LICENSE-APACHE) or
[MIT license](../../LICENSE-MIT) at your option.
