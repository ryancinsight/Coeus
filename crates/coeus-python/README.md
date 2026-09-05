# coeus-python

Tensors, automatic differentiation, and neural-network layers for Python,
backed by the [Coeus](https://github.com/ryancinsight/Coeus) Rust stack.

The API follows PyTorch's shape closely enough to read without a translation
guide — a `Tensor` that tracks gradients, `Linear` and convolution layers with
`forward`, optimizers with `step` and `zero_grad` — while the computation runs
in Rust and releases the GIL.

## Install

```sh
pip install coeus-python
```

Wheels are published for CPython 3.9 through 3.13 on Linux, Windows, and
macOS. There are no runtime dependencies.

## Use

```python
import pycoeus

# A tensor is data plus a shape; autograd is opt-in per tensor.
x = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0], shape=[2, 2], requires_grad=True)

y = (x * 2.0 + 1.0).sum()
y.backward()
print(x.grad)          # [2.0, 2.0, 2.0, 2.0]

# Layers and optimizers compose the way you would expect.
layer = pycoeus.Linear(2, 1)
opt = pycoeus.SGD(layer.parameters(), lr=0.01)

out = layer.forward(x)
loss = out.sum()
loss.backward()
opt.step()
opt.zero_grad()
```

The import name is `pycoeus`; the distribution is `coeus-python`.

## What is exposed

- **Tensors** — real and complex, N-dimensional, with the five arithmetic
  operators accepting either a tensor (elementwise) or a Python float (scalar
  broadcast), plus reductions, indexing, and layout operations.
- **Autograd** — `requires_grad`, `backward`, `grad`, and no-grad contexts.
- **Layers** — linear, convolution, normalization, pooling, attention,
  dropout, bilinear.
- **Training** — optimizers and learning-rate schedulers, losses,
  activations, initializers, state dicts.
- **Distributed** — collective helpers.

The wheel ships a hand-maintained type stub (`pycoeus.pyi`) covering the whole
surface. It is not yet a PEP 561 inline-stub distribution — the extension is a
single top-level module, which has no package directory for the `py.typed`
marker to live in — so editors that read a root-level stub will use it and
`mypy` will not.

## Why the numerics are not here

This package is a binding surface: it converts types, maps Rust errors onto
Python exceptions, and releases the GIL around compute. Every kernel lives in
the Rust crates it wraps. One implementation means one place to verify, and
the suite differences the bindings against PyTorch, JAX, and MLX on identical
inputs rather than against themselves.

## Links

- [Source and issues](https://github.com/ryancinsight/Coeus)

## Licence

MIT or Apache-2.0, at your option.
