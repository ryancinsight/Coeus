# 0067. `Linear::new` must break unit symmetry

Status: Accepted

Date: 2026-08-26

## Context

`Linear::new` sets every weight to `1.0` and every bias to `0.0`:

```rust
let w_tensor = Tensor::ones_on(w_shape, &backend);
```

A layer built this way has identical rows, so every unit in it computes the
same value from the same input, receives the same gradient, and applies the
same update. They are identical at step zero and stay identical for the life of
the network. A `Linear(3, 4)` has the expressive capacity of a `Linear(3, 1)`,
and no amount of training changes that.

Measured, not inferred. One forward and backward through
`Linear::<f32, MoiraiBackend>::new(3, 4, true)` on a `[2, 3]` input:

```
initial weights: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
weight gradient: [2.0, -0.75, 1.25,  2.0, -0.75, 1.25,
                  2.0, -0.75, 1.25,  2.0, -0.75, 1.25]
```

Four units, one gradient repeated four times.

This is not a gap in the crate's capability. `coeus_nn::init` already provides
`xavier_uniform`, `xavier_normal`, `kaiming_uniform`, `kaiming_normal`,
`uniform` and `normal`, each with a `_with_seed` variant, and
`attention/mha.rs` calls `kaiming_uniform` for exactly this reason. `Linear`
is the outlier.

Every network in this workspace and in kwavers' PINNs is built from `Linear`.
kwavers' `PINN3DNetwork::new` constructs its input layer, every hidden layer
and its output layer through `Linear::new` and does not re-initialise, so a
`hidden_layers: vec![64, 64]` configuration trains 2 effective units, not 128.
`KW-PINN-3D-NO-CONVERGENCE` diagnosed that solver's learning-rate schedule; this
is a second, independent reason those networks did not fit what they were asked
to fit, and it was underneath the first.

## Decision

`Linear::new` initialises weights by Kaiming uniform over `fan_in` through
`init::kaiming_uniform`, and biases stay zero. `Linear::with_seed` takes the
seed for callers who want to choose the draw.

`new` stays deterministic: `init::kaiming_uniform` draws from a fixed seed, so
a layer built the same way twice is the same layer. Reproducibility is not
traded away for symmetry-breaking; the two are independent, and today's
reproducibility comes from degeneracy rather than from anything worth keeping.

The constructors become fallible. `kaiming_uniform` rejects a zero `fan_in`
and the backend's draw can fail, and a library does not panic on
input-dependent paths. The draw also needs more of `T` and `B` than holding a
layer does, so those bounds go on the constructors rather than the type -- a
`Linear<T, B>` stays nameable for any scalar, and only building one requires a
sampleable scalar and a backend that can sample it.

This is the shape `MultiHeadAttention::new` already has in this crate, for
exactly this reason.

## Consequences

Measured by applying the change, not estimated:

| Where | Sites |
|---|---|
| `coeus-nn` lib (RNN cells, SwiGLU, transformer FFN) | 10 |
| `coeus-nn` tests and benches | 68 |
| `coeus-python` bindings | 11 |
| kwavers, 7 files | 21 |

The composite layers that build a `Linear` internally -- `GRUCell`,
`LSTMCell`, `RNNCell`, `SwiGlu`, `FeedForward` -- become fallible too, and
their wrappers after them. Their bounds sit on the constructors rather than the
impl blocks: `step`, `forward_seq` and the rest draw nothing and keep the
bounds they had.

Three of those constructors documented an initialisation they never performed.
`GRUCell::new`, `LSTMCell::new` and `RNNCell::new` each said "Xavier-initialized
weights" directly above a call to the all-ones constructor. The documentation
described the intent; the code had lost it. They now say Kaiming and do it.

Three tests out of 1130 failed, all of the predicted kind, and each was
re-derived rather than re-baselined:

- `swiglu_forward_matches_analytic` opened with "ones-weight projection => each
  output column equals the input row sum S". Its oracle came from the defect.
  It now sets both projections to all-ones explicitly, which keeps the closed
  form and makes the test independent of how a layer is initialised.
- `rnn_parity::{sequential,moirai}_rnn_match_reference` expected `tanh(3)` and
  `relu(3)`, derived from all-ones weights the same way. Same treatment.

In the Python bindings, three constructions sat inside `py.allow_threads`,
where a `PyErr` cannot be raised without the GIL. They are hoisted out. Each
overwrites every weight it just built with the tensors the wrapper already
holds, so the draw was discarded work on a compute path either way.

`tests/nn/linear_initialisation.rs` is the regression: units start distinct,
compute distinct outputs, take distinct gradients under a weight-dependent
loss, `with_seed` reproduces, and a zero `in_features` is rejected. Four of the
six fail against the old initialisation, which was checked by restoring it.

One correction worth recording, because the first version of that test asserted
something false. Under `L = sum(y)` the weight gradient is
`dL/dW[j,i] = sum_n x[n,i]` for every unit `j` -- identical across units
whatever `W` holds. That is correct arithmetic rather than a symmetry, and a
test asserting otherwise would fail against a correct implementation. The test
uses `L = sum(y * y)`, whose gradient `2 * y[n,j] * x[n,i]` does depend on the
unit's own weights.

Reproducibility improves rather than degrading. A network was reproducible
before because it was degenerate; it is reproducible now because
`kaiming_uniform` draws from a fixed seed and `with_seed` lets a caller choose
one -- the property kwavers' `collocation_seed` already establishes for the
other half of a PINN run.

kwavers follows per the co-evolution protocol: 21 sites across 7 files, after
which `PINN3DNetwork` stops training two effective units per layer.

## Alternatives rejected

**Leave `new` alone and add `new_kaiming`.** Two constructors where one is
correct and the default is not; every caller who does not know about the
problem keeps hitting it. The naming prohibition applies -- the initialisation
strategy is a variation dimension, not a name suffix -- and a replacement takes
the original's name.

**Default to `xavier_uniform`.** Defensible, and better for `tanh` networks
specifically, which is what the PINNs use. Kaiming is chosen because it is the
more common default and correct for the ReLU-family activations most consumers
reach for first; a caller who wants Xavier has `init::xavier_uniform` and one
line. Worth revisiting with measurements from the PINN suite, which is
`tanh`-based and may prefer the other.
