# 0067. `Linear::new` must break unit symmetry

Status: Proposed

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

This changes the numbers every fresh `Linear` produces, and 50 call sites
construct one — 29 in this workspace, 21 in kwavers. Tests that assert on the
output of an untrained layer will change, and each has to be re-derived rather
than re-baselined: an assertion that happened to hold for all-ones weights was
asserting on the defect.

Reproducibility gets better rather than worse. Today a network is reproducible
because it is degenerate; afterwards it is reproducible because the seed is
carried, which is the property kwavers' `collocation_seed` already establishes
for the other half of a PINN run.

The change is `[major]` and crosses a repository boundary, so it lands upstream
here first and kwavers follows per the co-evolution protocol.

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
