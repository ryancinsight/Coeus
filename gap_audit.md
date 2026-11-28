# Coeus Gap Audit

Risks nobody is actively fixing. A risk with a planned fix is a backlog item
instead (see `backlog.md`); a closed risk's residue lives in its ADR or PR,
not here.

## CUDA context creation failure remains unclassified

Consumer run `87532469-ce04-4b9a-8225-f53e7ea000a2` fails in `cuCtxCreate_v2`
with status 999 after releasing a successful temporary device. The isolated
reproduction passes, so no fix is confirmed; driver health inspection found
no matching Windows driver event.
Re-open trigger: a captured driver fault or a reproducible failing lifecycle.

## Slop pattern: per-element coordinate buffer in flat-index decode loops

A kernel walking a flat output index and decoding it into coordinates through
a `vec![0usize; ndim]` allocated *inside* the loop is waste: the buffer never
outlives the iteration that fills it. Fix: fuse the decode into the
accumulation instead of allocating per element.
Check (no lint covers this yet): `rg --multiline 'for \w+ in 0\.\.[^\n]*\{[^}]{0,200}?vec!\[0usize;'`.
Re-open trigger: the pattern recurs outside the five sites already fixed under the closed COEUS-OPS-INDEX-DECODE-ALLOC-001 — promote to a Clippy lint if so.

## Slop pattern: stale local `*.pyd` shadows the installed extension

pytest prepends the test directory to `sys.path`, so a leftover
`crates/coeus-python/tests/pycoeus*.pyd` build artifact silently overrides the
freshly `maturin develop`-installed module, pinning an out-of-date binary and
producing spurious `AttributeError`s for newly-added bindings.
Mitigation: keep built extensions out of `tests/`; the canonical module is the
site-packages install.
Re-open trigger: the artifact reappears (add a pre-test cleanup or a `.gitignore`/CI check if it recurs).

## G-049 residual: `gammaln` backward — resolved via provider `digamma`

`d/dx lgamma(x) = digamma(x)`. Eunomia now exposes `FloatElement::digamma`,
so the backward is no longer blocked: `coeus-ops::digamma` evaluates the
scalar provider function on the host (the provider opcode layer has no
`Digamma` kernel, same placement as `atan2`), and the tracked
`coeus_autograd::lgamma` composes `grad_out * digamma(x)`.
`lgamma_forward` and its Python binding entry point are unchanged and stay
forward-only; new Rust code uses the differentiable `lgamma`. Finite-difference
gradcheck coverage lives in `gradcheck/new_ops.rs`
(`lgamma_backward_matches_finite_differences`, f32+f64).

## K-001 residual: `bessel_k1` absent from the provider surface

`coeus-ops::bessel_k0` is forward-only. The derivative is
`d/dx K₀(x) = −K₁(x)`, and Eunomia exposes `bessel_j0`/`bessel_j1`/
`bessel_k0` but no `bessel_k1`, so a tracked node would need an invented
kernel. Re-open trigger: Eunomia gains `bessel_k1`, then add the tracked
wrapper with the `-K₁` backward and a finite-difference gradcheck.
Forward value coverage: `bessel_k0_matches_reference_and_rejects_nonpositive`
(1e-6 against the documented A&S error bound).

## Seq2Seq all-ones gradient has no convergent finite-difference oracle

`test_transformer_seq2seq` perturbs one `src` element of its all-ones fixture
and recomputes `sum(output)`. Central differences do not converge as the step
shrinks: numeric -1.64e4 at h=1e-3, -2.29e5 at h=1e-4, -3.08e7 at h=1e-6 and
-3.50e9 at h=1e-8, while `src.grad()[0]` is -1.166e11. The gradient cannot be
pinned to an independent oracle at this fixture, so the test asserts only its
shape and finiteness; pinning the observed value would certify an unverified
number.
Re-open trigger: a well-conditioned fixture with a convergent finite-difference
oracle, or a diagnosed cause for the singularity in the seq2seq forward/backward.
