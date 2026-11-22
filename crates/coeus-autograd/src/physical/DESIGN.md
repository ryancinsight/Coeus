# Physical reverse-mode derivatives

## Contract and proof

Let `X = s_x x + b_x` and `Y = s_y y + b_y` convert presented readings into
canonical SI coordinates, where both scales are fixed and positive. Let
`Y = F(X)` with canonical Jacobian `J_SI`. Differentiation gives

```text
dY = J_SI dX
dy/dx = (s_x / s_y) J_SI
dim(J_SI) = dim(dY) / dim(dX)
```

The chain rule yields the second identity using `dX = s_x dx` and
`dY = s_y dy`. Origins disappear from the differentials. A nonlinear law
can still depend on the point at which its derivative is evaluated.
Absolute-temperature differentials use interval units. The canonical
kelvin identity has derivative 1 and the K per °F coordinate slope is 5/9.

Aequitas subtracts exponent vectors and produces a base-semantic
quotient. Energy over length produces force. A dimensionless loss over
length produces inverse length. The gradient retains its input and
output quantity kinds for choosing coordinate units. A derived SI
quotient carries its exponents and does not infer a named domain kind
from equal exponents. The forward subtraction of two temperature points produces a typed interval.

In a composition, multiplying local derivatives cancels the intermediate
units. All reverse paths to the same input therefore contribute the same
quotient for one scalar objective. Its adjoint seed is 1. In raw Coeus
a vector output with seed `w` returns `Jᵀw`. A future physical VJP must
type the seed as `objective/output`, so the result has units
`objective/input`. A full Jacobian retains the output axes.

## Adapter

1. Accept Aequitas quantities holding canonical tensors or normalize
   homogeneous raw tensors using typed linear and affine coordinates.
   Reject rank-zero tensors at ingress. Shape `[1]` represents a scalar
   on the qualified CPU backend.
2. Start a fresh graph and call a closure with a tracked input and a scope
   for typed constants. Invariant lifetimes and private storage prevent
   graph escape and arbitrary raw variable attachment. Require recording
   at entry and during typed operations.
3. Reuse Aequitas's additive laws for addition and subtraction. Multiply
   and divide canonical values using existing dimensional composition.
   Sum requires a self-additive quantity. Log takes an ordinary
   dimensionless ratio such as `p/p_ref`. Numeric domains such as
   positive log arguments belong to the model. Backend errors propagate.
4. Run Coeus backward once for a one-element objective and return its
   value and input-shaped gradient. Independent objectives yield typed
   zero gradients. A new call uses independent buffers even for shared
   input data. A failed call discards its tape.

Eunomia materializes conversion coefficients in the selected scalar type.
Return a range error when a required scale or scale ratio is nonpositive
or non-finite in that type, or when an offset becomes non-finite. Data
arithmetic follows the backend's numeric semantics. The adapter does not
read tensor data back to the host.

Initial qualification is Rust CPU binary32 and binary64. Multi-input
reverse passes, vector-output VJPs, arbitrary operation registration,
other backends and Python need separate acceptance.

## Acceptance

| Case | Required value and type |
| --- | --- |
| Work at 1 N and 2 mm | 0.002 J; derivative 1 N; 0.001 J/mm and 1 mJ/mm |
| Vector work | Gradient equals each force component with the input shape |
| Pressure ramp | 108 kPa at 2 s; derivative 4000 Pa/s |
| Isothermal compression | 100 kPa at 2 L for PV = 200 J; −50 MPa/m³; normalized derivative −1 |
| Affine and interval | 68 °F = 293.15 K; 36 °F interval = 20 K; slopes 5/9 K/°F and 9/5 °F/K |
| Sensible heat | 10 kJ for 68 °F minus 32 °F at 500 J/K; derivative 500 J/K or 2500/9 J/°F |
| Dimensionless log pressure ratio | At 108 kPa, derivative 1/108000 Pa⁻¹ |
| Invalid operations | Compile errors for unlike addition, adding absolute points, dimensioned logarithm, affine interval or graph escape. Typed errors for rank-zero inputs, non-scalar objectives, incompatible broadcasting and disabled recording. |
| Independent objectives | Energy and dimensionless objectives on the same data produce isolated typed gradients; an independent scalar gives shaped zeros |

Check physical values against analytical derivatives and finite differences
with declared scales. Coeus's default step depends on `max(abs(x), 1)`
in the tested coordinates. For `P(V)=C/V` the exact secant at endpoints
`V₊,V₋` is `−C/(V₊ V₋)`. Checking `P/P₀ = 1/(V/V₀)` gives a better scaled
numerical problem. Recover the SI slope using `P₀/V₀` and retain the
raw-coordinate reproduction.

Verify each candidate with the workspace feature enabled and disabled.
Run Rustdoc, formatting, strict Clippy, locked resolution and an API
comparison against the base revision. Hosted compatibility and clean
published-consumer resolution belong to release qualification.
