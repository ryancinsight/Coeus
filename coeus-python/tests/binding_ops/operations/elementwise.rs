//! Element-wise tensor operation binding contracts.

use super::support::run_script;

#[test]
fn test_abs_sqrt_neg_pow() {
    run_script(
        r#"
import pycoeus

x = pycoeus.Tensor([-4.0, 9.0, -1.0, 4.0], [2, 2])

a = pycoeus.abs(x)
assert a.data == [4.0, 9.0, 1.0, 4.0]

s = pycoeus.sqrt(pycoeus.abs(x))
assert abs(s.data[0] - 2.0) < 1e-9
assert abs(s.data[1] - 3.0) < 1e-9

n = pycoeus.neg(x)
assert n.data == [4.0, -9.0, 1.0, -4.0]

p = pycoeus.pow(pycoeus.abs(x), 2.0)
assert abs(p.data[0] - 16.0) < 1e-6
assert abs(p.data[1] - 81.0) < 1e-6
"#,
    );
}

#[test]
fn test_recip_sign_floor_ceil_round_trunc() {
    run_script(
        r#"
import pycoeus

x = pycoeus.Tensor([2.0, -3.0, 0.5, -0.5, 0.0], [5])

# free function form
r = pycoeus.recip(x)
assert abs(r.data[0] - 0.5) < 1e-9
assert abs(r.data[1] + 1.0/3.0) < 1e-9
assert abs(r.data[2] - 2.0) < 1e-9

s = pycoeus.sign(x)
assert s.data == [1.0, -1.0, 1.0, -1.0, 0.0]

f = pycoeus.floor(x)
assert f.data == [2.0, -3.0, 0.0, -1.0, 0.0]

c = pycoeus.ceil(x)
assert c.data == [2.0, -3.0, 1.0, 0.0, 0.0]

ro = pycoeus.round(x)
assert ro.data[0] == 2.0
assert ro.data[1] == -3.0

t = pycoeus.trunc(x)
assert t.data == [2.0, -3.0, 0.0, 0.0, 0.0]

# tensor method form
x2 = pycoeus.Tensor([1.5, -2.3, 3.8], [3])
assert abs(x2.recip().data[0] - 1.0/1.5) < 1e-9
assert x2.sign().data == [1.0, -1.0, 1.0]
assert x2.floor().data == [1.0, -3.0, 3.0]
assert x2.ceil().data == [2.0, -2.0, 4.0]
assert x2.round().data == [2.0, -2.0, 4.0]
assert x2.trunc().data == [1.0, -2.0, 3.0]
"#,
    );
}

#[test]
fn test_clamp() {
    run_script(
        r#"
import pycoeus

x = pycoeus.Tensor([-3.0, -0.5, 0.5, 1.5, 2.5, 4.0], [2, 3])
c = pycoeus.clamp(x, -1.0, 2.0)
assert c.data == [-1.0, -0.5, 0.5, 1.5, 2.0, 2.0]
"#,
    );
}

#[test]
fn test_comparisons_and_where() {
    run_script(
        r#"
import pycoeus

a = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0], [4])
b = pycoeus.Tensor([2.0, 2.0, 1.0, 5.0], [4])

eq = pycoeus.eq(a, b)
assert eq.data == [0.0, 1.0, 0.0, 0.0], f"eq failed: {eq.data}"

lt = pycoeus.lt(a, b)
assert lt.data == [1.0, 0.0, 0.0, 1.0], f"lt failed: {lt.data}"

gt = pycoeus.gt(a, b)
assert gt.data == [0.0, 0.0, 1.0, 0.0], f"gt failed: {gt.data}"

ge = pycoeus.ge(a, b)
assert ge.data == [0.0, 1.0, 1.0, 0.0], f"ge failed: {ge.data}"

le = pycoeus.le(a, b)
assert le.data == [1.0, 1.0, 0.0, 1.0], f"le failed: {le.data}"

ne = pycoeus.ne(a, b)
assert ne.data == [1.0, 0.0, 1.0, 1.0], f"ne failed: {ne.data}"

# Tensor method forms
eq2 = a.eq(b)
assert eq2.data == [0.0, 1.0, 0.0, 0.0]

lt2 = a.lt(b)
assert lt2.data == [1.0, 0.0, 0.0, 1.0]

ge2 = a.ge(b)
assert ge2.data == [0.0, 1.0, 1.0, 0.0]

le2 = a.le(b)
assert le2.data == [1.0, 1.0, 0.0, 1.0]

ne2 = a.ne(b)
assert ne2.data == [1.0, 0.0, 1.0, 1.0]

# where_fn
cond  = pycoeus.Tensor([1.0, 0.0, 1.0, 0.0], [4])
on_t  = pycoeus.Tensor([10.0, 20.0, 30.0, 40.0], [4])
on_f  = pycoeus.Tensor([-1.0, -2.0, -3.0, -4.0], [4])
result = pycoeus.where_fn(cond, on_t, on_f)
assert result.data == [10.0, -2.0, 30.0, -4.0], f"where_fn failed: {result.data}"
"#,
    );
}

#[test]
fn test_softmax_cumsum_flip() {
    run_script(
        r#"
import pycoeus
import math

# softmax
x = pycoeus.Tensor([1.0, 2.0, 3.0], [3])
sm = pycoeus.softmax(x, 0)
total = sum(sm.data)
assert abs(total - 1.0) < 1e-6, f"softmax does not sum to 1: {total}"
for v in sm.data:
    assert 0.0 < v < 1.0, f"softmax value out of (0,1): {v}"

# cumsum along axis 0
cx = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0], [4])
cs = pycoeus.cumsum(cx, 0)
assert cs.data == [1.0, 3.0, 6.0, 10.0], f"cumsum wrong: {cs.data}"

# flip
fx = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
fl = pycoeus.flip(fx, 1)
assert fl.data == [3.0, 2.0, 1.0, 6.0, 5.0, 4.0], f"flip wrong: {fl.data}"
"#,
    );
}

#[test]
fn test_py_tensor_scalar_arithmetic() {
    // Every arithmetic Python operator must accept a Python `float`
    // and route through `coeus_autograd::scalar_*` (the dispatch
    // path `PyTensor::binop_dispatch`). Expectations match PyTorch /
    // JAX / MLX scalar-arithmetic semantics: scalar broadcast on
    // forward ops, negation+add composition for mirrored ops.
    run_script(
        r#"
import pycoeus

t = pycoeus.Tensor([3.0, -1.0, 5.0, 2.0, -4.0, 0.5], [2, 3])

# --- forward ops: Tensor op float ---
add_r = (t + 1.0).data
sub_r = (t - 1.0).data
mul_r = (t * 2.0).data
div_r = (t / 2.0).data
assert add_r == [4.0, 0.0, 6.0, 3.0, -3.0, 1.5], add_r
assert sub_r == [2.0, -2.0, 4.0, 1.0, -5.0, -0.5], sub_r
assert mul_r == [6.0, -2.0, 10.0, 4.0, -8.0, 1.0], mul_r
assert div_r == [1.5, -0.5, 2.5, 1.0, -2.0, 0.25], div_r

# --- mirrored ops: float op Tensor ---
assert (1.0 + t).data == [4.0, 0.0, 6.0, 3.0, -3.0, 1.5]
assert (1.0 - t).data == [-2.0, 2.0, -4.0, -1.0, 5.0, 0.5]
assert (2.0 * t).data == [6.0, -2.0, 10.0, 4.0, -8.0, 1.0]
assert (2.0 / t).data == [
    2.0 / 3.0, -2.0, 0.4, 1.0, -0.5, 4.0,
]

# --- __neg__ and __abs__ ---
assert (-t).data == [-3.0, 1.0, -5.0, -2.0, 4.0, -0.5]
assert abs(t).data == [3.0, 1.0, 5.0, 2.0, 4.0, 0.5]

# --- tensor-tensor arithmetic still works under the new dispatch path ---
u = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
assert (t + u).data == [4.0, 1.0, 8.0, 6.0, 1.0, 6.5]
assert (t - u).data == [2.0, -3.0, 2.0, -2.0, -9.0, -5.5]
"#,
    );
}
