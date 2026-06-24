// ── binding_tests_ops.rs ──
//
// Tests for functional ops exposed at the module level: stack, matmul, zeros,
// ones, full, arange, linspace, abs, sqrt, neg, clamp, max_axis, min_axis,
// sum, mean, reshape, permute, t, pow, log_sum_exp.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::ffi::CString;

fn run_script(script: &str) {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let pycoeus_module = pyo3::types::PyModule::new(py, "pycoeus").unwrap();
        pycoeus::pycoeus(&pycoeus_module).unwrap();
        let sys = py.import("sys").unwrap();
        let modules_any = sys.getattr("modules").unwrap();
        let modules = modules_any.downcast::<PyDict>().unwrap();
        modules.set_item("pycoeus", &pycoeus_module).unwrap();
        let script = CString::new(script).expect("test script must not contain interior NUL");
        py.run(script.as_c_str(), None, None)
            .unwrap_or_else(|e| panic!("Python script failed:\n{e}"));
    });
}

#[test]
fn test_constructors() {
    run_script(
        r#"
import pycoeus

z = pycoeus.zeros([2, 3])
assert z.shape == [2, 3]
assert all(v == 0.0 for v in z.data)

o = pycoeus.ones([2, 3])
assert o.shape == [2, 3]
assert all(v == 1.0 for v in o.data)

f = pycoeus.full([4], 3.14)
assert f.shape == [4]
assert abs(f.data[0] - 3.14) < 1e-9
assert abs(f.data[3] - 3.14) < 1e-9

a = pycoeus.arange(0.0, 5.0, 1.0)
assert a.shape == [5]
assert a.data == [0.0, 1.0, 2.0, 3.0, 4.0]

a2 = pycoeus.arange(0.0, 1.0, 0.25)
assert len(a2.data) == 4

lin = pycoeus.linspace(0.0, 1.0, 5)
assert lin.shape == [5]
assert abs(lin.data[0] - 0.0) < 1e-9
assert abs(lin.data[4] - 1.0) < 1e-9
assert abs(lin.data[2] - 0.5) < 1e-9
"#,
    );
}

#[test]
fn test_functional_matmul() {
    run_script(
        r#"
import pycoeus

a = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
b = pycoeus.Tensor([7.0, 8.0, 9.0, 10.0, 11.0, 12.0], [3, 2])
c = pycoeus.matmul(a, b)
assert c.shape == [2, 2]
assert abs(c.data[0] - 58.0) < 1e-9
assert abs(c.data[3] - 154.0) < 1e-9
"#,
    );
}

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
fn test_max_min_axis() {
    run_script(
        r#"
import pycoeus

x = pycoeus.Tensor([1.0, 3.0, 2.0, 4.0, 0.0, 5.0], [2, 3])

mx = pycoeus.max_axis(x, 1)
assert mx.shape == [2, 1]
assert abs(mx.data[0] - 3.0) < 1e-9
assert abs(mx.data[1] - 5.0) < 1e-9

mn = pycoeus.min_axis(x, 1)
assert mn.shape == [2, 1]
assert abs(mn.data[0] - 1.0) < 1e-9
assert abs(mn.data[1] - 0.0) < 1e-9
"#,
    );
}

#[test]
fn test_sum_mean_global() {
    run_script(
        r#"
import pycoeus

x = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0], [2, 2])

s = pycoeus.sum(x)
assert len(s.data) == 1
assert abs(s.data[0] - 10.0) < 1e-9

m = pycoeus.mean(x)
assert len(m.data) == 1
assert abs(m.data[0] - 2.5) < 1e-9
"#,
    );
}

#[test]
fn test_reshape_permute_t() {
    run_script(
        r#"
import pycoeus

x = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])

r = pycoeus.reshape(x, [3, 2])
assert r.shape == [3, 2]
assert r.data == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

tp = pycoeus.t(x)
assert tp.shape == [3, 2]
assert abs(tp.data[0] - 1.0) < 1e-9
assert abs(tp.data[1] - 4.0) < 1e-9

x3 = pycoeus.Tensor(list(range(24)), [2, 3, 4])
perm = pycoeus.permute(x3, [2, 0, 1])
assert perm.shape == [4, 2, 3]
"#,
    );
}

#[test]
fn test_stack_functional() {
    run_script(
        r#"
import pycoeus

a = pycoeus.Tensor([1.0, 2.0, 3.0], [3])
b = pycoeus.Tensor([4.0, 5.0, 6.0], [3])
s = pycoeus.stack([a, b], 0)
assert s.shape == [2, 3]
assert s.data == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

# Stack along dim=1
a2 = pycoeus.Tensor([1.0, 2.0], [2])
b2 = pycoeus.Tensor([3.0, 4.0], [2])
s2 = pycoeus.stack([a2, b2], 1)
assert s2.shape == [2, 2]
"#,
    );
}

#[test]
fn test_log_sum_exp() {
    run_script(
        r#"
import pycoeus
import math

x = pycoeus.Tensor([1.0, 2.0, 3.0, 0.0, 1.0, 2.0], [2, 3])
lse = pycoeus.log_sum_exp(x, 1)
assert lse.shape == [2, 1]
# log(exp(1)+exp(2)+exp(3)) ≈ 3.407606
expected_row0 = math.log(math.exp(1.0) + math.exp(2.0) + math.exp(3.0))
assert abs(lse.data[0] - expected_row0) < 1e-5
"#,
    );
}

#[test]
fn test_new_ops_backward() {
    run_script(
        r#"
import pycoeus

# abs backward
x = pycoeus.Tensor([-2.0, 1.0, -3.0, 4.0], [2, 2], requires_grad=True)
y = pycoeus.abs(x)
loss = pycoeus.sum(y)
loss.backward()
grad = x.grad
assert grad is not None
assert grad == [-1.0, 1.0, -1.0, 1.0]

# clamp backward
x2 = pycoeus.Tensor([-3.0, 0.5, 1.5, 4.0], [4], requires_grad=True)
y2 = pycoeus.clamp(x2, -1.0, 2.0)
loss2 = pycoeus.sum(y2)
loss2.backward()
grad2 = x2.grad
assert grad2 is not None
# -3 is below lo → 0; 0.5 inside → 1; 1.5 inside → 1; 4 above hi → 0
assert grad2 == [0.0, 1.0, 1.0, 0.0]

# stack backward
a = pycoeus.Tensor([1.0, 2.0], [2], requires_grad=True)
b = pycoeus.Tensor([3.0, 4.0], [2], requires_grad=True)
s = pycoeus.stack([a, b], 0)
loss_s = pycoeus.sum(s)
loss_s.backward()
assert a.grad == [1.0, 1.0]
assert b.grad == [1.0, 1.0]
"#,
    );
}

#[test]
fn test_topk_and_sort() {
    run_script(
        r#"
import pycoeus

x = pycoeus.Tensor([3.0, 1.0, 4.0, 1.5, 9.0, 2.6], [6])

# topk: top 3 values
vals, idxs = pycoeus.topk(x, 3, 0)
assert vals.shape == [3]
v = vals.data
assert v[0] >= v[1] >= v[2], f"topk values should be descending: {v}"

# sort: ascending
sv, si = pycoeus.sort(x, dim=0, descending=False)
assert sv.shape == [6]
sd = sv.data
for i in range(len(sd) - 1):
    assert sd[i] <= sd[i+1], f"sort not ascending at {i}: {sd}"

# sort: descending
sv2, si2 = pycoeus.sort(x, dim=0, descending=True)
sd2 = sv2.data
for i in range(len(sd2) - 1):
    assert sd2[i] >= sd2[i+1], f"sort not descending at {i}: {sd2}"
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

# Tensor method forms
eq2 = a.eq(b)
assert eq2.data == [0.0, 1.0, 0.0, 0.0]

lt2 = a.lt(b)
assert lt2.data == [1.0, 0.0, 0.0, 1.0]

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
fn test_randn_zeros_ones_like_eye() {
    run_script(
        r#"
import pycoeus

# randn: should produce different values each call (not all zero)
r = pycoeus.randn([3, 4])
assert r.shape == [3, 4]
assert not all(v == 0.0 for v in r.data), "randn returned all zeros"

# zeros_like
z = pycoeus.zeros([2, 3])
zl = pycoeus.zeros_like(z)
assert zl.shape == [2, 3]
assert all(v == 0.0 for v in zl.data)

# ones_like
ol = pycoeus.ones_like(z)
assert ol.shape == [2, 3]
assert all(v == 1.0 for v in ol.data)

# eye
e = pycoeus.eye(3)
assert e.shape == [3, 3]
assert e.data[0] == 1.0 and e.data[1] == 0.0 and e.data[4] == 1.0
"#,
    );
}

#[test]
fn test_gather_scatter() {
    run_script(
        r#"
import pycoeus

# gather: select from [10, 20, 30, 40] using index [2, 0, 1]
x = pycoeus.Tensor([10.0, 20.0, 30.0, 40.0], [4])
idx = pycoeus.Tensor([2.0, 0.0, 1.0], [3])
g = pycoeus.gather(x, 0, idx)
assert g.data == [30.0, 10.0, 20.0], f"gather wrong: {g.data}"

# 2-D gather along dim=1
x2 = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
idx2 = pycoeus.Tensor([2.0, 0.0, 1.0, 0.0], [2, 2])
g2 = pycoeus.gather(x2, 1, idx2)
assert g2.shape == [2, 2]
# row 0: [1,2,3] → indices [2,0] → [3,1]
# row 1: [4,5,6] → indices [1,0] → [5,4]
assert g2.data == [3.0, 1.0, 5.0, 4.0], f"2d gather wrong: {g2.data}"

# scatter_add
base = pycoeus.Tensor([0.0, 0.0, 0.0, 0.0], [4])
idx3 = pycoeus.Tensor([1.0, 2.0, 1.0], [3])
src  = pycoeus.Tensor([10.0, 20.0, 30.0], [3])
out = pycoeus.scatter_add(base, 0, idx3, src)
# idx=[1,2,1]: out[1] += 10+30 = 40, out[2] += 20
assert out.data == [0.0, 40.0, 20.0, 0.0], f"scatter_add wrong: {out.data}"
"#,
    );
}

#[test]
fn test_tensor_index_slice_and_iter() {
    run_script(
        r#"
import pycoeus

x = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2])

assert len(x) == 3, f"len wrong: {len(x)}"

row0 = x[0]
assert row0.shape == [2], f"row0 shape wrong: {row0.shape}"
assert row0.data == [1.0, 2.0], f"row0 data wrong: {row0.data}"

last = x[-1]
assert last.shape == [2], f"last shape wrong: {last.shape}"
assert last.data == [5.0, 6.0], f"last data wrong: {last.data}"

middle = x[1:3]
assert middle.shape == [2, 2], f"middle shape wrong: {middle.shape}"
assert middle.data == [3.0, 4.0, 5.0, 6.0], f"middle data wrong: {middle.data}"

rows = [row.data for row in x]
assert rows == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], f"iter rows wrong: {rows}"

try:
    _ = x[::2]
    raise AssertionError("slice step should fail")
except ValueError:
    pass

scalar = pycoeus.Tensor([7.0], [])
try:
    _ = len(scalar)
    raise AssertionError("len(scalar) should fail")
except TypeError:
    pass

try:
    _ = scalar[0]
    raise AssertionError("scalar indexing should fail")
except IndexError:
    pass
"#,
    );
}

#[test]
fn test_repeat_interleave_and_interpolate() {
    run_script(
        r#"
import pycoeus

# repeat_interleave along dim=0, repeats=2
x = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
r = pycoeus.repeat_interleave(x, 2, 0)
assert r.shape == [4, 2], f"repeat_interleave shape wrong: {r.shape}"
# Each row repeated: [1,2], [1,2], [3,4], [3,4]
assert r.data == [1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0], f"repeat_interleave data wrong: {r.data}"

# interpolate nearest 1-D: [N=1, C=1, L=4] → [N=1, C=1, L=8]
xL = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0], [1, 1, 4])
out1d = pycoeus.interpolate(xL, [8], mode="nearest")
assert out1d.shape == [1, 1, 8], f"interp1d shape: {out1d.shape}"
# Each element repeated ≈ 2 times
d = out1d.data
assert d[0] == d[1] == 1.0, f"interp1d elem 0 wrong: {d[:2]}"

# interpolate nearest 2-D: [1,1,2,2] → [1,1,4,4]
x2d = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0], [1, 1, 2, 2])
out2d = pycoeus.interpolate(x2d, [4, 4], mode="nearest")
assert out2d.shape == [1, 1, 4, 4]
d2 = out2d.data
assert d2[0] == 1.0, f"interp2d corner wrong: {d2[0]}"
assert d2[3] == 2.0, f"interp2d top-right wrong: {d2[3]}"
"#,
    );
}

#[test]
fn test_statistical_ops() {
    run_script(
        r#"
import pycoeus
import math

x = pycoeus.Tensor([2.0, 4.0, 6.0, 8.0], [4])

# std (unbiased)
s = pycoeus.std(x)
expected_std = math.sqrt(((4.0 + 0.0 + 4.0 + 16.0) / 3.0))  # wait, let me recompute
# mean=5, deviations: -3,-1,1,3; sq: 9,1,1,9; sum=20; N-1=3; var=20/3; std=sqrt(20/3)
expected_std2 = math.sqrt(20.0 / 3.0)
assert abs(s - expected_std2) < 1e-9, f"std wrong: {s} vs {expected_std2}"

# var (unbiased)
v = pycoeus.var(x)
assert abs(v - 20.0 / 3.0) < 1e-9, f"var wrong: {v}"

# norm (L2)
n = pycoeus.norm(x)
expected_norm = math.sqrt(4 + 16 + 36 + 64)
assert abs(n - expected_norm) < 1e-9, f"norm wrong: {n}"
"#,
    );
}
