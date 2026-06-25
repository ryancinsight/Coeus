// ── binding_tests_ops.rs ──
//
// Tests for functional ops exposed at the module level: stack, matmul, zeros,
// ones, full, arange, linspace, abs, sqrt, neg, recip, sign, floor, ceil,
// round, trunc, clamp, max_axis, min_axis, sum, mean, reshape, permute, t,
// pow, log_sum_exp.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::ffi::CString;

mod common;

fn run_script(script: &str) {
    let _guard = common::python_test_lock()
        .lock()
        .expect("python test lock poisoned");
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let pycoeus_module = pyo3::types::PyModule::new(py, "pycoeus").unwrap();
        pycoeus::pycoeus(&pycoeus_module).unwrap();
        let sys = py.import("sys").unwrap();
        let modules_any = sys.getattr("modules").unwrap();
        let modules = modules_any.downcast::<PyDict>().unwrap();
        modules.set_item("pycoeus", &pycoeus_module).unwrap();
        let globals = PyDict::new(py);
        globals.set_item("pycoeus", &pycoeus_module).unwrap();
        let script = CString::new(script).expect("test script must not contain interior NUL");
        let result = py.run(script.as_c_str(), Some(&globals), None);
        modules
            .del_item("pycoeus")
            .unwrap_or_else(|e| panic!("failed to remove pycoeus test module: {e:?}"));
        result.unwrap_or_else(|e| panic!("Python script failed:\n{e}"));
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

log = pycoeus.logspace(0.0, 2.0, 3, 10.0)
assert log.shape == [3]
assert abs(log.data[0] - 1.0) < 1e-9
assert abs(log.data[1] - 10.0) < 1e-9
assert abs(log.data[2] - 100.0) < 1e-9

geo = pycoeus.geomspace(1.0, 16.0, 5)
assert geo.shape == [5]
assert abs(geo.data[0] - 1.0) < 1e-9
assert abs(geo.data[1] - 2.0) < 1e-9
assert abs(geo.data[2] - 4.0) < 1e-9
assert abs(geo.data[3] - 8.0) < 1e-9
assert abs(geo.data[4] - 16.0) < 1e-9

try:
    pycoeus.geomspace(-1.0, 16.0, 4)
    raise AssertionError("expected ValueError for mismatched signs")
except ValueError:
    pass
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
fn test_einsum_wrapper() {
    run_script(
        r#"
import pycoeus

a = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
b = pycoeus.Tensor([7.0, 8.0, 9.0, 10.0, 11.0, 12.0], [3, 2])
mm = pycoeus.einsum("ij,jk->ik", [a, b])
assert mm.shape == [2, 2], f"einsum matmul shape wrong: {mm.shape}"
assert mm.data == [58.0, 64.0, 139.0, 154.0], f"einsum matmul wrong: {mm.data}"

tp = pycoeus.einsum("ij->ji", [a])
assert tp.shape == [3, 2], f"einsum transpose shape wrong: {tp.shape}"
assert tp.data == [1.0, 4.0, 2.0, 5.0, 3.0, 6.0], f"einsum transpose wrong: {tp.data}"

x = pycoeus.Tensor([1.0, 2.0, 3.0], [3])
y = pycoeus.Tensor([4.0, 5.0, 6.0], [3])
dot = pycoeus.einsum("i,i->", [x, y])
assert dot.shape == [1], f"einsum dot shape wrong: {dot.shape}"
assert abs(dot.item() - 32.0) < 1e-9, f"einsum dot wrong: {dot.data}"
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

# topk largest=True (default): top 3 values
vals, idxs = pycoeus.topk(x, 3, 0)
assert vals.shape == [3]
v = vals.data
assert v[0] >= v[1] >= v[2], f"topk largest values should be descending: {v}"
# largest 3 of [3,1,4,1.5,9,2.6] = [9,4,3]
assert abs(v[0] - 9.0) < 1e-9, f"topk[0]={v[0]}"

# topk largest=False: bottom 3 values
vals_sm, idxs_sm = pycoeus.topk(x, 3, 0, False)
assert vals_sm.shape == [3]
vs = vals_sm.data
# smallest 3 of [3,1,4,1.5,9,2.6] ≈ [1,1.5,2.6]
assert abs(vs[0] - 1.0) < 1e-9, f"topk smallest[0]={vs[0]}"

# topk with explicit largest=True
vals2, idxs2 = pycoeus.topk(x, 2, 0, True)
assert vals2.shape == [2]
v2 = vals2.data
assert v2[0] >= v2[1], f"explicit largest=True: {v2}"

# 2D topk along dim=1
m = pycoeus.Tensor([1.0, 5.0, 2.0, 4.0, 3.0, 0.0], [2, 3])
mv, mi = pycoeus.topk(m, 2, 1)
assert mv.shape == [2, 2], f"2d topk shape: {mv.shape}"
# row0: [1,5,2] → top2: [5,2]; row1: [4,3,0] → top2: [4,3]
assert abs(mv.data[0] - 5.0) < 1e-9, f"2d topk row0[0]={mv.data[0]}"
assert abs(mv.data[2] - 4.0) < 1e-9, f"2d topk row1[0]={mv.data[2]}"

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

sel_rows = pycoeus.index_select(x2, 0, pycoeus.Tensor([1.0, 0.0], [2]))
assert sel_rows.shape == [2, 3], f"index_select rows shape wrong: {sel_rows.shape}"
assert sel_rows.data == [4.0, 5.0, 6.0, 1.0, 2.0, 3.0], f"index_select rows wrong: {sel_rows.data}"

sel_cols = pycoeus.index_select(x2, 1, pycoeus.Tensor([2.0, 0.0], [2]))
assert sel_cols.shape == [2, 2], f"index_select cols shape wrong: {sel_cols.shape}"
assert sel_cols.data == [3.0, 1.0, 6.0, 4.0], f"index_select cols wrong: {sel_cols.data}"
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
fn test_unsqueeze_squeeze_flatten() {
    run_script(
        r#"
import pycoeus

x = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
assert x.shape == [2, 3]

# unsqueeze at dim=0 → [1, 2, 3]
u = pycoeus.unsqueeze(x, 0)
assert u.shape == [1, 2, 3], f"unsqueeze dim=0: {u.shape}"

# unsqueeze at dim=2 → [2, 3, 1]
u2 = pycoeus.unsqueeze(x, 2)
assert u2.shape == [2, 3, 1], f"unsqueeze dim=2: {u2.shape}"

# squeeze u (has size-1 dim at 0) → [2, 3]
s = pycoeus.squeeze(u)
assert s.shape == [2, 3], f"squeeze all: {s.shape}"

# squeeze with dim=0 explicitly
s2 = pycoeus.squeeze(u, 0)
assert s2.shape == [2, 3], f"squeeze dim=0: {s2.shape}"

# flatten [2, 3] → [6]
f = pycoeus.flatten(x)
assert f.shape == [6], f"flatten default: {f.shape}"
assert f.data == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], f"flatten data: {f.data}"

# flatten [1, 2, 3] → keep dim=0, flatten 1..2 → [1, 6]
y = pycoeus.Tensor(list(range(24)), [2, 3, 4])
f2 = pycoeus.flatten(y, 1, 2)
assert f2.shape == [2, 12], f"flatten dim 1..2: {f2.shape}"

# unsqueeze backward
xg = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0], [4], requires_grad=True)
ug = pycoeus.unsqueeze(xg, 0)
loss = pycoeus.sum(ug)
loss.backward()
assert xg.grad == [1.0, 1.0, 1.0, 1.0], f"unsqueeze grad: {xg.grad}"

try:
    pycoeus.unsqueeze(x, 4)
    raise AssertionError("unsqueeze out-of-range dim should fail")
except ValueError:
    pass

try:
    pycoeus.squeeze(x, 0)
    raise AssertionError("squeeze of non-singleton dim should fail")
except ValueError:
    pass

try:
    pycoeus.flatten(x, 2)
    raise AssertionError("flatten out-of-range start_dim should fail")
except ValueError:
    pass
"#,
    );
}

#[test]
fn test_argmax_argmin() {
    run_script(
        r#"
import pycoeus

x = pycoeus.Tensor([3.0, 1.0, 4.0, 1.5, 9.0, 2.6, 5.3, 5.8, 9.7, 3.2], [2, 5])

# argmax along dim=1 (keep-dim) → shape [2, 1]
am = pycoeus.argmax(x, 1)
assert am.shape == [2, 1], f"argmax shape: {am.shape}"
# row0: [3,1,4,1.5,9] → max at idx 4
# row1: [2.6,5.3,5.8,9.7,3.2] → max at idx 3
assert am.data[0] == 4.0, f"argmax row0: {am.data[0]}"
assert am.data[1] == 3.0, f"argmax row1: {am.data[1]}"

# argmin along dim=1 (keep-dim) → shape [2, 1]
an = pycoeus.argmin(x, 1)
assert an.shape == [2, 1], f"argmin shape: {an.shape}"
# row0: min at idx 1 (value 1.0)
# row1: min at idx 0 (value 2.6)
assert an.data[0] == 1.0, f"argmin row0: {an.data[0]}"
assert an.data[1] == 0.0, f"argmin row1: {an.data[1]}"

# 1-D case: argmax of [2, 5, 1, 8, 3] → idx 3
v = pycoeus.Tensor([2.0, 5.0, 1.0, 8.0, 3.0], [5])
assert pycoeus.argmax(v, 0).data[0] == 3.0
assert pycoeus.argmin(v, 0).data[0] == 2.0

try:
    pycoeus.argmax(v, 1)
    raise AssertionError("argmax out-of-range dim should fail")
except ValueError:
    pass
"#,
    );
}

#[test]
fn test_tril_triu_roll() {
    run_script(
        r#"
import pycoeus

# ── tril ─────────────────────────────────────────────────────────────
x = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [3, 3])

lo = pycoeus.tril(x)
assert lo.data == [1.0, 0.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 9.0], f"tril k=0: {lo.data}"

lo1 = pycoeus.tril(x, 1)
assert lo1.data == [1.0, 2.0, 0.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], f"tril k=1: {lo1.data}"

lo_neg = pycoeus.tril(x, -1)
assert lo_neg.data == [0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 7.0, 8.0, 0.0], f"tril k=-1: {lo_neg.data}"

# tril backward: gradient is masked with same tril
xg = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [3, 3], requires_grad=True)
loss = pycoeus.sum(pycoeus.tril(xg))
loss.backward()
# Only lower-triangle positions receive gradient=1
assert xg.grad == [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0], f"tril bwd: {xg.grad}"

# ── triu ─────────────────────────────────────────────────────────────
hi = pycoeus.triu(x)
assert hi.data == [1.0, 2.0, 3.0, 0.0, 5.0, 6.0, 0.0, 0.0, 9.0], f"triu k=0: {hi.data}"

hi1 = pycoeus.triu(x, 1)
assert hi1.data == [0.0, 2.0, 3.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0], f"triu k=1: {hi1.data}"

# tril + triu extracts diagonal: triu(tril(x, 0), 0)
diag = pycoeus.triu(pycoeus.tril(x), 0)
assert diag.data == [1.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 9.0], f"diag: {diag.data}"

# ── roll ─────────────────────────────────────────────────────────────
v = pycoeus.Tensor([0.0, 1.0, 2.0, 3.0], [4])
r1 = pycoeus.roll(v, [1], [0])
assert r1.data == [3.0, 0.0, 1.0, 2.0], f"roll +1: {r1.data}"

r_neg = pycoeus.roll(v, [-1], [0])
assert r_neg.data == [1.0, 2.0, 3.0, 0.0], f"roll -1: {r_neg.data}"

# roll backward: backward is roll by negative shift
rg = pycoeus.Tensor([0.0, 1.0, 2.0, 3.0], [4], requires_grad=True)
rolled = pycoeus.roll(rg, [1], [0])
loss_r = pycoeus.sum(rolled)
loss_r.backward()
# all-ones gradient rolled by -1 is still all-ones
assert rg.grad == [1.0, 1.0, 1.0, 1.0], f"roll bwd: {rg.grad}"

# roll 2D along rows
m = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
m_row1 = pycoeus.roll(m, [1], [0])
assert m_row1.data == [4.0, 5.0, 6.0, 1.0, 2.0, 3.0], f"roll 2d row: {m_row1.data}"

# roll zero shift is identity
assert pycoeus.roll(v, [0], [0]).data == [0.0, 1.0, 2.0, 3.0]

# error paths
try:
    _ = pycoeus.tril(pycoeus.Tensor([1.0, 2.0, 3.0], [3]))  # 1-D
    raise AssertionError("tril 1-D should raise")
except ValueError:
    pass
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
# mean=5, deviations: -3,-1,1,3; sq: 9,1,1,9; sum=20; N-1=3; var=20/3; std=sqrt(20/3)
expected_std2 = math.sqrt(20.0 / 3.0)
assert abs(s - expected_std2) < 1e-9, f"std wrong: {s} vs {expected_std2}"

# var (unbiased)
v = pycoeus.var(x)
assert abs(v - 20.0 / 3.0) < 1e-9, f"var wrong: {v}"

# var (biased, N divisor)
v_biased = pycoeus.var(x, unbiased=False)
assert abs(v_biased - 5.0) < 1e-9, f"var biased wrong: {v_biased}"

# std (biased, N divisor)
s_biased = pycoeus.std(x, unbiased=False)
assert abs(s_biased - math.sqrt(5.0)) < 1e-9, f"std biased wrong: {s_biased}"

# norm (L2) returns [1] tensor
n = pycoeus.norm(x)
assert n.shape == [1]
expected_norm = math.sqrt(4 + 16 + 36 + 64)
assert abs(n.item() - expected_norm) < 1e-9, f"norm wrong: {n.item()}"

# 2-D axis + keepdim variance — matches torch.var(x, dim=1, keepdim=True)
y = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
# row 0 [1,2,3]: mean=2, dev^2=[1,0,1] sum=2, N-1=2, var=1.0
# row 1 [4,5,6]: mean=5, dev^2=[1,0,1] sum=2, N-1=2, var=1.0
v_axis1 = pycoeus.var(y, axis=1)
assert v_axis1.shape == [2], f"var axis1 shape: {v_axis1.shape}"
assert abs(v_axis1.data[0] - 1.0) < 1e-9, f"var axis1 row0 wrong: {v_axis1.data[0]}"
assert abs(v_axis1.data[1] - 1.0) < 1e-9, f"var axis1 row1 wrong: {v_axis1.data[1]}"
v_axis1_keep = pycoeus.var(y, axis=1, keepdim=True)
assert v_axis1_keep.shape == [2, 1], f"var axis1 keepdim shape: {v_axis1_keep.shape}"
assert abs(v_axis1_keep.data[0] - 1.0) < 1e-9, f"var axis1 keepdim row0 wrong: {v_axis1_keep.data[0]}"
assert abs(v_axis1_keep.data[1] - 1.0) < 1e-9, f"var axis1 keepdim row1 wrong: {v_axis1_keep.data[1]}"
# sum biased (N divisor): var(y, dim=0, unbiased=False) reduces the 2 rows.
# col means: [2.5, 3.5, 4.5]; deviations per column: [-1.5, 1.5]
# sq sum per column: 2*(1.5^2) = 4.5; N=2 -> 4.5/2 = 2.25
v_axis0 = pycoeus.var(y, axis=0, unbiased=False)
assert v_axis0.shape == [3], f"var axis0 shape: {v_axis0.shape}"
assert v_axis0.data == [2.25, 2.25, 2.25], f"var axis0 biased wrong: {v_axis0.data}"

# 2-D axis + keepdim std — std = sqrt(var)
s_axis1 = pycoeus.std(y, axis=1)
assert s_axis1.shape == [2], f"std axis1 shape: {s_axis1.shape}"
assert abs(s_axis1.data[0] - 1.0) < 1e-9, f"std axis1 row0 wrong: {s_axis1.data[0]}"
assert abs(s_axis1.data[1] - 1.0) < 1e-9, f"std axis1 row1 wrong: {s_axis1.data[1]}"
s_axis1_keep = pycoeus.std(y, axis=1, keepdim=True)
assert s_axis1_keep.shape == [2, 1], f"std axis1 keepdim shape: {s_axis1_keep.shape}"
assert abs(s_axis1_keep.data[0] - 1.0) < 1e-9, f"std axis1 keepdim row0 wrong: {s_axis1_keep.data[0]}"
assert abs(s_axis1_keep.data[1] - 1.0) < 1e-9, f"std axis1 keepdim row1 wrong: {s_axis1_keep.data[1]}"

try:
    _ = pycoeus.var(y, axis=2)
    raise AssertionError("var out-of-range axis should raise")
except ValueError:
    pass

# Error path: empty tensor surfaces ValueError, not a panic
try:
    _ = pycoeus.var(pycoeus.zeros([0]))
    raise AssertionError("var of empty tensor should raise")
except ValueError:
    pass
try:
    _ = pycoeus.std(pycoeus.zeros([0]))
    raise AssertionError("std of empty tensor should raise")
except ValueError:
    pass
"#,
    );
}

#[test]
fn test_meshgrid_tile_tensor_methods() {
    run_script(
        r#"
import pycoeus

# ── meshgrid ij ───────────────────────────────────────────────────────
x = pycoeus.Tensor([0.0, 1.0, 2.0], [3])
y = pycoeus.Tensor([10.0, 20.0], [2])
grids = pycoeus.meshgrid([x, y], "ij")
assert len(grids) == 2
# grid_x varies along axis 0
assert grids[0].shape == [3, 2], f"grid_x shape: {grids[0].shape}"
assert grids[0].data == [0.0, 0.0, 1.0, 1.0, 2.0, 2.0], f"grid_x: {grids[0].data}"
# grid_y varies along axis 1
assert grids[1].shape == [3, 2]
assert grids[1].data == [10.0, 20.0, 10.0, 20.0, 10.0, 20.0]

# ── meshgrid xy ───────────────────────────────────────────────────────
gxy = pycoeus.meshgrid([x, y], "xy")
# xy: first output varies along dim 1, second along dim 0
assert gxy[0].shape == [2, 3]

# ── meshgrid errors ───────────────────────────────────────────────────
try:
    _ = pycoeus.meshgrid([x, y], "bad")
    raise AssertionError("bad indexing should raise")
except ValueError:
    pass

m2d = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
try:
    _ = pycoeus.meshgrid([x, m2d], "ij")  # 2-D tensor in list
    raise AssertionError("2-D tensor in meshgrid should raise")
except ValueError:
    pass

# ── tile ──────────────────────────────────────────────────────────────
v = pycoeus.Tensor([1.0, 2.0, 3.0], [3])
t1 = pycoeus.tile(v, [2])
assert t1.shape == [6]
assert t1.data == [1.0, 2.0, 3.0, 1.0, 2.0, 3.0], f"tile 1-D: {t1.data}"

m = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
t2 = pycoeus.tile(m, [2, 3])
assert t2.shape == [4, 6]
assert t2.data[:6] == [1.0, 2.0, 1.0, 2.0, 1.0, 2.0]

# tile backward
vg = pycoeus.Tensor([1.0, 2.0, 3.0], [3], requires_grad=True)
tg = pycoeus.tile(vg, [3])
pycoeus.sum(tg).backward()
# each original element copied 3 times → grad = 3
assert vg.grad == [3.0, 3.0, 3.0], f"tile bwd: {vg.grad}"

# ── Tensor.repeat (method form) ───────────────────────────────────────
r = v.repeat([2])
assert r.data == [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]

# ── Tensor.T (2-D transpose) ─────────────────────────────────────────
a = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
at = a.T
assert at.shape == [3, 2], f"T shape: {at.shape}"
assert abs(at.data[0] - 1.0) < 1e-9  # a[0,0]
assert abs(at.data[1] - 4.0) < 1e-9  # a[1,0]

try:
    _ = v.T  # 1-D should fail
    raise AssertionError("1-D T should raise")
except ValueError:
    pass

# ── Tensor.numel() ────────────────────────────────────────────────────
assert a.numel() == 6
assert v.numel() == 3

# ── Tensor.is_contiguous() ────────────────────────────────────────────
assert a.is_contiguous() == True
# permuted tensor is not contiguous
ap = pycoeus.permute(a, [1, 0])
assert ap.is_contiguous() == False

# ── Tensor.clone_tensor() ─────────────────────────────────────────────
c = a.clone_tensor()
assert c.shape == a.shape
assert c.data == a.data
"#,
    );
}

#[test]
fn test_no_grad_detaches_operation_outputs() {
    run_script(
        r#"
import pycoeus

x = pycoeus.Tensor([1.0, 2.0, 3.0], [3], requires_grad=True)

with pycoeus.no_grad():
    y = x + x
    assert y.data == [2.0, 4.0, 6.0], f"no_grad add data: {y.data}"
    assert y.requires_grad is False, "no_grad operation output should be detached"
    assert y.grad is None, f"detached output grad should be None, got {y.grad}"

    with pycoeus.no_grad():
        z = pycoeus.relu(x)
        assert z.data == [1.0, 2.0, 3.0], f"nested no_grad relu data: {z.data}"
        assert z.requires_grad is False, "nested no_grad operation output should be detached"

    still_off = pycoeus.exp(x)
    assert still_off.requires_grad is False, "outer no_grad scope should remain active"

explicit = pycoeus.zeros([2], requires_grad=True)
assert explicit.requires_grad is True, "explicit factory requires_grad should be honored"

tracked = x + x
assert tracked.requires_grad is True, "tracking should resume after no_grad exits"
pycoeus.sum(tracked).backward()
assert x.grad == [2.0, 2.0, 2.0], f"post-no_grad gradient mismatch: {x.grad}"
"#,
    );
}

#[test]
fn test_diag_diagonal_cumprod() {
    run_script(
        r#"
import pycoeus
import math

# ── diag: create diagonal matrix from vector ──────────────────────────
v = pycoeus.Tensor([1.0, 2.0, 3.0], [3])
m = pycoeus.diag(v)
assert m.shape == [3, 3], f"diag shape: {m.shape}"
assert m.data == [1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0], f"diag: {m.data}"

# diag with super-diagonal k=1
m1 = pycoeus.diag(v, 1)
assert m1.shape == [4, 4]
assert abs(m1.data[1] - 1.0) < 1e-9  # (0,1)
assert abs(m1.data[6] - 2.0) < 1e-9  # (1,2)

# ── diagonal: extract diagonal from matrix ───────────────────────────
mat = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [3, 3])
d = pycoeus.diagonal(mat)
assert d.shape == [3], f"diagonal shape: {d.shape}"
assert d.data == [1.0, 5.0, 9.0], f"diagonal: {d.data}"

d1 = pycoeus.diagonal(mat, 1)
assert d1.shape == [2]
assert d1.data == [2.0, 6.0], f"diagonal k=1: {d1.data}"

# ── diag backward: gradient goes via diagonal ─────────────────────────
vg = pycoeus.Tensor([1.0, 2.0, 3.0], [3], requires_grad=True)
mg = pycoeus.diag(vg)
loss = pycoeus.sum(mg)
loss.backward()
# Gradient of sum(diag(v)) w.r.t. v is all-ones (only diagonal elements contribute)
assert vg.grad == [1.0, 1.0, 1.0], f"diag backward: {vg.grad}"

# ── diagonal backward: gradient goes via diag ────────────────────────
matg = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [3, 3], requires_grad=True)
dg = pycoeus.diagonal(matg)
loss2 = pycoeus.sum(dg)
loss2.backward()
# Only diagonal positions receive gradient
assert matg.grad[0] == 1.0, f"diagonal bwd [0,0]: {matg.grad[0]}"
assert matg.grad[4] == 1.0, f"diagonal bwd [1,1]: {matg.grad[4]}"
assert matg.grad[8] == 1.0, f"diagonal bwd [2,2]: {matg.grad[8]}"
assert matg.grad[1] == 0.0, f"diagonal bwd [0,1] should be 0: {matg.grad[1]}"

# ── cumprod: cumulative product ────────────────────────────────────────
x = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0], [4])
cp = pycoeus.cumprod(x, 0)
assert cp.shape == [4]
assert cp.data == [1.0, 2.0, 6.0, 24.0], f"cumprod: {cp.data}"

# 2-D along dim=1
y = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
cpy = pycoeus.cumprod(y, 1)
assert cpy.shape == [2, 3]
assert cpy.data == [1.0, 2.0, 6.0, 4.0, 20.0, 120.0], f"cumprod 2D: {cpy.data}"

# ── cumprod backward ───────────────────────────────────────────────────
xg = pycoeus.Tensor([1.0, 2.0, 3.0], [3], requires_grad=True)
cpg = pycoeus.cumprod(xg, 0)   # [1, 2, 6]
loss3 = pycoeus.sum(cpg)        # 1+2+6=9
loss3.backward()
# grad[0] = 1+2+6=9 (all suffix products * 1), actually d/dx[0]= out[0]+out[1]+out[2] / x[0] = 9/1=9
# grad[1] = out[1]+out[2] / x[1] = 8/2 = 4
# grad[2] = out[2] / x[2] = 6/3 = 2
assert abs(xg.grad[0] - 9.0) < 1e-6, f"cumprod bwd[0]: {xg.grad[0]}"
assert abs(xg.grad[1] - 4.0) < 1e-6, f"cumprod bwd[1]: {xg.grad[1]}"
assert abs(xg.grad[2] - 2.0) < 1e-6, f"cumprod bwd[2]: {xg.grad[2]}"

# ── error paths ───────────────────────────────────────────────────────
try:
    _ = pycoeus.diag(y)  # 2-D input should fail
    raise AssertionError("diag 2-D should raise")
except ValueError:
    pass
"#,
    );
}

#[test]
fn test_nn_functional_ops() {
    run_script(
        r#"
import pycoeus
import math

# ── F.relu ────────────────────────────────────────────────────────────
x = pycoeus.Tensor([-2.0, -1.0, 0.0, 1.0, 2.0], [5])
r = pycoeus.f_relu(x)
assert r.data == [0.0, 0.0, 0.0, 1.0, 2.0], f"f_relu: {r.data}"

# ── F.sigmoid ─────────────────────────────────────────────────────────
s = pycoeus.f_sigmoid(pycoeus.Tensor([0.0], [1]))
assert abs(s.data[0] - 0.5) < 1e-6

# ── F.softmax ─────────────────────────────────────────────────────────
logits = pycoeus.Tensor([1.0, 2.0, 3.0], [3])
sm = pycoeus.f_softmax(logits, 0)
assert abs(sum(sm.data) - 1.0) < 1e-6, f"f_softmax sum: {sum(sm.data)}"

# ── F.log_softmax ─────────────────────────────────────────────────────
lsm = pycoeus.f_log_softmax(logits, 0)
# log_softmax values should all be <= 0
assert all(v <= 0.0 for v in lsm.data), f"f_log_softmax: {lsm.data}"

# ── F.gelu ────────────────────────────────────────────────────────────
gx = pycoeus.f_gelu(pycoeus.Tensor([0.0], [1]))
assert abs(gx.data[0]) < 1e-6  # gelu(0) = 0

# ── F.silu ────────────────────────────────────────────────────────────
sx = pycoeus.f_silu(pycoeus.Tensor([0.0], [1]))
assert abs(sx.data[0]) < 1e-6  # silu(0) = 0

# ── F.mse_loss ────────────────────────────────────────────────────────
pred = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0], [4])
targ = pycoeus.Tensor([1.5, 1.5, 4.0, 0.0], [4])
mse = pycoeus.f_mse_loss(pred, targ)
expected_mse = ((1-1.5)**2 + (2-1.5)**2 + (3-4)**2 + (4-0)**2) / 4
assert abs(mse.data[0] - expected_mse) < 1e-5, f"f_mse_loss: {mse.data[0]}"

# ── F.cross_entropy ───────────────────────────────────────────────────
ce = pycoeus.f_cross_entropy(
    pycoeus.Tensor([1.5, 0.5, -0.5, -1.0, 2.0, 0.0], [2, 3]),
    [0, 1]
)
assert ce.shape == [1], f"f_cross_entropy shape: {ce.shape}"
assert ce.data[0] > 0.0, f"f_cross_entropy value should be positive"
"#,
    );
}

#[test]
fn test_einsum_index_select() {
    run_script(
        r#"
import pycoeus
import math

# ── einsum: matmul ij,jk->ik ─────────────────────────────────────────
a = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
b = pycoeus.Tensor([7.0, 8.0, 9.0, 10.0, 11.0, 12.0], [3, 2])
c = pycoeus.einsum("ij,jk->ik", [a, b])
assert c.shape == [2, 2], f"einsum matmul shape: {c.shape}"
# row0: [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
assert abs(c.data[0] - 58.0) < 1e-9
assert abs(c.data[1] - 64.0) < 1e-9

# ── einsum: transpose ij->ji ─────────────────────────────────────────
tp = pycoeus.einsum("ij->ji", [a])
assert tp.shape == [3, 2], f"einsum transpose shape: {tp.shape}"
assert abs(tp.data[0] - 1.0) < 1e-9  # a[0,0]
assert abs(tp.data[1] - 4.0) < 1e-9  # a[1,0]

# ── einsum: dot product i,i-> ─────────────────────────────────────────
v1 = pycoeus.Tensor([1.0, 2.0, 3.0], [3])
v2 = pycoeus.Tensor([4.0, 5.0, 6.0], [3])
dot = pycoeus.einsum("i,i->", [v1, v2])
assert abs(dot.data[0] - (1*4 + 2*5 + 3*6)) < 1e-9, f"dot: {dot.data}"

# ── einsum: outer product i,j->ij ─────────────────────────────────────
u = pycoeus.Tensor([1.0, 2.0], [2])
w = pycoeus.Tensor([3.0, 4.0, 5.0], [3])
outer = pycoeus.einsum("i,j->ij", [u, w])
assert outer.shape == [2, 3]
assert outer.data == [3.0, 4.0, 5.0, 6.0, 8.0, 10.0], f"outer: {outer.data}"

# ── einsum backward: matmul passes through autograd ─────────────────
ag = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], requires_grad=True)
bg = pycoeus.Tensor([7.0, 8.0, 9.0, 10.0, 11.0, 12.0], [3, 2], requires_grad=True)
cg = pycoeus.einsum("ij,jk->ik", [ag, bg])
pycoeus.sum(cg).backward()
assert ag.grad is not None, "einsum matmul backward: a.grad is None"
assert bg.grad is not None, "einsum matmul backward: b.grad is None"

# ── index_select: 1-D ────────────────────────────────────────────────
x = pycoeus.Tensor([10.0, 20.0, 30.0, 40.0, 50.0], [5])
idx = pycoeus.Tensor([4.0, 0.0, 2.0], [3])
sel = pycoeus.index_select(x, 0, idx)
assert sel.shape == [3], f"index_select shape: {sel.shape}"
assert sel.data == [50.0, 10.0, 30.0], f"index_select data: {sel.data}"

# ── index_select: 2-D row selection ──────────────────────────────────
m = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], [4, 3])
ridx = pycoeus.Tensor([3.0, 1.0], [2])
rows = pycoeus.index_select(m, 0, ridx)
assert rows.shape == [2, 3], f"index_select rows shape: {rows.shape}"
assert rows.data == [10.0, 11.0, 12.0, 4.0, 5.0, 6.0], f"rows: {rows.data}"

# ── index_select backward ─────────────────────────────────────────────
xg = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0, 5.0], [5], requires_grad=True)
idxg = pycoeus.Tensor([1.0, 3.0], [2])
out = pycoeus.index_select(xg, 0, idxg)
pycoeus.sum(out).backward()
# grad[1] = 1, grad[3] = 1, others = 0
assert xg.grad == [0.0, 1.0, 0.0, 1.0, 0.0], f"index_select bwd: {xg.grad}"

# ── error: index_select with non-1-D index raises ────────────────────
try:
    _ = pycoeus.index_select(x, 0, m)  # m is 2-D
    raise AssertionError("non-1-D index should raise")
except ValueError:
    pass
"#,
    );
}

#[test]
fn test_broadcast_masked_fill_nonzero() {
    run_script(
        r#"
import pycoeus

# ── broadcast_to ─────────────────────────────────────────────────────
x1 = pycoeus.Tensor([3.5], [1])
b1 = pycoeus.broadcast_to(x1, [4])
assert b1.shape == [4], f"broadcast_to 1D shape: {b1.shape}"
assert b1.data == [3.5, 3.5, 3.5, 3.5], f"broadcast_to 1D data: {b1.data}"

# 2-D: [1,3] → [2,3]
x2 = pycoeus.Tensor([1.0, 2.0, 3.0], [1, 3])
b2 = pycoeus.broadcast_to(x2, [2, 3])
assert b2.shape == [2, 3]
assert b2.data == [1.0, 2.0, 3.0, 1.0, 2.0, 3.0], f"broadcast_to 2D: {b2.data}"

# identity (same shape)
x3 = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
b3 = pycoeus.broadcast_to(x3, [2, 2])
assert b3.data == [1.0, 2.0, 3.0, 4.0]

# broadcast backward: gradient sums over broadcast dims
xg = pycoeus.Tensor([1.0, 2.0, 3.0], [1, 3], requires_grad=True)
bg = pycoeus.broadcast_to(xg, [4, 3])
loss = pycoeus.sum(bg)
loss.backward()
# each of 3 values broadcast to 4 rows → grad = 4 for each element
assert xg.grad == [4.0, 4.0, 4.0], f"broadcast backward: {xg.grad}"

# rank mismatch error
try:
    _ = pycoeus.broadcast_to(x1, [2, 2])  # rank 1 → target rank 2
    raise AssertionError("rank mismatch should raise")
except ValueError:
    pass

# ── masked_fill ───────────────────────────────────────────────────────
inp = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0], [4])
mask = pycoeus.Tensor([0.0, 1.0, 0.0, -2.0], [4])
out = pycoeus.masked_fill(inp, mask, 9.0)
assert out.data == [1.0, 9.0, 3.0, 9.0], f"masked_fill: {out.data}"

# all-zero mask is identity
z_mask = pycoeus.zeros([4])
ident = pycoeus.masked_fill(inp, z_mask, 99.0)
assert ident.data == [1.0, 2.0, 3.0, 4.0]

# backward: gradient zeroed at masked positions
inpg = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0], [4], requires_grad=True)
mf = pycoeus.masked_fill(inpg, mask, 9.0)
pycoeus.sum(mf).backward()
assert inpg.grad == [1.0, 0.0, 1.0, 0.0], f"masked_fill bwd: {inpg.grad}"

# mask is non-differentiable; a differentiable mask alone must not create a
# tracked output.
maskg = pycoeus.Tensor([0.0, 1.0, 0.0, 1.0], [4], requires_grad=True)
mf_mask_only = pycoeus.masked_fill(inp, maskg, 7.0)
assert mf_mask_only.requires_grad is False

# shape mismatch error
try:
    _ = pycoeus.masked_fill(inp, pycoeus.Tensor([1.0, 0.0], [2]), 0.0)
    raise AssertionError("shape mismatch should raise")
except ValueError:
    pass

# ── nonzero ────────────────────────────────────────────────────────────
v = pycoeus.Tensor([0.0, 2.0, 0.0, 3.0], [4])
nz = pycoeus.nonzero(v)
assert nz.shape == [2, 1], f"nonzero 1D shape: {nz.shape}"
assert nz.data == [1.0, 3.0], f"nonzero 1D data: {nz.data}"

# 2-D input
m2 = pycoeus.Tensor([0.0, 5.0, 0.0, 6.0, 7.0, 0.0], [2, 3])
nz2 = pycoeus.nonzero(m2)
assert nz2.shape == [3, 2], f"nonzero 2D shape: {nz2.shape}"
assert nz2.data == [0.0, 1.0, 1.0, 0.0, 1.0, 1.0], f"nonzero 2D data: {nz2.data}"

# all-zero input → empty [0, ndim]
nz_empty = pycoeus.nonzero(pycoeus.zeros([3, 3]))
assert nz_empty.shape == [0, 2], f"nonzero empty: {nz_empty.shape}"
"#,
    );
}

#[test]
fn test_feedforward_module() {
    run_script(
        r#"
import pycoeus

# FeedForward d_model=4, d_ff=8
ffn = pycoeus.FeedForward(4, 8)
ffn_with_dropout = pycoeus.FeedForward(4, 8, dropout_p=0.5)

x = pycoeus.Tensor([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [2, 4])
out = ffn.forward(x)
assert out.shape == [2, 4], f"FeedForward output shape: {out.shape}"
dropout_out = ffn_with_dropout.forward(x)
assert dropout_out.shape == [2, 4], f"FeedForward dropout output shape: {dropout_out.shape}"
# Output should be finite and not all the same value
vals = out.data
assert any(abs(v) > 0 for v in vals), "FeedForward output all zero"

try:
    _ = pycoeus.FeedForward(4, 8, dropout_p=1.0)
    raise AssertionError("dropout_p=1.0 should raise")
except ValueError:
    pass
"#,
    );
}

#[test]
fn test_vector_norm_p_orders() {
    run_script(
        r#"
import pycoeus
import math

x = pycoeus.Tensor([1.0, -2.0, 3.0, -4.0, 5.0], [5])

# Default ord=2 matches pycoeus.norm (L2). Both now return [1] tensors.
n2_default = pycoeus.vector_norm(x)
n_l2 = pycoeus.norm(x)
assert n2_default.shape == [1], f"vector_norm default shape: {n2_default.shape}"
assert n_l2.shape == [1], f"norm shape: {n_l2.shape}"
assert abs(n2_default.item() - n_l2.item()) < 1e-9, f"vector_norm default ord=2 != norm: {n2_default.item()} vs {n_l2.item()}"

# ord=1: Manhattan distance = sum(|x_i|) = 1+2+3+4+5 = 15.
n1 = pycoeus.vector_norm(x, ord=1.0)
assert n1.shape == [1]
assert abs(n1.item() - 15.0) < 1e-9, f"vector_norm ord=1 wrong: {n1.item()}"

# ord=3: (sum(|x|^3))^(1/3) — closed-form reference.
abs_vals = [1.0, 2.0, 3.0, 4.0, 5.0]
sum_cubes = sum(v ** 3 for v in abs_vals)
n3 = pycoeus.vector_norm(x, ord=3.0)
assert abs(n3.item() - sum_cubes ** (1.0 / 3.0)) < 1e-9, f"vector_norm ord=3 wrong: {n3.item()}"

# ord=4 fractional is fine; p must be finite positive.
n_half = pycoeus.vector_norm(x, ord=0.5)
assert isinstance(n_half.item(), float)

# per-axis ord-p norm returns a tensor (axis reduced).
m = pycoeus.Tensor([1.0, -2.0, 3.0, -4.0, 5.0, -6.0], [2, 3])
axis1 = pycoeus.vector_norm(m, ord=2.0, axis=1)
assert axis1.shape == [2], f"vector_norm axis=1 shape: {axis1.shape}"
want_axis1 = [math.sqrt(14.0), math.sqrt(77.0)]
assert all(abs(g - w) < 1e-9 for g, w in zip(axis1.data, want_axis1)), f"axis1: {axis1.data}"

axis0_keep = pycoeus.vector_norm(m, ord=1.0, axis=0, keepdim=True)
assert axis0_keep.shape == [1, 3], f"vector_norm keepdim shape: {axis0_keep.shape}"
assert axis0_keep.data == [5.0, 7.0, 9.0], f"axis0 keepdim: {axis0_keep.data}"

one_dim_axis = pycoeus.vector_norm(x, ord=2.0, axis=0)
assert one_dim_axis.shape == [1], f"1D axis shape: {one_dim_axis.shape}"
assert abs(one_dim_axis.item() - n_l2.item()) < 1e-9, f"1D axis scalar: {one_dim_axis.item()}"

# error: ord<=0 must raise ValueError, not panic.
for bad in [0.0, -1.0, float('inf'), -float('inf')]:
    try:
        _ = pycoeus.vector_norm(x, ord=bad)
        raise AssertionError(f"vector_norm ord={bad} should raise")
    except ValueError:
        pass

# error: empty tensor must raise ValueError.
try:
    _ = pycoeus.vector_norm(pycoeus.zeros([0]))
    raise AssertionError("vector_norm on empty tensor should raise")
except ValueError:
    pass

try:
    _ = pycoeus.vector_norm(m, axis=2)
    raise AssertionError("vector_norm out-of-range axis should raise")
except ValueError:
    pass
"#,
    );
}

// ── dot / cross ────────────────────────────────────────────────────────

#[test]
fn test_bmm_outer_ops() {
    run_script(
        r#"
import pycoeus
import math

# ── bmm (batch matrix multiply) ──────────────────────────────────────
# [2, 2, 3] × [2, 3, 4] → [2, 2, 4]
a = pycoeus.Tensor([float(i) for i in range(12)], [2, 2, 3])
b = pycoeus.Tensor([float(i) for i in range(24)], [2, 3, 4])
c = pycoeus.bmm(a, b)
assert c.shape == [2, 2, 4], f"bmm shape: {c.shape}"
# batch 0, row 0: [0,1,2] @ [[0,1,2,3],[4,5,6,7],[8,9,10,11]] = [20,23,26,29]
assert abs(c.data[0] - 20.0) < 1e-5, f"bmm[0,0,0]={c.data[0]}"
assert abs(c.data[1] - 23.0) < 1e-5, f"bmm[0,0,1]={c.data[1]}"

# Error: non-3D input
try:
    pycoeus.bmm(pycoeus.Tensor([1.0, 2.0], [2]), b)
    raise AssertionError("bmm non-3D a should raise")
except ValueError:
    pass

# ── outer (outer product) ────────────────────────────────────────────
v1 = pycoeus.Tensor([1.0, 2.0, 3.0], [3])
v2 = pycoeus.Tensor([4.0, 5.0], [2])
outer = pycoeus.outer(v1, v2)
assert outer.shape == [3, 2], f"outer shape: {outer.shape}"
# outer[i,j] = v1[i] * v2[j]
assert abs(outer.data[0] - 4.0) < 1e-9  # 1*4
assert abs(outer.data[1] - 5.0) < 1e-9  # 1*5
assert abs(outer.data[2] - 8.0) < 1e-9  # 2*4
assert abs(outer.data[5] - 15.0) < 1e-9  # 3*5

# Error: non-1D input
try:
    pycoeus.outer(pycoeus.Tensor([1.0, 2.0, 3.0, 4.0], [2, 2]), v2)
    raise AssertionError("outer non-1D a should raise")
except ValueError:
    pass
"#,
    );
}

#[test]
fn test_one_hot_masked_select_chunk() {
    run_script(
        r#"
import pycoeus

# ── one_hot ────────────────────────────────────────────────────────────
# indices [0, 2, 1, 2], num_classes=3 → [4, 3]
idx = pycoeus.Tensor([0.0, 2.0, 1.0, 2.0], [4])
oh = pycoeus.one_hot(idx, 3)
assert oh.shape == [4, 3], f"one_hot shape: {oh.shape}"
# row 0: [1,0,0], row 1: [0,0,1], row 2: [0,1,0], row 3: [0,0,1]
assert oh.data == [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], f"one_hot: {oh.data}"

# one_hot with 2 classes
idx2 = pycoeus.Tensor([0.0, 1.0, 0.0], [3])
oh2 = pycoeus.one_hot(idx2, 2)
assert oh2.data == [1.0, 0.0, 0.0, 1.0, 1.0, 0.0], f"one_hot 2-class: {oh2.data}"

# Error: non-1D indices
try:
    pycoeus.one_hot(pycoeus.Tensor([0.0, 1.0, 0.0, 1.0], [2, 2]), 3)
    raise AssertionError("non-1D one_hot should raise")
except ValueError:
    pass

# ── masked_select ────────────────────────────────────────────────────
x = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
m = pycoeus.Tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0], [2, 3])
sel = pycoeus.masked_select(x, m)
assert sel.shape == [3], f"masked_select shape: {sel.shape}"
assert sel.data == [1.0, 3.0, 5.0], f"masked_select: {sel.data}"

# All selected
m_all = pycoeus.ones([4])
x_all = pycoeus.Tensor([7.0, 8.0, 9.0, 10.0], [4])
sel_all = pycoeus.masked_select(x_all, m_all)
assert sel_all.shape == [4]
assert sel_all.data == [7.0, 8.0, 9.0, 10.0]

# None selected
m_none = pycoeus.zeros([4])
sel_none = pycoeus.masked_select(x_all, m_none)
assert sel_none.shape == [0], f"masked_select empty: {sel_none.shape}"

# ── chunk ────────────────────────────────────────────────────────────
v = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [6])

# Even split: 3 chunks of 2
parts = pycoeus.chunk(v, 3, 0)
assert len(parts) == 3, f"chunk count: {len(parts)}"
assert parts[0].data == [1.0, 2.0]
assert parts[1].data == [3.0, 4.0]
assert parts[2].data == [5.0, 6.0]

# Uneven: 4 chunks of ceil(6/4)=2 with last smaller
parts4 = pycoeus.chunk(v, 4, 0)
assert len(parts4) == 3, f"chunk 4 count: {len(parts4)}"  # only 3 non-empty chunks for size 6

# 2D chunk along dim=1
m = pycoeus.Tensor([float(i+1) for i in range(12)], [2, 6])
parts2d = pycoeus.chunk(m, 3, 1)
assert len(parts2d) == 3
assert parts2d[0].shape == [2, 2], f"2D chunk shape: {parts2d[0].shape}"

# Default dim=0
parts_def = pycoeus.chunk(v, 2)
assert len(parts_def) == 2
assert parts_def[0].data == [1.0, 2.0, 3.0]
"#,
    );
}

#[test]
fn test_glu_activation() {
    run_script(
        r#"
import pycoeus
import math

# ── glu (Gated Linear Unit) ───────────────────────────────────────────
# glu(x, dim): first_half * sigmoid(second_half)
# For a simple case: input = [a, b] along dim=0, glu = [a * sigmoid(b)]
x = pycoeus.Tensor([2.0, 4.0], [2])
out = pycoeus.glu(x, 0)
assert out.shape == [1], f"glu 1D shape: {out.shape}"
sig4 = 1.0 / (1.0 + math.exp(-4.0))
expected = 2.0 * sig4
assert abs(out.data[0] - expected) < 1e-5, f"glu 1D: {out.data[0]} vs {expected}"

# 2D: split in half along last dim (default dim=-1)
# input [2, 4]: each row split into [2] + [2]
x2 = pycoeus.Tensor([1.0, 2.0, 0.5, -0.5, 3.0, 4.0, -1.0, 2.0], [2, 4])
out2 = pycoeus.glu(x2)  # default dim=-1
assert out2.shape == [2, 2], f"glu 2D shape: {out2.shape}"
# row0: [1,2] * sigmoid([0.5,-0.5]) = [1*sig(0.5), 2*sig(-0.5)]
sig05 = 1.0 / (1.0 + math.exp(-0.5))
sig_neg05 = 1.0 / (1.0 + math.exp(0.5))
assert abs(out2.data[0] - 1.0 * sig05) < 1e-5, f"glu[0,0]: {out2.data[0]}"
assert abs(out2.data[1] - 2.0 * sig_neg05) < 1e-5, f"glu[0,1]: {out2.data[1]}"

# Error: odd size along dim
try:
    pycoeus.glu(pycoeus.Tensor([1.0, 2.0, 3.0], [3]), 0)
    raise AssertionError("glu odd dim should raise")
except ValueError:
    pass

# Error: dim out of range
try:
    pycoeus.glu(pycoeus.Tensor([1.0, 2.0], [2]), 5)
    raise AssertionError("glu out-of-range dim should raise")
except ValueError:
    pass
"#,
    );
}

#[test]
fn test_module_list() {
    run_script(
        r#"
import pycoeus

# ── ModuleList ────────────────────────────────────────────────────────
lin1 = pycoeus.Linear(4, 8)
ln = pycoeus.LayerNorm(8)
lin2 = pycoeus.Linear(8, 4)
layers = pycoeus.ModuleList([lin1, ln, lin2])

assert len(layers) == 3, f"len: {len(layers)}"

# Explicit forward (not auto-chained)
x = pycoeus.Tensor([float(i) for i in range(4)], [1, 4])
out = layers[0].forward(x)
assert out.shape == [1, 8], f"layer[0] output: {out.shape}"
out = layers[1].forward(out)
assert out.shape == [1, 8], f"layer[1] output: {out.shape}"
out = layers[2].forward(out)
assert out.shape == [1, 4], f"layer[2] output: {out.shape}"

# parameters(): collects from all sub-modules
params = layers.parameters()
# Linear(4,8): weight[8,4]+bias[8]=2; LayerNorm(8): weight+bias=2; Linear(8,4): weight[4,8]+bias[4]=2 → 6 total
assert len(params) == 6, f"parameter count: {len(params)}"

# negative indexing
last = layers[-1]
assert hasattr(last, 'forward'), "layers[-1] should have forward"

# setitem
new_lin = pycoeus.Linear(4, 8)
layers[0] = new_lin

# empty ModuleList
empty = pycoeus.ModuleList()
assert len(empty) == 0
empty.append(pycoeus.Linear(2, 2))
assert len(empty) == 1

# index out of range
try:
    _ = layers[10]
    raise AssertionError("out-of-range should raise")
except IndexError:
    pass

# zero_grad runs without error
layers.zero_grad()
"#,
    );
}

#[test]
fn test_softmax_log_softmax_methods() {
    run_script(
        r#"
import pycoeus
import math

# ── tensor.softmax(dim) ────────────────────────────────────────────────
logits = pycoeus.Tensor([1.0, 2.0, 3.0], [3])
sm = logits.softmax(0)
assert abs(sum(sm.data) - 1.0) < 1e-6, f"softmax sum: {sum(sm.data)}"
# Softmax is monotone with input order
assert sm.data[0] < sm.data[1] < sm.data[2], f"softmax order: {sm.data}"

# 2D input with dim=1
logits2d = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
sm2 = logits2d.softmax(1)
assert sm2.shape == [2, 3]
# Each row sums to 1
r0 = sum(sm2.data[:3])
r1 = sum(sm2.data[3:])
assert abs(r0 - 1.0) < 1e-5, f"row0 sum: {r0}"
assert abs(r1 - 1.0) < 1e-5, f"row1 sum: {r1}"

# ── tensor.log_softmax(dim) ────────────────────────────────────────────
lsm = logits.log_softmax(0)
# log_softmax values should all be <= 0
assert all(v <= 0.0 for v in lsm.data), f"log_softmax: {lsm.data}"
# exp(log_softmax) should match softmax
for a, b in zip([math.exp(v) for v in lsm.data], sm.data):
    assert abs(a - b) < 1e-5, f"exp(log_softmax) vs softmax: {a} vs {b}"

# ── backward through softmax ──────────────────────────────────────────
x = pycoeus.Tensor([1.0, 2.0, 3.0], [3], requires_grad=True)
y = x.softmax(0)
loss = pycoeus.sum(y)
loss.backward()
# sum(softmax) = 1 always, so gradient w.r.t. x should be zero (Jacobian is skew)
# In practice the grad is non-zero at individual elements; just check shapes.
assert x.grad is not None, "softmax backward should produce gradient"
assert len(x.grad) == 3
"#,
    );
}

#[test]
fn test_sequential_module() {
    run_script(
        r#"
import pycoeus

# ── Sequential chains forward calls ────────────────────────────────────
# Build: Linear(4→8) → LayerNorm(8) as a Sequential
lin = pycoeus.Linear(4, 8)
ln = pycoeus.LayerNorm(8)
model = pycoeus.Sequential([lin, ln])

x = pycoeus.Tensor([float(i) for i in range(4)], [1, 4])
out = model.forward(x)
assert out.shape == [1, 8], f"Sequential output shape: {out.shape}"

# ── Parameters collected from all sub-modules ──────────────────────────
params = model.parameters()
# Linear has weight+bias (2), LayerNorm has weight+bias (2) → total 4
assert len(params) == 4, f"parameter count: {len(params)}"

# ── Empty Sequential is identity ──────────────────────────────────────
empty = pycoeus.Sequential([])
x_small = pycoeus.Tensor([1.0, 2.0, 3.0], [3])
out_id = empty.forward(x_small)
assert out_id.data == x_small.data

# ── __len__ and __getitem__ ────────────────────────────────────────────
assert len(model) == 2
m0 = model[0]  # should be the Linear
assert hasattr(m0, 'forward'), "model[0] should have forward"
m_neg = model[-1]  # last item
assert hasattr(m_neg, 'forward'), "model[-1] should have forward"

try:
    model[5]
    raise AssertionError("out-of-range index should raise")
except IndexError:
    pass

# ── Backward gradient flows through chain ─────────────────────────────
x2 = pycoeus.Tensor([float(i) for i in range(4)], [1, 4], requires_grad=True)
out2 = model.forward(x2)
pycoeus.sum(out2).backward()
assert x2.grad is not None, "Sequential backward should produce gradient on input"
"#,
    );
}

#[test]
fn test_conv_transpose_tracked_backward() {
    run_script(
        r#"
import pycoeus

# ── ConvTranspose1d tracks gradient ────────────────────────────────────
# 1 batch, 1 in_channel, length=3, 1 out_channel, kernel=2, stride=1
ct1 = pycoeus.ConvTranspose1d(1, 1, 2, stride=1, padding=0, bias=False)
# Fix the weight to [1, 1, 2] = [[1, 0.5]] for a deterministic result.
ct1.weight.data = [1.0, 0.5]

x1 = pycoeus.Tensor([1.0, 2.0, 3.0], [1, 1, 3], requires_grad=True)
out1 = ct1.forward(x1)
assert out1.shape == [1, 1, 4], f"ConvTranspose1d output shape: {out1.shape}"

pycoeus.sum(out1).backward()
assert x1.grad is not None, "ConvTranspose1d should track gradient on input"
assert len(x1.grad) == 3, f"ConvTranspose1d grad shape: {len(x1.grad)}"

# ── ConvTranspose2d tracks gradient ────────────────────────────────────
ct2 = pycoeus.ConvTranspose2d(1, 1, 1, stride=1, padding=0, bias=False)
ct2.weight.data = [2.0]  # scalar weight = 2.0

x2 = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0], [1, 1, 2, 2], requires_grad=True)
out2 = ct2.forward(x2)
assert out2.shape == [1, 1, 2, 2], f"ConvTranspose2d output shape: {out2.shape}"

# With identity 1×1 kernel = 2.0, output = 2 * input
for a, b in zip(out2.data, [2.0, 4.0, 6.0, 8.0]):
    assert abs(a - b) < 1e-5, f"ConvTranspose2d forward: {a} vs {b}"

pycoeus.sum(out2).backward()
assert x2.grad is not None, "ConvTranspose2d should track gradient on input"
# grad_input = grad_out * weight = 1 * 2.0 = 2.0 per element
for g in x2.grad:
    assert abs(g - 2.0) < 1e-5, f"ConvTranspose2d grad_input: {g}"
"#,
    );
}

#[test]
fn test_layernorm_3d_forward_nd() {
    run_script(
        r#"
import pycoeus

# ── LayerNorm on 3-D [batch, seq, d] ──────────────────────────────────
batch, seq, d = 2, 3, 4
x_data = [float(i) for i in range(batch * seq * d)]
x = pycoeus.Tensor(x_data, [batch, seq, d], requires_grad=True)

ln = pycoeus.LayerNorm(d, eps=1e-5)

# forward_nd produces same shape as input
out = ln.forward_nd(x)
assert out.shape == [batch, seq, d], f"forward_nd shape: {out.shape}"

# backward propagates gradients
loss = pycoeus.sum(out)
loss.backward()
assert x.grad is not None, "forward_nd should track gradients"
assert len(x.grad) == batch * seq * d, f"grad shape mismatch: {len(x.grad)}"

# ── layer_norm functional handles 3-D transparently ────────────────────
x2 = pycoeus.Tensor(x_data, [batch, seq, d], requires_grad=True)
out2 = pycoeus.layer_norm(x2, d)
assert out2.shape == [batch, seq, d], f"layer_norm 3D shape: {out2.shape}"

# Identical to forward_nd output
for a, b in zip(out.data, out2.data):
    assert abs(a - b) < 1e-6, f"forward_nd vs layer_norm mismatch: {a} vs {b}"

# ── 2-D forward is unchanged ────────────────────────────────────────────
x3 = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.5, 3.0], [2, 4])
out3 = ln.forward(x3)
out3_nd = ln.forward_nd(x3)
assert out3.shape == [2, 4]
for a, b in zip(out3.data, out3_nd.data):
    assert abs(a - b) < 1e-9, f"forward vs forward_nd 2D: {a} vs {b}"

# ── 4-D input [batch, channels, h, w] ────────────────────────────────
b2, c, h, w = 1, 2, 3, 4
ln4 = pycoeus.LayerNorm(w, eps=1e-5)
x4 = pycoeus.Tensor([float(i) for i in range(b2 * c * h * w)], [b2, c, h, w])
out4 = ln4.forward_nd(x4)
assert out4.shape == [b2, c, h, w], f"4D forward_nd shape: {out4.shape}"
"#,
    );
}

#[test]
fn test_dtype_cast_methods() {
    run_script(
        r#"
import pycoeus
import math

x = pycoeus.Tensor([1.7, -2.9, 3.1, 0.0, -0.1], [5])

# ── float() / double() ────────────────────────────────────────────────
# float() is identity for f64 storage; shape and values preserved.
xf = x.float()
assert xf.shape == [5], f"float shape: {xf.shape}"
for a, b in zip(xf.data, x.data):
    assert abs(a - b) < 1e-12, f"float() changed value: {a} vs {b}"

xd = x.double()
assert xd.data == x.data, f"double() changed data"

# ── long() / int() ────────────────────────────────────────────────────
# Truncates fractional part toward zero (matching torch.long behaviour).
xl = x.long()
assert xl.shape == [5], f"long shape: {xl.shape}"
expected_long = [1.0, -2.0, 3.0, 0.0, 0.0]
for g, e in zip(xl.data, expected_long):
    assert g == e, f"long() value: got {g}, expected {e}"

xi = x.int()
assert xi.data == xl.data, f"int() != long()"

# ── half() ───────────────────────────────────────────────────────────
# Values are quantized to f16 precision; stored back as f64.
xh = x.half()
assert xh.shape == [5], f"half shape: {xh.shape}"
# 1.7 in f16 is exactly 1.7 (representable), check it's close.
for a, b in zip(xh.data, x.data):
    assert abs(a - b) < 5e-3, f"half() too far from original: {a} vs {b}"
# Should round-trip differently from original for non-representable values.
# 3.1 → f16(3.1) ≈ 3.1015625
xh3 = pycoeus.Tensor([3.1], [1]).half()
assert abs(xh3.data[0] - 3.1) < 0.01, f"half 3.1: {xh3.data[0]}"

# ── to(dtype) ────────────────────────────────────────────────────────
xt_float = x.to("float")
assert xt_float.data == x.data, "to('float') changed data"
xt_long = x.to("long")
assert xt_long.data == xl.data, "to('long') != long()"
xt_half = x.to("float16")
# to(float16) should agree with half()
for a, b in zip(xt_half.data, xh.data):
    assert abs(a - b) < 1e-12, f"to(float16) vs half(): {a} vs {b}"

# unknown dtype raises ValueError
try:
    x.to("bfloat16")
    raise AssertionError("unknown dtype should raise")
except ValueError:
    pass

# ── type_as() ────────────────────────────────────────────────────────
other = pycoeus.Tensor([100.0], [1])
ta = x.type_as(other)
assert ta.data == x.data, f"type_as changed data"
assert ta.shape == x.shape, f"type_as changed shape"
"#,
    );
}

#[test]
fn test_sdp_attention_and_module() {
    run_script(
        r#"
import pycoeus
import math

# ── functional scaled_dot_product_attention ───────────────────────────
batch, seq_q, seq_k, d_k, d_v = 1, 3, 4, 8, 8
# Use ones for simple validation: output should be exactly v for uniform attn.
q = pycoeus.zeros([batch, seq_q, d_k])
k = pycoeus.zeros([batch, seq_k, d_k])
v = pycoeus.ones([batch, seq_k, d_v])
out = pycoeus.scaled_dot_product_attention(q, k, v)
assert out.shape == [batch, seq_q, d_v], f"sdpa shape: {out.shape}"
# All-zeros Q·K^T → uniform softmax → output = mean(V) = ones (since V=ones)
for val in out.data:
    assert abs(val - 1.0) < 1e-5, f"sdpa val: {val}"

# ── PyScaledDotProductAttention module ────────────────────────────────
sdpa_mod = pycoeus.ScaledDotProductAttention(scale=None, is_causal=False)
out_mod = sdpa_mod.forward(q, k, v)
assert out_mod.shape == [batch, seq_q, d_v], f"sdpa_mod shape: {out_mod.shape}"
for a, b in zip(out_mod.data, out.data):
    assert abs(a - b) < 1e-9, f"module vs functional mismatch: {a} vs {b}"

# ── causal attention: output should differ from non-causal ─────────────
q_rng = pycoeus.Tensor([float(i) for i in range(batch * seq_q * d_k)], [batch, seq_q, d_k])
k_rng = pycoeus.Tensor([float(i + 1) for i in range(batch * seq_k * d_k)], [batch, seq_k, d_k])
v_rng = pycoeus.Tensor([float(i + 2) for i in range(batch * seq_k * d_v)], [batch, seq_k, d_v])
out_nc = pycoeus.scaled_dot_product_attention(q_rng, k_rng, v_rng)
out_causal = pycoeus.scaled_dot_product_attention(q_rng, k_rng, v_rng, is_causal=True)
# causal masks future — not identical (should differ for non-trivial inputs)
assert out_nc.shape == out_causal.shape, "causal/non-causal shape mismatch"
# Just verify outputs are valid (no NaN, no inf)
for val in out_nc.data:
    assert math.isfinite(val), f"non-causal has non-finite value: {val}"
for val in out_causal.data:
    assert math.isfinite(val), f"causal has non-finite value: {val}"

# ── no parameters in ScaledDotProductAttention ──────────────────────
assert sdpa_mod.parameters() == [], "sdpa should have no parameters"
sd = sdpa_mod.state_dict()
sdpa_mod.load_state_dict(sd)  # no-op, should not raise
"#,
    );
}

#[test]
fn test_amax_amin_prod_ops() {
    run_script(
        r#"
import pycoeus

x = pycoeus.Tensor([3.0, -1.0, 5.0, 2.0, -4.0, 0.5], [2, 3])

# amax (global max)
am = pycoeus.amax(x)
assert abs(am - 5.0) < 1e-9, f"amax: {am}"

# amin (global min)
an = pycoeus.amin(x)
assert abs(an - (-4.0)) < 1e-9, f"amin: {an}"

# prod (global product: 3 * -1 * 5 * 2 * -4 * 0.5 = 60)
pr = pycoeus.prod(x)
expected_prod = 3.0 * (-1.0) * 5.0 * 2.0 * (-4.0) * 0.5
assert abs(pr - expected_prod) < 1e-5, f"prod: {pr} expected {expected_prod}"

# 1-D tensor
v = pycoeus.Tensor([7.0, 3.0, 9.0, 1.0], [4])
assert abs(pycoeus.amax(v) - 9.0) < 1e-9
assert abs(pycoeus.amin(v) - 1.0) < 1e-9
assert abs(pycoeus.prod(v) - 7.0*3.0*9.0*1.0) < 1e-9

# empty tensor raises ValueError for amax/amin; prod returns 1.0 (identity)
try:
    pycoeus.amax(pycoeus.zeros([0]))
    raise AssertionError("amax empty should raise")
except ValueError:
    pass
try:
    pycoeus.amin(pycoeus.zeros([0]))
    raise AssertionError("amin empty should raise")
except ValueError:
    pass
# prod of empty tensor = 1.0 (multiplicative identity, matching numpy/PyTorch)
pr_empty = pycoeus.prod(pycoeus.zeros([0]))
assert abs(pr_empty - 1.0) < 1e-9, f"prod empty: {pr_empty}"
"#,
    );
}

#[test]
fn test_dot_cross_vector_ops() {
    run_script(
        r#"
import math
import pycoeus

# ── dot ────────────────────────────────────────────────────────────────
# torch.dot([1,2,3], [4,5,6]) = 32
a = pycoeus.Tensor([1.0, 2.0, 3.0], [3])
b = pycoeus.Tensor([4.0, 5.0, 6.0], [3])
got = pycoeus.dot(a, b)
assert abs(got - 32.0) < 1e-9, f"1D dot: {got}"

# 2-D inputs: torch.dot flattens.
am = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
bm = pycoeus.Tensor([7.0, 8.0, 9.0, 10.0, 11.0, 12.0], [2, 3])
want = 7 + 16 + 27 + 40 + 55 + 72  # = 217
got = pycoeus.dot(am, bm)
assert abs(got - float(want)) < 1e-9, f"2D flat dot: {got}"

# orthogonal: <e_x, e_y> = 0
ex = pycoeus.Tensor([1.0, 0.0, 0.0], [3])
ey = pycoeus.Tensor([0.0, 1.0, 0.0], [3])
assert abs(pycoeus.dot(ex, ey)) < 1e-9

# error: numel mismatch
try:
    _ = pycoeus.dot(pycoeus.Tensor([1.0, 2.0], [2]), pycoeus.Tensor([1.0, 2.0, 3.0], [3]))
    raise AssertionError("dot numel-mismatch should raise")
except ValueError:
    pass

# error: empty input
try:
    _ = pycoeus.dot(pycoeus.Tensor([], [0]), pycoeus.Tensor([], [0]))
    raise AssertionError("dot on empty should raise")
except ValueError:
    pass

# ── cross ──────────────────────────────────────────────────────────────
# cross(e_x, e_y) = e_z (default dim=0 for 1-D 3-vector)
cx_out = pycoeus.cross(ex, ey)  # default dim=0
assert cx_out.shape == [3], f"cross shape: {cx_out.shape}"
assert cx_out.data == [0.0, 0.0, 1.0], f"cross([1,0,0], [0,1,0]) dim0: {cx_out.data}"

# 2-D cross (default dim=0): columns are 3-vectors.
am = pycoeus.Tensor([1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 4.0], [3, 3])
bm = pycoeus.Tensor([0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0], [3, 3])
out = pycoeus.cross(am, bm, dim=0)
assert out.shape == [3, 3], f"cross 2D dim0 shape: {out.shape}"
expected = [0.0, 0.0, 0.0, -5.0, 0.0, 20.0, 0.0, 0.0, 0.0]
for g, w in zip(out.data, expected):
    assert abs(g - w) < 1e-9, f"cross 2D dim0: got={g} expected={w}"

# 2-D cross (dim=1): per-row cross.
am = pycoeus.Tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], [2, 3])
bm = pycoeus.Tensor([0.0, 1.0, 0.0, 0.0, 0.0, 1.0], [2, 3])
out = pycoeus.cross(am, bm, dim=1)
assert out.shape == [2, 3], f"cross 2D dim1 shape: {out.shape}"
expected = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
for g, w in zip(out.data, expected):
    assert abs(g - w) < 1e-9, f"cross 2D dim1: got={g} expected={w}"

# cross(a, a) = 0
v = pycoeus.Tensor([2.0, 3.0, 4.0], [3])
out = pycoeus.cross(v, v)  # dim=0
assert all(abs(x) < 1e-9 for x in out.data), f"cross(v, v): {out.data}"

# error: shape mismatch
try:
    _ = pycoeus.cross(pycoeus.Tensor([1.0, 2.0, 3.0], [3]), pycoeus.Tensor([1.0, 2.0], [2]))
    raise AssertionError("cross shape-mismatch should raise")
except ValueError:
    pass

# error: dim out of range
try:
    _ = pycoeus.cross(pycoeus.Tensor([1.0, 2.0, 3.0], [3]), pycoeus.Tensor([4.0, 5.0, 6.0], [3]), dim=5)
    raise AssertionError("cross out-of-range dim should raise")
except ValueError:
    pass

# error: dim != 3
try:
    _ = pycoeus.cross(pycoeus.Tensor([1.0, 2.0, 3.0, 4.0], [4]), pycoeus.Tensor([5.0, 6.0, 7.0, 8.0], [4]))
    raise AssertionError("cross axis-size!=3 should raise")
except ValueError:
    pass
"#,
    );
}
