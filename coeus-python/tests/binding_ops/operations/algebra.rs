//! Linear-algebra and vector-operation binding contracts.

use super::support::run_script;

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

c = pycoeus.Tensor([9.0, 10.0, 11.0, 12.0], [2, 2])
chain = pycoeus.einsum("ij,jk,kl->il", [a, b, c])
assert chain.shape == [2, 2], f"einsum3 chain shape wrong: {chain.shape}"
assert chain.data == [1226.0, 1348.0, 2945.0, 3238.0], f"einsum3 chain wrong: {chain.data}"
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
assert got.shape == [1], f"1D dot shape: {got.shape}"
assert abs(got.item() - 32.0) < 1e-9, f"1D dot: {got.item()}"

# 2-D inputs: torch.dot flattens.
am = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
bm = pycoeus.Tensor([7.0, 8.0, 9.0, 10.0, 11.0, 12.0], [2, 3])
want = 7 + 16 + 27 + 40 + 55 + 72  # = 217
got = pycoeus.dot(am, bm)
assert abs(got.item() - float(want)) < 1e-9, f"2D flat dot: {got.item()}"

# orthogonal: <e_x, e_y> = 0
ex = pycoeus.Tensor([1.0, 0.0, 0.0], [3])
ey = pycoeus.Tensor([0.0, 1.0, 0.0], [3])
assert abs(pycoeus.dot(ex, ey).item()) < 1e-9

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
