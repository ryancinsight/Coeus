//! Indexing, masking, and mutation binding contracts.

use super::support::run_script;

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
fn test_index_put_op() {
    run_script(
        r#"
import pycoeus

# ── index_put: replace ────────────────────────────────────────────────
x = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0], [4])
idx = pycoeus.Tensor([1.0, 3.0], [2])
vals = pycoeus.Tensor([10.0, 20.0], [2])
out = pycoeus.index_put(x, idx, vals)
assert out.data == [1.0, 10.0, 3.0, 20.0], f"index_put replace: {out.data}"

# ── index_put: accumulate ──────────────────────────────────────────────
out_acc = pycoeus.index_put(x, idx, vals, accumulate=True)
# idx=[1,3], vals=[10,20], accumulate: x[1]+=10=12, x[3]+=20=24
assert abs(out_acc.data[1] - 12.0) < 1e-5, f"index_put acc[1]={out_acc.data[1]}"
assert abs(out_acc.data[3] - 24.0) < 1e-5, f"index_put acc[3]={out_acc.data[3]}"
# Non-indexed unchanged
assert out_acc.data[0] == 1.0
assert out_acc.data[2] == 3.0

# ── 2D tensor: replace rows ────────────────────────────────────────────
m = pycoeus.Tensor([float(i+1) for i in range(9)], [3, 3])
row_idx = pycoeus.Tensor([2.0], [1])
new_row = pycoeus.Tensor([100.0, 200.0, 300.0], [1, 3])
out2d = pycoeus.index_put(m, row_idx, new_row)
assert out2d.data[6:9] == [100.0, 200.0, 300.0], f"index_put 2D row: {out2d.data[6:9]}"
assert out2d.data[0] == 1.0  # Unchanged row

# ── Error: non-1D indices ──────────────────────────────────────────────
try:
    pycoeus.index_put(x, m, vals)
    raise AssertionError("non-1D indices should raise")
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
fn test_tensor_setitem() {
    run_script(
        r#"
import pycoeus

# ── Integer index assignment ───────────────────────────────────────────
t = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2])

# Assign scalar to a row
t[0] = 10.0
assert t.data[0] == 10.0, f"setitem scalar row[0][0]: {t.data[0]}"
assert t.data[1] == 10.0, f"setitem scalar row[0][1]: {t.data[1]}"
# Other rows unchanged
assert t.data[2] == 3.0, f"setitem did not change row1: {t.data[2]}"

# Assign tensor to a row
new_row = pycoeus.Tensor([99.0, 88.0], [2])
t[2] = new_row
assert t.data[4] == 99.0, f"setitem tensor row[2][0]: {t.data[4]}"
assert t.data[5] == 88.0, f"setitem tensor row[2][1]: {t.data[5]}"

# Negative index
t[-1] = 77.0
assert t.data[4] == 77.0, f"setitem negative index: {t.data[4]}"

# 1-D tensor
v = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0], [4])
v[2] = 99.0
assert v.data[2] == 99.0, f"1D setitem: {v.data[2]}"

# Out-of-range raises IndexError
try:
    t[100] = 0.0
    raise AssertionError("out-of-range should raise")
except IndexError:
    pass

# Non-int index raises TypeError
try:
    t[1:3] = 5.0
    raise AssertionError("slice index should raise TypeError")
except TypeError:
    pass

# Shape mismatch raises ValueError
try:
    wrong_row = pycoeus.Tensor([1.0, 2.0, 3.0], [3])
    t[0] = wrong_row
    raise AssertionError("shape mismatch should raise ValueError")
except ValueError:
    pass
"#,
    );
}
