//! Shape, layout, and view binding contracts.

use super::support::run_script;

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
