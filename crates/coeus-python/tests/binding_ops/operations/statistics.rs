//! Norm, validation, and statistical binding contracts.

use super::support::run_script;

#[test]
fn test_normalize_closeness_nan_and_grad_clipping() {
    run_script(
        r#"
import pycoeus

# module reductions default to keepdim=False, with explicit keepdim support.
x = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
row_sum = pycoeus.sum_axis(x, 1)
row_mean_keep = pycoeus.mean_axis(x, 1, keepdim=True)
assert row_sum.shape == [2], f"sum_axis squeeze shape: {row_sum.shape}"
assert row_sum.data == [3.0, 7.0], f"sum_axis data: {row_sum.data}"
assert row_mean_keep.shape == [2, 1], f"mean_axis keepdim shape: {row_mean_keep.shape}"
assert row_mean_keep.data == [1.5, 3.5], f"mean_axis keepdim data: {row_mean_keep.data}"

# normalize uses L2 norm per row and preserves zero rows through eps clamp.
n = pycoeus.normalize(pycoeus.Tensor([3.0, 4.0, 0.0, 0.0], [2, 2]), dim=1)
expected = [0.6, 0.8, 0.0, 0.0]
assert n.shape == [2, 2], f"normalize shape: {n.shape}"
for got, want in zip(n.data, expected):
    assert abs(got - want) < 1e-12, f"normalize data: {n.data}"
try:
    pycoeus.normalize(x, p=0.0)
    raise AssertionError("normalize should reject non-positive p")
except ValueError:
    pass

# isclose/allclose expose PyTorch-style value tolerance semantics.
a = pycoeus.Tensor([1.0, 2.0, 3.0], [3])
b = pycoeus.Tensor([1.0 + 1e-6, 2.0, 3.1], [3])
close = pycoeus.isclose(a, b, rtol=1e-5, atol=1e-8)
assert close.data == [1.0, 1.0, 0.0], f"isclose data: {close.data}"
assert pycoeus.allclose(a, b, rtol=1e-5, atol=1e-8) is False
assert pycoeus.allclose(a, a) is True

# nan_to_num replaces each special value according to the supplied contract.
special = pycoeus.Tensor([float("nan"), float("inf"), -float("inf"), 2.0], [4])
finite = pycoeus.nan_to_num(special, nan=-1.0, posinf=9.0, neginf=-9.0)
assert finite.data == [-1.0, 9.0, -9.0, 2.0], f"nan_to_num data: {finite.data}"

# gradient norm clipping returns the pre-clip norm and rescales gradients.
p = pycoeus.Tensor([3.0, 4.0], [2], requires_grad=True)
pycoeus.sum(p * p).backward()
norm = pycoeus.clip_grad_norm_([p], 5.0)
assert abs(norm - 10.0) < 1e-12, f"pre-clip norm: {norm}"
assert abs(p.grad[0] - 3.0) < 1e-6 and abs(p.grad[1] - 4.0) < 1e-6, f"clip norm grad: {p.grad}"

q = pycoeus.Tensor([2.0, -4.0], [2], requires_grad=True)
pycoeus.sum(q * q).backward()
pycoeus.clip_grad_value_([q], 2.5)
assert q.grad == [2.5, -2.5], f"clip value grad: {q.grad}"

shown = repr(pycoeus.Tensor([1.0, 2.0], [2]))
assert shown == "Tensor([1.0, 2.0], shape=[2])", f"repr: {shown}"
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

#[test]
fn test_matrix_norm_fro() {
    run_script(
        r#"
import math
import pycoeus

# ── torch.linalg.matrix_norm(input, ord='fro') parity ──────────────────
# Reference: torch.linalg.matrix_norm(reshape(arange(9), (3,3))) =
# sqrt(0+1+4+9+16+25+36+49+64) = sqrt(204) ≈ 14.2829.

# 2-D input → plain Python float.
flat = pycoeus.matrix_norm(pycoeus.Tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [3, 3]))
assert isinstance(flat, float), f"2-D should return float, got {type(flat)}"
assert abs(flat - math.sqrt(204.0)) < 1e-9, f"2-D frobenius: {flat}"

# 3x3 identity matrix → sqrt(3).
id_flat = pycoeus.matrix_norm(
    pycoeus.Tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], [3, 3]),
)
assert abs(id_flat - math.sqrt(3.0)) < 1e-9, f"identity frobenius: {id_flat}"

# Non-square (3x2) reduction: sqrt(1+4+9+16+25+36) = sqrt(91).
nsq = pycoeus.matrix_norm(
    pycoeus.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2]),
)
assert abs(nsq - math.sqrt(91.0)) < 1e-9, f"3x2 frobenius: {nsq}"

# 3-D batched input with two stacked copies — returns a [2] PyTensor
# carrying one Frobenius norm per batch slot.
stacked = pycoeus.Tensor(
    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
     0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    [2, 3, 3],
)
out = pycoeus.matrix_norm(stacked)
assert out.shape == [2], f"3-D batched shape: {out.shape}"
for got in out.data:
    assert abs(got - math.sqrt(204.0)) < 1e-9, f"3-D batched entry: {got}"

# 4-D batched input — returns a [2, 2] PyTensor of Frobenius norms for
# the leading 2x2 batch of identity matrices.
batch4 = pycoeus.Tensor(
    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
     1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
     1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
     1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    [2, 2, 3, 3],
)
out4 = pycoeus.matrix_norm(batch4)
assert out4.shape == [2, 2], f"4-D batched shape: {out4.shape}"
for got in out4.data:
    assert abs(got - math.sqrt(3.0)) < 1e-9, f"4-D batched entry: {got}"

# 1-D input → ValueError.
try:
    pycoeus.matrix_norm(pycoeus.Tensor([1.0, 2.0, 3.0], [3]))
    raise AssertionError("matrix_norm on 1-D should raise")
except ValueError:
    pass

# ord != 'fro' → ValueError (only Frobenius is currently shipped).
try:
    pycoeus.matrix_norm(pycoeus.Tensor([1.0, 2.0], [1, 2]), ord='nuc')
    raise AssertionError("matrix_norm with ord='nuc' should raise")
except ValueError:
    pass

# ord default is 'fro' (omit keyword → 2-D returns float).
default = pycoeus.matrix_norm(pycoeus.Tensor([3.0, 4.0], [1, 2]))
assert isinstance(default, float), f"default ord: {type(default)}"
assert abs(default - 5.0) < 1e-9, f"default ord frobenius: {default}"
"#,
    );
}
