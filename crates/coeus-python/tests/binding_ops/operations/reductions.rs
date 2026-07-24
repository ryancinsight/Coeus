//! Reduction and order-statistics binding contracts.

use super::support::run_script;

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
assert s.shape == [1], f"std scalar shape: {s.shape}"
assert abs(s.item() - expected_std2) < 1e-9, f"std wrong: {s.item()} vs {expected_std2}"

# var (unbiased)
v = pycoeus.var(x)
assert v.shape == [1], f"var scalar shape: {v.shape}"
assert abs(v.item() - 20.0 / 3.0) < 1e-9, f"var wrong: {v.item()}"

# var (biased, N divisor)
v_biased = pycoeus.var(x, unbiased=False)
assert abs(v_biased.item() - 5.0) < 1e-9, f"var biased wrong: {v_biased.item()}"

# std (biased, N divisor)
s_biased = pycoeus.std(x, unbiased=False)
assert abs(s_biased.item() - math.sqrt(5.0)) < 1e-9, f"std biased wrong: {s_biased.item()}"

# var_mean / std_mean without axis return scalar pairs.
vm_var, vm_mean = pycoeus.var_mean(x)
assert vm_var.shape == [1], f"var_mean scalar variance shape: {vm_var.shape}"
assert vm_mean.shape == [1], f"var_mean scalar mean shape: {vm_mean.shape}"
assert abs(vm_var.item() - v.item()) < 1e-9, f"var_mean scalar variance wrong: {vm_var.item()}"
assert abs(vm_mean.item() - 5.0) < 1e-9, f"var_mean scalar mean wrong: {vm_mean.item()}"
sm_std, sm_mean = pycoeus.std_mean(x, unbiased=False)
assert sm_std.shape == [1], f"std_mean scalar std shape: {sm_std.shape}"
assert sm_mean.shape == [1], f"std_mean scalar mean shape: {sm_mean.shape}"
assert abs(sm_std.item() - s_biased.item()) < 1e-9, f"std_mean scalar std wrong: {sm_std.item()}"
assert abs(sm_mean.item() - 5.0) < 1e-9, f"std_mean scalar mean wrong: {sm_mean.item()}"

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

vm_axis1, mean_axis1 = pycoeus.var_mean(y, axis=1)
assert vm_axis1.shape == [2], f"var_mean axis1 variance shape: {vm_axis1.shape}"
assert mean_axis1.shape == [2], f"var_mean axis1 mean shape: {mean_axis1.shape}"
assert vm_axis1.data == [1.0, 1.0], f"var_mean axis1 variance wrong: {vm_axis1.data}"
assert mean_axis1.data == [2.0, 5.0], f"var_mean axis1 mean wrong: {mean_axis1.data}"
sm_axis1_keep, sm_mean_axis1_keep = pycoeus.std_mean(y, axis=1, keepdim=True)
assert sm_axis1_keep.shape == [2, 1], f"std_mean keepdim std shape: {sm_axis1_keep.shape}"
assert sm_mean_axis1_keep.shape == [2, 1], f"std_mean keepdim mean shape: {sm_mean_axis1_keep.shape}"
assert sm_axis1_keep.data == [1.0, 1.0], f"std_mean keepdim std wrong: {sm_axis1_keep.data}"
assert sm_mean_axis1_keep.data == [2.0, 5.0], f"std_mean keepdim mean wrong: {sm_mean_axis1_keep.data}"

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
try:
    _ = pycoeus.var_mean(y, axis=2)
    raise AssertionError("var_mean out-of-range axis should raise")
except ValueError:
    pass
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
pr = pycoeus.prod(x).item()
expected_prod = 3.0 * (-1.0) * 5.0 * 2.0 * (-4.0) * 0.5
assert abs(pr - expected_prod) < 1e-5, f"prod: {pr} expected {expected_prod}"

# 1-D tensor
v = pycoeus.Tensor([7.0, 3.0, 9.0, 1.0], [4])
assert abs(pycoeus.amax(v) - 9.0) < 1e-9
assert abs(pycoeus.amin(v) - 1.0) < 1e-9
assert abs(pycoeus.prod(v).item() - 7.0*3.0*9.0*1.0) < 1e-9

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
pr_empty = pycoeus.prod(pycoeus.zeros([0])).item()
assert abs(pr_empty - 1.0) < 1e-9, f"prod empty: {pr_empty}"
"#,
    );
}
