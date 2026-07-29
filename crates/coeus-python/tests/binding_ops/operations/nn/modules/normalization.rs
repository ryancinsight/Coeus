//! Normalization module binding contracts.

use super::super::super::support::run_script;

#[test]
fn test_module_contract_failure_is_value_error() {
    run_script(
        r#"
import pycoeus

layer = pycoeus.LayerNorm(4)
rank_one = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0], [4])

try:
    layer.forward(rank_one)
except ValueError as error:
    message = str(error)
    assert "LayerNorm" in message
    assert "expected input rank 2" in message
    assert "got 1" in message
else:
    raise AssertionError("rank-one LayerNorm input must raise ValueError")
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
fn test_batchnorm_eval_mode() {
    run_script(
        r#"
import pycoeus

# During training, BN normalizes using batch stats.
# During eval, BN normalizes using stored running stats and must not mutate them.

def assert_eval_uses_running_stats_and_does_not_mutate(
    bn, x, running_mean, running_var, c0_prefix_len, shape_expected
):
    # Prime with one training step so forward path is exercised.
    out_train = bn.forward(x)
    assert out_train.shape == shape_expected, f"train shape: {out_train.shape}"

    bn.running_mean.data = list(running_mean)
    bn.running_var.data = list(running_var)
    before_mean = list(bn.running_mean.data)
    before_var = list(bn.running_var.data)

    out_eval = bn.eval_forward(x)
    assert out_eval.shape == shape_expected, f"eval shape: {out_eval.shape}"

    # Running stats must not change during eval_forward.
    assert bn.running_mean.data == before_mean, "eval_forward must not mutate running_mean"
    assert bn.running_var.data == before_var, "eval_forward must not mutate running_var"

    # Channel 0 values occupy the first contiguous block in [N, C, ...] layout.
    for v, xv in zip(out_eval.data[:c0_prefix_len], x.data[:c0_prefix_len]):
        expected = (xv - running_mean[0]) / (running_var[0] + 1e-5) ** 0.5
        assert abs(v - expected) < 1e-3, f"eval C0: got {v} expected {expected:.4f}"


# ── BatchNorm1d ───────────────────────────────────────────────────────
bn1 = pycoeus.BatchNorm1d(2, eps=1e-5, momentum=1.0)
x1 = pycoeus.Tensor([
    2.0, 4.0,   # C0
    5.0, 8.0,   # C1
], [1, 2, 2])  # [N, C, L]
assert_eval_uses_running_stats_and_does_not_mutate(
    bn1, x1, [10.0, 20.0], [1.0, 1.0], c0_prefix_len=2, shape_expected=[1, 2, 2]
)

# ── BatchNorm2d ───────────────────────────────────────────────────────
bn2 = pycoeus.BatchNorm2d(2, eps=1e-5, momentum=1.0)
x2 = pycoeus.Tensor([
    2.0, 4.0, 3.0, 5.0,   # C0
    5.0, 8.0, 6.0, 9.0,   # C1
], [1, 2, 2, 2])  # [N, C, H, W]
assert_eval_uses_running_stats_and_does_not_mutate(
    bn2, x2, [10.0, 20.0], [1.0, 1.0], c0_prefix_len=4, shape_expected=[1, 2, 2, 2]
)

# ── BatchNorm3d ───────────────────────────────────────────────────────
bn3 = pycoeus.BatchNorm3d(2, eps=1e-5, momentum=1.0)
x3 = pycoeus.Tensor([
    2.0, 4.0,   # C0
    5.0, 8.0,   # C1
], [1, 2, 1, 1, 2])  # [N, C, D, H, W]
assert_eval_uses_running_stats_and_does_not_mutate(
    bn3, x3, [10.0, 20.0], [1.0, 1.0], c0_prefix_len=2, shape_expected=[1, 2, 1, 1, 2]
)
"#,
    );
}

#[test]
fn test_instancenorm_forward_shape_and_value() {
    run_script(
        r#"
import pycoeus
import math

eps = 1e-5

# ── InstanceNorm1d [N, C, L] ──────────────────────────────────────────────────
in1 = pycoeus.InstanceNorm1d(2, eps=eps)
in1.weight.data = [1.0, 1.0]
in1.bias.data = [0.0, 0.0]
x1 = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1, 2, 3])  # [N=1, C=2, L=3]
y1 = in1.forward(x1)
assert y1.shape == [1, 2, 3], f"InstanceNorm1d shape: {y1.shape}"

# Channel 0: [1,2,3] → mean=2, population var=2/3
mean_c0 = 2.0
var_c0 = 2.0 / 3.0
std_c0 = math.sqrt(var_c0 + eps)
expected_c0 = [(v - mean_c0) / std_c0 for v in [1.0, 2.0, 3.0]]
for i, (got, exp) in enumerate(zip(y1.data[:3], expected_c0)):
    assert abs(got - exp) < 1e-4, f"in1 C0[{i}]: got {got:.6f} expected {exp:.6f}"

# ── InstanceNorm2d [N, C, H, W] ──────────────────────────────────────────────
in2 = pycoeus.InstanceNorm2d(2, eps=eps)
in2.weight.data = [1.0, 1.0]
in2.bias.data = [0.0, 0.0]
x2 = pycoeus.Tensor([
    1.0, 2.0, 3.0, 4.0,
    5.0, 6.0, 7.0, 8.0,
], [1, 2, 2, 2])
y2 = in2.forward(x2)
assert y2.shape == [1, 2, 2, 2], f"InstanceNorm2d shape: {y2.shape}"

vals_c0 = [1.0, 2.0, 3.0, 4.0]
mean2 = sum(vals_c0) / 4.0
var2 = sum((v - mean2) ** 2 for v in vals_c0) / 4.0
std2 = math.sqrt(var2 + eps)
expected2 = [(v - mean2) / std2 for v in vals_c0]
for i, (got, exp) in enumerate(zip(y2.data[:4], expected2)):
    assert abs(got - exp) < 1e-4, f"in2 C0[{i}]: got {got:.6f} expected {exp:.6f}"

# ── InstanceNorm3d [N, C, D, H, W] ───────────────────────────────────────────
in3 = pycoeus.InstanceNorm3d(2, eps=eps)
in3.weight.data = [1.0, 1.0]
in3.bias.data = [0.0, 0.0]
data3 = [float(v) for v in range(1, 17)]  # 1..16, [N=1, C=2, D=2, H=2, W=2]
x3 = pycoeus.Tensor(data3, [1, 2, 2, 2, 2])
y3 = in3.forward(x3)
assert y3.shape == [1, 2, 2, 2, 2], f"InstanceNorm3d shape: {y3.shape}"

vals3_c0 = [float(v) for v in range(1, 9)]
mean3 = sum(vals3_c0) / 8.0
var3 = sum((v - mean3) ** 2 for v in vals3_c0) / 8.0
std3 = math.sqrt(var3 + eps)
expected3 = [(v - mean3) / std3 for v in vals3_c0]
for i, (got, exp) in enumerate(zip(y3.data[:8], expected3)):
    assert abs(got - exp) < 1e-4, f"in3 C0[{i}]: got {got:.6f} expected {exp:.6f}"
"#,
    );
}
