//! Core neural-network module binding contracts.

use super::super::super::support::run_script;

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
fn test_module_base_class() {
    run_script(
        r#"
import pycoeus

# ── pycoeus.Module base class ─────────────────────────────────────────
# Can be instantiated directly (but forward raises NotImplementedError).
m = pycoeus.Module()
assert m.parameters() == [], "default parameters() should be empty"
assert m.is_training is True, "default training mode should be True"

m.train(False)
assert m.is_training is False, "train(False) should set eval mode"
m.eval()
assert m.is_training is False

# Resetting back to training mode
m.train()
assert m.is_training is True, "train() with no args should set training=True"

# forward raises NotImplementedError
try:
    x = pycoeus.Tensor([1.0], [1])
    m.forward(x)
    raise AssertionError("Module.forward() should raise NotImplementedError")
except NotImplementedError:
    pass

# ── Module works in Sequential / ModuleList as a duck-typed interface ─
# pycoeus modules (Linear, LayerNorm, etc.) satisfy the Module protocol
# without formally inheriting; Sequential accepts any object with forward().
lin = pycoeus.Linear(4, 4)
assert hasattr(lin, 'forward'), "Linear has forward"
assert hasattr(lin, 'parameters'), "Linear has parameters"
assert hasattr(lin, 'zero_grad'), "Linear has zero_grad"

# Custom module (pure Python, no inheritance needed for protocol use):
class ScaleLayer:
    def __init__(self, scale_val):
        self.scale = pycoeus.Tensor([scale_val], [1])
    def forward(self, x):
        return pycoeus.Tensor([v * self.scale.data[0] for v in x.data], x.shape)
    def parameters(self):
        return []
    def zero_grad(self):
        pass

sl = ScaleLayer(2.0)
x = pycoeus.Tensor([1.0, 2.0, 3.0], [3])
out = sl.forward(x)
assert out.data == [2.0, 4.0, 6.0], f"scale layer: {out.data}"

# Combine in Sequential (duck typing).
seq = pycoeus.Sequential([sl])
out2 = seq.forward(x)
assert out2.data == [2.0, 4.0, 6.0], f"sequential scale: {out2.data}"
"#,
    );
}

#[test]
fn test_bilinear_module() {
    run_script(
        r#"
import pycoeus

# ── Bilinear forward ──────────────────────────────────────────────────
# Bilinear(in1=2, in2=2, out=1) with identity weight: W[0,:,:] = I_2
bil = pycoeus.Bilinear(2, 2, 1, bias=False)
# Set W[0] to identity
bil.weight.data = [1.0, 0.0, 0.0, 1.0]

x1 = pycoeus.Tensor([1.0, 2.0], [1, 2])
x2 = pycoeus.Tensor([3.0, 4.0], [1, 2])

# out[0,0] = x1 @ W[0] @ x2.T = [1,2] @ [[1,0],[0,1]] @ [3,4].T = 1*3+2*4=11
out = bil.bilinear_forward(x1, x2)
assert out.shape == [1, 1], f"bilinear shape: {out.shape}"
assert abs(out.data[0] - 11.0) < 1e-5, f"bilinear value: {out.data[0]}"

# With bias
bil_b = pycoeus.Bilinear(2, 2, 1, bias=True)
bil_b.weight.data = [1.0, 0.0, 0.0, 1.0]
bil_b.bias.data = [5.0]
out_b = bil_b.bilinear_forward(x1, x2)
assert abs(out_b.data[0] - 16.0) < 1e-5, f"bilinear+bias: {out_b.data[0]}"

# Parameters
params = bil.parameters()
assert len(params) == 1, f"no-bias params: {len(params)}"
params_b = bil_b.parameters()
assert len(params_b) == 2, f"with-bias params: {len(params_b)}"

# state_dict roundtrip
sd = bil.state_dict()
bil2 = pycoeus.Bilinear(2, 2, 1, bias=False)
bil2.load_state_dict(sd)
out2 = bil2.bilinear_forward(x1, x2)
assert abs(out2.data[0] - 11.0) < 1e-5, f"state_dict roundtrip: {out2.data[0]}"
"#,
    );
}
