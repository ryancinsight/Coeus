//! Autograd and gradient-state binding contracts.

use super::support::run_script;

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
fn test_cat_where_backward_parity() {
    run_script(
        r#"
import pycoeus

# ── cat backward ──────────────────────────────────────────────────────
x = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], requires_grad=True)
y = pycoeus.Tensor([7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0], [3, 3], requires_grad=True)
out = pycoeus.cat([x, y], 0)
assert out.shape == [5, 3], f"cat shape: {out.shape}"
pycoeus.sum(out).backward()
# Every element contributed once → all grads = 1.
for v in x.grad:
    assert abs(v - 1.0) < 1e-9, f"cat x.grad element: {v}"
for v in y.grad:
    assert abs(v - 1.0) < 1e-9, f"cat y.grad element: {v}"

# ── where_cond backward ───────────────────────────────────────────────
cond = pycoeus.Tensor([1.0, 0.0, 1.0, 0.0], [4])
on_true = pycoeus.Tensor([10.0, 11.0, 12.0, 13.0], [4], requires_grad=True)
on_false = pycoeus.Tensor([20.0, 21.0, 22.0, 23.0], [4], requires_grad=True)
where_out = pycoeus.where_cond(cond, on_true, on_false)
assert where_out.data == [10.0, 21.0, 12.0, 23.0], f"where_cond fwd: {where_out.data}"
pycoeus.sum(where_out).backward()
# on_true gets grad at positions where cond == 1
assert on_true.grad == [1.0, 0.0, 1.0, 0.0], f"where on_true.grad: {on_true.grad}"
# on_false gets grad at positions where cond == 0
assert on_false.grad == [0.0, 1.0, 0.0, 1.0], f"where on_false.grad: {on_false.grad}"
"#,
    );
}

#[test]
fn test_gradient_accumulation() {
    run_script(
        r#"
import pycoeus

# ── Gradient accumulation: N backward passes before zero_grad ─────────
# sum(x * x).backward() gives grad = 2 * x.
# Accumulating over N steps without zeroing should give N * single_step.

p = pycoeus.Tensor([1.0, 2.0, 3.0], [3], requires_grad=True)
single_step_grad = [2.0, 4.0, 6.0]
N = 3

for _ in range(N):
    loss = pycoeus.sum(p * p)
    loss.backward()

assert p.grad is not None, "gradient must exist"
for got, want in zip(p.grad, [v * N for v in single_step_grad]):
    assert abs(got - want) < 1e-5, f"accumulated grad: got {got} want {want}"

# After zero_grad, accumulated grad is cleared.
p.zero_grad()
assert p.grad is None or all(v == 0.0 for v in (p.grad or [])), "zero_grad must clear"

# One more forward+backward gives single_step again.
pycoeus.sum(p * p).backward()
for got, want in zip(p.grad, single_step_grad):
    assert abs(got - want) < 1e-5, f"post-zero grad: got {got} want {want}"
"#,
    );
}
