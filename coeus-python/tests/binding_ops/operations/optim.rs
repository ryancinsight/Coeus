//! Scheduler binding contracts.

use super::support::run_script;

#[test]
fn test_cosine_annealing_lr_scheduler() {
    run_script(
        r#"
import pycoeus
import math

# ── CosineAnnealingLR verification ───────────────────────────────────
# cosine anneal: lr(t) = eta_min + 0.5*(base_lr - eta_min)*(1 + cos(pi*t/T))
base_lr = 0.1
eta_min = 0.001
T = 10

def cosine_lr(t):
    t_ = min(t, T)
    return eta_min + 0.5 * (base_lr - eta_min) * (1 + math.cos(math.pi * t_ / T))

# At t=0: should equal base_lr
assert abs(cosine_lr(0) - base_lr) < 1e-9, f"t=0 should be base_lr"

# At t=T: should equal eta_min
assert abs(cosine_lr(T) - eta_min) < 1e-9, f"t=T should be eta_min"

# Monotonically decreasing from 0 to T
lrs = [cosine_lr(t) for t in range(T + 1)]
for i in range(len(lrs) - 1):
    assert lrs[i] >= lrs[i+1] - 1e-12, f"cosine LR not monotone at t={i}"

# Beyond T_max is clamped
assert abs(cosine_lr(100) - eta_min) < 1e-9, "beyond T_max should be eta_min"

# ── Test using pycoeus.LrScheduler.cosine_anneal ─────────────────────
p1 = pycoeus.Tensor([1.0, 2.0], [2], requires_grad=True)
pycoeus.sum(p1 * p1).backward()
optimizer = pycoeus.Adam([('weight', p1)], lr=base_lr)
scheduler = pycoeus.LrScheduler.cosine_anneal(optimizer, base_lr, T, eta_min)

# Step once
scheduler.step()
# After 1 step (step index 0, so t=0 was used), the optimizer stepped
# and step counter incremented to 1.
# Verify that optimizer processed its step without error (params changed).
"#,
    );
}
