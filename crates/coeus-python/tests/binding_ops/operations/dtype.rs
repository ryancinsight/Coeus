//! Dtype conversion binding contracts.

use super::support::run_script;

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
