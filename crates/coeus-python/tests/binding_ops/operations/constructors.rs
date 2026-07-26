//! Constructor and module-initialization binding contracts.

use super::support::run_script;

#[test]
fn test_constructors() {
    run_script(
        r#"
import pycoeus

z = pycoeus.zeros([2, 3])
assert z.shape == [2, 3]
assert all(v == 0.0 for v in z.data)

o = pycoeus.ones([2, 3])
assert o.shape == [2, 3]
assert all(v == 1.0 for v in o.data)

f = pycoeus.full([4], 3.14)
assert f.shape == [4]
assert abs(f.data[0] - 3.14) < 1e-9
assert abs(f.data[3] - 3.14) < 1e-9

a = pycoeus.arange(0.0, 5.0, 1.0)
assert a.shape == [5]
assert a.data == [0.0, 1.0, 2.0, 3.0, 4.0]

a2 = pycoeus.arange(0.0, 1.0, 0.25)
assert len(a2.data) == 4

lin = pycoeus.linspace(0.0, 1.0, 5)
assert lin.shape == [5]
assert abs(lin.data[0] - 0.0) < 1e-9
assert abs(lin.data[4] - 1.0) < 1e-9
assert abs(lin.data[2] - 0.5) < 1e-9

log = pycoeus.logspace(0.0, 2.0, 3, 10.0)
assert log.shape == [3]
assert abs(log.data[0] - 1.0) < 1e-9
assert abs(log.data[1] - 10.0) < 1e-9
assert abs(log.data[2] - 100.0) < 1e-9

geo = pycoeus.geomspace(1.0, 16.0, 5)
assert geo.shape == [5]
assert abs(geo.data[0] - 1.0) < 1e-9
assert abs(geo.data[1] - 2.0) < 1e-9
assert abs(geo.data[2] - 4.0) < 1e-9
assert abs(geo.data[3] - 8.0) < 1e-9
assert abs(geo.data[4] - 16.0) < 1e-9

try:
    pycoeus.geomspace(-1.0, 16.0, 4)
    raise AssertionError("expected ValueError for mismatched signs")
except ValueError:
    pass
"#,
    );
}

#[test]
fn test_init_submodule_mutates_tensor_values() {
    run_script(
        r#"
import pycoeus

t = pycoeus.zeros([2, 3], requires_grad=True)
pycoeus.init.constant_(t, 2.5)
assert t.shape == [2, 3]
assert t.requires_grad is True
assert t.data == [2.5] * 6, f"constant_: {t.data}"

pycoeus.init.zeros_(t)
assert t.data == [0.0] * 6, f"zeros_: {t.data}"

pycoeus.init.ones_(t)
assert t.data == [1.0] * 6, f"ones_: {t.data}"

pycoeus.init.uniform_(t, -0.25, 0.25)
assert all(-0.25 <= v <= 0.25 for v in t.data), f"uniform_ range: {t.data}"
assert any(v != 0.0 for v in t.data), f"uniform_ should write nonzero values: {t.data}"

pycoeus.init.normal_(t, 1.0, 0.5)
assert len(t.data) == 6 and all(v == v for v in t.data), f"normal_: {t.data}"

pycoeus.init.xavier_uniform_(t, 3, 5)
limit = (6.0 / 8.0) ** 0.5
assert all(-limit <= v <= limit for v in t.data), f"xavier_uniform_ range: {t.data}"

pycoeus.init.xavier_normal_(t, 3, 5)
assert len(t.data) == 6 and all(v == v for v in t.data), f"xavier_normal_: {t.data}"

pycoeus.init.kaiming_uniform_(t, 3)
k_limit = (6.0 / 3.0) ** 0.5
assert all(-k_limit <= v <= k_limit for v in t.data), f"kaiming_uniform_ range: {t.data}"

pycoeus.init.kaiming_normal_(t, 3)
assert len(t.data) == 6 and all(v == v for v in t.data), f"kaiming_normal_: {t.data}"
"#,
    );
}

#[test]
fn test_randn_zeros_ones_like_eye() {
    run_script(
        r#"
import pycoeus

# randn: should produce different values each call (not all zero)
r = pycoeus.randn([3, 4])
assert r.shape == [3, 4]
assert not all(v == 0.0 for v in r.data), "randn returned all zeros"

# zeros_like
z = pycoeus.zeros([2, 3])
zl = pycoeus.zeros_like(z)
assert zl.shape == [2, 3]
assert all(v == 0.0 for v in zl.data)

# ones_like
ol = pycoeus.ones_like(z)
assert ol.shape == [2, 3]
assert all(v == 1.0 for v in ol.data)

# eye
e = pycoeus.eye(3)
assert e.shape == [3, 3]
assert e.data[0] == 1.0 and e.data[1] == 0.0 and e.data[4] == 1.0

# rand: uniform [0, 1)
u = pycoeus.rand([2, 3], requires_grad=True)
assert u.shape == [2, 3]
assert u.requires_grad is True
assert all(0.0 <= v < 1.0 for v in u.data), f"rand range: {u.data}"

# randint: integer values in [low, high)
ri = pycoeus.randint(2, 5, [8])
assert ri.shape == [8]
assert all(v in (2.0, 3.0, 4.0) for v in ri.data), f"randint range: {ri.data}"
try:
    pycoeus.randint(5, 5, [1])
    raise AssertionError("randint should reject empty range")
except ValueError:
    pass

# bernoulli deterministic boundaries
b0 = pycoeus.bernoulli([4], p=0.0)
b1 = pycoeus.bernoulli([4], p=1.0)
assert b0.data == [0.0, 0.0, 0.0, 0.0], f"bernoulli p=0: {b0.data}"
assert b1.data == [1.0, 1.0, 1.0, 1.0], f"bernoulli p=1: {b1.data}"
try:
    pycoeus.bernoulli([1], p=1.5)
    raise AssertionError("bernoulli should reject p > 1")
except ValueError:
    pass
"#,
    );
}
