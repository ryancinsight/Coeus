//! Boundary contracts for the tensor's indexing and in-place mutators.
//!
//! Each assertion is written against the NumPy/PyTorch behavior a user brings
//! to `t[-1] = x` or `t.fill_(0)`, and calls out in a comment where this
//! binding intentionally differs. The restrictions pinned here — slice steps,
//! empty slices, slice assignment — are the binding's own documented limits,
//! so pinning them makes a later widening a deliberate change rather than an
//! accident.

use super::support::run_script;

#[test]
fn test_getitem_index_boundaries() {
    run_script(
        r#"
import pycoeus

t = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2])

# Negative indices normalize against dim 0 the way NumPy/PyTorch do.
assert t[-1].data == [5.0, 6.0], f"t[-1]: {t[-1].data}"
assert t[-3].data == [1.0, 2.0], f"t[-3]: {t[-3].data}"
assert t[2].data == [5.0, 6.0], f"t[2]: {t[2].data}"

# Squeezing dim 0 leaves the trailing dimensions intact.
assert t[0].shape == [2], f"t[0].shape: {t[0].shape}"
assert t[-1].shape == [2], f"t[-1].shape: {t[-1].shape}"

# Both overflows are IndexError, matching torch's "index out of range".
for bad in (-4, 3):
    try:
        t[bad]
        raise AssertionError(f"index {bad} should be out of range")
    except IndexError:
        pass

# 0-D tensors cannot be indexed at all.
scalar = pycoeus.Tensor([7.0], [])
try:
    scalar[0]
    raise AssertionError("indexing a 0-D tensor should raise IndexError")
except IndexError:
    pass
"#,
    );
}

#[test]
fn test_getitem_slice_boundaries() {
    run_script(
        r#"
import pycoeus

t = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2])

# Python slice clipping: bounds past the extent clamp instead of failing.
assert t[1:].shape == [2, 2], f"t[1:].shape: {t[1:].shape}"
assert t[0:100].shape == [3, 2], f"t[0:100].shape: {t[0:100].shape}"
assert t[-1:].shape == [1, 2], f"t[-1:].shape: {t[-1:].shape}"
assert t[-2:].data == [3.0, 4.0, 5.0, 6.0], f"t[-2:]: {t[-2:].data}"
assert t[-2:-1].shape == [1, 2], f"t[-2:-1].shape: {t[-2:-1].shape}"

# Binding restriction: only unit-stride slices exist. torch would return rows
# 0 and 2 for `t[::2]`, and an empty tensor for `t[2:2]`; both are rejected
# here, and these assertions pin that as intentional rather than incidental.
try:
    t[::2]
    raise AssertionError("step != 1 should raise ValueError")
except ValueError as error:
    assert "step" in str(error), f"unexpected step message: {error}"

try:
    t[2:2]
    raise AssertionError("empty slice should raise ValueError")
except ValueError as error:
    assert "empty slice" in str(error), f"unexpected empty-slice message: {error}"
"#,
    );
}

#[test]
fn test_getitem_rejects_non_integral_index() {
    run_script(
        r#"
import pycoeus

t = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0], [4])

# torch accepts only ints, slices and tensors here; floats raise TypeError and
# strings are never valid indexing keys.
for bad in ("0", 1.5, None):
    try:
        t[bad]
        raise AssertionError(f"index {bad!r} should raise TypeError")
    except TypeError:
        pass
"#,
    );
}

#[test]
fn test_setitem_offset_boundaries() {
    run_script(
        r#"
import pycoeus

v = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0], [4])

# Negative offset normalizes, as in torch.
v[-1] = 40.0
assert v.data == [1.0, 2.0, 3.0, 40.0], f"v[-1] = 40.0: {v.data}"
v[-4] = 10.0
assert v.data == [10.0, 2.0, 3.0, 40.0], f"v[-4] = 10.0: {v.data}"

# Python ints and bools are accepted as scalar values (torch coerces them too).
v[1] = 7
assert v.data[1] == 7.0, f"int value: {v.data[1]}"
v[2] = True
assert v.data[2] == 1.0, f"bool value: {v.data[2]}"

# Overflows in either direction are IndexError.
for bad in (-5, 4, 100):
    try:
        v[bad] = 0.0
        raise AssertionError(f"index {bad} should be out of range")
    except IndexError:
        pass

# Non-integral and slice keys raise TypeError: this binding assigns whole dim-0
# slabs only, where torch would accept `v[1:3] = 5.0`.
for bad in (1.5, slice(1, 3)):
    try:
        v[bad] = 0.0
        raise AssertionError(f"index {bad!r} should raise TypeError")
    except TypeError:
        pass

# A 0-D tensor has no dim 0 to assign into.
scalar = pycoeus.Tensor([7.0], [])
try:
    scalar[0] = 1.0
    raise AssertionError("assigning into a 0-D tensor should raise IndexError")
except IndexError:
    pass
"#,
    );
}

#[test]
fn test_setitem_row_contract() {
    run_script(
        r#"
import pycoeus

t = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2])

# A scalar fills the whole selected row and leaves the others alone.
t[0] = 10.0
assert t.data == [10.0, 10.0, 3.0, 4.0, 5.0, 6.0], f"scalar row fill: {t.data}"

# A tensor value must carry exactly one row of elements.
t[2] = pycoeus.Tensor([99.0, 88.0], [2])
assert t.data == [10.0, 10.0, 3.0, 4.0, 99.0, 88.0], f"tensor row: {t.data}"

# A single-element tensor broadcasts across the row, the result torch gives a
# size-1 value.
t[0] = pycoeus.Tensor([5.0], [1])
assert t.data == [5.0, 5.0, 3.0, 4.0, 99.0, 88.0], f"broadcast row: {t.data}"

# Any other element count is ValueError (torch raises RuntimeError here).
for wrong in (pycoeus.Tensor([1.0, 2.0, 3.0], [3]), pycoeus.Tensor([1.0, 2.0, 3.0, 4.0], [4])):
    try:
        t[0] = wrong
        raise AssertionError(f"row of {len(wrong.data)} elements should be rejected")
    except ValueError:
        pass

# Binding laxness, pinned deliberately: the value is matched by element count,
# not by shape, so a [2, 1] value fills a length-2 row where torch would reject
# the shape mismatch.
t[1] = pycoeus.Tensor([7.0, 8.0], [2, 1])
assert t.data == [5.0, 5.0, 7.0, 8.0, 99.0, 88.0], f"reshaped row value: {t.data}"
"#,
    );
}

#[test]
fn test_inplace_mutators() {
    run_script(
        r#"
import pycoeus

# fill_/zero_/one_ mutate the receiver and return it, so chaining and
# `t.zero_() is t` behave as torch's in-place methods do.
t = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
returned = t.fill_(2.5)
assert returned is t, "fill_ must return the receiver"
assert t.data == [2.5, 2.5, 2.5, 2.5], f"fill_ receiver: {t.data}"

assert t.zero_() is t, "zero_ must return the receiver"
assert t.data == [0.0, 0.0, 0.0, 0.0], f"zero_ receiver: {t.data}"
assert t.one_() is t, "one_ must return the receiver"
assert t.data == [1.0, 1.0, 1.0, 1.0], f"one_ receiver: {t.data}"

# Combination assignment mutates in place, so an alias sees the update — the
# contract torch's `add_`/`sub_`/`mul_` give their callers.
x = pycoeus.Tensor([1.0, 2.0], [2])
alias = x
x += pycoeus.Tensor([10.0, 20.0], [2])
assert x is alias, "+= must not rebind the tensor"
assert alias.data == [11.0, 22.0], f"+= alias: {alias.data}"

x -= pycoeus.Tensor([1.0, 1.0], [2])
assert alias.data == [10.0, 21.0], f"-= alias: {alias.data}"

x *= pycoeus.Tensor([2.0, 2.0], [2])
assert alias.data == [20.0, 42.0], f"*= alias: {alias.data}"

# Incompatible shapes fail instead of silently reallocating (torch raises
# RuntimeError; the backend's broadcast error surfaces as ValueError here).
for bad in (
    lambda: x.__iadd__(pycoeus.Tensor([1.0, 2.0, 3.0], [3])),
    lambda: x.__isub__(pycoeus.Tensor([1.0, 2.0, 3.0], [3])),
    lambda: x.__imul__(pycoeus.Tensor([1.0, 2.0, 3.0], [3])),
):
    try:
        bad()
        raise AssertionError("mismatched in-place operand should raise ValueError")
    except ValueError:
        pass
assert alias.data == [20.0, 42.0], f"failed in-place ops must not mutate: {alias.data}"
"#,
    );
}

#[test]
fn test_inplace_rejects_tracked_tensors() {
    run_script(
        r#"
import pycoeus

# torch raises for an in-place op on a leaf that requires grad; this binding
# refuses with ValueError (its own contract-error class) and names the path.
g = pycoeus.Tensor([1.0, 2.0], [2], requires_grad=True)
other = pycoeus.Tensor([1.0, 1.0], [2])

attempts = {
    "fill_": lambda: g.fill_(0.0),
    "zero_": lambda: g.zero_(),
    "one_": lambda: g.one_(),
    "__setitem__": lambda: g.__setitem__(0, 1.0),
    "__iadd__": lambda: g.__iadd__(other),
    "__isub__": lambda: g.__isub__(other),
    "__imul__": lambda: g.__imul__(other),
}
for name, attempt in attempts.items():
    try:
        attempt()
        raise AssertionError(f"{name} on a requires_grad tensor must be refused")
    except ValueError as error:
        assert "requires grad" in str(error), f"{name} message: {error}"

# The refusal is total: no path may have partially rewritten the value.
assert g.data == [1.0, 2.0], f"refused mutations changed the value: {g.data}"

# Stricter than torch, which allows in-place ops on a tracked non-leaf result:
# a backward node saves clones of its operands at forward time, so rewriting the
# result would make the next backward describe the old value.
product = g * 2.0
try:
    product.zero_()
    raise AssertionError("in-place on a tracked operation result must be refused")
except ValueError:
    pass

# detach() yields an untracked copy, which is the supported way to mutate.
detached = g.detach()
assert detached.fill_(7.0) is detached, "detached copy must stay mutable"
assert detached.data == [7.0, 7.0], f"detached copy: {detached.data}"
assert g.data == [1.0, 2.0], f"detached mutation leaked into the tracked tensor: {g.data}"

# And requires_grad_(False) on the original lifts the restriction, returning the
# receiver so the flag change and the mutation chain on one object.
assert g.requires_grad_(False) is g, "requires_grad_ must return the receiver"
assert g.zero_() is g, "zero_ after requires_grad_(False) must mutate in place"
assert g.data == [0.0, 0.0], f"zero_ after requires_grad_(False): {g.data}"
"#,
    );
}

#[test]
fn test_empty_and_zero_element_tensors() {
    run_script(
        r#"
import pycoeus

# `Tensor([])` infers a length-0 dim 0, the shape torch.tensor([]) gives.
t = pycoeus.Tensor([])
assert t.shape == [0], f"Tensor([]) shape: {t.shape}"
assert t.ndim == 1, f"Tensor([]) ndim: {t.ndim}"
assert t.numel() == 0, f"Tensor([]) numel: {t.numel()}"
assert t.data == [], f"Tensor([]) data: {t.data}"
assert t.tolist() == [], f"Tensor([]) tolist: {t.tolist()}"
assert t.is_contiguous(), "a zero-element tensor is trivially contiguous"
assert str(t) == "Tensor([], shape=[0])", f"empty repr: {t!r}"

# A zero extent in any position carries zero elements, whatever the other dims.
for shape in ([0], [0, 3], [2, 0], [3, 0, 2]):
    z = pycoeus.Tensor([], shape)
    assert z.shape == shape, f"empty shape {shape}: {z.shape}"
    assert z.numel() == 0, f"empty numel {shape}: {z.numel()}"
    assert z.data == [], f"empty data {shape}: {z.data}"
    # Iteration walks dim 0, so a zero *leading* extent yields no items while a
    # zero inner extent still yields one empty row per leading index, as in
    # torch's `list(torch.empty(2, 0))`.
    rows = list(z)
    assert len(rows) == shape[0], f"empty iter len {shape}: {len(rows)}"
    assert all(row.shape == shape[1:] for row in rows), f"empty row shape {shape}"

# Data and shape are validated against each other rather than trusted: a
# zero-element payload in a 0-D shape (which holds exactly one element) is a
# ValueError, not a Rust panic leaking out of the constructor.
for data, shape in (([], []), ([1.0], [0]), ([1.0, 2.0], [3])):
    try:
        pycoeus.Tensor(data, shape)
        raise AssertionError(f"Tensor({data}, {shape}) should raise ValueError")
    except ValueError:
        pass

# Dim 0 has size 0, so no index exists in either direction. torch reports the
# same case as "index 0 is out of bounds for dimension 0 with size 0".
z = pycoeus.Tensor([], [0])
for bad in (0, -1):
    for attempt, label in (
        (lambda: z[bad], f"read at {bad}"),
        (lambda: z.__setitem__(bad, 1.0), f"assign at {bad}"),
    ):
        try:
            attempt()
            raise AssertionError(f"{label} on an empty tensor should raise IndexError")
        except IndexError:
            pass

# A zero-extent slice is ValueError here where torch returns shape [0]; the
# unit-stride/non-empty slice restriction is pinned in
# test_getitem_slice_boundaries and this is its zero-extent corner.
try:
    z[0:]
    raise AssertionError("slicing an empty dim should raise ValueError")
except ValueError as error:
    assert "empty slice" in str(error), f"unexpected empty-slice message: {error}"

# The in-place mutators are total on zero elements and still hand back the
# receiver; there is nothing to write, so the tensor stays empty.
assert z.fill_(3.0) is z, "fill_ must return the receiver for an empty tensor"
assert z.data == [], f"empty fill_: {z.data}"
assert z.zero_() is z, "zero_ must return the receiver for an empty tensor"
assert z.one_() is z, "one_ must return the receiver for an empty tensor"
assert z.data == [], f"empty one_: {z.data}"

# The tracked-tensor guard applies to an empty tensor too: the refusal is about
# the autograd state, not about how many elements there are to rewrite.
tracked = pycoeus.Tensor([], [0], requires_grad=True)
assert tracked.requires_grad is True
for attempt, name in (
    (lambda: tracked.zero_(), "zero_"),
    (lambda: tracked.fill_(0.0), "fill_"),
    (lambda: tracked.__setitem__(0, 0.0), "__setitem__"),
):
    try:
        attempt()
        raise AssertionError(f"{name} on a tracked empty tensor must be refused")
    except ValueError as error:
        assert "requires grad" in str(error), f"{name} message: {error}"
assert tracked.requires_grad_(False) is tracked, "requires_grad_ must return the receiver"
assert tracked.zero_() is tracked, "zero_ after requires_grad_(False) must mutate in place"

# The dual corner: shape [] is a 0-D tensor holding exactly one element.
scalar = pycoeus.Tensor([7.0], [])
assert scalar.shape == [] and scalar.ndim == 0, f"0-D shape: {scalar.shape}"
assert scalar.numel() == 1, f"0-D numel: {scalar.numel()}"
assert scalar.item() == 7.0, f"0-D item: {scalar.item()}"
assert scalar.zero_() is scalar, "zero_ must return the receiver for a 0-D tensor"
assert scalar.item() == 0.0, f"0-D zero_: {scalar.item()}"
"#,
    );
}
