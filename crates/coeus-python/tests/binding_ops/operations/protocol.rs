//! Python protocol contracts: truthiness, length, and numeric conversion.
//!
//! A tensor is only truthy when it holds exactly one element. The empty and
//! multi-element cases are refused rather than answered, because the answer a
//! user reads into them (`False`/`True`) is where silent logic bugs live — the
//! Conv3d parity check in `parity_comparison.py` skipped a bias assignment for
//! months by testing a one-element zero bias as if it were `None`.

use super::support::run_script;

#[test]
fn test_tensor_truthiness_requires_one_element() {
    run_script(
        r#"
import pycoeus

# A single-element tensor has a truth value, whether 1-D or 0-D.
assert bool(pycoeus.Tensor([2.0], [1])) is True
assert bool(pycoeus.Tensor([0.0], [1])) is False
assert bool(pycoeus.Tensor([-0.5], [1])) is True
assert bool(pycoeus.Tensor([7.0], [])) is True
assert bool(pycoeus.Tensor([0.0], [])) is False
assert not pycoeus.Tensor([0.0], [1])

# Only a one-element tensor has a truth value. torch refuses both of these with
# "Boolean value of Tensor with no values / more than one value is ambiguous";
# this binding refuses with ValueError (its contract-error class) and the same
# wording.
for label, tensor, expected in (
    ("empty", pycoeus.Tensor([], [0]), "no values"),
    ("two", pycoeus.Tensor([1.0, 0.0], [2]), "more than one value"),
    ("four", pycoeus.Tensor([0.0] * 4, [2, 2]), "more than one value"),
):
    try:
        bool(tensor)
        raise AssertionError(f"bool({label}) should raise")
    except ValueError as error:
        message = str(error)
        assert "ambiguous" in message, f"{label} message: {message}"
        assert expected in message, f"{label} message: {message}"

# The refusal is about element count, not content: an all-zero multi-element
# tensor is as ambiguous as one holding nonzeros, so an `if tensor:` guard can
# never silently take the false branch on a whole batch.
try:
    if pycoeus.Tensor([0.0, 0.0], [2]):
        raise AssertionError("an all-zero multi-element tensor must not be truthy")
except ValueError:
    pass

# `is not None` and `numel()`/`len()` remain the ways to test presence and
# emptiness, which is what the suites use.
empty = pycoeus.Tensor([], [0, 3])
assert empty is not None
assert len(empty) == 0
assert empty.numel() == 0

# Conversion protocols keep their single-element rule.
assert float(pycoeus.Tensor([2.5], [1])) == 2.5
assert int(pycoeus.Tensor([3.9], [1])) == 3
for bad in (pycoeus.Tensor([], [0]), pycoeus.Tensor([1.0, 2.0], [2])):
    try:
        float(bad)
        raise AssertionError("conversion of a multi-element tensor should raise")
    except ValueError:
        pass
"#,
    );
}
