"""MLX output-parity tests for the pycoeus Python bindings.

Each test verifies that pycoeus and MLX produce numerically equivalent
forward output (loss) given identical weight values.  Tests are skipped
automatically when MLX is absent.

Run via::

    pytest coeus-python/tests/test_mlx_parity.py -v

Forward-only note:
MLX currently does not expose f64 arrays on any platform (CPU/CPU only
supports f32/f16 in the open-source build).  Pycoeus' standard pytest
parity harness uses f64.  We therefore verify forward-loss parity at the
MLX-native f32 precision with the analytical MSELoss formula and a
``1e-3`` tolerance consistent with MS-126's conv2d f32 ceiling.  Backward
parity is intentionally not asserted — the precision mismatch across
backends makes f32-vs-f64 gradient comparison unreliable, and the
existing PyTorch/JAX parity tests cover the autograd path at f64.

Weight-convention note:
MLX computes ``x @ w.T`` for the linear projection, matching pycoeus'
``[out_features, in_features]`` storage; weights are copied directly
without transposition.
"""

import os
import sys

import pytest

# Locate pycoeus.pyd alongside this test file.
_HERE = os.path.dirname(__file__)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import pycoeus  # noqa: E402

try:
    import mlx.core as mx  # noqa: E402
except ImportError:
    mx = None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ATOL = 1e-3  # MLX f32 precision ceiling (matches MS-126 conv2d tolerance)


def _allclose(label: str, got, expected, atol: float = _ATOL) -> None:
    """Assert each scalar in ``got`` matches the corresponding entry in ``expected``."""
    expected_list = list(expected)
    assert len(got) == len(expected_list), (
        f"{label}: length {len(got)} != {len(expected_list)}"
    )
    for i, (a, e) in enumerate(zip(got, expected_list)):
        a_f = float(a)
        e_f = float(e)
        diff = abs(a_f - e_f)
        assert diff <= atol, (
            f"{label}[{i}]: got={a_f:.8g}, expected={e_f:.8g}, "
            f"diff={diff:.3e}, atol={atol:.3e}"
        )


# ---------------------------------------------------------------------------
# Linear + ReLU forward (loss)
# ---------------------------------------------------------------------------


def test_linear_matches_mlx() -> None:
    """Forward parity: Linear(256→64) + ReLU + MSELoss.

    Mirrors ``test_pytorch_parity.py::test_linear_matches_pytorch`` and
    ``test_jax_parity.py::test_linear_matches_jax`` at f32 (MLX native
    precision); asserts value-semantic equality on the scalar loss within
    ``1e-3``.
    """
    if mx is None:
        pytest.skip("MLX is not installed")

    in_f, out_f, batch = 256, 64, 128

    linear_pyc = pycoeus.Linear(in_f, out_f, bias=True)
    w_data = linear_pyc.weight.data  # [out_f, in_f] flat
    b_data = linear_pyc.bias.data  # [out_f] flat

    x_data = [float(i) * 0.01 for i in range(batch * in_f)]
    tgt_data = [1.0] * (batch * out_f)

    # pycoeus forward at f64 (pycoeus default precision)
    x_pyc = pycoeus.Tensor(x_data, [batch, in_f], requires_grad=True)
    out_pyc = linear_pyc.forward(x_pyc)
    act_pyc = pycoeus.relu(out_pyc)
    tgt_pyc = pycoeus.Tensor(tgt_data, [batch, out_f])
    loss_pyc = pycoeus.mse_loss(act_pyc, tgt_pyc)

    # MLX forward at f32 (MLX native precision).
    # MLX does not expose f64; we promote pycoeus' weight (f64) to the same
    # f32 inputs MLX uses and compare scalar loss within the f32 tolerance.
    x_mlx = mx.array(x_data).reshape(batch, in_f)
    w_mlx = mx.array(w_data).reshape(out_f, in_f)
    b_mlx = mx.array(b_data).reshape(out_f)
    tgt_mlx = mx.array(tgt_data).reshape(batch, out_f)

    out_mlx = x_mlx @ w_mlx.T + b_mlx
    act_mlx = mx.maximum(out_mlx, 0.0)
    diff_mlx = act_mlx - tgt_mlx
    loss_mlx = mx.mean(diff_mlx * diff_mlx)
    mx.eval(loss_mlx)

    # Promote MLX's f32 scalar to a Python float for the comparison.
    loss_mlx_val = float(loss_mlx)
    loss_pyc_val = float(loss_pyc.data[0])

    diff = abs(loss_pyc_val - loss_mlx_val)
    assert diff <= _ATOL, (
        f"loss: got={loss_pyc_val:.8g}, expected={loss_mlx_val:.8g}, "
        f"diff={diff:.3e}, atol={_ATOL:.3e}"
    )
