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
import math

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


# ---------------------------------------------------------------------------
# MultiHeadAttention forward parity
# ---------------------------------------------------------------------------


def test_mha_matches_mlx() -> None:
    """Forward parity: MultiHeadAttention(d_model=4, H=2), self-attention, no bias."""
    if mx is None:
        pytest.skip("MLX is not installed")

    d_model, num_heads, batch, seq = 4, 2, 1, 3
    d_head = d_model // num_heads

    wq = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
          0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    wk = [0.2, 0.1, 0.4, 0.3, 0.6, 0.5, 0.8, 0.7,
          0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6]
    wv = [0.3, 0.3, 0.3, 0.3, 0.7, 0.7, 0.7, 0.7,
          0.4, 0.4, 0.4, 0.4, 0.8, 0.8, 0.8, 0.8]
    wo = [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0,
          0.5, 0.5, 0.5, 0.5, 0.1, 0.2, 0.3, 0.4]
    x_data = [0.1 * i - 0.3 for i in range(batch * seq * d_model)]

    mha_pyc = pycoeus.MultiHeadAttention(d_model=d_model, num_heads=num_heads, bias=False)
    mha_pyc.w_q.data = wq
    mha_pyc.w_k.data = wk
    mha_pyc.w_v.data = wv
    mha_pyc.w_o.data = wo
    x_pyc = pycoeus.Tensor(x_data, [batch, seq, d_model], requires_grad=False)
    out_pyc = mha_pyc.forward(x_pyc)

    x_mlx = mx.array(x_data).reshape(batch, seq, d_model)
    wq_mlx = mx.array(wq).reshape(d_model, d_model)
    wk_mlx = mx.array(wk).reshape(d_model, d_model)
    wv_mlx = mx.array(wv).reshape(d_model, d_model)
    wo_mlx = mx.array(wo).reshape(d_model, d_model)

    q = x_mlx @ wq_mlx.T
    k = x_mlx @ wk_mlx.T
    v = x_mlx @ wv_mlx.T

    qh = mx.transpose(q.reshape(batch, seq, num_heads, d_head), (0, 2, 1, 3))
    kh = mx.transpose(k.reshape(batch, seq, num_heads, d_head), (0, 2, 1, 3))
    vh = mx.transpose(v.reshape(batch, seq, num_heads, d_head), (0, 2, 1, 3))

    scores = mx.matmul(qh, mx.transpose(kh, (0, 1, 3, 2))) / math.sqrt(d_head)
    attn = mx.softmax(scores, axis=-1)
    ctx = mx.matmul(attn, vh)
    merged = mx.transpose(ctx, (0, 2, 1, 3)).reshape(batch, seq, d_model)
    out_mlx = merged @ wo_mlx.T
    mx.eval(out_mlx)

    _allclose("mha_out", list(out_pyc.data), out_mlx.flatten().tolist(), atol=1e-3)
