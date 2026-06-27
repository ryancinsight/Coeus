"""PyTorch output-parity tests for the pycoeus Python bindings.

Each test verifies that pycoeus and PyTorch produce numerically equivalent
outputs (forward and, where applicable, backward/gradient) given identical
weight values.  Tests are skipped automatically when PyTorch is absent.

Run via::

    pytest coeus-python/tests/test_pytorch_parity.py -v

Weight-convention note:
Both pycoeus and PyTorch Linear/MHA store projection weights in
``[out_features, in_features]`` order and compute ``x @ W.T``, so weights
are copied directly without transposition.
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

torch = pytest.importorskip("torch")  # skip entire module if PyTorch absent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ATOL = 1e-5


def _allclose(label: str, got: list, expected: list, atol: float = _ATOL) -> None:
    assert len(got) == len(expected), f"{label}: length {len(got)} != {len(expected)}"
    for i, (a, e) in enumerate(zip(got, expected)):
        diff = abs(a - e)
        assert diff <= atol, (
            f"{label}[{i}]: got={a:.8g}, expected={e:.8g}, diff={diff:.3e}, atol={atol:.3e}"
        )


# ---------------------------------------------------------------------------
# Linear + ReLU forward + backward
# ---------------------------------------------------------------------------


def test_linear_matches_pytorch() -> None:
    """Forward and gradient parity: Linear(256→64) + ReLU + MSELoss."""
    in_f, out_f, batch = 256, 64, 128

    linear_pyc = pycoeus.Linear(in_f, out_f, bias=True)
    w_data = linear_pyc.weight.data  # [out_f, in_f] flat
    b_data = linear_pyc.bias.data    # [out_f] flat

    x_data = [float(i) * 0.01 for i in range(batch * in_f)]
    tgt_data = [1.0] * (batch * out_f)

    # pycoeus forward + backward
    x_pyc = pycoeus.Tensor(x_data, [batch, in_f], requires_grad=True)
    out_pyc = linear_pyc.forward(x_pyc)
    act_pyc = pycoeus.relu(out_pyc)
    tgt_pyc = pycoeus.Tensor(tgt_data, [batch, out_f])
    loss_pyc = pycoeus.mse_loss(act_pyc, tgt_pyc)
    loss_pyc.backward()

    # PyTorch forward + backward (f64 to match pycoeus default precision)
    x_t = torch.tensor(x_data, dtype=torch.float64).reshape(batch, in_f).requires_grad_(True)
    w_t = torch.tensor(w_data, dtype=torch.float64).reshape(out_f, in_f).requires_grad_(True)
    b_t = torch.tensor(b_data, dtype=torch.float64).requires_grad_(True)
    out_t = torch.nn.functional.linear(x_t, w_t, b_t)
    act_t = torch.relu(out_t)
    tgt_t = torch.tensor(tgt_data, dtype=torch.float64).reshape(batch, out_f)
    loss_t = torch.nn.functional.mse_loss(act_t, tgt_t)
    loss_t.backward()

    assert abs(loss_pyc.data[0] - loss_t.item()) < _ATOL, (
        f"loss: got={loss_pyc.data[0]:.8g}, expected={loss_t.item():.8g}"
    )
    _allclose("dx", list(x_pyc.grad), x_t.grad.flatten().tolist())
    _allclose("dW", list(linear_pyc.weight.grad), w_t.grad.flatten().tolist())
    _allclose("db", list(linear_pyc.bias.grad), b_t.grad.flatten().tolist())


# ---------------------------------------------------------------------------
# MultiHeadAttention forward parity
# ---------------------------------------------------------------------------


def test_mha_matches_pytorch() -> None:
    """Forward parity: MultiHeadAttention(d_model=4, H=2), self-attention, no bias.

    Both pycoeus and PyTorch store projection weights as ``[d_out, d_in]`` and
    compute ``x @ W.T``; weights are copied directly.
    """
    d_model, num_heads, batch, seq = 4, 2, 1, 3

    # Fixed weights: deterministic, non-trivial.
    wq = [0.1, 0.2, 0.3, 0.4,  0.5, 0.6, 0.7, 0.8,
          0.9, 1.0, 0.1, 0.2,  0.3, 0.4, 0.5, 0.6]
    wk = [0.2, 0.1, 0.4, 0.3,  0.6, 0.5, 0.8, 0.7,
          0.1, 0.9, 0.2, 0.8,  0.3, 0.7, 0.4, 0.6]
    wv = [0.3, 0.3, 0.3, 0.3,  0.7, 0.7, 0.7, 0.7,
          0.4, 0.4, 0.4, 0.4,  0.8, 0.8, 0.8, 0.8]
    wo = [1.0, 0.0, 0.0, 1.0,  0.0, 1.0, 1.0, 0.0,
          0.5, 0.5, 0.5, 0.5,  0.1, 0.2, 0.3, 0.4]

    x_data = [0.1 * i - 0.3 for i in range(batch * seq * d_model)]

    # pycoeus
    mha_pyc = pycoeus.MultiHeadAttention(d_model=d_model, num_heads=num_heads, bias=False)
    mha_pyc.w_q.data = wq
    mha_pyc.w_k.data = wk
    mha_pyc.w_v.data = wv
    mha_pyc.w_o.data = wo
    x_pyc = pycoeus.Tensor(x_data, [batch, seq, d_model], requires_grad=False)
    out_pyc = mha_pyc.forward(x_pyc)

    # PyTorch: in_proj_weight rows are [Wq, Wk, Wv], each [d_model, d_model].
    mha_t = torch.nn.MultiheadAttention(
        embed_dim=d_model, num_heads=num_heads, bias=False,
        batch_first=True, dtype=torch.float64,
    )
    with torch.no_grad():
        mha_t.in_proj_weight[:d_model, :] = (
            torch.tensor(wq, dtype=torch.float64).reshape(d_model, d_model)
        )
        mha_t.in_proj_weight[d_model : 2 * d_model, :] = (
            torch.tensor(wk, dtype=torch.float64).reshape(d_model, d_model)
        )
        mha_t.in_proj_weight[2 * d_model :, :] = (
            torch.tensor(wv, dtype=torch.float64).reshape(d_model, d_model)
        )
        mha_t.out_proj.weight[:] = (
            torch.tensor(wo, dtype=torch.float64).reshape(d_model, d_model)
        )
    x_t = torch.tensor(x_data, dtype=torch.float64).reshape(batch, seq, d_model)
    out_t, _ = mha_t(x_t, x_t, x_t, need_weights=False)

    _allclose("mha_out", list(out_pyc.data), out_t.flatten().tolist(), atol=1e-10)
