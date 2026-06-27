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


# ---------------------------------------------------------------------------
# Conv1d forward + backward
# ---------------------------------------------------------------------------


def test_conv1d_matches_pytorch() -> None:
    """Forward and gradient parity: Conv1d(in=2, out=3, k=3, stride=1, pad=0, bias)."""
    w_data = [
        0.5, -0.5, 1.0, 0.0, 1.0, 0.0,
        0.1, 0.2, 0.3, -0.1, -0.2, -0.3,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    ]
    b_data = [0.1, -0.1, 0.5]
    x_data = [1.0, 2.0, 3.0, 4.0, -1.0, 0.0, 1.0, 2.0]

    conv_pyc = pycoeus.Conv1d(2, 3, 3, 1, 0, 1, True)
    conv_pyc.weight.data = w_data
    conv_pyc.bias.data = b_data
    x_pyc = pycoeus.Tensor(x_data, [1, 2, 4], requires_grad=True)
    out_pyc = conv_pyc.forward(x_pyc)
    out_pyc.backward()

    conv_t = torch.nn.Conv1d(2, 3, kernel_size=3, stride=1, padding=0, dilation=1, bias=True).double()
    with torch.no_grad():
        conv_t.weight[:] = torch.tensor(w_data, dtype=torch.float64).reshape(3, 2, 3)
        conv_t.bias[:] = torch.tensor(b_data, dtype=torch.float64)
    x_t = torch.tensor(x_data, dtype=torch.float64).reshape(1, 2, 4).requires_grad_(True)
    out_t = conv_t(x_t)
    out_t.sum().backward()

    _allclose("conv1d_out", list(out_pyc.data), out_t.flatten().tolist())
    _allclose("conv1d_dx", list(x_pyc.grad), x_t.grad.flatten().tolist())
    _allclose("conv1d_dW", list(conv_pyc.weight.grad), conv_t.weight.grad.flatten().tolist())
    _allclose("conv1d_db", list(conv_pyc.bias.grad), conv_t.bias.grad.flatten().tolist())


# ---------------------------------------------------------------------------
# Conv2d forward + backward
# ---------------------------------------------------------------------------


def test_conv2d_matches_pytorch() -> None:
    """Forward and gradient parity: Conv2d(in=2, out=2, k=2, stride=1, pad=0, bias)."""
    w_data = [
        0.5, -0.5, 1.0, 0.0,
        0.1, 0.2, 0.3, -0.1,
        -0.2, 0.5, 0.0, 1.0,
        1.0, -1.0, 0.2, 0.8,
    ]
    b_data = [0.1, -0.2]
    x_data = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
        -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0,
    ]

    conv_pyc = pycoeus.Conv2d(2, 2, 2, 1, 0, 1, True)
    conv_pyc.weight.data = w_data
    conv_pyc.bias.data = b_data
    x_pyc = pycoeus.Tensor(x_data, [1, 2, 3, 3], requires_grad=True)
    out_pyc = conv_pyc.forward(x_pyc)
    out_pyc.backward()

    conv_t = torch.nn.Conv2d(2, 2, kernel_size=2, stride=1, padding=0, dilation=1, bias=True).double()
    with torch.no_grad():
        conv_t.weight[:] = torch.tensor(w_data, dtype=torch.float64).reshape(2, 2, 2, 2)
        conv_t.bias[:] = torch.tensor(b_data, dtype=torch.float64)
    x_t = torch.tensor(x_data, dtype=torch.float64).reshape(1, 2, 3, 3).requires_grad_(True)
    out_t = conv_t(x_t)
    out_t.sum().backward()

    _allclose("conv2d_out", list(out_pyc.data), out_t.flatten().tolist())
    _allclose("conv2d_dx", list(x_pyc.grad), x_t.grad.flatten().tolist())
    _allclose("conv2d_dW", list(conv_pyc.weight.grad), conv_t.weight.grad.flatten().tolist())
    _allclose("conv2d_db", list(conv_pyc.bias.grad), conv_t.bias.grad.flatten().tolist())


# ---------------------------------------------------------------------------
# LayerNorm forward + backward
# ---------------------------------------------------------------------------


def test_layernorm_matches_pytorch() -> None:
    """Forward and gradient parity: LayerNorm(4, eps=1e-5)."""
    _ATOL_LN = 1e-4  # LN backward accumulates over the normalized dimension
    data = [1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.5, 3.0]
    gamma = [1.2, 0.8, 1.0, 0.9]
    beta = [0.1, -0.1, 0.2, 0.0]

    ln_pyc = pycoeus.LayerNorm(4, 1e-5)
    ln_pyc.weight.data = gamma
    ln_pyc.bias.data = beta
    x_pyc = pycoeus.Tensor(data, [2, 4], requires_grad=True)
    out_pyc = ln_pyc.forward(x_pyc)
    out_pyc.backward()

    ln_t = torch.nn.LayerNorm(4, eps=1e-5).double()
    with torch.no_grad():
        ln_t.weight[:] = torch.tensor(gamma, dtype=torch.float64)
        ln_t.bias[:] = torch.tensor(beta, dtype=torch.float64)
    x_t = torch.tensor(data, dtype=torch.float64).reshape(2, 4).requires_grad_(True)
    out_t = ln_t(x_t)
    out_t.sum().backward()

    _allclose("ln_out", list(out_pyc.data), out_t.flatten().tolist(), atol=_ATOL_LN)
    _allclose("ln_dx", list(x_pyc.grad), x_t.grad.flatten().tolist(), atol=_ATOL_LN)
    _allclose("ln_dgamma", list(ln_pyc.weight.grad), ln_t.weight.grad.flatten().tolist(), atol=_ATOL_LN)
    _allclose("ln_dbeta", list(ln_pyc.bias.grad), ln_t.bias.grad.flatten().tolist(), atol=_ATOL_LN)


# ---------------------------------------------------------------------------
# MultiHeadAttention backward (dx + dW_q)
# ---------------------------------------------------------------------------


def test_mha_backward_matches_pytorch() -> None:
    """Backward parity: MHA(d_model=4, H=2, no bias) — dx and dW_q after sum loss."""
    d_model, num_heads, batch, seq = 4, 2, 1, 3

    wq = [0.1, 0.2, 0.3, 0.4,  0.5, 0.6, 0.7, 0.8,
          0.9, 1.0, 0.1, 0.2,  0.3, 0.4, 0.5, 0.6]
    wk = [0.2, 0.1, 0.4, 0.3,  0.6, 0.5, 0.8, 0.7,
          0.1, 0.9, 0.2, 0.8,  0.3, 0.7, 0.4, 0.6]
    wv = [0.3, 0.3, 0.3, 0.3,  0.7, 0.7, 0.7, 0.7,
          0.4, 0.4, 0.4, 0.4,  0.8, 0.8, 0.8, 0.8]
    wo = [1.0, 0.0, 0.0, 1.0,  0.0, 1.0, 1.0, 0.0,
          0.5, 0.5, 0.5, 0.5,  0.1, 0.2, 0.3, 0.4]
    x_data = [0.1 * i - 0.3 for i in range(batch * seq * d_model)]

    mha_pyc = pycoeus.MultiHeadAttention(d_model=d_model, num_heads=num_heads, bias=False)
    mha_pyc.w_q.data = wq
    mha_pyc.w_k.data = wk
    mha_pyc.w_v.data = wv
    mha_pyc.w_o.data = wo
    x_pyc = pycoeus.Tensor(x_data, [batch, seq, d_model], requires_grad=True)
    out_pyc = mha_pyc.forward(x_pyc)
    out_pyc.backward()

    mha_t = torch.nn.MultiheadAttention(
        embed_dim=d_model, num_heads=num_heads, bias=False,
        batch_first=True, dtype=torch.float64,
    )
    with torch.no_grad():
        mha_t.in_proj_weight[:d_model, :] = torch.tensor(wq, dtype=torch.float64).reshape(d_model, d_model)
        mha_t.in_proj_weight[d_model : 2 * d_model, :] = torch.tensor(wk, dtype=torch.float64).reshape(d_model, d_model)
        mha_t.in_proj_weight[2 * d_model :, :] = torch.tensor(wv, dtype=torch.float64).reshape(d_model, d_model)
        mha_t.out_proj.weight[:] = torch.tensor(wo, dtype=torch.float64).reshape(d_model, d_model)
    x_t = torch.tensor(x_data, dtype=torch.float64).reshape(batch, seq, d_model).requires_grad_(True)
    out_t, _ = mha_t(x_t, x_t, x_t, need_weights=False)
    out_t.sum().backward()

    # dx: pycoeus MHA is self-attn so the same input contributes to Q, K, V.
    _allclose("mha_bwd_dx", list(x_pyc.grad), x_t.grad.flatten().tolist(), atol=1e-5)
    # dW_q: pycoeus [d_model, d_model] flat; PyTorch in_proj_weight[:d_model, :].
    _allclose(
        "mha_bwd_dWq",
        list(mha_pyc.w_q.grad),
        mha_t.in_proj_weight.grad[:d_model, :].flatten().tolist(),
        atol=1e-5,
    )


# ---------------------------------------------------------------------------
# TransformerEncoderLayer shape contract
# ---------------------------------------------------------------------------
#
# NOTE: pycoeus.TransformerEncoderLayer is a stateless wrapper — it
# re-initialises weights with Kaiming random values on every forward() call
# and does not expose its parameters, so direct weight-matching parity is not
# possible through this binding.  A weight-exposure refactor is tracked in
# gap_audit.md (PyTransformerEncoderLayer stateless binding defect).
# This test verifies only the shape contract and that the forward runs.


@pytest.mark.skipif(
    not hasattr(pycoeus, "TransformerEncoderLayer"),
    reason="pycoeus.TransformerEncoderLayer not available in this wheel build",
)
def test_transformer_encoder_layer_shape_contract() -> None:
    """Shape contract: TransformerEncoderLayer(d_model=4, H=2, d_ff=8) preserves [B, S, D]."""
    d_model, num_heads, d_ff = 4, 2, 8
    batch, seq = 2, 5
    tel = pycoeus.TransformerEncoderLayer(d_model=d_model, d_ff=d_ff, num_heads=num_heads)
    x_pyc = pycoeus.Tensor(
        [0.1 * i for i in range(batch * seq * d_model)],
        [batch, seq, d_model],
        requires_grad=False,
    )
    out = tel.forward(x_pyc)
    assert len(out.data) == batch * seq * d_model, (
        f"shape mismatch: expected {batch * seq * d_model}, got {len(out.data)}"
    )
