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

import math
import os
import sys

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
    b_data = linear_pyc.bias.data  # [out_f] flat

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
    x_t = (
        torch.tensor(x_data, dtype=torch.float64)
        .reshape(batch, in_f)
        .requires_grad_(True)
    )
    w_t = (
        torch.tensor(w_data, dtype=torch.float64)
        .reshape(out_f, in_f)
        .requires_grad_(True)
    )
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
    wq = [
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
    ]
    wk = [
        0.2,
        0.1,
        0.4,
        0.3,
        0.6,
        0.5,
        0.8,
        0.7,
        0.1,
        0.9,
        0.2,
        0.8,
        0.3,
        0.7,
        0.4,
        0.6,
    ]
    wv = [
        0.3,
        0.3,
        0.3,
        0.3,
        0.7,
        0.7,
        0.7,
        0.7,
        0.4,
        0.4,
        0.4,
        0.4,
        0.8,
        0.8,
        0.8,
        0.8,
    ]
    wo = [
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        1.0,
        1.0,
        0.0,
        0.5,
        0.5,
        0.5,
        0.5,
        0.1,
        0.2,
        0.3,
        0.4,
    ]

    x_data = [0.1 * i - 0.3 for i in range(batch * seq * d_model)]

    # pycoeus
    mha_pyc = pycoeus.MultiHeadAttention(
        d_model=d_model, num_heads=num_heads, bias=False
    )
    mha_pyc.w_q.data = wq
    mha_pyc.w_k.data = wk
    mha_pyc.w_v.data = wv
    mha_pyc.w_o.data = wo
    x_pyc = pycoeus.Tensor(x_data, [batch, seq, d_model], requires_grad=False)
    out_pyc = mha_pyc.forward(x_pyc)

    # PyTorch: in_proj_weight rows are [Wq, Wk, Wv], each [d_model, d_model].
    mha_t = torch.nn.MultiheadAttention(
        embed_dim=d_model,
        num_heads=num_heads,
        bias=False,
        batch_first=True,
        dtype=torch.float64,
    )
    with torch.no_grad():
        mha_t.in_proj_weight[:d_model, :] = torch.tensor(
            wq, dtype=torch.float64
        ).reshape(d_model, d_model)
        mha_t.in_proj_weight[d_model : 2 * d_model, :] = torch.tensor(
            wk, dtype=torch.float64
        ).reshape(d_model, d_model)
        mha_t.in_proj_weight[2 * d_model :, :] = torch.tensor(
            wv, dtype=torch.float64
        ).reshape(d_model, d_model)
        mha_t.out_proj.weight[:] = torch.tensor(wo, dtype=torch.float64).reshape(
            d_model, d_model
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
        0.5,
        -0.5,
        1.0,
        0.0,
        1.0,
        0.0,
        0.1,
        0.2,
        0.3,
        -0.1,
        -0.2,
        -0.3,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ]
    b_data = [0.1, -0.1, 0.5]
    x_data = [1.0, 2.0, 3.0, 4.0, -1.0, 0.0, 1.0, 2.0]

    conv_pyc = pycoeus.Conv1d(2, 3, 3, 1, 0, 1, True)
    conv_pyc.weight.data = w_data
    conv_pyc.bias.data = b_data
    x_pyc = pycoeus.Tensor(x_data, [1, 2, 4], requires_grad=True)
    out_pyc = conv_pyc.forward(x_pyc)
    out_pyc.backward()

    conv_t = torch.nn.Conv1d(
        2, 3, kernel_size=3, stride=1, padding=0, dilation=1, bias=True
    ).double()
    with torch.no_grad():
        conv_t.weight[:] = torch.tensor(w_data, dtype=torch.float64).reshape(3, 2, 3)
        conv_t.bias[:] = torch.tensor(b_data, dtype=torch.float64)
    x_t = (
        torch.tensor(x_data, dtype=torch.float64).reshape(1, 2, 4).requires_grad_(True)
    )
    out_t = conv_t(x_t)
    out_t.sum().backward()

    _allclose("conv1d_out", list(out_pyc.data), out_t.flatten().tolist())
    _allclose("conv1d_dx", list(x_pyc.grad), x_t.grad.flatten().tolist())
    _allclose(
        "conv1d_dW", list(conv_pyc.weight.grad), conv_t.weight.grad.flatten().tolist()
    )
    _allclose(
        "conv1d_db", list(conv_pyc.bias.grad), conv_t.bias.grad.flatten().tolist()
    )


# ---------------------------------------------------------------------------
# Conv2d forward + backward
# ---------------------------------------------------------------------------


def test_conv2d_matches_pytorch() -> None:
    """Forward and gradient parity: Conv2d(in=2, out=2, k=2, stride=1, pad=0, bias)."""
    w_data = [
        0.5,
        -0.5,
        1.0,
        0.0,
        0.1,
        0.2,
        0.3,
        -0.1,
        -0.2,
        0.5,
        0.0,
        1.0,
        1.0,
        -1.0,
        0.2,
        0.8,
    ]
    b_data = [0.1, -0.2]
    x_data = [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        -1.0,
        -2.0,
        -3.0,
        -4.0,
        -5.0,
        -6.0,
        -7.0,
        -8.0,
        -9.0,
    ]

    conv_pyc = pycoeus.Conv2d(2, 2, 2, 1, 0, 1, True)
    conv_pyc.weight.data = w_data
    conv_pyc.bias.data = b_data
    x_pyc = pycoeus.Tensor(x_data, [1, 2, 3, 3], requires_grad=True)
    out_pyc = conv_pyc.forward(x_pyc)
    out_pyc.backward()

    conv_t = torch.nn.Conv2d(
        2, 2, kernel_size=2, stride=1, padding=0, dilation=1, bias=True
    ).double()
    with torch.no_grad():
        conv_t.weight[:] = torch.tensor(w_data, dtype=torch.float64).reshape(2, 2, 2, 2)
        conv_t.bias[:] = torch.tensor(b_data, dtype=torch.float64)
    x_t = (
        torch.tensor(x_data, dtype=torch.float64)
        .reshape(1, 2, 3, 3)
        .requires_grad_(True)
    )
    out_t = conv_t(x_t)
    out_t.sum().backward()

    _allclose("conv2d_out", list(out_pyc.data), out_t.flatten().tolist())
    _allclose("conv2d_dx", list(x_pyc.grad), x_t.grad.flatten().tolist())
    _allclose(
        "conv2d_dW", list(conv_pyc.weight.grad), conv_t.weight.grad.flatten().tolist()
    )
    _allclose(
        "conv2d_db", list(conv_pyc.bias.grad), conv_t.bias.grad.flatten().tolist()
    )


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
    _allclose(
        "ln_dgamma",
        list(ln_pyc.weight.grad),
        ln_t.weight.grad.flatten().tolist(),
        atol=_ATOL_LN,
    )
    _allclose(
        "ln_dbeta",
        list(ln_pyc.bias.grad),
        ln_t.bias.grad.flatten().tolist(),
        atol=_ATOL_LN,
    )


# ---------------------------------------------------------------------------
# MultiHeadAttention backward (dx + dW_q)
# ---------------------------------------------------------------------------


def test_mha_backward_matches_pytorch() -> None:
    """Backward parity: MHA(d_model=4, H=2, no bias) — dx and dW_q after sum loss."""
    d_model, num_heads, batch, seq = 4, 2, 1, 3

    wq = [
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
    ]
    wk = [
        0.2,
        0.1,
        0.4,
        0.3,
        0.6,
        0.5,
        0.8,
        0.7,
        0.1,
        0.9,
        0.2,
        0.8,
        0.3,
        0.7,
        0.4,
        0.6,
    ]
    wv = [
        0.3,
        0.3,
        0.3,
        0.3,
        0.7,
        0.7,
        0.7,
        0.7,
        0.4,
        0.4,
        0.4,
        0.4,
        0.8,
        0.8,
        0.8,
        0.8,
    ]
    wo = [
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        1.0,
        1.0,
        0.0,
        0.5,
        0.5,
        0.5,
        0.5,
        0.1,
        0.2,
        0.3,
        0.4,
    ]
    x_data = [0.1 * i - 0.3 for i in range(batch * seq * d_model)]

    mha_pyc = pycoeus.MultiHeadAttention(
        d_model=d_model, num_heads=num_heads, bias=False
    )
    mha_pyc.w_q.data = wq
    mha_pyc.w_k.data = wk
    mha_pyc.w_v.data = wv
    mha_pyc.w_o.data = wo
    x_pyc = pycoeus.Tensor(x_data, [batch, seq, d_model], requires_grad=True)
    out_pyc = mha_pyc.forward(x_pyc)
    out_pyc.backward()

    mha_t = torch.nn.MultiheadAttention(
        embed_dim=d_model,
        num_heads=num_heads,
        bias=False,
        batch_first=True,
        dtype=torch.float64,
    )
    with torch.no_grad():
        mha_t.in_proj_weight[:d_model, :] = torch.tensor(
            wq, dtype=torch.float64
        ).reshape(d_model, d_model)
        mha_t.in_proj_weight[d_model : 2 * d_model, :] = torch.tensor(
            wk, dtype=torch.float64
        ).reshape(d_model, d_model)
        mha_t.in_proj_weight[2 * d_model :, :] = torch.tensor(
            wv, dtype=torch.float64
        ).reshape(d_model, d_model)
        mha_t.out_proj.weight[:] = torch.tensor(wo, dtype=torch.float64).reshape(
            d_model, d_model
        )
    x_t = (
        torch.tensor(x_data, dtype=torch.float64)
        .reshape(batch, seq, d_model)
        .requires_grad_(True)
    )
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
# TransformerEncoderLayer / TransformerEncoder — shared PyTorch helper
# ---------------------------------------------------------------------------


def _torch_preln_layer_fwd(
    x_t: "torch.Tensor",
    layer: "pycoeus.TransformerEncoderLayer",
    d_model: int,
    num_heads: int,
) -> "torch.Tensor":
    """PyTorch Pre-LN encoder forward assembled from a pycoeus layer's weights.

    ``x_t``   – ``[batch, seq, d_model]`` float64 tensor.
    ``layer`` – a fully-stateful ``pycoeus.TransformerEncoderLayer``.

    Returns ``[batch, seq, d_model]`` float64 tensor.
    """
    d_ff = layer.d_ff
    wq = list(layer.self_attn.w_q.data)
    wk = list(layer.self_attn.w_k.data)
    wv = list(layer.self_attn.w_v.data)
    wo = list(layer.self_attn.w_o.data)
    gamma1 = list(layer.norm1.weight.data)
    beta1 = list(layer.norm1.bias.data)
    gamma2 = list(layer.norm2.weight.data)
    beta2 = list(layer.norm2.bias.data)
    wff1 = list(layer.ffn.linear1.weight.data)
    bff1 = list(layer.ffn.linear1.bias.data) if layer.ffn.linear1.bias else [0.0] * d_ff
    wff2 = list(layer.ffn.linear2.weight.data)
    bff2 = (
        list(layer.ffn.linear2.bias.data) if layer.ffn.linear2.bias else [0.0] * d_model
    )

    mha_t = torch.nn.MultiheadAttention(
        embed_dim=d_model,
        num_heads=num_heads,
        bias=False,
        batch_first=True,
        dtype=torch.float64,
    )
    with torch.no_grad():
        mha_t.in_proj_weight[:d_model, :] = torch.tensor(
            wq, dtype=torch.float64
        ).reshape(d_model, d_model)
        mha_t.in_proj_weight[d_model : 2 * d_model, :] = torch.tensor(
            wk, dtype=torch.float64
        ).reshape(d_model, d_model)
        mha_t.in_proj_weight[2 * d_model :, :] = torch.tensor(
            wv, dtype=torch.float64
        ).reshape(d_model, d_model)
        mha_t.out_proj.weight[:] = torch.tensor(wo, dtype=torch.float64).reshape(
            d_model, d_model
        )
    mha_t.in_proj_bias = None
    mha_t.out_proj.bias = None

    ln1_t = torch.nn.LayerNorm(d_model, eps=1e-5, dtype=torch.float64)
    ln2_t = torch.nn.LayerNorm(d_model, eps=1e-5, dtype=torch.float64)
    with torch.no_grad():
        ln1_t.weight[:] = torch.tensor(gamma1, dtype=torch.float64)
        ln1_t.bias[:] = torch.tensor(beta1, dtype=torch.float64)
        ln2_t.weight[:] = torch.tensor(gamma2, dtype=torch.float64)
        ln2_t.bias[:] = torch.tensor(beta2, dtype=torch.float64)

    ff1_t = torch.nn.Linear(d_model, d_ff, bias=True, dtype=torch.float64)
    ff2_t = torch.nn.Linear(d_ff, d_model, bias=True, dtype=torch.float64)
    with torch.no_grad():
        ff1_t.weight[:] = torch.tensor(wff1, dtype=torch.float64).reshape(d_ff, d_model)
        ff1_t.bias[:] = torch.tensor(bff1, dtype=torch.float64)
        ff2_t.weight[:] = torch.tensor(wff2, dtype=torch.float64).reshape(d_model, d_ff)
        ff2_t.bias[:] = torch.tensor(bff2, dtype=torch.float64)

    normed1 = ln1_t(x_t)
    attn_out, _ = mha_t(normed1, normed1, normed1, need_weights=False)
    x1_t = x_t + attn_out
    normed2 = ln2_t(x1_t)
    ffn_out = ff2_t(torch.nn.functional.gelu(ff1_t(normed2)))
    return x1_t + ffn_out


# ---------------------------------------------------------------------------
# TransformerEncoderLayer (Pre-LN) forward parity
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not hasattr(pycoeus, "TransformerEncoderLayer"),
    reason="pycoeus.TransformerEncoderLayer not available in this wheel build",
)
def test_transformer_encoder_layer_matches_pytorch() -> None:
    """Forward parity: TransformerEncoderLayer(d_model=4, H=2, d_ff=8, dropout=0).

    Pre-LN forward:
      x₁ = x + MHA(LN1(x))
      out = x₁ + FFN(LN2(x₁))

    Weights are extracted from the stateful pycoeus sub-modules and copied to
    individually assembled PyTorch components (same weight convention — no
    transposition needed).
    """
    d_model, num_heads = 4, 2
    batch, seq = 1, 3
    _ATOL_ENC = 2e-4

    tel = pycoeus.TransformerEncoderLayer(d_model=d_model, d_ff=8, num_heads=num_heads)

    x_data = [0.1 * i - 0.3 for i in range(batch * seq * d_model)]
    x_pyc = pycoeus.Tensor(x_data, [batch, seq, d_model], requires_grad=False)
    out_pyc = tel.forward(x_pyc)

    x_t = torch.tensor(x_data, dtype=torch.float64).reshape(batch, seq, d_model)
    out_t = _torch_preln_layer_fwd(x_t, tel, d_model, num_heads)

    _allclose(
        "encoder_layer_fwd",
        list(out_pyc.data),
        out_t.flatten().tolist(),
        atol=_ATOL_ENC,
    )


# ---------------------------------------------------------------------------
# TransformerEncoder (Pre-LN N-layer stack) forward parity
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not hasattr(pycoeus, "TransformerEncoder"),
    reason="pycoeus.TransformerEncoder not available in this wheel build",
)
def test_transformer_encoder_stack_matches_pytorch() -> None:
    """Forward parity: TransformerEncoder(d_model=4, H=2, d_ff=8, num_layers=2, dropout=0).

    Each stateful pycoeus layer is independently assembled as a PyTorch Pre-LN
    forward and chained sequentially.  Confirms both the weight-extraction path
    and the N-layer composition logic.
    """
    d_model, num_heads, num_layers = 4, 2, 2
    batch, seq = 1, 3
    _ATOL_ENC = 2e-4

    enc = pycoeus.TransformerEncoder(
        d_model=d_model,
        d_ff=8,
        num_heads=num_heads,
        num_layers=num_layers,
    )
    assert enc.num_layers == num_layers
    assert len(enc.parameters()) == 16 * num_layers

    x_data = [0.1 * i - 0.3 for i in range(batch * seq * d_model)]
    x_pyc = pycoeus.Tensor(x_data, [batch, seq, d_model], requires_grad=False)
    out_pyc = enc.forward(x_pyc)

    x_t = torch.tensor(x_data, dtype=torch.float64).reshape(batch, seq, d_model)
    for layer in enc.layers:
        x_t = _torch_preln_layer_fwd(x_t, layer, d_model, num_heads)

    _allclose(
        "encoder_stack_fwd", list(out_pyc.data), x_t.flatten().tolist(), atol=_ATOL_ENC
    )


# ── TransformerDecoder parity ────────────────────────────────────────────────


def _torch_preln_decoder_layer_fwd(
    tgt_t: "torch.Tensor",
    memory_t: "torch.Tensor",
    layer: "pycoeus.TransformerDecoderLayer",
    d_model: int,
    num_heads: int,
) -> "torch.Tensor":
    """PyTorch Pre-LN decoder forward from a pycoeus decoder layer's weights.

    Implements:
      x1  = tgt + self_attn(norm1(tgt))          # causal self-attention
      x2  = x1  + cross_attn(norm2(x1), memory)  # cross-attention
      out = x2  + ffn(norm3(x2))                 # position-wise FFN

    MHA biases are zero-initialised in coeus (additive no-ops), excluded here.
    """
    import torch.nn.functional as F

    d_ff = layer.d_ff
    seq_tgt = tgt_t.shape[1]

    sa_wq = list(layer.self_attn.w_q.data)
    sa_wk = list(layer.self_attn.w_k.data)
    sa_wv = list(layer.self_attn.w_v.data)
    sa_wo = list(layer.self_attn.w_o.data)
    ca_wq = list(layer.cross_attn.w_q.data)
    ca_wk = list(layer.cross_attn.w_k.data)
    ca_wv = list(layer.cross_attn.w_v.data)
    ca_wo = list(layer.cross_attn.w_o.data)
    gamma1 = list(layer.norm1.weight.data)
    beta1 = list(layer.norm1.bias.data)
    gamma2 = list(layer.norm2.weight.data)
    beta2 = list(layer.norm2.bias.data)
    gamma3 = list(layer.norm3.weight.data)
    beta3 = list(layer.norm3.bias.data)
    wff1 = list(layer.ffn.linear1.weight.data)
    bff1 = list(layer.ffn.linear1.bias.data) if layer.ffn.linear1.bias else [0.0] * d_ff
    wff2 = list(layer.ffn.linear2.weight.data)
    bff2 = (
        list(layer.ffn.linear2.bias.data) if layer.ffn.linear2.bias else [0.0] * d_model
    )

    def _make_mha(wq, wk, wv, wo):
        mha = torch.nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            bias=False,
            batch_first=True,
            dtype=torch.float64,
        )
        with torch.no_grad():
            mha.in_proj_weight[:d_model, :] = torch.tensor(
                wq, dtype=torch.float64
            ).reshape(d_model, d_model)
            mha.in_proj_weight[d_model : 2 * d_model, :] = torch.tensor(
                wk, dtype=torch.float64
            ).reshape(d_model, d_model)
            mha.in_proj_weight[2 * d_model :, :] = torch.tensor(
                wv, dtype=torch.float64
            ).reshape(d_model, d_model)
            mha.out_proj.weight[:] = torch.tensor(wo, dtype=torch.float64).reshape(
                d_model, d_model
            )
        mha.in_proj_bias = None
        mha.out_proj.bias = None
        return mha

    sa_mha = _make_mha(sa_wq, sa_wk, sa_wv, sa_wo)
    ca_mha = _make_mha(ca_wq, ca_wk, ca_wv, ca_wo)

    ln1 = torch.nn.LayerNorm(d_model, eps=1e-5, dtype=torch.float64)
    ln2 = torch.nn.LayerNorm(d_model, eps=1e-5, dtype=torch.float64)
    ln3 = torch.nn.LayerNorm(d_model, eps=1e-5, dtype=torch.float64)
    with torch.no_grad():
        ln1.weight[:] = torch.tensor(gamma1, dtype=torch.float64)
        ln1.bias[:] = torch.tensor(beta1, dtype=torch.float64)
        ln2.weight[:] = torch.tensor(gamma2, dtype=torch.float64)
        ln2.bias[:] = torch.tensor(beta2, dtype=torch.float64)
        ln3.weight[:] = torch.tensor(gamma3, dtype=torch.float64)
        ln3.bias[:] = torch.tensor(beta3, dtype=torch.float64)

    ff1 = torch.nn.Linear(d_model, d_ff, bias=True, dtype=torch.float64)
    ff2 = torch.nn.Linear(d_ff, d_model, bias=True, dtype=torch.float64)
    with torch.no_grad():
        ff1.weight[:] = torch.tensor(wff1, dtype=torch.float64).reshape(d_ff, d_model)
        ff1.bias[:] = torch.tensor(bff1, dtype=torch.float64)
        ff2.weight[:] = torch.tensor(wff2, dtype=torch.float64).reshape(d_model, d_ff)
        ff2.bias[:] = torch.tensor(bff2, dtype=torch.float64)

    # causal mask: True = mask out (future positions)
    causal = torch.triu(torch.ones(seq_tgt, seq_tgt, dtype=torch.bool), diagonal=1)

    normed1 = ln1(tgt_t)
    sa_out, _ = sa_mha(normed1, normed1, normed1, attn_mask=causal, need_weights=False)
    x1 = tgt_t + sa_out
    ca_out, _ = ca_mha(ln2(x1), memory_t, memory_t, need_weights=False)
    x2 = x1 + ca_out
    ffn_out = ff2(F.gelu(ff1(ln3(x2))))
    return x2 + ffn_out


@pytest.mark.skipif(
    not hasattr(pycoeus, "TransformerDecoderLayer"),
    reason="pycoeus.TransformerDecoderLayer not available",
)
def test_transformer_decoder_layer_matches_pytorch() -> None:
    """Forward parity: TransformerDecoderLayer(d_model=4, H=2, d_ff=8, dropout=0)."""
    d_model, num_heads = 4, 2
    batch, seq_tgt, seq_src = 1, 3, 5
    _ATOL = 2e-4

    dec = pycoeus.TransformerDecoderLayer(d_model=d_model, d_ff=8, num_heads=num_heads)
    assert dec.num_heads == num_heads
    assert dec.d_model == d_model
    assert len(dec.parameters()) == 26

    tgt_data = [0.1 * i - 0.3 for i in range(batch * seq_tgt * d_model)]
    mem_data = [0.05 * i for i in range(batch * seq_src * d_model)]
    tgt_pyc = pycoeus.Tensor(tgt_data, [batch, seq_tgt, d_model], requires_grad=False)
    mem_pyc = pycoeus.Tensor(mem_data, [batch, seq_src, d_model], requires_grad=False)
    out_pyc = dec.forward(tgt_pyc, mem_pyc)

    tgt_t = torch.tensor(tgt_data, dtype=torch.float64).reshape(batch, seq_tgt, d_model)
    mem_t = torch.tensor(mem_data, dtype=torch.float64).reshape(batch, seq_src, d_model)
    out_t = _torch_preln_decoder_layer_fwd(tgt_t, mem_t, dec, d_model, num_heads)

    _allclose("dec_layer_fwd", list(out_pyc.data), out_t.flatten().tolist(), atol=_ATOL)


@pytest.mark.skipif(
    not hasattr(pycoeus, "TransformerDecoder"),
    reason="pycoeus.TransformerDecoder not available",
)
def test_transformer_decoder_stack_matches_pytorch() -> None:
    """Forward parity: TransformerDecoder(d_model=4, H=2, d_ff=8, num_layers=2, dropout=0)."""
    d_model, num_heads, num_layers = 4, 2, 2
    batch, seq_tgt, seq_src = 1, 3, 5
    _ATOL = 2e-4

    dec = pycoeus.TransformerDecoder(
        d_model=d_model,
        d_ff=8,
        num_heads=num_heads,
        num_layers=num_layers,
    )
    assert dec.num_layers == num_layers
    assert len(dec.parameters()) == 26 * num_layers

    tgt_data = [0.1 * i - 0.3 for i in range(batch * seq_tgt * d_model)]
    mem_data = [0.05 * i for i in range(batch * seq_src * d_model)]
    tgt_pyc = pycoeus.Tensor(tgt_data, [batch, seq_tgt, d_model], requires_grad=False)
    mem_pyc = pycoeus.Tensor(mem_data, [batch, seq_src, d_model], requires_grad=False)
    out_pyc = dec.forward(tgt_pyc, mem_pyc)

    tgt_t = torch.tensor(tgt_data, dtype=torch.float64).reshape(batch, seq_tgt, d_model)
    mem_t = torch.tensor(mem_data, dtype=torch.float64).reshape(batch, seq_src, d_model)
    for layer in dec.layers:
        tgt_t = _torch_preln_decoder_layer_fwd(tgt_t, mem_t, layer, d_model, num_heads)

    _allclose(
        "decoder_stack_fwd", list(out_pyc.data), tgt_t.flatten().tolist(), atol=_ATOL
    )


# ── Transformer seq2seq composition test ─────────────────────────────────────


@pytest.mark.skipif(
    not hasattr(pycoeus, "Transformer"),
    reason="pycoeus.Transformer not available",
)
def test_transformer_seq2seq_composition() -> None:
    """Transformer.forward(src, tgt) == encoder.forward(src) → decoder.forward(tgt, memory).

    Confirms the seq2seq chaining is bitwise-identical to manual composition
    via the stored sub-modules, and that parameter count is 16*E + 26*D.
    """
    d_model, num_heads, num_enc, num_dec = 4, 2, 1, 1
    batch, seq_src, seq_tgt = 1, 5, 3

    t = pycoeus.Transformer(
        d_model=d_model,
        d_ff=8,
        num_heads=num_heads,
        num_enc_layers=num_enc,
        num_dec_layers=num_dec,
    )
    assert t.num_enc_layers == num_enc
    assert t.num_dec_layers == num_dec
    assert t.d_model == d_model
    assert len(t.parameters()) == 16 * num_enc + 26 * num_dec

    src_data = [0.05 * i for i in range(batch * seq_src * d_model)]
    tgt_data = [0.1 * i - 0.3 for i in range(batch * seq_tgt * d_model)]
    src_pyc = pycoeus.Tensor(src_data, [batch, seq_src, d_model], requires_grad=False)
    tgt_pyc = pycoeus.Tensor(tgt_data, [batch, seq_tgt, d_model], requires_grad=False)

    # Full transformer forward
    out_t = t.forward(src_pyc, tgt_pyc)

    # Manual composition via stored sub-modules
    memory = t.encoder.forward(src_pyc)
    out_manual = t.decoder.forward(tgt_pyc, memory)

    # Same computation path → bitwise identical (tolerance = 1e-12)
    _allclose(
        "transformer_seq2seq", list(out_t.data), list(out_manual.data), atol=1e-12
    )


# ---------------------------------------------------------------------------
# RNN cell PyTorch parity (MS-132)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not hasattr(pycoeus, "LSTMCell"),
    reason="pycoeus.LSTMCell not available",
)
def test_lstm_cell_step_matches_pytorch() -> None:
    """PyTorch parity for LSTMCell.step.

    Weight-injection approach: copy w_ih/b_ih/w_hh/b_hh from a freshly
    constructed pycoeus.LSTMCell (zero-initialized biases) into a PyTorch
    LSTMCell; run one step on both and compare h_new and c_new.

    Gate ordering matches: both coeus and PyTorch use [i, f, g, o] for
    weight_ih [4H, I] and weight_hh [4H, H].
    """
    input_size, hidden_size = 4, 6
    batch = 2

    lstm = pycoeus.LSTMCell(input_size=input_size, hidden_size=hidden_size, bias=True)
    assert len(lstm.parameters()) == 4  # w_ih, w_hh, b_ih, b_hh

    x_data = [0.05 * i for i in range(batch * input_size)]
    h_data = [0.0] * (batch * hidden_size)
    c_data = [0.0] * (batch * hidden_size)

    x_pyc = pycoeus.Tensor(x_data, [batch, input_size], requires_grad=False)
    h_pyc = pycoeus.Tensor(h_data, [batch, hidden_size], requires_grad=False)
    c_pyc = pycoeus.Tensor(c_data, [batch, hidden_size], requires_grad=False)

    h_new_pyc, c_new_pyc = lstm.step(x_pyc, h_pyc, c_pyc)

    torch_lstm = torch.nn.LSTMCell(input_size, hidden_size).double()
    with torch.no_grad():
        torch_lstm.weight_ih[:] = torch.tensor(
            list(lstm.w_ih.data), dtype=torch.float64
        ).reshape(4 * hidden_size, input_size)
        torch_lstm.weight_hh[:] = torch.tensor(
            list(lstm.w_hh.data), dtype=torch.float64
        ).reshape(4 * hidden_size, hidden_size)
        torch_lstm.bias_ih[:] = (
            torch.tensor(list(lstm.b_ih.data), dtype=torch.float64)
            if lstm.b_ih is not None
            else torch.zeros(4 * hidden_size, dtype=torch.float64)
        )
        torch_lstm.bias_hh[:] = (
            torch.tensor(list(lstm.b_hh.data), dtype=torch.float64)
            if lstm.b_hh is not None
            else torch.zeros(4 * hidden_size, dtype=torch.float64)
        )

    x_t = torch.tensor(x_data, dtype=torch.float64).reshape(batch, input_size)
    h_t = torch.tensor(h_data, dtype=torch.float64).reshape(batch, hidden_size)
    c_t = torch.tensor(c_data, dtype=torch.float64).reshape(batch, hidden_size)

    h_new_t, c_new_t = torch_lstm(x_t, (h_t, c_t))

    _allclose(
        "lstm_h_new", list(h_new_pyc.data), h_new_t.flatten().tolist(), atol=1e-10
    )
    _allclose(
        "lstm_c_new", list(c_new_pyc.data), c_new_t.flatten().tolist(), atol=1e-10
    )


@pytest.mark.skipif(
    not hasattr(pycoeus, "GRUCell"),
    reason="pycoeus.GRUCell not available",
)
def test_gru_cell_step_matches_pytorch() -> None:
    """PyTorch parity for GRUCell.step.

    Gate ordering: both coeus and PyTorch use [r, z, n] for
    weight_ih [3H, I] and weight_hh [3H, H].  The new-gate formula
    n = tanh(W_in@x + b_in + r * (W_hn@h + b_hn)) matches in both.
    """
    input_size, hidden_size = 4, 6
    batch = 2

    gru = pycoeus.GRUCell(input_size=input_size, hidden_size=hidden_size, bias=True)
    assert len(gru.parameters()) == 4  # w_ih, w_hh, b_ih, b_hh

    x_data = [0.05 * i for i in range(batch * input_size)]
    h_data = [0.0] * (batch * hidden_size)

    x_pyc = pycoeus.Tensor(x_data, [batch, input_size], requires_grad=False)
    h_pyc = pycoeus.Tensor(h_data, [batch, hidden_size], requires_grad=False)

    h_new_pyc = gru.step(x_pyc, h_pyc)

    torch_gru = torch.nn.GRUCell(input_size, hidden_size).double()
    with torch.no_grad():
        torch_gru.weight_ih[:] = torch.tensor(
            list(gru.w_ih.data), dtype=torch.float64
        ).reshape(3 * hidden_size, input_size)
        torch_gru.weight_hh[:] = torch.tensor(
            list(gru.w_hh.data), dtype=torch.float64
        ).reshape(3 * hidden_size, hidden_size)
        torch_gru.bias_ih[:] = (
            torch.tensor(list(gru.b_ih.data), dtype=torch.float64)
            if gru.b_ih is not None
            else torch.zeros(3 * hidden_size, dtype=torch.float64)
        )
        torch_gru.bias_hh[:] = (
            torch.tensor(list(gru.b_hh.data), dtype=torch.float64)
            if gru.b_hh is not None
            else torch.zeros(3 * hidden_size, dtype=torch.float64)
        )

    x_t = torch.tensor(x_data, dtype=torch.float64).reshape(batch, input_size)
    h_t = torch.tensor(h_data, dtype=torch.float64).reshape(batch, hidden_size)

    h_new_t = torch_gru(x_t, h_t)

    _allclose("gru_h_new", list(h_new_pyc.data), h_new_t.flatten().tolist(), atol=1e-10)


# ── Optimizer step PyTorch parity ─────────────────────────────────────────────


def test_sgd_step_matches_pytorch() -> None:
    """SGD vanilla step (momentum=0) parity against torch.optim.SGD.

    Setup: w=[1.0], target=[0.0], mse_loss → loss=w², grad=2w=2.0.
    SGD(lr=0.1): w_new = w − lr·grad = 1.0 − 0.1·2.0 = 0.8.
    Evidence tier: differential / empirical (compared to PyTorch reference).
    """
    lr = 0.1

    w_pyc = pycoeus.Tensor([1.0], [1], requires_grad=True)
    target_pyc = pycoeus.Tensor([0.0], [1])
    loss_pyc = pycoeus.mse_loss(w_pyc, target_pyc)
    loss_pyc.backward()  # grad = 2 * 1.0 / 1 = 2.0
    opt_pyc = pycoeus.SGD([w_pyc], lr=lr, momentum=0.0)
    opt_pyc.step()

    w_t = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    loss_t = torch.nn.functional.mse_loss(w_t, torch.zeros(1, dtype=torch.float64))
    loss_t.backward()
    opt_t = torch.optim.SGD([w_t], lr=lr, momentum=0.0)
    opt_t.step()

    _allclose("sgd_w", list(w_pyc.data), w_t.detach().flatten().tolist(), atol=1e-10)


def test_adam_step_matches_pytorch() -> None:
    """Adam first-step parity against torch.optim.Adam.

    Setup: w=[1.0], target=[0.0], mse_loss → grad=2.0.
    Adam(lr=1e-2): m̂=2.0, v̂=4.0, step≈lr → w_new≈0.99.
    Evidence tier: differential / empirical (compared to PyTorch reference).
    """
    lr = 1e-2

    w_pyc = pycoeus.Tensor([1.0], [1], requires_grad=True)
    target_pyc = pycoeus.Tensor([0.0], [1])
    loss_pyc = pycoeus.mse_loss(w_pyc, target_pyc)
    loss_pyc.backward()  # grad = 2.0
    opt_pyc = pycoeus.Adam([w_pyc], lr=lr, beta1=0.9, beta2=0.999, eps=1e-8)
    opt_pyc.step()

    w_t = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    loss_t = torch.nn.functional.mse_loss(w_t, torch.zeros(1, dtype=torch.float64))
    loss_t.backward()
    opt_t = torch.optim.Adam([w_t], lr=lr, betas=(0.9, 0.999), eps=1e-8)
    opt_t.step()

    _allclose("adam_w", list(w_pyc.data), w_t.detach().flatten().tolist(), atol=1e-10)


def test_adamw_step_matches_pytorch() -> None:
    """AdamW first-step parity against torch.optim.AdamW.

    AdamW decouples weight decay: p = p − lr·(m̂/(√v̂+ε) + λ·p).
    Setup: w=[1.0], target=[0.0], mse_loss → grad=2.0, wd=0.01.
    Evidence tier: differential / empirical (compared to PyTorch reference).
    """
    lr = 1e-2
    wd = 0.01

    w_pyc = pycoeus.Tensor([1.0], [1], requires_grad=True)
    target_pyc = pycoeus.Tensor([0.0], [1])
    loss_pyc = pycoeus.mse_loss(w_pyc, target_pyc)
    loss_pyc.backward()  # grad = 2.0
    opt_pyc = pycoeus.AdamW([w_pyc], lr=lr, weight_decay=wd)
    opt_pyc.step()

    w_t = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    loss_t = torch.nn.functional.mse_loss(w_t, torch.zeros(1, dtype=torch.float64))
    loss_t.backward()
    opt_t = torch.optim.AdamW([w_t], lr=lr, weight_decay=wd)
    opt_t.step()

    _allclose("adamw_w", list(w_pyc.data), w_t.detach().flatten().tolist(), atol=1e-10)


# ── Bilinear PyTorch parity ───────────────────────────────────────────────────


def test_bilinear_forward_matches_pytorch() -> None:
    """Bilinear forward parity against torch.nn.Bilinear.

    Weight layout [out, in1, in2] is identical between pycoeus and PyTorch,
    so weights are copied flat without transposition.
    Evidence tier: differential / empirical.
    """
    in1, in2, out, batch = 3, 4, 2, 5

    bil = pycoeus.Bilinear(in1, in2, out, bias=True)
    w_flat = list(bil.weight.data)  # [out*in1*in2] row-major
    b_flat = list(bil.bias.data)  # [out]

    x1_data = [float(i) * 0.1 for i in range(batch * in1)]
    x2_data = [float(i) * 0.05 - 0.5 for i in range(batch * in2)]

    # pycoeus forward
    x1_pyc = pycoeus.Tensor(x1_data, [batch, in1])
    x2_pyc = pycoeus.Tensor(x2_data, [batch, in2])
    out_pyc = bil.bilinear_forward(x1_pyc, x2_pyc)

    # PyTorch reference (double precision to match pycoeus f64)
    torch_bil = torch.nn.Bilinear(in1, in2, out, bias=True).double()
    with torch.no_grad():
        torch_bil.weight[:] = torch.tensor(w_flat, dtype=torch.float64).reshape(
            out, in1, in2
        )
        torch_bil.bias[:] = torch.tensor(b_flat, dtype=torch.float64)
    x1_t = torch.tensor(x1_data, dtype=torch.float64).reshape(batch, in1)
    x2_t = torch.tensor(x2_data, dtype=torch.float64).reshape(batch, in2)
    out_t = torch_bil(x1_t, x2_t)

    _allclose(
        "bilinear_out",
        list(out_pyc.data),
        out_t.flatten().tolist(),
        atol=1e-10,
    )


# ── RMSNorm PyTorch parity ─────────────────────────────────────────────────


def test_rmsnorm_matches_pytorch() -> None:
    """Forward and gradient parity: RMSNorm(4, eps=1e-8), no bias.

    PyTorch 2.12 does not yet ship ``torch.nn.RMSNorm`` as stable, so the
    oracle is the canonical formula
    ::

        rms = sqrt(mean(x * x, dim=-1, keepdim=True) + eps)
        y = (x / rms) * gamma

    which is exactly what ``torch.nn.RMSNorm`` (since PyTorch 2.4) computes.
    Forward output, input gradient, and gamma gradient agree to bitwise
    precision between pycoeus and the formula at f64.

    Evidence tier: differential / empirical against PyTorch's RMSNorm
    canonical formula at f64. Tolerance 1e-10.
    """
    d_model = 4
    data = [1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.5, 3.0]
    gamma = [1.2, 0.8, 1.0, 0.9]
    eps = 1e-8

    rn_pyc = pycoeus.RMSNorm(d_model, eps)
    rn_pyc.weight.data = gamma
    x_pyc = pycoeus.Tensor(data, [2, d_model], requires_grad=True)
    y_pyc = rn_pyc.forward(x_pyc)
    loss_pyc = pycoeus.sum(y_pyc)
    loss_pyc.backward()

    x_t = (
        torch.tensor(data, dtype=torch.float64).reshape(2, d_model).requires_grad_(True)
    )
    w_t = (
        torch.tensor(gamma, dtype=torch.float64)
        .reshape(1, d_model)
        .requires_grad_(True)
    )
    rms_t = torch.sqrt((x_t * x_t).mean(dim=-1, keepdim=True) + eps)
    y_t = (x_t / rms_t) * w_t
    loss_t = y_t.sum()
    loss_t.backward()

    _allclose(
        "rmsnorm_y", pyc_y_data := list(y_pyc.data), y_t.flatten().tolist(), atol=1e-10
    )
    _allclose("rmsnorm_dx", list(x_pyc.grad), x_t.grad.flatten().tolist(), atol=1e-10)
    _allclose(
        "rmsnorm_dgamma",
        list(rn_pyc.weight.grad),
        w_t.grad.flatten().tolist(),
        atol=1e-10,
    )


# ── Embedding PyTorch parity ────────────────────────────────────────────────


def test_embedding_matches_pytorch() -> None:
    """Forward and gradient parity: Embedding(num=6, dim=4).

    Sets a non-trivial weight matrix, forwards on a fixed index sequence,
    computes ``mse_loss`` against a target like for like ``torch.nn.Embedding``.
    PyTorch's ``nn.Embedding`` uses sparse-index backward; ``torch.autograd``
    gathers gradients into the weight matrix only at the rows that were
    looked up.  Pycoeus' embedding parity follows the same contract.

    Evidence tier: differential / empirical against ``torch.nn.Embedding`` at f64.
    Tolerance 1e-10.
    """
    num_embeddings, embedding_dim = 6, 4
    weight_data = [
        0.5,
        -0.5,
        1.0,
        0.0,
        0.1,
        0.2,
        0.3,
        -0.1,
        -0.2,
        0.5,
        0.0,
        1.0,
        1.0,
        -1.0,
        0.2,
        0.8,
        0.7,
        0.3,
        -0.7,
        0.4,
        0.0,
        1.0,
        0.5,
        -0.5,
    ]
    indices = [0, 2, 4, 1, 3, 5]
    target_data = [1.0] * (len(indices) * embedding_dim)  # [6, 4]

    # pycoeus forward + backward
    emb_pyc = pycoeus.Embedding(num_embeddings, embedding_dim)
    emb_pyc.weight.data = weight_data
    idx_pyc = pycoeus.Tensor(
        [float(i) for i in indices], [len(indices)], requires_grad=False
    )
    y_pyc = emb_pyc.forward(idx_pyc)
    target_pyc = pycoeus.Tensor(target_data, [len(indices), embedding_dim])
    loss_pyc = pycoeus.mse_loss(y_pyc, target_pyc)
    loss_pyc.backward()

    # PyTorch reference
    emb_t = torch.nn.Embedding(num_embeddings, embedding_dim).double()
    with torch.no_grad():
        emb_t.weight.copy_(
            torch.tensor(weight_data, dtype=torch.float64).reshape(
                num_embeddings, embedding_dim
            )
        )
    idx_t = torch.tensor(indices, dtype=torch.long)
    y_t = emb_t(idx_t)
    target_t = torch.tensor(target_data, dtype=torch.float64).reshape(
        len(indices), embedding_dim
    )
    loss_t = torch.nn.functional.mse_loss(y_t, target_t)
    loss_t.backward()

    _allclose("embedding_y", list(y_pyc.data), y_t.flatten().tolist(), atol=1e-10)
    _allclose(
        "embedding_dweight",
        list(emb_pyc.weight.grad),
        emb_t.weight.grad.flatten().tolist(),
        atol=1e-10,
    )


# ── InstanceNorm{1d,2d,3d} PyTorch parity (MS-144) ──────────────────────────


def test_instancenorm2d_matches_pytorch() -> None:
    """Forward and gradient parity: InstanceNorm2d(2, eps=1e-5) on [N=2, C=2, H=2, W=2].

    Differential against ``torch.nn.functional.instance_norm`` at f64, with
    affine weight/bias injected from pycoeus into the PyTorch reference via
    ``F.instance_norm(..., weight, bias)``.  Weight layout
    ``[num_features]`` is identical between pycoeus and PyTorch.
    """
    n, c, h, w = 2, 2, 2, 2
    eps = 1e-5
    data = [0.1 * i - 0.2 for i in range(n * c * h * w)]
    gamma = [1.5, 0.5]
    beta = [0.1, -0.1]

    in_pyc = pycoeus.InstanceNorm2d(c, eps=eps)
    in_pyc.weight.data = gamma
    in_pyc.bias.data = beta
    x_pyc = pycoeus.Tensor(data, [n, c, h, w], requires_grad=True)
    out_pyc = in_pyc.forward(x_pyc)
    out_pyc.sum().backward()

    x_t = (
        torch.tensor(data, dtype=torch.float64).reshape(n, c, h, w).requires_grad_(True)
    )
    g_t = torch.tensor(gamma, dtype=torch.float64, requires_grad=True)
    b_t = torch.tensor(beta, dtype=torch.float64, requires_grad=True)
    out_t = torch.nn.functional.instance_norm(
        x_t,
        weight=g_t,
        bias=b_t,
        use_input_stats=True,
        eps=eps,
    )
    out_t.sum().backward()

    _allclose("in2d_out", list(out_pyc.data), out_t.flatten().tolist(), atol=1e-10)
    _allclose("in2d_dx", list(x_pyc.grad), x_t.grad.flatten().tolist(), atol=1e-10)
    _allclose(
        "in2d_dgamma",
        list(in_pyc.weight.grad),
        g_t.grad.flatten().tolist(),
        atol=1e-10,
    )
    _allclose(
        "in2d_dbeta",
        list(in_pyc.bias.grad),
        b_t.grad.flatten().tolist(),
        atol=1e-10,
    )


def test_instancenorm1d_matches_pytorch() -> None:
    """Forward and gradient parity: InstanceNorm1d(2, eps=1e-5) on [N=1, C=2, L=4].

    Differential against ``torch.nn.functional.instance_norm`` at f64.
    """
    n, c, l = 1, 2, 4
    eps = 1e-5
    data = [0.1 * i - 0.1 for i in range(n * c * l)]
    gamma = [1.2, 0.8]
    beta = [0.05, -0.05]

    in_pyc = pycoeus.InstanceNorm1d(c, eps=eps)
    in_pyc.weight.data = gamma
    in_pyc.bias.data = beta
    x_pyc = pycoeus.Tensor(data, [n, c, l], requires_grad=True)
    out_pyc = in_pyc.forward(x_pyc)
    out_pyc.sum().backward()

    x_t = torch.tensor(data, dtype=torch.float64).reshape(n, c, l).requires_grad_(True)
    g_t = torch.tensor(gamma, dtype=torch.float64, requires_grad=True)
    b_t = torch.tensor(beta, dtype=torch.float64, requires_grad=True)
    out_t = torch.nn.functional.instance_norm(
        x_t,
        weight=g_t,
        bias=b_t,
        use_input_stats=True,
        eps=eps,
    )
    out_t.sum().backward()

    _allclose("in1d_out", list(out_pyc.data), out_t.flatten().tolist(), atol=1e-10)
    _allclose("in1d_dx", list(x_pyc.grad), x_t.grad.flatten().tolist(), atol=1e-10)
    _allclose(
        "in1d_dgamma",
        list(in_pyc.weight.grad),
        g_t.grad.flatten().tolist(),
        atol=1e-10,
    )
    _allclose(
        "in1d_dbeta",
        list(in_pyc.bias.grad),
        b_t.grad.flatten().tolist(),
        atol=1e-10,
    )


def test_instancenorm3d_matches_pytorch() -> None:
    """Forward and gradient parity: InstanceNorm3d(2, eps=1e-5) on [N=1, C=2, D=2, H=2, W=2].

    Differential against ``torch.nn.functional.instance_norm`` at f64.
    """
    n, c, d, h, w = 1, 2, 2, 2, 2
    eps = 1e-5
    data = [0.05 * i - 0.1 for i in range(n * c * d * h * w)]
    gamma = [1.3, 0.7]
    beta = [0.2, -0.2]

    in_pyc = pycoeus.InstanceNorm3d(c, eps=eps)
    in_pyc.weight.data = gamma
    in_pyc.bias.data = beta
    x_pyc = pycoeus.Tensor(data, [n, c, d, h, w], requires_grad=True)
    out_pyc = in_pyc.forward(x_pyc)
    out_pyc.sum().backward()

    x_t = (
        torch.tensor(data, dtype=torch.float64)
        .reshape(n, c, d, h, w)
        .requires_grad_(True)
    )
    g_t = torch.tensor(gamma, dtype=torch.float64, requires_grad=True)
    b_t = torch.tensor(beta, dtype=torch.float64, requires_grad=True)
    out_t = torch.nn.functional.instance_norm(
        x_t,
        weight=g_t,
        bias=b_t,
        use_input_stats=True,
        eps=eps,
    )
    out_t.sum().backward()

    _allclose("in3d_out", list(out_pyc.data), out_t.flatten().tolist(), atol=1e-10)
    _allclose("in3d_dx", list(x_pyc.grad), x_t.grad.flatten().tolist(), atol=1e-10)
    _allclose(
        "in3d_dgamma",
        list(in_pyc.weight.grad),
        g_t.grad.flatten().tolist(),
        atol=1e-10,
    )
    _allclose(
        "in3d_dbeta",
        list(in_pyc.bias.grad),
        b_t.grad.flatten().tolist(),
        atol=1e-10,
    )


# ── Optimizer step parity closure (RMSProp + AdaGrad) (MS-144) ──────────────


def test_rmsprop_step_matches_pytorch() -> None:
    """RMSProp first-step parity against torch.optim.RMSprop.

    Setup: w=[1.0], target=[0.0], mse_loss → grad=2.0.
    RMSProp(lr=1e-2, alpha=0.99, eps=1e-8) update:
      square_avg = α·0 + (1−α)·g² = 0.01·4 = 0.04
      lr_step = lr / (sqrt(square_avg) + eps) = 0.01 / (0.2 + 1e-8) ≈ 0.05
      w_new = w − lr_step ≈ 0.95.
    """
    lr = 1e-2
    alpha = 0.99
    eps = 1e-8

    w_pyc = pycoeus.Tensor([1.0], [1], requires_grad=True)
    target_pyc = pycoeus.Tensor([0.0], [1])
    loss_pyc = pycoeus.mse_loss(w_pyc, target_pyc)
    loss_pyc.backward()  # grad = 2.0
    opt_pyc = pycoeus.RMSProp([w_pyc], lr=lr, alpha=alpha, eps=eps)
    opt_pyc.step()

    w_t = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    loss_t = torch.nn.functional.mse_loss(w_t, torch.zeros(1, dtype=torch.float64))
    loss_t.backward()
    opt_t = torch.optim.RMSprop(
        [w_t], lr=lr, alpha=alpha, eps=eps, momentum=0.0, centered=False
    )
    opt_t.step()

    _allclose(
        "rmsprop_w", list(w_pyc.data), w_t.detach().flatten().tolist(), atol=1e-10
    )


def test_adagrad_step_matches_pytorch() -> None:
    """AdaGrad first-step parity against torch.optim.Adagrad.

    Setup: w=[1.0], target=[0.0], mse_loss → grad=2.0.
    AdaGrad(lr=1e-2, eps=1e-10):
      accumulated = g² = 4
      update = lr · g / (sqrt(accumulated) + eps) = 0.01 · 2 / (2 + 1e-10) ≈ 0.01
      w_new = w − update ≈ 0.99.
    """
    lr = 1e-2
    eps = 1e-10

    w_pyc = pycoeus.Tensor([1.0], [1], requires_grad=True)
    target_pyc = pycoeus.Tensor([0.0], [1])
    loss_pyc = pycoeus.mse_loss(w_pyc, target_pyc)
    loss_pyc.backward()  # grad = 2.0
    opt_pyc = pycoeus.AdaGrad([w_pyc], lr=lr, eps=eps)
    opt_pyc.step()

    w_t = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    loss_t = torch.nn.functional.mse_loss(w_t, torch.zeros(1, dtype=torch.float64))
    loss_t.backward()
    opt_t = torch.optim.Adagrad([w_t], lr=lr, eps=eps, weight_decay=0.0)
    opt_t.step()

    _allclose(
        "adagrad_w", list(w_pyc.data), w_t.detach().flatten().tolist(), atol=1e-10
    )
