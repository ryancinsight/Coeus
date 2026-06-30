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


def test_bilinear_backward_matches_pytorch() -> None:
    """Backward parity: Bilinear(in1=3, in2=4, out=2, bias=True) — dweight + dbias.

    The pycoeus Bilinear is implemented by autograd-tracked composition
    (matmul + mul + sum_axis + cat + add), so no custom backward is needed.
    The differential check vs ``torch.nn.Bilinear`` at f64 exercises the full
    composition chain at machine precision.

    Evidence tier: differential / empirical against PyTorch's autograd at f64.
    Tolerance 1e-10.
    """
    in1, in2, out, batch = 3, 4, 2, 5

    bil = pycoeus.Bilinear(in1, in2, out, bias=True)
    w_flat = list(bil.weight.data)
    b_flat = list(bil.bias.data)

    x1_data = [float(i) * 0.1 for i in range(batch * in1)]
    x2_data = [float(i) * 0.05 - 0.5 for i in range(batch * in2)]

    # pycoeus forward + backward
    x1_pyc = pycoeus.Tensor(x1_data, [batch, in1], requires_grad=True)
    x2_pyc = pycoeus.Tensor(x2_data, [batch, in2], requires_grad=True)
    out_pyc = bil.bilinear_forward(x1_pyc, x2_pyc)
    loss_pyc = pycoeus.sum(out_pyc)
    loss_pyc.backward()

    # PyTorch reference (double precision; weight layout identical: [out, in1, in2])
    torch_bil = torch.nn.Bilinear(in1, in2, out, bias=True).double()
    with torch.no_grad():
        torch_bil.weight[:] = torch.tensor(w_flat, dtype=torch.float64).reshape(
            out, in1, in2
        )
        torch_bil.bias[:] = torch.tensor(b_flat, dtype=torch.float64)
    x1_t = (
        torch.tensor(x1_data, dtype=torch.float64)
        .reshape(batch, in1)
        .requires_grad_(True)
    )
    x2_t = (
        torch.tensor(x2_data, dtype=torch.float64)
        .reshape(batch, in2)
        .requires_grad_(True)
    )
    out_t = torch_bil(x1_t, x2_t)
    out_t.sum().backward()

    # dweight: pycoeus flat [out, in1, in2] == PyTorch weight.grad same layout.
    _allclose(
        "bilinear_bwd_dweight",
        list(bil.weight.grad),
        torch_bil.weight.grad.flatten().tolist(),
        atol=1e-10,
    )
    # dbias: [out] == [out].
    _allclose(
        "bilinear_bwd_dbias",
        list(bil.bias.grad),
        torch_bil.bias.grad.flatten().tolist(),
        atol=1e-10,
    )
    # dx1, dx2: parity of input gradients.
    _allclose(
        "bilinear_bwd_dx1",
        list(x1_pyc.grad),
        x1_t.grad.flatten().tolist(),
        atol=1e-10,
    )
    _allclose(
        "bilinear_bwd_dx2",
        list(x2_pyc.grad),
        x2_t.grad.flatten().tolist(),
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


# ── GroupNorm PyTorch parity (MS-149) ───────────────────────────────────────


def test_groupnorm_matches_pytorch() -> None:
    """Forward and gradient parity: GroupNorm(num_groups=2, C=4) on [N=2, C=4, H=2, W=2].

    Differential against ``torch.nn.functional.group_norm`` at f64, with affine
    weight/bias injected from pycoeus into the PyTorch reference.  GroupNorm
    normalizes over the C/G channels of each group plus all spatial positions;
    the affine ``weight``/``bias`` layout ``[num_features]`` is identical between
    pycoeus and PyTorch.
    """
    n, c, h, w = 2, 4, 2, 2
    num_groups = 2
    eps = 1e-5
    data = [0.1 * i - 0.5 for i in range(n * c * h * w)]
    gamma = [1.5, 0.5, 1.2, 0.8]
    beta = [0.1, -0.1, 0.2, -0.2]

    gn_pyc = pycoeus.GroupNorm(num_groups, c, eps=eps)
    gn_pyc.weight.data = gamma
    gn_pyc.bias.data = beta
    x_pyc = pycoeus.Tensor(data, [n, c, h, w], requires_grad=True)
    out_pyc = gn_pyc.forward(x_pyc)
    out_pyc.sum().backward()

    x_t = (
        torch.tensor(data, dtype=torch.float64).reshape(n, c, h, w).requires_grad_(True)
    )
    g_t = torch.tensor(gamma, dtype=torch.float64, requires_grad=True)
    b_t = torch.tensor(beta, dtype=torch.float64, requires_grad=True)
    out_t = torch.nn.functional.group_norm(
        x_t, num_groups, weight=g_t, bias=b_t, eps=eps
    )
    out_t.sum().backward()

    _allclose("gn_out", list(out_pyc.data), out_t.flatten().tolist(), atol=1e-10)
    _allclose("gn_dx", list(x_pyc.grad), x_t.grad.flatten().tolist(), atol=1e-10)
    _allclose(
        "gn_dgamma",
        list(gn_pyc.weight.grad),
        g_t.grad.flatten().tolist(),
        atol=1e-10,
    )
    _allclose(
        "gn_dbeta",
        list(gn_pyc.bias.grad),
        b_t.grad.flatten().tolist(),
        atol=1e-10,
    )


# ── MaxPool2d / AvgPool2d PyTorch parity (MS-151) ───────────────────────────


def test_maxpool2d_matches_pytorch() -> None:
    """Forward and input-gradient parity: MaxPool2d(k=2, stride=2) on [1, 2, 4, 4].

    Differential against ``torch.nn.functional.max_pool2d`` at f64, atol=1e-10.
    MaxPool routes the upstream gradient only to the argmax position in each
    window; the test exercises that routing via ``out.sum().backward()``.
    """
    n, c, h, w = 1, 2, 4, 4
    data = [0.1 * i - 0.7 for i in range(n * c * h * w)]

    x_pyc = pycoeus.Tensor(data, [n, c, h, w], requires_grad=True)
    out_pyc = pycoeus.MaxPool2d(2, stride=2, padding=0).forward(x_pyc)
    out_pyc.sum().backward()

    x_t = (
        torch.tensor(data, dtype=torch.float64).reshape(n, c, h, w).requires_grad_(True)
    )
    out_t = torch.nn.functional.max_pool2d(x_t, 2, stride=2, padding=0)
    out_t.sum().backward()

    _allclose("maxpool2d_out", list(out_pyc.data), out_t.flatten().tolist(), atol=1e-10)
    _allclose("maxpool2d_dx", list(x_pyc.grad), x_t.grad.flatten().tolist(), atol=1e-10)


def test_avgpool2d_matches_pytorch() -> None:
    """Forward and input-gradient parity: AvgPool2d(k=2, stride=2) on [1, 2, 4, 4].

    Differential against ``torch.nn.functional.avg_pool2d`` at f64, atol=1e-10.
    AvgPool distributes the upstream gradient uniformly (1/window_size) across
    each window; verified via ``out.sum().backward()``.
    """
    n, c, h, w = 1, 2, 4, 4
    data = [0.1 * i - 0.7 for i in range(n * c * h * w)]

    x_pyc = pycoeus.Tensor(data, [n, c, h, w], requires_grad=True)
    out_pyc = pycoeus.AvgPool2d(2, stride=2, padding=0).forward(x_pyc)
    out_pyc.sum().backward()

    x_t = (
        torch.tensor(data, dtype=torch.float64).reshape(n, c, h, w).requires_grad_(True)
    )
    out_t = torch.nn.functional.avg_pool2d(x_t, 2, stride=2, padding=0)
    out_t.sum().backward()

    _allclose("avgpool2d_out", list(out_pyc.data), out_t.flatten().tolist(), atol=1e-10)
    _allclose("avgpool2d_dx", list(x_pyc.grad), x_t.grad.flatten().tolist(), atol=1e-10)


# ── GlobalAvgPool2d / GlobalMaxPool2d PyTorch parity (MS-166) ────────────────


def test_global_avg_pool2d_matches_pytorch() -> None:
    """Forward and input-gradient parity: GlobalAvgPool2d on [2, 3, 4, 4].

    Differential against ``torch.nn.functional.adaptive_avg_pool2d(x, 1)`` at f64,
    atol=1e-10. Global average pooling reduces every spatial position to one value
    per channel (output ``[N, C, 1, 1]``); the backward distributes the upstream
    gradient uniformly (1/(H*W)) across the window.
    """
    n, c, h, w = 2, 3, 4, 4
    data = [0.1 * i - 1.0 for i in range(n * c * h * w)]

    x_pyc = pycoeus.Tensor(data, [n, c, h, w], requires_grad=True)
    out_pyc = pycoeus.GlobalAvgPool2d().forward(x_pyc)
    out_pyc.sum().backward()

    x_t = (
        torch.tensor(data, dtype=torch.float64).reshape(n, c, h, w).requires_grad_(True)
    )
    out_t = torch.nn.functional.adaptive_avg_pool2d(x_t, 1)
    out_t.sum().backward()

    _allclose("gap2d_out", list(out_pyc.data), out_t.flatten().tolist(), atol=1e-10)
    _allclose("gap2d_dx", list(x_pyc.grad), x_t.grad.flatten().tolist(), atol=1e-10)


def test_global_max_pool2d_matches_pytorch() -> None:
    """Forward and input-gradient parity: GlobalMaxPool2d on [2, 3, 4, 4].

    Differential against ``torch.nn.functional.adaptive_max_pool2d(x, 1)`` at f64,
    atol=1e-10. Global max pooling takes the maximum over all spatial positions
    per channel (output ``[N, C, 1, 1]``); the backward routes the upstream
    gradient to the single argmax position in each window.
    """
    n, c, h, w = 2, 3, 4, 4
    data = [0.1 * i - 1.0 for i in range(n * c * h * w)]

    x_pyc = pycoeus.Tensor(data, [n, c, h, w], requires_grad=True)
    out_pyc = pycoeus.GlobalMaxPool2d().forward(x_pyc)
    out_pyc.sum().backward()

    x_t = (
        torch.tensor(data, dtype=torch.float64).reshape(n, c, h, w).requires_grad_(True)
    )
    out_t = torch.nn.functional.adaptive_max_pool2d(x_t, 1)
    out_t.sum().backward()

    _allclose("gmp2d_out", list(out_pyc.data), out_t.flatten().tolist(), atol=1e-10)
    _allclose("gmp2d_dx", list(x_pyc.grad), x_t.grad.flatten().tolist(), atol=1e-10)


# ── Activation function PyTorch parity (SiLU/Mish/ELU/Softplus/LeakyReLU) (MS-167) ──


def _assert_activation_parity(name, pyc_fn, torch_fn, data, atol=1e-9):
    """Differential forward + input-gradient parity for an elementwise activation.

    Drives `out.sum().backward()` so the full elementwise derivative is compared
    against the PyTorch reference at f64.
    """
    x_pyc = pycoeus.Tensor(data, [len(data)], requires_grad=True)
    out_pyc = pyc_fn(x_pyc)
    out_pyc.sum().backward()

    x_t = torch.tensor(data, dtype=torch.float64, requires_grad=True)
    out_t = torch_fn(x_t)
    out_t.sum().backward()

    _allclose(f"{name}_out", list(out_pyc.data), out_t.detach().tolist(), atol=atol)
    _allclose(f"{name}_dx", list(x_pyc.grad), x_t.grad.tolist(), atol=atol)


# Mixed-sign inputs exercise both regimes of each nonlinearity. SiLU/Mish/ELU/
# Softplus are C1 everywhere (0.0 is safe to include); LeakyReLU has a kink at 0,
# where the subgradient convention is implementation-defined, so 0.0 is excluded.
_ACTIVATION_INPUT = [-2.0, -0.5, 0.0, 0.3, 1.5, 3.0]
_ACTIVATION_INPUT_NO_ZERO = [-2.0, -0.5, 0.3, 1.5, 3.0]


def test_silu_matches_pytorch() -> None:
    _assert_activation_parity(
        "silu", lambda x: pycoeus.silu(x), torch.nn.functional.silu, _ACTIVATION_INPUT
    )


def test_log_sigmoid_matches_pytorch() -> None:
    _assert_activation_parity(
        "log_sigmoid",
        lambda x: pycoeus.log_sigmoid(x),
        torch.nn.functional.logsigmoid,
        _ACTIVATION_INPUT,
    )


def test_tanhshrink_matches_pytorch() -> None:
    _assert_activation_parity(
        "tanhshrink",
        lambda x: pycoeus.tanhshrink(x),
        torch.nn.functional.tanhshrink,
        _ACTIVATION_INPUT,
    )


def test_mish_matches_pytorch() -> None:
    _assert_activation_parity(
        "mish", lambda x: pycoeus.mish(x), torch.nn.functional.mish, _ACTIVATION_INPUT
    )


def test_elu_matches_pytorch() -> None:
    _assert_activation_parity(
        "elu", lambda x: pycoeus.elu(x), torch.nn.functional.elu, _ACTIVATION_INPUT
    )


def test_softplus_matches_pytorch() -> None:
    _assert_activation_parity(
        "softplus",
        lambda x: pycoeus.softplus(x),
        torch.nn.functional.softplus,
        _ACTIVATION_INPUT,
    )


def test_leaky_relu_matches_pytorch() -> None:
    # Default negative slope 0.01 on both sides; 0.0 excluded (kink subgradient).
    _assert_activation_parity(
        "leaky_relu",
        lambda x: pycoeus.leaky_relu(x),
        torch.nn.functional.leaky_relu,
        _ACTIVATION_INPUT_NO_ZERO,
    )


# ── G-037 extended activation family PyTorch parity ────────────────────────
#
# Kink/subgradient notes (PyTorch subgradient conventions):
#   - Hardswish: gradient is (2x+3)/6 inside [-3, 3]; 0 at x<-3, 1 at x>3.
#     At x=-3 and x=3 the function is closed-interval continuous; gradient
#     is (2x+3)/6 which evaluates to -0.5 at x=-3 and 1.5 at x=3.
#   - Hardsigmoid: 1/6 inside (-3, 3); 0 at the open exterior. PyTorch
#     uses open-interval convention so x = -3, 3 are excluded.
#   - Hardtanh: 1 inside (min, max); 0 at saturating positions. The
#     kink endpoints (x = min, x = max) are excluded.
#   - Hardshrink / Softshrink: 1 if |x| > λ, else 0. The boundary
#     |x| = λ is excluded (PyTorch convention).
#   - Softsign: smooth everywhere, no kink exclusion.
#   - Threshold: PyTorch uses x > threshold, so x = threshold is in
#     the replacement region (gradient 0). The threshold is excluded.
#   - Celu: continuously differentiable (gradient = exp(x/α) for x<0,
#     gradient = 1 for x ≥ 0; both agree at x = 0). No exclusion needed.
#   - PReLU: 1 for x ≥ 0, α for x < 0. PyTorch passes gradient at x=0
#     (subgradient = 1). No exclusion needed for the default α = 0.25.

_HARDSWISH_INPUT = [-4.0, -3.0, -1.0, 0.0, 1.0, 3.0, 4.0]
_HARDSIGMOID_INPUT = [-4.0, -1.0, 0.0, 1.0, 4.0]  # exclude ±3
_HARDTANH_INPUT = [-2.0, -0.5, 0.0, 0.5, 2.0]  # exclude ±1
_HARDSHRINK_INPUT = [-2.0, -0.6, 0.0, 0.6, 2.0]  # exclude ±λ=0.5
_SOFTSHRINK_INPUT = [-2.0, -0.6, 0.0, 0.6, 2.0]
_SOFTSIGN_INPUT = [-2.0, -1.0, 0.0, 1.0, 2.0]
_THRESHOLD_INPUT = [-2.0, -0.5, 0.0, 0.5, 2.0]  # exclude x = threshold = 0
_CELU_INPUT = [-2.0, -1.0, 0.0, 1.0, 2.0]
_PRELU_INPUT = [-2.0, -1.0, 0.0, 1.0, 2.0]


def test_hardswish_matches_pytorch() -> None:
    _assert_activation_parity(
        "hardswish",
        lambda x: pycoeus.hardswish(x),
        torch.nn.functional.hardswish,
        _HARDSWISH_INPUT,
    )


def test_hardsigmoid_matches_pytorch() -> None:
    _assert_activation_parity(
        "hardsigmoid",
        lambda x: pycoeus.hardsigmoid(x),
        torch.nn.functional.hardsigmoid,
        _HARDSIGMOID_INPUT,
    )


def test_hardtanh_matches_pytorch() -> None:
    _assert_activation_parity(
        "hardtanh",
        lambda x: pycoeus.hardtanh(x, -1.0, 1.0),
        torch.nn.functional.hardtanh,
        _HARDTANH_INPUT,
    )


def test_hardshrink_matches_pytorch() -> None:
    _assert_activation_parity(
        "hardshrink",
        lambda x: pycoeus.hardshrink(x, 0.5),
        lambda x: torch.nn.functional.hardshrink(x, lambd=0.5),
        _HARDSHRINK_INPUT,
    )


def test_softshrink_matches_pytorch() -> None:
    _assert_activation_parity(
        "softshrink",
        lambda x: pycoeus.softshrink(x, 0.5),
        lambda x: torch.nn.functional.softshrink(x, lambd=0.5),
        _SOFTSHRINK_INPUT,
    )


def test_softsign_matches_pytorch() -> None:
    _assert_activation_parity(
        "softsign",
        lambda x: pycoeus.softsign(x),
        torch.nn.functional.softsign,
        _SOFTSIGN_INPUT,
    )


def test_threshold_matches_pytorch() -> None:
    _assert_activation_parity(
        "threshold",
        lambda x: pycoeus.threshold(x, 0.0, -1.0),
        lambda x: torch.nn.functional.threshold(x, threshold=0.0, value=-1.0),
        _THRESHOLD_INPUT,
    )


def test_celu_matches_pytorch() -> None:
    _assert_activation_parity(
        "celu",
        lambda x: pycoeus.celu(x, 1.0),
        lambda x: torch.nn.functional.celu(x, alpha=1.0),
        _CELU_INPUT,
    )


def test_prelu_matches_pytorch() -> None:
    _assert_activation_parity(
        "prelu",
        lambda x: pycoeus.prelu(x, 0.25),
        lambda x: torch.nn.functional.prelu(x, torch.tensor(0.25, dtype=torch.float64)),
        _PRELU_INPUT,
    )


# ── Classification loss PyTorch parity (CrossEntropy / NLL) (MS-153) ─────────


def test_cross_entropy_loss_matches_pytorch() -> None:
    """Forward and logit-gradient parity for cross_entropy_loss on [N=3, C=4].

    Differential against ``torch.nn.functional.cross_entropy`` (default mean
    reduction) at f64, `atol=1e-10`.  coeus' ``cross_entropy_loss`` fuses
    log-softmax + NLL internally; the test pins both the scalar loss and the
    full softmax-minus-onehot gradient routed back to the logits.
    """
    logits = [2.0, 1.0, 0.1, -0.5, 0.3, 2.2, 1.1, 0.0, -1.0, 0.5, 3.0, 1.5]
    targets = [0, 1, 2]

    x_pyc = pycoeus.Tensor(logits, [3, 4], requires_grad=True)
    loss_pyc = pycoeus.cross_entropy_loss(x_pyc, targets)
    loss_pyc.backward()

    x_t = torch.tensor(logits, dtype=torch.float64).reshape(3, 4).requires_grad_(True)
    t_t = torch.tensor(targets, dtype=torch.long)
    loss_t = torch.nn.functional.cross_entropy(x_t, t_t)
    loss_t.backward()

    _allclose("ce_loss", list(loss_pyc.data), [loss_t.item()], atol=1e-10)
    _allclose("ce_dx", list(x_pyc.grad), x_t.grad.flatten().tolist(), atol=1e-10)


def test_nll_loss_matches_pytorch() -> None:
    """Forward and gradient parity for nll_loss over log-softmax on [N=3, C=4].

    Differential against ``torch.nn.functional.nll_loss`` (default mean
    reduction) at f64, `atol=1e-10`.  Verifies that ``nll_loss(log_softmax(x))``
    composes to the same value and gradient as the fused cross-entropy path.
    """
    logits = [2.0, 1.0, 0.1, -0.5, 0.3, 2.2, 1.1, 0.0, -1.0, 0.5, 3.0, 1.5]
    targets = [0, 1, 2]

    x_pyc = pycoeus.Tensor(logits, [3, 4], requires_grad=True)
    log_probs_pyc = pycoeus.log_softmax(x_pyc, 1)
    loss_pyc = pycoeus.nll_loss(log_probs_pyc, targets)
    loss_pyc.backward()

    x_t = torch.tensor(logits, dtype=torch.float64).reshape(3, 4).requires_grad_(True)
    t_t = torch.tensor(targets, dtype=torch.long)
    log_probs_t = torch.nn.functional.log_softmax(x_t, 1)
    loss_t = torch.nn.functional.nll_loss(log_probs_t, t_t)
    loss_t.backward()

    _allclose("nll_loss", list(loss_pyc.data), [loss_t.item()], atol=1e-10)
    _allclose("nll_dx", list(x_pyc.grad), x_t.grad.flatten().tolist(), atol=1e-10)


# ── Regression/binary loss PyTorch parity (BCE / Huber) (MS-156) ────────────


def test_binary_cross_entropy_matches_pytorch() -> None:
    """Forward and prediction-gradient parity for binary_cross_entropy on [4].

    Differential against ``torch.nn.functional.binary_cross_entropy`` (default
    mean reduction) at f64, `atol=1e-9`.  ``pred`` are probabilities in (0, 1)
    held away from the 0/1 extremes so the eps-clamp contract does not diverge
    from PyTorch's log-clamp.  Pins the −[t·log p + (1−t)·log(1−p)] forward and
    the (p−t)/(p·(1−p)) gradient.
    """
    pred = [0.8, 0.3, 0.6, 0.1]
    target = [1.0, 0.0, 1.0, 0.0]

    p_pyc = pycoeus.Tensor(pred, [4], requires_grad=True)
    t_pyc = pycoeus.Tensor(target, [4])
    loss_pyc = pycoeus.binary_cross_entropy(p_pyc, t_pyc)
    loss_pyc.backward()

    p_t = torch.tensor(pred, dtype=torch.float64, requires_grad=True)
    t_t = torch.tensor(target, dtype=torch.float64)
    loss_t = torch.nn.functional.binary_cross_entropy(p_t, t_t)
    loss_t.backward()

    _allclose("bce_loss", list(loss_pyc.data), [loss_t.item()], atol=1e-9)
    _allclose("bce_dx", list(p_pyc.grad), p_t.grad.flatten().tolist(), atol=1e-9)


def test_huber_loss_matches_pytorch() -> None:
    """Forward and prediction-gradient parity for huber_loss(delta=1.0) on [4].

    Differential against ``torch.nn.functional.huber_loss`` (default mean
    reduction) at f64, `atol=1e-10`.  The four samples straddle the transition:
    errors −0.2 and −0.5 fall in the quadratic region (|e| ≤ δ) and 2.5, −3.0 in
    the linear region (|e| > δ), exercising both pieces of the loss and its
    gradient (e in the quadratic region, δ·sign(e) in the linear region).
    """
    pred = [0.0, 2.5, 1.0, -3.0]
    target = [0.2, 0.0, 1.5, 0.0]

    p_pyc = pycoeus.Tensor(pred, [4], requires_grad=True)
    t_pyc = pycoeus.Tensor(target, [4])
    loss_pyc = pycoeus.huber_loss(p_pyc, t_pyc, 1.0)
    loss_pyc.backward()

    p_t = torch.tensor(pred, dtype=torch.float64, requires_grad=True)
    t_t = torch.tensor(target, dtype=torch.float64)
    loss_t = torch.nn.functional.huber_loss(p_t, t_t, delta=1.0)
    loss_t.backward()

    _allclose("huber_loss", list(loss_pyc.data), [loss_t.item()], atol=1e-10)
    _allclose("huber_dx", list(p_pyc.grad), p_t.grad.flatten().tolist(), atol=1e-10)


# ── KL / MarginRanking loss PyTorch parity (MS-182) ───────────────────────────


def test_kl_divergence_matches_pytorch() -> None:
    """Forward and input-gradient parity for KL divergence on [N=2, C=3].

    Differential against ``torch.nn.functional.kl_div`` with
    ``reduction='mean'`` and ``log_target=False`` at f64. ``input`` carries
    log-probabilities and ``target`` carries probabilities.
    """
    log_probs = [
        math.log(0.7),
        math.log(0.2),
        math.log(0.1),
        math.log(0.3),
        math.log(0.6),
        math.log(0.1),
    ]
    target = [0.6, 0.2, 0.2, 0.0, 0.3, 0.7]

    i_pyc = pycoeus.Tensor(log_probs, [2, 3], requires_grad=True)
    t_pyc = pycoeus.Tensor(target, [2, 3])
    loss_pyc = pycoeus.kl_divergence(i_pyc, t_pyc)
    loss_pyc.backward()

    i_t = torch.tensor(log_probs, dtype=torch.float64).reshape(2, 3).requires_grad_(True)
    t_t = torch.tensor(target, dtype=torch.float64).reshape(2, 3)
    loss_t = (
        torch.nn.functional.kl_div(i_t, t_t, reduction="sum", log_target=False) / i_t.numel()
    )
    loss_t.backward()

    _allclose("kl_loss", list(loss_pyc.data), [loss_t.item()], atol=1e-10)
    _allclose("kl_dinput", list(i_pyc.grad), i_t.grad.flatten().tolist(), atol=1e-10)


def test_margin_ranking_loss_matches_pytorch() -> None:
    """Forward and input-gradient parity for margin_ranking_loss on [4].

    Differential against ``torch.nn.functional.margin_ranking_loss`` with
    ``reduction='mean'`` at f64. Samples span active and inactive hinge regions.
    """
    input1 = [0.1, 1.3, -0.4, 0.3]
    input2 = [0.5, 1.0, 0.2, -0.6]
    target = [1.0, 1.0, -1.0, -1.0]
    margin = 0.2

    i1_pyc = pycoeus.Tensor(input1, [4], requires_grad=True)
    i2_pyc = pycoeus.Tensor(input2, [4], requires_grad=True)
    loss_pyc = pycoeus.margin_ranking_loss(i1_pyc, i2_pyc, target, margin)
    loss_pyc.backward()

    i1_t = torch.tensor(input1, dtype=torch.float64, requires_grad=True)
    i2_t = torch.tensor(input2, dtype=torch.float64, requires_grad=True)
    t_t = torch.tensor(target, dtype=torch.float64)
    loss_t = torch.nn.functional.margin_ranking_loss(
        i1_t, i2_t, t_t, margin=margin, reduction="mean"
    )
    loss_t.backward()

    _allclose("margin_loss", list(loss_pyc.data), [loss_t.item()], atol=1e-10)
    _allclose("margin_dinput1", list(i1_pyc.grad), i1_t.grad.flatten().tolist(), atol=1e-10)
    _allclose("margin_dinput2", list(i2_pyc.grad), i2_t.grad.flatten().tolist(), atol=1e-10)


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


# ── ConvTranspose3d PyTorch parity (MS-183) ─────────────────────────────────


@pytest.mark.skipif(
    not hasattr(pycoeus, "ConvTranspose3d"),
    reason="pycoeus.ConvTranspose3d not available in this wheel build",
)
def test_conv_transpose3d_matches_pytorch() -> None:
    """Forward + gradient parity: ConvTranspose3d(in=2, out=2, k=2, stride=1, pad=0, bias).

    Differential against ``torch.nn.ConvTranspose3d`` at f64 with weights copied
    flat (pycoeus ``[in_channels, out_channels, kD, kH, kW]`` matches PyTorch's
    transposed-conv convention — no transposition needed).  Pins forward output,
    input gradient, weight gradient, and bias gradient.

    Evidence tier: differential / empirical at f64, atol=1e-10.
    """
    in_channels, out_channels = 2, 2
    kernel_size, stride, padding, dilation = 2, 1, 0, 1

    w_data = [
        # ic=0, oc=0 (kD,kH,kW)
        0.5, -0.5, 1.0, 0.0,    0.1, 0.2, 0.3, -0.1,
        # ic=0, oc=1
        -0.2, 0.5, 0.0, 1.0,   1.0, -1.0, 0.2, 0.8,
        # ic=1, oc=0
        0.7, 0.3, -0.7, 0.4,   0.0, 1.0, 0.5, -0.5,
        # ic=1, oc=1
        0.5, -0.5, 1.0, 0.0,   0.1, 0.2, 0.3, -0.1,
    ]
    assert len(w_data) == in_channels * out_channels * kernel_size ** 3
    b_data = [0.1, -0.2]
    x_data = [
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        -1.0, -2.0, -3.0, -4.0,
        -5.0, -6.0, -7.0, -8.0,
    ]
    assert len(x_data) == 1 * in_channels * 2 * 2 * 2
    input_shape = [1, in_channels, 2, 2, 2]  # N=1, C_in=2, D=H=W=2

    ct_pyc = pycoeus.ConvTranspose3d(
        in_channels, out_channels, kernel_size,
        stride=stride, padding=padding, output_padding=0,
        dilation=dilation, bias=True,
    )
    ct_pyc.weight.data = w_data
    ct_pyc.bias.data = b_data
    x_pyc = pycoeus.Tensor(x_data, input_shape, requires_grad=True)
    out_pyc = ct_pyc.forward(x_pyc)
    out_pyc.sum().backward()

    ct_t = torch.nn.ConvTranspose3d(
        in_channels, out_channels, kernel_size,
        stride=stride, padding=padding, output_padding=0,
        dilation=dilation, bias=True,
    ).double()
    with torch.no_grad():
        ct_t.weight[:] = torch.tensor(w_data, dtype=torch.float64).reshape(
            in_channels, out_channels, kernel_size, kernel_size, kernel_size,
        )
        ct_t.bias[:] = torch.tensor(b_data, dtype=torch.float64)
    x_t = (
        torch.tensor(x_data, dtype=torch.float64)
        .reshape(*input_shape)
        .requires_grad_(True)
    )
    out_t = ct_t(x_t)
    out_t.sum().backward()

    _allclose("ct3d_out", list(out_pyc.data), out_t.flatten().tolist(), atol=1e-10)
    _allclose("ct3d_dx", list(x_pyc.grad), x_t.grad.flatten().tolist(), atol=1e-10)
    _allclose(
        "ct3d_dW",
        list(ct_pyc.weight.grad),
        ct_t.weight.grad.flatten().tolist(),
        atol=1e-10,
    )
    _allclose(
        "ct3d_db",
        list(ct_pyc.bias.grad),
        ct_t.bias.grad.flatten().tolist(),
        atol=1e-10,
    )


# ---------------------------------------------------------------------------
# G-038 loss/distance family — forward + input-gradient parity vs torch
# ---------------------------------------------------------------------------

import torch.nn.functional as _F  # noqa: E402


def _loss_grad_parity(label, pyc_loss_fn, torch_loss_fn, x_data, t_data, shape):
    """Compare a (input, target)->scalar loss: value + d/d_input vs torch f64."""
    x_pyc = pycoeus.Tensor(x_data, shape, requires_grad=True)
    t_pyc = pycoeus.Tensor(t_data, shape)
    loss_pyc = pyc_loss_fn(x_pyc, t_pyc)
    loss_pyc.backward()

    n = 1
    for s in shape:
        n *= s
    x_t = torch.tensor(x_data, dtype=torch.float64).reshape(shape).requires_grad_(True)
    t_t = torch.tensor(t_data, dtype=torch.float64).reshape(shape)
    loss_t = torch_loss_fn(x_t, t_t)
    loss_t.backward()
    assert abs(loss_pyc.data[0] - loss_t.item()) < _ATOL, (
        f"{label}: got={loss_pyc.data[0]:.8g}, expected={loss_t.item():.8g}"
    )
    _allclose(f"{label}_dx", list(x_pyc.grad), x_t.grad.flatten().tolist())


def test_l1_loss_matches_pytorch() -> None:
    _loss_grad_parity(
        "l1_loss",
        lambda a, b: pycoeus.l1_loss(a, b),
        lambda a, b: _F.l1_loss(a, b),
        [0.5, -1.2, 3.0, 0.1, -2.0, 1.5],
        [1.0, 0.0, 2.0, 0.0, -1.0, 1.0],
        [2, 3],
    )


def test_bce_with_logits_matches_pytorch() -> None:
    _loss_grad_parity(
        "bce_with_logits",
        lambda a, b: pycoeus.bce_with_logits(a, b),
        lambda a, b: _F.binary_cross_entropy_with_logits(a, b),
        [0.5, -1.2, 0.3, 2.0],
        [1.0, 0.0, 1.0, 0.0],
        [4],
    )


def test_poisson_nll_matches_pytorch() -> None:
    _loss_grad_parity(
        "poisson_nll",
        lambda a, b: pycoeus.poisson_nll(a, b),
        lambda a, b: _F.poisson_nll_loss(a, b, log_input=True, full=False),
        [0.0, 1.0, -0.5, 0.7],
        [2.0, 0.0, 3.0, 1.0],
        [4],
    )


def test_soft_margin_matches_pytorch() -> None:
    _loss_grad_parity(
        "soft_margin",
        lambda a, b: pycoeus.soft_margin(a, b),
        lambda a, b: _F.soft_margin_loss(a, b),
        [0.5, -1.2, 2.0, -0.3],
        [1.0, -1.0, 1.0, -1.0],
        [4],
    )


def test_pairwise_distance_matches_pytorch() -> None:
    # Vector-output [N]; compare the distance vector (forward).
    x1d = [1.0, 2.0, 3.0, 4.0]
    x2d = [0.0, 0.0, 1.0, 1.0]
    d_pyc = pycoeus.pairwise_distance(
        pycoeus.Tensor(x1d, [2, 2]), pycoeus.Tensor(x2d, [2, 2]), 2.0, 1e-6
    )
    x1_t = torch.tensor(x1d, dtype=torch.float64).reshape(2, 2)
    x2_t = torch.tensor(x2d, dtype=torch.float64).reshape(2, 2)
    d_t = _F.pairwise_distance(x1_t, x2_t, p=2.0, eps=1e-6)
    _allclose("pairwise_distance", list(d_pyc.data), d_t.flatten().tolist())


def test_triplet_margin_matches_pytorch() -> None:
    a = [0.0, 0.0, 1.0, 1.0]
    p = [2.0, 0.0, 1.0, 2.0]
    n = [0.0, 2.5, 3.0, 1.0]
    a_pyc = pycoeus.Tensor(a, [2, 2], requires_grad=True)
    loss_pyc = pycoeus.triplet_margin_loss(
        a_pyc, pycoeus.Tensor(p, [2, 2]), pycoeus.Tensor(n, [2, 2]), 1.0, 2.0, 1e-6
    )
    loss_pyc.backward()
    a_t = torch.tensor(a, dtype=torch.float64).reshape(2, 2).requires_grad_(True)
    loss_t = _F.triplet_margin_loss(
        a_t,
        torch.tensor(p, dtype=torch.float64).reshape(2, 2),
        torch.tensor(n, dtype=torch.float64).reshape(2, 2),
        margin=1.0,
        p=2.0,
        eps=1e-6,
    )
    loss_t.backward()
    assert abs(loss_pyc.data[0] - loss_t.item()) < _ATOL, (
        f"triplet: got={loss_pyc.data[0]:.8g}, expected={loss_t.item():.8g}"
    )
    _allclose("triplet_da", list(a_pyc.grad), a_t.grad.flatten().tolist())


def test_multi_margin_matches_pytorch() -> None:
    x_data = [0.5, 0.8, -0.6, 1.0, 0.2, 0.3]
    targets = [0, 1]
    x_pyc = pycoeus.Tensor(x_data, [2, 3], requires_grad=True)
    loss_pyc = pycoeus.multi_margin(x_pyc, targets, 1.0, 1.0)
    loss_pyc.backward()
    x_t = torch.tensor(x_data, dtype=torch.float64).reshape(2, 3).requires_grad_(True)
    loss_t = _F.multi_margin_loss(x_t, torch.tensor(targets, dtype=torch.long), p=1, margin=1.0)
    loss_t.backward()
    assert abs(loss_pyc.data[0] - loss_t.item()) < _ATOL, (
        f"multi_margin: got={loss_pyc.data[0]:.8g}, expected={loss_t.item():.8g}"
    )
    _allclose("multi_margin_dx", list(x_pyc.grad), x_t.grad.flatten().tolist())


# ---------------------------------------------------------------------------
# EmbeddingBag forward parity (G-041)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not hasattr(pycoeus, "EmbeddingBag"),
    reason="pycoeus.EmbeddingBag not available in this build",
)
def test_embeddingbag_sum_matches_pytorch() -> None:
    """Forward parity: EmbeddingBag(vocab=6, dim=4, mode='sum') with two bags.

    Evidence tier: differential/empirical against ``torch.nn.EmbeddingBag`` at f64.
    Tolerance 1e-10.
    """
    num_embeddings, embedding_dim = 6, 4
    weight_data = [
        1.0, 0.0, -1.0, 0.5,    # row 0
        0.5, 0.5,  0.5, 0.5,    # row 1
        0.0, 1.0,  0.0, 1.0,    # row 2
       -1.0, 0.0,  1.0, 0.0,    # row 3
        0.2, 0.3, -0.2, -0.3,   # row 4
        0.7,-0.7,  0.7, -0.7,   # row 5
    ]
    # Two bags: bag0 = [0, 2, 4], bag1 = [1, 3, 5]
    flat_indices = [0, 2, 4, 1, 3, 5]
    offsets = [0, 3]

    eb_pyc = pycoeus.EmbeddingBag(num_embeddings, embedding_dim, "sum")
    eb_pyc.weight.data = weight_data
    out_pyc = eb_pyc.forward_with_offsets(flat_indices, offsets)

    # PyTorch reference
    eb_t = torch.nn.EmbeddingBag(num_embeddings, embedding_dim, mode="sum").double()
    with torch.no_grad():
        eb_t.weight.copy_(
            torch.tensor(weight_data, dtype=torch.float64).reshape(num_embeddings, embedding_dim)
        )
    idx_t = torch.tensor(flat_indices, dtype=torch.long)
    off_t = torch.tensor(offsets, dtype=torch.long)
    out_t = eb_t(idx_t, off_t)

    _allclose(
        "embeddingbag_sum_fwd",
        list(out_pyc.data),
        out_t.flatten().tolist(),
        atol=1e-10,
    )


@pytest.mark.skipif(
    not hasattr(pycoeus, "EmbeddingBag"),
    reason="pycoeus.EmbeddingBag not available in this build",
)
def test_embeddingbag_mean_matches_pytorch() -> None:
    """Forward parity: EmbeddingBag(vocab=6, dim=4, mode='mean') with two bags."""
    num_embeddings, embedding_dim = 6, 4
    weight_data = [
        1.0, 0.0, -1.0, 0.5,
        0.5, 0.5,  0.5, 0.5,
        0.0, 1.0,  0.0, 1.0,
       -1.0, 0.0,  1.0, 0.0,
        0.2, 0.3, -0.2, -0.3,
        0.7,-0.7,  0.7, -0.7,
    ]
    flat_indices = [0, 2, 4, 1, 3, 5]
    offsets = [0, 3]

    eb_pyc = pycoeus.EmbeddingBag(num_embeddings, embedding_dim, "mean")
    eb_pyc.weight.data = weight_data
    out_pyc = eb_pyc.forward_with_offsets(flat_indices, offsets)

    eb_t = torch.nn.EmbeddingBag(num_embeddings, embedding_dim, mode="mean").double()
    with torch.no_grad():
        eb_t.weight.copy_(
            torch.tensor(weight_data, dtype=torch.float64).reshape(num_embeddings, embedding_dim)
        )
    idx_t = torch.tensor(flat_indices, dtype=torch.long)
    off_t = torch.tensor(offsets, dtype=torch.long)
    out_t = eb_t(idx_t, off_t)

    _allclose(
        "embeddingbag_mean_fwd",
        list(out_pyc.data),
        out_t.flatten().tolist(),
        atol=1e-10,
    )


@pytest.mark.skipif(
    not hasattr(pycoeus, "EmbeddingBag"),
    reason="pycoeus.EmbeddingBag not available in this build",
)
def test_embeddingbag_max_matches_pytorch() -> None:
    """Forward parity: EmbeddingBag(vocab=6, dim=4, mode='max') with two bags."""
    num_embeddings, embedding_dim = 6, 4
    weight_data = [
        1.0, 0.0, -1.0, 0.5,
        0.5, 0.5,  0.5, 0.5,
        0.0, 1.0,  0.0, 1.0,
       -1.0, 0.0,  1.0, 0.0,
        0.2, 0.3, -0.2, -0.3,
        0.7,-0.7,  0.7, -0.7,
    ]
    flat_indices = [0, 2, 4, 1, 3, 5]
    offsets = [0, 3]

    eb_pyc = pycoeus.EmbeddingBag(num_embeddings, embedding_dim, "max")
    eb_pyc.weight.data = weight_data
    out_pyc = eb_pyc.forward_with_offsets(flat_indices, offsets)

    eb_t = torch.nn.EmbeddingBag(num_embeddings, embedding_dim, mode="max").double()
    with torch.no_grad():
        eb_t.weight.copy_(
            torch.tensor(weight_data, dtype=torch.float64).reshape(num_embeddings, embedding_dim)
        )
    idx_t = torch.tensor(flat_indices, dtype=torch.long)
    off_t = torch.tensor(offsets, dtype=torch.long)
    out_t = eb_t(idx_t, off_t)

    _allclose(
        "embeddingbag_max_fwd",
        list(out_pyc.data),
        out_t.flatten().tolist(),
        atol=1e-10,
    )



# ---------------------------------------------------------------------------
# AdaptiveAvgPool1d and AdaptiveAvgPool2d parity (MS-213)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not hasattr(pycoeus, "AdaptiveAvgPool1d"),
    reason="pycoeus.AdaptiveAvgPool1d not available in this build",
)
def test_adaptive_avg_pool1d_matches_pytorch() -> None:
    """Forward parity: AdaptiveAvgPool1d(output_size=3) on [2, 4, 8]."""
    n, c, l = 2, 4, 8
    output_size = 3
    data = [float(i) * 0.25 - 1.0 for i in range(n * c * l)]

    m_pyc = pycoeus.AdaptiveAvgPool1d(output_size)
    x_pyc = pycoeus.Tensor(data, [n, c, l])
    y_pyc = m_pyc.forward(x_pyc)

    x_t = torch.tensor(data, dtype=torch.float64).reshape(n, c, l)
    m_t = torch.nn.AdaptiveAvgPool1d(output_size)
    y_t = m_t(x_t)

    _allclose(
        "adaptive_avg_pool1d",
        list(y_pyc.data),
        y_t.flatten().tolist(),
        atol=1e-10,
    )


@pytest.mark.skipif(
    not hasattr(pycoeus, "AdaptiveAvgPool2d"),
    reason="pycoeus.AdaptiveAvgPool2d not available in this build",
)
def test_adaptive_avg_pool2d_global_matches_pytorch() -> None:
    """Forward parity: AdaptiveAvgPool2d(1) (global avg) on [2, 3, 6, 6]."""
    n, c, h, w = 2, 3, 6, 6
    data = [float(i) * 0.1 - 3.0 for i in range(n * c * h * w)]

    m_pyc = pycoeus.AdaptiveAvgPool2d(1)
    x_pyc = pycoeus.Tensor(data, [n, c, h, w])
    y_pyc = m_pyc.forward(x_pyc)

    x_t = torch.tensor(data, dtype=torch.float64).reshape(n, c, h, w)
    m_t = torch.nn.AdaptiveAvgPool2d(1)
    y_t = m_t(x_t)

    _allclose(
        "adaptive_avg_pool2d_global",
        list(y_pyc.data),
        y_t.flatten().tolist(),
        atol=1e-10,
    )


@pytest.mark.skipif(
    not hasattr(pycoeus, "AdaptiveAvgPool2d"),
    reason="pycoeus.AdaptiveAvgPool2d not available in this build",
)
def test_adaptive_avg_pool2d_non_trivial_matches_pytorch() -> None:
    """Forward parity: AdaptiveAvgPool2d(3, 4) on [1, 2, 6, 8] (non-square output)."""
    n, c, h, w = 1, 2, 6, 8
    out_h, out_w = 3, 4
    data = [float(i) * 0.05 - 1.5 for i in range(n * c * h * w)]

    m_pyc = pycoeus.AdaptiveAvgPool2d(out_h, out_w)
    x_pyc = pycoeus.Tensor(data, [n, c, h, w])
    y_pyc = m_pyc.forward(x_pyc)

    x_t = torch.tensor(data, dtype=torch.float64).reshape(n, c, h, w)
    m_t = torch.nn.AdaptiveAvgPool2d((out_h, out_w))
    y_t = m_t(x_t)

    _allclose(
        "adaptive_avg_pool2d_non_trivial",
        list(y_pyc.data),
        y_t.flatten().tolist(),
        atol=1e-10,
    )


def test_adaptive_avg_pool_backward_matches_pytorch() -> None:
    """Gradient parity vs torch: AdaptiveAvgPool1d/2d are now differentiable
    (G-045). Both use overlapping adaptive regions so the gradient genuinely
    sums region contributions."""
    # 1d: [2, 4, 7] -> 3 (overlapping regions)
    n, c, length = 2, 4, 7
    d1 = [math.sin(i * 0.11) for i in range(n * c * length)]
    x1 = pycoeus.Tensor(d1, [n, c, length], requires_grad=True)
    y1 = pycoeus.AdaptiveAvgPool1d(3).forward(x1)
    pycoeus.mse_loss(y1, pycoeus.Tensor([0.1] * (n * c * 3), [n, c, 3])).backward()
    x1t = (
        torch.tensor(d1, dtype=torch.float64)
        .reshape(n, c, length)
        .requires_grad_(True)
    )
    y1t = torch.nn.AdaptiveAvgPool1d(3)(x1t)
    torch.nn.functional.mse_loss(
        y1t, torch.full((n, c, 3), 0.1, dtype=torch.float64)
    ).backward()
    _allclose("adaptive1d_forward", list(y1.data), y1t.detach().flatten().tolist())
    _allclose("adaptive1d_dx", list(x1.grad), x1t.grad.flatten().tolist())

    # 2d: [2, 3, 5, 5] -> (2, 2) (overlapping on both axes)
    n, c, h, w = 2, 3, 5, 5
    d2 = [math.cos(i * 0.07) for i in range(n * c * h * w)]
    x2 = pycoeus.Tensor(d2, [n, c, h, w], requires_grad=True)
    y2 = pycoeus.AdaptiveAvgPool2d(2, 2).forward(x2)
    pycoeus.mse_loss(y2, pycoeus.Tensor([0.2] * (n * c * 2 * 2), [n, c, 2, 2])).backward()
    x2t = (
        torch.tensor(d2, dtype=torch.float64)
        .reshape(n, c, h, w)
        .requires_grad_(True)
    )
    y2t = torch.nn.AdaptiveAvgPool2d((2, 2))(x2t)
    torch.nn.functional.mse_loss(
        y2t, torch.full((n, c, 2, 2), 0.2, dtype=torch.float64)
    ).backward()
    _allclose("adaptive2d_forward", list(y2.data), y2t.detach().flatten().tolist())
    _allclose("adaptive2d_dx", list(x2.grad), x2t.grad.flatten().tolist())


def test_adaptive_max_pool_backward_matches_pytorch() -> None:
    """Forward + gradient parity vs torch: AdaptiveMaxPool1d/2d are differentiable
    (G-045). Distinct values ((i*13)%211 is a permutation, gcd(13,211)=1) so each
    region's argmax is unique and matches torch's, making dx unambiguous."""
    # 1d: [2, 4, 7] -> 3
    n, c, length = 2, 4, 7
    d1 = [((i * 13) % 211) * 0.07 for i in range(n * c * length)]
    x1 = pycoeus.Tensor(d1, [n, c, length], requires_grad=True)
    y1 = pycoeus.AdaptiveMaxPool1d(3).forward(x1)
    pycoeus.mse_loss(y1, pycoeus.Tensor([0.5] * (n * c * 3), [n, c, 3])).backward()
    x1t = (
        torch.tensor(d1, dtype=torch.float64)
        .reshape(n, c, length)
        .requires_grad_(True)
    )
    y1t = torch.nn.AdaptiveMaxPool1d(3)(x1t)
    torch.nn.functional.mse_loss(
        y1t, torch.full((n, c, 3), 0.5, dtype=torch.float64)
    ).backward()
    _allclose("adaptivemax1d_forward", list(y1.data), y1t.detach().flatten().tolist())
    _allclose("adaptivemax1d_dx", list(x1.grad), x1t.grad.flatten().tolist())

    # 2d: [2, 3, 5, 5] -> (2, 2)
    n, c, h, w = 2, 3, 5, 5
    d2 = [((i * 13) % 211) * 0.07 for i in range(n * c * h * w)]
    x2 = pycoeus.Tensor(d2, [n, c, h, w], requires_grad=True)
    y2 = pycoeus.AdaptiveMaxPool2d(2, 2).forward(x2)
    pycoeus.mse_loss(y2, pycoeus.Tensor([0.5] * (n * c * 4), [n, c, 2, 2])).backward()
    x2t = (
        torch.tensor(d2, dtype=torch.float64)
        .reshape(n, c, h, w)
        .requires_grad_(True)
    )
    y2t = torch.nn.AdaptiveMaxPool2d((2, 2))(x2t)
    torch.nn.functional.mse_loss(
        y2t, torch.full((n, c, 2, 2), 0.5, dtype=torch.float64)
    ).backward()
    _allclose("adaptivemax2d_forward", list(y2.data), y2t.detach().flatten().tolist())
    _allclose("adaptivemax2d_dx", list(x2.grad), x2t.grad.flatten().tolist())


# ---------------------------------------------------------------------------
# Unfold2d parity (MS-213)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not hasattr(pycoeus, "Unfold2d"),
    reason="pycoeus.Unfold2d not available in this build",
)
def test_unfold2d_matches_pytorch() -> None:
    """Forward parity: Unfold2d(kernel=3, stride=1, padding=0) on [1, 2, 5, 5]."""
    n, c, h, w = 1, 2, 5, 5
    kernel, stride, padding = 3, 1, 0
    data = [float(i) * 0.1 for i in range(n * c * h * w)]

    m_pyc = pycoeus.Unfold2d(kernel, stride, padding, 1)
    x_pyc = pycoeus.Tensor(data, [n, c, h, w])
    y_pyc = m_pyc.forward(x_pyc)

    x_t = torch.tensor(data, dtype=torch.float64).reshape(n, c, h, w)
    m_t = torch.nn.Unfold(kernel_size=kernel, stride=stride, padding=padding)
    y_t = m_t(x_t)

    _allclose(
        "unfold2d",
        list(y_pyc.data),
        y_t.flatten().tolist(),
        atol=1e-10,
    )


def test_unfold2d_backward_matches_pytorch() -> None:
    """Forward + gradient parity vs torch.nn.Unfold (Unfold2d is now
    differentiable: backward is the fold2d col2im transpose)."""
    n, c, h, w = 1, 2, 5, 5
    kernel, stride, padding = 3, 1, 0
    h_out = (h + 2 * padding - kernel) // stride + 1
    w_out = (w + 2 * padding - kernel) // stride + 1
    out_ch, length = c * kernel * kernel, h_out * w_out
    data = [math.sin(i * 0.1) for i in range(n * c * h * w)]

    x_pyc = pycoeus.Tensor(data, [n, c, h, w], requires_grad=True)
    y_pyc = pycoeus.Unfold2d(kernel, stride, padding, 1).forward(x_pyc)
    tgt_pyc = pycoeus.Tensor([0.1] * (n * out_ch * length), [n, out_ch, length])
    pycoeus.mse_loss(y_pyc, tgt_pyc).backward()

    x_t = (
        torch.tensor(data, dtype=torch.float64)
        .reshape(n, c, h, w)
        .requires_grad_(True)
    )
    y_t = torch.nn.Unfold(kernel_size=kernel, stride=stride, padding=padding)(x_t)
    tgt_t = torch.full((n, out_ch, length), 0.1, dtype=torch.float64)
    torch.nn.functional.mse_loss(y_t, tgt_t).backward()

    _allclose("unfold2d_forward", list(y_pyc.data), y_t.detach().flatten().tolist())
    _allclose("unfold2d_dx", list(x_pyc.grad), x_t.grad.flatten().tolist())


def test_fold2d_backward_matches_pytorch() -> None:
    """Forward + gradient parity vs torch.nn.Fold (Fold2d is differentiable:
    backward is the unfold2d im2col adjoint)."""
    n, c = 1, 2
    oh, ow = 5, 5
    kernel, stride, padding = 3, 1, 0
    h_out = (oh + 2 * padding - kernel) // stride + 1
    length = h_out * h_out
    in_ch = c * kernel * kernel
    data = [math.sin(i * 0.1) for i in range(n * in_ch * length)]

    x_pyc = pycoeus.Tensor(data, [n, in_ch, length], requires_grad=True)
    y_pyc = pycoeus.Fold2d(oh, ow, kernel, stride, padding, 1).forward(x_pyc)
    tgt_pyc = pycoeus.Tensor([0.1] * (n * c * oh * ow), [n, c, oh, ow])
    pycoeus.mse_loss(y_pyc, tgt_pyc).backward()

    x_t = (
        torch.tensor(data, dtype=torch.float64)
        .reshape(n, in_ch, length)
        .requires_grad_(True)
    )
    y_t = torch.nn.Fold(
        output_size=(oh, ow), kernel_size=kernel, stride=stride, padding=padding
    )(x_t)
    tgt_t = torch.full((n, c, oh, ow), 0.1, dtype=torch.float64)
    torch.nn.functional.mse_loss(y_t, tgt_t).backward()

    _allclose("fold2d_forward", list(y_pyc.data), y_t.detach().flatten().tolist())
    _allclose("fold2d_dx", list(x_pyc.grad), x_t.grad.flatten().tolist())


def test_fold1d_forward_and_backward() -> None:
    """Fold1d binding smoke test: torch has no 1D Fold, so this verifies the
    newly-bound layer reconstructs the non-overlapping tiling and propagates a
    gradient, matching the Rust analytic test."""
    # output_size=6, kernel=2, stride=2 (non-overlapping); input [1, C*k=2, blocks=3].
    m = pycoeus.Fold1d(6, 2, 2, 0, 1)
    data = [float(i + 1) for i in range(6)]
    x = pycoeus.Tensor(data, [1, 2, 3], requires_grad=True)
    y = m.forward(x)
    # Non-overlapping fold reorders the 6 input values into [1, 1, 6].
    assert sorted(list(y.data)) == sorted(data)
    pycoeus.mse_loss(y, pycoeus.Tensor([0.0] * 6, [1, 1, 6])).backward()
    grad = list(x.grad)
    assert len(grad) == 6
    assert all(math.isfinite(g) for g in grad)
    assert any(abs(g) > 1e-12 for g in grad), "Fold1d gradient is all-zero"


# ---------------------------------------------------------------------------
# Unfold1d parity (MS-214)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not hasattr(pycoeus, "Unfold1d"),
    reason="pycoeus.Unfold1d not available in this build",
)
def test_unfold1d_matches_pytorch() -> None:
    """Forward parity: Unfold1d(kernel=3, stride=1) on [2, 3, 7].

    PyTorch equivalent: x.unfold(dim=2, size=3, step=1) then permute + reshape.
    """
    n, c, l = 2, 3, 7
    kernel, stride = 3, 1
    data = [float(i) * 0.1 - 1.5 for i in range(n * c * l)]

    m_pyc = pycoeus.Unfold1d(kernel, stride, 0, 1)
    x_pyc = pycoeus.Tensor(data, [n, c, l])
    y_pyc = m_pyc.forward(x_pyc)

    # PyTorch: x.unfold(dim, size, step) → [N, C, L_out, kernel]
    # Coeus output: [N, C*kernel, L_out] with layout [n, c*k+ki, lo]
    # ki is the SLOW index: row 0..kernel covers ki for c=0,
    # then ki for c=1, etc.
    x_t = torch.tensor(data, dtype=torch.float64).reshape(n, c, l)
    y_t_raw = x_t.unfold(2, kernel, stride)  # [N, C, L_out, kernel]
    # PyTorch stores [N, C, L_out, kernel]; Coeus stores [N, C*kernel, L_out]
    # where the inner dimension is (ci * kernel + ki).
    # Reshape PyTorch output to [N, C*kernel, L_out]:
    y_t = y_t_raw.permute(0, 1, 3, 2).reshape(n, c * kernel, -1)

    _allclose(
        "unfold1d",
        list(y_pyc.data),
        y_t.flatten().tolist(),
        atol=1e-10,
    )


# ---------------------------------------------------------------------------
# SwiGLU forward + gradient parity
# ---------------------------------------------------------------------------


def test_swiglu_matches_pytorch() -> None:
    """Forward + gradient parity: SwiGLU(64→128), no bias, via MSELoss.

    PyTorch has no built-in SwiGLU, so the reference is composed from
    primitives: ``silu(x @ Wi.T) * (x @ Wo.T)``. Both layers' weights are read
    from pycoeus and injected into the torch reference so the only difference
    measured is the implementation, not the initialisation.
    """
    d_in, d_out, batch = 64, 128, 32

    sg_pyc = pycoeus.SwiGlu(d_in, d_out, bias=False)
    wi_data = sg_pyc.linear_inner.weight.data  # [d_out, d_in] flat
    wo_data = sg_pyc.linear_outer.weight.data  # [d_out, d_in] flat

    x_data = [math.sin(i * 0.013) for i in range(batch * d_in)]
    tgt_data = [0.5] * (batch * d_out)

    # pycoeus forward + backward
    x_pyc = pycoeus.Tensor(x_data, [batch, d_in], requires_grad=True)
    out_pyc = sg_pyc.forward(x_pyc)
    tgt_pyc = pycoeus.Tensor(tgt_data, [batch, d_out])
    loss_pyc = pycoeus.mse_loss(out_pyc, tgt_pyc)
    loss_pyc.backward()

    # PyTorch reference (f64 to match pycoeus default precision)
    x_t = (
        torch.tensor(x_data, dtype=torch.float64)
        .reshape(batch, d_in)
        .requires_grad_(True)
    )
    wi_t = (
        torch.tensor(wi_data, dtype=torch.float64)
        .reshape(d_out, d_in)
        .requires_grad_(True)
    )
    wo_t = (
        torch.tensor(wo_data, dtype=torch.float64)
        .reshape(d_out, d_in)
        .requires_grad_(True)
    )
    inner_t = torch.nn.functional.linear(x_t, wi_t)
    outer_t = torch.nn.functional.linear(x_t, wo_t)
    out_t = torch.nn.functional.silu(inner_t) * outer_t
    tgt_t = torch.tensor(tgt_data, dtype=torch.float64).reshape(batch, d_out)
    loss_t = torch.nn.functional.mse_loss(out_t, tgt_t)
    loss_t.backward()

    _allclose("swiglu_forward", list(out_pyc.data), out_t.detach().flatten().tolist())
    assert abs(loss_pyc.data[0] - loss_t.item()) < _ATOL, (
        f"swiglu loss: got={loss_pyc.data[0]:.8g}, expected={loss_t.item():.8g}"
    )
    _allclose("swiglu_dx", list(x_pyc.grad), x_t.grad.flatten().tolist())
    _allclose(
        "swiglu_dWi",
        list(sg_pyc.linear_inner.weight.grad),
        wi_t.grad.flatten().tolist(),
    )
    _allclose(
        "swiglu_dWo",
        list(sg_pyc.linear_outer.weight.grad),
        wo_t.grad.flatten().tolist(),
    )


# ---------------------------------------------------------------------------
# LocalResponseNorm forward + gradient parity
# ---------------------------------------------------------------------------


def test_local_response_norm_matches_pytorch() -> None:
    """Forward + gradient parity: LocalResponseNorm(size=5) on [2, 8, 4, 4].

    Cross-channel LRN has no learnable parameters; pycoeus and torch share the
    ``alpha/size`` convention and defaults (alpha=1e-4, beta=0.75, k=1.0), so a
    direct comparison needs no weight injection. coeus's LRN is differentiable
    (autograd-graph forward), so ``dx`` parity is verified too (gap G-044
    closed).
    """
    n, c, h, w = 2, 8, 4, 4
    size = 5

    lrn_pyc = pycoeus.LocalResponseNorm(size)
    x_data = [math.sin(i * 0.05) for i in range(n * c * h * w)]
    tgt_data = [0.3] * (n * c * h * w)

    # pycoeus forward + backward
    x_pyc = pycoeus.Tensor(x_data, [n, c, h, w], requires_grad=True)
    out_pyc = lrn_pyc.forward(x_pyc)
    tgt_pyc = pycoeus.Tensor(tgt_data, [n, c, h, w])
    loss_pyc = pycoeus.mse_loss(out_pyc, tgt_pyc)
    loss_pyc.backward()

    # PyTorch forward + backward (f64)
    x_t = (
        torch.tensor(x_data, dtype=torch.float64)
        .reshape(n, c, h, w)
        .requires_grad_(True)
    )
    out_t = torch.nn.LocalResponseNorm(size)(x_t)  # alpha=1e-4, beta=0.75, k=1.0
    tgt_t = torch.tensor(tgt_data, dtype=torch.float64).reshape(n, c, h, w)
    loss_t = torch.nn.functional.mse_loss(out_t, tgt_t)
    loss_t.backward()

    _allclose("lrn_forward", list(out_pyc.data), out_t.detach().flatten().tolist())
    assert abs(loss_pyc.data[0] - loss_t.item()) < _ATOL, (
        f"lrn loss: got={loss_pyc.data[0]:.8g}, expected={loss_t.item():.8g}"
    )
    _allclose("lrn_dx", list(x_pyc.grad), x_t.grad.flatten().tolist())
