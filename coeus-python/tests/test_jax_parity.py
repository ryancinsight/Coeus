"""JAX output-parity tests for the pycoeus Python bindings.

Each test verifies that pycoeus and JAX produce numerically equivalent
forward and backward gradients given identical weight values.  Tests are
skipped automatically when JAX is absent, and the entire module is skipped
when JAX's f64 path is not available (CPU-only JAX without
``JAX_ENABLE_X64=1`` silently truncates to f32 which breaks parity with
pycoeus' f64 default precision).

Run via::

    pytest coeus-python/tests/test_jax_parity.py -v

Weight-convention note:
JAX computes ``x @ w.T`` for the linear projection (transposes the
weight), matching pycoeus' ``[out_features, in_features]`` storage; weights
are copied directly without transposition.
"""

import math
import os
import sys

import pytest

# JAX_ENABLE_X64 must be set before JAX is imported to enable f64 arrays.
# JAX silently downcasts to f32 on CPU without this, breaking parity with
# pycoeus' f64 default precision.
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# Locate pycoeus.pyd alongside this test file.
_HERE = os.path.dirname(__file__)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import pycoeus  # noqa: E402

jax = pytest.importorskip("jax")  # noqa: F401 — module-level skip if absent
# importorskip returns the module only when present; require jax explicitly
# so the type checker has the symbol bound below.
jax = sys.modules["jax"]
import jax.numpy as jnp  # noqa: E402

# Sanity-check that f64 actually worked: a fresh JAX array literal of dtype
# float64 must report float64; if JAX silently downcasted to f32 (e.g. XLA
# backend ignoring JAX_ENABLE_X64), skip the entire module.
_probe = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)
if str(_probe.dtype) != "float64":
    pytest.skip(
        "JAX f64 path unavailable on this platform "
        "(set JAX_ENABLE_X64=1, or use a backend that supports float64).",
        allow_module_level=True,
    )
del _probe


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ATOL = 1e-5


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
# Linear + ReLU forward + backward
# ---------------------------------------------------------------------------


def test_linear_matches_jax() -> None:
    """Forward and gradient parity: Linear(256→64) + ReLU + MSELoss.

    Mirrors ``test_pytorch_parity.py::test_linear_matches_pytorch``.
    """
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

    # JAX forward + backward (f64 to match pycoeus default precision)
    x_jax = jnp.asarray(x_data, dtype=jnp.float64).reshape(batch, in_f)
    w_jax = jnp.asarray(w_data, dtype=jnp.float64).reshape(out_f, in_f)
    b_jax = jnp.asarray(b_data, dtype=jnp.float64).reshape(out_f)
    tgt_jax = jnp.asarray(tgt_data, dtype=jnp.float64).reshape(batch, out_f)

    def jax_loss(x, w, b):
        out = x @ w.T + b
        act = jnp.maximum(out, 0.0)
        diff = act - tgt_jax
        return jnp.mean(diff * diff)

    grad_fn = jax.value_and_grad(jax_loss, argnums=(0, 1, 2))
    loss_jax, (dx_jax, dw_jax, db_jax) = grad_fn(x_jax, w_jax, b_jax)
    # Force materialization so timing of value access is consistent
    # (JAX is lazy; .tolist()/float() forces evaluation).
    loss_jax.block_until_ready()

    assert abs(loss_pyc.data[0] - float(loss_jax)) < _ATOL, (
        f"loss: got={loss_pyc.data[0]:.8g}, expected={float(loss_jax):.8g}"
    )
    _allclose("dx", list(x_pyc.grad), dx_jax.flatten().tolist())
    _allclose("dW", list(linear_pyc.weight.grad), dw_jax.flatten().tolist())
    _allclose("db", list(linear_pyc.bias.grad), db_jax.flatten().tolist())


# ---------------------------------------------------------------------------
# MultiHeadAttention forward parity
# ---------------------------------------------------------------------------


def test_mha_matches_jax() -> None:
    """Forward parity: MultiHeadAttention(d_model=4, H=2), self-attention, no bias."""
    d_model, num_heads, batch, seq = 4, 2, 1, 3
    d_head = d_model // num_heads

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
    x_pyc = pycoeus.Tensor(x_data, [batch, seq, d_model], requires_grad=False)
    out_pyc = mha_pyc.forward(x_pyc)

    x_jax = jnp.asarray(x_data, dtype=jnp.float64).reshape(batch, seq, d_model)
    wq_jax = jnp.asarray(wq, dtype=jnp.float64).reshape(d_model, d_model)
    wk_jax = jnp.asarray(wk, dtype=jnp.float64).reshape(d_model, d_model)
    wv_jax = jnp.asarray(wv, dtype=jnp.float64).reshape(d_model, d_model)
    wo_jax = jnp.asarray(wo, dtype=jnp.float64).reshape(d_model, d_model)

    q = x_jax @ wq_jax.T
    k = x_jax @ wk_jax.T
    v = x_jax @ wv_jax.T

    qh = q.reshape(batch, seq, num_heads, d_head).transpose(0, 2, 1, 3)
    kh = k.reshape(batch, seq, num_heads, d_head).transpose(0, 2, 1, 3)
    vh = v.reshape(batch, seq, num_heads, d_head).transpose(0, 2, 1, 3)

    scores = jnp.matmul(qh, jnp.swapaxes(kh, -1, -2)) / math.sqrt(d_head)
    attn = jax.nn.softmax(scores, axis=-1)
    ctx = jnp.matmul(attn, vh)
    merged = jnp.swapaxes(ctx, 1, 2).reshape(batch, seq, d_model)
    out_jax = merged @ wo_jax.T

    _allclose("mha_out", list(out_pyc.data), out_jax.flatten().tolist(), atol=1e-10)


# ---------------------------------------------------------------------------
# TransformerDecoderLayer forward parity
# ---------------------------------------------------------------------------


def _jax_layer_norm(x, gamma, beta, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    xhat = (x - mean) / jnp.sqrt(var + eps)
    return xhat * gamma.reshape(1, 1, -1) + beta.reshape(1, 1, -1)


def _jax_mha_forward(q_in, k_in, v_in, wq, wk, wv, wo, num_heads):
    batch, seq_q, d_model = q_in.shape
    seq_k = k_in.shape[1]
    d_head = d_model // num_heads

    q = q_in @ wq.T
    k = k_in @ wk.T
    v = v_in @ wv.T

    qh = q.reshape(batch, seq_q, num_heads, d_head).transpose(0, 2, 1, 3)
    kh = k.reshape(batch, seq_k, num_heads, d_head).transpose(0, 2, 1, 3)
    vh = v.reshape(batch, seq_k, num_heads, d_head).transpose(0, 2, 1, 3)

    scores = jnp.matmul(qh, jnp.swapaxes(kh, -1, -2)) / math.sqrt(d_head)
    attn = jax.nn.softmax(scores, axis=-1)
    ctx = jnp.matmul(attn, vh)
    merged = jnp.swapaxes(ctx, 1, 2).reshape(batch, seq_q, d_model)
    return merged @ wo.T


@pytest.mark.skipif(
    not hasattr(pycoeus, "TransformerDecoderLayer"),
    reason="pycoeus.TransformerDecoderLayer not available",
)
def test_transformer_decoder_layer_matches_jax() -> None:
    """Forward parity: TransformerDecoderLayer(d_model=4, H=2, d_ff=8, dropout=0)."""
    d_model, num_heads, d_ff = 4, 2, 8
    batch, seq_tgt, seq_src = 1, 3, 5
    _ATOL_DEC = 2e-4

    dec = pycoeus.TransformerDecoderLayer(
        d_model=d_model, d_ff=d_ff, num_heads=num_heads
    )

    tgt_data = [0.1 * i - 0.3 for i in range(batch * seq_tgt * d_model)]
    mem_data = [0.05 * i for i in range(batch * seq_src * d_model)]
    tgt_pyc = pycoeus.Tensor(tgt_data, [batch, seq_tgt, d_model], requires_grad=False)
    mem_pyc = pycoeus.Tensor(mem_data, [batch, seq_src, d_model], requires_grad=False)
    out_pyc = dec.forward(tgt_pyc, mem_pyc)

    tgt = jnp.asarray(tgt_data, dtype=jnp.float64).reshape(batch, seq_tgt, d_model)
    memory = jnp.asarray(mem_data, dtype=jnp.float64).reshape(batch, seq_src, d_model)

    sa_wq = jnp.asarray(list(dec.self_attn.w_q.data), dtype=jnp.float64).reshape(
        d_model, d_model
    )
    sa_wk = jnp.asarray(list(dec.self_attn.w_k.data), dtype=jnp.float64).reshape(
        d_model, d_model
    )
    sa_wv = jnp.asarray(list(dec.self_attn.w_v.data), dtype=jnp.float64).reshape(
        d_model, d_model
    )
    sa_wo = jnp.asarray(list(dec.self_attn.w_o.data), dtype=jnp.float64).reshape(
        d_model, d_model
    )
    ca_wq = jnp.asarray(list(dec.cross_attn.w_q.data), dtype=jnp.float64).reshape(
        d_model, d_model
    )
    ca_wk = jnp.asarray(list(dec.cross_attn.w_k.data), dtype=jnp.float64).reshape(
        d_model, d_model
    )
    ca_wv = jnp.asarray(list(dec.cross_attn.w_v.data), dtype=jnp.float64).reshape(
        d_model, d_model
    )
    ca_wo = jnp.asarray(list(dec.cross_attn.w_o.data), dtype=jnp.float64).reshape(
        d_model, d_model
    )

    n1_g = jnp.asarray(list(dec.norm1.weight.data), dtype=jnp.float64)
    n1_b = jnp.asarray(list(dec.norm1.bias.data), dtype=jnp.float64)
    n2_g = jnp.asarray(list(dec.norm2.weight.data), dtype=jnp.float64)
    n2_b = jnp.asarray(list(dec.norm2.bias.data), dtype=jnp.float64)
    n3_g = jnp.asarray(list(dec.norm3.weight.data), dtype=jnp.float64)
    n3_b = jnp.asarray(list(dec.norm3.bias.data), dtype=jnp.float64)

    ff1_w = jnp.asarray(list(dec.ffn.linear1.weight.data), dtype=jnp.float64).reshape(
        d_ff, d_model
    )
    ff1_b = jnp.asarray(list(dec.ffn.linear1.bias.data), dtype=jnp.float64)
    ff2_w = jnp.asarray(list(dec.ffn.linear2.weight.data), dtype=jnp.float64).reshape(
        d_model, d_ff
    )
    ff2_b = jnp.asarray(list(dec.ffn.linear2.bias.data), dtype=jnp.float64)

    n1 = _jax_layer_norm(tgt, n1_g, n1_b, eps=1e-5)
    x1 = tgt + _jax_mha_forward(n1, n1, n1, sa_wq, sa_wk, sa_wv, sa_wo, num_heads)
    n2 = _jax_layer_norm(x1, n2_g, n2_b, eps=1e-5)
    x2 = x1 + _jax_mha_forward(
        n2, memory, memory, ca_wq, ca_wk, ca_wv, ca_wo, num_heads
    )
    n3 = _jax_layer_norm(x2, n3_g, n3_b, eps=1e-5)
    ff = jax.nn.gelu(n3 @ ff1_w.T + ff1_b)
    out_jax = x2 + (ff @ ff2_w.T + ff2_b)

    _allclose(
        "decoder_layer_fwd",
        list(out_pyc.data),
        out_jax.flatten().tolist(),
        atol=_ATOL_DEC,
    )


# ---------------------------------------------------------------------------
# Elementwise activation forward + backward (mirrors the PyTorch activation
# parity tests, against jax.nn references)
# ---------------------------------------------------------------------------


def _assert_activation_matches_jax(name, pyc_fn, jax_fn, data) -> None:
    """Differential forward + input-gradient parity for an elementwise activation.

    pycoeus drives ``out.sum().backward()``; JAX uses ``jax.grad`` of the summed
    activation. Compared at f64, ``_ATOL``.
    """
    x_pyc = pycoeus.Tensor(data, [len(data)], requires_grad=True)
    out_pyc = pyc_fn(x_pyc)
    out_pyc.sum().backward()

    x_jax = jnp.array(data, dtype=jnp.float64)
    out_jax = jax_fn(x_jax)
    grad_jax = jax.grad(lambda z: jnp.sum(jax_fn(z)))(x_jax)

    _allclose(f"{name}_out", list(out_pyc.data), list(out_jax))
    _allclose(f"{name}_dx", list(x_pyc.grad), list(grad_jax))


# Mixed-sign inputs span both regimes of each nonlinearity. SiLU/Mish/ELU/
# Softplus are C1 everywhere (0.0 safe); LeakyReLU has a kink at 0 where the
# subgradient convention is implementation-defined, so 0.0 is excluded.
_ACT_INPUT = [-2.0, -0.5, 0.0, 0.3, 1.5, 3.0]
_ACT_INPUT_NO_ZERO = [-2.0, -0.5, 0.3, 1.5, 3.0]


def test_silu_matches_jax() -> None:
    _assert_activation_matches_jax(
        "silu", lambda x: pycoeus.silu(x), jax.nn.silu, _ACT_INPUT
    )


def test_log_sigmoid_matches_jax() -> None:
    _assert_activation_matches_jax(
        "log_sigmoid", lambda x: pycoeus.log_sigmoid(x), jax.nn.log_sigmoid, _ACT_INPUT
    )


def test_tanhshrink_matches_jax() -> None:
    _assert_activation_matches_jax(
        "tanhshrink",
        lambda x: pycoeus.tanhshrink(x),
        lambda x: x - jnp.tanh(x),
        _ACT_INPUT,
    )


def test_mish_matches_jax() -> None:
    _assert_activation_matches_jax(
        "mish", lambda x: pycoeus.mish(x), jax.nn.mish, _ACT_INPUT
    )


def test_elu_matches_jax() -> None:
    _assert_activation_matches_jax(
        "elu", lambda x: pycoeus.elu(x), jax.nn.elu, _ACT_INPUT
    )


def test_softplus_matches_jax() -> None:
    _assert_activation_matches_jax(
        "softplus", lambda x: pycoeus.softplus(x), jax.nn.softplus, _ACT_INPUT
    )


def test_leaky_relu_matches_jax() -> None:
    # Default negative slope 0.01 on both sides; 0.0 excluded (kink subgradient).
    _assert_activation_matches_jax(
        "leaky_relu",
        lambda x: pycoeus.leaky_relu(x),
        jax.nn.leaky_relu,
        _ACT_INPUT_NO_ZERO,
    )


def test_prelu_matches_jax() -> None:
    """PReLU forward + input-gradient parity vs JAX.

    PReLU contract: y = x if x > 0 else alpha * x; dx = 1 if x > 0 else alpha.
    Includes x = 0.0 to exercise the kink: Coeus and JAX both return alpha
    there (gradient convention matches PyTorch, since both PyTorch and JAX
    evaluate the negative-side branch at x = 0).
    """
    alpha = 0.25
    data = [-2.0, -1.0, 0.0, 0.5, 1.0]

    x_pyc = pycoeus.Tensor(data, [len(data)], requires_grad=True)
    out_pyc = pycoeus.prelu(x_pyc, alpha)
    out_pyc.sum().backward()

    x_jax = jnp.array(data, dtype=jnp.float64)
    jax_prelu = lambda z: jnp.where(z > 0.0, z, alpha * z)
    out_jax = jax_prelu(x_jax)
    grad_jax = jax.grad(lambda z: jnp.sum(jax_prelu(z)))(x_jax)

    _allclose("prelu_out", list(out_pyc.data), list(out_jax))
    _allclose("prelu_dx", list(x_pyc.grad), list(grad_jax))


def test_relu_matches_jax() -> None:
    # ReLU kink at 0 -> exclude 0.0 (subgradient convention).
    _assert_activation_matches_jax(
        "relu", lambda x: pycoeus.relu(x), jax.nn.relu, _ACT_INPUT_NO_ZERO
    )


def test_sigmoid_matches_jax() -> None:
    _assert_activation_matches_jax(
        "sigmoid", lambda x: pycoeus.sigmoid(x), jax.nn.sigmoid, _ACT_INPUT
    )


def test_tanh_matches_jax() -> None:
    _assert_activation_matches_jax(
        "tanh", lambda x: pycoeus.tanh(x), jnp.tanh, _ACT_INPUT
    )


def test_gelu_matches_jax() -> None:
    # Exact GELU (erf form); JAX exact via approximate=False.
    _assert_activation_matches_jax(
        "gelu",
        lambda x: pycoeus.gelu(x),
        lambda z: jax.nn.gelu(z, approximate=False),
        _ACT_INPUT,
    )


def test_glu_matches_jax() -> None:
    """Gated Linear Unit over dim=1: [2, 4] -> [2, 2] (a * sigmoid(b) split)."""
    data = [0.1 * i - 0.5 for i in range(2 * 4)]
    x_pyc = pycoeus.Tensor(data, [2, 4], requires_grad=True)
    out_pyc = pycoeus.glu(x_pyc, 1)
    out_pyc.sum().backward()

    x_jax = jnp.array(data, dtype=jnp.float64).reshape(2, 4)
    out_jax = jax.nn.glu(x_jax, axis=1)
    grad_jax = jax.grad(lambda z: jnp.sum(jax.nn.glu(z, axis=1)))(x_jax)

    _allclose("glu_out", list(out_pyc.data), out_jax.flatten().tolist())
    _allclose("glu_dx", list(x_pyc.grad), grad_jax.flatten().tolist())


# ---------------------------------------------------------------------------
# Softmax / log-softmax / cross-entropy forward + backward (mirrors the
# PyTorch loss/softmax parity, against jax.nn references)
# ---------------------------------------------------------------------------

_LOGITS = [2.0, 1.0, 0.1, -0.5, 0.3, 2.2, 1.1, 0.0, -1.0, 0.5, 3.0, 1.5]
_LOGITS_SHAPE = [3, 4]
_TARGETS = [0, 1, 2]


def test_softmax_matches_jax() -> None:
    """Forward and input-gradient parity: softmax over dim=1 on [3, 4]."""
    x_pyc = pycoeus.Tensor(_LOGITS, _LOGITS_SHAPE, requires_grad=True)
    out_pyc = pycoeus.softmax(x_pyc, 1)
    out_pyc.sum().backward()

    x_jax = jnp.array(_LOGITS, dtype=jnp.float64).reshape(3, 4)
    out_jax = jax.nn.softmax(x_jax, axis=1)
    grad_jax = jax.grad(lambda z: jnp.sum(jax.nn.softmax(z, axis=1)))(x_jax)

    _allclose("softmax_out", list(out_pyc.data), out_jax.flatten().tolist())
    _allclose("softmax_dx", list(x_pyc.grad), grad_jax.flatten().tolist())


def test_log_softmax_matches_jax() -> None:
    """Forward and input-gradient parity: log-softmax over dim=1 on [3, 4]."""
    x_pyc = pycoeus.Tensor(_LOGITS, _LOGITS_SHAPE, requires_grad=True)
    out_pyc = pycoeus.log_softmax(x_pyc, 1)
    out_pyc.sum().backward()

    x_jax = jnp.array(_LOGITS, dtype=jnp.float64).reshape(3, 4)
    out_jax = jax.nn.log_softmax(x_jax, axis=1)
    grad_jax = jax.grad(lambda z: jnp.sum(jax.nn.log_softmax(z, axis=1)))(x_jax)

    _allclose("log_softmax_out", list(out_pyc.data), out_jax.flatten().tolist())
    _allclose("log_softmax_dx", list(x_pyc.grad), grad_jax.flatten().tolist())


def test_cross_entropy_loss_matches_jax() -> None:
    """Forward and logit-gradient parity: cross_entropy_loss (mean) on [3, 4].

    JAX reference fuses log-softmax + negative-log-likelihood with mean reduction,
    matching pycoeus' ``cross_entropy_loss``.
    """

    def _jax_ce(z):
        log_probs = jax.nn.log_softmax(z, axis=1)
        return -jnp.mean(log_probs[jnp.arange(3), jnp.array(_TARGETS)])

    x_pyc = pycoeus.Tensor(_LOGITS, _LOGITS_SHAPE, requires_grad=True)
    loss_pyc = pycoeus.cross_entropy_loss(x_pyc, _TARGETS)
    loss_pyc.backward()

    x_jax = jnp.array(_LOGITS, dtype=jnp.float64).reshape(3, 4)
    loss_jax = _jax_ce(x_jax)
    grad_jax = jax.grad(_jax_ce)(x_jax)

    _allclose("ce_loss", list(loss_pyc.data), [float(loss_jax)])
    _allclose("ce_dx", list(x_pyc.grad), grad_jax.flatten().tolist())


# ---------------------------------------------------------------------------
# LayerNorm / RMSNorm forward + backward (mirrors the PyTorch norm parity,
# against inline JAX references)
# ---------------------------------------------------------------------------

_NORM_DATA = [0.1 * i - 0.3 for i in range(2 * 4)]
_NORM_GAMMA = [1.5, 0.5, 1.2, 0.8]
_NORM_BETA = [0.1, -0.1, 0.2, -0.2]
_NORM_EPS = 1e-5


def test_layernorm_matches_jax() -> None:
    """Forward + input/weight/bias gradient parity: LayerNorm(4) on [2, 4]."""
    ln = pycoeus.LayerNorm(4, eps=_NORM_EPS)
    ln.weight.data = _NORM_GAMMA
    ln.bias.data = _NORM_BETA
    x_pyc = pycoeus.Tensor(_NORM_DATA, [2, 4], requires_grad=True)
    out_pyc = ln.forward(x_pyc)
    out_pyc.sum().backward()

    def _jax_ln(z, g, b):
        mu = jnp.mean(z, axis=-1, keepdims=True)
        var = jnp.mean((z - mu) ** 2, axis=-1, keepdims=True)
        return (z - mu) / jnp.sqrt(var + _NORM_EPS) * g + b

    x_jax = jnp.array(_NORM_DATA, dtype=jnp.float64).reshape(2, 4)
    g_jax = jnp.array(_NORM_GAMMA, dtype=jnp.float64)
    b_jax = jnp.array(_NORM_BETA, dtype=jnp.float64)
    out_jax = _jax_ln(x_jax, g_jax, b_jax)
    gx, gg, gb = jax.grad(lambda z, g, b: jnp.sum(_jax_ln(z, g, b)), argnums=(0, 1, 2))(
        x_jax, g_jax, b_jax
    )

    _allclose("ln_out", list(out_pyc.data), out_jax.flatten().tolist())
    _allclose("ln_dx", list(x_pyc.grad), gx.flatten().tolist())
    _allclose("ln_dgamma", list(ln.weight.grad), gg.tolist())
    _allclose("ln_dbeta", list(ln.bias.grad), gb.tolist())


def test_rmsnorm_matches_jax() -> None:
    """Forward + input/weight gradient parity: RMSNorm(4) on [2, 4] (no bias)."""
    rms = pycoeus.RMSNorm(4, eps=_NORM_EPS)
    rms.weight.data = _NORM_GAMMA
    x_pyc = pycoeus.Tensor(_NORM_DATA, [2, 4], requires_grad=True)
    out_pyc = rms.forward(x_pyc)
    out_pyc.sum().backward()

    def _jax_rms(z, g):
        return z / jnp.sqrt(jnp.mean(z**2, axis=-1, keepdims=True) + _NORM_EPS) * g

    x_jax = jnp.array(_NORM_DATA, dtype=jnp.float64).reshape(2, 4)
    g_jax = jnp.array(_NORM_GAMMA, dtype=jnp.float64)
    out_jax = _jax_rms(x_jax, g_jax)
    gx, gg = jax.grad(lambda z, g: jnp.sum(_jax_rms(z, g)), argnums=(0, 1))(
        x_jax, g_jax
    )

    _allclose("rms_out", list(out_pyc.data), out_jax.flatten().tolist())
    _allclose("rms_dx", list(x_pyc.grad), gx.flatten().tolist())
    _allclose("rms_dgamma", list(rms.weight.grad), gg.tolist())


# ---------------------------------------------------------------------------
# Regression / binary loss forward + backward (mirrors the PyTorch loss parity,
# against inline JAX references)
# ---------------------------------------------------------------------------


def test_mse_loss_matches_jax() -> None:
    """Forward and prediction-gradient parity: mse_loss (mean) on [4]."""
    pred = [0.8, 0.3, 0.6, 0.1]
    target = [1.0, 0.0, 1.0, 0.0]

    p_pyc = pycoeus.Tensor(pred, [4], requires_grad=True)
    loss_pyc = pycoeus.mse_loss(p_pyc, pycoeus.Tensor(target, [4]))
    loss_pyc.backward()

    t_jax = jnp.array(target, dtype=jnp.float64)

    def _jax_mse(z):
        return jnp.mean((z - t_jax) ** 2)

    p_jax = jnp.array(pred, dtype=jnp.float64)
    _allclose("mse_loss", list(loss_pyc.data), [float(_jax_mse(p_jax))])
    _allclose("mse_dx", list(p_pyc.grad), jax.grad(_jax_mse)(p_jax).tolist())


def test_binary_cross_entropy_matches_jax() -> None:
    """Forward and prediction-gradient parity: binary_cross_entropy (mean) on [4].

    Probabilities in (0, 1) held away from 0/1 so the eps-clamp does not diverge.
    """
    pred = [0.8, 0.3, 0.6, 0.1]
    target = [1.0, 0.0, 1.0, 0.0]

    p_pyc = pycoeus.Tensor(pred, [4], requires_grad=True)
    loss_pyc = pycoeus.binary_cross_entropy(p_pyc, pycoeus.Tensor(target, [4]))
    loss_pyc.backward()

    t_jax = jnp.array(target, dtype=jnp.float64)

    def _jax_bce(z):
        return -jnp.mean(t_jax * jnp.log(z) + (1.0 - t_jax) * jnp.log(1.0 - z))

    p_jax = jnp.array(pred, dtype=jnp.float64)
    _allclose("bce_loss", list(loss_pyc.data), [float(_jax_bce(p_jax))])
    _allclose("bce_dx", list(p_pyc.grad), jax.grad(_jax_bce)(p_jax).tolist())


def test_huber_loss_matches_jax() -> None:
    """Forward and prediction-gradient parity: huber_loss(delta=1.0) on [4].

    Samples straddle the transition so both the quadratic (|e| <= delta) and
    linear (|e| > delta) regions and their gradients are exercised.
    """
    pred = [0.0, 2.5, 1.0, -3.0]
    target = [0.2, 0.0, 1.5, 0.0]
    delta = 1.0

    p_pyc = pycoeus.Tensor(pred, [4], requires_grad=True)
    loss_pyc = pycoeus.huber_loss(p_pyc, pycoeus.Tensor(target, [4]), delta)
    loss_pyc.backward()

    t_jax = jnp.array(target, dtype=jnp.float64)

    def _jax_huber(z):
        d = z - t_jax
        a = jnp.abs(d)
        return jnp.mean(jnp.where(a <= delta, 0.5 * d * d, delta * (a - 0.5 * delta)))

    p_jax = jnp.array(pred, dtype=jnp.float64)
    _allclose("huber_loss", list(loss_pyc.data), [float(_jax_huber(p_jax))])
    _allclose("huber_dx", list(p_pyc.grad), jax.grad(_jax_huber)(p_jax).tolist())


def test_kl_divergence_matches_jax() -> None:
    """KL divergence (input = log-probs, target = probs) on [2, 3], mean reduction.

    Mirrors the PyTorch parity: ``kl_div(log_input, target, reduction='sum')/numel``
    = mean of ``target * (log target - log_input)``, with the ``target == 0`` terms
    contributing 0 (the JAX reference masks them so ``0 * log 0`` is not a NaN).
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

    t_jax = jnp.array(target, dtype=jnp.float64).reshape(2, 3)

    def _jax_kl(inp):
        terms = jnp.where(t_jax > 0.0, t_jax * (jnp.log(t_jax) - inp), 0.0)
        return jnp.sum(terms) / inp.size

    i_jax = jnp.array(log_probs, dtype=jnp.float64).reshape(2, 3)
    _allclose("kl_loss", list(loss_pyc.data), [float(_jax_kl(i_jax))])
    _allclose(
        "kl_dinput", list(i_pyc.grad), jax.grad(_jax_kl)(i_jax).flatten().tolist()
    )


def test_margin_ranking_loss_matches_jax() -> None:
    """MarginRanking loss on [4], mean reduction: mean(relu(-t*(i1-i2) + margin))."""
    input1 = [0.1, 1.3, -0.4, 0.3]
    input2 = [0.5, 1.0, 0.2, -0.6]
    target = [1.0, 1.0, -1.0, -1.0]
    margin = 0.2

    i1_pyc = pycoeus.Tensor(input1, [4], requires_grad=True)
    i2_pyc = pycoeus.Tensor(input2, [4], requires_grad=True)
    loss_pyc = pycoeus.margin_ranking_loss(i1_pyc, i2_pyc, target, margin)
    loss_pyc.backward()

    t_jax = jnp.array(target, dtype=jnp.float64)

    def _jax_margin(x1, x2):
        return jnp.mean(jnp.maximum(0.0, -t_jax * (x1 - x2) + margin))

    x1 = jnp.array(input1, dtype=jnp.float64)
    x2 = jnp.array(input2, dtype=jnp.float64)
    g1, g2 = jax.grad(_jax_margin, argnums=(0, 1))(x1, x2)
    _allclose("margin_loss", list(loss_pyc.data), [float(_jax_margin(x1, x2))])
    _allclose("margin_dinput1", list(i1_pyc.grad), g1.tolist())
    _allclose("margin_dinput2", list(i2_pyc.grad), g2.tolist())


# ---------------------------------------------------------------------------
# Embedding lookup and GroupNorm (mirrors the PyTorch parity, against inline
# JAX references)
# ---------------------------------------------------------------------------


def test_embedding_matches_jax() -> None:
    """Embedding lookup forward + weight-gradient parity on [6] indices -> [6, 4].

    The backward scatter-adds the upstream gradient to the looked-up rows; JAX's
    advanced-index gradient does the same, so a sum-loss gradient is compared.
    """
    num_embeddings, embedding_dim = 6, 4
    weight = [0.1 * i - 1.0 for i in range(num_embeddings * embedding_dim)]
    indices = [0, 2, 4, 1, 3, 5]

    emb = pycoeus.Embedding(num_embeddings, embedding_dim)
    emb.weight.data = weight
    idx_pyc = pycoeus.Tensor(
        [float(i) for i in indices], [len(indices)], requires_grad=False
    )
    out_pyc = emb.forward(idx_pyc)
    out_pyc.sum().backward()

    w_jax = jnp.array(weight, dtype=jnp.float64).reshape(num_embeddings, embedding_dim)
    idx_jax = jnp.array(indices)

    def _jax_emb(w):
        return jnp.sum(w[idx_jax])

    _allclose("emb_out", list(out_pyc.data), w_jax[idx_jax].flatten().tolist())
    _allclose(
        "emb_dweight",
        list(emb.weight.grad),
        jax.grad(_jax_emb)(w_jax).flatten().tolist(),
    )


def test_groupnorm_matches_jax() -> None:
    """GroupNorm(groups=2, C=4) on [2, 4, 2, 2]: forward + input/gamma/beta grads."""
    n, c, h, w = 2, 4, 2, 2
    groups = 2
    eps = 1e-5
    data = [0.1 * i - 0.5 for i in range(n * c * h * w)]
    gamma = [1.5, 0.5, 1.2, 0.8]
    beta = [0.1, -0.1, 0.2, -0.2]

    gn = pycoeus.GroupNorm(groups, c, eps=eps)
    gn.weight.data = gamma
    gn.bias.data = beta
    x_pyc = pycoeus.Tensor(data, [n, c, h, w], requires_grad=True)
    out_pyc = gn.forward(x_pyc)
    out_pyc.sum().backward()

    def _jax_gn(z, ga, be):
        zr = z.reshape(n, groups, c // groups, h, w)
        mu = jnp.mean(zr, axis=(2, 3, 4), keepdims=True)
        var = jnp.mean((zr - mu) ** 2, axis=(2, 3, 4), keepdims=True)
        norm = ((zr - mu) / jnp.sqrt(var + eps)).reshape(n, c, h, w)
        return norm * ga.reshape(1, c, 1, 1) + be.reshape(1, c, 1, 1)

    x_jax = jnp.array(data, dtype=jnp.float64).reshape(n, c, h, w)
    ga_jax = jnp.array(gamma, dtype=jnp.float64)
    be_jax = jnp.array(beta, dtype=jnp.float64)
    out_jax = _jax_gn(x_jax, ga_jax, be_jax)
    gx, gg, gb = jax.grad(
        lambda z, ga, be: jnp.sum(_jax_gn(z, ga, be)), argnums=(0, 1, 2)
    )(x_jax, ga_jax, be_jax)

    _allclose("gn_out", list(out_pyc.data), out_jax.flatten().tolist())
    _allclose("gn_dx", list(x_pyc.grad), gx.flatten().tolist())
    _allclose("gn_dgamma", list(gn.weight.grad), gg.tolist())
    _allclose("gn_dbeta", list(gn.bias.grad), gb.tolist())


# ---------------------------------------------------------------------------
# G-038 loss/distance family — forward + input-gradient parity vs JAX
# (loss formulas defined directly in jnp; gradients via jax.grad; optax absent)
# ---------------------------------------------------------------------------


def _jax_loss_parity(label, pyc_fn, jax_fn, x_data, t_data, shape):
    """Compare a (input, target)->scalar loss: value + d/d_input vs jax.grad."""
    x_pyc = pycoeus.Tensor(x_data, list(shape), requires_grad=True)
    t_pyc = pycoeus.Tensor(t_data, list(shape))
    loss_pyc = pyc_fn(x_pyc, t_pyc)
    loss_pyc.backward()

    x_j = jnp.asarray(x_data, dtype=jnp.float64).reshape(shape)
    t_j = jnp.asarray(t_data, dtype=jnp.float64).reshape(shape)
    f = lambda x: jax_fn(x, t_j)  # noqa: E731
    assert abs(loss_pyc.data[0] - float(f(x_j))) < _ATOL, (
        f"{label}: got={loss_pyc.data[0]:.8g}, expected={float(f(x_j)):.8g}"
    )
    _allclose(f"{label}_dx", list(x_pyc.grad), jax.grad(f)(x_j).flatten().tolist())


def test_l1_loss_matches_jax() -> None:
    _jax_loss_parity(
        "l1_loss",
        lambda a, b: pycoeus.l1_loss(a, b),
        lambda x, t: jnp.mean(jnp.abs(x - t)),
        [0.5, -1.2, 3.0, 0.1, -2.0, 1.5],
        [1.0, 0.0, 2.0, 0.0, -1.0, 1.0],
        (2, 3),
    )


def test_bce_with_logits_matches_jax() -> None:
    _jax_loss_parity(
        "bce_with_logits",
        lambda a, b: pycoeus.bce_with_logits(a, b),
        lambda z, y: jnp.mean(
            jnp.maximum(z, 0.0) - z * y + jnp.log1p(jnp.exp(-jnp.abs(z)))
        ),
        [0.5, -1.2, 0.3, 2.0],
        [1.0, 0.0, 1.0, 0.0],
        (4,),
    )


def test_poisson_nll_matches_jax() -> None:
    _jax_loss_parity(
        "poisson_nll",
        lambda a, b: pycoeus.poisson_nll(a, b),
        lambda x, t: jnp.mean(jnp.exp(x) - t * x),
        [0.0, 1.0, -0.5, 0.7],
        [2.0, 0.0, 3.0, 1.0],
        (4,),
    )


def test_soft_margin_matches_jax() -> None:
    _jax_loss_parity(
        "soft_margin",
        lambda a, b: pycoeus.soft_margin(a, b),
        lambda x, y: jnp.mean(jnp.log1p(jnp.exp(-y * x))),
        [0.5, -1.2, 2.0, -0.3],
        [1.0, -1.0, 1.0, -1.0],
        (4,),
    )


def test_pairwise_distance_matches_jax() -> None:
    x1d = [1.0, 2.0, 3.0, 4.0]
    x2d = [0.0, 0.0, 1.0, 1.0]
    d_pyc = pycoeus.pairwise_distance(
        pycoeus.Tensor(x1d, [2, 2]), pycoeus.Tensor(x2d, [2, 2]), 2.0, 1e-6
    )
    x1_j = jnp.asarray(x1d, dtype=jnp.float64).reshape(2, 2)
    x2_j = jnp.asarray(x2d, dtype=jnp.float64).reshape(2, 2)
    d_j = jnp.sqrt(jnp.sum((x1_j - x2_j) ** 2, axis=1) + 1e-6)
    _allclose("pairwise_distance", list(d_pyc.data), d_j.tolist())


def test_triplet_margin_matches_jax() -> None:
    a = [0.0, 0.0, 1.0, 1.0]
    p = [2.0, 0.0, 1.0, 2.0]
    n = [0.0, 2.5, 3.0, 1.0]
    a_pyc = pycoeus.Tensor(a, [2, 2], requires_grad=True)
    loss_pyc = pycoeus.triplet_margin_loss(
        a_pyc, pycoeus.Tensor(p, [2, 2]), pycoeus.Tensor(n, [2, 2]), 1.0, 2.0, 1e-6
    )
    loss_pyc.backward()
    p_j = jnp.asarray(p, dtype=jnp.float64).reshape(2, 2)
    n_j = jnp.asarray(n, dtype=jnp.float64).reshape(2, 2)

    def f(av):
        d_ap = jnp.sqrt(jnp.sum((av - p_j) ** 2, axis=1) + 1e-6)
        d_an = jnp.sqrt(jnp.sum((av - n_j) ** 2, axis=1) + 1e-6)
        return jnp.mean(jnp.maximum(0.0, d_ap - d_an + 1.0))

    a_j = jnp.asarray(a, dtype=jnp.float64).reshape(2, 2)
    assert abs(loss_pyc.data[0] - float(f(a_j))) < _ATOL
    _allclose("triplet_da", list(a_pyc.grad), jax.grad(f)(a_j).flatten().tolist())


def test_multi_margin_matches_jax() -> None:
    x_data = [0.5, 0.8, -0.6, 1.0, 0.2, 0.3]
    targets = [0, 1]
    x_pyc = pycoeus.Tensor(x_data, [2, 3], requires_grad=True)
    loss_pyc = pycoeus.multi_margin(x_pyc, targets, 1.0, 1.0)
    loss_pyc.backward()

    def f(x):  # x: [2, 3]
        rows = []
        for i in range(2):
            y = targets[i]
            m = 1.0 - x[i, y] + x[i]  # [C]
            hinge = jnp.maximum(0.0, m)
            rows.append((jnp.sum(hinge) - hinge[y]) / 3.0)
        return jnp.mean(jnp.stack(rows))

    x_j = jnp.asarray(x_data, dtype=jnp.float64).reshape(2, 3)
    assert abs(loss_pyc.data[0] - float(f(x_j))) < _ATOL
    _allclose("multi_margin_dx", list(x_pyc.grad), jax.grad(f)(x_j).flatten().tolist())


# ---------------------------------------------------------------------------
# AdaptiveAvgPool2d parity (MS-214)
# ---------------------------------------------------------------------------


def test_adaptive_avg_pool2d_global_matches_jax() -> None:
    """Forward parity: AdaptiveAvgPool2d(1) (global avg) on [2, 3, 5, 5].

    JAX reference: jnp.mean over last two spatial axes with keepdims.
    """
    n, c, h, w = 2, 3, 5, 5
    data = [float(i) * 0.1 - 2.5 for i in range(n * c * h * w)]

    if not hasattr(pycoeus, "AdaptiveAvgPool2d"):
        pytest.skip("pycoeus.AdaptiveAvgPool2d not available")

    m_pyc = pycoeus.AdaptiveAvgPool2d(1)
    x_pyc = pycoeus.Tensor(data, [n, c, h, w])
    y_pyc = m_pyc.forward(x_pyc)

    x_j = jnp.asarray(data, dtype=jnp.float64).reshape(n, c, h, w)
    y_j = jnp.mean(x_j, axis=(-2, -1), keepdims=True)

    _allclose(
        "adaptive_avg_pool2d_global_jax",
        list(y_pyc.data),
        y_j.flatten().tolist(),
        atol=1e-10,
    )


# ---------------------------------------------------------------------------
# SwiGLU forward + gradient parity
# ---------------------------------------------------------------------------


def test_swiglu_matches_jax() -> None:
    """Forward + gradient parity: SwiGLU(64→128), no bias, via MSELoss.

    Mirrors ``test_pytorch_parity.py::test_swiglu_matches_pytorch``. JAX has no
    built-in SwiGLU, so the reference is composed from primitives:
    ``silu(x @ Wi.T) * (x @ Wo.T)`` with silu computed as ``z * sigmoid(z)``.
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

    # JAX reference (f64 to match pycoeus default precision)
    x_jax = jnp.asarray(x_data, dtype=jnp.float64).reshape(batch, d_in)
    wi_jax = jnp.asarray(wi_data, dtype=jnp.float64).reshape(d_out, d_in)
    wo_jax = jnp.asarray(wo_data, dtype=jnp.float64).reshape(d_out, d_in)
    tgt_jax = jnp.asarray(tgt_data, dtype=jnp.float64).reshape(batch, d_out)

    def _silu(z):
        return z / (1.0 + jnp.exp(-z))

    def jax_loss(x, wi, wo):
        out = _silu(x @ wi.T) * (x @ wo.T)
        diff = out - tgt_jax
        return jnp.mean(diff * diff)

    grad_fn = jax.value_and_grad(jax_loss, argnums=(0, 1, 2))
    loss_jax, (dx_jax, dwi_jax, dwo_jax) = grad_fn(x_jax, wi_jax, wo_jax)
    loss_jax.block_until_ready()

    out_jax = _silu(x_jax @ wi_jax.T) * (x_jax @ wo_jax.T)

    _allclose("swiglu_forward_jax", list(out_pyc.data), out_jax.flatten().tolist())
    assert abs(loss_pyc.data[0] - float(loss_jax)) < _ATOL, (
        f"swiglu loss: got={loss_pyc.data[0]:.8g}, expected={float(loss_jax):.8g}"
    )
    _allclose("swiglu_dx_jax", list(x_pyc.grad), dx_jax.flatten().tolist())
    _allclose(
        "swiglu_dWi_jax",
        list(sg_pyc.linear_inner.weight.grad),
        dwi_jax.flatten().tolist(),
    )
    _allclose(
        "swiglu_dWo_jax",
        list(sg_pyc.linear_outer.weight.grad),
        dwo_jax.flatten().tolist(),
    )


# ---------------------------------------------------------------------------
# LocalResponseNorm forward + gradient parity
# ---------------------------------------------------------------------------


def test_local_response_norm_matches_jax() -> None:
    """Forward + gradient parity: LocalResponseNorm(size=5) on [2, 8, 4, 4].

    Mirrors ``test_pytorch_parity.py::test_local_response_norm_matches_pytorch``.
    JAX has no built-in LRN, so the reference composes the same cross-channel
    windowed sum-of-squares used by coeus — a band matrix ``M[i,j]=1 iff
    |i-j|<=size//2`` contracted against the squared activations — then
    ``(k + (alpha/size)*windowed)**beta`` and ``x/denom``. coeus's LRN is
    differentiable, so ``dx`` parity is verified via ``jax.value_and_grad``.
    """
    n, c, h, w = 2, 8, 4, 4
    size, alpha, beta, k = 5, 1e-4, 0.75, 1.0

    lrn_pyc = pycoeus.LocalResponseNorm(size)
    x_data = [math.sin(i * 0.05) for i in range(n * c * h * w)]
    tgt_data = [0.3] * (n * c * h * w)

    # pycoeus forward + backward
    x_pyc = pycoeus.Tensor(x_data, [n, c, h, w], requires_grad=True)
    out_pyc = lrn_pyc.forward(x_pyc)
    tgt_pyc = pycoeus.Tensor(tgt_data, [n, c, h, w])
    loss_pyc = pycoeus.mse_loss(out_pyc, tgt_pyc)
    loss_pyc.backward()

    # JAX reference (f64)
    x_jax = jnp.asarray(x_data, dtype=jnp.float64).reshape(n, c, h, w)
    tgt_jax = jnp.asarray(tgt_data, dtype=jnp.float64).reshape(n, c, h, w)
    idx = jnp.arange(c)
    band = (jnp.abs(idx[:, None] - idx[None, :]) <= size // 2).astype(jnp.float64)

    def lrn_forward(x):
        sq = (x * x).reshape(n, c, h * w)
        windowed = jnp.einsum("cj,njs->ncs", band, sq).reshape(n, c, h, w)
        return x / (k + (alpha / size) * windowed) ** beta

    def lrn_loss(x):
        diff = lrn_forward(x) - tgt_jax
        return jnp.mean(diff * diff)

    out_jax = lrn_forward(x_jax)
    loss_jax, dx_jax = jax.value_and_grad(lrn_loss)(x_jax)
    loss_jax.block_until_ready()

    _allclose("lrn_forward_jax", list(out_pyc.data), out_jax.flatten().tolist())
    assert abs(loss_pyc.data[0] - float(loss_jax)) < _ATOL, (
        f"lrn loss: got={loss_pyc.data[0]:.8g}, expected={float(loss_jax):.8g}"
    )
    _allclose("lrn_dx_jax", list(x_pyc.grad), dx_jax.flatten().tolist())


# ---------------------------------------------------------------------------
# AdaptiveAvgPool forward + gradient parity
# ---------------------------------------------------------------------------


def _avg_pool_matrix(in_len: int, out_len: int):
    """Averaging matrix ``P[out, in]`` for adaptive pooling — PyTorch's region
    convention ``[floor(o*in/out), ceil((o+1)*in/out))``, ``1/region`` per cell."""
    rows = []
    for o in range(out_len):
        start = o * in_len // out_len
        end = -(-(o + 1) * in_len // out_len)  # ceil
        rows.append(
            [1.0 / (end - start) if start <= li < end else 0.0 for li in range(in_len)]
        )
    return jnp.asarray(rows, dtype=jnp.float64)


def test_adaptive_avg_pool_matches_jax() -> None:
    """Forward + gradient parity vs a composed JAX reference (averaging matmul),
    mirroring the PyTorch dx test. AdaptiveAvgPool is differentiable (G-045)."""
    # 1d: [2, 4, 7] -> 3
    n, c, length, out = 2, 4, 7, 3
    d1 = [math.sin(i * 0.11) for i in range(n * c * length)]
    x1p = pycoeus.Tensor(d1, [n, c, length], requires_grad=True)
    y1 = pycoeus.AdaptiveAvgPool1d(out).forward(x1p)
    pycoeus.mse_loss(y1, pycoeus.Tensor([0.1] * (n * c * out), [n, c, out])).backward()

    p1 = _avg_pool_matrix(length, out)
    x1j = jnp.asarray(d1, dtype=jnp.float64).reshape(n, c, length)
    tgt1 = jnp.full((n, c, out), 0.1, dtype=jnp.float64)

    def loss1(x):
        return jnp.mean((jnp.einsum("ol,ncl->nco", p1, x) - tgt1) ** 2)

    _, dx1 = jax.value_and_grad(loss1)(x1j)
    y1j = jnp.einsum("ol,ncl->nco", p1, x1j)
    _allclose("adaptive1d_forward_jax", list(y1.data), y1j.flatten().tolist())
    _allclose("adaptive1d_dx_jax", list(x1p.grad), dx1.flatten().tolist())

    # 2d: [2, 3, 5, 5] -> (2, 2)
    n, c, h, w, oh, ow = 2, 3, 5, 5, 2, 2
    d2 = [math.cos(i * 0.07) for i in range(n * c * h * w)]
    x2p = pycoeus.Tensor(d2, [n, c, h, w], requires_grad=True)
    y2 = pycoeus.AdaptiveAvgPool2d(oh, ow).forward(x2p)
    pycoeus.mse_loss(
        y2, pycoeus.Tensor([0.2] * (n * c * oh * ow), [n, c, oh, ow])
    ).backward()

    ph, pw = _avg_pool_matrix(h, oh), _avg_pool_matrix(w, ow)
    x2j = jnp.asarray(d2, dtype=jnp.float64).reshape(n, c, h, w)
    tgt2 = jnp.full((n, c, oh, ow), 0.2, dtype=jnp.float64)

    def loss2(x):
        return jnp.mean((jnp.einsum("ah,bw,nchw->ncab", ph, pw, x) - tgt2) ** 2)

    _, dx2 = jax.value_and_grad(loss2)(x2j)
    y2j = jnp.einsum("ah,bw,nchw->ncab", ph, pw, x2j)
    _allclose("adaptive2d_forward_jax", list(y2.data), y2j.flatten().tolist())
    _allclose("adaptive2d_dx_jax", list(x2p.grad), dx2.flatten().tolist())


def test_adaptive_max_pool_matches_jax() -> None:
    """Forward + gradient parity vs a JAX per-region-max reference, mirroring the
    PyTorch dx test. AdaptiveMaxPool is differentiable (G-045). Distinct values
    ((i*13)%211, gcd=1) keep each region's argmax unique."""

    def ceil_div(a: int, b: int) -> int:
        return -(-a // b)

    # 1d: [2, 4, 7] -> 3
    n, c, length, out = 2, 4, 7, 3
    d1 = [((i * 13) % 211) * 0.07 for i in range(n * c * length)]
    x1p = pycoeus.Tensor(d1, [n, c, length], requires_grad=True)
    y1 = pycoeus.AdaptiveMaxPool1d(out).forward(x1p)
    pycoeus.mse_loss(y1, pycoeus.Tensor([0.5] * (n * c * out), [n, c, out])).backward()

    x1j = jnp.asarray(d1, dtype=jnp.float64).reshape(n, c, length)
    tgt1 = jnp.full((n, c, out), 0.5, dtype=jnp.float64)

    def fwd1(x):
        cols = [
            jnp.max(
                x[:, :, o * length // out : ceil_div((o + 1) * length, out)], axis=2
            )
            for o in range(out)
        ]
        return jnp.stack(cols, axis=2)

    _, dx1 = jax.value_and_grad(lambda x: jnp.mean((fwd1(x) - tgt1) ** 2))(x1j)
    _allclose("adaptivemax1d_forward_jax", list(y1.data), fwd1(x1j).flatten().tolist())
    _allclose("adaptivemax1d_dx_jax", list(x1p.grad), dx1.flatten().tolist())

    # 2d: [2, 3, 5, 5] -> (2, 2)
    n, c, h, w, oh, ow = 2, 3, 5, 5, 2, 2
    d2 = [((i * 13) % 211) * 0.07 for i in range(n * c * h * w)]
    x2p = pycoeus.Tensor(d2, [n, c, h, w], requires_grad=True)
    y2 = pycoeus.AdaptiveMaxPool2d(oh, ow).forward(x2p)
    pycoeus.mse_loss(
        y2, pycoeus.Tensor([0.5] * (n * c * oh * ow), [n, c, oh, ow])
    ).backward()

    x2j = jnp.asarray(d2, dtype=jnp.float64).reshape(n, c, h, w)
    tgt2 = jnp.full((n, c, oh, ow), 0.5, dtype=jnp.float64)

    def fwd2(x):
        rows = []
        for oi in range(oh):
            cols = [
                jnp.max(
                    x[
                        :,
                        :,
                        oi * h // oh : ceil_div((oi + 1) * h, oh),
                        oj * w // ow : ceil_div((oj + 1) * w, ow),
                    ],
                    axis=(2, 3),
                )
                for oj in range(ow)
            ]
            rows.append(jnp.stack(cols, axis=2))
        return jnp.stack(rows, axis=2)

    _, dx2 = jax.value_and_grad(lambda x: jnp.mean((fwd2(x) - tgt2) ** 2))(x2j)
    _allclose("adaptivemax2d_forward_jax", list(y2.data), fwd2(x2j).flatten().tolist())
    _allclose("adaptivemax2d_dx_jax", list(x2p.grad), dx2.flatten().tolist())


# ---------------------------------------------------------------------------
# Smooth L1 (Huber-β) parity (G-038 closure)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
def test_smooth_l1_loss_matches_jax(beta: float) -> None:
    """Forward + gradient parity for SmoothL1 on `[-2, -1, -0.5, 0.5, 1, 1.5]`.

    Differential against an explicit `jnp.where` reference (the JAX
    equivalent of PyTorch's `F.smooth_l1_loss` piecewise). `jax.grad`
    routes through the same `jnp.where`, so the resulting seed matches
    pycoeus at f64 with `atol=1e-10`.
    """
    pred = [-2.0, -1.0, -0.5, 0.5, 1.0, 1.5]
    target = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    p_pyc = pycoeus.Tensor(pred, [6], requires_grad=True)
    t_pyc = pycoeus.Tensor(target, [6])
    loss_pyc = pycoeus.smooth_l1_loss(p_pyc, t_pyc, beta)
    loss_pyc.backward()

    p_jax = jnp.asarray(pred, dtype=jnp.float64)
    t_jax = jnp.asarray(target, dtype=jnp.float64)

    def jax_smooth(x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        z = x - t
        return jnp.mean(
            jnp.where(
                jnp.abs(z) < beta,
                0.5 * z * z / beta,
                jnp.abs(z) - 0.5 * beta,
            )
        )

    loss_jax = jax_smooth(p_jax, t_jax)
    grad_x = jax.grad(jax_smooth, argnums=0)(p_jax, t_jax)
    loss_jax.block_until_ready()
    grad_x.block_until_ready()

    _allclose("smooth_l1_out_jax", list(loss_pyc.data), [float(loss_jax)], atol=1e-10)
    _allclose("smooth_l1_dx_jax", list(p_pyc.grad), list(grad_x), atol=1e-10)


def test_cosine_similarity_matches_jax() -> None:
    """Row-wise cosine similarity parity vs JAX (`dim=1`, default `eps=1e-8`).

    Mirrors `test_pytorch_parity.py::test_cosine_similarity_matches_pytorch`
    but the gradient is taken via `jax.grad` over `sum(cos)` to mirror the
    PyTorch differential.
    """
    x1d = [3.0, 4.0, 1.0, 0.0]
    x2d = [4.0, 3.0, 0.0, 1.0]

    x1_pyc = pycoeus.Tensor(x1d, [2, 2], requires_grad=True)
    x2_pyc = pycoeus.Tensor(x2d, [2, 2], requires_grad=True)
    out_pyc = pycoeus.cosine_similarity(x1_pyc, x2_pyc, dim=1)
    out_pyc.sum().backward()

    x1_j = jnp.asarray(x1d, dtype=jnp.float64).reshape(2, 2)
    x2_j = jnp.asarray(x2d, dtype=jnp.float64).reshape(2, 2)

    def jax_cos(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        dot = jnp.sum(a * b, axis=1)
        n1 = jnp.sqrt(jnp.sum(a * a, axis=1))
        n2 = jnp.sqrt(jnp.sum(b * b, axis=1))
        return dot / (n1 * n2 + 1e-8)

    out_j = jax_cos(x1_j, x2_j)
    grad_fn = jax.grad(lambda a, b: jnp.sum(jax_cos(a, b)), argnums=(0, 1))
    g_x1, g_x2 = grad_fn(x1_j, x2_j)
    out_j.block_until_ready()
    g_x1.block_until_ready()
    g_x2.block_until_ready()

    _allclose("cos_out_jax", list(out_pyc.data), list(out_j), atol=1e-10)
    _allclose("cos_dx1_jax", list(x1_pyc.grad), list(g_x1.flatten()), atol=1e-10)
    _allclose("cos_dx2_jax", list(x2_pyc.grad), list(g_x2.flatten()), atol=1e-10)


# ---------------------------------------------------------------------------
# Mish activation parity (MS-228)
# ---------------------------------------------------------------------------


def test_mish_matches_jax() -> None:
    """Mish forward and gradient parity: x * tanh(softplus(x)).

    JAX reference: x * jnp.tanh(jnp.log1p(jnp.exp(x))) at f64.
    Input spans positive, negative, and zero to cover all regions.
    """
    x_data = [-2.0, -0.5, 0.0, 0.5, 1.5, 3.0]

    x_pyc = pycoeus.Tensor(x_data, [6], requires_grad=True)
    y_pyc = pycoeus.mish(x_pyc)
    # backward via sum
    y_pyc.backward()

    x_j = jnp.asarray(x_data, dtype=jnp.float64)

    def mish_j(x):
        return jnp.sum(x * jnp.tanh(jnp.log1p(jnp.exp(x))))

    val_j, dx_j = jax.value_and_grad(mish_j)(x_j)

    # forward: compare element-wise
    def mish_scalar(x):
        return x * jnp.tanh(jnp.log1p(jnp.exp(x)))

    y_j = jax.vmap(mish_scalar)(x_j)
    _allclose("mish_forward", list(y_pyc.data), y_j.flatten().tolist(), atol=1e-9)
    _allclose("mish_grad", list(x_pyc.grad), dx_j.flatten().tolist(), atol=1e-9)


# ---------------------------------------------------------------------------
# Dropout eval identity parity (MS-228)
# ---------------------------------------------------------------------------


def test_dropout_eval_matches_jax() -> None:
    """Dropout in eval mode is an identity: output == input.

    Both pycoeus and JAX dropout (with rate=0 i.e. deterministic eval) should
    produce outputs equal to the input at any dtype.
    """
    if not hasattr(pycoeus, "Dropout"):
        pytest.skip("pycoeus.Dropout not available")

    x_data = [-1.0, 0.5, 2.0, -0.3, 1.1]
    x_pyc = pycoeus.Tensor(x_data, [5])
    m = pycoeus.Dropout(p=0.5)
    y_pyc = m.forward(x_pyc)

    x_j = jnp.asarray(x_data, dtype=jnp.float64)
    # Eval mode: no dropout applied → output == input
    _allclose("dropout_eval", list(y_pyc.data), x_j.flatten().tolist(), atol=1e-15)


# ---------------------------------------------------------------------------
# SinusoidalEncoding JAX parity (MS-231)
# ---------------------------------------------------------------------------


def test_sinusoidal_encoding_matches_jax() -> None:
    """SinusoidalEncoding forward parity against inline JAX formula.

    PE(pos, 2i) = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

    Input: zero tensor [1, 6, 8]; output = PE rows 0..5.
    """
    if not hasattr(pycoeus, "SinusoidalEncoding"):
        pytest.skip("pycoeus.SinusoidalEncoding not available")

    batch, seq, d_model = 1, 6, 8
    max_len = 16
    data = [0.0] * (batch * seq * d_model)

    pe_pyc = pycoeus.SinusoidalEncoding(max_len=max_len, d_model=d_model)
    x_pyc = pycoeus.Tensor(data, [batch, seq, d_model])
    y_pyc = pe_pyc.forward(x_pyc)

    # JAX reference
    positions = jnp.arange(seq, dtype=jnp.float64)
    dims = jnp.arange(d_model // 2, dtype=jnp.float64)
    denom = 10000.0 ** (2.0 * dims / d_model)
    angles = positions[:, None] / denom[None, :]  # [seq, d_model//2]
    pe_sin = jnp.sin(angles)  # [seq, d_model//2]
    pe_cos = jnp.cos(angles)  # [seq, d_model//2]
    # interleave: even=sin, odd=cos
    pe = jnp.stack([pe_sin, pe_cos], axis=-1).reshape(seq, d_model)  # [seq, d_model]
    pe_batch = jnp.tile(pe[None, :, :], [batch, 1, 1])  # [batch, seq, d_model]

    _allclose("sinusoidal_jax", list(y_pyc.data), pe_batch.flatten().tolist(), atol=1e-10)


# ---------------------------------------------------------------------------
# BatchNorm1d eval-mode JAX parity (MS-231)
# ---------------------------------------------------------------------------


def test_batchnorm1d_eval_matches_jax() -> None:
    """BatchNorm1d eval-mode parity: running_mean/var normalization.

    JAX reference: (x - mean) / sqrt(var + eps) * gamma + beta.
    Uses fixed running_mean=0, running_var=1, gamma=1, beta=0.
    """
    if not hasattr(pycoeus, "BatchNorm1d"):
        pytest.skip("pycoeus.BatchNorm1d not available")

    n, c, length = 2, 3, 4
    data = [float(i) * 0.1 - 0.5 for i in range(n * c * length)]

    bn_pyc = pycoeus.BatchNorm1d(c, eps=1e-5, momentum=0.1)
    # pycoeus eval_forward() uses running stats (mean=0, var=1 initially)
    x_pyc = pycoeus.Tensor(data, [n, c, length])
    y_pyc = bn_pyc.eval_forward(x_pyc)

    # JAX: (x - 0) / sqrt(1 + 1e-5) * 1 + 0 = x / sqrt(1+eps)
    x_j = jnp.asarray(data, dtype=jnp.float64).reshape(n, c, length)
    y_j = x_j / jnp.sqrt(1.0 + 1e-5)

    _allclose("bn1d_eval_jax", list(y_pyc.data), y_j.flatten().tolist(), atol=1e-10)



# ---------------------------------------------------------------------------
# RotaryEmbedding (RoPE) JAX parity (MS-232)
# ---------------------------------------------------------------------------


def test_rotary_embedding_matches_jax() -> None:
    """RotaryEmbedding forward parity against inline JAX RoPE formula.

    GPT-NeoX/LLaMA-style RoPE:
    theta_i = base^(-2i/d) for i in [0, d//2)
    angle(pos, i) = pos * theta_i
    cos_table[pos, i] = cos_table[pos, i + d//2] = cos(angle(pos, i))
    sin_table[pos, i] = sin_table[pos, i + d//2] = sin(angle(pos, i))

    x_rotated = x * cos + rotate_half(x) * sin
    rotate_half([x1, x2]) = [-x2, x1] (split in halves, not alternating)
    """
    if not hasattr(pycoeus, "RotaryEmbedding"):
        pytest.skip("pycoeus.RotaryEmbedding not available")

    batch, seq, heads, d_head = 1, 4, 2, 8
    max_len = 16
    base = 10000.0
    data = [float(i) * 0.05 - 0.5 for i in range(batch * seq * heads * d_head)]

    rope_pyc = pycoeus.RotaryEmbedding(max_len=max_len, d_head=d_head, base=base)
    x_pyc = pycoeus.Tensor(data, [batch, seq, heads, d_head])
    y_pyc = rope_pyc.forward(x_pyc)

    # JAX reference — GPT-NeoX style duplicate cos/sin
    x_j = jnp.asarray(data, dtype=jnp.float64).reshape(batch, seq, heads, d_head)
    half = d_head // 2
    positions = jnp.arange(seq, dtype=jnp.float64)
    freqs = base ** (-2.0 * jnp.arange(half, dtype=jnp.float64) / d_head)
    angles = positions[:, None] * freqs[None, :]  # [seq, half]
    cos_t = jnp.concatenate([jnp.cos(angles), jnp.cos(angles)], axis=-1)  # [seq, d_head]
    sin_t = jnp.concatenate([jnp.sin(angles), jnp.sin(angles)], axis=-1)

    # reshape for [batch, seq, 1, d_head] broadcast
    cos_t = cos_t[None, :, None, :]  # [1, seq, 1, d_head]
    sin_t = sin_t[None, :, None, :]

    # rotate_half: [x[:half], x[half:]] → [-x[half:], x[:half]]
    x_first = x_j[..., :half]
    x_second = x_j[..., half:]
    x_rot = jnp.concatenate([-x_second, x_first], axis=-1)

    y_j = x_j * cos_t + x_rot * sin_t

    _allclose("rope", list(y_pyc.data), y_j.flatten().tolist(), atol=1e-9)


# ---------------------------------------------------------------------------
# Shape ops JAX parity (MS-235): movedim / flatten
# ---------------------------------------------------------------------------


def test_movedim_matches_jax() -> None:
    """jnp.moveaxis parity on [2, 3, 4]: move axis 0 to axis 2."""
    n, c, d = 2, 3, 4
    data = [float(i) * 0.1 for i in range(n * c * d)]

    x_pyc = pycoeus.Tensor(data, [n, c, d], requires_grad=True)
    y_pyc = pycoeus.movedim(x_pyc, 0, 2)
    y_pyc.backward()

    x_j = jnp.asarray(data, dtype=jnp.float64).reshape(n, c, d)
    y_j = jnp.moveaxis(x_j, 0, 2)
    dx_j = jax.grad(lambda x: jnp.sum(jnp.moveaxis(x, 0, 2)))(x_j)

    _allclose("movedim_fwd_jax", list(y_pyc.data), y_j.flatten().tolist(), atol=1e-10)
    _allclose("movedim_dx_jax", list(x_pyc.grad), dx_j.flatten().tolist(), atol=1e-10)


def test_flatten_matches_jax() -> None:
    """jnp.reshape parity: flatten [2,3,4] → [2,12] (flatten(1,2))."""
    n, c, d = 2, 3, 4
    data = [float(i) * 0.1 for i in range(n * c * d)]

    x_pyc = pycoeus.Tensor(data, [n, c, d], requires_grad=True)
    y_pyc = pycoeus.flatten(x_pyc, 1, 2)
    y_pyc.backward()

    x_j = jnp.asarray(data, dtype=jnp.float64).reshape(n, c, d)
    y_j = x_j.reshape(n, c * d)
    dx_j = jax.grad(lambda x: jnp.sum(x.reshape(n, c * d)))(x_j)

    _allclose("flatten_fwd_jax", list(y_pyc.data), y_j.flatten().tolist(), atol=1e-10)
    _allclose("flatten_dx_jax", list(x_pyc.grad), dx_j.flatten().tolist(), atol=1e-10)


# ---------------------------------------------------------------------------
# diff / cumsum / cumprod JAX parity (MS-236)
# ---------------------------------------------------------------------------


def test_diff_matches_jax() -> None:
    """jnp.diff parity: n=1 on [1,4,9,16]."""
    data = [1.0, 4.0, 9.0, 16.0]

    x_pyc = pycoeus.Tensor(data, [4], requires_grad=True)
    y_pyc = pycoeus.diff(x_pyc, n=1, dim=0)
    y_pyc.backward()

    x_j = jnp.asarray(data, dtype=jnp.float64)
    y_j = jnp.diff(x_j, n=1, axis=0)
    dx_j = jax.grad(lambda x: jnp.sum(jnp.diff(x, n=1, axis=0)))(x_j)

    _allclose("diff_jax_fwd", list(y_pyc.data), y_j.flatten().tolist(), atol=1e-10)
    _allclose("diff_jax_dx", list(x_pyc.grad), dx_j.flatten().tolist(), atol=1e-10)


def test_cumsum_matches_jax() -> None:
    """jnp.cumsum parity on [8] along axis=0."""
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

    x_pyc = pycoeus.Tensor(data, [8], requires_grad=True)
    y_pyc = pycoeus.cumsum(x_pyc, 0)
    y_pyc.backward()

    x_j = jnp.asarray(data, dtype=jnp.float64)
    y_j = jnp.cumsum(x_j, axis=0)
    dx_j = jax.grad(lambda x: jnp.sum(jnp.cumsum(x, axis=0)))(x_j)

    _allclose("cumsum_jax_fwd", list(y_pyc.data), y_j.flatten().tolist(), atol=1e-10)
    _allclose("cumsum_jax_dx", list(x_pyc.grad), dx_j.flatten().tolist(), atol=1e-10)


# ---------------------------------------------------------------------------
# tril / triu / roll / flip parity
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not hasattr(pycoeus, "tril"), reason="pycoeus.tril not available")
def test_tril_matches_jax() -> None:
    """jnp.tril(x) vs pycoeus.tril(x, 0)."""
    data = [float(i) for i in range(16)]
    x_pyc = pycoeus.Tensor(data, [4, 4], requires_grad=False)
    got = pycoeus.tril(x_pyc, 0)
    t = jnp.array(data, dtype=jnp.float64).reshape(4, 4)
    exp = jnp.tril(t)
    _allclose("tril", list(got.data), jnp.ravel(exp).tolist())


@pytest.mark.skipif(not hasattr(pycoeus, "triu"), reason="pycoeus.triu not available")
def test_triu_matches_jax() -> None:
    """jnp.triu(x) vs pycoeus.triu(x, 0)."""
    data = [float(i) for i in range(16)]
    x_pyc = pycoeus.Tensor(data, [4, 4], requires_grad=False)
    got = pycoeus.triu(x_pyc, 0)
    t = jnp.array(data, dtype=jnp.float64).reshape(4, 4)
    exp = jnp.triu(t)
    _allclose("triu", list(got.data), jnp.ravel(exp).tolist())


@pytest.mark.skipif(not hasattr(pycoeus, "roll"), reason="pycoeus.roll not available")
def test_roll_matches_jax() -> None:
    """jnp.roll(x, shift=1, axis=0) vs pycoeus.roll(x, [1], [0])."""
    data = [float(i) for i in range(9)]
    x_pyc = pycoeus.Tensor(data, [3, 3], requires_grad=False)
    got = pycoeus.roll(x_pyc, [1], [0])
    t = jnp.array(data, dtype=jnp.float64).reshape(3, 3)
    exp = jnp.roll(t, 1, axis=0)
    _allclose("roll", list(got.data), jnp.ravel(exp).tolist())


@pytest.mark.skipif(not hasattr(pycoeus, "flip"), reason="pycoeus.flip not available")
def test_flip_matches_jax() -> None:
    """jnp.flip(x, axis=0) vs pycoeus.flip(x, 0)."""
    data = [float(i) for i in range(9)]
    x_pyc = pycoeus.Tensor(data, [3, 3], requires_grad=False)
    got = pycoeus.flip(x_pyc, 0)
    t = jnp.array(data, dtype=jnp.float64).reshape(3, 3)
    exp = jnp.flip(t, axis=0)
    _allclose("flip", list(got.data), jnp.ravel(exp).tolist())

# ---------------------------------------------------------------------------
# argmax / argmin / topk parity
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not hasattr(pycoeus, "argmax"), reason="pycoeus.argmax not available")
def test_argmax_matches_jax() -> None:
    """jnp.argmax(x, axis=0, keepdims=True) vs pycoeus.argmax(x, 0)."""
    data = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0, 5.0, 8.0]
    x_pyc = pycoeus.Tensor(data, [3, 4], requires_grad=False)
    got = pycoeus.argmax(x_pyc, 0)
    t = jnp.array(data, dtype=jnp.float64).reshape(3, 4)
    exp = jnp.argmax(t, axis=0, keepdims=True)
    _allclose("argmax", list(got.data), jnp.ravel(exp.astype(jnp.float64)).tolist())


@pytest.mark.skipif(not hasattr(pycoeus, "argmin"), reason="pycoeus.argmin not available")
def test_argmin_matches_jax() -> None:
    """jnp.argmin(x, axis=1, keepdims=True) vs pycoeus.argmin(x, 1)."""
    data = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0, 5.0, 8.0]
    x_pyc = pycoeus.Tensor(data, [3, 4], requires_grad=False)
    got = pycoeus.argmin(x_pyc, 1)
    t = jnp.array(data, dtype=jnp.float64).reshape(3, 4)
    exp = jnp.argmin(t, axis=1, keepdims=True)
    _allclose("argmin", list(got.data), jnp.ravel(exp.astype(jnp.float64)).tolist())

# ---------------------------------------------------------------------------
# sort / norm / outer / clamp / where_cond parity
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not hasattr(pycoeus, "sort"), reason="pycoeus.sort not available")
def test_sort_matches_jax() -> None:
    """jnp.sort(x, axis=1) vs pycoeus.sort(x, dim=1)."""
    data = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0, 5.0, 8.0]
    x_pyc = pycoeus.Tensor(data, [3, 4], requires_grad=False)
    vals_pyc, _ = pycoeus.sort(x_pyc, dim=1, descending=False)
    t = jnp.array(data, dtype=jnp.float64).reshape(3, 4)
    exp = jnp.sort(t, axis=1)
    _allclose("sort", list(vals_pyc.data), jnp.ravel(exp).tolist())


@pytest.mark.skipif(not hasattr(pycoeus, "outer"), reason="pycoeus.outer not available")
def test_outer_matches_jax() -> None:
    """jnp.outer(a, b) vs pycoeus.outer(a, b)."""
    a = [1.0, 2.0, 3.0]
    b = [4.0, 5.0]
    a_pyc = pycoeus.Tensor(a, [3], requires_grad=False)
    b_pyc = pycoeus.Tensor(b, [2], requires_grad=False)
    got = pycoeus.outer(a_pyc, b_pyc)
    exp = jnp.outer(jnp.array(a, dtype=jnp.float64), jnp.array(b, dtype=jnp.float64))
    _allclose("outer", list(got.data), jnp.ravel(exp).tolist())


@pytest.mark.skipif(not hasattr(pycoeus, "clamp"), reason="pycoeus.clamp not available")
def test_clamp_matches_jax() -> None:
    """jnp.clip(x, -1.0, 2.0) vs pycoeus.clamp(x, -1.0, 2.0)."""
    data = [-3.0, -1.0, 0.5, 1.5, 2.5, 4.0]
    x_pyc = pycoeus.Tensor(data, [6], requires_grad=False)
    got = pycoeus.clamp(x_pyc, -1.0, 2.0)
    t = jnp.array(data, dtype=jnp.float64)
    exp = jnp.clip(t, -1.0, 2.0)
    _allclose("clamp", list(got.data), exp.tolist())

# ---------------------------------------------------------------------------
# bmm / log_sum_exp parity
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not hasattr(pycoeus, "bmm"), reason="pycoeus.bmm not available")
def test_bmm_matches_jax() -> None:
    """jnp.matmul(a, b) on batch vs pycoeus.bmm — [2,3,4] x [2,4,5]."""
    a_data = [float(i) * 0.1 for i in range(2 * 3 * 4)]
    b_data = [float(i) * 0.05 for i in range(2 * 4 * 5)]
    a_pyc = pycoeus.Tensor(a_data, [2, 3, 4], requires_grad=False)
    b_pyc = pycoeus.Tensor(b_data, [2, 4, 5], requires_grad=False)
    out_pyc = pycoeus.bmm(a_pyc, b_pyc)
    a_j = jnp.array(a_data, dtype=jnp.float64).reshape(2, 3, 4)
    b_j = jnp.array(b_data, dtype=jnp.float64).reshape(2, 4, 5)
    exp = jnp.matmul(a_j, b_j)
    _allclose("bmm", list(out_pyc.data), jnp.ravel(exp).tolist())


@pytest.mark.skipif(not hasattr(pycoeus, "log_sum_exp"), reason="pycoeus.log_sum_exp not available")
def test_log_sum_exp_matches_jax() -> None:
    """jax.scipy.special.logsumexp(x, axis=1) vs pycoeus.log_sum_exp(x, 1)."""
    import jax.scipy.special as jsp
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    x_pyc = pycoeus.Tensor(data, [3, 3], requires_grad=False)
    out_pyc = pycoeus.log_sum_exp(x_pyc, 1)
    t = jnp.array(data, dtype=jnp.float64).reshape(3, 3)
    exp = jsp.logsumexp(t, axis=1)
    _allclose("logsumexp", list(out_pyc.data), exp.tolist())

# ---------------------------------------------------------------------------
# tile / broadcast_to / index_select parity
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not hasattr(pycoeus, "tile"), reason="pycoeus.tile not available")
def test_tile_matches_jax() -> None:
    """jnp.tile(x, reps) vs pycoeus.tile(x, reps)."""
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    x_pyc = pycoeus.Tensor(data, [2, 3], requires_grad=False)
    got = pycoeus.tile(x_pyc, [2, 3])
    t = jnp.array(data, dtype=jnp.float64).reshape(2, 3)
    exp = jnp.tile(t, (2, 3))
    _allclose("tile", list(got.data), jnp.ravel(exp).tolist())


@pytest.mark.skipif(not hasattr(pycoeus, "broadcast_to"), reason="pycoeus.broadcast_to not available")
def test_broadcast_to_matches_jax() -> None:
    """jnp.broadcast_to(x, shape) vs pycoeus.broadcast_to(x, shape)."""
    data = [1.0, 2.0, 3.0]
    x_pyc = pycoeus.Tensor(data, [1, 3], requires_grad=False)
    got = pycoeus.broadcast_to(x_pyc, [4, 3])
    t = jnp.array(data, dtype=jnp.float64).reshape(1, 3)
    exp = jnp.broadcast_to(t, (4, 3))
    _allclose("broadcast_to", list(got.data), jnp.ravel(exp).tolist())

# ---------------------------------------------------------------------------
# creation ops: arange / eye / linspace parity
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not hasattr(pycoeus, "arange"), reason="pycoeus.arange not available")
def test_arange_matches_jax() -> None:
    """jnp.arange(0, 10, 2) vs pycoeus.arange(0, 10, 2)."""
    got = pycoeus.arange(0.0, 10.0, 2.0)
    exp = jnp.arange(0, 10, 2, dtype=jnp.float64)
    _allclose("arange", list(got.data), exp.tolist())


@pytest.mark.skipif(not hasattr(pycoeus, "eye"), reason="pycoeus.eye not available")
def test_eye_matches_jax() -> None:
    """jnp.eye(4) vs pycoeus.eye(4)."""
    got = pycoeus.eye(4)
    exp = jnp.eye(4, dtype=jnp.float64)
    _allclose("eye", list(got.data), jnp.ravel(exp).tolist())

# ---------------------------------------------------------------------------
# gelu_tanh / pow parity
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not hasattr(pycoeus, "gelu_tanh"), reason="pycoeus.gelu_tanh not available")
def test_gelu_tanh_matches_jax() -> None:
    """jax.nn.gelu(x, approximate=True) vs pycoeus.gelu_tanh(x)."""
    import jax.nn
    data = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    x_pyc = pycoeus.Tensor(data, [7], requires_grad=False)
    got = pycoeus.gelu_tanh(x_pyc)
    t = jnp.array(data, dtype=jnp.float64)
    exp = jax.nn.gelu(t, approximate=True)
    _allclose("gelu_tanh", list(got.data), exp.tolist(), atol=1e-6)


@pytest.mark.skipif(not hasattr(pycoeus, "erf"), reason="pycoeus.erf not available")
def test_erf_matches_jax() -> None:
    """jax.scipy.special.erf(x) vs pycoeus.erf(x)."""
    import jax.scipy.special
    data = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    x_pyc = pycoeus.Tensor(data, [7], requires_grad=False)
    got = pycoeus.erf(x_pyc)
    t = jnp.array(data, dtype=jnp.float64)
    exp = jax.scipy.special.erf(t)
    _allclose("erf", list(got.data), exp.tolist(), atol=1e-12)


@pytest.mark.skipif(not hasattr(pycoeus, "erfc"), reason="pycoeus.erfc not available")
def test_erfc_matches_jax() -> None:
    """jax.scipy.special.erfc(x) vs pycoeus.erfc(x)."""
    import jax.scipy.special

    data = [-2.0, -1.0, 0.0, 1.0, 2.0]
    x_pyc = pycoeus.Tensor(data, [5], requires_grad=False)
    got = pycoeus.erfc(x_pyc)
    t = jnp.array(data, dtype=jnp.float64)
    exp = jax.scipy.special.erfc(t)
    _allclose("erfc", list(got.data), exp.tolist(), atol=1e-12)


@pytest.mark.skipif(not hasattr(pycoeus, "pow"), reason="pycoeus.pow not available")
def test_pow_matches_jax() -> None:
    """jnp.power(x, 3) vs pycoeus.pow(x, 3.0)."""
    data = [1.0, 2.0, -1.0, 0.5]
    x_pyc = pycoeus.Tensor(data, [4], requires_grad=False)
    got = pycoeus.pow(x_pyc, 3.0)
    t = jnp.array(data, dtype=jnp.float64)
    exp = jnp.power(t, 3)
    _allclose("pow", list(got.data), exp.tolist())

# ---------------------------------------------------------------------------
# einsum / norm parity
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not hasattr(pycoeus, "einsum"), reason="pycoeus.einsum not available")
def test_einsum_dot_matches_jax() -> None:
    """jnp.einsum('i,i->',a,b) vs pycoeus.einsum."""
    a = [1.0, 2.0, 3.0, 4.0]
    b = [4.0, 3.0, 2.0, 1.0]
    a_pyc = pycoeus.Tensor(a, [4], requires_grad=False)
    b_pyc = pycoeus.Tensor(b, [4], requires_grad=False)
    got = pycoeus.einsum("i,i->", [a_pyc, b_pyc])
    a_j = jnp.array(a, dtype=jnp.float64)
    b_j = jnp.array(b, dtype=jnp.float64)
    exp = jnp.einsum("i,i->", a_j, b_j)
    _allclose("einsum_dot", list(got.data), [float(exp)])


@pytest.mark.skipif(not hasattr(pycoeus, "norm"), reason="pycoeus.norm not available")
def test_norm_global_l2_matches_jax() -> None:
    """jnp.linalg.norm(x) global L2 vs pycoeus.norm(x)."""
    data = [3.0, 4.0, 0.0, 5.0, 12.0, 13.0]
    x_pyc = pycoeus.Tensor(data, [6], requires_grad=False)
    out_pyc = pycoeus.norm(x_pyc)
    t = jnp.array(data, dtype=jnp.float64)
    exp = jnp.linalg.norm(t)
    _allclose("norm_l2", list(out_pyc.data), [float(exp)])

# ---------------------------------------------------------------------------
# sin / cos parity
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not hasattr(pycoeus, "sin"), reason="pycoeus.sin not available")
def test_sin_matches_jax() -> None:
    """jnp.sin(x) vs pycoeus.sin(x)."""
    data = [-3.14159, -1.5708, 0.0, 1.5708, 3.14159]
    x_pyc = pycoeus.Tensor(data, [5], requires_grad=False)
    got = pycoeus.sin(x_pyc)
    exp = jnp.sin(jnp.array(data, dtype=jnp.float64))
    _allclose("sin", list(got.data), exp.tolist(), atol=1e-10)


@pytest.mark.skipif(not hasattr(pycoeus, "cos"), reason="pycoeus.cos not available")
def test_cos_matches_jax() -> None:
    """jnp.cos(x) vs pycoeus.cos(x)."""
    data = [-3.14159, -1.5708, 0.0, 1.5708, 3.14159]
    x_pyc = pycoeus.Tensor(data, [5], requires_grad=False)
    got = pycoeus.cos(x_pyc)
    exp = jnp.cos(jnp.array(data, dtype=jnp.float64))
    _allclose("cos", list(got.data), exp.tolist(), atol=1e-10)


@pytest.mark.skipif(not hasattr(pycoeus, "asin"), reason="pycoeus.asin not available")
def test_asin_matches_jax() -> None:
    data = [-0.9, -0.5, 0.0, 0.5, 0.9]
    x_pyc = pycoeus.Tensor(data, [5], requires_grad=False)
    got = pycoeus.asin(x_pyc)
    exp = jnp.arcsin(jnp.array(data, dtype=jnp.float64))
    _allclose("asin", list(got.data), exp.tolist(), atol=1e-12)


@pytest.mark.skipif(not hasattr(pycoeus, "tan"), reason="pycoeus.tan not available")
def test_tan_matches_jax() -> None:
    data = [-1.0, -0.5, 0.0, 0.5, 1.0]
    x_pyc = pycoeus.Tensor(data, [5], requires_grad=False)
    got = pycoeus.tan(x_pyc)
    exp = jnp.tan(jnp.array(data, dtype=jnp.float64))
    _allclose("tan", list(got.data), exp.tolist(), atol=1e-12)


@pytest.mark.skipif(not hasattr(pycoeus, "atan"), reason="pycoeus.atan not available")
def test_atan_matches_jax() -> None:
    data = [-2.0, -1.0, 0.0, 1.0, 2.0]
    x_pyc = pycoeus.Tensor(data, [5], requires_grad=False)
    got = pycoeus.atan(x_pyc)
    exp = jnp.arctan(jnp.array(data, dtype=jnp.float64))
    _allclose("atan", list(got.data), exp.tolist(), atol=1e-12)

# ---------------------------------------------------------------------------
# hardshrink / mish / celu parity (JAX stax/nn equivalents)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not hasattr(pycoeus, "mish"), reason="pycoeus.mish not available")
def test_mish_matches_jax() -> None:
    """x * tanh(softplus(x)) vs pycoeus.mish(x)."""
    import jax.nn
    data = [-2.0, -1.0, 0.0, 1.0, 2.0]
    x_pyc = pycoeus.Tensor(data, [5], requires_grad=False)
    got = pycoeus.mish(x_pyc)
    t = jnp.array(data, dtype=jnp.float64)
    # JAX mish: x * tanh(softplus(x))
    exp = t * jnp.tanh(jnp.log1p(jnp.exp(t)))
    _allclose("mish", list(got.data), exp.tolist(), atol=1e-8)


@pytest.mark.skipif(not hasattr(pycoeus, "softsign"), reason="pycoeus.softsign not available")
def test_softsign_matches_jax() -> None:
    """jax.nn.soft_sign(x) vs pycoeus.softsign(x)."""
    import jax.nn
    data = [-2.0, -1.0, 0.0, 1.0, 2.0]
    x_pyc = pycoeus.Tensor(data, [5], requires_grad=False)
    got = pycoeus.softsign(x_pyc)
    t = jnp.array(data, dtype=jnp.float64)
    exp = jax.nn.soft_sign(t)
    _allclose("softsign", list(got.data), exp.tolist())

# ---------------------------------------------------------------------------
# hardsigmoid / hardswish / sign / recip parity
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not hasattr(pycoeus, "hardsigmoid"), reason="pycoeus.hardsigmoid not available")
def test_hardsigmoid_matches_jax() -> None:
    """jax.nn.hard_sigmoid(x) vs pycoeus.hardsigmoid(x)."""
    import jax.nn
    data = [-4.0, -1.5, 0.0, 1.5, 4.0]
    x_pyc = pycoeus.Tensor(data, [5], requires_grad=False)
    got = pycoeus.hardsigmoid(x_pyc)
    t = jnp.array(data, dtype=jnp.float64)
    exp = jax.nn.hard_sigmoid(t)
    _allclose("hardsigmoid", list(got.data), exp.tolist(), atol=1e-10)


@pytest.mark.skipif(not hasattr(pycoeus, "hardswish"), reason="pycoeus.hardswish not available")
def test_hardswish_matches_jax() -> None:
    """jax.nn.hard_swish(x) vs pycoeus.hardswish(x)."""
    import jax.nn
    data = [-4.0, -1.5, 0.0, 1.5, 4.0]
    x_pyc = pycoeus.Tensor(data, [5], requires_grad=False)
    got = pycoeus.hardswish(x_pyc)
    t = jnp.array(data, dtype=jnp.float64)
    exp = jax.nn.hard_swish(t)
    _allclose("hardswish", list(got.data), exp.tolist(), atol=1e-10)


@pytest.mark.skipif(not hasattr(pycoeus, "sign"), reason="pycoeus.sign not available")
def test_sign_matches_jax() -> None:
    """jnp.sign(x) vs pycoeus.sign(x)."""
    data = [-2.0, -0.0, 0.0, 0.5, 3.0]
    x_pyc = pycoeus.Tensor(data, [5], requires_grad=False)
    got = pycoeus.sign(x_pyc)
    t = jnp.array(data, dtype=jnp.float64)
    exp = jnp.sign(t)
    _allclose("sign", list(got.data), exp.tolist())

# ---------------------------------------------------------------------------
# abs / sqrt / floor / ceil parity
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not hasattr(pycoeus, "abs"), reason="pycoeus.abs not available")
def test_abs_matches_jax() -> None:
    """jnp.abs(x) vs pycoeus.abs(x)."""
    data = [-3.0, -1.0, 0.0, 1.0, 3.0]
    x_pyc = pycoeus.Tensor(data, [5], requires_grad=False)
    got = pycoeus.abs(x_pyc)
    exp = jnp.abs(jnp.array(data, dtype=jnp.float64))
    _allclose("abs", list(got.data), exp.tolist())


@pytest.mark.skipif(not hasattr(pycoeus, "sqrt"), reason="pycoeus.sqrt not available")
def test_sqrt_matches_jax() -> None:
    """jnp.sqrt(x) vs pycoeus.sqrt(x)."""
    data = [0.25, 1.0, 4.0, 9.0, 16.0]
    x_pyc = pycoeus.Tensor(data, [5], requires_grad=False)
    got = pycoeus.sqrt(x_pyc)
    exp = jnp.sqrt(jnp.array(data, dtype=jnp.float64))
    _allclose("sqrt", list(got.data), exp.tolist())


@pytest.mark.skipif(not hasattr(pycoeus, "floor"), reason="pycoeus.floor not available")
def test_floor_matches_jax() -> None:
    """jnp.floor(x) vs pycoeus.floor(x)."""
    data = [-1.9, -1.0, 0.5, 1.0, 1.9]
    x_pyc = pycoeus.Tensor(data, [5], requires_grad=False)
    got = pycoeus.floor(x_pyc)
    exp = jnp.floor(jnp.array(data, dtype=jnp.float64))
    _allclose("floor", list(got.data), exp.tolist())


@pytest.mark.skipif(not hasattr(pycoeus, "ceil"), reason="pycoeus.ceil not available")
def test_ceil_matches_jax() -> None:
    """jnp.ceil(x) vs pycoeus.ceil(x)."""
    data = [-1.9, -1.0, 0.5, 1.0, 1.9]
    x_pyc = pycoeus.Tensor(data, [5], requires_grad=False)
    got = pycoeus.ceil(x_pyc)
    exp = jnp.ceil(jnp.array(data, dtype=jnp.float64))
    _allclose("ceil", list(got.data), exp.tolist())

# ---------------------------------------------------------------------------
# var / std parity
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not hasattr(pycoeus, "var"), reason="pycoeus.var not available")
def test_var_matches_jax() -> None:
    """jnp.var(x) (population) vs pycoeus.var(x, unbiased=False)."""
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    x_pyc = pycoeus.Tensor(data, [5], requires_grad=False)
    got = pycoeus.var(x_pyc, unbiased=False)
    t = jnp.array(data, dtype=jnp.float64)
    exp = jnp.var(t)
    _allclose("var", list(got.data), [float(exp)])

# ---------------------------------------------------------------------------
# exp / log / neg parity
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not hasattr(pycoeus, "exp"), reason="pycoeus.exp not available")
def test_exp_matches_jax() -> None:
    """jnp.exp(x) vs pycoeus.exp(x)."""
    data = [-2.0, -1.0, 0.0, 1.0, 2.0]
    x_pyc = pycoeus.Tensor(data, [5], requires_grad=False)
    got = pycoeus.exp(x_pyc)
    exp = jnp.exp(jnp.array(data, dtype=jnp.float64))
    _allclose("exp", list(got.data), exp.tolist(), atol=1e-12)


@pytest.mark.skipif(not hasattr(pycoeus, "log"), reason="pycoeus.log not available")
def test_log_matches_jax() -> None:
    """jnp.log(x) vs pycoeus.log(x)."""
    import math
    data = [0.1, 0.5, 1.0, 2.0, math.e]
    x_pyc = pycoeus.Tensor(data, [5], requires_grad=False)
    got = pycoeus.log(x_pyc)
    exp = jnp.log(jnp.array(data, dtype=jnp.float64))
    _allclose("log", list(got.data), exp.tolist(), atol=1e-12)


@pytest.mark.skipif(not hasattr(pycoeus, "neg"), reason="pycoeus.neg not available")
def test_neg_matches_jax() -> None:
    """jnp.negative(x) vs pycoeus.neg(x)."""
    data = [-2.0, 0.0, 1.0, 3.5]
    x_pyc = pycoeus.Tensor(data, [4], requires_grad=False)
    got = pycoeus.neg(x_pyc)
    exp = jnp.negative(jnp.array(data, dtype=jnp.float64))
    _allclose("neg", list(got.data), exp.tolist())

@pytest.mark.skipif(not hasattr(pycoeus, "sinh"), reason="pycoeus.sinh not available")
def test_sinh_matches_jax() -> None:
    data = [-2.0, -1.0, 0.0, 1.0, 2.0]
    x_pyc = pycoeus.Tensor(data, [5], requires_grad=False)
    got = pycoeus.sinh(x_pyc)
    exp = jnp.sinh(jnp.array(data, dtype=jnp.float64))
    _allclose("sinh", list(got.data), exp.tolist(), atol=1e-12)

@pytest.mark.skipif(not hasattr(pycoeus, "log2"), reason="pycoeus.log2 not available")
def test_log2_matches_jax() -> None:
    data = [0.5, 1.0, 2.0, 4.0, 8.0]
    x_pyc = pycoeus.Tensor(data, [5], requires_grad=False)
    got = pycoeus.log2(x_pyc)
    exp = jnp.log2(jnp.array(data, dtype=jnp.float64))
    _allclose("log2", list(got.data), exp.tolist(), atol=1e-12)

# ---------------------------------------------------------------------------
# sinh / cosh / log2 / log10 parity
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not hasattr(pycoeus, "sinh"), reason="pycoeus.sinh not available")
def test_sinh_matches_jax() -> None:
    """jnp.sinh(x) vs pycoeus.sinh(x)."""
    data = [-2.0, -1.0, 0.0, 1.0, 2.0]
    x_pyc = pycoeus.Tensor(data, [5], requires_grad=False)
    got = pycoeus.sinh(x_pyc)
    exp = jnp.sinh(jnp.array(data, dtype=jnp.float64))
    _allclose("sinh", list(got.data), exp.tolist(), atol=1e-12)


@pytest.mark.skipif(not hasattr(pycoeus, "cosh"), reason="pycoeus.cosh not available")
def test_cosh_matches_jax() -> None:
    """jnp.cosh(x) vs pycoeus.cosh(x)."""
    data = [-2.0, -1.0, 0.0, 1.0, 2.0]
    x_pyc = pycoeus.Tensor(data, [5], requires_grad=False)
    got = pycoeus.cosh(x_pyc)
    exp = jnp.cosh(jnp.array(data, dtype=jnp.float64))
    _allclose("cosh", list(got.data), exp.tolist(), atol=1e-12)


@pytest.mark.skipif(not hasattr(pycoeus, "log2"), reason="pycoeus.log2 not available")
def test_log2_matches_jax() -> None:
    """jnp.log2(x) vs pycoeus.log2(x)."""
    data = [0.5, 1.0, 2.0, 4.0, 8.0]
    x_pyc = pycoeus.Tensor(data, [5], requires_grad=False)
    got = pycoeus.log2(x_pyc)
    exp = jnp.log2(jnp.array(data, dtype=jnp.float64))
    _allclose("log2", list(got.data), exp.tolist(), atol=1e-12)


@pytest.mark.skipif(not hasattr(pycoeus, "log10"), reason="pycoeus.log10 not available")
def test_log10_matches_jax() -> None:
    """jnp.log10(x) vs pycoeus.log10(x)."""
    data = [0.1, 1.0, 10.0, 100.0, 1000.0]
    x_pyc = pycoeus.Tensor(data, [5], requires_grad=False)
    got = pycoeus.log10(x_pyc)
    exp = jnp.log10(jnp.array(data, dtype=jnp.float64))
    _allclose("log10", list(got.data), exp.tolist(), atol=1e-12)

# ---------------------------------------------------------------------------
# stack / cat / matmul parity
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not hasattr(pycoeus, "stack"), reason="pycoeus.stack not available")
def test_stack_matches_jax() -> None:
    """jnp.stack([a,b], axis=0) vs pycoeus.stack([a,b], dim=0)."""
    a = [1.0, 2.0, 3.0]
    b = [4.0, 5.0, 6.0]
    a_pyc = pycoeus.Tensor(a, [3], requires_grad=False)
    b_pyc = pycoeus.Tensor(b, [3], requires_grad=False)
    got = pycoeus.stack([a_pyc, b_pyc], dim=0)
    a_j = jnp.array(a, dtype=jnp.float64)
    b_j = jnp.array(b, dtype=jnp.float64)
    exp = jnp.stack([a_j, b_j], axis=0)
    _allclose("stack", list(got.data), jnp.ravel(exp).tolist())


@pytest.mark.skipif(not hasattr(pycoeus, "matmul"), reason="pycoeus.matmul not available")
def test_matmul_matches_jax() -> None:
    """jnp.matmul(a, b) vs pycoeus.matmul(a, b)."""
    a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    b = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    a_pyc = pycoeus.Tensor(a, [2, 3], requires_grad=False)
    b_pyc = pycoeus.Tensor(b, [3, 2], requires_grad=False)
    got = pycoeus.matmul(a_pyc, b_pyc)
    a_j = jnp.array(a, dtype=jnp.float64).reshape(2, 3)
    b_j = jnp.array(b, dtype=jnp.float64).reshape(3, 2)
    exp = jnp.matmul(a_j, b_j)
    _allclose("matmul", list(got.data), jnp.ravel(exp).tolist())
