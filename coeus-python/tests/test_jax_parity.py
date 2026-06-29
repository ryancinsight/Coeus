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

import os
import sys
import math

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

    dec = pycoeus.TransformerDecoderLayer(d_model=d_model, d_ff=d_ff, num_heads=num_heads)

    tgt_data = [0.1 * i - 0.3 for i in range(batch * seq_tgt * d_model)]
    mem_data = [0.05 * i for i in range(batch * seq_src * d_model)]
    tgt_pyc = pycoeus.Tensor(tgt_data, [batch, seq_tgt, d_model], requires_grad=False)
    mem_pyc = pycoeus.Tensor(mem_data, [batch, seq_src, d_model], requires_grad=False)
    out_pyc = dec.forward(tgt_pyc, mem_pyc)

    tgt = jnp.asarray(tgt_data, dtype=jnp.float64).reshape(batch, seq_tgt, d_model)
    memory = jnp.asarray(mem_data, dtype=jnp.float64).reshape(batch, seq_src, d_model)

    sa_wq = jnp.asarray(list(dec.self_attn.w_q.data), dtype=jnp.float64).reshape(d_model, d_model)
    sa_wk = jnp.asarray(list(dec.self_attn.w_k.data), dtype=jnp.float64).reshape(d_model, d_model)
    sa_wv = jnp.asarray(list(dec.self_attn.w_v.data), dtype=jnp.float64).reshape(d_model, d_model)
    sa_wo = jnp.asarray(list(dec.self_attn.w_o.data), dtype=jnp.float64).reshape(d_model, d_model)
    ca_wq = jnp.asarray(list(dec.cross_attn.w_q.data), dtype=jnp.float64).reshape(d_model, d_model)
    ca_wk = jnp.asarray(list(dec.cross_attn.w_k.data), dtype=jnp.float64).reshape(d_model, d_model)
    ca_wv = jnp.asarray(list(dec.cross_attn.w_v.data), dtype=jnp.float64).reshape(d_model, d_model)
    ca_wo = jnp.asarray(list(dec.cross_attn.w_o.data), dtype=jnp.float64).reshape(d_model, d_model)

    n1_g = jnp.asarray(list(dec.norm1.weight.data), dtype=jnp.float64)
    n1_b = jnp.asarray(list(dec.norm1.bias.data), dtype=jnp.float64)
    n2_g = jnp.asarray(list(dec.norm2.weight.data), dtype=jnp.float64)
    n2_b = jnp.asarray(list(dec.norm2.bias.data), dtype=jnp.float64)
    n3_g = jnp.asarray(list(dec.norm3.weight.data), dtype=jnp.float64)
    n3_b = jnp.asarray(list(dec.norm3.bias.data), dtype=jnp.float64)

    ff1_w = jnp.asarray(list(dec.ffn.linear1.weight.data), dtype=jnp.float64).reshape(d_ff, d_model)
    ff1_b = jnp.asarray(list(dec.ffn.linear1.bias.data), dtype=jnp.float64)
    ff2_w = jnp.asarray(list(dec.ffn.linear2.weight.data), dtype=jnp.float64).reshape(d_model, d_ff)
    ff2_b = jnp.asarray(list(dec.ffn.linear2.bias.data), dtype=jnp.float64)

    n1 = _jax_layer_norm(tgt, n1_g, n1_b, eps=1e-5)
    x1 = tgt + _jax_mha_forward(n1, n1, n1, sa_wq, sa_wk, sa_wv, sa_wo, num_heads)
    n2 = _jax_layer_norm(x1, n2_g, n2_b, eps=1e-5)
    x2 = x1 + _jax_mha_forward(n2, memory, memory, ca_wq, ca_wk, ca_wv, ca_wo, num_heads)
    n3 = _jax_layer_norm(x2, n3_g, n3_b, eps=1e-5)
    ff = jax.nn.gelu(n3 @ ff1_w.T + ff1_b)
    out_jax = x2 + (ff @ ff2_w.T + ff2_b)

    _allclose("decoder_layer_fwd", list(out_pyc.data), out_jax.flatten().tolist(), atol=_ATOL_DEC)


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
    _assert_activation_matches_jax("silu", lambda x: pycoeus.silu(x), jax.nn.silu, _ACT_INPUT)


def test_mish_matches_jax() -> None:
    _assert_activation_matches_jax("mish", lambda x: pycoeus.mish(x), jax.nn.mish, _ACT_INPUT)


def test_elu_matches_jax() -> None:
    _assert_activation_matches_jax("elu", lambda x: pycoeus.elu(x), jax.nn.elu, _ACT_INPUT)


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
    _assert_activation_matches_jax("tanh", lambda x: pycoeus.tanh(x), jnp.tanh, _ACT_INPUT)


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
    gx, gg = jax.grad(lambda z, g: jnp.sum(_jax_rms(z, g)), argnums=(0, 1))(x_jax, g_jax)

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
        math.log(0.7), math.log(0.2), math.log(0.1),
        math.log(0.3), math.log(0.6), math.log(0.1),
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
    _allclose("kl_dinput", list(i_pyc.grad), jax.grad(_jax_kl)(i_jax).flatten().tolist())


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
    idx_pyc = pycoeus.Tensor([float(i) for i in indices], [len(indices)], requires_grad=False)
    out_pyc = emb.forward(idx_pyc)
    out_pyc.sum().backward()

    w_jax = jnp.array(weight, dtype=jnp.float64).reshape(num_embeddings, embedding_dim)
    idx_jax = jnp.array(indices)

    def _jax_emb(w):
        return jnp.sum(w[idx_jax])

    _allclose("emb_out", list(out_pyc.data), w_jax[idx_jax].flatten().tolist())
    _allclose("emb_dweight", list(emb.weight.grad), jax.grad(_jax_emb)(w_jax).flatten().tolist())


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
    gx, gg, gb = jax.grad(lambda z, ga, be: jnp.sum(_jax_gn(z, ga, be)), argnums=(0, 1, 2))(
        x_jax, ga_jax, be_jax
    )

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
        lambda z, y: jnp.mean(jnp.maximum(z, 0.0) - z * y + jnp.log1p(jnp.exp(-jnp.abs(z)))),
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
