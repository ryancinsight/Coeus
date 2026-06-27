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
