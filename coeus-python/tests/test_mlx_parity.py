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


# ---------------------------------------------------------------------------
# MS-384: MLX parity expansion — activations, math, losses, shapes (30 tests)
# ---------------------------------------------------------------------------


def _skip_if_no_mlx():
    if mx is None:
        pytest.skip("MLX not installed")


def _f32(data, shape):
    """Create pycoeus Tensor at default (f64) precision and matching MLX f32 array."""
    x_pyc = pycoeus.Tensor([float(v) for v in data], list(shape), requires_grad=False)
    x_mlx = mx.array([float(v) for v in data]).reshape(*shape)
    return x_pyc, x_mlx


def test_relu_matches_mlx() -> None:
    """relu forward matches mlx.core.maximum(x, 0)."""
    _skip_if_no_mlx()
    data = [-2.0, -0.5, 0.0, 0.3, 1.5]
    x_pyc, x_mlx = _f32(data, [5])
    out_pyc = pycoeus.relu(x_pyc)
    out_mlx = mx.maximum(x_mlx, 0.0)
    mx.eval(out_mlx)
    _allclose("relu", list(out_pyc.data), out_mlx.flatten().tolist())


def test_sigmoid_matches_mlx() -> None:
    """sigmoid forward matches mlx.core.sigmoid."""
    _skip_if_no_mlx()
    data = [-2.0, -1.0, 0.0, 1.0, 2.0]
    x_pyc, x_mlx = _f32(data, [5])
    out_pyc = pycoeus.sigmoid(x_pyc)
    out_mlx = mx.sigmoid(x_mlx)
    mx.eval(out_mlx)
    _allclose("sigmoid", list(out_pyc.data), out_mlx.flatten().tolist())


def test_tanh_matches_mlx() -> None:
    """tanh forward matches mlx.core.tanh."""
    _skip_if_no_mlx()
    data = [-1.0, -0.5, 0.0, 0.5, 1.0]
    x_pyc, x_mlx = _f32(data, [5])
    out_pyc = pycoeus.tanh(x_pyc)
    out_mlx = mx.tanh(x_mlx)
    mx.eval(out_mlx)
    _allclose("tanh", list(out_pyc.data), out_mlx.flatten().tolist())


def test_gelu_matches_mlx() -> None:
    """gelu forward matches mlx.core.erf-based GELU."""
    _skip_if_no_mlx()
    data = [-1.5, -0.5, 0.0, 0.5, 1.5]
    x_pyc, x_mlx = _f32(data, [5])
    out_pyc = pycoeus.gelu(x_pyc)
    # MLX: x * 0.5 * (1 + erf(x / sqrt(2)))
    out_mlx = x_mlx * 0.5 * (1.0 + mx.erf(x_mlx / math.sqrt(2)))
    mx.eval(out_mlx)
    _allclose("gelu", list(out_pyc.data), out_mlx.flatten().tolist(), atol=2e-3)


def test_silu_matches_mlx() -> None:
    """silu forward: x * sigmoid(x) matches MLX."""
    _skip_if_no_mlx()
    data = [-1.0, -0.5, 0.0, 0.5, 1.0]
    x_pyc, x_mlx = _f32(data, [5])
    out_pyc = pycoeus.silu(x_pyc)
    out_mlx = x_mlx * mx.sigmoid(x_mlx)
    mx.eval(out_mlx)
    _allclose("silu", list(out_pyc.data), out_mlx.flatten().tolist())


def test_softmax_matches_mlx() -> None:
    """softmax(dim=1) forward matches mlx.core.softmax."""
    _skip_if_no_mlx()
    data = [1.0, 2.0, 3.0, 0.5, 1.5, 2.5]
    x_pyc = pycoeus.Tensor(data, [2, 3], requires_grad=False)
    x_mlx = mx.array(data).reshape(2, 3)
    out_pyc = pycoeus.softmax(x_pyc, 1)
    out_mlx = mx.softmax(x_mlx, axis=1)
    mx.eval(out_mlx)
    _allclose("softmax", list(out_pyc.data), out_mlx.flatten().tolist())


def test_log_softmax_matches_mlx() -> None:
    """log_softmax(dim=1) forward matches MLX log_softmax."""
    _skip_if_no_mlx()
    data = [1.0, 2.0, 3.0, 0.5, 1.5, 2.5]
    x_pyc = pycoeus.Tensor(data, [2, 3], requires_grad=False)
    x_mlx = mx.array(data).reshape(2, 3)
    out_pyc = pycoeus.log_softmax(x_pyc, 1)
    out_mlx = mx.log(mx.softmax(x_mlx, axis=1))
    mx.eval(out_mlx)
    _allclose("log_softmax", list(out_pyc.data), out_mlx.flatten().tolist())


def test_abs_matches_mlx() -> None:
    """abs forward matches mlx.core.abs."""
    _skip_if_no_mlx()
    data = [-3.0, -1.0, 0.0, 1.5, 2.5]
    x_pyc, x_mlx = _f32(data, [5])
    out_pyc = pycoeus.abs(x_pyc)
    out_mlx = mx.abs(x_mlx)
    mx.eval(out_mlx)
    _allclose("abs", list(out_pyc.data), out_mlx.flatten().tolist())


def test_sqrt_matches_mlx() -> None:
    """sqrt forward matches mlx.core.sqrt."""
    _skip_if_no_mlx()
    data = [0.25, 1.0, 2.25, 4.0, 9.0]
    x_pyc, x_mlx = _f32(data, [5])
    out_pyc = pycoeus.sqrt(x_pyc)
    out_mlx = mx.sqrt(x_mlx)
    mx.eval(out_mlx)
    _allclose("sqrt", list(out_pyc.data), out_mlx.flatten().tolist())


def test_exp_matches_mlx() -> None:
    """exp forward matches mlx.core.exp."""
    _skip_if_no_mlx()
    data = [-1.0, 0.0, 0.5, 1.0, 2.0]
    x_pyc, x_mlx = _f32(data, [5])
    out_pyc = pycoeus.exp(x_pyc)
    out_mlx = mx.exp(x_mlx)
    mx.eval(out_mlx)
    _allclose("exp", list(out_pyc.data), out_mlx.flatten().tolist())


def test_log_matches_mlx() -> None:
    """log forward matches mlx.core.log."""
    _skip_if_no_mlx()
    data = [0.5, 1.0, 2.0, 4.0, 8.0]
    x_pyc, x_mlx = _f32(data, [5])
    out_pyc = pycoeus.log(x_pyc)
    out_mlx = mx.log(x_mlx)
    mx.eval(out_mlx)
    _allclose("log", list(out_pyc.data), out_mlx.flatten().tolist())


def test_neg_matches_mlx() -> None:
    """neg forward matches MLX element-wise negation."""
    _skip_if_no_mlx()
    data = [1.0, -2.0, 3.5, -0.5]
    x_pyc, x_mlx = _f32(data, [4])
    out_pyc = pycoeus.neg(x_pyc)
    out_mlx = -x_mlx
    mx.eval(out_mlx)
    _allclose("neg", list(out_pyc.data), out_mlx.flatten().tolist())


def test_matmul_matches_mlx() -> None:
    """matmul (2x3) @ (3x2) forward matches mlx.core.matmul."""
    _skip_if_no_mlx()
    a_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    b_data = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    a_pyc = pycoeus.Tensor(a_data, [2, 3], requires_grad=False)
    b_pyc = pycoeus.Tensor(b_data, [3, 2], requires_grad=False)
    out_pyc = pycoeus.matmul(a_pyc, b_pyc)
    a_mlx = mx.array(a_data).reshape(2, 3)
    b_mlx = mx.array(b_data).reshape(3, 2)
    out_mlx = a_mlx @ b_mlx
    mx.eval(out_mlx)
    _allclose("matmul", list(out_pyc.data), out_mlx.flatten().tolist())


def test_add_matches_mlx() -> None:
    """element-wise add matches MLX."""
    _skip_if_no_mlx()
    a = [1.0, 2.0, 3.0]; b = [4.0, 5.0, 6.0]
    a_pyc = pycoeus.Tensor(a, [3], requires_grad=False)
    b_pyc = pycoeus.Tensor(b, [3], requires_grad=False)
    out_pyc = a_pyc + b_pyc
    out_mlx = mx.array(a) + mx.array(b)
    mx.eval(out_mlx)
    _allclose("add", list(out_pyc.data), out_mlx.flatten().tolist())


def test_mul_matches_mlx() -> None:
    """element-wise mul matches MLX."""
    _skip_if_no_mlx()
    a = [1.0, 2.0, 3.0]; b = [4.0, 5.0, 6.0]
    a_pyc = pycoeus.Tensor(a, [3], requires_grad=False)
    b_pyc = pycoeus.Tensor(b, [3], requires_grad=False)
    out_pyc = a_pyc * b_pyc
    out_mlx = mx.array(a) * mx.array(b)
    mx.eval(out_mlx)
    _allclose("mul", list(out_pyc.data), out_mlx.flatten().tolist())


def test_mse_loss_matches_mlx() -> None:
    """mse_loss forward matches MLX mean squared error."""
    _skip_if_no_mlx()
    pred = [1.5, -0.5, 2.0, 0.1]
    tgt = [1.0, 0.0, 1.5, -0.2]
    p_pyc = pycoeus.Tensor(pred, [4], requires_grad=False)
    t_pyc = pycoeus.Tensor(tgt, [4], requires_grad=False)
    loss_pyc = pycoeus.mse_loss(p_pyc, t_pyc)
    p_mlx = mx.array(pred); t_mlx = mx.array(tgt)
    diff = p_mlx - t_mlx
    loss_mlx = mx.mean(diff * diff)
    mx.eval(loss_mlx)
    _allclose("mse", [float(loss_pyc.data[0])], [float(loss_mlx)])


def test_l1_loss_matches_mlx() -> None:
    """l1_loss forward matches MLX mean absolute error."""
    _skip_if_no_mlx()
    pred = [1.5, -0.5, 2.0, 0.1]
    tgt = [1.0, 0.0, 1.5, -0.2]
    p_pyc = pycoeus.Tensor(pred, [4], requires_grad=False)
    t_pyc = pycoeus.Tensor(tgt, [4], requires_grad=False)
    loss_pyc = pycoeus.l1_loss(p_pyc, t_pyc)
    p_mlx = mx.array(pred); t_mlx = mx.array(tgt)
    loss_mlx = mx.mean(mx.abs(p_mlx - t_mlx))
    mx.eval(loss_mlx)
    _allclose("l1", [float(loss_pyc.data[0])], [float(loss_mlx)])


def test_reshape_matches_mlx() -> None:
    """reshape (2x6 → 3x4) matches MLX reshape."""
    _skip_if_no_mlx()
    data = [float(i) for i in range(12)]
    x_pyc = pycoeus.Tensor(data, [2, 6], requires_grad=False)
    out_pyc = pycoeus.reshape(x_pyc, [3, 4])
    x_mlx = mx.array(data).reshape(2, 6)
    out_mlx = x_mlx.reshape(3, 4)
    mx.eval(out_mlx)
    _allclose("reshape", list(out_pyc.data), out_mlx.flatten().tolist())


def test_permute_matches_mlx() -> None:
    """permute (0,2,1) matches MLX transpose."""
    _skip_if_no_mlx()
    data = [float(i) for i in range(24)]
    x_pyc = pycoeus.Tensor(data, [2, 3, 4], requires_grad=False)
    out_pyc = pycoeus.permute(x_pyc, [0, 2, 1])
    x_mlx = mx.array(data).reshape(2, 3, 4)
    out_mlx = mx.transpose(x_mlx, (0, 2, 1))
    mx.eval(out_mlx)
    _allclose("permute", list(out_pyc.data), out_mlx.flatten().tolist())


def test_cat_matches_mlx() -> None:
    """cat(dim=0) matches MLX concatenate."""
    _skip_if_no_mlx()
    a = [1.0, 2.0, 3.0]; b = [4.0, 5.0, 6.0]
    a_pyc = pycoeus.Tensor(a, [3], requires_grad=False)
    b_pyc = pycoeus.Tensor(b, [3], requires_grad=False)
    out_pyc = pycoeus.cat([a_pyc, b_pyc], 0)
    out_mlx = mx.concatenate([mx.array(a), mx.array(b)], axis=0)
    mx.eval(out_mlx)
    _allclose("cat", list(out_pyc.data), out_mlx.flatten().tolist())


def test_sum_matches_mlx() -> None:
    """sum() global matches MLX sum."""
    _skip_if_no_mlx()
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    x_pyc = pycoeus.Tensor(data, [2, 3], requires_grad=False)
    out_pyc = pycoeus.sum(x_pyc)
    out_mlx = mx.sum(mx.array(data).reshape(2, 3))
    mx.eval(out_mlx)
    _allclose("sum", list(out_pyc.data), [float(out_mlx)])


def test_mean_matches_mlx() -> None:
    """mean() global matches MLX mean."""
    _skip_if_no_mlx()
    data = [1.0, 2.0, 3.0, 4.0]
    x_pyc = pycoeus.Tensor(data, [4], requires_grad=False)
    out_pyc = pycoeus.mean(x_pyc)
    out_mlx = mx.mean(mx.array(data))
    mx.eval(out_mlx)
    _allclose("mean", list(out_pyc.data), [float(out_mlx)])


def test_layer_norm_matches_mlx() -> None:
    """LayerNorm(4) forward matches MLX layer_norm."""
    _skip_if_no_mlx()
    n, d = 2, 4
    data = [1.0, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5]
    x_pyc = pycoeus.Tensor(data, [n, d], requires_grad=False)
    ln = pycoeus.LayerNorm(d)
    out_pyc = ln.forward(x_pyc)
    x_mlx = mx.array(data).reshape(n, d)
    # MLX layer_norm: (x - mean) / sqrt(var + eps) * weight + bias
    # with weight=ones, bias=zeros
    mean = mx.mean(x_mlx, axis=-1, keepdims=True)
    var = mx.mean((x_mlx - mean) ** 2, axis=-1, keepdims=True)
    out_mlx = (x_mlx - mean) / mx.sqrt(var + 1e-5)
    mx.eval(out_mlx)
    _allclose("ln", list(out_pyc.data), out_mlx.flatten().tolist(), atol=2e-3)


def test_embedding_matches_mlx() -> None:
    """Embedding(8, 4) forward matches MLX embedding lookup."""
    _skip_if_no_mlx()
    vocab, dim = 8, 4
    indices = [0, 2, 5, 1]
    w_data = [float(i) * 0.1 for i in range(vocab * dim)]
    embed = pycoeus.Embedding(num_embeddings=vocab, embedding_dim=dim)
    embed.parameters()[0].data = w_data
    idx_pyc = pycoeus.Tensor([float(i) for i in indices], [4], requires_grad=False)
    out_pyc = embed.forward(idx_pyc)
    w_mlx = mx.array(w_data).reshape(vocab, dim)
    out_mlx = w_mlx[mx.array(indices)]
    mx.eval(out_mlx)
    _allclose("embed", list(out_pyc.data), out_mlx.flatten().tolist())


def test_clamp_matches_mlx() -> None:
    """clamp(-0.5, 0.5) forward matches MLX clip."""
    _skip_if_no_mlx()
    data = [-1.0, -0.3, 0.0, 0.4, 1.0]
    x_pyc, x_mlx = _f32(data, [5])
    out_pyc = pycoeus.clamp(x_pyc, -0.5, 0.5)
    out_mlx = mx.clip(x_mlx, -0.5, 0.5)
    mx.eval(out_mlx)
    _allclose("clamp", list(out_pyc.data), out_mlx.flatten().tolist())


def test_erf_matches_mlx() -> None:
    """erf forward matches mlx.core.erf."""
    _skip_if_no_mlx()
    data = [-1.5, -0.5, 0.0, 0.5, 1.5]
    x_pyc, x_mlx = _f32(data, [5])
    out_pyc = pycoeus.erf(x_pyc)
    out_mlx = mx.erf(x_mlx)
    mx.eval(out_mlx)
    _allclose("erf", list(out_pyc.data), out_mlx.flatten().tolist())


def test_cumsum_matches_mlx() -> None:
    """cumsum(dim=0) forward matches MLX cumsum."""
    _skip_if_no_mlx()
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    x_pyc, x_mlx = _f32(data, [5])
    out_pyc = pycoeus.cumsum(x_pyc, 0)
    out_mlx = mx.cumsum(x_mlx, axis=0)
    mx.eval(out_mlx)
    _allclose("cumsum", list(out_pyc.data), out_mlx.flatten().tolist())


def test_max_axis_matches_mlx() -> None:
    """max_axis(dim=1) forward matches MLX max."""
    _skip_if_no_mlx()
    data = [3.0, 1.0, 4.0, 2.0, 5.0, 0.0]
    x_pyc = pycoeus.Tensor(data, [2, 3], requires_grad=False)
    out_pyc = pycoeus.max_axis(x_pyc, 1)
    x_mlx = mx.array(data).reshape(2, 3)
    out_mlx = mx.max(x_mlx, axis=1, keepdims=True)
    mx.eval(out_mlx)
    _allclose("max_axis", list(out_pyc.data), out_mlx.flatten().tolist())


def test_flip_matches_mlx() -> None:
    """flip(axis=0) forward matches MLX flip."""
    _skip_if_no_mlx()
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    x_pyc = pycoeus.Tensor(data, [2, 3], requires_grad=False)
    out_pyc = pycoeus.flip(x_pyc, 0)
    x_mlx = mx.array(data).reshape(2, 3)
    # MLX flip: mlx.core.flip(x, axis)
    try:
        out_mlx = mx.flip(x_mlx, axis=0)
    except AttributeError:
        # older MLX uses slice indexing
        out_mlx = x_mlx[::-1]
    mx.eval(out_mlx)
    _allclose("flip", list(out_pyc.data), out_mlx.flatten().tolist())


def test_tile_matches_mlx() -> None:
    """tile([2,2]) matches MLX tile."""
    _skip_if_no_mlx()
    data = [1.0, 2.0, 3.0, 4.0]
    x_pyc = pycoeus.Tensor(data, [2, 2], requires_grad=False)
    out_pyc = pycoeus.tile(x_pyc, [2, 2])
    x_mlx = mx.array(data).reshape(2, 2)
    out_mlx = mx.tile(x_mlx, (2, 2))
    mx.eval(out_mlx)
    _allclose("tile", list(out_pyc.data), out_mlx.flatten().tolist())


# ---------------------------------------------------------------------------
# MS-403: MLX parity expansion toward 50 tests
# ---------------------------------------------------------------------------


def test_min_axis_matches_mlx() -> None:
    """min_axis(dim=1) forward matches MLX min."""
    _skip_if_no_mlx()
    data = [3.0, 1.0, 4.0, 2.0, 5.0, 0.0]
    x_pyc = pycoeus.Tensor(data, [2, 3], requires_grad=False)
    out_pyc = pycoeus.min_axis(x_pyc, 1)
    x_mlx = mx.array(data).reshape(2, 3)
    out_mlx = mx.min(x_mlx, axis=1, keepdims=True)
    mx.eval(out_mlx)
    _allclose("min_axis", list(out_pyc.data), out_mlx.flatten().tolist())


def test_sum_axis_matches_mlx() -> None:
    """sum_axis(dim=0) forward matches MLX sum."""
    _skip_if_no_mlx()
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    x_pyc = pycoeus.Tensor(data, [2, 3], requires_grad=False)
    out_pyc = pycoeus.sum_axis(x_pyc, 0)
    x_mlx = mx.array(data).reshape(2, 3)
    out_mlx = mx.sum(x_mlx, axis=0, keepdims=False)
    mx.eval(out_mlx)
    _allclose("sum_axis", list(out_pyc.data), out_mlx.flatten().tolist())


def test_mean_axis_matches_mlx() -> None:
    """mean_axis(dim=1) forward matches MLX mean."""
    _skip_if_no_mlx()
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    x_pyc = pycoeus.Tensor(data, [2, 3], requires_grad=False)
    out_pyc = pycoeus.mean_axis(x_pyc, 1)
    x_mlx = mx.array(data).reshape(2, 3)
    out_mlx = mx.mean(x_mlx, axis=1, keepdims=True)
    mx.eval(out_mlx)
    _allclose("mean_axis", list(out_pyc.data), out_mlx.flatten().tolist())


def test_stack_matches_mlx() -> None:
    """stack(dim=0) forward matches MLX stack."""
    _skip_if_no_mlx()
    a = [1.0, 2.0, 3.0]; b = [4.0, 5.0, 6.0]
    a_pyc = pycoeus.Tensor(a, [3], requires_grad=False)
    b_pyc = pycoeus.Tensor(b, [3], requires_grad=False)
    out_pyc = pycoeus.stack([a_pyc, b_pyc], 0)
    out_mlx = mx.stack([mx.array(a), mx.array(b)], axis=0)
    mx.eval(out_mlx)
    _allclose("stack", list(out_pyc.data), out_mlx.flatten().tolist())


def test_softplus_matches_mlx() -> None:
    """softplus forward matches MLX log(1+exp(x))."""
    _skip_if_no_mlx()
    data = [-2.0, -1.0, 0.0, 1.0, 2.0]
    x_pyc, x_mlx = _f32(data, [5])
    out_pyc = pycoeus.softplus(x_pyc)
    out_mlx = mx.log(1.0 + mx.exp(x_mlx))
    mx.eval(out_mlx)
    _allclose("softplus", list(out_pyc.data), out_mlx.flatten().tolist())


def test_hardsigmoid_matches_mlx() -> None:
    """hardsigmoid forward matches MLX clip((x+3)/6, 0, 1)."""
    _skip_if_no_mlx()
    data = [-4.0, -1.0, 0.0, 1.5, 4.0]
    x_pyc, x_mlx = _f32(data, [5])
    out_pyc = pycoeus.hardsigmoid(x_pyc)
    out_mlx = mx.clip((x_mlx + 3.0) / 6.0, 0.0, 1.0)
    mx.eval(out_mlx)
    _allclose("hardsig", list(out_pyc.data), out_mlx.flatten().tolist())


def test_hardswish_matches_mlx() -> None:
    """hardswish forward matches MLX x * hardsigmoid(x)."""
    _skip_if_no_mlx()
    data = [-4.0, -1.0, 0.0, 1.5, 4.0]
    x_pyc, x_mlx = _f32(data, [5])
    out_pyc = pycoeus.hardswish(x_pyc)
    hs = mx.clip((x_mlx + 3.0) / 6.0, 0.0, 1.0)
    out_mlx = x_mlx * hs
    mx.eval(out_mlx)
    _allclose("hardswish", list(out_pyc.data), out_mlx.flatten().tolist())


def test_leaky_relu_matches_mlx() -> None:
    """leaky_relu(0.1) forward matches MLX where."""
    _skip_if_no_mlx()
    data = [-2.0, -0.5, 0.0, 0.5, 2.0]
    x_pyc, x_mlx = _f32(data, [5])
    out_pyc = pycoeus.leaky_relu(x_pyc, 0.1)
    out_mlx = mx.where(x_mlx > 0, x_mlx, 0.1 * x_mlx)
    mx.eval(out_mlx)
    _allclose("leaky_relu", list(out_pyc.data), out_mlx.flatten().tolist())


def test_elu_matches_mlx() -> None:
    """elu(alpha=1) forward matches MLX where."""
    _skip_if_no_mlx()
    data = [-2.0, -0.5, 0.0, 0.5, 1.5]
    x_pyc, x_mlx = _f32(data, [5])
    out_pyc = pycoeus.elu(x_pyc)
    out_mlx = mx.where(x_mlx >= 0, x_mlx, mx.exp(x_mlx) - 1.0)
    mx.eval(out_mlx)
    _allclose("elu", list(out_pyc.data), out_mlx.flatten().tolist())


def test_prod_matches_mlx() -> None:
    """prod() global matches MLX product."""
    _skip_if_no_mlx()
    data = [1.0, 2.0, 3.0, 4.0]
    x_pyc = pycoeus.Tensor(data, [4], requires_grad=False)
    out_pyc = pycoeus.prod(x_pyc)
    out_mlx = mx.prod(mx.array(data))
    mx.eval(out_mlx)
    _allclose("prod", list(out_pyc.data), [float(out_mlx)])


def test_norm_matches_mlx() -> None:
    """L2 norm forward matches MLX sqrt(sum(x**2))."""
    _skip_if_no_mlx()
    data = [3.0, 4.0, 0.0, 12.0]
    x_pyc, x_mlx = _f32(data, [4])
    out_pyc = pycoeus.norm(x_pyc)
    out_mlx = mx.sqrt(mx.sum(x_mlx ** 2))
    mx.eval(out_mlx)
    _allclose("norm", list(out_pyc.data), [float(out_mlx)])


def test_floor_ceil_matches_mlx() -> None:
    """floor/ceil forward matches MLX floor/ceil."""
    _skip_if_no_mlx()
    data = [0.3, 1.7, -0.5, 2.9, -1.1]
    x_pyc, x_mlx = _f32(data, [5])
    floor_pyc = pycoeus.floor(x_pyc)
    ceil_pyc = pycoeus.ceil(x_pyc)
    _allclose("floor", list(floor_pyc.data), mx.floor(x_mlx).flatten().tolist())
    _allclose("ceil", list(ceil_pyc.data), mx.ceil(x_mlx).flatten().tolist())


def test_arange_matches_mlx() -> None:
    """arange(0, 5, 1) forward matches MLX arange."""
    _skip_if_no_mlx()
    out_pyc = pycoeus.arange(0.0, 5.0, 1.0)
    out_mlx = mx.arange(0, 5, 1, dtype=mx.float32)
    mx.eval(out_mlx)
    _allclose("arange", list(out_pyc.data), out_mlx.tolist())


def test_scatter_add_matches_mlx() -> None:
    """scatter_add forward matches MLX scatter-accumulate."""
    _skip_if_no_mlx()
    src = [1.0, 2.0, 3.0]
    inp = [0.0, 0.0, 0.0, 0.0, 0.0]
    idx = [4.0, 1.0, 3.0]
    src_pyc = pycoeus.Tensor(src, [3], requires_grad=False)
    inp_pyc = pycoeus.Tensor(inp, [5], requires_grad=False)
    idx_pyc = pycoeus.Tensor(idx, [3], requires_grad=False)
    out_pyc = pycoeus.scatter_add(inp_pyc, 0, idx_pyc, src_pyc)
    # MLX: scatter + add
    out_arr = mx.array(inp)
    idx_arr = mx.array([4, 1, 3])
    src_arr = mx.array(src)
    out_mlx = out_arr.at[idx_arr].add(src_arr)
    mx.eval(out_mlx)
    _allclose("scatter_add", list(out_pyc.data), out_mlx.tolist())