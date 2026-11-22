"""Forward parity for the full-size Linear, ReLU and mean-square loss graph."""

import numpy as np
import pytest

import pycoeus

mx = pytest.importorskip("mlx.core")


def _comparison_roundoff(roundings: int, dtype) -> float:
    """Sum the relative error bounds for the backend and the f64 reference."""

    def gamma(arithmetic_dtype) -> float:
        u = np.finfo(arithmetic_dtype).eps / 2
        return roundings * u / (1 - roundings * u)

    reference_error = gamma(np.float64)
    # The reference magnitude used to scale the bound can round down.
    return (gamma(dtype) + reference_error) / (1 - reference_error)


def test_linear_matches_mlx() -> None:
    """Verify every activation and both scalar losses at their native precision."""
    in_f, out_f, batch = 256, 64, 128
    linear_pyc = pycoeus.Linear(in_f, out_f, bias=True)
    w_data = linear_pyc.weight.data
    b_data = linear_pyc.bias.data
    x_data = [float(i) * 0.01 for i in range(batch * in_f)]
    tgt_data = [1.0] * (batch * out_f)

    x_pyc = pycoeus.Tensor(x_data, [batch, in_f], requires_grad=True)
    act_pyc = pycoeus.relu(linear_pyc.forward(x_pyc))
    tgt_pyc = pycoeus.Tensor(tgt_data, [batch, out_f])
    loss_pyc = pycoeus.mse_loss(act_pyc, tgt_pyc)

    x_mlx = mx.array(x_data, dtype=mx.float32).reshape(batch, in_f)
    w_mlx = mx.array(w_data, dtype=mx.float32).reshape(out_f, in_f)
    b_mlx = mx.array(b_data, dtype=mx.float32)
    tgt_mlx = mx.array(tgt_data, dtype=mx.float32).reshape(batch, out_f)
    act_mlx = mx.maximum(x_mlx @ w_mlx.T + b_mlx, 0.0)
    residual_mlx = act_mlx - tgt_mlx
    loss_mlx = mx.mean(residual_mlx * residual_mlx)
    mx.eval(act_mlx, loss_mlx)

    x_ref = np.asarray(x_data, dtype=np.float64).reshape(batch, in_f)
    w_ref = np.asarray(w_data, dtype=np.float64).reshape(out_f, in_f)
    b_ref = np.asarray(b_data, dtype=np.float64)
    targets = np.asarray(tgt_data, dtype=np.float64).reshape(batch, out_f)
    act_ref = np.maximum(x_ref @ w_ref.T + b_ref, 0.0)
    dot_scale = np.abs(x_ref) @ np.abs(w_ref).T + np.abs(b_ref)

    observations = (
        ("Coeus", np.asarray(act_pyc.data).reshape(batch, out_f), loss_pyc.data[0], np.float64),
        ("MLX", np.asarray(act_mlx.tolist()), float(loss_mlx), np.float32),
    )
    for label, activations, loss, dtype in observations:
        # A product term encounters up to two input roundings, a multiply,
        # in_f - 1 sum steps and a bias addition. An FMA can only shorten
        # this path. ReLU is 1-Lipschitz, so this also bounds activation error.
        activation_bound = _comparison_roundoff(in_f + 3, dtype) * dot_scale
        activation_error = np.abs(activations - act_ref)
        assert np.all(activation_error <= activation_bound), (
            f"{label}: activation error exceeds its roundoff bound by "
            f"{np.max(activation_error - activation_bound):.3e}"
        )

        # Use each backend's activations in an independent f64 loss oracle.
        # The residual's rounding counts twice in its square, followed by
        # one multiply, N - 1 sum steps and up to two mean-scaling steps.
        # The nonnegative squared terms introduce no reduction cancellation.
        loss_ref = np.mean(np.square(activations - targets))
        loss_bound = _comparison_roundoff(activations.size + 4, dtype) * loss_ref
        assert abs(loss - loss_ref) <= loss_bound, (
            f"{label}: loss {loss:.12g}, reference {loss_ref:.12g}, "
            f"roundoff bound {loss_bound:.3e}"
        )
