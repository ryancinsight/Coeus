"""CPU gamma-log acceptance against real PyTorch, JAX and SciPy contracts."""
import math
import json
from pathlib import Path
import numpy as np
import pytest
import torch
import scipy.special
import jax
import jax.numpy as jnp
import pycoeus

jax.config.update("jax_enable_x64", True)

# Additional tight differential regression envelope. It is not a cancellation
# proof. Per-input propagated provider bounds are checked separately below.
REFERENCE = json.loads((Path(__file__).parent / "fixtures/gammaln_reference.json").read_text())
CASES = {row["input"]: row for row in REFERENCE["cases"]}
U = 2.0 ** -53
GAMMA = 132 * U / (1 - 132 * U)
DATA = [1e-30, 1e-6, 0.125, 0.5, 1., 1.4616321449683623,
        math.nextafter(1.4616321449683623, math.inf), 2., 16., 1e6, 1e30,
        -1e-12, -.125, -.5, -.5040830082644554, -1.25, -2.5, -10.25,
        math.nextafter(-1., 0.), math.nextafter(-1., -math.inf)]


def derived_gradient(actual, inputs, weights):
    """Provider absolute enclosure plus rounded VJP and reference products."""
    for value, x, weight in zip(actual, inputs, weights, strict=True):
        row = CASES[x]
        reference = weight * row["digamma"]
        # Round each positive budget operation outward. The denominator
        # translates exact |w*d_ref| to its rounded reference magnitude.
        def up(number):
            return math.nextafter(number, math.inf)
        factor = up(1 + U)
        absolute = up(up(factor * abs(weight)) * row["absolute"])
        relative = up(up(up(factor * row["relative"]) + 2 * U) / (1 - U))
        limit = up(up(absolute + up(relative * abs(reference))) + 2 * math.ulp(0.))
        assert up(abs(value - reference)) <= limit


def framework_comparison(actual, observed, inputs, weights):
    """Preserve the exact known SciPy/PyTorch pole counterexample explicitly."""
    for value, oracle, x, weight in zip(actual, observed, inputs, weights, strict=True):
        if x == math.nextafter(-1., 0.) and oracle == weight * -5545090608933970.0:
            # Recorded real-framework result; it is not the mathematical oracle.
            reference = weight * CASES[x]["digamma"]
            assert value == reference
        else:
            close([value], [oracle])


def close(actual, expected, bound=GAMMA):
    np.testing.assert_allclose(actual, expected, atol=bound, rtol=bound, equal_nan=True)


@pytest.mark.parametrize("name", ["gammaln", "lgamma"])
def test_tracked_alias_weighted_gradients(name):
    weights = np.array([(-1.0 if i % 2 else 1.0) * (i + 1) / 8 for i in range(len(DATA))])
    x = pycoeus.Tensor(DATA, [len(DATA)], requires_grad=True)
    out = getattr(pycoeus, name)(x)
    (out * pycoeus.Tensor(weights.tolist(), [len(DATA)], requires_grad=False)).sum().backward()
    t = torch.tensor(DATA, dtype=torch.float64, requires_grad=True)
    torch_out = torch.lgamma(t)
    (torch_out * torch.tensor(weights, dtype=torch.float64)).sum().backward()
    jax_gradient = jax.grad(lambda z: jnp.sum(jax.scipy.special.gammaln(z) * weights))(jnp.array(DATA))
    derived_gradient(list(x.grad), DATA, weights)
    framework_comparison(list(x.grad), scipy.special.digamma(DATA) * weights, DATA, weights)
    framework_comparison(list(x.grad), t.grad.numpy(), DATA, weights)
    close(list(x.grad), jax_gradient)
    # lgamma forward remains libm-owned, with independent framework algorithms.
    close(list(out.data), torch_out.detach().numpy(), 64 * U)
    assert out.requires_grad


@pytest.mark.parametrize("name", ["gammaln", "lgamma"])
def test_alias_transposed_layout_and_shared_accumulation(name):
    data = [.5, 1., 2., -.5, 3., -1.5]
    x = pycoeus.Tensor(data, [2, 3], requires_grad=True)
    view = x.transpose(0, 1)
    left, right = getattr(pycoeus, name)(view), getattr(pycoeus, name)(view)
    weights = [2., -3., .5, 4., -2., .25]
    ((left + right) * pycoeus.Tensor(weights, [3, 2], requires_grad=False)).sum().backward()
    t = torch.tensor(data, dtype=torch.float64, requires_grad=True)
    transposed = t.reshape(2, 3).transpose(0, 1)
    ((torch.lgamma(transposed) + torch.lgamma(transposed)) * torch.tensor(weights).reshape(3, 2)).sum().backward()
    close(list(x.grad), t.grad.numpy(), 2 * GAMMA)


@pytest.mark.parametrize("name", ["gammaln", "lgamma"])
def test_alias_broadcast_gradient(name):
    x = pycoeus.Tensor([.5, 2.], [1, 2], requires_grad=True)
    getattr(pycoeus, name)(x.broadcast_to([3, 2])).sum().backward()
    close(list(x.grad), 3 * scipy.special.digamma([.5, 2.]), 2 * GAMMA)


@pytest.mark.parametrize("name", ["gammaln", "lgamma"])
def test_alias_domain_signed_zero_poles_and_extreme_inputs(name):
    data = [0., -0., -1., -2., math.inf, -math.inf, math.nan,
            1e-300, -1e-300, 1e300]
    x = pycoeus.Tensor(data, [len(data)], requires_grad=True)
    getattr(pycoeus, name)(x).backward()
    close(list(x.grad), scipy.special.digamma(data))
    assert list(x.grad)[0] == -math.inf
    assert list(x.grad)[1] == math.inf
    # JAX's -0 convention differs; finite and other special cases agree.
    jax_grad = np.asarray(jax.scipy.special.digamma(jnp.array(data)))
    mask = np.arange(len(data)) != 1
    close(np.array(list(x.grad))[mask], jax_grad[mask])


@pytest.mark.parametrize("name", ["gammaln", "lgamma"])
def test_alias_nontracked_forward_preserves_values(name):
    x = pycoeus.Tensor([.5, 1., 2., -.5, -1.25], [5], requires_grad=False)
    result = getattr(pycoeus, name)(x)
    close(list(result.data), scipy.special.gammaln([.5, 1., 2., -.5, -1.25]), 64 * U)
    assert not result.requires_grad


@pytest.mark.parametrize("name", ["gammaln", "lgamma"])
def test_alias_rank_zero_and_empty_shapes(name):
    scalar = pycoeus.Tensor([.5], [], requires_grad=True)
    result = getattr(pycoeus, name)(scalar)
    (result * -2.5).backward()
    assert list(result.shape) == []
    close(list(scalar.grad), [-2.5 * scipy.special.digamma(.5)])
    empty = pycoeus.Tensor([], [0, 3], requires_grad=True)
    result = getattr(pycoeus, name)(empty)
    result.backward()
    assert list(result.shape) == [0, 3]
    assert list(empty.grad) == []


@pytest.mark.parametrize("name", ["gammaln", "lgamma"])
def test_alias_all_high_precision_fixtures_obey_propagated_budget(name):
    data = [row["input"] for row in REFERENCE["cases"]]
    weights = [(-1. if i % 2 else 1.) * ((i % 7) + 1) / 8 for i in range(len(data))]
    x = pycoeus.Tensor(data, [len(data)], requires_grad=True)
    result = getattr(pycoeus, name)(x)
    (result * pycoeus.Tensor(weights, [len(data)], requires_grad=False)).sum().backward()
    derived_gradient(list(x.grad), data, weights)


@pytest.mark.parametrize("name", ["gammaln", "lgamma"])
def test_alias_zero_upstream_keeps_ieee_pole_product(name):
    data = [0., -0., -1., math.inf, -math.inf, math.nan]
    x = pycoeus.Tensor(data, [len(data)], requires_grad=True)
    (getattr(pycoeus, name)(x) * 0.).sum().backward()
    t = torch.tensor(data, dtype=torch.float64, requires_grad=True)
    (torch.lgamma(t) * 0.).sum().backward()
    assert len(list(x.grad)) == len(data)
    assert all(math.isnan(value) for value in x.grad)
    assert all(math.isnan(value) for value in t.grad.tolist())
