"""Value-semantic PyTorch parity for sequential composition."""

from pathlib import Path
import sys

import pytest

_TEST_ROOT = Path(__file__).resolve().parents[1]
if str(_TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(_TEST_ROOT))

import pycoeus  # noqa: E402

torch = pytest.importorskip("torch")


def _assert_f64_close(label: str, actual: list[float], expected: list[float]) -> None:
    """Compare short f64 kernels under a rounding-error-derived bound."""
    assert len(actual) == len(expected), label
    epsilon = sys.float_info.epsilon
    for index, (got, want) in enumerate(zip(actual, expected, strict=True)):
        # Each output traverses two four-term affine kernels. The 64-epsilon
        # factor conservatively covers their multiply/add and backward trees.
        tolerance = 64.0 * epsilon * max(1.0, abs(got), abs(want))
        assert abs(got - want) <= tolerance, (
            f"{label}[{index}]: {got=} {want=} {tolerance=}"
        )


def test_sequential_matches_pytorch() -> None:
    """Composition, parameter order, and every gradient match PyTorch."""
    first = pycoeus.Linear(3, 4)
    second = pycoeus.Linear(4, 2)
    first_weight = [0.2, -0.1, 0.4, -0.3, 0.5, 0.7, 0.6, -0.2, 0.1, -0.8, 0.9, 0.3]
    first_bias = [0.05, -0.15, 0.25, -0.35]
    second_weight = [0.4, -0.6, 0.8, 0.2, -0.5, 0.7, -0.1, 0.9]
    second_bias = [0.3, -0.4]
    first.weight.data = first_weight
    first.bias.data = first_bias
    second.weight.data = second_weight
    second.bias.data = second_bias

    sequential = pycoeus.Sequential([first])
    sequential.append(second)
    assert len(sequential) == 2
    assert sequential[0] is first
    assert sequential[-1] is second
    parameters = sequential.parameters()
    assert parameters == [first.weight, first.bias, second.weight, second.bias]

    input_values = [0.5, -1.0, 1.5, -0.25, 0.75, 2.0]
    coeus_input = pycoeus.Tensor(input_values, [2, 3], requires_grad=True)
    coeus_output = sequential.forward(coeus_input)
    coeus_output.sum().backward()

    torch_first = torch.nn.Linear(3, 4, dtype=torch.float64)
    torch_second = torch.nn.Linear(4, 2, dtype=torch.float64)
    with torch.no_grad():
        torch_first.weight.copy_(
            torch.tensor(first_weight, dtype=torch.float64).reshape(4, 3)
        )
        torch_first.bias.copy_(torch.tensor(first_bias, dtype=torch.float64))
        torch_second.weight.copy_(
            torch.tensor(second_weight, dtype=torch.float64).reshape(2, 4)
        )
        torch_second.bias.copy_(torch.tensor(second_bias, dtype=torch.float64))
    torch_sequential = torch.nn.Sequential(torch_first, torch_second)
    torch_input = (
        torch.tensor(input_values, dtype=torch.float64)
        .reshape(2, 3)
        .requires_grad_(True)
    )
    torch_output = torch_sequential(torch_input)
    torch_output.sum().backward()

    _assert_f64_close("output", coeus_output.data, torch_output.flatten().tolist())
    _assert_f64_close("input gradient", coeus_input.grad, torch_input.grad.flatten().tolist())
    for index, (coeus_parameter, torch_parameter) in enumerate(
        zip(parameters, torch_sequential.parameters(), strict=True)
    ):
        _assert_f64_close(
            f"parameter gradient {index}",
            coeus_parameter.grad,
            torch_parameter.grad.flatten().tolist(),
        )

    sequential.zero_grad()
    for parameter in parameters:
        assert parameter.grad is not None
        assert all(value == 0.0 for value in parameter.grad)

    empty = pycoeus.Sequential([])
    identity_input = pycoeus.Tensor(input_values, [2, 3], requires_grad=True)
    identity = empty.forward(identity_input)
    identity.sum().backward()
    assert identity.shape == identity_input.shape
    assert identity.data == identity_input.data
    assert identity_input.grad == [1.0] * len(input_values)
