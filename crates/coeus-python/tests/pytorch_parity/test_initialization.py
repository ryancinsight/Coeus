"""PyTorch contract evidence for in-place tensor initialization."""

import math
from pathlib import Path
import statistics
import sys

import pytest

_TEST_ROOT = Path(__file__).resolve().parents[1]
if str(_TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(_TEST_ROOT))

import pycoeus  # noqa: E402

torch = pytest.importorskip("torch")


def _assert_distribution(
    label: str,
    values: list[float],
    expected_mean: float,
    expected_std: float,
) -> None:
    """Check moments under an eight-standard-error sampling bound."""
    count = len(values)
    mean = statistics.fmean(values)
    std = statistics.pstdev(values)
    rounding = 128.0 * sys.float_info.epsilon * max(
        1.0, abs(expected_mean), expected_std
    )
    mean_bound = 8.0 * expected_std / math.sqrt(count) + rounding
    std_bound = 8.0 * expected_std / math.sqrt(2.0 * (count - 1)) + rounding
    assert abs(mean - expected_mean) <= mean_bound, (
        f"{label} mean: {mean=} {expected_mean=} {mean_bound=}"
    )
    assert abs(std - expected_std) <= std_bound, (
        f"{label} std: {std=} {expected_std=} {std_bound=}"
    )


def _compare_distribution(
    label: str,
    shape: list[int],
    coeus_initializer,
    torch_initializer,
    expected_mean: float,
    expected_std: float,
    support: tuple[float, float] | None = None,
) -> None:
    count = math.prod(shape)
    coeus_tensor = pycoeus.Tensor([0.0] * count, shape)
    torch_tensor = torch.empty(shape, dtype=torch.float64)
    coeus_initializer(coeus_tensor)
    torch.manual_seed(42)
    torch_initializer(torch_tensor)
    coeus_values = coeus_tensor.data
    torch_values = torch_tensor.flatten().tolist()
    _assert_distribution(f"Coeus {label}", coeus_values, expected_mean, expected_std)
    _assert_distribution(f"PyTorch {label}", torch_values, expected_mean, expected_std)
    if support is not None:
        lower, upper = support
        for implementation, values in (("Coeus", coeus_values), ("PyTorch", torch_values)):
            assert min(values) >= lower, f"{implementation} {label} violated lower support"
            assert max(values) < upper, f"{implementation} {label} violated upper support"


def test_initializers_match_pytorch_contract() -> None:
    """Every exposed initializer satisfies the corresponding PyTorch law."""
    constant = pycoeus.Tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
    torch_constant = torch.empty([2, 2], dtype=torch.float64)
    for coeus_fn, torch_fn, expected in (
        (
            lambda: pycoeus.init.constant_(constant, -2.5),
            lambda: torch.nn.init.constant_(torch_constant, -2.5),
            -2.5,
        ),
        (lambda: pycoeus.init.zeros_(constant), lambda: torch.nn.init.zeros_(torch_constant), 0.0),
        (lambda: pycoeus.init.ones_(constant), lambda: torch.nn.init.ones_(torch_constant), 1.0),
    ):
        coeus_fn()
        torch_fn()
        assert constant.data == torch_constant.flatten().tolist() == [expected] * 4

    lower, upper = -1.25, 2.75
    uniform_std = (upper - lower) / math.sqrt(12.0)
    _compare_distribution(
        "uniform",
        [16_384],
        lambda tensor: pycoeus.init.uniform_(tensor, lower, upper),
        lambda tensor: torch.nn.init.uniform_(tensor, lower, upper),
        (lower + upper) / 2.0,
        uniform_std,
        (lower, upper),
    )
    normal_mean, normal_std = 0.75, 1.5
    _compare_distribution(
        "normal",
        [16_384],
        lambda tensor: pycoeus.init.normal_(tensor, normal_mean, normal_std),
        lambda tensor: torch.nn.init.normal_(tensor, normal_mean, normal_std),
        normal_mean,
        normal_std,
    )

    fan_in, fan_out = 128, 64
    shape = [fan_out, fan_in]
    xavier_bound = math.sqrt(6.0 / (fan_in + fan_out))
    xavier_std = math.sqrt(2.0 / (fan_in + fan_out))
    _compare_distribution(
        "Xavier uniform",
        shape,
        lambda tensor: pycoeus.init.xavier_uniform_(tensor, fan_in, fan_out),
        torch.nn.init.xavier_uniform_,
        0.0,
        xavier_bound / math.sqrt(3.0),
        (-xavier_bound, xavier_bound),
    )
    _compare_distribution(
        "Xavier normal",
        shape,
        lambda tensor: pycoeus.init.xavier_normal_(tensor, fan_in, fan_out),
        torch.nn.init.xavier_normal_,
        0.0,
        xavier_std,
    )

    kaiming_bound = math.sqrt(6.0 / fan_in)
    kaiming_std = math.sqrt(2.0 / fan_in)
    _compare_distribution(
        "Kaiming uniform",
        shape,
        lambda tensor: pycoeus.init.kaiming_uniform_(tensor, fan_in),
        lambda tensor: torch.nn.init.kaiming_uniform_(tensor, a=0.0),
        0.0,
        kaiming_bound / math.sqrt(3.0),
        (-kaiming_bound, kaiming_bound),
    )
    _compare_distribution(
        "Kaiming normal",
        shape,
        lambda tensor: pycoeus.init.kaiming_normal_(tensor, fan_in),
        lambda tensor: torch.nn.init.kaiming_normal_(tensor, a=0.0),
        0.0,
        kaiming_std,
    )


def test_initializers_reject_invalid_domains() -> None:
    """The Python trust boundary rejects values that could panic or poison output."""
    vector = pycoeus.Tensor([0.0, 0.0], [2])
    scalar = pycoeus.Tensor([0.0], [])
    rank_seven = pycoeus.Tensor([0.0], [1, 1, 1, 1, 1, 1, 1])
    original_vector = vector.data
    invalid_calls = (
        lambda: pycoeus.init.uniform_(vector, 2.0, 1.0),
        lambda: pycoeus.init.uniform_(vector, math.nan, 1.0),
        lambda: pycoeus.init.normal_(vector, 0.0, -1.0),
        lambda: pycoeus.init.normal_(vector, math.inf, 1.0),
        lambda: pycoeus.init.xavier_uniform_(vector, 0, 0),
        lambda: pycoeus.init.xavier_normal_(vector, 0, 0),
        lambda: pycoeus.init.kaiming_uniform_(vector, 0),
        lambda: pycoeus.init.kaiming_normal_(vector, 0),
        lambda: pycoeus.init.kaiming_normal_(vector, -1),
        lambda: pycoeus.init.kaiming_normal_(vector, 1 << 200),
        lambda: pycoeus.init.xavier_normal_(vector, 2 * sys.maxsize + 1, 1),
        lambda: pycoeus.init.uniform_(scalar, 0.0, 1.0),
        lambda: pycoeus.init.normal_(rank_seven, 0.0, 1.0),
        lambda: pycoeus.rand([]),
        lambda: pycoeus.randn([1, 1, 1, 1, 1, 1, 1]),
    )
    for call in invalid_calls:
        with pytest.raises(ValueError):
            call()
        assert vector.data == original_vector
