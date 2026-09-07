"""CTC value and gradient parity at the logit boundary."""

import math
import sys

import pytest
import pycoeus

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("targets", [[], [1], [1, 1]])
def test_ctc_logits_gradients_match_pytorch(targets: list[int]) -> None:
    logits = [0.0] * 6
    actual = pycoeus.Tensor(logits, [3, 1, 2], requires_grad=True)
    expected = torch.tensor(logits, dtype=torch.float64).reshape(3, 1, 2).requires_grad_(True)
    loss = pycoeus.ctc_loss(actual.log_softmax(2), targets, [3], [len(targets)], 0)
    reference = torch.nn.functional.ctc_loss(
        expected.log_softmax(2), torch.tensor(targets, dtype=torch.long),
        [3], [len(targets)], blank=0, reduction="mean",
    )
    loss.backward()
    reference.backward()
    # Each comparison includes two three-frame, five-state recurrences and
    # two binary log-softmax reductions; twelve operations per state and
    # eight per reduction bound the rounded operation count for this fixture.
    operations = 2 * 3 * 5 * 12 + 2 * 3 * 8
    epsilon = sys.float_info.epsilon
    gamma = operations * epsilon / (1.0 - operations * epsilon)
    assert abs(loss.data[0] - reference.item()) <= gamma * max(1.0, abs(reference.item()))
    for got, want in zip(actual.grad, expected.grad.flatten().tolist(), strict=True):
        assert abs(got - want) <= gamma * max(1.0, abs(want))
    if not targets:
        assert abs(loss.data[0] - 3 * math.log(2)) <= gamma


@pytest.mark.parametrize("targets,inputs,lengths,message", [
    ([], [2], [0], "input length 2 exceeds 1"),
    ([0], [1], [1], "invalid sequence label 0"),
    ([2], [1], [1], "invalid sequence label 2"),
    ([], [], [0], "length counts"),
    ([], [1], [1], "shape mismatch"),
])
def test_ctc_invalid_sequences_raise_value_error(targets, inputs, lengths, message) -> None:
    input_tensor = pycoeus.Tensor([-math.log(2)] * 2, [1, 1, 2])
    with pytest.raises(ValueError, match=message):
        pycoeus.ctc_loss(input_tensor, targets, inputs, lengths, 0)


def test_ctc_impossible_gradient_preserves_input() -> None:
    input_tensor = pycoeus.Tensor([-math.log(2)] * 2, [1, 1, 2], requires_grad=True)
    loss = pycoeus.ctc_loss(input_tensor, [1, 1], [1], [2], 0)
    assert loss.data == [math.inf]
    before = list(input_tensor.grad)
    with pytest.raises(RuntimeError, match="gradient is undefined"):
        loss.backward()
    assert input_tensor.grad == before == [0.0, 0.0]


def test_ctc_loss_matches_pytorch() -> None:
    """CTC loss forward parity against torch.nn.functional.ctc_loss.

    T=5, N=2, C=3 (3 classes, blank=0).
    Two samples with different target lengths.
    Compares pycoeus CTC loss (mean reduction) against PyTorch at f64.
    """
    import torch.nn.functional as F

    T, N, C = 5, 2, 3
    blank = 0

    # Seeded Gaussian logits produce deterministic log probabilities.
    torch.manual_seed(42)
    logits_t = torch.randn(T, N, C, dtype=torch.float64)
    log_probs_t = F.log_softmax(logits_t, dim=2)

    targets_t = torch.tensor([1, 2, 1, 2, 2], dtype=torch.long)  # flat
    input_lengths_t = torch.tensor([5, 5], dtype=torch.long)
    target_lengths_t = torch.tensor([2, 3], dtype=torch.long)

    loss_t = F.ctc_loss(
        log_probs_t,
        targets_t,
        input_lengths_t,
        target_lengths_t,
        blank=blank,
        reduction="mean",
    )

    # pycoeus — pass flat log_probs as [T, N, C] list
    lp_flat = log_probs_t.detach().flatten().tolist()
    x_pyc = pycoeus.Tensor(lp_flat, [T, N, C])
    targets_pyc = [1, 2, 1, 2, 2]
    input_lengths_pyc = [5, 5]
    target_lengths_pyc = [2, 3]

    loss_pyc = pycoeus.ctc_loss(
        x_pyc, targets_pyc, input_lengths_pyc, target_lengths_pyc, blank
    )

    # Five frames, at most seven extended states: each recurrence step
    # contributes at most twelve rounded/elementary operations. Compare two
    # independently ordered recurrences with gamma(2 * 5 * 7 * 12).
    operations = 2 * T * (2 * max(target_lengths_pyc) + 1) * 12
    epsilon = sys.float_info.epsilon
    gamma = operations * epsilon / (1.0 - operations * epsilon)
    assert abs(loss_pyc.data[0] - loss_t.item()) <= gamma * max(1.0, abs(loss_t.item()))
