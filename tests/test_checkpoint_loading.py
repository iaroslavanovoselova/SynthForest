import pytest

torch = pytest.importorskip("torch")

from synthforest.training.utils import load_trained_state_dict  # noqa: E402


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dinov3 = torch.nn.Linear(2, 2)
        self.head = torch.nn.Linear(2, 1)


def test_released_checkpoints_may_omit_the_frozen_backbone() -> None:
    source, target = _Model(), _Model()
    pretrained = target.dinov3.weight.clone()
    head_only = {k: v for k, v in source.state_dict().items() if not k.startswith("dinov3.")}

    load_trained_state_dict(target, head_only)

    assert torch.equal(target.head.weight, source.head.weight)
    assert torch.equal(target.dinov3.weight, pretrained)


def test_missing_trained_parameters_are_rejected() -> None:
    source, target = _Model(), _Model()
    without_head = {k: v for k, v in source.state_dict().items() if not k.startswith("head.")}
    with pytest.raises(RuntimeError, match="head.weight"):
        load_trained_state_dict(target, without_head)
