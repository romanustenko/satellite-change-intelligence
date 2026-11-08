"""Dense semantic change detection adapters: loss, epoch runner, dataset and transform."""

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from satchange.data.transforms import RandomFlipRotate
from satchange.models import FCSiamDiff
from satchange.models.factory import SEMANTIC_TASK, create_task_model
from satchange.semantic.scd_training import (
    NUM_CLASSES,
    SemanticSCDLoss,
    predict_maps,
    run_epoch,
    split_logits,
)
from satchange.training import fit_binary_change_model


def _loader(batches: int = 3, size: int = 16) -> list[dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(0)
    out = []
    for _ in range(batches):
        label = torch.randint(0, NUM_CLASSES, (2, size, size), generator=generator)
        out.append({
            "before": torch.rand(2, 3, size, size, generator=generator),
            "after": torch.rand(2, 3, size, size, generator=generator),
            "before_label": label,
            "after_label": label.clone(),
        })
    return out


def test_split_logits_validates_channels() -> None:
    first, second = split_logits(torch.zeros(1, 14, 4, 4))
    assert first.shape == second.shape == (1, 7, 4, 4)
    for bad in (torch.zeros(1, 13, 4, 4), torch.zeros(14, 4, 4)):
        try:
            split_logits(bad)
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


def test_loss_matches_manual_cross_entropy_and_weights_validated() -> None:
    logits = torch.randn(2, 14, 4, 4)
    label = torch.randint(0, 7, (2, 4, 4))
    loss = SemanticSCDLoss()(logits, label, label)
    first, second = logits[:, :7], logits[:, 7:]
    expected = torch.nn.functional.cross_entropy(first, label) + torch.nn.functional.cross_entropy(
        second, label
    )
    assert torch.allclose(loss, expected)
    for weights in ([1.0] * 6, [1.0] * 6 + [0.0]):
        try:
            SemanticSCDLoss(weights)
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


def test_predict_maps_shapes_and_range() -> None:
    first, second = predict_maps(torch.randn(2, 14, 5, 6))
    assert first.shape == second.shape == (2, 5, 6)
    assert first.dtype == torch.int64 and int(first.max()) < 7


def test_run_epoch_trains_and_reports_pooled_metrics() -> None:
    torch.manual_seed(0)
    model = FCSiamDiff(in_channels=3, out_channels=14, base_channels=4)
    loss_function = SemanticSCDLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    data = _loader()
    first = run_epoch(model, data, loss_function, torch.device("cpu"), optimizer=optimizer)  # type: ignore[arg-type]
    for _ in range(5):
        last = run_epoch(model, data, loss_function, torch.device("cpu"), optimizer=optimizer)  # type: ignore[arg-type]
    assert last["loss"] < first["loss"]
    evaluation = run_epoch(model, data, loss_function, torch.device("cpu"))  # type: ignore[arg-type]
    assert evaluation["selection_score"] == evaluation["sek"]
    assert evaluation["samples"] == 6
    assert {"miou", "fscd", "kappa_n0", "loss"} <= evaluation.keys()


def test_transform_flips_extra_label_maps_with_the_images() -> None:
    sample = {
        "before": torch.arange(48.0).reshape(3, 4, 4),
        "after": torch.zeros(3, 4, 4),
        "mask": torch.zeros(1, 4, 4),
        "before_label": torch.arange(16).reshape(4, 4),
        "after_label": torch.arange(16).reshape(4, 4),
        "sample_id": "x",
    }
    transform = RandomFlipRotate(extra_keys=("before_label", "after_label"))
    for seed in range(8):
        torch.manual_seed(seed)
        out = transform(sample)
        assert torch.equal(out["before_label"], out["before"][0].long())
        assert torch.equal(out["after_label"], out["before_label"])


def _samples(count: int = 4, size: int = 16) -> list[dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(1)
    samples = []
    for _ in range(count):
        label = torch.randint(0, NUM_CLASSES, (size, size), generator=generator)
        samples.append({
            "before": torch.rand(3, size, size, generator=generator),
            "after": torch.rand(3, size, size, generator=generator),
            "before_label": label,
            "after_label": label.clone(),
        })
    return samples


def test_semantic_model_needs_fourteen_output_channels() -> None:
    parameters = {"in_channels": 3, "out_channels": 14, "base_channels": 4}
    model = create_task_model("fc_siam_diff", parameters, SEMANTIC_TASK)
    assert model(torch.rand(1, 3, 16, 16), torch.rand(1, 3, 16, 16)).shape == (1, 14, 16, 16)
    with pytest.raises(ValueError, match="out_channels=14"):
        create_task_model("fc_siam_diff", {**parameters, "out_channels": 1}, SEMANTIC_TASK)
    with pytest.raises(ValueError):
        create_task_model("bit", parameters, SEMANTIC_TASK)


def test_shared_engine_fits_and_resumes_semantic_task(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = create_task_model(
        "fc_siam_diff", {"in_channels": 3, "out_channels": 14, "base_channels": 4}, SEMANTIC_TASK
    )
    train = DataLoader(_samples(), batch_size=2, generator=torch.Generator().manual_seed(3))
    validation = DataLoader(_samples(2), batch_size=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    kwargs = {
        "model": model, "train_loader": train, "validation_loader": validation,
        "loss_function": SemanticSCDLoss(), "optimizer": optimizer, "device": torch.device("cpu"),
        "thresholds": [], "output_root": tmp_path, "model_name": "fc_siam_diff",
        "run_name": "sem", "task": SEMANTIC_TASK,
    }
    history = fit_binary_change_model(number_of_epochs=2, **kwargs)  # type: ignore[arg-type]
    assert [row["epoch"] for row in history] == [1, 2]
    assert history[-1]["selection_score"] == history[-1]["sek"]
    checkpoint = torch.load(tmp_path / "fc_siam_diff/sem/last.pt", weights_only=False)
    assert checkpoint["task"] == SEMANTIC_TASK and checkpoint["thresholds"] == []
    assert checkpoint["best_metrics"]["selection_score"] >= history[0]["selection_score"] - 1e-12
    resumed = fit_binary_change_model(
        number_of_epochs=1, resume_from=tmp_path / "fc_siam_diff/sem/last.pt", **kwargs  # type: ignore[arg-type]
    )
    assert len(resumed) == 3
    with pytest.raises(ValueError, match="no thresholds"):
        fit_binary_change_model(number_of_epochs=1, **{**kwargs, "thresholds": [0.5]})  # type: ignore[arg-type]


def test_deterministic_cross_entropy_equals_torch_reference() -> None:
    logits = torch.randn(2, 14, 5, 6)
    label = torch.randint(0, 7, (2, 5, 6))
    weights = [0.5, 1.0, 2.0, 1.5, 3.0, 0.7, 4.0]
    loss = SemanticSCDLoss(weights)(logits, label, label.flip(-1))
    reference = torch.nn.functional.cross_entropy(
        logits[:, :7], label, weight=torch.tensor(weights)
    ) + torch.nn.functional.cross_entropy(
        logits[:, 7:], label.flip(-1), weight=torch.tensor(weights)
    )
    assert torch.allclose(loss, reference, atol=1e-6)
