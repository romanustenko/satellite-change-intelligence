"""Tests for evaluation metrics."""

import pytest
import torch
from torch import nn

from satchange.metrics import BinaryChangeMetrics, binary_metrics_from_counts
from satchange.training import evaluate_thresholds


def test_binary_metrics_from_known_confusion_counts() -> None:
    metrics = binary_metrics_from_counts(
        true_positive=1,
        false_positive=1,
        false_negative=1,
        true_negative=1,
    )

    assert metrics["true_positive"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["true_negative"] == 1
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)
    assert metrics["iou"] == pytest.approx(1 / 3)
    assert metrics["accuracy"] == pytest.approx(0.5)


def test_binary_metrics_zero_denominators_are_zero() -> None:
    metrics = binary_metrics_from_counts(
        true_positive=0,
        false_positive=0,
        false_negative=0,
        true_negative=4,
    )

    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
    assert metrics["iou"] == 0.0
    assert metrics["accuracy"] == 1.0


def test_binary_metrics_aggregate_globally_across_updates() -> None:
    probabilities = torch.tensor([0.9, 0.2, 0.8, 0.1])
    targets = torch.tensor([1.0, 1.0, 0.0, 0.0])

    one_update = BinaryChangeMetrics(threshold=0.5)
    one_update.update_probabilities(probabilities, targets)

    two_updates = BinaryChangeMetrics(threshold=0.5)
    two_updates.update_probabilities(probabilities[:2], targets[:2])
    two_updates.update_probabilities(probabilities[2:], targets[2:])

    assert two_updates.compute() == one_update.compute()
    assert one_update.compute()["f1"] == pytest.approx(0.5)


def test_binary_metrics_update_interprets_inputs_as_logits() -> None:
    metrics = BinaryChangeMetrics(threshold=0.7)
    metrics.update(torch.tensor([0.8]), torch.tensor([1.0]))

    assert metrics.compute()["false_negative"] == 1


class _BeforeLogitModel(nn.Module):
    def forward(
        self,
        before: torch.Tensor,
        after: torch.Tensor,
    ) -> torch.Tensor:
        del after
        return before


def test_threshold_evaluation_uses_global_canonical_metrics() -> None:
    probabilities = torch.tensor([0.9, 0.6, 0.4, 0.1])
    logits = torch.logit(probabilities).reshape(2, 1, 1, 2)
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0]).reshape(2, 1, 1, 2)
    batches = [
        {
            "before": logits[index : index + 1],
            "after": torch.zeros_like(logits[index : index + 1]),
            "mask": targets[index : index + 1],
        }
        for index in range(2)
    ]

    evaluation = evaluate_thresholds(
        model=_BeforeLogitModel(),
        data_loader=batches,  # type: ignore[arg-type]
        loss_function=nn.BCEWithLogitsLoss(),
        device=torch.device("cpu"),
        thresholds=[0.3, 0.7],
    )

    low_threshold, high_threshold = evaluation["threshold_results"]
    assert low_threshold == {
        "threshold": 0.3,
        "precision": pytest.approx(2 / 3),
        "recall": pytest.approx(1.0),
        "f1": pytest.approx(0.8),
        "iou": pytest.approx(2 / 3),
        "accuracy": pytest.approx(0.75),
        "tp": 2,
        "fp": 1,
        "fn": 0,
        "tn": 1,
        "predicted_changed_pixels": 3,
    }
    assert high_threshold["f1"] == pytest.approx(2 / 3)
    best_result = max(evaluation["threshold_results"], key=lambda result: result["f1"])
    assert best_result["threshold"] == pytest.approx(0.3)


def test_threshold_selection_keeps_first_threshold_on_an_f1_tie() -> None:
    probabilities = torch.tensor([0.1, 0.1]).reshape(1, 1, 1, 2)
    targets = torch.zeros_like(probabilities)
    batches = [
        {
            "before": torch.logit(probabilities),
            "after": torch.zeros_like(probabilities),
            "mask": targets,
        }
    ]

    evaluation = evaluate_thresholds(
        model=_BeforeLogitModel(),
        data_loader=batches,  # type: ignore[arg-type]
        loss_function=nn.BCEWithLogitsLoss(),
        device=torch.device("cpu"),
        thresholds=[0.3, 0.7],
    )
    threshold_results = evaluation["threshold_results"]
    best_result = max(threshold_results, key=lambda result: result["f1"])

    assert [result["threshold"] for result in threshold_results] == [0.3, 0.7]
    assert [result["f1"] for result in threshold_results] == [0.0, 0.0]
    assert best_result["threshold"] == pytest.approx(0.3)
