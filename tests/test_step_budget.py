"""Optimizer-step budget in the shared training engine (R2-V3-T02)."""

import shutil
from pathlib import Path
from typing import Any

import pytest
import torch
from test_scd_training import _samples
from torch.utils.data import DataLoader

from satchange.losses import BCEDiceLoss
from satchange.models import FCSiamDiff
from satchange.models.factory import SEMANTIC_TASK, create_task_model
from satchange.semantic.scd_training import SemanticSCDLoss, run_epoch
from satchange.step_budget import StepBudget
from satchange.training import fit_binary_change_model, train_one_epoch


def test_step_budget_counts_and_validates() -> None:
    budget = StepBudget(2)
    assert budget.remaining == 2 and not budget.exhausted
    budget.take(4)
    budget.take(3)
    assert budget.exhausted and budget.remaining == 0 and budget.samples_seen == 7
    with pytest.raises(RuntimeError, match="exhausted"):
        budget.take(1)
    assert StepBudget.from_state(budget.state_dict()) == budget
    unlimited = StepBudget()
    unlimited.take(5)
    assert unlimited.remaining is None and not unlimited.exhausted
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="max_optimizer_steps"):
            StepBudget(bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="exceeds"):
        StepBudget(1, optimizer_steps=2)
    with pytest.raises(ValueError, match="Unknown"):
        StepBudget.from_state({"max_optimizer_steps": 1, "steps": 0})


def test_binary_and_semantic_epoch_runners_stop_inside_the_epoch() -> None:
    torch.manual_seed(0)
    binary = [
        {"before": torch.rand(3, 16, 16), "after": torch.rand(3, 16, 16),
         "mask": (torch.rand(1, 16, 16) > 0.5).float()}
        for _ in range(6)
    ]
    model = FCSiamDiff(base_channels=4)
    budget = StepBudget(2)
    train_one_epoch(model, DataLoader(binary, batch_size=2), BCEDiceLoss(),
                    torch.optim.SGD(model.parameters(), lr=0.1), torch.device("cpu"),
                    budget=budget)
    assert (budget.optimizer_steps, budget.samples_seen) == (2, 4)
    semantic = create_task_model(
        "fc_siam_diff", {"in_channels": 3, "out_channels": 14, "base_channels": 4}, SEMANTIC_TASK
    )
    budget = StepBudget(1)
    optimizer = torch.optim.SGD(semantic.parameters(), lr=0.1)
    result = run_epoch(semantic, DataLoader(_samples(), batch_size=2), SemanticSCDLoss(),
                       torch.device("cpu"), optimizer=optimizer, budget=budget)
    assert budget.optimizer_steps == 1 and result["samples"] == 2
    with pytest.raises(ValueError, match="only when fitting"):
        run_epoch(semantic, DataLoader(_samples(), batch_size=2), SemanticSCDLoss(),
                  torch.device("cpu"), budget=StepBudget(1))


def _kwargs(tmp_path: Path) -> dict[str, Any]:
    torch.manual_seed(0)
    model = create_task_model(
        "fc_siam_diff", {"in_channels": 3, "out_channels": 14, "base_channels": 4}, SEMANTIC_TASK
    )
    return {
        "model": model,
        "train_loader": DataLoader(_samples(), batch_size=2),  # 2 steps per epoch
        "validation_loader": DataLoader(_samples(2), batch_size=2),
        "loss_function": SemanticSCDLoss(),
        "optimizer": torch.optim.AdamW(model.parameters(), lr=1e-2),
        "device": torch.device("cpu"), "thresholds": [], "output_root": tmp_path,
        "model_name": "fc_siam_diff", "run_name": "budget", "task": SEMANTIC_TASK,
        "checkpoint_interval": 1,
    }


def test_engine_stops_at_the_step_budget_and_resumes_exactly(tmp_path: Path) -> None:
    history = fit_binary_change_model(
        number_of_epochs=5, max_optimizer_steps=3, flops_per_sample=10.0, **_kwargs(tmp_path)
    )
    assert [row["epoch"] for row in history] == [1, 2]
    assert [row["optimizer_steps"] for row in history] == [2, 3]
    last = history[-1]
    assert last["samples_seen"] == 6 and last["training_flops"] == 60.0
    assert last["stop_reason"] == "step_budget" and history[0]["stop_reason"] is None
    run = tmp_path / "fc_siam_diff/budget"
    final = torch.load(run / "last.pt", weights_only=False)
    assert final["step_budget"] == {"max_optimizer_steps": 3, "optimizer_steps": 3,
                                    "samples_seen": 6}

    # A completed budget run is not restarted.
    again = fit_binary_change_model(number_of_epochs=1, max_optimizer_steps=3,
                                    flops_per_sample=10.0, resume_from=run / "last.pt",
                                    **_kwargs(tmp_path))
    assert len(again) == 2

    # Resuming from epoch 1 reproduces the uninterrupted final weights.
    epoch_one = tmp_path / "epoch1.pt"
    shutil.copy(run / "checkpoints/fc_siam_diff_epoch_0001.pt", epoch_one)
    resumed = fit_binary_change_model(number_of_epochs=4, max_optimizer_steps=3,
                                      flops_per_sample=10.0, resume_from=epoch_one,
                                      **_kwargs(tmp_path))
    assert [row["optimizer_steps"] for row in resumed] == [2, 3]
    replayed = torch.load(run / "last.pt", weights_only=False)
    for name, tensor in final["model_state_dict"].items():
        assert torch.equal(tensor, replayed["model_state_dict"][name]), name

    with pytest.raises(ValueError, match="differs from the checkpoint"):
        fit_binary_change_model(number_of_epochs=4, max_optimizer_steps=4,
                                resume_from=epoch_one, **_kwargs(tmp_path))
    with pytest.raises(ValueError, match="step budget mismatch"):
        fit_binary_change_model(number_of_epochs=4, resume_from=epoch_one, **_kwargs(tmp_path))


def test_engine_rejects_inconsistent_budget_options(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="excludes early stopping"):
        fit_binary_change_model(number_of_epochs=5, max_optimizer_steps=3, patience=2,
                                **_kwargs(tmp_path))
    with pytest.raises(ValueError, match="cannot reach"):
        fit_binary_change_model(number_of_epochs=1, max_optimizer_steps=3, **_kwargs(tmp_path))
    with pytest.raises(ValueError, match="only together"):
        fit_binary_change_model(number_of_epochs=1, flops_per_sample=1.0, **_kwargs(tmp_path))
    with pytest.raises(ValueError, match="positive number"):
        fit_binary_change_model(number_of_epochs=1, max_optimizer_steps=2, flops_per_sample=0.0,
                                **_kwargs(tmp_path))
    # Without a budget the engine behaves as before (no budget keys in the records).
    history = fit_binary_change_model(number_of_epochs=1, **_kwargs(tmp_path))
    assert "optimizer_steps" not in history[0] and history[0]["stop_reason"] == "max_epochs"
