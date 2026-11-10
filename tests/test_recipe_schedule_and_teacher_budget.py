"""LR schedule (D-057) and teacher-FLOP charging for SCanNet's pseudo labels (D-056)."""

import math
from pathlib import Path
from typing import Any

import pytest
import torch
from torch.utils.data import DataLoader

from satchange.lr_schedule import warmup_cosine_factor
from satchange.models.factory import SEMANTIC_TASK, create_task_model
from satchange.semantic.scd_losses import (
    SCanNetPseudoLabeler,
    create_scd_loss,
    pseudo_label_budget_steps,
    pseudo_label_start_step,
)
from satchange.semantic.scd_training import run_epoch
from satchange.step_budget import StepBudget
from satchange.training import fit_binary_change_model

SIZE = 64


def _samples(count: int = 4, seed: int = 0) -> list[dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(count):
        changed = torch.rand(SIZE, SIZE, generator=generator) < 0.4
        out.append(
            {
                "before": torch.rand(3, SIZE, SIZE, generator=generator),
                "after": torch.rand(3, SIZE, SIZE, generator=generator),
                "before_label": torch.randint(1, 7, (SIZE, SIZE), generator=generator) * changed,
                "after_label": torch.randint(1, 7, (SIZE, SIZE), generator=generator) * changed,
            }
        )
    return out


def test_warmup_cosine_shape() -> None:
    total = 100
    factors = [warmup_cosine_factor(step, total_steps=total) for step in range(total)]
    assert factors[0] == pytest.approx(1 / 5)  # 5 warm-up steps: 0.2 ... 1.0
    assert factors[4] == pytest.approx(1.0) and factors[5] == pytest.approx(1.0)
    assert all(b >= a for a, b in zip(factors[:4], factors[1:5], strict=True))
    assert all(b <= a + 1e-12 for a, b in zip(factors[5:-1], factors[6:], strict=True))
    assert factors[-1] < 1e-3 and factors[-1] > 0
    midpoint = 5 + (total - 5) // 2
    assert factors[midpoint] == pytest.approx(0.5, abs=0.02)
    assert warmup_cosine_factor(0, total_steps=1) == 1.0  # one step: single warm-up step
    assert warmup_cosine_factor(3, total_steps=10, warmup_fraction=0.0) == pytest.approx(
        0.5 * (1 + math.cos(math.pi * 3 / 10))
    )
    for bad in (
        {"step": -1, "total_steps": 5},
        {"step": 5, "total_steps": 5},
        {"step": 0, "total_steps": 0},
    ):
        with pytest.raises(ValueError):
            warmup_cosine_factor(**bad)
    with pytest.raises(ValueError, match="warmup_fraction"):
        warmup_cosine_factor(0, total_steps=5, warmup_fraction=1.0)


def test_run_epoch_applies_the_step_lr_and_is_resume_safe() -> None:
    torch.manual_seed(0)
    model = create_task_model("siam_unet_resnet18", {"head_layout": "multi_head"}, SEMANTIC_TASK)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    seen: list[float] = []

    def factor(step: int) -> float:
        seen.append(step)
        return warmup_cosine_factor(step, total_steps=4)

    budget = StepBudget(4)
    loader = DataLoader(_samples(4), batch_size=2)
    run_epoch(
        model,
        loader,
        create_scd_loss("siam_unet_resnet18"),
        torch.device("cpu"),
        optimizer=optimizer,
        budget=budget,
        semantic_argmax="land_cover",
        lr_factor=factor,
    )
    assert seen == [0, 1]  # 4 samples, batch 2: two steps, step index passed to the schedule
    assert optimizer.param_groups[0]["base_lr"] == 1e-3  # base never compounds
    assert optimizer.param_groups[0]["lr"] == pytest.approx(
        1e-3 * warmup_cosine_factor(1, total_steps=4)
    )
    with pytest.raises(ValueError, match="lr_factor needs"):
        run_epoch(
            model,
            loader,
            create_scd_loss("siam_unet_resnet18"),
            torch.device("cpu"),
            semantic_argmax="land_cover",
            lr_factor=factor,
        )


def test_pseudo_label_budget_steps_charges_the_teacher() -> None:
    step, teacher = 13.2, 17.2
    budget = 1000.0
    steps = pseudo_label_budget_steps(budget, step_flops=step, teacher_step_flops=teacher)

    def cost(n: int) -> float:
        return n * step + (n - pseudo_label_start_step(n)) * teacher

    assert cost(steps) >= budget > cost(steps - 1)
    plain = math.ceil(budget / step)
    assert steps < plain  # charging the teacher means fewer steps than without it
    assert pseudo_label_budget_steps(budget, step_flops=step, teacher_step_flops=0.0) == plain
    assert pseudo_label_budget_steps(1.0, step_flops=5.0, teacher_step_flops=3.0) == 1
    with pytest.raises(ValueError):
        pseudo_label_budget_steps(0.0, step_flops=1.0, teacher_step_flops=1.0)


def test_teacher_flops_are_counted_and_enter_training_flops(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = create_task_model("scannet", {}, SEMANTIC_TASK)
    labeler = SCanNetPseudoLabeler(start_step=pseudo_label_start_step(4), tta=False)
    with pytest.raises(RuntimeError, match="flops_per_sample was not set"):
        _ = labeler.teacher_flops
    probe = _samples(2)
    before = torch.stack([s["before"] for s in probe])
    after = torch.stack([s["after"] for s in probe])
    per_sample = labeler.count_flops(model, before, after)
    assert per_sample > 0 and not labeler.active and labeler.teacher_samples == 0
    tta_labeler = SCanNetPseudoLabeler(start_step=2, tta=True)
    assert tta_labeler.count_flops(model, before, after) == pytest.approx(4 * per_sample, rel=0.02)

    kwargs: dict[str, Any] = {
        "model": model,
        "train_loader": DataLoader(
            _samples(4), batch_size=2, generator=torch.Generator().manual_seed(1)
        ),
        "validation_loader": DataLoader(_samples(2, seed=5), batch_size=2),
        "loss_function": create_scd_loss("scannet"),
        "optimizer": torch.optim.AdamW(model.parameters(), lr=1e-4),
        "device": torch.device("cpu"),
        "number_of_epochs": 4,
        "thresholds": [],
        "output_root": tmp_path,
        "model_name": "scannet",
        "run_name": "psd",
        "config": {"model_parameters": {}},
        "task": SEMANTIC_TASK,
        "semantic_argmax": "land_cover",
        "pseudo_labeler": labeler,
        "max_optimizer_steps": 4,
        "flops_per_sample": 1000.0,
        "lr_factor": lambda step: warmup_cosine_factor(step, total_steps=4),
    }
    history = fit_binary_change_model(**kwargs)
    last = history[-1]
    assert last["optimizer_steps"] == 4 and last["samples_seen"] == 8
    assert labeler.teacher_samples == 4  # steps 2 and 3 (start step 2), batch 2
    assert last["teacher_flops"] == pytest.approx(per_sample * 4)
    assert last["training_flops"] == pytest.approx(1000.0 * 8 + per_sample * 4)


def test_engine_rejects_lr_factor_for_binary_task(tmp_path: Path) -> None:
    from satchange.models.fc_siam_diff import FCSiamDiff

    model = FCSiamDiff(in_channels=3, out_channels=1, base_channels=4)
    with pytest.raises(ValueError, match="only to the semantic task"):
        fit_binary_change_model(
            model=model,
            train_loader=[],
            validation_loader=[],  # type: ignore[arg-type]
            loss_function=torch.nn.BCEWithLogitsLoss(),
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            device=torch.device("cpu"),
            number_of_epochs=1,
            thresholds=[0.5],
            output_root=tmp_path,
            model_name="x",
            run_name="y",
            lr_factor=lambda step: 1.0,
        )
