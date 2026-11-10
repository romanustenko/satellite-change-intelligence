"""Multi-head SCD models (U-Net, SCanNet, MambaSCD) through the shared semantic engine (V4-T08)."""

from pathlib import Path
from typing import Any

import pytest
import torch
from torch.utils.data import DataLoader

from satchange.models.factory import (
    MULTI_HEAD_SEMANTIC_MODELS,
    SEMANTIC_MODELS,
    SEMANTIC_TASK,
    create_task_model,
)
from satchange.run import reconstruct_checkpoint
from satchange.semantic.scd_losses import (
    SCanNetPseudoLabeler,
    create_scd_loss,
    pseudo_label_start_step,
)
from satchange.semantic.scd_training import SemanticSCDLoss, run_epoch
from satchange.step_budget import StepBudget
from satchange.training import fit_binary_change_model

SIZE = 64
SMALL_PARAMETERS: dict[str, dict[str, Any]] = {
    "siam_unet_resnet18": {"head_layout": "multi_head"},
    "scannet": {},
    "mamba_scd": {
        "depths": [1, 1, 1, 1],
        "embed_dim": 16,
        "hidden_dim": 16,
        "scan_backend": "reference",
    },
}


def _samples(count: int = 4, seed: int = 0) -> list[dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    samples = []
    for _ in range(count):
        changed = torch.rand(SIZE, SIZE, generator=generator) < 0.4
        samples.append(
            {
                "before": torch.rand(3, SIZE, SIZE, generator=generator),
                "after": torch.rand(3, SIZE, SIZE, generator=generator),
                "before_label": torch.randint(1, 7, (SIZE, SIZE), generator=generator) * changed,
                "after_label": torch.randint(1, 7, (SIZE, SIZE), generator=generator) * changed,
            }
        )
    return samples


def test_factory_builds_multi_head_models_without_out_channels() -> None:
    assert SEMANTIC_MODELS == ("fc_siam_diff", "fc_siam_semantic")  # joint models unchanged
    assert set(MULTI_HEAD_SEMANTIC_MODELS) == set(SMALL_PARAMETERS)
    for name, parameters in SMALL_PARAMETERS.items():
        model = create_task_model(name, parameters, SEMANTIC_TASK)
        assert isinstance(model, torch.nn.Module)
    with pytest.raises(ValueError, match="out_channels=14"):
        create_task_model("fc_siam_diff", {"in_channels": 3}, SEMANTIC_TASK)


@pytest.mark.parametrize("name", sorted(SMALL_PARAMETERS))
def test_run_epoch_trains_each_model_with_its_loss(name: str) -> None:
    torch.manual_seed(0)
    model = create_task_model(name, SMALL_PARAMETERS[name], SEMANTIC_TASK)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    budget = StepBudget(1)
    loader = DataLoader(_samples(4), batch_size=2)
    result = run_epoch(
        model,
        loader,
        create_scd_loss(name),
        torch.device("cpu"),
        optimizer=optimizer,
        budget=budget,
        semantic_argmax="land_cover",
    )
    assert budget.optimizer_steps == 1 and result["samples"] == 2
    assert torch.isfinite(torch.tensor(result["loss"]))
    evaluation = run_epoch(
        model, loader, create_scd_loss(name), torch.device("cpu"), semantic_argmax="land_cover"
    )
    assert evaluation["samples"] == 4 and 0.0 <= evaluation["miou"] <= 1.0
    assert evaluation["selection_score"] == evaluation["sek"]


def test_run_epoch_contract_checks() -> None:
    model = create_task_model(
        "siam_unet_resnet18", SMALL_PARAMETERS["siam_unet_resnet18"], SEMANTIC_TASK
    )
    loader = DataLoader(_samples(2), batch_size=2)
    cpu = torch.device("cpu")
    with pytest.raises(ValueError, match="explicit semantic_argmax"):
        run_epoch(model, loader, create_scd_loss("siam_unet_resnet18"), cpu)
    with pytest.raises(ValueError, match="only to multi-head"):
        run_epoch(model, loader, SemanticSCDLoss(), cpu, semantic_argmax="land_cover")
    with pytest.raises(ValueError, match="SCanNet loss only"):
        run_epoch(
            model,
            loader,
            create_scd_loss("siam_unet_resnet18"),
            cpu,
            semantic_argmax="land_cover",
            pseudo_labeler=SCanNetPseudoLabeler(start_step=1),
        )


def test_scannet_pseudo_labels_start_at_the_fixed_step() -> None:
    torch.manual_seed(0)
    model = create_task_model("scannet", {}, SEMANTIC_TASK)
    loader = DataLoader(_samples(2), batch_size=2)  # one optimizer step per epoch
    loss = create_scd_loss("scannet")
    seen: list[dict[str, Any]] = []
    original = loss.forward

    def recording(*args: Any, **kwargs: Any) -> torch.Tensor:
        seen.append(kwargs)
        return original(*args, **kwargs)  # type: ignore[no-any-return]

    loss.forward = recording  # type: ignore[method-assign]
    labeler = SCanNetPseudoLabeler(start_step=2, tta=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    budget = StepBudget(10)
    cpu = torch.device("cpu")
    for _ in range(3):
        run_epoch(
            model,
            loader,
            loss,
            cpu,
            optimizer=optimizer,
            budget=budget,
            semantic_argmax="land_cover",
            pseudo_labeler=labeler,
        )
    assert [bool(call) for call in seen] == [False, False, True]  # steps 0, 1 plain; step 2 on
    assert set(seen[2]) == {"pseudo_before", "pseudo_after"}
    assert seen[2]["pseudo_before"].dtype == torch.int64
    with pytest.raises(ValueError, match="step budget is required"):
        run_epoch(
            model,
            loader,
            loss,
            cpu,
            optimizer=optimizer,
            semantic_argmax="land_cover",
            pseudo_labeler=labeler,
        )


def test_pseudo_labeler_teacher_refresh_schedule() -> None:
    model = create_task_model(
        "siam_unet_resnet18", SMALL_PARAMETERS["siam_unet_resnet18"], SEMANTIC_TASK
    )
    labeler = SCanNetPseudoLabeler(start_step=5)
    labeler.before_step(model, 4)
    assert not labeler.active
    labeler.after_validation(model, 0.3, 4)  # improvement before the start: tracked, no teacher
    assert not labeler.active and labeler.best_fscd == 0.3
    labeler.before_step(model, 5)
    first_teacher = labeler.teacher
    assert labeler.active
    labeler.before_step(model, 6)
    assert labeler.teacher is first_teacher  # started once
    labeler.after_validation(model, 0.2, 6)  # no improvement: same teacher
    assert labeler.teacher is first_teacher
    labeler.after_validation(model, 0.4, 7)  # improvement after the start: new teacher
    assert labeler.teacher is not first_teacher
    assert pseudo_label_start_step(10) == 5 and pseudo_label_start_step(7) == 4
    assert pseudo_label_start_step(1) == 1
    for bad in (0, True, 2.5):
        with pytest.raises(ValueError):
            pseudo_label_start_step(bad)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            SCanNetPseudoLabeler(start_step=bad)  # type: ignore[arg-type]


def test_engine_runs_scannet_with_pseudo_labels_and_rejects_resume(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = create_task_model("scannet", {}, SEMANTIC_TASK)
    labeler = SCanNetPseudoLabeler(start_step=pseudo_label_start_step(4), tta=False)
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
    }
    with pytest.raises(ValueError, match="set max_optimizer_steps"):
        fit_binary_change_model(**kwargs)
    history = fit_binary_change_model(max_optimizer_steps=4, **kwargs)
    assert history[-1]["optimizer_steps"] == 4
    assert labeler.active and labeler.best_fscd > float("-inf")
    with pytest.raises(ValueError, match="Resuming with pseudo labels"):
        fit_binary_change_model(
            max_optimizer_steps=4, resume_from=tmp_path / "scannet/psd/last.pt", **kwargs
        )


def test_shared_engine_fits_and_reconstructs_multi_head_model(tmp_path: Path) -> None:
    torch.manual_seed(0)
    parameters = SMALL_PARAMETERS["siam_unet_resnet18"]
    model = create_task_model("siam_unet_resnet18", parameters, SEMANTIC_TASK)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    history = fit_binary_change_model(
        model=model,
        train_loader=DataLoader(
            _samples(4), batch_size=2, generator=torch.Generator().manual_seed(1)
        ),
        validation_loader=DataLoader(_samples(2, seed=5), batch_size=2),
        loss_function=create_scd_loss("siam_unet_resnet18"),
        optimizer=optimizer,
        device=torch.device("cpu"),
        number_of_epochs=3,
        thresholds=[],
        output_root=tmp_path,
        model_name="siam_unet_resnet18",
        run_name="mh",
        config={"model_parameters": parameters},
        task=SEMANTIC_TASK,
        max_optimizer_steps=3,
        semantic_argmax="land_cover",
    )
    assert history[-1]["optimizer_steps"] == 3 and history[-1]["stop_reason"] == "step_budget"
    rebuilt, state = reconstruct_checkpoint(
        tmp_path / "siam_unet_resnet18/mh/best.pt", torch.device("cpu")
    )
    assert state["task"] == SEMANTIC_TASK
    assert type(rebuilt).__name__ == "SiamUNetResNet18"


def test_engine_rejects_semantic_options_for_binary_task(tmp_path: Path) -> None:
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
            semantic_argmax="land_cover",
        )
