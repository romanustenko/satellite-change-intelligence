"""Shared runtime tests using synthetic data and two small Gloo processes."""

import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from PIL import Image
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from satchange.data import RandomFlipRotate, create_dataset_dataloader
from satchange.execution import on_rank_zero, precision_context, seed_everything
from satchange.losses import BCEDiceLoss
from satchange.models import create_change_model
from satchange.optimization import run_parameter_search
from satchange.run import probe_training_step, reconstruct_checkpoint, run_training
from satchange.training import evaluate_thresholds, fit_binary_change_model
from satchange.utils import load_config


class TinyPairModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Conv2d(3, 1, 1)

    def forward(self, before, after):
        return self.head((before - after).abs())


def _distributed_worker(rank, directory):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=(Path(directory) / "rendezvous").as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    try:
        seed_everything(12)
        samples = [
            {
                "before": torch.rand(3, 8, 8),
                "after": torch.rand(3, 8, 8),
                "mask": torch.randint(0, 2, (1, 8, 8)).float(),
            }
            for _ in range(5)
        ]
        sampler = DistributedSampler(samples, num_replicas=2, rank=rank, seed=12)
        loader = DataLoader(
            samples,
            batch_size=2,
            sampler=sampler,
            generator=torch.Generator().manual_seed(12 + rank),
        )
        validation = DataLoader(samples[:3], batch_size=2)  # Odd length reveals padding bugs.
        model = DistributedDataParallel(TinyPairModel())
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=0, threshold=1.0
        )
        loss = BCEDiceLoss()
        history = fit_binary_change_model(
            model,
            loader,
            validation,
            loss,
            optimizer,
            torch.device("cpu"),
            2,
            [0.4, 0.5],
            directory,
            "tiny",
            "ddp",
            scheduler=scheduler,
        )
        assert sampler.epoch == 2
        for row in history:
            assert (
                sum(
                    row[key]
                    for key in (
                        "true_positive",
                        "true_negative",
                        "false_positive",
                        "false_negative",
                    )
                )
                == 3 * 64
            )
        tensors = [torch.empty_like(model.module.head.weight) for _ in range(2)]
        dist.all_gather(tensors, model.module.head.weight)
        torch.testing.assert_close(tensors[0], tensors[1], rtol=0, atol=0)
        checkpoint = Path(directory) / "tiny/ddp/last.pt"
        state = torch.load(checkpoint, weights_only=False)
        assert not any(key.startswith("module.") for key in state["model_state_dict"])
        assert len(state["rng_states"]) == 2
        assert state["world_size"] == 2
        assert optimizer.param_groups[0]["lr"] == 0.001
        history = fit_binary_change_model(
            model,
            loader,
            validation,
            loss,
            optimizer,
            torch.device("cpu"),
            1,
            [0.4, 0.5],
            directory,
            "tiny",
            "ddp",
            scheduler=scheduler,
            resume_from=checkpoint,
        )
        assert len(history) == 3 and sampler.epoch == 3
        # Failure reaches both peers without an unmatched validation collective.
        with pytest.raises(RuntimeError, match="Rank-zero operation failed"):
            on_rank_zero(
                lambda: evaluate_thresholds(
                    model.module, DataLoader([]), loss, torch.device("cpu"), [0.5]
                )
            )
        with pytest.raises(ValueError, match="cannot run inside"):
            run_parameter_search(
                {"seed": [1]}, {}, "tiny", directory, [0.5], 1, directory, "forbidden", [0]
            )
        Path(directory, f"rank_{rank}.json").write_text(
            json.dumps(
                [
                    {
                        key: row[key]
                        for key in (
                            "epoch",
                            "validation_f1",
                            "best_threshold",
                            "next_learning_rate",
                        )
                    }
                    for row in history
                ]
            )
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="PyTorch Gloo is unavailable")
def test_two_process_fit_validation_resume_and_failure_coordination(tmp_path):
    mp.spawn(_distributed_worker, args=(str(tmp_path),), nprocs=2, join=True)
    assert (tmp_path / "rank_0.json").read_text() == (tmp_path / "rank_1.json").read_text()
    assert len(list((tmp_path / "tiny/ddp").glob("best.pt"))) == 1


def test_precision_and_probe_guards():
    with precision_context(torch.device("cpu"), "fp32"):
        pass
    for mode in ("bf16", "fp16", "typo"):
        with pytest.raises(ValueError):
            precision_context(torch.device("cpu"), mode)
    with pytest.raises(ValueError, match="enabled"):
        probe_training_step({}, torch.device("cpu"))
    with pytest.raises(ValueError, match="CUDA"):
        probe_training_step({}, torch.device("cpu"), enabled=True)


def test_preprocessing_native_whole_resize_and_synchronized_geometry(tmp_path):
    image = np.zeros((1024, 1024, 3), dtype=np.uint8)
    image[:256, :256] = 255
    image[768:, 768:] = 255
    for folder in ("A", "B", "label"):
        path = tmp_path / "train" / folder
        path.mkdir(parents=True)
        Image.fromarray(image if folder != "label" else image[:, :, 0]).save(path / "scene.png")
    for regime, expected in (("resized", 256), ("full_resolution", 1024)):
        config = load_config(f"configs/siam_mit_b3_diff_{regime}.yaml")
        dataset, _ = create_dataset_dataloader(
            tmp_path, "train", config["data"]["image_size"], 1, 0
        )
        sample = dataset[0]
        assert sample["before"].shape == (3, expected, expected)
        assert sample["mask"].shape == (1, expected, expected)
        assert sample["before"][0, 0, 0] == sample["before"][0, -1, -1] == 1
        assert sample["before"][0, expected // 2, expected // 2] == 0
        assert set(sample["mask"].unique().tolist()) == {0, 1}
        torch.manual_seed(9)
        transformed = RandomFlipRotate(1, 1)(sample)
        assert torch.equal(transformed["before"], transformed["after"])
        assert torch.equal(transformed["before"][:1] >= 0.5, transformed["mask"].bool())
        assert all(transformed[key].is_contiguous() for key in ("before", "after", "mask"))


def test_generic_yaml_training_launcher_and_collision_guard(tmp_path):
    for split in ("train", "val"):
        for folder in ("A", "B", "label"):
            path = tmp_path / "data" / split / folder
            path.mkdir(parents=True)
            pixels = np.zeros((32, 32, 3), dtype=np.uint8)
            pixels[8:16] = 255 if folder != "A" else 100
            Image.fromarray(pixels).save(path / "one.png")
    config = load_config("configs/siam_mit_b3_diff_resized.yaml")
    config["model"]["parameters"]["encoder_depths"] = [1, 1, 1, 1]  # Launcher depth ablation.
    config["data"]["image_size"] = 32
    config["data"]["num_workers"] = 0
    config["data"]["batch_size"] = 1
    config["training"]["epochs"] = 1
    config["training"]["thresholds"] = [0.5]  # Explicit validation grid for the fixture.
    config["runtime"]["precision"] = "fp32"  # This launcher fixture runs on CPU.
    config["dataset"]["prepared_directory"] = "data"
    config["output"]["directory"] = "output"
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        history = run_training(config, tmp_path, torch.device("cpu"), run_name="synthetic")
        assert len(history) == 1
        with pytest.raises(FileExistsError):
            run_training(config, tmp_path, torch.device("cpu"), run_name="synthetic")
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize(
    "name,parameters",
    [
        ("fc_siam_diff", {"base_channels": 2}),
        (
            "bit",
            {
                "embedding_dim": 8,
                "num_heads": 2,
                "encoder_head_dim": 4,
                "decoder_head_dim": 4,
                "decoder_depth": 1,
            },
        ),
    ],
)
def test_legacy_model_common_checkpoint_regression(tmp_path, name, parameters):
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        model = create_change_model(name, parameters)
        loader = DataLoader(
            [
                {
                    "before": torch.rand(3, 32, 32),
                    "after": torch.rand(3, 32, 32),
                    "mask": torch.randint(0, 2, (1, 32, 32)).float(),
                }
            ]
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        fit_binary_change_model(
            model,
            loader,
            loader,
            BCEDiceLoss(),
            optimizer,
            torch.device("cpu"),
            1,
            [0.5],
            tmp_path,
            name,
            "legacy",
            config={"model_parameters": parameters},
        )
        fresh, _ = reconstruct_checkpoint(tmp_path / name / "legacy/last.pt", torch.device("cpu"))
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, fresh.state_dict()[key], rtol=0, atol=0)
    finally:
        torch.set_num_threads(previous)
