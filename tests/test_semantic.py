"""CPU-only synthetic implementation checks, not evidence of real semantic accuracy."""

import json
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from satchange.execution import seed_everything
from satchange.models import TinySiameseChangeClassifier, count_model_parameters
from satchange.models.factory import MODEL_REGISTRY
from satchange.models.tiny_siamese_change_classifier import masked_pool
from satchange.semantic.__main__ import main, synthetic_smoke
from satchange.semantic.data import (
    SemanticROIDataset,
    collate_rois,
    data_protocol,
    dominant_transition,
    prepare_dataset,
    read_manifest,
    read_map,
)
from satchange.semantic.roi import ROISettings, align_whole_image_probabilities, extract_rois
from satchange.semantic.training import (
    SemanticMetrics,
    create_loaders,
    hpo_parameters,
    logical_forward,
    run_epoch,
    semantic_loss,
    validate_parameters,
)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config():
    return {
        "classes": ["land", "built"],
        "class_ids": {1: "land", 2: "built"},
        "ignore_ids": [0, 255],
        "mask_source": "ground_truth",
        "detector": None,
        "roi": {"context_pixels": 1, "min_crop_size": 8},
        "label_policy": {"min_transition_purity": 0.8, "min_valid_fraction": 0.8},
        "dropout": 0.1,
        "training": {"epochs": 2, "batch_size": 4, "seed": 42, "num_workers": 0},
    }


def scene_tensors(height=171, width=197):
    before = torch.arange(height * width).reshape(1, height, width).float()
    before = (before / (height * width)).expand(3, -1, -1).contiguous()
    return before, 1 - before, torch.zeros(1, height, width)


def make_manifest(tmp_path):
    scenes = []
    for index, split in enumerate(("train", "train", "train", "val", "test")):
        height, width = 17 + index, 23 + index
        before = np.full((height, width, 3), 20 + index, dtype=np.uint8)
        after = np.full((height, width, 3), 200 - index, dtype=np.uint8)
        change = np.zeros((height, width), dtype=np.float32)
        change[0 : 3 + index, 1 : 4 + index] = 1
        semantic_before = np.ones((height, width), dtype=np.int16)
        semantic_after = np.full((height, width), 2, dtype=np.int16)
        prefix = f"scene_{index}"
        row = {"sample_id": prefix, "group_id": prefix, "split": split}
        for key, array in (
            ("before", before),
            ("after", after),
            ("before_semantic", semantic_before),
            ("after_semantic", semantic_after),
            ("change", change),
        ):
            name = f"{prefix}_{key}.{'png' if key in ('before', 'after') else 'npy'}"
            if key in ("before", "after"):
                Image.fromarray(array).save(tmp_path / name)
            else:
                np.save(tmp_path / name, array)
            row[key] = name
        scenes.append(row)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"scenes": scenes}), encoding="utf-8")
    return manifest, scenes


@pytest.mark.parametrize("size", [(128, 128), (17, 23), (9, 31), (4, 4)])
def test_shared_forward_shape_grid_and_swap(size):
    seed_everything(42)
    model = TinySiameseChangeClassifier(["a", "b", "c"]).eval()
    assert "tiny_siamese_change_classifier" not in MODEL_REGISTRY
    assert count_model_parameters(model)["total_parameters"] < 110_000
    assert sum(isinstance(layer, nn.Conv2d) for layer in model.modules()) == 3
    assert sum(isinstance(layer, nn.MaxPool2d) for layer in model.modules()) == 2
    calls = []
    hook = model.backbone.register_forward_hook(
        lambda module, args, output: calls.append(output.shape)
    )
    before, after = torch.rand(2, 3, *size), torch.rand(2, 3, *size)
    mask = torch.zeros(2, 1, *size)
    mask[:, :, -1, -1] = 1  # Tiny support survives actual odd-size area reduction.
    normal = model(before, after, mask)
    assert calls == [torch.Size([4, 128, size[0] // 4, size[1] // 4])]
    swapped = model(after, before, mask)
    hook.remove()
    assert normal["before_logits"].shape == (2, 3)
    torch.testing.assert_close(normal["before_logits"], swapped["after_logits"])
    torch.testing.assert_close(normal["after_logits"], swapped["before_logits"])


def test_both_losses_reach_one_backbone_and_head():
    model = TinySiameseChangeClassifier(["a", "b"], dropout=0)
    before, after = torch.rand(1, 3, 17, 19), torch.rand(1, 3, 17, 19)
    mask = torch.ones(1, 1, 17, 19)
    for key in ("before_logits", "after_logits"):
        model.zero_grad(set_to_none=True)
        output = model(before, after, mask)
        F.cross_entropy(output[key], torch.tensor([1])).backward()
        for parameters in (model.backbone.parameters(), model.head.parameters()):
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
        assert model.backbone[0].weight.grad.abs().sum() > 0
    assert int(model.backbone[1].num_batches_tracked) == 2  # One update per temporal pair.


def test_pool_hand_calculation_fp32_and_tiny_support():
    features = torch.tensor([[[[1, 2], [3, 4]]]], dtype=torch.bfloat16)
    mask = torch.tensor([[[[1.0, 0], [0.5, 0]]]])
    result = masked_pool(features, mask)
    assert result.dtype == torch.float32
    assert float(result) == pytest.approx(2.5 / 1.5)
    tiny = torch.zeros(1, 1, 17, 23)
    tiny[0, 0, -1, -1] = 0.7
    assert float(masked_pool(torch.ones(1, 2, 4, 5), tiny)[0, 0]) == pytest.approx(1, abs=1e-5)
    with pytest.raises(ValueError, match="nonempty"):
        masked_pool(features, torch.zeros_like(mask))


@pytest.mark.parametrize(
    "failure", ["empty", "nan", "negative", "above_one", "shape", "small", "dtype"]
)
def test_model_input_validation(failure):
    model = TinySiameseChangeClassifier(["a", "b"])
    before, after, mask = torch.rand(1, 3, 9, 11), torch.rand(1, 3, 9, 11), torch.ones(1, 1, 9, 11)
    if failure == "empty":
        mask.zero_()
    elif failure in ("nan", "negative", "above_one"):
        mask[0, 0, 0, 0] = {"nan": float("nan"), "negative": -1, "above_one": 2}[failure]
    elif failure == "shape":
        after = after[..., :-1]
    elif failure == "small":
        before, after, mask = before[:, :, :3], after[:, :, :3], mask[:, :, :3]
    else:
        before = before.double()
    with pytest.raises(ValueError):
        model(before, after, mask)


@pytest.mark.parametrize(
    "classes,dropout",
    [([], 0.1), (["a"], 0.1), (["a", "a"], 0.1), (["a", "b"], 1), (["a", "b"], float("nan"))],
)
def test_model_configuration_validation(classes, dropout):
    with pytest.raises(ValueError):
        TinySiameseChangeClassifier(classes, dropout)


@pytest.mark.parametrize("y,x", [(0, 0), (0, 196), (170, 0), (170, 196), (90, 100)])
def test_real_context_near_all_borders(y, x):
    before, after, probability = scene_tensors()
    probability[0, y, x] = 0.9
    (roi,) = extract_rois(before, after, probability)
    assert roi.before.shape == (3, 128, 128)
    assert not roi.minimum_extent_limited
    x0, y0, x1, y1 = roi.crop_bbox
    torch.testing.assert_close(roi.before, before[:, y0:y1, x0:x1])
    torch.testing.assert_close(roi.after, after[:, y0:y1, x0:x1])
    assert roi.component_bbox == (x, y, x + 1, y + 1)
    assert roi.mask.sum() == pytest.approx(0.9)
    assert x0 <= x < x1 and y0 <= y < y1


def test_scale_large_components_small_images_neighbors_and_order():
    before, after, probability = scene_tensors(180, 240)
    probability[0, 2:32, 4:44] = 0.8
    probability[0, 40:42, 50:52] = 0.7  # Included in context but excluded from first pooling mask.
    rois = extract_rois(before, after, probability)
    assert [roi.area for roi in rois] == [30 * 40, 4]
    assert rois[0].probabilities.sum() > rois[0].mask.sum()
    assert int(rois[0].support.sum()) == 30 * 40
    assert [roi.crop_bbox for roi in rois] == [
        roi.crop_bbox for roi in extract_rois(before, after, probability)
    ]
    large = torch.ones_like(probability)
    (roi,) = extract_rois(before, after, large)
    assert roi.before.shape == before.shape and roi.area == 180 * 240
    before, after, probability = scene_tensors(37, 51)
    probability[:, 1, 1] = 1
    (roi,) = extract_rois(before, after, probability)
    assert roi.minimum_extent_limited and roi.before.shape == (3, 37, 51)
    assert extract_rois(before, after, torch.zeros_like(probability)) == []


def test_connectivity_min_area_and_probability_alignment():
    before, after, probability = scene_tensors(10, 12)
    probability[0, 2, 2] = probability[0, 3, 3] = 1
    settings = ROISettings(min_crop_size=4, context_pixels=0)
    assert len(extract_rois(before, after, probability, settings)) == 1
    assert len(extract_rois(before, after, probability, replace(settings, connectivity=4))) == 2
    assert extract_rois(before, after, probability, replace(settings, min_component_area=3)) == []
    resized = align_whole_image_probabilities(torch.ones(1, 4, 5), (10, 12), (4, 5))
    assert resized.shape == (1, 10, 12)
    torch.testing.assert_close(resized, torch.ones_like(resized))
    with pytest.raises(ValueError):
        align_whole_image_probabilities(probability, (10, 12), (4, 5))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.1])
def test_roi_rejects_invalid_probabilities(bad):
    before, after, probabilities = scene_tensors(8, 8)
    probabilities[0, 1, 1] = bad
    with pytest.raises(ValueError, match="finite"):
        extract_rois(before, after, probabilities)


def test_roi_rejects_spatial_mismatch_and_small_input():
    before, after, probabilities = scene_tensors(8, 8)
    with pytest.raises(ValueError, match="align"):
        extract_rois(before, after, probabilities[:, :-1])
    with pytest.raises(ValueError, match="too small"):
        extract_rois(before[:, :3], after[:, :3], probabilities[:, :3])


def test_component_ids_stay_stable_after_area_filtering():
    before, after, probability = scene_tensors(20, 20)
    probability[0, 1, 1] = 1
    probability[0, 10:12, 10:12] = 1
    settings = ROISettings(min_crop_size=4, context_pixels=0)
    assert [roi.component_id for roi in extract_rois(before, after, probability, settings)] == [
        1,
        2,
    ]
    filtered = extract_rois(before, after, probability, replace(settings, min_component_area=2))
    assert [roi.component_id for roi in filtered] == [2]


def test_joint_label_selection_does_not_invent_independent_majority_transition():
    cfg = config()
    cfg["classes"].append("third")
    cfg["class_ids"][3] = "third"
    cfg["label_policy"]["min_transition_purity"] = 0.2
    # Marginal majorities are (1,1), which is absent. Actual dominant pair is (1,2).
    pairs = [(1, 2)] * 5 + [(1, 3)] * 4 + [(2, 1)] * 4 + [(3, 1)] * 4
    before, after = torch.tensor(pairs).T
    quality = dominant_transition(
        before.unsqueeze(0),
        after.unsqueeze(0),
        torch.ones(1, len(pairs), dtype=torch.bool),
        data_protocol(cfg),
    )
    assert before.mode().values == after.mode().values == 1
    assert (quality["before_label"], quality["after_label"]) == (0, 1)


def test_observed_joint_transition_purity_ignore_ties_and_same_class():
    cfg = config()
    cfg["label_policy"]["min_transition_purity"] = 0.4
    protocol = data_protocol(cfg)
    before = torch.tensor([[1, 1, 1, 2, 2, 0]])
    after = torch.tensor([[2, 2, 1, 1, 1, 0]])
    support = torch.ones_like(before, dtype=torch.bool)
    tied = dominant_transition(before, after, support, protocol)
    assert tied["reason"] == "tied_transition"
    before[0, 4] = 1
    quality = dominant_transition(before, after, support, protocol)
    # Two independent majorities would select (1,1); that is tied with (1,2), so still rejected.
    assert quality["reason"] == "tied_transition"
    after[0, 4] = 2
    quality = dominant_transition(before, after, support, protocol)
    assert (quality["before_label"], quality["after_label"]) == (0, 1)
    assert quality["transition_purity"] == 3 / 5
    strict = data_protocol(config())
    assert dominant_transition(before, after, support, strict)["reason"] == "mixed_transition"
    before[0, :3] = 0
    assert (
        dominant_transition(before, after, support, strict)["reason"] == "insufficient_annotation"
    )
    same = dominant_transition(
        torch.ones(2, 2, dtype=torch.long),
        torch.ones(2, 2, dtype=torch.long),
        torch.ones(2, 2, dtype=torch.bool),
        strict,
    )
    assert same["before_label"] == same["after_label"] == 0
    with pytest.raises(ValueError, match="undeclared"):
        dominant_transition(
            torch.full((2, 2), 99),
            torch.ones(2, 2, dtype=torch.long),
            torch.ones(2, 2, dtype=torch.bool),
            strict,
        )


def test_manifest_preparation_native_crops_and_synchronized_augmentation(tmp_path):
    manifest, scenes = make_manifest(tmp_path)
    cfg = config()
    document = prepare_dataset(cfg, manifest, tmp_path / "prepared")
    assert document["coverage"]["train"]["retained_fraction"] == 1
    assert document["coverage"]["train"]["transition_distribution"] == {"land -> built": 3}
    assert len(list((tmp_path / "prepared").glob("masks/*.npz"))) == 5
    assert not list((tmp_path / "prepared").glob("*.png"))
    dataset = SemanticROIDataset(tmp_path / "prepared/prepared.json", "train", cfg, augment=True)
    dataset.verify_sources()
    torch.manual_seed(3)
    sample = dataset[0]
    assert sample["before"].shape[-2:] == sample["mask"].shape[-2:]
    assert all(sample[key].is_contiguous() for key in ("before", "after", "mask"))
    assert sample["before_label"] == 0 and sample["after_label"] == 1
    # Exercise geometry with a spatial marker rather than constant fixture RGB.
    from satchange.data.transforms import RandomFlipRotate

    marker = torch.arange(8 * 11).reshape(1, 8, 11).float()
    transformed = RandomFlipRotate(1, 1)(
        {
            "before": marker.expand(3, -1, -1),
            "after": marker.expand(3, -1, -1) + 1,
            "mask": marker,
            "sample_id": "marker",
        }
    )
    torch.testing.assert_close(transformed["before"][:1], transformed["mask"])
    torch.testing.assert_close(transformed["before"] + 1, transformed["after"])
    assert all(transformed[key].is_contiguous() for key in ("before", "after", "mask"))
    with pytest.raises(FileExistsError):
        prepare_dataset(cfg, manifest, tmp_path / "prepared")
    with pytest.raises(ValueError, match="augmentation"):
        SemanticROIDataset(tmp_path / "prepared/prepared.json", "val", cfg, augment=True)
    changed = deepcopy(cfg)
    changed["classes"].reverse()
    with pytest.raises(ValueError, match="protocol"):
        SemanticROIDataset(tmp_path / "prepared/prepared.json", "train", changed)
    np.save(tmp_path / scenes[0]["change"], np.zeros((17, 23), dtype=np.float32))
    with pytest.raises(ValueError, match="source changed"):
        dataset.verify_sources()


def test_preparation_exclusion_reporting_and_soft_predicted_weights(tmp_path):
    manifest, scenes = make_manifest(tmp_path)
    cfg = config()
    cfg["mask_source"] = "predicted"
    cfg["detector"] = {"identity": "synthetic-test-detector", "training_groups": ["scene_0"]}
    probabilities = np.load(tmp_path / scenes[0]["change"]) * 0.7
    np.save(tmp_path / scenes[0]["change"], probabilities)
    labels = np.load(tmp_path / scenes[1]["before_semantic"])
    labels[:] = 0
    np.save(tmp_path / scenes[1]["before_semantic"], labels)
    document = prepare_dataset(cfg, manifest, tmp_path / "prepared")
    assert document["coverage"]["train"]["exclusions"] == {"insufficient_annotation": 1}
    assert document["coverage"]["train"]["retained_fraction"] == 2 / 3
    dataset = SemanticROIDataset(tmp_path / "prepared/prepared.json", "train", cfg)
    assert dataset[0]["mask"].max() == pytest.approx(0.7)
    assert any(not row["retained"] and row["quality"]["reason"] for row in document["records"])
    cfg["mask_source"], cfg["detector"] = "ground_truth", None
    with pytest.raises(ValueError, match="binary"):
        prepare_dataset(cfg, manifest)


def test_split_leakage_missing_semantics_and_detector_provenance(tmp_path):
    manifest, scenes = make_manifest(tmp_path)
    protocol = data_protocol(config())
    scenes[-1]["group_id"] = scenes[0]["group_id"]
    manifest.write_text(json.dumps({"scenes": scenes}), encoding="utf-8")
    with pytest.raises(ValueError, match="Cross-split"):
        read_manifest(manifest, protocol)
    scenes[-1]["group_id"] = "test"
    scenes[-1]["before_semantic"] = "missing.npy"
    manifest.write_text(json.dumps({"scenes": scenes}), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="before_semantic"):
        read_manifest(manifest, protocol)
    with pytest.raises(FileNotFoundError, match="LEVIR-CD"):
        prepare_dataset(config(), tmp_path / "missing.json")
    scenes[-1]["before_semantic"] = "scene_4_before_semantic.npy"
    manifest.write_text(json.dumps({"scenes": scenes}), encoding="utf-8")
    cfg = config()
    cfg["mask_source"] = "predicted"
    with pytest.raises(ValueError, match="detector identity"):
        data_protocol(cfg)
    cfg["detector"] = {"identity": "test", "training_groups": ["scene_3"]}
    with pytest.raises(ValueError, match="Detector fitting-group leakage"):
        read_manifest(manifest, data_protocol(cfg))


def test_weighted_ce_over_logical_batch_and_singleton_cancellation():
    output = {
        "before_logits": torch.tensor([[2.0, 0], [0, 1.0]], requires_grad=True),
        "after_logits": torch.tensor([[0.0, 2], [1.0, 0]], requires_grad=True),
    }
    before, after, weights = torch.tensor([0, 1]), torch.tensor([1, 0]), torch.tensor([1.0, 4.0])
    loss, totals = semantic_loss(output, before, after, weights)
    expected = sum(
        F.cross_entropy(output[key], labels, weight=weights)
        for key, labels in (("before_logits", before), ("after_logits", after))
    )
    torch.testing.assert_close(loss, expected)
    assert [denominator for _, denominator in totals] == [5, 5]
    loss.backward()
    singleton = {key: value[:1] for key, value in output.items()}
    weighted, _ = semantic_loss(singleton, before[:1], after[:1], weights)
    unweighted, _ = semantic_loss(singleton, before[:1], after[:1])
    torch.testing.assert_close(weighted, unweighted)


def test_variable_logical_batches_partial_batch_and_exact_loss_logging():
    model = TinySiameseChangeClassifier(["a", "b"]).eval()
    samples = [
        {
            "before": torch.rand(3, h, w),
            "after": torch.rand(3, h, w),
            "mask": torch.ones(1, h, w),
            "before_label": i % 2,
            "after_label": (i + 1) % 2,
        }
        for i, (h, w) in enumerate(((8, 9), (9, 10), (11, 7), (8, 8), (5, 13)))
    ]
    loader = DataLoader(samples, batch_size=4, collate_fn=collate_rois)
    assert [len(batch) for batch in loader] == [4, 1]
    weights = torch.tensor([1.0, 3.0])
    combined = run_epoch(model, loader, torch.device("cpu"), weights=weights)
    logits, before, after = logical_forward(model, samples, torch.device("cpu"))
    expected, totals = semantic_loss(logits, before, after, weights)
    assert combined["loss"] == pytest.approx(float(expected.detach()))
    np.testing.assert_allclose(combined["loss_totals"], totals, rtol=1e-6)
    assert combined["samples"] == 5
    optimizer = torch.optim.AdamW(model.parameters())
    assert (
        run_epoch(model, loader, torch.device("cpu"), weights=weights, optimizer=optimizer)[
            "samples"
        ]
        == 5
    )


def test_global_confusion_macro_f1_joint_hand_calculation():
    metrics = SemanticMetrics(("a", "b", "unsupported"))
    before, after = torch.tensor([0, 0, 1]), torch.tensor([0, 1, 1])
    logits = {
        "before_logits": F.one_hot(torch.tensor([0, 1, 1]), 3).float(),
        "after_logits": F.one_hot(torch.tensor([0, 1, 0]), 3).float(),
    }
    for start, stop in ((0, 2), (2, 3)):
        metrics.update(
            {key: value[start:stop] for key, value in logits.items()},
            before[start:stop],
            after[start:stop],
        )
    result = metrics.compute()
    assert result["before_confusion"] == [[1, 1, 0], [0, 1, 0], [0, 0, 0]]
    assert result["after_confusion"] == [[1, 0, 0], [1, 1, 0], [0, 0, 0]]
    assert result["before_macro_f1"] == pytest.approx(4 / 9)
    assert result["after_macro_f1"] == pytest.approx(4 / 9)
    assert result["joint_accuracy"] == 1 / 3
    assert result["before_per_class"][2]["support"] == 0
    with pytest.raises(ValueError, match="empty"):
        SemanticMetrics(("a", "b")).compute()


def test_synthetic_smoke_bounded_tiny_overfit():
    result = synthetic_smoke(60)
    assert result["evidence"] == "synthetic implementation check only"
    assert result["last_loss"] < result["first_loss"] * 0.1
    assert result["eval_joint_accuracy"] == 1


def test_cli_missing_data_and_no_independent_training(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "semantic",
            "validate",
            "--config",
            "configs/tiny_siamese_change_classifier.yaml",
            "--manifest",
            str(tmp_path / "missing.json"),
        ],
    )
    with pytest.raises(FileNotFoundError, match="Semantic manifest missing"):
        main()
    monkeypatch.setattr("sys.argv", ["semantic", "train"])
    with pytest.raises(SystemExit):
        main()


@pytest.mark.parametrize("failure", ["weights_length", "weights_zero", "seed", "precision"])
def test_training_configuration_errors(tmp_path, failure):
    manifest, _ = make_manifest(tmp_path)
    cfg = config()
    cfg["training"].update(
        {
            "weights_length": {"class_weights": [1]},
            "weights_zero": {"class_weights": [0, 1]},
            "seed": {"seed": 2**32},
            "precision": {"precision": "fp16"},
        }[failure]
    )
    prepare_dataset(cfg, manifest, tmp_path / "prepared")
    with pytest.raises(ValueError):
        validate_parameters(hpo_parameters(cfg, tmp_path / "prepared/prepared.json"))


@pytest.mark.parametrize("bad", [1.000000001, -0.000000001, float("nan"), 2 + 1j])
def test_map_validation_precedes_float32_conversion(tmp_path, bad):
    path = tmp_path / "probabilities.npy"
    np.save(path, np.full((4, 5), bad))
    with pytest.raises(ValueError, match="finite"):
        read_map(path)


def test_unannotated_unknown_and_no_retained_regions_fail_clearly(tmp_path):
    manifest, scenes = make_manifest(tmp_path)
    cfg = config()
    cfg["classes"][1] = "unknown"
    cfg["class_ids"][2] = "unknown"
    for row in scenes:
        path = tmp_path / row["after_semantic"]
        np.save(path, np.ones_like(np.load(path)))
    prepare_dataset(cfg, manifest, tmp_path / "unknown")
    with pytest.raises(ValueError, match="labeled training examples"):
        create_loaders(
            tmp_path / "unknown/prepared.json",
            hpo_parameters(cfg, tmp_path / "unknown/prepared.json"),
        )
    cfg["roi"]["min_component_area"] = 100
    doc = prepare_dataset(cfg, manifest, tmp_path / "filtered")
    assert doc["coverage"]["train"]["retained"] == 0
    assert doc["coverage"]["train"]["exclusions"] == {"small_component": 3}
    with pytest.raises(ValueError, match="No eligible"):
        SemanticROIDataset(tmp_path / "filtered/prepared.json", "train", cfg)
