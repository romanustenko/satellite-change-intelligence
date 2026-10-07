# Изменения: TinySiameseChangeClassifier

Сформировано 6 октября 2026 года из текущих файлов проекта.

Корень проекта: `D:\Projects\Projects\Satellite change intelligence\satellite-change-intelligence`. Все пути ниже указаны относительно него.

Ниже приведены полный код новых файлов, изменение существующего экспорта и добавленная документация. Это отчёт по реализации в этой беседе; в рабочей папке нет `.git`, поэтому он не является автоматически полученным Git diff. Исходники при формировании отчёта не изменялись.

## Добавленные файлы

- `src/satchange/models/tiny_siamese_change_classifier.py`
- `src/satchange/semantic/__init__.py`
- `src/satchange/semantic/__main__.py`
- `src/satchange/semantic/roi.py`
- `src/satchange/semantic/data.py`
- `src/satchange/semantic/training.py`
- `src/satchange/semantic/inference.py`
- `configs/tiny_siamese_change_classifier.yaml`
- `tests/test_semantic.py`
- `docs/semantic_classifier.md`

## Изменённый файл: `src/satchange/models/__init__.py`

Добавлены импорт и публичный экспорт классификатора. Реестр бинарных детекторов не менялся.

```diff
 from satchange.models.siam_mit_b3_diff import SiamMiTB3Diff
+from satchange.models.tiny_siamese_change_classifier import TinySiameseChangeClassifier
 
 __all__ = [
+    "TinySiameseChangeClassifier",
     "SiamMiTB3Diff",
```

### Полное текущее содержимое

````python
"""Public models."""

from satchange.models.bit import BIT
from satchange.models.factory import count_model_parameters, create_change_model
from satchange.models.fc_siam_diff import FCSiamDiff
from satchange.models.siam_mit_b3_diff import SiamMiTB3Diff
from satchange.models.tiny_siamese_change_classifier import TinySiameseChangeClassifier

__all__ = [
    "TinySiameseChangeClassifier",
    "SiamMiTB3Diff",
    "BIT",
    "FCSiamDiff",
    "count_model_parameters",
    "create_change_model",
]
````

## Новый файл: `src/satchange/models/tiny_siamese_change_classifier.py`

````python
"""A small shared CNN for semantic classification of native-scale changed ROIs."""

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def validate_classes(classes: Sequence[str]) -> tuple[str, ...]:
    """Validate the ordered, persisted taxonomy; no dataset classes are inferred."""
    if isinstance(classes, str) or len(classes) < 2:
        raise ValueError("classes must contain at least two ordered class names.")
    if any(not isinstance(name, str) or not name.strip() for name in classes):
        raise ValueError("Every class name must be a nonempty string.")
    if len(set(classes)) != len(classes):
        raise ValueError("Class names must be unique.")
    return tuple(classes)


def validate_unit_tensor(value: torch.Tensor, name: str) -> None:
    """Reject non-floating, nonfinite, or out-of-range normalized inputs."""
    if not value.is_floating_point():
        raise ValueError(f"{name} must be floating point, normalized to [0, 1].")
    if not torch.isfinite(value).all() or (value < 0).any() or (value > 1).any():
        raise ValueError(f"{name} must be finite and in [0, 1]; logits are not probabilities.")


def masked_pool(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Area-reduce [B,1,H,W] weights and pool [B,D,h,w] features in FP32.

    Area reduction retains even one positive source pixel on odd-sized grids.
    Empty support is an error, rather than an implicit global-pooling fallback.
    """
    if features.ndim != 4 or mask.ndim != 4 or mask.shape[:2] != (features.shape[0], 1):
        raise ValueError("Pooling expects features [B,D,h,w] and aligned mask [B,1,H,W].")
    if features.device != mask.device:
        raise ValueError("Features and mask must use the same device.")
    validate_unit_tensor(mask, "mask")
    weights = F.interpolate(mask.float(), size=features.shape[-2:], mode="area")
    denominator = weights.sum(dim=(-2, -1))
    if (denominator <= 0).any():
        raise ValueError("Every ROI requires a nonempty effective mask.")
    return (features.float() * weights).sum(dim=(-2, -1)) / (denominator + 1e-8)


class TinySiameseChangeClassifier(nn.Module):
    """Shared RGB backbone and head, with limited local receptive-field context.

    forward(before, after, mask): images [B,3,H,W], mask [B,1,H,W],
    raw before/after logits [B,C]. RGB remains unmasked and at native scale.
    Two pooling stages produce floor(H/4) x floor(W/4) feature grids.
    Both dates share one BatchNorm update by concatenating on the batch axis.
    """

    def __init__(self, classes: Sequence[str], dropout: float = 0.1) -> None:
        super().__init__()
        self.classes = validate_classes(classes)
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        self.dropout = float(dropout)
        layers: list[nn.Module] = []
        for index, (incoming, outgoing) in enumerate(((3, 32), (32, 64), (64, 128))):
            layers.extend(
                [
                    nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False),
                    nn.BatchNorm2d(outgoing),
                    nn.ReLU(),
                ]
            )
            if index < 2:
                layers.append(nn.MaxPool2d(2))
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, len(self.classes))
        )

    def forward(
        self, before: torch.Tensor, after: torch.Tensor, mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Return two semantic-logit tensors using exactly the same parameters."""
        if before.ndim != 4 or before.shape[1] != 3 or before.shape != after.shape:
            raise ValueError("Images must be aligned [B,3,H,W] tensors.")
        if before.shape[0] == 0 or min(before.shape[-2:]) < 4:
            raise ValueError("Images require a nonempty batch and height/width of at least 4.")
        if mask.shape != (before.shape[0], 1, *before.shape[-2:]):
            raise ValueError("mask must be aligned [B,1,H,W].")
        if before.device != after.device or before.device != mask.device:
            raise ValueError("Images and mask must use the same device.")
        if before.dtype != torch.float32 or after.dtype != torch.float32:
            raise ValueError("RGB inputs must be float32 normalized to [0, 1].")
        for name, value in (("before", before), ("after", after), ("mask", mask)):
            validate_unit_tensor(value, name)
        if (mask.float().sum(dim=(-2, -1)) <= 0).any():
            raise ValueError("Every ROI requires a nonempty effective mask.")
        features = self.backbone(torch.cat((before, after), dim=0))
        vectors = masked_pool(features, torch.cat((mask, mask), dim=0))
        before_logits, after_logits = self.head(vectors).chunk(2, dim=0)
        return {"before_logits": before_logits, "after_logits": after_logits}
````

## Новый файл: `src/satchange/semantic/__init__.py`

````python
"""Native-scale ROI semantic classification, separate from binary detection."""

from satchange.semantic.roi import (
    ROI,
    ROISettings,
    align_whole_image_probabilities,
    extract_rois,
)

__all__ = ["ROI", "ROISettings", "align_whole_image_probabilities", "extract_rois"]
````

## Новый файл: `src/satchange/semantic/__main__.py`

````python
"""Small explicit CLI for semantic ROI preparation, training, evaluation, and inference."""

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from satchange.execution import seed_everything
from satchange.models import TinySiameseChangeClassifier, count_model_parameters
from satchange.semantic.data import prepare_dataset, read_map, read_rgb, write_json
from satchange.semantic.inference import evaluate_classifier, predict_checkpoint
from satchange.semantic.training import fit_classifier, semantic_loss, summarize_runs
from satchange.utils import load_config


def synthetic_smoke(steps: int = 60) -> dict[str, Any]:
    """Bounded CPU forward/backward and tiny-overfit check; no real-dataset evidence."""
    if not 1 <= steps <= 200:
        raise ValueError("Synthetic steps must be between 1 and 200.")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        seed_everything(42)
        model = TinySiameseChangeClassifier(["synthetic_dark", "synthetic_bright"], dropout=0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        before, after = torch.zeros(2, 3, 17, 23), torch.ones(2, 3, 17, 23)
        mask = torch.ones(2, 1, 17, 23)
        targets = (torch.zeros(2, dtype=torch.long), torch.ones(2, dtype=torch.long))
        losses = []
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss, _ = semantic_loss(model(before, after, mask), *targets)
            loss.backward()
            if any(
                p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()
            ):
                raise RuntimeError("Synthetic backward produced nonfinite gradients.")
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            output = model(before, after, mask)
            joint = (
                (
                    (output["before_logits"].argmax(1) == targets[0])
                    & (output["after_logits"].argmax(1) == targets[1])
                )
                .float()
                .mean()
            )
        return {
            "evidence": "synthetic implementation check only",
            "seed": 42,
            "steps": steps,
            "first_loss": losses[0],
            "last_loss": losses[-1],
            "eval_joint_accuracy": float(joint),
            **count_model_parameters(model),
        }
    finally:
        torch.set_num_threads(previous_threads)


def main() -> None:
    """Run only the operation explicitly selected by the caller."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "prepare"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", required=True)
        sub.add_argument("--manifest", required=True)
        if name == "prepare":
            sub.add_argument("--output", required=True)
    train = commands.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--prepared", required=True)
    train.add_argument("--run", required=True)
    train.add_argument("--resume", help="Trusted last.pt; --epochs is the total target.")
    train.add_argument("--epochs", type=int)
    train.add_argument("--seed", type=int)
    train.add_argument("--device", default="cpu")
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--prepared", required=True)
    evaluation.add_argument("--split", choices=("val", "test"), required=True)
    evaluation.add_argument("--device", default="cpu")
    evaluation.add_argument("--output", required=True)
    inference = commands.add_parser("infer")
    inference.add_argument("--checkpoint", required=True)
    for key in ("before", "after", "change", "output"):
        inference.add_argument(f"--{key}", required=True)
    inference.add_argument("--device", default="cpu")
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--steps", type=int, default=60)
    summary = commands.add_parser("summarize")
    summary.add_argument("--results", nargs="+", required=True)
    summary.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command in ("evaluate", "infer", "summarize"):
        if Path(args.output).exists():
            raise FileExistsError(f"Output already exists: {args.output}")
    if args.command in ("validate", "prepare"):
        result = prepare_dataset(
            load_config(args.config),
            args.manifest,
            args.output if args.command == "prepare" else None,
        )
        print(
            json.dumps({"coverage": result["coverage"], "protocol": result["protocol"]}, indent=2)
        )
        return
    if args.command == "train":
        config = load_config(args.config)
        for key in ("epochs", "seed"):
            if getattr(args, key) is not None:
                config.setdefault("training", {})[key] = getattr(args, key)
        history = fit_classifier(
            config, args.prepared, args.run, torch.device(args.device), resume=args.resume
        )
        print(json.dumps({"completed_epoch": len(history), "last_validation": history[-1]["val"]}))
        return
    if args.command == "smoke":
        result = synthetic_smoke(args.steps)
    elif args.command == "evaluate":
        result = evaluate_classifier(
            args.checkpoint, args.prepared, args.split, torch.device(args.device)
        )
    elif args.command == "infer":
        result = predict_checkpoint(
            args.checkpoint,
            read_rgb(args.before),
            read_rgb(args.after),
            read_map(args.change).unsqueeze(0),
            torch.device(args.device),
        )
    else:
        result = summarize_runs(
            [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.results]
        )
    if args.command != "smoke":
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
````

## Новый файл: `src/satchange/semantic/roi.py`

````python
"""Deterministic component crops in the original image coordinate frame."""

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from satchange.models.tiny_siamese_change_classifier import validate_unit_tensor


@dataclass(frozen=True)
class ROISettings:
    """Starting crop settings, not optimized values; all distances are native pixels."""

    threshold: float = 0.5
    connectivity: int = 8
    context_pixels: int = 16
    min_crop_size: int = 128
    min_component_area: int = 1

    def __post_init__(self) -> None:
        if not 0 < self.threshold <= 1:
            raise ValueError("threshold must be in (0, 1], excluding zero-weight components.")
        if self.connectivity not in (4, 8):
            raise ValueError("connectivity must be 4 or 8.")
        for name, minimum in (
            ("context_pixels", 0),
            ("min_crop_size", 4),
            ("min_component_area", 1),
        ):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")


@dataclass
class ROI:
    """Aligned crops with component and crop boxes (x0,y0,x1,y1), upper bounds exclusive."""

    component_id: int
    component_bbox: tuple[int, int, int, int]
    crop_bbox: tuple[int, int, int, int]
    area: int
    before: torch.Tensor
    after: torch.Tensor
    probabilities: torch.Tensor
    support: torch.Tensor
    mask: torch.Tensor
    minimum_extent_limited: bool


DEFAULT_ROI_SETTINGS = ROISettings()


def validate_scene(before: torch.Tensor, after: torch.Tensor, probabilities: torch.Tensor) -> None:
    """Require native, aligned RGB [3,H,W] and probability [1,H,W] tensors."""
    if before.ndim != 3 or before.shape[0] != 3 or before.shape != after.shape:
        raise ValueError("Scene images must be aligned [3,H,W].")
    if min(before.shape[-2:]) < 4:
        raise ValueError("Native images are too small: the backbone requires height/width >= 4.")
    if probabilities.shape != (1, *before.shape[-2:]):
        raise ValueError("Probability map must already align with native RGB as [1,H,W].")
    if before.dtype != torch.float32 or after.dtype != torch.float32:
        raise ValueError("Native RGB must be float32 normalized to [0, 1].")
    if before.device != after.device or before.device != probabilities.device:
        raise ValueError("All scene tensors must use the same device.")
    for name, value in (("before", before), ("after", after), ("probabilities", probabilities)):
        validate_unit_tensor(value, name)


def _extend_axis(low: int, high: int, extent: int, settings: ROISettings) -> tuple[int, int]:
    length = min(extent, max(high - low + 2 * settings.context_pixels, settings.min_crop_size))
    start = max(0, min((low + high - length) // 2, extent - length))
    return start, start + length


def extract_rois(
    before: torch.Tensor,
    after: torch.Tensor,
    probabilities: torch.Tensor,
    settings: ROISettings = DEFAULT_ROI_SETTINGS,
) -> list[ROI]:
    """Extract real context without RGB resizing/padding or truncating components.

    Components are sorted by their first row-major pixel, independently of OpenCV's
    label numbering. Only the selected component contributes soft pooling weights.
    Detector output geometry must be mapped by the caller before using this API.
    """
    validate_scene(before, after, probabilities)
    binary = (probabilities[0] >= settings.threshold).detach().cpu().numpy().astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=settings.connectivity
    )
    components = []
    unique_labels, first_pixels = np.unique(labels, return_index=True)
    first_by_label = dict(zip(unique_labels.tolist(), first_pixels.tolist(), strict=True))
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        first = first_by_label[label]
        components.append((first, label, area))
    height, width = before.shape[-2:]
    results = []
    for component_id, (_, label, area) in enumerate(sorted(components), start=1):
        if area < settings.min_component_area:
            continue
        x0, y0, w, h = (int(value) for value in stats[label, :4])
        x1, y1 = x0 + w, y0 + h
        left, right = _extend_axis(x0, x1, width, settings)
        top, bottom = _extend_axis(y0, y1, height, settings)
        support = torch.from_numpy((labels[top:bottom, left:right] == label).copy())
        support = support.unsqueeze(0).to(device=before.device)
        probability_crop = probabilities[:, top:bottom, left:right].contiguous()
        results.append(
            ROI(
                component_id=component_id,
                component_bbox=(x0, y0, x1, y1),
                crop_bbox=(left, top, right, bottom),
                area=area,
                before=before[:, top:bottom, left:right].contiguous(),
                after=after[:, top:bottom, left:right].contiguous(),
                probabilities=probability_crop,
                support=support,
                mask=(probability_crop * support).float().contiguous(),
                minimum_extent_limited=height < settings.min_crop_size
                or width < settings.min_crop_size,
            )
        )
    return results


def align_whole_image_probabilities(
    probabilities: torch.Tensor,
    original_size: tuple[int, int],
    detector_input_size: tuple[int, int],
) -> torch.Tensor:
    """Map a known *whole-image resize* detector probability grid back to native (H,W).

    Caller must verify that preprocessing was only a whole-image resize. Cropped,
    padded, and tiled outputs need their own known inverse geometry; this helper
    must not be used to guess it. RGB is never resampled by this function.
    """
    if probabilities.shape != (1, *detector_input_size):
        raise ValueError("Probability grid must match the documented detector input size.")
    if len(original_size) != 2 or any(
        type(value) is not int or value < 4 for value in original_size
    ):
        raise ValueError("original_size must be native (height,width), both >= 4.")
    validate_unit_tensor(probabilities, "probabilities")
    return (
        F.interpolate(
            probabilities.float().unsqueeze(0),
            size=original_size,
            mode="bilinear",
            align_corners=False,
        )[0]
        .clamp(0, 1)
        .contiguous()
    )
````

## Новый файл: `src/satchange/semantic/data.py`

````python
"""Manifest-based semantic supervision; binary LEVIR masks cannot supply these labels."""

import hashlib
import json
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from satchange.data.transforms import RandomFlipRotate
from satchange.models.tiny_siamese_change_classifier import validate_classes
from satchange.semantic.roi import ROISettings, extract_rois

NORMALIZATION = "RGB uint8 / 255; native resolution"


def write_json(path: Path, value: Any) -> None:
    """Save readable JSON with finite numeric values."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def file_digest(path: Path) -> str:
    """Hash source content so a resume cannot silently use changed annotations/images."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rgb(path: str | Path) -> torch.Tensor:
    """Load an actual RGB uint8 image as native [3,H,W] float32 in [0,1]."""
    with Image.open(path) as image:
        if image.mode != "RGB":
            raise ValueError(f"Expected RGB imagery, not {image.mode}: {path}")
        pixels = np.array(image, dtype=np.uint8, copy=True)
    return torch.from_numpy(pixels).permute(2, 0, 1).float().div_(255).contiguous()


def read_map(path: str | Path, *, semantic: bool = False) -> torch.Tensor:
    """Read a non-pickled .npy [H,W] map, preserving integer semantic IDs."""
    if Path(path).suffix.lower() != ".npy":
        raise ValueError(f"Maps must be .npy [H,W] arrays: {path}")
    array = np.load(path, allow_pickle=False)
    if array.ndim != 2:
        raise ValueError(f"Map must have shape [H,W]: {path}")
    if semantic and not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"Semantic maps require integer IDs: {path}")
    if not semantic and not (
        np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_)
    ):
        raise ValueError(f"Change maps require numeric probabilities: {path}")
    if not semantic and (
        np.iscomplexobj(array)
        or not np.isfinite(array).all()
        or (array < 0).any()
        or (array > 1).any()
    ):
        raise ValueError(f"Change probabilities must be finite and in [0,1]: {path}")
    if semantic and np.issubdtype(array.dtype, np.unsignedinteger):
        if array.size and array.max() > np.iinfo(np.int64).max:
            raise ValueError(f"Semantic IDs must fit signed int64: {path}")
    # int64 also allows explicit negative ignore IDs without binarizing labels.
    return torch.from_numpy(array.astype(np.int64 if semantic else np.float32, copy=True))


def data_protocol(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve the single explicit taxonomy, ROI, annotation, and mask-source contract."""
    classes = validate_classes(config.get("classes", []))
    mapping = config.get("class_ids")
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("class_ids must explicitly map raw integer IDs to class names.")
    ids: dict[str, int] = {}
    for raw, name in mapping.items():
        if str(int(raw)) != str(raw) or name not in classes:
            raise ValueError("class_ids keys must be integer IDs and values declared class names.")
        ids[str(raw)] = classes.index(name)
    if set(ids.values()) != set(range(len(classes))):
        raise ValueError("Every declared class needs an explicit raw-ID mapping.")
    ignore_ids = config.get("ignore_ids")
    if not isinstance(ignore_ids, list) or any(type(value) is not int for value in ignore_ids):
        raise ValueError("ignore_ids must explicitly list integer unlabeled/background IDs.")
    if set(map(str, ignore_ids)) & ids.keys():
        raise ValueError("An ID cannot be both a learned class and ignored.")
    source = config.get("mask_source")
    if source not in ("ground_truth", "predicted"):
        raise ValueError("mask_source must be 'ground_truth' or 'predicted'.")
    settings = ROISettings(**config.get("roi", {}))
    policy = {"min_transition_purity": 0.8, "min_valid_fraction": 0.8}
    supplied = config.get("label_policy", {})
    if set(supplied) - policy.keys():
        raise ValueError("Unknown label_policy key.")
    policy.update(supplied)
    if any(not 0 < value <= 1 for value in policy.values()):
        raise ValueError("Label purity and valid-fraction thresholds must be in (0,1].")
    detector = config.get("detector")
    if source == "predicted":
        if not isinstance(detector, dict) or not detector.get("identity"):
            raise ValueError(
                "Predicted masks need detector identity and training_groups provenance."
            )
        groups = detector.get("training_groups")
        if not isinstance(groups, list) or any(not isinstance(group, str) for group in groups):
            raise ValueError(
                "detector.training_groups must list the frozen detector's fitting groups."
            )
    elif detector is not None:
        raise ValueError("ground_truth experiments must set detector: null.")
    return {
        "classes": list(classes),
        "class_ids": ids,
        "ignore_ids": sorted(set(ignore_ids)),
        "roi": asdict(settings),
        "label_policy": policy,
        "mask_source": source,
        "detector": detector,
        "normalization": NORMALIZATION,
    }


def read_manifest(path: str | Path, protocol: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate scene splits, unique IDs, source files, and detector fitting-group provenance."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Semantic manifest missing: {path}. Supply real before/after semantic annotations; "
            "LEVIR-CD binary labels are insufficient. See docs/semantic_classifier.md."
        )
    document = json.loads(path.read_text(encoding="utf-8"))
    scenes = document.get("scenes") if isinstance(document, dict) else None
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("Manifest must contain a nonempty 'scenes' list.")
    group_splits: dict[str, str] = {}
    ids: set[str] = set()
    result = []
    fields = ("before", "after", "before_semantic", "after_semantic", "change")
    for scene in scenes:
        required = set(fields + ("sample_id", "group_id", "split"))
        if not isinstance(scene, dict) or required - scene.keys():
            raise ValueError("Each scene needs RGB, both semantic labels, change, split, and IDs.")
        sample_id, group, split = (scene[key] for key in ("sample_id", "group_id", "split"))
        if not all(isinstance(value, str) and value for value in (sample_id, group, split)):
            raise ValueError("sample_id, group_id, and split must be nonempty strings.")
        if sample_id in ids or split not in ("train", "val", "test"):
            raise ValueError("sample_id must be unique and split must be train, val, or test.")
        if group in group_splits and group_splits[group] != split:
            raise ValueError(f"Cross-split group leakage: {group}")
        if split != "train" and group in (protocol.get("detector") or {}).get(
            "training_groups", []
        ):
            raise ValueError(f"Detector fitting-group leakage into {split}: {group}")
        ids.add(sample_id)
        group_splits[group] = split
        resolved = dict(scene)
        for field in fields:
            if not isinstance(scene[field], str) or not scene[field]:
                raise ValueError(
                    f"{sample_id}: missing {field} path; no fabricated-label fallback."
                )
            source = (path.parent / scene[field]).resolve()
            if not source.is_file():
                raise FileNotFoundError(f"{sample_id}: missing {field}: {source}")
            resolved[field] = str(source)
        result.append(resolved)
    return result


def dominant_transition(
    before: torch.Tensor,
    after: torch.Tensor,
    support: torch.Tensor,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Select an observed joint label pair only within component support; report exclusions."""
    if before.shape != after.shape or support.shape != before.shape:
        raise ValueError("Semantic maps and component support must align as [H,W].")
    if before.ndim != 2 or support.dtype != torch.bool:
        raise ValueError("Component support must be boolean [H,W].")
    integer_types = (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64)
    if before.dtype not in integer_types or after.dtype not in integer_types:
        raise ValueError("Semantic maps require integer class IDs.")
    mapped = []
    for labels in (before, after):
        values = torch.full_like(labels, -1, dtype=torch.long)
        known = torch.zeros_like(labels, dtype=torch.bool)
        for raw, target in protocol["class_ids"].items():
            selected = labels == int(raw)
            values[selected] = target
            known |= selected
        for raw in protocol["ignore_ids"]:
            known |= labels == raw
        if not known.all():
            raise ValueError(
                "Semantic map contains undeclared class IDs; set class_ids/ignore_ids."
            )
        mapped.append(values)
    valid = support.bool() & (mapped[0] >= 0) & (mapped[1] >= 0)
    area, valid_count = int(support.sum()), int(valid.sum())
    fraction = valid_count / area if area else 0.0
    quality: dict[str, Any] = {"valid_pixels": valid_count, "valid_fraction": fraction}
    if not valid_count or fraction < protocol["label_policy"]["min_valid_fraction"]:
        return {**quality, "reason": "insufficient_annotation"}
    pairs = torch.stack((mapped[0][valid], mapped[1][valid]), dim=1)
    unique, counts = torch.unique(pairs, dim=0, return_counts=True)
    largest = int(counts.max())
    purity = largest / valid_count
    quality["transition_purity"] = purity
    if int((counts == largest).sum()) != 1:
        return {**quality, "reason": "tied_transition"}
    if purity < protocol["label_policy"]["min_transition_purity"]:
        return {**quality, "reason": "mixed_transition"}
    pair = unique[int(counts.argmax())].tolist()
    return {**quality, "reason": None, "before_label": pair[0], "after_label": pair[1]}


def prepare_dataset(
    config: dict[str, Any],
    manifest: str | Path,
    destination: str | Path | None = None,
) -> dict[str, Any]:
    """Validate real sources and generate ROI records/masks, without duplicating RGB imagery.

    destination=None validates and reports without writing. An existing output is never replaced.
    Coverage is annotation-eligible ROI coverage, not detection matching or end-to-end accuracy.
    """
    protocol = data_protocol(config)
    scenes = read_manifest(manifest, protocol)
    settings = ROISettings(**protocol["roi"])
    output = Path(destination).resolve() if destination is not None else None
    if output is not None and output.exists():
        raise FileExistsError(f"Preparation output already exists: {output}")
    records = []
    reports: dict[str, Any] = {
        split: {
            "proposed": 0,
            "retained": 0,
            "exclusions": Counter(),
            "before_distribution": Counter(),
            "after_distribution": Counter(),
            "transition_distribution": Counter(),
            "scenes_without_components": 0,
        }
        for split in ("train", "val", "test")
    }
    hashes = {}
    for scene in scenes:
        for key in ("before", "after", "before_semantic", "after_semantic", "change"):
            source = scene[key]
            if source not in hashes:
                hashes[source] = file_digest(Path(source))
        before, after = read_rgb(scene["before"]), read_rgb(scene["after"])
        probabilities = read_map(scene["change"]).unsqueeze(0)
        if protocol["mask_source"] == "ground_truth":
            if not ((probabilities == 0) | (probabilities == 1)).all():
                raise ValueError("ground_truth change maps must contain binary 0/1 support.")
        semantic_before = read_map(scene["before_semantic"], semantic=True)
        semantic_after = read_map(scene["after_semantic"], semantic=True)
        if semantic_before.shape != before.shape[-2:] or semantic_after.shape != before.shape[-2:]:
            raise ValueError(f"{scene['sample_id']}: native semantic/RGB spatial mismatch.")
        # Validate the whole taxonomy even when no component is proposed.
        dominant_transition(
            semantic_before,
            semantic_after,
            torch.zeros_like(semantic_before, dtype=torch.bool),
            protocol,
        )
        rois = extract_rois(before, after, probabilities, replace(settings, min_component_area=1))
        report = reports[scene["split"]]
        report["scenes_without_components"] += int(not rois)
        for roi in rois:
            report["proposed"] += 1
            x0, y0, x1, y1 = roi.crop_bbox
            quality = dominant_transition(
                semantic_before[y0:y1, x0:x1],
                semantic_after[y0:y1, x0:x1],
                roi.support[0],
                protocol,
            )
            reason = (
                "small_component" if roi.area < settings.min_component_area else quality["reason"]
            )
            if reason:
                report["exclusions"][reason] += 1
                records.append(
                    {
                        **scene,
                        "component_id": roi.component_id,
                        "crop_bbox": list(roi.crop_bbox),
                        "area": roi.area,
                        "component_bbox": list(roi.component_bbox),
                        "retained": False,
                        "minimum_extent_limited": roi.minimum_extent_limited,
                        "quality": {**quality, "reason": reason},
                    }
                )
                continue
            mask_name = f"masks/roi_{len(records):08d}.npz"
            if output is not None:
                (output / "masks").mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    output / mask_name,
                    probabilities=roi.probabilities.numpy(),
                    support=roi.support.numpy(),
                )
            record = {
                **scene,
                "component_id": roi.component_id,
                "crop_bbox": list(roi.crop_bbox),
                "component_bbox": list(roi.component_bbox),
                "area": roi.area,
                "mask_file": mask_name,
                "retained": True,
                "quality": quality,
                "minimum_extent_limited": roi.minimum_extent_limited,
                "before_label": quality["before_label"],
                "after_label": quality["after_label"],
            }
            records.append(record)
            report["retained"] += 1
            left, right = (
                protocol["classes"][record[key]] for key in ("before_label", "after_label")
            )
            report["before_distribution"][left] += 1
            report["after_distribution"][right] += 1
            report["transition_distribution"][f"{left} -> {right}"] += 1
    for report in reports.values():
        report["retained_fraction"] = (
            report["retained"] / report["proposed"] if report["proposed"] else 0.0
        )
    provenance = {
        "manifest": str(Path(manifest).resolve()),
        "scenes": scenes,
        "source_hashes": hashes,
    }
    document = {
        "schema_version": 1,
        "protocol": protocol,
        "provenance": provenance,
        "records": records,
        "coverage": reports,
        "coverage_definition": "annotation-eligible / proposed connected components",
    }
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        document["mask_hashes"] = {
            record["mask_file"]: file_digest(output / record["mask_file"])
            for record in records
            if record["retained"]
        }
        write_json(output / "prepared.json", document)
    return document


class SemanticROIDataset(Dataset):
    """Load variable-size real RGB crops and component weights from prepared records."""

    def __init__(
        self,
        prepared: str | Path,
        split: str,
        config: dict[str, Any],
        *,
        augment: bool = False,
    ) -> None:
        self.path = Path(prepared).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(
                f"Prepared semantic data missing: {self.path}. Run semantic prepare with "
                "real before/after semantic annotations; no synthetic-label fallback is provided."
            )
        self.document = json.loads(self.path.read_text(encoding="utf-8"))
        if self.document.get("schema_version") != 1:
            raise ValueError("Unsupported prepared-data schema.")
        if self.document["protocol"] != data_protocol(config):
            raise ValueError(
                "Prepared data protocol differs from configuration; prepare a new dataset."
            )
        if split not in ("train", "val", "test") or (augment and split != "train"):
            raise ValueError("Use train/val/test and augmentation only on train.")
        self.records = [
            row for row in self.document["records"] if row["retained"] and row["split"] == split
        ]
        if not self.records:
            raise ValueError(f"No eligible semantic ROIs in {split}; supply annotated real data.")
        self.transform = RandomFlipRotate() if augment else None
        self.split = split

    def verify_sources(self) -> None:
        """Check input and generated-mask identities once before a run/evaluation."""
        for source, expected in self.document["provenance"]["source_hashes"].items():
            if file_digest(Path(source)) != expected:
                raise ValueError(f"Prepared source changed: {source}; regenerate prepared data.")
        for source, expected in self.document["mask_hashes"].items():
            if file_digest(self.path.parent / source) != expected:
                raise ValueError(f"Prepared ROI mask changed: {source}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records[index]
        x0, y0, x1, y1 = row["crop_bbox"]
        with np.load(self.path.parent / row["mask_file"], allow_pickle=False) as archive:
            mask = torch.from_numpy((archive["probabilities"] * archive["support"]).copy()).float()
        sample = {
            "before": read_rgb(row["before"])[:, y0:y1, x0:x1].contiguous(),
            "after": read_rgb(row["after"])[:, y0:y1, x0:x1].contiguous(),
            "mask": mask.contiguous(),
            "sample_id": row["sample_id"],
            "before_label": row["before_label"],
            "after_label": row["after_label"],
        }
        return self.transform(sample) if self.transform is not None else sample


def collate_rois(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Preserve different native crop sizes; never resize or pad logical batches."""
    return samples
````

## Новый файл: `src/satchange/semantic/training.py`

````python
"""Fixed-budget semantic training and global metrics, independent of binary segmentation."""

import hashlib
import math
import platform
import statistics
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from satchange.execution import capture_rng, precision_context, restore_rng, seed_everything
from satchange.models import TinySiameseChangeClassifier, count_model_parameters
from satchange.models.tiny_siamese_change_classifier import validate_classes
from satchange.semantic.data import (
    NORMALIZATION,
    SemanticROIDataset,
    collate_rois,
    data_protocol,
    file_digest,
    write_json,
)
from satchange.training import _save_checkpoint, load_training_checkpoint

TASK = "semantic_roi_classification"
ARCHITECTURE = "tiny_siamese_change_classifier_v1_32_64_128"
SELECTION_RULE = "mean(before_macro_f1, after_macro_f1) on val; earlier epoch wins ties"
CPU_DEVICE = torch.device("cpu")


class SemanticMetrics:
    """Two global CxC confusion matrices: rows true, columns predicted.

    Undefined precision/recall/F1 are zero; macro F1 includes every declared class,
    including classes with zero support. Joint accuracy requires both dates correct.
    """

    def __init__(self, classes: tuple[str, ...]) -> None:
        self.classes = validate_classes(classes)
        self.matrices = [
            torch.zeros(len(classes), len(classes), dtype=torch.int64) for _ in range(2)
        ]
        self.joint_correct = 0
        self.samples = 0

    def update(
        self,
        logits: dict[str, torch.Tensor],
        before: torch.Tensor,
        after: torch.Tensor,
    ) -> None:
        """Accumulate counts, never per-batch F1 averages."""
        correct = []
        for index, (key, target) in enumerate((("before_logits", before), ("after_logits", after))):
            values = logits[key].detach().float().cpu()
            target = target.detach().cpu()
            if values.shape != (target.numel(), len(self.classes)) or target.ndim != 1:
                raise ValueError("Metrics require logits [B,C] and labels [B].")
            if (
                target.dtype != torch.long
                or (target < 0).any()
                or (target >= len(self.classes)).any()
            ):
                raise ValueError("Metric labels must be int64 indices in the declared class order.")
            if not torch.isfinite(values).all():
                raise ValueError("Semantic logits must be finite.")
            predicted = values.argmax(dim=1)
            count = torch.bincount(
                target * len(self.classes) + predicted, minlength=len(self.classes) ** 2
            )
            self.matrices[index] += count.reshape(len(self.classes), len(self.classes))
            correct.append(predicted == target)
        self.samples += before.numel()
        self.joint_correct += int((correct[0] & correct[1]).sum())

    def compute(self) -> dict[str, Any]:
        """Return counts, class supports, per-date metrics, and the declared selection score."""
        if not self.samples:
            raise ValueError("Cannot evaluate an empty semantic split.")
        result: dict[str, Any] = {
            "classes": list(self.classes),
            "samples": self.samples,
            "joint_accuracy": self.joint_correct / self.samples,
            "zero_denominator_convention": "zero; macro includes all declared classes",
        }
        for date, matrix in zip(("before", "after"), self.matrices, strict=True):
            counts = matrix.double()
            tp, support, predicted = counts.diag(), counts.sum(1), counts.sum(0)
            precision = torch.where(predicted > 0, tp / predicted, 0)
            recall = torch.where(support > 0, tp / support, 0)
            f1 = torch.where(
                precision + recall > 0, 2 * precision * recall / (precision + recall), 0
            )
            result[f"{date}_accuracy"] = float(tp.sum()) / self.samples
            result[f"{date}_macro_f1"] = float(f1.mean())
            result[f"{date}_confusion"] = matrix.tolist()
            result[f"{date}_per_class"] = [
                {
                    "class": name,
                    "precision": float(precision[i]),
                    "recall": float(recall[i]),
                    "f1": float(f1[i]),
                    "support": int(support[i]),
                }
                for i, name in enumerate(self.classes)
            ]
        result["selection_score"] = (result["before_macro_f1"] + result["after_macro_f1"]) / 2
        result["selection_rule"] = SELECTION_RULE
        return result


def logical_forward(
    model: TinySiameseChangeClassifier,
    samples: list[dict[str, Any]],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Forward each variable-size pair, then concatenate logits over the logical batch.

    BatchNorm statistics cover each temporal pair, not the full logical minibatch.
    """
    if not samples:
        raise ValueError("Logical batches must not be empty.")
    outputs = [
        model(*(sample[key].unsqueeze(0).to(device) for key in ("before", "after", "mask")))
        for sample in samples
    ]
    logits = {
        key: torch.cat([output[key] for output in outputs])
        for key in ("before_logits", "after_logits")
    }
    labels = [
        torch.tensor([sample[key] for sample in samples], dtype=torch.long, device=device)
        for key in ("before_label", "after_label")
    ]
    return logits, labels[0], labels[1]


def semantic_loss(
    logits: dict[str, torch.Tensor],
    before: torch.Tensor,
    after: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[tuple[float, float]]]:
    """Sum date-wise weighted CE over the logical batch; return exact logging denominators.

    For a singleton logical batch, normalized weighted CE cancels its class weight.
    Split logging sums numerators/denominators separately for each date.
    """
    terms = []
    totals = []
    for key, labels in (("before_logits", before), ("after_logits", after)):
        numerator = F.cross_entropy(logits[key].float(), labels, weight=weights, reduction="sum")
        denominator = (
            weights[labels].sum() if weights is not None else numerator.new_tensor(labels.numel())
        )
        if not torch.isfinite(numerator) or not torch.isfinite(denominator) or denominator <= 0:
            raise ValueError("Semantic CE needs finite logits and a positive loss denominator.")
        terms.append(numerator / denominator)
        totals.append((float(numerator.detach()), float(denominator.detach())))
    return terms[0] + terms[1], totals


def run_epoch(
    model: TinySiameseChangeClassifier,
    loader: DataLoader,
    device: torch.device,
    *,
    weights: torch.Tensor | None = None,
    precision: str = "fp32",
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    """Fit or evaluate a split, handling final partial logical batches correctly."""
    model.train(optimizer is not None)
    metrics = SemanticMetrics(model.classes)
    totals = [[0.0, 0.0], [0.0, 0.0]]
    for samples in loader:
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None), precision_context(device, precision):
            logits, before, after = logical_forward(model, samples, device)
            loss, batch_totals = semantic_loss(logits, before, after, weights)
        if optimizer is not None:
            loss.backward()
            optimizer.step()
        metrics.update(logits, before, after)
        for total, batch_total in zip(totals, batch_totals, strict=True):
            total[0] += batch_total[0]
            total[1] += batch_total[1]
    result = metrics.compute()
    result["loss"] = sum(numerator / denominator for numerator, denominator in totals)
    result["loss_totals"] = totals
    return result


def resolve_config(config: dict[str, Any]) -> dict[str, Any]:
    """Fill only supported baseline settings; epochs means a total target, not additional epochs."""
    result = deepcopy(config)
    data_protocol(result)
    result.setdefault("dropout", 0.1)
    training: dict[str, Any] = {
        "epochs": 50,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 4,
        "num_workers": 0,
        "precision": "fp32",
        "seed": 42,
        "class_weights": None,
    }
    supplied = result.get("training", {})
    if set(supplied) - training.keys():
        raise ValueError("Unknown semantic training setting; scheduler/HPO are not part of v1.")
    training.update(supplied)
    for key, lower in (("epochs", 1), ("batch_size", 1), ("num_workers", 0), ("seed", 0)):
        if type(training[key]) is not int or training[key] < lower:
            raise ValueError(f"training.{key} must be an integer >= {lower}.")
    if training["seed"] >= 2**32:
        raise ValueError("seed must be below 2**32 for the shared NumPy seeding contract.")
    for key, lower in (("learning_rate", 0), ("weight_decay", 0)):
        value = training[key]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < lower:
            raise ValueError(f"training.{key} must be finite and nonnegative.")
    if training["learning_rate"] == 0:
        raise ValueError("learning_rate must be positive.")
    if training["precision"] not in ("fp32", "bf16"):
        raise ValueError("Only fp32 and CUDA bf16 are supported.")
    weights = training["class_weights"]
    if weights is not None:
        if not isinstance(weights, list) or len(weights) != len(result["classes"]):
            raise ValueError("class_weights must be a vector aligned to classes, or null.")
        if any(not isinstance(w, (int, float)) or not math.isfinite(w) or w <= 0 for w in weights):
            raise ValueError("Explicit class weights must be finite and strictly positive.")
    if not isinstance(result["dropout"], (int, float)) or not 0 <= result["dropout"] < 1:
        raise ValueError("dropout must be in [0,1).")
    result["training"] = training
    return result


def _runtime_metadata(device: torch.device) -> dict[str, Any]:
    package = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted((package / "semantic").glob("*.py")) + [
        package / "models/tiny_siamese_change_classifier.py"
    ]:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=package, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "device": str(device),
        "code_revision": revision,
        "semantic_source_sha256": digest.hexdigest(),
    }


def validate_checkpoint(state: dict[str, Any]) -> None:
    """Validate semantic identity and required resume/inference state before loading parameters."""
    required = {
        "metadata",
        "history",
        "best_score",
        "best_epoch",
        "best_weights",
        "rng_state",
        "epoch",
        "model_state_dict",
        "optimizer_state_dict",
    }
    if (
        required - state.keys()
        or state.get("task") != TASK
        or state.get("architecture") != ARCHITECTURE
    ):
        raise ValueError("Not a compatible complete semantic classifier checkpoint.")
    metadata = state["metadata"]
    metadata_keys = {
        "config",
        "protocol",
        "normalization",
        "selection_rule",
        "runtime",
        "compatibility_config",
        "data_identity",
    }
    if not isinstance(metadata, dict) or metadata_keys - metadata.keys():
        raise ValueError("Checkpoint semantic metadata is incomplete.")
    config = resolve_config(metadata["config"])
    if metadata["protocol"] != data_protocol(config) or metadata["normalization"] != NORMALIZATION:
        raise ValueError("Checkpoint class mapping / data protocol is inconsistent.")
    if (
        type(state["epoch"]) is not int
        or state["epoch"] < 1
        or not isinstance(state["history"], list)
        or len(state["history"]) != state["epoch"]
        or type(state["best_epoch"]) is not int
        or not 1 <= state["best_epoch"] <= state["epoch"]
        or not isinstance(state["best_score"], (int, float))
        or not math.isfinite(state["best_score"])
        or not 0 <= state["best_score"] <= 1
    ):
        raise ValueError("Checkpoint epoch/history/best-score state is invalid.")
    if metadata["selection_rule"] != SELECTION_RULE:
        raise ValueError("Checkpoint selection rule is incompatible.")
    scores = []
    for epoch, row in enumerate(state["history"], start=1):
        if (
            not isinstance(row, dict)
            or row.get("epoch") != epoch
            or not isinstance(row.get("val"), dict)
        ):
            raise ValueError("Checkpoint history must contain consecutive completed epochs.")
        score = row["val"].get("selection_score")
        if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Checkpoint history selection scores are invalid.")
        scores.append(score)
    if state["best_score"] != max(scores) or state["best_epoch"] != scores.index(max(scores)) + 1:
        raise ValueError("Checkpoint best score/epoch contradicts the declared tie rule.")
    current, best = state["model_state_dict"], state["best_weights"]
    if not isinstance(current, dict) or not isinstance(best, dict) or current.keys() != best.keys():
        raise ValueError("Checkpoint best weights are incomplete.")
    for key, value in current.items():
        if (
            not isinstance(value, torch.Tensor)
            or not isinstance(best[key], torch.Tensor)
            or value.shape != best[key].shape
            or value.dtype != best[key].dtype
        ):
            raise ValueError(f"Checkpoint best-weight shape/dtype mismatch: {key}")
    if (
        not isinstance(state["rng_state"], dict)
        or {"python", "numpy", "torch", "cuda", "loader"} - state["rng_state"].keys()
    ):
        raise ValueError("Checkpoint RNG state is incomplete.")


def fit_classifier(
    config: dict[str, Any],
    prepared: str | Path,
    run_directory: str | Path,
    device: torch.device = CPU_DEVICE,
    *,
    resume: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Run a fixed epoch budget; resume is explicitly epoch-boundary and total-target based.

    A portable last checkpoint embeds the best weights so resuming from a later epoch
    preserves the correct best model, including when continuing in a new directory.
    Only locally produced/trusted full checkpoints may be supplied.
    """
    config = resolve_config(config)
    options = config["training"]
    precision_context(device, options["precision"])
    if device.type == "cuda":
        torch.cuda.set_device(device)
    seed_everything(options["seed"])
    train = SemanticROIDataset(prepared, "train", config, augment=True)
    val = SemanticROIDataset(prepared, "val", config)
    train.verify_sources()
    if "unknown" in config["classes"]:
        unknown = config["classes"].index("unknown")
        if not any(unknown in (row["before_label"], row["after_label"]) for row in train.records):
            raise ValueError("A learned unknown class requires labeled training examples.")
    generator = torch.Generator().manual_seed(options["seed"])
    train_loader = DataLoader(
        train,
        batch_size=options["batch_size"],
        shuffle=True,
        num_workers=options["num_workers"],
        collate_fn=collate_rois,
        generator=generator,
    )
    val_loader = DataLoader(
        val,
        batch_size=options["batch_size"],
        collate_fn=collate_rois,
        num_workers=options["num_workers"],
        generator=torch.Generator().manual_seed(options["seed"] + 1),
    )
    model = TinySiameseChangeClassifier(config["classes"], config["dropout"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=options["learning_rate"], weight_decay=options["weight_decay"]
    )
    weights = (
        torch.tensor(options["class_weights"], device=device, dtype=torch.float32)
        if options["class_weights"] is not None
        else None
    )
    compatibility = deepcopy(config)
    compatibility["training"].pop("epochs")
    metadata = {
        "config": config,
        "compatibility_config": compatibility,
        "protocol": data_protocol(config),
        "normalization": NORMALIZATION,
        "data_identity": file_digest(Path(prepared)),
        "provenance": train.document["provenance"],
        "coverage": train.document["coverage"],
        "selection_rule": SELECTION_RULE,
        "parameters": count_model_parameters(model),
        "runtime": _runtime_metadata(device),
        "scheduler": None,
    }
    directory = Path(run_directory).resolve()
    history: list[dict[str, Any]] = []
    best_score, best_epoch, start = -1.0, 0, 0
    best_weights: dict[str, torch.Tensor] = {}
    state = None
    if resume is not None:
        state = load_training_checkpoint(resume)
        validate_checkpoint(state)
        if state.get("inference_only"):
            raise ValueError("Resume requires last.pt; best.pt is inference-only.")
        saved = state["metadata"]
        for key in ("compatibility_config", "protocol", "data_identity", "runtime"):
            if saved[key] != metadata[key]:
                raise ValueError(
                    f"Resume class mapping, architecture, or data/protocol mismatch: {key}"
                )
        start = state["epoch"]
        if options["epochs"] < start:
            raise ValueError("epochs is a total target and cannot be less than the saved epoch.")
        if directory.exists() and Path(resume).resolve().parent != directory:
            raise FileExistsError("Resume into an existing directory requires its own last.pt.")
        if directory.exists() and Path(resume).resolve() != directory / "last.pt":
            raise ValueError("In-place resume must use last.pt, not rewind an existing run.")
        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        history, best_score, best_epoch = state["history"], state["best_score"], state["best_epoch"]
        best_weights = state["best_weights"]
        restore_rng(state["rng_state"], train_loader)
    elif directory.exists():
        raise FileExistsError(f"Run already exists; use explicit resume: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "resolved_config.json", metadata)
    write_json(
        directory / "class_mapping.json",
        {"classes": config["classes"], "class_ids": config["class_ids"]},
    )
    # best.pt is an inference checkpoint containing the best epoch's actual model weights.
    if state is not None:
        state = {**state, "metadata": metadata}
        _save_checkpoint(directory / "last.pt", state)
        _save_checkpoint(
            directory / "best.pt",
            {**state, "model_state_dict": best_weights, "inference_only": True},
        )
        write_json(directory / "history.json", history)
        write_json(
            directory / "best_validation.json",
            {
                **history[best_epoch - 1]["val"],
                "epoch": best_epoch,
                "mask_source": config["mask_source"],
                "coverage": train.document["coverage"]["val"],
                "coverage_definition": train.document["coverage_definition"],
            },
        )
    for epoch in range(start + 1, options["epochs"] + 1):
        training = run_epoch(
            model,
            train_loader,
            device,
            weights=weights,
            precision=options["precision"],
            optimizer=optimizer,
        )
        validation = run_epoch(
            model, val_loader, device, weights=weights, precision=options["precision"]
        )
        history.append({"epoch": epoch, "train": training, "val": validation})
        improved = validation["selection_score"] > best_score
        if improved:
            best_score, best_epoch = validation["selection_score"], epoch
            best_weights = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        state = {
            "task": TASK,
            "architecture": ARCHITECTURE,
            "metadata": metadata,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "rng_state": capture_rng(train_loader),
            "history": history,
            "best_score": best_score,
            "best_epoch": best_epoch,
            "best_weights": best_weights,
            "scheduler_state_dict": None,
        }
        if improved:
            _save_checkpoint(
                directory / "best.pt",
                {**state, "model_state_dict": best_weights, "inference_only": True},
            )
            write_json(
                directory / "best_validation.json",
                {
                    **validation,
                    "epoch": epoch,
                    "mask_source": config["mask_source"],
                    "coverage": train.document["coverage"]["val"],
                    "coverage_definition": train.document["coverage_definition"],
                },
            )
        _save_checkpoint(directory / "last.pt", state)
        write_json(directory / "history.json", history)
    return history


def summarize_runs(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Report mean/sample standard deviation across seeds without choosing a best test seed."""
    if len(results) < 2 or len({row["seed"] for row in results}) != len(results):
        raise ValueError("Supply at least two independently seeded evaluation results.")
    for key in ("classes", "protocol", "data_identity", "split"):
        if any(row[key] != results[0][key] for row in results[1:]):
            raise ValueError(f"Run summaries require the same {key}.")
    if len({row["checkpoint_identity"] for row in results}) != len(results):
        raise ValueError("Each seed must come from a different trained checkpoint.")
    keys = (
        "before_accuracy",
        "after_accuracy",
        "before_macro_f1",
        "after_macro_f1",
        "joint_accuracy",
    )
    return {
        "seeds": [row["seed"] for row in results],
        "split": results[0]["split"],
        "classes": results[0]["classes"],
        "metrics": {
            key: {
                "mean": statistics.mean(row[key] for row in results),
                "sample_std": statistics.stdev(row[key] for row in results),
            }
            for key in keys
        },
    }
````

## Новый файл: `src/satchange/semantic/inference.py`

````python
"""Trusted classifier checkpoint evaluation and native-scale ROI inference."""

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from satchange.models import TinySiameseChangeClassifier
from satchange.semantic.data import SemanticROIDataset, collate_rois, file_digest
from satchange.semantic.roi import DEFAULT_ROI_SETTINGS, ROISettings, extract_rois
from satchange.semantic.training import run_epoch, validate_checkpoint
from satchange.training import load_training_checkpoint

CPU_DEVICE = torch.device("cpu")


def load_classifier(
    checkpoint: str | Path,
    device: torch.device = CPU_DEVICE,
) -> tuple[TinySiameseChangeClassifier, dict[str, Any]]:
    """Load only a locally produced/trusted semantic checkpoint, deliberately via CPU."""
    state = load_training_checkpoint(checkpoint)
    validate_checkpoint(state)
    config = state["metadata"]["config"]
    model = TinySiameseChangeClassifier(config["classes"], config["dropout"])
    model.load_state_dict(state["model_state_dict"], strict=True)
    return model.to(device).eval(), state


@torch.inference_mode()
def predict_rois(
    model: TinySiameseChangeClassifier,
    before: torch.Tensor,
    after: torch.Tensor,
    probabilities: torch.Tensor,
    settings: ROISettings = DEFAULT_ROI_SETTINGS,
) -> list[dict[str, Any]]:
    """Classify native RGB components; softmax scores are uncalibrated confidence scores.

    Return component/crop boxes in original coordinates. An empty change map returns [].
    No detector runs, annotation access, or region discovery happens in the network.
    """
    model.eval()
    device = next(model.parameters()).device
    result = []
    for roi in extract_rois(before, after, probabilities, settings):
        output = model(
            *(value.unsqueeze(0).to(device) for value in (roi.before, roi.after, roi.mask))
        )
        dates = []
        for key in ("before_logits", "after_logits"):
            score, index = output[key].float().softmax(dim=1)[0].max(dim=0)
            dates.append((model.classes[int(index)], float(score)))
        result.append(
            {
                "component_id": roi.component_id,
                "bbox": list(roi.component_bbox),
                "crop_bbox": list(roi.crop_bbox),
                "area": roi.area,
                "minimum_extent_limited": roi.minimum_extent_limited,
                "before_class": dates[0][0],
                "before_score": dates[0][1],
                "after_class": dates[1][0],
                "after_score": dates[1][1],
                "transition": f"{dates[0][0]} -> {dates[1][0]}",
            }
        )
    return result


def predict_checkpoint(
    checkpoint: str | Path,
    before: torch.Tensor,
    after: torch.Tensor,
    probabilities: torch.Tensor,
    device: torch.device = CPU_DEVICE,
) -> dict[str, Any]:
    """Use the saved class mapping, normalization, ROI settings, and declared mask source."""
    model, state = load_classifier(checkpoint, device)
    protocol = state["metadata"]["protocol"]
    if protocol["mask_source"] == "ground_truth":
        if not ((probabilities == 0) | (probabilities == 1)).all():
            raise ValueError("This oracle checkpoint requires binary ground_truth support.")
    predictions = predict_rois(model, before, after, probabilities, ROISettings(**protocol["roi"]))
    return {
        "classes": list(model.classes),
        "mask_source": protocol["mask_source"],
        "detector": protocol["detector"],
        "normalization": protocol["normalization"],
        "confidence_definition": "uncalibrated per-date softmax scores",
        "predictions": predictions,
    }


def evaluate_classifier(
    checkpoint: str | Path,
    prepared: str | Path,
    split: str,
    device: torch.device = CPU_DEVICE,
) -> dict[str, Any]:
    """Evaluate labeled ROI coverage; this is not an end-to-end detector accuracy measure."""
    if split not in ("val", "test"):
        raise ValueError("Evaluation split must be val or untouched test.")
    model, state = load_classifier(checkpoint, device)
    metadata = state["metadata"]
    config = metadata["config"]
    dataset = SemanticROIDataset(prepared, split, config)
    dataset.verify_sources()
    identity = file_digest(Path(prepared))
    if identity != metadata["data_identity"]:
        raise ValueError("Evaluation requires the declared prepared-data/split provenance.")
    loader = DataLoader(
        dataset, batch_size=config["training"]["batch_size"], collate_fn=collate_rois
    )
    weights = config["training"]["class_weights"]
    result = run_epoch(
        model,
        loader,
        device,
        weights=(
            torch.tensor(weights, device=device, dtype=torch.float32)
            if weights is not None
            else None
        ),
    )
    return {
        **result,
        "split": split,
        "seed": config["training"]["seed"],
        "mask_source": config["mask_source"],
        "protocol": metadata["protocol"],
        "data_identity": identity,
        "checkpoint_identity": file_digest(Path(checkpoint)),
        "coverage": dataset.document["coverage"][split],
        "coverage_definition": dataset.document["coverage_definition"],
        "evaluated_epoch": state["best_epoch"] if state.get("inference_only") else state["epoch"],
    }
````

## Новый файл: `configs/tiny_siamese_change_classifier.yaml`

````yaml
# EXAMPLE taxonomy only. Replace IDs/names with your real dataset's declared taxonomy.
# Background must explicitly be a learned class or listed in ignore_ids.
classes: [building, road, vegetation, water, bare_ground]
class_ids: {1: building, 2: road, 3: vegetation, 4: water, 5: bare_ground}
ignore_ids: [0, 255]
mask_source: ground_truth
detector: null
roi:
  threshold: 0.5
  connectivity: 8
  context_pixels: 16
  min_crop_size: 128
  min_component_area: 1
label_policy:
  min_transition_purity: 0.8
  min_valid_fraction: 0.8
dropout: 0.1
training:
  epochs: 50
  learning_rate: 0.001
  weight_decay: 0.0001
  batch_size: 4
  num_workers: 0
  precision: fp32
  seed: 42
  class_weights: null
````

## Новый файл: `tests/test_semantic.py`

````python
"""CPU-only synthetic implementation checks, not evidence of real semantic accuracy."""

import json
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch
import yaml
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
from satchange.semantic.inference import evaluate_classifier, load_classifier, predict_checkpoint
from satchange.semantic.roi import ROISettings, align_whole_image_probabilities, extract_rois
from satchange.semantic.training import (
    SemanticMetrics,
    fit_classifier,
    logical_forward,
    resolve_config,
    run_epoch,
    semantic_loss,
    summarize_runs,
    validate_checkpoint,
)
from satchange.training import load_training_checkpoint


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


def test_checkpoint_roundtrip_exact_resume_best_weights_and_inference(tmp_path):
    manifest, _ = make_manifest(tmp_path)
    cfg = config()
    prepare_dataset(cfg, manifest, tmp_path / "prepared")
    prepared = tmp_path / "prepared/prepared.json"
    full_history = fit_classifier(cfg, prepared, tmp_path / "full")
    one_epoch = deepcopy(cfg)
    one_epoch["training"]["epochs"] = 1
    fit_classifier(one_epoch, prepared, tmp_path / "staged")
    history = fit_classifier(cfg, prepared, tmp_path / "staged", resume=tmp_path / "staged/last.pt")
    assert full_history == history
    full = load_training_checkpoint(tmp_path / "full/last.pt")
    resumed = load_training_checkpoint(tmp_path / "staged/last.pt")
    for key, value in full["model_state_dict"].items():
        torch.testing.assert_close(value, resumed["model_state_dict"][key], rtol=0, atol=0)
    assert resumed["best_epoch"] == next(
        row["epoch"] for row in history if row["val"]["selection_score"] == resumed["best_score"]
    )
    model, best = load_classifier(tmp_path / "staged/best.pt")
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, resumed["best_weights"][key])
    assert best["inference_only"]
    metrics = evaluate_classifier(tmp_path / "staged/best.pt", prepared, "test")
    assert metrics["samples"] == 1 and metrics["mask_source"] == "ground_truth"
    before, after, probability = scene_tensors(17, 23)
    empty = predict_checkpoint(tmp_path / "staged/best.pt", before, after, probability)
    assert empty["predictions"] == []
    probability[0, 2:4, 3:5] = 1
    predictions = predict_checkpoint(tmp_path / "staged/best.pt", before, after, probability)
    assert predictions["predictions"][0]["bbox"] == [3, 2, 5, 4]
    assert " -> " in predictions["predictions"][0]["transition"]
    assert 0 <= predictions["predictions"][0]["before_score"] <= 1
    with pytest.raises(FileExistsError):
        fit_classifier(cfg, prepared, tmp_path / "staged")
    changed = deepcopy(cfg)
    changed["dropout"] = 0.3
    with pytest.raises(ValueError, match="mismatch"):
        fit_classifier(changed, prepared, tmp_path / "staged", resume=tmp_path / "staged/last.pt")
    changed = deepcopy(cfg)
    changed["classes"].reverse()
    with pytest.raises(ValueError, match="protocol"):
        fit_classifier(changed, prepared, tmp_path / "staged", resume=tmp_path / "staged/last.pt")
    fit_classifier(cfg, prepared, tmp_path / "portable", resume=tmp_path / "staged/last.pt")
    portable, _ = load_classifier(tmp_path / "portable/best.pt")
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, portable.state_dict()[key])


def test_synthetic_smoke_bounded_tiny_overfit():
    result = synthetic_smoke(60)
    assert result["evidence"] == "synthetic implementation check only"
    assert result["last_loss"] < result["first_loss"] * 0.1
    assert result["eval_joint_accuracy"] == 1


def test_seed_summary_and_cli_missing_data(tmp_path, monkeypatch):
    base = {
        "classes": ["a", "b"],
        "protocol": {},
        "data_identity": "data",
        "split": "test",
        "before_accuracy": 0.5,
        "after_accuracy": 0.5,
        "before_macro_f1": 0.5,
        "after_macro_f1": 0.5,
        "joint_accuracy": 0.5,
    }
    results = [
        {**base, "seed": seed, "checkpoint_identity": str(seed), "joint_accuracy": score}
        for seed, score in ((42, 0), (123, 0.5), (777, 1))
    ]
    summary = summarize_runs(results)
    assert summary["metrics"]["joint_accuracy"] == {"mean": 0.5, "sample_std": 0.5}
    with pytest.raises(ValueError, match="independently"):
        summarize_runs([results[0], results[0]])
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


@pytest.mark.parametrize(
    "failure", ["weights_length", "weights_zero", "seed", "precision", "epochs"]
)
def test_training_configuration_errors(failure):
    cfg = config()
    cfg["training"].update(
        {
            "weights_length": {"class_weights": [1]},
            "weights_zero": {"class_weights": [0, 1]},
            "seed": {"seed": 2**32},
            "precision": {"precision": "fp16"},
            "epochs": {"epochs": 0},
        }[failure]
    )
    with pytest.raises(ValueError):
        resolve_config(cfg)


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
        fit_classifier(cfg, tmp_path / "unknown/prepared.json", tmp_path / "run")
    cfg["roi"]["min_component_area"] = 100
    doc = prepare_dataset(cfg, manifest, tmp_path / "filtered")
    assert doc["coverage"]["train"]["retained"] == 0
    assert doc["coverage"]["train"]["exclusions"] == {"small_component": 3}
    with pytest.raises(ValueError, match="No eligible"):
        SemanticROIDataset(tmp_path / "filtered/prepared.json", "train", cfg)


def test_cli_preparation_training_resume_evaluation_and_inference(tmp_path, monkeypatch, capsys):
    manifest, _ = make_manifest(tmp_path)
    config_file = tmp_path / "config.yaml"
    cfg = config()
    cfg["training"]["class_weights"] = [1, 2]
    config_file.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    prepared_dir, run = tmp_path / "prepared", tmp_path / "cli_run"
    prepared = prepared_dir / "prepared.json"

    def invoke(arguments):
        monkeypatch.setattr("sys.argv", ["semantic", *map(str, arguments)])
        main()
        return json.loads(capsys.readouterr().out)

    report = invoke(["validate", "--config", config_file, "--manifest", manifest])
    assert report["coverage"]["train"]["retained"] == 3
    invoke(["prepare", "--config", config_file, "--manifest", manifest, "--output", prepared_dir])
    invoke(["train", "--config", config_file, "--prepared", prepared, "--run", run, "--epochs", 1])
    invoke(
        [
            "train",
            "--config",
            config_file,
            "--prepared",
            prepared,
            "--run",
            run,
            "--resume",
            run / "last.pt",
            "--epochs",
            2,
        ]
    )
    result = invoke(
        [
            "evaluate",
            "--checkpoint",
            run / "best.pt",
            "--prepared",
            prepared,
            "--split",
            "test",
            "--output",
            run / "test.json",
        ]
    )
    assert result["samples"] == 1
    prediction = invoke(
        [
            "infer",
            "--checkpoint",
            run / "best.pt",
            "--before",
            tmp_path / "scene_4_before.png",
            "--after",
            tmp_path / "scene_4_after.png",
            "--change",
            tmp_path / "scene_4_change.npy",
            "--output",
            run / "prediction.json",
        ]
    )
    assert len(prediction["predictions"]) == 1
    with pytest.raises(ValueError, match="inference-only"):
        fit_classifier(cfg, prepared, tmp_path / "bad_resume", resume=run / "best.pt")
    state = load_training_checkpoint(run / "last.pt")
    for key in ("architecture", "rng_state", "best_weights"):
        invalid = deepcopy(state)
        invalid.pop(key)
        with pytest.raises(ValueError, match="semantic classifier"):
            validate_checkpoint(invalid)
    invalid = deepcopy(state)
    invalid["metadata"]["config"]["classes"].reverse()
    with pytest.raises(ValueError, match="class mapping"):
        validate_checkpoint(invalid)
    for key, value in (("best_score", -1), ("best_epoch", 0)):
        invalid = deepcopy(state)
        invalid[key] = value
        with pytest.raises(ValueError, match="epoch/history"):
            validate_checkpoint(invalid)
    assert not (tmp_path / "bad_resume").exists()
````

## Новый файл: `docs/semantic_classifier.md`

````markdown
# TinySiameseChangeClassifier

This second-stage baseline classifies an already proposed ROI before and after a change.
It does not propose changes, produce segmentation masks, or retrain a first-stage detector.

## Reuse and scope

The implementation imports `load_config`, `seed_everything`, `precision_context`,
`capture_rng`/`restore_rng`, `RandomFlipRotate`, `count_model_parameters`,
`training._save_checkpoint`, and `load_training_checkpoint`. The new model is exported from
`satchange.models`, without registering it as a binary detector. New code lives in the model
file and `satchange.semantic`: native component crops, manifest supervision, multiclass
metrics, a short semantic training loop, inference, and a CLI. LEVIR-specific loaders,
BCE/Dice, binary metrics, and detector trainers have incompatible contracts and are not used.
No new production dependencies are needed: connected components use declared OpenCV.

## Tensor flow and trade-offs

Native RGB is float32 `[3,H,W]` in `[0,1]`, obtained from RGB uint8 files by division by 255.
The model receives images `[B,3,H,W]` and effective weights `[B,1,H,W]`.
It concatenates dates on the batch axis, producing `[2B,3,H,W]`, then runs one backbone:
three Conv3x3/BatchNorm/ReLU blocks with channels 32/64/128 and two MaxPool2d(2) layers.
The final feature grid is `[2B,128,floor(H/4),floor(W/4)]`: 128x128 becomes 32x32.
Only the weights are area-downsampled to the actual feature dimensions. FP32 weighted
pooling uses `sum(features * weights) / (sum(weights) + 1e-8)` and rejects empty support.
One shared `Linear(128,64)/ReLU/Dropout/Linear(64,C)` head produces raw logits for both dates.
The small CNN has local receptive-field context, not global scene understanding.

ROIs use `(x0,y0,x1,y1)` boxes in original-image coordinates, with exclusive upper bounds.
`extract_rois` receives aligned native `[3,H,W]` RGB and `[1,H,W]` probabilities. It thresholds
with `>=`, finds components using configurable 4/8 connectivity, sorts by first row-major pixel,
and expands into real context. Initial values are threshold 0.5, context 16 pixels, minimum
crop 128 pixels, minimum area one pixel, and 8 connectivity. They are starting settings,
not optimized claims. Crops shift inward at borders and retain entire large components.
Images below the minimum use available real pixels and record that limitation. H/W below
four is rejected. There is no RGB resizing, letterboxing, synthetic padding, or fixed-size
truncation. A 30x40 object stays 30x40. The saved effective weights are probability times
selected-component support; neighboring changed objects inside the context do not pool directly.

An already resized detector map may use
`satchange.semantic.roi.align_whole_image_probabilities(map, original_size, detector_input_size)`
only when its preprocessing was a documented whole-image resize; sizes are `(H,W)`.
Cropped/padded/tiled outputs need their own verified inverse geometry. The default APIs
reject spatial mismatches, nonfinite/out-of-range probabilities, and logits passed as probabilities.

The collate function returns a list. Differently sized pairs run individually; their logits
are concatenated into one logical batch before CE. Thus BatchNorm sees each temporal pair,
not the entire logical batch. Default logical batch size is four; one is supported.
Weighted CE is normalized over each date's full logical batch. For a singleton logical batch
the normalized class weight cancels. Split loss logs sum numerators/weight denominators
separately per date, handling final partial batches without averaging batch means.

## One explicit data contract

LEVIR-CD binary building-change masks cannot provide multiclass before/after labels.
Supply real aligned RGB images, integer semantic maps, and change maps. No dataset is
downloaded and no missing annotations are fabricated. The diagnostic archive is unnecessary
when this working source is available; checkpoint metadata alone is not a trained model.

The JSON manifest has a `scenes` list. Every scene supplies exactly these required fields
(file paths are relative to the manifest, or absolute):

```json
{
  "scenes": [
    {
      "sample_id": "unique_pair_id",
      "group_id": "geographic_or_source_scene_group",
      "split": "train",
      "before": "before.png",
      "after": "after.png",
      "before_semantic": "before_semantic.npy",
      "after_semantic": "after_semantic.npy",
      "change": "change.npy"
    }
  ]
}
```

RGB files must actually be RGB uint8 images. All three maps are non-pickled `.npy` arrays
of shape `[H,W]`, aligned to the original RGB. Semantic maps must have integer dtype;
palette images must be explicitly converted to integer IDs by the dataset owner first.
Ground-truth change maps contain exactly 0/1. Predicted change maps contain actual frozen-detector
probabilities in `[0,1]`, preserving soft weights. There is no automatic binarization of labels
or conversion of logits. Each manifest uses the single `mask_source` declared in its config.
Do not combine oracle and predicted map files in one experiment.

The YAML declares the ordered `classes`, raw-ID-to-name `class_ids`, and `ignore_ids`.
Every observed semantic ID must be declared. Background can be explicitly learned or ignored;
the example ignores 0 and 255. These are examples, not a claimed dataset taxonomy. Edit the
YAML mapping to match real annotations; no Python source edits are needed. Missing labels,
ignored pixels, a learned `unknown` class, and low softmax scores are different concepts.
`unknown` is allowed only with corresponding labeled training examples.

Split source groups before ROI generation and preserve official splits. Duplicate sample IDs
and any group spanning splits are rejected. For `predicted`, set, for example:

```yaml
mask_source: predicted
detector:
  identity: YOUR_FROZEN_DETECTOR_CHECKPOINT_HASH_OR_RUN_ID
  training_groups: [ACTUAL_FITTING_GROUP_ID]
```

This is declared provenance, not independent proof of how the detector was fitted.
Validation/test group overlap with that fitting list is rejected. The pipeline consumes saved
maps and never fits a detector. Generate maps without fitting either model on validation/test.

Scalar labels come from actual observed `(before_class,after_class)` pairs on the selected
component's valid pixels, not independent date majorities or the full context box.
Defaults require valid annotation on at least 80% of the component and a dominant transition
purity of at least 80% of valid pixels. Ties, insufficient labels, mixed transitions, and small
components are excluded with reasons. Same-class transitions remain valid. The same rules
apply to all splits. Preparation reports proposed/retained counts, retained fractions, exclusions,
and date/transition distributions. It saves paths, component/crop coordinates, quality metadata,
provenance hashes, and compressed per-ROI probability/support masks, not duplicate RGB imagery.

Predicted evaluation covers annotation-eligible proposed ROIs. It does not perform object matching
against a separate change-region ground truth. Coverage/exclusions accompany classification metrics;
missed regions and false positives still require separate accounting. Do not call selected-ROI
accuracy end-to-end system accuracy. Oracle ROI results do not establish detector-pipeline results.
A building-change detector does not automatically propose road/water/vegetation changes; the
classifier cannot classify changes that were never proposed.

## Training, selection, and artifacts

Starting settings are AdamW, 50 total epochs, LR 1e-3, weight decay 1e-4, dropout 0.1,
seed 42, FP32, and zero DataLoader workers. CUDA BF16 may be selected explicitly using the
existing precision helper, but must be supported by the selected hardware. An explicit positive
class-weight vector follows saved class order. There is no automatic weight estimation;
if providing calculated weights, compute them from training labels only and address zero-count
classes explicitly. No scheduler, early stopping, HPO, or multi-seed run launches automatically.

Loss is `CE(before_logits,before_label) + CE(after_logits,after_label)`.
Metrics are before/after accuracy and macro F1, joint accuracy, two global CxC confusion
matrices (rows true, columns predicted), and per-class precision/recall/F1/support.
Zero denominators yield zero; macro F1 includes every declared class, including absent ones.
Checkpoint `selection_score` is mean before/after validation macro F1; exact ties retain the
earlier epoch. Test never controls selection. Seeds 42, 123, and 777 can be run independently
using `--seed` and separate run names, then summarized with mean/sample standard deviation.

Run outputs contain `resolved_config.json` (including parameter count/runtime/provenance),
`class_mapping.json`, `history.json`, `best_validation.json`, `best.pt`, and `last.pt`.
Evaluation JSON contains metrics, raw confusion matrices, class supports, and ROI coverage.
No existing artifacts are replaced without an explicit resume. `last.pt` saves optimizer,
RNG and loader state, history, best score/epoch, and a portable copy of the best weights.
`best.pt` is for inference/evaluation; it is not a resumable checkpoint.
Resume rejects changed class mappings, architecture/settings, source data/protocol, or runtime.
`--epochs` always means a **total target**, e.g. resuming epoch 20 with `--epochs 50`
runs 30 additional epochs. Reproducible continuation is verified with `num_workers=0` on CPU;
no exact mid-epoch or multi-worker/CUDA continuation claim is made.
Only load full checkpoints from your own runs or another trusted source; existing loading uses
Python deserialization with `weights_only=False` and explicit CPU mapping.

## Copy-paste commands (PowerShell, repository root)

The local `.venv` used for implementation checks has Python 3.12 and CPU PyTorch.
Replace the `REPLACE_WITH...` paths below with actual annotations/images. The example taxonomy
must also match those annotations before real preparation. Paths remain configurable inputs.

```powershell
$python = '.\.venv\Scripts\python.exe'
$config = 'configs\tiny_siamese_change_classifier.yaml'
$manifest = 'REPLACE_WITH_REAL_SEMANTIC_MANIFEST.json'
$preparedDirectory = 'data\semantic_rois_oracle_v1'
$prepared = "$preparedDirectory\prepared.json"
$run = 'artifacts\semantic_classification\tiny_siamese_change_classifier\oracle_seed42_v1'

# Read-only validation and exclusion report; then prepare distinct ROI metadata/masks.
& $python -m satchange.semantic validate --config $config --manifest $manifest
& $python -m satchange.semantic prepare --config $config --manifest $manifest --output $preparedDirectory

# Synthetic only; no data download and no persisted model artifacts.
& $python -m satchange.semantic smoke --steps 60
& $python -m pytest tests/test_semantic.py

# Real training: run explicitly only when real semantic annotations exist.
& $python -m satchange.semantic train --config $config --prepared $prepared --run $run --seed 42 --device cpu

# Epoch-boundary resume; this is a total epoch target.
& $python -m satchange.semantic train --config $config --prepared $prepared --run $run --resume "$run\last.pt" --epochs 50 --device cpu

# Validation, then untouched final test evaluation. Output files must be new.
& $python -m satchange.semantic evaluate --checkpoint "$run\best.pt" --prepared $prepared --split val --output "$run\evaluation_val.json"
& $python -m satchange.semantic evaluate --checkpoint "$run\best.pt" --prepared $prepared --split test --output "$run\evaluation_test.json"

# This oracle config requires binary aligned native support. Predicted runs retain soft maps.
& $python -m satchange.semantic infer --checkpoint "$run\best.pt" --before 'REPLACE_WITH_BEFORE_RGB.png' --after 'REPLACE_WITH_AFTER_RGB.png' --change 'REPLACE_WITH_ALIGNED_NATIVE_CHANGE.npy' --output "$run\inference_scene.json"

# After three independent runs/evaluations; never cherry-pick the best test seed.
& $python -m satchange.semantic summarize --results 'REPLACE_WITH_SEED42_EVALUATION.json' 'REPLACE_WITH_SEED123_EVALUATION.json' 'REPLACE_WITH_SEED777_EVALUATION.json' --output 'artifacts\semantic_classification\three_seed_summary.json'
```

Python callers can use `prepare_dataset`, `SemanticROIDataset`, `fit_classifier`,
`evaluate_classifier`, `predict_checkpoint`, and lower-level `extract_rois`/`predict_rois`.
`predict_checkpoint` uses saved preprocessing/classes/settings. Inference runs in eval/inference
mode and reports original-image boxes, class names, transition strings, and separate uncalibrated
softmax scores. It invents neither a joint confidence probability nor an unknown-class detector.
An empty native change map returns an empty prediction list.

## Implementation verification (2026-10-01)

Added files: `src/satchange/models/tiny_siamese_change_classifier.py`,
`src/satchange/semantic/{__init__,__main__,roi,data,training,inference}.py`,
`configs/tiny_siamese_change_classifier.yaml`, `tests/test_semantic.py`, and this guide.
The only modified existing file is `src/satchange/models/__init__.py`, adding the public export.
The five-class example has 102,053 trainable parameters; the synthetic two-class check has 101,858.

Checks ran with an isolated Python 3.12.14 `.venv`, CPU PyTorch 2.14.1+cpu, NumPy 2.5.3,
and OpenCV 5.0.0. No production dependency declarations/versions were changed. The editable
installation used `--no-deps`; only the dependencies needed for these checks were installed.

| Executed command | Result |
| --- | --- |
| `.\.venv\Scripts\python -m pytest tests/test_semantic.py` | **53 passed**, 5.46 seconds |
| `.\.venv\Scripts\python -m satchange.semantic smoke --steps 60` | Passed: finite backward, loss 1.309796 to 0, toy eval joint accuracy 1.0 |
| `.\.venv\Scripts\python -m ruff check src/satchange/semantic src/satchange/models/tiny_siamese_change_classifier.py src/satchange/models/__init__.py tests/test_semantic.py` | **Passed** |
| `.\.venv\Scripts\python -m mypy --follow-imports=silent src/satchange/semantic src/satchange/models/tiny_siamese_change_classifier.py` | **Passed**, 7 source files; imported existing modules' errors suppressed for this focused check |
| `.\.venv\Scripts\python -m pytest` | **119 passed, 3 skipped, 1 failed**, 40.17 seconds |
| `.\.venv\Scripts\python -m ruff check . --output-format concise` | Failed: **147 issues in unchanged files**, including old notebooks and dormant modules |
| `.\.venv\Scripts\python -m mypy src` | Failed: **47 errors in 9 unchanged files**, 29 source files checked |

The full-suite failure is
`tests/test_execution.py::test_generic_yaml_training_launcher_and_collision_guard`:
it explicitly launches on CPU using the existing MiT YAML's BF16 setting, which the existing
precision helper rejects. Those files were left unchanged. Two GPU tests skipped because CUDA
and explicit GPU allocation were unavailable. The notebook execution check skipped because
`nbformat` was not installed; notebook execution was not verified.

The new tests cover native border/context crops, large/small/odd/rectangular extents, stable
component IDs, neighboring-component isolation, tiny-mask pooling, observed joint labels,
ignore/purity exclusions, split and detector-group leakage checks, weighted logical batches,
partial-batch logging, hand-calculated confusion/F1, trusted checkpoint round trips, tie handling,
exact CPU resume, best-weight portability, input/metadata rejection, and all main CLI operations
on temporary synthetic fixtures. Synthetic fitting/smoke results establish implementation behavior
only; real-data training/accuracy, GPU/BF16 behavior, and multi-worker resume were not verified.

The workspace has no `.git` directory, so a Git diff/revision was unavailable. Before/after
SHA256 comparisons confirmed the existing-file change scope and unchanged notebooks/configs.
No real dataset was downloaded/prepared, no real training or HPO was launched, and existing
experiment/model artifacts were not rewritten. Runtime checkpoints record a null Git revision
when unavailable and also hash the semantic source files.
````
