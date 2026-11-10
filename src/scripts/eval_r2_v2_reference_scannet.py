"""R2-V4-T05 (SCanNet half): published SCanNet SECOND checkpoint through our re-implementation,
on OUR validation split. Fidelity reference only (the checkpoint trained on 361 of our 444
validation tiles and was selected on the authors' own evaluation split; D-047). Never reads the
test split. Inputs use the authors' per-date SECOND statistics (their ``RS_ST.py``), not ImageNet.

Usage: python src/scripts/eval_r2_v2_reference_scannet.py --device cuda:0 --output <json>
"""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from satchange.data.second_semantic import SECONDSemanticDataset
from satchange.execution import resolve_device
from satchange.models.scannet import SCanNet
from satchange.semantic.scd_metrics import SemanticChangeMetrics
from satchange.semantic.scd_outputs import maps_from_change_and_semantic
from satchange.semantic.second import SECOND_COLORS

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/SECOND"
REFERENCE = ROOT / "artifacts/research2/v2/reference/scannet"
CHECKPOINT = REFERENCE / "SCanNet_32e_mIoU73.37_Sek23.94_Fscd63.66_OA87.86.pth"
CHECKPOINT_SHA256 = "05e2b17bfa984525af0075c7fd27dc97ee55848e7d36bd08aa9e72640072a7d2"
# SCanNet label ids by RGB colour (ST_COLORMAP in their datasets/RS_ST.py); ours differ.
SCANNET_COLORS = {
    0: (255, 255, 255), 1: (0, 0, 255), 2: (128, 128, 128), 3: (0, 128, 0), 4: (0, 255, 0),
    5: (128, 0, 0), 6: (255, 0, 0),
}
# Per-date normalization statistics (0-255 scale) of the authors' loader.
MEAN = {"before": (113.40, 114.08, 116.45), "after": (111.07, 114.04, 118.18)}
STD = {"before": (48.30, 46.27, 48.14), "after": (49.41, 47.01, 47.94)}


def normalize(image: torch.Tensor, date: str) -> torch.Tensor:
    """[0,1] RGB batch → the authors' per-date standardized input."""
    mean = torch.tensor(MEAN[date]).view(1, 3, 1, 1)
    std = torch.tensor(STD[date]).view(1, 3, 1, 1)
    return (image * 255.0 - mean) / std


def scannet_channels_in_our_order() -> list[int]:
    """Index list: our class id j is the published model's channel ``result[j]``."""
    ours = {colour: our_id for colour, our_id in SECOND_COLORS.items()}
    theirs = {ours[colour]: their_id for their_id, colour in SCANNET_COLORS.items()}
    if sorted(theirs) != list(range(7)):
        raise ValueError("Class colour tables do not form a permutation.")
    return [theirs[our_id] for our_id in range(7)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--semantic-argmax", choices=("all", "land_cover"), default="all")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if _sha256(CHECKPOINT) != CHECKPOINT_SHA256:
        raise ValueError("Reference checkpoint hash mismatch.")
    device = resolve_device(args.device)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model = SCanNet()
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=True))
    model.to(device).eval()
    dataset = SECONDSemanticDataset(DATA, "val", None)
    loader = DataLoader(dataset, batch_size=4, num_workers=4)
    order = scannet_channels_in_our_order()
    metrics = {rule: SemanticChangeMetrics() for rule in ("all", "land_cover")}
    started = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            change, first, second = model(normalize(batch["before"], "before").to(device),
                                          normalize(batch["after"], "after").to(device))
            first, second = first[:, order], second[:, order]
            for rule, accumulator in metrics.items():
                pred_1, pred_2 = maps_from_change_and_semantic(
                    change, first, second, semantic_argmax=rule  # type: ignore[arg-type]
                )
                accumulator.update(pred_1.cpu(), batch["before_label"])
                accumulator.update(pred_2.cpu(), batch["after_label"])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=ROOT, check=False).stdout.strip()
    record = {
        "task": "R2-V4-T05 (SCanNet half)",
        "label": "OPTIMISTIC fidelity reference: trained on 361/444 val tiles, own-split selected",
        "split": "val", "samples": len(dataset),
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)), "checkpoint_sha256": CHECKPOINT_SHA256,
        "input_normalization": "authors per-date SECOND statistics", "tf32": False,
        "precision": "fp32", "class_channel_order_from_scannet": order,
        "metrics": {rule: m.compute() for rule, m in metrics.items()},
        "primary_rule_as_published_code": "all",
        "wall_seconds": seconds, "device": torch.cuda.get_device_name(device)
        if device.type == "cuda" else "cpu",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "git_revision": revision, "torch": torch.__version__, "cuda": torch.version.cuda,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    with temporary.open("w") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, args.output)
    computed = {rule: m.compute() for rule, m in metrics.items()}
    print(json.dumps({rule: {k: round(v, 4) for k, v in m.items() if k in ("sek", "miou", "fscd")}
                      for rule, m in computed.items()}), flush=True)


if __name__ == "__main__":
    main()
