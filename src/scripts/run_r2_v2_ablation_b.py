"""Research 2 ablation B (H6, D-059 / D-082): train one SCD model on ground-truth ROI windows.

Same recipe as ``run_r2_v2_scd.py`` (ImageNet init, AdamW, 5 % warm-up + cosine per step, the
model's own loss, fp32 / TF32 off, HPO-winner lr and weight decay, training-FLOP budget) — only the
training data differ: instead of whole 512² tiles the model sees the windows the two-stage
inference path (``batch_ctx``) would cut around **ground-truth** change boxes:

* boxes = case-2 ``frontier0.5`` ROI settings applied to the binary change mask of each tile;
* each box → the smallest 128/256/384/512 square window containing it, placed as in
  ``bench_r2_v2_two_stage_speedups.scene("batch_ctx")``; labels of the whole window are used.

Batches hold windows of one size only, with the pixel count of 8 tiles (8 × 512², 384-windows:
14 × 384² = 0.98 of it), so a step processes about as many pixels as a case-1 step. The FLOP budget
is turned into a step count with the mean counted step FLOPs of the epoch's bucket mix (SCanNet:
teacher passes charged, scaled by pixels). Selection: validation SeK on ground-truth windows of the
444 validation tiles (same geometry, no augmentation). Train and validation splits only.

Usage: python src/scripts/run_r2_v2_ablation_b.py --model scannet --flop-budget 3.2e17 \
    --lr 2.5e-4 --weight-decay 0.01 --seed 42 --device cuda:0
"""

import argparse
import functools
import json
import math
import multiprocessing
import os
import time
from typing import Any

import numpy as np
import r2_v2_systems_lib as lib
import run_r2_v2_scd as scd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from satchange.data.levir_cd import create_dataset_dataloader
from satchange.data.normalization import normalize_image
from satchange.data.transforms import RandomFlipRotate
from satchange.execution import resolve_device, seed_everything
from satchange.lr_schedule import WARMUP_FRACTION, warmup_cosine_factor
from satchange.semantic.compute import count_training_step_flops, steps_for_budget
from satchange.semantic.scd_losses import (
    SCanNetPseudoLabeler,
    create_scd_loss,
    pseudo_label_budget_steps,
    pseudo_label_start_step,
)
from satchange.training import fit_binary_change_model

GROUP = "ablation_b"
CACHE = lib.ROOT / "artifacts/research2/v2/ablation_b/tile_cache"
POLICY = lib.policies_for("case2")["frontier0.5"]
TILE = lib.TILE
BUCKET_BATCH = {512: 8, 384: 14, 256: 32, 128: 128}


def windows_of(mask: np.ndarray) -> list[tuple[int, int, int]]:
    """``batch_ctx`` windows (x, y, side) around the ground-truth boxes of one tile."""
    detections = lib.rs.detections_by_threshold(mask.astype(np.float32), (POLICY.threshold,))
    boxes = lib.rs.generate_boxes(detections, POLICY, TILE, TILE)
    out = []
    for x0, y0, x1, y1 in boxes.tolist():
        side = next(b for b in lib.BUCKETS if b >= max(x1 - x0, y1 - y0))
        wx = min(max((x0 + x1) // 2 - side // 2, 0), TILE - side)
        wy = min(max((y0 + y1) // 2 - side // 2, 0), TILE - side)
        out.append((wx, wy, side))
    return out


_DATASETS: dict[str, Any] = {}


def _load_tile(args: tuple[str, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(images uint8 [2,3,H,W], labels int8 [2,H,W], change mask bool [H,W]) of one tile."""
    split, index = args
    torch.set_num_threads(1)
    if split not in _DATASETS:
        _DATASETS[split], _ = create_dataset_dataloader(lib.DATA, split, None, 1, 0,
                                                        semantic_labels=True)
    sample = _DATASETS[split][index]
    images = np.stack([torch.round(sample[k] * 255).to(torch.uint8).numpy()
                       for k in ("before", "after")])
    labels = np.stack([sample[k].numpy().astype(np.int8) for k in ("before_label", "after_label")])
    return images, labels, sample["mask"][0].numpy() > 0.5


class TileStore:
    """All tiles of a split in memory as uint8 (images) and int8 (labels); cached as one .npz."""

    def __init__(self, split: str, workers: int = 16) -> None:
        cache = CACHE / f"{split}.npz"
        if cache.is_file():
            data = np.load(cache)
            self.images, self.labels, masks = data["images"], data["labels"], data["masks"]
            self.ids = data["ids"].tolist()
        else:
            dataset, _ = create_dataset_dataloader(lib.DATA, split, None, 1, 0,
                                                   semantic_labels=True)
            self.ids = list(dataset.sample_ids)
            with multiprocessing.get_context("fork").Pool(workers) as pool:
                rows = pool.map(_load_tile, [(split, i) for i in range(len(dataset))], chunksize=8)
            self.images = np.stack([r[0] for r in rows])
            self.labels = np.stack([r[1] for r in rows])
            masks = np.stack([r[2] for r in rows])
            CACHE.mkdir(parents=True, exist_ok=True)
            temporary = cache.with_name(f"{cache.stem}.{os.getpid()}.tmp.npz")
            np.savez(temporary, images=self.images, labels=self.labels, masks=masks,
                     ids=np.array(self.ids))
            os.replace(temporary, cache)
        self.windows: list[tuple[int, int, int, int]] = []
        for index, mask in enumerate(masks):
            self.windows.extend((index, *w) for w in windows_of(mask))


class WindowDataset(Dataset):
    """One ground-truth window per item, in the sample format of the SECOND semantic loader."""

    def __init__(self, store: TileStore, transform: RandomFlipRotate | None) -> None:
        self.store, self.transform = store, transform

    def __len__(self) -> int:
        return len(self.store.windows)

    def __getitem__(self, item: int) -> dict[str, Any]:
        tile, wx, wy, side = self.store.windows[item]
        rows, cols = slice(wy, wy + side), slice(wx, wx + side)
        images = torch.from_numpy(self.store.images[tile, :, :, rows, cols].copy()).float() / 255
        labels = torch.from_numpy(self.store.labels[tile, :, rows, cols].astype(np.int64))
        sample = {
            "before": normalize_image(images[0], "imagenet"),
            "after": normalize_image(images[1], "imagenet"),
            "mask": ((labels[0] > 0) | (labels[1] > 0)).float()[None],
            "before_label": labels[0],
            "after_label": labels[1],
            "sample_id": f"{self.store.ids[tile]}@{wx},{wy},{side}",
        }
        return self.transform(sample) if self.transform is not None else sample


class BucketBatchSampler(Sampler[list[int]]):
    """Batches of equal-size windows; reshuffled per epoch from ``seed + epoch`` when training."""

    def __init__(self, store: TileStore, *, seed: int, train: bool) -> None:
        self.seed, self.train, self.epoch = seed, train, 0
        self.groups: dict[int, list[int]] = {}
        for item, (_, _, _, side) in enumerate(store.windows):
            self.groups.setdefault(side, []).append(item)

    def batches_per_bucket(self) -> dict[int, int]:
        size = {side: len(items) / BUCKET_BATCH[side] for side, items in self.groups.items()}
        return {side: math.floor(n) if self.train else math.ceil(n) for side, n in size.items()}

    def __len__(self) -> int:
        return sum(self.batches_per_bucket().values())

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        batches = []
        for side in sorted(self.groups):
            items = np.array(self.groups[side])
            if self.train:
                items = rng.permutation(items)
            size = BUCKET_BATCH[side]
            for start in range(0, self.batches_per_bucket()[side] * size, size):
                chunk = items[start:start + size]
                if len(chunk):
                    batches.append(chunk.tolist())
        order = rng.permutation(len(batches)) if self.train else range(len(batches))
        return iter([batches[i] for i in order])


def bucket_batch(dataset: WindowDataset, sampler: BucketBatchSampler, side: int) -> dict:
    """One full batch of windows of ``side`` (for FLOP counting)."""
    items = sampler.groups[side][:BUCKET_BATCH[side]]
    samples = [dataset[i] for i in items]
    return {k: torch.stack([s[k] for s in samples]) for k in ("before", "after", "before_label",
                                                                 "after_label")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(scd.MODELS))
    parser.add_argument("--flop-budget", type=float, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", required=True)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=0, help="smoke test: cap the step count")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    name = f"{lib.SHORT[args.model]}_final_s{args.seed}"  # layout read by eval via R2_SCD_ROOT
    group = GROUP if not args.max_steps else GROUP + "_smoke"
    resume_from = scd.RUNS / group / args.model / name / "last.pt" if args.resume else None
    if resume_from is not None and not resume_from.is_file():
        raise FileNotFoundError(f"nothing to resume: {resume_from}")

    device = resolve_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed_everything(args.seed)
    started = time.perf_counter()
    train_store, val_store = TileStore("train"), TileStore("val")
    print(f"windows: train {len(train_store.windows)}, val {len(val_store.windows)} "
          f"({time.perf_counter() - started:.0f}s)", flush=True)
    transform = RandomFlipRotate(extra_keys=("before_label", "after_label"))
    train_set = WindowDataset(train_store, transform)
    train_sampler = BucketBatchSampler(train_store, seed=args.seed, train=True)
    val_set = WindowDataset(val_store, None)
    val_sampler = BucketBatchSampler(val_store, seed=args.seed, train=False)
    train_loader = DataLoader(train_set, batch_sampler=train_sampler, num_workers=args.num_workers,
                              pin_memory=True, persistent_workers=args.num_workers > 0)
    validation_loader = DataLoader(val_set, batch_sampler=val_sampler,
                                   num_workers=args.num_workers, pin_memory=True)

    model, encoder_report = scd.build_model(args.model, device)
    loss_function = create_scd_loss(args.model).to(device)
    per_bucket = train_sampler.batches_per_bucket()
    bucket_flops = {}
    for side in per_bucket:
        batch = bucket_batch(train_set, train_sampler, side)
        labels = batch["before_label"].to(device), batch["after_label"].to(device)
        bucket_flops[side] = count_training_step_flops(
            model, [batch["before"].to(device), batch["after"].to(device)],
            lambda out, labels=labels: loss_function(out, *labels)).total
    step_flops = sum(per_bucket[s] * bucket_flops[s] for s in per_bucket) / sum(per_bucket.values())
    labeler: SCanNetPseudoLabeler | None = None
    teacher_step_flops = 0.0
    if args.model == "scannet":
        batch = bucket_batch(train_set, train_sampler, max(per_bucket))
        probe = SCanNetPseudoLabeler(start_step=1)
        side = max(per_bucket)
        tile_pixels = BUCKET_BATCH[side] * side ** 2
        per_pixel = probe.count_flops(model, batch["before"].to(device)[:1],
                                      batch["after"].to(device)[:1]) / side ** 2
        teacher_step_flops = per_pixel * tile_pixels * (step_flops / bucket_flops[side])
        steps = pseudo_label_budget_steps(args.flop_budget, step_flops=step_flops,
                                          teacher_step_flops=teacher_step_flops)
        labeler = SCanNetPseudoLabeler(start_step=pseudo_label_start_step(steps))
    else:
        steps = steps_for_budget(args.flop_budget, round(step_flops))
    if args.max_steps:
        steps = min(steps, args.max_steps)
        if labeler is not None:
            labeler = SCanNetPseudoLabeler(start_step=pseudo_label_start_step(steps))
    epochs = math.ceil(steps / len(train_loader))
    torch.cuda.empty_cache()
    print(json.dumps({"bucket_batches_per_epoch": per_bucket, "bucket_step_flops": bucket_flops,
                      "mean_step_flops": step_flops, "teacher_step_flops": teacher_step_flops,
                      "steps": steps, "epochs": epochs}), flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    config = {
        "task": scd.SEMANTIC_TASK,
        "model_parameters": scd.MODELS[args.model]["parameters"],
        "recipe": {
            "training_data": "ground-truth ROI windows (case-2 frontier0.5 boxes, batch_ctx "
                             "windows 128/256/384/512), whole-window labels",
            "batch": {str(k): v for k, v in BUCKET_BATCH.items()},
            "input_normalization": "imagenet", "optimizer": "AdamW", "lr": args.lr,
            "weight_decay": args.weight_decay,
            "schedule": f"warmup {WARMUP_FRACTION} + cosine per step",
            "precision": "fp32 (TF32 off)", "semantic_argmax": scd.SEMANTIC_ARGMAX,
            "augmentation": "flips + rot90 (train only)",
            "selection": "validation SeK on ground-truth windows",
        },
        "roi_settings": lib.settings_to_json(POLICY),
        "flop_budget": args.flop_budget, "optimizer_steps": steps, "seed": args.seed,
        "windows": {"train": len(train_store.windows), "val": len(val_store.windows)},
        "bucket_batches_per_epoch": {str(k): v for k, v in per_bucket.items()},
        "bucket_step_flops": {str(k): v for k, v in bucket_flops.items()},
        "mean_step_flops": step_flops, "teacher_step_flops": teacher_step_flops,
        "pseudo_label_start_step": None if labeler is None else labeler.start_step,
        "encoder_init": {k: encoder_report[k] for k in ("path", "sha256") if k in encoder_report},
    }
    history = fit_binary_change_model(
        model=model, train_loader=train_loader, validation_loader=validation_loader,
        loss_function=loss_function, optimizer=optimizer, device=device,
        number_of_epochs=epochs, thresholds=[], output_root=scd.RUNS / group,
        model_name=args.model, run_name=name, config=config, task=scd.SEMANTIC_TASK,
        max_optimizer_steps=steps, semantic_argmax=scd.SEMANTIC_ARGMAX, pseudo_labeler=labeler,
        lr_factor=functools.partial(warmup_cosine_factor, total_steps=steps),
        resume_from=resume_from,
    )
    best = max(history, key=lambda row: row["selection_score"])
    last = history[-1]
    start = labeler.start_step if labeler is not None else steps
    flops = last["optimizer_steps"] * step_flops + max(0, last["optimizer_steps"] - start) * (
        teacher_step_flops)
    record = {
        "group": group, "name": name, "model": args.model, "seed": args.seed, "lr": args.lr,
        "weight_decay": args.weight_decay, "flop_budget": args.flop_budget,
        "optimizer_steps": last["optimizer_steps"], "epochs_run": len(history),
        "samples_seen": last["samples_seen"], "training_flops_estimate": flops,
        "step_flops": step_flops, "best_epoch": best["epoch"], "best_sek_windows": best["sek"],
        "best_miou_windows": best["miou"], "best_fscd_windows": best["fscd"],
        "curve": [{"epoch": r["epoch"], "steps": r["optimizer_steps"], "sek": r["sek"],
                   "miou": r["miou"], "fscd": r["fscd"], "train_loss": r["train_loss"]}
                  for r in history],
        "wall_seconds": time.perf_counter() - started,
        "gpu": torch.cuda.get_device_name(device), "device": args.device,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "git_revision": scd._git("rev-parse", "HEAD"),
        "git_dirty": bool(scd._git("status", "--porcelain", "--", "src", "tests")),
        "finished_utc": time.strftime("%FT%TZ", time.gmtime()),
    }
    run_dir = scd.RUNS / group / args.model / name
    (run_dir / "run_record.json").write_text(json.dumps({**record, "config": config}, indent=2))
    if not args.max_steps:
        with scd.LEDGER.open("a") as handle:
            handle.write(json.dumps({k: v for k, v in record.items() if k != "curve"}) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "curve"}), flush=True)


if __name__ == "__main__":
    main()
