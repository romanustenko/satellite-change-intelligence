"""Explicit ROI preparation/evaluation/inference CLI; training uses the common HPO entry."""

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from satchange.execution import seed_everything
from satchange.models import TinySiameseChangeClassifier, count_model_parameters
from satchange.semantic.data import prepare_dataset, read_map, read_rgb, write_json
from satchange.semantic.inference import evaluate_classifier, predict_checkpoint
from satchange.semantic.training import semantic_loss
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
    inference.add_argument("--mask-source", choices=("ground_truth", "predicted"), required=True)
    inference.add_argument("--detector-identity")
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--steps", type=int, default=60)
    args = parser.parse_args()
    if args.command in ("evaluate", "infer"):
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
            inference_mask_source=args.mask_source,
            detector={"identity": args.detector_identity} if args.detector_identity else None,
        )
    if args.command != "smoke":
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
