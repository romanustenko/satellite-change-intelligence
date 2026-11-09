"""Research 2 system B on the SECOND *validation* split: detector -> ROI crops -> WHAT -> scene SeK.

Loads a frozen WHERE detector (LEVIR-initialized Light/Medium checkpoint of ``where_hpo``) and a
trained WHAT model (``run_r2_scd.py --stage what``), caches the detector probability maps once, and
grid-searches the ROI policy (threshold / context / minimum component area) by validation SeK
(``semantic.roi_policy``). Also reports the oracle-ROI upper bound and stage-1 coverage
diagnostics. The test split is not accepted by this script: the confirmatory test pass is a
separate script wired to the freeze manifest (R2-P11-T06). Result: ``<out>.json``.

    python src/scripts/eval_r2_pipeline.py --gpu 1 --tier light --seed 42 --out <json>
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from satchange.data.levir_cd import create_dataset_dataloader
from satchange.data.second_semantic import SECONDSemanticDataset
from satchange.execution import resolve_device
from satchange.models.factory import SEMANTIC_TASK, create_task_model
from satchange.run import reconstruct_checkpoint
from satchange.semantic.inference import detector_probability_map
from satchange.semantic.roi import DEFAULT_ROI_SETTINGS
from satchange.semantic.roi_policy import coverage_diagnostics, select_roi_policy
from satchange.semantic.what_stage import evaluate_what_pipeline
from satchange.training import _best_weights

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "SECOND"
WHERE = ROOT / "artifacts/research2/where_hpo"
SCD = ROOT / "artifacts/research2/scd"
WHERE_SUMMARY = {
    "light": (WHERE / "r2_where_light_v2/final_training_summary.json", "run_name"),
    "medium": (WHERE / "r2_where_medium_v2/seed_validation_summary.json", "trial"),
}


def detector_checkpoint(tier: str, seed: int) -> tuple[Path, float]:
    """Best checkpoint and its validation-selected threshold of the final WHERE run of ``seed``."""
    summary, key = WHERE_SUMMARY[tier]
    rows = [r for r in json.loads(summary.read_text())["results"] if r[key] == f"seed_{seed}"]
    if len(rows) != 1 or rows[0]["status"] != "completed":
        raise ValueError(f"Expected exactly one completed {tier} WHERE run for seed {seed}.")
    return Path(rows[0]["best_checkpoint"]), float(rows[0]["best_threshold"])


def load_what(run: str, model_name: str, device: torch.device) -> torch.nn.Module:
    last = SCD / model_name / run / "last.pt"
    state = torch.load(last, map_location="cpu", weights_only=False)
    config = state["config"]
    model = create_task_model(model_name, config["model_parameters"], SEMANTIC_TASK)
    model.load_state_dict(_best_weights(state, last))  # best epoch, never the last
    return model.to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--tier", choices=tuple(WHERE_SUMMARY), required=True)
    parser.add_argument("--seed", type=int, default=42, help="WHERE seed")
    parser.add_argument("--what-run", help="run name under scd/<model>; default: same seed")
    parser.add_argument("--model", default="fc_siam_semantic")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    parser.add_argument("--contexts", type=int, nargs="+", default=[16, 48])
    parser.add_argument("--min-areas", type=int, nargs="+", default=[1, 100, 400])
    parser.add_argument("--limit", type=int, default=0, help="debug: first N scenes only")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)

    device = resolve_device(f"cuda:{args.gpu}")
    what_run = args.what_run or f"what_sem_lr0.001_e60_s{args.seed}_v1"
    detector_path, detector_threshold = detector_checkpoint(args.tier, args.seed)
    detector, _ = reconstruct_checkpoint(detector_path, device)
    what = load_what(what_run, args.model, device)
    dataset, _ = create_dataset_dataloader(DATA, "val", None, 1, 0, semantic_labels=True)
    assert isinstance(dataset, SECONDSemanticDataset)
    count = args.limit or len(dataset)
    started = time.perf_counter()
    probabilities: dict[str, torch.Tensor] = {}
    for index in range(count):
        sample = dataset[index]
        probabilities[sample["sample_id"]] = detector_probability_map(
            detector, sample["before"], sample["after"], input_size=(512, 512),
            output_kind="logits",
        )
    del detector
    torch.cuda.empty_cache()
    subset = _Subset(dataset, count)

    def evaluate(settings: Any) -> dict[str, float]:
        return evaluate_what_pipeline(
            what, subset, device,  # type: ignore[arg-type]
            probability_source=lambda sample: probabilities[sample["sample_id"]],
            settings=settings,
        )

    policy = select_roi_policy(
        evaluate, thresholds=args.thresholds, context_pixels=args.contexts,
        min_component_areas=args.min_areas,
    )
    best_settings = type(DEFAULT_ROI_SETTINGS)(**policy["best"]["settings"])
    diagnostics = [
        coverage_diagnostics(probabilities[subset.sample_ids[i]], subset[i]["mask"], best_settings)
        for i in range(count)
    ]
    summary = {
        key: sum(row[key] for row in diagnostics) / count for key in diagnostics[0]
    }
    result = {
        "split": "val", "scenes": count, "tier": args.tier, "where_seed": args.seed,
        "where_checkpoint": str(detector_path), "where_threshold_f1": detector_threshold,
        "what_run": what_run, "policy": policy,
        "oracle": evaluate_what_pipeline(what, subset, device),  # type: ignore[arg-type]
        "coverage_mean_per_scene": summary,
        "wall_hours": (time.perf_counter() - started) / 3600,
        "finished_utc": time.strftime("%FT%TZ", time.gmtime()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ("tier", "where_seed", "wall_hours")}
                     | {"best": policy["best"]}))


class _Subset:
    """First ``count`` scenes of a dataset (debug limit) with the attributes the evaluator uses."""

    def __init__(self, dataset: SECONDSemanticDataset, count: int) -> None:
        self.dataset, self.count = dataset, count
        self.sample_ids = dataset.sample_ids[:count]

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.dataset[index]


if __name__ == "__main__":
    main()
