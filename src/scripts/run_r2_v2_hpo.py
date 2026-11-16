"""Hyper-parameter search for one SCD model under a fixed per-trial FLOP budget (R2-V6-T01, D-072).

Every trial is one run of ``run_r2_v2_scd.py`` (the shared trainer with the step budget, per-model
loss,
ledger row), so the search uses exactly the training code of the final runs. The trial list is drawn
once
from a fixed seed and is **identical for all three models** (same space, same size, same budget):
learning
rate in [1e-4, 1e-3] (log scale, one draw per sixth of the range), weight decay from {0.001, 0.01,
0.05}
(each used equally often), seed 42. The winner is the trial
with the best validation SeK (best epoch). A finished trial (``run_record.json`` present) is never
rerun; an
interrupted one is restarted from scratch under the same name after its directory is moved aside by
hand.
Validation only; the SECOND test split is not read.

Usage: python src/scripts/run_r2_v2_hpo.py --model scannet --device cuda:2 [--trials 6] [--budget
8e16]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "src/scripts/run_r2_v2_scd.py"
RUNS = ROOT / "artifacts/research2/v2/runs"
GROUP = "hpo"
WEIGHT_DECAYS = (0.001, 0.01, 0.05)
SEARCH_SEED = 20261116
SHORT = {"siam_unet_resnet18": "unet", "scannet": "scannet", "mamba_scd": "mamba"}


def draw_trials(count: int) -> list[dict[str, float]]:
    """Fixed trial list shared by all models: stratified log-uniform lr, balanced weight decay."""
    rng = np.random.default_rng(SEARCH_SEED)
    strata = rng.permutation(count)
    learning_rates = 10 ** (-4.0 + (strata + rng.uniform(0.0, 1.0, count)) / count)
    decays = rng.permutation(np.resize(np.array(WEIGHT_DECAYS), count))
    return [
        {"lr": float(f"{lr:.2g}"), "weight_decay": float(wd)}
        for lr, wd in zip(learning_rates, decays, strict=True)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(SHORT))
    parser.add_argument("--device", required=True)
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--budget", type=float, default=8e16, help="training FLOPs per trial")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    trials = draw_trials(args.trials)
    folder = RUNS / GROUP / args.model
    folder.mkdir(parents=True, exist_ok=True)
    definition = {
        "model": args.model,
        "budget_flops_per_trial": args.budget,
        "seed": args.seed,
        "search_seed": SEARCH_SEED,
        "trials": trials,
    }
    definition_path = folder / "search.json"
    if definition_path.exists() and json.loads(definition_path.read_text()) != definition:
        raise ValueError(f"{definition_path} holds a different search definition; use a new group")
    definition_path.write_text(json.dumps(definition, indent=2))

    results = []
    for index, trial in enumerate(trials):
        name = f"{SHORT[args.model]}_hpo_t{index}_lr{trial['lr']:g}_wd{trial['weight_decay']:g}"
        record_path = folder / name / "run_record.json"
        if record_path.exists():
            print(f"trial {index}: {name} already finished, skipped", flush=True)
        else:
            command = [
                sys.executable, str(RUNNER), "--model", args.model, "--flop-budget",
                str(args.budget), "--lr", str(trial["lr"]), "--weight-decay",
                str(trial["weight_decay"]), "--seed", str(args.seed), "--device", args.device,
                "--group", GROUP, "--name", name, "--num-workers", str(args.num_workers),
            ]
            print(f"trial {index}: {' '.join(command)}", flush=True)
            subprocess.run(command, check=True, cwd=ROOT)
        record = json.loads(record_path.read_text())
        results.append(
            {
                "trial": index, "name": name, **trial,
                "best_sek": record["best_sek"], "best_epoch": record["best_epoch"],
                "final_sek": record["final_sek"], "optimizer_steps": record["optimizer_steps"],
                "training_flops": record["training_flops"], "wall_seconds": record["wall_seconds"],
            }
        )
    winner = max(results, key=lambda row: row["best_sek"])
    (folder / "search_result.json").write_text(
        json.dumps({**definition, "results": results, "selected": winner}, indent=2)
    )
    print("selected", json.dumps(winner), flush=True)


if __name__ == "__main__":
    main()
