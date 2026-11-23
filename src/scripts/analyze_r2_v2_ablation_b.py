"""Ablation B (H6) analysis on SECOND validation: SCD trained on tiles (A, case 1) vs on
ground-truth ROI windows (B), each applied to whole tiles, detector ROIs and oracle ROIs.

Inputs: per-tile confusion matrices written by ``eval_r2_v2_systems.py --split val`` for the case-1
checkpoints (``v9/eval/val``, seed 42) and for the ablation-B checkpoints (``ablation_b/eval/val``,
``R2_SCD_ROOT`` redirected). Paired tile bootstrap (one seed, 444 tiles) of SeK differences;
density strata with the frozen edges. Exploratory (D-059 / D-082): checkpoints and ROI policies
were also selected on validation; the test split is not read.

Usage: python src/scripts/analyze_r2_v2_ablation_b.py [--resamples 2000]
"""

import argparse
import json

import numpy as np
import r2_v2_systems_lib as lib

from satchange.semantic.analysis import assign_strata, metrics_from_histogram

OUT = lib.ROOT / "artifacts/research2/v2/ablation_b"
SEED = 42
INPUTS = {"tile": "single", "detector ROI (case 2)": "case2__frontier0.5",
          "detector ROI (case 3)": "case3__frontier0.5",
          "detector ROI (case 4)": "case4__frontier0.5", "oracle ROI": "oracle__balanced"}


def sek(hist: np.ndarray, index: np.ndarray | None = None) -> float:
    pooled = hist.sum(0) if index is None else hist[index].sum(0)
    return float(metrics_from_histogram(pooled)["sek"])


def diff(first: np.ndarray, second: np.ndarray, draws: np.ndarray) -> dict:
    """SeK(first) − SeK(second) with a paired 95 % bootstrap interval and a one-sided p (≤ 0)."""
    values = np.array([sek(first, idx) - sek(second, idx) for idx in draws])
    return {"diff": sek(first) - sek(second),
            "ci95": [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))],
            "p_not_better": float(np.mean(values <= 0))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args()
    frozen = json.loads((lib.V9 / "analysis_test.json").read_text())
    delta, edges = frozen["delta"], frozen["density_edges"]
    rng = np.random.default_rng(0)
    result: dict = {"split": "val", "seed": SEED, "resamples": args.resamples, "delta": delta,
                    "density_edges": edges, "models": {}}
    for model in lib.SCD_MODELS:
        a = np.load(lib.V9 / "eval/val" / f"{model}_s{SEED}.npz")
        b = np.load(OUT / "eval/val" / f"{model}_s{SEED}.npz")
        if not np.array_equal(a["ids"], b["ids"]):
            raise ValueError(f"{model}: tile order differs between A and B")
        tiles = a["ids"].size
        draws = rng.integers(0, tiles, (args.resamples, tiles))
        hist = {(train, name): src[f"hist__{key}"]
                for train, src in (("A", a), ("B", b)) for name, key in INPUTS.items()}
        table = {f"{train} · {name}": sek(h) for (train, name), h in hist.items()}
        two_a, two_b = hist[("A", "detector ROI (case 2)")], hist[("B", "detector ROI (case 2)")]
        single_a = hist[("A", "tile")]
        comparisons = {
            "H6: B two-stage − A two-stage (case 2)": diff(two_b, two_a, draws),
            "B two-stage − A single-stage": diff(two_b, single_a, draws),
            "A two-stage − A single-stage (reference)": diff(two_a, single_a, draws),
            "B on tiles − A on tiles": diff(hist[("B", "tile")], single_a, draws),
            "B oracle ROI − A oracle ROI": diff(hist[("B", "oracle ROI")],
                                                hist[("A", "oracle ROI")], draws),
        }
        for case in ("case 3", "case 4"):
            name = f"detector ROI ({case})"
            comparisons[f"B − A, {case}"] = diff(hist[("B", name)], hist[("A", name)], draws)
        comparisons["B two-stage − A single-stage"]["non_inferior"] = bool(
            comparisons["B two-stage − A single-stage"]["ci95"][0] > -delta)
        strata = assign_strata(a["density"], edges)
        by_density = []
        for stratum in range(len(edges) + 1):
            members = np.flatnonzero(strata == stratum)
            sub = members[rng.integers(0, members.size, (args.resamples, members.size))]
            by_density.append({"tiles": int(members.size),
                               "B − A two-stage": diff(two_b, two_a, sub),
                               "B two-stage − A single": diff(two_b, single_a, sub)})
        result["models"][model] = {"sek": table, "comparisons": comparisons,
                                   "by_density": by_density}
        print(model, json.dumps({k: round(v["diff"], 4) for k, v in comparisons.items()}),
              flush=True)
    path = OUT / "analysis_val.json"
    path.write_text(json.dumps(result, indent=2))
    print("written", path.relative_to(lib.ROOT), flush=True)


if __name__ == "__main__":
    main()
