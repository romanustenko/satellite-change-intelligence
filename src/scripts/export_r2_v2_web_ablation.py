"""Export the ablation-H6 summary for the web demo's expert view (numbers copied, never typed).

Reads ``results/research2/ablation_h6/analysis_val.json`` (validation, seed 42; written by
``analyze_r2_v2_ablation_b.py``) and writes a small JSON with, per SCD model, the SeK of every
training-data x input combination, the H6 comparison, the comparison with the single-stage system,
the oracle-ROI comparison and the density strata.

Usage: python src/scripts/export_r2_v2_web_ablation.py --out <site>/public/data/ablation_h6.json
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/research2/ablation_h6/analysis_val.json"
MODELS = ("siam_unet_resnet18", "scannet", "mamba_scd")
INPUTS = {"tile": "tile", "case2": "detector ROI (case 2)", "oracle": "oracle ROI"}
COMPARISONS = {
    "h6": "H6: B two-stage − A two-stage (case 2)",
    "vs_single": "B two-stage − A single-stage",
    "oracle": "B oracle ROI − A oracle ROI",
}


def diff(entry: dict) -> dict:
    return {"diff": entry["diff"], "ci95": entry["ci95"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(SOURCE.read_text())
    models = {}
    for name in MODELS:
        entry = source["models"][name]
        comparisons = {key: diff(entry["comparisons"][label])
                       for key, label in COMPARISONS.items()}
        comparisons["vs_single"]["non_inferior"] = entry["comparisons"][
            COMPARISONS["vs_single"]]["non_inferior"]
        models[name] = {
            "sek": {train: {key: entry["sek"][f"{train} · {label}"]
                            for key, label in INPUTS.items()} for train in ("A", "B")},
            "comparisons": comparisons,
            "strata": [{"tiles": s["tiles"], "h6": diff(s["B − A two-stage"])}
                       for s in entry["by_density"]],
        }
    result = {"split": "SECOND validation", "tiles": 444, "seed": source["seed"],
              "delta": source["delta"], "density_edges": source["density_edges"],
              "resamples": source["resamples"], "models": models}
    if sum(s["tiles"] for s in models[MODELS[0]]["strata"]) != result["tiles"]:
        raise ValueError("strata do not add up to the validation split")
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print("written", args.out, flush=True)


if __name__ == "__main__":
    main()
