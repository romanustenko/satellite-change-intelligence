"""Statistics for Research 2 A-vs-B comparisons from saved per-scene results (R2-P12-T01).

Reads only ``scene_results.npz`` / ``run.json`` written by ``eval_r2_frozen.py`` (never a dataset)
and the analysis parameters frozen in the manifest (``systems.B.config.analysis``: ``delta``,
``density_edges``, optional ``resamples``/``bootstrap_seed``). Optional ``--latency`` is the JSON of
``bench_r2_latency.py``. Writes ``analysis.json`` next to the results (refuses to overwrite).

    python src/scripts/analyze_r2_frozen.py --results <dir> --manifest <m.json> [--latency <json>]
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from satchange.semantic.analysis import (
    assign_strata,
    error_decomposition,
    latency_ratio_by_stratum,
    metrics_from_histogram,
    non_inferior,
    paired_bootstrap_seeds,
)

METRICS = ("sek", "miou", "fscd")


def per_seed(hists: np.ndarray) -> dict[str, dict[str, Any]]:
    """Pooled metrics of each seed plus mean and sample SD over seeds."""
    rows = [metrics_from_histogram(h.sum(0)) for h in hists]
    out: dict[str, dict[str, Any]] = {}
    for metric in METRICS:
        values = np.array([row[metric] for row in rows])
        out[metric] = {
            "per_seed": [float(v) for v in values], "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--latency", type=Path)
    args = parser.parse_args()
    out_path = args.results / "analysis.json"
    if out_path.exists():
        raise FileExistsError(out_path)
    frozen = json.loads(args.manifest.read_text())["systems"]["B"]["config"]["analysis"]
    delta, edges = float(frozen["delta"]), [float(e) for e in frozen["density_edges"]]
    resamples, boot_seed = int(frozen.get("resamples", 2000)), int(frozen.get("bootstrap_seed", 0))

    data = np.load(args.results / "scene_results.npz")
    a, b, oracle = data["hist_a"], data["hist_b"], data["hist_oracle"]
    density = data["density"]
    strata = assign_strata(density, edges)
    groups: dict[str, np.ndarray | None] = {"all": None}
    groups.update({f"stratum_{k}": np.flatnonzero(strata == k) for k in range(len(edges) + 1)})

    comparisons: dict[str, Any] = {}
    for name, subset in groups.items():
        row: dict[str, Any] = {"scenes": int(density.size if subset is None else subset.size)}
        for metric in METRICS:
            row[f"B_minus_A_{metric}"] = paired_bootstrap_seeds(
                b, a, metric=metric, resamples=resamples, seed=boot_seed, subset=subset
            )
        row["B_minus_A_sek"]["non_inferior"] = non_inferior(row["B_minus_A_sek"], delta)
        row["oracle_minus_B_sek"] = paired_bootstrap_seeds(
            oracle, b, metric="sek", resamples=resamples, seed=boot_seed, subset=subset
        )
        comparisons[name] = row

    inside, outside = data["hist_b_inside"].sum((0, 1)), data["hist_b_outside"].sum((0, 1))
    result: dict[str, Any] = {
        "split": json.loads((args.results / "run.json").read_text())["split"],
        "delta": delta, "density_edges": edges, "seeds": data["seeds"].tolist(),
        "systems": {"A": per_seed(a), "B": per_seed(b), "B_oracle": per_seed(oracle)},
        "comparisons": comparisons,
        "error_decomposition_hc": error_decomposition(inside, outside),
    }
    if args.latency:
        latency = json.loads(args.latency.read_text())
        index = {s: i for i, s in enumerate(data["sample_ids"].tolist())}
        positions = np.array([index[s] for s in latency["sample_ids"]])
        result["latency"] = latency_ratio_by_stratum(
            np.array(latency["latency_b_ms"]), np.array(latency["latency_a_ms"]),
            density[positions], edges, resamples=resamples, seed=boot_seed,
        )
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps({"A_sek": result["systems"]["A"]["sek"]["mean"],
                      "B_sek": result["systems"]["B"]["sek"]["mean"],
                      "B_minus_A": comparisons["all"]["B_minus_A_sek"]}))


if __name__ == "__main__":
    main()
