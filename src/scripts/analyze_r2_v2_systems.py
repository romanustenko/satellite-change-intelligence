"""Analysis of the 12 systems from the per-tile histograms of ``eval_r2_v2_systems.py`` (V7/V8/V9).

Everything is computed from saved outputs (no model, no data read). Statistics follow the protocol
(§8): paired scene bootstrap nested with seeds - one tile resample is shared by all seeds and all
systems, so any difference of seed-mean metrics is paired; compute ratios use the same resamples.
Primary comparisons (frontier0.5): P1-P3 case 2 vs case 1 (non-inferiority with delta + compute),
P4-P6 case 2 vs case 3, P7-P9 case 4 vs case 2; Holm adjustment over the nine (one-sided bootstrap
p-values: P(diff <= -delta) for non-inferiority, P(diff <= 0) for superiority). Everything else is
exploratory. Strata are terciles of the changed-pixel fraction with edges taken from validation.

Usage: python src/scripts/analyze_r2_v2_systems.py --split val [--delta 0.005] [--resamples 2000]
"""

import argparse
import json

import numpy as np
import r2_v2_systems_lib as lib

from satchange.semantic.analysis import assign_strata, metrics_from_histogram, quantile_edges

PRIMARY_POLICY = "frontier0.5"
POLICIES = ("balanced", "frontier0.7", "frontier0.5", "frontier0.3")
CASES = ("case2", "case3", "case4")
BOOT_SEED = 0


def load(split: str, models: tuple[str, ...]) -> dict:
    """data[model] = {key: array [seeds, tiles, ...]}; keys as in the evaluator (field__system)."""
    data: dict = {}
    folder = lib.V9 / "eval" / split
    for model in models:
        files = [np.load(folder / f"{model}_s{seed}.npz") for seed in lib.SEEDS]
        ids = files[0]["ids"]
        if any(not np.array_equal(f["ids"], ids) for f in files):
            raise ValueError(f"{model}: tile order differs between seeds")
        keys = [k for k in files[0].files if k not in ("ids", "density")]
        data[model] = {k: np.stack([f[k] for f in files]) for k in keys}
        data[model]["density"] = files[0]["density"]
        data[model]["ids"] = ids
    return data


def seed_mean_metric(hist: np.ndarray, index: np.ndarray, metric: str) -> float:
    """Mean over seeds of the pooled metric of the resampled tiles ``index`` (hist [S,N,7,7])."""
    return float(np.mean([metrics_from_histogram(h[index].sum(0))[metric] for h in hist]))


def bootstrap(
    hists: dict[str, np.ndarray], draws: np.ndarray, metric: str = "sek"
) -> dict[str, np.ndarray]:
    return {name: np.array([seed_mean_metric(h, idx, metric) for idx in draws])
            for name, h in hists.items()}


def interval(values: np.ndarray, level: float = 0.95) -> list[float]:
    tail = (1 - level) / 2 * 100
    return [float(np.percentile(values, tail)), float(np.percentile(values, 100 - tail))]


def holm(p_values: dict[str, float]) -> dict[str, float]:
    order = sorted(p_values, key=lambda k: p_values[k])
    adjusted, running = {}, 0.0
    for rank, name in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * p_values[name]))
        adjusted[name] = running
    return adjusted


def compute_tile_flops(data: dict, flops: dict, model: str, key: str) -> np.ndarray:
    """Inference FLOPs per tile [seeds, tiles] of a system (SCD windows + detector forward)."""
    scd = flops["scd"][model]
    if key == "single":
        return np.full(data[model]["hist__single"].shape[:2], float(scd["512"]))
    case = key.split("__")[0]
    buckets = data[model][f"buckets__{key}"]  # [S, N, 4]
    per_bucket = np.array([scd[str(b)] for b in lib.BUCKETS], dtype=float)
    detector = 0.0
    if case in ("case2", "case3"):
        detector = flops["detectors"]["medium_ft"]["forward_flops"]
    elif case == "case4":
        partner = json.loads((lib.V9 / "case4_partner.json").read_text())["partner"]
        detector = (flops["detectors"]["medium_ft"]["forward_flops"]
                    + flops["detectors"][partner]["forward_flops"])
    elif case == "oracle":
        detector = 0.0
    return buckets @ per_bucket + detector


def ratio_boot(numerator: np.ndarray, denominator: np.ndarray, draws: np.ndarray) -> np.ndarray:
    num, den = numerator.mean(0), denominator.mean(0)  # seed-mean per tile
    return np.array([num[idx].sum() / den[idx].sum() for idx in draws])


def compare(
    entry: dict, boot: dict[str, np.ndarray], name: str, a: str, b: str, margin: float | None,
    superiority: bool,
) -> None:
    diff = boot[a] - boot[b]
    observed = entry["systems"][a]["sek"]["mean"] - entry["systems"][b]["sek"]["mean"]
    comparison = {"a": a, "b": b, "diff_sek": float(observed), "ci95": interval(diff)}
    if superiority:
        comparison["p_one_sided"] = float(np.mean(diff <= 0))
    elif margin is not None:
        comparison["p_one_sided"] = float(np.mean(diff <= -margin))
        comparison["non_inferior"] = bool(interval(diff)[0] > -margin)
    entry["comparisons"][name] = comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--delta", type=float, help="non-inferiority margin (SeK); optional on val")
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--edges", type=float, nargs=2, help="frozen density tercile edges")
    parser.add_argument("--models", nargs="+", default=list(lib.SCD_MODELS))
    args = parser.parse_args()
    models = tuple(args.models)
    data = load(args.split, models)
    flops = json.loads((lib.V9 / "inference_flops.json").read_text())
    tiles = len(data[models[0]]["density"])
    rng = np.random.default_rng(BOOT_SEED)
    draws = rng.integers(0, tiles, (args.resamples, tiles))
    keys = ["single"] + [f"{c}__{p}" for c in CASES for p in POLICIES] + ["oracle__balanced"]
    result: dict = {"split": args.split, "tiles": tiles, "resamples": args.resamples,
                    "delta": args.delta, "primary_policy": PRIMARY_POLICY, "models": {}}
    latency_path = lib.V9 / f"latency_{args.split}.json"
    latency = json.loads(latency_path.read_text()) if latency_path.exists() else {}
    seed_sd = {}
    for model in models:
        hist = {k: data[model][f"hist__{k}"] for k in keys}
        boot = bootstrap(hist, draws)
        tile_flops = {k: compute_tile_flops(data, flops, model, k) for k in keys}
        entry: dict = {"systems": {}, "comparisons": {}}
        for key in keys:
            per_seed = [metrics_from_histogram(h.sum(0)) for h in hist[key]]
            entry["systems"][key] = {
                **{m: {"mean": float(np.mean([p[m] for p in per_seed])),
                       "sd": float(np.std([p[m] for p in per_seed], ddof=1))}
                   for m in ("sek", "miou", "fscd")},
                "ci95_sek": interval(boot[key]),
                "crops_per_tile": float(data[model][f"crops__{key}"].mean()),
                "box_area_ratio": float(data[model][f"box_area__{key}"].mean() / lib.TILE ** 2),
                "union_area_ratio": float(data[model][f"union_area__{key}"].mean() / lib.TILE ** 2),
                "window_area_ratio": float(
                    data[model][f"window_area__{key}"].mean() / lib.TILE ** 2),
                "flops_per_tile": float(tile_flops[key].mean()),
                "flops_ratio_vs_single": float(
                    tile_flops[key].mean() / tile_flops["single"].mean()),
                "flops_ratio_ci95": interval(ratio_boot(tile_flops[key], tile_flops["single"],
                                                        draws)),
                "latency_ms": latency.get(model, {}).get(key),
            }
        seed_sd[model] = entry["systems"]["single"]["sek"]["sd"]

        for case in CASES:
            for policy in POLICIES:
                compare(entry, boot, f"{case}__{policy}_vs_single", f"{case}__{policy}", "single",
                        args.delta, False)
        for policy in POLICIES:
            compare(entry, boot, f"case2_vs_case3__{policy}", f"case2__{policy}",
                    f"case3__{policy}", None, True)
            compare(entry, boot, f"case4_vs_case2__{policy}", f"case4__{policy}",
                    f"case2__{policy}", None, True)
        entry["strata"] = {}
        if args.edges:
            edges = list(args.edges)
        else:
            edges = quantile_edges(data[model]["density"], 3)
            result["edges_note"] = "terciles of the analysed split (val: freeze these edges)"
        result["density_edges"] = edges
        stratum = assign_strata(data[model]["density"], edges)
        for number in range(3):
            subset = np.flatnonzero(stratum == number)
            sub_draws = subset[rng.integers(0, subset.size, (args.resamples, subset.size))]
            sub = bootstrap({k: hist[k] for k in ("single", f"case2__{PRIMARY_POLICY}",
                                                  f"case3__{PRIMARY_POLICY}",
                                                  f"case4__{PRIMARY_POLICY}")}, sub_draws)
            entry["strata"][str(number)] = {
                "tiles": int(subset.size),
                **{f"{c}_minus_single": {
                    "diff": float(np.mean(sub[f"{c}__{PRIMARY_POLICY}"] - sub["single"])),
                    "ci95": interval(sub[f"{c}__{PRIMARY_POLICY}"] - sub["single"])}
                   for c in CASES},
            }
        result["models"][model] = entry
    result["single_stage_seed_sd_sek"] = seed_sd
    result["delta_rule_value"] = max(
        0.005, float(np.ceil(2 * np.mean(list(seed_sd.values())) * 1000) / 1000))
    primary: dict[str, float] = {}
    for model in models:
        comps = result["models"][model]["comparisons"]
        primary[f"P1-3 {model} case2 vs single"] = comps[
            f"case2__{PRIMARY_POLICY}_vs_single"].get("p_one_sided", float("nan"))
        primary[f"P4-6 {model} case2 vs case3"] = comps[
            f"case2_vs_case3__{PRIMARY_POLICY}"]["p_one_sided"]
        primary[f"P7-9 {model} case4 vs case2"] = comps[
            f"case4_vs_case2__{PRIMARY_POLICY}"]["p_one_sided"]
    result["primary_holm"] = {"raw_p": primary,
                              "adjusted_p": holm({k: v for k, v in primary.items() if v == v})}
    out = lib.V9 / f"analysis_{args.split}.json"
    out.write_text(json.dumps(result, indent=2))
    print("wrote", out, "delta rule value", result["delta_rule_value"], flush=True)
    lines = [f"# 12 systems, split {args.split}, {tiles} tiles\n",
             "| model | system | SeK | mIoU | Fscd | dSeK vs single (CI95) | area | crops | "
             "FLOPs / single (CI95) |", "|---|---|---|---|---|---|---|---|---|"]
    for model in models:
        entry = result["models"][model]
        for key in keys:
            s = entry["systems"][key]
            comp = entry["comparisons"].get(f"{key}_vs_single")
            delta = (f"{comp['diff_sek']:+.4f} ({comp['ci95'][0]:+.4f}…{comp['ci95'][1]:+.4f})"
                     if comp else "—")
            lines.append(
                f"| {model} | {key} | {s['sek']['mean']:.4f} ± {s['sek']['sd']:.4f} | "
                f"{s['miou']['mean']:.4f} | {s['fscd']['mean']:.4f} | {delta} | "
                f"{s['union_area_ratio']:.2f} | {s['crops_per_tile']:.1f} | "
                f"{s['flops_ratio_vs_single']:.2f} ({s['flops_ratio_ci95'][0]:.2f}…"
                f"{s['flops_ratio_ci95'][1]:.2f}) |")
    (lib.V9 / f"analysis_{args.split}.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
