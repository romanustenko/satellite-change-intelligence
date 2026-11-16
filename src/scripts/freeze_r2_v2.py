"""Freeze manifest v2 for the 12 systems (R2-V7-T02, D-010, D-074).

Collects everything chosen on validation: the nine SCD checkpoints (final runs, best-epoch
``best.pt``), the detector checkpoints of the three variants (case 2 ``medium_ft``, case 3
``medium_scratch``, case-4 partner) for the three seeds, the case-4 partner decision, the ROI
policies, the non-inferiority margin delta and the density-stratum edges. The values are bundled
into one evidence file (hashed by the manifest) together with the hashes of the HPO results; the
protocol file hash is recorded too. Refuses a dirty ``src/``/``tests`` tree and never overwrites.
It reads no dataset. After this the test split may be read once by
``eval_r2_v2_systems.py --split test --manifest <output>``.

Usage: python src/scripts/freeze_r2_v2.py --delta 0.005 --edges 0.12 0.31 \
    --protocol docs/research2/EXPERIMENTAL_PROTOCOL.md \
    --output artifacts/research2/v2/v9/freeze_manifest_v2.json
"""

import argparse
import json
from pathlib import Path

import r2_v2_systems_lib as lib

from satchange.semantic.freeze import build_freeze_manifest, sha256_file


def detector_checkpoint(variant: str, seed: int) -> Path:
    group, detector, template, _ = lib.DETECTOR_RUNS[variant]
    return lib.RUNS / group / detector / template.format(seed=seed) / "best.pt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delta", type=float, required=True)
    parser.add_argument("--edges", type=float, nargs=2, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.delta <= 0 or args.edges[0] >= args.edges[1]:
        raise ValueError("delta must be positive and the two density edges ascending")
    partner_file = lib.V9 / "case4_partner.json"
    policy_file = lib.V9 / "roi_policies.json"
    flops_file = lib.V9 / "inference_flops.json"
    partner = lib.partner_name()
    hpo_results = {m: lib.RUNS / "hpo" / m / "search_result.json" for m in lib.SCD_MODELS}
    evidence = {
        "delta_sek": args.delta, "density_edges": list(args.edges),
        "case4_partner": partner, "case4_partner_file": str(partner_file),
        "case4_partner_sha256": sha256_file(partner_file),
        "roi_policies_file": str(policy_file), "roi_policies_sha256": sha256_file(policy_file),
        "inference_flops_sha256": sha256_file(flops_file),
        "hpo_results": {m: {"path": str(p), "sha256": sha256_file(p)}
                        for m, p in hpo_results.items()},
        "inference_path": "batch_ctx", "primary_policy": "frontier0.5",
        "seed_pairing": "two-stage system of SCD seed s uses detector seed s",
    }
    evidence_path = lib.V9 / "freeze_evidence.json"
    if evidence_path.exists():
        raise FileExistsError(f"{evidence_path} exists; the freeze evidence is written once")
    evidence_path.write_text(json.dumps(evidence, indent=2))
    scd = {m: [lib.scd_run_dir(m, s) / "best.pt" for s in lib.SEEDS] for m in lib.SCD_MODELS}
    detectors = {
        "case2": [detector_checkpoint("medium_ft", s) for s in lib.SEEDS],
        "case3": [detector_checkpoint("medium_scratch", s) for s in lib.SEEDS],
        "case4": [detector_checkpoint("medium_ft", s) for s in lib.SEEDS]
        + [detector_checkpoint(partner, s) for s in lib.SEEDS],
    }
    policies = json.loads(policy_file.read_text())
    systems = {}
    for model in lib.SCD_MODELS:
        systems[f"case1_{model}"] = {
            "checkpoints": scd[model], "selection_evidence": evidence_path,
            "config": {"model": model, "stage": "single"},
        }
        for case, files in detectors.items():
            systems[f"{case}_{model}"] = {
                "checkpoints": [*scd[model], *files], "selection_evidence": evidence_path,
                "config": {"model": model, "stage": "two", "case": case,
                           "roi_policies": policies[case], "inference_path": "batch_ctx"},
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_freeze_manifest(systems, root=lib.ROOT, protocol=args.protocol,
                                     output=args.output)
    print(json.dumps({"output": str(args.output), "git_revision": manifest["git_revision"],
                      "systems": len(manifest["systems"])}))


if __name__ == "__main__":
    main()
