"""Build the Research 2 freeze manifest (D-010, R2-P11-T05) for systems A and B.

Everything is selected beforehand on validation: the WHERE tier and ROI policy come from
``eval_r2_pipeline.py`` output (``--evidence``), the density strata and the non-inferiority margin
from the validation analysis. Checkpoints are the per-seed ``best.pt`` files (best validation
epoch). The manifest records their SHA-256, the git revision (clean tree required), the protocol
hash and the analysis parameters; it never reads any dataset. Refuses to overwrite.

    python src/scripts/freeze_r2.py --tier medium --evidence <eval json> --delta 0.01 \
        --density-edges 0.1 0.2 --protocol docs/research2/EXPERIMENTAL_PROTOCOL.md \
        --output artifacts/research2/freeze/manifest_v1.json
"""

import argparse
import json
from pathlib import Path

from eval_r2_pipeline import SCD, detector_checkpoint  # same directory (script dir on sys.path)

from satchange.semantic.freeze import build_freeze_manifest

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (42, 123, 777)
A_RUN = "a_sem_lr0.001_e60_s{seed}_v1"
WHAT_RUN = "what_sem_lr0.001_e60_s{seed}_v1"
MODEL = "fc_siam_semantic"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=("light", "medium"), required=True)
    parser.add_argument("--evidence", type=Path, required=True,
                        help="validation output of eval_r2_pipeline.py that chose the ROI policy")
    parser.add_argument("--delta", type=float, required=True, help="non-inferiority margin (SeK)")
    parser.add_argument("--density-edges", type=float, nargs="+", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args()
    if args.delta <= 0 or sorted(args.density_edges) != args.density_edges:
        raise ValueError("delta must be positive and density edges ascending.")

    evidence = json.loads(args.evidence.read_text())
    if evidence["split"] != "val" or evidence["tier"] != args.tier:
        raise ValueError("The evidence must be a validation evaluation of the same tier.")
    roi_settings = evidence["policy"]["best"]["settings"]

    a_paths = {str(s): SCD / MODEL / A_RUN.format(seed=s) / "best.pt" for s in SEEDS}
    what_paths = {str(s): SCD / MODEL / WHAT_RUN.format(seed=s) / "best.pt" for s in SEEDS}
    detector_paths = {str(s): detector_checkpoint(args.tier, s)[0] for s in SEEDS}
    systems = {
        "A": {
            "checkpoints": list(a_paths.values()), "selection_evidence": args.evidence,
            "config": {"checkpoint_by_seed": {k: str(v) for k, v in a_paths.items()},
                       "model": MODEL, "decision": "D-040"},
        },
        "B": {
            "checkpoints": [*detector_paths.values(), *what_paths.values()],
            "selection_evidence": args.evidence,
            "config": {
                "tier": args.tier, "roi_settings": roi_settings,
                "detector_by_seed": {k: str(v) for k, v in detector_paths.items()},
                "what_by_seed": {k: str(v) for k, v in what_paths.items()},
                "detector_input_size": 512, "detector_output_kind": "logits",
                "analysis": {"delta": args.delta, "density_edges": args.density_edges,
                             "resamples": args.resamples, "bootstrap_seed": 0},
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_freeze_manifest(
        systems, root=ROOT, protocol=args.protocol, output=args.output  # type: ignore[arg-type]
    )
    print(json.dumps({"output": str(args.output), "git_revision": manifest["git_revision"]}))


if __name__ == "__main__":
    main()
