"""Validation-only pipeline watcher for the 12 systems (R2-V9, D-075); stops before any freeze.

Waits for the files of each stage and runs it once, in order:
1. all six case-3/4 detector finals exist -> detector maps on val, case-4 partner, ROI policies
   (``prepare_r2_v2_val.py``), forward-FLOP table (already measured separately);
2. all nine SCD finals exist -> ``eval_r2_v2_systems.py --split val`` (one process per model on the
   GPU with the most free memory), then ``analyze_r2_v2_systems.py --split val``.
Never touches the test split, never freezes (the user confirms the protocol first), never kills
anything; a failing stage is logged and stops the watcher.

Usage: nohup python src/scripts/r2_v2_pipeline_val.py > artifacts/research2/v2/runs/pipeline_val.log &
"""

import subprocess
import sys
import time

import r2_v2_systems_lib as lib

ROOT = lib.ROOT
SCRIPTS = ROOT / "src/scripts"
PY = sys.executable
POLL = 300


def detector_finals_ready() -> bool:
    paths = [
        lib.RUNS / group / detector / template.format(seed=seed) / "run_record.json"
        for variant in ("medium_scratch", "light_resized")
        for group, detector, template, _ in [lib.DETECTOR_RUNS[variant]]
        for seed in lib.SEEDS
    ]
    return all(p.exists() for p in paths)


def scd_finals_ready() -> bool:
    return all((lib.scd_run_dir(m, s) / "run_record.json").exists()
               for m in lib.SCD_MODELS for s in lib.SEEDS)


def free_gpus() -> list[int]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()
    rows = sorted((float(line.split(",")[1]), int(line.split(",")[0])) for line in out)
    return [index for _, index in rows]


def run(command: list[str], name: str) -> None:
    print(time.strftime("%F %T"), "running", name, flush=True)
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode:
        print(f"STAGE FAILED: {name} (exit {result.returncode})", flush=True)
        raise SystemExit(result.returncode)


def main() -> None:
    stage1 = stage2 = False
    while not (stage1 and stage2):
        if not stage1 and detector_finals_ready():
            gpu = free_gpus()[0]
            maps_done = all(lib.map_path("val", v, s).exists()
                            for v in lib.DETECTOR_RUNS for s in lib.SEEDS)
            if not maps_done:
                run([PY, str(SCRIPTS / "prepare_r2_v2_val.py"), "maps", "--device", f"cuda:{gpu}"],
                    "val maps")
            if not (lib.V9 / "case4_partner.json").exists():
                run([PY, str(SCRIPTS / "prepare_r2_v2_val.py"), "partner"], "case-4 partner")
            if not (lib.V9 / "roi_policies.json").exists():
                run([PY, str(SCRIPTS / "prepare_r2_v2_val.py"), "policies"], "ROI policies")
            stage1 = True
        if stage1 and not stage2 and scd_finals_ready():
            gpus = free_gpus()
            procs = []
            for position, model in enumerate(lib.SCD_MODELS):
                command = [PY, str(SCRIPTS / "eval_r2_v2_systems.py"), "--split", "val",
                           "--devices", f"cuda:{gpus[position % len(gpus)]}", "--models", model]
                print(time.strftime("%F %T"), "starting", " ".join(command[2:]), flush=True)
                procs.append(subprocess.Popen(command, cwd=ROOT))
            if any(p.wait() for p in procs):
                print("STAGE FAILED: val evaluation", flush=True)
                raise SystemExit(1)
            run([PY, str(SCRIPTS / "analyze_r2_v2_systems.py"), "--split", "val"],
                "val analysis")
            stage2 = True
        if not (stage1 and stage2):
            time.sleep(POLL)
    print(time.strftime("%F %T"), "validation pipeline done; waiting for the user's protocol "
          "confirmation before the freeze", flush=True)


if __name__ == "__main__":
    main()
