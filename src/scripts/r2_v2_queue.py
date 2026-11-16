"""GPU job queue for the V6 searches and finals (D-072, D-075): stateless, restart-safe scheduler.

Jobs: 6 HPO trials per SCD model (the trial list of ``run_r2_v2_hpo.py``) and, once all six
trials of
a model have a ``run_record.json``, its 3 final runs (seeds 42/123/777, b* = 3.2e17 FLOPs, the
winning lr / weight decay by best validation SeK, ties -> lower trial index). Every loop re-reads
the
disk and ``ps``/``nvidia-smi``; a job whose process is already alive (matched by ``--name``)
counts as
running, a job with a ``run_record.json`` as done. Nothing is killed or paused, foreign processes
are
never touched, a job is never rerun or restarted automatically (a vanished job without a record is
reported in ``queue_state.json`` for a human decision).

Shared-GPU rule: foreign memory = used - own; a job is placed when the estimated
own+foreign
memory after placing stays <= 76 GB (<= 40 GB of own memory if foreign use exceeds 8 GB).

Usage: python src/scripts/r2_v2_queue.py [--dry-run] [--gpus 0 1 2 3] [--poll 120]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_r2_v2_hpo as hpo  # noqa: E402

ROOT = hpo.ROOT
RUNS = hpo.RUNS
RUNNER = hpo.RUNNER
LOG_DIR = RUNS / "queue_logs"
STATE = RUNS / "queue_state.json"
HPO_BUDGET = 8e16
FINAL_BUDGET = 3.2e17
SEEDS = (42, 123, 777)
NEED_GB = {"mamba_scd": 56.0, "siam_unet_resnet18": 21.0, "scannet": 22.0}
PRIORITY = ("mamba_scd", "siam_unet_resnet18", "scannet")  # longest jobs first
HPO_TRIALS = 6
TOTAL_CAP_GB, BUSY_CAP_GB, BUSY_FOREIGN_GB = 76.0, 40.0, 8.0
WORKERS = 16
# GPUs kept free of small jobs while a MambaSCD job (needs ~54 GB) is waiting, so it cannot starve.
MAMBA_GPUS = (1, 3)


@dataclass
class Job:
    name: str
    model: str
    group: str
    lr: float
    weight_decay: float
    seed: int
    budget: float
    kind: str  # "hpo" | "final"
    trial: int = -1
    depends: list[str] = field(default_factory=list)

    @property
    def record(self) -> Path:
        return RUNS / self.group / self.model / self.name / "run_record.json"


def hpo_name(model: str, index: int, trial: dict[str, float]) -> str:
    return f"{hpo.SHORT[model]}_hpo_t{index}_lr{trial['lr']:g}_wd{trial['weight_decay']:g}"


def hpo_jobs(model: str) -> list[Job]:
    return [
        Job(hpo_name(model, i, t), model, hpo.GROUP, t["lr"], t["weight_decay"], 42, HPO_BUDGET,
            "hpo", i)
        for i, t in enumerate(hpo.draw_trials(HPO_TRIALS))
    ]


def winner(model: str) -> dict | None:
    """Best trial by validation SeK once all trials are finished (writes search_result.json)."""
    jobs = hpo_jobs(model)
    if not all(job.record.exists() for job in jobs):
        return None
    rows = []
    for job in jobs:
        record = json.loads(job.record.read_text())
        rows.append({
            "trial": job.trial, "name": job.name, "lr": job.lr, "weight_decay": job.weight_decay,
            "best_sek": record["best_sek"], "best_epoch": record["best_epoch"],
            "final_sek": record["final_sek"], "optimizer_steps": record["optimizer_steps"],
            "training_flops": record["training_flops"], "wall_seconds": record["wall_seconds"],
        })
    best = max(rows, key=lambda row: (row["best_sek"], -row["trial"]))
    path = RUNS / hpo.GROUP / model / "search_result.json"
    if not path.exists():
        path.write_text(json.dumps({
            "model": model, "budget_flops_per_trial": HPO_BUDGET, "seed": 42,
            "search_seed": hpo.SEARCH_SEED, "trials": hpo.draw_trials(HPO_TRIALS),
            "results": rows, "selected": best, "written_by": "r2_v2_queue.py",
        }, indent=2))
    return best


def final_jobs(model: str) -> list[Job]:
    best = winner(model)
    if best is None:
        return []
    return [
        Job(f"{hpo.SHORT[model]}_final_s{seed}", model, "final", best["lr"], best["weight_decay"],
            seed, FINAL_BUDGET, "final")
        for seed in SEEDS
    ]


def all_jobs() -> list[Job]:
    jobs: list[Job] = []
    for model in PRIORITY:
        jobs += hpo_jobs(model)
    for model in PRIORITY:
        jobs += final_jobs(model)
    return jobs


def process_table() -> list[tuple[int, str]]:
    out = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True, check=True)
    rows = []
    for line in out.stdout.splitlines()[1:]:
        pid, _, args = line.strip().partition(" ")
        rows.append((int(pid), args))
    return rows


def alive_jobs(processes: list[tuple[int, str]]) -> dict[str, tuple[int, str]]:
    """job name -> (pid, device) for live SCD runner processes."""
    found = {}
    for pid, args in processes:
        if "run_r2_v2_scd.py" not in args or "--name" not in args:
            continue
        tokens = args.split()
        name = tokens[tokens.index("--name") + 1]
        device = tokens[tokens.index("--device") + 1] if "--device" in tokens else "cuda:0"
        found[name] = (pid, device)
    return found


def alive_detectors(processes: list[tuple[int, str]]) -> list[int]:
    """GPU indices of live detector runs (case 3-4 chains) - they share the GPUs."""
    found = []
    for _, args in processes:
        if "run_r2_v2_detector.py" in args and "--device" in args.split():
            tokens = args.split()
            found.append(int(tokens[tokens.index("--device") + 1].split(":")[1]))
    return found


def gpu_state(gpus: list[int], processes: list[tuple[int, str]]) -> dict[int, dict[str, float]]:
    """used / own memory (GB) per GPU; own = compute processes whose command is in this checkout."""
    uuids = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.used",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True,
                           check=True).stdout.strip().splitlines()
    index_of, used = {}, {}
    for line in uuids:
        index, uuid, memory = [part.strip() for part in line.split(",")]
        index_of[uuid] = int(index)
        used[int(index)] = float(memory) / 1024
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                           "--format=csv,noheader,nounits"], capture_output=True, text=True,
                          check=True).stdout.strip().splitlines()
    commands = dict(processes)
    own = dict.fromkeys(used, 0.0)
    for line in apps:
        if not line.strip():
            continue
        uuid, pid, memory = [part.strip() for part in line.split(",")]
        if str(ROOT) in commands.get(int(pid), "") or "src/scripts/" in commands.get(int(pid), ""):
            own[index_of[uuid]] += float(memory) / 1024
    return {g: {"used": used[g], "own": own[g], "foreign": used[g] - own[g]} for g in gpus}


def place(job: Job, gpus: dict[int, dict[str, float]], reserved: dict[int, float]) -> int | None:
    """GPU index where the job fits under the memory rules, or None."""
    need = NEED_GB[job.model]
    best = None
    for gpu, state in gpus.items():
        own_after = max(state["own"], reserved[gpu]) + need
        total_after = state["foreign"] + own_after
        cap_total = TOTAL_CAP_GB
        if state["foreign"] > BUSY_FOREIGN_GB and own_after > BUSY_CAP_GB:
            continue
        if total_after <= cap_total and (best is None or total_after < best[0]):
            best = (total_after, gpu)
    return None if best is None else best[1]


def launch(job: Job, gpu: int) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(RUNNER), "--model", job.model, "--flop-budget", str(job.budget),
        "--lr", str(job.lr), "--weight-decay", str(job.weight_decay), "--seed", str(job.seed),
        "--device", f"cuda:{gpu}", "--group", job.group, "--name", job.name,
        "--num-workers", str(WORKERS),
    ]
    with (LOG_DIR / f"{job.name}.log").open("ab") as log:
        subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                         start_new_session=True)
    print(time.strftime("%F %T"), "launched", job.name, "on cuda:", gpu, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--poll", type=int, default=120)
    parser.add_argument("--max-own-jobs-per-gpu", type=int, default=2)
    args = parser.parse_args()
    seen_alive: set[str] = set()
    while True:
        processes = process_table()
        alive = alive_jobs(processes)
        jobs = all_jobs()
        pending = [j for j in jobs if not j.record.exists() and j.name not in alive]
        lost = [j.name for j in pending if j.name in seen_alive]
        seen_alive = set(alive)
        gpus = gpu_state(args.gpus, processes)
        reserved = dict.fromkeys(args.gpus, 0.0)
        count = dict.fromkeys(args.gpus, 0)
        for gpu in alive_detectors(processes):
            if gpu in count:
                count[gpu] += 1
        for name, (_, device) in alive.items():
            gpu = int(device.split(":")[1])
            job = next((j for j in jobs if j.name == name), None)
            if gpu in reserved and job is not None:
                reserved[gpu] += NEED_GB[job.model]
                count[gpu] += 1
        started = []
        # never start a job that previously ran and vanished without a record (needs a human)
        fresh = [j for j in pending if not (RUNS / j.group / j.model / j.name).exists()]
        mamba_waiting = any(j.model == "mamba_scd" for j in fresh)
        for job in fresh:
            candidates = {g: s for g, s in gpus.items() if count[g] < args.max_own_jobs_per_gpu}
            if job.model != "mamba_scd" and mamba_waiting:
                candidates = {g: s for g, s in candidates.items() if g not in MAMBA_GPUS}
            if job.model == "mamba_scd":
                candidates = {g: s for g, s in candidates.items() if count[g] == 0}
            gpu = place(job, candidates, reserved) if candidates else None
            if gpu is None:
                continue
            if not args.dry_run:
                launch(job, gpu)
            started.append(f"{job.name}@{gpu}")
            if job.model == "mamba_scd":
                launched = {x.split("@")[0] for x in started}
                mamba_waiting = any(
                    j.model == "mamba_scd" and j.name not in launched for j in fresh
                )
            reserved[gpu] += NEED_GB[job.model]
            count[gpu] += 1
            gpus[gpu]["own"] += 0.0
        stalled = [j.name for j in pending
                   if (RUNS / j.group / j.model / j.name).exists() and j.name not in alive]
        done = sum(1 for j in jobs if j.record.exists())
        state = {
            "time": time.strftime("%FT%TZ", time.gmtime()), "jobs_known": len(jobs), "done": done,
            "alive": {n: d for n, (_, d) in alive.items()}, "started_now": started,
            "stalled_without_record": stalled, "vanished_since_last_poll": lost,
            "gpu_gb": {g: {k: round(v, 1) for k, v in s.items()} for g, s in gpus.items()},
        }
        STATE.write_text(json.dumps(state, indent=1))
        if args.dry_run:
            print(json.dumps(state, indent=1))
            return
        if done == len(jobs) and len(jobs) == len(PRIORITY) * (HPO_TRIALS + len(SEEDS)):
            print("all jobs done", flush=True)
            return
        time.sleep(args.poll)
        os.sched_yield()


if __name__ == "__main__":
    main()
