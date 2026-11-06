# Research 2 — Readiness Audit (task R2-P01-T01)

**Status:** local part COMPLETED 2026-11-06; training-server part BLOCKED (no server access from this
workstation) — commands in §4 remain to be run. Read-only except the two local provenance tags
recorded in F2 (created with the user's approval). Labels as in [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md).

## 1. Readiness table

| # | Item | Local (this checkout, 2026-11-06) | Training server | Evidence |
|---|---|---|---|---|
| 1 | Source version | HEAD `b72cd6c` on `main`; root commit `9ed0fc7` (2026-10-06) — **history before 2026-10-06 is not on `main`** | UNKNOWN | `git rev-parse HEAD`, `git rev-list --max-parents=0 HEAD` |
| 2 | Uncommitted work | `M notebooks/02_fc_siam_diff_full_hpo.ipynb`; `?? .clinerules/`, `Instruction.md`, `docs/research2/` | UNKNOWN | `git status --short` |
| 3 | Checkout shape | sparse (`configs docs notebooks scripts src tests`), partial clone `blob:none`; `artifacts/`, `data/` not materialized | UNKNOWN | `git sparse-checkout list`, `git remote -v` |
| 4 | HPO v3 code signature (current code) | `3c676c65…eae1a` — computed on **CRLF** working files (see finding F3) | UNKNOWN | `SatelliteBackend().code_signature()` |
| 5 | Final-model checkpoints | 22 `.pt` tracked as Git LFS pointers; expected SHA256/size in §2 | UNKNOWN (presence, LFS smudged or pointer, hash match) | `git show HEAD:<path>` (pointer text only) |
| 6 | Training-code provenance of final models | see F1–F2: two training commits unreachable, one missing | — | `metadata.json`, `training_results.json` |
| 7 | Unfinished Research 1 HPO searches | not available (`artifacts/hyperparameter_search/` untracked) | UNKNOWN — needed for R2-P06-T01 | — |
| 8 | Research 1 hypothesis outputs | not available (`artifacts/hypothesis_tests/` untracked) | UNKNOWN — needed for R2-P01-T03 | — |
| 9 | SECOND data | absent (expected, D-027) | UNKNOWN | — |
| 10 | SECOND loader/prep | present: `semantic/second.py` (`bootstrap_second`, `discover_second`, `prepare_second`); coupled to ROI-classifier prep | same code if HEAD matches | `PROJECT_CONTEXT.md` §13 |
| 11 | WHAT / single-stage SCD model | missing | missing (same code) | `PROJECT_CONTEXT.md` §10–11 |
| 12 | Dense SCD metrics, reconstruction, end-to-end latency | missing | missing | idem |
| 13 | Reproducibility hooks | present (seeds, deterministic algorithms, generators, epoch-boundary RNG restore); TF32 explicit only on semantic path | defaults UNKNOWN (torch version) | `PROJECT_CONTEXT.md` §17 |
| 14 | GPU-hour accounting hooks | epoch-compute `gpu_hours` only; no budget/ledger | same | `training.py:1083`, `hpo_backend.py:785, 826` |
| 15 | GPUs | none: Python 3.12.14, torch `2.14.1+cpu`, CUDA unavailable | UNKNOWN (reportedly 4× A100 80 GB) | `torch.__version__`, `torch.cuda.is_available()` |
| 16 | Running jobs | none relevant locally | UNKNOWN | — |
| 17 | Git LFS client | `git-lfs/3.7.1` | UNKNOWN | `git lfs version` |

## 2. Expected checkpoint identities (VERIFIED from LFS pointers in HEAD)

The LFS `oid` is the SHA256 of the file content, so the server files can be verified against it.

| Path under `artifacts/final_models/` | SHA256 (LFS oid) | Size (bytes) |
|---|---|---|
| `heavy_full/deployment_weights.pt` | `03f8c745a9adc109bf394233d5be43ca4c5e8f696074f9a1398d6de1199d250a` | 183318350 |
| `heavy_full/seed_123_best.pt` | `70bb72f4edbc21cc52f3043b48787ca938f58aa040cf9d98c38ff89ae8194db4` | 733329115 |
| `heavy_full/seed_42_best.pt` | `738c757b644174501fcf8584de05db94bcb702d43ff8b2523cbba1b477bd6b42` | 733332443 |
| `heavy_full/seed_777_best.pt` | `b61d891b6b870c06b8e7f03cbadbea8f5ac9604703386f4204f427602de2a904` | 733332891 |
| `heavy_resized/deployment_weights.pt` | `acdbfba44845cb3b9ead09a122e98e22c38678f7a1578f6d956983348367abd0` | 183318350 |
| `heavy_resized/seed_42_best.pt` | `c024f68cc6050db7e1ec4a25808dcce24d1075c837dfceb065eca08577d6bd6c` | 733348379 |
| `light_full/deployment_weights.pt` | `3a786b6de6743179614c28a06e019063a22f503dfbe21cba2c1bc7ff0e979d77` | 31058517 |
| `light_full/seed_123_best.pt` | `b8312630436f35dca810089c775c99cef5e2ca5ea2315a5d6edc9701515ce10c` | 93200853 |
| `light_full/seed_2024_best.pt` | `12fd0449554cdbddb04258dd0f312a3479768d23fd51d83ca55c9429f13f306c` | 93201109 |
| `light_full/seed_777_best.pt` | `94b784644fd14e7da2d3af70b1490224383e90c661dd9d1407b468db7c40453b` | 93182613 |
| `light_resized/deployment_weights.pt` | `ac304d1d6543d4909dcaaaedca890fcd1e541a0a5639b60f2a18f247317230b7` | 7779605 |
| `light_resized/seed_123_best.pt` | `6e8db6e51a08f1aaaeac4a0cddd9d51f4c0a3fc22cdcff20829544187b538f02` | 23372885 |
| `light_resized/seed_42_best.pt` | `a4becfc385ad7382159cfce9259de2738d549d0161cb8b42f79c247318f0aa6a` | 23371157 |
| `light_resized/seed_777_best.pt` | `e4a8dbec3eb6dc89527688b7bce94db938f7ff01e09ce22f87ec070ff1a22793` | 23369045 |
| `medium_full/deployment_weights.pt` | `379cdafddb1353bfc3762b8f2ed6b9181b7f4220fdf7d9b80b334e939c16260e` | 17300484 |
| `medium_full/seed_123_best.pt` | `8680db04cea68e588217906ba93d78d3422cba9f6b8e11b031b4ce2992bd922a` | 51896786 |
| `medium_full/seed_42_best.pt` | `056d735c940e273ed7b6483c0aae5e805122aea862a9da24871026d68dece0d6` | 51889938 |
| `medium_full/seed_777_best.pt` | `9f98561fbe784d8292fedefdd17378041e3358fe3d0bec8df2a617e336712cb9` | 51890258 |
| `medium_resized/deployment_weights.pt` | `1492cad2f23e2d463635899fe45be0bbf498b337b8f8e9574f72bdefb87f38ef` | 14078852 |
| `medium_resized/seed_123_best.pt` | `70b588d2d67e5a2fecc23355f764d1f4463e0eff7c72c4c97690755b53e94f78` | 42231826 |
| `medium_resized/seed_42_best.pt` | `c273a74afcc1fd600d898ddede7264b6d2dc33f33f78b0cfa22dcaa5fbb0be79` | 42231442 |
| `medium_resized/seed_777_best.pt` | `bcb394476f7bd6a0c6c0961825ff43cec1a8994a34a048af4c4b66fcb8e56127` | 42228306 |

Deployment selections recorded in `metadata.json` (VERIFIED HISTORICAL): light_full seed 2024 /
threshold 0.8; light_resized 42 / 0.7; medium_full 123 / 0.7; medium_resized 123 / 0.7;
heavy_full 42 / 0.8; heavy_resized 42 / 0.85.

## 3. Findings

**F1 — Light models: training commit missing.** `light_full` and `light_resized` `metadata.json`
record `git_commit 7f38560c7e845ee6478ae3adcdff9f0aa1d22253`. The object is absent locally and the
remote refuses it (`upload-pack: not our ref`). The exact training code of both Light models is
therefore **not recoverable from git** (unless another copy exists on the server). Impact: D-010
freeze manifests for Light checkpoints can cite weights and `model_parameters` but not code
identity; the planned pilot uses Light weights (D-020).

**F2 — Medium and Heavy-resized training commits exist only as unreferenced local objects.**
`02b72d7` (2026-09-09, used by `medium_full`/`medium_resized` `training_results.json`) and
`81a5e53` (2026-09-28, `heavy_resized`) exist in this local object store but are were contained in no
branch, tag, reflog or other ref (`git for-each-ref --contains` → empty) and could be removed by a
future `git gc`/prune. `heavy_full` (`9ed0fc7`) is reachable (root of `main`). Remote presence of
`02b72d7`/`81a5e53` is UNKNOWN. `main` history starts at `9ed0fc7`, i.e. the pre-October history
was not carried over. **Resolved locally 2026-11-06 (user-approved):** identities verified (author,
date, message; SHA referenced by `medium_full/training_results.json`, `medium_resized/metadata.json`,
`heavy_resized/metadata.json`) and annotated local tags created:
`provenance/r1-medium-training-02b72d7`, `provenance/r1-heavy-resized-training-81a5e53`. The latter
also preserves the whole pre-October history (19 commits, root `886bc7d` 2026-07-27, "First Commit!").
`7f38560` is not in that history. Tags are **not pushed**; pushing is a separate decision.

**F3 — The HPO code signature depends on line endings.** `SatelliteBackend.code_signature`
(`hpo_backend.py:268–290`) hashes `path.read_bytes()` of working-tree files. In this Windows
checkout the files are CRLF in the working tree and LF in the index (`git ls-files --eol`:
`i/lf w/crlf`), so the same commit produces different signatures on Windows and Linux. Consequences
for D-022: a signature recorded on the server cannot be reproduced on this machine from the same
commit; any migration evidence must hash normalized content (e.g. git blob contents) rather than
raw working-tree bytes. Recorded as a new open design question for R2-P06-T01.

**F4 — Research 1 HPO and hypothesis outputs are server-only.** Whether any v3 search is
unfinished — the key input for R2-P06-T01 — can only be answered on the server (§4).

## 4. Server part (to be run by the user)

Run from the project root on the training server with the project environment active. All
commands only read; `sha256sum` reads ~4 GB of checkpoints. **Review the output before sharing**
(process command lines may contain paths you consider private; no environment variables are
printed).

```bash
set -u
echo "== git"; git rev-parse HEAD; git branch --show-current; git status --short | head -60
git sparse-checkout list 2>/dev/null; git lfs version 2>/dev/null
for c in 7f38560c7e845ee6478ae3adcdff9f0aa1d22253 02b72d7fc9f7deebc696662474cae8de564b8530 81a5e53351305c644823eb1301ac86f10d234675; do printf "%s: " "$c"; git cat-file -t "$c" 2>/dev/null || echo absent; done
echo "== final model checkpoints (pointer or real file?)"; git lfs ls-files 2>/dev/null | grep final_models
find artifacts/final_models -name '*.pt' -print0 | sort -z | xargs -0 sha256sum
echo "== environment"
python - <<'EOF'
import sys, torch
print("python", sys.version.split()[0], "| torch", torch.__version__, "| cuda", torch.version.cuda,
      "| cudnn", torch.backends.cudnn.version(), "| available", torch.cuda.is_available())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
print("tf32 defaults: matmul", torch.backends.cuda.matmul.allow_tf32, "| cudnn", torch.backends.cudnn.allow_tf32)
from satchange.hpo_backend import SatelliteBackend
print("current backend_code_signature", SatelliteBackend().code_signature())
EOF
python -m pip list 2>/dev/null | grep -iE "^(torch|torchvision|numpy|opencv-python|pillow|gdown|scikit-image|scipy) "
nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
echo "== running project processes"
ps -eo pid,etime,args | grep -E "satchange|torchrun|ipykernel|jupyter" | grep -v grep | cut -c1-200
echo "== HPO v3 searches"
python - <<'EOF'
import json, pathlib
root = pathlib.Path("artifacts/hyperparameter_search")
protocols = sorted(root.rglob("pipeline_protocol.json")) if root.exists() else []
for proto in protocols:
    d = proto.parent
    sig = json.loads(proto.read_text(encoding="utf-8")).get("backend_code_signature")
    marks = [m for m in ("PAUSE_AFTER_EPOCH", "STOP_AFTER_ACTIVE_TRIALS") if (d / m).exists()]
    print(d, "| signature", sig, "| markers", marks or "-")
    for r in sorted(d.rglob("results.json")):
        try:
            j = json.loads(r.read_text(encoding="utf-8"))
        except Exception as error:
            print("   ", r.relative_to(d), "unreadable", type(error).__name__); continue
        print("   ", r.relative_to(d), "complete", j.get("complete"), "paused", j.get("paused"),
              "unfinished", len(j.get("unfinished_job_ids", [])), "failed", len(j.get("failed_trials", [])))
v3 = {p.parent for p in protocols}
others = sorted(p for p in root.glob("*/*") if p.is_dir() and not any(p == q or p in q.parents for q in v3)) if root.exists() else []
print("depth-2 directories without a v3 protocol (likely legacy):", *others, sep="\n  ")
EOF
echo "== datasets and research outputs"
ls -la data 2>/dev/null; ls artifacts/hypothesis_tests 2>/dev/null
python - <<'EOF'
from pathlib import Path
from satchange.semantic.second import discover_second
root = Path("data/SECOND")
if not root.exists():
    print("data/SECOND absent")
else:
    try:
        found = discover_second(root)
        print({split: len(rows) for split, rows in found.items()})
    except Exception as error:
        print("discover_second failed:", type(error).__name__, error)
EOF
df -h . 2>/dev/null
```

## 5. Remaining work for this task

**BLOCKED until server access.** When available, any session should:
1. Run §4 on the server (the user or an agent with server access).
2. Fill the "Training server" column; compare checkpoint hashes with §2; list unfinished searches
   and whether their recorded signature equals the server's current signature.
3. Update `PROJECT_CONTEXT.md` (§11, §13, §20), `PROGRESS.md`, and set this task to COMPLETED.
