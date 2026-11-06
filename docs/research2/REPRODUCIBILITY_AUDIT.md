# Research 2 — Reproducibility Audit (task R2-P03-T01, read-only)

**Status:** COMPLETED 2026-11-06 for code analysis and CPU evidence; GPU evidence pending (no GPU
on this workstation; commands in `READINESS_AUDIT.md` §4 collect the server stack). No code or
determinism setting was changed (D-019). Line numbers refer to HEAD `b72cd6c`.

## A. Classification

**Deterministic under defined conditions — by design, empirically confirmed on CPU only.**
Bitwise repeatability is expected only when *all* of the following are identical: GPU model,
CUDA/cuDNN/cuBLAS/PyTorch build, code, data, configuration (including `batch_size`,
`num_workers`, `precision`, `world_size`) and seed. Across GPU architectures, software versions or
world sizes only **statistical** reproducibility is achievable. Two settings are uncontrolled on
the binary path (TF32, cuBLAS workspace override) and the GPU model is not recorded per training
run — so historical Research 1 runs are *partially* reproducible: the stack can be reconstructed
from `benchmark.json` (A100-SXM4-80GB, CUDA 13.0, torch 2.12.1+cu130) but not proven per run.

## B. Code evidence

| Mechanism | Location | Effect |
|---|---|---|
| Global seeding + deterministic mode | `execution.seed_everything` (`execution.py:77–87`) | Python/NumPy/Torch/CUDA seeds; `torch.use_deterministic_algorithms(True)` (no `warn_only` → nondeterministic ops **raise**); cuDNN deterministic, benchmark off; `CUBLAS_WORKSPACE_CONFIG=:4096:8` via `os.environ.setdefault` (an existing value wins) |
| HPO v3 worker | `hpo_backend.py:570–582` | `torch.cuda.set_device` then `seed_everything(seed)`; TF32 set explicitly **only** for the semantic task |
| Legacy HPO worker | `optimization._run_single_trial` (`optimization.py:416–447`) | forces `CUBLAS_WORKSPACE_CONFIG` (assignment, not `setdefault`), same deterministic flags, own seeding — duplicate of `seed_everything` |
| DDP | `run.py:106, 150–171` | `DistributedSampler(seed=seed)`, loader generator `seed + rank`, `seed_everything(seed + rank)` |
| Shuffling | `data/levir_cd.py:515–529` | explicit `torch.Generator(seed)` only when `shuffle and seed is not None` |
| Augmentation | `data/transforms.py:54–65` | `torch.rand`/`torch.randint` from the process (or worker) Torch RNG |
| DataLoader workers | no `worker_init_fn` | PyTorch default: each worker's Torch seed derives from the loader generator; sample→worker assignment depends on `num_workers` → `num_workers` changes augmentation draws (it is part of the HPO identity, `hpo_backend.py:47, 65`) |
| Resume | `execution.capture_rng`/`restore_rng` (`execution.py:90–111`); RNG saved per rank in checkpoints (`training.py:1085`) | epoch-boundary resume restores Python/NumPy/Torch/CUDA/loader-generator state; mid-epoch resume unsupported |
| Precision | `execution.precision_context` | fp32 or bf16 autocast (CUDA only); Heavy final models trained **bf16**, all six evaluated fp32 |
| Evaluation numerics | `final_benchmark._runtime` (`final_benchmark.py:670–696`) | forces TF32 off, autocast off, cuDNN deterministic during evaluation — evaluation is in a different numeric regime than bf16/TF32 training |
| Nondeterministic-op guard | bilinear `F.interpolate` in `models/bit.py:452, 518`, `models/siam_mit_b3_diff.py:315, 323` | under `use_deterministic_algorithms(True)` PyTorch either runs a deterministic kernel or raises; historical CUDA training completed, so these ops did not raise on torch 2.12.1 |
| Recorded metadata | `hpo_backend.py:712–725` | `torch_version`, `cuda_version`, `world_size`, `execution_mode`, `code_revision`; **not** GPU model, driver, cuDNN version, TF32 flags, `CUBLAS_WORKSPACE_CONFIG` |

**CPU evidence (2026-11-06, torch 2.14.1+cpu, 1 thread).** Using only project APIs
(`seed_everything`, `FCSiamDiff`, `RandomFlipRotate`, seeded shuffling, `fit_binary_change_model`)
on synthetic 32×32 data: (1) two identical 4-epoch runs → bitwise identical weights; (2) 2 epochs +
`resume_from=last.pt` for 2 more → bitwise identical to the uninterrupted 4-epoch run; (3) sanity:
2 ≠ 4 epochs. This confirms the RNG/optimizer/loader restore logic, not GPU kernels.

## C. Same GPU, repeated runs

Same A100 model, same software stack, same configuration → expected **bitwise identical**
training trajectories for HPO v3 and `run.py` paths (deterministic mode + seeded generators +
cuDNN deterministic). Caveats: (1) binary-path TF32 follows PyTorch defaults (cuDNN conv TF32 is
on by default in the legacy flag API; the server value is printed by `READINESS_AUDIT.md` §4) —
deterministic, but numerically different from strict FP32 and differently than evaluation;
(2) a pre-set `CUBLAS_WORKSPACE_CONFIG` (e.g. `:16:8`) is kept by `seed_everything` but
overwritten by the legacy path — both are deterministic settings but may select different cuBLAS
algorithms; (3) not empirically verified on GPU.

## D. Changing GPUs

- **A100 → another A100 of the same SKU** (same software stack): kernels and heuristics are the same
  → bitwise identity expected (to be confirmed with one short repeat run when a GPU is available).
- **Different A100 SKU (PCIe vs SXM, 40 vs 80 GB) or different driver/CUDA/cuDNN/PyTorch:** algorithm
  selection may differ → small numerical differences that compound over training → not bitwise;
  results remain statistically comparable.
- **Different architecture (e.g. H100, consumer GPUs):** different kernels and bf16/TF32 behaviour →
  never bitwise; statistical consistency only.
- **Different number of GPUs (DDP world size):** changes data partitioning (`DistributedSampler`),
  per-rank seeds (`seed + rank`) and effective batch → a different training trajectory, not a
  reproduction; exact resume requires the same world size. HPO v3 always uses one GPU per job.

Changing GPU *device IDs* on the same machine does not by itself break reproducibility.

## E. HPO reproducibility

- **Trial generation is deterministic:** grid expansion from the frozen `pipeline_protocol.json`;
  job identities are canonical parameter sets; seeds are explicit.
- **Scheduling is asynchronous but numerically irrelevant on a homogeneous pool:** jobs run in
  separate spawned processes, one per device, assigned by free memory (`hpo_core.py:483–519`);
  each job's result depends only on its own parameters and seed. On a **heterogeneous** pool the
  GPU model a job lands on is nondeterministic → the result would depend on scheduling.
- **OOM retries** resume the job from `last.pt` at an epoch boundary on another device; per the
  CPU evidence this is trajectory-preserving on identical hardware.
- **Selection** is a deterministic function of recorded validation metrics (first threshold wins
  ties; earlier epoch wins semantic ties).
- **Pause/resume** of whole phases follows the same epoch-boundary logic.

## F. Final answer

**Yes — you can use different GPUs in Research 2 without compromising scientific validity, under
these conditions:**
1. Within any single comparison (A vs B, scratch vs fine-tune, seeds of one system) all runs use the
   same GPU model and software stack — or, if impossible, GPU type is recorded and balanced across
   compared conditions rather than confounded with one of them.
2. Claims are **statistical** (multiple training seeds, paired evaluation, CIs), not bitwise; bitwise
   identity is a debugging property, not a scientific requirement.
3. All **latency/throughput** measurements of compared systems are made on one fixed GPU model and
   stack (D-026); GPU-hours from different GPU types are reported separately, not pooled silently.
4. The GPU model, driver, CUDA/cuDNN/PyTorch versions and TF32/precision flags are recorded for every
   run (currently missing — see R2-P03-T02).
5. Research 1 checkpoints are reused as fixed artifacts (hash-identified), so their historical
   non-recorded settings do not affect Research 2 comparisons except as documented provenance.

## G. Recommendations (task R2-P03-T02)

Implemented now (test only, no setting changed): `tests/test_execution.py::
test_repeat_and_epoch_boundary_resume_are_bitwise_identical` turns the CPU evidence above into a
permanent regression test (tests are not part of the HPO code signature).

Proposed, **not implemented** — each changes files covered by `backend_code_signature` (D-022) and
the numerics-affecting ones change the training regime, so they wait for the compatibility
mechanism (R2-P06-T03) and, where marked, a user decision:

| # | Proposal | Affects numerics? | Gate |
|---|---|---|---|
| 1 | Record GPU model, driver, cuDNN version, effective TF32 flags, `CUBLAS_WORKSPACE_CONFIG`, deterministic-mode state in every training config/summary (extend the existing metadata block in `hpo_backend.py:712–725`, `run.py`) | no | technical (after R2-P06-T03) |
| 2 | Set TF32 explicitly on the binary path from configuration (default = documented value) instead of PyTorch defaults | yes for new runs; Research 1 unaffected (immutable) | **protocol decision** (D-026) |
| 3 | Make `CUBLAS_WORKSPACE_CONFIG` handling identical in `seed_everything` and the legacy worker (warn when a pre-set value differs) | possibly | technical + note in protocol |
| 4 | Hash normalized (LF) source content in the code signature (READINESS_AUDIT F3) | no | part of R2-P06-T01 design |
| 5 | One short GPU repeat run (2 epochs, same seed, twice; plus 1+1 resume) on the target A100s to confirm C/D empirically | — | small GPU use; run with the first approved GPU session |
| 6 | Research 2 latency benchmarks on one fixed GPU model, recorded per D-026 | — | protocol (D-026) |
