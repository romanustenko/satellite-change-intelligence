# AGENTS.md — satellite-change-intelligence

Primary persistent knowledge base for AI coding agents. Verified against the repository on
2026-11-01. Items marked **(unverified)** / **unknown** could not be confirmed from files.
Note: `AGENTS.md` is listed in `.gitignore` (local-only); update it manually when facts change.

---

## 0. Research 2 — mandatory context restoration (added 2026-11-06)

The git-tracked source of truth for Research 2 is `docs/research2/` (this file is only a local
pointer). Before any development, every session must:

1. Read this file, then `docs/research2/PROJECT_CONTEXT.md` (scientific memory; §0 = procedure).
2. Review `docs/research2/DECISION_LOG.md` — APPROVED decisions are binding; never change one
   silently; record conflicts and propose reconsideration.
3. Review `docs/research2/PROGRESS.md` — milestone, pending approvals, blockers.
4. Locate the assigned task (`R2-Pxx-Tyy`) in `docs/research2/IMPLEMENTATION_PLAN.md`; respect its
   dependencies and approval gates (G1–G11). `docs/research2/EXPERIMENTAL_PROTOCOL.md` is a
   PROPOSED draft until gate G11.
5. Inspect the relevant current code; confirm it still matches the documents.
6. State: global objective → milestone → scientific requirement → existing implementation →
   assigned task → expected evidence. If the purpose is unclear, stop and report.
7. Reuse existing components; no duplicated trainers/HPO/loaders/metrics; preserve Research 1
   artifacts and results; verify before marking a task COMPLETED/VERIFIED.
8. After meaningful work, update `PROGRESS.md`, the task status and, for decisions, `DECISION_LOG.md`.

**Working policy (D-028, supersedes "ask/plan before each technical step" in §10 and
`.clinerules`):** work autonomously on approved tasks — technical choices, files, tests, docs, safe
local git operations need no confirmation; after each task validate → update docs → short report
(Completed/Why/Verified/Remaining/Next, Russian) → continue with the next unblocked task. Scientific
decisions, GPU budgets, pushes/history rewrites and Research 1 artifacts remain gated. Full text:
`docs/research2/PROJECT_CONTEXT.md` §0.
**D-030 (autonomy v2):** gates G1–G11 are verification checkpoints; ask the user only for the
research question, hypotheses, evaluation methodology, evaluation splits, final comparison rules,
substantial GPU spend, destructive operations on historical artifacts, irreversible remote changes.
**D-032:** work as one autonomous agent — no subagents or automatic delegation; avoid repeated
audits, excessive planning and unnecessary documentation; reuse existing context; brief Russian
reports (done / why / remaining).

Corrections to the sections below (user decisions 2026-11-06):
- **Research 1 is ongoing**, not frozen; Research 2 reuses its checkpoints only via a versioned
  freeze manifest (D-010).
- **TinySiameseChangeClassifier was intentionally abandoned.** Do not restore or reimplement it;
  removal of its remaining code is task R2-P07-T01 (D-006). References to its notebook are outdated.
- The absence of `data/` locally is expected (populated by notebook code); the local machine is not
  the training environment — no speculative CUDA changes (D-027).
- One shared `src/satchange/` for both studies; no duplicate infrastructure (D-021).
- **HPO code-signature coupling:** editing `training.py`, `execution.py`, `losses.py`, `metrics.py`,
  `hpo_*.py`, `automated_hpo.py`, `optimization.py`, `run.py`, `data/`, `models/` or `semantic/`
  changes `backend_code_signature` and blocks resuming existing v3 searches under the same
  `search_name`; the signature mechanism may only change after the design review (D-022, G9).

---

## 1. Project Overview

- **Package:** `satchange` 0.1.0 (`pyproject.toml`, setuptools, `src/` layout, editable install
  in `.venv`). Declared description mentions "uncertainty estimation" — **no such code exists**.
- **Research goal:** compare binary change-detection models on **LEVIR-CD** (bi-temporal RGB,
  1024×1024) in two input regimes — **Full** (native) and **Resized** (whole scene → 256×256,
  no crops) — then study ROI extraction: binary "WHERE" detector → native-scale ROI crops →
  semantic "WHAT" classifier trained on **SECOND** (6 classes).
- **Model tiers** (names used in artifacts/notebooks):

  | Tier | Registry name | Class | File (`src/satchange/models/`) |
  |---|---|---|---|
  | Light | `fc_siam_diff` | `FCSiamDiff` | `fc_siam_diff.py` |
  | Medium | `bit` | `BIT` (BIT-S4-DD8, ResNet18-S4, scratch only) | `bit.py` |
  | Heavy | `siam_mit_b3_diff` | `SiamMiTB3Diff` (MiT-B3, 45,772,961 params default) | `siam_mit_b3_diff.py` |

- **Primary workflow:** prepare data → optional tiny-overfit → single training or 3-phase HPO
  (`hpo` → `seeds` → `final`) → `run_final_benchmark` (validate all seeds, then test, export) →
  research notebooks 05–11 analyse the six finalized models (`{light,medium,heavy}_{full,resized}`)
  without retraining.

## 2. Repository Architecture

```
configs/          YAML experiment configs + one JSON HPO search space
docs/             heavy_model.md (Russian; Heavy report + commands), research2/ (Research 2 docs)
notebooks/        research_1/ (01–11 + Heavy staged-resume / HPO notebooks), research_2/ (SECOND)
src/satchange/    the library — all reusable logic belongs here
src/scripts/      install_hpo_v3.py (rewrites package sources!), calculate_class_balance.py (broken)
tests/            pytest suite (CPU, synthetic data)
artifacts/        outputs; git-ignored except artifacts/final_models/** (Git LFS for weights)
data/             datasets; git-ignored (only data/.gitkeep tracked)
source_backups/   tracked pre-HPO-v3 source backups (not part of the package)
changes.md        ~315 KB historical change log/diffs — read selectively, never rewrite
```

The working copy is a **git sparse checkout** (`configs docs notebooks scripts src tests` +
root files); `artifacts/`, `source_backups/` and `data/.gitkeep` are tracked but not
materialized locally, and no local `data/` directory exists (datasets are not present here).

### Core modules (`src/satchange/`)

| Module | Responsibility |
|---|---|
| `utils.py` | `load_config(path)` — strict YAML loader (root must be a mapping). |
| `execution.py` | `resolve_device` (never silently falls back to CPU), `precision_context` (`fp32`/`bf16` only), `seed_everything` (deterministic algorithms, `CUBLAS_WORKSPACE_CONFIG=:4096:8`), RNG capture/restore, DDP helpers `rank`, `world_size`, `unwrap`, `on_rank_zero`, `gather_objects`. |
| `data/levir_cd.py` | LEVIR-CD gdown download/extract/validate, `LEVIRCDDataset`, `create_dataset_dataloader`. Sample: `before`/`after` float32 `[3,H,W]` in [0,1], `mask` float32 `[1,H,W]` in {0,1}, `sample_id`. RGB bilinear resize, mask nearest. No mean/std normalization. Splits `train/val/test`, subdirs `A/B/label`. |
| `data/transforms.py` | `RandomFlipRotate` — synchronized flips + rot90 (training only). |
| `losses.py` | `DiceLoss`, `BCEDiceLoss(bce_weight, dice_weight, smooth=1.0, pos_weight=None)` on logits. |
| `metrics.py` | `BinaryChangeMetrics` — global confusion counts (never batch-averaged F1). |
| `models/factory.py` | `MODEL_REGISTRY` (binary only), `create_change_model`, `create_task_model`, `model_class_for_task`, `BINARY_TASK` (binary task only since R2-P07-T01). |
| `training.py` | Shared engine for **both** tasks: `train_one_epoch`, `evaluate_thresholds`, `fit_binary_change_model(task=...)`, `EarlyStopping`, atomic `_save_checkpoint`, checkpoint discovery/loading, best-weight recovery. Ends with `HPO_TRAINER_PATCH_VERSION = 1` (marker for `install_hpo_v3.py`). |
| `optimization.py` | Worker-level/legacy HPO: `_expand_search_space`, `_build_trial_parameters` (`model__<param>`, `loss_weights`), `_create_scheduler` (`none`/`reduce_on_plateau`, mode=max), `run_parameter_search` (process pool per GPU, CUDA-OOM retry). |
| `hpo_core.py` | Model-agnostic HPO v3 controller; stdlib only **by design** (no torch/satchange imports). |
| `hpo_backend.py` | `SatelliteBackend` — the only v3 part that knows the training/checkpoint schema (`TRAIN_KEYS`, `SEMANTIC_TRAIN_KEYS`, `GENERATED_KEYS`); legacy runs are read-only. |
| `automated_hpo.py` | Public v3 API: `run_automated_parameter_search`, `request_hpo_pause`. |
| `final_benchmark.py` | `run_final_benchmark` → `artifacts/final_models/<variant>/`; validation before test, no threshold search on test, latency benchmark, transactional output. |
| `run.py` | CLI `python -m satchange.run {train,hpo,probe}`; `resolve_parameters` (YAML → flat HPO schema), `reconstruct_checkpoint`, `run_training`, `probe_training_step`. |
| `experiments.py` | `load_fc_siam_diff_experiment` (FC-Siam-Diff-only; notebooks 01/02). |
| `semantic/` | `roi.py` (OpenCV components → crops; keep byte-identical — notebooks 08–11 hash it), `second.py` (SECOND install/discovery/decoding, `prepare_second_splits` manifests), `data.py` (hash/RGB/map/JSON helpers), `inference.py` (`detector_probability_map`). |
| `data.py`, `loader.py`, `models/blocks.py` | **Dead/placeholder.** `data.py` is shadowed by the `data/` package; `loader.py` is commented out; `blocks.py` is imported nowhere. |

### Execution flows

- **Binary:** `YAML → load_config → resolve_parameters → create_dataset_dataloader` (train with
  `RandomFlipRotate`, val without) `→ create_change_model → BCEDiceLoss + AdamW`
  (+ optional ReduceLROnPlateau) `→ fit_binary_change_model → evaluate_thresholds` each epoch
  (best validation F1 over the threshold grid; first threshold wins ties) `→ checkpoints`.
  Forward contract: `model(before, after) -> raw logits [B,1,H,W]`.
- **Semantic (Research 2):** no semantic model/task exists yet (TinySiamese removed, R2-P07-T01);
  the `task` argument of the training/HPO/benchmark APIs accepts only the binary task.
- **Dependency direction:** `hpo_core` ← `automated_hpo` → `hpo_backend` → `optimization` →
  `training` → `execution`/`metrics`; `run.py` and `final_benchmark.py` sit on top. Research
  notebooks 06–11 import **private** helpers `final_benchmark._runtime`, `_load_checkpoint`,
  `_instantiate` — renaming them breaks those notebooks.

### Entry points (verified)

- `python -m satchange.run {train,hpo,probe}` — `train` and `hpo` read
  `config["training"]["thresholds"]`, which **no shipped YAML defines**; it must be added to the
  config (tests do this) or the run fails with `KeyError`. `hpo` writes to
  `artifacts/hyperparameter_search/<run-name>`; `probe` writes `artifacts/profiling/<uuid>.json`.
- Python APIs used from notebooks: `run_automated_parameter_search`, `request_hpo_pause`,
  `run_final_benchmark`, `run_parameter_search` (legacy), `fit_binary_change_model`.

### Fragile components / known technical debt

- `training.py` (~1.2k lines) is the single engine for binary + semantic tasks, DDP, resume,
  early stopping and checkpoints; `src/scripts/install_hpo_v3.py` patches it by AST/text anchors
  (`HPO_TRAINER_PATCH_VERSION`). Refactors can break the patcher and legacy-checkpoint resume.
- Two HPO stacks coexist: legacy `optimization.run_parameter_search` (notebooks 01–03, CLI `hpo`)
  and v3 `automated_hpo` → `hpo_core` → `hpo_backend` (Heavy, semantic). v3 reuses private
  `optimization`/`training` helpers (`_build_trial_parameters`, `_create_scheduler`,
  `_save_checkpoint`, …) — treat them as internal API.
- Determinism setup is duplicated (`execution.seed_everything` vs `_run_single_trial`).
- Dead code: `satchange/data.py`, `loader.py`, `models/blocks.py`; FC-Siam-Diff defines its own
  `ConvBlock`. `src/scripts/calculate_class_balance.py` lacks its `DataLoader` import.
- Research analysis code is duplicated inline across notebooks 06–11 rather than in `src/`.
- No CI; `tests/test_data.py` is empty; GPU paths (BF16, NCCL, native 1024) are not covered by
  default tests.

## 3. Technology Stack (verified)

| Item | Evidence / status |
|---|---|
| Python `>=3.11,<3.13` | `pyproject.toml`; local `.venv` = 3.12.14; notebook metadata mostly 3.11.6 |
| PyTorch (unpinned), torchvision | local `.venv`: `torch 2.14.1+cpu` — **CUDA unavailable on this machine** |
| CUDA / GPU | Code supports CUDA, BF16 autocast, NCCL DDP (`torchrun`), multi-GPU independent HPO. Target GPU type **unknown**; `max_trial_memory_mb=80_000` default and docs mentioning A100 suggest 80 GB A100s **(unverified)** |
| Data / imaging | numpy, Pillow, opencv-python (`semantic/roi.py`), gdown (downloads); pandas/matplotlib in notebooks; scipy used in research notebooks (not declared) |
| Declared but unused in `src/` | albumentations, torchmetrics, timm, einops, pydantic, rich, tqdm, scikit-learn |
| Configuration | Plain YAML (PyYAML) → dicts. No Hydra/OmegaConf/pydantic models, no inheritance |
| Experiment tracking | Filesystem only (JSON/CSV/checkpoints). No W&B/MLflow/TensorBoard |
| Testing / QA | pytest (`testpaths=["tests"]`, `addopts="-ra -q"`), ruff (line 100; E,F,I,B,UP; py311), mypy (`disallow_untyped_defs`); nbformat/nbclient/ipykernel for notebook tests |
| CI / pre-commit | **None** (no `.github/`, no `.pre-commit-config.yaml`, despite `pre-commit` in dev extras) |
| Git LFS | `*.pt, *.pth, *.ckpt, *.onnx` |

## 4. Development Guidelines

- **Reusable logic lives in `src/satchange/`.** Reuse `create_dataset_dataloader`,
  `create_change_model`, `BCEDiceLoss`, `fit_binary_change_model`, `evaluate_thresholds`,
  `run_parameter_search`/`run_automated_parameter_search`, `run_final_benchmark`. The project
  deliberately has **no per-model trainer/HPO/dataset**; do not add parallel pipelines.
- **Adding a binary model:** `nn.Module` with `forward(before, after) -> [B,1,H,W]` logits;
  validate constructor args with `ValueError`; register in `MODEL_REGISTRY`; export in
  `models/__init__.py`; keep constructor parameters JSON-serializable (they are stored as
  `model_parameters` in checkpoints and HPO identities, and bound via `inspect.signature`).
- **Style (observed):** full type hints, `pathlib.Path`, built-in generics and `X | None`,
  one-line docstrings, explicit `ValueError`/`RuntimeError`/`FileNotFoundError` with descriptive
  messages, keyword-only (`*`) arguments for new options, `zip(..., strict=True)`. Older modules
  (`losses.py`, `levir_cd.py`, parts of `training.py`/`optimization.py`) use a vertical
  one-argument-per-line style; newer modules (`execution.py`, `run.py`, `hpo_*`, `semantic/`)
  are compact. **Match the style of the file you edit**; do not reformat unrelated code.
- **Naming:** snake_case modules/functions, PascalCase classes, private helpers prefixed `_`,
  UPPER_CASE module constants; registry names are lowercase snake_case.
- **Language:** new code, comments and docs in English. Existing Russian comments
  (`models/blocks.py`), Russian `docs/heavy_model.md` and Russian notebook markdown stay as is.
- **Logging:** `print(..., flush=True)` (rank-0 only in training) and
  `warnings.warn(..., stacklevel=2)`. No `logging` setup exists — do not introduce one unasked.
- **Fail loudly:** no silent CPU fallback, no silent restart when checkpoints are unreadable, no
  implicit label/color fallbacks, unknown parameters are rejected. Preserve this behaviour.
- **No side effects on import**; actions only through explicit calls/CLI.
- **Backward compatibility:** keep `__all__` exports, the `fit_binary_change_model` name (used by
  both tasks), checkpoint keys, HPO parameter keys, `model__` routing, `PHASES`, and private
  helpers imported by notebooks. Old checkpoints/legacy HPO folders must remain loadable.
- **Atomic writes:** follow the existing temp-file + `fsync` + `os.replace` pattern
  (`training._save_checkpoint`, `hpo_core.write_json`) for any new persistent output.
- **Line endings/encoding:** many files are CRLF, some start with a UTF-8 BOM. Do not normalize.
- **Dependencies:** no additions/upgrades without explicit approval; do not assume declared but
  unused packages (timm, einops, albumentations, …) are integrated.

## 5. ML Engineering Rules

- **Tensor contracts:** float32 RGB `[B,3,H,W]` in [0,1] (no ImageNet normalization; BIT rejects
  `pretrained_backbone=True` for this reason, Heavy raises `NotImplementedError`); masks
  `[B,1,H,W]` in {0,1}. `SiamMiTB3Diff` requires H,W ≥ 32 and divisible by 32. Losses/metrics
  raise on shape mismatch — keep and extend such checks.
- **Precision:** only `"fp32"` (default) and `"bf16"` (CUDA with BF16 support). Logits are cast
  with `.float()` before loss/metrics. There is **no FP16/GradScaler**. The YAML key
  `runtime.mixed_precision` (FC-Siam, changeformer) is **not read by any code**.
- **TF32:** set only on the semantic path from config `tf32` (default false);
  `final_benchmark._runtime` disables TF32/autocast during evaluation and restores flags.
- **Determinism:** `seed_everything` and HPO workers enable
  `torch.use_deterministic_algorithms(True)`, cudnn deterministic, benchmark off. New ops must be
  deterministic-compatible. Loaders use explicit `torch.Generator` seeds; DDP uses `seed + rank`.
  Resume restores Python/NumPy/Torch/CUDA/loader RNG at epoch boundaries.
- **Devices & memory:** use `resolve_device`; `torch.load(..., map_location="cpu")` then move;
  keep best weights on CPU (`_copy_weights_to_cpu`), never a second GPU model copy. HPO workers
  release memory after trials and retry transient CUDA OOM; never auto-reduce resolution/depth
  on OOM. Heavy native 1024 depends on `use_checkpointing` (activation checkpointing).
- **DDP:** only `torchrun ... -m satchange.run train --ddp`; HPO/probe refuse to run inside DDP;
  exact resume requires identical world size, precision, task and architecture parameters.
- **Thresholds:** chosen on validation only (documented grid 0.05…0.95 step 0.05, passed
  explicitly). Test evaluation uses the frozen validation threshold. Never tune on test.
- **Checkpoint format** (`training.py`, `checkpoint_schema_version: 2`): dict with `epoch`,
  `model_name`, `run_name`, `thresholds`, `stop_reason`,
  `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `config`
  (incl. `model_parameters`), `best_metrics`, `current_metrics`, `history`, `rng_states`,
  `precision`, `world_size`, `early_stopping`, optional `best_model_state_dict`/`task`/`runtime`.
  Saved atomically on rank 0. Loaded with `weights_only=False` → **only trusted, own files**.
  `deployment_weights.pt` = plain CPU FP32 `state_dict`.
- **Reconstruction** (`run.reconstruct_checkpoint`) strictly rebuilds from `model_name` +
  `config["model_parameters"]` with `strict=True`. Renaming parameters/modules or changing
  constructor defaults silently changes or breaks old checkpoints and final models — avoid, or
  add an explicit compatibility path plus a regression test.

## 6. Experiment Management

- **Binary YAML schema** (`configs/{fc_siam_diff,bit,siam_mit_b3_diff}_*.yaml`):
  `project{name, seed}`, `runtime{device: auto|cpu|cuda[:N], precision}`,
  `model{name, parameters}`, `data{image_size|null, batch_size, num_workers}`,
  `training{epochs, learning_rate, weight_decay, loss{bce_weight, dice_weight, pos_weight},
  scheduler, thresholds*}`, `output{directory}`, `dataset{name, download_url,
  download_directory, prepared_directory}` (*`thresholds` must be added by the caller).
  `resolve_parameters` maps it to the flat HPO schema (`model_parameters, seed, image_size,
  batch_size, num_workers, learning_rate, weight_decay, scheduler, bce_weight, dice_weight,
  positive_weight, precision`). No inheritance; CLI `--device/--precision/--epochs` override.
- **Known config inconsistencies — document, do not silently "fix":**
  `bit_full.yaml` uses `image_size: 1024` (other Full configs use `null`); `bit_full` and
  `bit_resized` share `output.directory: artifacts/bit`; `siam_mit_b3_diff_full_resolution.yaml`
  outputs to `artifacts/final_models/heavy_resized` and sets `use_checkpointing: false` although
  `docs/heavy_model.md` states it is enabled; `changeformer.yaml` and `research.yaml` are not
  used by any code (no ChangeFormer model exists); `output.checkpoint_name` and
  `runtime.mixed_precision` are ignored.
- **Search spaces:** dict of lists (Cartesian product) or list of dicts (union of grids);
  `model__<param>` targets constructor args; `loss_weights: [[bce, dice], ...]` must not be
  combined with separate weight keys. Example: `configs/siam_mit_b3_diff_search.json`.
- **HPO v3 rules (`automated_hpo.py`):** `stage_epochs` are **absolute** targets (default
  `[40, 60, 80]`; re-running never adds epochs); phases `hpo → seeds → final`; seeds default
  `(42, 123, 777)`; any change of model/data/code/search definition requires a **new
  `search_name`** (reuse old results via `reuse_from`); `pipeline_protocol.json` and
  `.hpo_v3.lock` guard each search directory; pause with `request_hpo_pause` (markers
  `PAUSE_AFTER_EPOCH` / `STOP_AFTER_ACTIVE_TRIALS`); missing `last.pt` for a matching run is an
  error, not a silent restart. Never edit the package while an HPO process imports it.
- **Early stopping:** `patience=None` = fixed budget; monitor `validation_f1` (binary) or
  `selection_score` (semantic: mean of before/after macro-F1 on val; earlier epoch wins ties).
- **Output layout:**
  - training: `<output.directory>/<model_name>/<run_name>/{best.pt, last.pt,
    checkpoints/<model_name>_epoch_NNNN.pt}`; `run_<uuid12>` names; existing run directories are
    never reused for fresh runs (`FileExistsError`); tiny-overfit runs go under `tiny_overfit/`.
  - HPO: `artifacts/hyperparameter_search/...` (e.g. `heavy/heavy_full`), seed runs `seed_<seed>`.
  - final: `artifacts/final_models/<variant>/` (`seed_<seed>_best.pt`, `deployment_weights.pt`,
    `benchmark.json`, `metadata.json`, `training_results.json`, `validation_results.json`,
    `final_test_results.json`); variants `{light,medium,heavy}_{full,resized}`; summary CSVs
    `final_models_metrics.csv`, `final_models_gpu_cost.csv`. Existing output is replaced only with
    `overwrite=True` (previous folder preserved as backup).
  - research: `artifacts/hypothesis_tests/<study>/` (caches, `protocol_frozen.json`, reports);
    profiling: `artifacts/profiling/<uuid>.json`.
- **Reproducibility metadata** is recorded automatically (git revision, torch/CUDA versions,
  world size, precision, dataset root, thresholds). Keep recording it for new outputs.
- **Historical results are immutable.** New experiments → new directories/`search_name`s;
  compare against existing JSON/CSV instead of regenerating them.

## 7. Jupyter Notebook Guidelines

- **Location (since 2026-11-06):** all notebooks below live in `notebooks/research_1/`; Research 2
  notebooks go to `notebooks/research_2/` (`notebooks/README.md`). `tests/test_notebooks.py`
  discovers notebooks recursively.
- **Inventory:** `01_fc_siam_diff`, `02_fc_siam_diff_full_hpo` (legacy `run_parameter_search`);
  `03_bit_full`, `03_bit_resized` (BIT training/HPO; `03_bit_resized` also contains inline
  benchmark helpers); `04_mit_full`, `04_mit_resized`, `Heavy_Full_HPO_v3_metadata_fix`,
  `heavy_full_staged_resume`, `heavy_resized_staged_resume` (Heavy v3 HPO + final benchmark);
  `05`–`11` "Research I": hypotheses H1–H5, ROI model selection (`08_levir_roi_model_selection`
  and its executable `_v2`), Medium topology
  and matched cost (`09`), ROI generator ablations (`10`), grid tile sizes for all six models
  (`11`). Research notebooks perform **no training**; they read `artifacts/final_models/` and
  `artifacts/hypothesis_tests/` caches and are chained (07–11 reuse notebook 06 code and/or its
  cached probability arrays and reports).
- Notebooks 06–11 embed a large duplicated analysis toolkit (`binary_metrics`,
  `match_components`, `Experiment`, `Settings`, …), partly SHA-checked against notebook 06 —
  editing those cells can break the hash checks of later notebooks.
- Edit **source cells only**; preserve metadata, kernelspec, cell ids/order and existing outputs
  (outputs are historical evidence). Do not clear, re-run or re-save notebooks unasked.
- Expensive operations sit behind flags (`RUN_DATA_SETUP`, `RUN_PARAMETER_SEARCH`,
  `RUN_HPO_BLOCKS`, `RUN_TEST`, `RUN_TEST_EVALUATION`, `RUN_MEDIUM_SEEDS`, …). New expensive cells
  must be guarded with default `False`. Several research notebooks default flags to **`True`**
  (incl. `RUN_TEST=True` in 06/07) — never "Run All" them as a check.
- Heavy/HPO notebooks require: stop old HPO processes → change package → restart kernel.
- Move reusable notebook logic into `src/satchange/` only on request, keeping the notebook working.
- Docs/tests reference notebooks absent from this checkout (`03_heavy.ipynb`, `02_bit.ipynb`).
- `ruff check .` also lints notebooks (~1.6k pre-existing findings); never reformat notebooks
  to satisfy it.

## 8. Testing and Validation

Verified commands (repo root, PowerShell; tests use relative `configs/...` paths):

```powershell
.\.venv\Scripts\python -m pytest                          # full CPU suite (~136 tests, minutes)
.\.venv\Scripts\python -m pytest tests/test_heavy.py      # focused file
.\.venv\Scripts\python -m ruff check src tests --output-format concise
.\.venv\Scripts\python -m mypy src
.\.venv\Scripts\python -m compileall -q src tests
.\.venv\Scripts\python -m satchange.run --help
git diff --check
```

**Baseline (2026-11-01, this machine, torch 2.14.1+cpu)** — compare against it, do not
introduce new failures/findings in touched files:
- pytest (updated 2026-11-06): 134 passed, 3 skipped (2 CUDA opt-in, 1 missing `03_heavy.ipynb`),
  0 failed. `mypy src`: 40 pre-existing errors in 8 files (unchanged vs HEAD `b72cd6c`).
- `ruff check src tests`: 41 pre-existing findings (mostly E501, I001, B905, 1 F821).
  `ruff check .` adds ~1.6k notebook findings.
- `mypy src`: fails on `src/scripts/calculate_class_balance.py` (`DataLoader` undefined).

Test map: `test_metrics`, `test_models`, `test_optimization`, `test_execution` (2-process Gloo
DDP, launcher, preprocessing, legacy checkpoint regression), `test_heavy` (architecture,
attention reference, fit/resume/HPO), `test_semantic`, `test_semantic_hpo` (v3 controller with
mocked CPU "GPU" queue, final benchmark), `test_notebooks` (syntax of every notebook + fresh-kernel
safe paths). `tests/test_data.py` is empty.

- **Minimum after a change:** focused pytest for touched modules + ruff/mypy on touched files.
- **Full pytest** when touching `training.py`, `optimization.py`, `hpo_*`, `automated_hpo.py`,
  `run.py`, `execution.py`, `final_benchmark.py`, `models/factory.py`, model constructors,
  checkpoint formats, or `semantic/`.
- **GPU tests are opt-in:** `SATCHANGE_TEST_CUDA_DEVICE=cuda:N` (+ `SATCHANGE_TEST_NATIVE=1` for
  native 1024). Use only when CUDA/BF16/memory paths change **and** the user approves; this
  machine has no CUDA.
- Notebook edits: `tests/test_notebooks.py` validates syntax of all notebooks (no execution of
  research notebooks).

## 9. Safety and Data Protection

- **Never modify, delete, move or overwrite:** `data/**`, `artifacts/**` (especially
  `final_models/`, `hyperparameter_search/`, `hypothesis_tests/`), `*.pt/*.pth/*.ckpt/*.onnx`,
  `source_backups/**`, notebook outputs, `changes.md`, `.venv/`.
- Do not open binary checkpoints or large notebook outputs wholesale; inspect notebook source
  cells programmatically.
- Weights are Git LFS; `hpo_backend._load` rejects LFS pointer files. `artifacts/` is outside the
  current sparse checkout — do not change sparse-checkout settings unasked.
- `.env`/`.env.*` are git-ignored (`.env.example` is empty). Never read, print, copy or commit
  secrets; never place credentials in configs/docs/notebooks. Public dataset URLs are not secrets.
- Without an explicit request, do **not** run: dataset downloads (`prepare_levir_cd`, SECOND
  bootstrap), training, HPO, `probe`, `run_final_benchmark`, test-set evaluation, research
  notebooks, or `src/scripts/install_hpo_v3.py` (it rewrites `src/satchange/*.py`).
- Git: no history rewriting, force operations or pushes; no mass reformatting or line-ending
  changes. The working tree may hold the user's uncommitted edits (e.g. notebooks) — leave them.

## 10. AI Agent Development Workflow

1. Read this file and relevant `docs/*.md`; identify affected modules, configs, tests, notebooks.
2. Inspect existing implementations and all call sites (incl. notebooks importing private helpers)
   before adding code; prefer extending shared components.
3. Check constraints: checkpoint/HPO-identity compatibility, determinism, precision/device
   handling, rank-0/DDP behaviour, validation-vs-test separation, atomic writes.
4. State the plan and assumptions; mark unknowns instead of guessing APIs or configs.
5. Make minimal, focused, reversible changes; add/adjust CPU tests with synthetic data.
6. Run the cheapest sufficient validation (§8); never start GPU/training workloads unasked.
7. Review `git status`/`git diff` for unrelated edits, encoding/line-ending churn, artifacts.
8. Report changes, rationale, validation results versus the baseline, and open risks.
