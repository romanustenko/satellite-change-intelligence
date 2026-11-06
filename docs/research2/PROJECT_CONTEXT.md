# Research 2 — Project Context (persistent scientific memory)

**Document status:** initialized 2026-11-06 from a read-only inspection of the live repository
(branch `main`, HEAD `b72cd6c`) plus the user's decisions of 2026-11-06.
**Companion documents:** [`DECISION_LOG.md`](DECISION_LOG.md) ·
[`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) · [`PROGRESS.md`](PROGRESS.md) ·
[`EXPERIMENTAL_PROTOCOL.md`](EXPERIMENTAL_PROTOCOL.md) (draft, PROPOSED).

This file is git-tracked on purpose: the root `AGENTS.md` is listed in `.gitignore`, so it is a
local pointer only. Everything an agent needs to continue Research 2 must be here or in the
companion files.

### Evidence labels used in all Research 2 documents

| Label | Meaning |
|---|---|
| **VERIFIED CURRENT** | Checked in the live repository on the date given (file/symbol/line cited). |
| **VERIFIED HISTORICAL** | Measured result read from a tracked artifact (e.g. `artifacts/final_models/**` via `git show`). |
| **REPORTED** | Stated in the hand-off or earlier AI summaries; source archive not available here; **not re-verified**. |
| **USER-APPROVED** | Explicit user decision, recorded in `DECISION_LOG.md` with date. |
| **PROPOSED** | Recommendation awaiting user approval. |
| **UNKNOWN** | Could not be determined from available sources. |

Line numbers refer to HEAD `b72cd6c` and drift with edits; re-check before relying on them.

---

## 0. Mandatory context restoration (every new session)

1. Read the root `AGENTS.md` (local, untracked) if present — repository conventions, commands,
   safety rules.
2. Read this file completely.
3. Read `DECISION_LOG.md`. Approved decisions are binding; never change one silently. If new
   evidence contradicts one, record the conflict and propose reconsideration.
4. Read `PROGRESS.md`: current milestone, pending approvals, blocked tasks.
5. Locate the assigned task in `IMPLEMENTATION_PLAN.md` (`R2-Pxx-Tyy`); check its dependencies,
   approval requirement and status.
6. Inspect the relevant current source code. Line numbers and claims in these documents may be
   stale.
7. Confirm the implementation still matches the documented state; record any drift.

Before writing code, state the chain explicitly:

**Global objective → current milestone → scientific requirement → existing implementation →
assigned task → expected evidence.**

If the scientific purpose of a change cannot be established, stop and report the uncertainty.

### Working policy (USER-APPROVED 2026-11-06, D-028 + D-030 — supersedes older "ask before each technical step" rules)

**D-030 (autonomy v2):** the agent is the lead autonomous engineer; approval gates are
verification/documentation checkpoints. Ask the user only for changes to the research question,
hypotheses, evaluation methodology, dataset evaluation splits or final comparison rules, for
substantial GPU expenditure, and before destructive operations on historical artifacts or
irreversible remote changes. **D-032:** work as a single autonomous agent (no subagents); avoid
repeated audits, excessive planning and unnecessary documentation; reuse existing context. The
rules below remain in force where they do not conflict.

- The agent is the primary implementation engineer: it chooses technical approaches, creates
  ordinary files/tests/docs, fixes obvious problems and performs safe, reversible, local git
  operations for approved tasks **without asking**.
- After each task: run relevant tests/validation → update `PROGRESS.md` and the task status →
  short report (**Completed / Why / Verified / Remaining / Next**, in Russian) → continue with the
  next approved, unblocked task from `IMPLEMENTATION_PLAN.md`. If a task is blocked, record the
  blocker and continue with another independent task.
- Find discoverable answers yourself (repository, docs, tests, git history, external technical
  docs). If the GPU server is inaccessible, prepare the commands, record what remains to run, and
  never claim unrun checks as done; do not repeatedly ask the user to run them.
- **Scientific decisions stay with the user:** research question, hypotheses/protocol, split
  definitions, primary metrics, model-selection criteria, comparison rules, confirmatory budgets.
  For an unresolved one, document the recommendation and consequences and continue with
  independent work. No substantial GPU training without an approved budget (G8).
- Integrity: never overwrite Research 1 artifacts/results, never bypass HPO compatibility checks,
  never delete unrelated files, never push/force-push/rewrite remote history without permission,
  never fabricate results, preserve uncommitted user changes, one shared `src/satchange/`.
Conversation history and auto-compaction are not a substitute for these files: after any
meaningful work, update `PROGRESS.md`, the task status in `IMPLEMENTATION_PLAN.md`, and
`DECISION_LOG.md` when a decision was made.

---

## 1. Overall purpose

The project studies change intelligence on bi-temporal satellite image pairs. It has two
scientifically distinct but connected studies sharing one codebase (`src/satchange/`):

- **Research 1** — binary change detection ("WHERE did something change?") on LEVIR-CD.
- **Research 2** — semantic change detection ("WHAT changed into what?") on SECOND, comparing a
  single-stage system with a two-stage WHERE → ROI → WHAT system.

## 2. Scientific motivation

A semantic model applied to the whole image pays the full cost on every pixel, while in many
scenes only a small fraction of the area changes. A two-stage system could spend semantic
computation only where a cheap binary detector finds change. Whether this helps in practice is
open: stage-1 misses are unrecoverable, crops cost overhead, many small crops can be slower than
one full-image pass, and the benefit should depend on change density. Research 2 measures this
trade-off honestly; **a negative result (no advantage) is a valid outcome.**

## 3. Research 1 — objectives and completed investigations

**Status: USER-APPROVED (2026-11-06) — Research 1 is ongoing, not frozen.** The user continues
to investigate hypotheses. Research 2 may reuse Research 1 checkpoints, but only through the
versioned selection/freeze policy in D-010 (see `DECISION_LOG.md`).

Six finalized binary detectors (VERIFIED HISTORICAL, `artifacts/final_models/` in git HEAD):

| Variant | Registry name | Input | Notes |
|---|---|---|---|
| `light_full` | `fc_siam_diff` | native 1024×1024 | `base_channels=32`; seed checkpoints 123, 777, 2024; deployment seed 2024, threshold 0.8 |
| `light_resized` | `fc_siam_diff` | whole scene → 256×256 | `base_channels=16`; seeds 42, 123, 777; deployment seed 42, threshold 0.7 |
| `medium_full` | `bit` | 1024 | seeds 42, 123, 777 |
| `medium_resized` | `bit` | 256 | seeds 42, 123, 777 |
| `heavy_full` | `siam_mit_b3_diff` | 1024 | seeds 42, 123, 777 |
| `heavy_resized` | `siam_mit_b3_diff` | 256 | **only `seed_42_best.pt`** |

Per-seed metrics, recorded training GPU-hours and provenance: `CHECKPOINT_INVENTORY.md`.
Historical benchmark hardware (VERIFIED HISTORICAL, `benchmark.json`): NVIDIA A100-SXM4-80GB,
CUDA 13.0, torch 2.12.1+cu130 (where recorded).

"Resized" = downsampling the whole scene, not cropping. Architecture parameters also differ
between Full and Resized (e.g. FC-Siam `base_channels` 32 vs 16; BIT embeddings reportedly
64 vs 32), so Full/Resized differences cannot be attributed to resolution alone.

Light-full seed spread is large (VERIFIED HISTORICAL, `light_full/metadata.json`): validation F1
0.8235 (seed 123) vs 0.7695 (seed 777) with the same hyperparameters. Seed variance is a real
effect for Research 2 statistics.

Research workflow (VERIFIED CURRENT; notebooks in `notebooks/research_1/` since 2026-11-06):
training/HPO notebooks 01–04 and `heavy_*`; research
notebooks 05–11 analyse the finalized models without retraining. Their outputs live in
`artifacts/hypothesis_tests/` on the GPU server (not tracked by git, not available locally).

## 4. Historical H1–H11 findings

**All numbers in this section are REPORTED / NOT REVERIFIED.** Source: the hand-off document
`PROJECT_NEW_CHAT_HANDOFF_2026-10-31.md` (outside the repository; user's local copy), which cites
ZIP archives (`H1_H5_v1_results.zip`, `H1_H5_followup_v2_results.zip`,
`H08_roi_model_selection_v2_results.zip`, `H09_medium_topology_matched_cost_results.zip`,
`H10_roi_generator_topology_ablations_results.zip`, `H11_grid_tile_size_all_models_results.zip`,
`08_medium_diagnostic_context.zip`). **None of these archives is available in this workspace.**
Use the numbers as historical orientation and starting points only — never as Research 2
settings without re-measurement on SECOND.

| ID | Claim (short) | Evidence source (reported) | Data / split | Limitations |
|---|---|---|---|---|
| H1 | Resolution effect on small changes is metric/threshold/family dependent; no clean causal effect | H1_H5_v1 | LEVIR test (128 pairs) | Architecture parameters co-vary with resolution |
| H2 | Heavy has better boundary quality; does not explain the whole F1 gain | H1_H5_v1 | LEVIR test | Boundary importance may shrink with crop context |
| H3 | Light and Heavy errors are partly complementary; OR ensemble only worth it if it wins on a complete-system Pareto | H1_H5_v1, H08 | LEVIR test/val | OR costs an extra detector forward |
| H4 | Strong inversion Pixel F1 vs strict component recall for Medium | `reports/H4_test_rankings.csv` | LEVIR test, 7 027 GT components | Test set already used → not confirmatory |
| H5 | Quality and cost must be evaluated together | H1_H5_v1 | LEVIR test | Old latencies are forward-only |
| H08 | Medium Full component recall pattern repeats across seeds 123/42/777 (0.646/0.646/0.629); Medium Resized 0.072/0.066/0.059 | `medium_all_seed_validation_metrics.csv` | LEVIR val | 3 seeds do not identify hyperparameter effects |
| H09 | At ≈97.5 % ROI coverage Medium Full needs ≈34.4 crops/pair vs 44.4 (Light Full), processed area ≈54.8 % vs 70.2 % | H09 archive | LEVIR val | Part of coverage comes from rectangular-crop context, not localisation quality |
| H10 | ROI policy matters more for workload than mask morphology (e.g. Medium Full thr 0.70: grid-128 ≈16.9 crops/pair, area ≈26.4 %); opening 5×5 raises strict recall but also fully-missed components; **watershed not executed** (`scikit-image` missing) | `10_run_manifest.json` | 32 internal audit pairs of LEVIR val | Internal audit ⊂ old validation; not independent |
| H11 | Bare 32-px tiles: very low single-crop coverage (Medium Full thr 0.70: 5.16 %), ≈104 crops/pair; grid256+margin16 dev-selected for Full models (audit single-crop ≥95 % coverage ≈98.3–98.7 %, ≈7–7.7 crops/pair, ≈52–57 % area) | `11_selected_audit_results.csv`, `11_run_manifest.json` (`validation_only: true`, `uses_test: false`) | 32 dev + 32 audit pairs of LEVIR val (seed 1029) | Exploratory; selection minimised a source-area proxy, not WHAT GPU time |

Reported H4 test values (thresholds selected earlier on validation): Pixel F1 / strict
component recall (IoU≥0.5) — Light Full 0.841 / 0.733, Medium Full 0.848 / 0.593, Heavy Full
0.864 / 0.753, Light Resized 0.759 / 0.278, Medium Resized 0.706 / 0.050, Heavy Resized
0.748 / 0.188. **These conflict with tracked artifacts for at least Light Full** (§21, item 1).

Reported working WHERE shortlist from the hand-off (PROPOSED, **not approved**, D-012):
Medium Full (efficient ROI contender), Heavy Full (binary-quality reference), Light Resized
(low-cost baseline); Light Full as a controlled comparison; OR Light+Heavy optional. Rankings
must be re-derived on SECOND.

Reported candidate ROI policies for SECOND measurement (PROPOSED): grid 256 + context 16;
grid 128 + context 16; possibly adaptive 64 + context 16. Not frozen.

## 5. Methodological limitations of Research 1

- LEVIR test was analysed in H1–H5 and follow-ups; later LEVIR findings are not independent
  confirmation. H08–H11 are mostly validation-only; the 32-pair audit is part of the old
  validation split.
- HPO budgets, search spaces and reproducibility settings differed between model families and
  evolved during development; final quality alone is not a fair comparison.
- Historical latencies (`final_models_gpu_cost.csv`, VERIFIED HISTORICAL) are synchronized
  forward-only, batch 1, on the server GPU; preprocessing, transfers, ROI extraction, WHAT and
  merging are excluded. They cannot prove an end-to-end speed-up.
- LEVIR-CD is building-centric; SECOND has six land-cover classes. LEVIR rankings do not transfer
  without adaptation and re-evaluation.

## 6. Unsuccessful experiments and lessons (REPORTED)

- `08_levir_roi_model_selection.ipynb` (v1) was a diagnostic stub (~5–10 s run); `_v2` is the
  real ROI sweep. Do not use v1 as the experiment of record.
- The original H11 notebook failed at `DataFrame.pivot` (duplicate dev+audit rows); the fixed
  notebook filters `internal_split == 'audit'` first (`11_levir_grid_tile_size_all_models_fixed.ipynb`
  is the only H11 notebook in the repository — VERIFIED CURRENT).
- H10 watershed never ran; topology alternatives are not exhausted.
- Early discussions mixed union coverage, crop area and per-object detection; keep them apart (§14).
- The removed TinySiamese ROI classifier line of work (notebooks deleted in commits `60f4fca`,
  `b72cd6c`) is abandoned by user decision (D-006). Its design (ROI-level dominant-transition
  classification) is not the Research 2 semantic model.

## 7. How Research 1 motivates Research 2

Research 1 showed that ROI suitability depends strongly on mask topology and crop policy rather
than on pixel F1 alone, and that crop count, single-crop coverage and processed area trade off
against each other. The open question is whether these effects survive on SECOND and whether a
complete two-stage pipeline achieves a better semantic-quality / cost frontier than a
single-stage model.

## 8. Central Research 2 question

**USER-APPROVED (G1, 2026-11-06):** *Under which accuracy and computational-budget conditions is
a two-stage semantic change detection system preferable to a single-stage semantic change
detection model?* The exact hypotheses remain PROPOSED (`EXPERIMENTAL_PROTOCOL.md`, D-002).

## 9. Architectural concepts

- **Architecture A — single-stage:** `(T1, T2) → full-image semantic change model → semantic
  change map`.
- **Architecture B — two-stage:** `(T1, T2) → WHERE binary detector → probability/mask → ROI
  extraction → paired T1/T2 crops → WHAT semantic model → reconstruction into original image
  coordinates`. Pixels outside all ROIs are predicted as no-change; stage-1 misses must count as
  errors in the end-to-end evaluation.
- Oracle ROIs (from ground truth) are an upper-bound diagnostic, never a deployable system.
- The semantic prediction target (dense per-pixel maps vs ROI-level labels) and the single-stage
  baseline architecture are **open** (D-013).

## 10. Existing model families and their roles

| Family | File | Role in Research 2 |
|---|---|---|
| `FCSiamDiff` | `src/satchange/models/fc_siam_diff.py` | Pilot model (scratch vs LEVIR fine-tune); candidate WHERE detector. Constructor `(in_channels=3, out_channels=1, base_channels=16)`; head `output_layer` = 1×1 conv (VERIFIED CURRENT). |
| `BIT` | `models/bit.py` | Candidate WHERE detector (scratch only; rejects `pretrained_backbone=True`). |
| `SiamMiTB3Diff` | `models/siam_mit_b3_diff.py` | Candidate WHERE detector / quality reference; H,W must be multiples of 32. |
| `TinySiameseChangeClassifier` | — | **Removed 2026-11-06 (R2-P07-T01).** Only a 1-function shim `models/tiny_siamese_change_classifier.py` (`validate_unit_tensor`) remains so `semantic/roi.py` stays byte-identical for the H08–H11 hash checks. Do not restore. |

No semantic change detection model (dense, single-stage or WHAT) exists in the repository
(VERIFIED CURRENT).

## 11. Current repository architecture (VERIFIED CURRENT, 2026-11-06)

See root `AGENTS.md` §2 for the full module table. Research-2-relevant facts:

**HPO v3** — `automated_hpo.run_automated_parameter_search` (`automated_hpo.py:75`) →
`hpo_core.Experiment` → `hpo_backend.SatelliteBackend`.
- Phases `hpo → seeds → final`; `stage_epochs` are absolute targets; default seeds (42, 123, 777);
  `dry_run`, `resume_from`, `reuse_from`, `checkpoint_interval`, `max_trial_memory_mb` exist.
  **No GPU-hour budget, limit or expenditure ledger exists.**
- Scheduler `Experiment._execute` (`hpo_core.py:423–607`): one `ProcessPoolExecutor(max_workers=1,
  spawn)` per device; each job runs on exactly one device (`execution_mode="independent_hpo"`,
  `world_size=1`, `hpo_backend.py:712–725`). Idle devices are ordered by free memory from
  `torch.cuda.mem_get_info` (`hpo_backend.py:525–538`). CUDA OOM → retry on another device keeping
  `last.pt`. Markers `PAUSE_AFTER_EPOCH` / `STOP_AFTER_ACTIVE_TRIALS`; Ctrl+C touches the pause
  marker. `results.json` is rewritten atomically after each completed job (`save()`).
- Search directory guard: `pipeline_protocol.json` + `.hpo_v3.lock`. `ensure_protocol`
  (`hpo_core.py:246–268`) refuses any difference, including `backend_code_signature`.
- DDP training exists only via `torchrun ... -m satchange.run train --ddp`; HPO refuses DDP.

**Training engine** — `training.fit_binary_change_model` (shared by binary and the old semantic
task). Epoch timer `training.py:863–942` covers train + validation of the epoch (CUDA-synchronized).
Per-epoch `gpu_hours = max(epoch_seconds over ranks) × world_size / 3600` on CUDA, else 0
(`training.py:1083`).

**SECOND** — `semantic/second.py` (see §13).

**ROI / inference** — `semantic/roi.py`: `ROISettings` (defaults threshold 0.5, connectivity 8,
context 16, min crop 128, min component area 1), `extract_rois`, `align_whole_image_probabilities`.
`semantic/inference.py`: `detector_probability_map` (binary detector → probability map on the
native grid). All TinySiamese classifier/benchmark/CLI code was removed (R2-P07-T01).

**Final benchmark** — `final_benchmark.run_final_benchmark`: validation before test, frozen
validation threshold, forward-only latency; `_runtime` (`final_benchmark.py:670–696`) disables
TF32/autocast during evaluation and restores flags.

**Not present (VERIFIED CURRENT):** dense SCD metrics (mIoU / SeK / Fscd), reconstruction of crop
predictions into image coordinates, end-to-end two-stage latency measurement, a single-stage SCD
model, a semantic head for FC-Siam-Diff, a SECOND loader for binary training
(`data.levir_cd.create_dataset_dataloader` expects the LEVIR `A/B/label` layout), an
initialize-from-checkpoint (fine-tune) path in HPO (`resume_from` continues a run with identical
parameters — whether it can serve as initialization is UNKNOWN, to be checked in `R2-P08-T01`).

## 12. Reusable infrastructure

Reuse, extend, never duplicate: `fit_binary_change_model`, `evaluate_thresholds`,
`BinaryChangeMetrics`, `BCEDiceLoss`, `create_change_model`/`MODEL_REGISTRY`,
`run_automated_parameter_search` (HPO v3), `run_final_benchmark`, `execution.seed_everything`,
`capture_rng`/`restore_rng`, `resolve_device`, `precision_context`, atomic writers
(`training._save_checkpoint`, `hpo_core.write_json`), `semantic/second.py` installation and
discovery, `semantic/roi.py`, `semantic/inference.detector_probability_map`.

**USER-APPROVED (D-021):** Research 1 and Research 2 share one `src/satchange/`. No Research-2
copies of `training.py`, `hpo_core.py`, `hpo_backend.py` or other shared modules, no separate
repositories, worktrees or duplicated source trees.

## 13. Datasets and compatibility

**LEVIR-CD** (`data/levir_cd.py`, VERIFIED CURRENT): RGB float32 in [0,1], no mean/std
normalization, 1024×1024, splits `train/val/test` with `A/B/label`, binary masks.

**SECOND** (`semantic/second.py`, VERIFIED CURRENT):
- Sources: `SECOND_SOURCE = https://captain-whu.github.io/SCD/`; official archive via `gdown`
  (Google Drive id embedded in `bootstrap_second`); palette reference `PALETTE_SOURCE`.
- Classes (`SECOND_CLASSES`): non-vegetated ground surface, tree, low vegetation, water,
  buildings, playgrounds; canonical IDs 1–6, 0 = non-change (white), from exact RGB colors
  (`SECOND_COLORS`). Unknown colors fail.
- `bootstrap_second(root, archive=None, allow_download=False)`: reuses a valid layout; otherwise
  installs from a local ZIP or downloads (only with `allow_download=True`); safe extraction to a
  staging directory; writes `source.json` with a **local** SHA256 (not an author checksum).
- `discover_second(root)`: requires leaves `train|val|test` with `im1/im2/label1/label2`, exact
  stem matching; an unlabeled `test` leaf yields an empty list.
- `prepare_second_splits(root, destination, *, split_seed, val_fraction=None, test_fraction=None,
  allow_missing_test=False, source_groups=None)` (R2-P05-T02): writes a persistent split manifest;
  official labeled splits are always preserved; a missing split is an error unless the policy is
  explicit (fractions carve whole filename groups from official train with one seeded shuffle);
  filename groups claim no geographic independence; byte-identical RGB pairs and shared stems
  across splits are reported. The old ROI-coupled `prepare_second` was removed with TinySiamese.
- **Label-mapping caution:** the removed TinySiamese config (`configs/tiny_siamese_change_classifier.yaml`
  in git history) declared palette-index names (1 = water) in an order different from the
  canonical IDs (4 = water). `decode_label(path, indexed_ids)` needs the correct mapping for
  palette-indexed PNGs; verify on real label files before any training.
- Official split structure / test-label availability: **UNKNOWN locally.** (Commonly reported in
  the literature: the public release provides labeled training pairs only — to be verified on the
  installed data in `R2-P05-T01`, not assumed.)
- `data/` is absent locally by design (git-ignored, created by notebook code) — USER-APPROVED fact,
  not a defect.

**Compatibility LEVIR → SECOND:** different resolution (1024 vs 512), sensors/regions, label
semantics (binary building change vs six-class from-to). Binary change on SECOND can be derived
as `label > 0` (both label maps share the same change support in SECOND).

## 14. Metric definitions (precise meanings; do not substitute)

Binary / ROI (Research 1, as defined in the hand-off; implementation lives in notebooks 06–11,
not in `src/`):

- **Pixel F1 / IoU:** global confusion counts (`metrics.BinaryChangeMetrics`), never batch-averaged.
- **Strict component recall:** fraction of GT connected components matched one-to-one with IoU ≥ 0.5.
- **Raw-mask GT coverage:** fraction of GT pixels inside the predicted mask.
- **ROI union coverage ≥95 %:** fraction of GT components with ≥95 % of pixels inside the union of ROIs.
- **ROI single-crop coverage ≥95 %:** fraction of GT components with ≥95 % inside at least one
  single crop — critical when WHAT processes crops independently.
- **Crops / pair:** number of spatial ROI locations per image pair (each yields T1 and T2 patches).
- **Processed-area ratio:** sum of crop areas / scene area (overlaps counted repeatedly); unique
  area counts overlaps once. Neither equals latency.
- ROI coverage does not prove correct semantic classification.

Semantic metrics for Research 2 are PROPOSED (`EXPERIMENTAL_PROTOCOL.md` §5, D-014). The old
`semantic/training.SemanticMetrics` computes ROI-level before/after macro-F1 for the removed
classifier and is not a dense SCD metric.

## 15. Scientific fairness requirements

- Selection of models, thresholds, ROI policies and hyperparameters on train/validation only;
  the SECOND test split is untouched until the frozen final evaluation (USER-APPROVED, G3).
- Compared systems receive pre-declared, recorded optimization budgets (policy PROPOSED, D-004).
- Equal GPU-hours, wall time, epochs, steps, trials, capacity and inference budget are different
  constraints; the protocol must say which one is controlled for each comparison.
- Two-stage systems contain two trained models; their budget must cover both stages.
- LEVIR pretraining cost is not zero; it is reported separately as historical pretraining cost
  (USER-APPROVED, G2).

## 16. Computational budget considerations

- **USER-APPROVED (G2):** primary accounting unit = allocated GPU-hours,
  `GPU-hours = Σ_i T_i / 3600` over allocated GPUs (Instruction.md §6.2). Separate categories:
  training/HPO, inference, historical pretraining.
- Numerical limits are **not approved**; they will be informed by the pilot (D-005).
- Current accounting (§11) measures only epoch compute, not allocation; it omits data/model setup,
  checkpoint writes, queue time and failed OOM attempts; trial summaries sum the whole history
  (`hpo_backend.py:785, 826`) including epochs inherited from earlier phases → **summing per-phase
  reports double-counts**. Legacy `optimization.py:683, 1572` counts single-GPU epoch seconds.
- The HPO config records `torch_version` and `cuda_version` but not the GPU model; only
  `final_benchmark` records `gpu_name` (`final_benchmark.py:796`).

## 17. Reproducibility (preliminary, read-only; full audit = `R2-P03-T01`)

- `seed_everything` (`execution.py:77–87`): Python/NumPy/Torch/CUDA seeds,
  `torch.use_deterministic_algorithms(True)`, cuDNN deterministic, benchmark off,
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` via `setdefault` (an existing environment value wins).
- DataLoader shuffling uses an explicit `torch.Generator(seed)` only when `shuffle and seed`
  (`levir_cd.py:517–529`); augmentation `RandomFlipRotate` draws from `torch.rand`
  (`transforms.py:54–65`); worker seeding follows PyTorch defaults.
- RNG states saved/restored at epoch boundaries only (`execution.py:90–111`).
- TF32 is set explicitly only on the semantic path (`hpo_backend.py:580–582`); the binary path
  inherits PyTorch defaults (version-dependent; to be verified on the server). Evaluation in
  `final_benchmark._runtime` forces TF32 off and autocast off → training and evaluation numerics
  may differ.
- HPO job identities and parameters are deterministic, but device assignment depends on free
  memory at dispatch time; bitwise identity across different GPU models or driver/library
  versions is not expected; statistical reproducibility across seeds is the realistic target.
- Historical Research 1 reproducibility limitations are accepted as provenance; no retroactive
  repair without evidence (USER-APPROVED, D-021).

## 18. Implementation constraints

- Reusable logic in `src/satchange/`; notebooks orchestrate. No parallel trainers, HPO systems,
  model loaders, dataset loaders or metric implementations.
- Fail loudly; no silent CPU fallback; atomic writes; deterministic-compatible operations.
- Keep checkpoint keys, `model_parameters`, HPO parameter keys, `model__` routing and private
  helpers used by notebooks backward compatible.
- **HPO code-signature coupling (VERIFIED CURRENT):** `SatelliteBackend.code_signature`
  (`hpo_backend.py:268–290`) is one SHA256 over `training.py`, `execution.py`, `losses.py`,
  `metrics.py`, `hpo_backend.py`, `hpo_core.py`, `automated_hpo.py`, `optimization.py`, `run.py`
  and every file under `data/`, `models/`, `semantic/` (a test,
  `tests/test_semantic_hpo.py:528`, asserts semantic files and the controller are included).
  Any edit to these files makes every existing v3 search directory refuse to continue under the
  same `search_name`. Checkpoints do not store this signature (they store
  `code_revision = git HEAD`, a generated key not used for matching, `hpo_backend.py:737`);
  runs are matched by canonical parameters (`hpo_core.py:272–297`). The existing escape route —
  new `search_name` + `reuse_from` the old directory — continues matching runs from `last.pt`
  **without any code check of the old directory**. The check is therefore too broad inside a
  directory and absent across directories. A versioned compatibility design is required before
  any shared-code change while Research 1 HPO may still need resuming (`R2-P06-T01`, D-022).
- Final models are unaffected by the signature: `run.reconstruct_checkpoint` and
  `run_final_benchmark` rebuild from `model_name` + `model_parameters` with `strict=True`.
- Local machine: no CUDA (`torch 2.14.1+cpu`); documentation and CPU-only validation run locally,
  GPU work runs later on the training server. Absence of local GPU must not motivate speculative
  CUDA code changes (USER-APPROVED).

## 19. Current status (2026-11-06)

- Research 1: ongoing (user-driven); six finalized models tracked in git.
- Research 2: planning initialized; documents created; protocol draft v0.1 PROPOSED; no Research 2
  code, notebooks or experiments exist. See `PROGRESS.md`.

## 20. Open questions and unknowns

1. Semantic prediction target and single-stage baseline architecture (D-013).
2. Rules for selecting Research 1 checkpoints for Research 2 (D-012).
3. Full-system GPU budget accounting policy and numbers (D-004, D-005).
4. Primary accuracy–efficiency criteria and non-inferiority margin (D-014, D-015).
5. Official SECOND split structure on the installed copy; validation (and possibly test) creation
   policy (D-008, D-009); spatial/scene leakage (no grouping metadata known).
6. Whether any general training augmentation/noise is justified (D-018).
7. Which Research 1 HPO searches are unfinished and must remain resumable (D-022).
8. Whether `resume_from` can initialize a new condition from a LEVIR checkpoint (fine-tune) or a
   minimal initialization option is needed (R2-P08-T01).
9. Availability and integrity of the H1–H11 archives (REPORTED numbers unverified).
10. Target GPU environment (type, count, driver/CUDA/torch versions) for Research 2 — UNKNOWN;
    previous work reportedly used four 80 GB A100 GPUs.

## 21. Recorded discrepancies between sources

1. **Light Full test F1 — RESOLVED** (`CHECKPOINT_INVENTORY.md`): `final_models_gpu_cost.csv`
   `test_F1` (0.8054) is the 3-seed mean; hand-off H4 (0.841149) is the deployment seed 2024. Same
   pattern for all variants. Separate open item: per-variant `benchmark.json` latencies disagree
   with the CSV latencies (different benchmark protocols/code versions).
2. `artifacts/final_models/final_models_metrics.csv` column `F1` mixes validation and test values
   (`f1_key` column: `recomputed_validation_f1`, `validation_f1`, `test_f1`, …). Do not read it as one metric.
3. TinySiamese: user states it was intentionally removed; its notebooks are deleted, but model code,
   factory routing, the semantic branch in `training.py`/`hpo_backend.py`/`automated_hpo.py`/
   `final_benchmark.py`, `semantic/*` helpers, `configs/tiny_siamese_change_classifier.yaml`,
   `docs/semantic_classifier.md` and tests remained — **resolved 2026-11-06 (R2-P07-T01)**.
4. Docs/tests reference absent notebooks: `docs/heavy_model.md` (`03_heavy.ipynb`, `02_bit.ipynb`),
   `docs/semantic_classifier.md` and `tests/test_notebooks.py:53`
   (`tiny_siamese_second_staged_resume.ipynb`), `tests/test_notebooks.py:21` (`03_heavy.ipynb`).
5. The hand-off places `ROISettings`/`extract_rois` in a historical snapshot; they exist in the live
   `semantic/roi.py` with the same defaults. The H11 `roi_module_sha256` was not compared.
6. Notebooks 06 and 07 default `RUN_TEST=True`; 08_v2, 09, 10, 11 default several `RUN_*` flags to
   `True`. Never "Run All" them as a check.
7. **Training-code provenance gaps** (`READINESS_AUDIT.md` F1–F2, 2026-11-06): the Light models'
   training commit `7f38560` is absent locally and on the remote; the Medium (`02b72d7`) and
   Heavy-resized (`81a5e53`) training commits were unreferenced local objects — now protected by
   local tags (D-029; not pushed);
   `main` history starts at `9ed0fc7` (2026-10-06).
8. **HPO code signature is line-ending dependent** (`READINESS_AUDIT.md` F3): it hashes raw
   working-tree bytes; Windows (CRLF) and Linux (LF) checkouts of one commit give different values.

## 22. Major risks and threats to validity

- **Construct validity:** ROI coverage or processed area mistaken for semantic quality or latency.
- **Internal validity:** unequal optimization opportunity; architecture capacity confounded with the
  one- vs two-stage factor; Full/Resized confounded with architecture parameters.
- **Statistical validity:** few training seeds; large seed variance (Light Full); spatial
  dependence between SECOND tiles; pilot results treated as confirmatory.
- **External validity:** conclusions are conditional on SECOND, the tested models, hardware and budget.
- **Leakage:** no geographic grouping metadata; filename-level grouping only.
- **Operational:** shared-code edits blocking Research 1 HPO resume (code signature); Research 1
  continuing while Research 2 depends on its checkpoints (versioned freeze policy, D-010);
  `src/scripts/install_hpo_v3.py` rewrites sources.

## 23. Rationale of key historical decisions

- One shared engine (`fit_binary_change_model`) and one HPO system to avoid divergent pipelines
  (AGENTS.md §4).
- Thresholds chosen on validation only; test evaluation with frozen thresholds (final_benchmark).
- Absolute `stage_epochs` and strict protocol guards to make HPO resumable and auditable.
- Strict fail-loud policy (no silent restart when checkpoints are missing; unknown labels fail).
- TinySiamese abandoned by the user; Research 2 needs a newly discussed semantic model (D-013).
