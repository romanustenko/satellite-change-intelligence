# Research 2 — Master Implementation Plan

**Created:** 2026-11-06 · Context: [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md) · Decisions:
[`DECISION_LOG.md`](DECISION_LOG.md) · State: [`PROGRESS.md`](PROGRESS.md).

Statuses: NOT STARTED · IN PROGRESS · BLOCKED · COMPLETED · VERIFIED. A task is COMPLETED only
when its deliverable exists, VERIFIED only when its validation was actually run. Tasks are executed
under the D-028 working policy (autonomous for technical work, next unblocked task follows
automatically); "Approval" names the remaining scientific/protected gates. Token estimates are
rough (±50 %) and cover one agent session including reading.

## Approval gates

Since D-030 the gates are verification/documentation checkpoints; only scientific decisions
(question, hypotheses, evaluation methodology, evaluation splits, final comparison rules),
substantial GPU spend, destructive operations on historical artifacts and irreversible remote
changes require the user.

| Gate | Scope | State (2026-11-06) |
|---|---|---|
| G1 | Hypotheses and primary outcome | Question APPROVED; hypotheses PROPOSED |
| G2 | Budget policy and GPU-hour definition | Unit/categories APPROVED; policy and numbers PROPOSED |
| G3 | SECOND splits and leakage policy | APPROVED in principle; validation/test creation PROPOSED |
| G4 | Metrics | PROPOSED |
| G5 | Model/ROI/checkpoint selection rules | PROPOSED |
| G6 | Notebook reorganization | APPROVED |
| G7 | HPO changes (accounting) | APPROVED 2026-11-06 (design authorized) |
| G8 | Pilot launch / any substantial GPU run | NOT APPROVED |
| G9 | Change of the HPO code-signature / compatibility mechanism | APPROVED 2026-11-06 (D-022, Q1–Q3) |
| G10 | Any change that may invalidate Research 1 results or block Research 1 HPO resume | Always explicit |
| G11 | Final protocol v1.0 freeze | Pending |

## Dependency overview

```text
P1 verification ──┬─> P2 protocol ──> P2-T04 freeze (G11) ──> P10 SCD infra ──> P11 experiments ──> P12 report
                  │        ▲
P3 repro audit ───┘        │ (inputs)
P4 notebooks (G6) ──> P5-T03 notebook location
P6-T01 compat design (G9) ──> P6-T03 ──┬─> P5-T02 second.py decoupling ──> P5-T03/T04
P6-T02 accounting design (G7) ─> P6-T04 ┤                                 └─> P7 TinySiamese removal
                                        └─> P8 pilot prep (needs P5-T04) ──> P9 pilot (G8) ──> budget (D-005)
```

Rule from D-022/G10: any edit to `training.py`, `execution.py`, `losses.py`, `metrics.py`,
`hpo_*.py`, `automated_hpo.py`, `optimization.py`, `run.py`, `data/`, `models/` or `semantic/`
changes the current `backend_code_signature`. Until P6-T03 lands, such edits must be scheduled
with the user (unfinished Research 1 searches would then need a new `search_name`). Notebooks,
tests, configs and docs are not part of the signature.

---

## Phase 1 — Repository and environment verification

### R2-P01-T01 — Training-environment readiness audit
- **Scientific purpose:** establish what really exists where experiments will run (data, weights, GPUs, running jobs) before any decision depends on it.
- **Research dependency:** all experiments; budget estimation (D-005).
- **Current implementation:** local checkout is sparse (no `artifacts/`, no `data/`, CPU-only torch). Server state UNKNOWN.
- **Required work:** read-only on the GPU server: `git status`/HEAD, LFS state of `artifacts/final_models/**`, SHA256 of `.pt` files, presence of `artifacts/hyperparameter_search/` and unfinished searches (`results.json` `complete=false`, pause markers), SECOND presence, GPU model/count/driver/CUDA/torch, running processes. Short table with confirmed/missing/unknown.
- **Relevant files:** `artifacts/final_models/*/metadata.json`, `artifacts/hyperparameter_search/**/pipeline_protocol.json`, `results.json`.
- **Dependencies:** user access to the server.
- **Deliverables:** `docs/research2/READINESS_AUDIT.md`.
- **Acceptance:** every item has evidence (command/output/hash) or is marked UNKNOWN.
- **Validation:** cross-check against `PROJECT_CONTEXT.md` §3, §11; record drift.
- **Complexity:** Low. **Tokens:** 15–40k. **GPU:** none (inspection only).
- **Risks:** accidentally touching running HPO; reading checkpoint contents (use metadata JSON only).
- **Approval:** user runs or authorizes commands on the server.
- **Status:** BLOCKED (partial) — local part COMPLETED 2026-11-06 (`READINESS_AUDIT.md` §1–3, provenance tags D-029); server part blocked: no server access; commands ready in §4.

### R2-P01-T02 — Final-model checkpoint inventory and provenance
- **Scientific purpose:** candidate Research 1 checkpoints need exact provenance before any freeze (D-010).
- **Research dependency:** D-010, D-012, pilot (D-020).
- **Current implementation:** tracked metadata per variant (seeds, thresholds, `model_parameters`, git commit). VERIFIED HISTORICAL subset in `PROJECT_CONTEXT.md` §3.
- **Required work:** table per variant/seed: path, SHA256, `model_parameters`, best epoch, validation threshold/F1, training git commit, test metric source; flag discrepancies (light_full test F1; mixed `F1` column). No selection.
- **Relevant files:** `artifacts/final_models/**/{metadata,validation_results,final_test_results,training_results}.json`, `final_models_*.csv`.
- **Dependencies:** R2-P01-T01 (hashes on server).
- **Deliverables:** `docs/research2/CHECKPOINT_INVENTORY.md`.
- **Acceptance:** every reused value cites its JSON key; no value from test used for ranking.
- **Validation:** spot-check with `git show HEAD:<path>`.
- **Complexity:** Low. **Tokens:** 20–50k. **GPU:** none.
- **Risks:** confusing validation and test fields. **Approval:** none beyond assignment.
- **Status:** COMPLETED 2026-11-06 (`CHECKPOINT_INVENTORY.md`; server hash match pending with R2-P01-T01).

### R2-P01-T03 — Re-verify reported H1–H11 claims (optional)
- **Scientific purpose:** turn REPORTED numbers into VERIFIED HISTORICAL where they influence Research 2 choices (ROI candidates, shortlist).
- **Research dependency:** D-012 inputs only.
- **Current implementation:** archives not available in this workspace; outputs live on the server under `artifacts/hypothesis_tests/`.
- **Required work:** read manifests/CSVs named in `PROJECT_CONTEXT.md` §4; compare; record mismatches. No re-execution.
- **Relevant files:** H1–H11 archives or server `artifacts/hypothesis_tests/**`.
- **Dependencies:** user provides archives/access.
- **Deliverables:** updated labels in `PROJECT_CONTEXT.md` §4.
- **Acceptance:** each updated number cites file + row/column.
- **Validation:** independent second read of two numbers.
- **Complexity:** Low. **Tokens:** 20–60k. **GPU:** none.
- **Risks:** mixing dev/audit or val/test rows. **Approval:** none. **Status:** BLOCKED (archives unavailable).

## Phase 2 — Experimental protocol

### R2-P02-T01 — Draft protocol v0.1
- **Scientific purpose:** explicit, falsifiable plan before expensive work.
- **Research dependency:** G1–G5.
- **Current implementation:** —
- **Required work:** draft hypotheses, dataset, budget, metrics, selection, statistics, baselines, decision rules.
- **Relevant files:** `docs/research2/EXPERIMENTAL_PROTOCOL.md`.
- **Dependencies:** none.
- **Deliverables:** draft v0.1 (PROPOSED).
- **Acceptance:** every choice linked to a D-entry; nothing marked approved without source.
- **Validation:** consistency review with `DECISION_LOG.md` (2026-11-06).
- **Complexity:** Medium. **Tokens:** done. **GPU:** none.
- **Risks:** draft mistaken for frozen protocol. **Approval:** content needs G1–G5.
- **Status:** COMPLETED (draft created 2026-11-06; **not approved**).

### R2-P02-T02 — Methodology discussion of open decisions
- **Scientific purpose:** user decides D-002, D-004, D-008, D-009, D-012, D-014, D-015, D-018, D-024–D-026.
- **Research dependency:** G1–G5.
- **Current implementation:** proposals in `EXPERIMENTAL_PROTOCOL.md`.
- **Required work:** present options with trade-offs; record answers in `DECISION_LOG.md`.
- **Relevant files:** protocol, decision log.
- **Dependencies:** R2-P02-T01.
- **Deliverables:** updated decision statuses.
- **Acceptance:** each decision APPROVED/REJECTED with source, or explicitly deferred.
- **Validation:** no decision changed without a dated user source.
- **Complexity:** Medium. **Tokens:** 20–60k. **GPU:** none.
- **Risks:** premature numbers before pilot. **Approval:** user. **Status:** NOT STARTED.

### R2-P02-T03 — Semantic prediction target and single-stage baseline
- **Scientific purpose:** A and B must produce comparable outputs; defines metrics and models.
- **Research dependency:** D-013 → D-014, P10.
- **Current implementation:** no SCD model; TinySiamese abandoned (D-006).
- **Required work:** compare dense per-pixel maps vs ROI labels; candidate backbones (incl. FC-Siam-Diff with semantic heads, external SCD models); implementation cost; dependency constraints (no new packages without approval).
- **Relevant files:** `models/`, `semantic/roi.py`, `semantic/inference.py`.
- **Dependencies:** R2-P02-T01.
- **Deliverables:** decision note in `DECISION_LOG.md` (D-013) + protocol §5 update.
- **Acceptance:** chosen representation yields identical output grids for A and B.
- **Validation:** check against SECOND label semantics (R2-P05-T01 results if available).
- **Complexity:** High. **Tokens:** 40–100k. **GPU:** none.
- **Risks:** capacity confound between A and B. **Approval:** G4/G5. **Status:** NOT STARTED.

### R2-P02-T04 — Freeze protocol v1.0
- **Scientific purpose:** pre-registration before confirmatory experiments.
- **Research dependency:** G11.
- **Current implementation:** draft v0.1.
- **Required work:** integrate approved decisions and pilot-based budget; version, date.
- **Relevant files:** `EXPERIMENTAL_PROTOCOL.md`.
- **Dependencies:** R2-P02-T02, R2-P02-T03, R2-P05-T04, R2-P09-T02.
- **Deliverables:** protocol v1.0 APPROVED.
- **Acceptance:** user approval recorded; no PROPOSED items in confirmatory scope.
- **Validation:** cross-document consistency check.
- **Complexity:** Medium. **Tokens:** 20–50k. **GPU:** none.
- **Risks:** later changes without versioning. **Approval:** G11. **Status:** BLOCKED.

## Phase 3 — Reproducibility audit (read-only)

### R2-P03-T01 — Reproducibility report
- **Scientific purpose:** know whether results can be reproduced across runs, GPUs and resume, and what "reproducible" means for Research 2 statistics.
- **Research dependency:** D-025, D-026, hardware policy.
- **Current implementation:** preliminary findings in `PROJECT_CONTEXT.md` §17.
- **Required work:** code-based answer to Instruction.md §8 (same GPU, A100↔A100, other architectures, different GPU counts, HPO scheduling, bitwise vs statistical, resume): seeds, generators, workers, deterministic algorithms, cuDNN, cuBLAS env, TF32 defaults per torch version, bf16, non-deterministic ops, RNG restore, metadata.
- **Relevant files:** `execution.py`, `training.py`, `hpo_backend.py`, `hpo_core.py`, `optimization.py` (`_run_single_trial`), `data/levir_cd.py`, `data/transforms.py`, `final_benchmark.py`.
- **Dependencies:** none (may run in parallel with P2).
- **Deliverables:** `docs/research2/REPRODUCIBILITY_AUDIT.md` with classification and line references.
- **Acceptance:** sections A–F of Instruction.md §8 answered with evidence; unknowns marked.
- **Validation:** every claim cites path:line; optional CPU-only determinism check only if approved.
- **Complexity:** Medium. **Tokens:** 40–90k. **GPU:** none.
- **Risks:** over-claiming bitwise reproducibility. **Approval:** none (read-only, D-019).
- **Status:** COMPLETED 2026-11-06 (`REPRODUCIBILITY_AUDIT.md` A–F; CPU repeat/resume evidence; GPU confirmation pending server access).

### R2-P03-T02 — Reproducibility recommendations
- **Scientific purpose:** propose controls for Research 2 (e.g. record GPU model, explicit TF32 on binary path).
- **Research dependency:** D-026.
- **Current implementation:** —
- **Required work:** list of proposed changes with impact on code signature and Research 1.
- **Relevant files:** as T01.
- **Dependencies:** R2-P03-T01.
- **Deliverables:** recommendation section; new D-entries PROPOSED.
- **Acceptance:** no code changed.
- **Validation:** review by user.
- **Complexity:** Low. **Tokens:** 10–25k. **GPU:** none.
- **Risks:** — **Approval:** any implementation needs explicit permission (D-019).
- **Status:** COMPLETED 2026-11-06 (`REPRODUCIBILITY_AUDIT.md` §G; regression test added; code changes listed as proposals, gated by R2-P06-T03 / D-026).

## Phase 4 — Notebook organization (G6 approved)

### R2-P04-T01 — Notebook inventory and dependency map
- **Scientific purpose:** move safely without changing Research 1 behaviour.
- **Research dependency:** D-016.
- **Current implementation:** 17 notebooks in `notebooks/`; root resolution: 01–04 and `heavy_*_staged_resume` derive the root from `Path.cwd()` (parent/`next(...)` patterns); 04_mit_resized, 05–11, `Heavy_*` contain the absolute server path; `tests/test_notebooks.py:15` uses non-recursive `glob("*.ipynb")`, `:21`/`:53` reference absent notebooks; docs reference absent notebooks.
- **Required work:** per notebook: root-resolution code, relative paths, imports, notebook-to-notebook references (06→07–11 SHA checks), flags; list of required edits.
- **Relevant files:** `notebooks/*.ipynb` (source cells only), `tests/test_notebooks.py`, `docs/*.md`.
- **Dependencies:** none.
- **Deliverables:** inventory table (in task report / PROGRESS).
- **Acceptance:** every path-dependent cell identified.
- **Validation:** programmatic scan of source cells.
- **Complexity:** Low. **Tokens:** 20–50k. **GPU:** none.
- **Risks:** SHA-checked cells in 06–11 must not be edited. **Approval:** G6 (given).
- **Status:** COMPLETED 2026-11-06 — only 01–04 (root resolution) and 08 v1 (notebook-06 lookup) need edits; 05 uses `rglob`; 04_mit_resized, 05–11, `Heavy_*`, `heavy_*` use absolute server paths or a `src/satchange` walk-up and need none; no cell of 06–11 touched.

### R2-P04-T02 — Move notebooks and repair references
- **Scientific purpose:** separate the two studies in one repository.
- **Research dependency:** D-016.
- **Current implementation:** flat `notebooks/`.
- **Required work:** `git mv` notebooks to `notebooks/research_1/`; create `notebooks/research_2/` (no placeholders); minimal root-resolution fixes in source cells that would break; recursive discovery in `tests/test_notebooks.py`; update doc links; keep the uncommitted edit of `02_fc_siam_diff_full_hpo.ipynb` intact.
- **Relevant files:** `notebooks/**`, `tests/test_notebooks.py`, `docs/heavy_model.md`, `docs/semantic_classifier.md` (until P7 removes it).
- **Dependencies:** R2-P04-T01.
- **Deliverables:** moved notebooks; passing notebook tests.
- **Acceptance:** `test_notebooks.py` discovers all notebooks; outputs/metadata unchanged; no scientific behaviour changed.
- **Validation:** `pytest tests/test_notebooks.py`; diff shows only source-cell path edits; `git diff --check`.
- **Complexity:** Medium. **Tokens:** 40–90k. **GPU:** none.
- **Risks:** breaking 06–11 hash checks; losing uncommitted work. **Approval:** G6 (given).
- **Status:** VERIFIED 2026-11-06 — 17 notebooks moved with `git mv`; 19 source-line edits in 7 notebooks via count-checked raw-JSON replacement (outputs, metadata, cell ids unchanged); the user's uncommitted edit of `02_fc_siam_diff_full_hpo.ipynb` preserved (SHA256 identical before/after the move); `tests/test_notebooks.py` recursive discovery + non-empty assertion; `docs/heavy_model.md` paths updated (CRLF kept); `notebooks/README.md`, `notebooks/research_2/.gitkeep` added. Checks: root resolution simulated from `notebooks/research_1` and project root for all edited cells; `pytest tests/test_notebooks.py` = baseline (17 notebooks syntax-validated, 1 skip, known Tiny failure); ruff clean; `git diff --check` clean. Not committed.

## Phase 5 — SECOND dataset preparation (D-011)

### R2-P05-T01 — Inspect installed SECOND structure
- **Scientific purpose:** split policy depends on what the official release contains.
- **Research dependency:** D-007, D-008, D-009.
- **Current implementation:** `bootstrap_second`, `discover_second` (`semantic/second.py`).
- **Required work:** in the data environment: install/reuse SECOND, run `discover_second`, report leaves, counts, labeled/unlabeled test, image sizes, palette values vs `SECOND_COLORS`, `indexed_id_names` mismatch check.
- **Relevant files:** `src/satchange/semantic/second.py`, `configs/tiny_siamese_change_classifier.yaml` (mapping reference only).
- **Dependencies:** user authorizes download (or provides archive).
- **Deliverables:** structure facts in `PROJECT_CONTEXT.md` §13; D-008/D-009 inputs.
- **Acceptance:** counts and label values observed, not assumed.
- **Validation:** checksums recorded in `source.json`.
- **Complexity:** Low. **Tokens:** 15–40k. **GPU:** none.
- **Risks:** quota/auth failures of Google Drive. **Approval:** download authorization. **Status:** COMPLETED 2026-11-07 — installed in `data/SECOND` (release is nested: `SECOND_total_test.zip` + `SECOND_train_set.rar`, which `bootstrap_second` cannot unpack; installed by hand with bsdtar, nested SHA256 in `source.json`); audit in `docs/research2/second_audit.json` (`src/scripts/second_audit.py`).

### R2-P05-T02 — Decouple split/conversion from the ROI-classifier preparation
- **Scientific purpose:** shared, model-independent split manifests and decoded labels for every Research 2 model.
- **Research dependency:** D-011, D-006.
- **Current implementation:** `prepare_second` requires `data_protocol(config)` with ROI settings and builds the TinySiamese ROI dataset via `prepare_dataset`.
- **Required work:** minimal change in `semantic/second.py`: expose discovery + split assignment + label conversion + manifest writing without ROI preparation; keep hash-identified atomic cache; tests.
- **Relevant files:** `semantic/second.py`, `semantic/data.py`, `tests/test_semantic.py`.
- **Dependencies:** R2-P05-T01; R2-P06-T03 or explicit user acceptance of the code-signature consequence (G10).
- **Deliverables:** extended API + CPU tests on synthetic data.
- **Acceptance:** split manifest reproducible from seed; existing behaviour preserved until P7.
- **Validation:** focused pytest, ruff, mypy on touched files.
- **Complexity:** Medium. **Tokens:** 40–90k. **GPU:** none.
- **Risks:** breaking existing caches; signature change. **Approval:** G10. **Status:** COMPLETED 2026-11-06 — `semantic.second.prepare_second_splits` / `read_second_split_manifest` (explicit split policy, required seed, official splits never overridden, duplicate-pair diagnostics), `tests/test_second.py` (16 tests); `prepare_second` behaviour unchanged.

### R2-P05-T03 — Implement `notebooks/research_2/01_second_dataset_preparation.ipynb`
- **Scientific purpose:** one reproducible, documented dataset foundation (D-011).
- **Research dependency:** D-007, D-008, D-009, D-018.
- **Current implementation:** none.
- **Required work:** notebook calling `bootstrap_second` (download guarded by a flag defaulting to `False`), integrity checks, documentation of source/version/structure/classes/ignored label, split manifests (official preserved; validation/test creation only per approved D-008/D-009), diagnostics (counts, sizes, class/transition distribution, changed vs unchanged, corrupt files, example visualisations), augmentation section stating "disabled" unless D-018 approves; Markdown explanation of every change.
- **Relevant files:** `notebooks/research_2/`, `semantic/second.py`.
- **Dependencies:** R2-P04-T02, R2-P05-T02, D-008/D-009 decided.
- **Deliverables:** notebook (no fabricated outputs).
- **Acceptance:** fresh-kernel syntax test passes; expensive cells guarded; unexecuted steps stated.
- **Validation:** `tests/test_notebooks.py`; CPU run of non-download cells on synthetic layout if feasible.
- **Complexity:** Medium. **Tokens:** 50–120k. **GPU:** none.
- **Risks:** duplicating `second.py` logic in the notebook. **Approval:** D-008/D-009 (decided 2026-11-07). **Status:** NOT STARTED (unblocked).

### R2-P05-T04 — Execute preparation and freeze the dataset manifest
- **Scientific purpose:** frozen splits used by every model.
- **Research dependency:** G3, G11.
- **Current implementation:** —
- **Required work:** user runs the notebook in the data environment; record archive hash, manifest hash, counts.
- **Relevant files:** `data/` (git-ignored), manifest location to be decided (git-tracked copy of ID lists recommended).
- **Dependencies:** R2-P05-T03.
- **Deliverables:** frozen manifest + `PROGRESS.md` entry.
- **Acceptance:** counts reported; hashes recorded; no test-driven choice.
- **Validation:** rerun yields identical manifest hash.
- **Complexity:** Low. **Tokens:** 10–30k. **GPU:** none.
- **Risks:** — **Approval:** G3 freeze. **Status:** BLOCKED.

## Phase 6 — HPO compatibility and GPU-hour accounting (shared code, D-021/D-022/D-017)

### R2-P06-T01 — Versioned compatibility design (signatures, migration, checkpoint implications)
- **Scientific purpose:** allow necessary shared-code evolution while keeping every HPO result traceable; prevent both false blocking and silent bypass.
- **Research dependency:** D-021, D-022, G9, G10.
- **Current implementation:** `SatelliteBackend.code_signature` (`hpo_backend.py:268–290`) hashes a broad file set; `Experiment.ensure_protocol` (`hpo_core.py:246–268`) requires exact protocol equality; `reuse_from` sources are discovered by parameters only (`hpo_core.py:240–243, 272–297`) with no code check; checkpoints store `code_revision` (git HEAD) but not the signature; final-model loading is signature-independent.
- **Required work:** concrete design document covering: component signatures (training-semantics vs infrastructure) and their file sets per task/model; rule that infrastructure mismatches block by default; how monitoring/logging-only changes are classified and reviewed; inclusion of `reuse_from` sources in verification (including legacy folders without v3 protocol); new protocol schema and `code_history` audit record; legacy migration (reproduce the recorded old signature from a git commit as evidence of code identity — explicitly not of behavioural equivalence; separate migration record, old protocols never rewritten; otherwise read-only + explicit incompatibility report); effect on existing checkpoints, `best.pt`/`last.pt`, final models and tests (`tests/test_semantic_hpo.py:528`); interaction with `src/scripts/install_hpo_v3.py`.
- **Open questions:** the current signature hashes raw working-tree bytes and therefore differs between CRLF (Windows) and LF (Linux) checkouts of the same commit (`READINESS_AUDIT.md` F3) — evidence must hash normalized content; training commits of Light models are missing and those of Medium/Heavy-resized are unreferenced (F1, F2); which Research 1 searches are unfinished; file-level granularity given that `training.py` mixes training semantics with DDP/checkpoint code; whether a reviewed version constant is acceptable evidence; how to verify sources patched by `install_hpo_v3.py`.
- **Relevant files:** `hpo_core.py`, `hpo_backend.py`, `automated_hpo.py`, `tests/test_semantic_hpo.py`, server `artifacts/hyperparameter_search/**/pipeline_protocol.json`.
- **Dependencies:** R2-P01-T01 (list of unfinished searches).
- **Deliverables:** `docs/research2/HPO_COMPATIBILITY_DESIGN.md` (PROPOSED).
- **Acceptance:** every requirement of D-022 mapped to a mechanism; explicit list of what can / cannot resume after the change.
- **Validation:** user review (G9).
- **Complexity:** High. **Tokens:** 60–140k. **GPU:** none.
- **Risks:** false equivalence claims; scope creep into a framework. **Approval:** G9.
- **Status:** COMPLETED (design PROPOSED) 2026-11-06 — `HPO_COMPATIBILITY_DESIGN.md` Part 1; awaiting user review (Q1–Q3).

### R2-P06-T02 — GPU-hour accounting design
- **Scientific purpose:** auditable budget control (D-003, D-017).
- **Research dependency:** D-004, D-005.
- **Current implementation:** epoch-compute `gpu_hours` only; history sums double-count across phases; no ledger/limit; GPU model not recorded.
- **Required work:** define measured interval (device allocation from job dispatch to completion in the controller, plus failed/OOM attempts), ledger format (append-only, atomic, job/attempt IDs for de-duplication), cumulative/remaining budget, pre-dispatch check with expected trial cost, enforcement (stop dispatch; pause marker for active jobs at epoch boundary), overshoot reporting, crash/restart semantics and uncertainty recording, phase de-duplication, API (`gpu_hour_budget=None` default), relation to existing `gpu_hours` fields (kept for compatibility).
- **Relevant files:** `hpo_core.py` (`_execute`, `save`), `automated_hpo.py`, `hpo_backend.py` (`_train` summary), `training.py:1083`.
- **Dependencies:** R2-P06-T01 (where the code lives and how it affects signatures).
- **Deliverables:** design section in `HPO_COMPATIBILITY_DESIGN.md` or separate `GPU_ACCOUNTING_DESIGN.md`.
- **Acceptance:** covers Instruction.md §6.1–6.5; zero-overshoot not promised.
- **Validation:** user review (G7).
- **Complexity:** Medium. **Tokens:** 40–90k. **GPU:** none.
- **Risks:** counting available instead of allocated GPUs. **Approval:** G7 review.
- **Status:** COMPLETED (design PROPOSED) 2026-11-06 — `HPO_COMPATIBILITY_DESIGN.md` Part 2; awaiting user review.

### R2-P06-T03 — Implement the compatibility mechanism
- **Scientific purpose:** unblock shared-code changes safely.
- **Research dependency:** D-022.
- **Current implementation:** see T01.
- **Required work:** implement approved design in the existing modules; migration record writer; no rewriting of old protocols.
- **Relevant files:** `hpo_core.py`, `hpo_backend.py`, `automated_hpo.py`.
- **Dependencies:** R2-P06-T01 approved (G9).
- **Deliverables:** code + regression tests (legacy directory fixtures).
- **Acceptance:** old protocol directories behave exactly as documented (resume / read-only + report).
- **Validation:** full pytest vs baseline (AGENTS.md §8), ruff, mypy.
- **Complexity:** High. **Tokens:** 80–180k. **GPU:** none.
- **Risks:** breaking legacy resume. **Approval:** G9, G10. **Status:** VERIFIED 2026-11-06 — schema-4 protocol with per-file normalized code manifest (`SatelliteBackend.code_identity`), `code_history.json`, `acknowledge_code_change`, `CodeCompatibilityError`, provenance verification of `reuse_from`/`resume_from` sources, `unverified_reuse='exploratory'` labelling (Q3); old protocols never rewritten.

### R2-P06-T04 — Implement GPU-hour accounting and enforcement
- **Scientific purpose:** budgeted, auditable HPO for Research 2.
- **Research dependency:** D-003, D-017.
- **Current implementation:** none.
- **Required work:** approved T02 design in `hpo_core`/`automated_hpo` (and minimal backend hooks); record GPU model.
- **Relevant files:** `hpo_core.py`, `automated_hpo.py`, `hpo_backend.py`.
- **Dependencies:** R2-P06-T02 approved, R2-P06-T03.
- **Deliverables:** code; backwards-compatible default.
- **Acceptance:** Instruction.md §6.1 list satisfied; no budget reset on resume.
- **Validation:** R2-P06-T05.
- **Complexity:** High. **Tokens:** 80–160k. **GPU:** none.
- **Risks:** concurrency on ledger writes. **Approval:** G7. **Status:** VERIFIED 2026-11-06 — `src/satchange/hpo_accounting.py` ledger (`gpu_ledger.jsonl`), dispatch-time budget check, mid-run pause via `PAUSE_AFTER_EPOCH`, overshoot records, orphan closing, `gpu_hour_budget`/`budget_reason` API, GPU name in `resources()`, `epoch_compute_gpu_hours_this_attempt`.

### R2-P06-T05 — Tests for compatibility and accounting
- **Scientific purpose:** evidence that accounting and compatibility behave as designed.
- **Research dependency:** D-017, D-022.
- **Current implementation:** CPU "GPU" queue mocks in `tests/test_semantic_hpo.py`.
- **Required work:** mocked clocks/synthetic workers: single GPU, four parallel GPUs, unallocated GPUs, budget boundary, mid-trial exhaustion, crash/resume, failed and OOM attempts, duplicate prevention, graceful stop, backward compatibility, legacy protocol directories, `reuse_from` verification.
- **Relevant files:** `tests/test_semantic_hpo.py` or a new focused test file.
- **Dependencies:** R2-P06-T03, R2-P06-T04.
- **Deliverables:** tests.
- **Acceptance:** all pass on CPU; no real GPU used.
- **Validation:** full pytest vs baseline.
- **Complexity:** Medium. **Tokens:** 50–110k. **GPU:** none.
- **Risks:** flaky timing tests (use injected clocks). **Approval:** — **Status:** VERIFIED 2026-11-06 — `tests/test_hpo_compatibility.py` (12 tests, fake backend + injected clock); found and fixed a torn-ledger-line append bug.

### R2-P06-T06 — Usage documentation
- **Scientific purpose:** future sessions use the budget correctly.
- **Research dependency:** D-017.
- **Current implementation:** —
- **Required work:** document API, ledger, limitations, migration procedure.
- **Relevant files:** `docs/research2/`, `AGENTS.md` pointer.
- **Dependencies:** R2-P06-T05.
- **Deliverables:** docs.
- **Acceptance:** matches implemented behaviour.
- **Validation:** review against code.
- **Complexity:** Low. **Tokens:** 15–30k. **GPU:** none.
- **Risks:** — **Approval:** — **Status:** COMPLETED 2026-11-06 — usage in the `run_automated_parameter_search` docstring and `HPO_COMPATIBILITY_DESIGN.md` (deviations listed there).

## Phase 7 — TinySiamese cleanup (D-006)

### R2-P07-T01 — Remove remaining TinySiamese components
- **Scientific purpose:** remove an abandoned model line so it cannot be confused with the Research 2 semantic model; reduce maintenance surface.
- **Research dependency:** D-006, D-013.
- **Current implementation:** `models/tiny_siamese_change_classifier.py`; export in `models/__init__.py`; `SEMANTIC_TASK`/`model_class_for_task` in `models/factory.py`; semantic branch in `training.py` (TinySiamese import near line 870); semantic paths in `hpo_backend.py`, `automated_hpo.py`, `final_benchmark.py`; `semantic/training.py`, `semantic/inference.py` (`load_classifier`, `predict_rois`, `evaluate_classifier`), `semantic/benchmark.py` (`finalize_semantic`), CLI `semantic/__main__.py`, ROI dataset parts of `semantic/data.py`; `configs/tiny_siamese_change_classifier.yaml`; `docs/semantic_classifier.md`; `tests/test_semantic.py`, `tests/test_semantic_hpo.py`, `tests/test_notebooks.py:53`.
- **Required work:** full call-site search (src, tests, notebooks) before deletion; remove TinySiamese-specific code, configs, tests, factory registrations and docs; preserve `semantic/second.py` (installation, discovery, decoding, decoupled splits), `semantic/roi.py`, `semantic/inference.detector_probability_map`/`predict_detector_checkpoint`; keep tests of the preserved parts and of HPO infrastructure that currently use the semantic path as fixtures (rewrite them to the binary task if needed); keep `fit_binary_change_model` name and checkpoint keys; protect Research 1 functionality and existing HPO experiment directories.
- **Relevant files:** as listed.
- **Dependencies:** R2-P05-T02 (decoupled SECOND preparation), R2-P06-T03 (compatibility mechanism; removal changes the signature).
- **Deliverables:** removal change + updated tests/docs + note in `PROGRESS.md`.
- **Acceptance:** no remaining references to `tiny_siamese`/`semantic_roi_classification` except historical docs/changelog; Research 1 binary paths unchanged.
- **Validation:** full pytest vs baseline, ruff, mypy, `compileall`; `git grep` for leftovers.
- **Complexity:** High. **Tokens:** 80–180k. **GPU:** none.
- **Risks:** removing infrastructure tests that also cover the controller; breaking notebook imports. **Approval:** D-006 (given); G10 check. **Status:** VERIFIED 2026-11-06 — removed the model, factory routing, the semantic task in `training`/`optimization`/`hpo_backend`/`automated_hpo`/`final_benchmark` (the `task` argument now accepts only the binary task), `semantic/{training,benchmark,__main__}.py`, ROI-dataset parts of `semantic/data.py`, `prepare_second`, classifier inference, Tiny config/doc/tests. Kept byte-identical `semantic/roi.py` (notebook hash checks) via a one-function shim; controller tests ported to the binary task in `tests/test_hpo_v3.py` (incl. the first end-to-end binary `run_final_benchmark` test). Full pytest 134 passed / 3 skipped / 0 failed; mypy unchanged vs HEAD.

## Phase 8 — FC-Siam-Diff pilot preparation (D-020, D-024)

### R2-P08-T01 — Checkpoint compatibility and initialization path
- **Scientific purpose:** conditions must differ only in initialization.
- **Research dependency:** H-D, D-020.
- **Current implementation:** `FCSiamDiff(in_channels, out_channels, base_channels)`; light_full `base_channels=32`, light_resized 16; head `output_layer`; `run.reconstruct_checkpoint` strict rebuild; `resume_from` continues identical runs (initialization use UNKNOWN).
- **Required work:** determine which LEVIR checkpoint and which layers load (all for binary target; all but `output_layer` for a semantic head); find or propose the minimal initialization option in existing code.
- **Relevant files:** `models/fc_siam_diff.py`, `run.py`, `hpo_backend.py`, `automated_hpo.py`, `artifacts/final_models/light_*`.
- **Dependencies:** R2-P01-T02.
- **Deliverables:** compatibility note with layer map.
- **Acceptance:** exact loaded/unloaded parameter lists.
- **Validation:** CPU load test with `strict` reporting on a tracked checkpoint (server or LFS).
- **Complexity:** Medium. **Tokens:** 30–70k. **GPU:** none.
- **Risks:** silent partial loads. **Approval:** —
- **Status:** COMPLETED locally 2026-11-06 (`CHECKPOINT_INVENTORY.md` pilot section); strict load of real `.pt` pending server/LFS.

### R2-P08-T02 — SECOND binary data path for detectors
- **Scientific purpose:** WHERE adaptation and the pilot need SECOND binary masks in the existing training engine.
- **Research dependency:** D-024, P11-T01.
- **Current implementation:** `create_dataset_dataloader` supports the LEVIR layout only; `prepare_second` derives `change = label > 0`.
- **Required work:** choose the smallest extension (e.g. manifest-driven split support in the existing loader) — no parallel dataset pipeline.
- **Relevant files:** `data/levir_cd.py`, `semantic/second.py`.
- **Dependencies:** R2-P05-T04, R2-P06-T03.
- **Deliverables:** extension + CPU tests.
- **Acceptance:** same tensor contract as LEVIR samples.
- **Validation:** focused + full pytest.
- **Complexity:** Medium. **Tokens:** 40–100k. **GPU:** none.
- **Risks:** duplicating loader logic. **Approval:** G10. **Status:** BLOCKED.

### R2-P08-T03 — Pilot configuration proposal
- **Scientific purpose:** bounded, fair pilot.
- **Research dependency:** D-020, D-024, D-005.
- **Current implementation:** —
- **Required work:** runs, seeds, base_channels, resolution (native 512), epochs, validation schedule, checkpoint frequency, max GPU-hours, stopping, quality thresholds for "GPU-hours to target", expected overhead.
- **Relevant files:** `EXPERIMENTAL_PROTOCOL.md` §2 (H-D).
- **Dependencies:** R2-P08-T01, R2-P08-T02, R2-P06-T04.
- **Deliverables:** pilot spec.
- **Acceptance:** only initialization differs; budget cap explicit.
- **Validation:** user review.
- **Complexity:** Medium. **Tokens:** 20–50k. **GPU:** none.
- **Risks:** — **Approval:** G8. **Status:** BLOCKED.

### R2-P08-T04 — CPU smoke tests of both pilot conditions
- **Scientific purpose:** catch wiring errors before GPU time.
- **Research dependency:** D-020.
- **Required work / Current implementation:** tiny synthetic SECOND layout; both conditions run 1–2 epochs on CPU through the real HPO entry point with `dry_run` first.
- **Relevant files:** tests.
- **Dependencies:** R2-P08-T03.
- **Deliverables:** passing smoke tests.
- **Acceptance:** ledger and metrics written; no GPU used.
- **Validation:** pytest.
- **Complexity:** Low. **Tokens:** 20–50k. **GPU:** none.
- **Risks:** — **Approval:** — **Status:** BLOCKED.

## Phase 9 — Pilot execution and budget estimation

### R2-P09-T01 — Run the bounded pilot
- **Scientific purpose:** measure real cost and convergence (H-D).
- **Research dependency:** D-005, D-020.
- **Current implementation:** —
- **Required work:** run approved spec on the server; record ledger, curves vs epochs and GPU-hours, peak memory, stability. No test set.
- **Relevant files:** `notebooks/research_2/` pilot notebook (created only when approved).
- **Dependencies:** R2-P08-T04, G8.
- **Deliverables:** pilot results under a new artifacts directory; `PROGRESS.md` entry.
- **Acceptance:** spend within cap or overshoot reported.
- **Validation:** ledger reconciles with per-epoch records.
- **Complexity:** Medium. **Tokens:** 30–80k. **GPU:** bounded (cap set in R2-P08-T03).
- **Risks:** treating pilot as confirmatory. **Approval:** G8. **Status:** BLOCKED.

### R2-P09-T02 — Budget options
- **Scientific purpose:** fix D-005 on evidence.
- **Research dependency:** D-004, D-005.
- **Required work / Current implementation:** extrapolate costs (scratch, fine-tune, HPO, final training, repetitions, both stages, single-stage, validation, benchmarking); three options with supported conclusions.
- **Relevant files:** pilot outputs.
- **Dependencies:** R2-P09-T01.
- **Deliverables:** budget proposal in `EXPERIMENTAL_PROTOCOL.md`.
- **Acceptance:** each option states what it can and cannot conclude.
- **Validation:** user review.
- **Complexity:** Medium. **Tokens:** 20–50k. **GPU:** none.
- **Risks:** extrapolation from semantic-free pilot to semantic models. **Approval:** G2. **Status:** BLOCKED.

## Phase 10 — Semantic change detection infrastructure (only as approved by the protocol)

### R2-P10-T01 — Dense SCD metrics
- **Purpose / dependency:** primary metric (D-014). **Current:** none (only binary metrics, old ROI-level metrics).
- **Required work:** SeK, mIoU, Fscd, per-class IoU/F1 from global confusion counts, verified against a reference implementation on fixtures. **Files:** `metrics.py` or `semantic/`. **Depends on:** R2-P02-T03, D-014.
- **Deliverables / acceptance:** functions + tests reproducing reference values. **Validation:** pytest.
- **Complexity:** Medium. **Tokens:** 40–90k. **GPU:** none. **Risks:** definition mismatch. **Approval:** G4. **Status:** BLOCKED.

### R2-P10-T02 — Semantic model(s)
- **Purpose / dependency:** WHAT and single-stage models per D-013. **Current:** none.
- **Required work:** minimal implementation registered via the existing factory pattern; JSON-serializable constructor; tensor-contract checks. **Files:** `models/`, `models/factory.py`. **Depends on:** R2-P02-T03, R2-P06-T03.
- **Deliverables / acceptance:** model + CPU tests. **Validation:** pytest.
- **Complexity:** High. **Tokens:** 60–150k. **GPU:** none for tests. **Risks:** capacity confound. **Approval:** G5. **Status:** BLOCKED.

### R2-P10-T03 — Reconstruction of crop predictions into image coordinates
- **Purpose / dependency:** end-to-end evaluation of B. **Current:** ROI boxes `(x0,y0,x1,y1)` exclusive upper bounds in `semantic/roi.py`; no reconstruction.
- **Required work:** deterministic merge of overlapping crops; outside-ROI = no-change. **Files:** `semantic/`. **Depends on:** R2-P10-T02.
- **Deliverables / acceptance:** function + tests (overlap, borders, Resized alignment). **Validation:** pytest.
- **Complexity:** Medium. **Tokens:** 30–70k. **GPU:** none. **Risks:** overlap policy bias. **Approval:** — **Status:** BLOCKED.

### R2-P10-T04 — End-to-end latency benchmark
- **Purpose / dependency:** H-A, H-B efficiency outcome (D-026). **Current:** forward-only benchmarks (`final_benchmark`, `semantic/benchmark._speed`).
- **Required work:** extend existing benchmark code to time all six stages for B and the full pass for A under identical conditions. **Files:** `final_benchmark.py` / `semantic/benchmark.py`. **Depends on:** R2-P10-T03.
- **Deliverables / acceptance:** benchmark + CPU tests of the timing harness. **Validation:** pytest; GPU run only under G8-type approval.
- **Complexity:** Medium. **Tokens:** 40–90k. **GPU:** later, small. **Risks:** asynchronous CUDA timing errors. **Approval:** — **Status:** BLOCKED.

## Phase 11 — Controlled experiments

| Task | Title | Depends on | Approval | Status |
|---|---|---|---|---|
| R2-P11-T01 | WHERE adaptation of shortlisted detectors on SECOND (binary) under the budget policy | R2-P08-T02, R2-P09-T02, R2-P02-T04 | G8-type, G2 | BLOCKED |
| R2-P11-T02 | Single-stage model training (3 seeds) | R2-P10-T02, R2-P02-T04 | G2 | BLOCKED |
| R2-P11-T03 | Stage-2 (WHAT) training on crops (3 seeds) | R2-P10-T02/T03, R2-P02-T04 | G2 | BLOCKED |
| R2-P11-T04 | ROI policy and threshold selection on validation (incl. oracle ROIs) | R2-P11-T01, R2-P11-T03 | G5 | BLOCKED |
| R2-P11-T05 | Freeze manifest: checkpoints, configs, code version, preprocessing, selection evidence (D-010) | R2-P11-T01..T04 | G11 | BLOCKED |
| R2-P11-T06 | Single final test evaluation of frozen systems + latency benchmark | R2-P11-T05, R2-P10-T04 | explicit | BLOCKED |

Common fields for P11 tasks — **purpose:** produce the confirmatory evidence for H-A/H-B/H-C;
**current implementation:** HPO v3, final benchmark; **deliverables:** new artifact directories
(never overwriting Research 1), ledger entries, `PROGRESS.md` updates; **acceptance:** spend
within budget, selection on validation only, provenance recorded; **validation:** reconciliation
of ledger, manifest hashes, rerun of evaluation from frozen checkpoints; **complexity:** High;
**tokens:** 50–150k each; **GPU:** substantial (set by D-005); **risks:** budget overruns,
test leakage, Research 1 checkpoint drift.

## Phase 12 — Scientific evaluation and reporting

| Task | Title | Depends on | Status |
|---|---|---|---|
| R2-P12-T01 | Statistical analysis: paired bootstrap, seeds, density strata, Pareto fronts, H-C error decomposition | R2-P11-T06 | BLOCKED |
| R2-P12-T02 | Reproducibility verification (rerun of a frozen evaluation) and final report with limitations | R2-P12-T01 | BLOCKED |

Purpose: answer the central question conditionally (domain, models, hardware, budget), including
a possibly negative result. Complexity Medium–High; tokens 60–150k each; no new training.
