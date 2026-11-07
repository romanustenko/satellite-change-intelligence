# Research 2 — Progress

**Last updated:** 2026-11-07. Update after every meaningful task: what changed, evidence, new
blockers. Keep it a state record, not a transcript.

## Current milestone

**M1 — Foundation + SECOND data ready (2026-11-07).** Docs, HPO compatibility/GPU-hour ledger,
TinySiamese removal, SECOND binary loader and split manifest done; SECOND installed locally and
audited. Protocol draft v0.1 still PROPOSED (gate G11). No SECOND notebook, pilot, training or HPO yet.

## Research 1 (historical foundation, ongoing)

- **Status:** ongoing — the user continues hypothesis work (USER-APPROVED, D-010). Research 2 does
  not freeze Research 1 models; it reuses checkpoints only through a versioned freeze manifest.
- **Verified historical:** six finalized LEVIR-CD detectors tracked in git under
  `artifacts/final_models/` with metadata, validation/test results and forward-only GPU cost
  (`final_models_metrics.csv`, `final_models_gpu_cost.csv`).
- **Reported, not re-verified:** H1–H11 results (hand-off of 2026-10-31; archives not available
  here) — see `PROJECT_CONTEXT.md` §4.

## Research 2 work log

| Date | Item | Result | Evidence |
|---|---|---|---|
| 2026-11-06 | Read-only repository inspection | Findings recorded in `PROJECT_CONTEXT.md` §11–§18, §21 | HEAD `b72cd6c` |
| 2026-11-06 | Created `PROJECT_CONTEXT.md`, `EXPERIMENTAL_PROTOCOL.md` (draft), `DECISION_LOG.md`, `IMPLEMENTATION_PLAN.md`, `PROGRESS.md`; added §0 to local `AGENTS.md` | Documentation only; no code/notebook/config changes | `git status` |
| 2026-11-06 | User decisions on G1–G8, shared codebase, HPO compatibility requirements, TinySiamese removal | Recorded as D-001…D-027 | `DECISION_LOG.md` |
| 2026-11-06 | Working policy D-028 recorded (PROJECT_CONTEXT §0, AGENTS.md §0, `.clinerules`); provenance tags D-029 created locally (not pushed) | — | `git tag -l 'provenance/*'` |
| 2026-11-06 | R2-P07-T01 TinySiamese removed | Semantic task removed from shared engine/HPO/benchmark; roi.py untouched (shim); tests ported to binary | `tests/test_hpo_v3.py`; pytest 134/3/0 |
| 2026-11-06 | R2-P06-T03..T06 implemented | Schema-4 code identity + exception audit trail + `reuse_from` provenance; GPU-hour ledger and budget enforcement; 12 new tests | `hpo_core.py`, `hpo_accounting.py`, `tests/test_hpo_compatibility.py` |
| 2026-11-06 | R2-P05-T02 SECOND split manifest | `prepare_second_splits` decoupled from ROI prep (explicit policy) | `semantic/second.py`, `tests/test_second.py` |
| 2026-11-06 | Workflow | Subagents created, then removed (D-031 → D-032 single agent) | `DECISION_LOG.md` |
| 2026-11-06 | R2-P06-T01/T02 designs | Per-file normalized code manifest, experiment-specific dependency set, `reuse_from` verification, legacy migration by commit matching; controller-side allocation ledger with heartbeat, budget enforcement via existing pause marker | `HPO_COMPATIBILITY_DESIGN.md` (PROPOSED) |
| 2026-11-06 | R2-P08-T01 pilot compatibility | Architecture counts match final models; 512 px OK; no init-from-weights path — minimal `initial_weights` key proposed | `CHECKPOINT_INVENTORY.md` |
| 2026-11-06 | R2-P03-T01/T02 reproducibility audit | Deterministic under defined conditions; CPU evidence: repeat run and 2+2 resume bitwise identical; different GPUs OK under 5 conditions; new regression test | `REPRODUCIBILITY_AUDIT.md`; `pytest tests/test_execution.py` |
| 2026-11-06 | R2-P04-T01/T02 notebook reorganization | `notebooks/research_1/` (17 moved), `notebooks/research_2/`; paths and test discovery fixed | IMPLEMENTATION_PLAN status; `pytest tests/test_notebooks.py` |
| 2026-11-06 | R2-P01-T02 checkpoint inventory | Per-seed candidates, recorded training GPU-hours, provenance; Light Full F1 discrepancy resolved (CSV = 3-seed mean); latency sources disagree | `CHECKPOINT_INVENTORY.md` |
| 2026-11-06 | R2-P01-T01 local part (read-only) | Expected SHA256 of all 22 final checkpoints from LFS pointers; provenance gaps F1–F2; line-ending-dependent HPO signature F3; server commands prepared | `READINESS_AUDIT.md` |

## Task status summary

- COMPLETED/VERIFIED: R2-P02-T01 (protocol draft v0.1 — not approved), R2-P01-T02 (checkpoint inventory),
  R2-P04-T01, R2-P04-T02 (notebook reorganization; uncommitted), R2-P03-T01, R2-P03-T02,
  R2-P06-T01, R2-P06-T02 (designs, awaiting review), R2-P08-T01 (local part).
- BLOCKED (partial): R2-P01-T01 — local part done; server part needs server access (`READINESS_AUDIT.md` §4).
- NOT STARTED (assignable): R2-P02-T02, R2-P02-T03 (need user discussion),
  R2-P05-T01 (needs SECOND download authorization or data environment).
- BLOCKED: R2-P01-T03 (archives), R2-P02-T04, R2-P05-T02..T04, R2-P06-T03..T06, R2-P07-T01,
  R2-P08-T02..T04, all of P9–P12.

## Pending user decisions

1. Semantic prediction target and single-stage baseline (D-013).
2. Rules for selecting Research 1 checkpoints (D-012; principles in D-010 approved).
3. Full-system GPU budget policy (D-004); numbers after pilot (D-005).
4. Primary accuracy–efficiency criteria and decision rules (D-014, D-015).
5. ~~SECOND split structure (D-008, D-009)~~ — resolved 2026-11-07 (see Update 2026-11-07).
6. General augmentation/noise (D-018) — proposed: disabled.
7. Hypothesis formulations (D-002), pilot target (D-024), seeds/statistics (D-025), benchmarking (D-026).
8. HPO compatibility design review (D-022, G9) once R2-P06-T01 is delivered.

## Blockers and outstanding validation

- Server state unknown (artifacts/LFS, unfinished Research 1 searches, GPUs, SECOND) → R2-P01-T01.
- Any shared-code edit changes the HPO code signature → coordinate with Research 1 until R2-P06-T03.
- `prepare_second` coupled to the abandoned ROI-classifier preparation → R2-P05-T02.
- SECOND label palette mapping (`indexed_id_names` vs canonical IDs) unverified on real data.
- Latency sources disagree (`benchmark.json` vs `final_models_gpu_cost.csv`); re-measure under D-026.
- Provenance: Light training commit `7f38560` missing (F1); `02b72d7`/`81a5e53` now tagged locally (D-029).
- HPO signature differs between CRLF and LF checkouts (F3) — input for R2-P06-T01.

## Update 2026-11-06 (late) — R2-P08-T02 partial

Done: `src/satchange/data/second_binary.py` (SECOND root with `second_split_manifest.json` →
binary change detection through `create_dataset_dataloader`, HPO and `run_final_benchmark`);
optional HPO identity key `initial_weights = {path, sha256, exclude}` for LEVIR→SECOND fine-tuning
(fresh start only, hash-checked, strict except excluded prefixes). Tests: `tests/test_second_binary.py`.
Full pytest 137 passed / 3 skipped / 0 failed. **Open:** `mypy src` shows 43 errors vs 40 at HEAD
`b72cd6c` — 3 new typing errors (likely `final_benchmark.py`/`run.py` after the loader return-type
change). **Resolved 2026-11-07:** the errors came from `final_benchmark.py:922` using
`before_files/after_files/label_files`, which `SECONDBinaryDataset` lacked (a real `AttributeError`
for SECOND final benchmarks, not only typing); the attributes were added with a test. `mypy src` is
back to 40 (= HEAD baseline); full pytest 137 passed / 3 skipped / 0 failed.
Next: R2-P08-T03 pilot spec (needs D-024), R2-P05-T03 notebook (needs D-008/D-009).

## Session handoff (2026-11-06, end of session) — superseded by the 2026-11-07 handoff at the end of this file

**State:** committed on local branch `research2/foundation` (not pushed). Not committed on purpose:
the user's own edit in `notebooks/research_1/02_fc_siam_diff_full_hpo.ipynb` (2 lines), the
user's `Instruction.md` and `.clinerules/`, and the git-ignored local `AGENTS.md`.

**Done this session:** Research 2 docs (this folder); provenance tags (D-029, local only);
notebook split `research_1/` / `research_2/` (R2-P04); checkpoint inventory (R2-P01-T02);
reproducibility audit + regression test (R2-P03); HPO schema-4 code identity, compatibility
exceptions, `reuse_from` provenance, GPU-hour ledger and budget (R2-P06, `hpo_core.py`,
`hpo_accounting.py`, `hpo_backend.py`, `automated_hpo.py`); SECOND split manifests
(`semantic/second.prepare_second_splits`, R2-P05-T02); TinySiamese removal (R2-P07-T01).

**Tests:** full `pytest` 134 passed / 3 skipped / 0 failed; `mypy src` 40 errors identical to HEAD
`b72cd6c`; `ruff` clean on touched files. Key test files: `tests/test_hpo_compatibility.py`,
`tests/test_hpo_v3.py`, `tests/test_second.py`, `tests/test_semantic.py`.

**Important decisions:** D-022 (Q1–Q3), D-028/D-030/D-032 (autonomous single agent; ask only for
scientific decisions, big GPU spend, destructive/remote ops), D-006 shim for `roi.py` hash checks.

**Blocked / waiting:** GPU server (R2-P01-T01 server part, unfinished Research 1 searches — they
now need `acknowledge_code_change` or a new search), SECOND download, scientific decisions D-013,
D-008/D-009, D-024, D-005.

**Exact next action:** read `PROJECT_CONTEXT.md` §0, then this file; if the user has decided
D-008/D-009, implement R2-P05-T03 (`notebooks/research_2/01_second_dataset_preparation.ipynb`,
guarded, unexecuted) on top of `prepare_second_splits`; otherwise start the R2-P08-T02/T03 design
(SECOND binary loader via split manifest, minimal `initial_weights` key) and ask the user for D-013.

## Validation (2026-11-06, historical; see the 2026-11-07 handoff for the latest)

Full `pytest`: 134 passed, 3 skipped, 0 failed. `mypy src`: 40 pre-existing errors, identical to
HEAD `b72cd6c`. `ruff` clean on touched files. Nothing committed; provenance tags local only.

## Known technical limitations (current code)

- No GPU-hour budget/ledger; epoch-only `gpu_hours`; double counting across phases if summed.
- No dense SCD metrics, reconstruction, end-to-end latency, semantic models, SECOND binary loader.
- Local machine: CPU-only torch; datasets and artifacts not present locally (by design).

## Recommended next step

Locally executable, approved implementation work is exhausted. Remaining work needs input:
- **SECOND data** (download authorization or server) → R2-P05-T01, R2-P05-T03 notebook execution,
  R2-P08-T02 (SECOND binary loader).
- **Server access** → R2-P01-T01 server part, strict checkpoint loads, GPU reproducibility check.
- **Scientific decisions** → D-013 (semantic target / single-stage baseline) unblocks P10; D-008/D-009
  (splits), D-024 (pilot target), D-005 (budget, after pilot).
Next step that can proceed without data: R2-P05-T03 notebook skeleton (guarded, unexecuted) once
D-008/D-009 are decided; R2-P08-T02/T03 design can start in parallel.

## Update 2026-11-07 — R2-P05-T01 completed, SECOND splits decided

- **Installed** `data/SECOND` (git-ignored; 3.79 GB archive downloaded with the existing
  `bootstrap_second` download path). The release is nested (`SECOND_total_test.zip` →
  `test/`, `SECOND_train_set.rar` → `train/`), which `bootstrap_second` cannot extract; installed
  by hand with bsdtar into the layout `discover_second` expects; nested archive SHA256 are in
  `data/SECOND/source.json` (local, not author-published). Follow-up gap: teach `bootstrap_second`
  the nested layout (touches `semantic/`; HPO signature; not urgent).
- **Facts (observed):** train 2968 + test 1694 labeled pairs (= 4662), no official val; 512×512 RGB
  tiles, RGB-colour labels (all pixels match `SECOND_COLORS`, 0 unknown); T1/T2 change support
  identical in every tile; all six classes in both splits (playground rarest: 229 train / 101 test
  tiles); 5 train tiles without change, 0 in test; ~20 % changed pixels in both splits.
- **Leakage audit** (`second_audit.json`): 0 byte-identical images or shared stems; 2 near-duplicate
  pairs (1 within train, 1 train/test: 04041↔04070); 10 weak adjacent-edge candidates (4 within
  train, 6 train/test). Official IDs interleave train/test (random tile assignment, no geographic
  metadata) → geographic independence cannot be verified; impact on the official test is
  negligible (≤7 of 1694 tiles) and is reported, not repaired.
- **Decisions:** D-008 val = 15 % of train groups (444 tiles), seed 2099, within-train candidate
  pairs kept together; D-009 official test used. Manifest `data/SECOND/second_split_manifest.json`
  (sha256 `7d46718c…`): train 2524 / val 444 / test 1694; verified test == official test, no overlap,
  class presence proportional, loader reads all three splits.
- **Next:** R2-P05-T03 notebook (unexecuted skeleton using `prepare_second_splits` and
  `second_audit.py`); R2-P08-T03 pilot spec needs D-024.

## Session handoff (2026-11-07, end of session) — START HERE in a new window

**Restore context:** AGENTS.md §0 → `PROJECT_CONTEXT.md` §0 → `DECISION_LOG.md` → this section →
`IMPLEMENTATION_PLAN.md` (task `R2-P05-T03`). Reply to the user briefly in Russian (D-032).

**Git state:** branch `research2/foundation`, local only (not pushed). HEAD `cd0f2da` (SECOND install +
audit + D-008/D-009), previous `8e9e630` (SECONDBinaryDataset file maps), `54e4f77`, `dc33444`.
Deliberately uncommitted user files: `notebooks/research_1/02_fc_siam_diff_full_hpo.ipynb` (user edit),
`Instruction.md`, `.clinerules/`, git-ignored local `AGENTS.md`. Never commit/modify them.

**Done this session (2026-11-07):**
1. Fixed `SECONDBinaryDataset` missing `before_files/after_files/label_files` (real `AttributeError` in
   `final_benchmark` for SECOND, not only typing); test added in `tests/test_second_binary.py`.
   `mypy src` back to 40 (= baseline). Full pytest at that point: 137 passed / 3 skipped / 0 failed.
2. R2-P05-T01: SECOND downloaded (existing `bootstrap_second` download path) and installed by hand in
   `data/SECOND` (git-ignored). Release is nested (`SECOND_total_test.zip` → `test/`,
   `SECOND_train_set.rar` → `train/`); `bootstrap_second` cannot unpack it; I used Windows
   `C:\Windows\System32\tar.exe` (bsdtar, reads RAR). Nested SHA256 in `data/SECOND/source.json`.
3. Audit script `src/scripts/second_audit.py` → `docs/research2/second_audit.json` (counts, class
   shares, duplicates, near-duplicate/adjacent-edge candidates, `merged_groups`;
   `source_groups_from_report()` rebuilds the full group map).
4. D-008 / D-009 decided (see `DECISION_LOG.md`): official test (1694) used as is; val = 15 % of train
   groups, `split_seed=2099` → train 2524 / val 444 / test 1694; manifest
   `data/SECOND/second_split_manifest.json` (sha256 `7d46718c…`), frozen. Verified: test == official
   test, no overlap, merged groups intact, `create_dataset_dataloader` reads all three splits.

**Data facts to reuse (observed):** 512×512 RGB tiles; labels are RGB colours matching `SECOND_COLORS`
(0 unknown px); T1/T2 change support identical in every tile; ~20 % changed pixels; rarest class
playground (229 train / 101 test tiles); 5 train tiles without change; official IDs interleave
train/test (random assignment, no geography) → geographic independence unverifiable; leakage ≈ 1
cross-split near-duplicate pair (`04041`↔`04070`) + 6 weak cross-split adjacency candidates.

**Environment notes:** local machine is CPU-only; use `.\.venv\Scripts\python` (no system python on
PATH in the bash tool: call `.venv/Scripts/python`). Docs are CRLF — edit with a script that preserves
line endings. `data/` still holds temporary install leftovers (`.SECOND.incomplete-*` with the nested
archives, and the outer `SECOND-*.zip`, ~7.6 GB); safe to delete only with the user's OK (data/** rule).
Scratch analysis files (rows.pkl etc.) were in the session scratchpad and are not needed.

**Not done / open:**
- **R2-P05-T03** `notebooks/research_2/01_second_dataset_preparation.ipynb` — UNBLOCKED, not started:
  guarded (download flag default False), unexecuted, reuse `prepare_second_splits` +
  `second_audit.py`, document source/structure/classes/splits/leakage/"augmentation disabled" (D-018),
  no fabricated outputs; validate with `tests/test_notebooks.py`.
- Teach `bootstrap_second` the nested release layout (touches `semantic/` → HPO code signature; low priority).
- Full `pytest` was not re-run after adding `src/scripts/second_audit.py` and docs (only mypy/ruff);
  run it before the next commit that touches `src/satchange`.
- **User decisions still pending:** D-013 (semantic target / single-stage baseline, unblocks P10),
  D-024 (pilot target, needed for R2-P08-T03), D-005 (GPU budget after pilot), D-012, D-014/D-015,
  D-018, D-002, D-025/D-026, D-022/G9 HPO design review.
- Server-dependent: R2-P01-T01 server part, GPU reproducibility check, unfinished Research 1 searches.

**Exact next action:** implement R2-P05-T03 (notebook skeleton), run `tests/test_notebooks.py` and the
full pytest, update this file and the task status, commit locally; then ask the user for D-024 / D-013.
