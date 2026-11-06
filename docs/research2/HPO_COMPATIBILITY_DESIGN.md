# HPO Compatibility and GPU-Hour Accounting — Design

**Status: IMPLEMENTED 2026-11-06 (R2-P06-T03..T05), approved with Q1–Q3 (D-022).** Deviations from
the text below: a standalone `resume_from` checkpoint outside any search directory has unknown
provenance and cannot be continued; the commit-matching migration helper (§1.2 f) is not yet
implemented — `acknowledge_code_change` with evidence is the migration path; the binary
identity covers the LEVIR loader only — add the SECOND loader once it exists (R2-P08-T02).
Requirements: D-021, D-022 (`DECISION_LOG.md`). Evidence: `PROJECT_CONTEXT.md` §18,
`READINESS_AUDIT.md` F1–F3.

---

## Part 1 — Versioned code-compatibility mechanism (R2-P06-T01)

### 1.1 Current behaviour (VERIFIED CURRENT)

| Aspect | Code | Behaviour |
|---|---|---|
| Search identity | `hpo_core.Experiment.ensure_protocol` (`hpo_core.py:246–268`) | `pipeline_protocol.json` = schema 3, backend, **one** `backend_code_signature`, context, base parameters, stages, seeds, metrics, gaps, `reuse_from`, initial checkpoints; any difference → `ValueError` ("use a new search_name") |
| Signature | `SatelliteBackend.code_signature` (`hpo_backend.py:268–290`) | SHA256 over **raw working-tree bytes** of 9 named files + everything in `data/`, `models/`, `semantic/` → depends on line endings (F3), includes code irrelevant to the experiment (e.g. TinySiamese for a binary search) |
| Run matching | `hpo_core._matching/_source` (`hpo_core.py:272–297`), `hpo_backend._inspect` (`:292–417`) | canonical parameters + strict checks of model name, dataset, task, thresholds, precision, world size, metric consistency, best/last coherence — **no code check** |
| `reuse_from` | `hpo_backend.discover` (`:448–523`), `_train` (`:584–600`) | other search directories are scanned; their `pipeline_protocol.json` context (model, thresholds, task, dataset) is checked, **their code signature is not**; matching partial runs are continued by copying `last.pt`/`best.pt` into the new search (`_atomic_copy`, source untouched) |
| Legacy pre-v3 folders | `discover` (`automated_hpo_protocol.json`) | only with `allow_legacy_import=True`; model, thresholds, dataset checked; no code identity |
| Checkpoints | `training.py`, `hpo_backend.py:737` | store `config.code_revision = git HEAD` (not compared), no signature |

Net effect: the guard is **too broad** inside one directory (blocks on unrelated edits, differs
between Windows and Linux) and **absent** across directories (silent cross-code continuation via
`reuse_from`).

### 1.2 Design

**(a) Per-file code manifest instead of one hash.** The protocol records
`code_identity = {scheme: "satchange-code-v2", normalization: "CRLF→LF", files: {relpath: sha256}}`
for the experiment's dependency set. Hashing normalized content makes the identity equal for the
same git commit on Windows and Linux. A mismatch is reported as an exact list of changed / added /
removed files instead of an opaque hash difference.

**(b) Experiment-specific dependency set.** Instead of "all of `data/`, `models/`, `semantic/`":

| Category | Files (binary task) | Mismatch |
|---|---|---|
| `training` | `training.py`, `execution.py`, `losses.py`, `metrics.py`, `optimization.py`, `hpo_backend.py`, `data/__init__.py`, `data/levir_cd.py` (or the SECOND loader once it exists), `data/transforms.py`, `models/factory.py`, `models/__init__.py`, the module of `MODEL_REGISTRY[model_name]` and the `satchange` modules it imports | **blocks** |
| `control` | `hpo_core.py`, `automated_hpo.py` | **blocks** (trial generation, scheduling, stopping, selection) |
| `monitoring` | one new stdlib module for measurement and ledger I/O only (Part 2) | recorded in `code_history`, does not block |
| semantic task | adds `semantic/*` modules actually imported by the semantic path | blocks |

Model-specific closure: for `fc_siam_diff` the BIT and MiT files are not part of the identity, so
editing another model does not block. `models/__init__.py` and `factory.py` still import all
models — any edit there blocks every search (acceptable and honest).

**(c) What "monitoring" may contain.** Only code that reads clocks/devices, appends ledger records
and formats reports; it must not return values that the scheduler or trainer uses for decisions
(enforcement decisions live in `hpo_core`, category `control`). A unit test asserts the boundary
(the monitoring module imports no `satchange` training code; scheduler decisions do not depend on
its return values except the explicit budget numbers from Part 2). This is the only automatic
non-blocking path, satisfying "harmless monitoring changes ≠ behaviour changes".

**(d) Explicit, recorded acknowledgement — never automatic.** For a reviewed change in a blocking
file (e.g. a bug fix that provably does not alter trajectories), an explicit call
`acknowledge_code_change(search_directory, reason=..., evidence=...)` appends to
`code_history.json`: old/new file maps, changed files, reason, evidence (e.g. a test name), date.
Resume then proceeds and every later result row records the code-history index it ran under.
Without such an entry the run refuses. **Whether this escape path is acceptable at all is a
decision for the user (question Q1).** Without it, the only routes are finish-first or a new search.

**(e) `reuse_from` under the same rules.** For every reused source directory the controller
derives the source's code identity:

| Source | Identity available | Completed results reused | Partial runs continued |
|---|---|---|---|
| v4 (`code_identity`) | per-file map | yes, if no blocking difference, else only with acknowledgement | same rule |
| v3 (`backend_code_signature` only) | single hash; per-file map only via migration record (f) | with migration record: as v4; without: allowed **as historical results only**, flagged `code_identity: unknown` in the report | **refused** without migration record or acknowledgement |
| legacy `automated_hpo_protocol.json` | none | as v3 without record (requires `allow_legacy_import` as today) | refused |

A `reuse_manifest.json` in the new search lists each source directory, its identity, verdict and
the job ids that used it.

**(f) Legacy migration (evidence of identity, not equivalence).** A helper (read-only for the old
directory) searches git commits — including the local `provenance/*` tags — for one whose
recomputed **schema-3** signature (raw bytes, tried with LF and CRLF materialization) equals the
recorded value. On a match it writes a sidecar `code_identity_migration.json` *next to, not into,*
the old protocol: commit, per-file map of that commit, method. This proves which code produced the
directory; it does **not** claim the current code is equivalent — continuing still follows rules
(b)–(e). If no commit matches (uncommitted edits, `install_hpo_v3.py` patches, missing commit
`7f38560`), the directory stays read-only, its results remain usable as historical evidence and
the incompatibility is reported.

**(g) Unchanged guarantees.** Strict parameter/model/dataset/task/threshold/precision/world-size
checks in `_inspect`/`_train` stay as they are. Old `pipeline_protocol.json` files are never
rewritten. `PHASES`, parameter keys, `model__` routing, checkpoint keys stay.

### 1.3 Implications

| Object | Effect |
|---|---|
| `.pt` checkpoints (all) | none — not read for identity, not modified |
| `artifacts/final_models/**` | none — `reconstruct_checkpoint`/`run_final_benchmark` do not use signatures |
| Finished Research 1 searches | readable as before; reusable as results (with identity flag) |
| **Unfinished** Research 1 v3 searches | after the change, schema 3 ≠ schema 4 → same `search_name` cannot continue under new code (as today after any edit). Options: finish them **before** the change; or continue in a new search via `reuse_from` + migration record (+ acknowledgement if blocking files changed) |
| Legacy pre-v3 folders | unchanged behaviour plus explicit identity flag |
| Tests | `tests/test_semantic_hpo.py:528` rewritten to assert the new property (semantic/controller edits are detected for the tasks that use them); new tests for normalization, closure, reuse verdicts, migration matching, acknowledgement log |
| `src/scripts/install_hpo_v3.py` | patches sources → produces identities matching no commit; documented as unsupported for new searches |

### 1.4 Open questions (need the user — G9)

- **Q1.** Allow the explicit acknowledgement path (d), or require finish-first / new search only?
- **Q2.** Which Research 1 searches are unfinished and must stay resumable? (Server data,
  `READINESS_AUDIT.md` §4 — blocked.) Finishing them before the change removes most migration work.
- **Q3.** Default for reusing *completed* results from sources with unknown code identity: allow
  with a flag (recommended — they are immutable historical evidence) or refuse?
- File-level granularity is accepted as a known limitation: `training.py` mixes trainer semantics
  with DDP/checkpoint code, so any edit there blocks.

---

## Part 2 — GPU-hour accounting and budget enforcement (R2-P06-T02)

### 2.1 Definition and scope (D-003)

`GPU-hours = Σ_allocations (end − start) × number_of_devices / 3600`, measured **in the controller**
from job dispatch to observed completion. An allocation covers worker start, CUDA init, data and
model setup, checkpoint copy/writes, training, validation and teardown; failed and OOM attempts
are allocations too. Queue time (pending, not dispatched) is excluded. Polling (`wait(timeout=0.5)`,
`hpo_core.py:540`) can over-count by ≤ 0.5 s per allocation — conservative. HPO v3 jobs use one
device; the formula supports lists of devices for future multi-GPU jobs. DDP runs via `torchrun`
are outside the HPO controller and not covered (documented limitation).

The existing per-epoch `gpu_hours` (epoch compute) is kept unchanged for compatibility and reported
alongside as `epoch_compute_gpu_hours`; per-phase reports add the **incremental** value (epochs
`> source_epoch` only), removing today's double counting across phases.

### 2.2 Records

- **Ledger:** `<search>/gpu_ledger.jsonl`, append-only, one JSON object per line, `flush` + `fsync`
  per record. Single writer: the controller process (the `.hpo_v3.lock` already guarantees one
  controller per search directory); workers never write the ledger.
- **Events:** `start` (allocation id = uuid4, job id, attempt index, operation id, phase, parameters
  identity, seed, model, devices with GPU name, wall-clock UTC, code identity, environment ids),
  `heartbeat` (every 60 s per active allocation), `end` (status: completed / paused / oom / failed /
  interrupted, duration, checkpoint references, validation metrics, error), `budget` (set/raise with
  reason), `overshoot` (amount, cause).
- **Derived summary:** `gpu_budget.json` (budget, spent, active estimate, remaining, overshoot),
  rewritten atomically with the existing `write_json`; always recomputable from the ledger.
- **Per-trial:** result rows gain `allocated_gpu_hours`, `allocation_ids`; operation reports gain
  `operation_allocated_gpu_hours`.

### 2.3 Enforcement

1. `run_automated_parameter_search(..., gpu_hour_budget: float | None = None)` — `None` keeps today's
   behaviour (accounting still recorded). The budget is stored in the ledger, not in
   `pipeline_protocol.json`, so raising it is an explicit recorded event, never a protocol change.
2. **Before dispatch:** `remaining = budget − spent − Σ active elapsed`. A job is dispatched only if
   `remaining > expected_cost`, where `expected_cost = (target − source_epoch) × seconds_per_epoch`
   estimated from completed allocations of the same model/phase (none known → dispatch only if
   `remaining > 0`, and mark the estimate as unknown).
3. **During execution:** after each wait cycle the controller updates the active estimate; when
   `spent + active ≥ budget` it touches the existing `PAUSE_AFTER_EPOCH` marker → active jobs save
   at the next epoch boundary (existing exact-resume path) and no new job is dispatched.
4. **Overshoot** = final spent − budget; bounded by about one epoch plus teardown per active job;
   always written as an `overshoot` event and shown in the report. Zero overshoot is not promised.
5. A later explicit call with an exhausted budget refuses to dispatch (clear error) unless the
   budget was raised with a recorded reason.

### 2.4 Crash and resume

| Situation | Handling |
|---|---|
| Normal completion | `start` + `end` |
| Controller restart | ledger replay; allocations with `start` but no `end` are closed as `interrupted` at their last heartbeat (charged), flagged `uncertain_seconds ≤ 60` |
| Worker crash / CUDA OOM | the future raises or returns `oom` → `end` with that status; time charged; retry is a new allocation |
| Ctrl+C | existing pause path; active allocations end as `paused` |
| Budget exhaustion | §2.3 steps 3–5 |
| Interrupted ledger write | last line without newline/invalid JSON is ignored on replay and reported; at most one event lost, covered by the heartbeat rule |
| Duplicate prevention | allocation ids are unique; replay counts each id once; `end` for an unknown or already-closed id is ignored and reported |

### 2.5 Tests (R2-P06-T05)

Injected clock and synthetic CPU "GPU" queue (as in `tests/test_semantic_hpo.py`): single device,
four parallel devices, unallocated devices (no charge), budget boundary (exact and ±ε), mid-trial
exhaustion (pause + overshoot record), controller crash + replay (heartbeat charge), failed and OOM
attempts charged, duplicate ids, graceful stop, `gpu_hour_budget=None` backward compatibility,
`reuse_from` provenance records, legacy directories untouched.

### 2.6 Placement and order

Ledger/measurement in the new `monitoring` module (stdlib only, like `hpo_core`); enforcement in
`hpo_core._execute`; parameter in `automated_hpo`; GPU name capture through the backend's
`resources()` (already queries each device). Implement Part 1 first, then Part 2, in the shared
codebase; both touch `control` files, so they follow the rules of Part 1 themselves.
