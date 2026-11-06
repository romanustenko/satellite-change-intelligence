# Research 2 — Decision Log

Statuses: **PROPOSED · APPROVED · REJECTED · SUPERSEDED.** "APPROVED" is used only with explicit
user approval; the source is cited. AI recommendations are never approved by default. An agent
must never silently change an APPROVED decision: if new evidence conflicts with one, add a note
under the decision and propose reconsideration.

Sources: **U-2026-11-06** = the user's chat message of 2026-11-06 (decisions on G1–G8, sections
1–6, and two follow-ups on shared code/HPO compatibility and TinySiamese cleanup);
**Instruction.md** = root `Instruction.md` (untracked), which the user designated the approved
source of implementation requirements but not of scientific decisions.

| ID | Topic | Status |
|---|---|---|
| D-001 | Central research question | APPROVED (G1) |
| D-002 | Hypothesis formulations | PROPOSED |
| D-003 | GPU-hours as primary accounting unit; cost categories | APPROVED (G2, partial) |
| D-004 | Full-system budget fairness policy | PROPOSED |
| D-005 | Numerical GPU-hour limits | PROPOSED (deferred to pilot) |
| D-006 | TinySiamese abandoned; remove remaining components | APPROVED |
| D-007 | Official SECOND splits preferred; test excluded | APPROVED in principle (G3) |
| D-008 | Validation split creation if missing | PROPOSED |
| D-009 | Test split if official test labels unavailable | PROPOSED |
| D-010 | Versioned Research 1 checkpoint selection/freeze policy | APPROVED (principles) |
| D-011 | Dedicated SECOND preparation notebook | APPROVED |
| D-012 | Selection rules for WHERE models, ROI policies, checkpoints | PROPOSED (G5 not approved) |
| D-013 | Semantic prediction target and single-stage baseline | PROPOSED (open) |
| D-014 | Primary metrics | PROPOSED (G4 not finalized) |
| D-015 | Decision rules and non-inferiority margin | PROPOSED |
| D-016 | Notebook reorganization research_1 / research_2 | APPROVED (G6) |
| D-017 | GPU-hour accounting inside existing HPO | APPROVED in principle (G7) |
| D-018 | General augmentation / noise policy | PROPOSED |
| D-019 | Reproducibility audit is read-only | APPROVED |
| D-020 | FC-Siam-Diff pilot (scratch vs LEVIR fine-tune) | PROPOSED; launch NOT approved (G8) |
| D-021 | One shared codebase; provenance over bitwise compatibility | APPROVED |
| D-022 | Versioned HPO compatibility mechanism | APPROVED (design + Q1–Q3, 2026-11-06) |
| D-023 | Source hierarchy and persistence of documentation | APPROVED |
| D-024 | Pilot target representation | PROPOSED |
| D-025 | Seeds and statistical analysis | PROPOSED |
| D-026 | Hardware and latency benchmarking conditions | PROPOSED |
| D-027 | Environment facts (local `data/`, local GPU) | APPROVED |
| D-028 | Autonomous working policy for technical work | APPROVED |
| D-029 | Local provenance tags for unreferenced Research 1 training commits | APPROVED |
| D-030 | Autonomy policy v2 — gates are verification checkpoints | APPROVED |
| D-031 | Specialized subagents (SCD, SECOND, HPO) | SUPERSEDED by D-032 |
| D-032 | Single autonomous agent, lean workflow | APPROVED |

---

### D-001 — Central research question
- **Source:** U-2026-11-06 (G1 "provisionally approved").
- **Question:** What does Research 2 investigate?
- **Decision:** *Under which accuracy and computational-budget conditions is a two-stage semantic
  change detection system preferable to a single-stage model?* Two-stage superiority is not assumed.
- **Status:** APPROVED. Hypotheses themselves remain PROPOSED (D-002).
- **Impact:** All tasks. **Revisit:** only by the user.

### D-002 — Hypothesis formulations
- **Source:** Instruction.md §5.2A (provisional H1–H4); proposal in `EXPERIMENTAL_PROTOCOL.md` §2.
- **Alternatives:** keep H1–H4 separately; consolidate into H-A (conditional non-inferiority at
  lower cost), H-B (density dependence), H-C (WHERE bottleneck, supporting), H-D (transfer, pilot).
- **Recommendation:** consolidation — H1 and H2 test the same conditional trade-off; separate
  hypotheses inflate multiplicity without new information.
- **Status:** PROPOSED. **Impact:** R2-P02-T02..T04, P11, P12.
- **Revisit:** after pilot or if the semantic representation (D-013) changes measurability.

### D-003 — Accounting unit and cost categories
- **Source:** U-2026-11-06 (G2 partially approved); Instruction.md §6.2.
- **Decision:** primary unit = allocated GPU-hours, `Σ_i T_i / 3600` over GPUs actually allocated
  (not available GPUs, not utilization). Report separately: training/HPO, inference, historical
  pretraining. NVML utilization is supplementary only.
- **Evidence:** current code measures epoch compute only (`training.py:1083`), not allocation.
- **Status:** APPROVED. **Impact:** R2-P06-*, R2-P09-*, P11.
- **Revisit:** if allocation time cannot be measured reliably for some execution mode.

### D-004 — Full-system budget fairness policy
- **Source:** U-2026-11-06 asks to *recommend* a shared total optimization budget per compared
  system with component-level reporting; Instruction.md §5.2C.
- **Alternatives:** (1) equal HPO GPU-hours per candidate; (2) equal total development GPU-hours per
  system, B splitting its budget between stages, split pre-declared; (3) equal training budgets with
  HPO reported separately.
- **Recommendation:** (2), component costs always reported; LEVIR pretraining in a sensitivity analysis.
- **Status:** PROPOSED. **Impact:** P9, P11. **Revisit:** after pilot measurements.

### D-005 — Numerical GPU-hour limits
- **Source:** U-2026-11-06 ("numerical limit NOT approved yet; informed by the pilot").
- **Status:** PROPOSED (deferred). Three budget options will follow the pilot (R2-P09-T02).

### D-006 — TinySiamese abandoned; remaining components to be removed
- **Source:** U-2026-11-06 (model intentionally removed; do not restore or reimplement; removal of
  remaining components approved, in a later implementation phase).
- **Evidence:** notebooks deleted in commits `60f4fca`, `b72cd6c`; code remains
  (`models/tiny_siamese_change_classifier.py`, `factory.model_class_for_task`, semantic branches of
  `training.py`, `hpo_backend.py`, `automated_hpo.py`, `final_benchmark.py`, `semantic/*`,
  `configs/tiny_siamese_change_classifier.yaml`, `docs/semantic_classifier.md`, `tests/test_semantic*.py`).
- **Decision:** remove TinySiamese-specific code, configs, tests, factory registrations and docs;
  preserve reusable SECOND preparation, ROI extraction and detector inference components.
- **Status:** APPROVED. **Impact:** R2-P07-T01 (after R2-P05-T02 and R2-P06).
- **Executed 2026-11-06.** Technical decision recorded: `models/tiny_siamese_change_classifier.py`
  is kept as a one-function shim (`validate_unit_tensor`) because `semantic/roi.py` imports it and
  must stay byte-identical for the Research 1 H08–H11 ROI hash checks (notebooks 08–11).
- **Revisit:** never restore; a new semantic model follows D-013.

### D-007 — Official SECOND splits; test excluded
- **Source:** U-2026-11-06 (G3 approved in principle).
- **Decision:** prefer official splits, subject to verification of structure and leakage. The test
  split is excluded from HPO, model selection and evaluation-driven preprocessing decisions. No
  random repartition of the whole dataset and no redistribution of official test samples without
  explicit approval.
- **Status:** APPROVED (in principle). **Impact:** R2-P05-*, all experiments.

### D-008 — Validation split if missing
- **Source:** U-2026-11-06 §4.2; existing `prepare_second` local holdout (15 % of groups per missing
  split, `split_seed=2099`).
- **Recommendation:** carve validation from official train only, fixed seed, group-disjoint,
  persisted manifest of sample IDs, sample counts reported; investigate tile adjacency/near-duplicates
  first. Fraction (15 % as in existing code, or another) to be decided with the data in hand.
- **Status:** PROPOSED. **Impact:** R2-P05-T03/T04.

### D-009 — Test split if official test labels are unavailable
- **Source:** finding — `discover_second` treats an unlabeled `test` leaf as empty; public test-label
  availability is UNKNOWN locally (commonly reported as unavailable — not verified).
- **Alternatives:** (a) local labeled test from official train (group-disjoint, frozen early);
  (b) report only validation results (no confirmatory test) — scientifically weak.
- **Recommendation:** (a), created once and frozen before any model development on SECOND.
- **Status:** PROPOSED; requires explicit approval because it changes what "test" means.

### D-010 — Versioned Research 1 checkpoint selection policy
- **Source:** U-2026-11-06 §1.
- **Decision (principles):** Research 1 continues independently; Research 2 may reuse selected
  checkpoints; before confirmatory experiments freeze checkpoint, configuration, code version,
  preprocessing and selection criteria (freeze manifest); new Research 1 results never change an
  ongoing Research 2 experiment; any later replacement is a separate documented experiment; no
  checkpoint selection by test performance. Research 1 metrics are historical evidence/starting
  points, not approved Research 2 settings.
- **Status:** APPROVED. **Impact:** R2-P01-T02, R2-P11-T05.

### D-011 — Dedicated SECOND preparation notebook
- **Source:** U-2026-11-06 §4.
- **Decision:** `notebooks/research_2/01_second_dataset_preparation.ipynb` installs (reusing
  `bootstrap_second`), verifies, documents source/version/structure/classes, establishes reusable
  split manifests, diagnostics, essential preprocessing and (only if justified) a documented general
  augmentation policy. Small improvements go into `src/satchange/semantic/second.py`, no parallel
  pipeline. Experiment-specific transforms stay in experiment notebooks. No fabricated statistics.
- **Status:** APPROVED. **Impact:** R2-P05-*.

### D-012 — Selection rules (WHERE, ROI, checkpoints)
- **Source:** U-2026-11-06 (G5 not approved); hand-off shortlist (Medium Full, Heavy Full, Light
  Resized; Light Full control; OR optional) — REPORTED, non-binding.
- **Recommendation:** pre-declared validation-only Pareto rule on end-to-end quality vs cost on
  SECOND; ROI policy candidates from H11 (grid256+ctx16, grid128+ctx16, adaptive64+ctx16) measured,
  not inherited.
- **Status:** PROPOSED. **Impact:** P11.

### D-013 — Semantic prediction target and single-stage baseline
- **Source:** U-2026-11-06 (architecture/output must be discussed first).
- **Alternatives:** (a) dense per-pixel T1/T2 class maps with no-change (SECOND benchmark
  convention); (b) ROI-level transition labels (old TinySiamese approach — not comparable to a dense
  single-stage output). Baseline: (i) the same semantic backbone on the full image (isolates the
  ROI-gating effect); (ii) an external published SCD architecture (credibility, extra cost).
- **Recommendation:** (a) + (i) mandatory; (ii) optional.
- **Status:** PROPOSED (open). **Impact:** R2-P10-*, D-014.

### D-014 — Primary metrics
- **Source:** U-2026-11-06 (G4 not finalized).
- **Recommendation:** semantic: SeK primary with mIoU and Fscd (definitions to be verified against a
  reference implementation); efficiency: end-to-end latency per pair (median, p95). Diagnostics per
  `EXPERIMENTAL_PROTOCOL.md` §5.
- **Status:** PROPOSED. Depends on D-013.

### D-015 — Decision rules and non-inferiority margin
- **Recommendation:** per-density-stratum rules in `EXPERIMENTAL_PROTOCOL.md` §10; δ fixed before
  test from validation seed variability and practical relevance.
- **Status:** PROPOSED.

### D-016 — Notebook reorganization
- **Source:** U-2026-11-06 (G6 approved); Instruction.md §3.
- **Decision:** move existing notebooks to `notebooks/research_1/` (git-aware moves, no renames),
  create `notebooks/research_2/`; preserve the user's uncommitted modification of
  `02_fc_siam_diff_full_hpo.ipynb` and other uncommitted work; fix root resolution, documentation
  references and the non-recursive discovery in `tests/test_notebooks.py`; no change to Research 1
  training behaviour; no placeholder notebooks.
- **Status:** APPROVED. **Impact:** R2-P04-*. Not executed yet.

### D-017 — GPU-hour accounting inside existing HPO
- **Source:** U-2026-11-06 (G7 approved in principle); Instruction.md §6.
- **Decision:** extend `run_automated_parameter_search` / `hpo_core` / `hpo_backend`; no second HPO
  framework; backward compatible; no double counting of epochs inherited across phases; persistent
  ledger; resume without budget reset; implementation only after the accounting design is reviewed.
- **Status:** APPROVED in principle. **Impact:** R2-P06-T02..T05. Depends on D-022.

### D-018 — General augmentation / noise
- **Source:** U-2026-11-06 §4.4.
- **Recommendation:** disabled by default; enable synchronized flips/rot90 (existing
  `RandomFlipRotate`) only if justified by the protocol and applied identically to all compared
  systems; no artificial noise; never on validation/test.
- **Status:** PROPOSED.

### D-019 — Reproducibility audit is read-only
- **Source:** U-2026-11-06 (G7 note); Instruction.md §8.
- **Decision:** the audit modifies no code and no determinism settings; fixes require explicit permission.
- **Status:** APPROVED. **Impact:** R2-P03-*.

### D-020 — FC-Siam-Diff pilot
- **Source:** Instruction.md §7; U-2026-11-06 (G8 not approved).
- **Decision so far:** only initialization differs between conditions (scratch vs compatible
  LEVIR weights); identical architecture, data, schedule, budget, seeds; no test set.
- **Status:** design PROPOSED; launch **NOT APPROVED** until checkpoint compatibility, SECOND
  configuration, controls and pilot budget are reviewed.

### D-021 — One shared codebase
- **Source:** U-2026-11-06 (follow-up "Shared codebase and HPO compatibility").
- **Decision:** both studies use the same `src/satchange/`; no Research-2 copies of `training.py`,
  `hpo_core.py`, `hpo_backend.py` or other shared modules to preserve historical signatures; no
  separate repositories, worktrees or duplicate source trees. Scientific traceability and
  maintainability take priority over bitwise compatibility. Historical experiments are not repaired
  retroactively without evidence; their provenance is preserved.
- **Status:** APPROVED.

### D-022 — Versioned HPO compatibility mechanism
- **Source:** U-2026-11-06 follow-ups; finding in `PROJECT_CONTEXT.md` §18.
- **Problem:** `backend_code_signature` (one hash over a broad file set) blocks resume after any
  edit, while `reuse_from` bypasses code checks entirely.
- **Approved requirements:** distinguish training-semantics changes from infrastructure-only changes;
  keep strict validation of architecture, optimizer, scheduler, dataset configuration and other
  training-critical parameters; preserve historical signatures and metadata (no rewriting of old
  protocols); a mismatch in an infrastructure signature does **not** automatically permit resume —
  `hpo_core.py`/`automated_hpo.py` changes can alter trial generation, scheduling, stopping or
  checkpoint selection; only changes demonstrably limited to monitoring/logging may be admitted;
  `reuse_from` sources are included in compatibility verification; a matching historical signature
  is evidence of code identity, not of behavioural equivalence with the current implementation;
  legacy experiments that cannot safely resume keep their results and the incompatibility is
  reported; GPU-hour accounting integrates without duplicating infrastructure; minimal design.
- **Preliminary direction (PROPOSED):** protocol records component signatures —
  `training_signature` (training/loss/metric/execution/backend setup, dataset module, model module +
  factory) and `infrastructure_signature` (controller); both block by default; an explicit,
  reviewed classification admits monitoring-only changes and appends to a `code_history` audit log;
  accounting lives in the controller. Legacy migration: find the git commit that reproduces the
  recorded old signature (evidence of identity) and write a separate migration record; otherwise
  the directory stays read-only.
- **Open design questions:** see R2-P06-T01; concrete design delivered 2026-11-06 in
  `HPO_COMPATIBILITY_DESIGN.md` (questions Q1–Q3 for the user).
- **Status:** APPROVED 2026-11-06 — design `HPO_COMPATIBILITY_DESIGN.md` authorized for
  implementation with the user's answers: **Q1** explicitly documented compatibility exceptions are
  allowed only when supported by evidence, with preserved provenance and audit records; critical
  checks are never bypassed silently. **Q2** the GPU server is unavailable to the agent: record the
  missing evidence (unfinished Research 1 searches unknown) and continue locally. **Q3** historical
  results with unknown code provenance may be reused for exploratory analysis only, clearly
  labelled; never treated as verified reproducible results or used in confirmatory comparisons
  without validation.

### D-023 — Source hierarchy and persistence
- **Source:** U-2026-11-06 §5–6.
- **Decision:** `Instruction.md` = approved implementation requirements, not approval of scientific
  choices; the hand-off = historical context, cross-checked against the repository; missing H1–H11
  archives → REPORTED / NOT REVERIFIED; core requirements live in git-tracked `docs/research2/`;
  `AGENTS.md` (git-ignored) is only a pointer.
- **Status:** APPROVED.

### D-024 — Pilot target representation
- **Alternatives:** (a) binary change on SECOND (`label > 0`), identical 1-channel head, only
  initialization differs — measures the WHERE-adaptation cost needed by every two-stage system and
  does not depend on D-013; (b) semantic head on FC-Siam-Diff — requires head adaptation and the
  open semantic target decision.
- **Recommendation:** (a) first; (b) only after D-013.
- **Status:** PROPOSED.

### D-025 — Seeds and statistics
- **Recommendation:** 3 seeds (42, 123, 777) for confirmatory components; paired bootstrap over test
  pairs nested with seeds; mean ± SD, 95 % CIs, effect sizes; pilot separate.
- **Status:** PROPOSED.

### D-026 — Hardware and latency benchmarking
- **Recommendation:** same GPU type for all compared systems; record GPU model/driver/CUDA/torch;
  dedicated GPU for latency, warm-up, synchronized timing, identical precision/TF32 for A and B.
- **Status:** PROPOSED.

### D-027 — Environment facts
- **Source:** U-2026-11-06 §2.
- **Decision:** absence of `data/` locally is expected (git-ignored; populated by notebook code);
  the local machine is not the training environment; documentation and CPU-only validation run
  locally; GPU experiments run later elsewhere; no speculative CUDA code changes because of the
  local lack of GPU.
- **Status:** APPROVED.

### D-028 — Autonomous working policy for technical work
- **Source:** user message of 2026-11-06 ("maximum autonomous execution speed").
- **Decision:** the agent independently makes technical decisions, implements, verifies and
  coordinates tasks via `IMPLEMENTATION_PLAN.md`; no confirmation needed for approved tasks,
  ordinary files/tests/docs, obvious fixes or safe reversible local git operations; after each task:
  validate, update `PROGRESS.md`/plan, short Russian report (Completed/Why/Verified/Remaining/Next),
  continue with the next unblocked task. Server-only checks are prepared and recorded, never claimed.
  Scientific decisions (question, hypotheses/protocol, splits, primary metrics, selection criteria,
  comparison rules, confirmatory budgets), substantial GPU runs, pushes/remote history rewrites and
  Research 1 artifacts remain gated. Supersedes older instructions requiring approval after every
  ordinary technical task (`AGENTS.md` §10, `.clinerules` "present a plan / ask").
- **Status:** APPROVED. **Impact:** all tasks; text in `PROJECT_CONTEXT.md` §0.

### D-029 — Local provenance tags for unreferenced Research 1 training commits
- **Source:** user approval of 2026-11-06; finding `READINESS_AUDIT.md` F2.
- **Decision:** annotated local tags `provenance/r1-medium-training-02b72d7` and
  `provenance/r1-heavy-resized-training-81a5e53` (identities verified). Not pushed; historical
  commits unchanged. Pushing remains a separate decision.
- **Status:** APPROVED and executed 2026-11-06.

### D-030 — Autonomy policy v2: approval gates are verification checkpoints
- **Source:** user message of 2026-11-06 ("permanently change our approval workflow").
- **Decision:** the agent is the lead autonomous engineer. It makes technical and architectural
  decisions, reviews its own designs,
  implements and proceeds to the next approved or technically necessary task without asking.
  Gates G1–G11 act as documentation/verification checkpoints. The user is asked **only** for
  decisions that change the research question, hypotheses, evaluation methodology, dataset
  evaluation splits or final model comparison rules, for substantial GPU expenditure, and before
  destructive operations on historical research artifacts or irreversible remote changes.
  Everything else: decide, record (this log / `PROGRESS.md`), continue. Missing information is
  documented and independent work continues. Milestone reports are brief, in Russian.
- **Supersedes:** the approval-related parts of D-028 (its other rules remain).
- **Status:** APPROVED.

### D-031 — Specialized subagents
- **Source:** user message of 2026-11-06.
- **Decision:** three project subagents defined in `.claude/agents/`: `scd-researcher` (Sonnet;
  read-only + web; SCD model research), `second-specialist` (Sonnet; SECOND analysis and
  `semantic/second.py` implementation), `hpo-specialist` (Opus; read-only reviews of HPO,
  reproducibility, accounting, compatibility). Each writes only its own report/files; the lead
  engineer integrates results and keeps scientific decisions with the user.
- **Status:** SUPERSEDED by D-032 (2026-11-06). The agents were stopped and `.claude/agents/`
  removed. Kept from their partial work: `semantic/second.py` split-manifest decoupling +
  `tests/test_second.py` (reviewed, tested). Not delivered: SCD model review, SECOND dataset
  notes, HPO design review.

### D-032 — Single autonomous agent, lean workflow
- **Source:** user message of 2026-11-06 (optimize limited Claude Pro usage).
- **Decision:** no subagents or automatic delegation; one agent owns the whole Research 2 workflow
  (HPO/GPU-hours, SECOND, reproducibility/compatibility, SCD models, tests, docs). Avoid repeated
  audits, excessive planning and unnecessary documentation; reuse existing code and context;
  continue to the next approved, unblocked task; brief Russian reports. Scientific approval for
  fundamental methodological decisions and expensive GPU experiments is unchanged (D-030).
- **Status:** APPROVED.
