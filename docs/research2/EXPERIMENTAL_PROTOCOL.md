# Research 2 — Experimental Protocol

**Version:** 0.1 (draft) · **Status: PROPOSED — NOT APPROVED.** · Created 2026-11-06.

Nothing in this document is a frozen methodological decision unless the corresponding entry in
[`DECISION_LOG.md`](DECISION_LOG.md) is APPROVED. Approval of the final protocol (v1.0) is gate
`R2-P02-T04`. Context: [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md).

Already approved inputs (USER-APPROVED 2026-11-06): the central question (D-001); allocated
GPU-hours as the accounting unit with separate training/HPO, inference and historical pretraining
costs (D-003); official SECOND splits preferred, test excluded from HPO/selection (D-007);
versioned checkpoint freeze policy (D-010).

---

## 1. Research question

*Under which accuracy and computational-budget conditions is a two-stage semantic change
detection system preferable to a single-stage semantic change detection model?* (D-001)

## 2. Hypotheses (PROPOSED, D-002)

The four provisional hypotheses from `Instruction.md` §5.2A are consolidated: H1 and H2 both
describe the conditional efficiency/accuracy trade-off and are merged into **H-A**; H3 becomes
**H-B**; a supporting diagnostic **H-C** explains failures; H4 becomes the pilot question **H-D**.

### H-A (central) — conditional non-inferiority at lower cost

For image pairs whose changed-area fraction ρ is below some density, the two-stage system B
reaches semantic quality non-inferior to the single-stage system A while having lower measured
end-to-end inference cost.

- Independent variable: system (A vs B). Moderator: ρ, binned into strata fixed before testing.
- Dependent: primary semantic metric (D-014), end-to-end latency per pair (p50/p95), GPU-ms per pair.
- Controls: same test pairs, same output grid, same hardware/precision/benchmark protocol, budget
  policy D-004, same semantic backbone where possible (D-013).
- Falsified if in no pre-declared stratum B is both non-inferior (lower CI bound of the quality
  difference > −δ) and faster (CI of the latency difference excludes 0).

### H-B (central, density dependence) — diminishing benefit

The latency ratio B/A increases with ρ and exceeds 1 above some density (more/larger crops,
overhead).

- Falsified if the ratio does not increase with ρ (no monotone trend across strata, CI-based), or if
  it is ≥ 1 in all strata (then B has no efficiency regime at all — itself a reportable result).

### H-C (supporting) — WHERE is the bottleneck

The quality gap between B with predicted ROIs and B with oracle (GT) ROIs is explained mainly by
change pixels outside predicted ROIs (stage-1 misses) rather than by WHAT errors inside ROIs.

- Measured by decomposing B's errors into outside-ROI misses vs inside-ROI errors. Diagnostic only;
  no confirmatory claim.

### H-D (pilot) — transfer from LEVIR-CD

FC-Siam-Diff initialized from a LEVIR-CD checkpoint needs fewer allocated GPU-hours than
training from scratch to reach a pre-declared validation quality on SECOND.

- Independent: initialization only (architecture, data, budget, schedule identical).
- Dependent: GPU-hours to threshold, validation curve vs epochs and vs GPU-hours, final validation
  quality within the budget.
- Falsified if scratch reaches the threshold within the same or lower GPU-hours, or neither
  condition reaches it within the bounded budget (inconclusive). Pilot = exploratory (few seeds).

## 3. Dataset protocol (G3 approved in principle; details PROPOSED)

1. Install SECOND once with `semantic.second.bootstrap_second` into the git-ignored `data/`
   directory; record `source.json` (local SHA256) — notebook
   `notebooks/research_2/01_second_dataset_preparation.ipynb` (D-011, `R2-P05-*`).
2. Inspect the actual split structure with `discover_second`. Preserve official labeled splits.
3. If no labeled validation split exists: carve validation from official train with a fixed seed
   (default `split_seed=2099` from the existing code), group-disjoint, persisted manifest (D-008).
4. If official test labels are unavailable, a labeled local test split from official train is
   unavoidable; this changes the meaning of "test" and needs explicit approval (D-009).
5. Grouping: only filename-level groups are available (`group_id` = stem). Investigate scene/
   location leakage (e.g. near-duplicate or adjacent tiles) before freezing; if grouping metadata
   is found, use grouped splitting.
6. Freeze dataset version (archive SHA256), split manifests (sample IDs), class mapping, ignored
   labels and preprocessing before any confirmatory experiment. All models use the same manifests.
7. Mandatory preprocessing only: exact label decoding (canonical IDs 1–6, 0 = no-change), scale to
   [0,1] float32 as in the existing loaders, no resizing at native 512×512 unless a model requires
   it (model-specific, documented in the experiment).
8. Augmentation: none by default. A general training augmentation (synchronized flips/rot90, as
   `RandomFlipRotate`) only if justified (D-018); applied on the fly with explicit seeds, never to
   validation/test, never baked into files.

## 4. Computational budget policy (PROPOSED, D-004/D-005)

| Option | Description | Pro | Contra |
|---|---|---|---|
| 1. Equal HPO GPU-hours per candidate model | each trained model gets the same HPO budget | simple | B gets 2× (two models) — unfair to A |
| **2. Equal total development GPU-hours per compared system (recommended)** | A and B get the same total optimization budget; B splits it between WHERE adaptation and WHAT; the split is declared beforehand and component costs are reported | answers the practical question "same development cost, which system is better?" | B's split is an extra degree of freedom; must be pre-declared |
| 3. Equal training budget, HPO reported separately | equal final-training GPU-hours; HPO cost only reported | cheap | HPO opportunity can differ silently |

Cost categories (D-003): **adaptation** (GPU-hours to adapt existing checkpoints to SECOND),
**full training** (including attributable LEVIR pretraining cost, reported separately and in a
sensitivity analysis), **inference** (per-pair end-to-end). Existing Research 1 checkpoints may be
used in a separately defined transfer condition; their LEVIR cost is never counted as zero.

Numerical limits: none yet (D-005). After the pilot, three options (resource-constrained /
balanced / comprehensive) will be proposed with the conclusions each can support.

## 5. Metrics (PROPOSED, D-014)

The metric set depends on the prediction representation (D-013). Recommended representation:
dense per-pixel T1 and T2 class maps with a no-change label, as in the SECOND benchmark, so that
A and B produce directly comparable outputs on the original grid.

**Primary (proposed, minimal):**
1. Semantic quality: **SeK** (separated kappa) — the standard SECOND metric — with **mIoU**
   (mean of change / no-change IoU) and **Fscd** as co-reported values. Definitions must be taken
   from and checked against a published reference implementation before adoption (UNVERIFIED here).
2. Efficiency: **end-to-end latency per image pair (median, p95)** on the frozen benchmark protocol.

**Secondary / diagnostics:** per-class IoU/F1, binary change precision/recall/F1/IoU, ROI union and
single-crop coverage ≥95 %, missed change pixels outside ROIs, false-crop fraction, crops per pair,
processed-area ratio, peak GPU memory, throughput, training/HPO GPU-hours.

**Two-stage latency must include:** (1) stage-1 inference, (2) mask/probability processing,
(3) ROI extraction, (4) crop preprocessing/batching, (5) stage-2 inference, (6) reconstruction.
Processed area is never used as a latency proxy.

## 6. Hardware and benchmarking (PROPOSED, D-026)

- Training: same GPU type for all compared conditions; record GPU model, count, driver, CUDA,
  torch versions (GPU model is currently not recorded in HPO configs — to add with accounting).
- Precision fixed per comparison (fp32 default; bf16 only if used for all compared systems).
- Latency: one dedicated GPU, no concurrent jobs, warm-up, CUDA-synchronized timing, batch size
  policy declared (batch 1 per pair as primary; crop batching inside B allowed and documented),
  TF32 state identical for A and B.
- Allocated GPU time (budget) ≠ GPU utilization (NVML, supplementary only) ≠ wall time.

## 7. Model selection (PROPOSED, D-010/D-012)

- Research 1 checkpoints are candidates, not settings. Before confirmatory experiments a **freeze
  manifest** records for each reused checkpoint: artifact path, SHA256, git commit, config,
  `model_parameters`, preprocessing, threshold source, and the selection rule applied. Later
  Research 1 results never change a running Research 2 experiment; replacements are new,
  documented experiments. No selection by test performance.
- WHERE candidates selected on SECOND validation by a pre-declared Pareto rule (end-to-end quality
  vs cost), not by pixel F1 alone.
- ROI threshold/policy, semantic hyperparameters and checkpoints (best validation epoch) selected
  on validation only; frozen before test.
- Architecture-capacity confound: compare A and B with the same semantic backbone where possible
  (D-013); report parameter counts and FLOPs where reliably measurable.

## 8. Repetitions and statistics (PROPOSED, D-025)

- Confirmatory systems: 3 training seeds (project default 42, 123, 777) per trained component.
- Report mean ± SD over seeds; paired bootstrap over test image pairs (same pairs for A and B),
  nested with seeds; 95 % CIs on differences; effect sizes.
- Non-inferiority margin δ fixed before test, from validation seed variability and practical
  relevance (not chosen after seeing test results).
- Pilot and exploratory findings are labelled as such and never pooled with confirmatory results.

## 9. Baselines and ablations

**Mandatory (proposed):** A — full-image semantic model; B-real — predicted ROIs from the frozen
WHERE detector; B-oracle — GT ROIs (upper bound, non-deployable); density stratification of all.
**Optional:** OR-ensemble WHERE, ROI threshold sensitivity, crop size/context ablation, an external
published SCD model as reference, adaptive ROI merging.

## 10. Decision rules (PROPOSED, D-015)

Per pre-declared density stratum:
- **B preferable:** non-inferior on the primary semantic metric and lower end-to-end latency (CI).
- **Trade-off:** B faster but inferior (or vice versa) → report the Pareto position; no winner.
- **A preferable:** B dominated (not faster and not better).
- **Stage-1 failure:** B inferior and H-C attributes the gap mainly to outside-ROI misses.
- **No demonstrated advantage:** CIs inconclusive.

## 11. Stopping and freeze conditions

- Budget exhaustion per D-004/D-005 stops new trials (accounting `R2-P06-*`).
- Dominated variants are dropped on validation evidence only.
- The test split is evaluated once per frozen system; re-evaluation after changes is a new,
  documented experiment.

## 12. Threats to validity

See `PROJECT_CONTEXT.md` §22. Key: capacity confounding, unequal optimization opportunity,
few seeds, spatial leakage, metric substitution (coverage/area as quality/latency), test reuse.
