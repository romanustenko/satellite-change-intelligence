# Satellite Change Intelligence — full report

> The short version is the [README](README.md). This document is for readers
> who want the whole story: how the study was designed, what was found, which decisions were taken
> and why, and what went wrong on the way.

## Contents

1. [Motivation and question](#1-motivation-and-question)
2. [Timeline](#2-timeline)
3. [Study design](#3-study-design)
4. [Hypotheses and outcomes](#4-hypotheses-and-outcomes)
5. [Results](#5-results)
6. [When does a two-stage system pay off?](#6-when-does-a-two-stage-system-pay-off)
7. [Key decisions](#7-key-decisions)
8. [Mistakes and lessons](#8-mistakes-and-lessons)
9. [Limitations](#9-limitations)
10. [Future work](#10-future-work)
11. [Reproducibility](#11-reproducibility)

---

## 1. Motivation and question

Semantic change detection answers two questions at once: **where** did something change between
two images, and **what** was it before and after. Models that do both are expensive. A natural
engineering idea is to split the work: a cheap binary detector finds candidate regions, and the
expensive semantic model looks only at those regions.

**Central question:** *under which accuracy and compute conditions is a two-stage semantic change
detection system preferable to a single-stage one?* A negative answer was declared a valid outcome
from the start.

## 2. Timeline

| When (2026) | Phase | Outcome |
|---|---|---|
| Aug 24 – Oct 30 | **Research 1:** binary change detection on LEVIR-CD (542 hand-run HPO trials, then hypotheses H1–H11), three tiers (Light FC-Siam-Diff, Medium BIT, Heavy MiT-B3) × Full / Resized input | six final detectors; Full resolution clearly better than Resized; Heavy best F1 |
| Nov 6–10 | **Research 2, iteration v1:** one self-built semantic model vs one two-stage system | two-stage slower and less accurate — but the study compared the wrong things (§8.1); archived |
| Nov 10 | **Scope correction:** 4 system types × 3 published SCD architectures, equal-FLOP budget | design v2 written down before any new code |
| Nov 10–12 | Detector screening on SECOND validation | Medium (BIT) chosen for the two-stage systems |
| Nov 12–13 | Pilot: learning rate and common training budget | budget b* = 3.2 × 10¹⁷ FLOPs per final run |
| Nov 16–18 | HPO (6 trials per model, equal FLOPs) | learning rate / weight decay per model |
| Nov 18–21 | Final training, 3 seeds × 3 models; ROI policies on validation; freeze | protocol v0.4 frozen with hashes |
| Nov 21–22 | **Test split read once**; analysis; latency | results below |
| Nov 22–23 | Web demo; exploratory cost checks (§6) | this report |

## 3. Study design

### 3.1 Systems (12)

| Case | System | Detector (*where*) | Semantic model (*what*) |
|---|---|---|---|
| 1 | single-stage | — | whole 512² tile |
| 2 | two-stage | BIT, pre-trained on LEVIR-CD, fine-tuned on SECOND | context windows around detected boxes |
| 3 | two-stage | BIT trained on SECOND only | same |
| 4 | two-stage | OR-ensemble: BIT ∨ FC-Siam-Diff (Resized) | same |

× three semantic models, ordered by complexity:
**Siamese U-Net + ResNet-18** (CNN), **SCanNet + ResNet-34** (CNN + transformer),
**MambaSCD-Tiny** (state-space model). The two-stage systems reuse exactly the single-stage
checkpoint — only the input changes — so any difference comes from the detector and the cropping.

**Two-stage mechanics:** detector probability map → threshold → connected components → boxes
(margin, merge gap, minimum side) → each box is processed in the smallest 128/256/384/512 window
that contains it → predictions merged back; everything outside the boxes is "no change", so a missed
change is an error.

### 3.2 Data

SECOND: 4 662 pairs of 512 × 512 aerial images, six land-cover classes (non-vegetated ground, tree,
low vegetation, water, building, playground) plus "no change". The official training set was split
into train 2 524 / validation 444 by filename groups (frozen manifest); the official test set
(1 694 tiles) was used once.

### 3.3 Metrics

- **SeK** (primary): Cohen's kappa on the semantic confusion matrix with the dominant
  "no change → no change" cell removed, multiplied by `exp(IoU_change − 1)` — it punishes both wrong
  classes and wrong localisation. Typical published values on SECOND are ≈ 0.20–0.25.
- mIoU and Fscd (secondary); forward FLOPs per tile (incl. the detector); wall-clock latency
  (batch 1, fp32, A100, paired per tile).

### 3.4 Fairness rules

- **Equal training FLOPs** for every semantic model (HPO included), turned into a fixed step count.
  Equal FLOPs means the light model makes ~8× more optimizer steps than SCanNet — that is intended.
- Selection (hyperparameters, checkpoints, thresholds, ROI policies) only on validation.
- 3 seeds; paired scene bootstrap nested with seeds; 9 pre-declared primary comparisons with Holm
  correction; everything else labelled exploratory.
- **Decision rule (frozen before the test):** two-stage is *preferable* only if it is non-inferior
  in SeK (95 % CI lower bound > −δ, δ = 0.007) **and** cheaper in FLOPs; better but dearer = trade-off;
  non-inferior but dearer = single-stage preferable.

## 4. Hypotheses and outcomes

| | Hypothesis | Outcome |
|---|---|---|
| H1 | At equal training FLOPs the three single-stage models differ | **yes:** SCanNet 0.238 ≳ MambaSCD 0.233 ≫ U-Net 0.185 SeK |
| H2 | Two-stage is non-inferior in SeK (central) | **yes** for all three — but more expensive, so never *preferable* |
| H3 | LEVIR-CD pre-training of the detector helps (case 2 > case 3) | **not shown** (≤ +0.001 SeK, Holm p ≥ 0.19) |
| H4 | The OR-ensemble detector helps (case 4 > case 2) | **not shown** (Holm p ≥ 0.23) |
| H5 | The two-stage effect depends on the share of changed pixels | **yes:** U-Net gains only on sparse tiles; strong models lose least on dense tiles |
| H6 | Training the semantic model on ROI crops does not help (ablation) | **not run yet** (§10) |

## 5. Results

### 5.1 Main table (test, primary ROI policy)

| Model | Case | SeK | ΔSeK vs single (CI95) | FLOPs × | Box / window area | Verdict |
|---|---|---|---|---|---|---|
| U-Net | 1 | 0.1853 | — | 1.00 | — | — |
| U-Net | 2 | 0.1871 | +0.0018 (+0.0009…+0.0027) | 2.14 | 0.53 / 1.05 | trade-off |
| U-Net | 3 | 0.1870 | +0.0017 (+0.0006…+0.0027) | 2.13 | 0.53 / 1.04 | trade-off |
| U-Net | 4 | 0.1878 | +0.0025 (+0.0015…+0.0035) | 2.06 | 0.49 / 0.90 | trade-off |
| SCanNet | 1 | 0.2381 | — | 1.00 | — | — |
| SCanNet | 2 | 0.2341 | −0.0040 (−0.0053…−0.0027) | 1.28 | 0.53 / 1.05 | single |
| SCanNet | 3 | 0.2331 | −0.0049 (−0.0065…−0.0034) | 1.27 | 0.53 / 1.04 | single |
| SCanNet | 4 | 0.2337 | −0.0044 (−0.0060…−0.0029) | 1.15 | 0.49 / 0.90 | single |
| MambaSCD | 1 | 0.2329 | — | 1.00 | — | — |
| MambaSCD | 2 | 0.2292 | −0.0038 (−0.0051…−0.0025) | 1.62 | 0.53 / 1.05 | single |
| MambaSCD | 3 | 0.2284 | −0.0046 (−0.0061…−0.0030) | 1.61 | 0.53 / 1.04 | single |
| MambaSCD | 4 | 0.2287 | −0.0043 (−0.0057…−0.0028) | 1.51 | 0.49 / 0.90 | single |

Latency (case 2 vs single, median of per-tile ratios): U-Net ×2.45, SCanNet ×1.92, MambaSCD ×1.99.

### 5.2 Accuracy–compute curve (case 2, all four ROI policies)

| Model | balanced | frontier0.7 | frontier0.5 (primary) | frontier0.3 |
|---|---|---|---|---|
| U-Net | +0.0001 / ×2.12 | +0.0011 / ×2.09 | +0.0018 / ×2.14 | −0.0100 / ×1.76 |
| SCanNet | −0.0001 / ×1.27 | −0.0006 / ×1.24 | −0.0040 / ×1.28 | −0.0307 / ×0.91 |
| MambaSCD | −0.0003 / ×1.61 | −0.0003 / ×1.57 | −0.0038 / ×1.62 | −0.0295 / ×1.24 |

Only one configuration is cheaper than the single stage (SCanNet, frontier0.3), and it loses
0.031 SeK.

### 5.3 Density strata (H5; ΔSeK case 2 − single)

| Share of changed pixels ρ | U-Net | SCanNet | MambaSCD |
|---|---|---|---|
| ρ < 0.115 (574 tiles) | **+0.0030** | −0.0062 | −0.0073 |
| 0.115 ≤ ρ < 0.228 (539) | +0.0007 | −0.0071 | −0.0069 |
| ρ ≥ 0.228 (581) | −0.0005 | −0.0025 | −0.0034 |

On dense tiles the boxes cover almost the whole tile and both systems converge. On sparse tiles the
detector matters most: it seems to remove U-Net's false alarms (gain) but, for the stronger models,
its misses outweigh that (loss). `This mechanism is an interpretation, not measured separately.`

### 5.4 Headroom: oracle ROIs

With boxes taken from the ground truth, every model would gain +0.018 to +0.021 SeK at ≈ 0.98×
FLOPs (no detector charged). The idea has headroom; the real detector (pixel recall 0.86–0.89)
does not capture it.

## 6. When does a two-stage system pay off?

Exploratory checks after the test (validation data only, compute only, no accuracy).

**Cost model.** For a scene of area A, with detector cost `C_det`, semantic-model cost `C_scd` and
processed window area W:

```
FLOPs(two-stage) / FLOPs(single) = C_det / C_scd + W / A
```

**Image size cancels out.** Measured forward FLOPs grow exactly with the pixel count for all four
networks (512 → 800 → 1024 → 2048 px), so `C_det / C_scd` is constant: U-Net 1.09, MambaSCD 0.57,
SCanNet 0.24. U-Net can therefore *never* be cheaper in a two-stage system with this detector.

**Window share on real scenes** (ROI settings of the primary policy):

| Scenes | Changed share (mean) | Boxes from | U-Net | SCanNet | MambaSCD |
|---|---|---|---|---|---|
| SECOND val, 512² | 19 % | ground truth | ×2.09 | ×1.24 | ×1.58 |
| SECOND val, 512² | 19 % | case-2 detector | ×2.13 | ×1.27 | ×1.61 |
| LEVIR-CD val, 1024² | 4 % | ground truth | ×1.51 | **×0.67** | ×1.00 |
| LEVIR-CD val, 1024² | 4 % | LEVIR-trained BIT | ×1.52 | **×0.67** | ×1.00 |
| LEVIR-CD val, 1024² | 4 % | SECOND-trained BIT | ×2.01 | ×1.16 | ×1.49 |
| same LEVIR scenes cut to 512² | 4 % | ground truth | ×1.46 | **×0.61** | ×0.95 |

Binned by changed share, the SECOND and LEVIR-CD curves agree. Conclusions:

- Two-stage becomes cheaper only when **changes are rare**: below ≈ 5–10 % of the scene for SCanNet,
  below ≈ 3 % for MambaSCD, never for U-Net.
- The detector must be **in-domain**; a SECOND detector on LEVIR-CD imagery fires on ≈ 70 % of the
  area.
- Larger scenes do not help by themselves (cutting into tiles was even slightly cheaper). Real
  large scenes usually *have* a lower changed share than curated benchmark tiles — that is where the
  savings would come from.
- Open: whether accuracy holds at such low change shares. On SECOND's sparse stratum the strong
  models lost 0.006–0.007 SeK.

## 7. Key decisions

| Decision | Why | Alternative rejected |
|---|---|---|
| Measure training compute in FLOPs and stop by step count | GPU-hours on a shared server depend on other people's jobs | project-wide GPU-hour cap (used in v1) |
| Train all three semantic models ourselves | published checkpoints had seen our validation split; MambaSCD's was selected on the test split | compare against published weights |
| Reimplement SCanNet from the paper | its code has no licence | copy the authors' code |
| Use the *same* semantic checkpoint inside the two-stage systems | isolates the effect of detector + cropping | train a separate crop model (kept as ablation H6) |
| Choose the detector by a frozen screening rule on validation | avoid picking the detector that happens to look best later | choose by test or by intuition |
| Context windows of fixed sizes (128/256/384/512) | windows of equal size can be batched on the GPU | exact box crops (one forward pass per box, used in v1) |
| Non-inferiority margin δ and density strata fixed by a mechanical rule before the test | no room to tune the conclusion | choose δ after seeing results |
| Read the test split exactly once, through a hashed freeze manifest | the test result must not influence any choice | iterate on test |
| Publish only precomputed masks in the demo, no weights | dataset and model licences | interactive inference online |

## 8. Mistakes and lessons

1. **The first iteration studied the wrong thing.** The intended design (four system types, three
   published architectures) was in my head but not on paper; v1 compared one self-built model with
   one two-stage system. Four days and ≈ 80 GPU-hours were archived.
   *Lesson:* write the design — what is compared and how it is judged — before writing code.
2. **A GPU-hour budget on a shared server is not a fair budget.** Throughput changed with other
   people's load. Switched to training FLOPs and fixed step counts.
3. **The test split was read in v1 for preliminary systems.** The v2 systems are new and frozen
   separately, but aggregate v1 numbers were known — documented as a threat to validity.
4. **The first ROI-settings grid could not reach small area budgets** (minimum 0.47 of the tile), so
   one robustness metric measured the grid edge, not noise. Found during noise calibration, *before*
   the rule was frozen; the grid was extended from 180 to 324 settings.
5. **Learning-rate probes ended at the grid edge** for all three detectors (best = largest value) and
   the grid was not extended. Possibly none of the detectors is at its optimal learning rate.
6. **Equal FLOPs ≠ equal opportunity.** At the screening budget the Heavy detector got only ≈ 6
   epochs and was undertrained; a 5× longer run (one seed) reached F1 0.694 vs Medium 0.688.
7. **The cost could have been predicted on paper.** The formula in §6 with the measured detector
   cost and window area explains the main result. A back-of-the-envelope cost model before the
   experiments would have shown which conditions to test.
8. **Long runs on a shared server get interrupted.** Five final runs were resumed exactly from
   their last checkpoint after restarts; one detector run was started twice by mistake (duplicate
   killed, integrity checked). Resumable training paid off.
9. **Latency on a shared GPU:** foreign load appeared during the measurement; ratios are paired per
   tile and robust, absolute milliseconds may be inflated.
10. **Notebooks are not a lab journal.** In Research 1 I edited the training notebooks again and
    again while the training functions changed, so their code no longer matches their saved
    outputs. What saved the history were the per-trial records in `artifacts/`. *Lesson:* keep code
    in a versioned library, let notebooks only call it, and write every run's parameters and
    results to disk.
11. **Demo details matter:** an inverted colour legend in dark mode, maps blocking touch scrolling,
    and a "processed area" column showing the box instead of the window area (documented in the
    README, not yet changed on the page) — all found by testing the page as a user.

What I learned from all this is summarised in the [README](README.md#what-i-learned).

## 9. Limitations

- One dataset (SECOND), one implementation per architecture, three seeds; no geographic metadata,
  so spatial dependence between tiles is not modelled.
- ROI policies, ensemble partner and δ were chosen on the same validation tiles used to select
  checkpoints (optimistic for the two-stage systems).
- The semantic models were not trained on crops (deliberately unfavourable for two-stage; ablation
  H6 pending).
- Our absolute SeK is below published numbers: smaller training budget and a shared recipe.
- §6 measures compute only; accuracy on large, sparse scenes is untested.

## 10. Future work

- **H6 — crop-trained semantic model:** does training on ROI windows recover the accuracy lost by
  the strong models?
- **Accuracy on large real scenes** with low change share (e.g. HRSCD, 10 000² px, semantic labels).
- **A cheaper detector:** U-Net would need a detector well below its own cost.
- **Engineering the latency:** box generation, host round trips and small batches cost more time
  than the saved area.

## 11. Reproducibility

- Every run records git revision, PyTorch/CUDA versions, GPU model, steps and FLOPs.
- Frozen protocol and code are hashed in a freeze manifest; the single test read is recorded.
- MambaSCD's `selective_scan` backward pass is not bitwise deterministic; its runs rely on seeds.
- Compute: 1.44 × 10¹⁸ training FLOPs per semantic model (HPO 4.8 × 10¹⁷ + finals 9.6 × 10¹⁷),
  4 × A100-80GB shared server.

