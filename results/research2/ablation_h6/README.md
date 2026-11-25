# Ablation H6 — semantic models trained on crops (exploratory)

**Hypothesis H6.** Training the semantic (SCD) model on ROI crops instead of whole tiles does *not*
improve the SeK of the two-stage system. Prediction recorded before the runs: "crops ≈ tiles or
worse".

**Design.** Each of the three SCD models was trained again from ImageNet initialization only on
context windows (128/256/384/512 px) around the ground-truth changes of the SECOND train tiles,
with the budget of the main study (3.2 × 10¹⁷ training FLOPs) and the HPO-winning learning rate /
weight decay. 2 × 2 design: training data {whole tiles (main-study checkpoint, seed 42), crops}
× input {whole tile, detector ROIs}, plus ground-truth ROIs. Validation only (444 tiles), one seed
(42), checkpoint selected on validation; the test split was not read. Paired tile bootstrap
(2000 resamples), δ = 0.007 SeK, density strata fixed on validation.

**Result.** H6 holds: crop − tile ΔSeK of the two-stage system (case 2) is −0.0023 (U-Net),
−0.0044 (SCanNet), −0.0002 (MambaSCD), all 95 % CIs include 0. With crop training the two-stage
system is non-inferior to the single-stage one for none of the models. Tables: main
[README](../../../README.md#ablation-train-the-semantic-model-on-crops-h6-exploratory).

| File | Content |
|---|---|
| `analysis_val.json` | SeK of every train × input combination, all paired comparisons with 95 % CI and one-sided p, density strata |
| `training_runs.json` | training records of the three crop-trained runs: steps, FLOPs, epochs, validation curve, configuration, software versions |

Code: [`src/scripts/run_r2_v2_ablation_b.py`](../../../src/scripts/run_r2_v2_ablation_b.py)
(training), [`src/scripts/analyze_r2_v2_ablation_b.py`](../../../src/scripts/analyze_r2_v2_ablation_b.py)
(analysis); commands in [REPRODUCE.md](../../../REPRODUCE.md#6-ablation-h6-optional).

Note: `git_revision` in `training_runs.json` is a commit of the development branch, read when a run
finishes (`2c750f5` = `7fc10ab` on `main`). Training started at `37c6061` (= `5318473` on `main`);
the commits in between changed only documentation, the Research 2 notebook and two evaluation
scripts that training does not use.
