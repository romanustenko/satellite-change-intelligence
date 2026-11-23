# Research 1 notebooks — guide

Research 1 (binary change detection on LEVIR-CD) was developed interactively in these notebooks.
The training and HPO notebooks were edited many times while the training functions in
`src/satchange/` changed: cells were removed or rewritten between runs, and later runs moved into
the library. **The saved outputs are historical evidence; the code cells do not always reproduce
them.** The reliable record of every run is in `artifacts/` (parameters, curves, metrics, GPU time,
checkpoint hashes).

## Training and hyperparameter search (Aug 24 – Oct 7, 2026)

| Notebook | Model / regime | What it did | Records |
|---|---|---|---|
| `01_fc_siam_diff.ipynb` | Light (FC-Siam-Diff), Resized + Full | dataset preparation, first training runs, a hand-written search grid with threshold selection on validation | `artifacts/training_runs/fc_siam_diff*` |
| `02_fc_siam_diff_full_hpo.ipynb` | Light, Full | staged HPO: learning-rate and batch-size screens, limited grid, loss, weight decay, model capacity, candidates a–h over seeds 2024/123/777, deterministic reproducibility checks | `artifacts/hyperparameter_search/fc_siam_diff_*`, `full_*` |
| `03_bit_full.ipynb` | Medium (BIT-S4-DD8), Full | staged HPO in six stages (lr × batch → architecture → BCE/Dice weights → weight decay → lr refinement → batch × lr), multi-seed confirmation, final 100-epoch training | `artifacts/hyperparameter_search/bit_full_*`, `bit_medium_*` |
| `03_bit_resized.ipynb` | Medium, Resized 256 | same procedure at 256 × 256 | `artifacts/hyperparameter_search/bit_medium_*` |
| `04_mit_full.ipynb`, `04_mit_resized.ipynb` | Heavy (Siamese MiT-B3) | staged search with the project's own HPO controller (`automated_hpo`; stage 1 = batch size × learning rate, …) and the final benchmark | `artifacts/hyperparameter_search/heavy/` |
| `heavy_full_staged_resume.ipynb`, `heavy_resized_staged_resume.ipynb` | Heavy, Full / Resized | resumable staged HPO (40 → 60 → 80 epochs) after interrupted runs | `artifacts/hyperparameter_search/heavy/` |
| `Heavy_Full_HPO_v3_metadata_fix.ipynb` | Heavy, Full | continuation after a checkpoint-import fix; final seeds | `artifacts/hyperparameter_search/heavy/all_models_metrics.csv` |

Final models (weights, thresholds selected on validation, test metrics, SHA-256):
`artifacts/final_models/{light,medium,heavy}_{full,resized}/`.

## Hypothesis studies (Oct 20 – Oct 30, 2026)

These notebooks perform no training; they read the final models and write to
`artifacts/hypothesis_tests/`.

| Notebook | Question | Records |
|---|---|---|
| `05_levir_hypotheses_preflight.ipynb` | preparation and checks for H1–H3 | — |
| `06_levir_hypotheses_H1_H5.ipynb` | H1 resolution vs object size, H2 boundary quality, H3 complementary errors (OR ensemble), H4 pixel F1 vs object recall, H5 quality vs cost | `H1_H5_v1/` |
| `07_levir_followup_experiments_H1_H5.ipynb` | follow-up experiments for H1–H5 | `H1_H5_followup_v2/` |
| `08_levir_roi_model_selection*.ipynb` | which detector gives the best ROIs; Medium diagnosis | `H08_*`, `08_medium_diagnostic/` |
| `09_levir_medium_topology_matched_cost.ipynb` | ROI geometry and comparison at matched cost | `H09_medium_topology_matched_cost/` |
| `10_levir_roi_generator_topology_ablations.ipynb` | ROI generator alternatives, mask morphology ablations | `H10_roi_generator_topology_ablations/` |
| `11_levir_grid_tile_size_all_models_fixed.ipynb` | fixed-grid tile size vs adaptive ROI for all six models | `H11_grid_tile_size_all_models/` |

Several notebooks default expensive flags (for example `RUN_TEST`) to `True`. Do not "Run All" them
as a check.
