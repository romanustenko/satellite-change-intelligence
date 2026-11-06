# Research 2 — Research 1 Checkpoint Inventory (task R2-P01-T02)

**Status:** COMPLETED 2026-11-06 from git-tracked metadata at HEAD `b72cd6c` (VERIFIED HISTORICAL).
Checkpoint contents were not opened. File identities (SHA256 = Git LFS oid, size) are listed in
[`READINESS_AUDIT.md`](READINESS_AUDIT.md) §2; whether the server files match them is pending
(R2-P01-T01 server part). **No selection is made here** (D-010, D-012): this is the candidate list
and its provenance. Test values are shown for traceability only and must never be used to rank or
select Research 2 candidates.

Source files per variant: `artifacts/final_models/<variant>/{metadata,training_results,
validation_results,final_test_results,benchmark}.json`. Validation: 64 LEVIR-CD pairs; test: 128.
"Train GPU-h (recorded)" is the historical `gpu_hours` field (epoch compute, see
`PROJECT_CONTEXT.md` §16) — a lower bound of historical pretraining cost, not allocation time.

## Per-variant candidates

### light_full

- model: `fc_siam_diff`; image_size: None; model_parameters: `{"in_channels": 3, "out_channels": 1, "base_channels": 32}`
- deployment: seed 2024, threshold 0.8; code revisions recorded: `7f38560`
- validation threshold selection: original validation thresholds; no new threshold search; test threshold policy: original validation thresholds; no new threshold search; pairs val/test: 64/128

| Seed | Best epoch | Val threshold | Val F1 | Val IoU | Test F1 (report only) | Train GPU-h (recorded) | Train seconds |
|---|---|---|---|---|---|---|---|
| 123 | 109 | 0.70 | 0.8235 | 0.6999 | 0.8102 | 2.652 | 9547 |
| 777 | 43 | 0.45 | 0.7695 | 0.6254 | 0.7648 | 2.632 | 9476 |
| 2024 | 110 | 0.80 | 0.8509 | 0.7405 | 0.8411 | 2.607 | 9385 |

### light_resized

- model: `fc_siam_diff`; image_size: 256; model_parameters: `{"in_channels": 3, "out_channels": 1, "base_channels": 16}`
- deployment: seed 42, threshold 0.7; code revisions recorded: `7f38560`
- validation threshold selection: saved validation thresholds; no new threshold search; test threshold policy: saved validation thresholds; no new threshold search; pairs val/test: 64/128

| Seed | Best epoch | Val threshold | Val F1 | Val IoU | Test F1 (report only) | Train GPU-h (recorded) | Train seconds |
|---|---|---|---|---|---|---|---|
| 42 | 104 | 0.70 | 0.7549 | 0.6062 | 0.7567 | 4.620 | 16632 |
| 123 | 110 | 0.80 | 0.7267 | 0.5707 | 0.7219 | 4.635 | 16686 |
| 777 | 96 | 0.90 | 0.7401 | 0.5875 | 0.7336 | 4.649 | 16737 |

### medium_full

- model: `bit`; image_size: 1024; model_parameters: `{"in_channels": 3, "out_channels": 1, "embedding_dim": 64, "token_length": 4, "num_heads": 8, "encoder_head_dim": 64, "decoder_head_dim": 64, "encoder_depth": 1, "decoder_depth": 8, "dropout": 0.0, "pretrained_backbone": false}`
- deployment: seed 123, threshold 0.7; code revisions recorded: `02b72d7`
- validation threshold selection: saved validation threshold; no test tuning; test threshold policy: saved validation threshold; no test tuning; pairs val/test: 64/128
- HPO search names: `bit_medium_full_final_100ep_seeds42_123_777_v1`, `bit_medium_full_final_100ep_seed42_rerun_v1`

| Seed | Best epoch | Val threshold | Val F1 | Val IoU | Test F1 (report only) | Train GPU-h (recorded) | Train seconds |
|---|---|---|---|---|---|---|---|
| 42 | 81 | 0.50 | 0.8602 | 0.7547 | 0.8513 | 5.184 | 18661 |
| 123 | 100 | 0.70 | 0.8609 | 0.7558 | 0.8481 | 6.440 | 23184 |
| 777 | 82 | 0.45 | 0.8488 | 0.7373 | 0.8377 | 9.817 | 35341 |

### medium_resized

- model: `bit`; image_size: 256; model_parameters: `{"in_channels": 3, "out_channels": 1, "embedding_dim": 32, "token_length": 4, "num_heads": 8, "encoder_head_dim": 64, "decoder_head_dim": 64, "encoder_depth": 1, "decoder_depth": 8, "dropout": 0.0, "pretrained_backbone": false}`
- deployment: seed 123, threshold 0.7; code revisions recorded: `02b72d7`
- validation threshold selection: saved validation thresholds; no new threshold search; test threshold policy: saved validation thresholds; no new threshold search; pairs val/test: 64/128

| Seed | Best epoch | Val threshold | Val F1 | Val IoU | Test F1 (report only) | Train GPU-h (recorded) | Train seconds |
|---|---|---|---|---|---|---|---|
| 42 | 98 | 0.85 | 0.7093 | 0.5495 | 0.6846 | 0.906 | 3262 |
| 123 | 99 | 0.70 | 0.7189 | 0.5612 | 0.7065 | 0.777 | 2796 |
| 777 | 89 | 0.65 | 0.7048 | 0.5442 | 0.6616 | 1.053 | 3791 |

### heavy_full

- model: `siam_mit_b3_diff`; image_size: None; model_parameters: `{"attention_dropout": 0.0, "decoder_channels": 128, "drop_path_rate": 0.1, "dropout": 0.0, "encoder_depths": [3, 4, 18, 3], "in_channels": 3, "norm_groups": 8, "out_channels": 1, "pretrained_backbone": false, "refinement_channels": [64, 32], "use_checkpointing": false}`
- deployment: seed 42, threshold 0.8; code revisions recorded: `9ed0fc7`
- validation threshold selection: saved or explicitly supplied validation threshold; no test tuning; test threshold policy: saved or explicitly supplied validation threshold; no test tuning; pairs val/test: 64/128

| Seed | Best epoch | Val threshold | Val F1 | Val IoU | Test F1 (report only) | Train GPU-h (recorded) | Train seconds |
|---|---|---|---|---|---|---|---|
| 42 | 79 | 0.80 | 0.8702 | 0.7702 | 0.8640 | 7.730 | 27829 |
| 123 | 70 | 0.80 | 0.8728 | 0.7744 | 0.8665 | 7.376 | 26554 |
| 777 | 80 | 0.80 | 0.8715 | 0.7722 | 0.8693 | 5.046 | 18167 |

### heavy_resized

- model: `siam_mit_b3_diff`; image_size: 256; model_parameters: `{"attention_dropout": 0.0, "decoder_channels": 128, "drop_path_rate": 0.1, "dropout": 0.0, "encoder_depths": [3, 4, 18, 3], "in_channels": 3, "norm_groups": 8, "out_channels": 1, "pretrained_backbone": false, "refinement_channels": [64, 32], "use_checkpointing": false}`
- deployment: seed 42, threshold 0.85; code revisions recorded: `81a5e53`
- validation threshold selection: saved or explicitly supplied validation threshold; no test tuning; test threshold policy: saved or explicitly supplied validation threshold; no test tuning; pairs val/test: 64/128

| Seed | Best epoch | Val threshold | Val F1 | Val IoU | Test F1 (report only) | Train GPU-h (recorded) | Train seconds |
|---|---|---|---|---|---|---|---|
| 42 | 59 | 0.85 | 0.7757 | 0.6335 | 0.7463 | 2.061 | 7418 |

## Findings

1. **Resolved discrepancy (Light Full test F1).** `final_models_gpu_cost.csv` `test_F1` is the
   **mean over the three seeds** for every variant (checked against `final_test_results.json`
   `test_f1_mean`: identical for all six). The hand-off H4 value 0.841149 is the **deployment seed
   2024** result (0.8411 above). Both are correct; they measure different things.
2. **Deployment seed choice is not uniform.** For five variants the deployment seed has the best
   validation F1; for `heavy_full` seed 42 (val F1 0.8702) was deployed although seed 123 has
   0.8728 — `metadata.json` records `deployment_selection: "explicitly supplied before test
   evaluation"`. No test-based selection is indicated, but the rule differs from the others.
3. **Large seed variance in Light Full** (val F1 0.7695–0.8509; seed 777 best epoch 43) — relevant
   for D-025 (seeds/statistics) and for choosing a pilot initialization.
4. **Historical training cost** (recorded epoch GPU-hours, sum over 3 seeds): Light Full ≈7.9,
   Light Resized ≈13.9, Medium Full ≈21.4, Medium Resized ≈2.7, Heavy Full ≈20.2, Heavy Resized ≈2.1
   (one seed). HPO search cost before the final runs is not included. Light Resized (256 px) recorded more
   GPU-hours than Light Full (1024 px), consistent with the epoch timer including data-loading wall
   time (`num_workers: 0`, on-the-fly resizing) — the field is not pure GPU compute. These feed the
   "historical pretraining cost" category (D-003) as lower bounds only.
5. **Two incompatible latency sources.** Per-variant `benchmark.json` latencies (e.g. Light Full
   128.6 ms/pair, Heavy Full 149.6 ms/pair) differ strongly from `final_models_gpu_cost.csv`
   (16.0 ms, 92.9 ms); the JSON schema also differs between variants (some record precision/TF32,
   some do not), i.e. the benchmarks were produced by different code versions/protocols. Hardware
   recorded: NVIDIA A100-SXM4-80GB, CUDA 13.0, torch 2.12.1+cu130 where recorded. Neither source
   is an end-to-end two-stage measurement; Research 2 must re-measure under D-026.
6. **Code provenance:** Light `7f38560` missing (READINESS_AUDIT F1); Medium `02b72d7` and
   Heavy-resized `81a5e53` preserved by local tags (D-029); Heavy Full `9ed0fc7` on `main`.
7. **Pilot-relevant (D-020/D-024):** FC-Siam-Diff candidates are Light Full
   (`base_channels=32`, native 1024 training) and Light Resized (`base_channels=16`, 256 input).
   SECOND is 512×512, so neither matches its input regime exactly; the choice is a pilot-design
   decision (R2-P08-T03), recorded here without selection.

## Pilot compatibility audit (task R2-P08-T01)

**Status:** COMPLETED locally 2026-11-06 except the strict key-by-key load of the real `.pt`
files (LFS objects not materialized here; pending server access or an LFS fetch).

- **Architecture identity (VERIFIED CURRENT vs VERIFIED HISTORICAL):** the current `FCSiamDiff`
  builds 46 state tensors under `encoder_blocks`, `upconvs`, `decoder_blocks`, `output_layer`;
  parameter counts 7,760,097 (`base_channels=32`) and 1,941,105 (`base_channels=16`) equal the
  recorded counts of `light_full` / `light_resized` (`final_models_gpu_cost.csv`). Strong evidence
  that the checkpoints load with `strict=True`; to be confirmed with `run.reconstruct_checkpoint`.
- **Input size:** fully convolutional; 512×512 (SECOND native) and 256×256 run; sizes not divisible
  by 8 fail loudly (500 → decoder/skip mismatch). SECOND needs no resizing for FC-Siam-Diff.
- **Binary target on SECOND (D-024 option a, recommended):** head `output_layer` = 1×1 conv,
  32→1 (or 16→1). All 46 tensors load; scratch and fine-tune differ **only** in initialization.
- **Semantic head (option b):** a different `out_channels` changes only `output_layer.weight`
  `[C,32,1,1]` and `output_layer.bias` `[C]`; the other 44 tensors (encoder, up-convolutions,
  decoder) load; the head is re-initialized identically in both conditions. Head layout depends on
  D-013.
- **Which LEVIR checkpoint (decision for R2-P08-T03, not made here):** `light_full`
  (`base_channels=32`, trained at native 1024 px) or `light_resized` (`base_channels=16`, trained on
  scenes downsampled to 256 px, i.e. a coarser apparent scale). The choice fixes the architecture of
  *both* pilot conditions (scratch must use the same `base_channels`). Relevant evidence: SECOND's
  ground sampling distance relative to LEVIR (to be documented in R2-P05-T01), seed variance of
  `light_full` (val F1 0.77–0.85).
- **Initialization path in HPO: missing (VERIFIED CURRENT).** `TRAIN_KEYS` (`hpo_backend.py:42–55`)
  has no weight-initialization key and unknown keys are rejected. `resume_from` is **not** a
  fine-tune: it requires identical parameters (incl. seed, model, dataset, thresholds), continues the
  epoch counter, optimizer/scheduler/early-stopping/RNG state and copies the LEVIR history into the
  run. **Minimal proposal (R2-P08-T02/T03, after R2-P06-T03):** one new identity key, e.g.
  `initial_weights = {"path", "sha256", "exclude": [...]}`, applied only at a fresh start
  (`source_epoch == 0`) with `load_state_dict(strict=True)` on the non-excluded keys and a hash check;
  recorded in the checkpoint config; scratch = key absent. No parallel trainer.
