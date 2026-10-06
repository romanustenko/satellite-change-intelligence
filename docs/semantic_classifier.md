# Tiny Siamese + SECOND

Open `notebooks/tiny_siamese_second_staged_resume.ipynb` in a project Python 3.11/3.12 kernel.
**Restart Kernel** after package updates. Do not edit the shared package while other HPO jobs
use it. The notebook defaults to CPU implementation smoke; dataset operations, GPU stages,
extra seeds and final test require their explicit configuration flags.

Training has one user entry point:

```python
from satchange.automated_hpo import run_automated_parameter_search
from satchange.semantic.training import hpo_parameters

base = hpo_parameters(config, prepared_json)
run_automated_parameter_search(
    task="semantic_roi_classification",
    stages=[("baseline", {"learning_rate": [base["learning_rate"]]})],
    base_parameters=base,
    model_name="tiny_siamese_change_classifier",
    dataset_root=prepared_json,
    stage_epochs=[40, 60, 80],
    seeds=[42, 123, 777],
    additional_seeds=True,
    output_root=hpo_root,
    search_name="second_oracle_hpo_v3_40_60_80",
    gpu_ids=[0, 1, 2, 3],
    selection_gap=0.04,
    stage="hpo", hpo_stage="baseline",
)
```

Omit segmentation thresholds/BCE/Dice/positive_weight/image_size. There is no semantic
training CLI, independent `fit_classifier`, second seed summarizer or GPU pool. The common
controller, existing independent queue and `training.fit_binary_change_model` lifecycle handle
both tasks. The legacy function name remains for binary API compatibility; its `task` keyword
selects the semantic epoch adapter. Tiny is not in the binary detector `MODEL_REGISTRY`.

## Data and native geometry

[SECOND's authors](https://captain-whu.github.io/SCD/) describe 512×512 images and six classes.
The saved class order is non-vegetated ground surface, tree, low vegetation, water, buildings,
playgrounds. White denotes non-change; same-class changes are valid. The exact color mapping
is documented in [the SCD-SAM authors' loader](https://raw.githubusercontent.com/yzygit1230/SCD-SAM/main/datasets/RS_ST.py).
Our canonical IDs are 1..6 in the above order; ID 0 is ignored. Original palette indices
are decoded through their RGB colors. Scalar indexed labels require an explicit verified
`indexed_ids` mapping; undocumented IDs/colors fail, with no grayscale/nearest-color fallback.

`semantic.second.bootstrap_second` validates an installed `train/val/test` layout beneath
`data/SECOND`, or explicitly accepts a local ZIP/official download. Its gdown file downloader
reuses the installed dependency; the LEVIR folder downloader is incompatible. ZIP paths and
symlinks are checked before extraction. Partial installations do not count as complete.
Quota/authentication errors propagate with a local-archive alternative. No access workaround
is implemented. Stored SHA256 values are computed locally, not published author checksums.

`prepare_second` discovers matched im1/im2/label1/label2 stems, converts annotations automatically
and calls the existing `prepare_dataset`. It checks equal positive annotation support on both
dates. Support is **not** `before_label != after_label`: the latter would lose same-class change.
Native RGB remains uint8/255; no resizing or padding enters classifier inputs.

Supplied labeled splits are retained. Missing labeled val/test requires explicit
`local_holdout=True`: a fixed `split_seed` assigns source groups from train, independently of
training seeds. Without supplied `source_groups`, groups are filenames, and geographic
independence is explicitly unverified. These local splits are not official test results.
Optional sensor/GSD metadata is recorded as supplied; no physical-scale alignment is inferred.

Cache directories are content/protocol versions. A completion marker and matching prepared
hash are required, followed by verification of source and ROI-mask hashes. Changed source,
ROI policy or split settings create a new version. A failed preparation stays under a private
`.incomplete-*` directory without a completion marker. Existing completed versions are preserved.

Cross-split group, normalized RGB-path and RGB-file-content pair checks include temporal reversal.
Identical empty annotation masks are not leakage evidence. Scene RGB hashes are reused during
preparation. No semantic labels are fabricated from LEVIR's binary masks.

## Tensor flow and metrics

Each selected connected component gets deterministic original-coordinate component/crop boxes
with exclusive upper bounds. Real context is 16 pixels, minimum crop extent 128, connectivity 8,
minimum component area 1. Borders shift context inward; large components remain complete, and
small images record their limited available extent. Component IDs remain stable after filtering.

Each list sample supplies before/after float32 `[3,H,W]`, selected-component probability weights
`[1,H,W]`, and two class indices. A logical minibatch can contain different native sizes. Each
temporal pair is concatenated to `[2,3,H,W]`, passed through one shared 32→64→128 CNN with two
pooling stages, then component-weighted feature pooling in FP32 yields `[2,128]`. Only the mask
is reduced to the actual feature grid. Neighboring components in the context crop cannot
directly contribute to pooling; RGB is not zeroed. Shared 128→64→C MLP returns raw logits.

Logits from all samples are concatenated to `[B,C]` before the two date-wise CE terms. Positive
class weights follow the saved class order; the default is explicitly unweighted CE. If users
derive weights externally, they must use training ROI labels only and record their rule and
absent-class policy. Weighted CE normalizes over the full logical batch separately per date;
logging accumulates the exact numerator/denominator, including the final partial batch.
Singleton weighted CE cancels its sample's weight. BatchNorm statistics cover each temporal
pair rather than the full logical minibatch; this is a deliberate small-model tradeoff.

Metrics accumulate two global C×C confusion matrices (true rows, predicted columns), date-wise
accuracy/macro F1, per-class precision/recall/F1/support and joint accuracy. Undefined ratios are
zero; macro F1 includes every declared class, including absent classes. Joint accuracy requires
both dates correct. Selection maximizes `(before_macro_f1 + after_macro_f1)/2` on val. Exact ties
retain the earlier epoch. Scheduler and early stopping use this same `selection_score`, and
secondary metrics come from that same selected checkpoint. No semantic IoU is invented.

## HPO, resume and numerical settings

The merged implementation uses GitHub's HPO v3 controller (`hpo_core.py`) and its
`SatelliteBackend`, with task-specific data/loss/metric adapters. No old controller replaces v3.
The notebook configures batch×LR, LR refinement, weight_decay×dropout, and scheduler stages.
Pass the complete stage list each time; use `stage="hpo", hpo_stage="01_batch_lr"` for a block.
`stage="seeds"` requires all HPO blocks complete, compares complete families on seeds
42/123/777, and selects mean validation `selection_score`. Optional `joint_tiebreaker=True`
adds mean joint accuracy only as a family tie-breaker. `additional_seeds=False` is an explicit
exploratory one-seed mode; its sample standard deviation is null.

`stage_epochs=[40,60,80]` specifies ABSOLUTE HPO, seeds, and final budgets. Every promotion
continues only its own seed checkpoint (optimizer/scheduler/loader/RNG/history/best weights).
Final freezes one configuration in `final_choice.json`, always fills all three seed runs,
disables early stopping, and requires all seeds to reach 80 epochs. Historical best weights
can be from an earlier epoch and receive final-budget completion metadata. Repeating a call
reuses completed trials and resumes interrupted `last.pt`, rather than adding epochs again.
Increasing a phase target is explicit via `stage_epochs` or an idempotent extension operation.
Policy promotion resets only early-stopping state, preserving the rest of the training state.
Inference-only best-weight exports can be evaluated but cannot resume fitting.

`dry_run=True` saves a plan without training. For a fresh search use `pending_policy="all"`.
`finish_started` retains only genuinely started secondary seed identities; missing seeds are
reported as incomplete families and cannot win. `request_hpo_pause(search_directory)` pauses
at an epoch boundary. OS locks now work on Windows and POSIX and release on process exit.

The existing v3 GPU queue schedules one independent worker per visible allocated GPU; no DDP
or second pool is added. Capacity/OOM retries and preservation of completed siblings are retained.
The code does not change CUDA_VISIBLE_DEVICES. Unsupported CUDA/BF16 fails without CPU fallback.
CPU `devices=["cpu"]` is available for small explicitly requested research runs and tests.

`pipeline_protocol.json` keeps the v3 model/data/search definition and backend source signature.
Tiny identity includes class order, dropout, weights, prepared SHA256, ROI policy, precision/TF32,
and optimizer settings. Source hashing includes semantic adapters and the controller. A source
change requires a new search name and explicit `reuse_from` directories; saved signatures and
old artifacts are never rewritten. The old local schema-4 controller is replaced by current v3;
its own complete Tiny checkpoints may be imported explicitly into a new search.

## Evaluation and detector composition

`semantic.inference.evaluate_classifier` reads the common checkpoint schema and reproduces saved
precision/TF32. An explicit precision override is recorded with actual device/precision. CPU
evaluation of a BF16 checkpoint needs a declared FP32 override; there is no silent fallback.
Final test is a separate `RUN_TEST_BENCHMARK` notebook flag after three completed final runs.
The common `run_final_benchmark(task="semantic_roi_classification", ...)` entry point dispatches
to the semantic evaluation/export adapter. It recomputes all validation metrics before any test
and refuses test if strict validation differs. Deployment seed is fixed before test (42 in the
notebook); test never selects a seed/configuration. Five JSON documents, byte-exact checkpoint
copies and a CPU deployment state_dict use the common transactional output publisher.
Reports contain per-date confusion/per-class metrics, macro F1, joint accuracy, ROI coverage,
identities, mean/sample std and saved training metadata. Speed uses one native validation ROI,
including its mask; it excludes detector, proposal extraction, I/O and device transfer.

GT-trained weights accept explicitly declared predicted inference probabilities. `predict_checkpoint`
separates training/inference mask source and records detector identity/proposal settings.
Alternate predicted-ROI evaluation requires `evaluation_config`, preserving training and evaluation
data identities, taxonomy, normalization, eligibility rule and split provenance. Evaluation pairs
and groups cannot overlap fitting data. ROI eligibility/exclusions and coverage are reported;
accuracy on eligible proposals is not end-to-end detector accuracy.

`predict_rois` consumes an aligned map directly without GT. `predict_detector_checkpoint` reuses
`run.reconstruct_checkpoint` for the frozen detector, explicitly converts declared logits or
probabilities, restores a known whole-image resize map, and classifies crops from original RGB.
This composition supports the project's RGB/255 normalization only. Unknown crop/tile/padding or
custom-normalization mappings require an explicitly aligned saved map. Confidence is uncalibrated
per-date softmax. Empty maps return no proposals.

The implementation was checked on synthetic CPU fixtures, including actual controller, queue
routing with mocked GPU/executor resources, workers, shared lifecycle and a fresh notebook kernel.
Real SECOND installation, download access, trained detector accuracy, CUDA/BF16 execution and
four-A100 integration have not been run. No real training/HPO or dataset download runs automatically.
