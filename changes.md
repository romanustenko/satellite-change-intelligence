# changes — Tiny + GitHub HPO v3

Объединено с GitHub main `9ed0fc78f448a872dae6a46e1f68aeeaeee8be82`.
Сохранены текущие HPO v3 controller/backend, promotion, pause, OOM queue и binary APIs.
Tiny использует общий controller и trainer: HPO → seeds → final → run_final_benchmark.
Бюджеты notebook [40,60,80] абсолютные; final seeds 42/123/777 без early stopping.
Отбор: среднее macro F1 before/after; benchmark сначала сверяет все validation metrics,
затем оценивает test, измеряет classifier latency на native ROI и экспортирует weights.
ROI RGB не resize/pad; две CE, variable-size list batches и data/split provenance сохранены.

## Проверки

- Полный pytest: **133 passed, 3 skipped** (136 collected), exit 0.
- Два пропуска требуют CUDA и явного allocation; один — отсутствующий legacy 03_heavy.ipynb.
- Tiny notebook выполнен в свежем kernel с безопасными default flags; output cells не сохранены.
- Regression: реальные synthetic CPU epochs через v3 backend/controller; promotion 1→2→3,
  три собственных seed states, strict resume/cache, final benchmark/export, validation gate.
- Ruff: затронутые HPO/Tiny/shared modules/tests + Tiny notebook — **pass**.
- Mypy: semantic package + hpo_core/hpo_backend/automated_hpo, follow-imports silent:
  **Success, 11 source files**.
- Полный Ruff: 143 прежних diagnostics; исходный GitHub main — 264.
- Полный mypy src: 43 прежних errors / 8 files; исходный main — 57 / 11.
- Сравнение diagnostics по file/code/message: **0 новых**.
- git diff --check: pass. Raw datasets, checkpoints и artifacts не входят в commit.
- Реальный SECOND/HPO/GPU/BF16/download/детектор не запускались; они не проверены.

CPU launcher fixture явно задаёт fp32, один batch, num_workers=0 и validation threshold.
Notebook syntax проверяется для всех существующих notebooks; отсутствующий legacy smoke пропускается.
Полный код добавлений/изменений ниже в unified diff: заголовок каждого блока содержит путь,
@@ содержит точные строки; + добавлено, - удалено. Notebook приведён как исполняемые ячейки.

## Файлы

- `configs/tiny_siamese_change_classifier.yaml`
- `docs/semantic_classifier.md`
- `notebooks/tiny_siamese_second_staged_resume.ipynb`
- `src/satchange/automated_hpo.py`
- `src/satchange/execution.py`
- `src/satchange/final_benchmark.py`
- `src/satchange/hpo_backend.py`
- `src/satchange/hpo_core.py`
- `src/satchange/models/__init__.py`
- `src/satchange/models/factory.py`
- `src/satchange/models/tiny_siamese_change_classifier.py`
- `src/satchange/optimization.py`
- `src/satchange/run.py`
- `src/satchange/semantic/__init__.py`
- `src/satchange/semantic/__main__.py`
- `src/satchange/semantic/benchmark.py`
- `src/satchange/semantic/data.py`
- `src/satchange/semantic/inference.py`
- `src/satchange/semantic/roi.py`
- `src/satchange/semantic/second.py`
- `src/satchange/semantic/training.py`
- `src/satchange/training.py`
- `tests/test_execution.py`
- `tests/test_notebooks.py`
- `tests/test_semantic.py`
- `tests/test_semantic_hpo.py`

## Код

```diff
diff --git a/configs/tiny_siamese_change_classifier.yaml b/configs/tiny_siamese_change_classifier.yaml
new file mode 100644
index 0000000..9a2bc9c
--- /dev/null
+++ b/configs/tiny_siamese_change_classifier.yaml
@@ -0,0 +1,53 @@
+# SECOND canonical IDs after exact conversion; indexed input mapping must match your copy.
+classes:
+- non-vegetated ground surface
+- tree
+- low vegetation
+- water
+- buildings
+- playgrounds
+class_ids:
+  1: non-vegetated ground surface
+  2: tree
+  3: low vegetation
+  4: water
+  5: buildings
+  6: playgrounds
+ignore_ids:
+- 0
+mask_source: ground_truth
+detector: null
+roi:
+  threshold: 0.5
+  connectivity: 8
+  context_pixels: 16
+  min_crop_size: 128
+  min_component_area: 1
+label_policy:
+  min_transition_purity: 0.8
+  min_valid_fraction: 0.8
+dropout: 0.1
+dataset:
+  name: SECOND
+  directory: data/SECOND
+  source: https://captain-whu.github.io/SCD/
+  palette_source: https://raw.githubusercontent.com/yzygit1230/SCD-SAM/main/datasets/RS_ST.py
+  split_seed: 2099
+  indexed_id_names:
+    0: non-change
+    1: water
+    2: non-vegetated ground surface
+    3: low vegetation
+    4: tree
+    5: buildings
+    6: playgrounds
+training:
+  learning_rate: 0.001
+  weight_decay: 0.0001
+  batch_size: 4
+  num_workers: 0
+  scheduler: none
+  precision: fp32
+  tf32: false
+  seed: 42
+  class_weights: null
diff --git a/docs/semantic_classifier.md b/docs/semantic_classifier.md
new file mode 100644
index 0000000..a0a235f
--- /dev/null
+++ b/docs/semantic_classifier.md
@@ -0,0 +1,174 @@
+# Tiny Siamese + SECOND
+
+Open `notebooks/tiny_siamese_second_staged_resume.ipynb` in a project Python 3.11/3.12 kernel.
+**Restart Kernel** after package updates. Do not edit the shared package while other HPO jobs
+use it. The notebook defaults to CPU implementation smoke; dataset operations, GPU stages,
+extra seeds and final test require their explicit configuration flags.
+
+Training has one user entry point:
+
+```python
+from satchange.automated_hpo import run_automated_parameter_search
+from satchange.semantic.training import hpo_parameters
+
+base = hpo_parameters(config, prepared_json)
+run_automated_parameter_search(
+    task="semantic_roi_classification",
+    stages=[("baseline", {"learning_rate": [base["learning_rate"]]})],
+    base_parameters=base,
+    model_name="tiny_siamese_change_classifier",
+    dataset_root=prepared_json,
+    stage_epochs=[40, 60, 80],
+    seeds=[42, 123, 777],
+    additional_seeds=True,
+    output_root=hpo_root,
+    search_name="second_oracle_hpo_v3_40_60_80",
+    gpu_ids=[0, 1, 2, 3],
+    selection_gap=0.04,
+    stage="hpo", hpo_stage="baseline",
+)
+```
+
+Omit segmentation thresholds/BCE/Dice/positive_weight/image_size. There is no semantic
+training CLI, independent `fit_classifier`, second seed summarizer or GPU pool. The common
+controller, existing independent queue and `training.fit_binary_change_model` lifecycle handle
+both tasks. The legacy function name remains for binary API compatibility; its `task` keyword
+selects the semantic epoch adapter. Tiny is not in the binary detector `MODEL_REGISTRY`.
+
+## Data and native geometry
+
+[SECOND's authors](https://captain-whu.github.io/SCD/) describe 512×512 images and six classes.
+The saved class order is non-vegetated ground surface, tree, low vegetation, water, buildings,
+playgrounds. White denotes non-change; same-class changes are valid. The exact color mapping
+is documented in [the SCD-SAM authors' loader](https://raw.githubusercontent.com/yzygit1230/SCD-SAM/main/datasets/RS_ST.py).
+Our canonical IDs are 1..6 in the above order; ID 0 is ignored. Original palette indices
+are decoded through their RGB colors. Scalar indexed labels require an explicit verified
+`indexed_ids` mapping; undocumented IDs/colors fail, with no grayscale/nearest-color fallback.
+
+`semantic.second.bootstrap_second` validates an installed `train/val/test` layout beneath
+`data/SECOND`, or explicitly accepts a local ZIP/official download. Its gdown file downloader
+reuses the installed dependency; the LEVIR folder downloader is incompatible. ZIP paths and
+symlinks are checked before extraction. Partial installations do not count as complete.
+Quota/authentication errors propagate with a local-archive alternative. No access workaround
+is implemented. Stored SHA256 values are computed locally, not published author checksums.
+
+`prepare_second` discovers matched im1/im2/label1/label2 stems, converts annotations automatically
+and calls the existing `prepare_dataset`. It checks equal positive annotation support on both
+dates. Support is **not** `before_label != after_label`: the latter would lose same-class change.
+Native RGB remains uint8/255; no resizing or padding enters classifier inputs.
+
+Supplied labeled splits are retained. Missing labeled val/test requires explicit
+`local_holdout=True`: a fixed `split_seed` assigns source groups from train, independently of
+training seeds. Without supplied `source_groups`, groups are filenames, and geographic
+independence is explicitly unverified. These local splits are not official test results.
+Optional sensor/GSD metadata is recorded as supplied; no physical-scale alignment is inferred.
+
+Cache directories are content/protocol versions. A completion marker and matching prepared
+hash are required, followed by verification of source and ROI-mask hashes. Changed source,
+ROI policy or split settings create a new version. A failed preparation stays under a private
+`.incomplete-*` directory without a completion marker. Existing completed versions are preserved.
+
+Cross-split group, normalized RGB-path and RGB-file-content pair checks include temporal reversal.
+Identical empty annotation masks are not leakage evidence. Scene RGB hashes are reused during
+preparation. No semantic labels are fabricated from LEVIR's binary masks.
+
+## Tensor flow and metrics
+
+Each selected connected component gets deterministic original-coordinate component/crop boxes
+with exclusive upper bounds. Real context is 16 pixels, minimum crop extent 128, connectivity 8,
+minimum component area 1. Borders shift context inward; large components remain complete, and
+small images record their limited available extent. Component IDs remain stable after filtering.
+
+Each list sample supplies before/after float32 `[3,H,W]`, selected-component probability weights
+`[1,H,W]`, and two class indices. A logical minibatch can contain different native sizes. Each
+temporal pair is concatenated to `[2,3,H,W]`, passed through one shared 32→64→128 CNN with two
+pooling stages, then component-weighted feature pooling in FP32 yields `[2,128]`. Only the mask
+is reduced to the actual feature grid. Neighboring components in the context crop cannot
+directly contribute to pooling; RGB is not zeroed. Shared 128→64→C MLP returns raw logits.
+
+Logits from all samples are concatenated to `[B,C]` before the two date-wise CE terms. Positive
+class weights follow the saved class order; the default is explicitly unweighted CE. If users
+derive weights externally, they must use training ROI labels only and record their rule and
+absent-class policy. Weighted CE normalizes over the full logical batch separately per date;
+logging accumulates the exact numerator/denominator, including the final partial batch.
+Singleton weighted CE cancels its sample's weight. BatchNorm statistics cover each temporal
+pair rather than the full logical minibatch; this is a deliberate small-model tradeoff.
+
+Metrics accumulate two global C×C confusion matrices (true rows, predicted columns), date-wise
+accuracy/macro F1, per-class precision/recall/F1/support and joint accuracy. Undefined ratios are
+zero; macro F1 includes every declared class, including absent classes. Joint accuracy requires
+both dates correct. Selection maximizes `(before_macro_f1 + after_macro_f1)/2` on val. Exact ties
+retain the earlier epoch. Scheduler and early stopping use this same `selection_score`, and
+secondary metrics come from that same selected checkpoint. No semantic IoU is invented.
+
+## HPO, resume and numerical settings
+
+The merged implementation uses GitHub's HPO v3 controller (`hpo_core.py`) and its
+`SatelliteBackend`, with task-specific data/loss/metric adapters. No old controller replaces v3.
+The notebook configures batch×LR, LR refinement, weight_decay×dropout, and scheduler stages.
+Pass the complete stage list each time; use `stage="hpo", hpo_stage="01_batch_lr"` for a block.
+`stage="seeds"` requires all HPO blocks complete, compares complete families on seeds
+42/123/777, and selects mean validation `selection_score`. Optional `joint_tiebreaker=True`
+adds mean joint accuracy only as a family tie-breaker. `additional_seeds=False` is an explicit
+exploratory one-seed mode; its sample standard deviation is null.
+
+`stage_epochs=[40,60,80]` specifies ABSOLUTE HPO, seeds, and final budgets. Every promotion
+continues only its own seed checkpoint (optimizer/scheduler/loader/RNG/history/best weights).
+Final freezes one configuration in `final_choice.json`, always fills all three seed runs,
+disables early stopping, and requires all seeds to reach 80 epochs. Historical best weights
+can be from an earlier epoch and receive final-budget completion metadata. Repeating a call
+reuses completed trials and resumes interrupted `last.pt`, rather than adding epochs again.
+Increasing a phase target is explicit via `stage_epochs` or an idempotent extension operation.
+Policy promotion resets only early-stopping state, preserving the rest of the training state.
+Inference-only best-weight exports can be evaluated but cannot resume fitting.
+
+`dry_run=True` saves a plan without training. For a fresh search use `pending_policy="all"`.
+`finish_started` retains only genuinely started secondary seed identities; missing seeds are
+reported as incomplete families and cannot win. `request_hpo_pause(search_directory)` pauses
+at an epoch boundary. OS locks now work on Windows and POSIX and release on process exit.
+
+The existing v3 GPU queue schedules one independent worker per visible allocated GPU; no DDP
+or second pool is added. Capacity/OOM retries and preservation of completed siblings are retained.
+The code does not change CUDA_VISIBLE_DEVICES. Unsupported CUDA/BF16 fails without CPU fallback.
+CPU `devices=["cpu"]` is available for small explicitly requested research runs and tests.
+
+`pipeline_protocol.json` keeps the v3 model/data/search definition and backend source signature.
+Tiny identity includes class order, dropout, weights, prepared SHA256, ROI policy, precision/TF32,
+and optimizer settings. Source hashing includes semantic adapters and the controller. A source
+change requires a new search name and explicit `reuse_from` directories; saved signatures and
+old artifacts are never rewritten. The old local schema-4 controller is replaced by current v3;
+its own complete Tiny checkpoints may be imported explicitly into a new search.
+
+## Evaluation and detector composition
+
+`semantic.inference.evaluate_classifier` reads the common checkpoint schema and reproduces saved
+precision/TF32. An explicit precision override is recorded with actual device/precision. CPU
+evaluation of a BF16 checkpoint needs a declared FP32 override; there is no silent fallback.
+Final test is a separate `RUN_TEST_BENCHMARK` notebook flag after three completed final runs.
+The common `run_final_benchmark(task="semantic_roi_classification", ...)` entry point dispatches
+to the semantic evaluation/export adapter. It recomputes all validation metrics before any test
+and refuses test if strict validation differs. Deployment seed is fixed before test (42 in the
+notebook); test never selects a seed/configuration. Five JSON documents, byte-exact checkpoint
+copies and a CPU deployment state_dict use the common transactional output publisher.
+Reports contain per-date confusion/per-class metrics, macro F1, joint accuracy, ROI coverage,
+identities, mean/sample std and saved training metadata. Speed uses one native validation ROI,
+including its mask; it excludes detector, proposal extraction, I/O and device transfer.
+
+GT-trained weights accept explicitly declared predicted inference probabilities. `predict_checkpoint`
+separates training/inference mask source and records detector identity/proposal settings.
+Alternate predicted-ROI evaluation requires `evaluation_config`, preserving training and evaluation
+data identities, taxonomy, normalization, eligibility rule and split provenance. Evaluation pairs
+and groups cannot overlap fitting data. ROI eligibility/exclusions and coverage are reported;
+accuracy on eligible proposals is not end-to-end detector accuracy.
+
+`predict_rois` consumes an aligned map directly without GT. `predict_detector_checkpoint` reuses
+`run.reconstruct_checkpoint` for the frozen detector, explicitly converts declared logits or
+probabilities, restores a known whole-image resize map, and classifies crops from original RGB.
+This composition supports the project's RGB/255 normalization only. Unknown crop/tile/padding or
+custom-normalization mappings require an explicitly aligned saved map. Confidence is uncalibrated
+per-date softmax. Empty maps return no proposals.
+
+The implementation was checked on synthetic CPU fixtures, including actual controller, queue
+routing with mocked GPU/executor resources, workers, shared lifecycle and a fresh notebook kernel.
+Real SECOND installation, download access, trained detector accuracy, CUDA/BF16 execution and
+four-A100 integration have not been run. No real training/HPO or dataset download runs automatically.
diff --git a/src/satchange/automated_hpo.py b/src/satchange/automated_hpo.py
index c5a65e9..588a1c4 100644
--- a/src/satchange/automated_hpo.py
+++ b/src/satchange/automated_hpo.py
@@ -6,0 +7 @@ not add epochs twice. run_final_benchmark remains a separate, read-only stage.
+
@@ -8,0 +10 @@ from __future__ import annotations
+from collections.abc import Mapping, Sequence
@@ -11 +13 @@ from pathlib import Path
-from typing import Any, Mapping, Sequence
+from typing import Any
@@ -14,2 +16,6 @@ from .hpo_core import (
-    Experiment, ParameterSearchMemoryError, PipelineRunError, PipelineSettings,
-    experiment_lock, load_backend,
+    Experiment,
+    ParameterSearchMemoryError,
+    PipelineRunError,
+    PipelineSettings,
+    experiment_lock,
+    load_backend,
@@ -18 +24,6 @@ from .hpo_core import (
-__all__ = ["run_automated_parameter_search", "request_hpo_pause", "ParameterSearchMemoryError", "PipelineRunError"]
+__all__ = [
+    "run_automated_parameter_search",
+    "request_hpo_pause",
+    "ParameterSearchMemoryError",
+    "PipelineRunError",
+]
@@ -30 +41 @@ def _budgets(values: Sequence[int] | Mapping[str, int]) -> dict[str, int]:
-        result = dict(zip(PHASES, values))
+        result = dict(zip(PHASES, values, strict=False))
@@ -56 +67,3 @@ def request_hpo_pause(search_directory: str | Path, *, after: str = "epoch") ->
-        raise ValueError("No v3 protocol here. The legacy process does not understand pause markers.")
+        raise ValueError(
+            "No v3 protocol here. The legacy process does not understand pause markers."
+        )
@@ -68 +81 @@ def run_automated_parameter_search(
-    thresholds: Sequence[float],
+    thresholds: Sequence[float] = (),
@@ -89,2 +102,2 @@ def run_automated_parameter_search(
-    f1_gap: float = .04,
-    iou_gap: float = .05,
+    f1_gap: float = 0.04,
+    iou_gap: float = 0.05,
@@ -92,3 +105,3 @@ def run_automated_parameter_search(
-    max_trial_memory_mb: float = 80_000.,
-    oom_retry_delay_seconds: float = 30.,
-    oom_retry_memory_margin_mb: float = 512.,
+    max_trial_memory_mb: float = 80_000.0,
+    oom_retry_delay_seconds: float = 30.0,
+    oom_retry_memory_margin_mb: float = 512.0,
@@ -100,0 +114,3 @@ def run_automated_parameter_search(
+    task: str = "binary_change_detection",
+    selection_gap: float = 0.04,
+    joint_tiebreaker: bool = False,
@@ -142,0 +159,20 @@ def run_automated_parameter_search(
+    if task not in ("binary_change_detection", "semantic_roi_classification"):
+        raise ValueError(f"Unsupported task: {task}")
+    semantic = task == "semantic_roi_classification"
+    if base_parameters.get("task", "binary_change_detection") != task:
+        raise ValueError("The requested task differs from base_parameters.")
+    if type(joint_tiebreaker) is not bool:
+        raise ValueError("joint_tiebreaker must be boolean.")
+    if not math.isfinite(selection_gap) or not 0 <= selection_gap <= 1:
+        raise ValueError("selection_gap must be in [0, 1].")
+    if semantic and thresholds:
+        raise ValueError("Semantic ROI classification uses no segmentation thresholds.")
+    if semantic and any(
+        key in {"task", "semantic_config", "prepared_identity", "class_weights"}
+        or key == "model__classes"
+        or key.startswith("semantic_config__")
+        for _, space in stages
+        for key in space
+    ):
+        raise ValueError("Keep the semantic taxonomy, data and class weights fixed during HPO.")
+
@@ -155 +191,3 @@ def run_automated_parameter_search(
-    if len(names) != len(set(names)) or any(not name or Path(name).name != name or name in (*PHASES, ".", "..") for name in names):
+    if len(names) != len(set(names)) or any(
+        not name or Path(name).name != name or name in (*PHASES, ".", "..") for name in names
+    ):
@@ -166 +204,3 @@ def run_automated_parameter_search(
-        raise ValueError("This API never silently restarts partial trials. Leave resume_incomplete=True.")
+        raise ValueError(
+            "This API never silently restarts partial trials. Leave resume_incomplete=True."
+        )
@@ -181,2 +221,2 @@ def run_automated_parameter_search(
-    for name, value in (("patience", patience), ("seed_patience", seed_patience)):
-        if value is not None and (type(value) is not int or value < 1):
+    for name, stopping_count in (("patience", patience), ("seed_patience", seed_patience)):
+        if stopping_count is not None and (type(stopping_count) is not int or stopping_count < 1):
@@ -184 +224,5 @@ def run_automated_parameter_search(
-    if type(min_epochs) is not int or min_epochs < 1 or (patience is not None and min_epochs > epochs["hpo"]):
+    if (
+        type(min_epochs) is not int
+        or min_epochs < 1
+        or (patience is not None and min_epochs > epochs["hpo"])
+    ):
@@ -186 +230,4 @@ def run_automated_parameter_search(
-    if not thresholds or any(isinstance(t, bool) or not math.isfinite(t) or not 0 < t < 1 for t in thresholds):
+    if not semantic and (
+        not thresholds
+        or any(isinstance(t, bool) or not math.isfinite(t) or not 0 < t < 1 for t in thresholds)
+    ):
@@ -192,3 +239,5 @@ def run_automated_parameter_search(
-    for name, value, positive in (("max_trial_memory_mb", max_trial_memory_mb, True),
-                                  ("oom_retry_delay_seconds", oom_retry_delay_seconds, True),
-                                  ("oom_retry_memory_margin_mb", oom_retry_memory_margin_mb, False)):
+    for name, value, positive in (
+        ("max_trial_memory_mb", max_trial_memory_mb, True),
+        ("oom_retry_delay_seconds", oom_retry_delay_seconds, True),
+        ("oom_retry_memory_margin_mb", oom_retry_memory_margin_mb, False),
+    ):
@@ -203,2 +252,8 @@ def run_automated_parameter_search(
-    context.update(model_name=model_name.strip().lower(), dataset_root=str(Path(dataset_root).expanduser().resolve()),
-                   thresholds=list(thresholds), allow_legacy_import=allow_legacy_import)
+    context.update(
+        model_name=model_name.strip().lower(),
+        dataset_root=str(Path(dataset_root).expanduser().resolve()),
+        thresholds=list(thresholds),
+        allow_legacy_import=allow_legacy_import,
+    )
+    if semantic:
+        context["task"] = task
@@ -209,3 +264,8 @@ def run_automated_parameter_search(
-        name: None if count is None else {"monitor": "validation_f1", "patience": count,
-                                           "min_epochs": min_epochs if name == "hpo" else 1,
-                                           "min_delta": float(min_delta)}
+        name: None
+        if count is None
+        else {
+            "monitor": "selection_score" if semantic else "validation_f1",
+            "patience": count,
+            "min_epochs": min_epochs if name == "hpo" else 1,
+            "min_delta": float(min_delta),
+        }
@@ -215,7 +275,27 @@ def run_automated_parameter_search(
-        root=root, backend=backend, context=context, base_parameters=base,
-        stages=deepcopy(list(stages)), stage_epochs=epochs, policies=policies, seeds=list(seeds),
-        devices=list(devices), gaps={"f1": f1_gap, "iou": iou_gap},
-        reuse_from=_paths(reuse_from), checkpoints=_paths(resume_from),
-        additional_seeds=additional_seeds, checkpoint_interval=checkpoint_interval,
-        max_trial_memory_mib=max_trial_memory_mb, retry_seconds=oom_retry_delay_seconds,
-        memory_margin_mib=oom_retry_memory_margin_mb, max_oom_attempts=max_oom_attempts, verbose=verbose)
+        root=root,
+        backend=backend,
+        context=context,
+        base_parameters=base,
+        stages=deepcopy(list(stages)),
+        stage_epochs=epochs,
+        policies=policies,
+        seeds=list(seeds),
+        devices=list(devices),
+        metrics=(
+            ("selection_score", "joint_accuracy") if joint_tiebreaker else ("selection_score",)
+        )
+        if semantic
+        else ("f1", "iou"),
+        gaps={"selection_score": selection_gap, "joint_accuracy": 1.0}
+        if semantic
+        else {"f1": f1_gap, "iou": iou_gap},
+        reuse_from=_paths(reuse_from),
+        checkpoints=_paths(resume_from),
+        additional_seeds=additional_seeds,
+        checkpoint_interval=checkpoint_interval,
+        max_trial_memory_mib=max_trial_memory_mb,
+        retry_seconds=oom_retry_delay_seconds,
+        memory_margin_mib=oom_retry_memory_margin_mb,
+        max_oom_attempts=max_oom_attempts,
+        verbose=verbose,
+    )
@@ -223,3 +303,9 @@ def run_automated_parameter_search(
-        return Experiment(settings).run(stage, hpo_stage=hpo_stage, pending_policy=pending_policy,
-                                         dry_run=dry_run, additional_epochs=additional_epochs,
-                                         extension_id=extension_id, final_parameters=final_parameters)
+        return Experiment(settings).run(
+            stage,
+            hpo_stage=hpo_stage,
+            pending_policy=pending_policy,
+            dry_run=dry_run,
+            additional_epochs=additional_epochs,
+            extension_id=extension_id,
+            final_parameters=final_parameters,
+        )
diff --git a/src/satchange/execution.py b/src/satchange/execution.py
index 4ff77cb..c2dac88 100644
--- a/src/satchange/execution.py
+++ b/src/satchange/execution.py
@@ -16,0 +17,15 @@ from torch.utils.data import DataLoader
+def resolve_device(device: str | torch.device) -> torch.device:
+    """Resolve cuda to its current visible index; never silently fall back to CPU."""
+    resolved = torch.device(device)
+    if resolved.type not in ("cpu", "cuda"):
+        raise ValueError("Use cpu, cuda, or cuda:N.")
+    if resolved.type == "cuda":
+        if not torch.cuda.is_available():
+            raise ValueError("CUDA was requested but is unavailable.")
+        index = torch.cuda.current_device() if resolved.index is None else resolved.index
+        if not 0 <= index < torch.cuda.device_count():
+            raise ValueError(f"CUDA index {index} is not visible to this process.")
+        resolved = torch.device("cuda", index)
+    return resolved
+
+
diff --git a/src/satchange/final_benchmark.py b/src/satchange/final_benchmark.py
index a310fa3..d5410c4 100644
--- a/src/satchange/final_benchmark.py
+++ b/src/satchange/final_benchmark.py
@@ -84,0 +85 @@ def run_final_benchmark(
+    task: str = "binary_change_detection",
@@ -127,0 +129,14 @@ def run_final_benchmark(
+    if task == "semantic_roi_classification":
+        from satchange.semantic.benchmark import finalize_semantic
+        if thresholds is not None or image_size != "auto":
+            raise ValueError("Semantic benchmark uses native ROIs and no binary thresholds.")
+        return finalize_semantic(
+            paths=paths, prepared=dataset, output=output, variant=variant, root=root,
+            training_results=training_results, metadata=metadata, device=device,
+            precision=precision, evaluation_batch_size=evaluation_batch_size,
+            num_workers=num_workers, deployment_seed=deployment_seed,
+            warmup=warmup, repetitions=repetitions, strict_validation=strict_validation,
+            validation_atol=validation_atol, overwrite=overwrite,
+        )
+    if task != "binary_change_detection":
+        raise ValueError(f"Unsupported benchmark task: {task}")
diff --git a/src/satchange/hpo_backend.py b/src/satchange/hpo_backend.py
index ec7de1f..2fed03e 100644
--- a/src/satchange/hpo_backend.py
+++ b/src/satchange/hpo_backend.py
@@ -6,0 +7 @@ All .pt inputs must be your own checkpoints (the project uses full torch.load).
+
@@ -15 +16 @@ import subprocess
-import time
+from collections.abc import Mapping, Sequence
@@ -19 +20 @@ from pathlib import Path
-from typing import Any, Mapping, Sequence
+from typing import Any, cast
@@ -24 +25 @@ from satchange.data import RandomFlipRotate, create_dataset_dataloader
-from satchange.execution import seed_everything
+from satchange.execution import resolve_device, seed_everything
@@ -27,3 +28,13 @@ from satchange.losses import BCEDiceLoss
-from satchange.models.factory import MODEL_REGISTRY, count_model_parameters, create_change_model
-from satchange.optimization import _build_trial_parameters, _create_scheduler, _is_cuda_oom_error, _parse_cuda_oom_message
-from satchange.training import EarlyStopping, fit_binary_change_model, _save_checkpoint
+from satchange.models.factory import (
+    MODEL_REGISTRY,
+    SEMANTIC_TASK,
+    count_model_parameters,
+    create_task_model,
+)
+from satchange.optimization import (
+    _build_trial_parameters,
+    _create_scheduler,
+    _is_cuda_oom_error,
+    _parse_cuda_oom_message,
+)
+from satchange.training import EarlyStopping, _save_checkpoint, fit_binary_change_model
@@ -32,3 +43,27 @@ TRAIN_KEYS = {
-    "seed", "model_parameters", "image_size", "batch_size", "num_workers",
-    "learning_rate", "weight_decay", "scheduler", "positive_weight", "bce_weight",
-    "dice_weight", "precision",
+    "seed",
+    "model_parameters",
+    "image_size",
+    "batch_size",
+    "num_workers",
+    "learning_rate",
+    "weight_decay",
+    "scheduler",
+    "positive_weight",
+    "bce_weight",
+    "dice_weight",
+    "precision",
+}
+SEMANTIC_TRAIN_KEYS = {
+    "task",
+    "semantic_config",
+    "prepared_identity",
+    "class_weights",
+    "tf32",
+    "seed",
+    "model_parameters",
+    "batch_size",
+    "num_workers",
+    "learning_rate",
+    "weight_decay",
+    "scheduler",
+    "precision",
@@ -37,4 +72,19 @@ GENERATED_KEYS = {
-    "epochs", "number_of_epochs", "code_revision", "torch_version", "cuda_version",
-    "thresholds", "world_size", "execution_mode", "architecture_variant", "model_name",
-    "early_stopping", "patience", "min_epochs", "min_delta", "resume_from",
-    "source_checkpoint", "target_epoch", "hpo_phase", "run_name",
+    "epochs",
+    "number_of_epochs",
+    "code_revision",
+    "torch_version",
+    "cuda_version",
+    "thresholds",
+    "world_size",
+    "execution_mode",
+    "architecture_variant",
+    "model_name",
+    "early_stopping",
+    "patience",
+    "min_epochs",
+    "min_delta",
+    "resume_from",
+    "source_checkpoint",
+    "target_epoch",
+    "hpo_phase",
+    "run_name",
@@ -43 +93,4 @@ GENERATED_KEYS = {
-    "budget_epochs", "dataset_root",
+    "budget_epochs",
+    "dataset_root",
+    "training_data",
+    "gpu_id",
@@ -56,2 +109,3 @@ def _load(path: Path) -> dict[str, Any]:
-    payload = torch.load(path, map_location="cpu", weights_only=False,
-                         mmap=prefix.startswith(b"PK"))
+    payload = torch.load(
+        path, map_location="cpu", weights_only=False, mmap=prefix.startswith(b"PK")
+    )
@@ -70 +124,4 @@ def _policy(value: Any) -> dict[str, Any] | None:
-    return EarlyStopping(**{key: value[key] for key in ("patience", "min_epochs", "min_delta")}).policy
+    return EarlyStopping(
+        **{key: value[key] for key in ("patience", "min_epochs", "min_delta")},
+        monitor=value.get("monitor", "validation_f1"),
+    ).policy
@@ -74,2 +131,5 @@ def _stopping_arguments(policy: dict[str, Any] | None) -> dict[str, Any]:
-    return {"patience": None} if policy is None else {
-        key: policy[key] for key in ("patience", "min_epochs", "min_delta")}
+    return (
+        {"patience": None}
+        if policy is None
+        else {key: policy[key] for key in ("patience", "min_epochs", "min_delta")}
+    )
@@ -91,0 +152 @@ def _check_dataset(document: Mapping[str, Any], context: Mapping[str, Any], sour
+        base: Path | None
@@ -96,3 +157,8 @@ def _check_dataset(document: Mapping[str, Any], context: Mapping[str, Any], sour
-            base = next((parent for parent in source.resolve().parents
-                         if (parent / "src/satchange").is_dir()
-                         or (parent / "pyproject.toml").is_file()), None)
+            base = next(
+                (
+                    parent
+                    for parent in source.resolve().parents
+                    if (parent / "src/satchange").is_dir() or (parent / "pyproject.toml").is_file()
+                ),
+                None,
+            )
@@ -102 +168,2 @@ def _check_dataset(document: Mapping[str, Any], context: Mapping[str, Any], sour
-                'Supply backend_context={"project_root": <source project root>}.')
+                'Supply backend_context={"project_root": <source project root>}.'
+            )
@@ -106 +173,3 @@ def _check_dataset(document: Mapping[str, Any], context: Mapping[str, Any], sour
-        raise ValueError(f"Dataset path differs in {source}: saved={saved.resolve()}, expected={expected}")
+        raise ValueError(
+            f"Dataset path differs in {source}: saved={saved.resolve()}, expected={expected}"
+        )
@@ -109,2 +178,3 @@ def _check_dataset(document: Mapping[str, Any], context: Mapping[str, Any], sour
-def _saved_budget(summary: Mapping[str, Any], checkpoint: Mapping[str, Any],
-                  fallback: int | None, source: Path) -> int | None:
+def _saved_budget(
+    summary: Mapping[str, Any], checkpoint: Mapping[str, Any], fallback: int | None, source: Path
+) -> int | None:
@@ -115,3 +185,8 @@ def _saved_budget(summary: Mapping[str, Any], checkpoint: Mapping[str, Any],
-    candidates = (summary.get("budget_epochs"), checkpoint.get("training_budget_epochs"),
-                  config.get("budget_epochs"), config.get("target_epoch"), fallback,
-                  config.get("epochs"))
+    candidates = (
+        summary.get("budget_epochs"),
+        checkpoint.get("training_budget_epochs"),
+        config.get("budget_epochs"),
+        config.get("target_epoch"),
+        fallback,
+        config.get("epochs"),
+    )
@@ -130 +205,2 @@ class SatelliteBackend:
-        unknown = set(merged) - TRAIN_KEYS - GENERATED_KEYS
+        keys = SEMANTIC_TRAIN_KEYS if merged.get("task") == SEMANTIC_TASK else TRAIN_KEYS
+        unknown = set(merged) - keys - GENERATED_KEYS
@@ -133 +209 @@ class SatelliteBackend:
-        result = {key: deepcopy(value) for key, value in merged.items() if key in TRAIN_KEYS}
+        result = {key: deepcopy(value) for key, value in merged.items() if key in keys}
@@ -136 +212 @@ class SatelliteBackend:
-        return json.loads(canonical(result))
+        return cast(dict[str, Any], json.loads(canonical(result)))
@@ -143,0 +220,16 @@ class SatelliteBackend:
+        if context.get("task") == SEMANTIC_TASK:
+            from satchange.semantic.data import file_digest
+            from satchange.semantic.training import validate_parameters
+
+            if context["model_name"] != "tiny_siamese_change_classifier":
+                raise ValueError("The semantic task requires tiny_siamese_change_classifier.")
+            validate_parameters(parameters)
+            prepared = Path(context["dataset_root"])
+            prepared = prepared / "prepared.json" if prepared.is_dir() else prepared
+            if file_digest(prepared) != parameters["prepared_identity"]:
+                raise ValueError(
+                    "Prepared semantic data identity differs from the training parameters."
+                )
+            if context["thresholds"]:
+                raise ValueError("Semantic trials use no thresholds.")
+            return
@@ -163 +255,6 @@ class SatelliteBackend:
-            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
+            if (
+                isinstance(value, bool)
+                or not isinstance(value, (int, float))
+                or not math.isfinite(value)
+                or value < 0
+            ):
@@ -165,2 +262,5 @@ class SatelliteBackend:
-        BCEDiceLoss(bce_weight=parameters["bce_weight"], dice_weight=parameters["dice_weight"],
-                    pos_weight=parameters["positive_weight"])
+        BCEDiceLoss(
+            bce_weight=parameters["bce_weight"],
+            dice_weight=parameters["dice_weight"],
+            pos_weight=parameters["positive_weight"],
+        )
@@ -170,3 +270,16 @@ class SatelliteBackend:
-        names = {"training.py", "execution.py", "losses.py", "metrics.py", "hpo_backend.py", "optimization.py"}
-        files = [p for p in package.rglob("*.py")
-                 if p.name in names or p.parent.name in ("data", "models")]
+        names = {
+            "training.py",
+            "execution.py",
+            "losses.py",
+            "metrics.py",
+            "hpo_backend.py",
+            "hpo_core.py",
+            "automated_hpo.py",
+            "optimization.py",
+            "run.py",
+        }
+        files = [
+            p
+            for p in package.rglob("*.py")
+            if p.name in names or p.parent.name in ("data", "models", "semantic")
+        ]
@@ -179,2 +292,3 @@ class SatelliteBackend:
-    def _inspect(self, directory: Path, context: dict, budget: int | None,
-                 explicit: Path | None = None) -> dict | None:
+    def _inspect(
+        self, directory: Path, context: dict, budget: int | None, explicit: Path | None = None
+    ) -> dict | None:
@@ -206 +320,6 @@ class SatelliteBackend:
-        thresholds = raw.get("thresholds", context["thresholds"] if context.get("allow_legacy_import") else None)
+        semantic = context.get("task") == SEMANTIC_TASK
+        if checkpoint.get("task") != (SEMANTIC_TASK if semantic else None):
+            raise ValueError(f"Checkpoint task differs: {path}")
+        thresholds = raw.get(
+            "thresholds", [] if semantic or context.get("allow_legacy_import") else None
+        )
@@ -209 +328,4 @@ class SatelliteBackend:
-        if checkpoint.get("precision", "fp32") != parameters["precision"] or checkpoint.get("world_size", 1) != 1:
+        if (
+            checkpoint.get("precision", "fp32") != parameters["precision"]
+            or checkpoint.get("world_size", 1) != 1
+        ):
@@ -239 +361 @@ class SatelliteBackend:
-        metrics = {"f1": float(best["validation_f1"]), "iou": float(best["validation_iou"])}
+        metrics = self._metrics(best, semantic)
@@ -246,2 +368,4 @@ class SatelliteBackend:
-            if (best_payload.get("model_name") != context["model_name"]
-                    or self.normalize(best_payload.get("config", {})) != parameters):
+            if (
+                best_payload.get("model_name") != context["model_name"]
+                or self.normalize(best_payload.get("config", {})) != parameters
+            ):
@@ -255 +379,3 @@ class SatelliteBackend:
-                saved_best = best_payload.get("current_metrics") or best_payload.get("best_metrics", {})
+                saved_best = best_payload.get("current_metrics") or best_payload.get(
+                    "best_metrics", {}
+                )
@@ -257 +383 @@ class SatelliteBackend:
-                    saved_value = saved_best.get(f"validation_{metric}")
+                    saved_value = saved_best.get(metric if semantic else f"validation_{metric}")
@@ -259 +385,3 @@ class SatelliteBackend:
-                        raise ValueError(f"best.pt {metric} disagrees with last.pt metadata: {directory}")
+                        raise ValueError(
+                            f"best.pt {metric} disagrees with last.pt metadata: {directory}"
+                        )
@@ -264 +392,3 @@ class SatelliteBackend:
-            raise FileNotFoundError(f"Missing best.pt for historical epoch {best_epoch}: {directory}")
+            raise FileNotFoundError(
+                f"Missing best.pt for historical epoch {best_epoch}: {directory}"
+            )
@@ -266,3 +396,8 @@ class SatelliteBackend:
-            "parameters": parameters, "epoch": epoch, "policy": policy, "stop_reason": reason,
-            "status": "completed" if complete else "partial", "budget_epochs": budget,
-            "run_directory": str(directory.resolve()), "run_name": checkpoint["run_name"],
+            "parameters": parameters,
+            "epoch": epoch,
+            "policy": policy,
+            "stop_reason": reason,
+            "status": "completed" if complete else "partial",
+            "budget_epochs": budget,
+            "run_directory": str(directory.resolve()),
+            "run_name": checkpoint["run_name"],
@@ -270,5 +405,9 @@ class SatelliteBackend:
-            "best_checkpoint": str(best_path.resolve()), "best_epoch": best_epoch,
-            "best_f1": metrics["f1"], "best_iou": metrics["iou"], "metrics": metrics,
-            "best_threshold": float(best["best_threshold"]), "model_name": context["model_name"],
-            "training_seconds": summary.get("training_seconds"), "gpu_hours": summary.get("gpu_hours"),
-            "imported_from": str(path.resolve()), "summary_path": str(summary_path.resolve()),
+            "best_checkpoint": str(best_path.resolve()),
+            "best_epoch": best_epoch,
+            **self._summary_metrics(best, semantic),
+            "metrics": metrics,
+            "model_name": context["model_name"],
+            "training_seconds": summary.get("training_seconds"),
+            "gpu_hours": summary.get("gpu_hours"),
+            "imported_from": str(path.resolve()),
+            "summary_path": str(summary_path.resolve()),
@@ -279 +418,33 @@ class SatelliteBackend:
-    def discover(self, roots: Sequence[str], checkpoints: Sequence[str], context: dict) -> list[dict]:
+    @staticmethod
+    def _metrics(best: Mapping[str, Any], semantic: bool) -> dict[str, float]:
+        if semantic:
+            score = float(best["selection_score"])
+            expected = (float(best["before_macro_f1"]) + float(best["after_macro_f1"])) / 2
+            if not math.isclose(score, expected, abs_tol=1e-12):
+                raise ValueError("Semantic selection score disagrees with per-date macro F1.")
+            return {"selection_score": score, "joint_accuracy": float(best["joint_accuracy"])}
+        return {"f1": float(best["validation_f1"]), "iou": float(best["validation_iou"])}
+
+    @classmethod
+    def _summary_metrics(cls, best: Mapping[str, Any], semantic: bool) -> dict[str, Any]:
+        if semantic:
+            return {
+                "task": SEMANTIC_TASK,
+                **cls._metrics(best, True),
+                "selected_metrics": {
+                    key: deepcopy(value)
+                    for key, value in best.items()
+                    if key.startswith(("before_", "after_"))
+                    or key
+                    in ("selection_score", "selection_rule", "joint_accuracy", "classes", "samples")
+                },
+            }
+        return {
+            "best_f1": float(best["validation_f1"]),
+            "best_iou": float(best["validation_iou"]),
+            "best_threshold": float(best["best_threshold"]),
+        }
+
+    def discover(
+        self, roots: Sequence[str], checkpoints: Sequence[str], context: dict
+    ) -> list[dict]:
@@ -289,2 +460,7 @@ class SatelliteBackend:
-                    raise ValueError("Set allow_legacy_import=True to import your old experiment explicitly.")
-                if protocol.get("model") != context["model_name"] or protocol.get("thresholds") != context["thresholds"]:
+                    raise ValueError(
+                        "Set allow_legacy_import=True to import your old experiment explicitly."
+                    )
+                if (
+                    protocol.get("model") != context["model_name"]
+                    or protocol.get("thresholds") != context["thresholds"]
+                ):
@@ -292 +468,4 @@ class SatelliteBackend:
-                if Path(protocol["dataset_root"]).resolve() != Path(context["dataset_root"]).resolve():
+                if (
+                    Path(protocol["dataset_root"]).resolve()
+                    != Path(context["dataset_root"]).resolve()
+                ):
@@ -300,2 +479,4 @@ class SatelliteBackend:
-                if (saved_context.get("model_name") != context["model_name"]
-                        or saved_context.get("thresholds") != context["thresholds"]):
+                if (
+                    saved_context.get("model_name") != context["model_name"]
+                    or saved_context.get("thresholds") != context["thresholds"]
+                ):
@@ -302,0 +484,2 @@ class SatelliteBackend:
+                if saved_context.get("task") != context.get("task"):
+                    raise ValueError(f"Pipeline task differs: {pipeline_path}")
@@ -312,2 +495,6 @@ class SatelliteBackend:
-                directories[:] = [d for d in directories if not d.startswith(".")
-                                   and d not in {"checkpoints", "operations", "__pycache__", "diagnostics"}]
+                directories[:] = [
+                    d
+                    for d in directories
+                    if not d.startswith(".")
+                    and d not in {"checkpoints", "operations", "__pycache__", "diagnostics"}
+                ]
@@ -315 +502,3 @@ class SatelliteBackend:
-                if path.parent.name != context["model_name"] or not ({"last.pt", "best.pt"} & set(filenames)):
+                if path.parent.name != context["model_name"] or not (
+                    {"last.pt", "best.pt"} & set(filenames)
+                ):
@@ -322 +511,3 @@ class SatelliteBackend:
-                row = self._inspect(path, context, budgets.get(request_name, protocol.get("epochs")))
+                row = self._inspect(
+                    path, context, budgets.get(request_name, protocol.get("epochs"))
+                )
@@ -339 +530 @@ class SatelliteBackend:
-                result[name] = {"free_mib": float("inf"), "total_mib": 0.}
+                result[name] = {"free_mib": float("inf"), "total_mib": 0.0}
@@ -358,4 +549,13 @@ class SatelliteBackend:
-            required = allocated + requested if allocated is not None and requested is not None else None
-            return {"status": "oom", "error": str(error), "required_mib": required,
-                    "requirement_kind": "live_tensors_plus_failed_allocation_lower_bound" if required else "unknown",
-                    "parameters": job["parameters"], "target_epoch": job["target_epoch"]}
+            required = (
+                allocated + requested if allocated is not None and requested is not None else None
+            )
+            return {
+                "status": "oom",
+                "error": str(error),
+                "required_mib": required,
+                "requirement_kind": "live_tensors_plus_failed_allocation_lower_bound"
+                if required
+                else "unknown",
+                "parameters": job["parameters"],
+                "target_epoch": job["target_epoch"],
+            }
@@ -366 +566,3 @@ class SatelliteBackend:
-            raise RuntimeError("Run scripts/install_hpo_v3.py and restart the kernel; trainer patch is missing.")
+            raise RuntimeError(
+                "Run scripts/install_hpo_v3.py and restart the kernel; trainer patch is missing."
+            )
@@ -370 +572,3 @@ class SatelliteBackend:
-        device = torch.device(device_name)
+        semantic = context.get("task") == SEMANTIC_TASK
+        score_key = "selection_score" if semantic else "validation_f1"
+        device = resolve_device(device_name)
@@ -375,0 +580,3 @@ class SatelliteBackend:
+        if semantic:
+            torch.backends.cuda.matmul.allow_tf32 = parameters["tf32"]
+            torch.backends.cudnn.allow_tf32 = parameters["tf32"]
@@ -389 +596,4 @@ class SatelliteBackend:
-                raise ValueError("Source checkpoint changed after plan creation. Stop the old training process first.")
+                raise ValueError(
+                    "Source checkpoint changed after plan creation. "
+                    "Stop the old training process first."
+                )
@@ -399 +609,7 @@ class SatelliteBackend:
-            required = {"optimizer_state_dict", "scheduler_state_dict", "rng_states", "history", "early_stopping"}
+            required = {
+                "optimizer_state_dict",
+                "scheduler_state_dict",
+                "rng_states",
+                "history",
+                "early_stopping",
+            }
@@ -402,2 +618,7 @@ class SatelliteBackend:
-                raise ValueError(f"Cannot perform exact state resume; missing {sorted(missing)} in {local_last}")
-            if saved.get("world_size", 1) != 1 or saved.get("precision", "fp32") != parameters["precision"]:
+                raise ValueError(
+                    f"Cannot perform exact state resume; missing {sorted(missing)} in {local_last}"
+                )
+            if (
+                saved.get("world_size", 1) != 1
+                or saved.get("precision", "fp32") != parameters["precision"]
+            ):
@@ -406 +627,4 @@ class SatelliteBackend:
-                raise ValueError("Resume training parameters differ; do not transplant optimizer state across trials.")
+                raise ValueError(
+                    "Resume training parameters differ; do "
+                    "not transplant optimizer state across trials."
+                )
@@ -412 +636 @@ class SatelliteBackend:
-            saved_best = saved.get("best_metrics") or max(history, key=lambda row: row["validation_f1"])
+            saved_best = saved.get("best_metrics") or max(history, key=lambda row: row[score_key])
@@ -417 +641,3 @@ class SatelliteBackend:
-                raise ValueError("Historical best checkpoint is missing or inconsistent; refusing to lose it.")
+                raise ValueError(
+                    "Historical best checkpoint is missing or inconsistent; refusing to lose it."
+                )
@@ -424 +650,3 @@ class SatelliteBackend:
-            raise ValueError("Local checkpoint is beyond the requested target; refusing to truncate history.")
+            raise ValueError(
+                "Local checkpoint is beyond the requested target; refusing to truncate history."
+            )
@@ -428,2 +656,8 @@ class SatelliteBackend:
-            self_message = f"seed={parameters['seed']}: epoch {source_epoch} -> {job['target_epoch']} (+{remaining})"
-            print(f"[RESUME] {self_message}" if source_epoch else f"[FRESH] {self_message}", flush=True)
+            self_message = (
+                f"seed={parameters['seed']}: epoch {source_epoch} "
+                f"-> {job['target_epoch']} (+{remaining})"
+            )
+            print(
+                f"[RESUME] {self_message}" if source_epoch else f"[FRESH] {self_message}",
+                flush=True,
+            )
@@ -431,8 +665,34 @@ class SatelliteBackend:
-                print("[POLICY] New phase stopping policy; optimizer, scheduler and RNG are retained.", flush=True)
-            data_options = dict(dataset_root=context["dataset_root"], image_size=parameters["image_size"],
-                                batch_size=parameters["batch_size"], num_workers=parameters["num_workers"],
-                                pin_memory=device.type == "cuda")
-            _, train_loader = create_dataset_dataloader(
-                **data_options, split="train", shuffle=True, seed=parameters["seed"],
-                transform=RandomFlipRotate(horizontal_flip_probability=.5, vertical_flip_probability=.5))
-            _, val_loader = create_dataset_dataloader(**data_options, split="val")
+                print(
+                    "[POLICY] New phase stopping policy; "
+                    "optimizer, scheduler and RNG are retained.",
+                    flush=True,
+                )
+            loss: torch.nn.Module
+            if semantic:
+                from satchange.semantic.training import SemanticCELoss, create_loaders
+
+                train_loader, val_loader = create_loaders(context["dataset_root"], parameters)
+                loss = SemanticCELoss(parameters["class_weights"]).to(device)
+            else:
+                data_options = dict(
+                    dataset_root=context["dataset_root"],
+                    image_size=parameters["image_size"],
+                    batch_size=parameters["batch_size"],
+                    num_workers=parameters["num_workers"],
+                    pin_memory=device.type == "cuda",
+                )
+                _, train_loader = create_dataset_dataloader(
+                    **data_options,
+                    split="train",
+                    shuffle=True,
+                    seed=parameters["seed"],
+                    transform=RandomFlipRotate(
+                        horizontal_flip_probability=0.5, vertical_flip_probability=0.5
+                    ),
+                )
+                _, val_loader = create_dataset_dataloader(**data_options, split="val")
+                loss = BCEDiceLoss(
+                    bce_weight=parameters["bce_weight"],
+                    dice_weight=parameters["dice_weight"],
+                    pos_weight=parameters["positive_weight"],
+                ).to(device)
@@ -441,2 +701,6 @@ class SatelliteBackend:
-                construction["pretrained_backbone"] = False  # Trained state is about to be restored.
-            model = create_change_model(context["model_name"], construction).to(device)
+                construction["pretrained_backbone"] = (
+                    False  # Trained state is about to be restored.
+                )
+            model = create_task_model(
+                context["model_name"], construction, context.get("task", "binary_change_detection")
+            ).to(device)
@@ -444,4 +708,5 @@ class SatelliteBackend:
-            loss = BCEDiceLoss(bce_weight=parameters["bce_weight"], dice_weight=parameters["dice_weight"],
-                               pos_weight=parameters["positive_weight"]).to(device)
-            optimizer = torch.optim.AdamW(model.parameters(), lr=parameters["learning_rate"],
-                                           weight_decay=parameters["weight_decay"])
+            optimizer = torch.optim.AdamW(
+                model.parameters(),
+                lr=parameters["learning_rate"],
+                weight_decay=parameters["weight_decay"],
+            )
@@ -449,4 +714,22 @@ class SatelliteBackend:
-            config = {**parameters, "thresholds": context["thresholds"], "epochs": job["target_epoch"],
-                      "budget_epochs": job["target_epoch"], "dataset_root": context["dataset_root"],
-                      "execution_mode": "independent_hpo", "world_size": 1, "hpo_phase": job["phase"],
-                      "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda}
+            config = {
+                **parameters,
+                "thresholds": context["thresholds"],
+                "epochs": job["target_epoch"],
+                "budget_epochs": job["target_epoch"],
+                "dataset_root": context["dataset_root"],
+                "execution_mode": "independent_hpo",
+                "world_size": 1,
+                "hpo_phase": job["phase"],
+                "torch_version": str(torch.__version__),
+                "cuda_version": torch.version.cuda,
+            }
+            if semantic:
+                from satchange.semantic.data import SemanticROIDataset
+
+                semantic_dataset = train_loader.dataset
+                assert isinstance(semantic_dataset, SemanticROIDataset)
+                config.pop("thresholds")
+                config["training_data"] = {
+                    "scenes": deepcopy(semantic_dataset.document["provenance"]["scenes"]),
+                    "protocol": deepcopy(semantic_dataset.document["protocol"]),
+                }
@@ -455 +738,2 @@ class SatelliteBackend:
-                    ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
+                    ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
+                ).stdout.strip()
@@ -460,9 +744,22 @@ class SatelliteBackend:
-                    model=model, train_loader=train_loader, validation_loader=val_loader,
-                    loss_function=loss, optimizer=optimizer, device=device,
-                    number_of_epochs=remaining, thresholds=context["thresholds"],
-                    output_root=job["directory"], model_name=context["model_name"], run_name=run_name,
-                    checkpoint_interval=job["checkpoint_interval"], config=config,
-                    resume_from=local_last if source_epoch else None, scheduler=scheduler,
-                    precision=parameters["precision"], log_prefix=f"[{job['id']} | {device}] ",
-                    stop_file=stop_file, reset_early_stopping=reset_stopper,
-                    **_stopping_arguments(job["policy"]))
+                    model=model,
+                    train_loader=train_loader,
+                    validation_loader=val_loader,
+                    loss_function=loss,
+                    optimizer=optimizer,
+                    device=device,
+                    number_of_epochs=remaining,
+                    thresholds=context["thresholds"],
+                    output_root=job["directory"],
+                    model_name=context["model_name"],
+                    run_name=run_name,
+                    checkpoint_interval=job["checkpoint_interval"],
+                    config=config,
+                    resume_from=local_last if source_epoch else None,
+                    scheduler=scheduler,
+                    precision=parameters["precision"],
+                    log_prefix=f"[{job['id']} | {device}] ",
+                    stop_file=stop_file,
+                    reset_early_stopping=reset_stopper,
+                    task=context.get("task", "binary_change_detection"),
+                    **_stopping_arguments(job["policy"]),
+                )
@@ -473,2 +770,7 @@ class SatelliteBackend:
-            config = {**parameters, "thresholds": context["thresholds"], "epochs": job["target_epoch"],
-                      "budget_epochs": job["target_epoch"], "dataset_root": context["dataset_root"]}
+            config = {
+                **parameters,
+                "thresholds": context["thresholds"],
+                "epochs": job["target_epoch"],
+                "budget_epochs": job["target_epoch"],
+                "dataset_root": context["dataset_root"],
+            }
@@ -478 +780 @@ class SatelliteBackend:
-        best = max(history, key=lambda row: row["validation_f1"])
+        best = max(history, key=lambda row: row[score_key])
@@ -480 +782,3 @@ class SatelliteBackend:
-        complete = epoch == job["target_epoch"] or (reason == "early_stopping" and job["policy"] is not None)
+        complete = epoch == job["target_epoch"] or (
+            reason == "early_stopping" and job["policy"] is not None
+        )
@@ -482 +786 @@ class SatelliteBackend:
-        metrics = {"f1": float(best["validation_f1"]), "iou": float(best["validation_iou"])}
+        metrics = self._metrics(best, semantic)
@@ -488,5 +792,10 @@ class SatelliteBackend:
-            best_payload.setdefault("original_training_config", deepcopy(best_payload.get("config", {})))
-            best_payload["config"] = {**best_payload.get("config", {}),
-                                      "epochs": job["target_epoch"], "hpo_phase": job["phase"],
-                                      "budget_epochs": job["target_epoch"],
-                                      "dataset_root": context["dataset_root"]}
+            best_payload.setdefault(
+                "original_training_config", deepcopy(best_payload.get("config", {}))
+            )
+            best_payload["config"] = {
+                **best_payload.get("config", {}),
+                "epochs": job["target_epoch"],
+                "hpo_phase": job["phase"],
+                "budget_epochs": job["target_epoch"],
+                "dataset_root": context["dataset_root"],
+            }
@@ -498,2 +807,5 @@ class SatelliteBackend:
-            "trial": run_name, "status": "completed" if complete else "paused",
-            "epoch": epoch, "epochs_completed": epoch, "budget_epochs": job["target_epoch"],
+            "trial": run_name,
+            "status": "completed" if complete else "paused",
+            "epoch": epoch,
+            "epochs_completed": epoch,
+            "budget_epochs": job["target_epoch"],
@@ -502,3 +814,6 @@ class SatelliteBackend:
-            "early_stopping": job["policy"], "policy": job["policy"], "parameters": config,
-            "best_epoch": best["epoch"], "best_threshold": best["best_threshold"],
-            "best_f1": metrics["f1"], "best_iou": metrics["iou"], "metrics": metrics,
+            "early_stopping": job["policy"],
+            "policy": job["policy"],
+            "parameters": config,
+            "best_epoch": best["epoch"],
+            **self._summary_metrics(best, semantic),
+            "metrics": metrics,
@@ -506,2 +821,4 @@ class SatelliteBackend:
-            "best_precision": best.get("validation_precision"), "best_recall": best.get("validation_recall"),
-            "training_seconds": seconds, "seconds_per_epoch": seconds / epoch,
+            "best_precision": best.get("validation_precision"),
+            "best_recall": best.get("validation_recall"),
+            "training_seconds": seconds,
+            "seconds_per_epoch": seconds / epoch,
@@ -511,6 +828,11 @@ class SatelliteBackend:
-            "run_directory": str(run_dir), "run_name": run_name,
-            "last_checkpoint": str(local_last), "best_checkpoint": str(local_best),
-            "model_name": context["model_name"], "device": device_name,
-            "resume_source": source["last_checkpoint"] if source else None,
-            "resumed_from_epoch": source_epoch, "new_epochs_completed": epoch - source_epoch,
-            "early_stopping_reset": reset_stopper, **counts,
+            "run_directory": str(run_dir),
+            "run_name": run_name,
+            "last_checkpoint": str(local_last),
+            "best_checkpoint": str(local_best),
+            "model_name": context["model_name"],
+            "device": device_name,
+            "resume_source": str(local_last) if source_epoch else None,
+            "resumed_from_epoch": source_epoch,
+            "new_epochs_completed": epoch - source_epoch,
+            "early_stopping_reset": reset_stopper,
+            **counts,
diff --git a/src/satchange/hpo_core.py b/src/satchange/hpo_core.py
index 6fe081c..94871df 100644
--- a/src/satchange/hpo_core.py
+++ b/src/satchange/hpo_core.py
@@ -6,0 +7 @@ Budgets are ABSOLUTE target epochs, never implicitly added on every restart.
+
@@ -19,0 +21 @@ import traceback
+from collections.abc import Iterator, Mapping, Sequence
@@ -22 +23,0 @@ from contextlib import contextmanager
-from copy import deepcopy
@@ -26 +27 @@ from pathlib import Path
-from typing import Any, Mapping, Protocol, Sequence
+from typing import Any, Protocol, cast
@@ -61,4 +62,2 @@ def write_json(path: str | Path, value: Any) -> None:
-def experiment_lock(root: Path):
-    """OS releases this advisory lock even after an interrupted kernel (POSIX)."""
-    import fcntl
-
+def experiment_lock(root: Path) -> Iterator[None]:
+    """Hold a nonblocking OS lock; kernel/process exit releases it on Windows and POSIX."""
@@ -66 +65,23 @@ def experiment_lock(root: Path):
-    with (root / ".hpo_v3.lock").open("a+") as handle:
+    with (root / ".hpo_v3.lock").open("a+b") as handle:
+        if os.name == "nt":
+            import msvcrt
+
+            if handle.seek(0, os.SEEK_END) == 0:
+                handle.write(b"\0")
+                handle.flush()
+            handle.seek(0)
+
+            def acquire() -> None:
+                return msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
+
+            def release() -> None:
+                return msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
+        else:
+            fcntl = importlib.import_module("fcntl")
+
+            def acquire() -> None:
+                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
+
+            def release() -> None:
+                fcntl.flock(handle, fcntl.LOCK_UN)
+
@@ -68,2 +89,2 @@ def experiment_lock(root: Path):
-            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
-        except BlockingIOError as error:
+            acquire()
+        except OSError as error:
@@ -74 +95 @@ def experiment_lock(root: Path):
-            handle.write(f"{socket.gethostname()} pid={os.getpid()}\n")
+            handle.write(f"{socket.gethostname()} pid={os.getpid()}\n".encode())
@@ -78 +99,2 @@ def experiment_lock(root: Path):
-            fcntl.flock(handle, fcntl.LOCK_UN)
+            handle.seek(0)
+            release()
@@ -88 +110,3 @@ class Backend(Protocol):
-    def discover(self, roots: Sequence[str], checkpoints: Sequence[str], context: Row) -> list[Row]: ...
+    def discover(
+        self, roots: Sequence[str], checkpoints: Sequence[str], context: Row
+    ) -> list[Row]: ...
@@ -95 +119 @@ def load_backend(reference: str) -> Backend:
-    return getattr(importlib.import_module(module), name)()
+    return cast(Backend, getattr(importlib.import_module(module), name)())
@@ -104,2 +128,5 @@ def _worker(reference: str, job: Row, device: str, stop_file: str) -> Row:
-        return {"status": "failed", "error": f"{type(error).__name__}: {error}",
-                "traceback": traceback.format_exc()}
+        return {
+            "status": "failed",
+            "error": f"{type(error).__name__}: {error}",
+            "traceback": traceback.format_exc(),
+        }
@@ -130 +157 @@ class PipelineSettings:
-    gaps: dict[str, float] = field(default_factory=lambda: {"f1": .04, "iou": .05})
+    gaps: dict[str, float] = field(default_factory=lambda: {"f1": 0.04, "iou": 0.05})
@@ -154,3 +181,6 @@ def can_reuse(record: Row, target: int, policy: Row | None) -> bool:
-    return (record.get("stop_reason") == "early_stopping"
-            and policy is not None and record.get("policy") == policy
-            and record.get("budget_epochs") == target)
+    return (
+        record.get("stop_reason") == "early_stopping"
+        and policy is not None
+        and record.get("policy") == policy
+        and record.get("budget_epochs") == target
+    )
@@ -159 +189,3 @@ def can_reuse(record: Row, target: int, policy: Row | None) -> bool:
-def summarize_families(records: list[Row], seeds: Sequence[int], metrics: Sequence[str]) -> tuple[list[Row], list[Row]]:
+def summarize_families(
+    records: list[Row], seeds: Sequence[int], metrics: Sequence[str]
+) -> tuple[list[Row], list[Row]]:
@@ -170,2 +202,7 @@ def summarize_families(records: list[Row], seeds: Sequence[int], metrics: Sequen
-            incomplete.append({"parameters": json.loads(key), "present_seeds": sorted(by_seed),
-                               "missing_seeds": sorted(set(seeds) - set(by_seed))})
+            incomplete.append(
+                {
+                    "parameters": json.loads(key),
+                    "present_seeds": sorted(by_seed),
+                    "missing_seeds": sorted(set(seeds) - set(by_seed)),
+                }
+            )
@@ -174,2 +211,6 @@ def summarize_families(records: list[Row], seeds: Sequence[int], metrics: Sequen
-        item = {"parameters": json.loads(key), "n_seeds": len(seeds), "seeds": list(seeds),
-                "seed_results": ordered}
+        item = {
+            "parameters": json.loads(key),
+            "n_seeds": len(seeds),
+            "seeds": list(seeds),
+            "seed_results": ordered,
+        }
@@ -183 +224,3 @@ def summarize_families(records: list[Row], seeds: Sequence[int], metrics: Sequen
-    ranking.sort(key=lambda row: tuple(-row[f"mean_{m}"] for m in metrics) + (canonical(row["parameters"]),))
+    ranking.sort(
+        key=lambda row: tuple(-row[f"mean_{m}"] for m in metrics) + (canonical(row["parameters"]),)
+    )
@@ -200 +243,2 @@ class Experiment:
-            [*self.s.reuse_from, str(self.root)], self.s.checkpoints, self.s.context)
+            [*self.s.reuse_from, str(self.root)], self.s.checkpoints, self.s.context
+        )
@@ -203,7 +247,14 @@ class Experiment:
-        fixed = {"schema_version": 3, "backend": self.s.backend,
-                 "backend_code_signature": self.backend.code_signature(),
-                 "context": self.s.context, "base": self.s.base_parameters,
-                 "stages": self.s.stages, "seeds": self.s.seeds,
-                 "metrics": self.s.metrics, "gaps": self.s.gaps,
-                 "additional_seeds": self.s.additional_seeds,
-                 "reuse_from": self.s.reuse_from, "initial_checkpoints": self.s.checkpoints}
+        fixed = {
+            "schema_version": 3,
+            "backend": self.s.backend,
+            "backend_code_signature": self.backend.code_signature(),
+            "context": self.s.context,
+            "base": self.s.base_parameters,
+            "stages": self.s.stages,
+            "seeds": self.s.seeds,
+            "metrics": self.s.metrics,
+            "gaps": self.s.gaps,
+            "additional_seeds": self.s.additional_seeds,
+            "reuse_from": self.s.reuse_from,
+            "initial_checkpoints": self.s.checkpoints,
+        }
@@ -213,2 +264,4 @@ class Experiment:
-            raise ValueError("Model/data/code/search definition changed. Use a new search_name; "
-                             "reuse_from can point to the old experiment.")
+            raise ValueError(
+                "Model/data/code/search definition changed. Use a new search_name; "
+                "reuse_from can point to the old experiment."
+            )
@@ -231,3 +284,7 @@ class Experiment:
-                raise ValueError("Changing to a different early-stopping rule at an already-reached "
-                                 "target is not an exact resume. Use a larger target or a new fresh run.")
-            return sorted(resumable, key=lambda row: (-row["epoch"], row["run_directory"]))[0], False
+                raise ValueError(
+                    "Changing to a different early-stopping rule at an already-reached "
+                    "target is not an exact resume. Use a larger target or a new fresh run."
+                )
+            return sorted(resumable, key=lambda row: (-row["epoch"], row["run_directory"]))[
+                0
+            ], False
@@ -235,3 +292,5 @@ class Experiment:
-            raise RuntimeError("A matching run exists, but its last.pt is missing. Refusing a silent "
-                               f"restart/best.pt fallback for seed={parameters['seed']}. "
-                               "Supply a resume checkpoint explicitly or restore last.pt.")
+            raise RuntimeError(
+                "A matching run exists, but its last.pt is missing. Refusing a silent "
+                f"restart/best.pt fallback for seed={parameters['seed']}. "
+                "Supply a resume checkpoint explicitly or restore last.pt."
+            )
@@ -241 +300,3 @@ class Experiment:
-        if not space or any(not isinstance(values, (list, tuple)) or not values for values in space.values()):
+        if not space or any(
+            not isinstance(values, (list, tuple)) or not values for values in space.values()
+        ):
@@ -244 +305,3 @@ class Experiment:
-        variations = [dict(zip(space, values)) for values in itertools.product(*space.values())]
+        variations = [
+            dict(zip(space, values, strict=False)) for values in itertools.product(*space.values())
+        ]
@@ -246 +309,2 @@ class Experiment:
-            for variation in ([{}] if controls else []) + variations:
+            defaults: list[Row] = [{}] if controls else []
+            for variation in defaults + variations:
@@ -252,5 +316,22 @@ class Experiment:
-    def _operation(self, phase: str, label: str, candidates: list[Row], target: int,
-                   policy: Row | None, pending_policy: str, *, dry_run: bool,
-                   require_existing: bool = False) -> Row:
-        operation_id = identity({"phase": phase, "label": label, "candidates": candidates,
-                                 "target": target, "policy": policy, "pending_policy": pending_policy})
+    def _operation(
+        self,
+        phase: str,
+        label: str,
+        candidates: list[Row],
+        target: int,
+        policy: Row | None,
+        pending_policy: str,
+        *,
+        dry_run: bool,
+        require_existing: bool = False,
+    ) -> Row:
+        operation_id = identity(
+            {
+                "phase": phase,
+                "label": label,
+                "candidates": candidates,
+                "target": target,
+                "policy": policy,
+                "pending_policy": pending_policy,
+            }
+        )
@@ -266,2 +347,5 @@ class Experiment:
-                started_families = {canonical(family(row["parameters"])) for row in self.records
-                                    if row["parameters"]["seed"] in secondary}
+                started_families = {
+                    canonical(family(row["parameters"]))
+                    for row in self.records
+                    if row["parameters"]["seed"] in secondary
+                }
@@ -281,10 +365,26 @@ class Experiment:
-                jobs.append({"id": task_id, "parameters": parameters, "target_epoch": target,
-                             "policy": policy, "context": self.s.context, "phase": phase,
-                             "directory": str(self.root / "runs" / task_id),
-                             "checkpoint_interval": self.s.checkpoint_interval,
-                             "source": source, "reused": reused})
-            plan = {"phase": phase, "label": label, "target_epoch": target, "policy": policy,
-                    "pending_policy": pending_policy, "planned_candidates": len(candidates),
-                    "jobs": jobs, "skipped_unstarted": skipped,
-                    "selection_scope": "all_candidates" if not skipped else "started_subset_only",
-                    "test_used": False}
+                jobs.append(
+                    {
+                        "id": task_id,
+                        "parameters": parameters,
+                        "target_epoch": target,
+                        "policy": policy,
+                        "context": self.s.context,
+                        "phase": phase,
+                        "directory": str(self.root / "runs" / task_id),
+                        "checkpoint_interval": self.s.checkpoint_interval,
+                        "source": source,
+                        "reused": reused,
+                    }
+                )
+            plan = {
+                "phase": phase,
+                "label": label,
+                "target_epoch": target,
+                "policy": policy,
+                "pending_policy": pending_policy,
+                "planned_candidates": len(candidates),
+                "jobs": jobs,
+                "skipped_unstarted": skipped,
+                "selection_scope": "all_candidates" if not skipped else "started_subset_only",
+                "test_used": False,
+            }
@@ -292 +392,3 @@ class Experiment:
-                raise RuntimeError(f"Earlier HPO block '{label}' is incomplete. Run it explicitly first.")
+                raise RuntimeError(
+                    f"Earlier HPO block '{label}' is incomplete. Run it explicitly first."
+                )
@@ -298 +400,3 @@ class Experiment:
-                raise RuntimeError(f"Earlier HPO block '{label}' is incomplete. Run it explicitly first.")
+                raise RuntimeError(
+                    f"Earlier HPO block '{label}' is incomplete. Run it explicitly first."
+                )
@@ -301,2 +405,4 @@ class Experiment:
-        self.announce(f"{label}: target={target}, retained={len(plan['jobs'])}, "
-                      f"skipped={len(plan['skipped_unstarted'])}; plan={plan_path}")
+        self.announce(
+            f"{label}: target={target}, retained={len(plan['jobs'])}, "
+            f"skipped={len(plan['skipped_unstarted'])}; plan={plan_path}"
+        )
@@ -304,3 +410,9 @@ class Experiment:
-            return {"complete": False, "dry_run": True, "plan_path": str(plan_path),
-                    "retained_trials": len(plan["jobs"]), "skipped_trials": len(plan["skipped_unstarted"]),
-                    "jobs": plan["jobs"], "selection_scope": plan["selection_scope"]}
+            return {
+                "complete": False,
+                "dry_run": True,
+                "plan_path": str(plan_path),
+                "retained_trials": len(plan["jobs"]),
+                "skipped_trials": len(plan["skipped_unstarted"]),
+                "jobs": plan["jobs"],
+                "selection_scope": plan["selection_scope"],
+            }
@@ -317 +429,3 @@ class Experiment:
-                raise FileNotFoundError(f"Saved completed result lost its checkpoint: {row['best_checkpoint']}")
+                raise FileNotFoundError(
+                    f"Saved completed result lost its checkpoint: {row['best_checkpoint']}"
+                )
@@ -329 +443,3 @@ class Experiment:
-                pending.append({"job": job, "attempts": [], "blocked": set(), "ready": 0., "need": 0.})
+                pending.append(
+                    {"job": job, "attempts": [], "blocked": set(), "ready": 0.0, "need": 0.0}
+                )
@@ -337,10 +453,17 @@ class Experiment:
-            report = {"complete": not pending and not active and not failures and not paused,
-                      "paused": paused, "phase": plan["phase"], "label": plan["label"],
-                      "target_epoch": plan["target_epoch"], "policy": plan["policy"],
-                      "selection_scope": plan["selection_scope"],
-                      "skipped_unstarted": plan["skipped_unstarted"],
-                      "results": list(results.values()), "failed_trials": list(failures.values()),
-                      "failure_history": failure_history,
-                      "unfinished_job_ids": [s["job"]["id"] for s in pending] +
-                                            [s["job"]["id"] for _, s in active.values()],
-                      "test_used": False, "report_path": str(report_path)}
+            report = {
+                "complete": not pending and not active and not failures and not paused,
+                "paused": paused,
+                "phase": plan["phase"],
+                "label": plan["label"],
+                "target_epoch": plan["target_epoch"],
+                "policy": plan["policy"],
+                "selection_scope": plan["selection_scope"],
+                "skipped_unstarted": plan["skipped_unstarted"],
+                "results": list(results.values()),
+                "failed_trials": list(failures.values()),
+                "failure_history": failure_history,
+                "unfinished_job_ids": [s["job"]["id"] for s in pending]
+                + [s["job"]["id"] for _, s in active.values()],
+                "test_used": False,
+                "report_path": str(report_path),
+            }
@@ -353 +476,3 @@ class Experiment:
-        limit = min(self.s.max_trial_memory_mib, largest) if largest else self.s.max_trial_memory_mib
+        limit = (
+            min(self.s.max_trial_memory_mib, largest) if largest else self.s.max_trial_memory_mib
+        )
@@ -355 +480 @@ class Experiment:
-        last_wait = 0.
+        last_wait = 0.0
@@ -364 +489,5 @@ class Experiment:
-                        ordered = sorted(idle, key=lambda d: resource[d].get("free_mib", float("inf")), reverse=True)
+                        ordered = sorted(
+                            idle,
+                            key=lambda d: resource[d].get("free_mib", float("inf")),
+                            reverse=True,
+                        )
@@ -368,3 +497,11 @@ class Experiment:
-                            chosen = next((i for i, s in enumerate(pending)
-                                           if s["ready"] <= time.monotonic() and device not in s["blocked"]
-                                           and available >= min(s["need"], max(0, total - self.s.memory_margin_mib))), None)
+                            chosen = next(
+                                (
+                                    i
+                                    for i, s in enumerate(pending)
+                                    if s["ready"] <= time.monotonic()
+                                    and device not in s["blocked"]
+                                    and available
+                                    >= min(s["need"], max(0, total - self.s.memory_margin_mib))
+                                ),
+                                None,
+                            )
@@ -375,2 +512,9 @@ class Experiment:
-                                executors[device] = ProcessPoolExecutor(max_workers=1, mp_context=context)
-                            state["attempts"].append({"device": device, "free_mib": available if math.isfinite(available) else None})
+                                executors[device] = ProcessPoolExecutor(
+                                    max_workers=1, mp_context=context
+                                )
+                            state["attempts"].append(
+                                {
+                                    "device": device,
+                                    "free_mib": available if math.isfinite(available) else None,
+                                }
+                            )
@@ -378,3 +522,7 @@ class Experiment:
-                            self.announce(f"[RUN] {job['id']} seed={job['parameters']['seed']} "
-                                          f"-> {device}, target={job['target_epoch']}")
-                            future = executors[device].submit(_worker, self.s.backend, job, device, str(stop))
+                            self.announce(
+                                f"[RUN] {job['id']} seed={job['parameters']['seed']} "
+                                f"-> {device}, target={job['target_epoch']}"
+                            )
+                            future = executors[device].submit(
+                                _worker, self.s.backend, job, device, str(stop)
+                            )
@@ -386 +534,3 @@ class Experiment:
-                            self.announce("[WAIT] Waiting for an allowed device with enough free memory.")
+                            self.announce(
+                                "[WAIT] Waiting for an allowed device with enough free memory."
+                            )
@@ -388 +538 @@ class Experiment:
-                        time.sleep(min(self.s.retry_seconds, 1.))
+                        time.sleep(min(self.s.retry_seconds, 1.0))
@@ -390 +540 @@ class Experiment:
-                    done, _ = wait(active, timeout=.5, return_when=FIRST_COMPLETED)
+                    done, _ = wait(active, timeout=0.5, return_when=FIRST_COMPLETED)
@@ -406 +556 @@ class Experiment:
-                            # Terminating the failed worker releases its CUDA context, not only cache.
+                            # Termination releases the failed worker's CUDA context.
@@ -409,2 +559,4 @@ class Experiment:
-                            exhausted = (self.s.max_oom_attempts is not None and
-                                         len(state["attempts"]) >= self.s.max_oom_attempts)
+                            exhausted = (
+                                self.s.max_oom_attempts is not None
+                                and len(state["attempts"]) >= self.s.max_oom_attempts
+                            )
@@ -425 +577,3 @@ class Experiment:
-                                self.announce(f"[OOM -> RETRY] {job['id']}; keep last.pt, try another device.")
+                                self.announce(
+                                    f"[OOM -> RETRY] {job['id']}; keep last.pt, try another device."
+                                )
@@ -432 +586,4 @@ class Experiment:
-                    self.announce("[PAUSE] No new trials. Waiting for active trials to save the current epoch.")
+                    self.announce(
+                        "[PAUSE] No new trials. Waiting for active trials "
+                        "to save the current epoch."
+                    )
@@ -438,4 +595,12 @@ class Experiment:
-            error_class = ParameterSearchMemoryError if any(
-                row.get("status", "").startswith("failed_cuda_oom") for row in failures.values()) else PipelineRunError
-            raise error_class(f"{len(failures)} trial(s) failed. Successful results saved: {report_path}",
-                              str(report_path), list(failures.values()))
+            error_class = (
+                ParameterSearchMemoryError
+                if any(
+                    row.get("status", "").startswith("failed_cuda_oom") for row in failures.values()
+                )
+                else PipelineRunError
+            )
+            raise error_class(
+                f"{len(failures)} trial(s) failed. Successful results saved: {report_path}",
+                str(report_path),
+                list(failures.values()),
+            )
@@ -451,2 +616,8 @@ class Experiment:
-        survivors = [row for row in rows if all(
-            maxima[m] - row["metrics"][m] <= self.s.gaps.get(m, 0) + 1e-12 for m in self.s.metrics)]
+        survivors = [
+            row
+            for row in rows
+            if all(
+                maxima[m] - row["metrics"][m] <= self.s.gaps.get(m, 0) + 1e-12
+                for m in self.s.metrics
+            )
+        ]
@@ -461,2 +632,3 @@ class Experiment:
-    def _hpo(self, stop_at: str | None, pending_policy: str, dry_run: bool,
-             execute: bool, target: int) -> Row:
+    def _hpo(
+        self, stop_at: str | None, pending_policy: str, dry_run: bool, execute: bool, target: int
+    ) -> Row:
@@ -471,3 +643,10 @@ class Experiment:
-            result = self._operation("hpo", name, candidates, target, self.s.policies["hpo"],
-                                     pending_policy if current else "all", dry_run=dry_run and current,
-                                     require_existing=not current)
+            result = self._operation(
+                "hpo",
+                name,
+                candidates,
+                target,
+                self.s.policies["hpo"],
+                pending_policy if current else "all",
+                dry_run=dry_run and current,
+                require_existing=not current,
+            )
@@ -482,4 +661,11 @@ class Experiment:
-    def run(self, stage: str, *, hpo_stage: str | None = None,
-            pending_policy: str = "all", dry_run: bool = False,
-            additional_epochs: int = 0, extension_id: str | None = None,
-            final_parameters: Row | None = None) -> Row:
+    def run(
+        self,
+        stage: str,
+        *,
+        hpo_stage: str | None = None,
+        pending_policy: str = "all",
+        dry_run: bool = False,
+        additional_epochs: int = 0,
+        extension_id: str | None = None,
+        final_parameters: Row | None = None,
+    ) -> Row:
@@ -489,2 +675,8 @@ class Experiment:
-            if not extension_id or Path(extension_id).name != extension_id or extension_id in (".", ".."):
-                raise ValueError("additional_epochs requires a simple extension_id (idempotency key).")
+            if (
+                not extension_id
+                or Path(extension_id).name != extension_id
+                or extension_id in (".", "..")
+            ):
+                raise ValueError(
+                    "additional_epochs requires a simple extension_id (idempotency key)."
+                )
@@ -492,2 +684,7 @@ class Experiment:
-            contract = {"stage": stage, "hpo_stage": hpo_stage, "base_target": target,
-                        "additional_epochs": additional_epochs, "target_epoch": target + additional_epochs}
+            contract = {
+                "stage": stage,
+                "hpo_stage": hpo_stage,
+                "base_target": target,
+                "additional_epochs": additional_epochs,
+                "target_epoch": target + additional_epochs,
+            }
@@ -495 +692,3 @@ class Experiment:
-                raise ValueError("This extension_id already has a different target; use a new extension_id.")
+                raise ValueError(
+                    "This extension_id already has a different target; use a new extension_id."
+                )
@@ -501,2 +700,5 @@ class Experiment:
-            raise ValueError(f"{stage} already targets {active_targets[stage]} epochs; requested {target}. "
-                             "Keep the extension arguments or update stage_epochs; targets never shrink silently.")
+            raise ValueError(
+                f"{stage} already targets {active_targets[stage]} epochs; requested {target}. "
+                "Keep the extension arguments or update stage_epochs; "
+                "targets never shrink silently."
+            )
@@ -507,2 +709,4 @@ class Experiment:
-                raise ValueError(f"Update stage_epochs[{previous_phase!r}] to its extended target "
-                                 f"{active_targets[previous_phase]} before using its selection.")
+                raise ValueError(
+                    f"Update stage_epochs[{previous_phase!r}] to its extended target "
+                    f"{active_targets[previous_phase]} before using its selection."
+                )
@@ -521,2 +725,9 @@ class Experiment:
-            report = self._operation(stage, stage, candidates, target, self.s.policies[stage],
-                                     pending_policy, dry_run=dry_run)
+            report = self._operation(
+                stage,
+                stage,
+                candidates,
+                target,
+                self.s.policies[stage],
+                pending_policy,
+                dry_run=dry_run,
+            )
@@ -527 +738,3 @@ class Experiment:
-            report.update(ranking=ranking, incomplete_families=incomplete, seeds=seeds, selected=selected)
+            report.update(
+                ranking=ranking, incomplete_families=incomplete, seeds=seeds, selected=selected
+            )
@@ -533,2 +746,5 @@ class Experiment:
-                raise RuntimeError("No configuration has the complete required seed set. Partial families "
-                                   "are saved, but cannot win. Use pending_policy='all' or choose final_parameters explicitly.")
+                raise RuntimeError(
+                    "No configuration has the complete required seed set. Partial families "
+                    "are saved, but cannot win. Use pending_policy='all' "
+                    "or choose final_parameters explicitly."
+                )
@@ -542,2 +758,7 @@ class Experiment:
-            if final_parameters is not None and family(self.backend.normalize(final_parameters)) != parameters:
-                raise ValueError("Final configuration already frozen. Use a new search_name to change it.")
+            if (
+                final_parameters is not None
+                and family(self.backend.normalize(final_parameters)) != parameters
+            ):
+                raise ValueError(
+                    "Final configuration already frozen. Use a new search_name to change it."
+                )
@@ -551 +772,3 @@ class Experiment:
-                    raise RuntimeError("Run stage='seeds' first, or provide explicit final_parameters.")
+                    raise RuntimeError(
+                        "Run stage='seeds' first, or provide explicit final_parameters."
+                    )
@@ -553,2 +776,5 @@ class Experiment:
-                if (not report.get("complete") or report.get("seeds") != self.s.seeds
-                        or report.get("target_epoch", 0) < self.s.stage_epochs["seeds"]):
+                if (
+                    not report.get("complete")
+                    or report.get("seeds") != self.s.seeds
+                    or report.get("target_epoch", 0) < self.s.stage_epochs["seeds"]
+                ):
@@ -565,2 +791,5 @@ class Experiment:
-                if (exploratory.get("complete") and selected is not None
-                        and exploratory.get("seeds") == [self.s.base_parameters["seed"]]):
+                if (
+                    exploratory.get("complete")
+                    and selected is not None
+                    and exploratory.get("seeds") == [self.s.base_parameters["seed"]]
+                ):
@@ -571 +800,4 @@ class Experiment:
-                    best = max(hpo["results"], key=lambda row: tuple(row["metrics"][m] for m in self.s.metrics))
+                    best = max(
+                        hpo["results"],
+                        key=lambda row: tuple(row["metrics"][m] for m in self.s.metrics),
+                    )
@@ -580,2 +812,3 @@ class Experiment:
-        report = self._operation("final", "final", candidates, target, self.s.policies["final"],
-                                 "all", dry_run=dry_run)
+        report = self._operation(
+            "final", "final", candidates, target, self.s.policies["final"], "all", dry_run=dry_run
+        )
@@ -589,3 +822,9 @@ class Experiment:
-        report.update(seeds=self.s.seeds, ranking=ranking, selected=ranking[0],
-                      best_checkpoints={str(row["parameters"]["seed"]): row["best_checkpoint"] for row in report["results"]},
-                      choice=choice)
+        report.update(
+            seeds=self.s.seeds,
+            ranking=ranking,
+            selected=ranking[0],
+            best_checkpoints={
+                str(row["parameters"]["seed"]): row["best_checkpoint"] for row in report["results"]
+            },
+            choice=choice,
+        )
diff --git a/src/satchange/models/__init__.py b/src/satchange/models/__init__.py
index 42d2e79..b2bc106 100644
--- a/src/satchange/models/__init__.py
+++ b/src/satchange/models/__init__.py
@@ -4 +4 @@ from satchange.models.bit import BIT
-from satchange.models.factory import count_model_parameters, create_change_model
+from satchange.models.factory import count_model_parameters, create_change_model, create_task_model
@@ -6,0 +7 @@ from satchange.models.siam_mit_b3_diff import SiamMiTB3Diff
+from satchange.models.tiny_siamese_change_classifier import TinySiameseChangeClassifier
@@ -8,0 +10 @@ __all__ = [
+    "TinySiameseChangeClassifier",
@@ -13,0 +16 @@ __all__ = [
+    "create_task_model",
diff --git a/src/satchange/models/factory.py b/src/satchange/models/factory.py
index 37f4a4b..f300b9f 100644
--- a/src/satchange/models/factory.py
+++ b/src/satchange/models/factory.py
@@ -3 +2,0 @@
-
@@ -10,0 +10,23 @@ from satchange.models.siam_mit_b3_diff import SiamMiTB3Diff
+from satchange.models.tiny_siamese_change_classifier import TinySiameseChangeClassifier
+
+BINARY_TASK = "binary_change_detection"
+SEMANTIC_TASK = "semantic_roi_classification"
+
+
+def model_class_for_task(model_name: str, task: str = BINARY_TASK) -> type[nn.Module]:
+    """Select a supported task contract without registering a classifier as a detector."""
+    if task == SEMANTIC_TASK and model_name == "tiny_siamese_change_classifier":
+        return TinySiameseChangeClassifier
+    if task == BINARY_TASK and model_name in MODEL_REGISTRY:
+        return MODEL_REGISTRY[model_name]
+    raise ValueError(f"Unsupported model/task pair: {model_name}/{task}")
+
+
+def create_task_model(
+    model_name: str,
+    parameters: dict[str, Any],
+    task: str = BINARY_TASK,
+) -> nn.Module:
+    """Construct a model with an explicit detector or ROI-classifier contract."""
+    return model_class_for_task(model_name, task)(**parameters)
+
@@ -42,2 +64 @@ def create_change_model(
-            f"Unknown model: '{model_name}'. "
-            f"Available models: {tuple(MODEL_REGISTRY)}."
+            f"Unknown model: '{model_name}'. Available models: {tuple(MODEL_REGISTRY)}."
@@ -46,3 +67 @@ def create_change_model(
-    model_class = MODEL_REGISTRY[
-        model_name
-    ]
+    model_class = MODEL_REGISTRY[model_name]
@@ -50,3 +69 @@ def create_change_model(
-    return model_class(
-        **parameters
-    )
+    return model_class(**parameters)
diff --git a/src/satchange/models/tiny_siamese_change_classifier.py b/src/satchange/models/tiny_siamese_change_classifier.py
new file mode 100644
index 0000000..9c93ef0
--- /dev/null
+++ b/src/satchange/models/tiny_siamese_change_classifier.py
@@ -0,0 +1,99 @@
+"""A small shared CNN for semantic classification of native-scale changed ROIs."""
+
+from collections.abc import Sequence
+
+import torch
+from torch import nn
+from torch.nn import functional as F
+
+
+def validate_classes(classes: Sequence[str]) -> tuple[str, ...]:
+    """Validate the ordered, persisted taxonomy; no dataset classes are inferred."""
+    if isinstance(classes, str) or len(classes) < 2:
+        raise ValueError("classes must contain at least two ordered class names.")
+    if any(not isinstance(name, str) or not name.strip() for name in classes):
+        raise ValueError("Every class name must be a nonempty string.")
+    if len(set(classes)) != len(classes):
+        raise ValueError("Class names must be unique.")
+    return tuple(classes)
+
+
+def validate_unit_tensor(value: torch.Tensor, name: str) -> None:
+    """Reject non-floating, nonfinite, or out-of-range normalized inputs."""
+    if not value.is_floating_point():
+        raise ValueError(f"{name} must be floating point, normalized to [0, 1].")
+    if not torch.isfinite(value).all() or (value < 0).any() or (value > 1).any():
+        raise ValueError(f"{name} must be finite and in [0, 1]; logits are not probabilities.")
+
+
+def masked_pool(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
+    """Area-reduce [B,1,H,W] weights and pool [B,D,h,w] features in FP32.
+
+    Area reduction retains even one positive source pixel on odd-sized grids.
+    Empty support is an error, rather than an implicit global-pooling fallback.
+    """
+    if features.ndim != 4 or mask.ndim != 4 or mask.shape[:2] != (features.shape[0], 1):
+        raise ValueError("Pooling expects features [B,D,h,w] and aligned mask [B,1,H,W].")
+    if features.device != mask.device:
+        raise ValueError("Features and mask must use the same device.")
+    validate_unit_tensor(mask, "mask")
+    weights = F.interpolate(mask.float(), size=features.shape[-2:], mode="area")
+    denominator = weights.sum(dim=(-2, -1))
+    if (denominator <= 0).any():
+        raise ValueError("Every ROI requires a nonempty effective mask.")
+    return (features.float() * weights).sum(dim=(-2, -1)) / (denominator + 1e-8)
+
+
+class TinySiameseChangeClassifier(nn.Module):
+    """Shared RGB backbone and head, with limited local receptive-field context.
+
+    forward(before, after, mask): images [B,3,H,W], mask [B,1,H,W],
+    raw before/after logits [B,C]. RGB remains unmasked and at native scale.
+    Two pooling stages produce floor(H/4) x floor(W/4) feature grids.
+    Both dates share one BatchNorm update by concatenating on the batch axis.
+    """
+
+    def __init__(self, classes: Sequence[str], dropout: float = 0.1) -> None:
+        super().__init__()
+        self.classes = validate_classes(classes)
+        if not 0 <= dropout < 1:
+            raise ValueError("dropout must be in [0, 1).")
+        self.dropout = float(dropout)
+        layers: list[nn.Module] = []
+        for index, (incoming, outgoing) in enumerate(((3, 32), (32, 64), (64, 128))):
+            layers.extend(
+                [
+                    nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False),
+                    nn.BatchNorm2d(outgoing),
+                    nn.ReLU(),
+                ]
+            )
+            if index < 2:
+                layers.append(nn.MaxPool2d(2))
+        self.backbone = nn.Sequential(*layers)
+        self.head = nn.Sequential(
+            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, len(self.classes))
+        )
+
+    def forward(
+        self, before: torch.Tensor, after: torch.Tensor, mask: torch.Tensor
+    ) -> dict[str, torch.Tensor]:
+        """Return two semantic-logit tensors using exactly the same parameters."""
+        if before.ndim != 4 or before.shape[1] != 3 or before.shape != after.shape:
+            raise ValueError("Images must be aligned [B,3,H,W] tensors.")
+        if before.shape[0] == 0 or min(before.shape[-2:]) < 4:
+            raise ValueError("Images require a nonempty batch and height/width of at least 4.")
+        if mask.shape != (before.shape[0], 1, *before.shape[-2:]):
+            raise ValueError("mask must be aligned [B,1,H,W].")
+        if before.device != after.device or before.device != mask.device:
+            raise ValueError("Images and mask must use the same device.")
+        if before.dtype != torch.float32 or after.dtype != torch.float32:
+            raise ValueError("RGB inputs must be float32 normalized to [0, 1].")
+        for name, value in (("before", before), ("after", after), ("mask", mask)):
+            validate_unit_tensor(value, name)
+        if (mask.float().sum(dim=(-2, -1)) <= 0).any():
+            raise ValueError("Every ROI requires a nonempty effective mask.")
+        features = self.backbone(torch.cat((before, after), dim=0))
+        vectors = masked_pool(features, torch.cat((mask, mask), dim=0))
+        before_logits, after_logits = self.head(vectors).chunk(2, dim=0)
+        return {"before_logits": before_logits, "after_logits": after_logits}
diff --git a/src/satchange/optimization.py b/src/satchange/optimization.py
index 11967ab..ec4ac46 100644
--- a/src/satchange/optimization.py
+++ b/src/satchange/optimization.py
@@ -10,2 +9,0 @@ import re
-import warnings
-from uuid import uuid4
@@ -13,0 +12 @@ import time
+import warnings
@@ -22,0 +22 @@ from typing import Any
+from uuid import uuid4
@@ -30,0 +31 @@ from satchange.data import (
+from satchange.execution import resolve_device
@@ -33 +34,2 @@ from satchange.models.factory import (
-    MODEL_REGISTRY,
+    BINARY_TASK,
+    SEMANTIC_TASK,
@@ -35 +37,2 @@ from satchange.models.factory import (
-    create_change_model,
+    create_task_model,
+    model_class_for_task,
@@ -39 +42,2 @@ from satchange.training import (
-    fit_binary_change_model,
+    _best_weights,
+    _save_checkpoint,
@@ -40,0 +45 @@ from satchange.training import (
+    fit_binary_change_model,
@@ -43,2 +47,0 @@ from satchange.training import (
-    _best_weights,
-    _save_checkpoint,
@@ -186,0 +190,19 @@ def _format_trial_parameters(parameters: dict[str, Any]) -> str:
+    if parameters.get("task") == SEMANTIC_TASK:
+        return " | ".join(
+            [
+                f"task={SEMANTIC_TASK}",
+                *[
+                    f"{key}={parameters.get(key)}"
+                    for key in (
+                        "seed",
+                        "learning_rate",
+                        "weight_decay",
+                        "batch_size",
+                        "scheduler",
+                        "precision",
+                        "tf32",
+                    )
+                ],
+                f"dropout={parameters['model_parameters']['dropout']}",
+            ]
+        )
@@ -242 +264,6 @@ def _resume_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
-    values = {key: parameters[key] for key in _RESUME_TRAINING_KEYS if key in parameters}
+    keys = _RESUME_TRAINING_KEYS
+    if parameters.get("task", BINARY_TASK) == SEMANTIC_TASK:
+        keys = {"task", "model_parameters", "semantic_config", "prepared_identity", "class_weights",
+                "seed", "batch_size", "num_workers", "learning_rate", "weight_decay", "scheduler",
+                "precision", "tf32"}
+    values = {key: parameters[key] for key in keys if key in parameters}
@@ -256 +283,3 @@ def read_hpo_checkpoint(
-    if not isinstance(saved, dict) or (_RESUME_TRAINING_KEYS - {"precision"}) - saved.keys():
+    semantic = isinstance(saved, dict) and saved.get("task") == SEMANTIC_TASK
+    if not isinstance(saved, dict) or (not semantic and
+            (_RESUME_TRAINING_KEYS - {"precision"}) - saved.keys()):
@@ -257,0 +287,5 @@ def read_hpo_checkpoint(
+    if semantic:
+        from satchange.semantic.training import validate_parameters
+        validate_parameters(saved)
+        if state.get("task") != SEMANTIC_TASK or thresholds:
+            raise ValueError("Semantic checkpoint task/threshold contract mismatch.")
@@ -264 +298 @@ def read_hpo_checkpoint(
-    if saved.get("thresholds", state.get("thresholds")) != list(thresholds):
+    if not semantic and saved.get("thresholds", state.get("thresholds")) != list(thresholds):
@@ -272 +306 @@ def read_hpo_checkpoint(
-            RuntimeWarning,
+            RuntimeWarning, stacklevel=2,
@@ -280 +314,3 @@ def read_hpo_checkpoint(
-    expected_best = max(history, key=lambda row: row["validation_f1"])
+    expected_best = max(
+        history, key=lambda row: row["selection_score" if semantic else "validation_f1"]
+    )
@@ -283 +319,6 @@ def read_hpo_checkpoint(
-    stopper = EarlyStopping(early_stopping.patience, early_stopping.min_epochs, early_stopping.min_delta)
+    stopper = EarlyStopping(
+        early_stopping.patience,
+        early_stopping.min_epochs,
+        early_stopping.min_delta,
+        monitor=early_stopping.monitor,
+    )
@@ -307 +348,6 @@ def _prepare_trial_resume(
-    restored_stopper = EarlyStopping(early_stopping.patience, early_stopping.min_epochs, early_stopping.min_delta)
+    restored_stopper = EarlyStopping(
+        early_stopping.patience,
+        early_stopping.min_epochs,
+        early_stopping.min_delta,
+        monitor=early_stopping.monitor,
+    )
@@ -337 +383,3 @@ def _write_json_atomic(path: Path, value: Any) -> None:
-        temporary.write_text(json.dumps(value, indent=2, default=str, allow_nan=False), encoding="utf-8")
+        temporary.write_text(
+            json.dumps(value, indent=2, default=str, allow_nan=False), encoding="utf-8"
+        )
@@ -361 +409,4 @@ def _run_single_trial(
-    early_stopping = EarlyStopping(patience, min_epochs, min_delta)
+    task = parameters.get("task", BINARY_TASK)
+    semantic = task == SEMANTIC_TASK
+    score_key = "selection_score" if semantic else "validation_f1"
+    early_stopping = EarlyStopping(patience, min_epochs, min_delta, monitor=score_key)
@@ -372,7 +423,6 @@ def _run_single_trial(
-    device = torch.device(
-        f"cuda:{gpu_id}"
-    )
-
-    torch.cuda.set_device(
-        device
-    )
+    device = resolve_device(f"cuda:{gpu_id}")
+    if device.type == "cuda":
+        torch.cuda.set_device(device)
+    if semantic:
+        torch.backends.cuda.matmul.allow_tf32 = parameters["tf32"]
+        torch.backends.cudnn.allow_tf32 = parameters["tf32"]
@@ -399,4 +449,9 @@ def _run_single_trial(
-    training_transform = RandomFlipRotate(
-        horizontal_flip_probability=0.5,
-        vertical_flip_probability=0.5,
-    )
+    if semantic:
+        from satchange.semantic.training import create_loaders, validate_parameters
+        validate_parameters(parameters)
+        train_loader, validation_loader = create_loaders(dataset_root, parameters)
+    else:
+        training_transform = RandomFlipRotate(
+            horizontal_flip_probability=0.5,
+            vertical_flip_probability=0.5,
+        )
@@ -404,17 +459,18 @@ def _run_single_trial(
-    _, train_loader = (
-        create_dataset_dataloader(
-            dataset_root=dataset_root,
-            split="train",
-            image_size=parameters[
-                "image_size"
-            ],
-            batch_size=parameters[
-                "batch_size"
-            ],
-            num_workers=parameters[
-                "num_workers"
-            ],
-            transform=training_transform,
-            shuffle=True,
-            pin_memory=True,
-            seed=seed,
+        _, train_loader = (
+            create_dataset_dataloader(
+                dataset_root=dataset_root,
+                split="train",
+                image_size=parameters[
+                    "image_size"
+                ],
+                batch_size=parameters[
+                    "batch_size"
+                ],
+                num_workers=parameters[
+                    "num_workers"
+                ],
+                transform=training_transform,
+                shuffle=True,
+                pin_memory=True,
+                seed=seed,
+            )
@@ -422 +477,0 @@ def _run_single_trial(
-    )
@@ -424,14 +479,15 @@ def _run_single_trial(
-    _, validation_loader = (
-        create_dataset_dataloader(
-            dataset_root=dataset_root,
-            split="val",
-            image_size=parameters[
-                "image_size"
-            ],
-            batch_size=parameters[
-                "batch_size"
-            ],
-            num_workers=parameters[
-                "num_workers"
-            ],
-            pin_memory=True,
+        _, validation_loader = (
+            create_dataset_dataloader(
+                dataset_root=dataset_root,
+                split="val",
+                image_size=parameters[
+                    "image_size"
+                ],
+                batch_size=parameters[
+                    "batch_size"
+                ],
+                num_workers=parameters[
+                    "num_workers"
+                ],
+                pin_memory=True,
+            )
@@ -439 +494,0 @@ def _run_single_trial(
-    )
@@ -441,2 +496,2 @@ def _run_single_trial(
-    model = create_change_model(
-        model_name=model_name,
+    model = create_task_model(
+        model_name=model_name, task=task,
@@ -447,0 +503,8 @@ def _run_single_trial(
+    if semantic:
+        from satchange.semantic.data import SemanticROIDataset
+        assert isinstance(train_loader.dataset, SemanticROIDataset)
+        parameters["training_data"] = {
+            "scenes": [{key: row[key] for key in ("sample_id", "group_id", "split", "rgb_hashes")}
+                       for row in train_loader.dataset.document["provenance"]["scenes"]],
+            "protocol": train_loader.dataset.document["protocol"],
+        }
@@ -453 +516,2 @@ def _run_single_trial(
-    parameters["thresholds"] = list(thresholds)
+    if not semantic:
+        parameters["thresholds"] = list(thresholds)
@@ -458,11 +522,16 @@ def _run_single_trial(
-    loss_function = BCEDiceLoss(
-        bce_weight=parameters[
-            "bce_weight"
-        ],
-        dice_weight=parameters[
-            "dice_weight"
-        ],
-        pos_weight=parameters[
-            "positive_weight"
-        ],
-    ).to(device)
+    loss_function: torch.nn.Module
+    if semantic:
+        from satchange.semantic.training import SemanticCELoss
+        loss_function = SemanticCELoss(parameters["class_weights"]).to(device)
+    else:
+        loss_function = BCEDiceLoss(
+            bce_weight=parameters[
+                "bce_weight"
+            ],
+            dice_weight=parameters[
+                "dice_weight"
+            ],
+            pos_weight=parameters[
+                "positive_weight"
+            ],
+        ).to(device)
@@ -491,3 +560,2 @@ def _run_single_trial(
-    torch.cuda.reset_peak_memory_stats(
-        device
-    )
+    if device.type == "cuda":
+        torch.cuda.reset_peak_memory_stats(device)
@@ -504 +572,3 @@ def _run_single_trial(
-    already_stopped = bool(previous_history and previous_history[-1].get("stop_reason") == "early_stopping")
+    already_stopped = bool(
+        previous_history and previous_history[-1].get("stop_reason") == "early_stopping"
+    )
@@ -524 +594 @@ def _run_single_trial(
-            precision=parameters.get("precision", "fp32"),
+            precision=parameters.get("precision", "fp32"), task=task,
@@ -536,3 +606,2 @@ def _run_single_trial(
-    torch.cuda.synchronize(
-        device
-    )
+    if device.type == "cuda":
+        torch.cuda.synchronize(device)
@@ -552 +621 @@ def _run_single_trial(
-            result["validation_f1"]
+            result[score_key]
@@ -571,2 +640,3 @@ def _run_single_trial(
-        "timing_scope": "retained_epoch_compute_plus_last_attempt_wall" if saved_epoch else "attempt_wall",
-
+        "timing_scope": "retained_epoch_compute_plus_last_attempt_wall"
+        if saved_epoch
+        else "attempt_wall",
@@ -577,46 +647,33 @@ def _run_single_trial(
-        "best_epoch": (
-            best_result["epoch"]
-        ),
-        "best_threshold": (
-            best_result[
-                "best_threshold"
-            ]
-        ),
-        "best_validation_loss": (
-            best_result[
-                "validation_loss"
-            ]
-        ),
-        "best_precision": (
-            best_result[
-                "validation_precision"
-            ]
-        ),
-        "best_recall": (
-            best_result[
-                "validation_recall"
-            ]
-        ),
-        "best_f1": (
-            best_result[
-                "validation_f1"
-            ]
-        ),
-        "best_iou": (
-            best_result[
-                "validation_iou"
-            ]
-        ),
-        "training_seconds": (
-            training_seconds
-        ),
-        "seconds_per_epoch": (
-            training_seconds
-            / len(history)
-        ),
-        "gpu_hours": (
-            training_seconds
-            / 3600.0
-        ),
-        "peak_gpu_memory_mb": (
-            peak_gpu_memory_mb
+        "best_epoch": (best_result["epoch"]),
+        **(
+            {
+                "task": task,
+                "selection_score": best_result[score_key],
+                "best_validation_loss": best_result["validation_loss"],
+                "selected_metrics": {
+                    key: best_result[key]
+                    for key in (
+                        "before_macro_f1",
+                        "after_macro_f1",
+                        "before_accuracy",
+                        "after_accuracy",
+                        "joint_accuracy",
+                        "before_confusion",
+                        "after_confusion",
+                        "before_per_class",
+                        "after_per_class",
+                        "selection_rule",
+                        "classes",
+                        "samples",
+                    )
+                },
+            }
+            if semantic
+            else {
+                "best_threshold": (best_result["best_threshold"]),
+                "best_validation_loss": (best_result["validation_loss"]),
+                "best_precision": (best_result["validation_precision"]),
+                "best_recall": (best_result["validation_recall"]),
+                "best_f1": (best_result["validation_f1"]),
+                "best_iou": (best_result["validation_iou"]),
+            }
@@ -623,0 +681,4 @@ def _run_single_trial(
+        "training_seconds": (training_seconds),
+        "seconds_per_epoch": (training_seconds / len(history)),
+        "gpu_hours": (training_seconds * (device.type == "cuda") / 3600.0),
+        "peak_gpu_memory_mb": (peak_gpu_memory_mb),
@@ -962 +1023,3 @@ def _trial_directory(
-def _archive_partial_trial_output(search_output_root: str | Path, model_name: str, trial_number: int) -> None:
+def _archive_partial_trial_output(
+    search_output_root: str | Path, model_name: str, trial_number: int
+) -> None:
@@ -1028 +1091,5 @@ def run_parameter_search(
-    early_stopping = EarlyStopping(patience, min_epochs, min_delta)
+    search_trials = _expand_search_space(search_space)
+    task = _build_trial_parameters(fixed_parameters, search_trials[0]).get("task", BINARY_TASK)
+    semantic = task == SEMANTIC_TASK
+    early_stopping = EarlyStopping(patience, min_epochs, min_delta,
+                                  monitor="selection_score" if semantic else "validation_f1")
@@ -1048,2 +1115 @@ def run_parameter_search(
-    if model_name not in MODEL_REGISTRY:
-        raise ValueError(f"Unknown model: {model_name}")
+    model_class = model_class_for_task(model_name, task)
@@ -1051 +1116,0 @@ def run_parameter_search(
-    search_trials = _expand_search_space(search_space)
@@ -1059,0 +1125,2 @@ def run_parameter_search(
+        if parameters.get("task", BINARY_TASK) != task:
+            raise ValueError("A queue cannot mix different training tasks.")
@@ -1064,0 +1132,4 @@ def run_parameter_search(
+        if semantic:
+            from satchange.semantic.training import validate_parameters
+            validate_parameters(parameters)
+            allowed_training = set(_resume_parameters(parameters)) | {"execution_mode"}
@@ -1071 +1142 @@ def run_parameter_search(
-        supported = inspect.signature(MODEL_REGISTRY[model_name]).parameters
+        supported = inspect.signature(model_class).parameters
@@ -1340,2 +1411,6 @@ def run_parameter_search(
-                        print("F1:", f"{result['best_f1']:.6f}")
-                        print("IoU:", f"{result['best_iou']:.6f}")
+                        if semantic:
+                            print("Selection score:", f"{result['selection_score']:.6f}")
+                        else:
+                            print("F1:", f"{result['best_f1']:.6f}")
+                        if not semantic:
+                            print("IoU:", f"{result['best_iou']:.6f}")
@@ -1493 +1568 @@ def run_parameter_search(
-        key=lambda result: result["best_f1"],
+        key=lambda result: result["selection_score" if semantic else "best_f1"],
diff --git a/src/satchange/run.py b/src/satchange/run.py
index c6a415d..c52b046 100644
--- a/src/satchange/run.py
+++ b/src/satchange/run.py
@@ -27,0 +28 @@ from satchange.execution import (
+    resolve_device,
@@ -63 +64,5 @@ def reconstruct_checkpoint(
-    state = torch.load(path, map_location=device, weights_only=False)
+    from satchange.models.factory import BINARY_TASK, create_task_model
+    device = resolve_device(device)
+    state = torch.load(path, map_location="cpu", weights_only=False)
+    if not isinstance(state, dict):
+        raise ValueError("Checkpoint must contain a project metadata mapping.")
@@ -75 +80,4 @@ def reconstruct_checkpoint(
-    model = create_change_model(state["model_name"], parameters).to(device)
+    task = state.get("task", BINARY_TASK)
+    if config.get("task", task) != task:
+        raise ValueError("Checkpoint task disagrees with config.")
+    model = create_task_model(state["model_name"], parameters, task).to(device)
diff --git a/src/satchange/semantic/__init__.py b/src/satchange/semantic/__init__.py
new file mode 100644
index 0000000..01830c8
--- /dev/null
+++ b/src/satchange/semantic/__init__.py
@@ -0,0 +1,10 @@
+"""Native-scale ROI semantic classification, separate from binary detection."""
+
+from satchange.semantic.roi import (
+    ROI,
+    ROISettings,
+    align_whole_image_probabilities,
+    extract_rois,
+)
+
+__all__ = ["ROI", "ROISettings", "align_whole_image_probabilities", "extract_rois"]
diff --git a/src/satchange/semantic/__main__.py b/src/satchange/semantic/__main__.py
new file mode 100644
index 0000000..6b0ae26
--- /dev/null
+++ b/src/satchange/semantic/__main__.py
@@ -0,0 +1,129 @@
+"""Explicit ROI preparation/evaluation/inference CLI; training uses the common HPO entry."""
+
+import argparse
+import json
+from pathlib import Path
+from typing import Any
+
+import torch
+
+from satchange.execution import seed_everything
+from satchange.models import TinySiameseChangeClassifier, count_model_parameters
+from satchange.semantic.data import prepare_dataset, read_map, read_rgb, write_json
+from satchange.semantic.inference import evaluate_classifier, predict_checkpoint
+from satchange.semantic.training import semantic_loss
+from satchange.utils import load_config
+
+
+def synthetic_smoke(steps: int = 60) -> dict[str, Any]:
+    """Bounded CPU forward/backward and tiny-overfit check; no real-dataset evidence."""
+    if not 1 <= steps <= 200:
+        raise ValueError("Synthetic steps must be between 1 and 200.")
+    previous_threads = torch.get_num_threads()
+    torch.set_num_threads(1)
+    try:
+        seed_everything(42)
+        model = TinySiameseChangeClassifier(["synthetic_dark", "synthetic_bright"], dropout=0)
+        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
+        before, after = torch.zeros(2, 3, 17, 23), torch.ones(2, 3, 17, 23)
+        mask = torch.ones(2, 1, 17, 23)
+        targets = (torch.zeros(2, dtype=torch.long), torch.ones(2, dtype=torch.long))
+        losses = []
+        for _ in range(steps):
+            optimizer.zero_grad(set_to_none=True)
+            loss, _ = semantic_loss(model(before, after, mask), *targets)
+            loss.backward()
+            if any(
+                p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()
+            ):
+                raise RuntimeError("Synthetic backward produced nonfinite gradients.")
+            optimizer.step()
+            losses.append(float(loss.detach()))
+        model.eval()
+        with torch.inference_mode():
+            output = model(before, after, mask)
+            joint = (
+                (
+                    (output["before_logits"].argmax(1) == targets[0])
+                    & (output["after_logits"].argmax(1) == targets[1])
+                )
+                .float()
+                .mean()
+            )
+        return {
+            "evidence": "synthetic implementation check only",
+            "seed": 42,
+            "steps": steps,
+            "first_loss": losses[0],
+            "last_loss": losses[-1],
+            "eval_joint_accuracy": float(joint),
+            **count_model_parameters(model),
+        }
+    finally:
+        torch.set_num_threads(previous_threads)
+
+
+def main() -> None:
+    """Run only the operation explicitly selected by the caller."""
+    parser = argparse.ArgumentParser(description=__doc__)
+    commands = parser.add_subparsers(dest="command", required=True)
+    for name in ("validate", "prepare"):
+        sub = commands.add_parser(name)
+        sub.add_argument("--config", required=True)
+        sub.add_argument("--manifest", required=True)
+        if name == "prepare":
+            sub.add_argument("--output", required=True)
+    evaluation = commands.add_parser("evaluate")
+    evaluation.add_argument("--checkpoint", required=True)
+    evaluation.add_argument("--prepared", required=True)
+    evaluation.add_argument("--split", choices=("val", "test"), required=True)
+    evaluation.add_argument("--device", default="cpu")
+    evaluation.add_argument("--output", required=True)
+    inference = commands.add_parser("infer")
+    inference.add_argument("--checkpoint", required=True)
+    for key in ("before", "after", "change", "output"):
+        inference.add_argument(f"--{key}", required=True)
+    inference.add_argument("--device", default="cpu")
+    inference.add_argument("--mask-source", choices=("ground_truth", "predicted"), required=True)
+    inference.add_argument("--detector-identity")
+    smoke = commands.add_parser("smoke")
+    smoke.add_argument("--steps", type=int, default=60)
+    args = parser.parse_args()
+    if args.command in ("evaluate", "infer"):
+        if Path(args.output).exists():
+            raise FileExistsError(f"Output already exists: {args.output}")
+    if args.command in ("validate", "prepare"):
+        result = prepare_dataset(
+            load_config(args.config),
+            args.manifest,
+            args.output if args.command == "prepare" else None,
+        )
+        print(
+            json.dumps({"coverage": result["coverage"], "protocol": result["protocol"]}, indent=2)
+        )
+        return
+    if args.command == "smoke":
+        result = synthetic_smoke(args.steps)
+    elif args.command == "evaluate":
+        result = evaluate_classifier(
+            args.checkpoint, args.prepared, args.split, torch.device(args.device)
+        )
+    elif args.command == "infer":
+        result = predict_checkpoint(
+            args.checkpoint,
+            read_rgb(args.before),
+            read_rgb(args.after),
+            read_map(args.change).unsqueeze(0),
+            torch.device(args.device),
+            inference_mask_source=args.mask_source,
+            detector={"identity": args.detector_identity} if args.detector_identity else None,
+        )
+    if args.command != "smoke":
+        output = Path(args.output)
+        output.parent.mkdir(parents=True, exist_ok=True)
+        write_json(output, result)
+    print(json.dumps(result, indent=2))
+
+
+if __name__ == "__main__":
+    main()
diff --git a/src/satchange/semantic/benchmark.py b/src/satchange/semantic/benchmark.py
new file mode 100644
index 0000000..e4667fb
--- /dev/null
+++ b/src/satchange/semantic/benchmark.py
@@ -0,0 +1,316 @@
+"""Semantic evaluation/export adapter for the shared run_final_benchmark entry point."""
+
+import math
+import shutil
+import statistics
+import time
+from copy import deepcopy
+from pathlib import Path
+from typing import Any
+
+import torch
+
+from satchange.execution import precision_context
+from satchange.hpo_backend import SatelliteBackend
+from satchange.hpo_core import family
+from satchange.models import count_model_parameters
+from satchange.models.factory import SEMANTIC_TASK
+from satchange.semantic.data import SemanticROIDataset, file_digest
+from satchange.semantic.inference import evaluate_classifier, load_classifier
+from satchange.training import _save_checkpoint
+
+_METRICS = (
+    "loss",
+    "selection_score",
+    "before_macro_f1",
+    "after_macro_f1",
+    "before_accuracy",
+    "after_accuracy",
+    "joint_accuracy",
+)
+
+
+def _speed(
+    checkpoint: Path,
+    sample: dict[str, Any],
+    weights: Path,
+    device: torch.device,
+    precision: str,
+    warmup: int,
+    repetitions: int,
+) -> dict[str, Any]:
+    """Time one native ROI pair, excluding proposal extraction, I/O and device transfer."""
+    from satchange.final_benchmark import _release
+
+    _release(device)
+    cuda = device.type == "cuda"
+    baseline = torch.cuda.memory_allocated(device) if cuda else 0
+    if cuda:
+        torch.cuda.reset_peak_memory_stats(device)
+    model, state = load_classifier(checkpoint, device)
+    torch.backends.cuda.matmul.allow_tf32 = state["config"]["tf32"]
+    torch.backends.cudnn.allow_tf32 = state["config"]["tf32"]
+    values = [sample[key].unsqueeze(0).to(device) for key in ("before", "after", "mask")]
+    counts = count_model_parameters(model)
+    times = []
+    try:
+        with torch.inference_mode(), precision_context(device, precision):
+            for _ in range(warmup):
+                model(*values)
+            for _ in range(repetitions):
+                if cuda:
+                    torch.cuda.synchronize(device)
+                start = time.perf_counter()
+                model(*values)
+                if cuda:
+                    torch.cuda.synchronize(device)
+                times.append((time.perf_counter() - start) * 1000)
+        mean = statistics.mean(times)
+        return {
+            **counts,
+            "deployment_seed": state["config"]["seed"],
+            "checkpoint_mb": checkpoint.stat().st_size / 1024**2,
+            "weights_only_mb": weights.stat().st_size / 1024**2,
+            "latency_ms_per_pair": mean,
+            "pairs_per_second": 1000 / mean,
+            "latency_median_ms": statistics.median(times),
+            "latency_samples_ms": times,
+            "peak_inference_vram_mb": (
+                (torch.cuda.max_memory_allocated(device) - baseline) / 1024**2 if cuda else None
+            ),
+            "device": str(device),
+            "pytorch_version": str(torch.__version__),
+            "cuda_version": torch.version.cuda,
+            "gpu_name": torch.cuda.get_device_name(device) if cuda else None,
+            "protocol": {
+                "task": SEMANTIC_TASK,
+                "precision": precision,
+                "tf32": state["config"]["tf32"],
+                "batch_size": 1,
+                "input_shape_per_image": list(values[0].shape),
+                "mask_shape": list(values[2].shape),
+                "warmup": warmup,
+                "repetitions": repetitions,
+                "input": "first eligible native validation ROI",
+                "timing": "synchronized classifier forward only" if cuda else "CPU forward only",
+                "includes_detector_roi_extraction_loading_h2d": False,
+                "scope": "classifier only; variable ROI sizes affect latency",
+                "memory_unit": "MiB; existing process allocations subtracted",
+            },
+        }
+    finally:
+        del model, state, values
+        _release(device)
+
+
+def finalize_semantic(
+    *,
+    paths: dict[int, Path],
+    prepared: Path,
+    output: Path,
+    variant: str,
+    root: Path,
+    training_results: Any,
+    metadata: Any,
+    device: torch.device,
+    precision: str,
+    evaluation_batch_size: int,
+    num_workers: int,
+    deployment_seed: int | None,
+    warmup: int,
+    repetitions: int,
+    strict_validation: bool,
+    validation_atol: float,
+    overwrite: bool,
+) -> dict[str, Any]:
+    """Validate all seeds, freeze deployment, evaluate untouched test, then publish a bundle."""
+    from satchange.final_benchmark import (
+        _output_transaction,
+        _read_mapping,
+        _runtime,
+        _seed,
+        _write_json,
+    )
+
+    prepared = prepared / "prepared.json" if prepared.is_dir() else prepared
+    if not prepared.is_file():
+        raise FileNotFoundError(prepared)
+    training = _read_mapping(training_results, root)
+    bundle = _read_mapping(metadata, root)
+    if training and (not training.get("complete") or set(training.get("seeds", [])) != set(paths)):
+        raise ValueError("Final training summary must be complete and cover every input seed.")
+    rows = {int(row["parameters"]["seed"]): row for row in training.get("results", [])}
+    if rows and set(rows) != set(paths):
+        raise ValueError("Final training rows differ from the supplied checkpoint seeds.")
+    reports: dict[str, Any] = {}
+    notes = []
+    with _output_transaction(output, overwrite) as (stage, backup), _runtime(device):
+        states = {}
+        copies = {}
+        hashes = set()
+        definition = None
+        for seed, source in paths.items():
+            copy = stage / f"seed_{seed}_best.pt"
+            shutil.copy2(source, copy)
+            model, state = load_classifier(copy)
+            del model
+            if state["config"]["seed"] != seed:
+                raise ValueError("Checkpoint seed differs from its mapping key.")
+            normalized = family(SatelliteBackend().normalize(state["config"]))
+            if definition is not None and definition != normalized:
+                raise ValueError(
+                    "Semantic seeds must have the same data/model/training definition."
+                )
+            definition = normalized
+            digest = file_digest(copy)
+            if digest in hashes:
+                raise ValueError("Identical checkpoint bytes assigned to different seeds.")
+            hashes.add(digest)
+            if seed in rows:
+                row = rows[seed]
+                if (
+                    row["epoch"] != training["target_epoch"]
+                    or state.get("training_completed_epoch") != row["epoch"]
+                    or state.get("training_budget_epochs") != training["target_epoch"]
+                    or Path(row["best_checkpoint"]).resolve() != source.resolve()
+                ):
+                    raise ValueError(
+                        "Checkpoint completion/path disagrees with final training summary."
+                    )
+                if family(SatelliteBackend().normalize(row["parameters"])) != normalized:
+                    raise ValueError("Final training parameters disagree with the checkpoint.")
+            states[seed], copies[seed] = state, copy
+        first = states[next(iter(states))]
+        dataset = SemanticROIDataset(prepared, "val", first["config"]["semantic_config"])
+        if not dataset.records:
+            raise ValueError("Validation contains no eligible semantic ROIs.")
+        selected = None
+        selection = ""
+        for split, filename in (("val", "validation_results"), ("test", "final_test_results")):
+            values = []
+            for seed, copy in copies.items():
+                result = evaluate_classifier(
+                    copy,
+                    prepared,
+                    split,
+                    device,
+                    precision=precision,
+                    evaluation_batch_size=evaluation_batch_size,
+                    num_workers=num_workers,
+                )
+                if split == "val":
+                    saved = states[seed]["best_metrics"]
+                    for metric in _METRICS:
+                        expected = saved["validation_loss" if metric == "loss" else metric]
+                        delta = float(result[metric]) - float(expected)
+                        result[f"{metric}_delta"] = delta
+                        if not math.isfinite(delta) or abs(delta) > validation_atol:
+                            message = f"Seed {seed}: validation {metric} differs by {delta:+.6f}."
+                            if strict_validation:
+                                raise ValueError(message + " No test evaluation was started.")
+                            notes.append(message)
+                values.append(result)
+            label = "validation" if split == "val" else "test"
+            report = {
+                "task": SEMANTIC_TASK,
+                "variant": variant,
+                "seeds": list(copies),
+                "results": values,
+                "std_ddof": 1,
+                "std_type": "sample",
+                "metric_aggregation": "global per-date ROI confusion counts; mean across seeds",
+                "evaluation_batch_size": evaluation_batch_size,
+                "num_workers": num_workers,
+                "precision": precision,
+                "number_of_rois": values[0]["samples"],
+                "scope": "eligible ROI semantic classification; excludes detector accuracy",
+            }
+            for metric in _METRICS:
+                scores = [float(value[metric]) for value in values]
+                report[f"{label}_{metric}_mean"] = statistics.mean(scores)
+                report[f"{label}_{metric}_std"] = (
+                    statistics.stdev(scores) if len(scores) > 1 else None
+                )
+            reports[filename] = report
+            _write_json(stage / f"{filename}.json", report)
+            if split == "val":
+                choice = (
+                    deployment_seed
+                    if deployment_seed is not None
+                    else bundle.get("deployment_seed")
+                )
+                if choice is None:
+                    choice = max(values, key=lambda row: (row["selection_score"], -row["seed"]))[
+                        "seed"
+                    ]
+                    selection = "highest validation mean per-date macro F1; ties: smallest seed"
+                else:
+                    choice = _seed(choice)
+                    selection = "explicit/saved deployment seed fixed before test"
+                if choice not in copies:
+                    raise ValueError("Deployment seed is absent from input checkpoints.")
+                selected = choice
+                _write_json(
+                    stage / "deployment_choice.json",
+                    {"seed": choice, "selection": selection, "test_used": False},
+                )
+        assert selected is not None
+        weights = stage / "deployment_weights.pt"
+        _save_checkpoint(
+            weights,
+            {
+                key: tensor.detach().cpu().float()
+                if tensor.is_floating_point()
+                else tensor.detach().cpu()
+                for key, tensor in states[selected]["model_state_dict"].items()
+            },
+        )
+        reports["benchmark"] = _speed(
+            copies[selected], dataset[0], weights, device, precision, warmup, repetitions
+        )
+        reports["training_results"] = (
+            deepcopy(training)
+            if training
+            else {
+                "results": [
+                    {"seed": seed, "best_epoch": state["epoch"], "parameters": state["config"]}
+                    for seed, state in states.items()
+                ]
+            }
+        )
+        reports["metadata"] = {
+            "schema_version": 1,
+            "task": SEMANTIC_TASK,
+            "model_name": first["model_name"],
+            "model_parameters": first["config"]["model_parameters"],
+            "training_parameters": definition,
+            "deployment_seed": selected,
+            "deployment_selection": selection,
+            "deployment_best_epoch": states[selected]["epoch"],
+            "dataset_root": str(prepared),
+            "prepared_identity": file_digest(prepared),
+            "protocol": dataset.document["protocol"],
+            "evaluation_complete": True,
+            "evaluation_precision": precision,
+            "std_ddof": 1,
+            "validation_check": {
+                "strict": strict_validation,
+                "absolute_tolerance": validation_atol,
+            },
+            "seeds": {
+                str(seed): {
+                    "checkpoint_file": copies[seed].name,
+                    "checkpoint_sha256": file_digest(copies[seed]),
+                    "source_checkpoint": str(paths[seed]),
+                    "best_epoch": state["epoch"],
+                }
+                for seed, state in states.items()
+            },
+            "deployment_weights_format": "plain CPU FP32 state_dict; no optimizer",
+            "scope": "semantic accuracy on eligible ROIs; detector performance is not evaluated",
+            "notes": notes,
+            "source_metadata": bundle,
+        }
+        for name in ("benchmark", "training_results", "metadata"):
+            _write_json(stage / f"{name}.json", reports[name])
+    return {"output_dir": str(output), "backup_dir": str(backup) if backup else None, **reports}
diff --git a/src/satchange/semantic/data.py b/src/satchange/semantic/data.py
new file mode 100644
index 0000000..9446d5e
--- /dev/null
+++ b/src/satchange/semantic/data.py
@@ -0,0 +1,455 @@
+"""Manifest-based semantic supervision; binary LEVIR masks cannot supply these labels."""
+
+import hashlib
+import json
+import os
+from collections import Counter
+from dataclasses import asdict, replace
+from pathlib import Path
+from typing import Any
+
+import numpy as np
+import torch
+from PIL import Image
+from torch.utils.data import Dataset
+
+from satchange.data.transforms import RandomFlipRotate
+from satchange.models.tiny_siamese_change_classifier import validate_classes
+from satchange.semantic.roi import ROISettings, extract_rois
+
+NORMALIZATION = "RGB uint8 / 255; native resolution"
+
+
+def write_json(path: Path, value: Any) -> None:
+    """Save readable JSON with finite numeric values."""
+    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
+
+
+def file_digest(path: Path) -> str:
+    """Hash source content so a resume cannot silently use changed annotations/images."""
+    digest = hashlib.sha256()
+    with path.open("rb") as stream:
+        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
+            digest.update(chunk)
+    return digest.hexdigest()
+
+
+def read_rgb(path: str | Path) -> torch.Tensor:
+    """Load an actual RGB uint8 image as native [3,H,W] float32 in [0,1]."""
+    with Image.open(path) as image:
+        if image.mode != "RGB":
+            raise ValueError(f"Expected RGB imagery, not {image.mode}: {path}")
+        pixels = np.array(image, dtype=np.uint8, copy=True)
+    return torch.from_numpy(pixels).permute(2, 0, 1).float().div_(255).contiguous()
+
+
+def read_map(path: str | Path, *, semantic: bool = False) -> torch.Tensor:
+    """Read a non-pickled .npy [H,W] map, preserving integer semantic IDs."""
+    if Path(path).suffix.lower() != ".npy":
+        raise ValueError(f"Maps must be .npy [H,W] arrays: {path}")
+    array = np.load(path, allow_pickle=False)
+    if array.ndim != 2:
+        raise ValueError(f"Map must have shape [H,W]: {path}")
+    if semantic and not np.issubdtype(array.dtype, np.integer):
+        raise ValueError(f"Semantic maps require integer IDs: {path}")
+    if not semantic and not (
+        np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_)
+    ):
+        raise ValueError(f"Change maps require numeric probabilities: {path}")
+    if not semantic and (
+        np.iscomplexobj(array)
+        or not np.isfinite(array).all()
+        or (array < 0).any()
+        or (array > 1).any()
+    ):
+        raise ValueError(f"Change probabilities must be finite and in [0,1]: {path}")
+    if semantic and np.issubdtype(array.dtype, np.unsignedinteger):
+        if array.size and array.max() > np.iinfo(np.int64).max:
+            raise ValueError(f"Semantic IDs must fit signed int64: {path}")
+    # int64 also allows explicit negative ignore IDs without binarizing labels.
+    return torch.from_numpy(array.astype(np.int64 if semantic else np.float32, copy=True))
+
+
+def data_protocol(config: dict[str, Any]) -> dict[str, Any]:
+    """Resolve the single explicit taxonomy, ROI, annotation, and mask-source contract."""
+    classes = validate_classes(config.get("classes", []))
+    mapping = config.get("class_ids")
+    if not isinstance(mapping, dict) or not mapping:
+        raise ValueError("class_ids must explicitly map raw integer IDs to class names.")
+    ids: dict[str, int] = {}
+    for raw, name in mapping.items():
+        if str(int(raw)) != str(raw) or name not in classes:
+            raise ValueError("class_ids keys must be integer IDs and values declared class names.")
+        ids[str(raw)] = classes.index(name)
+    if set(ids.values()) != set(range(len(classes))):
+        raise ValueError("Every declared class needs an explicit raw-ID mapping.")
+    ignore_ids = config.get("ignore_ids")
+    if not isinstance(ignore_ids, list) or any(type(value) is not int for value in ignore_ids):
+        raise ValueError("ignore_ids must explicitly list integer unlabeled/background IDs.")
+    if set(map(str, ignore_ids)) & ids.keys():
+        raise ValueError("An ID cannot be both a learned class and ignored.")
+    source = config.get("mask_source")
+    if source not in ("ground_truth", "predicted"):
+        raise ValueError("mask_source must be 'ground_truth' or 'predicted'.")
+    settings = ROISettings(**config.get("roi", {}))
+    policy = {"min_transition_purity": 0.8, "min_valid_fraction": 0.8}
+    supplied = config.get("label_policy", {})
+    if set(supplied) - policy.keys():
+        raise ValueError("Unknown label_policy key.")
+    policy.update(supplied)
+    if any(not 0 < value <= 1 for value in policy.values()):
+        raise ValueError("Label purity and valid-fraction thresholds must be in (0,1].")
+    detector = config.get("detector")
+    if source == "predicted":
+        if not isinstance(detector, dict) or not detector.get("identity"):
+            raise ValueError(
+                "Predicted masks need detector identity and training_groups provenance."
+            )
+        groups = detector.get("training_groups")
+        if not isinstance(groups, list) or any(not isinstance(group, str) for group in groups):
+            raise ValueError(
+                "detector.training_groups must list the frozen detector's fitting groups."
+            )
+    elif detector is not None:
+        raise ValueError("ground_truth experiments must set detector: null.")
+    return {
+        "classes": list(classes),
+        "class_ids": ids,
+        "ignore_ids": sorted(set(ignore_ids)),
+        "roi": asdict(settings),
+        "label_policy": policy,
+        "mask_source": source,
+        "detector": detector,
+        "normalization": NORMALIZATION,
+    }
+
+
+def read_manifest(path: str | Path, protocol: dict[str, Any]) -> list[dict[str, Any]]:
+    """Validate scene splits, unique IDs, source files, and detector fitting-group provenance."""
+    path = Path(path).resolve()
+    if not path.is_file():
+        raise FileNotFoundError(
+            f"Semantic manifest missing: {path}. Supply real before/after semantic annotations; "
+            "LEVIR-CD binary labels are insufficient. See docs/semantic_classifier.md."
+        )
+    document = json.loads(path.read_text(encoding="utf-8"))
+    scenes = document.get("scenes") if isinstance(document, dict) else None
+    if not isinstance(scenes, list) or not scenes:
+        raise ValueError("Manifest must contain a nonempty 'scenes' list.")
+    group_splits: dict[str, str] = {}
+    ids: set[str] = set()
+    result = []
+    pair_splits: dict[tuple[str, ...], str] = {}
+    content_splits: dict[tuple[str, ...], str] = {}
+    rgb_hashes: dict[str, str] = {}
+    fields = ("before", "after", "before_semantic", "after_semantic", "change")
+    for scene in scenes:
+        required = set(fields + ("sample_id", "group_id", "split"))
+        if not isinstance(scene, dict) or required - scene.keys():
+            raise ValueError("Each scene needs RGB, both semantic labels, change, split, and IDs.")
+        sample_id, group, split = (scene[key] for key in ("sample_id", "group_id", "split"))
+        if not all(isinstance(value, str) and value for value in (sample_id, group, split)):
+            raise ValueError("sample_id, group_id, and split must be nonempty strings.")
+        if sample_id in ids or split not in ("train", "val", "test"):
+            raise ValueError("sample_id must be unique and split must be train, val, or test.")
+        if group in group_splits and group_splits[group] != split:
+            raise ValueError(f"Cross-split group leakage: {group}")
+        if split != "train" and group in (protocol.get("detector") or {}).get(
+            "training_groups", []
+        ):
+            raise ValueError(f"Detector fitting-group leakage into {split}: {group}")
+        ids.add(sample_id)
+        group_splits[group] = split
+        resolved = dict(scene)
+        for field in fields:
+            if not isinstance(scene[field], str) or not scene[field]:
+                raise ValueError(
+                    f"{sample_id}: missing {field} path; no fabricated-label fallback."
+                )
+            source = (path.parent / scene[field]).resolve()
+            if not source.is_file():
+                raise FileNotFoundError(f"{sample_id}: missing {field}: {source}")
+            resolved[field] = str(source)
+        pair = tuple(
+            sorted((os.path.normcase(resolved["before"]), os.path.normcase(resolved["after"])))
+        )
+        for source in (resolved["before"], resolved["after"]):
+            if source not in rgb_hashes:
+                rgb_hashes[source] = file_digest(Path(source))
+        content = tuple(sorted((rgb_hashes[resolved["before"]], rgb_hashes[resolved["after"]])))
+        for identity, seen in ((pair, pair_splits), (content, content_splits)):
+            if identity in seen and seen[identity] != split:
+                raise ValueError(
+                    f"Cross-split RGB pair leakage (including temporal reversal): {sample_id}"
+                )
+            seen[identity] = split
+        resolved["rgb_hashes"] = {key: rgb_hashes[resolved[key]] for key in ("before", "after")}
+        result.append(resolved)
+    return result
+
+
+def dominant_transition(
+    before: torch.Tensor,
+    after: torch.Tensor,
+    support: torch.Tensor,
+    protocol: dict[str, Any],
+) -> dict[str, Any]:
+    """Select an observed joint label pair only within component support; report exclusions."""
+    if before.shape != after.shape or support.shape != before.shape:
+        raise ValueError("Semantic maps and component support must align as [H,W].")
+    if before.ndim != 2 or support.dtype != torch.bool:
+        raise ValueError("Component support must be boolean [H,W].")
+    integer_types = (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64)
+    if before.dtype not in integer_types or after.dtype not in integer_types:
+        raise ValueError("Semantic maps require integer class IDs.")
+    mapped = []
+    for labels in (before, after):
+        values = torch.full_like(labels, -1, dtype=torch.long)
+        known = torch.zeros_like(labels, dtype=torch.bool)
+        for raw, target in protocol["class_ids"].items():
+            selected = labels == int(raw)
+            values[selected] = target
+            known |= selected
+        for raw in protocol["ignore_ids"]:
+            known |= labels == raw
+        if not known.all():
+            raise ValueError(
+                "Semantic map contains undeclared class IDs; set class_ids/ignore_ids."
+            )
+        mapped.append(values)
+    valid = support.bool() & (mapped[0] >= 0) & (mapped[1] >= 0)
+    area, valid_count = int(support.sum()), int(valid.sum())
+    fraction = valid_count / area if area else 0.0
+    quality: dict[str, Any] = {"valid_pixels": valid_count, "valid_fraction": fraction}
+    if not valid_count or fraction < protocol["label_policy"]["min_valid_fraction"]:
+        return {**quality, "reason": "insufficient_annotation"}
+    pairs = torch.stack((mapped[0][valid], mapped[1][valid]), dim=1)
+    unique, counts = torch.unique(pairs, dim=0, return_counts=True)
+    largest = int(counts.max())
+    purity = largest / valid_count
+    quality["transition_purity"] = purity
+    if int((counts == largest).sum()) != 1:
+        return {**quality, "reason": "tied_transition"}
+    if purity < protocol["label_policy"]["min_transition_purity"]:
+        return {**quality, "reason": "mixed_transition"}
+    pair = unique[int(counts.argmax())].tolist()
+    return {**quality, "reason": None, "before_label": pair[0], "after_label": pair[1]}
+
+
+def prepare_dataset(
+    config: dict[str, Any],
+    manifest: str | Path,
+    destination: str | Path | None = None,
+) -> dict[str, Any]:
+    """Validate real sources and generate ROI records/masks, without duplicating RGB imagery.
+
+    destination=None validates and reports without writing. An existing output is never replaced.
+    Coverage is annotation-eligible ROI coverage, not detection matching or end-to-end accuracy.
+    """
+    protocol = data_protocol(config)
+    scenes = read_manifest(manifest, protocol)
+    settings = ROISettings(**protocol["roi"])
+    output = Path(destination).resolve() if destination is not None else None
+    if output is not None and output.exists():
+        raise FileExistsError(f"Preparation output already exists: {output}")
+    # A failed preparation never produces a completed prepared.json at the requested path.
+    final_output = output
+    if output is not None:
+        from uuid import uuid4
+
+        output = output.with_name(f".{output.name}.incomplete-{uuid4().hex}")
+    records = []
+    reports: dict[str, Any] = {
+        split: {
+            "proposed": 0,
+            "retained": 0,
+            "exclusions": Counter(),
+            "before_distribution": Counter(),
+            "after_distribution": Counter(),
+            "transition_distribution": Counter(),
+            "scenes_without_components": 0,
+            "crop_sizes": [],
+        }
+        for split in ("train", "val", "test")
+    }
+    hashes = {}
+    for scene in scenes:
+        for key in ("before", "after", "before_semantic", "after_semantic", "change"):
+            source = scene[key]
+            if source not in hashes:
+                hashes[source] = (
+                    scene["rgb_hashes"][key]
+                    if key in ("before", "after")
+                    else file_digest(Path(source))
+                )
+        before, after = read_rgb(scene["before"]), read_rgb(scene["after"])
+        probabilities = read_map(scene["change"]).unsqueeze(0)
+        if protocol["mask_source"] == "ground_truth":
+            if not ((probabilities == 0) | (probabilities == 1)).all():
+                raise ValueError("ground_truth change maps must contain binary 0/1 support.")
+        semantic_before = read_map(scene["before_semantic"], semantic=True)
+        semantic_after = read_map(scene["after_semantic"], semantic=True)
+        if semantic_before.shape != before.shape[-2:] or semantic_after.shape != before.shape[-2:]:
+            raise ValueError(f"{scene['sample_id']}: native semantic/RGB spatial mismatch.")
+        # Validate the whole taxonomy even when no component is proposed.
+        dominant_transition(
+            semantic_before,
+            semantic_after,
+            torch.zeros_like(semantic_before, dtype=torch.bool),
+            protocol,
+        )
+        rois = extract_rois(before, after, probabilities, replace(settings, min_component_area=1))
+        report = reports[scene["split"]]
+        report["scenes_without_components"] += int(not rois)
+        for roi in rois:
+            report["proposed"] += 1
+            x0, y0, x1, y1 = roi.crop_bbox
+            quality = dominant_transition(
+                semantic_before[y0:y1, x0:x1],
+                semantic_after[y0:y1, x0:x1],
+                roi.support[0],
+                protocol,
+            )
+            reason = (
+                "small_component" if roi.area < settings.min_component_area else quality["reason"]
+            )
+            if reason:
+                report["exclusions"][reason] += 1
+                records.append(
+                    {
+                        **scene,
+                        "component_id": roi.component_id,
+                        "crop_bbox": list(roi.crop_bbox),
+                        "area": roi.area,
+                        "component_bbox": list(roi.component_bbox),
+                        "retained": False,
+                        "minimum_extent_limited": roi.minimum_extent_limited,
+                        "quality": {**quality, "reason": reason},
+                    }
+                )
+                continue
+            mask_name = f"masks/roi_{len(records):08d}.npz"
+            if output is not None:
+                (output / "masks").mkdir(parents=True, exist_ok=True)
+                np.savez_compressed(
+                    output / mask_name,
+                    probabilities=roi.probabilities.numpy(),
+                    support=roi.support.numpy(),
+                )
+            record = {
+                **scene,
+                "component_id": roi.component_id,
+                "crop_bbox": list(roi.crop_bbox),
+                "component_bbox": list(roi.component_bbox),
+                "area": roi.area,
+                "mask_file": mask_name,
+                "retained": True,
+                "quality": quality,
+                "minimum_extent_limited": roi.minimum_extent_limited,
+                "before_label": quality["before_label"],
+                "after_label": quality["after_label"],
+            }
+            records.append(record)
+            report["retained"] += 1
+            report["crop_sizes"].append([y1 - y0, x1 - x0])
+            left, right = (
+                protocol["classes"][record[key]] for key in ("before_label", "after_label")
+            )
+            report["before_distribution"][left] += 1
+            report["after_distribution"][right] += 1
+            report["transition_distribution"][f"{left} -> {right}"] += 1
+    for report in reports.values():
+        report["retained_fraction"] = (
+            report["retained"] / report["proposed"] if report["proposed"] else 0.0
+        )
+    provenance = {
+        "manifest": str(Path(manifest).resolve()),
+        "scenes": scenes,
+        "source_hashes": hashes,
+    }
+    document = {
+        "schema_version": 1,
+        "protocol": protocol,
+        "provenance": provenance,
+        "records": records,
+        "coverage": reports,
+        "coverage_definition": "annotation-eligible / proposed connected components",
+    }
+    if output is not None:
+        output.mkdir(parents=True, exist_ok=True)
+        document["mask_hashes"] = {
+            record["mask_file"]: file_digest(output / record["mask_file"])
+            for record in records
+            if record["retained"]
+        }
+        write_json(output / "prepared.json", document)
+        assert final_output is not None
+        output.rename(final_output)
+    return document
+
+
+class SemanticROIDataset(Dataset):
+    """Load variable-size real RGB crops and component weights from prepared records."""
+
+    def __init__(
+        self,
+        prepared: str | Path,
+        split: str,
+        config: dict[str, Any],
+        *,
+        augment: bool = False,
+    ) -> None:
+        self.path = Path(prepared).resolve()
+        if not self.path.is_file():
+            raise FileNotFoundError(
+                f"Prepared semantic data missing: {self.path}. Run semantic prepare with "
+                "real before/after semantic annotations; no synthetic-label fallback is provided."
+            )
+        self.document = json.loads(self.path.read_text(encoding="utf-8"))
+        if self.document.get("schema_version") != 1:
+            raise ValueError("Unsupported prepared-data schema.")
+        if self.document["protocol"] != data_protocol(config):
+            raise ValueError(
+                "Prepared data protocol differs from configuration; prepare a new dataset."
+            )
+        if split not in ("train", "val", "test") or (augment and split != "train"):
+            raise ValueError("Use train/val/test and augmentation only on train.")
+        self.records = [
+            row for row in self.document["records"] if row["retained"] and row["split"] == split
+        ]
+        if not self.records:
+            raise ValueError(f"No eligible semantic ROIs in {split}; supply annotated real data.")
+        self.transform = RandomFlipRotate() if augment else None
+        self.split = split
+
+    def verify_sources(self) -> None:
+        """Check input and generated-mask identities once before a run/evaluation."""
+        for source, expected in self.document["provenance"]["source_hashes"].items():
+            if file_digest(Path(source)) != expected:
+                raise ValueError(f"Prepared source changed: {source}; regenerate prepared data.")
+        for source, expected in self.document["mask_hashes"].items():
+            if file_digest(self.path.parent / source) != expected:
+                raise ValueError(f"Prepared ROI mask changed: {source}")
+
+    def __len__(self) -> int:
+        return len(self.records)
+
+    def __getitem__(self, index: int) -> dict[str, Any]:
+        row = self.records[index]
+        x0, y0, x1, y1 = row["crop_bbox"]
+        with np.load(self.path.parent / row["mask_file"], allow_pickle=False) as archive:
+            mask = torch.from_numpy((archive["probabilities"] * archive["support"]).copy()).float()
+        sample = {
+            "before": read_rgb(row["before"])[:, y0:y1, x0:x1].contiguous(),
+            "after": read_rgb(row["after"])[:, y0:y1, x0:x1].contiguous(),
+            "mask": mask.contiguous(),
+            "sample_id": row["sample_id"],
+            "before_label": row["before_label"],
+            "after_label": row["after_label"],
+        }
+        return self.transform(sample) if self.transform is not None else sample
+
+
+def collate_rois(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
+    """Preserve different native crop sizes; never resize or pad logical batches."""
+    return samples
diff --git a/src/satchange/semantic/inference.py b/src/satchange/semantic/inference.py
new file mode 100644
index 0000000..869ea98
--- /dev/null
+++ b/src/satchange/semantic/inference.py
@@ -0,0 +1,313 @@
+"""Common-checkpoint evaluation and native RGB detector/classifier composition."""
+
+from dataclasses import asdict
+from pathlib import Path
+from typing import Any
+
+import torch
+from torch import nn
+from torch.nn import functional as F
+from torch.utils.data import DataLoader
+
+from satchange.execution import precision_context, resolve_device
+from satchange.models import TinySiameseChangeClassifier
+from satchange.models.factory import SEMANTIC_TASK
+from satchange.run import reconstruct_checkpoint
+from satchange.semantic.data import SemanticROIDataset, collate_rois, data_protocol, file_digest
+from satchange.semantic.roi import (
+    DEFAULT_ROI_SETTINGS,
+    ROISettings,
+    align_whole_image_probabilities,
+    extract_rois,
+)
+from satchange.semantic.training import run_epoch, validate_parameters
+
+
+def load_classifier(
+    checkpoint: str | Path, device: str | torch.device = "cpu"
+) -> tuple[TinySiameseChangeClassifier, dict[str, Any]]:
+    """Reconstruct a trusted HPO checkpoint via the shared loader."""
+    model, state = reconstruct_checkpoint(checkpoint, resolve_device(device))
+    if state.get("task") != SEMANTIC_TASK or not isinstance(model, TinySiameseChangeClassifier):
+        raise ValueError("Expected a semantic Tiny checkpoint from project HPO.")
+    validate_parameters(state["config"])
+    if state.get("precision", state["config"]["precision"]) != state["config"]["precision"]:
+        raise ValueError("Checkpoint precision metadata is inconsistent.")
+    return model.eval(), state
+
+
+def _numerics(state: dict[str, Any], device: torch.device, override: str | None) -> str:
+    precision = state["config"]["precision"] if override is None else override
+    precision_context(device, precision)
+    torch.backends.cuda.matmul.allow_tf32 = state["config"]["tf32"]
+    torch.backends.cudnn.allow_tf32 = state["config"]["tf32"]
+    return str(precision)
+
+
+@torch.inference_mode()
+def predict_rois(
+    model: TinySiameseChangeClassifier,
+    before: torch.Tensor,
+    after: torch.Tensor,
+    probabilities: torch.Tensor,
+    settings: ROISettings = DEFAULT_ROI_SETTINGS,
+    *,
+    precision: str = "fp32",
+) -> list[dict[str, Any]]:
+    """Classify native proposals without annotations; confidence is uncalibrated softmax."""
+    model.eval()
+    device = next(model.parameters()).device
+    precision_context(device, precision)
+    result = []
+    for roi in extract_rois(before, after, probabilities, settings):
+        with precision_context(device, precision):
+            output = model(*(v.unsqueeze(0).to(device) for v in (roi.before, roi.after, roi.mask)))
+        dates = []
+        for key in ("before_logits", "after_logits"):
+            score, index = output[key].float().softmax(dim=1)[0].max(dim=0)
+            dates.append((model.classes[int(index)], float(score)))
+        result.append(
+            {
+                "component_id": roi.component_id,
+                "bbox": list(roi.component_bbox),
+                "crop_bbox": list(roi.crop_bbox),
+                "area": roi.area,
+                "minimum_extent_limited": roi.minimum_extent_limited,
+                "before_class": dates[0][0],
+                "before_score": dates[0][1],
+                "after_class": dates[1][0],
+                "after_score": dates[1][1],
+                "transition": f"{dates[0][0]} -> {dates[1][0]}",
+            }
+        )
+    return result
+
+
+def predict_checkpoint(
+    checkpoint: str | Path,
+    before: torch.Tensor,
+    after: torch.Tensor,
+    probabilities: torch.Tensor,
+    device: str | torch.device = "cpu",
+    *,
+    inference_mask_source: str | None = None,
+    detector: dict[str, Any] | None = None,
+    settings: ROISettings | None = None,
+    precision: str | None = None,
+) -> dict[str, Any]:
+    """Explicit GT-to-predicted mask shift is supported with the same weights."""
+    device = resolve_device(device)
+    model, state = load_classifier(checkpoint, device)
+    training = data_protocol(state["config"]["semantic_config"])
+    source = training["mask_source"] if inference_mask_source is None else inference_mask_source
+    if source not in ("ground_truth", "predicted"):
+        raise ValueError("Declare ground_truth or predicted inference_mask_source.")
+    if source == "predicted" and (not isinstance(detector, dict) or not detector.get("identity")):
+        raise ValueError(
+            "Predicted inference needs frozen detector identity/preprocessing provenance."
+        )
+    if source == "ground_truth" and not ((probabilities == 0) | (probabilities == 1)).all():
+        raise ValueError("GT support must be binary; explicitly declare predicted probabilities.")
+    chosen = settings or ROISettings(**training["roi"])
+    actual = _numerics(state, device, precision)
+    return {
+        "classes": list(model.classes),
+        "training_mask_source": training["mask_source"],
+        "inference_mask_source": source,
+        "training_data_identity": state["config"]["prepared_identity"],
+        "detector": detector,
+        "proposal_settings": asdict(chosen),
+        "normalization": training["normalization"],
+        "inference_precision": actual,
+        "precision_override": precision,
+        "device": str(device),
+        "confidence_definition": "uncalibrated per-date softmax scores",
+        "predictions": predict_rois(model, before, after, probabilities, chosen, precision=actual),
+    }
+
+
+def evaluate_classifier(
+    checkpoint: str | Path,
+    prepared: str | Path,
+    split: str,
+    device: str | torch.device = "cpu",
+    *,
+    evaluation_config: dict[str, Any] | None = None,
+    precision: str | None = None,
+    evaluation_batch_size: int | None = None,
+    num_workers: int = 0,
+) -> dict[str, Any]:
+    """Evaluate eligible val/test ROIs, recording both identities and explicit mask-source shift.
+
+    Alternate prepared sets require explicit evaluation_config. Shared split assignments cannot
+    change; evaluation groups/pairs cannot overlap fitting data. This is ROI-only accuracy.
+    """
+    if split not in ("val", "test"):
+        raise ValueError("Evaluation split must be val or untouched test.")
+    device = resolve_device(device)
+    model, state = load_classifier(checkpoint, device)
+    parameters = state["config"]
+    config = parameters["semantic_config"] if evaluation_config is None else evaluation_config
+    training, evaluation = data_protocol(parameters["semantic_config"]), data_protocol(config)
+    for key in ("classes", "class_ids", "ignore_ids", "normalization", "label_policy"):
+        if training[key] != evaluation[key]:
+            raise ValueError(f"Evaluation {key} differs from saved semantic protocol.")
+    identity = file_digest(Path(prepared))
+    if evaluation_config is None and identity != parameters["prepared_identity"]:
+        raise ValueError("Alternate prepared identity needs explicit evaluation_config.")
+    dataset = SemanticROIDataset(prepared, split, config)
+    dataset.verify_sources()
+    original = parameters.get("training_data", {}).get("scenes", [])
+    if not original:
+        raise ValueError("Checkpoint lacks scene/split provenance for semantic evaluation.")
+    known = {scene["sample_id"]: scene for scene in original}
+    train_groups = {s["group_id"] for s in original if s["split"] == "train"}
+    train_pairs = {
+        tuple(sorted(s["rgb_hashes"].values())) for s in original if s["split"] == "train"
+    }
+    for scene in dataset.document["provenance"]["scenes"]:
+        if scene["split"] != split:
+            continue
+        if scene["sample_id"] in known and known[scene["sample_id"]]["split"] != split:
+            raise ValueError("Evaluation changes a saved scene split.")
+        if (
+            scene["group_id"] in train_groups
+            or tuple(sorted(scene["rgb_hashes"].values())) in train_pairs
+        ):
+            raise ValueError("Evaluation RGB/group leakage from classifier training data.")
+    weights = parameters["class_weights"]
+    actual = _numerics(state, device, precision)
+    result = run_epoch(
+        model,
+        DataLoader(
+            dataset,
+            batch_size=evaluation_batch_size or parameters["batch_size"],
+            num_workers=num_workers,
+            collate_fn=collate_rois,
+        ),
+        device,
+        precision=actual,
+        weights=torch.tensor(weights, device=device) if weights is not None else None,
+    )
+    return {
+        **result,
+        "split": split,
+        "seed": parameters["seed"],
+        "training_mask_source": training["mask_source"],
+        "evaluation_mask_source": evaluation["mask_source"],
+        "training_data_identity": parameters["prepared_identity"],
+        "evaluation_data_identity": identity,
+        "training_protocol": training,
+        "evaluation_protocol": evaluation,
+        "domain_shift": training["mask_source"] != evaluation["mask_source"],
+        "evaluation_precision": actual,
+        "precision_override": precision,
+        "device": str(device),
+        "checkpoint_identity": file_digest(Path(checkpoint)),
+        "coverage": dataset.document["coverage"][split],
+        "coverage_definition": dataset.document["coverage_definition"],
+        "evaluated_epoch": state["epoch"],
+    }
+
+
+@torch.inference_mode()
+def detector_probability_map(
+    detector: nn.Module,
+    before: torch.Tensor,
+    after: torch.Tensor,
+    *,
+    input_size: tuple[int, int],
+    output_kind: str,
+    geometry: str = "whole_image_resize",
+    precision: str = "fp32",
+    normalization: str = "rgb_0_1",
+) -> torch.Tensor:
+    """Frozen detector with declared preprocessing; only RGB/255 whole-image resize is supported.
+
+    Reuse reconstruct_checkpoint for detector loading. Native classifier RGB is untouched.
+    Cropped/tiled/padded/custom-normalized inference must provide a known aligned map instead.
+    """
+    if geometry != "whole_image_resize" or normalization != "rgb_0_1":
+        raise ValueError(
+            "Unsupported detector preprocessing; supply an explicitly aligned saved map."
+        )
+    if output_kind not in ("logits", "probabilities"):
+        raise ValueError("Declare detector output_kind as logits or probabilities.")
+    if before.ndim != 3 or before.shape[0] != 3 or before.shape != after.shape:
+        raise ValueError("Detector requires aligned native RGB [3,H,W].")
+    for value in (before, after):
+        if (
+            value.dtype != torch.float32
+            or not torch.isfinite(value).all()
+            or value.min() < 0
+            or value.max() > 1
+        ):
+            raise ValueError("Detector inputs must be float32 RGB in [0,1].")
+    if len(input_size) != 2 or any(type(v) is not int or v < 1 for v in input_size):
+        raise ValueError("Declare positive whole-image detector input_size (H,W).")
+    detector.eval()
+    device = next(detector.parameters()).device
+    inputs = [
+        F.interpolate(v.unsqueeze(0), size=input_size, mode="bilinear", align_corners=False).to(
+            device
+        )
+        for v in (before, after)
+    ]
+    with precision_context(device, precision):
+        output = detector(*inputs)
+    if output.shape != (1, 1, *input_size) or not torch.isfinite(output).all():
+        raise ValueError("Detector output must be finite [1,1,input H,input W].")
+    probabilities = output.float().sigmoid() if output_kind == "logits" else output.float()
+    return align_whole_image_probabilities(
+        probabilities[0].cpu(), (before.shape[-2], before.shape[-1]), input_size
+    )
+
+
+def predict_detector_checkpoint(
+    classifier_checkpoint: str | Path,
+    detector_checkpoint: str | Path,
+    before: torch.Tensor,
+    after: torch.Tensor,
+    *,
+    input_size: tuple[int, int],
+    output_kind: str,
+    device: str | torch.device = "cpu",
+    detector_precision: str | None = None,
+    classifier_precision: str | None = None,
+) -> dict[str, Any]:
+    """Compose the shared detector loader, aligned map, and native classifier."""
+    detector, detector_state = reconstruct_checkpoint(detector_checkpoint, resolve_device(device))
+    detector_config = detector_state["config"]
+    if detector_precision is None:
+        detector_precision = detector_state.get(
+            "precision", detector_config.get("precision", "fp32")
+        )
+    detector_tf32 = bool(detector_config.get("tf32", False))
+    torch.backends.cuda.matmul.allow_tf32 = detector_tf32
+    torch.backends.cudnn.allow_tf32 = detector_tf32
+    probabilities = detector_probability_map(
+        detector,
+        before,
+        after,
+        input_size=input_size,
+        output_kind=output_kind,
+        precision=detector_precision,
+    )
+    return predict_checkpoint(
+        classifier_checkpoint,
+        before,
+        after,
+        probabilities,
+        device,
+        inference_mask_source="predicted",
+        precision=classifier_precision,
+        detector={
+            "identity": file_digest(Path(detector_checkpoint)),
+            "input_size": list(input_size),
+            "output_kind": output_kind,
+            "precision": detector_precision,
+            "tf32": detector_tf32,
+            "geometry": "whole_image_resize",
+            "normalization": "rgb_0_1",
+        },
+    )
diff --git a/src/satchange/semantic/roi.py b/src/satchange/semantic/roi.py
new file mode 100644
index 0000000..b043fc5
--- /dev/null
+++ b/src/satchange/semantic/roi.py
@@ -0,0 +1,160 @@
+"""Deterministic component crops in the original image coordinate frame."""
+
+from dataclasses import dataclass
+
+import cv2
+import numpy as np
+import torch
+from torch.nn import functional as F
+
+from satchange.models.tiny_siamese_change_classifier import validate_unit_tensor
+
+
+@dataclass(frozen=True)
+class ROISettings:
+    """Starting crop settings, not optimized values; all distances are native pixels."""
+
+    threshold: float = 0.5
+    connectivity: int = 8
+    context_pixels: int = 16
+    min_crop_size: int = 128
+    min_component_area: int = 1
+
+    def __post_init__(self) -> None:
+        if not 0 < self.threshold <= 1:
+            raise ValueError("threshold must be in (0, 1], excluding zero-weight components.")
+        if self.connectivity not in (4, 8):
+            raise ValueError("connectivity must be 4 or 8.")
+        for name, minimum in (
+            ("context_pixels", 0),
+            ("min_crop_size", 4),
+            ("min_component_area", 1),
+        ):
+            value = getattr(self, name)
+            if type(value) is not int or value < minimum:
+                raise ValueError(f"{name} must be an integer >= {minimum}.")
+
+
+@dataclass
+class ROI:
+    """Aligned crops with component and crop boxes (x0,y0,x1,y1), upper bounds exclusive."""
+
+    component_id: int
+    component_bbox: tuple[int, int, int, int]
+    crop_bbox: tuple[int, int, int, int]
+    area: int
+    before: torch.Tensor
+    after: torch.Tensor
+    probabilities: torch.Tensor
+    support: torch.Tensor
+    mask: torch.Tensor
+    minimum_extent_limited: bool
+
+
+DEFAULT_ROI_SETTINGS = ROISettings()
+
+
+def validate_scene(before: torch.Tensor, after: torch.Tensor, probabilities: torch.Tensor) -> None:
+    """Require native, aligned RGB [3,H,W] and probability [1,H,W] tensors."""
+    if before.ndim != 3 or before.shape[0] != 3 or before.shape != after.shape:
+        raise ValueError("Scene images must be aligned [3,H,W].")
+    if min(before.shape[-2:]) < 4:
+        raise ValueError("Native images are too small: the backbone requires height/width >= 4.")
+    if probabilities.shape != (1, *before.shape[-2:]):
+        raise ValueError("Probability map must already align with native RGB as [1,H,W].")
+    if before.dtype != torch.float32 or after.dtype != torch.float32:
+        raise ValueError("Native RGB must be float32 normalized to [0, 1].")
+    if before.device != after.device or before.device != probabilities.device:
+        raise ValueError("All scene tensors must use the same device.")
+    for name, value in (("before", before), ("after", after), ("probabilities", probabilities)):
+        validate_unit_tensor(value, name)
+
+
+def _extend_axis(low: int, high: int, extent: int, settings: ROISettings) -> tuple[int, int]:
+    length = min(extent, max(high - low + 2 * settings.context_pixels, settings.min_crop_size))
+    start = max(0, min((low + high - length) // 2, extent - length))
+    return start, start + length
+
+
+def extract_rois(
+    before: torch.Tensor,
+    after: torch.Tensor,
+    probabilities: torch.Tensor,
+    settings: ROISettings = DEFAULT_ROI_SETTINGS,
+) -> list[ROI]:
+    """Extract real context without RGB resizing/padding or truncating components.
+
+    Components are sorted by their first row-major pixel, independently of OpenCV's
+    label numbering. Only the selected component contributes soft pooling weights.
+    Detector output geometry must be mapped by the caller before using this API.
+    """
+    validate_scene(before, after, probabilities)
+    binary = (probabilities[0] >= settings.threshold).detach().cpu().numpy().astype(np.uint8)
+    count, labels, stats, _ = cv2.connectedComponentsWithStats(
+        binary, connectivity=settings.connectivity
+    )
+    components = []
+    unique_labels, first_pixels = np.unique(labels, return_index=True)
+    first_by_label = dict(zip(unique_labels.tolist(), first_pixels.tolist(), strict=True))
+    for label in range(1, count):
+        area = int(stats[label, cv2.CC_STAT_AREA])
+        first = first_by_label[label]
+        components.append((first, label, area))
+    height, width = before.shape[-2:]
+    results = []
+    for component_id, (_, label, area) in enumerate(sorted(components), start=1):
+        if area < settings.min_component_area:
+            continue
+        x0, y0, w, h = (int(value) for value in stats[label, :4])
+        x1, y1 = x0 + w, y0 + h
+        left, right = _extend_axis(x0, x1, width, settings)
+        top, bottom = _extend_axis(y0, y1, height, settings)
+        support = torch.from_numpy((labels[top:bottom, left:right] == label).copy())
+        support = support.unsqueeze(0).to(device=before.device)
+        probability_crop = probabilities[:, top:bottom, left:right].contiguous()
+        results.append(
+            ROI(
+                component_id=component_id,
+                component_bbox=(x0, y0, x1, y1),
+                crop_bbox=(left, top, right, bottom),
+                area=area,
+                before=before[:, top:bottom, left:right].contiguous(),
+                after=after[:, top:bottom, left:right].contiguous(),
+                probabilities=probability_crop,
+                support=support,
+                mask=(probability_crop * support).float().contiguous(),
+                minimum_extent_limited=height < settings.min_crop_size
+                or width < settings.min_crop_size,
+            )
+        )
+    return results
+
+
+def align_whole_image_probabilities(
+    probabilities: torch.Tensor,
+    original_size: tuple[int, int],
+    detector_input_size: tuple[int, int],
+) -> torch.Tensor:
+    """Map a known *whole-image resize* detector probability grid back to native (H,W).
+
+    Caller must verify that preprocessing was only a whole-image resize. Cropped,
+    padded, and tiled outputs need their own known inverse geometry; this helper
+    must not be used to guess it. RGB is never resampled by this function.
+    """
+    if probabilities.shape != (1, *detector_input_size):
+        raise ValueError("Probability grid must match the documented detector input size.")
+    if len(original_size) != 2 or any(
+        type(value) is not int or value < 4 for value in original_size
+    ):
+        raise ValueError("original_size must be native (height,width), both >= 4.")
+    validate_unit_tensor(probabilities, "probabilities")
+    return (
+        F.interpolate(
+            probabilities.float().unsqueeze(0),
+            size=original_size,
+            mode="bilinear",
+            align_corners=False,
+        )[0]
+        .clamp(0, 1)
+        .contiguous()
+    )
diff --git a/src/satchange/semantic/second.py b/src/satchange/semantic/second.py
new file mode 100644
index 0000000..c76fe0f
--- /dev/null
+++ b/src/satchange/semantic/second.py
@@ -0,0 +1,356 @@
+"""Exact SECOND conversion and versioned preparation; never starts training on import."""
+
+import json
+import random
+import stat
+import zipfile
+from copy import deepcopy
+from hashlib import sha256
+from pathlib import Path
+from typing import Any
+from uuid import uuid4
+
+import numpy as np
+from PIL import Image
+
+from satchange.semantic.data import (
+    SemanticROIDataset,
+    data_protocol,
+    file_digest,
+    prepare_dataset,
+    read_rgb,
+    write_json,
+)
+
+SECOND_SOURCE = "https://captain-whu.github.io/SCD/"
+SECOND_DOWNLOAD = "https://drive.google.com/file/d/1mN8jzCKKK27p3ODGoDgepjiRYGQpB34u/view"
+PALETTE_SOURCE = "https://raw.githubusercontent.com/yzygit1230/SCD-SAM/main/datasets/RS_ST.py"
+SECOND_CLASSES = (
+    "non-vegetated ground surface",
+    "tree",
+    "low vegetation",
+    "water",
+    "buildings",
+    "playgrounds",
+)
+# Canonical local IDs, independent of a PNG's palette indices. White is non-change.
+SECOND_COLORS = {
+    (255, 255, 255): 0,
+    (128, 128, 128): 1,
+    (0, 255, 0): 2,
+    (0, 128, 0): 3,
+    (0, 0, 255): 4,
+    (128, 0, 0): 5,
+    (255, 0, 0): 6,
+}
+
+
+def decode_label(path: str | Path, indexed_ids: dict[int, str] | None = None) -> np.ndarray:
+    """Decode exact RGB/P colors or explicitly declared scalar IDs; unknown values fail."""
+    with Image.open(path) as image:
+        if image.mode in ("RGB", "P"):
+            pixels = np.asarray(image.convert("RGB"))
+            result = np.full(pixels.shape[:2], -1, dtype=np.int16)
+            for color, identifier in SECOND_COLORS.items():
+                result[(pixels == color).all(axis=2)] = identifier
+            if (result < 0).any():
+                unknown = np.unique(pixels[result < 0], axis=0)[:5].tolist()
+                raise ValueError(f"Unknown SECOND colors {unknown}: {path}; no color guessing.")
+            return result
+        if image.mode not in ("L", "I", "I;16") or indexed_ids is None:
+            raise ValueError(f"{path}: mode {image.mode} needs explicit indexed_id_names mapping.")
+        pixels = np.asarray(image)
+        result = np.full(pixels.shape, -1, dtype=np.int16)
+        for raw, name in indexed_ids.items():
+            if name != "non-change" and name not in SECOND_CLASSES:
+                raise ValueError(f"Undeclared SECOND class {name!r}; scalar ID {raw}.")
+            result[pixels == int(raw)] = (
+                0 if name == "non-change" else SECOND_CLASSES.index(name) + 1
+            )
+        if (result < 0).any():
+            raise ValueError(f"Unknown SECOND IDs {np.unique(pixels[result < 0])[:5]}: {path}")
+        return result
+
+
+def discover_second(root: str | Path) -> dict[str, list[dict[str, str]]]:
+    """Inspect installed train/val/test leaves with im1/im2/label1/label2; match exact stems.
+
+    Unlabeled test leaves are recorded as unavailable rather than invented annotations.
+    Duplicate/extra/missing IDs and ambiguous duplicate split directories fail.
+    """
+    root = Path(root).resolve()
+    found: dict[str, list[dict[str, str]]] = {}
+    for im1 in sorted(root.rglob("im1")):
+        if not im1.is_dir():
+            continue
+        leaf = im1.parent
+        split = leaf.name.lower()
+        if split not in ("train", "val", "test"):
+            raise ValueError(f"Unrecognized SECOND split directory: {leaf}")
+        if split in found:
+            raise ValueError(f"Multiple SECOND {split} directories beneath {root}.")
+        paths = {
+            key: leaf / name
+            for key, name in (
+                ("before", "im1"),
+                ("after", "im2"),
+                ("before_semantic", "label1"),
+                ("after_semantic", "label2"),
+            )
+        }
+        if split == "test" and not any(
+            paths[k].exists() for k in ("before_semantic", "after_semantic")
+        ):
+            found[split] = []
+            continue
+        indexes: dict[str, dict[str, str]] = {}
+        for key, directory in paths.items():
+            if not directory.is_dir():
+                raise FileNotFoundError(f"SECOND missing {directory}.")
+            index = {}
+            for file in sorted(directory.iterdir()):
+                if file.suffix.lower() not in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
+                    continue
+                if file.stem in index:
+                    raise ValueError(f"Duplicate SECOND identifier {file.stem}: {directory}")
+                index[file.stem] = str(file.resolve())
+            indexes[key] = index
+        identifiers = set(indexes["before"])
+        if not identifiers or any(set(index) != identifiers for index in indexes.values()):
+            raise ValueError(f"Missing/extra or empty SECOND pairs/annotations in {leaf}.")
+        found[split] = [
+            {
+                "sample_id": f"{split}/{identifier}",
+                "group_id": identifier,
+                "split": split,
+                **{k: v[identifier] for k, v in indexes.items()},
+            }
+            for identifier in sorted(identifiers)
+        ]
+    if not found.get("train"):
+        raise FileNotFoundError(f"No labeled SECOND train/im1,im2,label1,label2 beneath {root}.")
+    return found
+
+
+def bootstrap_second(
+    root: str | Path,
+    *,
+    archive: str | Path | None = None,
+    allow_download: bool = False,
+) -> Path:
+    """Reuse a valid layout or explicitly install a local/official ZIP with safe extraction.
+
+    A private staging directory and .part download never count as completed. Local SHA256
+    is provenance, not an author-published checksum. Authentication/quota errors propagate.
+    """
+    root = Path(root).resolve()
+    try:
+        discover_second(root)
+        return root
+    except FileNotFoundError:
+        if root.exists() and any(root.iterdir()):
+            raise ValueError(
+                f"Incomplete SECOND directory: {root}; choose a new installation path."
+            ) from None
+    if archive is None and not allow_download:
+        raise FileNotFoundError(
+            f"SECOND absent at {root}; provide local_archive or enable download."
+        )
+    root.parent.mkdir(parents=True, exist_ok=True)
+    if archive is None:
+        import gdown
+
+        partial = root.parent / f"SECOND-{uuid4().hex}.zip.part"
+        try:
+            downloaded = gdown.download(
+                id="1mN8jzCKKK27p3ODGoDgepjiRYGQpB34u", output=str(partial), quiet=False
+            )
+        except Exception as error:
+            raise RuntimeError(
+                f"Official SECOND download failed (access/quota/network): {error}; "
+                "provide a local archive or installed folder."
+            ) from error
+        if downloaded is None or not zipfile.is_zipfile(partial):
+            raise RuntimeError(
+                f"SECOND download incomplete/not ZIP: {partial}; use a local archive."
+            )
+        archive = partial.with_suffix("")
+        partial.rename(archive)
+    archive = Path(archive).resolve()
+    stage = root.parent / f".{root.name}.incomplete-{uuid4().hex}"
+    stage.mkdir()
+    with zipfile.ZipFile(archive) as bundle:
+        for entry in bundle.infolist():
+            name = entry.filename.replace("\\", "/")
+            target = (stage / name).resolve()
+            if (
+                not target.is_relative_to(stage)
+                or ":" in name
+                or stat.S_ISLNK(entry.external_attr >> 16)
+            ):
+                raise ValueError(f"Unsafe SECOND archive member: {entry.filename}")
+        bundle.extractall(stage)
+    discover_second(stage)
+    write_json(
+        stage / "source.json",
+        {
+            "source": SECOND_SOURCE,
+            "download": SECOND_DOWNLOAD,
+            "archive": str(archive),
+            "local_archive_sha256": file_digest(archive),
+            "checksum_is_local": True,
+            "complete": True,
+        },
+    )
+    if root.exists():
+        root.rmdir()  # Only an empty directory, never recursive removal.
+    stage.rename(root)
+    return root
+
+
+def prepare_second(
+    root: str | Path,
+    cache: str | Path,
+    config: dict[str, Any],
+    *,
+    split_seed: int = 2099,
+    local_holdout: bool = False,
+    indexed_ids: dict[int, str] | None = None,
+    source_groups: dict[str, str] | None = None,
+    sensor_metadata: dict[str, Any] | None = None,
+) -> Path:
+    """Convert native labels once and reuse only a verified complete content/protocol cache.
+
+    Preserve provided labeled splits. Missing val/test needs explicit local_holdout; groups
+    are source metadata when supplied, otherwise filenames, with no geographic-independence
+    claim. Training seeds never alter splits. Native SECOND images must be 512x512.
+    """
+    protocol = data_protocol(config)
+    if protocol["classes"] != list(SECOND_CLASSES) or protocol["mask_source"] != "ground_truth":
+        raise ValueError(
+            "SECOND oracle preparation requires the six documented classes and GT masks."
+        )
+    expected = {str(i + 1): i for i in range(6)}
+    if protocol["class_ids"] != expected or protocol["ignore_ids"] != [0]:
+        raise ValueError("SECOND conversion uses canonical IDs 1..6 with non-change ID 0 ignored.")
+    if type(split_seed) is not int or split_seed < 0:
+        raise ValueError("split_seed must be a fixed nonnegative integer.")
+    splits = discover_second(root)
+    scenes = [deepcopy(scene) for rows in splits.values() for scene in rows]
+    for scene in scenes:
+        if source_groups is not None:
+            if scene["sample_id"] not in source_groups:
+                raise ValueError(f"Missing source group: {scene['sample_id']}")
+            scene["group_id"] = source_groups[scene["sample_id"]]
+    missing = [split for split in ("val", "test") if not splits.get(split)]
+    if missing:
+        if not local_holdout:
+            raise ValueError(
+                f"SECOND labeled {missing} unavailable; explicitly enable local_holdout."
+            )
+        groups = sorted({scene["group_id"] for scene in scenes if scene["split"] == "train"})
+        if len(groups) < len(missing) + 1:
+            raise ValueError("Too few source groups for disjoint local train/val/test.")
+        random.Random(split_seed).shuffle(groups)
+        count = max(1, int(len(groups) * 0.15))
+        assignments = {}
+        for index, split in enumerate(missing):
+            assignments.update(dict.fromkeys(groups[index * count : (index + 1) * count], split))
+        for scene in scenes:
+            if scene["split"] == "train":
+                scene["split"] = assignments.get(scene["group_id"], "train")
+    hashes = {
+        scene[key]: file_digest(Path(scene[key]))
+        for scene in scenes
+        for key in ("before", "after", "before_semantic", "after_semantic")
+    }
+    identity = {
+        "adapter_schema": 1,
+        "protocol": protocol,
+        "sources": hashes,
+        "scenes": scenes,
+        "split_seed": split_seed,
+        "local_splits": missing,
+        "indexed_ids": indexed_ids,
+        "sensor_metadata": sensor_metadata,
+        "palette_source": PALETTE_SOURCE,
+    }
+    key = sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
+    cache = Path(cache).resolve()
+    destination = cache / key
+    marker = destination / "complete.json"
+    prepared = destination / "rois/prepared.json"
+    if marker.is_file():
+        completion = json.loads(marker.read_text(encoding="utf-8"))
+        if completion.get("identity") != key or completion.get("prepared_sha256") != file_digest(
+            prepared
+        ):
+            raise ValueError(f"SECOND cache completion identity corrupted: {destination}")
+        SemanticROIDataset(prepared, "train", config).verify_sources()
+        return prepared
+    if destination.exists():
+        raise ValueError(
+            f"Incomplete SECOND cache: {destination}; preserve it and choose another cache."
+        )
+    cache.mkdir(parents=True, exist_ok=True)
+    stage = cache / f".{key}.incomplete-{uuid4().hex}"
+    (stage / "maps").mkdir(parents=True)
+    converted = []
+    for index, scene in enumerate(scenes):
+        left, right = (
+            decode_label(scene[k], indexed_ids) for k in ("before_semantic", "after_semantic")
+        )
+        if left.shape != (512, 512) or right.shape != left.shape:
+            raise ValueError(f"SECOND requires native 512x512 labels: {scene['sample_id']}")
+        if any(read_rgb(scene[k]).shape != (3, 512, 512) for k in ("before", "after")):
+            raise ValueError(f"SECOND requires aligned native 512x512 RGB: {scene['sample_id']}")
+        if not np.array_equal(left > 0, right > 0):
+            raise ValueError(f"Inconsistent before/after change support: {scene['sample_id']}")
+        row = dict(scene)
+        for field, array in (
+            ("before_semantic", left),
+            ("after_semantic", right),
+            ("change", (left > 0).astype(np.float32)),
+        ):
+            filename = f"maps/{index:06d}_{field}.npy"
+            np.save(stage / filename, array, allow_pickle=False)
+            # Final absolute paths survive the atomic staging rename.
+            row[field] = str(destination / filename)
+        converted.append(row)
+    # Build against staging paths, then publish source paths to the final immutable directory.
+    staging_scenes = [
+        {
+            **row,
+            **{
+                k: str(stage / Path(row[k]).relative_to(destination))
+                for k in ("before_semantic", "after_semantic", "change")
+            },
+        }
+        for row in converted
+    ]
+    write_json(stage / "manifest.json", {"scenes": staging_scenes})
+    doc = prepare_dataset(config, stage / "manifest.json", stage / "rois")
+    encoded = json.dumps(doc).replace(
+        str(stage).replace("\\", "\\\\"), str(destination).replace("\\", "\\\\")
+    )
+    doc = json.loads(encoded)
+    doc["second"] = {
+        **identity,
+        "source": SECOND_SOURCE,
+        "checksum_is_local": True,
+        "group_basis": "source metadata" if source_groups else "filename only",
+        "geographic_independence_verified": False,
+        "change_support": "both labels non-white, checked equal; same-class retained",
+    }
+    write_json(stage / "rois/prepared.json", doc)
+    write_json(stage / "manifest.json", {"scenes": converted})
+    write_json(
+        stage / "complete.json",
+        {
+            "identity": key,
+            "prepared_sha256": file_digest(stage / "rois/prepared.json"),
+            "complete": True,
+        },
+    )
+    stage.rename(destination)
+    return prepared
diff --git a/src/satchange/semantic/training.py b/src/satchange/semantic/training.py
new file mode 100644
index 0000000..5d1c46e
--- /dev/null
+++ b/src/satchange/semantic/training.py
@@ -0,0 +1,325 @@
+"""Semantic batch/loss/metric adapters; epoch and checkpoint orchestration stays in training.py."""
+
+import math
+from copy import deepcopy
+from pathlib import Path
+from typing import Any
+
+import torch
+from torch import nn
+from torch.nn import functional as F
+from torch.utils.data import DataLoader
+
+from satchange.execution import precision_context
+from satchange.models import TinySiameseChangeClassifier
+from satchange.models.tiny_siamese_change_classifier import validate_classes
+
+TASK = "semantic_roi_classification"
+SELECTION_RULE = "mean(before_macro_f1, after_macro_f1) on val; earlier epoch wins ties"
+
+
+class SemanticCELoss(nn.Module):
+    """Loss settings consumed by the shared engine's semantic batch adapter."""
+
+    def __init__(self, class_weights: list[float] | None = None) -> None:
+        super().__init__()
+        self.register_buffer(
+            "weights",
+            torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None,
+        )
+
+
+def hpo_parameters(config: dict[str, Any], prepared: str | Path) -> dict[str, Any]:
+    """Resolve YAML into the existing HPO schema, including content/protocol identity."""
+    from satchange.semantic.data import data_protocol, file_digest
+
+    protocol = data_protocol(config)
+    training = config["training"]
+    return {
+        "task": TASK,
+        "model_parameters": {"classes": protocol["classes"], "dropout": config.get("dropout", 0.1)},
+        "semantic_config": {
+            key: deepcopy(config[key])
+            for key in (
+                "classes",
+                "class_ids",
+                "ignore_ids",
+                "mask_source",
+                "roi",
+                "label_policy",
+                "detector",
+            )
+            if key in config
+        },
+        "prepared_identity": file_digest(Path(prepared)),
+        "class_weights": training.get("class_weights"),
+        "seed": training.get("seed", 42),
+        "batch_size": training.get("batch_size", 4),
+        "num_workers": training.get("num_workers", 0),
+        "learning_rate": training.get("learning_rate", 1e-3),
+        "weight_decay": training.get("weight_decay", 1e-4),
+        "scheduler": training.get("scheduler", "none"),
+        "precision": training.get("precision", "fp32"),
+        "tf32": training.get("tf32", False),
+    }
+
+
+def validate_parameters(parameters: dict[str, Any]) -> None:
+    """Validate semantic settings before launching GPU workers; reject segmentation-only fields."""
+    from satchange.semantic.data import data_protocol
+
+    required = {
+        "task",
+        "semantic_config",
+        "prepared_identity",
+        "model_parameters",
+        "seed",
+        "batch_size",
+        "num_workers",
+        "learning_rate",
+        "weight_decay",
+        "scheduler",
+        "class_weights",
+        "precision",
+        "tf32",
+    }
+    if required - parameters.keys():
+        raise ValueError(f"Missing semantic HPO settings: {sorted(required - parameters.keys())}")
+    forbidden = {"image_size", "positive_weight", "bce_weight", "dice_weight", "loss_weights"}
+    if forbidden & parameters.keys():
+        raise ValueError("Semantic trials cannot use RGB resize, BCE/Dice or positive_weight.")
+    protocol = data_protocol(parameters["semantic_config"])
+    model_parameters = parameters["model_parameters"]
+    if set(model_parameters) != {"classes", "dropout"}:
+        raise ValueError("Tiny model_parameters must declare classes and dropout only.")
+    classes = validate_classes(model_parameters["classes"])
+    if not 0 <= model_parameters["dropout"] < 1:
+        raise ValueError("dropout must be in [0,1).")
+    if list(classes) != protocol["classes"]:
+        raise ValueError("Model class order differs from semantic data taxonomy.")
+    for key, lower in (("seed", 0), ("batch_size", 1), ("num_workers", 0)):
+        if type(parameters[key]) is not int or parameters[key] < lower:
+            raise ValueError(f"{key} must be an integer >= {lower}.")
+    if parameters["seed"] >= 2**32:
+        raise ValueError("seed must be below 2**32.")
+    for key in ("learning_rate", "weight_decay"):
+        if not math.isfinite(parameters[key]) or parameters[key] < 0:
+            raise ValueError(f"{key} must be finite and nonnegative.")
+    if parameters["learning_rate"] == 0 or parameters["scheduler"] not in (
+        "none",
+        "reduce_on_plateau",
+    ):
+        raise ValueError("Use a positive LR and a supported project scheduler.")
+    if parameters["precision"] not in ("fp32", "bf16") or type(parameters["tf32"]) is not bool:
+        raise ValueError("Use fp32/CUDA bf16 and an explicit boolean tf32 setting.")
+    weights = parameters["class_weights"]
+    if weights is not None and (
+        not isinstance(weights, list)
+        or len(weights) != len(classes)
+        or any(not math.isfinite(w) or w <= 0 for w in weights)
+    ):
+        raise ValueError(
+            "class_weights must be positive, finite, and aligned with the saved classes."
+        )
+
+
+def create_loaders(
+    prepared: str | Path, parameters: dict[str, Any]
+) -> tuple[DataLoader, DataLoader]:
+    """Build native-size list batches before a trial; no ROI generation runs on a worker GPU."""
+    from satchange.semantic.data import SemanticROIDataset, collate_rois, file_digest
+
+    path = Path(prepared)
+    if path.is_dir():
+        path /= "prepared.json"
+    if not path.is_file():
+        raise FileNotFoundError(
+            f"Prepared semantic data missing: {path}; run SECOND preparation first."
+        )
+    if file_digest(path) != parameters["prepared_identity"]:
+        raise ValueError("Prepared training identity changed; start a new declared dataset/trial.")
+    datasets = [
+        SemanticROIDataset(path, split, parameters["semantic_config"], augment=split == "train")
+        for split in ("train", "val")
+    ]
+    datasets[0].verify_sources()
+    if "unknown" in parameters["model_parameters"]["classes"]:
+        target = parameters["model_parameters"]["classes"].index("unknown")
+        if not any(target in (r["before_label"], r["after_label"]) for r in datasets[0].records):
+            raise ValueError("A learned unknown class requires labeled training examples.")
+    loaders = [
+        DataLoader(
+            dataset,
+            batch_size=parameters["batch_size"],
+            num_workers=parameters["num_workers"],
+            collate_fn=collate_rois,
+            shuffle=index == 0,
+            generator=torch.Generator().manual_seed(parameters["seed"] + index),
+        )
+        for index, dataset in enumerate(datasets)
+    ]
+    return loaders[0], loaders[1]
+
+
+class SemanticMetrics:
+    """Two global CxC confusion matrices: rows true, columns predicted.
+
+    Undefined precision/recall/F1 are zero; macro F1 includes every declared class,
+    including classes with zero support. Joint accuracy requires both dates correct.
+    """
+
+    def __init__(self, classes: tuple[str, ...]) -> None:
+        self.classes = validate_classes(classes)
+        self.matrices = [
+            torch.zeros(len(classes), len(classes), dtype=torch.int64) for _ in range(2)
+        ]
+        self.joint_correct = 0
+        self.samples = 0
+
+    def update(
+        self,
+        logits: dict[str, torch.Tensor],
+        before: torch.Tensor,
+        after: torch.Tensor,
+    ) -> None:
+        """Accumulate counts, never per-batch F1 averages."""
+        correct = []
+        for index, (key, target) in enumerate((("before_logits", before), ("after_logits", after))):
+            values = logits[key].detach().float().cpu()
+            target = target.detach().cpu()
+            if values.shape != (target.numel(), len(self.classes)) or target.ndim != 1:
+                raise ValueError("Metrics require logits [B,C] and labels [B].")
+            if (
+                target.dtype != torch.long
+                or (target < 0).any()
+                or (target >= len(self.classes)).any()
+            ):
+                raise ValueError("Metric labels must be int64 indices in the declared class order.")
+            if not torch.isfinite(values).all():
+                raise ValueError("Semantic logits must be finite.")
+            predicted = values.argmax(dim=1)
+            count = torch.bincount(
+                target * len(self.classes) + predicted, minlength=len(self.classes) ** 2
+            )
+            self.matrices[index] += count.reshape(len(self.classes), len(self.classes))
+            correct.append(predicted == target)
+        self.samples += before.numel()
+        self.joint_correct += int((correct[0] & correct[1]).sum())
+
+    def compute(self) -> dict[str, Any]:
+        """Return counts, class supports, per-date metrics, and the declared selection score."""
+        if not self.samples:
+            raise ValueError("Cannot evaluate an empty semantic split.")
+        result: dict[str, Any] = {
+            "classes": list(self.classes),
+            "samples": self.samples,
+            "joint_accuracy": self.joint_correct / self.samples,
+            "zero_denominator_convention": "zero; macro includes all declared classes",
+        }
+        for date, matrix in zip(("before", "after"), self.matrices, strict=True):
+            counts = matrix.double()
+            tp, support, predicted = counts.diag(), counts.sum(1), counts.sum(0)
+            precision = torch.where(predicted > 0, tp / predicted, 0)
+            recall = torch.where(support > 0, tp / support, 0)
+            f1 = torch.where(
+                precision + recall > 0, 2 * precision * recall / (precision + recall), 0
+            )
+            result[f"{date}_accuracy"] = float(tp.sum()) / self.samples
+            result[f"{date}_macro_f1"] = float(f1.mean())
+            result[f"{date}_confusion"] = matrix.tolist()
+            result[f"{date}_per_class"] = [
+                {
+                    "class": name,
+                    "precision": float(precision[i]),
+                    "recall": float(recall[i]),
+                    "f1": float(f1[i]),
+                    "support": int(support[i]),
+                }
+                for i, name in enumerate(self.classes)
+            ]
+        result["selection_score"] = (result["before_macro_f1"] + result["after_macro_f1"]) / 2
+        result["selection_rule"] = SELECTION_RULE
+        return result
+
+
+def logical_forward(
+    model: TinySiameseChangeClassifier,
+    samples: list[dict[str, Any]],
+    device: torch.device,
+) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
+    """Forward each variable-size pair, then concatenate logits over the logical batch.
+
+    BatchNorm statistics cover each temporal pair, not the full logical minibatch.
+    """
+    if not samples:
+        raise ValueError("Logical batches must not be empty.")
+    outputs = [
+        model(*(sample[key].unsqueeze(0).to(device) for key in ("before", "after", "mask")))
+        for sample in samples
+    ]
+    logits = {
+        key: torch.cat([output[key] for output in outputs])
+        for key in ("before_logits", "after_logits")
+    }
+    labels = [
+        torch.tensor([sample[key] for sample in samples], dtype=torch.long, device=device)
+        for key in ("before_label", "after_label")
+    ]
+    return logits, labels[0], labels[1]
+
+
+def semantic_loss(
+    logits: dict[str, torch.Tensor],
+    before: torch.Tensor,
+    after: torch.Tensor,
+    weights: torch.Tensor | None = None,
+) -> tuple[torch.Tensor, list[tuple[float, float]]]:
+    """Sum date-wise weighted CE over the logical batch; return exact logging denominators.
+
+    For a singleton logical batch, normalized weighted CE cancels its class weight.
+    Split logging sums numerators/denominators separately for each date.
+    """
+    terms = []
+    totals = []
+    for key, labels in (("before_logits", before), ("after_logits", after)):
+        numerator = F.cross_entropy(logits[key].float(), labels, weight=weights, reduction="sum")
+        denominator = (
+            weights[labels].sum() if weights is not None else numerator.new_tensor(labels.numel())
+        )
+        if not torch.isfinite(numerator) or not torch.isfinite(denominator) or denominator <= 0:
+            raise ValueError("Semantic CE needs finite logits and a positive loss denominator.")
+        terms.append(numerator / denominator)
+        totals.append((float(numerator.detach()), float(denominator.detach())))
+    return terms[0] + terms[1], totals
+
+
+def run_epoch(
+    model: TinySiameseChangeClassifier,
+    loader: DataLoader,
+    device: torch.device,
+    *,
+    weights: torch.Tensor | None = None,
+    precision: str = "fp32",
+    optimizer: torch.optim.Optimizer | None = None,
+) -> dict[str, Any]:
+    """Fit or evaluate a split, handling final partial logical batches correctly."""
+    model.train(optimizer is not None)
+    metrics = SemanticMetrics(model.classes)
+    totals = [[0.0, 0.0], [0.0, 0.0]]
+    for samples in loader:
+        if optimizer is not None:
+            optimizer.zero_grad(set_to_none=True)
+        with torch.set_grad_enabled(optimizer is not None), precision_context(device, precision):
+            logits, before, after = logical_forward(model, samples, device)
+            loss, batch_totals = semantic_loss(logits, before, after, weights)
+        if optimizer is not None:
+            loss.backward()
+            optimizer.step()
+        metrics.update(logits, before, after)
+        for total, batch_total in zip(totals, batch_totals, strict=True):
+            total[0] += batch_total[0]
+            total[1] += batch_total[1]
+    result = metrics.compute()
+    result["loss"] = sum(numerator / denominator for numerator, denominator in totals)
+    result["loss_totals"] = totals
+    return result
diff --git a/src/satchange/training.py b/src/satchange/training.py
index f0835fd..19eea34 100644
--- a/src/satchange/training.py
+++ b/src/satchange/training.py
@@ -6,2 +5,0 @@ import pickle
-import warnings
-from uuid import uuid4
@@ -8,0 +7 @@ import time
+import warnings
@@ -11,0 +11 @@ from typing import Any
+from uuid import uuid4
@@ -42,0 +43 @@ class EarlyStopping:
+    monitor: str = "validation_f1"
@@ -47,0 +49,2 @@ class EarlyStopping:
+        if self.monitor not in ("validation_f1", "selection_score"):
+            raise ValueError("Unsupported early-stopping monitor.")
@@ -69 +72 @@ class EarlyStopping:
-            "monitor": "validation_f1",
+            "monitor": self.monitor,
@@ -143,4 +146 @@ def _save_checkpoint(checkpoint_path: Path, checkpoint_state: dict[str, Any]) ->
-    """Commit one checkpoint atomically; an interrupted write keeps the old file."""
-    import os
-    from uuid import uuid4
-
+    """Commit a checkpoint atomically; a failed write leaves the previous file intact."""
@@ -152,4 +152,4 @@ def _save_checkpoint(checkpoint_path: Path, checkpoint_state: dict[str, Any]) ->
-        with temporary.open("wb") as handle:
-            torch.save(checkpoint_state, handle)
-            handle.flush()
-            os.fsync(handle.fileno())
+        with temporary.open("wb") as file:
+            torch.save(checkpoint_state, file)
+            file.flush()
+            os.fsync(file.fileno())
@@ -191,2 +191,4 @@ def find_training_checkpoint(directory: str | Path) -> tuple[Path, int] | None:
-        except (OSError, EOFError, RuntimeError, ValueError, KeyError, pickle.UnpicklingError) as error:
-            warnings.warn(f"Cannot resume {path}: {error}", RuntimeWarning)
+        except (
+            OSError, EOFError, RuntimeError, ValueError, KeyError, pickle.UnpicklingError
+        ) as error:
+            warnings.warn(f"Cannot resume {path}: {error}", RuntimeWarning, stacklevel=2)
@@ -257,2 +259,4 @@ def restore_best_checkpoint(state: dict[str, Any], source: Path, destination: Pa
-                    and candidate.get("best_metrics", {}).get("validation_f1")
-                        == state["best_metrics"]["validation_f1"]
+                    and candidate.get("best_metrics", {}).get("selection_score",
+                        candidate.get("best_metrics", {}).get("validation_f1"))
+                        == state["best_metrics"].get("selection_score",
+                           state["best_metrics"].get("validation_f1"))
@@ -262,0 +267,4 @@ def restore_best_checkpoint(state: dict[str, Any], source: Path, destination: Pa
+    if "task" in state:
+        best_state["task"] = state["task"]
+        best_state["runtime"] = state.get("runtime", {})
+        best_state["runtime_history"] = state.get("runtime_history", [])
@@ -567,0 +576 @@ def fit_binary_change_model(
+    task: str = "binary_change_detection",
@@ -578 +587,11 @@ def fit_binary_change_model(
-    early_stopping = EarlyStopping(patience, min_epochs, min_delta)
+    from satchange.models.factory import BINARY_TASK, SEMANTIC_TASK
+    if task not in (BINARY_TASK, SEMANTIC_TASK):
+        raise ValueError(f"Unsupported training task: {task}")
+    semantic = task == SEMANTIC_TASK
+    score_key = "selection_score" if semantic else "validation_f1"
+    if semantic and (thresholds or world_size() != 1):
+        raise ValueError("Semantic ROI trials use no thresholds and one worker per GPU.")
+    if semantic and config is None:
+        raise ValueError("Semantic training requires resolved HPO configuration.")
+    tf32 = bool((config or {}).get("tf32", False))
+    early_stopping = EarlyStopping(patience, min_epochs, min_delta, monitor=score_key)
@@ -647,0 +667 @@ def fit_binary_change_model(
+    runtime_history: list[dict[str, Any]] = []
@@ -710,0 +731,2 @@ def fit_binary_change_model(
+        if checkpoint.get("task", BINARY_TASK) != task:
+            raise ValueError("Resume task differs from the saved checkpoint.")
@@ -711,0 +734,5 @@ def fit_binary_change_model(
+        if semantic:
+            from satchange.optimization import _resume_parameters
+            if (config is None
+                    or _resume_parameters(config) != _resume_parameters(saved_config or {})):
+                raise ValueError("Resume semantic taxonomy/data/training protocol differs.")
@@ -761 +788 @@ def fit_binary_change_model(
-                best_metrics["validation_f1"]
+                best_metrics[score_key]
@@ -768 +795 @@ def fit_binary_change_model(
-                    result["validation_f1"]
+                    result[score_key]
@@ -773 +800 @@ def fit_binary_change_model(
-                best_metrics["validation_f1"]
+                best_metrics[score_key]
@@ -775,0 +803,8 @@ def fit_binary_change_model(
+        if semantic:
+            runtime_history = checkpoint.get("runtime_history", [])
+            previous_runtime = checkpoint.get("runtime", {})
+            current_runtime = {"device": str(device), "torch": str(torch.__version__),
+                               "cuda": torch.version.cuda, "tf32": tf32}
+            if previous_runtime != current_runtime:
+                runtime_history = [*runtime_history, {"resumed_epoch": saved_epoch,
+                                   "previous": previous_runtime, "current": current_runtime}]
@@ -777 +811,0 @@ def fit_binary_change_model(
-            # Explicit phase promotion: preserve weights/optimizer/scheduler/RNG/best history.
@@ -786 +820 @@ def fit_binary_change_model(
-
+        del checkpoint
@@ -835,8 +869,26 @@ def fit_binary_change_model(
-        train_loss = train_one_epoch(
-            model=model,
-            data_loader=train_loader,
-            loss_function=loss_function,
-            optimizer=optimizer,
-            device=device,
-            precision=precision,
-        )
+        if semantic:
+            from satchange.models import TinySiameseChangeClassifier
+            from satchange.semantic.training import SemanticCELoss, run_epoch
+            if not isinstance(bare_model, TinySiameseChangeClassifier) or not isinstance(
+                loss_function, SemanticCELoss
+            ):
+                raise ValueError(
+                    "Semantic task requires Tiny classifier and SemanticCELoss adapters."
+                )
+            weights = loss_function.weights
+            assert weights is None or isinstance(weights, torch.Tensor)
+            training_results = run_epoch(bare_model, train_loader, device, weights=weights,
+                                         precision=precision, optimizer=optimizer)
+            train_loss = training_results["loss"]
+            validation_results = run_epoch(bare_model, validation_loader, device,
+                                           weights=weights, precision=precision)
+            epoch_score = validation_results["selection_score"]
+        else:
+            train_loss = train_one_epoch(
+                model=model,
+                data_loader=train_loader,
+                loss_function=loss_function,
+                optimizer=optimizer,
+                device=device,
+                precision=precision,
+            )
@@ -844,9 +896,9 @@ def fit_binary_change_model(
-        train_loss = sum(gather_objects(train_loss)) / world_size()
-        validation_results = on_rank_zero(lambda: evaluate_thresholds(
-            model=bare_model,
-            data_loader=validation_loader,
-            loss_function=loss_function,
-            device=device,
-            thresholds=thresholds,
-            precision=precision,
-        ))
+            train_loss = sum(gather_objects(train_loss)) / world_size()
+            validation_results = on_rank_zero(lambda: evaluate_thresholds(
+                model=bare_model,
+                data_loader=validation_loader,
+                loss_function=loss_function,
+                device=device,
+                thresholds=thresholds,
+                precision=precision,
+            ))
@@ -854,6 +906,8 @@ def fit_binary_change_model(
-        best_threshold_result = max(
-            validation_results[
-                "threshold_results"
-            ],
-            key=lambda result: result["f1"],
-        )
+            best_threshold_result = max(
+                validation_results[
+                    "threshold_results"
+                ],
+                key=lambda result: result["f1"],
+            )
+
+            epoch_score = best_threshold_result["f1"]
@@ -871 +925 @@ def fit_binary_change_model(
-                    best_threshold_result["f1"]
+                    epoch_score
@@ -901,17 +955,20 @@ def fit_binary_change_model(
-        threshold_05_result = next(
-            (
-                result
-                for result
-                in validation_results[
-                    "threshold_results"
-                ]
-                if result["threshold"] == 0.50
-            ),
-            None,
-        )
-
-        mean_changed_probability = (
-            validation_results[
-                "mean_changed_probability"
-            ]
-        )
+        if semantic:
+            epoch_metrics = {
+                "epoch": epoch, "task": task, "train_loss": train_loss,
+                "validation_loss": validation_results["loss"],
+                **{k: v for k, v in validation_results.items() if k not in ("loss", "loss_totals")},
+                "learning_rate": learning_rate, "next_learning_rate": next_learning_rate,
+                "epoch_seconds": epoch_seconds, "peak_gpu_memory_mb": peak_gpu_memory_mb,
+            }
+        else:
+            threshold_05_result = next(
+                (
+                    result
+                    for result
+                    in validation_results[
+                        "threshold_results"
+                    ]
+                    if result["threshold"] == 0.50
+                ),
+                None,
+            )
@@ -919,5 +976,5 @@ def fit_binary_change_model(
-        mean_unchanged_probability = (
-            validation_results[
-                "mean_unchanged_probability"
-            ]
-        )
+            mean_changed_probability = (
+                validation_results[
+                    "mean_changed_probability"
+                ]
+            )
@@ -925,4 +982,5 @@ def fit_binary_change_model(
-        probability_gap = (
-            mean_changed_probability
-            - mean_unchanged_probability
-        )
+            mean_unchanged_probability = (
+                validation_results[
+                    "mean_unchanged_probability"
+                ]
+            )
@@ -930,42 +988 @@ def fit_binary_change_model(
-        epoch_metrics = {
-            "epoch": epoch,
-            "train_loss": train_loss,
-            "validation_loss": (
-                validation_results["loss"]
-            ),
-            "best_threshold": (
-                best_threshold_result["threshold"]
-            ),
-            "validation_precision": (
-                best_threshold_result["precision"]
-            ),
-            "validation_recall": (
-                best_threshold_result["recall"]
-            ),
-            "validation_f1": (
-                best_threshold_result["f1"]
-            ),
-            "validation_iou": (
-                best_threshold_result["iou"]
-            ),
-            "validation_accuracy": (
-                best_threshold_result["accuracy"]
-            ),
-            "true_positive": (
-                best_threshold_result["tp"]
-            ),
-            "false_positive": (
-                best_threshold_result["fp"]
-            ),
-            "false_negative": (
-                best_threshold_result["fn"]
-            ),
-            "true_negative": (
-                best_threshold_result["tn"]
-            ),
-            "f1_at_05": (
-                threshold_05_result["f1"]
-                if threshold_05_result is not None
-                else None
-            ),
-            "mean_changed_probability": (
+            probability_gap = (
@@ -973,10 +990,56 @@ def fit_binary_change_model(
-            ),
-            "mean_unchanged_probability": (
-                mean_unchanged_probability
-            ),
-            "probability_gap": probability_gap,
-            "learning_rate": learning_rate,
-            "next_learning_rate": next_learning_rate,
-            "epoch_seconds": epoch_seconds,
-            "peak_gpu_memory_mb": peak_gpu_memory_mb,
-        }
+                - mean_unchanged_probability
+            )
+
+            epoch_metrics = {
+                "epoch": epoch,
+                "train_loss": train_loss,
+                "validation_loss": (
+                    validation_results["loss"]
+                ),
+                "best_threshold": (
+                    best_threshold_result["threshold"]
+                ),
+                "validation_precision": (
+                    best_threshold_result["precision"]
+                ),
+                "validation_recall": (
+                    best_threshold_result["recall"]
+                ),
+                "validation_f1": (
+                    best_threshold_result["f1"]
+                ),
+                "validation_iou": (
+                    best_threshold_result["iou"]
+                ),
+                "validation_accuracy": (
+                    best_threshold_result["accuracy"]
+                ),
+                "true_positive": (
+                    best_threshold_result["tp"]
+                ),
+                "false_positive": (
+                    best_threshold_result["fp"]
+                ),
+                "false_negative": (
+                    best_threshold_result["fn"]
+                ),
+                "true_negative": (
+                    best_threshold_result["tn"]
+                ),
+                "f1_at_05": (
+                    threshold_05_result["f1"]
+                    if threshold_05_result is not None
+                    else None
+                ),
+                "mean_changed_probability": (
+                    mean_changed_probability
+                ),
+                "mean_unchanged_probability": (
+                    mean_unchanged_probability
+                ),
+                "probability_gap": probability_gap,
+                "learning_rate": learning_rate,
+                "next_learning_rate": next_learning_rate,
+                "epoch_seconds": epoch_seconds,
+                "peak_gpu_memory_mb": peak_gpu_memory_mb,
+            }
@@ -989 +1052 @@ def fit_binary_change_model(
-            epoch_metrics["validation_f1"]
+            epoch_metrics[score_key]
@@ -1055,0 +1119,7 @@ def fit_binary_change_model(
+        if semantic:
+            checkpoint_state.pop("thresholds")
+            checkpoint_state["task"] = task
+            checkpoint_state["runtime_history"] = runtime_history
+            checkpoint_state["runtime"] = {"device": str(device), "torch": str(torch.__version__),
+                                           "cuda": torch.version.cuda, "tf32": tf32}
+
@@ -1118,4 +1188,8 @@ def fit_binary_change_model(
-        print(
-            f"Best threshold:   "
-            f"{best_threshold_result['threshold']:.2f}"
-        )
+        if semantic:
+            print(f"Validation selection score: {epoch_score:.6f}; "
+                  f"joint accuracy: {validation_results['joint_accuracy']:.6f}")
+        else:
+            print(
+                f"Best threshold:   "
+                f"{best_threshold_result['threshold']:.2f}"
+            )
@@ -1123,4 +1197,4 @@ def fit_binary_change_model(
-        print(
-            f"Precision:        "
-            f"{best_threshold_result['precision']:.6f}"
-        )
+            print(
+                f"Precision:        "
+                f"{best_threshold_result['precision']:.6f}"
+            )
@@ -1128,4 +1202,4 @@ def fit_binary_change_model(
-        print(
-            f"Recall:           "
-            f"{best_threshold_result['recall']:.6f}"
-        )
+            print(
+                f"Recall:           "
+                f"{best_threshold_result['recall']:.6f}"
+            )
@@ -1133,4 +1207,4 @@ def fit_binary_change_model(
-        print(
-            f"F1:               "
-            f"{best_threshold_result['f1']:.6f}"
-        )
+            print(
+                f"F1:               "
+                f"{best_threshold_result['f1']:.6f}"
+            )
@@ -1138,4 +1212,4 @@ def fit_binary_change_model(
-        print(
-            f"IoU:              "
-            f"{best_threshold_result['iou']:.6f}"
-        )
+            print(
+                f"IoU:              "
+                f"{best_threshold_result['iou']:.6f}"
+            )
@@ -1147 +1221 @@ def fit_binary_change_model(
-                    f"{early_stopping.bad_epochs} epochs without validation F1 "
+                    f"{early_stopping.bad_epochs} epochs without {score_key} "
@@ -1149 +1223,2 @@ def fit_binary_change_model(
-                    f"Best checkpoint: epoch {best_metrics['epoch']}, F1={best_f1:.6f}."
+                    f"Best checkpoint: epoch {(best_metrics or {})['epoch']}, "
+                    f"{score_key}={best_f1:.6f}."
diff --git a/tests/test_execution.py b/tests/test_execution.py
index 927cc6f..1095465 100644
--- a/tests/test_execution.py
+++ b/tests/test_execution.py
@@ -208,0 +209,2 @@ def test_generic_yaml_training_launcher_and_collision_guard(tmp_path):
+    config["data"]["num_workers"] = 0
+    config["data"]["batch_size"] = 1
@@ -209,0 +212,2 @@ def test_generic_yaml_training_launcher_and_collision_guard(tmp_path):
+    config["training"]["thresholds"] = [0.5]  # Explicit validation grid for the fixture.
+    config["runtime"]["precision"] = "fp32"  # This launcher fixture runs on CPU.
diff --git a/tests/test_notebooks.py b/tests/test_notebooks.py
index 6b1c8ff..7bbebef 100644
--- a/tests/test_notebooks.py
+++ b/tests/test_notebooks.py
@@ -21 +21,4 @@ def test_notebook_syntax_and_heavy_safe_fresh_kernel():
-    notebook = nbformat.read(root / "notebooks/03_heavy.ipynb", as_version=4)
+    legacy = root / "notebooks/03_heavy.ipynb"
+    if not legacy.is_file():
+        pytest.skip("Legacy 03_heavy notebook is absent; all notebook syntax was validated.")
+    notebook = nbformat.read(legacy, as_version=4)
@@ -41,0 +45,35 @@ def test_notebook_syntax_and_heavy_safe_fresh_kernel():
+
+
+def test_tiny_second_safe_fresh_kernel():
+    """Execute default guards and CPU smoke; never downloads or runs real HPO/test."""
+    nbformat = pytest.importorskip("nbformat")
+    nbclient = pytest.importorskip("nbclient")
+    jupyter_client = pytest.importorskip("jupyter_client")
+    root = Path(__file__).resolve().parents[1]
+    path = root / "notebooks/tiny_siamese_second_staged_resume.ipynb"
+    notebook = nbformat.read(path, as_version=4)
+    nbformat.validate(notebook)
+    assert all(not cell.get("outputs") for cell in notebook.cells)
+    manager = jupyter_client.KernelManager(kernel_name="python3")
+    manager.kernel_spec.argv = [
+        sys.executable,
+        "-m",
+        "ipykernel_launcher",
+        "-f",
+        "{connection_file}",
+    ]
+    try:
+        nbclient.NotebookClient(
+            notebook,
+            km=manager,
+            timeout=60,
+            resources={"metadata": {"path": str(root / "notebooks")}},
+        ).execute()
+    finally:
+        if manager.has_kernel:
+            manager.shutdown_kernel(now=True)
+    outputs = " ".join(str(cell.get("outputs", [])) for cell in notebook.cells)
+    assert "Finite synthetic CE" in outputs and "Final test disabled" in outputs
+    assert "Preparation disabled" in outputs and "01_batch_lr disabled" in outputs
+    # Executed outputs remain in memory; the delivered notebook stays clean.
+    assert all(not cell.get("outputs") for cell in nbformat.read(path, as_version=4).cells)
diff --git a/tests/test_semantic.py b/tests/test_semantic.py
new file mode 100644
index 0000000..6450fc9
--- /dev/null
+++ b/tests/test_semantic.py
@@ -0,0 +1,549 @@
+"""CPU-only synthetic implementation checks, not evidence of real semantic accuracy."""
+
+import json
+from copy import deepcopy
+from dataclasses import replace
+
+import numpy as np
+import pytest
+import torch
+from PIL import Image
+from torch import nn
+from torch.nn import functional as F
+from torch.utils.data import DataLoader
+
+from satchange.execution import seed_everything
+from satchange.models import TinySiameseChangeClassifier, count_model_parameters
+from satchange.models.factory import MODEL_REGISTRY
+from satchange.models.tiny_siamese_change_classifier import masked_pool
+from satchange.semantic.__main__ import main, synthetic_smoke
+from satchange.semantic.data import (
+    SemanticROIDataset,
+    collate_rois,
+    data_protocol,
+    dominant_transition,
+    prepare_dataset,
+    read_manifest,
+    read_map,
+)
+from satchange.semantic.roi import ROISettings, align_whole_image_probabilities, extract_rois
+from satchange.semantic.training import (
+    SemanticMetrics,
+    create_loaders,
+    hpo_parameters,
+    logical_forward,
+    run_epoch,
+    semantic_loss,
+    validate_parameters,
+)
+
+
+@pytest.fixture(autouse=True)
+def cpu_threads():
+    previous = torch.get_num_threads()
+    torch.set_num_threads(1)
+    yield
+    torch.set_num_threads(previous)
+
+
+def config():
+    return {
+        "classes": ["land", "built"],
+        "class_ids": {1: "land", 2: "built"},
+        "ignore_ids": [0, 255],
+        "mask_source": "ground_truth",
+        "detector": None,
+        "roi": {"context_pixels": 1, "min_crop_size": 8},
+        "label_policy": {"min_transition_purity": 0.8, "min_valid_fraction": 0.8},
+        "dropout": 0.1,
+        "training": {"epochs": 2, "batch_size": 4, "seed": 42, "num_workers": 0},
+    }
+
+
+def scene_tensors(height=171, width=197):
+    before = torch.arange(height * width).reshape(1, height, width).float()
+    before = (before / (height * width)).expand(3, -1, -1).contiguous()
+    return before, 1 - before, torch.zeros(1, height, width)
+
+
+def make_manifest(tmp_path):
+    scenes = []
+    for index, split in enumerate(("train", "train", "train", "val", "test")):
+        height, width = 17 + index, 23 + index
+        before = np.full((height, width, 3), 20 + index, dtype=np.uint8)
+        after = np.full((height, width, 3), 200 - index, dtype=np.uint8)
+        change = np.zeros((height, width), dtype=np.float32)
+        change[0 : 3 + index, 1 : 4 + index] = 1
+        semantic_before = np.ones((height, width), dtype=np.int16)
+        semantic_after = np.full((height, width), 2, dtype=np.int16)
+        prefix = f"scene_{index}"
+        row = {"sample_id": prefix, "group_id": prefix, "split": split}
+        for key, array in (
+            ("before", before),
+            ("after", after),
+            ("before_semantic", semantic_before),
+            ("after_semantic", semantic_after),
+            ("change", change),
+        ):
+            name = f"{prefix}_{key}.{'png' if key in ('before', 'after') else 'npy'}"
+            if key in ("before", "after"):
+                Image.fromarray(array).save(tmp_path / name)
+            else:
+                np.save(tmp_path / name, array)
+            row[key] = name
+        scenes.append(row)
+    manifest = tmp_path / "manifest.json"
+    manifest.write_text(json.dumps({"scenes": scenes}), encoding="utf-8")
+    return manifest, scenes
+
+
+@pytest.mark.parametrize("size", [(128, 128), (17, 23), (9, 31), (4, 4)])
+def test_shared_forward_shape_grid_and_swap(size):
+    seed_everything(42)
+    model = TinySiameseChangeClassifier(["a", "b", "c"]).eval()
+    assert "tiny_siamese_change_classifier" not in MODEL_REGISTRY
+    assert count_model_parameters(model)["total_parameters"] < 110_000
+    assert sum(isinstance(layer, nn.Conv2d) for layer in model.modules()) == 3
+    assert sum(isinstance(layer, nn.MaxPool2d) for layer in model.modules()) == 2
+    calls = []
+    hook = model.backbone.register_forward_hook(
+        lambda module, args, output: calls.append(output.shape)
+    )
+    before, after = torch.rand(2, 3, *size), torch.rand(2, 3, *size)
+    mask = torch.zeros(2, 1, *size)
+    mask[:, :, -1, -1] = 1  # Tiny support survives actual odd-size area reduction.
+    normal = model(before, after, mask)
+    assert calls == [torch.Size([4, 128, size[0] // 4, size[1] // 4])]
+    swapped = model(after, before, mask)
+    hook.remove()
+    assert normal["before_logits"].shape == (2, 3)
+    torch.testing.assert_close(normal["before_logits"], swapped["after_logits"])
+    torch.testing.assert_close(normal["after_logits"], swapped["before_logits"])
+
+
+def test_both_losses_reach_one_backbone_and_head():
+    model = TinySiameseChangeClassifier(["a", "b"], dropout=0)
+    before, after = torch.rand(1, 3, 17, 19), torch.rand(1, 3, 17, 19)
+    mask = torch.ones(1, 1, 17, 19)
+    for key in ("before_logits", "after_logits"):
+        model.zero_grad(set_to_none=True)
+        output = model(before, after, mask)
+        F.cross_entropy(output[key], torch.tensor([1])).backward()
+        for parameters in (model.backbone.parameters(), model.head.parameters()):
+            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
+        assert model.backbone[0].weight.grad.abs().sum() > 0
+    assert int(model.backbone[1].num_batches_tracked) == 2  # One update per temporal pair.
+
+
+def test_pool_hand_calculation_fp32_and_tiny_support():
+    features = torch.tensor([[[[1, 2], [3, 4]]]], dtype=torch.bfloat16)
+    mask = torch.tensor([[[[1.0, 0], [0.5, 0]]]])
+    result = masked_pool(features, mask)
+    assert result.dtype == torch.float32
+    assert float(result) == pytest.approx(2.5 / 1.5)
+    tiny = torch.zeros(1, 1, 17, 23)
+    tiny[0, 0, -1, -1] = 0.7
+    assert float(masked_pool(torch.ones(1, 2, 4, 5), tiny)[0, 0]) == pytest.approx(1, abs=1e-5)
+    with pytest.raises(ValueError, match="nonempty"):
+        masked_pool(features, torch.zeros_like(mask))
+
+
+@pytest.mark.parametrize(
+    "failure", ["empty", "nan", "negative", "above_one", "shape", "small", "dtype"]
+)
+def test_model_input_validation(failure):
+    model = TinySiameseChangeClassifier(["a", "b"])
+    before, after, mask = torch.rand(1, 3, 9, 11), torch.rand(1, 3, 9, 11), torch.ones(1, 1, 9, 11)
+    if failure == "empty":
+        mask.zero_()
+    elif failure in ("nan", "negative", "above_one"):
+        mask[0, 0, 0, 0] = {"nan": float("nan"), "negative": -1, "above_one": 2}[failure]
+    elif failure == "shape":
+        after = after[..., :-1]
+    elif failure == "small":
+        before, after, mask = before[:, :, :3], after[:, :, :3], mask[:, :, :3]
+    else:
+        before = before.double()
+    with pytest.raises(ValueError):
+        model(before, after, mask)
+
+
+@pytest.mark.parametrize(
+    "classes,dropout",
+    [([], 0.1), (["a"], 0.1), (["a", "a"], 0.1), (["a", "b"], 1), (["a", "b"], float("nan"))],
+)
+def test_model_configuration_validation(classes, dropout):
+    with pytest.raises(ValueError):
+        TinySiameseChangeClassifier(classes, dropout)
+
+
+@pytest.mark.parametrize("y,x", [(0, 0), (0, 196), (170, 0), (170, 196), (90, 100)])
+def test_real_context_near_all_borders(y, x):
+    before, after, probability = scene_tensors()
+    probability[0, y, x] = 0.9
+    (roi,) = extract_rois(before, after, probability)
+    assert roi.before.shape == (3, 128, 128)
+    assert not roi.minimum_extent_limited
+    x0, y0, x1, y1 = roi.crop_bbox
+    torch.testing.assert_close(roi.before, before[:, y0:y1, x0:x1])
+    torch.testing.assert_close(roi.after, after[:, y0:y1, x0:x1])
+    assert roi.component_bbox == (x, y, x + 1, y + 1)
+    assert roi.mask.sum() == pytest.approx(0.9)
+    assert x0 <= x < x1 and y0 <= y < y1
+
+
+def test_scale_large_components_small_images_neighbors_and_order():
+    before, after, probability = scene_tensors(180, 240)
+    probability[0, 2:32, 4:44] = 0.8
+    probability[0, 40:42, 50:52] = 0.7  # Included in context but excluded from first pooling mask.
+    rois = extract_rois(before, after, probability)
+    assert [roi.area for roi in rois] == [30 * 40, 4]
+    assert rois[0].probabilities.sum() > rois[0].mask.sum()
+    assert int(rois[0].support.sum()) == 30 * 40
+    assert [roi.crop_bbox for roi in rois] == [
+        roi.crop_bbox for roi in extract_rois(before, after, probability)
+    ]
+    large = torch.ones_like(probability)
+    (roi,) = extract_rois(before, after, large)
+    assert roi.before.shape == before.shape and roi.area == 180 * 240
+    before, after, probability = scene_tensors(37, 51)
+    probability[:, 1, 1] = 1
+    (roi,) = extract_rois(before, after, probability)
+    assert roi.minimum_extent_limited and roi.before.shape == (3, 37, 51)
+    assert extract_rois(before, after, torch.zeros_like(probability)) == []
+
+
+def test_connectivity_min_area_and_probability_alignment():
+    before, after, probability = scene_tensors(10, 12)
+    probability[0, 2, 2] = probability[0, 3, 3] = 1
+    settings = ROISettings(min_crop_size=4, context_pixels=0)
+    assert len(extract_rois(before, after, probability, settings)) == 1
+    assert len(extract_rois(before, after, probability, replace(settings, connectivity=4))) == 2
+    assert extract_rois(before, after, probability, replace(settings, min_component_area=3)) == []
+    resized = align_whole_image_probabilities(torch.ones(1, 4, 5), (10, 12), (4, 5))
+    assert resized.shape == (1, 10, 12)
+    torch.testing.assert_close(resized, torch.ones_like(resized))
+    with pytest.raises(ValueError):
+        align_whole_image_probabilities(probability, (10, 12), (4, 5))
+
+
+@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.1])
+def test_roi_rejects_invalid_probabilities(bad):
+    before, after, probabilities = scene_tensors(8, 8)
+    probabilities[0, 1, 1] = bad
+    with pytest.raises(ValueError, match="finite"):
+        extract_rois(before, after, probabilities)
+
+
+def test_roi_rejects_spatial_mismatch_and_small_input():
+    before, after, probabilities = scene_tensors(8, 8)
+    with pytest.raises(ValueError, match="align"):
+        extract_rois(before, after, probabilities[:, :-1])
+    with pytest.raises(ValueError, match="too small"):
+        extract_rois(before[:, :3], after[:, :3], probabilities[:, :3])
+
+
+def test_component_ids_stay_stable_after_area_filtering():
+    before, after, probability = scene_tensors(20, 20)
+    probability[0, 1, 1] = 1
+    probability[0, 10:12, 10:12] = 1
+    settings = ROISettings(min_crop_size=4, context_pixels=0)
+    assert [roi.component_id for roi in extract_rois(before, after, probability, settings)] == [
+        1,
+        2,
+    ]
+    filtered = extract_rois(before, after, probability, replace(settings, min_component_area=2))
+    assert [roi.component_id for roi in filtered] == [2]
+
+
+def test_joint_label_selection_does_not_invent_independent_majority_transition():
+    cfg = config()
+    cfg["classes"].append("third")
+    cfg["class_ids"][3] = "third"
+    cfg["label_policy"]["min_transition_purity"] = 0.2
+    # Marginal majorities are (1,1), which is absent. Actual dominant pair is (1,2).
+    pairs = [(1, 2)] * 5 + [(1, 3)] * 4 + [(2, 1)] * 4 + [(3, 1)] * 4
+    before, after = torch.tensor(pairs).T
+    quality = dominant_transition(
+        before.unsqueeze(0),
+        after.unsqueeze(0),
+        torch.ones(1, len(pairs), dtype=torch.bool),
+        data_protocol(cfg),
+    )
+    assert before.mode().values == after.mode().values == 1
+    assert (quality["before_label"], quality["after_label"]) == (0, 1)
+
+
+def test_observed_joint_transition_purity_ignore_ties_and_same_class():
+    cfg = config()
+    cfg["label_policy"]["min_transition_purity"] = 0.4
+    protocol = data_protocol(cfg)
+    before = torch.tensor([[1, 1, 1, 2, 2, 0]])
+    after = torch.tensor([[2, 2, 1, 1, 1, 0]])
+    support = torch.ones_like(before, dtype=torch.bool)
+    tied = dominant_transition(before, after, support, protocol)
+    assert tied["reason"] == "tied_transition"
+    before[0, 4] = 1
+    quality = dominant_transition(before, after, support, protocol)
+    # Two independent majorities would select (1,1); that is tied with (1,2), so still rejected.
+    assert quality["reason"] == "tied_transition"
+    after[0, 4] = 2
+    quality = dominant_transition(before, after, support, protocol)
+    assert (quality["before_label"], quality["after_label"]) == (0, 1)
+    assert quality["transition_purity"] == 3 / 5
+    strict = data_protocol(config())
+    assert dominant_transition(before, after, support, strict)["reason"] == "mixed_transition"
+    before[0, :3] = 0
+    assert (
+        dominant_transition(before, after, support, strict)["reason"] == "insufficient_annotation"
+    )
+    same = dominant_transition(
+        torch.ones(2, 2, dtype=torch.long),
+        torch.ones(2, 2, dtype=torch.long),
+        torch.ones(2, 2, dtype=torch.bool),
+        strict,
+    )
+    assert same["before_label"] == same["after_label"] == 0
+    with pytest.raises(ValueError, match="undeclared"):
+        dominant_transition(
+            torch.full((2, 2), 99),
+            torch.ones(2, 2, dtype=torch.long),
+            torch.ones(2, 2, dtype=torch.bool),
+            strict,
+        )
+
+
+def test_manifest_preparation_native_crops_and_synchronized_augmentation(tmp_path):
+    manifest, scenes = make_manifest(tmp_path)
+    cfg = config()
+    document = prepare_dataset(cfg, manifest, tmp_path / "prepared")
+    assert document["coverage"]["train"]["retained_fraction"] == 1
+    assert document["coverage"]["train"]["transition_distribution"] == {"land -> built": 3}
+    assert len(list((tmp_path / "prepared").glob("masks/*.npz"))) == 5
+    assert not list((tmp_path / "prepared").glob("*.png"))
+    dataset = SemanticROIDataset(tmp_path / "prepared/prepared.json", "train", cfg, augment=True)
+    dataset.verify_sources()
+    torch.manual_seed(3)
+    sample = dataset[0]
+    assert sample["before"].shape[-2:] == sample["mask"].shape[-2:]
+    assert all(sample[key].is_contiguous() for key in ("before", "after", "mask"))
+    assert sample["before_label"] == 0 and sample["after_label"] == 1
+    # Exercise geometry with a spatial marker rather than constant fixture RGB.
+    from satchange.data.transforms import RandomFlipRotate
+
+    marker = torch.arange(8 * 11).reshape(1, 8, 11).float()
+    transformed = RandomFlipRotate(1, 1)(
+        {
+            "before": marker.expand(3, -1, -1),
+            "after": marker.expand(3, -1, -1) + 1,
+            "mask": marker,
+            "sample_id": "marker",
+        }
+    )
+    torch.testing.assert_close(transformed["before"][:1], transformed["mask"])
+    torch.testing.assert_close(transformed["before"] + 1, transformed["after"])
+    assert all(transformed[key].is_contiguous() for key in ("before", "after", "mask"))
+    with pytest.raises(FileExistsError):
+        prepare_dataset(cfg, manifest, tmp_path / "prepared")
+    with pytest.raises(ValueError, match="augmentation"):
+        SemanticROIDataset(tmp_path / "prepared/prepared.json", "val", cfg, augment=True)
+    changed = deepcopy(cfg)
+    changed["classes"].reverse()
+    with pytest.raises(ValueError, match="protocol"):
+        SemanticROIDataset(tmp_path / "prepared/prepared.json", "train", changed)
+    np.save(tmp_path / scenes[0]["change"], np.zeros((17, 23), dtype=np.float32))
+    with pytest.raises(ValueError, match="source changed"):
+        dataset.verify_sources()
+
+
+def test_preparation_exclusion_reporting_and_soft_predicted_weights(tmp_path):
+    manifest, scenes = make_manifest(tmp_path)
+    cfg = config()
+    cfg["mask_source"] = "predicted"
+    cfg["detector"] = {"identity": "synthetic-test-detector", "training_groups": ["scene_0"]}
+    probabilities = np.load(tmp_path / scenes[0]["change"]) * 0.7
+    np.save(tmp_path / scenes[0]["change"], probabilities)
+    labels = np.load(tmp_path / scenes[1]["before_semantic"])
+    labels[:] = 0
+    np.save(tmp_path / scenes[1]["before_semantic"], labels)
+    document = prepare_dataset(cfg, manifest, tmp_path / "prepared")
+    assert document["coverage"]["train"]["exclusions"] == {"insufficient_annotation": 1}
+    assert document["coverage"]["train"]["retained_fraction"] == 2 / 3
+    dataset = SemanticROIDataset(tmp_path / "prepared/prepared.json", "train", cfg)
+    assert dataset[0]["mask"].max() == pytest.approx(0.7)
+    assert any(not row["retained"] and row["quality"]["reason"] for row in document["records"])
+    cfg["mask_source"], cfg["detector"] = "ground_truth", None
+    with pytest.raises(ValueError, match="binary"):
+        prepare_dataset(cfg, manifest)
+
+
+def test_split_leakage_missing_semantics_and_detector_provenance(tmp_path):
+    manifest, scenes = make_manifest(tmp_path)
+    protocol = data_protocol(config())
+    scenes[-1]["group_id"] = scenes[0]["group_id"]
+    manifest.write_text(json.dumps({"scenes": scenes}), encoding="utf-8")
+    with pytest.raises(ValueError, match="Cross-split"):
+        read_manifest(manifest, protocol)
+    scenes[-1]["group_id"] = "test"
+    scenes[-1]["before_semantic"] = "missing.npy"
+    manifest.write_text(json.dumps({"scenes": scenes}), encoding="utf-8")
+    with pytest.raises(FileNotFoundError, match="before_semantic"):
+        read_manifest(manifest, protocol)
+    with pytest.raises(FileNotFoundError, match="LEVIR-CD"):
+        prepare_dataset(config(), tmp_path / "missing.json")
+    scenes[-1]["before_semantic"] = "scene_4_before_semantic.npy"
+    manifest.write_text(json.dumps({"scenes": scenes}), encoding="utf-8")
+    cfg = config()
+    cfg["mask_source"] = "predicted"
+    with pytest.raises(ValueError, match="detector identity"):
+        data_protocol(cfg)
+    cfg["detector"] = {"identity": "test", "training_groups": ["scene_3"]}
+    with pytest.raises(ValueError, match="Detector fitting-group leakage"):
+        read_manifest(manifest, data_protocol(cfg))
+
+
+def test_weighted_ce_over_logical_batch_and_singleton_cancellation():
+    output = {
+        "before_logits": torch.tensor([[2.0, 0], [0, 1.0]], requires_grad=True),
+        "after_logits": torch.tensor([[0.0, 2], [1.0, 0]], requires_grad=True),
+    }
+    before, after, weights = torch.tensor([0, 1]), torch.tensor([1, 0]), torch.tensor([1.0, 4.0])
+    loss, totals = semantic_loss(output, before, after, weights)
+    expected = sum(
+        F.cross_entropy(output[key], labels, weight=weights)
+        for key, labels in (("before_logits", before), ("after_logits", after))
+    )
+    torch.testing.assert_close(loss, expected)
+    assert [denominator for _, denominator in totals] == [5, 5]
+    loss.backward()
+    singleton = {key: value[:1] for key, value in output.items()}
+    weighted, _ = semantic_loss(singleton, before[:1], after[:1], weights)
+    unweighted, _ = semantic_loss(singleton, before[:1], after[:1])
+    torch.testing.assert_close(weighted, unweighted)
+
+
+def test_variable_logical_batches_partial_batch_and_exact_loss_logging():
+    model = TinySiameseChangeClassifier(["a", "b"]).eval()
+    samples = [
+        {
+            "before": torch.rand(3, h, w),
+            "after": torch.rand(3, h, w),
+            "mask": torch.ones(1, h, w),
+            "before_label": i % 2,
+            "after_label": (i + 1) % 2,
+        }
+        for i, (h, w) in enumerate(((8, 9), (9, 10), (11, 7), (8, 8), (5, 13)))
+    ]
+    loader = DataLoader(samples, batch_size=4, collate_fn=collate_rois)
+    assert [len(batch) for batch in loader] == [4, 1]
+    weights = torch.tensor([1.0, 3.0])
+    combined = run_epoch(model, loader, torch.device("cpu"), weights=weights)
+    logits, before, after = logical_forward(model, samples, torch.device("cpu"))
+    expected, totals = semantic_loss(logits, before, after, weights)
+    assert combined["loss"] == pytest.approx(float(expected.detach()))
+    np.testing.assert_allclose(combined["loss_totals"], totals, rtol=1e-6)
+    assert combined["samples"] == 5
+    optimizer = torch.optim.AdamW(model.parameters())
+    assert (
+        run_epoch(model, loader, torch.device("cpu"), weights=weights, optimizer=optimizer)[
+            "samples"
+        ]
+        == 5
+    )
+
+
+def test_global_confusion_macro_f1_joint_hand_calculation():
+    metrics = SemanticMetrics(("a", "b", "unsupported"))
+    before, after = torch.tensor([0, 0, 1]), torch.tensor([0, 1, 1])
+    logits = {
+        "before_logits": F.one_hot(torch.tensor([0, 1, 1]), 3).float(),
+        "after_logits": F.one_hot(torch.tensor([0, 1, 0]), 3).float(),
+    }
+    for start, stop in ((0, 2), (2, 3)):
+        metrics.update(
+            {key: value[start:stop] for key, value in logits.items()},
+            before[start:stop],
+            after[start:stop],
+        )
+    result = metrics.compute()
+    assert result["before_confusion"] == [[1, 1, 0], [0, 1, 0], [0, 0, 0]]
+    assert result["after_confusion"] == [[1, 0, 0], [1, 1, 0], [0, 0, 0]]
+    assert result["before_macro_f1"] == pytest.approx(4 / 9)
+    assert result["after_macro_f1"] == pytest.approx(4 / 9)
+    assert result["joint_accuracy"] == 1 / 3
+    assert result["before_per_class"][2]["support"] == 0
+    with pytest.raises(ValueError, match="empty"):
+        SemanticMetrics(("a", "b")).compute()
+
+
+def test_synthetic_smoke_bounded_tiny_overfit():
+    result = synthetic_smoke(60)
+    assert result["evidence"] == "synthetic implementation check only"
+    assert result["last_loss"] < result["first_loss"] * 0.1
+    assert result["eval_joint_accuracy"] == 1
+
+
+def test_cli_missing_data_and_no_independent_training(tmp_path, monkeypatch):
+    monkeypatch.setattr(
+        "sys.argv",
+        [
+            "semantic",
+            "validate",
+            "--config",
+            "configs/tiny_siamese_change_classifier.yaml",
+            "--manifest",
+            str(tmp_path / "missing.json"),
+        ],
+    )
+    with pytest.raises(FileNotFoundError, match="Semantic manifest missing"):
+        main()
+    monkeypatch.setattr("sys.argv", ["semantic", "train"])
+    with pytest.raises(SystemExit):
+        main()
+
+
+@pytest.mark.parametrize("failure", ["weights_length", "weights_zero", "seed", "precision"])
+def test_training_configuration_errors(tmp_path, failure):
+    manifest, _ = make_manifest(tmp_path)
+    cfg = config()
+    cfg["training"].update(
+        {
+            "weights_length": {"class_weights": [1]},
+            "weights_zero": {"class_weights": [0, 1]},
+            "seed": {"seed": 2**32},
+            "precision": {"precision": "fp16"},
+        }[failure]
+    )
+    prepare_dataset(cfg, manifest, tmp_path / "prepared")
+    with pytest.raises(ValueError):
+        validate_parameters(hpo_parameters(cfg, tmp_path / "prepared/prepared.json"))
+
+
+@pytest.mark.parametrize("bad", [1.000000001, -0.000000001, float("nan"), 2 + 1j])
+def test_map_validation_precedes_float32_conversion(tmp_path, bad):
+    path = tmp_path / "probabilities.npy"
+    np.save(path, np.full((4, 5), bad))
+    with pytest.raises(ValueError, match="finite"):
+        read_map(path)
+
+
+def test_unannotated_unknown_and_no_retained_regions_fail_clearly(tmp_path):
+    manifest, scenes = make_manifest(tmp_path)
+    cfg = config()
+    cfg["classes"][1] = "unknown"
+    cfg["class_ids"][2] = "unknown"
+    for row in scenes:
+        path = tmp_path / row["after_semantic"]
+        np.save(path, np.ones_like(np.load(path)))
+    prepare_dataset(cfg, manifest, tmp_path / "unknown")
+    with pytest.raises(ValueError, match="labeled training examples"):
+        create_loaders(
+            tmp_path / "unknown/prepared.json",
+            hpo_parameters(cfg, tmp_path / "unknown/prepared.json"),
+        )
+    cfg["roi"]["min_component_area"] = 100
+    doc = prepare_dataset(cfg, manifest, tmp_path / "filtered")
+    assert doc["coverage"]["train"]["retained"] == 0
+    assert doc["coverage"]["train"]["exclusions"] == {"small_component": 3}
+    with pytest.raises(ValueError, match="No eligible"):
+        SemanticROIDataset(tmp_path / "filtered/prepared.json", "train", cfg)
diff --git a/tests/test_semantic_hpo.py b/tests/test_semantic_hpo.py
new file mode 100644
index 0000000..a1e44ab
--- /dev/null
+++ b/tests/test_semantic_hpo.py
@@ -0,0 +1,652 @@
+"""Actual controller/worker/engine tests on synthetic CPU data; GPU queue boundary is mocked."""
+
+import json
+from copy import deepcopy
+from pathlib import Path
+
+import numpy as np
+import pytest
+import torch
+from PIL import Image
+from test_semantic import config, make_manifest, scene_tensors
+
+from satchange import automated_hpo as hpo
+from satchange import optimization
+from satchange.execution import resolve_device
+from satchange.models.factory import MODEL_REGISTRY, SEMANTIC_TASK
+from satchange.semantic.data import data_protocol, prepare_dataset, read_manifest
+from satchange.semantic.inference import (
+    detector_probability_map,
+    evaluate_classifier,
+    load_classifier,
+    predict_checkpoint,
+)
+from satchange.semantic.second import (
+    SECOND_CLASSES,
+    SECOND_COLORS,
+    bootstrap_second,
+    decode_label,
+    prepare_second,
+)
+from satchange.semantic.training import hpo_parameters
+from satchange.training import load_training_checkpoint
+
+
+@pytest.fixture(autouse=True)
+def threads():
+    previous = torch.get_num_threads()
+    torch.set_num_threads(1)
+    yield
+    torch.set_num_threads(previous)
+
+
+@pytest.fixture
+def cpu_queue(monkeypatch):
+    """Run the real v3 controller/backend/engine; replace only process and GPU boundaries."""
+    from concurrent.futures import Future
+
+    from satchange import hpo_backend, hpo_core
+
+    calls = []
+    monkeypatch.setattr(hpo_backend, "resolve_device", lambda _: torch.device("cpu"))
+    monkeypatch.setattr(
+        hpo_backend.SatelliteBackend,
+        "resources",
+        lambda self, devices: {
+            device: {"free_mib": 16000, "total_mib": 16000} for device in devices
+        },
+    )
+
+    class Executor:
+        def __init__(self, **kwargs):
+            assert kwargs["max_workers"] == 1
+            assert kwargs["mp_context"].get_start_method() == "spawn"
+
+        def submit(self, operation, *args):
+            calls.append({"job": args[1], "device": args[2], "gpu_ids": [0, 1, 2, 3]})
+            future = Future()
+            try:
+                future.set_result(operation(*args))
+            except Exception as error:
+                future.set_exception(error)
+            return future
+
+        def shutdown(self, **kwargs):
+            pass
+
+    monkeypatch.setattr(hpo_core, "ProcessPoolExecutor", Executor)
+    return calls
+
+
+def search(**args):
+    """Exercise the explicit v3 phases; attach only convenience fields used by these tests."""
+    stage = args.pop("stage", None)
+    if stage is None:
+        hpo.run_automated_parameter_search(**args, stage="hpo")
+        stage = "seeds"
+    result = hpo.run_automated_parameter_search(**args, stage=stage)
+    if "ranking" in result:
+        for item in result["ranking"]:
+            seed = args["base_parameters"]["seed"]
+            item["deployment_seed"] = seed
+            item["deployment_best_checkpoint"] = next(
+                row["best_checkpoint"]
+                for row in item["seed_results"]
+                if row["parameters"]["seed"] == seed
+            )
+        if args.get("task") == SEMANTIC_TASK:
+            result["selected_by_mean_selection_score"] = result["selected"]
+    return result
+
+
+def options(tmp_path, name="search", epochs=2):
+    manifest, _ = make_manifest(tmp_path)
+    prepared = tmp_path / "prepared/prepared.json"
+    if not prepared.exists():
+        prepare_dataset(config(), manifest, prepared.parent)
+    return dict(
+        stages=[("baseline", {"learning_rate": [0.001]})],
+        base_parameters=hpo_parameters(config(), prepared),
+        task=SEMANTIC_TASK,
+        model_name="tiny_siamese_change_classifier",
+        dataset_root=prepared,
+        stage_epochs=[epochs, epochs, epochs],
+        additional_seeds=False,
+        output_root=tmp_path / "searches",
+        search_name=name,
+        gpu_ids=[0, 1, 2, 3],
+        verbose=False,
+        checkpoint_interval=1,
+    )
+
+
+def test_actual_hpo_inference_cache_and_exact_epoch_resume(tmp_path, cpu_queue):
+    args = options(tmp_path)
+    first = search(**args)
+    chosen = first["selected_by_mean_selection_score"]
+    assert chosen["n_seeds"] == 1 and chosen["std_selection_score"] is None
+    assert "mean_iou" not in chosen and "best_iou" not in chosen["seed_results"][0]
+    path = Path(chosen["deployment_best_checkpoint"])
+    model, best = load_classifier(path)
+    history = best["history"]
+    score = max(row["selection_score"] for row in history)
+    assert best["epoch"] == next(r["epoch"] for r in history if r["selection_score"] == score)
+    selected = chosen["seed_results"][0]
+    assert selected["selected_metrics"]["joint_accuracy"] == best["best_metrics"]["joint_accuracy"]
+    assert best["task"] == SEMANTIC_TASK and not best.get("thresholds")
+    assert "tiny_siamese_change_classifier" not in MODEL_REGISTRY
+    assert evaluate_classifier(path, args["dataset_root"], "val")["selection_score"] == score
+    before, after, probabilities = scene_tensors(17, 23)
+    assert predict_checkpoint(path, before, after, probabilities)["predictions"] == []
+    probabilities[0, 2:5, 3:6] = 0.7
+    result = predict_checkpoint(
+        path,
+        before,
+        after,
+        probabilities,
+        inference_mask_source="predicted",
+        detector={"identity": "synthetic frozen detector"},
+    )
+    assert result["training_mask_source"] == "ground_truth"
+    assert result["inference_mask_source"] == "predicted" and len(result["predictions"]) == 1
+    assert result["inference_precision"] == "fp32"
+    with pytest.raises(ValueError, match="bf16"):
+        predict_checkpoint(
+            path,
+            before,
+            after,
+            probabilities,
+            precision="bf16",
+            inference_mask_source="predicted",
+            detector={"identity": "test"},
+        )
+    count = len(cpu_queue)
+    search(**args)
+    assert len(cpu_queue) == count  # Completed cache does not re-enter a worker.
+
+    one = search(**options(tmp_path, "one", 1))
+    initial = Path(one["ranking"][0]["deployment_best_checkpoint"]).parent / "last.pt"
+    resumed = search(**options(tmp_path, "continued", 2), resume_from=initial)
+    state = load_training_checkpoint(
+        Path(resumed["ranking"][0]["deployment_best_checkpoint"]).parent / "last.pt"
+    )
+    full = load_training_checkpoint(path.parent / "last.pt")
+    assert state["epoch"] == 2 and state["best_metrics"]["epoch"] == full["best_metrics"]["epoch"]
+    for key, value in full["model_state_dict"].items():
+        torch.testing.assert_close(value, state["model_state_dict"][key], rtol=0, atol=0)
+    for left, right in zip(full["history"], state["history"], strict=True):
+        for key in (
+            "train_loss",
+            "validation_loss",
+            "selection_score",
+            "before_confusion",
+            "after_confusion",
+        ):
+            assert left[key] == right[key]
+    assert cpu_queue[-1]["gpu_ids"] == [0, 1, 2, 3]
+
+
+def test_failed_trial_resumes_own_last_and_preserves_best(tmp_path, cpu_queue, monkeypatch):
+    args = options(tmp_path, "interrupt", 3)
+    save = optimization._save_checkpoint
+    from satchange import training
+
+    original = training._save_checkpoint
+    interrupted = False
+
+    def stop_after_epoch(path=None, state=None, **kwargs):
+        nonlocal interrupted
+        path = path or kwargs["checkpoint_path"]
+        state = state or kwargs["checkpoint_state"]
+        original(path, state)
+        if Path(path).name == "last.pt" and state["epoch"] == 1 and not interrupted:
+            interrupted = True
+            raise RuntimeError("synthetic interruption after saved epoch")
+
+    monkeypatch.setattr(training, "_save_checkpoint", stop_after_epoch)
+    with pytest.raises(RuntimeError, match=r"trial\(s\) failed"):
+        search(**args)
+    monkeypatch.setattr(training, "_save_checkpoint", original)
+    result = search(**args)
+    row = json.loads(Path(result["ranking"][0]["seed_results"][0]["summary_path"]).read_text())
+    assert row["resumed_from_epoch"] == 1 and row["new_epochs_completed"] == 2
+    assert Path(row["resume_source"]).name == "last.pt"
+    assert Path(result["ranking"][0]["deployment_best_checkpoint"]).is_file()
+    assert save is optimization._save_checkpoint
+
+
+def test_semantic_parameters_stages_seed_families_and_protocol_guard(tmp_path, cpu_queue):
+    args = options(tmp_path, epochs=1)
+    args["stages"] = [
+        ("baseline", {"learning_rate": [0.001]}),
+        ("dropout", {"model__dropout": [0.0, 0.2]}),
+    ]
+    search(**args, stage="baseline")
+    with pytest.raises(RuntimeError, match="incomplete"):
+        search(**args, stage="seeds")
+    search(**args, stage="dropout")
+    summary = search(**args, stage="seeds")
+    for row in summary["ranking"]:
+        params = row["parameters"]
+        assert (
+            "semantic_config" in params and "prepared_identity" in params and "seed" not in params
+        )
+    from satchange.hpo_backend import SatelliteBackend
+    from satchange.hpo_core import identity as trial_identity
+
+    identity = SatelliteBackend().normalize(args["base_parameters"])
+    altered = deepcopy(identity)
+    altered["model_parameters"]["dropout"] = 0.4
+    assert trial_identity(identity) != trial_identity(altered)
+    altered = deepcopy(identity)
+    altered["semantic_config"]["roi"]["context_pixels"] = 2
+    assert trial_identity(identity) != trial_identity(altered)
+    protocol = args["output_root"] / "search" / "pipeline_protocol.json"
+    previous = protocol.read_bytes()
+    broken = json.loads(previous)
+    broken["source_sha256"] = "unverified old Heavy source"
+    protocol.write_text(json.dumps(broken), encoding="utf-8")
+    snapshot = protocol.read_bytes()
+    with pytest.raises(ValueError, match="changed"):
+        search(**args)
+    assert protocol.read_bytes() == snapshot
+
+
+def test_rgb_pair_leak_reversal_copy_and_blank_masks(tmp_path):
+    manifest, scenes = make_manifest(tmp_path)
+    scenes[-1]["before"], scenes[-1]["after"] = scenes[0]["after"], scenes[0]["before"]
+    manifest.write_text(json.dumps({"scenes": scenes}), encoding="utf-8")
+    with pytest.raises(ValueError, match="RGB pair leakage"):
+        read_manifest(manifest, data_protocol(config()))
+    for key in ("before", "after"):
+        source = tmp_path / scenes[-1][key]
+        copy = tmp_path / f"copied_{key}.png"
+        copy.write_bytes(source.read_bytes())
+        scenes[-1][key] = copy.name
+    manifest.write_text(json.dumps({"scenes": scenes}), encoding="utf-8")
+    with pytest.raises(ValueError, match="RGB pair leakage"):
+        read_manifest(manifest, data_protocol(config()))
+
+
+def second_fixture(root, count=5):
+    for i in range(count):
+        for folder in ("im1", "im2", "label1", "label2"):
+            target = root / "train" / folder
+            target.mkdir(parents=True, exist_ok=True)
+            if folder.startswith("im"):
+                array = np.full((512, 512, 3), 20 + i + (50 if folder == "im2" else 0), np.uint8)
+            else:
+                array = np.full((512, 512, 3), 255, np.uint8)
+                array[2:5, 3:7] = (0, 255, 0)  # Same-class tree change remains changed.
+            Image.fromarray(array).save(target / f"{i:03d}.png")
+
+
+def second_config():
+    cfg = config()
+    cfg["classes"] = list(SECOND_CLASSES)
+    cfg["class_ids"] = {i + 1: name for i, name in enumerate(SECOND_CLASSES)}
+    cfg["ignore_ids"] = [0]
+    return cfg
+
+
+def test_second_exact_palette_ids_support_stable_cache(tmp_path):
+    colors = np.asarray(list(SECOND_COLORS), dtype=np.uint8).reshape(1, 7, 3)
+    path = tmp_path / "palette.png"
+    Image.fromarray(colors).save(path)
+    np.testing.assert_array_equal(
+        decode_label(path), np.asarray(list(SECOND_COLORS.values())).reshape(1, 7)
+    )
+    Image.fromarray(colors).convert("P", palette=Image.Palette.ADAPTIVE).save(path)
+    np.testing.assert_array_equal(
+        decode_label(path), np.asarray(list(SECOND_COLORS.values())).reshape(1, 7)
+    )
+    Image.fromarray(np.array([[1, 2]], dtype=np.uint8)).save(path)
+    with pytest.raises(ValueError, match="explicit"):
+        decode_label(path)
+    np.testing.assert_array_equal(decode_label(path, {1: "water", 2: "tree"}), [[4, 2]])
+    Image.fromarray(np.zeros((1, 1, 3), np.uint8)).save(path)
+    with pytest.raises(ValueError, match="Unknown SECOND colors"):
+        decode_label(path)
+    root = tmp_path / "SECOND"
+    second_fixture(root)
+    assert bootstrap_second(root) == root
+    with pytest.raises(ValueError, match="local_holdout"):
+        prepare_second(root, tmp_path / "cache", second_config())
+    prepared = prepare_second(root, tmp_path / "cache", second_config(), local_holdout=True)
+    doc = json.loads(prepared.read_text())
+    assert doc["second"]["local_splits"] == ["val", "test"]
+    assert not doc["second"]["geographic_independence_verified"]
+    assert all(r["before_label"] == r["after_label"] == 1 for r in doc["records"])
+    assert prepare_second(root, tmp_path / "cache", second_config(), local_holdout=True) == prepared
+    other = second_config()
+    other["roi"]["context_pixels"] = 2
+    assert prepare_second(root, tmp_path / "cache", other, local_holdout=True) != prepared
+    label = root / "train/label2/000.png"
+    Image.fromarray(np.full((512, 512, 3), 255, np.uint8)).save(label)
+    with pytest.raises(ValueError, match="Inconsistent"):
+        prepare_second(root, tmp_path / "cache", second_config(), local_holdout=True)
+
+
+def test_device_indices_and_detector_explicit_output_geometry(monkeypatch):
+    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
+    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
+    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
+    assert resolve_device("cuda") == torch.device("cuda:2")
+    assert resolve_device("cuda:3") == torch.device("cuda:3")
+    with pytest.raises(ValueError, match="not visible"):
+        resolve_device("cuda:4")
+
+    class Detector(torch.nn.Module):
+        def __init__(self):
+            super().__init__()
+            self.parameter = torch.nn.Parameter(torch.tensor(0.0))
+
+        def forward(self, before, after):
+            return torch.zeros_like(before[:, :1]) + self.parameter
+
+    detector = Detector()
+    before, after, _ = scene_tensors(17, 23)
+    output = detector_probability_map(
+        detector, before, after, input_size=(8, 8), output_kind="logits"
+    )
+    assert output.shape == (1, 17, 23)
+    torch.testing.assert_close(output, torch.full_like(output, 0.5))
+    assert not detector.training and detector.parameter.grad is None
+    output = detector_probability_map(
+        detector, before, after, input_size=(8, 8), output_kind="probabilities"
+    )
+    assert (output == 0).all()  # No double sigmoid or heuristic.
+    with pytest.raises(ValueError, match="Unsupported"):
+        detector_probability_map(
+            detector, before, after, input_size=(8, 8), output_kind="logits", geometry="tiles"
+        )
+
+
+def test_three_seeds_family_and_predicted_evaluation(tmp_path, cpu_queue):
+    args = options(tmp_path, epochs=1)
+    summary = search(**dict(args, additional_seeds=True))
+    family = summary["ranking"][0]
+    assert family["n_seeds"] == 3 and family["std_selection_score"] is not None
+    assert [r["parameters"]["seed"] for r in family["seed_results"]] == [42, 123, 777]
+    checkpoint = Path(family["deployment_best_checkpoint"])
+    assert family["deployment_seed"] == 42
+    cfg = config()
+    cfg["mask_source"] = "predicted"
+    cfg["detector"] = {"identity": "synthetic saved probabilities", "training_groups": []}
+    scenes = json.loads((tmp_path / "manifest.json").read_text())["scenes"]
+    for scene in scenes:
+        path = tmp_path / scene["change"]
+        np.save(path.with_name(path.stem + "_predicted.npy"), np.load(path) * 0.7)
+        scene["change"] = path.stem + "_predicted.npy"
+    alternate = tmp_path / "predicted_manifest.json"
+    alternate.write_text(json.dumps({"scenes": scenes}))
+    prepare_dataset(cfg, alternate, tmp_path / "predicted")
+    prepared = tmp_path / "predicted/prepared.json"
+    with pytest.raises(ValueError, match="explicit evaluation_config"):
+        evaluate_classifier(checkpoint, prepared, "val")
+    metrics = evaluate_classifier(
+        checkpoint, prepared, "val", evaluation_config=cfg, precision="fp32"
+    )
+    assert metrics["domain_shift"] and metrics["evaluation_mask_source"] == "predicted"
+    assert metrics["training_data_identity"] != metrics["evaluation_data_identity"]
+    assert metrics["evaluation_precision"] == "fp32" and metrics["precision_override"] == "fp32"
+
+
+def test_semantic_scheduler_early_stop_and_resume_validation(tmp_path, cpu_queue):
+    from satchange.training import EarlyStopping
+
+    args = options(tmp_path, epochs=4)
+    args["base_parameters"]["scheduler"] = "reduce_on_plateau"
+    summary = search(**args, patience=1, min_delta=1.0, seed_patience=1)
+    selected = summary["ranking"][0]
+    row = selected["seed_results"][0]
+    assert row["stop_reason"] == "early_stopping" and row["epoch"] == 2
+    checkpoint = Path(selected["deployment_best_checkpoint"]).parent / "last.pt"
+    state = load_training_checkpoint(checkpoint)
+    assert state["scheduler_state_dict"] is not None
+    assert state["early_stopping"]["policy"]["monitor"] == "selection_score"
+    parameters = deepcopy(args["base_parameters"])
+    parameters["model_parameters"]["dropout"] = 0.9
+    with pytest.raises(ValueError, match="parameters differ"):
+        optimization.read_hpo_checkpoint(
+            checkpoint,
+            model_name=args["model_name"],
+            dataset_root=args["dataset_root"],
+            thresholds=[],
+            budget=4,
+            early_stopping=EarlyStopping(1, 1, 1.0, monitor="selection_score"),
+            parameters=parameters,
+        )
+    parameters = deepcopy(args["base_parameters"])
+    parameters.update(gpu_id=3, code_revision="another compatible host")
+    optimization.read_hpo_checkpoint(
+        checkpoint,
+        model_name=args["model_name"],
+        dataset_root=args["dataset_root"],
+        thresholds=[],
+        budget=4,
+        early_stopping=EarlyStopping(1, 1, 1.0, monitor="selection_score"),
+        parameters=parameters,
+    )
+
+
+def test_binary_controller_worker_schema_and_metric_regression(tmp_path, cpu_queue):
+    from satchange.data import create_dataset_dataloader
+    from satchange.losses import BCEDiceLoss
+    from satchange.run import reconstruct_checkpoint
+    from satchange.training import evaluate_thresholds
+
+    for split in ("train", "val"):
+        for folder in ("A", "B", "label"):
+            directory = tmp_path / "binary" / split / folder
+            directory.mkdir(parents=True)
+            pixels = np.zeros((32, 32, 3), np.uint8)
+            pixels[8:16, 8:16] = 255 if folder != "A" else 20
+            Image.fromarray(pixels).save(directory / "scene.png")
+    base = dict(
+        model_parameters={},
+        seed=42,
+        image_size=32,
+        batch_size=1,
+        num_workers=0,
+        learning_rate=0.001,
+        weight_decay=0.0001,
+        scheduler="none",
+        positive_weight=None,
+        bce_weight=1.0,
+        dice_weight=1.0,
+        precision="fp32",
+    )
+    args = dict(
+        stages=[("baseline", {"learning_rate": [0.001]}), ("next", {"weight_decay": [0.0001]})],
+        base_parameters=base,
+        model_name="fc_siam_diff",
+        dataset_root=tmp_path / "binary",
+        thresholds=[0.4, 0.5],
+        stage_epochs=[1, 1, 1],
+        additional_seeds=False,
+        output_root=tmp_path / "searches",
+        search_name="binary",
+        gpu_ids=[0],
+    )
+    search(**args, stage="baseline")
+    search(**args, stage="next")
+    summary = search(**args, stage="seeds")
+    family = summary["selected_by_mean_validation_f1"]
+    row = family["seed_results"][0]
+    assert "best_f1" in row and "best_iou" in row and "selection_score" not in row
+    model, state = reconstruct_checkpoint(family["deployment_best_checkpoint"], torch.device("cpu"))
+    assert "task" not in state and state["thresholds"] == [0.4, 0.5]
+    _, loader = create_dataset_dataloader(tmp_path / "binary", "val", 32, 1, 0)
+    metrics = evaluate_thresholds(model, loader, BCEDiceLoss(), torch.device("cpu"), [0.4, 0.5])
+    best = max(metrics["threshold_results"], key=lambda r: r["f1"])
+    assert row["best_f1"] == best["f1"] and row["best_iou"] == best["iou"]
+
+
+def test_second_archive_safe_and_official_download_contract(tmp_path, monkeypatch):
+    import zipfile
+
+    import gdown
+
+    source = tmp_path / "source"
+    second_fixture(source, count=1)
+    archive = tmp_path / "SECOND.zip"
+    with zipfile.ZipFile(archive, "w") as bundle:
+        for path in source.rglob("*.png"):
+            bundle.write(path, str(path.relative_to(source)))
+    assert bootstrap_second(tmp_path / "local", archive=archive).is_dir()
+    calls = []
+
+    def download(**kwargs):
+        calls.append(kwargs)
+        Path(kwargs["output"]).write_bytes(archive.read_bytes())
+        return kwargs["output"]
+
+    monkeypatch.setattr(gdown, "download", download)
+    installed = bootstrap_second(tmp_path / "downloaded", allow_download=True)
+    assert calls[0]["id"] == "1mN8jzCKKK27p3ODGoDgepjiRYGQpB34u"
+    assert "fuzzy" not in calls[0]
+    assert json.loads((installed / "source.json").read_text())["checksum_is_local"]
+    malicious = tmp_path / "bad.zip"
+    with zipfile.ZipFile(malicious, "w") as bundle:
+        bundle.writestr("../escaped.txt", "synthetic")
+    with pytest.raises(ValueError, match="Unsafe"):
+        bootstrap_second(tmp_path / "unsafe", archive=malicious)
+    assert not (tmp_path / "escaped.txt").exists()
+
+
+def test_actual_v3_queue_dispatch_with_mock_resources(tmp_path, cpu_queue):
+    args = options(tmp_path, epochs=1)
+    args["stages"] = [("baseline", {"learning_rate": [0.001, 0.002, 0.003, 0.004, 0.005]})]
+    args["selection_gap"] = 1.0
+    summary = search(**args)
+    assert len(cpu_queue) == 5
+    assert {call["device"] for call in cpu_queue} == {"cuda:0", "cuda:1", "cuda:2", "cuda:3"}
+    assert len(summary["ranking"]) == 5
+
+
+def test_v3_source_fingerprint_includes_semantic_and_controller(monkeypatch):
+    from satchange.hpo_backend import SatelliteBackend
+
+    original = Path.read_bytes
+    baseline = SatelliteBackend().code_signature()
+
+    def changed(path):
+        content = original(path)
+        return content + b"\n# synthetic change" if path.name == "roi.py" else content
+
+    monkeypatch.setattr(Path, "read_bytes", changed)
+    assert SatelliteBackend().code_signature() != baseline
+
+
+def test_inference_only_best_export_remains_loadable(tmp_path, cpu_queue):
+    from satchange.training import restore_best_checkpoint
+
+    args = options(tmp_path)
+    summary = search(**args)
+    directory = Path(summary["ranking"][0]["deployment_best_checkpoint"]).parent
+    state = load_training_checkpoint(directory / "last.pt")
+    first = load_training_checkpoint(
+        directory / "checkpoints/tiny_siamese_change_classifier_epoch_0001.pt"
+    )
+    state["best_metrics"] = first["best_metrics"]
+    state["best_model_state_dict"] = first["model_state_dict"]
+    export = tmp_path / "isolated/best.pt"
+    restore_best_checkpoint(state, tmp_path / "isolated/last.pt", export)
+    model, loaded = load_classifier(export)
+    assert loaded["checkpoint_kind"] == "selection_weights" and loaded["task"] == SEMANTIC_TASK
+    for key, tensor in model.state_dict().items():
+        torch.testing.assert_close(tensor, first["model_state_dict"][key])
+    with pytest.raises(ValueError, match="complete training checkpoint"):
+        load_training_checkpoint(export)
+
+
+def test_v3_three_phase_promotion_final_benchmark_and_validation_gate(
+    tmp_path, cpu_queue, monkeypatch
+):
+    """Real small CPU epochs verify 1 -> 2 -> 3 totals, final export, and val-before-test."""
+    from satchange.final_benchmark import run_final_benchmark
+    from satchange.semantic import benchmark as semantic_benchmark
+
+    args = options(tmp_path, epochs=1)
+    args.update(stage_epochs=[1, 2, 3], additional_seeds=True, patience=1)
+    hpo.run_automated_parameter_search(**args, stage="hpo")
+    seeds = hpo.run_automated_parameter_search(**args, stage="seeds")
+    assert seeds["selected"]["seeds"] == [42, 123, 777]
+    plan = hpo.run_automated_parameter_search(**args, stage="final", dry_run=True)
+    assert len(plan["jobs"]) == 3
+    assert all(job["source"]["epoch"] == 2 and job["target_epoch"] == 3 for job in plan["jobs"])
+    final = hpo.run_automated_parameter_search(**args, stage="final")
+    assert final["complete"] and all(row["epoch"] == 3 for row in final["results"])
+    assert all(row["policy"] is None for row in final["results"])
+    assert len(cpu_queue) == 7  # 1 HPO + 3 seeds + 3 final; no fourth seed.
+    for row in final["results"]:
+        last = load_training_checkpoint(row["last_checkpoint"])
+        assert [r["epoch"] for r in last["history"]] == [1, 2, 3]
+        assert row["new_epochs_completed"] == 1
+        assert last["early_stopping"]["policy"] is None
+    saved_calls = len(cpu_queue)
+    hpo.run_automated_parameter_search(**args, stage="final")
+    assert len(cpu_queue) == saved_calls
+
+    calls = []
+    evaluate = semantic_benchmark.evaluate_classifier
+
+    def observed(*positional, **kwargs):
+        calls.append(positional[2])
+        return evaluate(*positional, **kwargs)
+
+    monkeypatch.setattr(semantic_benchmark, "evaluate_classifier", observed)
+    options_benchmark = dict(
+        task=SEMANTIC_TASK,
+        variant="tiny_test",
+        checkpoints=final["best_checkpoints"],
+        project_root=tmp_path,
+        dataset_root=args["dataset_root"],
+        training_results=final,
+        deployment_seed=42,
+        device="cpu",
+        warmup=0,
+        repetitions=2,
+    )
+    report = run_final_benchmark(**options_benchmark, output_dir=tmp_path / "final_bundle")
+    assert calls == ["val"] * 3 + ["test"] * 3
+    assert report["metadata"]["deployment_seed"] == 42
+    assert report["final_test_results"]["test_selection_score_std"] is not None
+    assert "test_iou_mean" not in report["final_test_results"]
+    assert report["benchmark"]["protocol"]["scope"].startswith("classifier only")
+    out = Path(report["output_dir"])
+    for path in (
+        "metadata.json",
+        "benchmark.json",
+        "training_results.json",
+        "validation_results.json",
+        "final_test_results.json",
+        "deployment_weights.pt",
+    ):
+        assert (out / path).is_file()
+    state = torch.load(out / "deployment_weights.pt", map_location="cpu", weights_only=True)
+    assert "optimizer_state_dict" not in state
+
+    calls.clear()
+
+    def changed_validation(*positional, **kwargs):
+        result = observed(*positional, **kwargs)
+        result["selection_score"] += 0.2
+        return result
+
+    monkeypatch.setattr(semantic_benchmark, "evaluate_classifier", changed_validation)
+    with pytest.raises(ValueError, match="No test evaluation"):
+        run_final_benchmark(**options_benchmark, output_dir=tmp_path / "rejected_bundle")
+    assert calls == ["val"] and not (tmp_path / "rejected_bundle").exists()
+
+
+def test_controller_lock_is_exclusive_and_released(tmp_path):
+    from satchange.hpo_core import experiment_lock
+
+    with experiment_lock(tmp_path):
+        with pytest.raises(RuntimeError, match="Another HPO"):
+            with experiment_lock(tmp_path):
+                pass
+    with experiment_lock(tmp_path):
+        pass
```

## notebooks/tiny_siamese_second_staged_resume.ipynb

### Markdown cell 0

# Tiny Siamese — SECOND: HPO v3 → seeds → final → benchmark

Порядок основан на Heavy_Full_HPO_v3_metadata_fix.ipynb. После обновления — Restart Kernel.
Абсолютные бюджеты [40,60,80]: HPO 40, seed validation 60, final training 80.
40→60 = ещё 20 эпох. Final: seeds 42,123,777, без early stopping. Test отдельно
через run_final_benchmark. По умолчанию данные и реальные запуски отключены.
GT ROI baseline измеряет семантику известных регионов, а не качество всей detector-системы.


### Cell 1

```python
import json
import sys
from pathlib import Path

ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src/satchange").is_dir()), None)
if ROOT is None:
    raise FileNotFoundError("Open this notebook inside satellite-change-intelligence.")
if not (3, 11) <= sys.version_info[:2] < (3, 13):
    raise RuntimeError("Use the project Python 3.11/3.12 kernel.")
sys.path.insert(0, str(ROOT / "src"))
print("Repository:", ROOT, "Interpreter:", sys.executable)
```

### Markdown cell 2

Все изменяемые пути и флаги — здесь. `LOCAL_HOLDOUT=True` явно разрешает фиксированный local
holdout из train, если размеченного val/test нет. Это не официальный test и не доказанная
географическая независимость. `SPLIT_SEED` не зависит от training seeds.
Для scalar indexed labels подтвердите документированную mapping локальной копии через
`USE_DOCUMENTED_INDEXED_IDS`; RGB/palette PNG проверяются автоматически.

### Cell 3

```python
DATA_ROOT = ROOT / "data/SECOND"
CACHE_ROOT = ROOT / "data/SECOND_semantic_cache"
CONFIG_FILE = ROOT / "configs/tiny_siamese_change_classifier.yaml"
LOCAL_ARCHIVE = None  # Path to a local official ZIP, or use the installed DATA_ROOT.
ALLOW_DOWNLOAD = False
PREPARE_DATA = False
LOCAL_HOLDOUT = True
SPLIT_SEED = 2099
USE_DOCUMENTED_INDEXED_IDS = False
SOURCE_GROUPS = None  # Optional mapping sample_id -> verified source scene group.
SENSOR_METADATA = None  # Only metadata actually available from your source.
INSTALL_MISSING = False  # Installs small missing notebook/data packages; never torch.

GPU_IDS = [0, 1, 2, 3]  # Visible numbering under the existing CUDA_VISIBLE_DEVICES.
HPO_ROOT = ROOT / "artifacts/hyperparameter_search/semantic/tiny_siamese_second"
SEARCH_NAME = "second_oracle_hpo_v3_40_60_80"
SEARCH_DIRECTORY = HPO_ROOT / SEARCH_NAME
STAGE_EPOCHS = [40, 60, 80]  # ABSOLUTE HPO / seeds / final targets.
FINAL_SEEDS = [42, 123, 777]
DEPLOYMENT_SEED = 42  # Fixed before test.
REUSE_DIRECTORIES = []  # Own compatible Tiny searches only.
CHECKPOINTS = []  # Own complete state checkpoints; automatic recovery prefers last.pt.
ALLOW_LEGACY_IMPORT = False
RESTORE_SAVED_PROTOCOL = True
SEED_QUEUE_POLICY = "all"  # Fresh search: all. finish_started requires started secondary seeds.
SELECTION_GAP = 0.04
PATIENCE = None
MIN_EPOCHS = 1
MIN_DELTA = 0.0
RUN_HPO_BLOCKS = False
PLAN_SEEDS = False
RUN_SEEDS = False
PLAN_FINAL = False
RUN_FINAL_TRAINING = False
RUN_TEST_BENCHMARK = False
RUN_PIPELINE_EXAMPLE = False

CLASSIFIER_CHECKPOINT = None  # None -> selected deployment best.pt once summary exists.
DETECTOR_CHECKPOINT = None
DETECTOR_INPUT_SIZE = (256, 256)  # Match this detector's actual whole-image preprocessing.
DETECTOR_OUTPUT_KIND = "logits"
DETECTOR_PRECISION = None  # Use saved detector precision; explicit override is recorded.
EXAMPLE_BEFORE = None
EXAMPLE_AFTER = None
EXAMPLE_MAP = None
EXAMPLE_MASK_SOURCE = "ground_truth"  # Or explicit predicted with DETECTOR_IDENTITY.
DETECTOR_IDENTITY = None
EVALUATION_PRECISION = None  # None reproduces the saved numerical mode.
```

### Cell 4

```python
import importlib.util
import subprocess

requirements = {
    "numpy": "numpy",
    "PIL": "pillow",
    "cv2": "opencv-python",
    "yaml": "PyYAML",
    "gdown": "gdown",
    "matplotlib": "matplotlib",
}
missing = [
    package for module, package in requirements.items() if importlib.util.find_spec(module) is None
]
if missing and INSTALL_MISSING:
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
elif missing:
    raise RuntimeError(
        f"Missing dependencies {missing}; select INSTALL_MISSING in the config cell."
    )
```

### Cell 5

```python
import torch

import satchange
from satchange.semantic.data import SemanticROIDataset, read_map, read_rgb
from satchange.semantic.second import bootstrap_second, discover_second, prepare_second
from satchange.utils import load_config

print("Package:", satchange.__file__, "Torch:", torch.__version__)
print("Visible CUDA GPUs:", torch.cuda.device_count())
CONFIG = load_config(CONFIG_FILE)
protocol_path = SEARCH_DIRECTORY / "pipeline_protocol.json"
SAVED_PROTOCOL = (
    json.loads(protocol_path.read_text())
    if RESTORE_SAVED_PROTOCOL and protocol_path.is_file()
    else None
)
if SAVED_PROTOCOL:
    CONFIG.update(SAVED_PROTOCOL["base"]["semantic_config"])
    CONFIG["dropout"] = SAVED_PROTOCOL["base"]["model_parameters"]["dropout"]
    for key in (
        "seed",
        "batch_size",
        "num_workers",
        "learning_rate",
        "weight_decay",
        "scheduler",
        "class_weights",
        "precision",
        "tf32",
    ):
        CONFIG["training"][key] = SAVED_PROTOCOL["base"][key]
print("Saved protocol loaded:", SAVED_PROTOCOL is not None)
if SAVED_PROTOCOL and (
    SAVED_PROTOCOL["context"].get("task") != "semantic_roi_classification"
    or SAVED_PROTOCOL["context"]["model_name"] != "tiny_siamese_change_classifier"
):
    raise ValueError("Saved search is not a Tiny semantic experiment.")
```

### Markdown cell 6

Проверка SECOND выполняется до запуска очереди. При установленном комплекте bootstrap пропускает
скачивание. Download использует ссылку страницы [авторов SECOND](https://captain-whu.github.io/SCD/);
quota/login ошибки требуют доступного локального ZIP. Local SHA256 — наша provenance, не checksum авторов.

### Cell 7

```python
PREPARED = None
if PREPARE_DATA:
    installed = bootstrap_second(DATA_ROOT, archive=LOCAL_ARCHIVE, allow_download=ALLOW_DOWNLOAD)
    layout = discover_second(installed)
    print("Labeled scenes by supplied split:", {k: len(v) for k, v in layout.items()})
    PREPARED = prepare_second(
        installed,
        CACHE_ROOT,
        CONFIG,
        split_seed=SPLIT_SEED,
        local_holdout=LOCAL_HOLDOUT,
        indexed_ids=CONFIG["dataset"]["indexed_id_names"] if USE_DOCUMENTED_INDEXED_IDS else None,
        source_groups=SOURCE_GROUPS,
        sensor_metadata=SENSOR_METADATA,
    )
    print("Verified prepared cache:", PREPARED)
elif SAVED_PROTOCOL and Path(SAVED_PROTOCOL["dataset_root"]).is_file():
    PREPARED = Path(SAVED_PROTOCOL["dataset_root"])
    SemanticROIDataset(PREPARED, "train", CONFIG).verify_sources()
else:
    print("Preparation disabled. Enable PREPARE_DATA after choosing the source.")
```

### Cell 8

```python
if PREPARED is not None:
    document = json.loads(PREPARED.read_text())
    print(json.dumps(document["coverage"], indent=2))
    print("Split provenance:", document.get("second", {}).get("local_splits"))
    print("ROI sizes [H,W] are native; visual display alone may scale them.")
    import matplotlib.pyplot as plt

    dataset = SemanticROIDataset(PREPARED, "train", CONFIG)
    for index in range(min(3, len(dataset))):
        sample = dataset[index]
        fig, axes = plt.subplots(1, 3, figsize=(10, 3))
        for ax, key in zip(axes, ("before", "after", "mask"), strict=True):
            values = (
                sample[key].permute(1, 2, 0).numpy() if key != "mask" else sample[key][0].numpy()
            )
            ax.imshow(values, cmap="gray" if key == "mask" else None)
            ax.set_title(key)
            ax.axis("off")
        fig.suptitle(
            f"{sample['sample_id']}: {CONFIG['classes'][sample['before_label']]} -> "
            f"{CONFIG['classes'][sample['after_label']]}"
        )
        plt.show()
else:
    print("No prepared cache selected; examples skipped.")
```

### Markdown cell 9

Безопасный implementation smoke: разные rectangular/odd ROI без resize/padding, shared
backbone `32→64→128`, feature grid примерно `H/4 × W/4`, маска уменьшается только для pooling,
raw logits `[B,C]`, две CE. Logical batch хранится списком, logits объединяются перед loss.
BatchNorm обновляется на temporal pair, не на всём logical minibatch. Это ограничение baseline.

### Cell 10

```python
from satchange.models import TinySiameseChangeClassifier, count_model_parameters
from satchange.semantic.training import logical_forward, semantic_loss

previous_threads = torch.get_num_threads()
torch.set_num_threads(1)
try:
    torch.manual_seed(42)
    model = TinySiameseChangeClassifier(CONFIG["classes"], CONFIG["dropout"])
    samples = [
        {
            "before": torch.rand(3, h, w),
            "after": torch.rand(3, h, w),
            "mask": torch.ones(1, h, w),
            "before_label": 0,
            "after_label": 1,
        }
        for h, w in [(17, 23), (19, 29)]
    ]
    logits, before_labels, after_labels = logical_forward(model, samples, torch.device("cpu"))
    loss, _ = semantic_loss(logits, before_labels, after_labels)
    loss.backward()
    assert torch.isfinite(loss) and all(
        p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
    )
    print("Synthetic shapes:", {k: tuple(v.shape) for k, v in logits.items()})
    print("Finite synthetic CE:", float(loss.detach()), count_model_parameters(model))
finally:
    torch.set_num_threads(previous_threads)
    del model
```

### Markdown cell 11

## Общие параметры v3

Native ROI → shared CNN → две группы logits [B,C] → две CE. Отбор: среднее macro F1
двух дат по всем объявленным классам. Batch/LR → LR → weight decay/dropout → scheduler.
BCE/Dice, image_size и threshold grid здесь не используются. Один worker на GPU;
пул четырёх GPU не добавляет четвёртый seed. Старые Heavy weights несовместимы.
После Restart читается сохранённая конфигурация и активные абсолютные бюджеты.


### Cell 12

```python
import inspect

from satchange.automated_hpo import run_automated_parameter_search
from satchange.semantic.training import hpo_parameters
from satchange.training import fit_binary_change_model

if "stage_epochs" not in inspect.signature(run_automated_parameter_search).parameters:
    raise RuntimeError("Install merged HPO v3 source and Restart Kernel.")
if not {"task", "stop_file"}.issubset(inspect.signature(fit_binary_change_model).parameters):
    raise RuntimeError("Shared trainer lacks Tiny/v3 support; Restart Kernel.")
STAGES = [
    ("01_batch_lr", {"batch_size": [4, 8], "learning_rate": [0.0005, 0.001]}),
    ("02_lr", {"learning_rate": [0.00075, 0.001]}),
    ("03_wd_dropout", {"weight_decay": [0.0001, 0.001], "model__dropout": [0.0, 0.1]}),
    ("04_scheduler", {"scheduler": ["none", "reduce_on_plateau"]}),
]
BASE = hpo_parameters(CONFIG, PREPARED) if PREPARED is not None else None
if SAVED_PROTOCOL:
    STAGES, BASE = SAVED_PROTOCOL["stages"], SAVED_PROTOCOL["base"]
    FINAL_SEEDS = SAVED_PROTOCOL["seeds"]
    SELECTION_GAP = SAVED_PROTOCOL["gaps"]["selection_score"]
    REUSE_DIRECTORIES = SAVED_PROTOCOL["reuse_from"]
    CHECKPOINTS = SAVED_PROTOCOL["initial_checkpoints"]
    target_file = SEARCH_DIRECTORY / "active_phase_targets.json"
    targets = json.loads(target_file.read_text()) if target_file.is_file() else {}
    STAGE_EPOCHS = [
        targets.get(phase, count)
        for phase, count in zip(("hpo", "seeds", "final"), STAGE_EPOCHS, strict=True)
    ]
OPTIONS = dict(
    task="semantic_roi_classification",
    stages=STAGES,
    base_parameters=BASE,
    model_name="tiny_siamese_change_classifier",
    dataset_root=PREPARED,
    stage_epochs=STAGE_EPOCHS,
    seeds=FINAL_SEEDS,
    additional_seeds=True,
    output_root=HPO_ROOT,
    search_name=SEARCH_NAME,
    gpu_ids=GPU_IDS,
    reuse_from=REUSE_DIRECTORIES,
    resume_from=CHECKPOINTS,
    resume_incomplete=True,
    allow_legacy_import=ALLOW_LEGACY_IMPORT,
    patience=PATIENCE,
    min_epochs=MIN_EPOCHS,
    min_delta=MIN_DELTA,
    seed_patience=None,
    selection_gap=SELECTION_GAP,
    joint_tiebreaker=False,
    checkpoint_interval=10,
    max_trial_memory_mb=80_000,
    oom_retry_delay_seconds=30,
    max_oom_attempts=12,
)
if (
    any((RUN_HPO_BLOCKS, PLAN_SEEDS, RUN_SEEDS, PLAN_FINAL, RUN_FINAL_TRAINING, RUN_TEST_BENCHMARK))
    and PREPARED is None
):
    raise FileNotFoundError("Prepare real SECOND first; no synthetic fallback.")
print("HPO / seeds / final targets:", STAGE_EPOCHS, "Final seeds:", FINAL_SEEDS)
print("Prepared:", PREPARED, "Search:", SEARCH_DIRECTORY)
```

### Markdown cell 13

## 01_batch_lr

Включите RUN_HPO_BLOCKS и выполните блок. Предыдущие блоки должны быть завершены.


### Cell 14

```python
if RUN_HPO_BLOCKS:
    hpo_result = run_automated_parameter_search(**OPTIONS, stage="hpo", hpo_stage="01_batch_lr")
    if not hpo_result["complete"]:
        raise RuntimeError("HPO paused; repeat this block before later phases.")
    print("Survivors:", len(hpo_result["survivors"]))
else:
    print("01_batch_lr disabled.")
```

### Markdown cell 15

## 02_lr

Включите RUN_HPO_BLOCKS и выполните блок. Предыдущие блоки должны быть завершены.


### Cell 16

```python
if RUN_HPO_BLOCKS:
    hpo_result = run_automated_parameter_search(**OPTIONS, stage="hpo", hpo_stage="02_lr")
    if not hpo_result["complete"]:
        raise RuntimeError("HPO paused; repeat this block before later phases.")
    print("Survivors:", len(hpo_result["survivors"]))
else:
    print("02_lr disabled.")
```

### Markdown cell 17

## 03_wd_dropout

Включите RUN_HPO_BLOCKS и выполните блок. Предыдущие блоки должны быть завершены.


### Cell 18

```python
if RUN_HPO_BLOCKS:
    hpo_result = run_automated_parameter_search(**OPTIONS, stage="hpo", hpo_stage="03_wd_dropout")
    if not hpo_result["complete"]:
        raise RuntimeError("HPO paused; repeat this block before later phases.")
    print("Survivors:", len(hpo_result["survivors"]))
else:
    print("03_wd_dropout disabled.")
```

### Markdown cell 19

## 04_scheduler

Включите RUN_HPO_BLOCKS и выполните блок. Предыдущие блоки должны быть завершены.


### Cell 20

```python
if RUN_HPO_BLOCKS:
    hpo_result = run_automated_parameter_search(**OPTIONS, stage="hpo", hpo_stage="04_scheduler")
    if not hpo_result["complete"]:
        raise RuntimeError("HPO paused; repeat this block before later phases.")
    print("Survivors:", len(hpo_result["survivors"]))
else:
    print("04_scheduler disabled.")
```

### Markdown cell 21

## План seed-фазы — без обучения

Включите PLAN_SEEDS и проверьте fresh/resume/reuse. Новый поиск требует policy all.
finish_started продолжает уже начатые вторичные seeds; неполные семьи не ранжируются.
dry_run записывает фиксированный план на диск.


### Cell 22

```python
if PLAN_SEEDS:
    seed_plan = run_automated_parameter_search(
        **OPTIONS,
        stage="seeds",
        pending_policy=SEED_QUEUE_POLICY,
        dry_run=True,
    )
    for item in seed_plan["jobs"]:
        saved = item["source"]["epoch"] if item["source"] else 0
        action = "reuse" if item["reused"] else "resume" if saved else "fresh"
        print(item["parameters"]["seed"], action, saved, "->", item["target_epoch"])
    print("Plan:", seed_plan["plan_path"], "Skipped:", seed_plan["skipped_trials"])
else:
    print("Seed plan disabled.")
```

### Markdown cell 23

## Seed validation — target=60

Сравниваются только полные группы с одинаковыми seeds/бюджетом/параметрами.
Победитель выбирается по validation mean per-date macro F1. Test не используется.


### Cell 24

```python
if RUN_SEEDS:
    seed_summary = run_automated_parameter_search(
        **OPTIONS,
        stage="seeds",
        pending_policy=SEED_QUEUE_POLICY,
    )
    if not seed_summary["complete"]:
        raise RuntimeError("Seed phase paused; repeat this cell before final training.")
else:
    path = SEARCH_DIRECTORY / "seed_validation_summary.json"
    seed_summary = json.loads(path.read_text()) if path.is_file() else None
if seed_summary:
    for rank, row in enumerate(seed_summary["ranking"], 1):
        print(rank, row["seeds"], row["mean_selection_score"], row["std_selection_score"])
    print("Incomplete families:", len(seed_summary["incomplete_families"]))
else:
    print("Seed validation summary absent.")
```

### Markdown cell 25

## Final — одна конфигурация, 80 эпох × 3 seeds

final_choice.json фиксируется до обучения и test. Каждый seed продолжает своё состояние
либо начинает с нуля. Early stopping отключён. Из другого kernel можно вызвать
request_hpo_pause(SEARCH_DIRECTORY); для продолжения повторите прежний вызов.
Target никогда не прибавляется сам.


### Cell 26

```python
if PLAN_FINAL:
    final_plan = run_automated_parameter_search(**OPTIONS, stage="final", dry_run=True)
    assert len(final_plan["jobs"]) == 3
    assert {job["parameters"]["seed"] for job in final_plan["jobs"]} == set(FINAL_SEEDS)
    for item in final_plan["jobs"]:
        saved = item["source"]["epoch"] if item["source"] else 0
        print(f"seed={item['parameters']['seed']}: {saved} -> {item['target_epoch']}")
    print("Final plan:", final_plan["plan_path"])
else:
    print("Final plan disabled.")
```

### Cell 27

```python
if RUN_FINAL_TRAINING:
    final_summary = run_automated_parameter_search(**OPTIONS, stage="final")
    if not final_summary["complete"]:
        raise RuntimeError("Final paused; repeat this cell before benchmark.")
else:
    path = SEARCH_DIRECTORY / "final_training_summary.json"
    final_summary = json.loads(path.read_text()) if path.is_file() else None
checkpoint = Path(CLASSIFIER_CHECKPOINT) if CLASSIFIER_CHECKPOINT else None
if final_summary:
    checkpoints = {
        int(seed): Path(path) for seed, path in final_summary["best_checkpoints"].items()
    }
    if set(checkpoints) != set(FINAL_SEEDS):
        raise RuntimeError("Incomplete final seed set.")
    if any(row["epoch"] != STAGE_EPOCHS[2] for row in final_summary["results"]):
        raise RuntimeError("Final results have a different target budget.")
    checkpoint = checkpoint or checkpoints[DEPLOYMENT_SEED]
    print("Final checkpoints:", checkpoints)
else:
    print("Final training disabled; no final summary selected.")
```

### Markdown cell 28

## run_final_benchmark — validation → test → скорость → export

Включите RUN_TEST_BENCHMARK после окончательного выбора и трёх завершённых final runs.
Сводка перечитывается с диска после Restart. best.pt может быть раньше 80-й эпохи,
если там был лучший validation. Строгую проверку validation сохраняйте включённой.
Сохраняются две confusion matrices, per-class metrics, macro F1 двух дат, joint accuracy,
coverage/provenance и sample std. Скорость измеряет classifier на первом native validation
ROI, без detector, extraction, I/O и device transfer; размеры ROI влияют на latency.


### Cell 29

```python
if RUN_TEST_BENCHMARK:
    from satchange.final_benchmark import run_final_benchmark

    summary_path = SEARCH_DIRECTORY / "final_training_summary.json"
    final_summary = json.loads(summary_path.read_text())
    checkpoints = {
        int(seed): Path(path) for seed, path in final_summary["best_checkpoints"].items()
    }
    if not final_summary["complete"] or set(checkpoints) != set(FINAL_SEEDS):
        raise RuntimeError("Benchmark requires all three completed final seeds.")
    if any(row["epoch"] != STAGE_EPOCHS[2] for row in final_summary["results"]):
        raise RuntimeError("Final target differs from notebook settings.")
    benchmark = run_final_benchmark(
        task="semantic_roi_classification",
        variant="tiny_siamese_second_oracle",
        checkpoints=checkpoints,
        project_root=ROOT,
        dataset_root=PREPARED,
        training_results=summary_path,
        deployment_seed=DEPLOYMENT_SEED,
        device=f"cuda:{GPU_IDS[0]}",
        evaluation_batch_size=1,
        num_workers=0,
        precision=EVALUATION_PRECISION or OPTIONS["base_parameters"]["precision"],
        warmup=20,
        repetitions=100,
        strict_validation=True,
        overwrite=False,
    )
    print("Final artifacts:", benchmark["output_dir"])
else:
    print("Final test disabled; enable RUN_TEST_BENCHMARK after final training.")
```

### Markdown cell 30

Pipeline example: original RGB → frozen detector → aligned probabilities → native crops.
Если detector checkpoint отсутствует, явно выбранная saved/GT map демонстрирует только вторую
часть pipeline. Detector preprocessing здесь поддерживает RGB/255 и whole-image resize;
для иных нормализаций, tiles/padding предоставьте map с известным обратным mapping.

### Cell 31

```python
from satchange.semantic.inference import predict_checkpoint, predict_detector_checkpoint

if RUN_PIPELINE_EXAMPLE:
    if checkpoint is None or EXAMPLE_BEFORE is None or EXAMPLE_AFTER is None:
        raise ValueError("Choose classifier checkpoint and aligned native RGB paths.")
    before, after = read_rgb(EXAMPLE_BEFORE), read_rgb(EXAMPLE_AFTER)
    if DETECTOR_CHECKPOINT is not None:
        prediction = predict_detector_checkpoint(
            checkpoint,
            DETECTOR_CHECKPOINT,
            before,
            after,
            input_size=DETECTOR_INPUT_SIZE,
            output_kind=DETECTOR_OUTPUT_KIND,
            device=f"cuda:{GPU_IDS[0]}" if torch.cuda.is_available() else "cpu",
            detector_precision=DETECTOR_PRECISION,
            classifier_precision=EVALUATION_PRECISION,
        )
    else:
        if EXAMPLE_MAP is None:
            raise ValueError("Choose aligned saved map; no detector is fabricated.")
        prediction = predict_checkpoint(
            checkpoint,
            before,
            after,
            read_map(EXAMPLE_MAP).unsqueeze(0),
            inference_mask_source=EXAMPLE_MASK_SOURCE,
            detector={"identity": DETECTOR_IDENTITY} if DETECTOR_IDENTITY else None,
            precision=EVALUATION_PRECISION,
        )
    print(json.dumps(prediction, indent=2))
else:
    print("Pipeline example disabled.")
```
