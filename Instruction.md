# Research 2 — Experimental Protocol, GPU Budget Control, Pilot Study & Repository Organization


Structure of the project
satellite-change-intelligence/
src/satchange/ — общий код и HPO
notebooks/
research_1/ — существующие ноутбуки
research_2/ — новые эксперименты
artifacts/ — модели и результаты
docs/ — протоколы исследований



## 0. Your role and primary instructions

Act as a senior Machine Learning Research Engineer and scientific research advisor.

You are working in my existing repository:

`satellite-change-intelligence`

Your responsibilities:

1. Analyze the existing codebase and the additional project-context file I will provide.
2. Establish a scientifically rigorous Research 2 Experimental Protocol, discussing methodological decisions with me.
3. Extend the existing HPO infrastructure with reliable GPU-hour accounting and budget enforcement.
4. Prepare a controlled pilot experiment comparing FC-Siam-Diff training from scratch on SECOND with fine-tuning from LEVIR-CD.
5. Perform a separate, read-only reproducibility audit.
6. Organize the repository to clearly separate the notebooks of Research 1 and Research 2.

**Do not blindly implement everything at once.**

The experimental methodology must be discussed and approved before expensive experiments are launched.

Preserve existing functionality and scientific results.

---

# 1. Additional project-context file

I will provide a separate file containing additional context about my project.

This file may describe:

- The history and objectives of Research 1.
- Previously trained models and their characteristics.
- Experiments already conducted.
- Existing results, observations and conclusions.
- The transition from Research 1 to Research 2.
- My intended scientific research questions.
- Previous architectural decisions.
- Technical limitations and problems encountered during development.
- Planned experiments and future directions.

**Read this file carefully before designing or implementing anything.**

However, use it intelligently.

## 1.1 The context file is not an implementation specification

Its purpose is to help you understand the project, not to authorize additional work.

In particular:

- Do not assume everything described in the file still needs to be implemented.
- Some features may already exist.
- Some previous ideas may have been abandoned.
- Some experiments may have already been completed.
- Some descriptions may refer to older code versions.
- Some historical decisions may not apply to Research 2.
- Some findings may have been superseded by later experiments.

Do not turn historical explanations into new requirements.

Do not create components simply because they are mentioned in the context file.

## 1.2 Cross-check against the actual repository

Use the following distinction:

**The context file explains the history and scientific intentions. The repository shows the current implementation.**

When they disagree:

1. Inspect the relevant code.
2. Determine whether the context describes an outdated implementation.
3. Identify the discrepancy.
4. Explain its implications.
5. Ask me only when the discrepancy affects an important scientific or architectural decision.

Do not invent missing functions, metrics, models, checkpoints or completed experiments.

Distinguish verified information from assumptions.

## 1.3 Avoid unnecessary development

Before implementing any component, determine:

- Does an equivalent implementation already exist?
- Can it be reused?
- Can the existing component be extended?
- Is this functionality actually required for the current research?
- Does implementing it improve scientific validity or maintainability?

Prefer modifying a well-established existing module over creating a competing implementation.

Do not overengineer.

Do not redesign the project merely because a different architecture might seem cleaner.

**The additional context file is background information. The explicit requirements in this prompt define the scope of the task.**

---

# 2. Research background

## 2.1 Research 1 — Binary Change Detection

The first research investigates binary change detection in satellite image pairs using LEVIR-CD.

The trained models predict **where changes occurred**, returning binary masks.

The project has investigated different model configurations:

- Light.
- Medium.
- Heavy.

Some variants use different model architectures and image resolutions.

Research 1 includes:

- Training.
- Hyperparameter optimization.
- Validation and test evaluation.
- Checkpoint selection.
- F1 and IoU measurement.
- Inference benchmarks.
- Resource-efficiency comparisons.
- Additional ROI extraction experiments.

Some experiments analyze how predicted change masks can be converted into cropped regions for further processing.

Relevant metrics may include:

- `roi_union_coverage95`
- `roi_single_coverage95`
- `mean_crops_per_pair`
- `processed_area_ratio`
- `fixed_input_work_ratio`
- `false_crop_fraction`

Verify the actual implementation and definitions rather than assuming they match the descriptions above.

**Research 1 is a separate, largely completed research project.**

Its existing trained models, checkpoints and results may be used as a starting point for Research 2.

Do not silently retrain Research 1 models or overwrite finalized results.

## 2.2 Research 2 — Semantic Change Detection

The research question is:

**Under which accuracy and computational-budget conditions is a two-stage semantic change detection system preferable to a single-stage semantic change detection model?**

Two approaches will be compared.

### Architecture A — Single-stage system

A semantic change detection model directly processes the complete satellite image pair.

Input:

- Image at T1.
- Image at T2.

Output:

- Spatially localized semantic information identifying what changed.

### Architecture B — Two-stage system

**Stage 1 — Binary Change Detection**

A model identifies where changes may have occurred.

Its prediction is processed by an ROI extraction algorithm.

The algorithm selects regions for further analysis.

**Stage 2 — Semantic Change Analysis**

A semantic model analyzes the selected image crops and predicts what changed.

The results are reconstructed into the original image coordinate system.

The intended advantage is potentially lower computation when changes occupy only a small portion of an image.

However:

- Stage 1 may miss relevant changes.
- Incorrect ROIs may cause semantic errors.
- Crop creation and reconstruction introduce overhead.
- Processing numerous crops may be less efficient than full-image inference.
- The benefit may depend strongly on change density.

These are hypotheses to investigate, not assumptions to accept.

The two-stage system must be evaluated end-to-end using actual predicted masks.

## 2.3 Available infrastructure

The project includes an automated HPO system:

```python
from satchange.automated_hpo import run_automated_parameter_search
```

Potentially relevant modules:

- `src/satchange/automated_hpo.py`
- `src/satchange/hpo_core.py`
- Other modules under `src/satchange/`
- `notebooks/`
- `artifacts/final_models/`

Verify their actual presence and behavior.

Previous experiments have used four NVIDIA A100 GPUs, generally 80 GB each.

Verify the hardware available in the current environment.

**Reuse the existing HPO and model-training infrastructure. Do not build another HPO system.**

---

# 3. Repository organization — two independent studies, one codebase

The repository contains two scientifically distinct investigations:

- Research 1 — Binary Change Detection.
- Research 2 — Semantic Change Detection and the single-stage versus two-stage comparison.

They must coexist in the same Git repository.

I do not want two completely separate repositories or duplicated training infrastructures.

## 3.1 Required notebook directory structure

Currently, the repository has a directory:

`notebooks/`

It contains the existing notebooks from Research 1.

Reorganize it into:

```text
satellite-change-intelligence/
├── src/
│   └── satchange/
│       └── ...                 # Shared implementation
│
├── notebooks/
│   ├── research_1/
│   │   └── ...                 # Existing Research 1 notebooks
│   │
│   └── research_2/
│       └── ...                 # New Research 2 notebooks
│
├── artifacts/
│   └── ...                     # Existing artifacts
│
├── docs/
│   └── ...                     # Research documentation
│
└── ...
```

The structure above is illustrative.

Preserve the rest of the existing repository layout wherever possible.

## 3.2 Move existing notebooks

Move the existing Research 1 notebooks from:

`notebooks/`

to:

`notebooks/research_1/`

Create:

`notebooks/research_2/`

for Research 2 experiments.

Use Git-aware file moves where practical to preserve meaningful file history.

Do not rename individual notebooks unless necessary.

Do not duplicate the notebooks.

Do not delete outputs, metadata or previous experimental results unnecessarily.

Before moving files, inspect their purpose and dependencies.

If the existing `notebooks/` directory contains supporting files, determine whether they belong to Research 1 or are shared resources.

Do not silently discard or incorrectly relocate shared resources.

## 3.3 Preserve notebook functionality

Moving the notebooks changes their location and may break relative paths.

Inspect and update affected references where necessary, including:

- Imports.
- Dataset paths.
- Checkpoint paths.
- Artifact directories.
- Configuration paths.
- Relative file access.
- Notebook-to-notebook references.
- README links.
- Documentation links.
- Scripts that invoke notebooks.
- Any automated tests or tools referencing their previous locations.

Do not change the scientific behavior of existing Research 1 notebooks.

Use robust project-root resolution if needed, but avoid large refactoring solely to support the move.

Verify that representative Research 1 notebooks can still resolve their dependencies.

Do not execute expensive training cells merely to verify paths.

## 3.4 Research separation

All new experimental notebooks for Research 2 should be placed under:

`notebooks/research_2/`

For example:

```text
notebooks/research_2/
├── 01_second_dataset_audit.ipynb
├── 02_fc_siam_diff_pilot.ipynb
└── ...
```

These filenames are examples, not mandatory requirements.

Create only notebooks needed for the currently approved experiments.

Do not generate empty placeholder notebooks for imagined future research.

Keep reusable Python implementations under the shared `src/satchange/` package.

Do not duplicate model implementations, evaluation functions or the HPO controller between Research 1 and Research 2.

## 3.5 Documentation

Research 2 should have its own experimental protocol and results documentation.

Suggested location:

`docs/research2/EXPERIMENTAL_PROTOCOL.md`

If the repository already has documentation conventions, follow those instead.

There is no need to reorganize all historical Research 1 documentation unless required for clarity.

**Important: this is a focused notebook reorganization, not permission to restructure the entire repository.**

---

# 4. Why Research 2 needs a stricter methodology

Research 1 revealed methodological limitations in how computational resources were allocated during HPO.

For example:

- Different models received different hyperparameter search spaces.
- Some HPO parameters were changed to accelerate training.
- Different searches required substantially different amounts of time.
- Reproducibility-related settings evolved throughout development.
- Some later experiments benefited from optimizations unavailable during earlier experiments.

Consequently, comparing final model quality alone does not necessarily establish a fair comparison.

For Research 2, we must define the rules before serious experimentation.

The objective is not to force all architectures to use identical hyperparameters.

It is to give competing approaches a scientifically defensible optimization opportunity and record their resource consumption.

Distinguish:

- Equal GPU-hours.
- Equal wall-clock time.
- Equal training epochs.
- Equal optimization steps.
- Equal HPO trials.
- Equal model capacity.
- Equal inference budget.

These constraints are not equivalent.

Recommend which ones matter for this research.

---

# 5. Task A — Research 2 Experimental Protocol

## 5.1 Initial audit

Inspect:

1. Existing architecture implementations.
2. LEVIR-CD training/evaluation pipelines.
3. SECOND dataset integration, if present.
4. Existing HPO configurations.
5. GPU scheduling and worker allocation.
6. Checkpointing and resume mechanisms.
7. Existing experiment result formats.
8. ROI extraction and crop-generation algorithms.
9. Metric implementations.
10. Existing final model checkpoints and their provenance.

Identify which components can be reused.

Provide source paths and function references.

## 5.2 Discuss and propose the scientific protocol

Prepare a proposal for:

`docs/research2/EXPERIMENTAL_PROTOCOL.md`

Do not finalize it without my approval.

For important methodological decisions, provide:

- Recommended option.
- Scientific justification.
- Realistic alternatives.
- Advantages and disadvantages.
- Computational cost implications.

### A. Hypotheses

Consider the following provisional hypotheses.

**H1 — Computational efficiency**

A two-stage system reduces end-to-end inference cost under certain distributions of changes.

**H2 — Accuracy-efficiency trade-off**

A two-stage architecture may provide a better accuracy-versus-computation trade-off when changes are spatially sparse.

**H3 — ROI limitations**

As the proportion of changed image area increases, ROI extraction produces diminishing computational benefits or may increase processing cost.

**H4 — Transfer learning**

LEVIR-CD pretraining may reduce the GPU-hours needed to achieve a specified validation quality on SECOND compared with training from scratch.

Critically evaluate these hypotheses.

Reformulate, combine or reject them where justified.

Distinguish central hypotheses from supporting investigations.

For each retained hypothesis, define:

- Independent variables.
- Dependent variables.
- Controls.
- Expected measurable effect.
- Falsification conditions.
- Required experiments.

Do not create hypotheses simply to increase their number.

### B. Dataset protocol

Analyze SECOND:

- Actual dataset structure.
- Image pairs.
- Semantic labels.
- Class mapping.
- Official splits.
- Geographic or scene leakage risks.
- Class imbalance.
- No-change examples.
- Preprocessing.
- Augmentation.
- Compatibility with LEVIR-CD.

Recommend whether to retain the official splits.

Freeze dataset version, split manifests and preprocessing before final experiments.

The test set must not be used for HPO or model selection.

Do not assume SECOND and LEVIR-CD have identical labels or image properties.

### C. Computational budget fairness

Compare at least:

1. Equal HPO GPU-hours per candidate.
2. Equal total development GPU-hours per approach.
3. Equal training budgets with separately reported HPO costs.

Recommend a scientifically defensible policy.

Address the fact that the two-stage system contains two trained models, while the single-stage system contains one.

Explain whether the two-stage system should have a shared total optimization budget across both stages.

Distinguish:

**Adaptation budget**

GPU-hours required to adapt existing models to SECOND.

**Full training budget**

Training costs including relevant pretraining when measurable and attributable.

**Inference budget**

Computational resources required for processing unseen image pairs.

LEVIR-CD pretraining must not automatically be represented as zero historical cost.

At the same time, existing checkpoints may legitimately be used in a separately defined transfer-learning experiment.

### D. Hardware comparability

Define the relevant controls:

- GPU type and number.
- Precision.
- Software environment.
- Training and validation schedules.
- Batch-size policy.
- Data-loading behavior.
- Parallelism.
- Inference benchmarking conditions.

Explain how GPU allocation time differs from utilization-weighted GPU metrics.

### E. Evaluation metrics

Propose a minimal primary metric set and supplementary diagnostics.

Consider:

**Semantic quality**

- mIoU.
- Macro F1.
- Per-class IoU/F1.
- Dataset-appropriate semantic change metrics.

**Binary/ROI quality**

- Change detection recall and precision.
- Binary F1/IoU.
- ROI coverage.
- Missed changes.
- False crops.
- Crop count.
- Processed area ratio.

**Computational efficiency**

- End-to-end inference latency.
- GPU-hours.
- Peak memory.
- Throughput.
- Computational work when reliably measurable.

The two-stage latency measurement must include:

1. Stage 1 inference.
2. Binary-mask processing.
3. ROI extraction.
4. Crop preprocessing.
5. Stage 2 inference.
6. Reconstruction.

Do not treat reduced processed area as proof of reduced measured latency.

### F. Model selection

Define predetermined rules for selecting:

- Research 1 binary models.
- Single-stage semantic baseline.
- Stage 2 semantic model.
- Hyperparameters.
- ROI thresholds.
- Crop configurations.
- Final checkpoints.

Use validation data for model selection.

Consider Pareto-front analysis instead of forcing all outcomes into one weighted score.

Discuss architecture-capacity confounding.

### G. Repetition and statistics

Recommend:

- Number of seeds.
- Which experiments must be repeated.
- Mean and standard deviation reporting.
- Appropriate uncertainty intervals.
- Paired evaluation on identical held-out examples.

Consider the available budget.

Separate pilot findings from confirmatory findings.

### H. Baselines and ablations

Consider:

- Single-stage full-image semantic model.
- Real two-stage pipeline using predicted ROIs.
- Oracle ROI experiment using ground-truth regions.
- Same semantic model with and without ROI filtering, if comparable.
- Low-change-density and high-change-density cases.
- ROI threshold sensitivity.

Clearly distinguish mandatory baselines from optional ablations.

Do not present oracle ground-truth cropping as a deployable architecture.

### I. Decision rules

Define how we determine when one approach is preferable.

Possible criteria:

- Comparable semantic accuracy at lower latency.
- Higher accuracy under a fixed inference budget.
- Better trade-offs in sparse-change scenes.
- Failure due to insufficient Stage 1 recall.
- No statistically or practically meaningful advantage.

Specify accuracy-versus-cost trade-offs before final testing.

Avoid arbitrary thresholds without justification.

---

# 6. Task B — GPU-hour accounting in existing HPO

Extend:

```python
from satchange.automated_hpo import run_automated_parameter_search
```

Do not create an independent HPO framework.

## 6.1 Required functionality

The HPO controller must support:

- Configurable maximum GPU-hour budget.
- Per-trial accounting.
- Cumulative accounting.
- Multiple simultaneous workers.
- Persistent expenditure records.
- Resume without resetting the budget.
- Prevention of new trials after budget exhaustion.
- Safe termination when necessary.
- Intermediate checkpoint preservation.
- Reporting of unavoidable budget overruns.

Choose a compatible API based on existing conventions.

A possible parameter is:

`gpu_hour_budget`

but this is not mandatory.

Preserve backwards compatibility.

## 6.2 Definition of GPU-hours

Use the conceptual definition:

\[
\text{GPU-hours}=\frac{\sum_i T_i}{3600}
\]

where \(T_i\) is the allocation time in seconds of GPU \(i\).

Examples:

- 1 GPU for 2 hours = 2 GPU-hours.
- 4 GPUs for 30 minutes = 2 GPU-hours.

Do not multiply controller runtime by the number of GPUs merely because those GPUs are available.

Count actual allocations.

Distinguish:

- Allocated GPU time.
- GPU utilization.
- Wall-clock time.
- Queued/idle time.
- Training and validation.
- Failed or interrupted trials.

Prefer conservative, auditable measurement.

NVML may provide supplementary utilization data, but utilization should not automatically determine the primary budget accounting metric.

## 6.3 Enforcement

The controller should:

1. Check remaining budget before scheduling trials.
2. Account for the expected cost of additional trials where possible.
3. Monitor active GPU allocations.
4. Update cumulative consumption periodically.
5. Stop scheduling after exhaustion.
6. Gracefully stop or checkpoint running trials when necessary.
7. Preserve partial results.
8. Report any unavoidable overshoot.

Inspect existing worker and stop-file behavior first.

Do not promise exact zero-overshoot enforcement when the trainer cannot support it.

## 6.4 Persistent records

Store sufficient information to reconstruct GPU expenditure.

Include:

- Experiment ID.
- Trial ID.
- Architecture.
- Hyperparameters.
- Dataset and split.
- Seed.
- GPU identifiers.
- GPU count.
- Start and end times.
- Allocation duration.
- GPU-hours consumed.
- Cumulative expenditure.
- Remaining budget.
- Trial status.
- Checkpoint references.
- Validation metrics.
- Errors and stop reasons.
- Environment identifiers.

Use an existing compatible storage format where possible.

Prevent double counting after restart.

Handle concurrent writes safely.

## 6.5 Crash and resume

Cover:

- Normal completion.
- Controller restart.
- Worker crash.
- CUDA OOM.
- Keyboard interruption.
- Budget exhaustion.
- Interrupted checkpoint writes.
- Partial trial records.

Historical spending must remain counted after resume.

Record uncertainty when exact expenditure cannot be recovered.

## 6.6 Tests

Implement focused tests for:

- Single-GPU accounting.
- Four parallel GPUs.
- Multi-GPU trial allocation.
- Unallocated GPUs.
- Budget boundaries.
- Mid-trial exhaustion.
- Crash and resume.
- Failed trial accounting.
- Duplicate prevention.
- Graceful stopping.
- Backwards compatibility.

Prefer mocked clocks and synthetic workers.

Do not consume real A100 resources for elementary controller tests.

---

# 7. Task C — FC-Siam-Diff pilot

Before choosing the overall research budget, measure real training cost.

## 7.1 Research question

Compare:

**Condition A — Scratch**

FC-Siam-Diff trained from random initialization on SECOND.

**Condition B — Transfer learning**

FC-Siam-Diff initialized using compatible pretrained weights from a LEVIR-CD checkpoint and then trained on SECOND.

The objective is to measure:

- Training cost.
- Convergence.
- Validation quality.
- Transfer-learning benefit.
- GPU-hours required for comparable quality.

The pilot is not intended to prove the final Research 2 hypothesis.

## 7.2 Compatibility audit

FC-Siam-Diff was originally used for binary change detection.

SECOND uses semantic labels.

Inspect:

- Model implementation.
- Output channels.
- Semantic label representation.
- Checkpoint architecture.
- Reusable pretrained layers.
- Target head requirements.
- Weight-loading compatibility.

Do not assume binary and semantic heads are interchangeable.

If architectural adaptation is required, keep the target architecture identical between the two pilot conditions.

Ideally, only parameter initialization should differ.

Document exactly which layers load pretrained weights.

If the model cannot reasonably support the target task, propose the smallest justified adaptation first.

## 7.3 Experimental controls

Keep comparable:

- Dataset split.
- Class mapping.
- Preprocessing.
- Architecture.
- Target representation.
- Batch-size policy.
- Training precision.
- Augmentation.
- Hardware.
- Validation schedule.
- Available training budget.
- Seed policy.

If initialization-specific hyperparameters differ, explain why the comparison remains fair.

Avoid giving one condition more HPO opportunities.

## 7.4 Pilot measurements

Record:

- Epoch duration.
- GPU-hours per epoch.
- Data loading/initialization overhead.
- Validation performance over training.
- Peak GPU memory.
- Total GPU-hours consumed.
- GPU-hours to reach selected quality thresholds, if reached.
- Stability.
- Convergence behavior.

Plot quality against both epochs and GPU-hours.

Do not treat a short pilot as statistically conclusive.

Do not use the final test set.

## 7.5 Pilot budget

First inspect available resources and infrastructure.

Then recommend a small, bounded pilot.

Specify:

- Number of runs.
- Whether minimal HPO is justified.
- Maximum GPU-hours.
- Stopping mechanism.
- Checkpoint frequency.
- Expected overhead.

Obtain my approval before launching substantial GPU training.

Pilot costs must be recorded separately from confirmatory experiment costs.

## 7.6 Full budget recommendation

After the pilot, estimate resources for:

- Scratch training.
- Fine-tuning.
- HPO.
- Final training.
- Required repetitions.
- Both two-stage components.
- Single-stage baselines.
- Validation.
- Final inference benchmarking.

Recommend three realistic options:

1. Resource-constrained.
2. Balanced.
3. Comprehensive.

Explain what scientific conclusions each budget can support.

Freeze the selected budget before confirmatory experiments.

---

# 8. Separate task — Reproducibility audit (READ-ONLY)

**Do not modify any code as part of this section.**

Inspect:

```python
run_automated_parameter_search(...)
```

and all relevant downstream functionality.

The question is:

**Is reproducibility achievable with the current parameters and implementation, or is it inherently tied to a specific GPU, making identical results impossible when changing GPUs?**

Answer based on actual code.

Distinguish:

1. Repeated runs on the same GPU.
2. Runs on different A100 GPUs.
3. Runs on different GPU architectures.
4. Runs using different numbers of GPUs.
5. HPO sampling and scheduling reproducibility.
6. Numerical/bitwise reproducibility versus statistical reproducibility.
7. Reproducibility after checkpoint resume.

Inspect relevant settings:

- Python, NumPy and PyTorch seeds.
- CUDA seeds.
- DataLoader workers and generators.
- Deterministic algorithms.
- cuDNN settings.
- CUDA/cuBLAS environment.
- TF32.
- AMP.
- Non-deterministic operations.
- GPU allocation.
- Parallel HPO scheduling.
- Checkpoint RNG state restoration.
- Library, driver and CUDA versions.
- Dataset augmentation randomness.

Do not assume seed equality guarantees identical outcomes.

Do not assume moving between GPU device IDs makes all reproducibility impossible.

## Required report

### A. Current status

Classify reproducibility as:

- Deterministic under defined conditions.
- Partially reproducible.
- Not reliably reproducible.
- Unknown due to insufficient evidence.

### B. Code evidence

Provide exact paths, functions and relevant lines.

### C. Same-GPU behavior

Explain expected reproducibility on repeated A100 runs.

### D. GPU switching

Explain the difference between:

- A100 to another A100.
- A100 to a different GPU architecture.

Distinguish bitwise identity from statistical consistency.

### E. HPO reproducibility

Determine whether identical configurations and execution order can be reproduced.

Analyze asynchronous worker effects.

### F. Final answer

Answer directly:

**Can I use different GPUs during Research 2 without compromising scientific validity, and under which conditions?**

You may recommend improvements.

**Do not implement reproducibility fixes without my explicit permission.**

---

# 9. Implementation requirements

## 9.1 Reuse existing components

Before adding anything, inspect the relevant code under:

`src/satchange/`

Avoid duplicate:

- Training loops.
- HPO controllers.
- Checkpoint managers.
- GPU schedulers.
- Model loaders.
- Dataset loaders.
- Metric implementations.

Prefer small, tested extensions.

## 9.2 Preserve scientific artifacts

Do not overwrite finalized Research 1 checkpoints or results.

Research 2 must preserve its own experiment identity and provenance.

Record where practical:

- Git commit.
- Model/checkpoint identity.
- Dataset version and split.
- Seed.
- Configuration.
- Software environment.
- GPU hardware.
- Resource budget.
- Validation and test results.

## 9.3 Minimize scope

Do not implement features solely because they are mentioned in the additional context file.

Do not introduce unnecessary frameworks, abstractions or dependencies.

Do not create new models or extensive experimental notebooks before the related experiments are approved.

## 9.4 Safe Git operations

Inspect the working tree before editing.

Preserve unrelated changes.

Use safe moves for existing notebooks.

Do not automatically push commits.

Do not upload large checkpoints without explicit approval.

## 9.5 Testing

Use unit tests and brief smoke tests before GPU training.

Do not launch full HPO sweeps or expensive experiments without approval.

---

# 10. Expected deliverables

## Deliverable 1 — Repository audit

Identify:

- Current architecture and reusable code.
- Existing research assets.
- HPO implementation.
- GPU scheduling.
- Checkpointing.
- Dataset handling.
- Missing components.

Incorporate relevant information from my additional context file, but distinguish it from verified repository facts.

## Deliverable 2 — Notebook reorganization

Implement:

- `notebooks/research_1/`
- `notebooks/research_2/`

Move existing Research 1 notebooks.

Repair affected paths and references.

Verify that the reorganization has not broken their functionality.

Do not reorganize unrelated directories.

## Deliverable 3 — Experimental Protocol proposal

Prepare a reasoned proposal covering:

- Research question.
- Hypotheses.
- Dataset splits.
- Baselines.
- Computational budgets.
- Model selection.
- Evaluation metrics.
- Repetitions.
- Stopping conditions.
- Statistical analysis.
- Threats to validity.

Discuss alternatives and wait for my approval before finalizing methodological decisions.

## Deliverable 4 — GPU-hour controller

After approval:

- Extend the existing HPO infrastructure.
- Add GPU budget enforcement.
- Store expenditure records.
- Support resume.
- Add tests.
- Document usage and limitations.

## Deliverable 5 — FC-Siam-Diff pilot

After approval:

- Verify SECOND compatibility.
- Prepare scratch and fine-tuning conditions.
- Run the bounded experiment.
- Measure GPU-hours and learning curves.
- Recommend the main experiment budget.

## Deliverable 6 — Reproducibility report

Provide a code-based, read-only analysis.

Do not modify reproducibility settings.

---

# 11. Execution order

Follow the phases below.

## Phase 1 — Context, repository audit and reproducibility

1. Read my additional project-context file.
2. Inspect the repository.
3. Reconcile the historical context with the actual code.
4. Identify reusable implementations.
5. Evaluate current reproducibility.
6. Present the reproducibility report with specific source references.
7. Identify any architectural or methodological risks.

Do not modify reproducibility settings.

## Phase 2 — Repository notebook reorganization

1. Inspect the current `notebooks/` directory.
2. Inventory notebook files and dependencies.
3. Create `notebooks/research_1/` and `notebooks/research_2/`.
4. Move existing Research 1 notebooks.
5. Correct affected paths and references.
6. Verify imports and file access.
7. Report changes and any limitations.

This phase does not require redesigning the remaining repository.

## Phase 3 — Experimental Protocol discussion

1. Propose suitable hypotheses.
2. Discuss dataset splits and leakage.
3. Recommend a fair computational budget policy.
4. Discuss primary metrics.
5. Recommend model selection rules.
6. Propose repetitions and statistical methods.
7. Identify the minimum necessary baselines.
8. Explain trade-offs and ask for my decisions.

**Stop for my approval before finalizing the protocol.**

## Phase 4 — GPU budget implementation

After approval:

1. Implement GPU-hour accounting in the existing HPO system.
2. Add persistent records.
3. Add budget enforcement.
4. Implement resume and safe termination.
5. Add tests.
6. Verify backwards compatibility.
7. Document behavior.

Do not implement unrelated reproducibility improvements.

## Phase 5 — Pilot preparation

1. Verify FC-Siam-Diff checkpoint compatibility.
2. Analyze SECOND requirements.
3. Prepare scratch and transfer-learning conditions.
4. Choose a small pilot budget.
5. Run smoke tests.
6. Present the exact proposed pilot configuration.

Obtain approval before substantial GPU computation.

## Phase 6 — Pilot experiment and budget recommendation

After approval:

1. Run the pilot.
2. Save intermediate and final results.
3. Record actual GPU-hour expenditure.
4. Compare scratch and fine-tuning.
5. Estimate full Research 2 costs.
6. Recommend a final budget.

Do not automatically run the full Research 2 experimental series.

---

# 12. Communication requirements

Treat me as the researcher responsible for the scientific decisions.

I want to participate in methodology design rather than receive unexplained implementation choices.

For each significant decision:

- Explain the available options.
- Recommend one.
- Explain why.
- Identify the relevant trade-offs.
- Ask for approval when necessary.

Clearly separate:

- Verified codebase facts.
- Historical context from my file.
- Methodological recommendations.
- Open scientific decisions.
- Work actually implemented.
- Work not yet executed.

Do not invent results or claim experiments have been completed without executing them.

Prioritize scientific validity, computational efficiency, maintainability and simplicity.

**Start with Phase 1. Read the supplied context file, inspect the actual repository, and perform the reproducibility audit. Then proceed through the phases in the specified order, respecting approval gates.**