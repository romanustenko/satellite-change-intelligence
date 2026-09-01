# Repository instructions

## Scope and working agreement

- These instructions apply to the entire repository.
- Treat the checked-out repository as the source of truth. Inspect the current code, configs,
  tests, notebooks, and Git status before drawing conclusions from conversation history.
- This is both a research implementation and a learning project. When explaining or changing a
  significant ML component, make the data flow, tensor shapes, design reason, and trade-offs clear.
- For analysis and reviews, cite concrete file paths and symbols, describe current behavior, and
  separate confirmed facts from recommendations.
- Do not claim that code works unless the relevant command was run successfully. State exactly
  what was and was not verified.
- Keep the repository read-only unless the user explicitly asks for a change. An analysis,
  diagnosis, review, or teaching request alone does not authorize edits.
- When a change is requested, keep it narrowly scoped. Do not perform unrelated cleanup, broad
  refactors, dependency upgrades, dataset operations, or experiment runs.

## Project purpose and current implementation

`satchange` is a Python/PyTorch research package for binary change detection in aligned satellite
image pairs. The implemented baseline loads LEVIR-CD before/after images and masks, applies paired
augmentations, predicts one change logit per pixel with FC-Siam-Diff, trains with BCE plus Dice,
and evaluates pixel-level metrics over multiple probability thresholds.

The current code does not yet implement a general uncertainty-estimation system despite the
package description. Probability summaries and threshold sweeps are diagnostics, not a calibrated
uncertainty model.

## Repository map

- `pyproject.toml`: Python `>=3.11,<3.13`, runtime and development dependencies, Ruff, pytest, and
  mypy settings. This is the dependency and tool-configuration source of truth.
- `src/satchange/data/levir_cd.py`: LEVIR-CD download, extraction, validation, indexing, dataset,
  image/mask conversion, and `DataLoader` construction.
- `src/satchange/data/transforms.py`: synchronized flips and 90-degree rotations for both temporal
  images and their mask.
- `src/satchange/models/fc_siam_diff.py`: the active shared-weight Siamese encoder, absolute
  feature-difference decoder, and binary-logit output.
- `src/satchange/models/factory.py`: model registry. Only `fc_siam_diff` is currently registered.
- `src/satchange/losses.py`: `DiceLoss` and `BCEDiceLoss`, both consuming logits.
- `src/satchange/metrics.py`: reusable accumulated binary confusion counts and derived metrics.
- `src/satchange/training.py`: epoch training, threshold evaluation, checkpointing, resume logic,
  scheduler stepping, timing, and GPU-memory reporting.
- `src/satchange/optimization.py`: Cartesian hyperparameter search with one spawned worker per
  selected CUDA device. It is GPU-only and potentially expensive.
- `src/satchange/experiments.py`: selection and validation of the `full` and `resized`
  FC-Siam-Diff configs.
- `configs/fc_siam_diff_{resized,full}.yaml`: active experiment inputs. The `bit.yaml` and
  `changeformer.yaml` files are design placeholders; their models are not implemented or
  registered.
- `notebooks/`: exploratory orchestration and the current experiment entry points. Keep reusable
  behavior in `src/satchange`; notebooks should call it rather than duplicate it.
- `artifacts/hyperparameter_search/`: tracked experiment summaries. Treat them as historical run
  outputs, not hand-maintained configuration.
- `scripts/calculate_class_balance.py`: unfinished helper logic, not a runnable CLI in its current
  form.
- `tests/`: currently empty placeholder files. Their presence is not evidence of test coverage.

The following files are currently dormant or incomplete: `src/satchange/data.py`,
`src/satchange/loader.py`, and `src/satchange/models/blocks.py`. The `satchange.data` import
resolves to the `src/satchange/data/` package. Do not route new code through the dormant files or
repair them opportunistically unless the task explicitly concerns them.

## Environment and commands

Use Python 3.11 or 3.12. Do not use Python 3.13+ for this project merely because it is the system
default. Prefer an isolated environment and invoke tools through that interpreter:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev,notebooks]"
```

Equivalent POSIX activation/paths are fine. Do not install optional `api` or `geo` dependencies
unless the requested work uses them.

Run the narrowest relevant check while iterating, then all applicable checks before handoff:

```powershell
.\.venv\Scripts\python -m pytest
.\.venv\Scripts\python -m ruff check .
.\.venv\Scripts\python -m mypy src
```

If the environment or dependencies are unavailable, do not silently install large packages or
pretend the checks passed. Report the missing prerequisite. The current repository has no stable
CLI for preparing data, training, or HPO; the notebooks call the library APIs directly.

## Data and tensor contracts

Preserve these contracts across datasets, transforms, models, losses, metrics, and training:

- A sample is a mapping with `before`, `after`, `mask`, and `sample_id`.
- `before` and `after` are aligned `float32` tensors shaped `[3, H, W]`, normalized to `[0, 1]`.
- `mask` is a binary `float32` tensor shaped `[1, H, W]`. Batched tensors therefore use
  `[B, C, H, W]`.
- Image pairs and masks must receive identical random spatial transforms. Use bilinear resampling
  for RGB imagery and nearest-neighbor resampling for masks.
- LEVIR-CD samples are matched by filename stem across `A`, `B`, and `label`; fail clearly on
  missing, extra, or duplicate identifiers.
- `FCSiamDiff.forward(before, after)` returns raw logits shaped `[B, 1, H, W]`. Do not apply a
  sigmoid before `BCEWithLogitsLoss`, `BCEDiceLoss`, or any other logits-based loss.
- The current encoder pools four times. Inputs must retain compatible spatial dimensions
  (normally each dimension divisible by 16), or decoder/skip shapes will not match.
- The two temporal images must pass through the same encoder weights. Change evidence is the
  absolute difference between corresponding feature maps.
- Metrics convert logits to probabilities internally. Keep threshold selection explicit and use
  the same comparison convention (`probability >= threshold`) throughout the project.

## ML experiment discipline

- Use `train` for fitting and class-balance estimates, `val` for model/epoch/hyperparameter and
  threshold selection, and `test` only for final unbiased evaluation. Never tune on `test`.
- Because changed pixels are sparse, do not report accuracy alone. Include at least F1 and IoU,
  and usually precision, recall, threshold, and confusion counts.
- Compute `pos_weight` from the training split only. Record it with BCE/Dice weights instead of
  treating it as an undocumented constant.
- Preserve reproducibility inputs: Python, NumPy, PyTorch, CUDA, and `DataLoader` seeds as relevant.
  Record model parameters, data resolution, batch size, optimizer, scheduler, loss weights, seed,
  thresholds, and code revision with experiment results.
- Compare experiments under the same split and evaluation protocol. Treat a validation gain as a
  hypothesis until it is reproduced and checked on the untouched test split.
- Select checkpoints by the declared primary metric. The current training and HPO code uses the
  best validation F1 across supplied thresholds; changing that affects checkpoint selection,
  scheduler behavior, result ranking, and comparability with tracked artifacts.
- A configured feature is not necessarily implemented. In particular, current YAML fields for
  mixed precision and gradient accumulation are not consumed by `training.py`.

## Safety around data, GPUs, and artifacts

- Do not call `prepare_levir_cd`, download or extract datasets, delete/recreate dataset splits,
  start training, or launch `run_parameter_search` unless the user explicitly requests that
  operation. These actions can use network, disk, multiple processes, and substantial GPU time.
- Never commit raw datasets, credentials, checkpoints, or other large generated files. Respect
  `.gitignore`; keep `data/.gitkeep` only as the directory marker.
- Do not delete or rewrite tracked experiment summaries under `artifacts/` as cleanup. New run
  outputs must use a distinct, descriptive run/search name and preserve the full configuration.
- Load checkpoints only from trusted sources: `torch.load(..., weights_only=False)` may deserialize
  arbitrary Python objects. Use `map_location` deliberately and validate checkpoint identity and
  required keys before resuming.
- Do not assume CUDA or a particular GPU count. Guard ordinary model/data tests for CPU; reserve
  multi-GPU assertions for explicitly requested HPO work.

## Change guidelines

- Read the whole affected path and search all callers before editing shared behavior. Reuse an
  existing project pattern or standard-library/PyTorch feature before adding an abstraction or
  dependency.
- Follow `src/` layout, absolute `satchange...` imports, type annotations, public docstrings, and
  the 100-character Ruff line limit. Keep comments accurate and useful for learning; remove or
  update comments that contradict the code.
- Add no production dependency without explaining why the standard library, PyTorch, or an
  installed dependency is insufficient. Update `pyproject.toml` only when code actually needs it.
- Keep public exports in the relevant `__init__.py` synchronized with supported APIs. Do not expose
  prototypes as stable public interfaces.
- When changing config shape or semantics, update every consumer and the relevant notebook in the
  same task. Do not add unused config keys.
- Prefer explicit validation and actionable exceptions at filesystem, configuration, shape, dtype,
  and device boundaries. Do not weaken checks that protect image/mask alignment or checkpoint
  integrity.
- Avoid premature model registries, base classes, generalized trainers, or framework migrations.
  Extend only when a second real implementation or concrete requirement makes the abstraction
  useful.

## Testing expectations

Add a focused regression test for non-trivial changed behavior:

- Dataset tests should build a tiny temporary `train/A`, `train/B`, and `train/label` fixture; they
  must not download LEVIR-CD.
- Transform tests must prove that before/after/mask geometry remains synchronized and outputs are
  contiguous.
- Model tests should cover input validation, output shape, shared two-input flow, and a finite
  backward pass on small CPU tensors with spatial dimensions divisible by 16.
- Loss and metric tests should use hand-computable tensors, cover empty-positive/empty-prediction
  cases, and distinguish logits from probabilities.
- Training/checkpoint tests should use tiny synthetic loaders and models. Do not require a GPU or a
  long optimization run for the default test suite.
- HPO helper tests should exercise pure validation and search-space expansion without spawning CUDA
  workers.

Before finishing, inspect the diff for accidental notebook output churn, generated artifacts,
dataset files, or unrelated formatting changes. Summarize changed files, checks run and their exact
results, plus any remaining limitation or unverified GPU/data path.

## Known baseline limitations

Do not hide or silently work around these when they affect a task:

- The test files contain no tests, so the repository currently has no effective automated coverage.
- `README.md` and `.env.example` are empty, and there is no packaged command-line entry point.
- `BIT` and `ChangeFormer` configs have no corresponding implementation in the model factory.
- The active trainer does not implement the configured mixed precision or gradient accumulation.
- Dormant prototype modules contain known defects and are not part of the active import path.

Fix a limitation only when it is in scope, and add the smallest check that demonstrates the fix.
