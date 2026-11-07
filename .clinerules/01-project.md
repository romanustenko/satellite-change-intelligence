# Cline Rules — satellite-change-intelligence

`AGENTS.md` (repo root) is the authoritative source of architecture, conventions, commands and
baselines. These rules are the short operational checklist for Cline; do not duplicate
`AGENTS.md` content here — update `AGENTS.md` when facts change.

## Communication
- Talk to the user in **Russian** unless asked otherwise.
- Write code, comments, docstrings, commit messages and project docs in **English**
  (do not translate existing Russian docs/notebook markdown).

## Before changing anything
1. Read `AGENTS.md` (at least §2, §4–§6, §8–§9) and relevant `docs/*.md` for the task.
2. Inspect the existing implementation and every call site (`src/`, `tests/`, notebooks — notebooks
   06–11 import private `final_benchmark` helpers) before editing.
3. Present a short plan; list assumptions and unknowns explicitly. Ask instead of guessing when
   behaviour, data or research intent is ambiguous.
   **Superseded for technical work (2026-11-06, D-028):** work autonomously on approved tasks; ask
   only for scientific decisions, GPU budgets and protected operations — see `AGENTS.md` §0 and
   `docs/research2/PROJECT_CONTEXT.md` §0.

## Implementation rules
- Respect the architecture: shared engine (`training.fit_binary_change_model`), model factory,
  `BCEDiceLoss`, `create_dataset_dataloader`, HPO v3 (`automated_hpo` → `hpo_core` →
  `hpo_backend`), `run_final_benchmark`. No parallel per-model trainers, datasets or HPO.
- Never invent APIs, modules, config keys, CLI flags, dependencies or test commands — verify
  them in the repository first.
- Prefer small, focused, reversible changes; match the style of the edited file; no unrelated
  refactors, reformatting, line-ending/BOM changes, or dependency installs/upgrades.
- Preserve backward compatibility: checkpoint keys and `model_parameters`, constructor
  signatures/defaults, HPO parameter keys and `model__` routing, `__all__` exports, private
  helpers used by notebooks, legacy HPO folders. If a break is unavoidable, explain it and add a
  regression test.
- Keep ML invariants: FP32 default / BF16 only on CUDA, deterministic algorithms, explicit seeds,
  CPU-side checkpoint loading, atomic writes, thresholds chosen on validation only.
- Notebooks: edit source cells only; keep metadata and outputs; guard expensive cells with flags
  defaulting to `False`; never execute research/HPO notebooks.

## Protection
- Never modify, delete or overwrite `data/`, `artifacts/` (esp. `final_models/`,
  `hyperparameter_search/`, `hypothesis_tests/`), `*.pt/*.pth/*.ckpt/*.onnx`, `source_backups/`,
  notebook outputs, `changes.md`, `.venv/`, `.env*`.
- Never read, print or store secrets/credentials.
- Do not run training, HPO, `satchange.run probe`, `run_final_benchmark`, dataset downloads,
  test-set evaluation, `src/scripts/install_hpo_v3.py` or any GPU workload unless the user
  explicitly asks. Opt-in CUDA tests need explicit approval.
- No git history rewriting, force operations or pushes; leave the user's uncommitted changes alone.

## Validation and reporting
- Run the cheapest sufficient checks from `AGENTS.md` §8 (focused pytest → full pytest for core
  modules; ruff/mypy on touched files) and compare with the recorded baseline; a pre-existing
  failure is not a regression, a new one is.
- Review `git status` / `git diff` before finishing; confirm only intended files changed.
- Final report: what changed and why, architectural impact (explain significant changes), test
  commands with results, assumptions, unresolved issues and risks.
