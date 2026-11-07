# Notebooks

Two studies share one codebase (`src/satchange/`); their notebooks are kept apart:

- `research_1/` — Research 1, binary change detection on LEVIR-CD (training/HPO notebooks 01–04 and
  `heavy_*`, research notebooks 05–11). Moved here unchanged on 2026-11-06 except for project-root
  resolution in source cells (01–04) and the notebook-06 lookup in `08_levir_roi_model_selection.ipynb`.
- `research_2/` — Research 2, semantic change detection on SECOND (single- vs two-stage). Notebooks are
  added only for approved tasks; see `docs/research2/IMPLEMENTATION_PLAN.md`.

Run notebooks with the kernel working directory set to the notebook's own folder or the project root.
Several Research 1 notebooks default expensive flags to `True` (e.g. `RUN_TEST` in 06/07) — never
"Run All" them as a check. Reusable logic belongs in `src/satchange/`, not in notebooks.
