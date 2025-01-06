# Experiment Artifacts

This directory is the single hub for experiment-related assets in the final submission.

## Top-level files

- `EVALUATION_FRAMEWORK.md`: current evaluation metric definitions.
- `EXPERIMENTS_SUMMARY.md`: report-ready experiment conclusions.
- `validity_analysis.md`: validity limits and defensible claims.
- `autoscore_recompute.py`: canonical AutoScore implementation used by runtime code.
- `summary_*.csv`, `experiment_*.json`, `wbs_snapshot_*.json`: archived run outputs.

## Subdirectories

- `*_experiment/`, `*_ablation/`, `*_comparison/`: experiment-specific runners, raw outputs, figures, and reports.
- `human_evaluation/`: human evaluation survey summaries and figures.
- `docs/`: experiment protocols, execution notes, and evaluation design drafts.
- `references/`: paper PDFs used as evaluation-method references.
- `report_assets/`: presentation/report source artifacts that are not runtime code.
- `_legacy/`: older experiment outputs retained for traceability.

Runtime debug responses are written to `generated/debug/` and are intentionally ignored by git.
