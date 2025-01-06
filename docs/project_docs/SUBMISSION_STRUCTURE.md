# Final Submission Structure

The repository is organized so runtime code, sample inputs, generated outputs, and experiment artifacts are separated.

## Runtime code

- `main.py`, `api.py`, `mcp_server.py`, `metrics.py`
- `agents/`
- `orchestration/`
- `data_pipeline/`
- `persona_engine/`
- `schemas/`
- `output/`
- `frontend/`

## Inputs and runtime outputs

- `sample_data/`: sample PRD, meeting transcript, member profiles, and eDISC PDFs used by the app.
- `generated/`: latest runtime outputs from normal app execution.
- `generated/debug/`: transient LLM debug dumps. This path is ignored by git.

## Evaluation and experiments

- `eval/`: reusable evaluation runner/analyzer code.
- `eval_results/`: single hub for experiment artifacts, reports, figures, survey outputs, paper references, and report assets.

## Documentation

- `README.md`: quick start and project summary.
- `PROJECT_OVERVIEW.md`: codebase map and current implementation notes.
- `MCP_GUIDE.md`: MCP/tool boundary guide.
- `eval_results/docs/`: experiment protocols, execution notes, and evaluation design drafts.
- `docs/archive/`: archived notes retained for traceability.
