<div align="center">

# symPO

### Multi-agent orchestration for intelligent work breakdown & assignment

*PRD, team context, and meeting notes in → debated WBS with roles, schedules, and buffers out.*

<br/>

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-orchestration-1f6feb?style=for-the-badge)](https://github.com/langchain-ai/langgraph)
[![License: MIT](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)](LICENSE)

**[한국어 README](README.ko.md)**

<br/>

[Overview](#overview) · [Pipeline](#pipeline) · [Outputs](#outputs) · [Validation](#validation) · [Stack](#tech-stack) · [Run](#quick-start)

</div>

---

## Overview

**symPO** (*symposium + project orchestration*) is a **research portfolio** exploring one question:

> *Can LLM agents—given real planning inputs—negotiate their way to a **usable WBS**, not just a plausible one-shot draft?*

Instead of a single prompt, the system runs a **five-stage pipeline**: ingest context → draft structure → route specialists → debate adjustments → finalize assignments. A PM-style supervisor mediates buffers, reassignments, and convergence.

| You provide | You receive |
|-------------|-------------|
| Product requirements | 3-level work breakdown (phase → feature group → task) |
| Team skills & resumes | Role / responsibility mapping per leaf task |
| Meeting notes (optional) | Debate transcript showing how the plan evolved |
| Behavioral profiles (optional, experimental) | Quality scores (automatic + human-validated) |

The repo ships the **full stack**: orchestration code, UI & API surfaces, MCP tool server, and an evaluation harness with ablation studies plus a **65-participant human survey**.

---

## Pipeline

<p align="center">
  <img src="docs/assets/slides/pipeline_overview.png" alt="symPO five-stage pipeline: Input → Drafting → Routing → Discussion → Assignment" width="920"/>
</p>

| Stage | What happens |
|:-----:|----------------|
| **Input** | Meeting audio or specs, plus team metadata, are parsed and indexed for retrieval. |
| **Drafting** | A generator agent proposes an initial hierarchical WBS. |
| **Routing** | A task-manager agent matches skills to work packages and summons the right member-agents. |
| **Discussion** | Agents review risks, buffers, and hand-offs; the supervisor intervenes when debate stalls or drifts. |
| **Assignment** | Feedback is merged into a final plan with schedules and locked responsibilities. |

Same engine powers the **web UI**, **live-streaming API**, and **MCP tool server**.

---

## Outputs

<p align="center">
  <img src="docs/assets/slides/project_output.png" alt="Example WBS cards and Gantt-style schedule produced by symPO" width="920"/>
</p>

- **Structured WBS** — L1 / L2 / L3 hierarchy with estimates and risk buffers  
- **Per-member task cards** — who owns what, for how long  
- **Schedule view** — dependency-aware timeline with color-coded ownership  
- **Debate log** — auditable record of agent reasoning and PM decisions  

---

## Validation

We did not stop at demo quality. The project includes backbone comparisons, RAG ablations, context-metadata studies, and a blind human survey by practitioners.

<p align="center">
  <img src="docs/assets/slides/human_evaluation.png" alt="Human evaluation survey results: 4.48 out of 5 average, 89.2% positive on feasibility" width="780"/>
</p>

| Signal | Takeaway |
|--------|----------|
| Multi-round debate | Judge scores rose vs. generate-only on most tested models |
| Backbone stability | Gemma-26B MoE was the most consistent overall in our setup |
| Resume vs. behavioral metadata | Skills / experience helped assignment more than eDISC-only in the pilot |
| Human survey (N=65) | Mean **4.48 / 5.00**, **89.2%** positive on deadline feasibility |

Full experiment write-ups → [`experiments/eval_results/EXPERIMENTS_SUMMARY.md`](experiments/eval_results/EXPERIMENTS_SUMMARY.md)

---

## Tech stack

| Area | Tools |
|------|-------|
| Core | Python 3.10+, LangChain, LangGraph |
| Interfaces | Streamlit · FastAPI (SSE) · MCP (FastMCP) |
| Retrieval | FAISS, sentence-transformers; hybrid / graph / agentic RAG variants tested |
| Models | Gemini, OpenAI, Anthropic, local & Ollama backends; offline mock mode |
| Evaluation | Custom AutoScore v2 · G-Eval-style LLM judge · cross-model judging |

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Default settings use a **mock LLM** — no API key needed to walk through the flow.

```bash
cd src
streamlit run main.py                 # browser UI
uvicorn api:app --reload              # API + live stream
python mcp_server.py                    # MCP tools
```

Smoke-test the evaluation harness:

```bash
cd src && python ../experiments/eval/experiment_runner.py --backend mock --runs 1
```

---

## Author

**Sukoji** — Human-Centered AI Engineering @ Sangmyung Univ. · Research @ [PIAI, POSTECH](https://piai.postech.ac.kr/english)

---

<div align="center">
<sub>Capstone / agent-systems research · 2026 · POSCO Youth AI·BigData Academy (team project origin)</sub>
</div>
