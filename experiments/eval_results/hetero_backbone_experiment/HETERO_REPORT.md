# Hetero-Backbone Experiment Report

생성일: 2026-04-25

## 0. Conditions

| 조건 | WBS Gen | Task Mgr | Sub-agents (Debate) |
|---|---|---|---|
| H_baseline   | Gemma26 | Gemma26 | Gemma26 |
| H_wbsgen     | **Gemini Flash Lite** | Gemma26 | Gemma26 |
| H_taskmgr    | Gemma26 | **Gemini Flash Lite** | Gemma26 |
| H_both       | **Gemini Flash Lite** | **Gemini Flash Lite** | Gemma26 |
| H_all_frontier | Gemini Flash Lite (Preview) | Gemini Flash Lite (Preview) | Gemini Flash Lite (Preview) |

**Judge**: `gemini-3.1-pro-preview` (env override)

## 1. 조건별 결과 (μ ± σ, N=3)

| 조건 | Structure | Assignment | Debate | Overall |
|---|---|---|---|---|
| All Gemma26 (baseline) | 0.76±0.06 | 0.40±0.11 | 0.81±0.11 | **0.65±0.06** |
| WBS Gen → Gemini | 0.76±0.03 | 0.29±0.03 | 0.89±0.05 | **0.63±0.02** |
| Task Mgr → Gemini | 0.79±0.08 | 0.34±0.01 | 0.88±0.04 | **0.66±0.04** |
| WBS Gen + Task Mgr → Gemini | 0.73±0.05 | 0.36±0.08 | 0.86±0.14 | **0.63±0.04** |
| All Gemini (frontier) | 0.78±0.01 | 0.37±0.07 | 0.80±0.13 | **0.64±0.03** |

## 2. Autoscore 결과

| 조건 | Quality | Allocation | Orchestration | Overall Auto |
|---|---|---|---|---|
| All Gemma26 (baseline) | 1.00±0.00 | 0.60±0.04 | 0.75±0.01 | **0.81±0.02** |
| WBS Gen → Gemini | 0.96±0.00 | 0.68±0.03 | 0.88±0.14 | **0.85±0.04** |
| Task Mgr → Gemini | 0.99±0.01 | 0.64±0.05 | 0.87±0.09 | **0.84±0.01** |
| WBS Gen + Task Mgr → Gemini | 1.00±0.00 | 0.65±0.02 | 0.87±0.13 | **0.85±0.03** |
| All Gemini (frontier) | 0.99±0.02 | 0.66±0.03 | 0.82±0.11 | **0.84±0.02** |

## 3. 핵심 분석

- **Baseline (Gemma26 all)**: 0.65
- **WBS Gen → Gemini**: 0.63 (Δ vs baseline = -0.02)
- **Task Mgr → Gemini**: 0.66 (Δ vs baseline = +0.01)
- **WBS Gen + Task Mgr → Gemini**: 0.63 (Δ vs baseline = -0.02)
- **All frontier**: 0.64 (Δ vs baseline = -0.01)


### Autoscore 관점 보완

- Baseline Auto = **0.81**, All frontier Auto = **0.84**
- H_both Auto = **0.85** → frontier orchestration 이득을 일부 흡수
- H_wbsgen Auto = **0.85**, H_taskmgr Auto = **0.84**

### 부분 특화의 marginal value

- WBS Gen만 frontier → all-frontier 대비 효과의 ?% 달성
- Task Mgr만 frontier → all-frontier 대비 효과의 ?% 달성
- 둘 다 frontier (H_both) → all-frontier에 가장 근접하나 sub-agents 비용 절약

## 4. 산출물

- `figures/fig1_overall.png`, `fig2_dimensions.png`
- `figures/fig3_autoscore_overall.png`, `fig4_autoscore_dimensions.png`
- `summary_hetero.csv`
- `snapshots/`, `logs/`, `run_hetero.py`, `rejudge_hetero.py`, `orchestrate.sh`
