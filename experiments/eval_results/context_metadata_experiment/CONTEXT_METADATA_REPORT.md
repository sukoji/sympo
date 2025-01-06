# Context Metadata Ablation — Report

생성일: 2026-04-27  |  Qwen API, N=3  |  Figure source: LLM-Judge median

## 0. 가설

- **H1**: M_both > M_resume > M_disc (skill match 정보 + behavior 모두 best)
- **H2**: M_disc < M_resume (skill 정보 없으면 assignment 약화)

## 1. 조건

| Mode | 이력서 (tech_stack, strengths, yoe) | eDISC (behavior type) | 베이스 |
|---|---|---|---|
| **M_resume** | ✅ 사용 | ❌ | gemma26 C3_3rounds (use_disc=False) 재활용 |
| **M_disc**   | ❌ stripped (monkey-patch) | ✅ | NEW: C4 + monkey-patch로 이력서 비움 |
| **M_both**   | ✅ 사용 | ✅ | gemma26 C4_with_disc (use_disc=True) 재활용 |

## 2. 결과 (Judge median ×3, N=3)

| Mode | Structure | Assignment | Debate | **Overall** |
|---|---|---|---|---|
| M_resume (resume only) | 0.78±0.08 | 0.31±0.03 | 0.80±0.13 | **0.62±0.05** |
| M_disc (eDISC only) | 0.71±0.03 | 0.22±0.08 | 0.82±0.08 | **0.57±0.03** |
| M_both (resume + eDISC) | 0.76±0.05 | 0.31±0.03 | 0.79±0.05 | **0.61±0.03** |

## 3. 핵심 분석

- **M_resume**: 0.619
- **M_disc**: 0.566
- **M_both**: 0.611

- Δ (both vs resume): -0.008
- Δ (resume vs disc): +0.053

## 4. 한계

- Gemma-4-26B의 phantom member ID 배정 문제로 Assignment 차원이 모든 mode에서 0 가까움 가능
- 시간차 confound: M_resume·M_both는 며칠 전 데이터 재활용
- N=3 pilot, 단일 PRD
- M_disc는 monkey-patch로 tech_stack 비움 — 시스템 fallback 동작에 영향
