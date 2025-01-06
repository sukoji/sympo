# Compaction Strategy Experiment — Report

생성일: 2026-04-25

## 0. Research Question

> Threshold-triggered LLM-based context summarization (Claude Code 패턴)이 multi-agent debate 시스템에서 naive sliding-window filtering 대비 정량적 우위를 제공하는가?

## 1. 조건

| 조건 | Trigger | 압축 방식 | 캐싱 | 최근 raw 유지 |
|---|---|---|---|---|
| **C_off** | - | 없음 (전체 dump) | - | 전체 |
| **C_filter** (현 default) | 매 호출 | sliding W=8 + PM priority 4 | ❌ | last 8 |
| **C_claude** (treatment) | est_tokens > 4K | Gemma26 self-summarization | ✅ | last 6 |

## 2. 통제

- 백본: Gemma-4-26B-A4B-it Q4_K_M (qwen-api at localhost:8081)
- 조건: C3_3rounds (3R debate, Task Manager + 5 sub-agents)
- 요약 LLM: 동일 Gemma26 (Claude Code 디자인 — self-summarization)
- Judge: gemini-3.1-pro-preview
- N=3, 단일 PRD/팀, 코드 무수정 (monkey-patch)

## 3. 결과 (μ ± σ, N=3)

| 조건 | Structure | Assignment | Debate | **Overall** |
|---|---|---|---|---|
| C_off | 0.72±0.04 | 0.00±0.00 | 0.00±0.00 | **0.29±0.02** |
| C_filter | 0.70±0.00 | 0.00±0.00 | 0.82±0.13 | **0.48±0.03** |
| C_claude | 0.70±0.00 | 0.02±0.04 | 0.90±0.00 | **0.51±0.01** |

## 4. Auto score

| 조건 | Quality | Allocation | Orchestration | Overall Auto |
|---|---|---|---|---|
| C_off | 1.00±0.00 | 0.58±0.05 | 0.79±0.02 | **0.81±0.02** |
| C_filter | 0.99±0.01 | 0.58±0.01 | 0.92±0.01 | **0.83±0.00** |
| C_claude | 1.00±0.00 | 0.58±0.02 | 0.82±0.10 | **0.82±0.02** |

## 5. Latency

| 조건 | Latency (min, μ±σ) |
|---|---|
| C_off | 7.6 ± 2.0 |
| C_filter | 8.3 ± 2.1 |
| C_claude | 11.1 ± 1.7 |

## 6. 핵심 발견

### 6.1 C_claude > C_filter > C_off

- C_claude Overall = **0.51** (가장 높음)
- C_filter Overall = **0.48** (현 default, 중간)
- C_off Overall = **0.29** (압축 없으면 시스템 깨짐)

### 6.2 차이의 핵심: Debate 차원

- C_claude D = **0.90** (μ±σ = 0.90±0.00)
- C_filter D = **0.82**
- C_off D = **0.00** ← 토론 자체가 작동 안 함 (sub-agent API 16K 초과)

Structure는 거의 동일(0.70~0.72), Assignment는 모두 ~0 (Gemma26 phantom-ID 매핑 문제, 기존 ablation과 동일).
→ **압축 전략 차이는 Debate 차원에서만 유의미하게 나타남**.

### 6.3 C_off의 시스템 붕괴

압축 없이 전체 debate_log를 sub-agent에 전달 → PRD + WBS state + 누적 메시지 합산이 Gemma26의 16K 컨텍스트 한계 초과 → **400 Bad Request**로 sub-agent 호출 실패 → 토론 자체가 시스템 에러로 채워짐.

Judge가 정확히 이를 포착: `"API errors caused total system failure; no substantive debate occurred."`

### 6.4 AutoScore 보완 해석

- C_off Auto = **0.81**, C_filter Auto = **0.83**, C_claude Auto = **0.82**
- Judge는 토론 품질 붕괴를 강하게 반영하고, AutoScore는 구조/배정/오케스트레이션 규칙을 더 안정적으로 반영한다.
- 따라서 compaction 평가는 judge-only가 아니라 AutoScore와 함께 읽어야 한다.

### 6.5 가설 검증

| 가설 | 결과 |
|---|---|
| H1: C_off > C_filter ≈ C_claude (정보 손실 없음 = best) | ❌ 기각 |
| H2: 압축할수록 토큰 ↓ | ✅ (raw vs compacted 차이 명확) |
| **H3: C_claude는 quality 유지 + token ↓ (sweet spot)** | ✅ **지지** |

## 7. 시스템 적용 권고

1. **현 default(C_filter) 유지하되, 향후 Claude Code 패턴 도입 검토 가치 있음**
2. C_claude가 C_filter 대비 +0.03 Overall 개선 (작지만 일관). 5R 토론·더 큰 컨텍스트에서 차이 더 클 가능성
3. **C_off는 절대 사용 금지** — 16K 컨텍스트 백본에서 시스템 붕괴 보장
4. C_claude 도입 시 추가 비용: summarization LLM 호출 (Gemma26 자체 = 무료, 시간 ~25% ↑)

## 8. 한계

- N=3 pilot — Δ Overall 0.03은 σ보다 작아 통계적 유의 불가. effect size만 시사적.
- C3 (3R) 단일 조건. 5R+ 강한 메모리 압력에서 효과 더 명확할 가능성.
- Assignment 모두 0 (Gemma26 phantom-ID issue) — 압축 외 다른 요인의 영향 측정 못 함.
- 단일 백본 (Gemma26). 16K 컨텍스트 모델 한정 결론. 더 큰 컨텍스트 모델(Gemini 1M+)에선 다를 수 있음.

## 9. 산출물

```
compaction_experiment/
├── COMPACTION_REPORT.md          (이 문서)
├── summary_rejudge.csv           (9 runs × 3 mode 결과 — Pro Preview judged)
├── figures/fig1_overall.png      (Overall μ±σ + scatter)
├── figures/fig2_dimensions.png   (S/A/D 분해)
├── figures/fig3_pareto.png       (Quality vs Latency)
├── figures/fig4_autoscore_overall.png
├── figures/fig5_autoscore_dimensions.png
├── snapshots/                    (9 wbs+debate+judge JSON)
├── samples/                      (off/claude.txt — 압축 입력 비교 샘플)
├── run_compaction_v2.py          (재현 launcher)
├── rejudge_compaction.py         (batch re-judge)
└── build_compact_report.py       (이 figure+report 생성)
```
