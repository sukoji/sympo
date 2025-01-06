# Compaction v4 — Quality × Efficiency Comparison (C_summary trigger fixed)

**Naming note**: 이전 "C_claude"는 Claude Code 패턴을 흉내낸 LLM-summary 모드. 실제로는 LangChain `ConversationSummaryBufferMemory` (token threshold + LLM 요약) 패턴이라 **`C_summary`로 개명**. (CSV/snapshot 파일명은 legacy `compactv4_claude` 유지.)

생성일: 2026-04-27  |  Gemma-4-26B Q4_K_M (32K ctx), N=3, C3 3R, Flash Lite Judge ×3 median

## 0. v3 버그 진단

v3 sample 분석 결과:

- **est_tokens 공식**: `chars × 1.2/3` = chars × **0.4** (영어 기준 underestimate)
- 한글 Gemma 토크나이저 실측: chars × **0.6~0.8**
- v3 threshold 4000 tokens + 위 공식 → 실제 chars ≥ 10,000 시점에야 trigger
- 우리 토론(~130 msg, ~30K chars 끝점)에서 **log≈44에서 첫 trigger** = 이미 1/3 진행 후
- 따라서 v3 결과는 사실상 "C_claude 거의 raw 상태로 측정" → C_filter와 비교 무효

## 1. v4 수정 사항

| 파라미터 | v3 | **v4** | 효과 |
|---|---|---|---|
| tokens/char | 0.4 | **0.7** | 한글 실측 반영 |
| TOKEN_THRESHOLD | 4000 | **1500** | trigger 빈도 ↑ |
| MAX_SUMMARY_CHARS | 2000 | **1500** | summary 더 압축 |
| KEEP_RECENT | 6 | 6 | 동일 |
| PM_RETAIN | 4 | 4 | 동일 |

→ 예상: log≈10-15에서 첫 trigger (이전 log≈44 대비 30개 메시지 일찍)

## 2. Trigger 작동 검증

| Mode | 첫 trigger log size | max calls | trigger fired (sample 단위) |
|---|---|---|---|
| C_minimal | N/A (no LLM summary) | 0 | 0 |
| C_filter | N/A (no LLM summary) | 0 | 0 |
| C_summary | 9 | 25 | 120 |

→ v3 (첫 trigger log≈44) 대비 v4는 log≈9에서 첫 trigger → **30+ 메시지 일찍 압축 시작**

## 3. 결과 (LLM-Judge × AutoScore)

| Mode | Structure | Assignment | Debate | **Judge Overall** | **Auto Overall** | Failures |
|---|---|---|---|---|---|---|
| C_minimal | 0.78±0.05 | 0.33±0.10 | 0.88±0.12 | **0.65±0.01** | **0.82±0.01** | 0.00±0.00 |
| C_filter | 0.79±0.07 | 0.33±0.00 | 0.92±0.06 | **0.66±0.03** | **0.81±0.02** | 0.00±0.00 |
| C_summary | 0.82±0.00 | 0.29±0.03 | 0.95±0.00 | **0.67±0.01** | **0.83±0.02** | 0.00±0.00 |

## 4. AutoScore 차원별

| Mode | Quality | Allocation | Orchestration |
|---|---|---|---|
| C_minimal | 0.95±0.01 | 0.59±0.03 | 0.94±0.10 |
| C_filter | 0.96±0.00 | 0.56±0.02 | 0.94±0.10 |
| C_summary | 0.95±0.05 | 0.59±0.03 | 0.99±0.01 |

## 5. 결론 (re-evaluation of v3)

- Judge Overall — C_minimal: 0.646 | C_filter: 0.662 | C_summary: 0.667
- Max Δ between modes = 0.021, max σ = 0.034

### v3 → v4 비교

| Mode | v3 Overall | v4 Overall | Δ |
|---|---|---|---|
| C_minimal | 0.53 | 0.65 | +0.12 |
| C_filter | 0.56 | 0.66 | +0.10 |
| C_summary | 0.52 | 0.67 | +0.15 |

### 해석

→ **C_summary marginal best (Δ < σ)** — v3 trigger 버그 artifact 확인됨. Quality는 동등, **차이는 효율성에서**.

## 6. 효율성 분석 (Quality × Cost)

| Mode | Judge | Wall time/run | LLM summary calls | 평균 압축 prompt size (log≥30) |
|---|---|---|---|---|
| C_minimal | 0.646 | 649s (10.8min) | 0 | 8421 chars |
| C_filter | 0.662 | 601s (10.0min) | 0 | 2572 chars |
| C_summary | 0.667 | 804s (13.4min) | 25 | 3418 chars |

- **최단 시간**: C_filter (601s/run)
- **최소 압축 prompt**: C_filter (2572 chars 평균)

### 핵심 학습

1. **v3 비교는 무효**: trigger가 거의 fire하지 않아 사실상 "raw vs sliding" 비교였음
2. **v4에서 fair 비교**: C_summary (실제로 LLM 요약 fire) vs C_filter (sliding) → quality 거의 동등 (Δ < σ)
3. **효율성에서 갈림**: C_filter가 wall time, LLM cost 모두 우세 → **quality 동등 시 효율성 우위**
4. **컨텍스트 안전성**: 32K + 모든 모드 failures=0 → 컨텍스트 한계가 진짜 bottleneck (compaction 종류 아님)

### 시스템 결정

현 default = **C_filter** 유지. 근거:
- Quality: C_summary와 동등 (Δ 0.005, σ 0.034 내)
- Wall time: C_filter ~600s/run vs C_summary ~800s/run (~33% 추가)
- LLM cost: C_filter 0 extra calls vs C_summary 25 extra calls/run
- Prompt size: C_filter steady ~1.8K vs C_summary 평균 더 큼 (요약 + recent + summary가 모두 들어감)
- 결론: **quality 동등 + 효율성 우세 → C_filter 채택 정당**
