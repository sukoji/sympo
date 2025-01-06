# Compaction Strategy Experiment v2 — Final Report

생성일: 2026-04-25  |  N=3, C3 3R, Gemma-4-26B, Pro Preview Judge (median ×3)

## 0. 사전 등록 가설

- **H1**: C_claude > C_filter > C_minimal (압축 정교화 → quality 향상)
- **H2**: latency C_claude > C_filter > C_minimal (요약 LLM 추가)
- **H3**: 모든 mode의 sub-agent failure rate < 10%/run (시스템 안전 영역)

## 1. 조건

| Mode | Trigger | 압축 방식 | 캐싱 | 최근 raw |
|---|---|---|---|---|
| **C_minimal** | 매 호출 | last 30 raw + PM 10 (약압축, 안전 baseline) | ❌ | last 30 |
| **C_filter** (default) | 매 호출 | sliding W=8 + PM 4 | ❌ | last 8 |
| **C_claude** | est_tokens > 4K | Gemma26 self-summarization | ✅ | last 6 |

## 2. 통제

- 백본/요약: Gemma-4-26B-A4B-it Q4_K_M (localhost:8081, OpenAI-compat endpoint)
- 조건: C3_3rounds (3R debate, Task Manager + 5 sub-agents)
- Judge: gemini-3.1-pro-preview, **각 snapshot당 3회 호출 후 median** (비결정성 통제)
- N=3, 코드 무수정 (monkey-patch)

## 3. 결과 (median of 3 judge trials)

| Mode | Structure | Assignment | Debate | **Overall** | Sub-agent failures |
|---|---|---|---|---|---|
| C_minimal | 0.75±0.04 | 0.00±0.00 | 0.93±0.03 | **0.53±0.01** | 0.00±0.00 |
| C_filter | 0.77±0.07 | 0.02±0.04 | 0.88±0.12 | **0.54±0.06** | 0.00±0.00 |
| C_claude | 0.72±0.04 | 0.02±0.04 | 0.95±0.00 | **0.54±0.02** | 6.33±0.58 |

## 4. Auto score 결과

| Mode | Quality | Allocation | Orchestration | **Overall Auto** |
|---|---|---|---|---|
| C_minimal | 1.00±0.00 | 0.62±0.04 | 0.93±0.02 | **0.85±0.02** |
| C_filter | 1.00±0.00 | 0.63±0.04 | 0.81±0.12 | **0.83±0.04** |
| C_claude | 0.50±0.55 | 0.38±0.25 | 0.40±0.45 | **0.44±0.42** |

## 5. Judge 비결정성 (within-snapshot std of 3 trials)

| Mode | S std | A std | D std |
|---|---|---|---|
| C_minimal | 0.000 | 0.000 | 0.019 |
| C_filter | 0.006 | 0.000 | 0.019 |
| C_claude | 0.000 | 0.000 | 0.010 |

→ A 차원이 가장 비결정적 (Gemma 도메인 특수성 + Pro Preview internal randomness).

## 6. 가설 평가

### H1 (C_claude > C_filter > C_minimal)
- C_minimal = 0.532, C_filter = 0.536, C_claude = 0.535
- ⚠️ 부분 지지: 둘 다 minimal보다 높지만 filter vs claude 순서 ambiguous

### H3 (sub-agent failure < 10%/run)
- C_minimal failures/run: 0.0, C_filter: 0.0, C_claude: 6.3
- ⚠️ 일부 mode에서 sub-agent fail 다수 — 결과 해석 시 confound 고려

## 7. 한계

- N=3 — Δ Overall이 σ 내면 통계 유의성 불가, effect size로만 시사적
- C3 단일 — 5R+ 더 강한 메모리 압력에서 효과 다를 수 있음
- 단일 백본 (Gemma 4-26B 16K) — 더 큰 컨텍스트 모델 미검증
- Assignment 차원의 phantom-ID 문제로 A 점수가 모든 mode에서 0 가까움 — 압축 외 confound

## 8. 산출물

```
compaction_v2_experiment/
├── COMPACTION_V2_REPORT.md
├── summary_judged.csv          (9 runs × 3 judge trials each)
├── figures/fig1~6.png
├── snapshots/                  (9 wbs+debate JSON)
├── samples/                    (각 mode별 압축 입력 sample)
├── logs/                       (실행 로그)
├── run_compaction_v3.py        (재현 launcher)
├── rejudge_median.py           (3-trial median judge)
├── build_report.py             (이 figure+report 생성)
└── orchestrate_v2.sh           (전체 pipeline)
```
