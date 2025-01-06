# Reasoning Mode Ablation Experiment — Final Report

생성일: 2026-04-25  |  N=3, C3 3R, Gemma-4-26B, gemini-3.1-pro-preview judge

## 0. 사전 등록 가설

- **H1**: R_max > R_high > R_none (Reasoning 정교화 → quality 향상)
- **H2**: latency R_max > R_high > R_none (요약 LLM 추가)
- **H3**: 모든 mode의 sub-agent failure rate < 10%/run (시스템 안전 영역)

## 1. 조건 (Reasoning instruction strength, prompt-engineered)

| Mode | 적용 범위 | Prompt prefix |
|---|---|---|
| **R_none** | 모든 agent | (없음 — default) |
| **R_high** | 모든 agent | "단계적으로 추론. 옵션 비교 + trade-off 명시." |
| **R_max**  | 모든 agent | CoT 4-step (제약·대안·trade-off·결정) 강제 |

## 2. 통제

- 백본/요약: Gemma-4-26B-A4B-it Q4_K_M (localhost:8081, OpenAI-compat endpoint)
- 조건: C3_3rounds (3R debate, Task Manager + 5 sub-agents)
- Judge: gemini-3.1-pro-preview, reasoning run 당시 저장된 archived judge output 사용
- N=3, 코드 무수정 (monkey-patch)

## 3. 결과 (archived judge outputs)

| Mode | Structure | Assignment | Debate | **Overall** | Sub-agent failures |
|---|---|---|---|---|---|
| R_none | 0.70±0.00 | 0.69±0.03 | 0.85±0.13 | **0.73±0.04** | 0.00±0.00 |
| R_high | 0.70±0.00 | 0.71±0.02 | 0.95±0.00 | **0.77±0.01** | 0.00±0.00 |
| R_max | 0.76±0.05 | 0.67±0.07 | 0.97±0.03 | **0.78±0.01** | 0.00±0.00 |

## 4. Auto score 결과

| Mode | Quality | Allocation | Orchestration | **Overall Auto** |
|---|---|---|---|---|
| R_none | 1.00±0.00 | 0.58±0.03 | 0.74±0.02 | **0.80±0.01** |
| R_high | 1.00±0.00 | 0.56±0.02 | 0.81±0.12 | **0.81±0.02** |
| R_max | 1.00±0.00 | 0.61±0.06 | 0.76±0.02 | **0.82±0.03** |

→ `Quality`는 세 mode 모두 **1.00**으로 포화되어, 비교 지표라기보다 **guardrail**로 해석하는 편이 맞습니다. 즉 reasoning mode 차이는 주로 `Allocation`과 `Orchestration`에서 읽어야 합니다.

## 5. 가설 평가

### H1 (R_max > R_high > R_none)
- R_none = 0.734, R_high = 0.766, R_max = 0.778
- ✅ 지지 (단조 순서 확인)

### H3 (sub-agent failure < 10%/run)
- R_none failures/run: 0.0, R_high: 0.0, R_max: 0.0
- ✅ 안전 영역 — 모든 mode의 sub-agent 호출 안정적

## 6. 한계

- N=3 — Δ Overall이 σ 내면 통계 유의성 불가, effect size로만 시사적
- C3 단일 — 5R+ 더 강한 메모리 압력에서 효과 다를 수 있음
- 단일 백본 (Gemma 4-26B 16K) — 더 큰 컨텍스트 모델 미검증
- Assignment 차이는 특히 보수적으로 해석해야 함. `R_max` assignment는 0.73/0.60/0.67로 분산이 상대적으로 크고, N=3 기준 95% CI가 넓어 (`약 0.51~0.83`) mode 간 우열을 강하게 주장하기 어렵습니다.
- 따라서 reasoning mode의 안정적 신호는 Assignment보다 **Debate/Overall 및 autoscore overall**에서 읽는 편이 타당합니다.

## 7. 산출물

```
reasoning_mode_experiment/
├── REASONING_REPORT.md
├── summary_judged.csv          (archived judge output canonicalized)
├── figures/fig1~8.png
├── snapshots/                  (9 wbs+debate JSON)
├── samples/                    (각 mode별 Reasoning 입력 sample)
├── logs/                       (실행 로그)
├── run_reasoning.py            (재현 launcher)
├── rejudge_median.py           (3-trial median judge)
├── build_report.py             (이 figure+report 생성)
└── orchestrate.sh              (전체 pipeline)
```
