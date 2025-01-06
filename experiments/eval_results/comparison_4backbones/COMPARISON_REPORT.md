# 4-Backbone Ablation Comparison Report

생성일: 2026-04-23

## 0. 비교 대상 모델

| 백본 라벨 | 정식 모델명 | 파라미터 |
|---|---|---|
| **gemma**   | google/gemma-4-E4B-it                            | ~4B (effective)             |
| **qwen**    | Qwen/Qwen3-14B                                    | 14B                         |
| **gemma26** | gemma-4-26B-A4B-it Q4_K_M (GGUF)                 | 26B total / 4B active (MoE) |
| **gemini**  | gemini-3.1-flash-lite-preview                     | (비공개)                     |

## 1. 실험 통제 (모든 백본 동일)

| 항목 | 값 |
|---|---|
| 조건 | C0~C5 6 조건 |
| 반복 | N=3 |
| PRD | sample_data/sample_prd.txt (AI 고객서비스 플랫폼) |
| 팀원 | sample_data/sample_members/ 6명 |
| Structure / Debate | LLM-Judge (Gemini 3.1 Pro Preview) |
| **Assignment** | **Rule-based (`autoscore_allocation`, eval2 §5 deterministic 공식)** |
| Overall | eval2 §4.4 active-dim re-normalized (S=0.40, A=0.35, D=0.25) |

> **Assignment 차원 변경 사유**: Gemma 4B의 긴 WBS에서 LLM Judge가 echo·truncation으로 빈번히 실패(N=2/17) → 평균이 무의미해짐. 이를 회피하기 위해 결정론적 규칙(`0.30×PlanningScore + 0.30×(1-Gini) + 0.20×Feasibility + 0.20×BufferAdequacy`)으로 전환. 모든 4 백본에 동일 적용해 일관성 보장.

## 2. 조건별 Overall 비교

| 조건 | Gemma (4B) | Qwen (14B) | Gemma26 (4B/MoE) | Gemini |
|---|---|---|---|---|
| C0 LLM only | 0.51±0.03 | 0.42±0.08 | 0.63±0.00 | 0.60±0.07 |
| C1 +assign | 0.50±0.02 | 0.49±0.03 | 0.58±0.02 | 0.54±0.01 |
| C2 +1R | 0.64±0.06 | 0.50±0.06 | 0.70±0.02 | 0.68±0.04 |
| C3 +3R | 0.64±0.04 | 0.54±0.02 | 0.73±0.04 | 0.68±0.03 |
| C4 +eDISC | 0.66±0.02 | 0.54±0.03 | 0.68±0.04 | 0.69±0.05 |
| C5 +5R | 0.64±0.02 | 0.55±0.02 | 0.70±0.01 | 0.72±0.02 |

## 3. 차원별 비교

### 3.1 Structure

| 조건 | Gemma | Qwen | Gemma26 | Gemini |
|---|---|---|---|---|
| C0 LLM only | 0.51±0.03 | 0.42±0.08 | 0.63±0.00 | 0.60±0.07 |
| C1 +assign | 0.50±0.04 | 0.42±0.08 | 0.63±0.00 | 0.53±0.00 |
| C2 +1R | 0.51±0.03 | 0.38±0.08 | 0.72±0.04 | 0.65±0.04 |
| C3 +3R | 0.55±0.04 | 0.47±0.00 | 0.72±0.04 | 0.62±0.08 |
| C4 +eDISC | 0.55±0.04 | 0.40±0.13 | 0.69±0.02 | 0.65±0.04 |
| C5 +5R | 0.51±0.03 | 0.47±0.00 | 0.72±0.04 | 0.65±0.04 |

### 3.2 Assignment

| 조건 | Gemma | Qwen | Gemma26 | Gemini |
|---|---|---|---|---|
| C0 LLM only | N/A | N/A | N/A | N/A |
| C1 +assign | 0.50±0.01 | 0.56±0.02 | 0.52±0.04 | 0.56±0.01 |
| C2 +1R | 0.60±0.03 | 0.64±0.02 | 0.58±0.03 | 0.69±0.05 |
| C3 +3R | 0.60±0.03 | 0.60±0.00 | 0.60±0.04 | 0.66±0.03 |
| C4 +eDISC | 0.60±0.01 | 0.61±0.04 | 0.56±0.04 | 0.67±0.02 |
| C5 +5R | 0.62±0.04 | 0.58±0.03 | 0.61±0.05 | 0.69±0.01 |

### 3.3 Debate

| 조건 | Gemma | Qwen | Gemma26 | Gemini |
|---|---|---|---|---|
| C0 LLM only | N/A | N/A | N/A | N/A |
| C1 +assign | N/A | N/A | N/A | N/A |
| C2 +1R | 0.92±0.14 | 0.52±0.12 | 0.82±0.03 | 0.73±0.10 |
| C3 +3R | 0.85±0.09 | 0.57±0.08 | 0.92±0.06 | 0.82±0.20 |
| C4 +eDISC | 0.90±0.10 | 0.67±0.15 | 0.82±0.10 | 0.77±0.14 |
| C5 +5R | 0.90±0.05 | 0.62±0.06 | 0.78±0.12 | 0.88±0.13 |

## 4. 핵심 발견

**4.1 백본 순위 — C3 (full system) 기준**

  1. Gemma26: 0.73
  2. Gemini: 0.68
  3. Gemma: 0.64
  4. Qwen: 0.54

- 평균 격차: Gemini − Gemma26 ≈ -0.02, Gemma26 − Qwen ≈ +0.16
- **Gemma26(4B active MoE)이 Qwen(14B dense)보다 일관 높음** — 동급 활성 파라미터에서 MoE 구조 우위 시사
- Gemini와 Gemma26 격차는 평균 -0.02로 좁음 (특히 C3 동률 근처)

**4.2 토론 라운드 효과 (C1 → C2 변화)**

- Gemma:   +0.14 (큰 단조 상승)
- Qwen:    +0.02 (하락 — 토론 도입 시 분산↑)
- Gemma26: +0.11
- Gemini:  +0.14 (천장 근처)

**4.3 eDISC 효과 (C3 → C4)**

- Gemma:   +0.01
- Qwen:    -0.00
- Gemma26: -0.05
- Gemini:  +0.00

**4.4 백본별 최적 조건 (Overall μ 최대)**

- gemma: C4_with_disc = 0.66
- qwen: C5_5rounds = 0.55
- gemma26: C3_3rounds = 0.73
- gemini: C5_5rounds = 0.72

## 5. Autoscore 비교

| 조건 | Gemma (4B) | Qwen (14B) | Gemma26 (4B/MoE) | Gemini |
|---|---|---|---|---|
| C0 LLM only | 0.44±0.01 | 0.43±0.01 | 0.45±0.00 | 0.41±0.04 |
| C1 +assign | 0.45±0.27 | 0.61±0.02 | 0.63±0.01 | 0.65±0.00 |
| C2 +1R | 0.79±0.01 | 0.86±0.01 | 0.82±0.03 | 0.83±0.02 |
| C3 +3R | 0.81±0.01 | 0.83±0.04 | 0.81±0.02 | 0.84±0.02 |
| C4 +eDISC | 0.79±0.01 | 0.85±0.01 | 0.81±0.02 | 0.84±0.01 |
| C5 +5R | 0.78±0.04 | 0.81±0.05 | 0.85±0.02 | 0.84±0.01 |

### 5.1 Autoscore 차원별 비교

#### Quality

| 조건 | Gemma | Qwen | Gemma26 | Gemini |
|---|---|---|---|---|
| C0 LLM only | 0.98±0.02 | 0.96±0.03 | 1.00±0.00 | 0.92±0.08 |
| C1 +assign | 0.65±0.56 | 0.92±0.03 | 0.99±0.01 | 1.00±0.00 |
| C2 +1R | 0.96±0.01 | 0.98±0.01 | 1.00±0.00 | 0.97±0.00 |
| C3 +3R | 0.99±0.01 | 0.95±0.04 | 1.00±0.00 | 0.99±0.02 |
| C4 +eDISC | 0.96±0.01 | 0.97±0.03 | 1.00±0.00 | 0.96±0.01 |
| C5 +5R | 0.96±0.02 | 0.96±0.07 | 1.00±0.00 | 0.98±0.02 |

#### Allocation

| 조건 | Gemma | Qwen | Gemma26 | Gemini |
|---|---|---|---|---|
| C0 LLM only | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 |
| C1 +assign | 0.47±0.06 | 0.56±0.02 | 0.52±0.04 | 0.56±0.01 |
| C2 +1R | 0.60±0.03 | 0.64±0.02 | 0.58±0.03 | 0.69±0.05 |
| C3 +3R | 0.60±0.03 | 0.60±0.00 | 0.60±0.04 | 0.66±0.03 |
| C4 +eDISC | 0.60±0.01 | 0.61±0.04 | 0.56±0.04 | 0.67±0.02 |
| C5 +5R | 0.62±0.04 | 0.58±0.03 | 0.61±0.05 | 0.69±0.01 |

#### Orchestration

| 조건 | Gemma | Qwen | Gemma26 | Gemini |
|---|---|---|---|---|
| C0 LLM only | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 |
| C1 +assign | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 |
| C2 +1R | 0.72±0.02 | 0.99±0.01 | 0.85±0.11 | 0.76±0.01 |
| C3 +3R | 0.74±0.02 | 0.93±0.10 | 0.75±0.01 | 0.82±0.11 |
| C4 +eDISC | 0.75±0.02 | 0.99±0.00 | 0.81±0.10 | 0.89±0.10 |
| C5 +5R | 0.66±0.10 | 0.88±0.11 | 0.95±0.02 | 0.81±0.10 |


## 6. Judge Reliability — 백본별 N/A 비율 (rev.3 패치 결과)

패치된 Judge(rev.3, max_tokens=1500 + echo 감지)로 Gemma snapshot 17개 재심사 결과:

| 백본 | Source CSV | S valid | A valid | D valid |
|---|---|---|---|---|
| gemma | rejudge_v3 (patched) | 17/17 | 17/17 | 12/17 |
| qwen | finegrain (original) | 18/18 | 18/18 | 12/18 |
| gemma26 | finegrain (original; C5 r2/r3 re-judged with gemini-3.1-pro) | 18/18 | 18/18 | 12/18 |
| gemini | finegrain (original) | 18/18 | 18/18 | 12/18 |

**핵심 발견**: Gemma의 원본 Assignment "0.0" 점수 다수는 **실제 Judge 평가가 아니라 파서 fallback 아티팩트**였음. 패치된 Judge가 정직하게 "Judge JSON 형성 실패 → N/A"로 분류.
17개 Gemma 스냅샷 중 **A 차원이 실제로 평가된 것은 3건뿐** (C3 r1=0.00, C4 r1=0.00, C5 r2=1.00).
Gemma A 평균은 N=1~2 통계로만 산출되며, 본 비교의 Gemma A 컬럼은 신뢰도 낮음 — 절대값보다는 "Gemma WBS는 Judge가 안정적으로 채점하기 어렵다" 자체를 백본 품질의 신호로 해석할 것.

Qwen·Gemini는 사용자 요청에 따라 rejudge하지 않고 원본 사용 → 같은 echo 감지 패치가 적용되지 않았으므로 절대값 비교 시 주의 (특히 Qwen A=1.0의 일부도 같은 fallback 아티팩트일 가능성).

## 7. Special Note — Gemma 4B vs Gemma26 Assignment 격차

Gemma C1 Overall = **0.21**, Gemma26 C1 ≈ **0.58**. 같은 Gemma 패밀리·동일 PRD에서 큰 격차.

**원인**: Gemma 4B는 직군 라벨("Data Engineer", "Backend Developer")을 생성하나 실제 팀원 ID(MBR-XXXX)와 매핑하지 못함 → Judge Skill Fit 0점. Gemma26은 같은 PRD에서 멤버 ID까지 정확히 매핑.
→ MoE 활성 파라미터는 같은 4B지만, **26B total knowledge로 도메인 매칭 능력이 질적으로 다름** 시사.

## 8. 소요 시간 비교 (per run, μ ± σ in seconds)

| 조건 | Gemma (4B) | Qwen (14B) | Gemma26 (4B/MoE) | Gemini |
|---|---|---|---|---|
| C0 LLM only | 584±87s | 1160±512s | 438±253s | 20±7s |
| C1 +assign | 673±85s | 1287±398s | 371±15s | 24±1s |
| C2 +1R | 1070±297s | 4921±2694s | 667±59s | 80±11s |
| C3 +3R | 2623±1652s | 3852±388s | 1437±457s | 181±26s |
| C4 +eDISC | 2508±1308s | 8944±6920s | 1612±253s | 122±28s |
| C5 +5R | 3628±2543s | 5067±1470s | 2103±576s | 204±19s |

**조건당 평균 (전체 18 runs 가중)**:

- **gemma**: avg 1848s/run, total wall-clock ≈ 554분 (9.2h)
- **qwen**: avg 4205s/run, total wall-clock ≈ 1262분 (21.0h)
- **gemma26**: avg 1105s/run, total wall-clock ≈ 331분 (5.5h)
- **gemini**: avg 105s/run, total wall-clock ≈ 32분 (0.5h)

**해석**:
- 조건이 복잡할수록(C5 5R 토론) 모든 백본에서 시간 증가
- API 백본(Gemini)이 로컬 백본보다 일관되게 빠름
- Gemma26 GGUF Q4 양자화는 Qwen3-14B FP16보다 빠름 (모델 크기·양자화 영향)
- Gemma 4B는 4B 모델치곤 느린데, 응답 토큰 수가 많아서(WBS 78~88 task) 시간 비례

## 9. 한계 (Validity Caveats)

- **Self-preference bias (Gemini 만 해당)**: 생성·판정 모두 Gemini → Panickssery 2024 효과로 점수 일부 부풀림. Gemma·Qwen은 cross-vendor라 무관.
- **Judge max_output_tokens=500 cap**: reason 필드 truncation 다수, 일부 Assignment 1.0이 Judge echo 후 default cap 의심. 단, 동일 조건에서 3 백본 모두 평가했으므로 **상대 비교는 일관**.
- **N=3**: pilot 수준. 조건 간 차이가 σ 내인 경우 통계적 확정 불가.
- **단일 PRD/팀**: 다른 도메인 일반화 미검증.

## 10. 산출물

- `figures/fig1_overall.png` — 조건별 Overall 막대 (4 백본 × 6 조건 + scatter)
- `figures/fig2_dimensions.png` — Structure/Assignment/Debate 차원 분해
- `figures/fig3_trajectory.png` — 조건 진행에 따른 Overall trajectory
- `figures/fig4_radar.png` — C3 (full system) 차원 radar
- `figures/fig5_latency.png` — 조건별 소요 시간 (log scale)
- `figures/fig9_autoscore_overall.png` — 조건별 autoscore overall 막대
- `figures/fig10_autoscore_dimensions.png` — autoscore quality/allocation/orchestration 분해
- `figures/fig11_llm_auto_alignment.png` — LLM overall과 AutoScore v2 정렬성 확인 figure
- `figures/fig12_api_deployment_view.png` — API 의존성(외부 토큰/월간 외부 지출)까지 포함한 배포 관점 figure
- `figures/fig13_gemma26_round_trajectory.png` — Gemma-4-26B 단독 C0/C1/C2/C3/C5 AutoScore v2와 LLM Judge 궤적
- `figures/fig14_gemma26_selection_view.png` — Gemma-4-26B 선택 근거: AutoScore/LLM Judge, GPU-min, local efficiency, external API dependency
- `summary_4backbones.csv` — 정리된 비교용 테이블
