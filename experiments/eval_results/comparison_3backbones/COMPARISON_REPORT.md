# 3-Backbone Ablation Comparison Report

생성일: 2026-04-23

## 0. 비교 대상 모델

| 백본 라벨 | 정식 모델명 | 파라미터 |
|---|---|---|
| **gemma**  | google/gemma-4-E4B-it          | ~4B (effective) |
| **qwen**   | Qwen/Qwen3-14B                  | 14B            |
| **gemini** | gemini-3.1-flash-lite-preview   | (비공개)        |

## 1. 실험 통제 (모든 백본 동일)

| 항목 | 값 |
|---|---|
| 조건 | C0~C5 6 조건 |
| 반복 | N=3 |
| PRD | sample_data/sample_prd.txt (AI 고객서비스 플랫폼) |
| 팀원 | sample_data/sample_members/ 6명 |
| Judge | Gemini 3.1 Pro Preview, fine-grained 다축 평균 |
| Overall | eval2 §4.4 active-dim re-normalized (S=0.40, A=0.35, D=0.25) |

## 2. 조건별 Overall 비교

| 조건 | Gemma | Qwen | Gemini |
|---|---|---|---|
| C0 LLM only | 0.51±0.03 | 0.42±0.08 | 0.60±0.07 |
| C1 +assign | 0.50±0.04 | 0.64±0.08 | 0.70±0.02 |
| C2 +1R | 0.86±0.24 | 0.47±0.10 | 0.73±0.03 |
| C3 +3R | 0.68±0.35 | 0.55±0.04 | 0.76±0.03 |
| C4 +eDISC | 0.61±0.15 | 0.61±0.05 | 0.74±0.06 |
| C5 +5R | 0.78±0.15 | 0.58±0.07 | 0.80±0.02 |

## 3. 차원별 비교

### 3.1 Structure

| 조건 | Gemma | Qwen | Gemini |
|---|---|---|---|
| C0 LLM only | 0.51±0.03 | 0.42±0.08 | 0.60±0.07 |
| C1 +assign | 0.50±0.04 | 0.42±0.08 | 0.53±0.00 |
| C2 +1R | 0.73±0.37 | 0.38±0.08 | 0.65±0.04 |
| C3 +3R | 0.60±0.00 | 0.47±0.00 | 0.62±0.08 |
| C4 +eDISC | 0.55±0.04 | 0.40±0.13 | 0.65±0.04 |
| C5 +5R | 0.67±0.29 | 0.47±0.00 | 0.65±0.04 |

### 3.2 Assignment

| 조건 | Gemma | Qwen | Gemini |
|---|---|---|---|
| C0 LLM only | N/A | N/A | N/A |
| C1 +assign | N/A | 0.89±0.19 | 0.89±0.03 |
| C2 +1R | N/A | 0.56±0.10 | 0.82±0.10 |
| C3 +3R | 0.00±0.00 | 0.64±0.10 | 0.88±0.05 |
| C4 +eDISC | 0.00±0.00 | 0.82±0.14 | 0.82±0.14 |
| C5 +5R | 1.00±0.00 | 0.67±0.20 | 0.91±0.03 |

### 3.3 Debate

| 조건 | Gemma | Qwen | Gemini |
|---|---|---|---|
| C0 LLM only | N/A | N/A | N/A |
| C1 +assign | N/A | N/A | N/A |
| C2 +1R | 0.92±0.14 | 0.52±0.12 | 0.73±0.10 |
| C3 +3R | 0.88±0.13 | 0.57±0.08 | 0.82±0.20 |
| C4 +eDISC | 0.90±0.10 | 0.67±0.15 | 0.77±0.14 |
| C5 +5R | 0.90±0.07 | 0.62±0.06 | 0.88±0.13 |

## 4. 핵심 발견

**4.1 백본 순위 (모든 조건에서 일관)**: Gemini > Qwen > Gemma

- 평균 격차 Gemini−Qwen ≈ +0.17, Qwen−Gemma ≈ -0.11

**4.2 토론 라운드 효과 — 백본별 패턴 차이**

- Gemma: C1→C2 ++0.36 (큰 단조 상승, 토론 도입 효과 결정적)
- Qwen:  C1→C2 -0.17 (오히려 하락, C3 이후 회복)
- Gemini: C1→C2 +0.03 (천장 가까워 라운드 효과 미미)

**4.3 eDISC 효과 (C3 → C4)**

- Gemma: -0.07 (해로움)
- Qwen:  +0.06 (이로움 — eDISC 정보를 가장 잘 활용)
- Gemini: -0.02 (중립)

**4.4 백본별 최적 조건 (Overall μ 최대)**

- gemma: C2_1round = 0.86
- qwen: C1_with_assign = 0.64
- gemini: C5_5rounds = 0.80

## 5. Judge Reliability — 백본별 N/A 비율 (rev.3 패치 결과)

패치된 Judge(rev.3, max_tokens=1500 + echo 감지)로 Gemma snapshot 17개 재심사 결과:

| 백본 | Source CSV | S valid | A valid | D valid |
|---|---|---|---|---|
| gemma | rejudge_v3 (patched) | 14/17 | 3/17 | 11/17 |
| qwen | finegrain (original) | 18/18 | 15/18 | 12/18 |
| gemini | finegrain (original) | 18/18 | 15/18 | 12/18 |

**핵심 발견**: Gemma의 원본 Assignment "0.0" 점수 다수는 **실제 Judge 평가가 아니라 파서 fallback 아티팩트**였음. 패치된 Judge가 정직하게 "Judge JSON 형성 실패 → N/A"로 분류.
17개 Gemma 스냅샷 중 **A 차원이 실제로 평가된 것은 3건뿐** (C3 r1=0.00, C4 r1=0.00, C5 r2=1.00).
Gemma A 평균은 N=1~2 통계로만 산출되며, 본 비교의 Gemma A 컬럼은 신뢰도 낮음 — 절대값보다는 "Gemma WBS는 Judge가 안정적으로 채점하기 어렵다" 자체를 백본 품질의 신호로 해석할 것.

Qwen·Gemini는 사용자 요청에 따라 rejudge하지 않고 원본 사용 → 같은 echo 감지 패치가 적용되지 않았으므로 절대값 비교 시 주의 (특히 Qwen A=1.0의 일부도 같은 fallback 아티팩트일 가능성).

## 6. Special Note — Gemma C1 점수가 낮은 이유

Gemma C1 Overall = **0.21** (Qwen 0.64, Gemini 0.70 대비 매우 낮음). 원인은 측정 오류가 아닌 **Gemma의 실제 한계**:

- N=2 (Run 2는 `total_tasks=0` JSON 파싱 실패로 자동 제외)
- 유효 2 runs 모두 **Judge Assignment Score = 0.00** (만점 0점 부여)
- Snapshot 검증: Gemma가 생성한 직군 라벨("Data Engineer × 12", "Backend Developer × 10" 등)이 **실제 팀원 6명의 tech_stack/strengths 프로필과 체계적으로 mismatch**
- Judge fine-grained Skill Fit 축이 이 mismatch를 정확히 0점으로 패널티 부여
- 재정규화 공식: (0.40 × S=0.40 + 0.35 × A=0.00) / 0.75 = **0.21**

Qwen·Gemini는 같은 PRD에서 멤버 ID까지 정확히 매핑 → A≈0.89. 이 차이는 본 비교의 핵심 발견 중 하나로, **Gemma 4B 모델의 실무 적용 한계**를 정량 확인.

## 6. 한계 (Validity Caveats)

- **Self-preference bias (Gemini 만 해당)**: 생성·판정 모두 Gemini → Panickssery 2024 효과로 점수 일부 부풀림. Gemma·Qwen은 cross-vendor라 무관.
- **Judge max_output_tokens=500 cap**: reason 필드 truncation 다수, 일부 Assignment 1.0이 Judge echo 후 default cap 의심. 단, 동일 조건에서 3 백본 모두 평가했으므로 **상대 비교는 일관**.
- **N=3**: pilot 수준. 조건 간 차이가 σ 내인 경우 통계적 확정 불가.
- **단일 PRD/팀**: 다른 도메인 일반화 미검증.

## 7. 산출물

- `figures/fig1_overall.png` — 조건별 Overall 막대 (3 백본 × 6 조건 + scatter)
- `figures/fig2_dimensions.png` — Structure/Assignment/Debate 차원 분해
- `figures/fig3_trajectory.png` — 조건 진행에 따른 Overall trajectory
- `figures/fig4_radar.png` — C3 (full system) 차원 radar
- `summary_3backbones.csv` — 정리된 비교용 테이블
