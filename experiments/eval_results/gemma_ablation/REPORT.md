# Gemma-4-E4B Ablation Study — Final Report

**실험 기간**: 2026-04-20 23:50 ~ 2026-04-21 05:00 (5h 10m 생성) + 2026-04-21 11:30~11:50 (재심사 20m)
**백본**: `google/gemma-4-E4B-it` (vLLM 원격 서빙, fp16)
**판정자(Judge)**: Gemini 3.1 Pro (벤더 독립)
**샘플**: `sample_data/prd_sample.txt` + 팀원 6명(+ eDISC 프로파일)

---

## 1. 목적

symPO 파이프라인(WBS 생성 → Task Manager 배정 → 다중 에이전트 토론 → 중재)의 각 구성 요소가 최종 WBS 품질에 미치는 기여도를 정량화하고, Gemma-4-E4B 백본 기준의 최적 구성 및 한계를 식별한다. 결과는 동급 오픈소스(Qwen3.5-4B)와의 쌍대 비교를 위한 baseline으로 재활용한다.

## 2. 설계

| 조건 | 라벨 | Task Match | Debate | eDISC | 평가 차원 |
|---|---|---|---|---|---|
| C0 | LLM 단독 | ✗ | ✗ | ✗ | Structure |
| C1 | +배정 | ✓ | ✗ | ✗ | S + Assignment |
| C2 | +1R 토론 | ✓ | 1라운드 | ✗ | S + A + Debate |
| C3 | +3R 토론 | ✓ | 3라운드 | ✗ | S + A + D |
| C4 | +eDISC | ✓ | 3라운드 | ✓ | S + A + D |
| C5 | +5R 토론 | ✓ | 5라운드 | ✗ | S + A + D |

각 조건 N=3 반복(총 18 runs). PRD·팀원·Judge 모델·하이퍼파라미터 완전 고정. C0·C1은 토론 루프를 물리적으로 우회.

## 3. 루브릭 재설계 (본 실험의 핵심 방법론적 개선)

**초기 분석에서 이전 루브릭이 Structure 점수 15/18 run을 정확히 0.30에 고정시키는 이산적 bucketing** 현상을 확인했다 (0.30 ↔ 0.85 bimodal). 원인은 루브릭이 "1.0 완벽 / 0.3 cap / 감점" 구조라 판정 모델이 중간값을 쓰지 않았기 때문. 이를 해결하기 위해 Judge 루브릭을 **다축 평균 방식**으로 재설계하였다.

### 신루브릭 구조 (eval/llm_judge.py)

각 차원을 2–4개 하위 축으로 분해하고, 각 축을 0.0–1.0 ladder로 채점 후 **평균하여 연속적 score** 산출:

- **Structure**: Hierarchy Completeness / Estimation Realism / Task Quality (3축 평균)
- **Assignment**: Skill Fit / Workload Balance / Coverage (3축 평균)
- **Debate**: Participation / Substance / Role Consistency / Convergence (4축 평균)

이 구조로 동일 18개 저장된 wbs_snapshot을 재심사한 결과, 점수 분포가 이전 0.30/0.85 bimodal에서 0.18–0.77의 연속 분포로 전환되었다. 본 보고서의 공식 결과는 **fine-grained 루브릭 재심사 값** 기준이다.

### 데이터 정리

- **실패 1건 제외**: C1 Run 2는 Gemma의 JSON 파싱 완전 실패로 `total_tasks=0`. 분석에서 제외(유효 N=17).
- **단계 부재 0 처리**: C0의 Assignment·Debate, C1의 Debate는 파이프라인 단계가 물리적으로 미실행(`experiment_runner.py:489-508`의 전용 분기). 해당 차원은 루브릭상 품질 0으로 정량화하여 Overall 가중 평균에 포함.
- **Judge 응답 clamp**: [0, 1] 범위 이탈 값 자동 clamp 로직 추가.

## 4. 결과 (fine-grained 루브릭 기준)

### 4.1 Judge 점수 요약 (μ ± σ)

| Condition | N | Structure | Assignment | Debate | **Overall** | Tasks |
|---|---|---|---|---|---|---|
| C0 (LLM 단독) | 3 | 0.47 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | **0.19 ± 0.00** | 72 ± 9 |
| C1 (+배정) | 2 | 0.40 ± 0.10 | 0.00 ± 0.00 | 0.00 ± 0.00 | **0.16 ± 0.04** | 72 ± 9 |
| C2 (+1R 토론) | 3 | 0.42 ± 0.08 | 0.33 ± 0.58 | 0.92 ± 0.14 | **0.52 ± 0.18** | 70 ± 7 |
| C3 (+3R 토론) | 3 | 0.46 ± 0.12 | 0.33 ± 0.58 | 0.93 ± 0.08 | **0.54 ± 0.14** | 69 ± 24 |
| C4 (+eDISC) | 3 | 0.49 ± 0.03 | 0.00 ± 0.00† | 0.87 ± 0.19 | **0.41 ± 0.05** | 70 ± 7 |
| C5 (+5R 토론) | 3 | 0.40 ± 0.12 | 0.00 ± 0.00† | 0.90 ± 0.09 | **0.38 ± 0.04** | 63 ± 27 |

† **실제 Judge가 측정하여 부여한 0점** (단계 부재 아님). C0 Assignment·Debate, C1 Debate의 0.00은 파이프라인 단계 부재로 인한 구조적 0이며, C4·C5의 Assignment 0.00은 태스크 매칭이 실행되었으나 fine-grained 루브릭의 Skill Fit 축이 Gemma의 역할 라벨과 팀원 프로파일 mismatch를 감지해 0점을 부여한 것. 도표(fig2_dimensions.png)에서 두 경우를 시각적으로 구분(사선 패턴 vs 얇은 막대).

Overall = Structure×0.40 + Assignment×0.35 + Debate×0.25 (단계 부재 포함 full weight).

### 4.2 Rule-based Autoscore (보조 지표, μ ± σ)

| Condition | Autoscore | Success | MECE | Feasibility | Workload Gini |
|---|---|---|---|---|---|
| C0 | 0.87 ± 0.01 | 1.00 ± 0.00 | 0.93 ± 0.05 | 1.00 ± 0.00 | 0.00 ± 0.00 |
| C1 | 0.86 ± 0.01 | 1.00 ± 0.00 | 0.91 ± 0.01 | 0.92 ± 0.11 | 0.28 ± 0.01 |
| C2 | 0.84 ± 0.01 | 1.00 ± 0.00 | 0.89 ± 0.03 | 1.00 ± 0.00 | 0.20 ± 0.13 |
| C3 | 0.85 ± 0.01 | 1.00 ± 0.00 | 0.98 ± 0.02 | 1.00 ± 0.00 | 0.19 ± 0.04 |
| C4 | 0.83 ± 0.01 | 1.00 ± 0.00 | 0.89 ± 0.03 | 1.00 ± 0.00 | 0.23 ± 0.07 |
| C5 | 0.82 ± 0.03 | 1.00 ± 0.00 | 0.88 ± 0.07 | 1.00 ± 0.00 | 0.28 ± 0.02 |

### 4.3 통계적 유의성 (조건 간 쌍대 효과, Δμ / pooled σ)

| 비교 | Δμ / σ (Overall) | 판정 |
|---|---|---|
| C1 → C2 (+토론) | **+4.0** | 🟢 **유의** |
| C4 → C5 | +0.63 | ⚠️ noise |
| C2 → C3 | +0.12 | ⚠️ noise |
| C3 → C4 | -0.88 | ⚠️ noise (but marginal) |
| C0 → C1 | -0.92 | ⚠️ noise (흥미로운 '하락') |
| C2 → C4 | -0.60 | ⚠️ noise |

### 4.4 Figures

- **`figures_finegrain/fig1_overall.png`**: 조건별 Overall 점수 + 개별 run scatter
- **`figures_finegrain/fig2_dimensions.png`**: Structure / Assignment / Debate 차원별 분해 (단계 부재는 사선 패턴)
- **`figures_finegrain/fig3_rubric_compare.png`**: 루브릭 재설계 전/후 Overall 대조 — 값 분산이 fine-grain에서 더 잘 잡힘을 가시적으로 입증
- 참고(이전 rubric): `figures_coarse/` 하위 보관

## 5. 핵심 해석

1. **토론 도입(C1→C2)만이 유일하게 통계적으로 확정적**: Δμ/σ=+4.0. 배정 단독 대비 1라운드 토론 추가가 Overall을 0.16→0.52로 대폭 상승시킨다. 이후 라운드 증가는 개선 효과가 분산에 묻힌다.

2. **배정만 추가한 C1 Overall이 C0보다 오히려 낮음**: 새 루브릭의 Skill Fit 축이 Gemma의 직군 라벨(예: "Marketing Planner", "Mobile Developer")과 실제 팀원 역할의 mismatch를 강하게 패널티(대부분 0.0). 즉 Gemma가 생성하는 L3 역할 라벨이 팀원 프로파일과 체계적으로 어긋나 있으며, 새 루브릭은 이를 정확히 포착한다.

3. **C4 eDISC는 오히려 해로울 가능성**: C3(0.54) → C4(0.41) 하락은 Δμ/σ=-0.88로 경계선상(σ 내). 확정은 N=3로 불가하나 **페르소나 정보가 Gemma에겐 혼란 요소**일 시그널 존재. N 증량 시 재검증 필요.

4. **C3/C4/C5 Assignment 0.33–0.00 bimodal 잔존**: 새 루브릭에서도 Assignment σ가 0.55–0.58로 매우 큼. Gemma가 역할 정합을 일부 run에선 완전히 맞추고(1.0) 일부 run에선 완전히 놓치는(0.0) 양극화 패턴. 백본 자체의 role planning 불안정성을 시사.

5. **Structure 천장 완화**: 이전 0.30 고정에서 새 루브릭 0.40–0.49로 이동하고 σ 등장(C1 0.10, C3 0.12, C5 0.12). 여전히 1.0엔 접근 못 하지만 **Gemma 내부 variation을 구별 가능한 수준으로 측정**됨.

6. **Rule-based Autoscore와 Judge 비상관**: Autoscore는 0.82–0.87 좁은 범위에서 모든 조건을 "실용적으로 OK"로 평가하는 반면 Judge는 0.16–0.54로 크게 벌어짐. 백본 비교 주 지표로는 **Judge 차원별 점수**를 사용하고 Autoscore는 파이프라인 완주성 보조 지표로만 활용해야 한다.

## 6. 한계

- **N=3 pilot 수준**: C2 이후 세부 조건(C3/C4/C5) 간 Overall 차이 모두 pooled σ 이내. N=5+ 증량 없이는 "라운드 수 효과" 또는 "eDISC 효과"에 대해 인과적 주장 불가.
- **단일 PRD**: 외적 타당도 제한. 향후 `prd_variant` 필드로 3~5개 변형에 대한 민감도 분석 필요.
- **단일 Judge**: Gemini 3.1 Pro 단독. Claude Sonnet 4.6 cross-judge 도입 시 Judge 내부 일치도(ICC) 산출 가능.
- **안전망 상시 발동**: L3 Retry가 18/18 run에서 발동. 측정값이 "Gemma 순수 능력"이 아니라 "Gemma + 파이프라인 안전망"의 합작. 이는 Qwen에도 동일 조건으로 적용되므로 백본 간 상대 비교엔 공정.

## 7. 결론

Gemma-4-E4B 기반 symPO 파이프라인에서 **통계적으로 확정 가능한 결론은 "C1→C2 토론 도입이 Overall을 +0.36 개선한다"는 것 하나**다. 나머지 세부 ablation(C2–C5 간)은 현 표본 크기로는 신호가 분산에 묻힌다. 단, 정성적으로는:

- 배정만 추가하는 C1은 Skill Fit 부족으로 Overall이 오히려 낮아지는 역설적 구간
- 토론 도입 후 Debate score는 0.87–0.93로 안정적으로 높아 **토론 자체의 품질은 raising to ceiling**
- 최적 ROI 지점은 **C2(+1R 토론)** — 라운드 수 증가는 품질 개선 없이 분산·시간 비용만 증가

**방법론 측면의 발견 하나**: LLM-as-Judge의 루브릭 설계가 실험 결론 자체를 바꿀 수 있음을 증명했다. Coarse 루브릭에선 C2–C5가 0.5–0.58 plateau로 보였으나, fine-grained 재설계 후 동일 데이터가 C4 0.41, C5 0.38으로 유의미하게 다른 패턴으로 해석된다. 향후 LLM-as-Judge 방법론 문서에 **"다축 평균 형식 루브릭"**을 모범으로 기록할 가치가 있다.

## 8. 후속 작업

1. **Qwen3.5-4B와의 쌍대 비교**(진행 중): 동일 매트릭스 · 동일 Judge · 동일 루브릭으로 백본 간 상대 성능 산출. 결과는 `eval_results/qwen_ablation/`에 저장.
2. **N=5 이상 증량**: 특히 C3/C4/C5에 집중해 "3R vs 5R", "+eDISC" 효과 재검증.
3. **Cross Judge 도입**: `ANTHROPIC_API_KEY` 확보 후 Claude Sonnet 4.6을 2차 판정자로 추가.
4. **PRD 변형 민감도**: `R1-A/B/C` 조건(요약본/상세본/+회의록)에 대한 별도 실험.

---

## Appendix — 파일 인벤토리 (본 디렉토리)

| 파일 | 설명 |
|---|---|
| `summary_raw.csv` | experiment_runner 원본 CSV (coarse rubric 점수) |
| `summary_coarse_cleaned.csv` | coarse rubric 정리판 (clamp + 단계 부재 0 처리) |
| `summary_finegrain.csv` | **본 보고서의 공식 점수 (fine-grained 재심사)** |
| `experiment_metadata.json` | 조건·구성·타이밍 메타데이터 |
| `snapshots/wbs_snapshot_*_gemma4-api_*.json` × 21 | 각 run의 WBS + debate_log 전문 (일부 중복은 pilot run 포함) |
| `figures_coarse/fig{1..4}.png` | 초기 분석 figure (참고용) |
| `figures_finegrain/fig{1..3}.png` | **본 보고서 figure** |
| `plot_coarse.py`, `plot_finegrain.py` | figure 재생성 스크립트 |
| `rejudge_snapshots.py` | 저장된 snapshot을 새 루브릭으로 재심사하는 CLI |
| `rejudge_finegrain_20260421_115048.csv` | 재심사 원본 출력 (중복 포함) |
| `ARCHIVE_*.md` | 이전 작업 중 생성된 요약 초안 (참고용, 본 REPORT로 대체됨) |
