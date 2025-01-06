# Gemma-4-E4B Ablation — 유의미 지표 Figure 해석

정리된 CSV(`summary_gemma4-api_piai_20260421_050058_cleaned.csv`) 기준.
**데이터 정리 내역**:
- 실패 1건(C1 Run 2, tasks=0) 제외
- Judge 점수 [0, 1] clamp 적용 (C4 Run 1 Debate 3.0→1.0, Run 2 Assignment 2.0→1.0)
- **파이프라인 단계 부재 차원 = 0으로 정량화** (허위 채움 아님, ablation 코드 경로상 해당 단계 실행 자체가 스킵되므로 루브릭상 quality=0과 동일):
  - C0: `use_task_match=False`, `max_rounds=0` → Assignment·Debate 모두 부재
  - C1: `use_task_match=True`, `max_rounds=0` → Debate만 부재 (배정만 수행)
  - 근거 코드: `eval/experiment_runner.py:489-508`의 조건별 분기

---

## Figure 1 — 조건별 Overall Judge 점수

![fig1](figures/fig1_overall_by_condition.png)

**읽는 법**: 보라색 막대 = 평균, 검은 수직선 = 표준편차, 오렌지 점 = 개별 run. N=3 (C1은 실패 1건 제외 N=2).

### 핵심 해석

- **C0 → C1 → C2 계단 상승** (0.120 → 0.365 → 0.580)
  - C0→C1 기여(+0.245): **Task Manager 배정 기능의 순수 기여**
  - C1→C2 기여(+0.215): **1라운드 토론 도입의 순수 기여**
  - 두 step 모두 유의미한 크기 — 각 파이프라인 요소가 실제로 가치를 더함
- **C3 vs C2 차이 없음** (0.579 ≈ 0.580): 3라운드가 1라운드보다 낫지 않음. σ 큼 → 통계적으로도 유의하지 않음
- **C4 (+eDISC) 0.499**: C2/C3보다 오히려 낮음. Gemma가 페르소나 정보를 효과적으로 활용 못함
- **C5 (+5R) 0.495**: 5라운드는 최악. 라운드 증가가 **한계수익 없이 분산만 증가**

### 결론

> Gemma-4-E4B의 경우 **C2(+1R 토론)이 최적점**. 추가 라운드 / eDISC는 ROI가 없다.
> 단일 구성 요소 중 **배정(+0.245)과 토론(+0.215)의 기여가 거의 대등**.

---

## Figure 2 — Judge 차원별 분해 (Structure · Assignment · Debate)

![fig2](figures/fig2_dimension_breakdown.png)

**읽는 법**: 불투명 막대 = 실제 Judge 측정값, 사선 패턴 막대 = **해당 파이프라인 단계 부재 (=0)**. "0 (단계 부재)" 라벨로 명시.

### 핵심 해석

1. **Structure (빨강) 0.30에 고정** — C3에서만 0.42로 살짝 오름(σ 크고 다른 run은 0.3).
   - **Gemma의 WBS 구조 생성 근본 한계**. 토론·배정·eDISC 어떤 것도 구조 품질을 끌어올리지 못함.
   - 원인 추정: 루브릭이 "5+ L1, 3+ L2, 3+ L3" 엄격 요구. Gemma는 L3 분해 실패로 매번 L3 Retry 안전망 발동 → Judge가 이를 "완전한 구조"로 인정 안 함.

2. **Assignment (초록) 0.7~0.83 안정** — 가장 재현성 높은 차원. 우리의 `_smart_assign` + 0-task 구제 + role compat 로직 일관 작동.

3. **Debate (파랑) 편차 극심**: C2 0.77 → C3 0.48 → C4 0.35 → C5 0.57.
   - C2가 debate 점수 최고. 라운드 늘수록 역할 이탈·빈 동의·주제 벗어남 누적.
   - **C4 (eDISC) Debate 0.35로 최저** — 페르소나가 오히려 토론 혼란 증폭 시사.

### 결론

> 백본 비교 시 **차원별로 나눠 보는 것이 필수**. Overall 평균만 보면 C3/C4/C5의 차이가 묻힘.
> Qwen 실험도 동일 포맷으로 차원별 비교 → 강점·약점이 정확히 어디서 드러나는지 확인 가능.

---

## Figure 3 — LLM Judge vs Rule-based Autoscore

![fig3](figures/fig3_judge_vs_autoscore.png)

### 핵심 해석

- **두 지표는 상관이 거의 없음**. y=x 대각선에서 완전히 벗어남.
- Judge 점수: 0.12 ~ 0.70 범위 (Gemma의 질적 차이를 민감하게 포착)
- Autoscore: 0.78 ~ 0.88 범위 (모든 run이 "실용적으로 OK" 판정)
- **C0 (LLM 단독)이 Autoscore는 가장 높음 (~0.87)**, Judge Overall은 가장 낮음 (0.12) → Autoscore는 "파이프라인 완주·커버리지"를 보지, "구조적 품질"을 못 봄.

### 결론

> **규칙 기반 지표만 보면 Gemma의 구조적 약점이 안 보인다.** 백본 비교의 주 지표로는 LLM Judge 점수(특히 차원별)를 쓰고, Autoscore는 보조로 활용해야 함.

---

## Figure 4 — 조건별 실행 시간 변동성

![fig4](figures/fig4_runtime.png)

### 핵심 해석

- **C0/C1**: 5~6분 (짧고 안정)
- **C2 (+1R)**: 7~11분 (평균 9)
- **C3 (+3R)**: 10~37분 — **3배 변동**
- **C4 (+eDISC)**: 12~33분 — 유사
- **C5 (+5R)**: 8~50분 — **최대 6배 변동**

### 결론

> 라운드 수 증가 → 평균·분산 동시 증가. C5 최장 run 50분/run 상한 고려.
> Qwen 실험은 V100에서 더 느릴 수 있으니 **N=3 시작 → 분산 보고 N=5 증량** 전략이 현실적.

---

## 종합 권고 (Qwen 실험 준비)

| 항목 | 권장 |
|---|---|
| 본 실험 주 지표 | **Judge Overall + per-dim (차원별)** |
| 보조 지표 | autoscore_final (완주성 확인), workload_gini (부하 균형) |
| 최저 유의미한 조건 | **C2** (1R) 또는 **C3** (3R) — Overall·Debate 모두 높음 |
| 회피할 조건 | **C5** (5R) — 비용↑ 품질↓ 분산↑ |
| eDISC 효과 | Gemma에선 **negative** — Qwen에서 재확인 필요 |
| N (runs) | 기본 3, 분산 큰 C3/C4/C5는 N=5로 증량 검토 |

### Qwen 비교 시 체크리스트
- [ ] 동일 PRD/팀원 고정 (`sample_data/`) — 자동 (Gemma와 동일)
- [ ] 동일 조건 매트릭스 (C0~C5)
- [ ] 동일 Judge (Gemini 3.1 Pro) — clamp 수정으로 range 문제 없음
- [ ] 차원별 쌍대 비교 (예: Gemma C3 Structure vs Qwen C3 Structure)
- [ ] 통계 검정: Wilcoxon signed-rank (N=3 pilot 수준), N=5면 Mann-Whitney 적용
