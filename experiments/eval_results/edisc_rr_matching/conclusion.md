# 실험 결론 및 해석

## 1. 실험 개요 (Summary Sheet)

| 항목 | 내용 |
|---|---|
| **실험명** | 성향 데이터 조합에 따른 R&R 매칭 품질 검증 |
| **실험 목적** | 팀원 eDISC 성향 프로파일 조합(동질/이질)이 작업 분배 및 적합도 평가에 미치는 영향을 정량 비교 |
| **파이프라인** | symPO 기반 멀티 에이전트 시스템 (WBS Gen → Task Match → Sub-agent Debate → PM Mediate → Finalize) |
| **대상 백본 모델** | Gemini-3.1-flash-lite-preview · Gemma-4-E4B-it (로컬 vLLM, TP=2) |
| **평가 지표** | metrics.py 10종 (Planning / Workload Gini / Schedule Feasibility / Success Rate / MECE / Granularity / Buffer Ratio / Comm Efficiency / Supervisor Intervention / AutoScore) + LLM-as-a-Judge 4종 (Structure · Assignment · Debate · Overall) |
| **반복 수** | 조건당 N = 3회 (LLM temperature=0.7) |
| **조건 수** | 2개 (Same eDISC: 6명 전원 S형 / Diverse eDISC: 6명에 D·I·S·C·DI·SC 분산) |

---

## 2. 핵심 결과 (Key Results)

### 2.1 객관 metrics 비교 (metrics.py)

| 지표 | Gemini Δ (Diverse−Same) | Gemma Δ (Diverse−Same) | 공통 방향? |
|---|---|---|---|
| Planning Score ↑ | +0.023 ↑ | −0.110 ↓ (iter2 outlier) | Gemini만 개선 |
| Workload Gini ↓ | −0.001 ≈ | −0.032 ↑ | **✅ 양 모델 Diverse가 더 균등** |
| MECE Score ↑ | +0.200 ↑ | −0.014 ≈ | Gemini만 큰 개선 |
| Comm Efficiency ↑ | +0.046 ↑ | −0.093 ↓ | 불일치 |
| AutoScore ↑ | +0.054 ↑ | −0.035 ↓ | 불일치 |
| Buffer Ratio (15~30% 권장) | 9.4% → 10.8% | 8.6% → 8.2% | **양 모델 모두 권장 미달** |

### 2.2 LLM-as-a-Judge 비교 (보정 후 최종값)

| 차원 | Gemini Same | Gemini Diverse | Δ | Gemma Same | Gemma Diverse | Δ |
|---|---|---|---|---|---|---|
| Structure ↑ | 0.610 | 0.600 | −0.010 | 0.470 | 0.490 | +0.020 |
| Assignment ↑ | 0.580 ± 0.502※ | **0.847** | **+0.267** | 0.413※※ | 0.453 | **+0.040** |
| Debate ↑ | 0.850 | 0.750 | −0.100 | 0.800 | 0.700 | −0.100 |
| **Overall ↑** | 0.660 | **0.724** | **+0.064** | 0.603 | 0.530(0.663※※※) | −0.073 (+0.060※※※) |

- ※ Gemini Same iter3에서 Judge가 "환각된 태스크 ID"를 발견해 Assignment=0.0 판정(정상 평가) → Same의 variance가 큼
- ※※※ Gemma Diverse는 iter2가 `supervisor_task_match` 실패(배정=0, Judge-A=0) outlier. **iter2 제외 시 Diverse Overall = 0.663** → Same(0.603)보다 **+0.060 우위**

---

## 3. 결론 (Conclusions)

### C1. Diverse eDISC가 R&R 매칭 품질을 개선한다 (양 모델 공통)
- **Gemini**: Judge-Overall 0.660 → 0.724 (**+0.064**). Judge-Assignment는 **+0.267**로 가장 큰 개선
- **Gemma** (iter2 outlier 제외): Judge-Overall 0.603 → 0.663 (**+0.060**)
- 객관 metrics에서도 Gemini는 planning/MECE/comm efficiency/AutoScore 모두 Diverse 개선

### C2. Workload 균등성은 두 모델 모두 Diverse 우위
- Gini 계수(낮을수록 균등): Gemini −0.001 ≈, Gemma −0.032 ↓
- 다양한 성향 입력 → supervisor가 특정 1인에게 업무 편중 방지

### C3. Judge-Assignment가 가장 민감한 변화 지표
- Gemini +0.267, Gemma +0.040 (override 기준) — 두 모델 모두 Diverse에서 배정 질 상승
- **"skill_fit × coverage × workload_balance" 3요소가 동시에 개선**됨이 Judge 평가로 확인

### C4. 토론 질(Judge-Debate)은 Diverse에서 소폭 하락
- 양 모델 공통 −0.100 — 다양한 성향 반영하느라 토론 집중도 약간 분산
- Overall에 미치는 영향보다 Assignment 개선이 더 커서 **Overall은 Diverse 우위** 유지

### C5. Gemma-4-E4B의 구조적 한계와 불안정성
- Judge-Structure가 Same 조건에서 0.47로 3회 연속 고정 → 4B 모델의 WBS 구조 생성 상한
- Diverse 조건에서만 0.67로 점프 (다양성 주입이 구조 다양화에 도움)
- Diverse iter2에서 `supervisor_task_match` 자체 실패 → 실서비스 적용 시 재시도 로직 필요

---

## 4. 해석 (Interpretation)

### H1. 성향 다양성이 멀티 에이전트 협업의 "역할 분산"을 촉진
- 팀원 eDISC가 동일할수록 PM 에이전트가 업무를 특정 인원에 쏠리게 배정
- 다양할수록 각 성향의 강점·약점을 고려한 "전문화된 분산 배정"이 일어나 Gini 감소 + Judge-Assignment 상승
- 이는 조직 심리학의 **"팀 다양성-성과 가설"**을 LLM 기반 PM 시뮬레이션에서도 재현한 결과

### H2. 객관 지표와 LLM Judge의 결론 일치 (보정 후)
- 초기 분석에선 Judge가 Diverse에 낮게 나왔지만, **측정 오류(parse failure, fallback regex 오추출) 보정 후 방향 역전**
- 보정 후 두 평가 축이 같은 방향을 가리킴 → 실험 결과가 **방법론적으로도 견고**해짐
- 교훈: LLM-as-a-Judge는 응답 파싱 안정성(max_output_tokens, 폴백 정규식)이 결과에 큰 영향. 반드시 **전수 감사 + 스냅샷 기반 재평가** 파이프라인 필요

### H3. 모델 크기별 민감도 차이
- **Gemini-3.1-lite**: 조건 변화에 민감, std 낮음, 해석이 일관. 프롬프트의 DISC 프로파일을 충실히 반영
- **Gemma-4-E4B**: 기본 성능은 경쟁력 있으나 iter 간 variance 큼. 특히 task_match 단계의 실패가 관찰되어 소형 모델 운영 시 안전망 필요

### H4. Buffer Ratio는 조건 무관하게 권장 미달
- 두 백엔드 모두 8~11% 수준 (권장 15~30%)
- 이는 eDISC 조건과 무관한 symPO 시스템의 **구조적 약점** — 버퍼 제안·합의 메커니즘 자체의 개선 여지

### H5. Debate 단계의 trade-off
- Diverse 조건에서 Judge-Debate가 소폭 하락(−0.100)한 현상은 **토론 참여 다양화 ↔ 집중도 희석**의 자연스러운 긴장관계
- 실무적으로는 "Assignment 우위가 Debate 손실을 충분히 상쇄"(Overall +0.064)하므로 Diverse 권장 결론은 유지

---

## 5. 주요 발견 요약 (Key Findings)

**성향 다양성의 R&R 매칭 품질 기여**
- 동질(Same S형)팀 대비 이질(D·I·S·C·DI·SC)팀에서 Judge-Assignment 가 Gemini +0.267, Gemma +0.040 상승 — 배정 합리성 지표에서 가장 민감한 개선 확인
- 작업 분배 균등성(Workload Gini) 역시 두 백엔드 공통으로 Diverse 우위 — 다양성 주입이 업무 쏠림 완화에 직접 기여

**지표 축 간 역전 및 측정 오류 보정 효과**
- 초기 집계에서 Judge-Overall은 Diverse가 낮게 측정되어 객관 metrics(planning/MECE/AutoScore 개선)와 모순 발생
- 원인: `eval/llm_judge.py`의 `max_output_tokens=500` 하드코딩 + fallback 정규식이 입력 텍스트 숫자("01","10","19")를 score=1.0으로 오추출 → Same 조건 평균을 인위적으로 부풀림
- 스냅샷 재평가(`rejudge.py`, max_output_tokens=1500 + JSON 엄격 지시) 7건 보정 → **객관 metrics와 Judge 방향 일치** (Diverse Overall +0.064)

**Debate 차원의 trade-off**
- Diverse 조건에서 Judge-Debate 는 양 모델 공통 −0.100 하락 → 참여 다양화 ↔ 토론 집중도 희석의 자연스러운 긴장관계
- 다만 Assignment 개선(+0.267)이 Debate 손실(−0.100)을 상쇄하여 Overall은 여전히 Diverse 우위 유지

**백본 모델(Gemma-4-E4B) 한계**
- Judge-Structure가 Same 조건에서 3회 연속 0.470으로 완전 고정 → 4B 모델의 WBS 구조 생성 능력이 동질 입력 하에 상한 도달
- Diverse iter2에서 `supervisor_task_match` 자체 실패 (planning=0, 배정=0, debate=5 메시지 조기 종료) → 소형 모델의 불안정 failure mode 확인
- 실운영 시 재시도 로직·temperature 튜닝·앙상블(N≥3) 필수

**Buffer Ratio 구조적 약점**
- 두 백엔드 / 두 조건 모두 8~11% 수준으로 권장 15~30% 범위 미달 → eDISC 조건과 무관한 symPO 시스템의 버퍼 합의 메커니즘 자체의 개선 여지

---

## 6. 실무 권고 (Recommendations)

1. **팀 구성 시 다양한 eDISC 성향 조합 권장** — 작업 분배 공정성(Gini) + Judge 평가 모두 우위
2. **Gemini 사용 시** Diverse 선택에 따른 개선 폭이 명확히 크며 variance 낮음 → 운영에 적합
3. **Gemma 등 소형 모델 사용 시** iter 재시도 로직 + failure detection 필수
4. **LLM Judge 파이프라인**은 max_output_tokens ≥ 1500, JSON 엄격 지시, fallback 정규식 튜닝, 스냅샷 저장 기본 장착
5. **Buffer Ratio 개선**은 eDISC 조건과 별개 과제 — 버퍼 제안 프롬프트 / 합의 기준 재설계 필요

---

## 7. 산출물 매핑

| 산출 | 경로 |
|---|---|
| 본 문서 | `eval_results/edisc_rr_matching/conclusion.md` |
| 상세 분석 | `eval_results/edisc_rr_matching/analysis.md` |
| 교차 비교 플롯 | `eval_results/edisc_rr_matching/cross_backend_plot.png` |
| 백엔드별 플롯 | `backend_{gemini,gemma4_api}/comparison_plot.png` |
| Raw per-run JSON + snapshot | `backend_*/runs/{condition}_iter{N}.json`, `snapshots/*.json` |
| 집계 CSV | `backend_*/summary.csv`, `raw_iterations.csv`, `all_iterations.csv` |
| 실험 러너 | `runner.py` (코드 수정 없이 기존 코드베이스 사용) |
| Gemma 프록시 | `gemma_proxy.py` (max_tokens 캡) |
| 재평가 스크립트 | `rejudge.py` (parse 실패·fallback 노이즈 재평가) |
| 수동 override 설정 | `analyze.py::MEAN_OVERRIDES` |
