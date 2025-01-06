# 실험 결론 및 해석 — RAG 전략 비교

## 1. 실험 개요 (Summary Sheet)

| 항목 | 내용 |
|---|---|
| **실험명** | RAG 전략 변화에 따른 WBS 생성 품질 검증 (eval2 §6 RQ2) |
| **실험 목적** | RAG 검색 전략 5종 (없음 / Vanilla / Hybrid / Graph / Agentic)이 WBS 생성 품질·근거 충실도·배정 합리성에 미치는 영향 정량 비교 |
| **파이프라인** | symPO 기반 멀티 에이전트 시스템 (2라운드 토론 + Task Manager 배정 고정) |
| **대상 백본 모델** | Gemini-3.1-flash-lite-preview · Gemma-4-E4B-it (vLLM TP=2) |
| **평가 지표** | metrics.py 11종 (RAGAS Faithfulness 포함) + LLM-as-a-Judge 4종 (Structure · Assignment · Debate · Overall) |
| **반복 수** | 조건당 N = 5회 (eval2 §6 최소 기준) |
| **조건 수** | 5개 (R0 none · R1 Vanilla · R2 Hybrid · R3 Graph · R4 Agentic) |

---

## 2. 핵심 결과 (Key Results)

### 2.1 백엔드 × 조건별 Judge-Overall (보정 후)

| Backend | R0 | R1 Vanilla | R2 Hybrid | R3 Graph | R4 Agentic |
|---|---|---|---|---|---|
| Gemini | 0.742 | 0.692 | **0.764** ★ | 0.746 | 0.687 |
| Gemma | 0.646 | 0.616 | 0.611 | 0.650 | **0.660** ★ |

### 2.2 RAGAS Faithfulness (RAG 근거 충실도)

| Backend | R0 | R1 | R2 | R3 | R4 |
|---|---|---|---|---|---|
| Gemini | N/A | 0.917 | 0.929 | **0.934** ★ | 0.913 |
| Gemma | N/A | 0.967 | **0.976** ★ | 0.970 | 0.969 |

### 2.3 통계 검정 (Mann-Whitney U, N=5, Holm-Bonferroni)

모든 R0 vs R{1,2,3,4} 비교에서 **p_holm = 1.0** (통계 유의성 없음).
단 Cliff's δ로 본 실질 효과:
- **Gemini R0 vs R1**: δ=−0.44 (중간) — Vanilla RAG가 Judge-Overall 저하
- **Gemini R0 vs R2**: δ=+0.28 (작음) — Hybrid 소폭 개선

---

## 3. 결론 (Conclusions)

### C1. RAG의 핵심 효과는 Faithfulness, 품질 지표는 미미
- Faithfulness는 R0 N/A → R1~R4 0.91~0.98로 급상승 — RAG의 본질적 기여
- MECE·Planning·AutoScore 등 구조·배정 품질은 조건 간 std 범위 내

### C2. Gemini는 Hybrid(R2), Gemma는 Agentic(R4) 선호
- 단일 최적 RAG 없음 — **백엔드별로 상반된 선호 확인**
- Hybrid(BM25+Dense+RRF): Gemini에서 Judge-Overall 최고 0.764
- Agentic(멀티홉 coverage 기반): Gemma에서 Judge-Overall 최고 0.660

### C3. Vanilla RAG의 역효과 경고
- Gemini R1 Vanilla가 R0 대비 Judge-Overall −0.050 저하 (δ=−0.44 중간 효과)
- Dense 검색 단독은 관련 낮은 문서가 프롬프트 노이즈로 작용할 위험

### C4. R0(RAG 없음)이 대부분 객관 지표에서 경쟁력
- Gemini R0: AutoScore 0.826 (최고), Judge-Structure 0.628 (최고)
- 상세 PRD가 주어지면 RAG 없이도 기본 품질 확보

### C5. Gemma의 Assignment failure mode 재확인
- rejudge에서 Gemma R1/R2 Assignment=1.0 중 다수가 실제로는 "환각된 태스크 ID 배정" (0.0)
- 4B 모델이 존재하지 않는 멤버 ID를 생성하는 실패 패턴 존재

### C6. N=5 표본은 통계 유의성 확보 불가
- Mann-Whitney U + Holm-Bonferroni 모두 p_holm=1.0
- eval2 §6 권장대로 N≥10 확장이 유의성 확보 전제조건

---

## 4. 해석 (Interpretation)

### H1. RAG는 "근거 확보" 도구, "품질 향상" 도구가 아니다
- Faithfulness 명확 개선 vs 다른 지표 영향 미미
- PRD 품질이 충분하면 RAG의 추가 가치는 제한적

### H2. 백엔드와 RAG 알고리즘의 궁합이 중요
- Gemini(강한 언어 이해)는 정확한 키워드+의미 결합(R2 Hybrid)에 시너지
- Gemma(제한된 용량)는 부족한 context를 멀티홉 확장(R4 Agentic)으로 보완
- → **모델별 RAG 맞춤 설정 필수**

### H3. Vanilla RAG 사용 시 관련성 검증 메커니즘 필요
- R1이 Gemini에서 δ=−0.44로 역효과
- Hybrid의 RRF 융합이나 Agentic의 coverage 평가 같은 **2차 필터**가 안전장치 역할

### H4. LLM Judge의 Fallback 노이즈는 재현되는 시스템적 문제
- `max_output_tokens=500` 한도 + 정규식 fallback이 score=1.0 오추출 유발
- **스냅샷 저장 + 재평가 파이프라인이 필수 인프라**
- 이번 실험에서도 12 run 보정 — 미보정 시 결론이 역전될 수 있음

### H5. Buffer Ratio 권장 미달은 RAG와 무관한 구조적 문제
- 모든 조건(R0~R4, 양 백엔드)에서 Buffer Ratio 8~13% — 권장 15~30% 미달
- RAG 도입으로 개선되지 않음 → 버퍼 합의 메커니즘 자체의 개선 필요

---

## 5. 주요 발견 요약 (Key Findings)

**RAG 도입의 실질적 기여**
- Faithfulness(근거 충실도)는 RAG 도입 시 일관되게 0.91~0.98로 확보되어 가장 뚜렷한 효과
- AutoScore·Planning·MECE 등 객관 품질 지표는 R0~R4 간 std 범위 내로 차이 미미

**백엔드별 최적 RAG 전략 역전**
- Gemini: R2 Hybrid (Judge-Overall 0.764) 우위, R1 Vanilla는 역효과 (δ=−0.44)
- Gemma: R4 Agentic (Judge-Overall 0.660) 우위, R2는 오히려 최저 (0.611)
- 단일 최적 RAG가 없으며 모델별 설정 필요

**Vanilla RAG의 역설적 저하**
- Gemini에서 R1(Dense 단독)이 R0 대비 Judge-Overall 저하 — 관련성 낮은 문서가 프롬프트 노이즈
- Hybrid(RRF)나 Agentic(coverage)처럼 2차 필터가 있는 전략이 안전

**Gemma의 Assignment 환각 failure mode**
- rejudge에서 Gemma의 Assignment=1.0 중 실제로 0.0인 케이스 다수 (존재하지 않는 팀원 ID 배정)
- 4B 모델의 멤버 식별자 환각 경향 → 실서비스 적용 시 배정 ID 검증 로직 필수

**통계적 유의성 확보 한계 (N=5)**
- Mann-Whitney U + Holm-Bonferroni 모두 p_holm=1.0
- eval2 §6 권장 N≥10 확장 필요 — 본 실험은 Cliff's δ 효과 크기로 경향성만 확인

---

## 6. 실무 권고 (Recommendations)

1. **RAG 도입 목적을 명확히**: "근거 인용(Faithfulness)" 목적이면 어느 전략이든 효과 있음 (0.91+)
2. **Gemini + R2 Hybrid** (BM25+Dense+RRF) 조합 추천 — Judge-Overall 최고
3. **Gemma + R4 Agentic** (멀티홉+coverage) 조합 추천 — 용량 한계 보완
4. **R1 Vanilla는 주의** — Dense 단독은 관련성 낮은 문서가 섞이는 위험 존재
5. Gemma 운영 시 **배정 태스크의 member_id 존재 검증** 로직 추가 (환각 방지)
6. **LLM Judge 파이프라인**은 max_output_tokens≥1500 + 스냅샷 + rejudge 필수
7. **후속 연구**: N≥10 확장, PRD 빈약 조건에서의 RAG 효과 검증, 다중 도메인 일반화 검증

---

## 7. 산출물 매핑

| 산출 | 경로 |
|---|---|
| 본 문서 | `eval_results/rag_strategy_comparison/conclusion.md` |
| 상세 분석 | `eval_results/rag_strategy_comparison/analysis.md` |
| 교차 비교 플롯 | `cross_backend_plot.png` |
| 백엔드별 플롯 | `backend_{gemini,gemma4_api}/comparison_plot.png` |
| Raw per-run JSON + snapshot | `backend_*/runs/{R0~R4}_iter{1~5}.json`, `snapshots/*.json` |
| 집계 CSV | `backend_*/summary.csv`, `raw_iterations.csv`, `all_iterations.csv` |
| 통계 검정 | `statistical_tests.json` (Mann-Whitney U, Cliff's δ, Holm-Bonferroni) |
| 실험 러너 | `runner.py` (기존 코드 미수정) |
| 재평가 스크립트 | `rejudge.py` (parse 오류 12건 보정) |
| 분석 생성기 | `analyze.py` (플롯/표/CSV/통계) |
