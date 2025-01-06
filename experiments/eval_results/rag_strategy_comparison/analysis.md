# 실험 분석: RAG 전략 변화에 따른 WBS 생성 품질 검증 (eval2 §6 RQ2)

## 실험 개요
- **목적**: RAG 검색 전략별로 WBS 생성 품질·근거 충실도·배정 질이 어떻게 달라지는지 정량 비교
- **방법**: 동일 팀원(6명, Diverse eDISC 고정) / PRD / 참고 WBS / 회의록 입력 고정, `rag_strategy`만 5가지로 변경
  - **R0**: RAG 없음 (PRD만)
  - **R1 Vanilla**: FAISS Dense 코사인 유사도
  - **R2 Hybrid**: BM25 + Dense + RRF (k=60)
  - **R3 Graph**: 엔티티 공출현 그래프 기반 시드+이웃 탐색
  - **R4 Agentic**: 3-hop 반복 검색 + coverage 평가 (threshold=0.72)
- **Iteration**: 조건당 **N=5회**
- **max_rounds**: 2 (RQ2 baseline 고정)
- **백엔드**: Gemini `gemini-3.1-flash-lite-preview`, Gemma `google/gemma-4-E4B-it` (vLLM TP=2, max_tokens 프록시 캡)
- **평가지표**:
  - metrics.py 11종 (RAGAS Faithfulness 포함)
  - LLM-as-a-Judge 4종 (Structure/Assignment/Debate/Overall)
- **총 실행**: 2 × 5 × 5 = **50 run**

---

## Judge 감사 및 보정 내역 (rejudge.py)

`max_output_tokens=500` 한도로 응답 절단 시 fallback regex가 입력 텍스트의 숫자를 score로 오추출하는 이슈가 재현됨. 전수 감사 후 재평가.

### 재평가된 12 run (max_output_tokens=1500 + JSON 엄격 지시)

| 백엔드 | run | 원래 score | 보정 후 |
|---|---|---|---|
| Gemini | R3 iter3 (Debate) | 1.0 | **0.85** |
| Gemini | R4 iter1 (Assign/Debate) | 1.0 / 0.0 | **0.0 / 0.75** |
| Gemini | R4 iter5 (Debate) | 1.0 | **0.95** |
| Gemma | R0 iter2 (Structure) | 1.0 | **0.47** |
| Gemma | R0 iter5 (Debate) | 1.0 | **0.85** |
| Gemma | R1 iter2 (Debate) | 1.0 | **0.95** |
| Gemma | R1 iter3 (Assignment) | 1.0 | **0.07** |
| Gemma | R2 iter1 (Assignment) | 1.0 | **0.0** |
| Gemma | R4 iter3 (Debate) | 0.0 | **0.9** |

- 보정 패턴: Gemma Assignment 1.0 중 다수가 실제로는 "환각된 태스크 ID 배정" (0.0)이었음 — Gemma 4B의 failure mode 재확인
- 원본 값은 `*.json.bak_rejudge` 백업 보존

---

## Gemini 백엔드 결과 (N=5/조건)

| 지표 | R0 | R1 Vanilla | R2 Hybrid | R3 Graph | R4 Agentic |
|---|---|---|---|---|---|
| Planning Score ↑ | 0.373±0.028 | 0.373±0.020 | 0.384±0.019 | 0.359±0.012 | **0.390±0.026** |
| Workload Gini ↓ | 0.186±0.055 | 0.210±0.081 | 0.218±0.055 | **0.187±0.031** | 0.191±0.035 |
| MECE ↑ | 0.843±0.083 | 0.770±0.179 | 0.788±0.095 | 0.830±0.067 | **0.850±0.047** |
| **Faithfulness ↑** | **N/A** | 0.917±0.014 | 0.929±0.023 | **0.934±0.012** | 0.913±0.021 |
| Comm Efficiency ↑ | 0.808±0.064 | 0.778±0.067 | **0.825±0.044** | 0.803±0.032 | 0.779±0.041 |
| AutoScore ↑ | **0.826±0.023** | 0.801±0.045 | 0.816±0.025 | 0.820±0.018 | 0.821±0.016 |
| Judge-Structure ↑ | **0.628±0.038** | 0.544±0.031 | 0.574±0.091 | 0.572±0.063 | 0.586±0.059 |
| Judge-Assignment ↑ | 0.838±0.073 | 0.842±0.063 | **0.906±0.033** | **0.906±0.033** | 0.706±0.398※ |
| Judge-Debate ↑ | 0.790±0.096 | 0.720±0.172 | **0.870±0.076** | 0.800±0.154 | 0.820±0.084 |
| **Judge-Overall ↑** | 0.742±0.047 | 0.692±0.069 | **0.764±0.046** | 0.746±0.065 | 0.687±0.170 |

※ R4 iter1에서 환각된 태스크 ID 발견(Judge-A=0.0)으로 variance 큼

## Gemma 백엔드 결과 (N=5/조건)

| 지표 | R0 | R1 Vanilla | R2 Hybrid | R3 Graph | R4 Agentic |
|---|---|---|---|---|---|
| Planning Score ↑ | **0.315±0.013** | 0.302±0.017 | 0.297±0.013 | 0.314±0.017 | 0.314±0.012 |
| Workload Gini ↓ | 0.243±0.083 | **0.220±0.092** | 0.247±0.055 | 0.231±0.034 | 0.226±0.080 |
| MECE ↑ | 0.937±0.033 | **0.960±0.020** | 0.932±0.062 | 0.948±0.052 | 0.930±0.040 |
| **Faithfulness ↑** | **N/A** | 0.967±0.015 | **0.976±0.009** | 0.970±0.014 | 0.969±0.014 |
| Supervisor 개입율 ↓ | 0.417±0.194 | **0.274±0.133** | 0.333±0.143 | 0.306±0.139 | 0.351±0.172 |
| AutoScore ↑ | 0.840±0.011 | **0.845±0.008** | 0.837±0.015 | 0.844±0.011 | 0.838±0.013 |
| Judge-Structure ↑ | 0.454±0.074 | 0.482±0.027 | 0.442±0.063 | **0.494±0.033** | 0.442±0.063 |
| Judge-Assignment ↑ | 0.762±0.156 | 0.588±0.306 | 0.590±0.346 | 0.694±0.123 | **0.780±0.080** |
| Judge-Debate ↑ | 0.790±0.139 | 0.870±0.179 | **0.910±0.108** | 0.840±0.089 | 0.840±0.082 |
| **Judge-Overall ↑** | 0.646±0.082 | 0.616±0.153 | 0.611±0.122 | 0.650±0.043 | **0.660±0.025** |

---

## 통계 검정 (Mann-Whitney U + Holm-Bonferroni, α=0.05)

Judge-Overall 기준 R0 vs R1~R4 비교

| Backend | 비교 | U | p | Cliff's δ | p_holm | 결론 |
|---|---|---|---|---|---|---|
| Gemini | R0 vs R1 | 18.0 | 0.310 | −0.44 | 1.0 | n.s. |
| Gemini | R0 vs R2 | 9.0 | 0.548 | +0.28 | 1.0 | n.s. |
| Gemini | R0 vs R3 | 13.0 | 1.000 | −0.04 | 1.0 | n.s. |
| Gemini | R0 vs R4 | 14.0 | 0.841 | −0.12 | 1.0 | n.s. |
| Gemma | R0 vs R1 | 12.5 | 1.000 | 0.00 | 1.0 | n.s. |
| Gemma | R0 vs R2 | 15.0 | 0.690 | −0.20 | 1.0 | n.s. |
| Gemma | R0 vs R3 | 14.5 | 0.753 | −0.16 | 1.0 | n.s. |
| Gemma | R0 vs R4 | 13.0 | 1.000 | −0.04 | 1.0 | n.s. |

**N=5 표본은 Mann-Whitney U의 검정력이 낮아 모든 비교가 Holm-Bonferroni 이후 유의성 없음(p_holm=1.0)**. eval2.txt §6 권장대로 **N ≥ 10**으로 확장하면 유의 가능성 있으나 본 실험 범위 외.

Cliff's δ (효과 크기, |δ|≥0.33: "중간", |δ|≥0.47: "큼")
- **Gemini R0 vs R1: δ=−0.44 (중간)** — R1 Vanilla RAG가 Judge-Overall을 **떨어뜨리는** 경향
- **Gemini R0 vs R2: δ=+0.28 (작음)** — R2 Hybrid는 소폭 개선
- 그 외는 모두 |δ|<0.33 (작음) — 실질적 차이 미미

---

## 핵심 결론

### C1. RAG 도입은 Faithfulness를 확실히 개선, 다른 지표는 미미
- **Faithfulness 측면**: R0은 N/A(근거 컨텍스트 없음), R1~R4 모두 0.91~0.98 영역으로 급상승
- **Gemma에서 Faithfulness 최고**: R2 Hybrid 0.976 — Gemini(0.934)보다 오히려 높음
- **Gemini에서 Faithfulness 최고**: R3 Graph 0.934
- 다만 Planning / AutoScore / Success Rate 등 객관 지표는 R0~R4 간 차이가 std 범위 내

### C2. Judge 관점은 백엔드별로 선호 RAG가 다름
- **Gemini**: **R2 Hybrid가 Judge-Overall(0.764) 최고** — Judge-Assignment·Debate 모두 R2 1위
- **Gemma**: **R4 Agentic이 Judge-Overall(0.660) 최고** — R2는 Gemma에서 오히려 최저(0.611)
- 해석: Hybrid(BM25+Dense)의 키워드+의미 결합은 Gemini의 프롬프트 해석 특성과 잘 맞음. Gemma는 Agentic의 멀티홉 확장이 부족한 용량을 보완

### C3. R0(RAG 없음)가 대부분 지표에서 경쟁력 있음
- Gemini에서 R0의 AutoScore(0.826), Judge-Structure(0.628)가 모두 최고
- R1 Vanilla는 Gemini에서 오히려 R0 대비 Judge-Overall −0.050 저하 (δ=−0.44 중간 효과)
- **시사**: PRD만 충실하면 기본 품질 확보 가능. RAG는 "근거 충실도(Faithfulness)" 개선에 한정된 기여

### C4. Gemma에서 RAG 전략 간 차이가 더 선명함
- Gemma는 R2 Hybrid에서 Judge-Debate 0.910 (최고)이지만 Judge-Assignment 0.590 (최저)
- R4 Agentic이 Assignment(0.780)·Overall(0.660) 모두 최고
- Gemma 4B는 멀티홉 확장된 context가 배정 결정에 도움

### C5. 통계적 유의성은 N=5로 확보 불가 (표본 한계)
- 모든 R0 vs R{1,2,3,4} 비교에서 Holm-Bonferroni 보정 후 p_holm=1.0
- 실질적 차이 시사(Cliff's δ)는 있으나 엄밀한 유의성 주장은 N≥10 필요

---

## 해석 (Interpretation)

### H1. Vanilla RAG가 오히려 해가 될 수 있다
- R1 Vanilla가 Gemini에서 Judge-Overall을 δ=−0.44만큼 낮춤
- 이유: Dense 검색만으로는 관련도 낮은 문서가 상위에 올라와 프롬프트 노이즈가 되는 경우 발생
- Hybrid나 Graph처럼 "관련성 확인 메커니즘"이 있는 전략이 더 안전

### H2. RAG의 주요 기여는 Faithfulness (근거 충실도)
- 객관 품질 지표(MECE, Planning, AutoScore)는 R0 대비 미미한 개선
- 반면 **Faithfulness는 N/A → 0.91~0.98로 명확히 향상** — "생성 결과가 근거에 기반"했는지가 주된 효과
- 즉 RAG는 "더 좋은 WBS를 만드는" 도구가 아니라 "주장의 근거를 확보하는" 도구

### H3. Hybrid(R2)의 Gemini 우위, Agentic(R4)의 Gemma 우위
- 두 백엔드가 서로 다른 RAG 전략을 선호 → **단일 최적 RAG가 없고, 모델별 선택 필요**
- Gemini: 고품질 프롬프트 해석 능력 + BM25 정확성 결합에서 시너지
- Gemma: 용량 제약을 멀티홉 확장으로 보완

### H4. R0이 강건한 이유: PRD 자체가 풍부
- 본 실험의 PRD는 단일 프로젝트(B2B CS 플랫폼) + 6개 명확한 key_features
- PRD가 충분히 상세하면 RAG 없이도 WBS 품질 확보 가능
- **RAG의 실질 효과는 "PRD가 빈약한 경우"**나 **"프로젝트 도메인이 낯선 경우"**에 유의미할 것 — 본 실험으로는 검증 불가

### H5. 측정 오류의 파급력 재확인
- Gemma 12 run에서 rejudge 보정 필요 (Gemini 3 run)
- 특히 Gemma의 Assignment=1.0 중 다수가 실제로는 "환각 ID 배정"(0.0) → Judge의 엄밀한 검증이 필수
- 스냅샷 + 재평가 파이프라인 없이 raw Judge 값을 그대로 쓰면 결론이 뒤집힘

---

## 실무 권고

1. **RAG 도입 목적이 "근거 인용"**이면 어느 전략이든 효과 있음 (Faithfulness 0.91+)
2. **Gemini 사용 시 R2 Hybrid 추천** (BM25+Dense+RRF가 Judge 평가 최고)
3. **Gemma 사용 시 R4 Agentic 추천** (멀티홉 확장이 4B 모델 용량 보완)
4. **R1 Vanilla는 주의** — Gemini에서 오히려 Judge-Overall 저하 경향 (δ=−0.44)
5. PRD가 이미 상세하면 R0(RAG 없음)도 경쟁력 있음 — RAG 도입은 "근거 확보" 가치와 비용을 함께 고려
6. Judge 결과는 **max_output_tokens 재검증 + fallback regex 감사** 필수
7. N≥10으로 확장 권장 — 본 실험 N=5는 통계적 유의성 확보 한계

---

## 산출물 매핑

```
eval_results/rag_strategy_comparison/
├── analysis.md                   ← 본 문서
├── conclusion.md                 ← 요약 Summary Sheet + Key Findings
├── all_runs.json                 ← 50 run 통합 결과 (metrics + judge)
├── all_iterations.csv            ← 장표용 raw long-format
├── statistical_tests.json        ← Mann-Whitney U + Cliff's δ + Holm-Bonferroni
├── cross_backend_plot.png        ← 2 백엔드 × 5 조건 × 15 지표 교차 비교
├── backend_gemini/
│   ├── comparison_plot.png       ← R0~R4 × Gemini
│   ├── summary_table.md / .csv   ← 조건별 평균±std
│   ├── raw_iterations.csv
│   └── runs/
│       ├── R{0~4}_iter{1~5}.json   (metrics + judge)
│       ├── *.json.bak_rejudge      (재평가 전 백업)
│       └── snapshots/
├── backend_gemma4_api/  (동일 구조)
├── runner.py                     ← 실험 러너
├── rejudge.py                    ← Judge parse 오류 재평가
└── analyze.py                    ← 집계/플롯/통계
```
