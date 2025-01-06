# RAG Strategy Comparison 결과 요약

기준 데이터: `eval_results/rag_strategy_comparison/all_runs.json`  
분석 기준일: 2026-04-22  
표본 수: 백엔드 2개 × 전략 5개 × 반복 5회 = 총 50 run

## Figure

- 핵심 지표 요약: [key_metrics_summary_20260422.png](/home/piai/ai_course/agent_test/eval_results/rag_strategy_comparison/key_metrics_summary_20260422.png)
- R0 대비 변화량 요약: [delta_vs_r0_summary_20260422.png](/home/piai/ai_course/agent_test/eval_results/rag_strategy_comparison/delta_vs_r0_summary_20260422.png)
- 전체 비교 플롯: [cross_backend_plot.png](/home/piai/ai_course/agent_test/eval_results/rag_strategy_comparison/cross_backend_plot.png)

## 먼저 봐야 할 결론

이번 결과에서 통계적으로 유의하다고 말할 수 있는 차이는 없다. `statistical_tests.json` 기준으로 R0 대비 모든 비교에서 Holm-Bonferroni 보정 후 `p_holm = 1.0`이다. 따라서 아래 내용은 유의성 확정이 아니라 경향성 해석이다.

그럼에도 실무적으로 읽을 만한 패턴은 분명하다.

- RAG를 켜면 `Faithfulness`는 확실히 올라간다.
- 하지만 `AutoScore`, `Judge Overall`, `Planning Score` 같은 전체 품질 지표는 크게 좋아지지 않는다.
- Gemini에서는 `R2 Hybrid`가 가장 안정적으로 좋다.
- Gemma4 API에서는 RAG가 전반 품질을 올렸다고 보기 어렵고, 오히려 `R0`가 `Judge Overall` 최고였다.
- 두 백엔드 모두 `R2 Hybrid`가 `Communication Efficiency`는 가장 높았다.

## 핵심 수치

### Gemini

| 지표 | 최고 전략 | 평균 |
|---|---:|---:|
| Judge Overall | R2 | 0.7717 |
| AutoScore | R0 | 0.8259 |
| Faithfulness | R3 | 0.9336 |
| Planning Score | R4 | 0.3899 |
| Communication Efficiency | R2 | 0.8246 |
| Workload Gini | R0 | 0.1860 |

### Gemma4 API

| 지표 | 최고 전략 | 평균 |
|---|---:|---:|
| Judge Overall | R0 | 0.6957 |
| AutoScore | R1 | 0.8451 |
| Faithfulness | R2 | 0.9757 |
| Planning Score | R0 | 0.3153 |
| Communication Efficiency | R2 | 0.8461 |
| Workload Gini | R1 | 0.2200 |

## 유의미한 해석

### 1. RAG의 가장 확실한 효과는 Faithfulness다

`R0`는 RAG 문맥이 없어서 Faithfulness가 N/A이고, RAG를 켠 `R1~R4`는 모두 높은 값을 보였다.

- Gemini: `R1 0.9172`, `R2 0.9291`, `R3 0.9336`, `R4 0.9134`
- Gemma4 API: `R1 0.9666`, `R2 0.9757`, `R3 0.9697`, `R4 0.9692`

즉, RAG는 “근거 충실도”에는 확실히 기여했다. 반대로 말하면, 이 실험에서 RAG의 1차 효과는 “더 좋은 WBS 생성”보다 “근거가 있는 WBS 생성”에 가깝다.

### 2. Gemini에서는 Hybrid가 가장 실전적이다

Gemini 기준 `R2 Hybrid`는 다음에서 강했다.

- `Judge Overall` 최고: `0.7717`
- `Communication Efficiency` 최고: `0.8246`
- `Faithfulness`도 높음: `0.9291`

반면 `AutoScore`는 `R0`가 더 높았다. 즉 Gemini에서는 RAG가 전체 품질을 강하게 끌어올린다기보다, `R2 Hybrid`가 성능 저하 없이 근거성과 토론 효율을 가장 잘 지킨 전략으로 보는 편이 맞다.

### 3. Gemini에서 Vanilla는 추천하기 어렵다

Gemini에서 `R1 Vanilla`는 `Judge Overall 0.6998`로 낮고, `R0 0.7420`보다도 떨어졌다. `statistical_tests.json`에서도 `R0 vs R1`의 Cliff's delta가 `-0.44`로, 표본은 작지만 체감 가능한 역효과 방향이다.

해석은 단순하다.

- Dense 검색 단독으로 뽑힌 문맥이 노이즈가 될 수 있다.
- Hybrid처럼 lexical + dense를 섞거나,
- Graph/Agentic처럼 추가 필터링이 있는 쪽이 더 안전하다.

### 4. Gemma4 API에서는 “RAG가 전체 품질을 올린다”는 결론이 안 나온다

Gemma4 API에서 `Judge Overall` 최고는 `R0 0.6957`이었다.

- `R1 0.6837`
- `R2 0.6808`
- `R3 0.6505`
- `R4 0.6148`

Faithfulness는 RAG에서 확실히 좋아졌지만, 최종 judge 관점 품질은 같이 올라가지 않았다. 따라서 Gemma4 API에서는 RAG를 “품질 향상 장치”로 보기보다 “근거 보강 장치”로 보는 해석이 더 맞다.

### 5. 두 백엔드 공통으로 Hybrid는 토론 효율이 좋다

`Communication Efficiency` 최고는 두 백엔드 모두 `R2 Hybrid`였다.

- Gemini: `0.8246`
- Gemma4 API: `0.8461`

즉 Hybrid는 최소한 “찾아온 문맥이 토론 흐름을 망치지 않고, 필요한 대화만 하게 만든다”는 관점에서는 가장 안정적이다.

### 6. 일부 지표는 실험 구분력이 거의 없었다

- `Schedule Feasibility`: 모든 조건에서 `1.0`
- `Buffer Ratio`: 대체로 `8~13%` 수준

따라서 이번 실험에서는 이 지표들로 RAG 전략 우열을 논하기 어렵다. 특히 Schedule Feasibility는 조건 변별력이 전혀 없었다.

## 추천 해석

실제로 발표나 문서에 넣을 문장은 아래 정도가 안전하다.

- “RAG 전략 변경은 근거 충실도에는 분명한 영향을 줬지만, 전체 WBS 품질 지표 개선은 제한적이었다.”
- “Gemini에서는 Hybrid가 가장 균형이 좋았고, Vanilla는 오히려 성능 저하 경향이 있었다.”
- “Gemma4 API에서는 RAG 도입이 Judge Overall 개선으로 이어지지 않았다.”
- “따라서 이 실험에서는 RAG를 품질 향상보다는 근거 보강 메커니즘으로 해석하는 것이 타당하다.”

## 주의

기존 [analysis.md](/home/piai/ai_course/agent_test/eval_results/rag_strategy_comparison/analysis.md)와 [conclusion.md](/home/piai/ai_course/agent_test/eval_results/rag_strategy_comparison/conclusion.md)는 일부 서술이 현재 `all_runs.json` 집계값과 맞지 않는다. 이번 요약은 기존 텍스트가 아니라 raw 결과를 다시 읽어 정리한 것이다.
