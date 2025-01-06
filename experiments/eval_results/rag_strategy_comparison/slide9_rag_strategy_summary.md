# 실험 결과9 - 다중 RAG 검색 전략별 성능 비교

## 좌측 실험 정보 표

| 항목 | 내용 |
|---|---|
| 실험명 | 다중 RAG 검색 전략별 성능 비교 |
| 실험 목적 | RAG 전략별 WBS 생성 품질·근거 충실도 변화 검증 |
| 파이프라인 | PRD/회의록/참고 WBS → RAG 검색 → WBS Generator → Task Manager → 2R Multi-Agent Debate |
| 대상 백본 모델 | Gemini 3.1 Flash Lite, Gemma-4-E4B-it |
| 평가 지표 | Judge Overall, AutoScore, Faithfulness, Planning Score, MECE, 실행 시간 |
| 반복 수 | 조건당 5회, 총 50 runs |
| 조건 수 | 5개: None, Vanilla, Hybrid, Graph, Agentic |

## 본문 요약 박스

- 근거성 ↑ → RAG의 핵심 효과는 Faithfulness 확보
  RAG 적용 시 Faithfulness가 Gemini 0.913~0.934, Gemma 0.967~0.976 수준으로 형성됨. R0(None)는 근거 문서가 없어 Faithfulness N/A이며, RAG의 실질 기여는 WBS 품질 상승보다 생성 결과의 근거성 확보에 가까움.
- 품질 개선 제한 → RAG가 항상 더 좋은 WBS를 만들지는 않음
  Judge Overall은 Gemini에서 Hybrid가 0.764로 최고였지만 R0도 0.742로 경쟁력 있음. AutoScore 역시 조건 간 차이가 작아, 상세 PRD가 이미 충분하면 RAG 추가가 품질 점수를 일관되게 끌어올리지는 않음.
- 모델별 최적 전략 상이 → 단일 최적 RAG 전략 없음
  Gemini는 Hybrid(BM25+Dense+RRF)가 Judge Overall 0.764로 최고, Gemma는 Agentic(멀티홉+coverage)이 0.660으로 최고. 백본 모델의 해석 능력과 컨텍스트 활용 방식에 따라 적합한 검색 전략이 달라짐.
- Vanilla RAG 주의 → 단순 Dense 검색은 프롬프트 노이즈 가능
  Gemini에서 Vanilla는 R0 대비 Judge Overall이 0.742→0.692로 하락. 관련성 검증 없이 유사도 기반 문서를 주입하면 오히려 WBS 구조 판단을 흐릴 수 있어 Hybrid/Agentic처럼 재정렬·coverage 검증이 있는 전략이 더 안전함.

## 하단 결론 문구

"RAG는 품질 부스터보다 근거 확보 장치 — Gemini는 Hybrid, Gemma는 Agentic 적용"
