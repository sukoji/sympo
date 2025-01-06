# symPO 실험 결과 타당성 검증 및 서술 가이드

정리일: 2026-04-27

이 문서는 현재 코드와 최신 실험 결과를 기준으로, 보고서/발표에서 가능한 주장과 피해야 할 주장을 정리한다. 과거 2026-04-16~19 Gemini 파일럿에서 발견된 SR coverage inflation, 표본 수 부족, 단일 PRD 한계는 여전히 유효하지만, 현재 코드는 embedding SR, NLI faithfulness, G-Eval judge, AutoScore v2를 포함하도록 갱신되었다.

## 1. 현재 평가 구현 기준

| 항목 | 현재 코드 기준 |
|---|---|
| Success Rate | sentence-transformers 임베딩 기반. 실패 시 keyword fallback |
| Faithfulness | NLI cross-encoder 기반. 실패 시 `keyword_fallback`으로 명시 |
| AutoScore | `v2_backfill_safe`, top-level `0.45/0.35/0.20` |
| LLM Judge | Structure/Assignment/Debate, scalar 또는 G-Eval |
| Cross Judge | `--cross-judge`로 2차 judge 저장 가능 |
| Tool trace | `mcp_tool_trace`로 내부 MCP-style tool boundary 기록 |

## 2. 핵심 타당성 이슈

### 2.1 표본 수 부족

대부분의 최신 실험은 N=3 수준의 pilot이다. 방향성 판단에는 유용하지만, 통계적으로 강한 결론에는 부족하다.

보고서 표현:

> 본 실험은 조건별 N=3 수준의 탐색적 pilot으로, 효과 방향과 시스템 병목을 확인하는 데 목적이 있다. 통계적 일반화는 다중 PRD와 N>=10 반복 실험이 필요하다.

### 2.2 단일 PRD/단일 팀 구성

대부분의 실험은 P마켓/고객서비스 계열 샘플 PRD와 6인 팀을 기준으로 한다. 다른 도메인과 팀 규모에서 같은 결과가 나온다고 단정할 수 없다.

보고서 표현:

> 결과는 단일 PRD와 고정 6인 팀 조건에서의 내부 비교이며, 도메인 일반화는 후속 실험으로 검증해야 한다.

### 2.3 eDISC 효과의 직접 근거 부족

Context metadata ablation 결과는 eDISC 개선을 강하게 지지하지 않는다.

| 조건 | Overall |
|---|---:|
| M_resume | 0.619 |
| M_disc | 0.566 |
| M_both | 0.611 |

보고서 표현:

> 현재 데이터에서는 성향 정보만으로는 업무배정 품질을 충분히 설명하기 어렵고, 이력서 기반 스킬/경험 정보가 더 직접적인 신호로 나타났다. eDISC는 업무배정의 보조 맥락으로 해석한다.

피해야 할 표현:

> eDISC 성향 반영으로 업무배정 품질이 향상되었다.

### 2.4 LLM Judge 편향

LLM Judge는 모델 선호, prompt 민감도, 긴 WBS truncation 영향을 받을 수 있다. `--cross-judge`와 `eval/reliability.py`가 구현되어 있으나, 모든 최신 실험이 cross-judge로 수행된 것은 아니다.

보고서 표현:

> Judge 결과는 자동지표와 함께 해석하며, cross-judge가 없는 실험은 단일 judge pilot으로 표기한다.

### 2.5 Faithfulness 용어

현재 코드는 NLI 모델 기반 faithfulness를 우선 사용하지만, 환경에 따라 `keyword_fallback`으로 내려갈 수 있다. 따라서 결과 표에는 반드시 `faithfulness_method`를 함께 보고한다.

보고서 표현:

> Faithfulness는 NLI 기반으로 계산했으며, 모델 로드 실패 시 keyword fallback 결과는 별도 표시하였다.

## 3. 주장 가능한 것

| 주장 | 근거 | 서술 강도 |
|---|---|---|
| 멀티 에이전트 토론은 단일/배정-only 조건보다 WBS 품질을 개선하는 경향이 있다 | 4-backbone C1->C2/C3 개선 | 가능 |
| Gemma4-26B baseline은 현재 조건에서 가장 안정적인 backbone이다 | 4-backbone, model-swap 결과 | 가능 |
| Reasoning 강도는 WBS/Judge 품질에 영향을 준다 | reasoning mode R_none<R_high<R_max | 가능 |
| C_filter compaction은 C_summary와 품질이 유사하면서 효율이 좋다 | compaction v4 | 가능 |
| eDISC는 현재 실험에서 보조 신호이며, 스킬/이력서 정보가 더 강하다 | context metadata ablation | 가능 |
| MCP는 현재 외부 분석 tool과 내부 tool boundary trace 수준으로 구현되어 있다 | `mcp_server.py`, `orchestration/mcp_tool_layer.py` | 가능 |

## 4. 주장하면 안 되는 것

| 주장 | 왜 안 되는가 |
|---|---|
| eDISC가 배정 품질을 향상시켰다 | M_both가 M_resume보다 높지 않음 |
| 모든 모델에서 토론 라운드가 단조 개선된다 | 백본별 차이와 분산 존재 |
| 특정 RAG 전략이 절대적으로 최고다 | 목적 지표별 1위가 다름 |
| 현재 agent가 외부 MCP tool을 직접 호출한다 | 현재는 Python call path 유지 + trace/catalog 구조 |
| AutoScore 수식은 0.40/0.35/0.25다 | 현재 canonical은 0.45/0.35/0.20 |
| N=3 결과로 통계적 유의성을 일반화할 수 있다 | 검정력 부족 |

## 5. 보고서용 서술 구조

### RQ1. 단일 LLM 대비 멀티 에이전트 토론

논리 흐름:

1. 단일 LLM은 WBS 구조 생성은 가능하지만, 배정/리스크/일정 조정 루프가 없다.
2. Task Manager 배정은 담당자 연결을 추가하지만, 일정 충돌과 workload imbalance를 스스로 해소하지 못한다.
3. 멀티 에이전트 토론은 역할별 검토를 통해 리스크, 버퍼, 재배정 근거를 만든다.
4. 실험에서는 C1 대비 C2/C3가 다수 백본에서 개선되는 경향을 보였다.
5. 단, 백본별 효과 차이가 있어 모델 안정성이 중요하다.

### RQ2. 팀원 메타데이터 기반 배정

논리 흐름:

1. 팀원 메타데이터는 스킬/경험과 행동 성향으로 나뉜다.
2. 현재 실험에서는 스킬/이력서 정보가 assignment에 더 강하게 작용했다.
3. eDISC는 단독 성능 개선 근거가 약하지만, 커뮤니케이션 방식/리스크 협업 맥락으로 확장 가능하다.
4. 따라서 eDISC는 “성능 향상 입증 완료”가 아니라 “반영 가능한 보조 맥락”으로 표현한다.

### RQ3. 오케스트레이션 효율

논리 흐름:

1. 멀티 에이전트 토론은 품질을 높일 수 있지만 토큰/시간 비용을 증가시킨다.
2. Compaction v4는 summary 방식과 filter 방식을 비교했다.
3. C_summary는 정상 작동했지만 품질 차이가 미미하고 비용이 증가했다.
4. 현재 default는 C_filter가 타당하다.

### RQ4. MCP/tool 구조

논리 흐름:

1. 현재 시스템은 LangGraph phase와 중앙 WBSState 기반으로 동작한다.
2. 외부 MCP 서버는 vector/RAG/eDISC/snapshot inspection tool을 제공한다.
3. 내부 오케스트레이션은 MCP-style tool boundary를 통해 추적 가능하다.
4. 단, LLM agent가 외부 MCP IPC로 tool을 직접 호출하는 완전한 구조는 후속 과제다.

## 6. Threats to Validity

### Internal Validity

- LLM 출력 비결정성이 조건 효과와 섞일 수 있다.
- Judge prompt와 모델 선택에 따른 평가 편향이 존재한다.
- 일부 실험은 과거 snapshot 재활용 또는 monkey-patch 조건을 포함한다.

### External Validity

- 단일 PRD/단일 팀/고정 샘플에 기반한다.
- 프로젝트 도메인과 팀 규모가 바뀌면 배정/토론 효과가 달라질 수 있다.
- 로컬/원격 모델 서버 상태, context limit, quantization 설정이 결과에 영향을 줄 수 있다.

### Construct Validity

- AutoScore는 실무 WBS 품질의 근사 지표다.
- Planning Score는 스킬 매칭을 수치화하지만 실제 PM 판단을 완전히 대체하지 못한다.
- eDISC는 행동 성향 자료이므로 기술 적합도와 동일한 축으로 해석하면 안 된다.

## 7. 보강 실험 우선순위

| 우선순위 | 실험 | 이유 |
|---:|---|---|
| 1 | none/resume/disc/both factorial, N>=10 | 성향 기반 배정 핵심 주장 검증 |
| 2 | Human PM blind evaluation | 자동지표와 실무 유용성 연결 |
| 3 | 다중 PRD/다중 팀 규모 | 외적 타당성 보강 |
| 4 | Cross-judge subset 확대 | Judge 편향 완화 |
| 5 | tool trace 평가 | MCP-style tool boundary의 추적성/효율성 입증 |
