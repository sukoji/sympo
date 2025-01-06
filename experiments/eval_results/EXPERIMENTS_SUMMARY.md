# symPO 실험 결과 통합 요약

정리일: 2026-04-27

이 문서는 현재 `eval_results/`에 남아 있는 주요 실험 산출물을 기준으로, 발표/보고서에 사용할 수 있는 결과와 아직 조심해야 할 해석을 구분해 정리한다. 과거 2026-04-16~19 Gemini 파일럿 결과는 참고 기록이며, 현재 결론은 4-backbone, model-swap, context metadata, reasoning, compaction, RAG 비교 실험을 우선한다.

## 1. 현재 코드 기준

주요 실행/평가 코드:

- 실행: `eval/experiment_runner.py`
- 자동 지표: `metrics.py`
- AutoScore canonical 재계산: `eval_results/autoscore_recompute.py`
- LLM Judge: `eval/llm_judge.py`
- 분석/신뢰도/민감도: `eval/analyze_results.py`, `eval/reliability.py`, `eval/sensitivity.py`

현재 AutoScore canonical 버전은 `v2_backfill_safe`이며 top-level weight는 Quality `0.45`, Allocation `0.35`, Orchestration `0.20`이다. Judge overall은 Structure `0.40`, Assignment `0.35`, Debate `0.25`를 사용한다.

## 2. 주요 실험 산출물

| 실험 | 경로 | 상태 | 핵심 용도 |
|---|---|---|---|
| 4-backbone ablation | `comparison_4backbones/COMPARISON_REPORT.md` | 완료 | 단일/배정/토론/eDISC/5R 조건과 백본 비교 |
| Model swap | `model_swap_experiment/wbs_taskmgr_model_comparison_20260426/SUMMARY.md` | 완료 | WBS Gen, Task Manager 백본 교체 영향 |
| Context metadata | `context_metadata_experiment/CONTEXT_METADATA_REPORT.md` | 완료 | 이력서/eDISC 메타데이터 효과 |
| Compaction v4 | `compaction_v4_experiment/COMPACTION_V4_REPORT.md` | 완료 | C_filter vs C_summary 품질/효율 비교 |
| Reasoning mode | `reasoning_mode_experiment/REASONING_REPORT.md` | 완료 | reasoning 강도별 품질 변화 |
| Hetero backbone | `hetero_backbone_experiment/HETERO_REPORT.md` | 완료 | 일부 agent frontier 교체 효과 |
| RAG strategy | `rag_strategy_comparison/conclusion.md` | 완료 | RAG 전략별 faithfulness/품질/비용 비교 |
| eDISC R&R matching | `edisc_rr_matching/conclusion.md` | 보조 | eDISC와 역할 매칭 가능성 탐색 |

## 3. 확실히 주장 가능한 결과

### 3.1 멀티 에이전트 토론 효과

4-backbone ablation에서 C1(+배정) 대비 C2(+1R), C3(+3R)는 대부분의 백본에서 Judge overall을 개선했다.

- Gemma: C1 0.50 -> C2 0.64
- Gemma26: C1 0.58 -> C2 0.70 -> C3 0.73
- Gemini: C1 0.54 -> C2 0.68
- Qwen은 개선 폭이 작아 백본 의존성이 존재한다.

보고서 표현:

> 멀티 에이전트 토론은 단일 생성/배정 결과를 그대로 쓰는 것보다 WBS 구조와 토론 품질을 개선하는 경향을 보였다. 다만 효과 크기는 백본 모델에 따라 달라졌다.

### 3.2 Gemma4-26B baseline의 안정성

4-backbone C3 기준 overall 순위:

1. Gemma26: 0.73
2. Gemini: 0.68
3. Gemma: 0.64
4. Qwen: 0.54

Model swap 실험에서도 Gemma4-26B baseline은 Judge 0.775, Auto 0.811로 Qwen 4B/EXAONE 교체 조건보다 높거나 안정적이었다.

보고서 표현:

> 본 프로젝트 조건에서는 Gemma4-26B baseline이 가장 안정적인 backbone으로 관찰되었고, 작은 역할별 모델 교체가 항상 성능 개선으로 이어지지는 않았다.

### 3.3 Reasoning mode 효과

Reasoning mode 실험은 C3 조건에서 reasoning 강도가 증가할수록 Judge overall이 상승하는 경향을 보였다.

- R_none: 0.73
- R_high: 0.77
- R_max: 0.78

보고서 표현:

> WBS 생성/토론 품질은 단순 모델 크기뿐 아니라 reasoning 설정에도 민감했다.

### 3.4 Compaction 정책

Compaction v4에서 C_summary가 Judge 0.667, C_filter가 0.662로 품질 차이는 표준편차 내였다. 효율성은 C_filter가 우세했다.

- C_filter: 601s/run, summary call 0, 평균 압축 prompt 2572 chars
- C_summary: 804s/run, summary call 25, 평균 압축 prompt 3418 chars

보고서 표현:

> LLM 요약 기반 compaction은 정상 작동했지만, 품질 차이가 미미하고 비용/시간이 증가해 현재 default는 C_filter가 타당하다.

### 3.5 Context metadata 결과

Context metadata ablation 결과:

- M_resume: 0.619
- M_disc: 0.566
- M_both: 0.611

현재 데이터에서는 eDISC 단독 또는 eDISC 추가가 이력서/스킬 기반 배정 대비 명확한 개선을 보이지 않았다.

보고서 표현:

> 현재 실험에서는 성향 정보보다 이력서 기반 기술/경험 정보가 배정 품질에 더 직접적인 영향을 보였다. eDISC는 보조 메타데이터로 해석하는 것이 안전하다.

## 4. 제한적으로만 주장 가능한 결과

### 4.1 RAG 전략

RAG 전략 비교는 근거 충실도 개선 가능성을 보여주지만, 전략별 품질 차이는 표본 수와 분산 때문에 강하게 단정하기 어렵다. 목적별 경향은 다음 정도로 표현한다.

- 근거 충실도: RAG 적용 조건이 유리
- 비용/시간: Hybrid 또는 C_filter류 단순 전략이 유리할 수 있음
- 리스크/버퍼: Agentic RAG가 일부 조건에서 높은 버퍼를 유도

주의:

- 현재 코드의 faithfulness는 NLI 우선, 실패 시 keyword fallback이다.
- 과거 리포트의 keyword-only 결과와 현재 NLI 기반 결과를 한 문장에 섞지 않는다.

### 4.2 eDISC 효과

보고서 배경에는 성향 기반 배정 필요성이 있으나, 현재 직접 실험은 eDISC 효과를 강하게 지지하지 않는다. 따라서 “eDISC가 성능을 향상시켰다”가 아니라 “성향 정보를 반영할 수 있는 구조를 구현했고, 현재 pilot에서는 이력서/스킬 정보가 더 강했다”로 써야 한다.

### 4.3 Tool/MCP

현재 프로젝트에는 두 가지 tool 관련 구조가 있다.

- 외부 MCP 서버: `mcp_server.py`가 vector/RAG/eDISC/snapshot inspection tool을 노출한다.
- 내부 MCP-style tool trace: `orchestration/mcp_tool_layer.py`가 LangGraph phase 호출을 stable tool name으로 감싸고 `mcp_tool_trace`에 기록한다.

다만 WBS 생성 agent가 외부 MCP IPC를 통해 도구를 호출하는 구조는 아직 아니다. 현재는 기존 Python call path를 보존하는 호환 계층과 추적 구조다.

보고서 표현:

> MCP는 현재 외부 분석/검색 도구와 내부 오케스트레이션 경계 추적에 적용되어 있으며, LLM agent의 완전한 외부 tool-calling 분리는 후속 확장 단계다.

## 5. 주장하면 안 되는 문장

| 피해야 할 주장 | 이유 |
|---|---|
| eDISC가 업무배정 품질을 향상시켰다 | context metadata 실험에서 M_both가 M_resume보다 높지 않음 |
| 모든 모델에서 토론 라운드가 단조 개선된다 | Qwen처럼 개선 폭이 작거나 조건별 분산이 있음 |
| Graph RAG 또는 Agentic RAG가 절대적으로 최고다 | 목적 지표별 1위가 다르고 통계 검정력이 부족함 |
| 현재 agent가 외부 MCP tool을 직접 호출한다 | 현재는 internal tool boundary trace와 외부 MCP server 제공 수준 |
| AutoScore 수식은 0.40/0.35/0.25다 | 현재 canonical 구현은 0.45/0.35/0.20 |

## 6. 발표용 핵심 메시지

1. 단일 LLM WBS 생성은 구조/누락/일정 현실성에서 한계가 있고, 멀티 에이전트 토론은 이를 보완하는 경향을 보였다.
2. 프로젝트의 현 baseline은 Gemma4-26B + C3 3R 조건이 가장 안정적이다.
3. 성향 정보(eDISC)는 구조적으로 반영 가능하지만, 현재 데이터에서는 스킬/이력서 정보가 배정 품질에 더 강한 신호였다.
4. Compaction은 LLM summary보다 sliding filter가 품질 대비 효율이 좋아 default로 적합하다.
5. MCP/tool 구조는 외부 분석 도구와 내부 trace boundary까지 구현되어 있으며, 완전한 agent tool-calling은 향후 확장 과제다.

## 7. 다음 실험 우선순위

| 우선순위 | 실험 | 목적 |
|---:|---|---|
| 1 | eDISC factorial 재실험: none/resume/disc/both | 프로젝트 핵심 배경인 성향 기반 배정 효과 직접 검증 |
| 2 | 다중 PRD/다중 팀 구성 N>=10 | 단일 PRD/6인 팀 한계 완화 |
| 3 | Human PM blind evaluation | 자동 지표와 실제 WBS 유용성 간 구성 타당성 검증 |
| 4 | tool trace 기반 평가 | MCP-style tool boundary가 오류 격리/추적성/비용에 주는 영향 측정 |
| 5 | RAG 전략 재실험 | NLI faithfulness 기준으로 RAG 효과 재검증 |
