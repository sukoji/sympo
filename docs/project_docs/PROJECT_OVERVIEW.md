# symPO 프로젝트 개요

이 문서는 2026-04-30 현재 코드베이스 기준으로 디렉토리 구조, 주요 상태, 실행 흐름을 정리한 개발자용 레퍼런스입니다.

## 1. 디렉토리 구조

```text
agent_test/
├── main.py
├── api.py
├── mcp_server.py
├── metrics.py
├── README.md
├── SUBMISSION_STRUCTURE.md
├── PROJECT_OVERVIEW.md
├── MCP_GUIDE.md
│
├── agents/
│   ├── state.py
│   ├── llm_config.py
│   ├── wbs_gen_agent.py
│   ├── supervisor_agent.py
│   ├── sub_agents.py
│   └── harness.py
│
├── orchestration/
│   ├── debate_loop.py
│   ├── graph_builder.py
│   └── mcp_tool_layer.py
│
├── data_pipeline/
│   ├── prd_parser.py
│   ├── member_parser.py
│   ├── disc_parser.py
│   ├── vector_store.py
│   └── rag_strategies.py
│
├── persona_engine/
│   ├── persona_builder.py
│   └── persona_templates.py
│
├── schemas/
│   ├── prd_schema.py
│   ├── member_schema.py
│   └── wbs_schema.py
│
├── output/
│   ├── wbs_generator.py
│   └── report_writer.py
│
├── eval/
│   ├── experiment_runner.py
│   ├── analyze_results.py
│   ├── llm_judge.py
│   ├── merge_results.py
│   ├── reliability.py
│   ├── sensitivity.py
│   ├── structural_checker.py
│   ├── constraint_checker.py
│   ├── baseline_runner.py
│   ├── benchmark_generator.py
│   └── run_evaluation.py
│
├── frontend/
│   └── index.html
├── generated/
│   └── debug/
├── eval_results/
│   ├── README.md
│   ├── references/
│   ├── docs/
│   └── report_assets/
├── sample_data/
├── docs/archive/
└── ref/
```

## 2. 실제 엔트리포인트

### `main.py`

- Streamlit 기반 메인 UI
- PRD/이력서/PDF 업로드, eDISC 캐시 로딩, 토론 UI 렌더링

### `api.py`

- FastAPI 앱
- `execute_sympo_flow()` 결과를 SSE로 스트리밍
- 이력서 파싱, 샘플 데이터, eDISC 상태 조회 제공

### `mcp_server.py`

- FastMCP 기반 서버
- 벡터 검색, RAG 비교, eDISC/실험 결과 조회용 툴 제공
- 오케스트레이션 tool catalog와 phase plan을 외부 MCP client에 노출

### `eval/experiment_runner.py`

- 조건별 반복 실험 실행기
- `C*`, `R*`, `A*`, `S2*` 조건군을 `CONDITIONS` dict로 관리

## 3. 런타임 데이터 플로우

```text
PRD / 팀원 / 회의록 / eDISC
  ->
Parser + PersonaBuilder
  ->
create_initial_state()
  ->
execute_sympo_flow()
  ->
phase_wbs_generation / wbs_gen_node
  ->
phase_task_match / supervisor_task_match
  ->
phase_l2_debate / L2 단위 후보 검토
  ->
phase_free_discussion + optional phase_critic_review
  ->
phase_supervisor_mediate
  ->
route_after_mediate()
  -> 재설계: phase_wbs_generation
  -> 다음 라운드: phase_task_match
  -> 종료: phase_finalize / supervisor_finalize
  ->
WBS/로그 저장 + 메트릭 계산 + optional Judge
```

SSE 스트림은 WBS/토론 로그뿐 아니라 `mcp_tool_trace`도 전달합니다. 프론트 `frontend/index.html`의 토론 로그 탭은 이 trace를 **Agent Tool Calls** 패널로 표시해, 에이전트가 어떤 tool boundary를 사용했는지 실시간으로 보여줍니다.

## 4. 오케스트레이션 구조

### `orchestration/debate_loop.py`

핵심 함수:

- `run_sequential_debate(state, max_rounds)`
- `execute_sympo_flow(state, max_rounds)`

주요 동작:

- Phase 1: `phase_wbs_generation()` → `wbs_gen_node`
- Phase 2: `phase_task_match()` → `supervisor_task_match`
- Phase 3: `phase_l2_debate()` → L2 중요도 순 후보 에이전트 검토
- Phase 4: `phase_free_discussion()` → 후보 에이전트 상호 검토
- Phase 5: `phase_critic_review()` → `critic_enabled=True`일 때만 교차 심사
- Phase 6: `phase_supervisor_mediate()` → 중재, 버퍼/재배정 반영, 라운드 증가
- Phase 7: `phase_finalize()` → 최종 확정
- 각 L2마다 2연속 `PASS_TOKEN`이면 조기 종료 가능
- 자유 토론(`free_discussion_agent`) 최대 턴 수는 `state["max_free_turns"]`
- 필요 시 `supervisor_check_and_intervene()`가 중간 개입
- `wbs_revision_needed`, `consensus_reached`, `current_round/max_rounds`에 따라 다음 phase를 결정
- 각 agent/phase 호출은 `orchestration.mcp_tool_layer.call_state_tool()`로 감싸져 `mcp_tool_trace`에 tool name, label, elapsed time, input/output key를 기록

### `orchestration/graph_builder.py`

- LangGraph가 있으면 phase 단위 `StateGraph`를 컴파일
- 그래프 구조: `START → wbs_generation → task_match → l2_debate → free_discussion → critic_review → supervisor_mediate → finalize → END`
- `wbs_generation` 이후에는 `max_rounds <= 0`이면 바로 `finalize`, 아니면 `task_match`로 이동
- `supervisor_mediate` 이후에는 `route_after_mediate()`가 재설계(`wbs_generation`), 다음 라운드(`task_match`), 종료(`finalize`)를 조건부 선택
- 각 LangGraph node는 `debate_loop.py`의 phase generator를 실행하고 `get_stream_writer()`로 중간 state를 custom stream에 전달
- LangGraph reducer 중복을 피하려고 node 반환값의 `debate_log`는 해당 phase에서 새로 추가된 로그만 반환
- `mcp_tool_trace`도 `operator.add` reducer 대상이므로 node 반환값에는 해당 phase에서 새로 추가된 trace만 반환
- LangGraph가 없으면 `execute_sympo_flow()`가 같은 phase 함수들을 사용하는 `run_sequential_debate()` 경로로 폴백

### `orchestration/mcp_tool_layer.py`

- 내부 실행 경계를 실제 MCP tool 이름과 맞추는 catalog 정의
- `wbs-server.generate_draft`, `assignment-server.match_tasks`, `debate-server.*`, `supervisor-server.*` tool spec 제공
- 기존 Python 직접 호출 경로를 유지하면서 호출 metadata를 `mcp_tool_trace`에 기록
- `mcp_server.py`의 `orchestration_tool_catalog()`가 같은 catalog를 외부 MCP client에 노출

## 5. 상태 모델

`agents/state.py`의 `WBSState`는 TypedDict 기반 중앙 상태입니다.

주요 필드:

- 입력: `prd`, `team_members`, `agent_personas`
- 현재 산출물: `current_wbs_draft`, `final_wbs`, `generation_summary`
- 로그: `debate_log`, `mcp_tool_trace`
- RAG: `rag_reference_wbs`, `rag_meeting_logs`, `disc_profiles`
- 라운드 제어: `current_round`, `min_rounds`, `max_rounds`, `consensus_reached`
- 호출/배정 제어: `called_agents`, `calling_context`, `member_role_map`, `locked_assignments`, `l2_agent_mapping`
- 실험 토글: `harness_enabled`, `max_free_turns`, `veto_enabled`, `critic_enabled`, `persona_strictness`, `prd_variant`, `model_class`, `prompting_strategy`
- 내부 플래그: `_free_turn_count`, `_anyone_spoke_in_free`, `_current_l2_task_id`, `_current_agent_acting`, `_l2_debate_cutoff`

`create_initial_state()`는 팀원 role을 `member_role_map`에 분리 저장하고, state에 넣는 `team_members`에서는 role을 제거합니다.

## 6. 에이전트 역할

### `agents/wbs_gen_agent.py`

- 3단계 WBS 초안 생성
- `repair_truncated_json()`로 잘린 JSON 복구
- 누락된 L3 보강 로직 포함

### `agents/supervisor_agent.py`

- `supervisor_task_match()`: L3 담당자 배정, `calling_context`, `l2_agent_mapping` 생성
- `supervisor_check_and_intervene()`: 경량 중간 개입
- `supervisor_mediate()`: 토론 후 조정
- `supervisor_finalize()`: 최종 요약과 `final_wbs` 확정

### `agents/sub_agents.py`

- Planner / Frontend / Backend / Designer / QA 서브에이전트
- 자유 토론용 `free_discussion_agent()`
- `PASS_TOKEN = "[PASS]"`

### `agents/harness.py`

- 서브에이전트 래퍼
- `harness_enabled=False`면 pass-through
- 기본 모드에서는 예외 격리, 역할 앵커 주입, 역할 탈선 관찰

## 7. 데이터 파이프라인과 RAG

### `data_pipeline/vector_store.py`

`WBSVectorStore`가 단일 저장소에 다음 문서 유형을 저장합니다.

- `prd`
- `member`
- `reference_wbs`
- `meeting_log`
- `disc_profile`

FAISS 초기화가 실패하면 in-memory 검색으로 폴백합니다.

### `data_pipeline/rag_strategies.py`

현재 구현된 전략:

- `vanilla`
- `hybrid`
- `graph`
- `agentic`
- `llm_rerank`

메타데이터는 `STRATEGY_INFO`에 정의돼 있으며 UI/SSE에서도 사용됩니다.

## 8. API 개요

`api.py` 기준 주요 엔드포인트:

| 메서드 | 경로 | 역할 |
|---|---|---|
| `GET` | `/` | `frontend/index.html` 반환 |
| `GET` | `/api/disc-status` | 로드된 eDISC 프로파일 목록 |
| `GET` | `/api/sample` | 샘플 PRD/팀원 |
| `GET` | `/api/sample/pmarket` | P마켓 샘플 응답 |
| `GET` | `/api/sample/meeting` | 샘플 회의록 |
| `POST` | `/api/parse-resume` | 텍스트/PDF 이력서 파싱 |
| `POST` | `/api/wbs/generate` | 스트림 시작 전 준비 응답 |
| `POST` | `/api/wbs/stream` | SSE 기반 WBS 생성 스트림 |

SSE payload에는 다음 정보가 포함될 수 있습니다.

- 현재 acting agent
- 현재 라운드
- 직렬화된 WBS draft
- debate log
- metrics
- llm_judge 결과
- rag metadata

## 9. 평가와 실험

### `metrics.py`

자동 지표 계산과 `compute_autoscore()` 제공. `compute_autoscore()`는 canonical 구현인 `eval_results.autoscore_recompute.recompute_autoscore()`에 위임한다.

대표 지표:

- `faithfulness`
- `success_rate`
- `planning_score`
- `buffer_ratio`
- `mece_score`
- `granularity_fitness`
- `workload_gini`
- `schedule_feasibility`
- `communication_efficiency`
- `token_cost`
- `harness_observability`

현재 canonical AutoScore:

- Version: `v2_backfill_safe`
- Top-level weights: Quality `0.45`, Allocation `0.35`, Orchestration `0.20`
- Quality: Success Rate, MECE, Granularity + optional faithfulness gate
- Allocation: Planning, Schedule Feasibility, Buffer Adequacy, Workload Balance
- Orchestration: Communication Efficiency, Convergence, Revision Yield, Failure Resilience

`mcp_tool_trace`는 현재 자동 점수에 직접 합산되지는 않지만, API/SSE와 snapshot inspection에서 tool boundary 추적 자료로 사용할 수 있다.

### `eval/llm_judge.py`

- 구조 / 배정 / 토론 3차원 평가
- `--judge gemini|claude|cross`
- `--judge-method scalar|geval`
- parse failure나 평가 실패는 `-1`로 표기될 수 있음

### `eval/experiment_runner.py`

주요 CLI:

```bash
python eval/experiment_runner.py --backend gemini --runs 3
python eval/experiment_runner.py --backend gemini --conditions C3_3rounds --judge-method geval --runs 3
python eval/experiment_runner.py --backend gemini --conditions C3_3rounds --harness both --runs 5
python eval/experiment_runner.py --backend gemini --conditions C3_3rounds --cross-judge --runs 3
```

조건군:

- Ablation: `C0_llm_only` ~ `C5_5rounds`
- RAG: `R0_no_rag`, `R1_vanilla`, `R2_hybrid`, `R3_graph`, `R4_agentic`, `R4_llm_rerank`
- 입력 변형: `R1_prd_summary`, `R1_prd_detailed`, `R1_prd_detailed_meeting`, `R5_meeting_regular`, `R5_meeting_no_schedule`
- 에이전트 설계: `A1_*`, `A2_*`, `A3_*`
- 모델/프롬프팅: `S2_*`

## 10. 산출물

### `generated/`

- 최신 단일 실행 결과
- `wbs_output.json`, `wbs_output.md`, `debate_log.md`, `metrics_report.json`, `metrics_history.csv`
- `debug/`: WBS 생성/태스크 매칭 LLM 응답 디버그 덤프. 최종 제출에서는 git ignore 대상.

### `eval_results/`

- 반복 실험 결과와 리포트의 단일 허브
- `summary_*.csv`
- `experiment_*.json`
- `wbs_snapshot_*.json`
- `analysis_report_*.md` 등
- `references/`: AlphaEval, G-Eval 등 평가 방법론 참고 PDF
- `report_assets/`: 발표/보고서 원본 PDF와 이미지 산출물
- `docs/`: 실험 프로토콜, 실행 메모, 평가 설계 초안
- `human_evaluation/`: 설문 집계 및 Human Evaluation 차트

현재 보고서/발표용 상위 문서:

- `eval_results/EXPERIMENTS_SUMMARY.md`
- `eval_results/EVALUATION_FRAMEWORK.md`
- `eval_results/validity_analysis.md`

## 11. 현재 코드 기준 주의사항

- `requirements.txt`는 코어 의존성 중심이며 API/MCP/일부 백엔드 패키지는 별도 설치가 필요할 수 있습니다.
- `.env.example`의 백엔드 설명은 `agents/llm_config.py`와 같이 해석해야 합니다.
- `ref/agent_test/`와 `docs/archive/`는 현재 실행 경로가 아니라 참고 스냅샷입니다.
- 외부 MCP 서버는 존재하지만, WBS 생성 LLM agent가 외부 MCP IPC로 직접 tool을 호출하는 완전 분리 구조는 아닙니다. 현재는 내부 MCP-style boundary trace와 외부 catalog/snapshot tool 제공 단계입니다.
