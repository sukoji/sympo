# symPO MCP Server 가이드

`symPO`의 MCP 서버는 벡터 검색, RAG 비교, eDISC/팀원 컨텍스트 조회, 실험 스냅샷 점검을 외부 에이전트 툴로 노출합니다. 이 문서는 2026-04-27 현재 코드 기준입니다.

## 1. 서버 개요

- 구현 파일: `mcp_server.py`
- 프레임워크: `mcp.server.fastmcp.FastMCP`
- 전송 방식:
  - `stdio` 기본
  - `sse` 선택

기본 실행:

```bash
python mcp_server.py
python mcp_server.py --transport sse --port 8001
```

## 2. 자동 인덱싱 데이터

서버는 첫 벡터 접근 시 `sample_data/`의 샘플 입력을 읽어 `WBSVectorStore`를 구성합니다.

자동 로드 대상:

- `sample_prd.txt`
- `sample_members/*.txt`
- `sample_meeting_transcript.txt`
- `eDISC_*.pdf`

기본 자동 로딩에는 `reference_wbs` 샘플이 포함되지 않습니다. 참고 WBS는 API/실험 경로에서 별도로 주입할 수 있습니다.

## 3. 제공 Tools

현재 `@mcp.tool()` 기준 12개입니다.

| Tool | 설명 |
|---|---|
| `vector_search(query, doc_type="", top_k=5)` | 벡터/폴백 검색 |
| `vector_stats()` | 인덱싱된 문서 유형 통계 |
| `rag_retrieve(query, strategy="vanilla", doc_type="meeting_log", top_k=3)` | 단일 RAG 전략 검색 |
| `rag_compare(query, doc_type="meeting_log", top_k=3)` | `vanilla/hybrid/graph/agentic` 비교 |
| `list_disc_profiles()` | 로드된 eDISC 프로파일 목록 |
| `get_member_context(member_name)` | 특정 팀원 컨텍스트 조회 |
| `evaluate_wbs_snapshot(snapshot_path)` | 저장된 WBS 스냅샷 요약 평가 |
| `list_experiments()` | `eval_results/` 파일 목록 |
| `orchestration_tool_catalog()` | LangGraph phase가 사용하는 MCP-compatible tool catalog |
| `orchestration_phase_plan()` | WBS 생성 phase와 MCP tool 대응 관계 |
| `inspect_wbs_snapshot_state(snapshot_path)` | 저장 snapshot을 state 관점에서 요약 |
| `compact_snapshot_debate_history(snapshot_path, max_messages=8, pm_decisions=4)` | snapshot 토론 로그를 현재 compaction 정책으로 축약 |

### 주의

- `evaluate_wbs_snapshot()`은 `metrics.compute_all_metrics()` 전체를 재실행하지 않고, 스냅샷 JSON에서 읽을 수 있는 경량 통계만 계산합니다.
- `rag_compare()`는 현재 4개 전략만 비교합니다. `llm_rerank`는 별도 비교 목록에 포함되지 않습니다.
- 오케스트레이션 MCP tools는 현재 안전한 단계적 적용을 위해 실제 phase 호출과 동일한 tool 이름을 공유합니다. 런타임은 `mcp_tool_trace`에 호출 기록을 남기고, 외부 MCP client는 같은 catalog를 조회할 수 있습니다.
- 코드 기준 알려진 불일치: `vector_search()`와 `get_member_context()`는 현재 `WBSVectorStore` 반환값을 `(doc, score)` 튜플처럼 순회하지만, `vector_store.py`는 dict 리스트를 반환합니다. 해당 tool은 호출 전 반환 포맷 정리가 필요합니다. RAG strategy 기반 tool과 orchestration catalog/inspection tool은 이 문제와 별개입니다.

## 4. 제공 Resources

현재 `@mcp.resource()` 기준 3개입니다.

| Resource URI | 내용 |
|---|---|
| `sympo://eval-framework` | `eval_results/EVALUATION_FRAMEWORK.md` |
| `sympo://sample-prd` | `sample_data/sample_prd.txt` |
| `sympo://meeting-transcript` | `sample_data/sample_meeting_transcript.txt` |

## 5. Claude Code 등록 예시

```json
{
  "mcpServers": {
    "sympo": {
      "command": "python3",
      "args": ["/home/piai/ai_course/agent_test/mcp_server.py"]
    }
  }
}
```

SSE 모드로 붙일 때는 `python mcp_server.py --transport sse --port 8001`로 띄운 뒤 클라이언트에서 `/sse` 엔드포인트를 사용합니다.

## 6. 오케스트레이션 MCP 구조

현재 WBS 생성 파이프라인은 LangGraph phase를 유지하면서, 각 phase 실행 경계를 MCP tool 이름과 1:1로 맞춘다.

```text
FastAPI / Frontend / Experiment Runner
  → LangGraph Orchestrator
  → WBSState
  → MCP-compatible tool boundary
  → Agent / deterministic function
```

핵심 원리는 다음과 같다.

- LangGraph는 `wbs_generation → task_match → l2_debate → free_discussion → supervisor_mediate → finalize` 순서를 제어한다.
- `WBSState`는 PRD, 팀원, WBS 초안, 토론 로그, 배정 결과를 중앙 상태로 보존한다.
- 각 phase 내부 호출은 `wbs-server.generate_draft`, `assignment-server.match_tasks` 같은 stable MCP tool 이름으로 기록된다.
- 실제 외부 MCP server는 같은 catalog를 `orchestration_tool_catalog()`로 노출한다.
- 런타임에는 `mcp_tool_trace`가 누적되어 어떤 phase에서 어떤 tool 경계가 실행됐는지 추적할 수 있다.
- FastAPI SSE는 `mcp_tool_trace`를 프론트로 전달하고, `frontend/index.html`은 토론 로그 탭의 **Agent Tool Calls** 패널에 tool name, phase label, elapsed time, input/output key를 실시간 표시한다.

현재 대응 관계:

| LangGraph phase | MCP tool boundary | 역할 |
|---|---|---|
| `wbs_generation` | `wbs-server.generate_draft` | WBS 초안 생성/재설계 |
| `task_match` | `assignment-server.match_tasks` | L3 배정, L2 후보 풀, 호출 에이전트 결정 |
| `l2_debate` | `debate-server.candidate_review`, `debate-server.role_review` | L2별 후보/전문가 검토 |
| `free_discussion` | `debate-server.free_discussion` | 상호 검토, PASS 수렴, critic |
| `supervisor_mediate` | `supervisor-server.mediate` | PM 중재, 재배정/버퍼/재설계 판단 |
| `finalize` | `supervisor-server.finalize` | 최종 WBS 확정 |

### 단계적 외부 MCP 전환 원칙

모든 에이전트 호출을 처음부터 외부 MCP IPC로 넘기면 state 직렬화, latency, LLM 호출 순서가 바뀌어 기존 실험 결과가 흔들릴 수 있다. 따라서 전환 순서는 아래가 안전하다.

1. 현재 단계: MCP server가 tool catalog와 snapshot inspection을 제공하고, 런타임은 같은 tool 이름으로 trace를 남긴다.
2. 다음 단계: `compact_snapshot_debate_history`, `inspect_wbs_snapshot_state`, autoscore처럼 결정론적 tool부터 외부 MCP client 경유로 바꾼다.
3. 마지막 단계: WBS 생성/Task Manager/PM 중재처럼 LLM 호출을 포함한 tool을 MCP server로 분리한다. 이때는 state schema version과 serialization contract를 고정해야 한다.

이 구조에서는 기존 FastAPI, 프론트, 실험 runner의 진입점은 유지되고, MCP는 관찰 가능한 tool boundary와 외부 연동 지점을 제공한다.

## 7. 사용 예시

### 벡터 검색

```text
"회의록에서 일정 버퍼 관련 내용을 찾아줘"
```

내부적으로:

```text
vector_search(query="일정 버퍼", doc_type="meeting_log", top_k=5)
```

### 전략별 검색 비교

```text
"고객 세그먼트 분석 쿼리로 RAG 전략별 차이를 비교해줘"
```

내부적으로:

```text
rag_compare(query="고객 세그먼트 분석")
```

### eDISC 목록 확인

```text
"로드된 eDISC 프로파일을 보여줘"
```

내부적으로:

```text
list_disc_profiles()
```

### 실험 결과 탐색

```text
"지금 저장된 실험 결과 파일을 보여줘"
```

내부적으로:

```text
list_experiments()
```

## 8. 직접 테스트

stdio 모드에서 JSON-RPC를 직접 흘려볼 수 있습니다.

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | python3 mcp_server.py
```

## 9. 관련 코드 위치

- 서버 진입점: `mcp_server.py`
- 오케스트레이션 tool catalog: `orchestration/mcp_tool_layer.py`
- LangGraph phase 연결: `orchestration/debate_loop.py`, `orchestration/graph_builder.py`
- 벡터 저장소: `data_pipeline/vector_store.py`
- RAG 전략: `data_pipeline/rag_strategies.py`
- 평가 스냅샷 포맷: `eval/experiment_runner.py`, `eval/llm_judge.py`
