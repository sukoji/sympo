# symPO 프로젝트 전수 감사 문서

작성일: 2026-04-22  
대상 경로: `/home/piai/ai_course/agent_test`

현재화 메모(2026-04-27): 이 문서는 2026-04-22 감사 스냅샷이다. 최신 실행/평가 기준은 `README.md`, `PROJECT_OVERVIEW.md`, `MCP_GUIDE.md`, `eval_results/EVALUATION_FRAMEWORK.md`, `eval_results/EXPERIMENTS_SUMMARY.md`, `eval_results/validity_analysis.md`를 우선한다. 특히 현재 코드는 `orchestration/mcp_tool_layer.py` 기반 `mcp_tool_trace`, AutoScore v2(`0.45/0.35/0.20`), G-Eval judge, context metadata 실험 결과를 반영한다.

이 문서는 현재 프로젝트 폴더를 기준으로 실제 실행 코드, 보조 모듈, 샘플 데이터, 평가 체계, 산출물, 보관 스냅샷까지 포함해 프로젝트의 동작 구조를 전수 정리한 감사 문서다.

## 1. 한 줄 요약

이 프로젝트는 PRD, 팀원 이력서/프로필, eDISC 성향 자료, 참고 WBS/회의록을 입력으로 받아 멀티에이전트 토론을 거쳐 3단계 WBS를 생성하고, 그 결과를 자동 지표와 LLM Judge로 평가하는 시스템이다.

## 2. 현재 폴더의 성격

- 현재 폴더는 Git 저장소가 아니다. `git status`는 동작하지 않았다.
- 최신 실행 코드와 과거 복제본이 같이 있다.
- 실제 최신 실행 축은 루트의 `main.py`, `api.py`, `agents/`, `orchestration/`, `data_pipeline/`, `schemas/`, `metrics.py`, `eval/`이다.
- `ref/agent_test/`는 예전 시점의 코드 복제본이다.
- `docs/archive/`는 과거 문서 스냅샷이다.
- `__pycache__/`, 이미지 파일(`image.png`, `image copy.png`), PDF, 텍스트 로그는 실행 보조/기록물이다.

## 3. 디렉토리 분류

### 3.1 실제 실행 코드

- `main.py`: Streamlit UI 메인 엔트리포인트
- `api.py`: FastAPI + SSE 백엔드 엔트리포인트
- `mcp_server.py`: 외부 에이전트 연동용 MCP 서버
- `agents/`: 생성/배정/토론/하네스/LLM 설정
- `orchestration/`: 실행 그래프와 토론 루프
- `data_pipeline/`: PRD/이력서/eDISC 파싱과 RAG 저장소/전략
- `persona_engine/`: 팀원 프로필을 에이전트 프롬프트로 변환
- `schemas/`: Pydantic 모델
- `output/`: WBS/토론 로그 저장 포맷터
- `metrics.py`: 자동 지표 계산
- `eval/`: 실험 러너, 분석기, Judge, 보조 평가 모듈
- `frontend/index.html`: FastAPI 루트에서 서빙되는 단일 페이지 UI

### 3.2 입력 데이터

- `sample_data/sample_prd.txt`: 기본 샘플 PRD
- `sample_data/sample_prd_summary.txt`: 요약 PRD
- `sample_data/sample_prd_detailed.txt`: 상세 PRD
- `sample_data/prd_sample.txt`: 별도 PRD 샘플
- `sample_data/sample_reference_wbs.txt`: RAG 참고용 WBS
- `sample_data/sample_meeting_transcript.txt`: 일정 정보 포함 회의록
- `sample_data/sample_meeting_no_schedule.txt`: 일정 정보 제거 회의록
- `sample_data/sample_members/*.txt`: 샘플 팀원 이력서/소개 텍스트
- `sample_data/sample_pmarket_members/*.txt`: P마켓 시나리오용 팀원 텍스트
- `sample_data/eDISC_*.pdf`: 팀원별 eDISC PDF

### 3.3 런타임 산출물

- `generated/wbs_output.json`
- `generated/wbs_output.md`
- `generated/debate_log.md`
- `generated/metrics_report.json`
- `generated/metrics_history.csv`

### 3.4 실험 산출물

- `eval_results/summary_*.csv`: 조건별 실행 결과 요약
- `eval_results/experiment_*.json`: 개별 실험 메타 결과
- `eval_results/wbs_snapshot_*.json`: 조건/반복별 WBS 스냅샷
- `eval_results/analysis_report_*.md`: 통계 분석 리포트
- `eval_results/figures/`: 그림 산출물
- `eval_results/gemma_ablation/`, `eval_results/rag_strategy_comparison/`, `eval_results/edisc_rr_matching/`: 서브 실험 세트

### 3.5 문서/보관본

- `README.md`: 현재 상위 SoT
- `PROJECT_OVERVIEW.md`: 상세 레퍼런스 문서
- `CLAUDE.md`, `MCP_GUIDE.md`, `exec.txt`, `eval2.txt`: 운영/실험 가이드
- `ref/agent_test/`: 코드 복제본
- `docs/archive/`: 과거 분석 문서

## 4. 시스템 핵심 아키텍처

프로젝트의 중심은 `WBSState`라는 공유 상태 객체다. 모든 에이전트는 이 상태를 읽고 일부 필드를 갱신하는 방식으로 동작한다.

전체 파이프라인은 다음 순서다.

1. 입력 파싱
2. RAG 컨텍스트 구축
3. 초기 상태 생성
4. WBS 초안 생성
5. 태스크-팀원 배정
6. 서브에이전트 토론 및 PM 중재
7. 합의 또는 라운드 종료
8. 최종 WBS 확정
9. 지표 계산 및 결과 저장

## 5. 실제 런타임 흐름

### 5.1 Streamlit 경로

`main.py`는 사용자 입력을 받아 다음 작업을 수행한다.

- `.env` 로드
- eDISC 프로파일 캐시 로드
- PRD 텍스트 또는 폼을 `PRDParser`로 구조화
- 팀원 텍스트/PDF를 `MemberParser`로 구조화
- `PersonaBuilder`로 팀원별 페르소나 생성
- `WBSVectorStore`를 만들고 PRD/팀원/참고 WBS/회의록/eDISC를 인덱싱
- 선택된 RAG 전략으로 참고 문맥 검색
- `create_initial_state()`로 `WBSState` 생성
- `execute_sympo_flow()` 제너레이터를 반복 소비하면서 UI 갱신
- 종료 후 `WBSGenerator`와 `ReportWriter`로 산출물 저장
- `compute_all_metrics()`로 정량 지표 계산

즉, `main.py`는 단순 프론트가 아니라 전체 오케스트레이션을 직접 구동하는 두꺼운 UI 엔트리포인트다.

### 5.2 FastAPI 경로

`api.py`는 거의 같은 로직을 API 형태로 감싼다.

- 시작 시 `.env` 로드
- 서버 부팅 시 `sample_data` 내 eDISC PDF를 선로드
- `GET /`: `frontend/index.html` 반환
- `GET /api/disc-status`: 로드된 eDISC 목록 반환
- `GET /api/sample`: 샘플 PRD/팀원 데이터 반환
- `GET /api/sample/pmarket`: P마켓용 하드코딩 샘플 반환
- `POST /api/wbs/generate`: 실제 생성 대신 준비 응답만 반환
- `POST /api/wbs/stream`: 핵심 엔드포인트

`/api/wbs/stream` 내부 동작:

1. 요청의 `llm_backend`를 환경변수에 반영
2. PRD를 폼 또는 raw text에서 파싱
3. 팀원 목록을 `MemberProfile` 리스트로 변환
4. `PersonaBuilder`로 supervisor/team persona 생성
5. `WBSVectorStore`에 PRD/팀원/참고 WBS/회의록/eDISC를 적재
6. `rag_strategy`에 따라 참고 WBS/회의록 검색
7. `create_initial_state()` 생성 후 RAG/eDISC 컨텍스트 주입
8. `execute_sympo_flow()`를 반복하면서 상태를 JSON으로 SSE 전송
9. 종료 후 `compute_all_metrics()` 호출
10. 옵션이 켜져 있으면 `eval.llm_judge.evaluate_wbs()` 실행
11. 마지막에 RAG 메타데이터까지 전송

즉 API 경로도 내부 핵심은 Streamlit과 동일한 제너레이터 기반 실행이다.

### 5.3 실험 러너 경로

`eval/experiment_runner.py`는 수동 UI 없이 조건 조합을 반복 실행한다.

- 조건 세트 `CONDITIONS`를 정의한다.
- C0~C5, R0~R4, A1~A3, S2, R1, R5 같은 실험군이 코드에 하드코딩되어 있다.
- 샘플 PRD/팀원/eDISC를 불러온다.
- 조건마다 `create_initial_state()`를 만든다.
- 조건에 따라 일부 단계를 생략하거나 전체 `execute_sympo_flow()`를 실행한다.
- 실행 후 `compute_all_metrics()`와 LLM Judge를 돌린다.
- 결과를 `eval_results/`에 CSV/JSON/스냅샷으로 저장한다.

프로젝트는 앱인 동시에 연구 실험 프레임워크다.

## 6. 오케스트레이션 계층

### 6.1 `agents/state.py`

`WBSState`는 중앙 공유 상태다. 핵심 필드는 다음 범주로 나뉜다.

- 입력: `prd`, `team_members`, `agent_personas`
- WBS: `current_wbs_draft`, `final_wbs`
- 로그: `debate_log`
- RAG: `rag_reference_wbs`, `rag_meeting_logs`, `disc_profiles`
- 제어: `current_round`, `min_rounds`, `max_rounds`, `consensus_reached`
- 배정: `assigned_tasks`, `called_agents`, `calling_context`, `locked_assignments`, `l2_agent_mapping`
- 임시 응답: `planner_response`, `frontend_response`, `backend_response`, `designer_response`, `qa_response`
- 재설계/수렴: `wbs_revision_needed`, `wbs_revision_hints`, `current_wbs_revision`, `total_days_history`
- 실험 변수: `harness_enabled`, `max_free_turns`, `veto_enabled`, `critic_enabled`, `persona_strictness`, `prd_variant`, `model_class`, `prompting_strategy`
- 내부 제어: `_free_turn_count`, `_current_l2_task_id`, `_current_agent_acting`, `_l2_debate_cutoff`

중요한 설계 포인트:

- `create_initial_state()`는 팀원 `role`을 `team_members`에서 제거하고 `member_role_map`에 따로 저장한다.
- 목적은 LLM에 role 정답을 그대로 노출하지 않고 supervisor가 독립적으로 배정하게 만드는 것이다.

### 6.2 `orchestration/debate_loop.py`

핵심 제너레이터는 `execute_sympo_flow(state, max_rounds)`다.

- LangSmith trace 컨텍스트를 연다.
- LangGraph가 있으면 `graph_builder.get_compiled_graph()`를 통해 custom stream 모드로 실행한다.
- LangGraph가 없으면 `run_sequential_debate()`를 직접 돈다.

실제 토론 로직은 `debate_loop.py`의 phase 함수들에 있다. LangGraph 경로와 sequential fallback 경로가 같은 phase 함수들을 공유하므로, 라운드 수·PASS 조기 종료·WBS 재설계·critic 토글·SSE 중간 상태 스트리밍 동작을 맞춘다.

외부 루프:

- WBS 초안 생성
- 필요 시 WBS 재설계 후 다시 생성

phase 순서:

1. `phase_wbs_generation`
2. `phase_task_match`
3. `phase_l2_debate`
4. `phase_free_discussion`
5. `phase_critic_review`
6. `phase_supervisor_mediate`
7. 조건부 라우팅: 재설계면 `phase_wbs_generation`, 추가 라운드면 `phase_task_match`, 종료면 `phase_finalize`

### 6.3 `orchestration/graph_builder.py`

이 파일은 현재 phase 단위 LangGraph DAG를 정의한다.

```text
START
  -> wbs_generation
  -> task_match
  -> l2_debate
  -> free_discussion
  -> critic_review
  -> supervisor_mediate
  -> route_after_mediate
       -> wbs_generation  # WBS 재설계
       -> task_match      # 다음 라운드
       -> finalize        # 합의/라운드 종료
  -> END
```

각 node는 `debate_loop.py`의 phase generator를 실행하고, 중간 state는 `get_stream_writer()`로 custom stream에 전달한다. `WBSState.debate_log`는 `operator.add` reducer를 사용하므로, LangGraph node 반환값에는 전체 로그가 아니라 해당 phase에서 새로 추가된 로그만 담아 중복 누적을 피한다.

## 7. 에이전트 계층

### 7.1 `agents/llm_config.py`

`get_llm()`이 백엔드를 선택한다.

지원 백엔드:

- `mock`
- `gemini`
- `gemma4`
- `gemma4-api`
- `qwen-api`
- `llama4`
- `openai`
- `ollama`
- `anthropic`

보조 기능:

- `normalize_content()`: 멀티파트 응답을 문자열로 평탄화
- 로컬 Gemma4 싱글톤
- OpenAI 호환 API용 Gemma4/Qwen 래퍼

### 7.2 `agents/wbs_gen_agent.py`

역할: Phase 1 WBS 초안 생성

핵심 로직:

- 프로젝트 유형을 `dev` 또는 `business`로 추론
- PRD, RAG WBS 문맥, 팀 정보를 프롬프트에 넣어 L1/L2/L3 구조를 생성
- JSON 응답을 파싱
- `repair_truncated_json()`으로 잘린 JSON 복구
- `_retry_missing_l3()`로 L3 누락 부분만 재생성 요청
- `_ensure_l3_coverage()`로 그래도 비는 L2에 최소 stub L3 합성

즉 약한 모델이 L3를 덜 만들더라도 후처리로 최소 구조를 보장한다.

### 7.3 `agents/supervisor_agent.py`

역할: PM/슈퍼바이저 전담

주요 함수:

- `supervisor_task_match()`
- `supervisor_check_and_intervene()`
- `supervisor_mediate()`
- `supervisor_finalize()`
- `_smart_assign()`

`supervisor_task_match()`:

- L3만 실제 배정 대상으로 본다.
- 팀원 기술, 강점, 회의록, eDISC를 모두 참고해 `allocations`, `called_agents`, `calling_context`, `l2_agent_mapping`을 생성한다.
- L1/L2는 배정 없이 요약 노드로 유지한다.
- 마지막에 `_smart_assign()`로 후처리 배정 균형을 잡는다.

`supervisor_check_and_intervene()`:

- 순수 동의만 반복될 때 조기 종료
- 구현 세부사항으로 새는 토론 리다이렉트
- 동일 발언자 독점 차단
- 마이크로 태스크 증식 차단
- 역할 사칭/반복 차감 패턴 제어

`supervisor_mediate()`:

- 최근 토론을 읽고 버퍼, 재배정, 신규 태스크를 추출
- veto/critic/재설계/수렴 판단을 반영
- `_apply_dynamic_buffers()`, `_apply_reassignments()`, `_smart_assign()`를 이용해 WBS 업데이트
- `locked_assignments`를 갱신해 PM 결정이 뒤집히지 않게 한다.

`supervisor_finalize()`:

- `final_wbs`, `generation_summary`, 최종 PM 메시지를 만든다.

### 7.4 `agents/sub_agents.py`

역할: 서브에이전트 토론

구성:

- 플래너
- 프론트엔드
- 백엔드
- 디자이너
- QA
- 크리틱
- 자유토론 에이전트

핵심 특징:

- `calling_context`를 우선 사용해 어떤 실제 멤버가 어느 역할로 말할지 결정
- `member_role_map`과 기술 스택을 함께 봐서 fallback 멤버 선택
- 최근 토론 이력과 PM 하드 제약을 프롬프트에 삽입
- PASS 규칙이 매우 강하게 설계되어 토큰 낭비를 줄이려 한다.

### 7.5 `agents/harness.py`

이 파일은 실험용 토글 래퍼다.

- `HARNESS_ENABLED=false` 또는 state에서 끄면 H0 baseline
- 기본은 H1
- 기능:
  - 예외 격리
  - 역할 앵커 주입
  - 출력 role drift 관찰

즉 서브에이전트의 안정성 자체가 실험 변수다.

## 8. 입력 파이프라인

### 8.1 `data_pipeline/prd_parser.py`

역할:

- 폼 입력을 `PRDInput`으로 변환
- 자유 텍스트 PRD에서 프로젝트명, 목표, 범위, 기능, 기술스택, 제약, 타깃 사용자를 추출

특징:

- 한국어 번호 목록 파싱
- 비즈니스 PRD도 처리하도록 휴리스틱 설계
- raw text를 그대로 `PRDInput.raw_text`에 보관

### 8.2 `data_pipeline/member_parser.py`

역할:

- 폼 입력 또는 이력서 텍스트를 `MemberProfile`로 변환

핵심:

- 기술 키워드 기반 역할 추론
- 이름 추출
- 경력, 기술, 강점, 약점 파싱
- 규칙 기반 버전과 LLM 기반 정밀 버전 둘 다 존재

### 8.3 `data_pipeline/disc_parser.py`

역할:

- eDISC PDF를 `DiscProfile`로 변환

파싱 항목:

- 이름
- DISC 스타일
- 주 유형
- 복합 코드
- 점수
- 강점 행동
- 보완 행동
- 행동 키워드
- 의사소통 스타일
- 의사결정 스타일
- 동기/비동기 요인
- 팀 역할

출력 메서드:

- `to_rag_text()`: 벡터스토어 적재용
- `to_agent_context()`: 프롬프트 직접 주입용

### 8.4 `data_pipeline/vector_store.py`

역할:

- PRD, 팀원, 참고 WBS, 회의록, eDISC를 한 저장소에 적재

구조:

- 가능하면 FAISS + HuggingFace 임베딩 사용
- 실패하면 메모리 키워드 검색으로 폴백

문서 타입:

- `prd`
- `member`
- `reference_wbs`
- `meeting_log`
- `disc_profile`

### 8.5 `data_pipeline/rag_strategies.py`

지원 전략:

- `vanilla`: dense similarity
- `hybrid`: BM25 + dense + RRF
- `graph`: 엔티티 공출현 그래프
- `agentic`: 반복적 쿼리 확장
- `llm_rerank`: dense 후보를 LLM으로 재정렬

즉 RAG 비교 실험이 코드에 직접 들어 있으며, 문서상 4종처럼 보이지만 실제 구현 레지스트리는 5종이다.

## 9. 페르소나 엔진

`persona_engine/persona_builder.py`는 `MemberProfile`을 에이전트 프롬프트 문자열로 바꾼다.

포함 정보:

- 경력
- 핵심 스킬
- 강점/약점
- 과거 프로젝트
- 병목 요인
- 투입률

역할이 없는 경우 기술스택으로 에이전트 타입을 역추론한다.

## 10. 스키마

### 10.1 `schemas/prd_schema.py`

`PRDInput`:

- `project_name`
- `project_goal`
- `target_users`
- `scope`
- `key_features`
- `tech_stack_requirements`
- `deadline`
- `team_size`
- `budget_weeks`
- `special_constraints`
- `raw_text`

### 10.2 `schemas/member_schema.py`

`MemberRole`:

- PM
- Planner
- Frontend Developer
- Backend Developer
- Fullstack Developer
- Designer
- QA Engineer
- Data Engineer
- DevOps
- Data Analyst
- Marketing Planner
- Business Analyst
- Mobile Developer
- Operations Manager

`MemberProfile`:

- `member_id`
- `name`
- `role`
- `sub_roles`
- `years_of_experience`
- `tech_stack`
- `primary_skills`
- `strengths`
- `weaknesses`
- `personality_traits`
- `past_projects`
- `preferred_task_types`
- `known_bottlenecks`
- `availability_percent`
- `raw_resume_text`

### 10.3 `schemas/wbs_schema.py`

`WBSTask`:

- `task_id`
- `level`
- `parent_id`
- `title`
- `description`
- `assigned_to`
- `required_role`
- `assigned_role`
- `estimated_days`
- `buffer_days`
- `total_days`
- `dependencies`
- `risk_factors`
- `buffer_rationale`
- `start_week`
- `end_week`
- `deliverables`
- `status`
- `importance`

`DebateMessage`:

- 발언 시각
- agent role
- agent name
- message
- message type
- related task id
- proposed buffer days

## 11. 출력 계층

### 11.1 `output/wbs_generator.py`

역할:

- WBS 태스크에 주차를 부여
- `WBSOutput` 패키지 생성
- 계층 딕셔너리 변환
- 크리티컬 패스 계산

일정 계산 방식:

- L1 순차
- L2는 부모 L1 안에서 순차
- L3는 부모 L2 안에서 순차

즉 현재는 정교한 CPM/리소스 제약 스케줄러가 아니라 계층 기반 순차 배정기다.

### 11.2 `output/report_writer.py`

역할:

- Markdown WBS 테이블 작성
- 토론 로그 Markdown 저장
- JSON 산출물 저장
- 개별 태스크 일정 설명 생성

## 12. 평가 체계

### 12.1 `metrics.py`

`compute_all_metrics()`가 최종 상태를 받아 여러 지표를 계산하고 저장한다.

핵심 지표 묶음:

- Faithfulness
- Interaction Turns
- Supervisor Intervention
- Success Rate
- Planning Score
- Buffer Ratio
- Convergence
- MECE
- Granularity Fitness
- Workload Gini
- Schedule Feasibility
- Communication Efficiency
- Token Cost
- AutoScore

결과 저장:

- `generated/metrics_report.json`
- `generated/metrics_history.csv`

### 12.2 `eval/llm_judge.py`

README와 API 흐름상 LLM Judge는 구조/배정/토론 3축 평가기로 쓰인다.

### 12.3 `eval/analyze_results.py`

역할:

- summary CSV 여러 개 로드
- 조건별 그룹화
- 평균/표준편차 계산
- Mann-Whitney U, Cliff's delta, Holm-Bonferroni 적용
- Markdown 분석 리포트 생성

### 12.4 `eval/experiment_runner.py`

이 파일은 프로젝트의 연구용 실험 운영본이다.

조건 예시:

- C0: LLM 단독
- C1: 배정 포함
- C2: 1라운드 토론
- C3: 3라운드 토론
- C4: eDISC 포함
- C5: 5라운드 토론
- R0~R4: RAG 전략 비교
- R4-B: LLM rerank 전략 비교
- R5: 회의록 변형 비교
- A1/A2/A3: critic, veto, persona strictness
- S2: 모델 급과 프롬프팅 조합
- R1: PRD 정보 밀도 조건

즉 앱 로직 자체보다 실험 설계 코드 양이 상당히 크다.

### 12.5 평가/실험 보조 모듈

`eval/` 디렉토리는 `experiment_runner.py`와 `llm_judge.py`만 있는 수준이 아니다. 실제로는 다음 보조 평가 축까지 구현돼 있다.

- `baseline_runner.py`: 단일 LLM zero-shot baseline WBS 생성
- `benchmark_generator.py`: planted constraints 기반 합성 벤치마크 생성
- `constraint_checker.py`: 심어진 제약 충족률 계산
- `structural_checker.py`: orphan, dependency, 8-80 rule, MECE rollup 등 구조 무결성 검사
- `red_team.py`: LLM 기반 적대적 결함 탐지
- `langsmith_evaluator.py`: LangSmith dataset/evaluation 연동
- `merge_results.py`: 여러 summary CSV 병합
- `reliability.py`: ICC, Cohen's kappa, Spearman, bootstrap CI
- `sensitivity.py`: Success Rate/AutoScore 하이퍼파라미터 감도 분석
- `run_evaluation.py`: GT-free 벤치마크 평가 파이프라인
- `generate_figures.py`: 논문/리포트용 figure 생성

즉 현재 프로젝트는 "앱 + 실험 러너"를 넘어, baseline·synthetic benchmark·적대적 평가·신뢰성 분석까지 포함한 비교적 넓은 연구 툴체인이다.

## 13. MCP 서버

`mcp_server.py`는 외부 LLM 도구로 프로젝트를 탐색하게 하기 위한 서버다.

기능:

- 벡터 검색
- 벡터 통계
- RAG retrieve
- RAG compare
- eDISC 목록
- 멤버 컨텍스트 조회
- WBS 스냅샷 간이 평가
- 실험 결과 파일 목록 조회
- 정적 resource 노출 (`sympo://eval-framework`, `sympo://sample-prd`, `sympo://meeting-transcript`)

샘플 데이터를 자동 인덱싱하여 tool 호출만으로 프로젝트 예시를 탐색할 수 있게 한다.

다만 현재 구현에는 중요한 불일치가 있다.

- `vector_search()`는 `vector_store.py`의 결과를 `(doc, score)` 튜플처럼 순회하지만, 실제 `retrieve_by_type()`/`retrieve_all_context()`는 dict 리스트를 반환한다.
- `get_member_context()`도 `retrieve_member_context()`를 `(doc, score)` 튜플처럼 가정한다.

즉 MCP 서버는 "존재한다"는 수준을 넘어, 일부 tool은 현재 코드 기준 실제 호출 시 깨질 가능성이 있다.

## 14. 환경변수

`.env.example` 기준 핵심 변수:

- `LLM_BACKEND`
- `GOOGLE_API_KEY`
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `OLLAMA_BASE_URL`
- `OLLAMA_MODEL`
- `GEMMA4_API_URL`
- `EMBEDDING_MODEL`
- `MAX_DEBATE_ROUNDS`
- `OUTPUT_DIR`
- `RUNNER_ID`
- `LANGCHAIN_*`, `LANGSMITH_*`
- `HF_TOKEN`

## 15. 실제 데이터 사용 방식

프로젝트는 다음 데이터를 서로 다른 방식으로 사용한다.

- PRD: 구조화 입력 + RAG 문맥 원천
- 팀원 이력서: 역할 추론, 배정 근거, 페르소나 생성
- eDISC PDF: `disc_profiles`와 RAG 문맥 둘 다로 사용
- 참고 WBS: WBS 생성 시 구조 참고용 RAG
- 회의록: 일정 교훈, 버퍼 근거, 배정 판단용 RAG
- debate_log: 중재와 평가의 입력
- final_wbs/current_wbs_draft: 최종 산출과 평가의 핵심 데이터

## 16. 폴더 내 중복/보관 구조

- `ref/agent_test/`: 과거 코드 전체 미러
- `docs/archive/`: 과거 분석 문서
- `main.py.utf8`: 루트 `main.py`의 별도 저장본으로 보이는 파일
- `server.txt`, `exec.txt`, `eval2.txt`, `TEAM_EXPERIMENT.md`, `EXPERIMENT_PROTOCOL.md` 등은 운영 기록/명세 문서

즉 전수검사 시 루트 최신 코드와 보관본을 혼동하면 안 된다.

## 17. 현재 코드 기준 잠재적 특이사항

감사 중 확인한 구조적 포인트:

- 실제 핵심 오케스트레이션은 LangGraph 세분 노드가 아니라 sequential loop다.
- `main.py`와 `api.py`가 거의 같은 핵심 파이프라인을 중복 구현한다.
- `output/` 모듈은 현재 Streamlit 경로에서 사용되며, API 경로는 SSE 전달 후 지표 계산까지 담당한다.
- `mcp_server.py`는 `vector_store.py`의 반환 형식을 일부 구간에서 튜플처럼 다루는 코드가 보여, 실제 호출 시 불일치 가능성이 있다.
- `team_members`에서 role을 제거하고 `member_role_map`으로 따로 들고 가는 설계가 프로젝트의 중요한 보안/실험 포인트다.
- WBS 일정 산정은 리소스 병렬화 최적화보다 계층 순차 배정에 가깝다.
- `orchestration/debate_loop.py`에는 현재 주 실행 경로 외에 `free_discussion_node()`, `sub_agent_debate_node()` 같은 LangGraph 지향 보조 노드도 남아 있다. 문서화는 해야 하지만, 메인 실행축과 동일 비중으로 보면 안 된다.
- `agents/harness.py`는 단순 래퍼가 아니라 RQ1-H 독립변수다. 예외 격리와 role drift 관찰은 실제 메트릭/실험 분석과 연결된다.
- `eval/experiment_runner.py`의 조건군은 README 수준 요약보다 더 넓다. C계열, RAG계열, R5 회의록 변형, A1 critic, A2 veto, A3 persona strictness, S2 모델급/프롬프팅, R1 PRD 밀도까지 포함한다.

## 18. 이 프로젝트를 이해할 때 가장 중요한 파일 순서

처음 읽을 때는 아래 순서가 가장 효율적이다.

1. `README.md`
2. `agents/state.py`
3. `orchestration/debate_loop.py`
4. `agents/wbs_gen_agent.py`
5. `agents/supervisor_agent.py`
6. `agents/sub_agents.py`
7. `api.py`
8. `main.py`
9. `data_pipeline/*`
10. `metrics.py`
11. `eval/experiment_runner.py`

## 19. 결론

이 프로젝트는 단순한 WBS 생성기가 아니라 다음이 결합된 복합 시스템이다.

- PRD/이력서/eDISC 파서
- RAG 기반 컨텍스트 검색기
- 멀티에이전트 토론 오케스트레이터
- PM 중재형 WBS 재배정 시스템
- 자동 정량 평가 시스템
- LLM Judge 기반 실험 프레임워크
- MCP 기반 외부 도구 서버

최신 실행 기준의 핵심 로직은 루트 코드에 있고, `ref/`와 `docs/archive/`는 참고용 보관 영역으로 보는 것이 맞다.
