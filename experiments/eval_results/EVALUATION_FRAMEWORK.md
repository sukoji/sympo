# symPO 평가 프레임워크

이 문서는 현재 코드 구현 기준의 평가/실험 체계를 요약한다. 세부 산식은 `metrics.py`, `eval_results/autoscore_recompute.py`, `eval/experiment_runner.py`, `eval/analyze_results.py`, `eval/llm_judge.py`를 우선한다.

최종 정리일: 2026-04-27

## 1. 평가 범위

symPO는 WBS 결과를 두 층위로 평가한다.

- 자동 지표: `metrics.py`
- LLM-as-a-Judge: `eval/llm_judge.py`

평가 철학은 AlphaEval의 multi-paradigm composition 방식처럼 단일 점수 하나가 아니라 여러 평가 패러다임을 결합한다. Judge 쪽은 G-Eval 스타일의 form-filling과 score-token probability weighted scoring을 지원한다.

## 2. 자동 지표

`compute_all_metrics()`는 최종 state, PRD, 팀원 목록을 받아 지표 묶음을 계산한다.

### 생성 품질

- `faithfulness`: RAG 컨텍스트 근거 충실도. NLI 모델(`cross-encoder/nli-MiniLM2-L6-H768`)을 우선 사용하고, 모델 로드 실패 시 `keyword_fallback`으로 기록한다.
- `success_rate`: PRD key feature 커버리지. sentence-transformers 임베딩 기반이며, 실패 시 키워드 fallback을 사용한다.
- `mece_score`: WBS 계층 구조의 중복/누락 완화 정도.
- `granularity_fitness`: 태스크 크기와 WBS 세분화 적합성.

### 배분 품질

- `planning_score`: 팀원 스킬/강점과 태스크 배정 간 적합도.
- `buffer_ratio`: 일정 버퍼 반영 비율.
- `workload_gini`: 팀원별 업무량 불균형. 낮을수록 좋다.
- `schedule_feasibility`: 시작/종료/의존성 기반 일정 타당성.

### 오케스트레이션

- `interaction_turns`: 토론 발화 수와 참여자 수.
- `supervisor_intervention`: PM/슈퍼바이저 개입 비율.
- `convergence`: 버퍼/배정/발화 흐름 기반 수렴도.
- `communication_efficiency`: PASS, 순수 동의, 시스템 메시지를 제외한 유효 발화 비율.
- `harness_observability`: 하네스 예외 포착과 role drift 관찰값.
- `mcp_tool_trace`: 내부 MCP-style tool boundary 호출 기록. 실제 외부 MCP IPC가 아니라 기존 Python 경로를 유지하면서 tool 이름, 입력 키, 출력 키, 성공/실패, 지연시간을 추적한다.

### 비용

- `token_cost`: 실제 API 사용량이 아니라 문자 수 기반 추정 토큰/비용.

## 3. AutoScore

런타임 `metrics.compute_autoscore()`는 canonical 구현인 `eval_results.autoscore_recompute.recompute_autoscore()`에 위임한다.

현재 canonical 버전:

- `autoscore_version = v2_backfill_safe`
- Top-level weights: Quality `0.45`, Allocation `0.35`, Orchestration `0.20`

카테고리 내부 가중치:

| Category | Components |
|---|---|
| Quality | `success_rate` 0.35, `mece_score` 0.35, `granularity_fitness` 0.30 |
| Allocation | `planning_score` 0.45, `schedule_feasibility` 0.25, `buffer_adequacy` 0.15, `workload_balance` 0.15 |
| Orchestration | `communication_efficiency` 0.25, `convergence` 0.30, `revision_yield` 0.25, `failure_resilience` 0.20 |

RAG faithfulness가 존재하면 Quality에 evidence gate를 적용한다. historical CSV에 없는 지표는 0점 처리하지 않고 N/A로 두며, 가능한 항목만 재정규화한다.

슬라이드나 보고서에 이전 수식 `0.40 x Quality + 0.35 x Allocation + 0.25 x Orchestration`을 사용할 경우, 이는 초기 설계안으로 표기해야 한다. 현재 코드 기준 canonical AutoScore는 `0.45/0.35/0.20`이다.

## 4. N/A 처리

현재 코드 기준으로 일부 지표는 `-1`을 사용해 N/A를 표현한다.

- `faithfulness = -1`: RAG 컨텍스트가 없는 조건
- `planning_score = -1`: 임베딩 기반 계산 불가
- Judge 항목 `-1`: 평가 호출 실패, 파싱 실패, 또는 해당 차원 제외

AutoScore와 Judge overall은 활성 차원만 대상으로 재정규화한다.

## 5. LLM-as-a-Judge

`eval/llm_judge.py`는 세 차원을 평가한다.

- Structure
- Assignment
- Debate

기본 가중치:

- Structure `0.40`
- Assignment `0.35`
- Debate `0.25`

지원 모드:

- `scalar`: 기존 JSON rubric judge
- `geval`: G-Eval 방식의 1~5 score-token 확률가중 채점. Claude judge는 현재 scalar로 fallback한다.
- `cross_judge=True`: 1차 judge와 2차 judge를 함께 저장한다.

기본 judge 환경변수:

- `JUDGE_MODEL_GEMINI=gemini-3.1-pro-preview`
- `JUDGE_MODEL_CLAUDE=claude-sonnet-4-6`
- `JUDGE_METHOD=geval`

CLI:

```bash
python eval/llm_judge.py eval_results/wbs_snapshot_*.json --judge gemini --judge-method geval
python eval/llm_judge.py eval_results/wbs_snapshot_*.json --judge claude --judge-method scalar
python eval/llm_judge.py eval_results/wbs_snapshot_*.json --judge cross --judge-method geval
```

## 6. 실험 조건군

`eval/experiment_runner.py`의 `CONDITIONS` 기준:

### Ablation

- `C0_llm_only`
- `C1_with_assign`
- `C2_1round`
- `C3_3rounds`
- `C4_with_disc`
- `C5_5rounds`

### RAG

- `R0_no_rag`
- `R1_vanilla`
- `R2_hybrid`
- `R3_graph`
- `R4_agentic`
- `R4_llm_rerank`

### 입력 변형

- `R1_prd_summary`
- `R1_prd_detailed`
- `R1_prd_detailed_meeting`
- `R5_meeting_regular`
- `R5_meeting_no_schedule`

### 에이전트 설계

- `A1_single`, `A1_critic`
- `A2_no_veto`, `A2_with_veto`
- `A3_defensive`, `A3_aggressive`

### 모델/프롬프팅

- `S2_frontier_single`
- `S2_frontier_chaining`
- `S2_frontier_cot`
- `S2_8b_single`
- `S2_8b_chaining`
- `S2_8b_cot`

## 7. 실험 실행 예시

```bash
python eval/experiment_runner.py --backend mock --runs 1
python eval/experiment_runner.py --backend gemini --conditions C0_llm_only C3_3rounds --runs 5
python eval/experiment_runner.py --backend qwen-api --conditions C3_3rounds --runs 3 --judge-method geval
python eval/experiment_runner.py --backend gemini --conditions C3_3rounds --harness both --runs 5
python eval/experiment_runner.py --backend gemini --conditions C3_3rounds --cross-judge --runs 3
```

지원 backend:

- `mock`
- `gemini`
- `openai`
- `gemma4`
- `gemma4-api`
- `qwen-api`
- `anthropic`
- `ollama`

## 8. 통계 및 후처리

현재 분석 스크립트 기준:

- 요약 통계: 평균, 표준편차, `n`, `n_na`
- 유의성 검정: Mann-Whitney U
- 효과 크기: Cliff's delta
- 다중 비교: Holm-Bonferroni
- 신뢰도: ICC, Cohen's kappa, Spearman rho
- 민감도: SR threshold, MECE threshold, AutoScore weight grid

관련 파일:

- `eval/analyze_results.py`
- `eval/reliability.py`
- `eval/sensitivity.py`

## 9. 산출물

### 단일 실행 산출물

- `generated/metrics_report.json`
- `generated/metrics_history.csv`

### 실험 산출물

- `eval_results/summary_*.csv`
- `eval_results/experiment_*.json`
- `eval_results/wbs_snapshot_*.json`
- `eval_results/analysis_report_*.md`

팀 단위 취합:

```bash
python eval/merge_results.py
python eval/analyze_results.py "eval_results/summary_*.csv"
python eval/reliability.py eval_results/merged_<timestamp>.csv
python eval/sensitivity.py eval_results/merged_<timestamp>.csv
```
