# symPO 실험 프로토콜 (Academic Protocol)

> 목적: `eval2.txt`에 정의된 **학술적·논리적으로 타당한 결과**를 재현 가능한 형태로 산출한다.
> 이 문서는 "무엇을 바꾸고(독립변수), 무엇을 고정하고(통제변수), 무엇을 측정하고(종속변수), 어떻게 통계 처리하는가"에 대한 실행 가능한 프로토콜이다.
> 현재화 기준일: 2026-04-27

---

## 0. 실험 전 체크 — eval2.txt 스펙 구현 현황

| 스펙 § | 요구사항 | 구현 파일 | 상태 |
|---|---|---|---|
| §3.A | 자동 지표 13개 (Faithfulness, SR, MECE, Granularity, Planning, Gini, Feasibility, Buffer, Convergence, CommEff, SupRatio, Turns, Cost) | `metrics.py` | ✅ |
| §3.A1 | SR τ 민감도(0.45/0.55/0.65) | `metrics.py::calc_success_rate` | ✅ |
| §3.A3 | MECE θ 캘리브레이션 | `metrics.py::calc_mece_score` | ✅ |
| §3.B1 | Planning Score + Skill Coverage + Length-norm | `metrics.py::calc_planning_score` | ✅ |
| §3.B3 | Schedule Feasibility 엄격 부등호(`>`) | `metrics.py::calc_schedule_feasibility` | ✅ |
| §3.C1 | Convergence 3축 (Buffer + Jaccard + Centroid) | `metrics.py::calc_convergence` | ✅ |
| §4 | 교차 Judge (Gemini × Claude), κ/ρ 보고 | `eval/llm_judge.py` | ✅ (`--cross-judge`) |
| §4.4 | 조건별 평가 차원 (C0=S만, C1=S+A, C2~=S+A+D) | `eval/llm_judge.py` | ✅ |
| §5 | AutoScore 가중 종합 + N/A 재정규화 | `metrics.py::compute_autoscore` → `eval_results/autoscore_recompute.py` | ✅ |
| §6 RQ1 | 5 조건(C0~C4) 러너 | `eval/experiment_runner.py --conditions` | ✅ |
| §6 RQ1-H | 하네스 on/off 러너 | `eval/experiment_runner.py --harness both` | ✅ |
| §6 통계 | Mann-Whitney U + Cliff's δ + Holm-Bonferroni | `eval/analyze_results.py` | ✅ |
| §6 재현성 | ICC, κ, ρ | `eval/reliability.py` | ✅ |
| §7 Failure Mode | F1~F5 탐지 | `metrics.py` + `analyze_results.py` | ✅ |
| §8 | 민감도 분석 | `eval/sensitivity.py` | ✅ |
| §9 산출물 | CSV, snapshot, 리포트, 그림 | `experiment_runner.py`, `generate_figures.py` | ✅ |
| MCP/Tool trace | phase별 tool boundary 추적 | `orchestration/mcp_tool_layer.py`, `api.py`, `frontend/index.html`, `mcp_server.py` | ✅ |

**주의**:
- §6 RQ2 (R0~R4 RAG 전략 비교)는 runner 설정과 기존 실험 리포트가 존재한다. 신규 본실험으로 재현할 때는 조건 조합을 수동 지정해야 한다 (§4.3 본문 참조).
- §8 전문가 평가(Human Evaluation)는 선택 사항이며 시니어 PM 3인 Best-Worst Scaling 수집 예정

### 0-1. 실행 전 Prerequisites (fresh agent 필수 체크)

아래가 모두 충족되어야 프로토콜의 어떤 조건이라도 완전 재현 가능.

1. **의존성**: `pip install -r requirements.txt` 완료
2. **`.env`**: `cp .env.example .env` 후 아래 키들 값 채움
   - `LLM_BACKEND=gemini` (기본 본 실험 백엔드)
   - `GOOGLE_API_KEY=<유효키>`
   - `RUNNER_ID=<본인 고유값>` — 팀 실험 시 결과 파일 충돌 방지용 (예: `kim`, `lee`)
   - `EMBEDDING_MODEL=all-MiniLM-L6-v2`, `MAX_DEBATE_ROUNDS=3`, `OUTPUT_DIR=./generated`
3. **조건별 추가 요건**
   - **S2_8b_*** 조건 실행 시 `GEMMA4_API_URL=<Colab ngrok URL>` 필수. Colab 노트북에서 Gemma-4 8B 서버가 떠 있고 ngrok 터널이 활성 상태여야 함. ngrok URL 만료 시 재발급·갱신. 없으면 S2_8b_* 조건은 실패 — frontier 비교만 수행하거나 `--backend mock`으로 드라이런.
   - **`qwen-api` 백엔드** 실행 시 `QWEN_API_URL` 필수. 필요하면 `QWEN_API_MODEL`, `QWEN_MAX_CONTEXT`, `QWEN_ENABLE_THINKING=false`도 명시한다.
   - **`--cross-judge`** 사용 시 Claude API 키 필요 (`ANTHROPIC_API_KEY`).
   - **`R3_graph`, `R4_agentic`, `R4_llm_rerank`** 는 추가 LLM 호출이 있어 비용·시간 2배 감안.
4. **드라이런 권장**: 첫 실행 전 `LLM_BACKEND=mock`으로 한 조건만 `--runs 1` 돌려서 파이프라인 정상 동작 확인.
5. **`eval_results/`** 디렉터리 비워두기 — 이전 실험 CSV와 섞이면 분석 단계에서 결측 매트릭스 해석이 꼬임.

### 0-2. MCP/tool-call trace 호환성

현재 런타임은 기존 실험 경로를 유지하면서 각 LangGraph phase 호출을 MCP-compatible tool boundary로 기록한다. 이 변경은 프롬프트, 모델 호출 순서, 배정 로직을 바꾸지 않기 때문에 기존 실험 조건의 의미는 유지된다.

- trace 필드: `WBSState.mcp_tool_trace`
- FastAPI SSE 전달 필드: `mcp_tool_trace`
- 프론트 표시 위치: 토론 로그 탭의 `Agent Tool Calls` 패널
- 외부 MCP catalog: `mcp_server.py`의 `orchestration_tool_catalog()`, `orchestration_phase_plan()`

주의: LLM 호출 자체를 외부 MCP IPC로 넘기는 단계는 아직 기본 경로가 아니다. 모든 LLM phase를 외부 MCP client 경유로 바꾸면 latency와 state 직렬화가 바뀌므로, 본 실험 재현 전에는 autoscore/compaction/snapshot inspection 같은 결정론적 tool부터 단계적으로 전환한다.

---

## 1. 실험 변수 정의 (Variables)

### 1-1. 독립변수 (IV · 이 값을 바꿔가며 실험)

| IV | 값 | 조작 방법 | 근거 |
|---|---|---|---|
| **condition** | C0, C1, C2, C3, C4 | `--conditions C2_1round ...` | eval2.txt §6 RQ1 |
| **harness_enabled** | H0(off), H1(on) | `--harness off|on|both` | eval2.txt §6 RQ1-H |
| **rag_strategy** | R0, R1(vanilla), R2(hybrid), R3(graph), R4(agentic) | runner 내 `rag_strategy` 필드 | eval2.txt §6 RQ2 |
| **backend** | mock, gemini, openai, gemma4, gemma4-api, qwen-api, anthropic, ollama | `--backend gemini` / `.env` | 외적 타당성 보강 |
| **cross_judge** | off, on | `--cross-judge` | self-preference 편향 통제 |

### 1-2. 통제변수 (Fixed · 비교 유효성을 위해 **반드시 고정**)

| 통제 | 값 | 이유 |
|---|---|---|
| PRD 입력 | 동일 파일 (예: `samples/prd_pmarket.json`) | 입력 편차 제거 |
| 팀원 프로필 | 동일 팀원 목록 | 배정 가능 집합 고정 |
| LLM 모델 | 동일 백엔드·모델·temperature | 생성 능력 편차 제거 |
| EMBEDDING_MODEL | `all-MiniLM-L6-v2` | 임베딩 축 고정 |
| MAX_DEBATE_ROUNDS | 조건별 명세 (C2=1, C3=3) 외에는 3 | RQ1 목적 외 라운드 변동 차단 |
| Random seed | LLM은 seed 없이 temp=0, 그 외 결정적 로직 | 가능한 최대한의 재현성 |
| 실행자 CPU·네트워크 | 기록 (runner_id에 자동 포함) | 실행 환경 추적 |

### 1-3. 종속변수 (DV · 측정 대상)

`eval2.txt §9`의 CSV 컬럼 전체가 DV이다. 핵심 요약:

- **자동 지표 13개** → `AutoScore` 종합 (§5)
- **LLM Judge 3차원** (Structure / Assignment / Debate) → `JudgeScore`
- **하네스 관측** → `role_drift_detected_count`, `harness_caught_exceptions`
- **Tool boundary trace** → `mcp_tool_trace` (현재 AutoScore에는 미합산)
- **Failure Mode 카운트** → F1~F5 분류 (§7)

---

## 2. 실험 설계 (Design)

### 2-1. 연구 질문별 실험 설계

```
RQ1  (condition IV) :  C0 / C1 / C2 / C3 / C4    ← between-condition, 각 N회 반복
RQ1-H(harness IV)   :  H0 / H1 (C3 고정)         ← between-harness, 각 N회 반복
RQ2  (rag IV)       :  R0 / R1 / R2 / R3 / R4    ← between-rag, 각 N회 반복, 2라운드 토론 고정
```

### 2-2. 반복 수 (N)

| 목적 | N | 근거 |
|---|---|---|
| 탐색 실험 (pilot) | 3 | 최소 분산 확인 |
| 본 실험 (권장) | **5** | eval2.txt §6 최소 요건 |
| 논문 수록 | **10** | ICC(2,k) 안정화 + bootstrap CI 신뢰도 |

> ⚠️ **Cohen's d로 N을 줄이지 말 것**. 효과크기와 재현성은 별개다 (eval2.txt §8 Internal Validity).

### 2-3. 조건 조합 매트릭스 (최소 본 실험)

| 실험 | condition | harness | rag | backend | N | 총 run |
|---|---|---|---|---|---|---|
| RQ1 | C0~C4 | H1 (default) | R1 (기본 RAG) | gemini | 5 | **25** |
| RQ1-H | C3 | H0, H1 | R1 | gemini | 5 | **10** |
| RQ2 | C3 고정 (2라운드) | H1 | R0~R4 | gemini | 5 | **25** |
| 교차심사 | C3 | H1 | R1 | gemini | 3 | **3** (cross-judge on) |

**최소 권장 러닝**: 63 runs × 조건 (하지만 팀원 병렬 분담 시 1인당 15~20 runs)

---

## 3. 실행 절차 (How to Run)

### 3-1. 환경 고정

```bash
cp .env.example .env
```

`.env`에 반드시 세팅:
```
LLM_BACKEND=gemini
GOOGLE_API_KEY=<유효키>
RUNNER_ID=<본인 고유 ID>       # 팀 실험 시 필수
EMBEDDING_MODEL=all-MiniLM-L6-v2
MAX_DEBATE_ROUNDS=3
OUTPUT_DIR=./generated
```

### 3-2. 명령 — 팀원 분담 예시

| 담당 | 명령 |
|---|---|
| 팀원 A | `python eval/experiment_runner.py --backend gemini --conditions C0_llm_only C1_with_assign --runs 5` |
| 팀원 B | `python eval/experiment_runner.py --backend gemini --conditions C2_1round --runs 5` |
| 팀원 C | `python eval/experiment_runner.py --backend gemini --conditions C3_3rounds --harness both --runs 5` *(RQ1-H 포함)* |
| 팀원 D | `python eval/experiment_runner.py --backend gemini --conditions C4_with_disc --runs 5` |
| 취합자 | `python eval/experiment_runner.py --backend gemini --conditions C3_3rounds --cross-judge --runs 3` |

### 3-3. 생성 파일 규칙

각 runner가 생성:
```
eval_results/summary_<backend>_<runner_id>_<timestamp>.csv
eval_results/experiment_<backend>_<runner_id>_<timestamp>.json
eval_results/wbs_snapshot_<cond>_r<i>_<backend>_<runner_id>_<timestamp>.json
```

- 파일명에 `runner_id`가 포함되어 팀원 간 충돌 없음
- CSV 첫 컬럼 `runner_id`로 필터 가능

---

## 4. 통계 분석 (Post-processing)

### 4-1. 취합 단계

```bash
# 팀원들이 공유한 summary_*.csv를 eval_results/에 모은 뒤
python eval/merge_results.py
# → eval_results/merged_<ts>.csv + 팀원×조건 매트릭스 출력으로 결측 확인
```

### 4-2. 본 분석

```bash
python eval/analyze_results.py eval_results/merged_<ts>.csv
# 산출:
#   - 조건별 mean ± std
#   - Mann-Whitney U (α=0.05)
#   - Cliff's δ (효과크기)
#   - Holm-Bonferroni 보정 p값
#   - Failure Mode 빈도
#   - 최적 조건 표시
# 저장: eval_results/analysis_report_<ts>.md
```

### 4-3. 재현성 리포트 (RQ1/RQ1-H 필수)

```bash
python eval/reliability.py eval_results/merged_<ts>.csv
# 산출:
#   - ICC(2,k) : 반복 실행 간 일관성
#   - Cohen's κ : 교차 Judge 범주 합의 (cross-judge run만)
#   - Spearman ρ : 교차 Judge 점수 상관
# 저장: eval_results/reliability_report.json
```

> κ < 0.4 또는 ρ < 0.5 이면 해당 Judge 차원은 **"판정 불가"**로 표기 (§4).

### 4-4. 민감도 분석 (논문 수록용)

```bash
python eval/sensitivity.py eval_results/merged_<ts>.csv
# 산출:
#   - SR τ (0.45/0.55/0.65)에서 AutoScore ranking 뒤집힘 여부
#   - MECE θ 캘리브레이션 결과
#   - AutoScore 가중치 ±0.10 grid search
# 저장: eval_results/sensitivity_report.md
```

### 4-5. Figure 생성

```bash
python eval/generate_figures.py eval_results/merged_<ts>.csv
# 저장: eval_results/figures/fig1~5.png
```

---

## 5. 학술적 타당성 체크리스트

각 실험 세트를 "결과"로 보고하기 전에 아래를 **전부** 확인한다.

### 5-1. Internal Validity (내적 타당성)

- [ ] 조건당 **N >= 5** 반복 실행했는가?
- [ ] `runner_id`로 실행자별 편차를 분리 가능한가?
- [ ] `AutoScore` N/A 재정규화가 조건별로 올바른가? (C0 = PlanScore·Gini·Feasibility 제외)
- [ ] LLM 비결정성 통제를 **Cohen's d가 아니라 반복 + CI + ICC**로 보고했는가?
- [ ] `LLM_BACKEND`, `EMBEDDING_MODEL`, `MAX_DEBATE_ROUNDS`가 모든 run에서 동일한가?

### 5-2. External Validity (외적 타당성)

- [ ] PRD·팀원 프로필이 모든 조건에서 동일한가?
- [ ] `harness_enabled`가 **독립변수**로서 층화(stratification) 되었는가? (병합 분석 금지)
- [ ] 백엔드 혼합 금지 — `summary_mock_*.csv`와 `summary_gemini_*.csv`는 **절대 합쳐 분석하지 않는다**
- [ ] 결론에 "단일 도메인·단일 모델 패밀리·팀 6인 한정" 한계를 명시했는가?

### 5-3. Construct Validity (구성 타당성)

- [ ] Planning Score와 함께 `planning_skill_coverage`, `planning_length_norm`를 보고하고 Spearman ρ를 계산했는가?
- [ ] LLM Judge는 **Gemini × Claude 교차** 결과를 사용했는가? (단일 Judge 결과는 부가 자료로만)
- [ ] Convergence는 3축(buffer, assignment, utterance) 통합 값으로 보고했는가?
- [ ] Structure/Assignment/Debate 각 차원의 Cohen's κ와 ρ가 임계값을 통과했는가?

### 5-4. Statistical Validity (통계 타당성)

- [ ] n < 30 조건에 **Mann-Whitney U**를 썼는가? (t-test 금지)
- [ ] **Cliff's δ** 또는 rank-biserial r을 함께 보고했는가?
- [ ] 다중 비교에 **Holm-Bonferroni** 또는 BH-FDR를 적용했는가? (단순 Bonferroni 금지)
- [ ] p값만이 아닌 효과크기 + 95% bootstrap CI를 함께 보고했는가?

### 5-5. Failure Mode 기록

- [ ] F1 (L1 < 3) 발생한 run을 제거 또는 별도 보고했는가?
- [ ] F2 (Role Hallucination) 빈도를 `role_drift_detected_count`로 집계했는가?
- [ ] F3~F5도 조건별 카운트로 테이블에 포함했는가?

---

## 6. 반-패턴 (하면 안 되는 것)

| ❌ 금지 | 이유 |
|---|---|
| 조건당 1~2회만 돌리고 "평균"으로 결론 | LLM 비결정성을 전혀 통제 못함 |
| mock + gemini 결과 섞어서 같은 차트에 | 모델별 생성 분포가 다름, 구성 타당성 파괴 |
| H0와 H1 합쳐서 "C3 평균"으로 보고 | 하네스가 독립변수인데 층화 안 함 |
| Judge가 Gemini 단독인데 "LLM Judge 결과" 로 단정 | self-preference bias 미통제 (Panickssery 2024) |
| 단일 τ 값(0.55)만으로 SR 결론 | 임계값 민감도 분석 누락 |
| p값만 보고, 효과크기 누락 | "통계적 유의"와 "실질적 유의"는 다름 |
| `MAX_DEBATE_ROUNDS`를 run별로 바꿈 | 통제변수 오염 → RQ1 비교 무효 |
| Bonferroni로 n<30에 엄격 보정 | 검정력 고갈, 대부분 차이를 놓침 |

---

## 7. 재현 가능한 보고 템플릿 (논문/리포트 §결과 섹션)

```markdown
## 결과

### 실험 설정
- 백엔드: Gemini 3.1 Pro Preview (temperature=0)
- 임베딩: sentence-transformers/all-MiniLM-L6-v2
- PRD: P마켓 커머스 v1.0 / 팀원 6명 /
- 반복: 각 조건 N=5 (총 63 runs)
- 참여자: <runner_id 리스트>

### RQ1 (Ablation)
조건 | AutoScore (mean±std) | Judge Overall | N | 유의(vs C1)
---|---|---|---|---
C0 | 0.xx ± 0.yy | 0.xx | 5 | Mann-Whitney U=w, p=q, Cliff's δ=d
...

- Holm-Bonferroni 보정 후 유의 조건: C2, C3
- 최고 AutoScore: C3 (0.xx)
- 재현성: ICC(2,k) = 0.xx
- 민감도: SR τ ∈ {0.45, 0.55, 0.65}에서 ranking 변화 없음 (robust)

### RQ1-H (하네스)
- 기본 가설(H0≈H1) 지지 여부: (p=0.xx, δ=0.xx)
- H1의 role_drift_detected_count = N건, harness_caught_exceptions = N건

### Failure Modes
F1: X회 / F2: Y회 / F3: Z회 ...

### Judge Reliability (Cross-judge subset)
- Structure: κ=0.xx, ρ=0.xx → 유효
- Assignment: κ=0.xx, ρ=0.xx → 유효
- Debate: κ=0.xx, ρ=0.xx → 유효/판정불가

### 한계
단일 도메인 · 단일 모델 패밀리 · 팀 6인 고정. 일반화 전 §8 외적 타당성 점검 필요.
```

---

## 8. 확장 실험 매트릭스 (10종 — 전부 구현 완료)

eval2.txt §6 (RQ1/RQ1-H/RQ2)를 넘어 추가 계획된 실험들. **2026-04-27 현재 10종 모두 `eval/experiment_runner.py`의 `CONDITIONS` dict에 조건 ID로 구현·배선 완료**. 새 에이전트도 아래 표의 조건 ID를 `--conditions`에 넘기면 즉시 실행 가능.

### 8-1. 실험 ↔ 조건 ID 매트릭스 (현재 구현 기준)

| # | 분류 | 실험명 | 독립변수 | 조건 ID | 상태 |
|---|---|---|---|---|---|
| E1 | structure | 토론 라운드 (1/3/5) | `max_rounds` | `C2_1round` / `C3_3rounds` / `C5_5rounds` | ✅ |
| E2 | structure | 모델급 × 프롬프팅 (Frontier/8B × Single/Chaining/CoT) | `model_class`, `prompting_strategy` | `S2_frontier_{single,chaining,cot}`, `S2_8b_{single,chaining,cot}` | ✅ |
| E3 | RAG | PRD 컨텍스트 범위 (요약/상세/상세+회의) | `prd_variant`, `use_meeting` | `R1_prd_summary` / `R1_prd_detailed` / `R1_prd_detailed_meeting` | ✅ |
| E4 | RAG | 프로필 메타데이터(이력서+eDISC) | `use_disc` | `C1_with_assign` (이력서만) vs `C4_with_disc` (이력서+eDISC) | ✅ |
| E5 | RAG | Hybrid Search (Vector / Vector+BM25) | `rag_strategy` | `R1_vanilla` / `R2_hybrid` | ✅ |
| E6 | RAG | LLM Reranking on/off | `rag_strategy` | `R1_vanilla` (off) vs `R4_llm_rerank` (on) | ✅ |
| E7 | RAG | 회의록 일정 언급 유무 | `meeting_variant` | `R5_meeting_regular` / `R5_meeting_no_schedule` | ✅ |
| E8 | agent | 비판 에이전트(Critic) 도입 | `critic_enabled` | `A1_single` / `A1_critic` | ✅ |
| E9 | agent | Veto(거부권) 메커니즘 | `veto_enabled` | `A2_no_veto` / `A2_with_veto` | ✅ |
| E10 | agent | 페르소나 강성(방어적/공격적) | `persona_strictness` | `A3_defensive` / `A3_aggressive` | ✅ |

**확장 실험 확장 매트릭스 (최소 본 실험 N=5 기준)**

| 실험 | 조건 | 총 run | 러너 명령 |
|---|---|---|---|
| E1 | C2, C3, C5 | 15 | `--conditions C2_1round C3_3rounds C5_5rounds --runs 5` |
| E2 | S2 × 6 | 30 | `--conditions S2_frontier_single S2_frontier_chaining S2_frontier_cot S2_8b_single S2_8b_chaining S2_8b_cot --runs 5` |
| E3 | R1 PRD 3종 | 15 | `--conditions R1_prd_summary R1_prd_detailed R1_prd_detailed_meeting --runs 5` |
| E5/E6 | R1, R2, R4-rerank | 15 | `--conditions R1_vanilla R2_hybrid R4_llm_rerank --runs 5` |
| E7 | R5 × 2 | 10 | `--conditions R5_meeting_regular R5_meeting_no_schedule --runs 5` |
| E8 | A1 × 2 | 10 | `--conditions A1_single A1_critic --runs 5` |
| E9 | A2 × 2 | 10 | `--conditions A2_no_veto A2_with_veto --runs 5` |
| E10 | A3 × 2 | 10 | `--conditions A3_defensive A3_aggressive --runs 5` |

### 8-2. 실험 조합 시 주의 — 실제 교란 변수

**⚠️ `R1_prd_detailed_meeting`은 2변수 동시 조작** (PRD 변종 + 회의록 on). R1(PRD 범위) vs R5(회의록 변종)와 단순 비교 불가. 이 조건은 **E3+E7 상호작용 탐색용**으로만 사용하고, 주효과 분석에서는 제외.

**⚠️ `S2_8b_*` 조건은 별도 백엔드 전제조건** — `§0-1` Prerequisites 참조. `GEMMA4_API_URL` 미설정 시 실행 실패.

**⚠️ `C4_with_disc`는 구현 조건이지 효과 입증이 아님** — 최신 context metadata pilot에서는 M_both가 M_resume보다 높지 않았다. 보고서에서는 eDISC를 "성능 향상 입증"이 아니라 "성향 메타데이터 반영 구조 및 보조 맥락"으로 표현한다.

**⚠️ 확장 실험 해석 시 주효과 분리**: 각 실험은 단일 IV만 변동시키는 between-group 설계. 여러 IV를 한 실험에 섞지 말 것 — 교차 효과를 보려면 별도 factorial design이 필요.

### 8-3. 확장된 독립변수 — 프로토콜 반영

기존 §1-1 IV 표에 다음을 추가한 것으로 간주한다 (전부 구현됨).

| IV | 값 | 조작 방법 (state 필드 / 조건 ID) |
|---|---|---|
| `max_rounds` | 1, 3, 5 | `C2_1round`/`C3_3rounds`/`C5_5rounds` |
| `prompting_strategy` | single, chaining, cot | state `prompting_strategy` — `S2_*_{single,chaining,cot}` |
| `prd_variant` | summary, detailed_full | state `prd_variant` — `R1_prd_*` |
| `use_disc` | off, on | CONDITIONS `use_disc` — `C1_with_assign` vs `C4_with_disc` |
| `rag_strategy` | vanilla, hybrid, graph, agentic, llm_rerank | CONDITIONS `rag_strategy` — `R0~R4` |
| `meeting_variant` | regular, no_schedule | CONDITIONS `meeting_variant` — `R5_meeting_*` |
| `critic_enabled` | off, on | state `critic_enabled` — `A1_single` vs `A1_critic` |
| `veto_enabled` | off, on | state `veto_enabled` — `A2_no_veto` vs `A2_with_veto` |
| `persona_strictness` | defensive, neutral, aggressive | state `persona_strictness` — `A3_*` |
| `model_class` | frontier, 8b | state `model_class` + 자동 backend override — `S2_*` |
| `backend` | mock, gemini, openai, gemma4, gemma4-api, qwen-api, anthropic, ollama | CLI `--backend` |

### 8-4. 확장 실험의 통계·타당성 원칙

§5 체크리스트는 그대로 적용. 추가 주의점:

- **E2 (모델급) 해석**: gemma4(8B) < gemini(Frontier)는 예상대로이며, 연구 가치는 **"경량 모델 + 프롬프트 전략으로 격차를 얼마나 좁히는가"**. 단순 "8B가 나쁘다"는 발견이 아니므로 **Δ품질 / Δ비용 평면에서 Pareto front**를 보고할 것.
- **E6 (Reranking)**: Reranker 자체 비용(토큰+지연)을 `est_tokens`·`elapsed_sec`에 반드시 포함하여 순이득을 판정.
- **E8 (Critic)**: Critic 도입 시 토론 라운드 수·토큰 증가가 필연적이므로 **"라운드 매칭"** 비교 (C3 vs C3+critic) 또는 **총 토큰 매칭** 비교 중 하나를 사전 선택하여 보고.
- **E9 (Veto)**: Veto 발동 빈도가 0이면 독립변수가 작동하지 않은 것. **Veto 발동율 (veto_triggered / total_rounds)** 을 부가 지표로 함께 보고.
- **E10 (Persona 강성)**: 공격적 페르소나는 `supervisor_intervention_ratio`를 인위적으로 올릴 수 있음. **Debate Score와 Convergence를 반드시 동시 보고**하여 "개입율이 높은 게 나쁜 것인지 구조상 필연인지" 구분.
- **E1/E2/E3 등 여러 IV를 한 실험에 섞지 말 것** — 교차 효과를 보려면 별도 factorial design 실험을 설계하고, 그 전까지는 항상 단일 IV만 변동시키는 between-group 실험으로 수행.

---

## 9. 참고

- 스펙 원본: `eval2.txt` (특히 §6 실험 설계, §8 타당성, §9 산출물)
- 팀 배포 절차: `TEAM_EXPERIMENT.md`
- 구현 파일 색인: `eval/` 디렉터리 + `metrics.py`
- 결과 저장 디렉터리: `eval_results/`
- 확장 실험 코드 구현: Phase A~D 우선순위에 따라 별도 PR로 추가
