# symPO 팀 실험 가이드

여러 팀원이 각자의 머신에서 Ablation Study를 돌리고, 결과를 한 사람이 모아 분석하기 위한 절차입니다.

현재화 기준일: 2026-04-27

---

## 0. 사전 준비 (각자 1회)

```bash
# 1) 저장소 클론
git clone <repo-url> sympo && cd sympo

# 2) 파이썬 환경
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3) 환경변수 설정
cp .env.example .env
# .env 파일을 열어 다음을 채운다
#   - LLM_BACKEND (mock | gemini | openai | gemma4 | gemma4-api | qwen-api | anthropic | ollama)
#   - GOOGLE_API_KEY (gemini 사용 시)
#   - RUNNER_ID=<본인 식별자>   ← 결과 파일 충돌 방지용 (예: kim, lee, park)
```

> **`RUNNER_ID`가 왜 중요한가?**
> 결과 파일명은 `summary_<backend>_<runner_id>_<timestamp>.csv` 형식이라,
> 팀원 간 식별자가 같으면 한 사람이 모았을 때 구별이 안 됩니다. **반드시 서로 다른 값**으로 설정하세요.

---

## 1. 실험 실행 — 담당 조건만

실험 매트릭스는 `eval/experiment_runner.py`에 정의되어 있으며, 조건은 다음과 같습니다.

### 코어 ablation (RQ1)
| 조건             | 설명                        |
| ---------------- | --------------------------- |
| `C0_llm_only`    | LLM 단독 (배정·토론 없음)   |
| `C1_with_assign` | +Supervisor 배정            |
| `C2_1round`      | +1라운드 토론               |
| `C3_3rounds`     | +3라운드 토론 (기본)        |
| `C4_with_disc`   | +eDISC 페르소나             |
| `C5_5rounds`     | +5라운드 토론 (S1-D 감도분석) |

### RAG 전략 (RQ2)
| 조건 | 설명 |
| --- | --- |
| `R0_no_rag` | RAG 없음 |
| `R1_vanilla` | FAISS Dense |
| `R2_hybrid` | BM25 + Dense RRF |
| `R3_graph` | 엔티티 그래프 |
| `R4_agentic` | 멀티홉 반복 검색 |
| `R4_llm_rerank` | Dense 후보 + LLM 재정렬 |

### 입력 변형 (RQ2-bis)
| 조건 | 설명 |
| --- | --- |
| `R1_prd_summary` | PRD 요약본 |
| `R1_prd_detailed` | PRD 확장 상세본 |
| `R1_prd_detailed_meeting` | 확장 PRD + 회의록 |
| `R5_meeting_regular` | 회의록(일정 포함) |
| `R5_meeting_no_schedule` | 회의록(일정 제거) |

### 에이전트 설계 (RQ3)
| 조건 | 설명 |
| --- | --- |
| `A1_single` / `A1_critic` | 단일 리뷰 vs Critic 교차 심사 |
| `A2_no_veto` / `A2_with_veto` | 거부권 없음 vs 단일 [VETO] 허용 |
| `A3_defensive` / `A3_aggressive` | 방어적 vs 공격적 페르소나 |

### 모델·프롬프팅 (RQ4)
| 조건 | 설명 |
| --- | --- |
| `S2_frontier_single` / `_chaining` / `_cot` | Gemini × 3가지 프롬프팅 |
| `S2_8b_single` / `_chaining` / `_cot` | Gemma4 (8B) × 3가지 프롬프팅 |

**팀 분담 예시 (4명)**

| 팀원 | 담당 조건            | 실행 명령                                                        |
| ---- | -------------------- | ----------------------------------------------------------------- |
| A    | C0, C1               | `python eval/experiment_runner.py --backend gemini --conditions C0_llm_only C1_with_assign --runs 3` |
| B    | C2                   | `python eval/experiment_runner.py --backend gemini --conditions C2_1round --runs 3` |
| C    | C3                   | `python eval/experiment_runner.py --backend gemini --conditions C3_3rounds --runs 3` |
| D    | C4                   | `python eval/experiment_runner.py --backend gemini --conditions C4_with_disc --runs 3` |

**RQ1-H (하네스 효과) 실험도 돌려야 할 때**

```bash
# 한 조건에 하네스 on/off 둘 다 돌리기 (조건당 runs × 2회)
python eval/experiment_runner.py --backend gemini --conditions C3_3rounds --harness both --runs 5
```

**교차 심사 (Claude로 검증)**

```bash
python eval/experiment_runner.py --backend gemini --conditions C3_3rounds --judge-method geval --runs 3
python eval/experiment_runner.py --backend gemini --conditions C3_3rounds --cross-judge --runs 3
```

### 실행 결과로 생기는 파일

`eval_results/` 디렉터리에 다음이 저장됩니다.

- `summary_<backend>_<runner_id>_<timestamp>.csv`  ← **이 파일만 모으면 됩니다**
- `experiment_<backend>_<runner_id>_<timestamp>.json` (전체 원본)
- `wbs_snapshot_<condition>_r<run_id>_<backend>_<runner_id>_<timestamp>.json`

---

## 2. 팀원 → 취합 담당자에게 결과 전달

팀원 각자는 `eval_results/summary_*.csv` 파일 **한 개 이상**을 공유 드라이브/슬랙에 업로드합니다.

취합 담당자는 받은 CSV들을 본인 레포의 `eval_results/` 아래에 모아둡니다.

```
eval_results/
├── summary_gemini_kim_20260419_101523.csv
├── summary_gemini_lee_20260419_110040.csv
├── summary_gemini_park_20260419_121511.csv
└── summary_gemini_anon_20260419_140021.csv
```

---

## 3. 병합 & 분석 (취합 담당자)

```bash
# 3-1. 병합 — eval_results/ 내 모든 summary_*.csv 자동 탐색
python eval/merge_results.py

# glob 명시도 가능
python eval/merge_results.py "eval_results/summary_gemini_*.csv"

# 결과: eval_results/merged_<timestamp>.csv + 콘솔에 팀원×조건 매트릭스 출력
```

`merge_results.py`는 팀원별·조건별 실행 수 매트릭스를 출력해 **결측 조합**을 바로 보여줍니다.

```bash
# 3-2. 분석 리포트 생성 (병합된 CSV 그대로 넣기)
python eval/analyze_results.py eval_results/merged_<timestamp>.csv

# 또는 병합 없이 glob으로 한 번에
python eval/analyze_results.py "eval_results/summary_gemini_*.csv"
```

결과: `eval_results/analysis_report_<timestamp>.md` — 조건별 평균·표준편차, 최적 조건 강조, Holm-Bonferroni 보정 p값 등.

---

## 4. 체크리스트 — 실험 나가기 전 확인

- [ ] `.env`의 `LLM_BACKEND`가 팀 합의된 백엔드(예: `gemini`)와 동일한가?
- [ ] `.env`의 `RUNNER_ID`가 팀 내 고유값인가?
- [ ] `GOOGLE_API_KEY` (또는 해당 백엔드 키)가 유효한가? → `LLM_BACKEND=mock`으로 드라이런 1회 추천
- [ ] `eval_results/` 디렉터리를 미리 비워두었는가? (과거 실험과 섞이지 않게)
- [ ] 조건당 **최소 3회, 권장 5회** 반복할 계획인가? (통계 검정 유효성)
- [ ] Qwen/Gemma API 백엔드를 쓰는 경우 `QWEN_API_URL` 또는 `GEMMA4_API_URL`이 현재 살아 있는가?

---

## 5. 자주 묻는 질문

**Q. 같은 조건을 여러 명이 돌려도 되나?**
A. 네. `runner_id`로 구분되며, 병합 시 합쳐서 샘플 수가 늘어납니다 (통계 파워↑).

**Q. 중간에 실패하면?**
A. 완료된 run 들은 이미 CSV에 append된 상태가 아니라 실행 종료 시점에 한 번에 저장됩니다. 그래서 **조건 단위로 재실행**해야 안전합니다. `--conditions`로 특정 조건만 돌리세요.

**Q. mock 백엔드로 돌린 결과도 합쳐도 되나?**
A. 합치는 건 가능하지만 **분석 리포트는 백엔드별로 분리**하는 게 맞습니다 (mock vs gemini는 같은 메트릭 축에서 비교 불가). `summary_mock_*.csv`와 `summary_gemini_*.csv`는 따로 merge → 따로 analyze 하세요.

**Q. Claude 교차심사(`--cross-judge`)는 누가 돌리나?**
A. 비용이 2배 드니 한 명(또는 취합 담당자)이 조건당 3run 정도만 추가로 돌리면 됩니다.

**Q. MCP/tool 사용 평가도 CSV에 들어가나?**
A. 현재 오케스트레이션은 `mcp_tool_trace`에 내부 MCP-style tool boundary를 남깁니다. 다만 기본 `summary_*.csv`의 AutoScore에는 아직 직접 합산되지 않으므로, tool usage 자체를 평가하려면 snapshot/API의 trace를 별도로 집계해야 합니다.
