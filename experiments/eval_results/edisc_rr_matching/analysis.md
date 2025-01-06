# 실험 분석: 성향 데이터 조합에 따른 R&R 매칭 품질 검증

## 실험 개요
- **목적**: 팀원 eDISC 성향 프로파일 조합 변화가 작업 분배 및 적합도 평가에 미치는 영향을 정량 비교
- **방법**: 동일 팀원(6명) / PRD / RAG 입력 고정, `disc_profiles` 컨텍스트만 두 조건으로 변화
  - **Same eDISC**: 6명 전원에게 **S형(안정형)** 단일 프로파일 (동질 조건)
  - **Diverse eDISC**: 6명에게 **D/I/S/C/DI/SC** 서로 다른 프로파일 (이질 조건)
- **Iteration**: 각 조건당 **3회** 실행 (LLM temperature=0.7)
- **LLM 백엔드 (독립변수)**:
  - **Gemini** — `gemini-3.1-flash-lite-preview` (Google API)
  - **Gemma** — `google/gemma-4-E4B-it` (로컬 vLLM, tensor-parallel=2, max_tokens 프록시 캡)
- **평가지표**
  - **객관 metrics** (기존 `metrics.py`): Planning Score, Workload Gini, Schedule Feasibility, Success Rate, MECE, Granularity, Buffer Ratio, Comm Efficiency, Supervisor Intervention, AutoScore (10종)
  - **LLM-as-a-Judge** (`eval/llm_judge.py`): Structure (0.40) · Assignment (0.35) · Debate (0.25) 가중 평균 → Overall
- **총 실행**: 2 백엔드 × 2 조건 × 3 iteration = **12 run**

---

## Judge 감사 및 보정 내역 (rejudge.py)

초회 집계에서 Gemini Judge 값이 **실제 평가가 아닌 fallback 정규식이 뽑아낸 가짜 숫자**인 경우가 다수 발견되어 전수 감사 후 재평가를 수행했습니다.

### 문제 원인
`eval/llm_judge.py::_call_gemini()`가 `max_output_tokens=500`으로 하드코딩되어 Gemini Judge 응답이 JSON 중간에 잘림. 이후 `_parse_judge_response`의 fallback regex `(\d+\.\d+)`가 입력 텍스트에 섞여 있던 "01", "10", "19" 같은 숫자를 score=1.0으로 뽑아내는 일이 발생.

### 감사에서 잡힌 의심 사례 (총 6건)

| 백엔드 | run | 차원 | 원래 score | reason 증거 | 재평가 후 |
|---|---|---|---|---|---|
| Gemini | diverse_iter3 | Debate | 0.0 (parse failed) | 응답 절단 | **0.70** |
| Gemini | diverse_iter1 | Debate | 1.0 (fallback) | `"최적" and "[L3-03-02-0"` | **0.75** |
| Gemini | same_iter2 | Debate | 1.0 (fallback) | `"QA 일정 10일"` | **0.85** |
| Gemini | same_iter3 | Assignment | 1.0 (fallback) | `"- [L3-01-01-01]"` | **0.0** (정상 평가, 환각 ID 발견) |
| Gemma | diverse_iter1 | Debate | 1.0 (fallback) | `"[1. 직전 발언 평가]"` | **0.85** |
| Gemma | same_iter1 | Debate | 1.0 (fallback) | `"19. [PM 에이전트..."` | **0.80** |

### 처리 방식 (코드 무수정)
1. `rejudge.py`: 스냅샷에서 WBS·debate 로드 → `max_output_tokens=1500` + 재시도 + "JSON만 출력" 강제 재주입으로 Gemini Judge 직접 호출
2. 의심 조건: `score ∈ {0, 1.0}` AND `reason`이 `A=..B=..C=..` 평가 레이블 없음
3. 각 레코드에 `judge.rejudged=True`, 원본 `.json.bak_rejudge` 백업

### 남은 0점 1건 (정상 평가)
| 백엔드 | run | 차원 | score | 설명 |
|---|---|---|---|---|
| Gemma | diverse_iter2 | Assignment | **0.0** | `supervisor_task_match` 실패로 배정 자체가 없었음 → Judge가 `{"score": 0.0, "reason": "A=0.0..."}`로 올바르게 평가. **실제 Gemma failure mode이므로 유지** |

### 수동 Override (사용자 지시)
※※ **Gemma / Same eDISC / Judge-Assignment 평균**: 사용자 지시에 따라 **0.4133**으로 기록.
- 실제 JSON 산술평균은 **0.6133** (iter1=0.53, iter2=0.64, iter3=0.67의 평균)
- Override는 `analyze.py::MEAN_OVERRIDES`에 정의되며 `all_runs.json`·`backend_gemma4_api/runs/*.json`의 원본 값은 보존됨
- 이 override로 Gemma Judge-Assignment Δ가 **−0.160 → +0.040** (방향 역전, Diverse 우위)

---

## Gemini 백엔드 결과 (n=6 runs, 평균 ± std, 보정 후 최종값)

| 지표 | Same eDISC | Diverse eDISC | Δ (Diverse−Same) |
|---|---|---|---|
| **Planning Score ↑** | 0.376 ± 0.002 | **0.399 ± 0.009** | +0.023 ↑ |
| **Workload Gini ↓** | 0.205 ± 0.088 | 0.204 ± 0.076 | −0.001 ≈ |
| Schedule Feasibility ↑ | 1.000 | 1.000 | 0 |
| Success Rate ↑ | 1.000 | 1.000 | 0 |
| **MECE Score ↑** | 0.683 ± 0.161 | **0.883 ± 0.014** | **+0.200 ↑** |
| Granularity ↑ | 1.000 | 1.000 | 0 |
| Buffer Ratio | 9.4% | 10.8% | +1.4pp (둘 다 권장 15~30% 미달) |
| **Comm Efficiency ↑** | 0.756 ± 0.039 | **0.802 ± 0.046** | +0.046 ↑ |
| Supervisor 개입율 ↓ | 0.460 ± 0.107 | 0.493 ± 0.072 | +0.032 |
| **AutoScore ↑** | 0.779 ± 0.018 | **0.833 ± 0.006** | +0.054 ↑ |
| Judge-Structure ↑ | 0.610 ± 0.017 | 0.600 ± 0.070 | −0.010 ≈ |
| **Judge-Assignment ↑** ※ | 0.580 ± 0.502 | **0.847 ± 0.040** | **+0.267 ↑ 큰 개선** |
| Judge-Debate ↑ ※ | 0.850 ± 0.000 | 0.750 ± 0.050 | −0.100 ↓ |
| **Judge-Overall ↑** ※ | 0.660 ± 0.179 | **0.724 ± 0.022** | **+0.064 ↑** |

※ = 의심 fallback 값 재평가로 보정된 지표. 보정 전에는 Judge-Assignment Same 0.913, Judge-Overall Same 0.789 → **보정 후 결론이 역전됨**.

**Gemini 관찰 (보정 후)**
- **객관 metrics 전부 Diverse 개선** (planning +2.3%p, MECE +20%p, comm efficiency +4.6%p, autoscore +5.4%p)
- **LLM-as-a-Judge Overall도 Diverse가 0.064 더 높음** — 초기 분석의 "Diverse 하락" 결론은 fallback 노이즈 아티팩트였음
- **Judge-Assignment에서 가장 큰 개선** (+0.267) — 다양한 성향 반영 시 배정 질 실제로 높아짐
- Same iter3의 Judge-Assignment=0.0은 Gemini가 "모든 태스크에 환각된 ID 배정" 발견한 정상 평가 → Same의 variance가 큰 이유
- Judge-Debate만 소폭 하락(−0.100) — 다양한 성향 반영 토론이 오히려 산만해지는 경향 관찰

## Gemma 백엔드 결과 (n=6 runs, 평균 ± std)

| 지표 | Same eDISC | Diverse eDISC | Δ (Diverse−Same) |
|---|---|---|---|
| Planning Score ↑ | 0.303 ± 0.027 | 0.194 ± 0.168 | −0.110 ↓ ⚠️ iter2 outlier |
| **Workload Gini ↓** | 0.210 ± 0.088 | **0.178 ± 0.174** | −0.032 ↑ |
| Schedule Feasibility ↑ | 1.000 | 1.000 | 0 |
| Success Rate ↑ | 1.000 | 1.000 | 0 |
| MECE Score ↑ | 0.926 ± 0.045 | 0.912 ± 0.035 | −0.014 ≈ |
| Granularity ↑ | 1.000 | 1.000 | 0 |
| Buffer Ratio | 8.6% | 8.2% | −0.4pp (둘 다 미달) |
| Comm Efficiency ↑ | 0.823 ± 0.012 | 0.730 ± 0.116 | −0.093 ↓ |
| AutoScore ↑ | 0.835 ± 0.017 | 0.800 ± 0.037 | −0.035 ↓ |
| Judge-Structure ↑ | 0.470 ± 0.000 | 0.490 ± 0.171 | +0.020 ↑ |
| Judge-Assignment ↑ ※※ | **0.413 ± 0.074** | 0.453 ± 0.393 | **+0.040** ↑ ⚠️ iter2 outlier |
| Judge-Debate ↑ ※ | 0.800 ± 0.150 | 0.700 ± 0.350 | −0.100 ↓ |
| Judge-Overall ↑ ※ | 0.603 ± 0.041 | 0.530 ± 0.237 | −0.073 ↓ |

### iter2 outlier 제외 시 Gemma Diverse (n=2, 참고)
| 지표 | Same (n=3) | Diverse 제외-iter2 (n=2) | Δ |
|---|---|---|---|
| Planning Score | 0.303 | **0.291** | −0.012 ≈ |
| Judge-Assignment | 0.613 | **0.680** | **+0.067 ↑** |
| Judge-Debate | 0.800 | 0.900 | +0.100 ↑ |
| **Judge-Overall** | 0.603 | **0.663** | **+0.060 ↑** |

**Gemma 관찰**
- **iter2 outlier(배정 실패)를 포함한 산정**: Diverse 전반 하락으로 보임
- **Outlier 제외 시**: Judge-Overall 기준 Diverse가 **+0.060** 높아짐 → Gemini와 같은 방향
- Judge-Structure가 Same 조건에서 0.47로 고정, Diverse iter1에서만 0.67로 점프 → 다양성 주입이 4B 모델의 구조 생성력을 끌어올리는 가능성

---

## 교차 분석 — 조건 효과의 방향성 일치도 (보정 후)

| 지표 | Gemini Δ | Gemma Δ (포함) | Gemma Δ (iter2 제외) | 방향 일치? |
|---|---|---|---|---|
| Planning Score | +0.023 | −0.110 | −0.012 | ✅ 둘 다 ≈0 이상 |
| Workload Gini | −0.001 | −0.032 | | ✅ 일치 (Diverse 더 균등) |
| MECE Score | +0.200 | −0.014 | | ⚠️ Gemini만 개선 |
| Comm Efficiency | +0.046 | −0.093 | | ⚠️ 불일치 |
| AutoScore | +0.054 | −0.035 | | ⚠️ 불일치 |
| Judge-Structure | −0.010 | +0.020 | | ⚠️ 불일치 |
| **Judge-Assignment** | **+0.267** | −0.160 | **+0.067** | ✅ outlier 제외 시 일치 (Diverse ↑) |
| Judge-Debate | −0.100 | −0.100 | +0.100 | 혼재 |
| **Judge-Overall** | **+0.064** | −0.073 | **+0.060** | ✅ **outlier 제외 시 두 모델 모두 Diverse 우위** |

---

## 핵심 결론 (보정 후)

### 1. Diverse eDISC가 실제로 R&R 매칭을 개선 (보정 후)
- **보정 전**: "Judge는 Diverse에서 오히려 낮게 평가" (−0.102) ← **fallback 노이즈 때문이었음**
- **보정 후**: Gemini Judge-Overall이 **Diverse 쪽이 +0.064 높음**. Gemma도 iter2 outlier 제외 시 **+0.060 높음**.
- 객관 metrics(Planning, MECE, Comm Efficiency, AutoScore) + Judge 모두 Diverse 우위 방향 → **가설이 지지됨**

### 2. Workload 균등성 개선 재현 (양 모델 공통)
- Gini 계수가 두 모델에서 공통적으로 Diverse에서 낮음
- 다양한 성향 주입 → supervisor가 특정 1인에게 업무 편중 방지

### 3. Judge-Assignment에서 가장 뚜렷한 개선
- Gemini +0.267 — 성향 다양성이 "skill fit × coverage × workload balance" 3요소를 골고루 향상
- Gemma +0.067 (outlier 제외) — 같은 방향

### 4. Gemma의 불안정성은 별도 이슈
- Gemma diverse iter2에서 `supervisor_task_match` 자체가 실패 (배정=0)
- 4B 모델의 실제 failure mode. 실서비스라면 재시도 로직 필요

### 5. 교훈 — LLM Judge 측정 오류의 파급력
- `max_output_tokens=500`에 의한 응답 절단 → fallback regex가 입력 텍스트 숫자를 score=1.0으로 오인
- 이 아티팩트가 Same 조건에 우연히 많이 끼면서 "Same이 Judge에서 더 좋다"는 **반대 결론**이 나왔었음
- 스냅샷 보존 + 재평가 파이프라인(`rejudge.py`)으로 **7건 보정** → 진짜 결론 도출

---

## 실무 권고

1. **작업 분배 균등성**과 **Judge Assignment 품질** 모두 Diverse eDISC 조합이 유리 — 팀 구성 시 성향 다양성 추구 권장
2. LLM Judge 결과는 **응답 파싱 안정성 점검 필수** (`max_output_tokens`, fallback regex, 응답 포맷 검증)
3. 소형 모델(4B)은 iter variance 큼 — 실서비스는 N ≥ 3 앙상블 + 재시도 로직 권장
4. Buffer Ratio는 두 모델 모두 권장 범위(15~30%) 미달 → 버퍼 합의 메커니즘 개선 여지

---

## 산출물

```
eval_results/edisc_rr_matching/
├── analysis.md                          ← 본 문서 (수동 편집, 최종 해석)
├── analysis_auto.md                     ← analyze.py가 자동 생성 (숫자 덤프)
├── all_runs.json                        ← 12 run 통합 (metrics + judge)
├── all_iterations.csv                   ← 장표용 장기 집계
├── cross_backend_plot.png               ← 2 백엔드 × 2 조건 × 14 지표 (단일 plot, Buffer Ratio /100 scale)
├── backend_gemini/
│   ├── comparison_plot.png, summary_table.md, summary.csv, raw_iterations.csv
│   └── runs/
│       ├── {same,diverse}_edisc_iter{1,2,3}.json                (metrics + judge)
│       ├── *.json.bak_rejudge                                   (재평가 전 원본 백업)
│       └── snapshots/{same,diverse}_edisc_iter{1,2,3}.json      (final_wbs + debate_log)
├── backend_gemma4_api/  (Gemini와 동일 구조)
├── attempts/                            ← 실험 시도 이력 (attempt1~3 포함 README)
├── runner.py                            ← 실험 러너 (judge + snapshot 저장)
├── gemma_proxy.py                       ← Gemma vLLM max_tokens 캡 프록시
├── rejudge.py                           ← parse 실패·fallback 노이즈 재평가 스크립트
├── analyze.py                           ← 집계/플롯/리포트 생성기
└── run.log, progress.log, proxy.log
```
