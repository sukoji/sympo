# Supervisor 경량화 독립 실험 (Colab Standalone)

현재화 메모(2026-04-27): 이 문서는 Colab 단독 supervisor 교체 파일럿 가이드입니다. 프로젝트 본체의 현재 실행/평가 기준은 루트 `README.md`, `PROJECT_OVERVIEW.md`, `eval_results/EVALUATION_FRAMEWORK.md`를 우선합니다. HuggingFace 모델 repo ID는 실행 시점의 공식 ID로 확인해야 합니다.

symPO 프로젝트의 supervisor 역할(주로 `task_match`) 을 오픈소스 LLM으로 교체했을 때의
**정확도 손실 vs 속도 증가** trade-off 를 Colab 단독으로 측정한다.

본 실험은 프로젝트 코어(`agents/`, `orchestration/`, `eval/`) 를 **전혀 수정하지 않는다**.
Colab 노트북 한 파일 + 벤치 케이스 한 파일로 완결된다.

## 목적

"supervisor 전체를 frontier(gemini)로 돌리는 기존 구성" 대비
"supervisor 만 오픈소스 8B급 모델로 대체하는 구성" 이 품질 유지하며 속도를 얻는지 확인.

## 실행 순서

1. Colab > Runtime > Change runtime type > **T4 GPU** 선택 (또는 A100)
2. `supervisor_oss_eval.ipynb` 업로드
3. (선택) `benchmark_cases.json` 도 함께 업로드 — 없어도 노트북 인라인 fallback 이 동작
4. 위에서부터 순차 실행
5. `GOOGLE_API_KEY` 입력 (baseline 용, getpass 프롬프트)
6. (선택) `HF_TOKEN` 입력 (Gemma / Llama 등 gated 모델용)

## 후보 모델 (2026-04 기준 최신 시리즈, 4-bit NF4, T4 16GB 여유)

| 모델 | 파라미터 | 4-bit VRAM | 역할 |
|---|---|---|---|
| `Qwen/Qwen3.5-8B-Instruct` | 8B | ~5 GB | 메인 후보 (Qwen 최신) |
| `Qwen/Qwen3.5-4B-Instruct` | 4B | ~2.5 GB | Qwen 경량 변종 |
| `google/gemma-4-E4B-it` | ~4B eff | ~3 GB | Gemma 4 경량 (프로젝트 기본) |
| `google/gemma-4-E2B-it` | ~2B eff | ~2 GB | Gemma 4 초경량 (symPO 현 배선) |
| `microsoft/Phi-4-mini-instruct` | 3.8B | ~2.3 GB | 타 패밀리 대조 |

**주의** — HuggingFace 의 정확한 repo ID 는 릴리스 시점에 따라 변경되므로
노트북 실행 시 `OSS_MODELS` 리스트에서 `from_pretrained` 가 404 로 실패하면
해당 모델의 **HF 페이지에서 공식 repo 이름으로 교체**하십시오
(예: `Qwen/Qwen3.5-8B-Instruct` → 실제 공개명이 `Qwen/Qwen3-Next-8B-Instruct` 등일 수 있음).

T4 16GB 에서는 Qwen3.5-14B, Gemma-4-12B 도 4-bit 로 돌아가지만
KV cache + 긴 supervisor 프롬프트 때문에 OOM 위험 → 기본 후보에서는 제외.
A100 을 쓸 수 있다면 노트북의 `OSS_MODELS` 리스트에 추가하면 된다.
Llama-4-Scout-17B-16E-Instruct (MoE, `llm_config.py:304` 에 이미 배선) 는 A100 한정.

## 지표

| 지표 | 설명 |
|---|---|
| `json_parse_rate` | 응답에서 JSON 블록이 파싱 가능한 비율 |
| `called_jaccard` | `called_agents` 집합의 Jaccard (vs Gemini gold) |
| `alloc_jaccard` | `allocations[task_id]` 멤버 집합의 평균 Jaccard |
| `latency_sec` | 응답 당 평균 생성 시간 |
| `quality` | 위 3개 품질 지표 평균 (0~1) |
| `speedup` | `gemini_latency / model_latency` |
| `quality_loss` | `(gemini_quality − model_quality) / gemini_quality` |
| `efficiency` | `speedup / quality_loss` |

## 판정 기준 (사전 등록)

| efficiency | quality_loss | 해석 |
|---|---|---|
| ≥ 2 | ≤ 0.10 | 교체 합당 |
| 1~2 | ≤ 0.15 | 조건부 (low-stakes only) |
| < 1 | — | 비합당 |

## 산출물

노트북 마지막 셀에서 자동 저장:

- `results_raw.csv` — case × model 단위 raw
- `results_summary.csv` — model 집계
- `fig_quality_vs_latency.png` — Pareto 플롯

## 한계

- 본 실험은 supervisor 의 `task_match` 단계만 검증. `mediate` / `finalize` 는 별도 케이스 추가 필요.
- Gemini 출력을 gold 로 삼음 → self-preference 편향 가능. 교차 Judge (Claude) 는 본 노트북 범위 밖.
- 벤치 케이스 3~5개 수준의 파일럿. 논문용 통계 결론엔 N 확장 필요 (`benchmark_cases.json` 에 케이스 추가).
