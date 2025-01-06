# Interpretation Figure Set

이 폴더는 기존 comparison figure를 발표 해석 단위로 다시 분해한 산출물입니다.

## 추천 사용 순서

1. `figA_llm_only_vs_c3_system_effect.png` — LLM-only 대비 시스템 구성의 효과를 먼저 보여줍니다.
2. `figB_gemma26_round_score_runtime.png` — Gemma-4-26B 내부에서 C3가 왜 선택되는지 설명합니다.
3. `figC_c3_model_selection_quality_signals.png` — C3 조건에서 모델별 품질 신호를 분리해 비교합니다.
4. `figD_c3_quality_cost_tradeoff.png` — C3에서 Gemma-4-26B가 local 운영점으로 타당함을 보여줍니다.
5. `figE_model_round_heatmap.png` 또는 `figF_gemma26_stepwise_gain_tradeoff.png` — 보조 근거로 사용합니다.

## 산출물

- `figA_llm_only_vs_c3_system_effect.png`: LLM-only(C0)와 선택 시스템(C3) 비교
- `figB_gemma26_round_score_runtime.png`: Gemma-4-26B 단일 모델 라운드별 점수/시간
- `figC_c3_model_selection_quality_signals.png`: C3에서 모델별 AutoScore와 LLM Judge 분리 비교
- `figD_c3_quality_cost_tradeoff.png`: C3 모델별 품질-비용 tradeoff
- `figE_model_round_heatmap.png`: 모델 x 라운드 패턴 heatmap
- `figF_gemma26_stepwise_gain_tradeoff.png`: Gemma-4-26B 단계별 증가분과 시간 비용

시각 규칙: Gemma-4-26B는 초록/굵은 테두리, 선택 C3는 금색, C0 baseline은 빗금, Gemini API는 빗금 또는 hollow marker로 표시했습니다.
