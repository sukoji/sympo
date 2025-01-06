# WBS / Task Manager Backbone Model Comparison

## 범위

이 폴더는 현재 완료된 백본 교체 실험을 모델별, 에이전트별로 비교한다. 모든 완료 실험은 `C3_3rounds` 조건을 유지하고, 지정된 agent만 해당 모델로 교체했다.
`Gemma4-26B baseline`은 모든 에이전트를 Gemma4-26B로 둔 기준선이다.

## 완료 실험 요약

| Agent | Model | Runs | Judge Overall | Structure | Assignment | Debate | Auto | Tasks | Time min | Workload Gini |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | Gemma4-26B baseline | 3 | 0.775 | 0.723 | 0.733 | 0.917 | 0.811 | 35.0 | 8.0 | 0.325 |
| WBS Gen | Qwen 4B | 3 | 0.756 | 0.613 | 0.803 | 0.920 | 0.799 | 64.3 | 11.9 | 0.162 |
| WBS Gen | EXAONE 7.8B | 3 | 0.637 | 0.493 | 0.593 | 0.927 | 0.752 | 18.3 | 10.8 | 0.496 |
| Task Manager | Qwen 4B | 3 | 0.759 | 0.740 | 0.653 | 0.937 | 0.735 | 42.3 | 18.4 | 0.490 |
| Task Manager | EXAONE 7.8B | 3 | 0.760 | 0.730 | 0.667 | 0.937 | 0.737 | 42.7 | 18.8 | 0.403 |

## 아직 미완료/대기 조건

| Agent | Model | 상태 |
|---|---|---|
| WBS Gen | Qwen LoRA | 8083/8084 서버 정상화 후 실행 예정 |
| Task Manager | Qwen LoRA | 8083/8084 서버 정상화 후 실행 예정 |

## 해석

- WBS Gen 교체에서는 **Qwen 4B 평균 overall 0.756**가 **EXAONE 7.8B 평균 overall 0.637**보다 높다.
- 그러나 Gemma4-26B baseline overall **0.775**과 비교하면 WBS Gen 교체 모델 둘 다 최종 overall을 넘지 못했다.
- EXAONE WBS는 평균 task 수가 **18.3**로 작고, 실험 로그상 `assigned_role` 타입 오류와 L1/L2 구조 부족이 반복됐다.
- Qwen 4B WBS는 평균 task 수가 **64.3**로 크지만 run 간 분산이 크다. 생성량은 많지만 안정성 해석이 필요하다.
- Task Manager-Qwen 4B는 평균 overall **0.759**로 수치상 양호하다. 다만 병목은 assignment 평균 **0.653**이며, judge reason상 skill-fit과 workload imbalance가 반복적으로 지적된다.
- Task Manager-EXAONE 7.8B는 overall **0.760**로 Qwen 4B와 유사하지만, baseline보다 느리고 assignment도 baseline보다 낮다.

## Figure

- `fig1_agent_model_completion_matrix.png`: agent-model 완료/대기 매트릭스
- `fig2_wbsgen_model_score_comparison.png`: Gemma baseline, Qwen 4B, EXAONE 7.8B의 WBS Gen 비교
- `fig3_taskmanager_model_score_comparison.png`: Gemma baseline, Qwen 4B, EXAONE 7.8B의 Task Manager 비교
- `fig4_qwen4b_role_swap_comparison.png`: Qwen 4B를 WBS Gen에 넣었을 때와 Task Manager에 넣었을 때 비교
- `fig5_efficiency_tradeoff_scatter.png`: 평균 성능, 소요 시간, task 규모 비교

## Raw

원본 summary CSV는 `raw/` 폴더에 복사했다. 집계 테이블은 `aggregate_model_comparison.csv`에 저장했다.
