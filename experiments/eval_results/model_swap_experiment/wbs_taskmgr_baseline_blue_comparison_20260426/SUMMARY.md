# Baseline-Inclusive Model Swap Comparison

## 요약 테이블

| Condition | Runs | Judge Overall | Structure | Assignment | Debate | Auto | Time min | Tokens | Cost USD | Tasks | Workload Gini |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemma4-26B baseline | 3 | 0.775 | 0.723 | 0.733 | 0.917 | 0.811 | 8.0 | 26170 | 0.0020 | 35.0 | 0.325 |
| WBS Gen Qwen 4B | 3 | 0.756 | 0.613 | 0.803 | 0.920 | 0.799 | 11.9 | 49492 | 0.0037 | 64.3 | 0.162 |
| WBS Gen EXAONE 7.8B | 3 | 0.637 | 0.493 | 0.593 | 0.927 | 0.752 | 10.8 | 41546 | 0.0031 | 18.3 | 0.496 |
| Task Manager Qwen 4B | 3 | 0.759 | 0.740 | 0.653 | 0.937 | 0.735 | 18.4 | 73127 | 0.0055 | 42.3 | 0.490 |
| Task Manager EXAONE 7.8B | 3 | 0.760 | 0.730 | 0.667 | 0.937 | 0.737 | 18.8 | 71169 | 0.0053 | 42.7 | 0.403 |

## 해석 포인트

- 기준선은 `Gemma4-26B baseline`으로 표기했다. 이는 C3 조건에서 모든 에이전트를 Gemma4-26B로 둔 결과다.
- 현재 최고 judge overall은 `Gemma4-26B baseline`의 `0.775`이다.
- 현재 가장 빠른 완료 조건은 `Gemma4-26B baseline`의 `8.0 min/run`이다.
- Task Manager EXAONE은 점수가 나오더라도 raw output의 JSON 주석, 미허용 assignee, 일부 L3 미배정 같은 schema-following 문제를 같이 해석해야 한다.
- `Debate` 점수는 교체하지 않은 토론 에이전트의 영향도 포함하므로, 백본 선택 근거에서는 `Overall`, `Assignment`, `Time`, `Token/Cost`, `Workload Gini`를 함께 보는 편이 더 타당하다.

## Figure

- `fig1_blue_baseline_score_comparison.png`: baseline 포함 품질 점수 비교
- `fig2_blue_runtime_resource_comparison.png`: 수행 시간, 토큰, 비용, 산출 task 수 비교
- `fig3_blue_efficiency_scatter.png`: 성능-속도-산출규모 효율 비교
- `fig4_blue_assignment_vs_workload_gini.png`: 배정 점수와 업무 편중도 비교

원본 CSV와 EXAONE Task Manager raw JSON/snapshot은 `raw/`에 복사했다.
