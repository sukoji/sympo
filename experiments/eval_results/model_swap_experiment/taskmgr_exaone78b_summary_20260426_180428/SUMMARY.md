# Task Manager EXAONE 7.8B Experiment

## 실험 조건

- 조건: `H_taskmgr_exaone78b`
- 변경 지점: Task Manager만 `EXAONE-3.5-7.8B-Instruct-Q4_K_M.gguf`
- 나머지 에이전트: Gemma4-26B baseline
- 반복: 3 runs

## 평균 결과

| Metric | Mean |
|---|---:|
| Judge Overall | 0.760 |
| Structure | 0.730 |
| Assignment | 0.667 |
| Debate | 0.937 |
| Auto Score | 0.737 |
| Time | 18.8 min/run |
| Tokens | 71,169 / run |
| Cost | 0.0053 USD / run |
| Tasks | 42.7 / run |
| Workload Gini | 0.403 |

## Run별 결과

| Run | Overall | Structure | Assignment | Debate | Auto | Time sec | Tasks | Workload Gini |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.708 | 0.72 | 0.52 | 0.95 | 0.727 | 1119.53 | 42 | 0.466 |
| 2 | 0.785 | 0.72 | 0.79 | 0.88 | 0.758 | 1151.27 | 44 | 0.297 |
| 3 | 0.787 | 0.75 | 0.69 | 0.98 | 0.727 | 1113.81 | 42 | 0.447 |

## 품질 이슈

- Run 2 raw output에서 JSON comment와 임시 assignee 표현이 감지되어 Task Manager의 schema-following 안정성이 약하다.
- Run 3은 `Standard JSON Parsing Error`와 `L1-04` 하위 L2 부족 경고가 있었다.
- Run 3 기준 L3 task 배정 분포는 `MBR-FC03=10`, `MBR-60D5=9`, `MBR-1934=4`, `MBR-B9BE=3`, `MBR-9707=2`, `MBR-1E4F=0`으로 DevOps가 idle이다.
- 점수상 Task Manager Qwen 4B와 유사하지만, Gemma4-26B baseline보다 느리고 전체 성능이 낮다.

## 연결 산출물

- 전체 baseline 포함 비교: `../wbs_taskmgr_baseline_blue_comparison_20260426/`
- Raw summary: `summary_qwen-api_taskmgr_exaone78b_20260426_180428.csv`
- Raw experiment JSON: `experiment_qwen-api_taskmgr_exaone78b_20260426_180428.json`
- Run snapshots: `wbs_snapshot_H_taskmgr_exaone78b_r*.json`
