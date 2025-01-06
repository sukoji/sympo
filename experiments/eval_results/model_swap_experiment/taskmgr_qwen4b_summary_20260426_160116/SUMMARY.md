# Qwen 4B Task Manager Backbone Swap Summary

## 실험 조건

- 조건: `C3_3rounds`
- 교체 지점: `Task Manager`만 Qwen 4B로 교체
- 유지 지점: WBS Gen, 토론 에이전트, 최종 정리, judge는 기존 설정 유지
- 모델 endpoint: `http://127.0.0.1:8082`
- raw CSV: `summary_qwen-api_taskmgr_qwen4b_20260426_160116.csv`

## 핵심 결과

- 평균 LLM Judge Overall: **0.759**
- 평균 Structure / Assignment / Debate: **0.740 / 0.653 / 0.937**
- 평균 Auto Score: **0.735**
- 평균 WBS 규모: **42.3 tasks**
- 평균 소요 시간: **18.4 min/run**
- 평균 workload gini: **0.490**

해석: Qwen 4B Task Manager는 JSON 구조와 팀원 ID 복사는 정상적으로 수행했다. 다만 judge reason 기준으로 assignment는 skill-fit과 workload balance가 병목이다. Debate 점수는 높지만 이는 후속 Gemma 기반 토론 단계의 보정 효과가 섞여 있으므로, Task Manager 단독 우수성으로 과해석하면 안 된다.

## Run별 지표

| Run | Tasks | Time min | Structure | Assignment | Debate | Overall | Auto | Workload Gini |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 42 | 18.4 | 0.75 | 0.63 | 0.85 | 0.733 | 0.726 | 0.524 |
| 2 | 43 | 18.4 | 0.77 | 0.63 | 0.98 | 0.773 | 0.736 | 0.514 |
| 3 | 42 | 18.3 | 0.70 | 0.70 | 0.98 | 0.770 | 0.742 | 0.433 |

## 산출물 품질 메모

- 모든 run에서 `bad assignee`는 발견되지 않음. 즉 Qwen 4B가 허용된 팀원 ID를 그대로 사용했다.
- 모든 L3 task에 담당자가 배정됨.
- judge가 반복적으로 지적한 약점은 QA/검증 성격 task의 skill mismatch와 업무량 불균형이다.
- WBS 생성은 Gemma 26B가 담당했으므로, 이 실험은 WBS 생성력 비교가 아니라 Task Manager 배정 판단력 비교로 해석해야 한다.

## Figure

- `fig1_qwen4b_taskmanager_score_profile.png`: judge score profile
- `fig2_qwen4b_taskmanager_run_dashboard.png`: run별 안정성 대시보드
- `fig3_qwen4b_taskmanager_quality_matrix.png`: 품질 지표 heatmap
- `fig4_qwen4b_taskmanager_workload_distribution.png`: Run 3 업무량 분포

## Snapshot

- `wbs_snapshot_H_taskmgr_qwen4b_r1_qwen-api_taskmgr_qwen4b_20260426_152234.json`
- `wbs_snapshot_H_taskmgr_qwen4b_r2_qwen-api_taskmgr_qwen4b_20260426_154201.json`
- `wbs_snapshot_H_taskmgr_qwen4b_r3_qwen-api_taskmgr_qwen4b_20260426_160116.json`
