# WBS Model Quality: Qualitative Sample Pack

비교 대상은 C3(3-round debate) 조건입니다. 각 모델은 C3 평균 LLM Judge 점수에 가장 가까운 저장 스냅샷을 대표 샘플로 자동 선택했습니다.

## Generated Figures

- `figQ1_c3_quality_signals.png`: C3 모델별 LLM Judge, AutoScore, 대표 샘플 점수
- `figQ2_sample_structure_profile.png`: 대표 샘플의 WBS 계층 구조, 배정 커버리지, L2 gap
- `figQ3_sample_assignment_heatmap.png`: 대표 샘플의 팀원명 기준 L3 작업량 heatmap
- `figQ4_sample_wbs_table.png`: 모델별 WBS 샘플을 표 형태로 포맷팅
- `figQ5_sample_debate_evidence.png`: 토론 로그의 참여자·task reference·risk/test 근거량
- `figQ6_assignment_reason_evidence.png`: 태스크·담당자·배정/재배정 근거 로그를 한 세트로 정리
- `figQ7_local_first_selection_view.png`: Gemma-4-26B 선택 논리를 강조한 발표용 figure

## Representative Samples

| Model | Snapshot | Sample Judge | C3 Judge Mean | C3 AutoScore | Tasks | L1/L2/L3 | Assigned L3 | Workload Gini | Buffer % |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemma-4B | `wbs_snapshot_C3_3rounds_r3_gemma4-api_piai_20260421_022430.json` | 0.613 | 0.644 | 0.806 | 77 | 6/18/53 | 100% | 0.123 | 6.7% |
| Qwen3-14B | `wbs_snapshot_C3_3rounds_r2_qwen-api_qwen_ablation_20260423_011311.json` | 0.569 | 0.541 | 0.826 | 26 | 3/6/17 | 100% | 0.317 | 7.5% |
| Gemma-4-26B | `wbs_snapshot_C3_3rounds_r3_qwen-api_gemma26_ablation_20260423_145328.json` | 0.738 | 0.729 | 0.811 | 33 | 3/8/22 | 100% | 0.377 | 7.6% |
| Gemini API | `wbs_snapshot_C3_3rounds_r2_gemini_gemini_ablation_20260422_210852.json` | 0.723 | 0.684 | 0.840 | 35 | 5/10/20 | 100% | 0.199 | 11.0% |

## Evaluation Metrics Used

- LLM Judge overall: Structure 0.40, Assignment 0.35, Debate 0.25 active-dimension weighting from the existing comparison pipeline.
- AutoScore v2: quality, allocation, orchestration deterministic score already present in `summary_4backbones.csv`.
- Qualitative sample metrics: task hierarchy counts, missing L2 children, L3 assignment coverage, workload Gini, top-assignee share, buffer ratio, dependency density, active debate agents, task-ID references, and risk/test term counts.

Team member display rule: figures use names recovered from Task Manager assignment logs or known sample-member mappings. When a snapshot uses model-generated MBR codes, raw codes are not shown; the display falls back to role-based labels.

## Per-Model Qualitative Notes

### Gemma-4B

- Structure/Assignment/Debate sample scores: 0.300 / 0.800 / 0.850
- WBS shape: 77 tasks, 6 L1, 18 L2, 53 L3, 0 L2 nodes without L3 children.
- R&R signal: 100% of L3 tasks assigned, 6 assignees, workload Gini 0.123.
- Debate trace: 5 active non-PM agents, 197 task-ID references, 113 risk/test terms.
- structure judge note: Judge reason text is truncated in the saved snapshot; numeric score is used.
- assignment judge note: Judge reason text is truncated in the saved snapshot; numeric score is used.
- debate judge note: Judge reason text is truncated in the saved snapshot; numeric score is used.

### Qwen3-14B

- Structure/Assignment/Debate sample scores: 0.470 / 0.730 / 0.500
- WBS shape: 26 tasks, 3 L1, 6 L2, 17 L3, 0 L2 nodes without L3 children.
- R&R signal: 100% of L3 tasks assigned, 6 assignees, workload Gini 0.317.
- Debate trace: 5 active non-PM agents, 68 task-ID references, 88 risk/test terms.
- structure judge note: Judge reason text is truncated in the saved snapshot; numeric score is used.
- assignment judge note: Judge reason text is truncated in the saved snapshot; numeric score is used.
- debate judge note: Judge reason text is truncated in the saved snapshot; numeric score is used.

### Gemma-4-26B

- Structure/Assignment/Debate sample scores: 0.700 / 0.700 / 0.850
- WBS shape: 33 tasks, 3 L1, 8 L2, 22 L3, 0 L2 nodes without L3 children.
- R&R signal: 100% of L3 tasks assigned, 6 assignees, workload Gini 0.377.
- Debate trace: 5 active non-PM agents, 92 task-ID references, 56 risk/test terms.
- structure judge note: A=0.5 B=0.6 C=1.0; 3 L1s with mostly good L3 coverage, all L3s 1-10d but buffer <10%,
titles are highly specific.
- assignment judge note: A=0.8 B=0.3 C=1.0; Good skill match, full coverage. Severe workload imbalance (32d vs
3d) and unlisted member assigned.
- debate judge note: A=0.8 B=0.8 C=1.0 D=0.8; 3 roles contributed concrete analysis with task IDs. Good
role consistency and convergence.

### Gemini API

- Structure/Assignment/Debate sample scores: 0.670 / 0.870 / 0.600
- WBS shape: 35 tasks, 5 L1, 10 L2, 20 L3, 0 L2 nodes without L3 children.
- R&R signal: 100% of L3 tasks assigned, 6 assignees, workload Gini 0.199.
- Debate trace: 4 active non-PM agents, 76 task-ID references, 101 risk/test terms.
- structure judge note: Judge reason text is truncated in the saved snapshot; numeric score is used.
- assignment judge note: Judge reason text is truncated in the saved snapshot; numeric score is used.
- debate judge note: Judge reason text is truncated in the saved snapshot; numeric score is used.

## Suggested Slide Order

1. Fig Q1 to anchor the model-level ranking.
2. Fig Q4 to show concrete WBS output differences in table form.
3. Fig Q2 and Fig Q3 to explain structure and R&R quality.
4. Fig Q6 to connect assignment decisions with log evidence.
5. Fig Q7 when presenting the Gemma-4-26B selection rationale.
6. Fig Q5 as aggregate debate trace evidence.
