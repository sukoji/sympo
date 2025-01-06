# LLM-only Limitation Sample: C0 vs C3

## Source snapshots

- C0: `gemma26_ablation/snapshots/wbs_snapshot_C0_llm_only_r1_qwen-api_gemma26_ablation_20260423_135834.json`
- C3: `gemma26_ablation/snapshots/wbs_snapshot_C3_3rounds_r2_qwen-api_gemma26_ablation_20260423_144446.json`

## Fair comparison principle

- C0는 LLM-only WBS draft 조건이므로 assignment/debate 차원을 결함으로 감점하지 않는다.
- C0에서 assignment/debate judge가 N/A인 것은 산출물 실패가 아니라 실험 조건의 범위 차이다.
- 따라서 슬라이드 주장은 "LLM-only가 WBS를 못 만든다"가 아니라 "C3가 WBS Structure rubric 안에서 estimate/buffer realism을 개선했다"로 둔다.

## Comparable WBS-quality evidence

| 항목 | C0 LLM-only | C3 SYMPo |
|---|---:|---:|
| Total tasks | 34 | 36 |
| L3 tasks | 23 | 25 |
| Dependency links | 31 | 33 |
| MECE score | 1.0 | 1.0 |
| Granularity fitness | 1.0 | 1.0 |
| Schedule feasibility | 1.0 | 1.0 |
| Buffer ratio | 0.0% | 12.6% |
| Structure judge | 0.630 | 0.770 |

## Structure judge sub-score evidence

| 항목 | C0 LLM-only | C3 SYMPo |
|---|---:|---:|
| A: hierarchy / depth | 0.5 | 0.5 |
| B: estimate / buffer realism | 0.4 | 0.8 |
| C: task specificity | 1.0 | 1.0 |

## Output-level interpretation

- C0도 계층형 WBS 초안, 역할명, 기간, 의존성을 생성하며 rule-based WBS quality 지표는 높다.
- 유의미한 차이는 Structure judge 내부의 estimate/buffer realism이다. C0는 no buffer로 B=0.4, C3는 12.6% buffer로 B=0.8이다.
- Assignment/debate는 C0 조건 밖이므로 이 슬라이드의 성능 주장 근거로 쓰지 않는다.
- 슬라이드 메시지: **LLM-only도 WBS draft는 가능하지만, C3는 동일 backbone에서 일정/버퍼 현실성이 더 높은 WBS로 개선된다.**
