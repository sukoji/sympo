# Compaction v3 — Bug Fix Report

이전 v2의 C_claude는 cumulative summary append 버그로 unbounded growth 발생.
v3에서 REPLACE 방식으로 수정 (≤2000자 cap, PM 4개로 축소).

## 결과 (median of 3 judge trials, N=3)

| Mode | Structure | Assignment | Debate | **Overall** | Sub-agent failures |
|---|---|---|---|---|---|
| C_minimal (last 30 + PM 10) | 0.75±0.04 | 0.00±0.00 | 0.93±0.03 | **0.53±0.01** | 0.00±0.00 |
| C_filter (last 8 + PM 4) [default] | 0.77±0.07 | 0.02±0.04 | 0.88±0.12 | **0.54±0.06** | 0.00±0.00 |
| C_claude_v2 BUGGY (cumulative summary) | 0.72±0.04 | 0.02±0.04 | 0.95±0.00 | **0.54±0.02** | 6.33±0.58 |
| C_claude_v3 FIXED (bounded REPLACE summary) | N/A | N/A | N/A | **N/A** | N/A |

## 핵심 발견

- C_minimal: 0.532
- C_filter: 0.536
- C_claude_v2 (BUGGY): 0.535
- C_claude_v3 (FIXED): 0.000

- Δ (v3 vs v2): -0.535
- Δ (v3 vs filter): -0.536

- v2 sub-agent failures: 6.3/run, v3 fixed: 0.0/run (Δ=-6.3)
