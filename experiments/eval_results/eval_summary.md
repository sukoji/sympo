# PO-Sync GT-Free 평가 보고서

> 현재화 메모(2026-04-27): 이 파일은 2026-04-14 초기 GT-Free 파일럿 결과 스냅샷입니다. 최신 평가 산식과 실험 결론은 `eval_results/EVALUATION_FRAMEWORK.md`, `eval_results/EXPERIMENTS_SUMMARY.md`, `eval_results/validity_analysis.md`를 우선합니다.

생성 시각: 2026-04-14 17:05:45
평가 케이스: 3개

## 종합 결과 요약

| 지표 | Baseline | PO-Sync | Delta (↑ PO-Sync 우위) |
|------|---------|---------|----------------------|
| 구조적 무결성 점수 | 87.0 | 97.5 | +10.5 |
| 제약 충족률 (CSR) | 83.3% | 91.7% | +8.3% |
| CSR (High 중요도) | 80.6% | 91.7% | +11.1% |
| SxS Judge 점수 | 15.0 | 19.3 | +4.3 |
| Red Team 결함 수 | 5.0 | 6.0 | -1.0 ↓ |

## SxS 판정 결과
- PO-Sync 승: **67%**
- Baseline 승: 33%
- 무승부: 0%

## 케이스별 상세

### BC-ECO-001 (ecommerce)
- WBS 태스크 수: Baseline=20, PO-Sync=22
- SI 점수: Baseline=87.5, PO-Sync=97.5
- CSR: Baseline=75.0%, PO-Sync=100.0%
- SxS 승자: **posync** (Baseline 15.0 vs PO-Sync 20.0)
- Red Team 결함: Baseline=5, PO-Sync=6

### BC-FIN-001 (fintech)
- WBS 태스크 수: Baseline=28, PO-Sync=23
- SI 점수: Baseline=87.5, PO-Sync=97.5
- CSR: Baseline=75.0%, PO-Sync=75.0%
- SxS 승자: **baseline** (Baseline 17.0 vs PO-Sync 15.0)
- Red Team 결함: Baseline=5, PO-Sync=6

### BC-BIG-001 (bigdata)
- WBS 태스크 수: Baseline=22, PO-Sync=30
- SI 점수: Baseline=86.0, PO-Sync=97.5
- CSR: Baseline=100.0%, PO-Sync=100.0%
- SxS 승자: **posync** (Baseline 13.0 vs PO-Sync 23.0)
- Red Team 결함: Baseline=5, PO-Sync=6
