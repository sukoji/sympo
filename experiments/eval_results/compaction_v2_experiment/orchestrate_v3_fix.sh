#!/bin/bash
# Run only fixed claude_v3, then re-judge + rebuild report (compare v2 vs v3)
set -e
ROOT=/home/piai/ai_course/agent_test
EXP=$ROOT/eval_results/compaction_v2_experiment
cd $ROOT

echo "=== Run claude_v3 (fixed) × 3 ==="
python $EXP/run_claude_v3_fixed.py 3 > $EXP/logs/claude_v3.log 2>&1
echo "  done"

mv $ROOT/eval_results/wbs_snapshot_*compactv2_claude_v3*.json $EXP/snapshots/ 2>/dev/null || true
mv $ROOT/eval_results/summary_qwen-api_compactv2_claude_v3*.csv $EXP/ 2>/dev/null || true
mv $ROOT/eval_results/experiment_qwen-api_compactv2_claude_v3*.json $EXP/ 2>/dev/null || true

echo ""
echo "=== Judge median ×3 (just claude_v3, then merge with existing v2 data) ==="
python $EXP/rejudge_median_v3.py 2>&1 | tail -25

echo ""
echo "=== Rebuild report with v3 ==="
python $EXP/build_report_v3.py 2>&1 | tail -10

echo ""
echo "=== ALL DONE ==="
