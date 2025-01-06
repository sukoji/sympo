#!/bin/bash
# Reasoning per-agent orchestration: 5 conditions × 3 runs → judge median ×3 → figures+report
set -e
ROOT=/home/piai/ai_course/agent_test
EXP=$ROOT/eval_results/reasoning_mode_experiment_per_agent
cd $ROOT

for cond in baseline wbs_high super_high subagents_high all_high; do
  echo ""
  echo "=== COND: $cond ==="
  python $EXP/run_reasoning_per_agent.py $cond 3 > $EXP/logs/$cond.log 2>&1
  echo "  done $cond"
done

echo ""
echo "=== Move outputs ==="
mv $ROOT/eval_results/wbs_snapshot_*reasoning_pa_*.json $EXP/snapshots/ 2>/dev/null || true
mv $ROOT/eval_results/summary_qwen-api_reasoning_pa_*.csv $EXP/ 2>/dev/null || true
mv $ROOT/eval_results/experiment_qwen-api_reasoning_pa_*.json $EXP/ 2>/dev/null || true
echo "snapshots: $(ls $EXP/snapshots/ | wc -l)"

echo ""
echo "=== Judge median ×3 per snapshot ==="
python $EXP/rejudge_median.py 2>&1 | tail -30

echo ""
echo "=== Build report + figures ==="
python $EXP/build_report.py 2>&1 | tail -10

echo ""
echo "=== ALL DONE ==="
