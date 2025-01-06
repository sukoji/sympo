#!/bin/bash
# Reasoning mode ablation orchestration: 3 modes × 3 runs → judge median ×3 → figures+report
set -e
ROOT=/home/piai/ai_course/agent_test
EXP=$ROOT/eval_results/reasoning_mode_experiment
cd $ROOT

for mode in none high max; do
  echo ""
  echo "=== MODE: $mode ==="
  python $EXP/run_reasoning.py $mode 3 > $EXP/logs/$mode.log 2>&1
  echo "  done $mode"
done

echo ""
echo "=== Move outputs ==="
mv $ROOT/eval_results/wbs_snapshot_*reasoning_*.json $EXP/snapshots/ 2>/dev/null || true
mv $ROOT/eval_results/summary_qwen-api_reasoning_*.csv $EXP/ 2>/dev/null || true
mv $ROOT/eval_results/experiment_qwen-api_reasoning_*.json $EXP/ 2>/dev/null || true
echo "snapshots: $(ls $EXP/snapshots/ | wc -l)"

echo ""
echo "=== Judge median ×3 per snapshot ==="
python $EXP/rejudge_median.py 2>&1 | tail -30

echo ""
echo "=== Build report + figures ==="
python $EXP/build_report.py 2>&1 | tail -10

echo ""
echo "=== ALL DONE ==="
