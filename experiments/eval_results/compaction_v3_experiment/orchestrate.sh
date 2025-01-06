#!/bin/bash
set -e
ROOT=/home/piai/ai_course/agent_test
EXP=$ROOT/eval_results/compaction_v3_experiment
cd $ROOT

for mode in minimal filter claude; do
  echo ""
  echo "=== MODE: $mode ==="
  python $EXP/run_compaction.py $mode 3 > $EXP/logs/$mode.log 2>&1
  echo "  done $mode"
done

mv $ROOT/eval_results/wbs_snapshot_*compactv3_*.json $EXP/snapshots/ 2>/dev/null || true
mv $ROOT/eval_results/summary_qwen-api_compactv3_*.csv $EXP/ 2>/dev/null || true
mv $ROOT/eval_results/experiment_qwen-api_compactv3_*.json $EXP/ 2>/dev/null || true

echo ""
echo "=== Judge median ×3 (Flash Lite to avoid Pro Preview hang) ==="
python $EXP/rejudge_median.py 2>&1 | tail -25

echo ""
echo "=== Build report ==="
python $EXP/build_report.py 2>&1 | tail -10

echo "=== ALL DONE ==="
