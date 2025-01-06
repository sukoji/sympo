#!/bin/bash
# Compaction v2 orchestration: 3 modes × 3 runs → organize → judge median (3x) → figures+report
set -e
ROOT=/home/piai/ai_course/agent_test
EXP=$ROOT/eval_results/compaction_v2_experiment
cd $ROOT

for mode in minimal filter claude; do
  echo ""
  echo "=== MODE: $mode ==="
  python $EXP/run_compaction_v3.py $mode 3 > $EXP/logs/$mode.log 2>&1
  echo "  done $mode"
done

echo ""
echo "=== Move outputs to v2 dir ==="
mv $ROOT/eval_results/wbs_snapshot_*compactv2_*.json $EXP/snapshots/ 2>/dev/null || true
mv $ROOT/eval_results/summary_qwen-api_compactv2_*.csv $EXP/ 2>/dev/null || true
mv $ROOT/eval_results/experiment_qwen-api_compactv2_*.json $EXP/ 2>/dev/null || true
echo "snapshots: $(ls $EXP/snapshots/ | wc -l)"

echo ""
echo "=== Re-judge with median (3x per snapshot) ==="
python $EXP/rejudge_median.py 2>&1 | tail -30

echo ""
echo "=== Build report + figures ==="
python $EXP/build_report.py 2>&1 | tail -10

echo ""
echo "=== ALL DONE ==="
