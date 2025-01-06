#!/bin/bash
set -e
ROOT=/home/piai/ai_course/agent_test
EXP=$ROOT/eval_results/compaction_v4_experiment
cd $ROOT

for mode in minimal filter claude; do
  echo ""
  echo "=== MODE: $mode ==="
  PYTHONUNBUFFERED=1 python -u $EXP/run_compaction.py $mode 3 > $EXP/logs/$mode.log 2>&1
  echo "  done $mode"
done

mv $ROOT/eval_results/wbs_snapshot_*compactv4_*.json $EXP/snapshots/ 2>/dev/null || true
mv $ROOT/eval_results/summary_qwen-api_compactv4_*.csv $EXP/ 2>/dev/null || true
mv $ROOT/eval_results/experiment_qwen-api_compactv4_*.json $EXP/ 2>/dev/null || true

echo ""
echo "=== Trigger sanity check ==="
for f in $EXP/samples/*.txt; do
  echo "--- $(basename $f) ---"
  grep -E "^# Mode" $f | head -5
  echo "summary triggered first at:"
  grep -B1 "=== 누적 요약" $f | grep "^# Mode" | head -1
  grep -oE "calls=[0-9]+" $f | sort | uniq -c
done

echo ""
echo "=== Judge median ×3 ==="
PYTHONUNBUFFERED=1 python -u $EXP/rejudge_median.py 2>&1 | tail -25

echo ""
echo "=== Build report ==="
python $EXP/build_report.py 2>&1 | tail -10

echo "=== ALL DONE ==="
