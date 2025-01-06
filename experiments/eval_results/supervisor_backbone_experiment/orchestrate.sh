#!/bin/bash
set -e

ROOT=/home/piai/ai_course/agent_test
EXP=$ROOT/eval_results/supervisor_backbone_experiment
mkdir -p "$EXP/logs" "$EXP/snapshots"

cd "$ROOT"

N=${1:-3}
MODES=${MODES:-"baseline supervisor_gemini wbsgen_gemini supervisor_wbsgen_gemini"}

echo "=== Supervisor backbone experiment: N=$N ==="
for mode in $MODES; do
  echo ""
  echo "── Mode: $mode ──"
  python "$EXP/run_supervisor_backbone.py" "$mode" "$N" > "$EXP/logs/$mode.log" 2>&1
  echo "  done $mode"
done

echo ""
echo "=== Collect outputs ==="
mv "$ROOT"/eval_results/wbs_snapshot_H_*_supbb_*.json "$EXP/snapshots/" 2>/dev/null || true
mv "$ROOT"/eval_results/summary_qwen-api_supbb_*.csv "$EXP/" 2>/dev/null || true
mv "$ROOT"/eval_results/experiment_qwen-api_supbb_*.json "$EXP/" 2>/dev/null || true

echo "Snapshots:"
find "$EXP/snapshots" -maxdepth 1 -name 'wbs_snapshot_*.json' | wc -l
echo "Output: $EXP"
