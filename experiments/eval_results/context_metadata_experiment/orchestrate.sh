#!/bin/bash
# Context metadata ablation: M_disc only (M_resume·M_both reuse existing data)
set -e
ROOT=/home/piai/ai_course/agent_test
EXP=$ROOT/eval_results/context_metadata_experiment
cd $ROOT

echo "=== Run M_disc (eDISC only) × 3 ==="
python $EXP/run_disc_only.py 3 > $EXP/logs/disc_only.log 2>&1
echo "  done"

mv $ROOT/eval_results/wbs_snapshot_*context_disc_only*.json $EXP/snapshots/ 2>/dev/null || true
mv $ROOT/eval_results/summary_qwen-api_context_disc_only*.csv $EXP/ 2>/dev/null || true
mv $ROOT/eval_results/experiment_qwen-api_context_disc_only*.json $EXP/ 2>/dev/null || true

echo ""
echo "=== Judge median ×3 (M_disc + reuse M_resume + M_both) ==="
python $EXP/rejudge_median.py 2>&1 | tail -30

echo ""
echo "=== Build report + figures ==="
python $EXP/build_report.py 2>&1 | tail -10

echo "=== ALL DONE ==="
