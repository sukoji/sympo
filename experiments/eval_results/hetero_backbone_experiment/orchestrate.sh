#!/bin/bash
# Orchestration: wait for C_claude rerun → check Pro Preview → run hetero × 3 → re-judge → report
set -e
ROOT=/home/piai/ai_course/agent_test
HETERO_DIR=$ROOT/eval_results/hetero_backbone_experiment
mkdir -p $HETERO_DIR/snapshots $HETERO_DIR/logs

cd $ROOT

echo "=== STEP 1: Wait for C_claude rerun (if running) ==="
while pgrep -f "run_compaction_v2.py claude" > /dev/null; do
  sleep 30
done
echo "C_claude done (or not running)"

echo ""
echo "=== STEP 2: Check Gemini Pro Preview availability ==="
PRO_OK=$(timeout 12 curl -sS \
  "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-pro-preview:generateContent?key=${GOOGLE_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"contents":[{"parts":[{"text":"ok"}]}],"generationConfig":{"maxOutputTokens":5}}' 2>/dev/null \
  | grep -c '"text"' || echo 0)

if [ "$PRO_OK" = "1" ]; then
  JUDGE_MODEL="gemini-3.1-pro-preview"
  echo "✅ Pro Preview OK — using as Judge"
else
  JUDGE_MODEL="gemini-3.1-flash-lite-preview"
  echo "⚠️ Pro Preview down (503) — falling back to Flash Lite as Judge"
fi
export JUDGE_MODEL_GEMINI=$JUDGE_MODEL

echo ""
echo "=== STEP 3: Run 3 hetero conditions × N=3 ==="
for cond in wbsgen taskmgr both; do
  echo ""
  echo "── Condition: H_$cond ──"
  python $HETERO_DIR/run_hetero.py $cond 3 > $HETERO_DIR/logs/$cond.log 2>&1
  echo "  done $cond"
done

echo ""
echo "=== STEP 4: Move outputs to hetero_backbone_experiment/ ==="
mv $ROOT/eval_results/wbs_snapshot_*hetero_*.json $HETERO_DIR/snapshots/ 2>/dev/null || true
mv $ROOT/eval_results/summary_qwen-api_hetero_*.csv $HETERO_DIR/ 2>/dev/null || true
mv $ROOT/eval_results/experiment_qwen-api_hetero_*.json $HETERO_DIR/ 2>/dev/null || true
ls $HETERO_DIR/snapshots/ | wc -l

echo ""
echo "=== STEP 5: Batch re-judge with $JUDGE_MODEL ==="
JUDGE_MODEL_GEMINI=$JUDGE_MODEL python $HETERO_DIR/rejudge_hetero.py 2>&1 | tail -25

echo ""
echo "=== STEP 6: Build comparison figures + report ==="
python $HETERO_DIR/build_hetero_report.py 2>&1 | tail -10

echo ""
echo "=== ALL DONE ==="
echo "Output: $HETERO_DIR/"
