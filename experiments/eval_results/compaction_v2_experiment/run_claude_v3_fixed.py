"""C_claude_v3_fixed only — 다른 mode (minimal/filter)는 v2 결과 그대로 재활용."""
import sys, os
sys.path.insert(0, '/home/piai/ai_course/agent_test')
from dotenv import load_dotenv; load_dotenv('/home/piai/ai_course/agent_test/.env')
os.environ['JUDGE_MODEL_GEMINI'] = 'gemini-3.1-pro-preview'

# Import the fixed compactor
sys.path.insert(0, '/home/piai/ai_course/agent_test/eval_results/compaction_v2_experiment')
from compactor_claude_v3_fixed import compactor_claude_v3

from agents import sub_agents

# Sample capture wrapper
def _wrap_capture(mode_name, fn):
    samples_dir = '/home/piai/ai_course/agent_test/eval_results/compaction_v2_experiment/samples'
    os.makedirs(samples_dir, exist_ok=True)
    sample_path = f'{samples_dir}/{mode_name}.txt'
    seen = set()
    def wrapped(state, max_messages=8):
        result = fn(state, max_messages=max_messages)
        log_len = len(state.get('debate_log', []))
        if log_len not in seen and log_len > 0:
            seen.add(log_len)
            with open(sample_path, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"# Mode: {mode_name} | log size: {log_len} | output chars: {len(result)}\n")
                f.write(f"{'='*80}\n{result}\n")
        return result
    return wrapped

if __name__ == '__main__':
    n_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    sub_agents._format_debate_history = _wrap_capture('claude_v3', compactor_claude_v3)
    print(f"🔧 Patched with FIXED claude_v3 (bounded summary REPLACE)")
    os.environ['RUNNER_ID'] = 'compactv2_claude_v3'
    from eval.experiment_runner import run_all_experiments
    run_all_experiments(
        backend='qwen-api',
        runs_per_condition=n_runs,
        conditions=['C3_3rounds'],
        harness_settings=None,
        cross_judge=False,
    )
