"""M_disc — eDISC only (이력서 stripped) condition.

Monkey-patch MemberParser.from_resume_text to strip:
- tech_stack → []
- strengths → []
- years_of_experience → 0
- past_projects → []

eDISC 정보(behavior type)는 그대로 유지.
조건: C4_with_disc (use_disc=True, 3R debate) 사용.
"""
import sys, os
sys.path.insert(0, '/home/piai/ai_course/agent_test')
from dotenv import load_dotenv; load_dotenv('/home/piai/ai_course/agent_test/.env')
os.environ['JUDGE_MODEL_GEMINI'] = 'gemini-3.1-pro-preview'

from data_pipeline import member_parser

_orig_from_resume = member_parser.MemberParser.from_resume_text

@classmethod
def patched_from_resume_text(cls, resume_text, name):
    profile = _orig_from_resume(resume_text, name)
    # Strip resume technical fields
    profile.tech_stack = []
    profile.strengths = []
    profile.years_of_experience = 0.0
    profile.past_projects = []
    profile.primary_skills = []
    return profile

member_parser.MemberParser.from_resume_text = patched_from_resume_text

if __name__ == '__main__':
    n_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    print(f"🔧 Patched MemberParser to strip resume tech fields (eDISC retained)")
    os.environ['RUNNER_ID'] = 'context_disc_only'
    from eval.experiment_runner import run_all_experiments
    run_all_experiments(
        backend='qwen-api',
        runs_per_condition=n_runs,
        conditions=['C4_with_disc'],  # use_disc=True + 3R debate
        harness_settings=None,
        cross_judge=False,
    )
