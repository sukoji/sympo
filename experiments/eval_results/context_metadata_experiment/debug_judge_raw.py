import json
import os
import sys

sys.path.insert(0, "/home/piai/ai_course/agent_test")
from dotenv import load_dotenv

load_dotenv("/home/piai/ai_course/agent_test/.env")
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

from langchain_google_genai import ChatGoogleGenerativeAI
from agents.llm_config import normalize_content
from data_pipeline.member_parser import MemberParser
from eval.llm_judge import (
    ASSIGNMENT_PROMPT,
    DEBATE_PROMPT,
    STRUCTURE_PROMPT,
    _format_debate,
    _format_team,
    _format_wbs,
    _parse_judge_response,
)

SNAPSHOT = (
    "/home/piai/ai_course/agent_test/eval_results/gemma26_ablation/"
    "snapshots/wbs_snapshot_C3_3rounds_r1_qwen-api_gemma26_ablation_20260423_143800.json"
)
MEMBERS = "/home/piai/ai_course/agent_test/sample_data/sample_members"


def load_team():
    team = []
    for name in sorted(os.listdir(MEMBERS)):
        if name.endswith(".txt"):
            path = os.path.join(MEMBERS, name)
            team.append(
                MemberParser.from_resume_text(
                    open(path, encoding="utf-8").read(),
                    name.replace("member_", "").replace(".txt", ""),
                )
            )
    return team


def raw_call(prompt: str) -> str:
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-pro-preview",
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        max_tokens=2500,
        retries=1,
        request_timeout=60,
    )
    return normalize_content(llm.invoke(prompt).content)


def main():
    data = json.load(open(SNAPSHOT, encoding="utf-8"))
    team = load_team()
    wbs_text = _format_wbs(data["wbs_tasks"], team_members=team, debate_log=data.get("debate_log", []))
    l3_lines = [line for line in wbs_text.split("\n") if "(L3," in line]
    prompts = {
        "structure": STRUCTURE_PROMPT.format(wbs_text=wbs_text[:3000]),
        "assignment": ASSIGNMENT_PROMPT.format(
            team_text=_format_team(team)[:1000],
            assignment_text="\n".join(l3_lines[:30]) or "(no L3)",
        ),
        "debate": DEBATE_PROMPT.format(debate_text=_format_debate(data.get("debate_log", []))[:3000]),
    }
    for name, prompt in prompts.items():
        print(f"\n===== {name.upper()} RAW =====")
        raw = raw_call(prompt)
        print(raw[:4000])
        print(f"----- PARSED {name} -----")
        print(_parse_judge_response(raw))


if __name__ == "__main__":
    main()
