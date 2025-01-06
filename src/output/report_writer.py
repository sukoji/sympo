"""WBS / debate log output writers."""
import json
import os
from typing import List

from schemas.wbs_schema import WBSOutput, DebateMessage, WBSTask


class ReportWriter:
    def __init__(self, output_dir: str = "./generated"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def write_wbs_markdown(self, wbs_output: WBSOutput, tasks: List[WBSTask]) -> str:
        path = os.path.join(self.output_dir, "wbs_output.md")
        lines = [
            f"# {wbs_output.project_name} — WBS",
            "",
            wbs_output.summary or "",
            "",
            f"**총 기간:** {wbs_output.total_weeks}주",
            "",
        ]
        for task in tasks:
            lvl = task.level.value if hasattr(task.level, "value") else str(task.level)
            assignees = ", ".join(task.assigned_to) or "미배정"
            lines.append(
                f"- `[{lvl}] {task.task_id}` **{task.title}** — "
                f"{task.estimated_days}일 + 버퍼 {task.buffer_days}일 | 담당: {assignees}"
            )
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return path

    def write_debate_log(self, debate_log: List, project_name: str) -> str:
        path = os.path.join(self.output_dir, "debate_log.md")
        lines = [f"# {project_name} — 토론 로그", ""]
        for msg in debate_log:
            if isinstance(msg, DebateMessage):
                role = msg.agent_name
                text = msg.message
                mtype = msg.message_type
            elif isinstance(msg, dict):
                role = msg.get("agent_name", "Agent")
                text = msg.get("message", "")
                mtype = msg.get("message_type", "comment")
            else:
                role, text, mtype = "Agent", str(msg), "comment"
            lines.append(f"### {role} ({mtype})\n\n{text}\n")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return path

    def write_json_output(self, wbs_output: WBSOutput) -> str:
        path = os.path.join(self.output_dir, "wbs_output.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(wbs_output.model_dump(), f, ensure_ascii=False, indent=2)
        return path

    def explain_task_schedule(self, task_id: str, wbs_output: WBSOutput) -> str:
        for raw in wbs_output.tasks:
            tid = raw.get("task_id") if isinstance(raw, dict) else getattr(raw, "task_id", "")
            if tid != task_id:
                continue
            if isinstance(raw, dict):
                title = raw.get("title", task_id)
                est = raw.get("estimated_days", "?")
                buf = raw.get("buffer_days", 0)
                start = raw.get("start_week")
                end = raw.get("end_week")
            else:
                title = raw.title
                est = raw.estimated_days
                buf = raw.buffer_days
                start = raw.start_week
                end = raw.end_week
            return (
                f"### {title} (`{task_id}`)\n\n"
                f"- 예상: **{est}일**, 버퍼 **{buf}일**\n"
                f"- 주차: {start or '?'} → {end or '?'}\n"
            )
        return f"`{task_id}` 태스크를 찾을 수 없습니다."
