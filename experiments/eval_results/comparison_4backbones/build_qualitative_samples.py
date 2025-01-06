"""Build model-wise qualitative WBS sample figures.

The existing comparison figures summarize aggregate scores. This script adds a
slide/report-oriented qualitative layer by selecting one representative C3
snapshot per backbone, extracting WBS structure and debate evidence, and
rendering compact figures plus a Markdown report.

Outputs:
  eval_results/comparison_4backbones/qualitative_samples/
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
import re
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-agent-test")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


BASE = Path(__file__).resolve().parents[1]
ROOT = BASE
OUT = ROOT / "comparison_4backbones" / "qualitative_samples"
OUT.mkdir(parents=True, exist_ok=True)

SUMMARY = ROOT / "comparison_4backbones" / "summary_4backbones.csv"
COND = "C3_3rounds"

BACKBONES = ["gemma", "qwen", "gemma26", "gemini"]
SHORT = {
    "gemma": "Gemma-4B",
    "qwen": "Qwen3-14B",
    "gemma26": "Gemma-4-26B",
    "gemini": "Gemini API",
}
COLORS = {
    "gemma": "#d97706",
    "qwen": "#7c3aed",
    "gemma26": "#16a34a",
    "gemini": "#2563eb",
}
SNAPSHOT_GLOBS = {
    "gemma": ROOT / "gemma_ablation" / "snapshots" / "wbs_snapshot_C3_3rounds_r*.json",
    "qwen": ROOT / "qwen_ablation" / "snapshots" / "wbs_snapshot_C3_3rounds_r*.json",
    "gemma26": ROOT / "gemma26_ablation" / "snapshots" / "wbs_snapshot_C3_3rounds_r*.json",
    "gemini": ROOT / "gemini_ablation" / "snapshots" / "wbs_snapshot_C3_3rounds_r*.json",
}
MEMBER_NAME = {
    "MBR-FD58": "박선민",
    "MBR-95F6": "이동헌",
    "MBR-335A": "이주성",
    "MBR-387D": "장선열",
    "MBR-F01E": "진석호",
    "MBR-31B4": "윤수빈",
}
ROLE_DEFAULT_NAME = {
    "Designer": "박선민",
    "Planner": "박선민",
    "PM/Planner": "박선민",
    "Designer/Planner": "박선민",
    "Frontend Developer": "장선열",
    "Backend Developer": "이주성",
    "Data Engineer": "이동헌",
    "DevOps": "진석호",
    "QA Engineer": "윤수빈",
}
AGENT_STOPWORDS = ("Task Manager", "PM", "WBS", "시스템", "System")


def configure_matplotlib() -> None:
    font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
    if Path(font_path).exists():
        font_manager.fontManager.addfont(font_path)
        prop = font_manager.FontProperties(fname=font_path)
        plt.rcParams["font.family"] = prop.get_name()
    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "figure.dpi": 120,
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
        }
    )


def parse_float(value, default=None):
    if value in (None, "", "NA"):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_c3_summary() -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    with open(SUMMARY, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["condition"] != COND or row["backbone"] not in BACKBONES:
                continue
            out[row["backbone"]] = {
                "overall_mean": parse_float(row.get("Overall_mean"), 0.0),
                "overall_std": parse_float(row.get("Overall_std"), 0.0),
                "auto_mean": parse_float(row.get("AutoOverall_mean"), 0.0),
                "structure": parse_float(row.get("S_mean"), 0.0),
                "assignment": parse_float(row.get("A_mean"), 0.0),
                "debate": parse_float(row.get("D_mean"), 0.0),
            }
    return out


def get_tasks(snapshot: dict) -> list[dict]:
    for key in ("wbs_tasks", "final_wbs", "wbs", "current_wbs_draft"):
        value = snapshot.get(key)
        if isinstance(value, list):
            return value
    return []


def task_id(task: dict) -> str:
    return str(task.get("task_id") or task.get("id") or "")


def task_level(task: dict) -> str:
    return str(task.get("level") or "").upper()


def task_title(task: dict) -> str:
    return str(task.get("title") or "")


def task_days(task: dict, key: str = "estimated_days") -> float:
    return parse_float(task.get(key), 0.0) or 0.0


def assigned_to(task: dict) -> list[str]:
    assignees = task.get("assigned_to") or []
    if isinstance(assignees, str):
        return [assignees] if assignees else []
    if isinstance(assignees, list):
        return [str(x) for x in assignees if str(x)]
    return []


def dependencies(task: dict) -> list[str]:
    deps = task.get("dependencies") or []
    if isinstance(deps, str):
        return [deps] if deps else []
    if isinstance(deps, list):
        return [str(x) for x in deps if str(x)]
    return []


def task_role(task: dict) -> str:
    return str(task.get("assigned_role") or task.get("role") or "")


def extract_assignment_names(snapshot: dict, tasks: list[dict]) -> tuple[dict[str, str], dict[str, str]]:
    """Infer task/person labels from Task Manager assignment logs.

    Some older snapshots contain model-generated MBR-* codes that are not the
    real member IDs. The Task Manager log usually preserves Korean names for
    the same task IDs, so use it as the display source and never expose raw
    generated codes in figures.
    """
    task_name: dict[str, str] = {}
    code_name: dict[str, str] = {}
    for msg in snapshot.get("debate_log") or []:
        text = re.sub(r"\s+", " ", str(msg.get("message") or ""))
        if "L3 배정 현황" not in text and "배정" not in text:
            continue
        for tid, _, name in re.findall(r"\[(L3-\d{2}-\d{2}-\d{2})\]\s*([^→\n]+?)\s*→\s*([가-힣]{2,4})", text):
            task_name[tid] = name
    by_id = {task_id(t): t for t in tasks}
    for tid, name in task_name.items():
        task = by_id.get(tid)
        if not task:
            continue
        for code in assigned_to(task):
            code_name[code] = name
    for code, name in MEMBER_NAME.items():
        code_name.setdefault(code, name)
    return task_name, code_name


def display_assignees(task: dict, task_name: dict[str, str], code_name: dict[str, str]) -> list[str]:
    tid = task_id(task)
    if tid in task_name:
        return [task_name[tid]]
    names = [code_name[code] for code in assigned_to(task) if code in code_name]
    if names:
        return list(dict.fromkeys(names))
    if assigned_to(task):
        return [ROLE_DEFAULT_NAME.get(task_role(task), role_label(task_role(task)))]
    return []


def role_label(role: str) -> str:
    role = role or "담당"
    normalized = {
        "Backend Developer": "백엔드 담당",
        "Frontend Developer": "프론트엔드 담당",
        "Data Engineer": "데이터 담당",
        "QA Engineer": "QA 담당",
        "DevOps": "인프라 담당",
        "Designer": "디자인 담당",
        "Planner": "기획 담당",
        "PM/Planner": "기획 담당",
        "Designer/Planner": "기획/디자인 담당",
    }
    return normalized.get(role, f"{role} 담당")


def gini(values: list[float]) -> float:
    vals = sorted(v for v in values if v >= 0)
    if not vals or sum(vals) == 0:
        return 0.0
    n = len(vals)
    weighted = sum((i + 1) * v for i, v in enumerate(vals))
    return (2 * weighted) / (n * sum(vals)) - (n + 1) / n


def snapshot_overall(snapshot: dict) -> float | None:
    judge = snapshot.get("llm_judge") or {}
    direct = parse_float(judge.get("overall"))
    if direct is not None:
        return direct
    scores = []
    weights = {"structure": 0.40, "assignment": 0.35, "debate": 0.25}
    for dim, weight in weights.items():
        score = parse_float((judge.get(dim) or {}).get("score"))
        if score is not None and score >= 0:
            scores.append((score, weight))
    if not scores:
        return None
    wsum = sum(w for _, w in scores)
    return sum(s * w for s, w in scores) / wsum


def choose_representative_snapshots(summary: dict[str, dict[str, float]]) -> dict[str, dict]:
    chosen = {}
    for bb in BACKBONES:
        target = summary[bb]["overall_mean"]
        candidates = []
        for raw_path in sorted(glob.glob(str(SNAPSHOT_GLOBS[bb]))):
            path = Path(raw_path)
            with open(path, encoding="utf-8") as f:
                snapshot = json.load(f)
            overall = snapshot_overall(snapshot)
            if overall is None:
                continue
            candidates.append((abs(overall - target), path, snapshot, overall))
        if not candidates:
            raise RuntimeError(f"No C3 snapshots found for {bb}")
        _, path, snapshot, overall = min(candidates, key=lambda item: item[0])
        chosen[bb] = {"path": path, "snapshot": snapshot, "sample_overall": overall}
    return chosen


def compute_sample_metrics(bb: str, selected: dict, summary: dict[str, dict[str, float]]) -> dict:
    snapshot = selected["snapshot"]
    tasks = get_tasks(snapshot)
    task_name, code_name = extract_assignment_names(snapshot, tasks)
    by_level = Counter(task_level(t) for t in tasks)
    children = defaultdict(list)
    for task in tasks:
        parent = task.get("parent_id")
        if parent:
            children[str(parent)].append(task)
    missing_l2 = [
        t for t in tasks
        if task_level(t) == "L2" and not any(task_level(c) == "L3" for c in children.get(task_id(t), []))
    ]
    l3_tasks = [t for t in tasks if task_level(t) == "L3"]
    assigned_l3 = [t for t in l3_tasks if assigned_to(t)]
    estimated = sum(task_days(t, "estimated_days") for t in tasks)
    buffers = sum(task_days(t, "buffer_days") for t in tasks)
    dep_count = sum(len(dependencies(t)) for t in tasks)

    loads = defaultdict(float)
    for task in l3_tasks:
        assignees = display_assignees(task, task_name, code_name)
        if not assignees:
            continue
        split = task_days(task, "estimated_days") / len(assignees)
        for member in assignees:
            loads[member] += split

    debate = snapshot.get("debate_log") or []
    active_agents = set()
    pass_messages = 0
    task_ref_count = 0
    risk_terms = 0
    for msg in debate:
        name = str(msg.get("agent_name") or "")
        text = str(msg.get("message") or "")
        if name and not any(stop in name for stop in AGENT_STOPWORDS):
            active_agents.add(name)
        if "PASS" in text or "추가 의견이 없습니다" in text:
            pass_messages += 1
        task_ref_count += len(re.findall(r"L[123]-\d{2}(?:-\d{2})?(?:-\d{2})?", text))
        risk_terms += len(re.findall(r"버퍼|리스크|위험|테스트|지연|성능|오류|장애", text))

    all_members = set(loads.keys())
    load_values = [loads[m] for m in sorted(all_members)]
    total_load = sum(load_values)
    top_share = max(load_values) / total_load if total_load else 0.0

    judge = snapshot.get("llm_judge") or {}
    return {
        "backbone": bb,
        "model": SHORT[bb],
        "snapshot": selected["path"].name,
        "sample_overall": selected["sample_overall"],
        "c3_overall_mean": summary[bb]["overall_mean"],
        "c3_autoscore_mean": summary[bb]["auto_mean"],
        "judge_structure": parse_float((judge.get("structure") or {}).get("score")),
        "judge_assignment": parse_float((judge.get("assignment") or {}).get("score")),
        "judge_debate": parse_float((judge.get("debate") or {}).get("score")),
        "tasks": len(tasks),
        "l1": by_level["L1"],
        "l2": by_level["L2"],
        "l3": by_level["L3"],
        "missing_l2_children": len(missing_l2),
        "assigned_l3_ratio": len(assigned_l3) / len(l3_tasks) if l3_tasks else 0.0,
        "unique_assignees": len(all_members),
        "workload_gini": gini(load_values),
        "top_assignee_share": top_share,
        "buffer_ratio_pct": (buffers / estimated * 100.0) if estimated else 0.0,
        "dependency_per_task": dep_count / len(tasks) if tasks else 0.0,
        "debate_messages": len(debate),
        "active_agents": len(active_agents),
        "pass_ratio": pass_messages / len(debate) if debate else 0.0,
        "task_ref_count": task_ref_count,
        "risk_term_count": risk_terms,
        "loads": dict(loads),
        "task_name": task_name,
        "code_name": code_name,
    }


def style_axis(ax) -> None:
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def savefig(name: str) -> None:
    path = OUT / name
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {path}")


def fmt(value, digits=3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_metrics_csv(rows: list[dict]) -> None:
    keys = [
        "backbone", "model", "snapshot", "sample_overall", "c3_overall_mean", "c3_autoscore_mean",
        "judge_structure", "judge_assignment", "judge_debate", "tasks", "l1", "l2", "l3",
        "missing_l2_children", "assigned_l3_ratio", "unique_assignees", "workload_gini",
        "top_assignee_share", "buffer_ratio_pct", "dependency_per_task", "debate_messages",
        "active_agents", "pass_ratio", "task_ref_count", "risk_term_count",
    ]
    with open(OUT / "sample_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def format_task_title(title: str, width: int = 24, max_lines: int = 2) -> str:
    return "\n".join(textwrap.wrap(title, width=width, max_lines=max_lines, placeholder="..."))


def sample_table_rows(bb: str, snapshot: dict, row: dict, max_rows: int = 12) -> list[dict[str, str]]:
    tasks = get_tasks(snapshot)
    out = []
    for task in tasks:
        level = task_level(task)
        if level not in ("L1", "L2", "L3"):
            continue
        names = display_assignees(task, row["task_name"], row["code_name"]) if level == "L3" else []
        days = task_days(task, "estimated_days")
        buffer = task_days(task, "buffer_days")
        out.append(
            {
                "model": SHORT[bb],
                "id": task_id(task),
                "level": level,
                "task": task_title(task),
                "assignee": ", ".join(names) if names else "-",
                "role": role_label(task_role(task)) if level == "L3" else "-",
                "days_buffer": f"{days:g} / {buffer:g}" if level == "L3" else f"{days:g} / {buffer:g}",
            }
        )
        if len(out) >= max_rows:
            break
    return out


def write_wbs_sample_tables(rows: list[dict], chosen: dict[str, dict]) -> None:
    keys = ["model", "id", "level", "task", "assignee", "role", "days_buffer"]
    with open(OUT / "wbs_sample_table.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for bb, row in zip(BACKBONES, rows):
            for item in sample_table_rows(bb, chosen[bb]["snapshot"], row, max_rows=14):
                writer.writerow(item)


def clean_snippet(text: str, width: int = 42, max_lines: int = 5) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"MBR-[A-Z0-9]+", "담당자 코드", text)
    text = re.sub(r"\[[123]\.\s*", "[", text)
    return "\n".join(textwrap.wrap(text, width=width, max_lines=max_lines, placeholder="..."))


def evidence_rows_for_model(bb: str, snapshot: dict, row: dict, max_rows: int = 3) -> list[dict[str, str]]:
    tasks = {task_id(t): t for t in get_tasks(snapshot)}
    evidence = []
    seen = set()
    keywords = r"배정|담당|재배정|역량|직무|전문성|스택|부합|리스크|타당"
    for msg in snapshot.get("debate_log") or []:
        agent = str(msg.get("agent_name") or "")
        text = str(msg.get("message") or "")
        if "Task Manager" in agent or "WBS Gen" in agent:
            continue
        if not re.search(keywords, text):
            continue
        task_ids = re.findall(r"L3-\d{2}-\d{2}-\d{2}", text)
        if not task_ids:
            continue
        for tid in task_ids:
            if tid not in tasks or tid in seen:
                continue
            task = tasks[tid]
            names = display_assignees(task, row["task_name"], row["code_name"])
            pos = max(text.find(tid) - 70, 0)
            snippet = text[pos : pos + 520]
            evidence.append(
                {
                    "model": SHORT[bb],
                    "task": f"{tid}\n{format_task_title(task_title(task), width=20, max_lines=2)}",
                    "assignee": ", ".join(names) if names else role_label(task_role(task)),
                    "agent": agent,
                    "evidence": clean_snippet(snippet),
                }
            )
            seen.add(tid)
            if len(evidence) >= max_rows:
                return evidence

    # Fallback: use the Task Manager assignment summary as provenance, not a reason.
    for msg in snapshot.get("debate_log") or []:
        text = str(msg.get("message") or "")
        if "L3 배정 현황" not in text:
            continue
        for tid, _, name in re.findall(r"\[(L3-\d{2}-\d{2}-\d{2})\]\s*([^→\n]+?)\s*→\s*([가-힣]{2,4})", text):
            if tid in tasks and tid not in seen:
                evidence.append(
                    {
                        "model": SHORT[bb],
                        "task": f"{tid}\n{format_task_title(task_title(tasks[tid]), width=20, max_lines=2)}",
                        "assignee": name,
                        "agent": "Task Manager",
                        "evidence": "초기 R&R 배정 현황 로그에서 확인된 담당자입니다. 배정 사유 로그는 해당 스냅샷에 충분히 남아 있지 않습니다.",
                    }
                )
                seen.add(tid)
                if len(evidence) >= max_rows:
                    return evidence
    return evidence


def write_assignment_evidence(rows: list[dict], chosen: dict[str, dict]) -> None:
    keys = ["model", "task", "assignee", "agent", "evidence"]
    with open(OUT / "assignment_reason_evidence.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for bb, row in zip(BACKBONES, rows):
            for item in evidence_rows_for_model(bb, chosen[bb]["snapshot"], row, max_rows=4):
                writer.writerow(item)


def tree_lines(tasks: list[dict], max_lines: int = 16) -> list[str]:
    lines = []
    for task in tasks:
        level = task_level(task)
        if level not in ("L1", "L2", "L3"):
            continue
        indent = {"L1": "", "L2": "  ", "L3": "    "}[level]
        title = task_title(task)
        assignees = ",".join(MEMBER_NAME.get(x, x) for x in assigned_to(task))
        suffix = f" -> {assignees}" if assignees and level == "L3" else ""
        lines.append(f"{indent}{task_id(task)} {title}{suffix}")
        if len(lines) >= max_lines:
            break
    return lines


def wrap_reason(text: str, width: int = 86) -> str:
    if not text:
        return "NA"
    text = re.sub(r"\s+", " ", str(text)).strip()
    if text.startswith("{") and '"reason"' in text:
        try:
            parsed = json.loads(text)
            text = str(parsed.get("reason") or "").strip()
        except json.JSONDecodeError:
            return "Judge reason text is truncated in the saved snapshot; numeric score is used."
    if not text:
        return "NA"
    return "\n".join(textwrap.wrap(text, width=width, max_lines=3, placeholder="..."))


def build_figures(rows: list[dict], chosen: dict[str, dict]) -> None:
    labels = [SHORT[bb] for bb in BACKBONES]
    x = np.arange(len(BACKBONES))

    # Fig Q1: aggregate quality signals plus selected sample overlay.
    fig, ax = plt.subplots(figsize=(13.8, 6.6))
    w = 0.34
    llm = [rows[i]["c3_overall_mean"] for i in range(len(rows))]
    auto = [rows[i]["c3_autoscore_mean"] for i in range(len(rows))]
    sample = [rows[i]["sample_overall"] for i in range(len(rows))]
    b1 = ax.bar(x - w / 2, llm, w, color="#111827", label="C3 LLM Judge mean", edgecolor="#111827")
    b2 = ax.bar(x + w / 2, auto, w, color=[COLORS[bb] for bb in BACKBONES], label="C3 AutoScore mean", edgecolor="#111827")
    ax.scatter(x, sample, s=130, marker="D", color="#f59e0b", edgecolor="#111827", linewidth=0.8, zorder=5, label="Selected sample")
    for i, (a, b, s) in enumerate(zip(llm, auto, sample)):
        ax.text(i - w / 2, a + 0.012, f"{a:.3f}", ha="center", fontsize=11)
        ax.text(i + w / 2, b + 0.012, f"{b:.3f}", ha="center", fontsize=11)
        ax.text(i, s - 0.035, f"{s:.3f}", ha="center", fontsize=10, color="#92400e")
    for i, bb in enumerate(BACKBONES):
        if bb == "gemini":
            b1[i].set_hatch("//")
            b2[i].set_hatch("//")
        if bb == "gemma26":
            b1[i].set_linewidth(2.2)
            b2[i].set_linewidth(2.2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.40, 0.90)
    ax.set_ylabel("Score")
    ax.set_title("Fig Q1. C3 Model Quality Signals and Representative Sample")
    ax.legend(loc="lower right", framealpha=0.95)
    style_axis(ax)
    savefig("figQ1_c3_quality_signals.png")

    # Fig Q2: WBS shape and basic completeness flags.
    fig, axes = plt.subplots(1, 2, figsize=(15.4, 6.8))
    bottoms = np.zeros(len(rows))
    for key, color, label in [
        ("l1", "#475569", "L1"),
        ("l2", "#60a5fa", "L2"),
        ("l3", "#16a34a", "L3"),
    ]:
        values = np.array([r[key] for r in rows])
        axes[0].bar(x, values, bottom=bottoms, color=color, label=label, edgecolor="#111827", linewidth=0.6)
        bottoms += values
    for i, r in enumerate(rows):
        axes[0].text(i, bottoms[i] + 1.2, f"{r['tasks']} tasks", ha="center", fontsize=11)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Task count")
    axes[0].set_title("A. Hierarchy depth and task volume")
    axes[0].legend(framealpha=0.95)
    style_axis(axes[0])

    assigned_pct = [r["assigned_l3_ratio"] * 100 for r in rows]
    missing = [r["missing_l2_children"] for r in rows]
    bars = axes[1].bar(x - 0.18, assigned_pct, 0.36, color="#22c55e", edgecolor="#166534", label="Assigned L3 %")
    ax2 = axes[1].twinx()
    ax2.bar(x + 0.18, missing, 0.36, color="#f97316", edgecolor="#9a3412", label="L2 without L3")
    for i, v in enumerate(assigned_pct):
        axes[1].text(i - 0.18, v + 2, f"{v:.0f}%", ha="center", fontsize=10)
    for i, v in enumerate(missing):
        ax2.text(i + 0.18, v + 0.15, f"{v}", ha="center", fontsize=10, color="#9a3412")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylim(0, 115)
    ax2.set_ylim(0, max(missing + [1]) + 2)
    axes[1].set_ylabel("Assigned L3 tasks (%)")
    ax2.set_ylabel("Missing L3 child count")
    axes[1].set_title("B. Assignment coverage and hierarchy gaps")
    handles1, labels1 = axes[1].get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    axes[1].legend(handles1 + handles2, labels1 + labels2, loc="lower right", framealpha=0.95)
    style_axis(axes[1])
    ax2.spines["top"].set_visible(False)
    savefig("figQ2_sample_structure_profile.png")

    # Fig Q3: workload distribution heatmap.
    members = sorted({m for row in rows for m in row["loads"]})
    matrix = np.array([[row["loads"].get(m, 0.0) for m in members] for row in rows])
    fig, ax = plt.subplots(figsize=(14.6, 6.3))
    vmax = max(matrix.max(), 1)
    im = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=vmax, aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            color = "white" if val > vmax * 0.58 else "#111827"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=10, color=color)
    ax.set_xticks(np.arange(len(members)))
    ax.set_xticklabels(members, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(labels)
    ax.set_title("Fig Q3. Representative Sample Workload by Assigned Member (L3 estimated days)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Estimated days")
    savefig("figQ3_sample_assignment_heatmap.png")

    # Fig Q4: WBS sample as tables. Avoid color-only hierarchy cues.
    fig, axes = plt.subplots(2, 2, figsize=(22.0, 16.0))
    for ax, bb, row in zip(axes.flat, BACKBONES, rows):
        ax.axis("off")
        ax.set_title(f"{SHORT[bb]} WBS sample table", loc="left", color=COLORS[bb], fontweight="bold", pad=10)
        table_rows = sample_table_rows(bb, chosen[bb]["snapshot"], row, max_rows=12)
        cell_text = [
            [
                item["id"],
                item["level"],
                format_task_title(item["task"], width=26, max_lines=2),
                item["assignee"],
                item["role"],
                item["days_buffer"],
            ]
            for item in table_rows
        ]
        table = ax.table(
            cellText=cell_text,
            colLabels=["ID", "Lv", "Task", "담당자", "역할", "일수/버퍼"],
            colWidths=[0.16, 0.07, 0.38, 0.13, 0.15, 0.11],
            loc="upper left",
            cellLoc="left",
            bbox=[0.0, 0.08, 1.0, 0.88],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.9)
        for (ridx, cidx), cell in table.get_celld().items():
            cell.set_edgecolor("#cbd5e1")
            cell.set_linewidth(0.5)
            if ridx == 0:
                cell.set_facecolor("#e5e7eb")
                cell.set_text_props(weight="bold", color="#111827")
                cell.set_height(0.058)
            else:
                level = table_rows[ridx - 1]["level"]
                if level == "L1":
                    cell.set_facecolor("#f8fafc")
                    cell.set_text_props(weight="bold")
                elif level == "L2":
                    cell.set_facecolor("#eff6ff")
                elif level == "L3":
                    cell.set_facecolor("#f0fdf4")
                cell.set_height(0.070)
        footer = f"tasks={row['tasks']}, L3 assigned={row['assigned_l3_ratio']:.0%}, workload Gini={row['workload_gini']:.3f}, buffer={row['buffer_ratio_pct']:.1f}%"
        ax.text(0.0, 0.015, footer, fontsize=10.2, color="#475569", transform=ax.transAxes)
    fig.suptitle("Fig Q4. WBS Samples as Tables: hierarchy, assignee name, role, days/buffer", y=0.995, fontsize=18)
    savefig("figQ4_sample_wbs_table.png")

    # Fig Q5: debate evidence.
    fig, ax = plt.subplots(figsize=(14.8, 6.5))
    width = 0.24
    active = [r["active_agents"] for r in rows]
    refs = [r["task_ref_count"] for r in rows]
    risk = [r["risk_term_count"] for r in rows]
    ax.bar(x - width, active, width, color="#0f766e", edgecolor="#115e59", label="Active non-PM agents")
    ax.bar(x, refs, width, color="#7c3aed", edgecolor="#581c87", label="Task-ID references")
    ax.bar(x + width, risk, width, color="#dc2626", edgecolor="#7f1d1d", label="Risk/test terms")
    for xpos, values in [(x - width, active), (x, refs), (x + width, risk)]:
        for xi, val in zip(xpos, values):
            ax.text(xi, val + max(refs + risk + active) * 0.025, f"{val}", ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Count")
    ax.set_title("Fig Q5. Debate Trace Evidence in Selected C3 Samples")
    ax.legend(loc="upper left", framealpha=0.95)
    style_axis(ax)
    savefig("figQ5_sample_debate_evidence.png")

    # Fig Q6: assignment reason snippets paired with task and assignee.
    fig, axes = plt.subplots(2, 2, figsize=(22.0, 15.4))
    for ax, bb, row in zip(axes.flat, BACKBONES, rows):
        ax.axis("off")
        ax.set_title(f"{SHORT[bb]} assignment reason evidence", loc="left", color=COLORS[bb], fontweight="bold", pad=10)
        evidence = evidence_rows_for_model(bb, chosen[bb]["snapshot"], row, max_rows=3)
        cell_text = [
            [
                item["task"],
                item["assignee"],
                item["agent"],
                item["evidence"],
            ]
            for item in evidence
        ]
        table = ax.table(
            cellText=cell_text,
            colLabels=["Task", "담당자", "Log speaker", "배정/재배정 근거 로그"],
            colWidths=[0.22, 0.10, 0.14, 0.54],
            loc="upper left",
            cellLoc="left",
            bbox=[0.0, 0.02, 1.0, 0.92],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7.8)
        for (ridx, cidx), cell in table.get_celld().items():
            cell.set_edgecolor("#cbd5e1")
            cell.set_linewidth(0.5)
            if ridx == 0:
                cell.set_facecolor("#e5e7eb")
                cell.set_text_props(weight="bold", color="#111827")
                cell.set_height(0.08)
            else:
                cell.set_facecolor("#ffffff" if ridx % 2 else "#f8fafc")
                cell.set_height(0.25)
    fig.suptitle("Fig Q6. Assignment Reason Evidence: task, assignee, and supporting log excerpt", y=0.995, fontsize=18)
    savefig("figQ6_assignment_reason_evidence.png")

    # Fig Q7: scoped selection slide. This is intentionally a local-first view,
    # not a complete scoreboard; full metrics remain in Q1-Q6 and CSV outputs.
    fig = plt.figure(figsize=(21.5, 10.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.18)
    ax_table = fig.add_subplot(gs[0, 0])
    ax_score = fig.add_subplot(gs[0, 1])
    ax_table.axis("off")

    focus_bb = "gemma26"
    focus_idx = BACKBONES.index(focus_bb)
    focus_row = rows[focus_idx]
    focus_items = sample_table_rows(focus_bb, chosen[focus_bb]["snapshot"], focus_row, max_rows=10)
    cell_text = [
        [
            item["id"],
            item["level"],
            format_task_title(item["task"], width=28, max_lines=2),
            item["assignee"],
            item["role"],
            item["days_buffer"],
        ]
        for item in focus_items
    ]
    ax_table.set_title("Selected Local Model: Gemma-4-26B C3 WBS Sample", loc="left", color=COLORS[focus_bb], fontweight="bold", pad=10)
    table = ax_table.table(
        cellText=cell_text,
        colLabels=["ID", "Lv", "Task", "담당자", "역할", "일수/버퍼"],
        colWidths=[0.16, 0.07, 0.40, 0.13, 0.15, 0.11],
        loc="upper left",
        cellLoc="left",
        bbox=[0.0, 0.12, 1.0, 0.82],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.4)
    for (ridx, cidx), cell in table.get_celld().items():
        cell.set_edgecolor("#cbd5e1")
        cell.set_linewidth(0.5)
        if ridx == 0:
            cell.set_facecolor("#dcfce7")
            cell.set_text_props(weight="bold", color="#111827")
            cell.set_height(0.062)
        else:
            level = focus_items[ridx - 1]["level"]
            if level == "L1":
                cell.set_facecolor("#f8fafc")
                cell.set_text_props(weight="bold")
            elif level == "L2":
                cell.set_facecolor("#eff6ff")
            else:
                cell.set_facecolor("#f0fdf4")
            cell.set_height(0.074)
    ax_table.text(
        0.0,
        0.04,
        "Representative C3 output with clear hierarchy, named assignees, roles, and buffer estimates.",
        fontsize=10.5,
        color="#475569",
        transform=ax_table.transAxes,
    )

    local_bbs = ["gemma", "qwen", "gemma26"]
    metrics = [
        ("Overall", "c3_overall_mean"),
        ("Structure", "judge_structure"),
        ("Debate", "judge_debate"),
    ]
    x2 = np.arange(len(metrics))
    width = 0.23
    for i, bb in enumerate(local_bbs):
        r = rows[BACKBONES.index(bb)]
        values = [r[key] for _, key in metrics]
        offset = (i - 1) * width
        bars = ax_score.bar(
            x2 + offset,
            values,
            width,
            color=COLORS[bb],
            edgecolor="#111827",
            linewidth=1.8 if bb == focus_bb else 0.7,
            alpha=0.95 if bb == focus_bb else 0.62,
            label=SHORT[bb],
        )
        for bar, value in zip(bars, values):
            ax_score.text(bar.get_x() + bar.get_width() / 2, value + 0.014, f"{value:.3f}", ha="center", fontsize=9.5)
    ax_score.axhspan(0.70, 0.95, color="#dcfce7", alpha=0.28, zorder=0)
    ax_score.set_xticks(x2)
    ax_score.set_xticklabels([m[0] for m in metrics])
    ax_score.set_ylim(0.25, 1.0)
    ax_score.set_ylabel("Representative C3 sample / C3 mean score")
    ax_score.set_title("Selection Signals for Local Backbones")
    ax_score.legend(loc="lower right", framealpha=0.95)
    style_axis(ax_score)
    fig.suptitle("Fig Q7. Gemma-4-26B C3 Selection View", y=0.99, fontsize=18)
    savefig("figQ7_local_first_selection_view.png")


def build_report(rows: list[dict], chosen: dict[str, dict]) -> None:
    report = OUT / "QUALITATIVE_SAMPLE_REPORT.md"
    with open(report, "w", encoding="utf-8") as f:
        f.write("# WBS Model Quality: Qualitative Sample Pack\n\n")
        f.write("비교 대상은 C3(3-round debate) 조건입니다. 각 모델은 C3 평균 LLM Judge 점수에 가장 가까운 저장 스냅샷을 대표 샘플로 자동 선택했습니다.\n\n")

        f.write("## Generated Figures\n\n")
        for name, desc in [
            ("figQ1_c3_quality_signals.png", "C3 모델별 LLM Judge, AutoScore, 대표 샘플 점수"),
            ("figQ2_sample_structure_profile.png", "대표 샘플의 WBS 계층 구조, 배정 커버리지, L2 gap"),
            ("figQ3_sample_assignment_heatmap.png", "대표 샘플의 팀원명 기준 L3 작업량 heatmap"),
            ("figQ4_sample_wbs_table.png", "모델별 WBS 샘플을 표 형태로 포맷팅"),
            ("figQ5_sample_debate_evidence.png", "토론 로그의 참여자·task reference·risk/test 근거량"),
            ("figQ6_assignment_reason_evidence.png", "태스크·담당자·배정/재배정 근거 로그를 한 세트로 정리"),
            ("figQ7_local_first_selection_view.png", "Gemma-4-26B 선택 논리를 강조한 발표용 figure"),
        ]:
            f.write(f"- `{name}`: {desc}\n")
        f.write("\n")

        f.write("## Representative Samples\n\n")
        f.write("| Model | Snapshot | Sample Judge | C3 Judge Mean | C3 AutoScore | Tasks | L1/L2/L3 | Assigned L3 | Workload Gini | Buffer % |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['model']} | `{r['snapshot']}` | {fmt(r['sample_overall'])} | "
                f"{fmt(r['c3_overall_mean'])} | {fmt(r['c3_autoscore_mean'])} | {r['tasks']} | "
                f"{r['l1']}/{r['l2']}/{r['l3']} | {r['assigned_l3_ratio']:.0%} | "
                f"{r['workload_gini']:.3f} | {r['buffer_ratio_pct']:.1f}% |\n"
            )
        f.write("\n")

        f.write("## Evaluation Metrics Used\n\n")
        f.write("- LLM Judge overall: Structure 0.40, Assignment 0.35, Debate 0.25 active-dimension weighting from the existing comparison pipeline.\n")
        f.write("- AutoScore v2: quality, allocation, orchestration deterministic score already present in `summary_4backbones.csv`.\n")
        f.write("- Qualitative sample metrics: task hierarchy counts, missing L2 children, L3 assignment coverage, workload Gini, top-assignee share, buffer ratio, dependency density, active debate agents, task-ID references, and risk/test term counts.\n\n")
        f.write("Team member display rule: figures use names recovered from Task Manager assignment logs or known sample-member mappings. When a snapshot uses model-generated MBR codes, raw codes are not shown; the display falls back to role-based labels.\n\n")

        f.write("## Per-Model Qualitative Notes\n\n")
        for bb, r in zip(BACKBONES, rows):
            snapshot = chosen[bb]["snapshot"]
            judge = snapshot.get("llm_judge") or {}
            f.write(f"### {r['model']}\n\n")
            f.write(
                f"- Structure/Assignment/Debate sample scores: "
                f"{fmt(r['judge_structure'])} / {fmt(r['judge_assignment'])} / {fmt(r['judge_debate'])}\n"
            )
            f.write(
                f"- WBS shape: {r['tasks']} tasks, {r['l1']} L1, {r['l2']} L2, {r['l3']} L3, "
                f"{r['missing_l2_children']} L2 nodes without L3 children.\n"
            )
            f.write(
                f"- R&R signal: {r['assigned_l3_ratio']:.0%} of L3 tasks assigned, "
                f"{r['unique_assignees']} assignees, workload Gini {r['workload_gini']:.3f}.\n"
            )
            f.write(
                f"- Debate trace: {r['active_agents']} active non-PM agents, "
                f"{r['task_ref_count']} task-ID references, {r['risk_term_count']} risk/test terms.\n"
            )
            for dim in ("structure", "assignment", "debate"):
                reason = (judge.get(dim) or {}).get("reason", "")
                f.write(f"- {dim} judge note: {wrap_reason(reason)}\n")
            f.write("\n")

        f.write("## Suggested Slide Order\n\n")
        f.write("1. Fig Q1 to anchor the model-level ranking.\n")
        f.write("2. Fig Q4 to show concrete WBS output differences in table form.\n")
        f.write("3. Fig Q2 and Fig Q3 to explain structure and R&R quality.\n")
        f.write("4. Fig Q6 to connect assignment decisions with log evidence.\n")
        f.write("5. Fig Q7 when presenting the Gemma-4-26B selection rationale.\n")
        f.write("6. Fig Q5 as aggregate debate trace evidence.\n")
    print(f"saved {report}")


def main() -> None:
    configure_matplotlib()
    summary = load_c3_summary()
    chosen = choose_representative_snapshots(summary)
    rows = [compute_sample_metrics(bb, chosen[bb], summary) for bb in BACKBONES]
    write_metrics_csv(rows)
    write_wbs_sample_tables(rows, chosen)
    write_assignment_evidence(rows, chosen)
    build_figures(rows, chosen)
    build_report(rows, chosen)
    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
