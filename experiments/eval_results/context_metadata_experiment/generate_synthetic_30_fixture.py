"""Generate clearly labeled synthetic 30-run fixtures for pipeline tests.

This does not represent executed experiments. Every output file contains
synthetic markers so it cannot be confused with real evaluation artifacts.
"""
from __future__ import annotations

import csv
import json
import math
import random
from datetime import datetime
from pathlib import Path


ROOT = Path("/home/piai/ai_course/agent_test/eval_results/context_metadata_experiment")
OUT = ROOT / "synthetic_30run_fixture"
SNAP = OUT / "snapshots"
LOGS = OUT / "logs"
FIG = OUT / "figures"

RANDOM_SEED = 260427
MODES = {
    "M_resume": {
        "label": "Resume only",
        "structure": (0.755, 0.035),
        "assignment": (0.385, 0.045),
        "debate": (0.790, 0.070),
        "tasks": (35, 2),
    },
    "M_disc": {
        "label": "eDISC only",
        "structure": (0.745, 0.040),
        "assignment": (0.405, 0.055),
        "debate": (0.840, 0.060),
        "tasks": (35, 2),
    },
    "M_both": {
        "label": "Resume + eDISC",
        "structure": (0.790, 0.030),
        "assignment": (0.485, 0.045),
        "debate": (0.880, 0.045),
        "tasks": (36, 2),
    },
}

WEIGHTS = {"S": 0.40, "A": 0.35, "D": 0.25}


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def sample_score(mean: float, sd: float) -> float:
    return round(clamp(random.gauss(mean, sd)), 3)


def median3(values: list[float]) -> float:
    return sorted(values)[1]


def synthetic_tasks(mode: str, run_id: int, n_tasks: int) -> list[dict]:
    roles = ["Planner", "Backend Developer", "Frontend Developer", "Data Engineer", "QA Engineer"]
    members = ["SYN-MBR-001", "SYN-MBR-002", "SYN-MBR-003", "SYN-MBR-004", "SYN-MBR-005"]
    tasks = [
        {
            "task_id": "SYN-L1-01",
            "title": "SYNTHETIC root phase for fixture testing",
            "level": "L1",
            "parent_id": None,
            "estimated_days": 20.0,
            "buffer_days": 2.0,
            "assigned_role": "",
            "assigned_to": [],
            "dependencies": [],
            "synthetic": True,
        },
        {
            "task_id": "SYN-L2-01-01",
            "title": "SYNTHETIC work package for fixture testing",
            "level": "L2",
            "parent_id": "SYN-L1-01",
            "estimated_days": 20.0,
            "buffer_days": 2.0,
            "assigned_role": "",
            "assigned_to": [],
            "dependencies": [],
            "synthetic": True,
        },
    ]
    for i in range(1, n_tasks + 1):
        tasks.append(
            {
                "task_id": f"SYN-L3-{run_id:02d}-{i:02d}",
                "title": f"SYNTHETIC task {i:02d} for {mode} fixture testing",
                "level": "L3",
                "parent_id": "SYN-L2-01-01",
                "estimated_days": float(1 + (i % 5)),
                "buffer_days": float(i % 2),
                "assigned_role": roles[(i + run_id) % len(roles)],
                "assigned_to": [members[(i + run_id) % len(members)]],
                "dependencies": [] if i == 1 else [f"SYN-L3-{run_id:02d}-{i-1:02d}"],
                "synthetic": True,
            }
        )
    return tasks


def synthetic_debate(mode: str, run_id: int) -> list[dict]:
    return [
        {
            "speaker": agent,
            "content": (
                f"SYNTHETIC fixture message for {mode} run {run_id}; "
                "not produced by the real multi-agent system."
            ),
            "round": idx + 1,
            "synthetic": True,
        }
        for idx, agent in enumerate(["planner", "backend", "qa", "supervisor"])
    ]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    random.seed(RANDOM_SEED)
    SNAP.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    rows = []
    for mode, cfg in MODES.items():
        for run_id in range(1, 31):
            n_tasks = max(28, int(round(random.gauss(*cfg["tasks"]))))
            s_trials = [sample_score(*cfg["structure"]) for _ in range(3)]
            a_trials = [sample_score(*cfg["assignment"]) for _ in range(3)]
            d_trials = [sample_score(*cfg["debate"]) for _ in range(3)]
            s_med = median3(s_trials)
            a_med = median3(a_trials)
            d_med = median3(d_trials)
            overall = round(
                s_med * WEIGHTS["S"] + a_med * WEIGHTS["A"] + d_med * WEIGHTS["D"], 4
            )

            snapshot_name = f"SYNTHETIC_wbs_snapshot_{mode}_r{run_id:02d}.json"
            snapshot = {
                "synthetic": True,
                "synthetic_notice": (
                    "Pipeline test fixture only. This is not an executed experiment result."
                ),
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "condition": mode,
                "run_id": run_id,
                "backend": "synthetic-fixture",
                "wbs_tasks": synthetic_tasks(mode, run_id, n_tasks),
                "debate_log": synthetic_debate(mode, run_id),
                "scores": {
                    "structure": s_med,
                    "assignment": a_med,
                    "debate": d_med,
                    "overall": overall,
                },
            }
            (SNAP / snapshot_name).write_text(
                json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            rows.append(
                {
                    "synthetic": True,
                    "mode": mode,
                    "run_id": run_id,
                    "snapshot": snapshot_name,
                    "n_tasks": n_tasks,
                    "S_t1": s_trials[0],
                    "S_t2": s_trials[1],
                    "S_t3": s_trials[2],
                    "S_med": s_med,
                    "A_t1": a_trials[0],
                    "A_t2": a_trials[1],
                    "A_t3": a_trials[2],
                    "A_med": a_med,
                    "D_t1": d_trials[0],
                    "D_t2": d_trials[1],
                    "D_t3": d_trials[2],
                    "D_med": d_med,
                    "overall_median": overall,
                    "fixture_note": "SYNTHETIC_TEST_FIXTURE_NOT_REAL_EXPERIMENT",
                }
            )

    write_csv(OUT / "summary_judged_synthetic_30.csv", rows)

    aggregates = []
    for mode in MODES:
        mode_rows = [r for r in rows if r["mode"] == mode]
        agg = {"synthetic": True, "mode": mode, "n": len(mode_rows)}
        for key in ("S_med", "A_med", "D_med", "overall_median"):
            vals = [float(r[key]) for r in mode_rows]
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
            ci95 = 1.96 * math.sqrt(var) / math.sqrt(len(vals))
            agg[f"{key}_mean"] = round(mean, 4)
            agg[f"{key}_sd"] = round(math.sqrt(var), 4)
            agg[f"{key}_ci95"] = round(ci95, 4)
        aggregates.append(agg)
    write_csv(OUT / "aggregate_synthetic_30.csv", aggregates)

    log_lines = [
        "SYNTHETIC TEST FIXTURE LOG",
        "This log was generated for pipeline testing only.",
        "No model inference, judge call, or real experiment execution occurred.",
        f"generated_at={datetime.now().isoformat(timespec='seconds')}",
        f"random_seed={RANDOM_SEED}",
        "conditions=M_resume,M_disc,M_both",
        "runs_per_condition=30",
        "bias_note=M_both parameters intentionally set higher for positive-control testing.",
    ]
    (LOGS / "synthetic_generation.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    report = [
        "# Synthetic 30-Run Context Metadata Fixture",
        "",
        "> This folder is a synthetic test fixture, not executed experiment evidence.",
        "",
        "Purpose: stress-test report/figure pipelines with 30 rows per condition.",
        "",
        "| Mode | N | Overall mean | Assignment mean | Structure mean | Debate mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        report.append(
            "| {mode} | {n} | {overall_median_mean:.3f} | {A_med_mean:.3f} | "
            "{S_med_mean:.3f} | {D_med_mean:.3f} |".format(**row)
        )
    report.extend(
        [
            "",
            "Interpretation for tests only: parameters were chosen so M_both is the positive-control winner.",
            "Do not cite this as an empirical result.",
        ]
    )
    (OUT / "README_SYNTHETIC.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"saved synthetic fixture: {OUT}")


if __name__ == "__main__":
    main()
