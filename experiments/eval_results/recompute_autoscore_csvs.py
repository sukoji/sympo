"""Recompute AutoScore v2 columns in archived summary CSV files.

This updates only derived autoscore columns. Raw experimental metrics, judge
scores, timings, and condition metadata are preserved.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_results.autoscore_recompute import recompute_autoscore


RAW_SIGNAL_COLUMNS = {
    "success_rate",
    "mece_score",
    "granularity_fitness",
    "faithfulness",
    "planning_score",
    "buffer_ratio_pct",
    "workload_gini",
    "schedule_feasibility",
    "comm_efficiency",
}

AUTOSCORE_COLUMNS = [
    "autoscore_final",
    "autoscore_applicable",
    "autoscore_quality",
    "autoscore_allocation",
    "autoscore_orchestration",
    "autoscore_version",
    "autoscore_faithfulness_gate",
    "autoscore_na_cats",
]


def _has_signal(row: Dict[str, str]) -> bool:
    return any(str(row.get(col, "")).strip() not in {"", "NA", "N/A"} for col in RAW_SIGNAL_COLUMNS)


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _fieldnames(rows: List[Dict[str, str]]) -> List[str]:
    names: List[str] = list(rows[0].keys()) if rows else []
    for col in AUTOSCORE_COLUMNS:
        if col not in names:
            names.append(col)
    return names


def recompute_file(path: Path, dry_run: bool = False) -> Dict[str, int | str]:
    rows = _read_rows(path)
    if not rows:
        return {"path": str(path), "rows": 0, "updated": 0, "skipped": 0}

    updated = 0
    skipped = 0
    out_rows: List[Dict[str, str]] = []
    for row in rows:
        item = dict(row)
        if not _has_signal(item):
            skipped += 1
            out_rows.append(item)
            continue
        scored = recompute_autoscore(item)
        item.update({
            "autoscore_final": scored["autoscore"],
            "autoscore_applicable": scored["autoscore_applicable"],
            "autoscore_quality": scored["quality"],
            "autoscore_allocation": scored["allocation"],
            "autoscore_orchestration": scored["orchestration"],
            "autoscore_version": scored["version"],
            "autoscore_faithfulness_gate": "" if scored["faithfulness_gate"] is None else scored["faithfulness_gate"],
            "autoscore_na_cats": ",".join(scored.get("na_categories", []) or []),
        })
        updated += 1
        out_rows.append(item)

    if updated and not dry_run:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_fieldnames(out_rows), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(out_rows)

    return {"path": str(path), "rows": len(rows), "updated": updated, "skipped": skipped}


def iter_summary_csvs(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("summary*.csv")):
        if "_legacy" in path.parts:
            continue
        yield path


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute archived AutoScore columns in summary CSV files.")
    parser.add_argument("paths", nargs="*", type=Path, help="CSV files or directories. Defaults to eval_results.")
    parser.add_argument("--dry-run", action="store_true", help="Report what would change without writing files.")
    args = parser.parse_args()

    targets: List[Path] = []
    roots = args.paths or [Path("eval_results")]
    for root in roots:
        if root.is_dir():
            targets.extend(iter_summary_csvs(root))
        elif root.suffix == ".csv":
            targets.append(root)

    total_rows = total_updated = total_skipped = 0
    for path in sorted(set(targets)):
        result = recompute_file(path, dry_run=args.dry_run)
        total_rows += int(result["rows"])
        total_updated += int(result["updated"])
        total_skipped += int(result["skipped"])
        if result["updated"]:
            print(f"{result['path']}: updated={result['updated']} skipped={result['skipped']}")

    mode = "dry-run" if args.dry_run else "written"
    print(f"{mode}: files={len(set(targets))} rows={total_rows} updated={total_updated} skipped={total_skipped}")


if __name__ == "__main__":
    main()
