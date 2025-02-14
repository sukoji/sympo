"""Export selected presentation slides to PNG via PowerPoint COM (Windows)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PPTX = ROOT / "presentation_source.pptx"
OUT = ROOT / "docs" / "assets" / "slides"

# Slides to export (1-based): title, pipeline overview, phase cards, results
SLIDES = {
    1: "hero_title.png",
    10: "pipeline_overview.png",
    36: "project_output.png",
    37: "human_evaluation.png",
    27: "key_result.png",
    40: "conclusion.png",
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        import win32com.client  # type: ignore
    except ImportError:
        print("pywin32 not installed", file=sys.stderr)
        sys.exit(1)

    app = win32com.client.Dispatch("PowerPoint.Application")
    pres = app.Presentations.Open(str(PPTX.resolve()), WithWindow=False)
    try:
        for num, name in SLIDES.items():
            out_path = str((OUT / name).resolve())
            pres.Slides(num).Export(out_path, "PNG", 1920, 1080)
            print("exported", name)
    finally:
        pres.Close()
        app.Quit()


if __name__ == "__main__":
    main()
