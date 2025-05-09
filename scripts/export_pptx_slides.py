"""Export selected presentation slides to PNG via PowerPoint COM (Windows).

Requires Microsoft PowerPoint + fonts used in the deck (run check_pptx_fonts.py first).
Manual "Save as Picture" in PowerPoint produces the same result when fonts are installed.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PPTX = ROOT / "presentation_source.pptx"
OUT = ROOT / "docs" / "assets" / "slides"

# Slides used in README (1-based)
README_SLIDES = {
    1: "hero_title.png",
    10: "pipeline_overview.png",
    36: "project_output.png",
    37: "human_evaluation.png",
}

# Extra slides (optional)
EXTRA_SLIDES = {
    27: "key_result.png",
    40: "conclusion.png",
}

# 16:9 slide at ~2x HD; increase to (3840, 2160) for retina README assets
EXPORT_W = 2560
EXPORT_H = 1440


def _preflight() -> None:
    checker = ROOT / "scripts" / "check_pptx_fonts.py"
    if checker.exists():
        rc = subprocess.call([sys.executable, str(checker)])
        if rc != 0:
            print("Aborting export — fix missing fonts first.", file=sys.stderr)
            sys.exit(rc)


def _load_patch_fn():
    import importlib.util

    patch_script = ROOT / "scripts" / "patch_pptx_for_readme.py"
    spec = importlib.util.spec_from_file_location("patch_pptx_for_readme", str(patch_script))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.patch_presentation


def main():
    if not PPTX.exists():
        print(f"missing: {PPTX}", file=sys.stderr)
        sys.exit(1)

    _preflight()
    OUT.mkdir(parents=True, exist_ok=True)

    export_pptx = PPTX
    patch_script = ROOT / "scripts" / "patch_pptx_for_readme.py"
    if patch_script.exists():
        export_pptx = _load_patch_fn()(PPTX)
        print("using patched deck", export_pptx)

    try:
        import win32com.client  # type: ignore
    except ImportError:
        print("pywin32 not installed", file=sys.stderr)
        sys.exit(1)

    app = win32com.client.Dispatch("PowerPoint.Application")
    pres = app.Presentations.Open(str(export_pptx.resolve()), WithWindow=False)
    try:
        for num, name in {**README_SLIDES, **EXTRA_SLIDES}.items():
            out_path = str((OUT / name).resolve())
            pres.Slides(num).Export(out_path, "PNG", EXPORT_W, EXPORT_H)
            print("exported", name, f"({EXPORT_W}x{EXPORT_H})")
    finally:
        pres.Close()
        app.Quit()

    crop_script = ROOT / "scripts" / "crop_slide_export.py"
    if crop_script.exists():
        subprocess.call([sys.executable, str(crop_script)])


if __name__ == "__main__":
    main()
