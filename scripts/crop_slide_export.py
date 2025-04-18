"""Crop PPT slide PNG exports: remove deck chrome (blue header, chapter bar, footer)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SLIDES_DIR = ROOT / "docs" / "assets" / "slides"
README_DIR = ROOT / "docs" / "assets" / "readme"

# source filename -> cropped output filename
CROPS = {
    "hero_title.png": ("hero.png", True),
    "pipeline_overview.png": ("pipeline.png", False),
    "project_output.png": ("outputs.png", False),
    "human_evaluation.png": ("validation.png", False),
}


def _is_blue_header_row(row: np.ndarray) -> bool:
    r, g, b = row[:, 0].astype(float), row[:, 1].astype(float), row[:, 2].astype(float)
    mask = (b > 120) & (b > r + 15) & (b > g + 10)
    return bool(mask.mean() > 0.25)


def _is_grey_chapter_row(row: np.ndarray) -> bool:
    r, g, b = row[:, 0].astype(float), row[:, 1].astype(float), row[:, 2].astype(float)
    grey = (np.abs(r - g) < 25) & (np.abs(g - b) < 25) & (r > 120) & (r < 230)
    return bool(grey.mean() > 0.18)


def _is_mostly_white_row(row: np.ndarray) -> bool:
    return bool(row.mean() > 248)


def crop_top(arr: np.ndarray, *, skip_chapter_bar: bool = True) -> int:
    h = arr.shape[0]
    y = 0
    while y < h and _is_mostly_white_row(arr[y]):
        y += 1
    while y < h and _is_blue_header_row(arr[y]):
        y += 1
    while y < h and _is_mostly_white_row(arr[y]):
        y += 1
    if skip_chapter_bar:
        while y < h and _is_grey_chapter_row(arr[y]):
            y += 1
        while y < h and _is_mostly_white_row(arr[y]):
            y += 1
    return max(0, y - 6)


def crop_bottom(arr: np.ndarray) -> int:
    h = arr.shape[0]
    y = h - 1
    while y > 0 and _is_mostly_white_row(arr[y]):
        y -= 1
    while y > 0:
        dark = (arr[y] < 200).any(axis=1).mean()
        if dark > 0.008:
            y -= 1
            continue
        break
    return min(h, y + 24)


def crop_to_content(img: Image.Image, *, pad: int = 32) -> Image.Image:
    arr = np.array(img)
    mask = (arr < 242).any(axis=2)
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return img
    top, bottom = int(ys.min()), int(ys.max())
    left, right = int(xs.min()), int(xs.max())
    top = max(0, top - pad)
    left = max(0, left - pad)
    bottom = min(arr.shape[0] - 1, bottom + pad)
    right = min(arr.shape[1] - 1, right + pad)
    return img.crop((left, top, right + 1, bottom + 1))


def crop_image(path: Path, *, trim_footer: bool = False) -> Image.Image:
    img = Image.open(path).convert("RGB")
    arr = np.array(img)
    top = crop_top(arr)
    bottom = crop_bottom(arr) if trim_footer else arr.shape[0]
    if bottom - top < 80:
        return img
    cropped = img.crop((0, top, arr.shape[1], bottom))
    if trim_footer:
        # Title slide: drop footer + empty lower half before tight bbox
        arr2 = np.array(cropped)
        cropped = cropped.crop((0, 0, arr2.shape[1], int(arr2.shape[0] * 0.58)))
        cropped = crop_to_content(cropped, pad=56)
    return cropped


def main() -> None:
    README_DIR.mkdir(parents=True, exist_ok=True)
    for src_name, (out_name, trim_footer) in CROPS.items():
        src = SLIDES_DIR / src_name
        if not src.exists():
            print("skip missing", src.relative_to(ROOT))
            continue
        cropped = crop_image(src, trim_footer=trim_footer)
        out = README_DIR / out_name
        cropped.save(out, format="PNG", optimize=True)
        print("wrote", out.relative_to(ROOT), cropped.size)


if __name__ == "__main__":
    main()
