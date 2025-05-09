"""Patch presentation slides for README export (spacing, labels, bullets)."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "presentation_source.pptx"


def _clear_bullets(paragraph) -> None:
    p_pr = paragraph._p.get_or_add_pPr()
    for child in list(p_pr):
        if child.tag in (
            qn("a:buChar"),
            qn("a:buAutoNum"),
            qn("a:buBlip"),
            qn("a:buFont"),
            qn("a:buClr"),
            qn("a:buSzPct"),
            qn("a:buSzPts"),
        ):
            p_pr.remove(child)


def _plain_bullet(shape, text: str) -> None:
    tf = shape.text_frame
    tf.margin_left = Pt(20)
    tf.margin_right = Pt(8)
    tf.word_wrap = True
    para = tf.paragraphs[0]
    _clear_bullets(para)
    para.level = 0
    para.text = f"• {text.lstrip('• ').strip()}"


def _patch_hero(slide) -> None:
    to_remove = []
    sympo_shape = None
    title_shape = None

    for sh in slide.shapes:
        if sh.shape_type == MSO_SHAPE_TYPE.PICTURE and sh.top > int(Inches(7.5)):
            to_remove.append(sh._element)
            continue
        if not sh.has_text_frame:
            continue
        text = sh.text_frame.text.strip()
        if text == "A3" or "박선민" in text:
            to_remove.append(sh._element)
        elif text == "SYMPO":
            sympo_shape = sh
        elif "Multi-Agent Orchestration" in text:
            title_shape = sh

    for el in to_remove:
        el.getparent().remove(el)

    if sympo_shape is not None:
        sympo_shape.width = int(Inches(11.2))

    if title_shape is not None:
        title_shape.left = int(Inches(12.8))
        title_shape.width = int(Inches(13.8))


def _patch_pipeline(slide) -> None:
    single_line = {
        "WBS 생성\n에이전트": ("WBS 생성 에이전트", 2.55),
        "태스크 관리\n에이전트": ("태스크 관리 에이전트", 2.55),
        "프로젝트\n명세서": ("프로젝트 명세서", 2.35),
        "팀원\n에이전트": ("팀원 에이전트", 2.2),
    }
    widen_only = {
        "태스크별 성향\u00a0기반\n팀원\u00a0에이전트 호출": 5.2,
        "계층형 프로젝트\u00a0\nWBS 초안 생성": 5.2,
        "회의 음성 및\u00a0\n프로젝트 명세서 입력": 4.8,
        "팀원 에이전트 간\n토의 후 피드백 요청": 4.8,
        "피드백 반영 후 업무\u00a0\n분배된 최종 WBS 도출": 5.0,
    }

    for sh in slide.shapes:
        if not sh.has_text_frame:
            continue
        text = sh.text_frame.text
        if text in single_line:
            new_text, w = single_line[text]
            sh.text_frame.text = new_text
            sh.width = int(Inches(w))
        elif text in widen_only:
            sh.width = int(Inches(widen_only[text]))
            tf = sh.text_frame
            tf.margin_left = Pt(4)
            tf.margin_right = Pt(4)


def _patch_outputs(slide) -> None:
    for sh in slide.shapes:
        if not sh.has_text_frame:
            continue
        text = sh.text_frame.text.strip()
        if not text:
            continue

        if text in ("WBS 최종본 산출 구조", "일정 시각화 특징"):
            sh.width = int(Inches(5.5))
            sh.height = int(Inches(0.85))
            sh.text_frame.margin_bottom = Pt(10)
            sh.text_frame.word_wrap = False
            continue

        if text.startswith(("3단계", "R&R", "프로젝트 메트릭", "직관적", "태스크 연결성")):
            _plain_bullet(sh, text)
            sh.width = int(Inches(11.6))
            sh.top += int(Inches(0.14))


def patch_presentation(src: Path = SOURCE) -> Path:
    if not src.exists():
        raise FileNotFoundError(src)

    tmp = Path(tempfile.mkdtemp(prefix="sympo_pptx_"))
    out = tmp / "readme_export.pptx"
    shutil.copy2(src, out)

    prs = Presentation(str(out))
    _patch_hero(prs.slides[0])
    _patch_pipeline(prs.slides[9])
    _patch_outputs(prs.slides[35])
    prs.save(str(out))
    return out


if __name__ == "__main__":
    path = patch_presentation()
    print("patched", path)
