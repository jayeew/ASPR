"""Editable vector components using the same scene primitives as Figures 1 and 2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from figure_pipeline.fig1_reference.export import convert, fonts
from figure_pipeline.fig1_reference.svg import Scene, measure

from .settings import COLORS, LABELS, METHODS, OUTPUT, PANELS, STUDY, read, roster

INK = "#172339"
NAVY = "#142f99"
LINE = "#9bcfeb"
PURPLE = "#8b2be2"


def text(s: Scene, x: float, y: float, value: str, size: float = 21, color: str = INK, bold: bool = False, anchor: str = "start") -> None:
    s.text(x, y, value, size, color, bold, anchor=anchor)


def wrap(s: Scene, x: float, y: float, value: str, width: float, size: float = 20, color: str = INK, bold: bool = False) -> None:
    line = ""
    for word in value.split():
        candidate = (line + " " + word).strip()
        if line and measure(candidate, size, bold) > width:
            text(s, x, y, line, size, color, bold)
            y += size * 1.16
            line = word
        else:
            line = candidate
    if line:
        text(s, x, y, line, size, color, bold)


def component(ident: str, width: float, height: float, title: str) -> Scene:
    s = Scene(width, height, ident)
    s.rect(0.8, 0.8, width - 1.6, height - 1.6, "#ffffff", "#c9e7f6", 6, sw=1)
    text(s, 13, 29, title, 24, NAVY, True)
    return s


def panel(ident: str, title: str) -> Scene:
    _, _, width, height = PANELS[ident]
    s = Scene(width, height, "panel_" + ident)
    s.rect(1, 1, width - 2, height - 2, "url(#bluewash)", LINE, 7, sw=1.4)
    text(s, 18, 42, ident, 47, "#111111", True)
    text(s, 84, 39, title, 36, NAVY, True)
    return s


def document_icon(s: Scene, x: float, y: float, color: str = PURPLE) -> None:
    s.rect(x, y, 37, 47, "#f8f5ff", color, 3, sw=2.4)
    for n, width in enumerate((21, 24, 20, 13)):
        s.line(x + 7, y + 13 + 8 * n, x + 7 + width, y + 13 + 8 * n, color, 2)


def scope() -> Scene:
    s = component("a1_scope", 404, 336, "Study scope and data")
    papers = roster()
    claims = sum(len(read(STUDY / "papers" / p["paper_id"] / "shared/claims.json")["claims"]) for p in papers)
    refs = [read(STUDY / "human_refs" / f"{p['paper_id']}.json") for p in papers]
    retained = [[r for r in p["references"] if r["use_in_main"]] for p in refs]
    n_ref = sum(bool(r) for r in retained)
    n_a = sum(any(r["tier"] == "A" for r in refs) for refs in retained)
    for y, label, subtitle in ((54, f"{len(papers)} papers", "Fixed original study cohort"), (161, f"{claims:,} shared anchors", "Aligned contribution identities")):
        s.rect(13, y, 378, 96, "#f9fcff", "#c5e4f7", 5, sw=1)
        document_icon(s, 29, y + 21)
        text(s, 83, y + 38, label, 28, NAVY, True)
        text(s, 83, y + 67, subtitle, 19)
    reports=sum((OUTPUT/"data/reports"/m/f"{p['paper_id']}.json").exists() for p in papers for m in METHODS)
    text(s, 16, 279, f"{reports:,} / 1,200 matched-task reports", 19, INK, True)
    text(s, 16, 304, f"{n_ref} papers with retained references", 19, NAVY)
    text(s, 16, 327, f"{n_a} with explicit innovation appraisals", 19, NAVY)
    return s


def permissions() -> Scene:
    s = component("a2_methods", 780, 336, "Methods and accessible information")
    widths = (126, 108, 140, 135, 124, 127)
    boundaries = [10]
    for width in widths:
        boundaries.append(boundaries[-1] + width)
    labels = ("Method", "Manuscript", "Prior-work retrieval", "Graph context", "Dedicated joint", "Held-out labels")
    for j, label in enumerate(labels):
        wrap(s, boundaries[j] + 5, 60, label, widths[j] - 9, 19, NAVY, True)
    access = ((1,0,0,0,0), (1,0,0,0,0), (1,0,0,0,0), (1,1,0,0,0), (1,0,1,1,0), (1,1,1,1,0))
    for i, method in enumerate(METHODS):
        y = 115 + 35 * i
        s.rect(10, y - 23, 760, 35, "#f0e8ff" if method == "fusion" else ("#f5faff" if i % 2 == 0 else "#ffffff"))
        text(s, 15, y + 2, LABELS[i], 19, INK if i < 3 else COLORS[method], True)
        for j, value in enumerate(access[i], 1):
            mid = (boundaries[j] + boundaries[j+1]) / 2
            if value:
                s.path(f"M{mid-7},{y-6} l5,6 l10,-16", "none", INK if i < 3 else COLORS[method], 2.5)
            else:
                text(s, mid, y, "–", 22, INK, anchor="middle")
    for x in boundaries:
        s.line(x, 40, x, 303, "#cfdeef", 0.8)
    for y in (92,127,162,197,232,267,302):
        s.line(10,y,770,y,"#cfdeef",0.8)
    return s


def channels() -> Scene:
    s = component("a3_channels", 500, 336, "Evaluation channels")
    entries = (("Rubric assessment", "LLM-based, source-grounded"), ("Source checking", "Atomic assertions and scope"),
               ("Published reviews", "Original reviewer histories"), ("Independent preference", "Separate judge; both orders"))
    for i, (title, subtitle) in enumerate(entries):
        x, y = 12 + (i % 2) * 242, 48 + (i // 2) * 139
        s.rect(x,y,234,128,"#fbfdff","#c9e7f6",5,sw=1)
        document_icon(s,x+12,y+17,"#3264d4" if i > 1 else PURPLE)
        wrap(s,x+60,y+33,title,162,22,NAVY,True)
        wrap(s,x+12,y+91,subtitle,210,19)
    return s


def separation() -> Scene:
    s = component("a4_separation", 306, 336, "Information separation")
    s.rect(12,49,282,81,"#eef6ff","#c3ddef",5)
    wrap(s,26,78,"Method-specific generation inputs",252,22,NAVY,True)
    text(s,153,171,"≠",47,"#ca4344",True,"middle")
    s.rect(12,190,282,117,"#f1eaff","#d0c0ef",5)
    wrap(s,26,220,"Reviewer judgments and evaluation labels held out from generation",250,21,PURPLE,True)
    return s


def save_panel(ident: str, title: str, pieces: list[tuple[Scene,float,float]]) -> None:
    s = panel(ident,title)
    for child,x,y in pieces:
        child.save(OUTPUT / "components" / f"{child.ident}.svg")
        convert(OUTPUT / "components" / f"{child.ident}.svg", 2)
        s.use(child,x,y)
    s.save(OUTPUT / "panels" / f"{ident}.svg")
    convert(OUTPUT / "panels" / f"{ident}.svg", 2)


def render_a() -> None:
    fonts(OUTPUT)
    save_panel("a", "Evaluation design and information boundaries", [(scope(),12,56),(permissions(),424,56),(channels(),1212,56),(separation(),1720,56)])


if __name__ == "__main__":
    render_a()
