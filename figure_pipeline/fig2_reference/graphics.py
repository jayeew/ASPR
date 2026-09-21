"""Editable SVG primitives and a component registry shared across five panels."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from figure_pipeline.fig1_reference.svg import Scene as BaseScene
from figure_pipeline.fig1_reference.svg import el, measure

from .data import read, write

INK = "#17202e"
BLUE = "#1269df"
NAVY = "#162a96"
ORANGE = "#e16c35"
RED = "#a93226"
CYAN = "#169dbb"
PURPLE = "#713bcb"
MUTED = "#202936"
LINE = "#b9def3"
COMMUNITIES = [
    "#398bea",
    "#fa993b",
    "#54a76a",
    "#eb596e",
    "#ab7bd2",
    "#e6b549",
    "#30b8c5",
    "#e985b6",
]
CLAIM_COLORS = [BLUE, ORANGE, "#239d72", "#d847a4"]


def text_size(size: float) -> float:
    """Enlarge body copy while retaining the reference's major heading scale."""
    if size < 10:
        return max(8.5, size * 1.24)
    return size * 1.12 if size < 14 else size


def text_color(color: str) -> str:
    if color in {"white", "#ffffff", "#fff"}:
        return color
    if color in {INK, MUTED, "#101018", "#0c1020"}:
        return color
    if color.startswith("#") and len(color) == 7:
        channels = [int(color[i : i + 2], 16) for i in (1, 3, 5)]
        return "#" + "".join(f"{round(v * 0.72):02x}" for v in channels)
    return color


class Scene(BaseScene):
    """Figure-2 typography policy; Figure 1's shared primitives stay untouched."""

    def text(
        self,
        x: float,
        y: float,
        value: str,
        size: float = 12,
        color: str = INK,
        bold: bool = False,
        italic: bool = False,
        anchor: str = "start",
        sans: bool = False,
    ) -> None:
        self.label(x, y, value, text_size(size), color, bold, italic, anchor, sans)

    def label(
        self,
        x: float,
        y: float,
        value: str,
        size: float,
        color: str = INK,
        bold: bool = False,
        italic: bool = False,
        anchor: str = "start",
        sans: bool = False,
    ) -> None:
        super().text(x, y, value, size, text_color(color), bold, italic, anchor, sans)

    def paragraph(
        self,
        x: float,
        y: float,
        value: str,
        width: float,
        size: float = 12,
        leading: float = 14,
        bold: bool = False,
        italic: bool = False,
        color: str = INK,
        max_lines: int | None = None,
    ) -> float:
        lines: list[str] = []
        for part in value.split("\n"):
            line = ""
            for word in part.split():
                candidate = f"{line} {word}".strip()
                if line and measure(candidate, size, bold, italic) > width:
                    lines.append(line)
                    line = word
                else:
                    line = candidate
            lines.append(line)
        if max_lines is not None and len(lines) > max_lines:
            raise ValueError(f"{self.ident}: paragraph exceeds {max_lines} lines")
        for i, line in enumerate(lines):
            self.label(x, y + i * leading, line, size, color, bold, italic)
        return y + len(lines) * leading


def fit(
    s: Scene,
    x: float,
    y: float,
    value: str,
    width: float,
    size: float = 11,
    color: str = INK,
    bold: bool = False,
) -> None:
    size = text_size(size)
    actual = min(size, size * width / max(measure(value, size, bold), 1))
    s.label(x, y, value, actual, color, bold)


def para(
    s: Scene,
    x: float,
    y: float,
    value: str,
    width: float,
    size: float = 8.7,
    color: str = INK,
    bold: bool = False,
    leading: float | None = None,
) -> float:
    size = text_size(size)
    return s.paragraph(
        x, y, value, width, size, leading or size * 1.2, bold=bold, color=color
    )


def box(
    s: Scene,
    title: str = "",
    tint: str = "#f4fbff",
    stroke: str = LINE,
    color: str = NAVY,
    title_size: float = 10.5,
) -> None:
    s.rect(0.4, 0.4, s.width - 0.8, s.height - 0.8, "#ffffff", stroke, 3, sw=0.55)
    if title:
        s.rect(0.8, 0.8, s.width - 1.6, 21, tint, radius=3)
        fit(s, 6, 14.5, title, s.width - 12, title_size, color, True)


def badge(
    s: Scene,
    x: float,
    y: float,
    text: str,
    color: str = BLUE,
    width: float = 30,
    height: float = 17,
    size: float = 9,
) -> None:
    s.rect(x, y, width, height, color, radius=3)
    s.text(
        x + width / 2, y + height * 0.72, text, size, "#ffffff", True, anchor="middle"
    )


def icon(
    s: Scene, kind: str, x: float, y: float, size: float = 25, color: str = BLUE
) -> None:
    q = size / 24
    a = Scene(24, 28, f"{s.ident}-{kind}-{x:g}-{y:g}")
    if kind == "doc":
        a.path("M4 1 H15 L21 7 V27 H4 Z", "#ffffff", color, 1.6)
        a.path("M15 1 V7 H21", "none", color, 1.2)
        for yy, xx in [(11, 17), (14, 18), (17, 17), (20, 15), (23, 18)]:
            a.line(7, yy, xx, yy, color, 0.85)
    elif kind == "db":
        for yy in [15, 10, 5]:
            a.path(
                f"M2 {yy} V{yy + 5} C2 {yy + 10} 22 {yy + 10} 22 {yy + 5} V{yy} Z",
                color,
                "#ffffff",
                0.7,
            )
            a.add(
                el(
                    "ellipse",
                    cx=12,
                    cy=yy,
                    rx=10,
                    ry=4,
                    fill=color,
                    stroke="#ffffff",
                    stroke_width=0.8,
                )
            )
    elif kind == "search":
        a.add(
            el("circle", cx=10, cy=10, r=7, fill="none", stroke=color, stroke_width=1.7)
        )
        a.line(15, 16, 22, 24, color, 2)
    elif kind == "shield":
        a.path("M2 3 Q12 7 22 3 V12 Q21 22 12 27 Q3 22 2 12 Z", "#f4f9ff", color, 1.5)
        a.path("M7 14 L11 18 L18 10", "none", color, 1.7)
    elif kind == "bulb":
        a.path(
            "M8 20 C8 15 3 15 3 9 C3 -1 21 -1 21 9 C21 15 16 15 16 20 Z",
            "#fff6d7",
            color,
            1.2,
        )
        a.line(8, 23, 16, 23, color, 1.5)
        a.line(10, 26, 14, 26, color, 1.5)
    elif kind == "bars":
        for xx, hh in [(4, 10), (11, 19), (18, 25)]:
            a.rect(xx, 27 - hh, 3.5, hh, color, radius=1)
    elif kind == "plus":
        a.line(3, 13, 22, 13, color, 2)
        a.line(12, 3, 12, 24, color, 2)
    elif kind == "clock":
        a.add(
            el(
                "circle",
                cx=12,
                cy=13,
                r=10,
                fill="none",
                stroke=color,
                stroke_width=1.3,
            )
        )
        a.path("M12 5 V13 L17 16", "none", color, 1.5)
    elif kind == "lock":
        a.rect(5, 12, 15, 14, color, radius=2)
        a.path("M8 12 V7 Q12 -1 17 7 V12", "none", color, 2)
        a.line(12.5, 17, 12.5, 22, "white", 1.5)
    elif kind == "gear":
        pts = [
            (
                12 + (12 if i % 4 < 2 else 9) * math.cos(i * math.pi / 16),
                14 + (12 if i % 4 < 2 else 9) * math.sin(i * math.pi / 16),
            )
            for i in range(32)
        ]
        a.add(
            el("polygon", points=" ".join(f"{xx},{yy}" for xx, yy in pts), fill=color)
        )
        a.add(el("circle", cx=12, cy=14, r=5, fill="white"))
    elif kind == "warning":
        a.path("M12 1 L24 25 H0 Z", "#e859b7")
        a.text(12, 22, "!", 19, "white", True, anchor="middle")
    elif kind == "link":
        a.path(
            "M10 18 L6 22 Q-2 22 1 15 L8 8 Q14 5 16 11 M9 16 L15 10 M10 7 L14 3 Q23 0 22 8 L16 15 Q10 19 7 13",
            "none",
            color,
            2,
        )
    else:
        pts = [(12, 2), (3, 14), (19, 12), (12, 25)]
        for i, j in [(0, 1), (0, 2), (1, 3), (2, 3), (1, 2)]:
            a.line(*pts[i], *pts[j], color, 1)
        for xx, yy in pts:
            a.node(xx, yy, fill="#6dc8fa", radius=3.2, stroke=color, sw=1)
    s.use(a, x, y, q)


class Figure:
    """One local coordinate system per panel and one editable file per subfigure."""

    def __init__(self, out: Path) -> None:
        self.out = out
        self.panels: dict[str, Scene] = {}
        self.components: dict[str, Scene] = {}
        self.manifest: list[dict[str, Any]] = []
        self.boxes = {
            "a": [10, 49, 1035, 584],
            "b": [12, 644, 500, 442],
            "c": [524, 644, 520, 442],
            "d": [12, 1098, 500, 344],
            "e": [524, 1098, 520, 344],
        }

    def panel(
        self, ident: str, title: str, color: str = NAVY, tint: str = "#eef9ff"
    ) -> Scene:
        _, _, width, height = self.boxes[ident]
        s = Scene(width, height, f"panel_{ident}")
        s.rect(
            0.5,
            0.5,
            width - 1,
            height - 1,
            "#ffffff",
            LINE if ident != "b" else "#f1caac",
            4,
            sw=0.8,
        )
        s.rect(1, 1, width - 2, 33, tint, radius=4)
        s.text(8, 26, ident, 28, "#101018", True, sans=True)
        fit(s, 42, 24, title, width - 51, 21, color, True)
        self.panels[ident] = s
        return s

    def component(
        self,
        panel: str,
        ident: str,
        x: float,
        y: float,
        scene: Scene,
        title: str,
        sources: list[str] | None = None,
    ) -> None:
        for child in scene.root.iter():
            old_id = child.get("id")
            if old_id and child is not scene.root:
                child.set("id", f"{ident}--{old_id}")
        scene.ident = ident
        scene.root.set("id", ident)
        scene.root.set("data-component", ident)
        self.components[ident] = scene
        self.panels[panel].use(scene, x, y)
        self.manifest.append(
            {
                "id": ident,
                "panel": panel,
                "title": title,
                "x": x,
                "y": y,
                "width": scene.width,
                "height": scene.height,
                "sources": sources or ["data/snapshot.json"],
                "svg": f"components/{ident}.svg",
                "layer": f"components/layers/{ident}.svg",
            }
        )

    def save(self) -> None:
        old_manifest = self.out / "layouts/components.json"
        if old_manifest.exists():
            retired = {r["id"] for r in read(old_manifest)} - set(self.components)
            for ident in retired:
                for ext in ["svg", "pdf", "png"]:
                    (self.out / "components" / f"{ident}.{ext}").unlink(missing_ok=True)
                (self.out / "components/layers" / f"{ident}.svg").unlink(
                    missing_ok=True
                )
        for ident, scene in self.components.items():
            scene.save(self.out / "components/layers" / f"{ident}.svg")
            scene.save(self.out / "components" / f"{ident}.svg")
        for ident, scene in self.panels.items():
            scene.save(self.out / "panels" / f"{ident}.svg")
            scene.save(self.out / "layouts/templates" / f"{ident}.svg")
        full = Scene(1055, 1491, "Fig2")
        full.rect(0, 0, 1055, 1491, "#ffffff")
        fit(
            full,
            8,
            31,
            "Fig. 2 | GEAR–Graph framework for evidence-based innovation analysis",
            1038,
            29,
            "#0c1020",
            True,
        )
        for ident, scene in self.panels.items():
            x, y, _, _ = self.boxes[ident]
            full.use(scene, x, y)
        full.save(self.out / "final/Fig2.svg", physical=True)
        full.save(self.out / "layouts/templates/Fig2.svg", physical=True)
        write(self.out / "layouts/components.json", self.manifest)
        write(
            self.out / "layouts/style.json",
            {
                "reference_size": [1055, 1491],
                "panel_boxes": self.boxes,
                "palette": {
                    "gear": ORANGE,
                    "graph": BLUE,
                    "joint": CYAN,
                    "fusion": PURPLE,
                },
                "community_colors": COMMUNITIES,
                "fonts": ["Times New Roman", "Arial"],
            },
        )
        write(self.out / "qa/text_boxes.json", full.text_boxes)
        write(
            self.out / "qa/component_text_boxes.json",
            {
                ident: {
                    "width": scene.width,
                    "height": scene.height,
                    "boxes": scene.text_boxes,
                }
                for ident, scene in self.components.items()
            },
        )
