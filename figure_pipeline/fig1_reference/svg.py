"""Small editable SVG scene graph with explicit fonts and component boundaries."""

from __future__ import annotations

import copy
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from PIL import ImageFont

NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", NS)
INK = "#101d3b"
FONTS = Path("/mnt/c/Windows/Fonts")


def el(tag: str, **attrs: Any) -> ET.Element:
    return ET.Element(
        f"{{{NS}}}{tag}",
        {k.replace("_", "-"): str(v) for k, v in attrs.items() if v is not None},
    )


def font_path(bold: bool = False, italic: bool = False, sans: bool = False) -> Path:
    if sans:
        return FONTS / ("arialbd.ttf" if bold else "arial.ttf")
    return FONTS / (
        "timesbi.ttf"
        if bold and italic
        else "timesbd.ttf"
        if bold
        else "timesi.ttf"
        if italic
        else "times.ttf"
    )


def measure(
    value: str,
    size: float,
    bold: bool = False,
    italic: bool = False,
    sans: bool = False,
) -> float:
    font = ImageFont.truetype(str(font_path(bold, italic, sans)), round(size * 16))
    return float(font.getlength(value)) / 16


class Scene:
    def __init__(self, width: float, height: float, ident: str) -> None:
        self.width, self.height, self.ident = width, height, ident
        self.root = el("g", id=ident)
        self.text_boxes: list[dict[str, Any]] = []

    def add(self, node: ET.Element) -> ET.Element:
        self.root.append(node)
        return node

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str = "none",
        stroke: str = "none",
        radius: float = 0,
        opacity: float = 1,
        sw: float = 0.65,
    ) -> None:
        self.add(
            el(
                "rect",
                x=x,
                y=y,
                width=w,
                height=h,
                fill=fill,
                stroke=stroke,
                rx=radius,
                opacity=opacity,
                stroke_width=sw,
            )
        )

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        color: str = "#abb3bd",
        width: float = 0.7,
        dash: str | None = None,
        opacity: float = 1,
    ) -> ET.Element:
        return self.add(
            el(
                "line",
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                stroke=color,
                stroke_width=width,
                stroke_dasharray=dash,
                opacity=opacity,
            )
        )

    def path(
        self,
        d: str,
        fill: str = "none",
        stroke: str = "none",
        width: float = 0.7,
        dash: str | None = None,
        opacity: float = 1,
    ) -> None:
        self.add(
            el(
                "path",
                d=d,
                fill=fill,
                stroke=stroke,
                stroke_width=width,
                stroke_dasharray=dash,
                opacity=opacity,
            )
        )

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
        node = el(
            "text",
            x=x,
            y=y,
            font_family="Arial" if sans else "Times New Roman",
            font_size=size,
            font_weight="bold" if bold else "normal",
            font_style="italic" if italic else "normal",
            fill=color,
            text_anchor=anchor,
        )
        node.text = value
        self.add(node)
        width = measure(value, size, bold, italic, sans)
        left = x - (
            width if anchor == "end" else width / 2 if anchor == "middle" else 0
        )
        self.text_boxes.append(
            {
                "text": value,
                "x": left,
                "y": y - size * 0.83,
                "width": width,
                "height": size,
                "font_size": size,
            }
        )

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
                test = f"{line} {word}".strip()
                if line and measure(test, size, bold, italic) > width:
                    lines.append(line)
                    line = word
                else:
                    line = test
            lines.append(line)
        if max_lines is not None and len(lines) > max_lines:
            raise ValueError(
                f"{self.ident}: paragraph requires {len(lines)} lines, limit {max_lines}: {value}"
            )
        for i, line in enumerate(lines):
            self.text(x, y + i * leading, line, size, color, bold, italic)
        return y + len(lines) * leading

    def node(
        self,
        x: float,
        y: float,
        role: str = "FINDING",
        fill: str = "#aeb6bf",
        radius: float = 3.5,
        stroke: str = "#82909f",
        sw: float = 0.55,
        ident: str | None = None,
    ) -> None:
        group = el("g", id=ident)
        attrs = {"fill": fill, "stroke": stroke, "stroke_width": sw}
        if role == "FINDING":
            shape = el("circle", cx=x, cy=y, r=radius, **attrs)
        elif role == "METHOD":
            shape = el(
                "rect",
                x=x - radius * 0.88,
                y=y - radius * 0.88,
                width=radius * 1.76,
                height=radius * 1.76,
                **attrs,
            )
        else:
            if role == "MECHANISM":
                angles = [-math.pi / 2, math.pi / 6, 5 * math.pi / 6]
                points = [
                    (x + radius * 1.2 * math.cos(a), y + radius * 1.2 * math.sin(a))
                    for a in angles
                ]
            elif role == "RESOURCE":
                points = [
                    (x, y - radius * 1.2),
                    (x + radius * 1.2, y),
                    (x, y + radius * 1.2),
                    (x - radius * 1.2, y),
                ]
            else:
                points = [
                    (
                        x
                        + radius
                        * (1.3 if i % 2 == 0 else 0.57)
                        * math.cos(-math.pi / 2 + i * math.pi / 5),
                        y
                        + radius
                        * (1.3 if i % 2 == 0 else 0.57)
                        * math.sin(-math.pi / 2 + i * math.pi / 5),
                    )
                    for i in range(10)
                ]
            shape = el(
                "polygon",
                points=" ".join(f"{a:.3f},{b:.3f}" for a, b in points),
                **attrs,
            )
        group.append(shape)
        self.add(group)

    def arrow(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        color: str = "#5f6774",
        width: float = 0.65,
        gap: float = 4,
    ) -> ET.Element:
        dx, dy = x2 - x1, y2 - y1
        length = max(math.hypot(dx, dy), 1e-8)
        ux, uy = dx / length, dy / length
        x1, y1, x2, y2 = x1 + ux * gap, y1 + uy * gap, x2 - ux * gap, y2 - uy * gap
        shaft = self.line(x1, y1, x2, y2, color, width)
        self.path(
            f"M{x2},{y2} L{x2 - ux * 4 - uy * 1.55},{y2 - uy * 4 + ux * 1.55} L{x2 - ux * 2.8},{y2 - uy * 2.8} L{x2 - ux * 4 + uy * 1.55},{y2 - uy * 4 - ux * 1.55} Z",
            color,
        )
        return shaft

    def use(self, other: Scene, x: float, y: float, scale: float = 1) -> None:
        group = el("g", transform=f"translate({x} {y}) scale({scale})")
        group.append(copy.deepcopy(other.root))
        self.add(group)
        for box in other.text_boxes:
            self.text_boxes.append(
                {
                    **box,
                    "x": x + box["x"] * scale,
                    "y": y + box["y"] * scale,
                    "width": box["width"] * scale,
                    "height": box["height"] * scale,
                    "component": other.ident,
                }
            )

    def document(self, physical: bool = False) -> ET.Element:
        width = "297mm" if physical else str(self.width)
        height = (
            f"{297 * self.height / self.width:.6f}mm" if physical else str(self.height)
        )
        root = el(
            "svg",
            width=width,
            height=height,
            viewBox=f"0 0 {self.width} {self.height}",
            version="1.1",
        )
        defs = el("defs")
        for ident, top, bottom in [
            ("bluewash", "#eaf7ff", "#ffffff"),
            ("pinkwash", "#f6e7f0", "#ffffff"),
            ("bluecard", "#cee9fb", "#eff8ff"),
            ("orangecard", "#ffdbbe", "#fff6ee"),
            ("greencard", "#c7ebd1", "#f2fbf4"),
            ("redcard", "#ffcdcd", "#fff4f3"),
        ]:
            grad = el("linearGradient", id=ident, x1="0", y1="0", x2="0", y2="1")
            grad.append(el("stop", offset="0", stop_color=top))
            grad.append(el("stop", offset="1", stop_color=bottom))
            defs.append(grad)
        root.append(defs)
        root.append(copy.deepcopy(self.root))
        return root

    def save(self, path: Path, physical: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        ET.ElementTree(self.document(physical)).write(
            path, encoding="utf-8", xml_declaration=True
        )


def safe_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", value)


def mix(a: str, b: str, fraction: float) -> str:
    aa = [int(a[i : i + 2], 16) for i in (1, 3, 5)]
    bb = [int(b[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(
        f"{round(x * (1 - fraction) + y * fraction):02x}" for x, y in zip(aa, bb)
    )
