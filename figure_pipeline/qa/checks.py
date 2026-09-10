"""Separate file integrity, data, statistics and visual checks."""
from __future__ import annotations
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def svg_check(path: Path) -> dict[str, Any]:
    tree = ET.parse(path)
    root = tree.getroot()
    width = float(re.sub(r'[^0-9.]', '', root.attrib['width']))
    texts = list(root.iter('{http://www.w3.org/2000/svg}text'))
    sizes = []
    for text in texts:
        style = text.attrib.get('style', '')
        match = re.search(r'font-size:\s*([0-9.]+)px', style) or re.search(r'font:\s*(?:[^;]*?\s)?([0-9.]+)px', style)
        if match:
            sizes.append(float(match.group(1)))
    scale = (180 / 25.4 * 72) / width
    return dict(file=path.name, width_pt=width, final_width_mm=180, text_count=len(texts), editable_text=bool(texts), raster_images=len(list(root.iter('{http://www.w3.org/2000/svg}image'))), measured_min_font_pt=min(sizes)*scale if sizes else None, font_check='pass' if sizes and min(sizes)*scale>=7.49 else 'requires_review', note='SVG font check does not certify absence of collisions; manual preview required.')


def audit_exports(output: Path) -> dict[str, Any]:
    results=[svg_check(p) for p in sorted(output.glob('*.svg'))]
    return dict(format=dict(svg_parse='pass',figures=results),data='see per-analysis audit files; no blanket certification',statistics='paper bootstrap recomputed; scientific correctness not measured',visual=dict(final_width_mm=180,minimum_font_target_pt=7.5,automatic_font_results=results,manual_review='recorded separately; automatic geometry cannot certify visual correctness'))
