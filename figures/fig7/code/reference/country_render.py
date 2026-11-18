"""Publication SVG panels below, not in place of, the saved Fig7 word clouds."""

from __future__ import annotations

import copy
import re
import textwrap
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any

import cairosvg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pycountry
import requests
import shapefile
from matplotlib import colors, font_manager
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from PIL import Image, ImageChops
from pyproj import Transformer

from .country_data import BASE, ROOT, TYPES, country_name, csv, digest, read, write

NS = "http://www.w3.org/2000/svg"
INK = "#101d3b"
BLUE = colors.LinearSegmentedColormap.from_list(
    "country_blue", ["#edf5fb", "#306b9b", "#123c63"]
)
SIGNED = colors.LinearSegmentedColormap.from_list(
    "preference", ["#a86521", "#fafafa", "#356f9e"]
)
TYPE_COLORS = ["#5984b3", "#56a9a0", "#d89962", "#c8b55a", "#9c88b7"]
PROJECTION = Transformer.from_crs(
    "EPSG:4326", "+proj=eqearth +datum=WGS84 +units=m", always_xy=True
)


def configure_fonts() -> None:
    for name in ("times.ttf", "timesbd.ttf", "arial.ttf", "arialbd.ttf"):
        path = Path("/mnt/c/Windows/Fonts") / name
        if path.exists():
            font_manager.fontManager.addfont(str(path))
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 13,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.edgecolor": "#b5c4cf",
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
        }
    )


def canvas(letter: str, title: str, width: int) -> Figure:
    plt.rcParams["svg.hashsalt"] = f"fig7-country-{letter}"
    fig = plt.figure(figsize=(width / 72, 450 / 72), facecolor="white")
    fig.text(0.018, 0.955, letter, fontsize=25, weight="bold", fontfamily="Arial")
    fig.text(0.075 if width == 500 else 0.053, 0.955, title, fontsize=20, weight="bold")
    return fig


def save_panel(fig: Figure, letter: str, out: Path) -> None:
    path = out / "panels" / letter
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".svg"))
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=160)
    plt.close(fig)


def map_shapes(out: Path) -> list[tuple[str, str, Any]]:
    asset = out / "assets/ne_50m_admin_0_map_units.zip"
    if not asset.exists():
        url = "https://naturalearth.s3.amazonaws.com/50m_cultural/ne_50m_admin_0_map_units.zip"
        response = requests.get(url, timeout=90)
        response.raise_for_status()
        asset.parent.mkdir(parents=True, exist_ok=True)
        asset.write_bytes(response.content)
    with zipfile.ZipFile(asset) as archive:
        versions = [n for n in archive.namelist() if n.endswith("VERSION.txt")]
        version = (
            archive.read(versions[0]).decode().strip() if versions else "unreported"
        )
        assert version == "5.1.1", f"Unexpected Natural Earth version: {version}"
        for name in archive.namelist():
            if Path(name).suffix in (".shp", ".shx", ".dbf", ".prj"):
                path = asset.parent / Path(name).name
                if not path.exists():
                    path.write_bytes(archive.read(name))
    reader = shapefile.Reader(str(asset.parent / "ne_50m_admin_0_map_units.shp"))
    shapes, mapping = [], []
    for shape in reader.iterShapeRecords():
        record = shape.record.as_dict()
        code = record.get("ISO_A2_EH") or record.get("ISO_A2")
        if code == "-99":
            entry = pycountry.countries.get(alpha_3=record.get("ADM0_A3", ""))
            code = entry.alpha_2 if entry else ""
        shapes.append((code, record["NAME"], shape.shape.__geo_interface__))
        mapping.append(
            {
                "map_name": record["NAME"],
                "country_code": code,
                "source_iso_a2": record.get("ISO_A2"),
            }
        )
    csv(out / "tables/map_country_mapping.csv", mapping)
    write(
        out / "assets/map_manifest.json",
        {
            "source": "Natural Earth",
            "scale": "1:50m",
            "layer": "admin_0_map_units",
            "version": version,
            "sha256": digest(asset),
            "projection": "Equal Earth",
            "url": "https://naturalearth.s3.amazonaws.com/50m_cultural/ne_50m_admin_0_map_units.zip",
        },
    )
    return shapes


def projected_paths(geometry: dict[str, Any]) -> list[MplPath]:
    polygons = (
        geometry["coordinates"]
        if geometry["type"] == "MultiPolygon"
        else [geometry["coordinates"]]
    )
    paths = []
    for polygon in polygons:
        vertices, codes = [], []
        for ring in polygon:
            lon, lat = np.array(ring).T
            x, y = PROJECTION.transform(lon, lat)
            vertices.extend(zip(x, y))
            codes.extend(
                [MplPath.MOVETO]
                + [MplPath.LINETO] * (len(ring) - 2)
                + [MplPath.CLOSEPOLY]
            )
        paths.append(MplPath(vertices, codes))
    return paths


def draw_map(
    ax: Axes,
    shapes: list[tuple[str, str, Any]],
    counts: dict[str, float],
    norm: colors.Normalize,
) -> None:
    for code, _, geometry in shapes:
        value = counts.get(code, 0)
        color = BLUE(norm(value)) if value else "#e5e9ed"
        for path in projected_paths(geometry):
            ax.add_patch(
                PathPatch(path, facecolor=color, edgecolor="white", linewidth=0.32)
            )
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.set_axis_off()


def panel_c(data: dict[str, Any], out: Path) -> None:
    fig = canvas(
        "c",
        "Research contribution by country",
        500,
    )
    counts = {r["country_code"]: r["fractional_papers"] for r in data["countries"]}
    maximum = max(counts.values())
    norm = colors.FuncNorm((np.log1p, np.expm1), vmin=0, vmax=maximum)
    shapes = map_shapes(out)
    ax = fig.add_axes([0.015, 0.365, 0.97, 0.48])
    draw_map(ax, shapes, counts, norm)
    ax.set_ylim(-7.5e6, 8.5e6)
    cb = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=BLUE),
        cax=fig.add_axes([0.16, 0.335, 0.70, 0.022]),
        orientation="horizontal",
    )
    ticks = [v for v in [0, 1, 10, 100, 1000, 10000] if v < maximum / 1.5] + [maximum]
    cb.set_ticks(ticks, labels=[f"{v:,.0f}" for v in ticks])
    cb.ax.tick_params(labelsize=11, length=2)
    cb.set_label("Fractional paper contribution", fontsize=11, labelpad=3)
    for box, bounds, name in [
        ([0.07, 0.055, 0.36, 0.17], (-13, 30, 38, 62), "Europe"),
        ([0.57, 0.055, 0.36, 0.17], (98, -3, 145, 39), "East / Southeast Asia"),
    ]:
        inset = fig.add_axes(box)
        draw_map(inset, shapes, counts, norm)
        lon, lat = np.meshgrid(
            np.linspace(bounds[0], bounds[2], 40), np.linspace(bounds[1], bounds[3], 40)
        )
        x, y = PROJECTION.transform(lon, lat)
        inset.set_xlim(float(x.min()), float(x.max()))
        inset.set_ylim(float(y.min()), float(y.max()))
        fig.text(box[0] + box[2] / 2, 0.233, name, fontsize=11, ha="center")
    mapped = {code for code, _, _ in shapes}
    csv(
        out / "tables/map_unmatched_countries.csv",
        [r for r in data["countries"] if r["country_code"] not in mapped],
        columns=list(data["countries"][0]),
    )
    save_panel(fig, "c", out)


def tidy_matrix(
    ax: Axes, labels: list[str], countries: list[str], bottom: float = 0
) -> None:
    ax.set_xticks(
        range(len(labels)),
        labels,
        rotation=50,
        ha="right",
        rotation_mode="anchor",
        fontsize=12,
    )
    ax.set_yticks(
        range(len(countries)), [country_name(c) for c in countries], fontsize=13
    )
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.set_ylim(len(countries) - 0.5, -0.5)
    ax.tick_params(length=0, pad=5)
    for spine in ax.spines.values():
        spine.set_visible(False)


def panel_d(data: dict[str, Any], out: Path) -> None:
    fig = canvas(
        "d",
        "Research-topic preferences",
        862,
    )
    countries = data["top_countries"]
    labels = [r["display_label"] for r in data["historical_topics"]]
    cells = {(r["country_code"], r["topic"]): r for r in data["topic_cells"]}
    maximum = max(cells[c, t]["fractional_papers"] for c in countries for t in labels)
    scale = 380 / maximum
    ax = fig.add_axes([0.16, 0.37, 0.79, 0.49])
    tidy_matrix(
        ax,
        [
            textwrap.fill(t, 24, break_long_words=False, break_on_hyphens=False)
            for t in labels
        ],
        countries,
    )
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(countries), 1), minor=True)
    ax.grid(which="minor", color="#e7edf1", linewidth=0.55)
    ax.tick_params(which="minor", length=0)
    for y, c in enumerate(countries):
        for x, t in enumerate(labels):
            cell = cells[c, t]
            amount = cell["fractional_papers"]
            if not amount:
                ax.text(
                    x, y, "–", ha="center", va="center", fontsize=8, color="#acb8c1"
                )
            elif cell["low_sample"]:
                ax.scatter(
                    x,
                    y,
                    s=amount * scale,
                    facecolors="none",
                    edgecolors="#8496a3",
                    linewidths=0.8,
                )
            else:
                ax.scatter(
                    x,
                    y,
                    s=amount * scale,
                    c=[cell["log2_preference"]],
                    cmap=SIGNED,
                    vmin=-2,
                    vmax=2,
                    edgecolors="white",
                    linewidths=0.35,
                )
    cb = fig.colorbar(
        plt.cm.ScalarMappable(norm=colors.Normalize(-2, 2), cmap=SIGNED),
        cax=fig.add_axes([0.18, 0.065, 0.32, 0.021]),
        orientation="horizontal",
        extend="both",
    )
    cb.set_ticks([-2, -1, 0, 1, 2], labels=["≤−2", "−1", "0", "+1", "≥+2"])
    cb.ax.tick_params(labelsize=11, length=2)
    cb.set_label("log₂(relative preference)", fontsize=11)
    sizes = [v for v in [1, 10, 50, 100] if v <= maximum][-3:]
    handles = [
        ax.scatter(
            [],
            [],
            s=v * scale,
            facecolors="#c1ccd4",
            edgecolors="#8496a3",
            linewidths=0.5,
        )
        for v in sizes
    ]
    fig.legend(
        handles,
        [str(v) for v in sizes],
        loc="lower right",
        bbox_to_anchor=(0.95, 0.045),
        ncol=len(sizes),
        frameon=False,
        title="Fractional papers",
        fontsize=11,
        title_fontsize=11,
    )
    save_panel(fig, "d", out)


def panel_e(data: dict[str, Any], out: Path) -> None:
    fig = canvas(
        "e",
        "Contribution-type profiles",
        500,
    )
    countries = data["top_countries"]
    cells = {
        (r["country_code"], r["claim_type"]): r["proportion"]
        for r in data["type_cells"]
    }
    totals = {r["country_code"]: r for r in data["countries"]}
    ax = fig.add_axes([0.26, 0.15, 0.48, 0.65])
    left = np.zeros(len(countries))
    for role, color in zip(TYPES, TYPE_COLORS):
        values = np.array([cells[c, role] * 100 for c in countries])
        ax.barh(
            range(len(countries)),
            values,
            left=left,
            height=0.64,
            color=color,
            edgecolor="white",
            linewidth=0.4,
            label=role.title(),
        )
        left += values
    ax.set_yticks(
        range(len(countries)), [country_name(c) for c in countries], fontsize=13
    )
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"], fontsize=11)
    ax.tick_params(axis="y", length=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    for y, c in enumerate(countries):
        r = totals[c]
        ax.text(
            104,
            y,
            f'{r["fractional_papers"]:,.1f} / {r["participating_papers"]:,}',
            va="center",
            fontsize=10.5,
        )
    fig.text(0.765, 0.815, "Fractional / papers", fontsize=10.5, color="#526779")
    fig.legend(
        loc="upper center",
        bbox_to_anchor=(0.51, 0.89),
        ncol=5,
        frameon=False,
        handlelength=0.85,
        handletextpad=0.3,
        columnspacing=0.7,
        fontsize=10.5,
    )
    save_panel(fig, "e", out)


def panel_f(data: dict[str, Any], out: Path) -> None:
    fig = canvas(
        "f",
        "Country sources of prospective directions",
        862,
    )
    countries = data["top_countries"] + ["Other", "Unresolved"]
    topics = [r["display_label"] for r in data["future_topics"]]
    cells = {(r["country_code"], r["topic"]): r for r in data["future_cells"]}
    labels = [
        textwrap.fill(t, 22, break_long_words=False, break_on_hyphens=False)
        for t in topics
    ]
    values = np.array(
        [[cells[c, t]["proportion"] * 100 for t in topics] for c in countries]
    )
    ax = fig.add_axes([0.16, 0.40, 0.79, 0.46])
    ax.imshow(
        values, cmap=BLUE, vmin=0, vmax=100, aspect="auto", interpolation="nearest"
    )
    tidy_matrix(ax, labels, countries)
    ax.set_xticks(np.arange(-0.5, len(topics), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(countries), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.9)
    ax.tick_params(which="minor", length=0)
    for y in range(len(countries)):
        for x in range(len(topics)):
            v = values[y, x]
            if v:
                ax.text(
                    x,
                    y,
                    f"{v:.0f}" if abs(v - round(v)) < 0.05 else f"{v:.1f}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="white" if v >= 50 else INK,
                )
    cb = fig.colorbar(
        plt.cm.ScalarMappable(norm=colors.Normalize(0, 100), cmap=BLUE),
        cax=fig.add_axes([0.18, 0.075, 0.32, 0.021]),
        orientation="horizontal",
    )
    cb.set_ticks([0, 25, 50, 75, 100], labels=["0", "25", "50", "75", "100%"])
    cb.ax.tick_params(labelsize=11, length=2)
    cb.set_label("Fractional share of cited source papers", fontsize=11)
    save_panel(fig, "f", out)


def prefix_ids(root: ET.Element, prefix: str) -> None:
    identifiers = {
        node.attrib["id"]: prefix + node.attrib["id"]
        for node in root.iter()
        if "id" in node.attrib
    }
    for node in root.iter():
        for key, value in list(node.attrib.items()):
            if key == "id":
                node.set(key, identifiers[value])
            elif value.startswith("#") and value[1:] in identifiers:
                node.set(key, "#" + identifiers[value[1:]])
            else:
                node.set(
                    key,
                    re.sub(
                        r"url\(#([^)]+)\)",
                        lambda m: f"url(#{identifiers.get(m[1], m[1])})",
                        value,
                    ),
                )


def compose(out: Path, data: dict[str, Any]) -> Path:
    ET.register_namespace("", NS)
    original = ET.parse(BASE / "final/Fig7.svg").getroot()
    root = copy.deepcopy(original)
    root.set("height", "1374pt")
    root.set("viewBox", "0 0 1400 1374")
    rect = ET.Element(
        f"{{{NS}}}rect",
        {"x": "0", "y": "890", "width": "1400", "height": "484", "fill": "white"},
    )
    root.append(rect)
    for letter, x, y, width in [
        ("c", 10, 906, 500),
        ("d", 528, 906, 862),
    ]:
        panel = ET.parse(out / "panels" / f"{letter}.svg").getroot()
        prefix_ids(panel, f"country_{letter}_")
        panel.set("x", str(x))
        panel.set("y", str(y))
        panel.set("width", str(width))
        panel.set("height", "450")
        root.append(panel)
    if not data["coverage"]["historical_gate_passed"]:
        note = ET.SubElement(
            root,
            f"{{{NS}}}text",
            {
                "x": "700",
                "y": "902",
                "text-anchor": "middle",
                "font-family": "Arial",
                "font-size": "12",
                "fill": "#ad5525",
            },
        )
        note.text = (
            "DRAFT — historical country attribution below the 95% coverage target"
        )
    target = out / "final/Fig7_country_extension.svg"
    target.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(target, encoding="utf-8", xml_declaration=True)
    assert [ET.tostring(c) for c in list(root)[: len(original)]] == [
        ET.tostring(c) for c in original
    ]
    return target


def export(target: Path, out: Path) -> None:
    from figure_pipeline.fig1_reference.export import fonts

    fonts(out)
    cairosvg.svg2pdf(url=str(target), write_to=str(target.with_suffix(".pdf")))
    cairosvg.svg2png(
        url=str(target),
        write_to=str(target.with_suffix(".png")),
        output_width=2100,
        output_height=2061,
    )
    high = target.with_name(target.stem + "_600dpi.png")
    width, height = round(1400 / 72 * 600), round(1374 / 72 * 600)
    cairosvg.svg2png(
        url=str(target), write_to=str(high), output_width=width, output_height=height
    )
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(high) as img:
        img.save(high, dpi=(600, 600))
    reference = out / "qa/original_same_renderer.png"
    cairosvg.svg2png(
        url=str(BASE / "final/Fig7.svg"),
        write_to=str(reference),
        output_width=2100,
        output_height=1335,
    )
    with Image.open(reference) as ref, Image.open(
        target.with_suffix(".png")
    ) as combined:
        diff = ImageChops.difference(
            ref.convert("RGB"), combined.crop((0, 0, 2100, 1335)).convert("RGB")
        )
        assert diff.getbbox() is None, "Original word clouds changed during composition"
    write(
        out / "qa/export_checks.json",
        {
            "original_svg_children_unchanged": True,
            "upper_crop_pixel_identical": True,
            "comparison_pixels": [2100, 1335],
            "png_pixels": [width, height],
            "dpi": 600,
            "new_model_calls": 0,
            "original_inputs_unchanged": True,
        },
    )


def methods(data: dict[str, Any], out: Path) -> None:
    n = data["historical_complete"]
    status = (
        "coverage target met"
        if data["coverage"]["historical_gate_passed"]
        else "DRAFT: coverage target not met"
    )
    text = f"""# Fig.7 — Country research differences / 国家科研差异

Status: **{status}**. Historical complete attribution: {n:,}/24,914 ({n/24914:.2%}).

## Scope and attribution / 统计总体和国家归属

Panels a and b are embedded unchanged from the accepted 1400 × 890 SVG. No labels,
font sizes, positions, source directions or model outputs were regenerated.
Panels c–d use the 2023–2025 claim-bearing corpus: 24,914 papers and 70,034 abstract
claims. Nature Communications contributes 24,219 papers (97.2%); these are corpus
differences, not representative estimates of national research strength.

Country/region is publication-time institutional affiliation, not citizenship,
current affiliation, publisher location or an assertion of contribution ownership.
Each complete paper receives weight 1/k in each of its k distinct countries/regions.
Author and institution counts do not multiply this weight. Partial, conflicting and
unmatched records are excluded from c–d and retained in the missing-data table.
OpenAlex responses, retrieval time, record update time and local source checks are retained.
An unmapped author's countries are repaired only if every raw affiliation address has an
explicit country/region suffix and its normalized address matches local manuscript text.
Normalization removes whitespace/punctuation and standardizes Unicode; trailing email
contacts are excluded from address matching. Exact original character spans are retained.
An API country counter explained fully by these locally verified extra countries is
recorded as corrected; unexplained count discrepancies remain conflicts.
Current metadata supplements attribution only; it does not change historical claims or forecasts.

## Panel definitions / 各图定义

**c.** Fractional paper contributions on Natural Earth 1:50m v5.1.1 map-unit geometry,
Equal Earth projection. The sequential color scale uses log1p and labels raw units.
Gray means no attributed record in this corpus, not absence of research. Geographic
codes follow the source's country/region units; map units distinguish overseas territories
that the country layer merges. Map mapping and unmatched units are supplied.

**d.** Top 12 countries by historical fractional contributions (ties by country code),
against the first 16 displayed historical phrases ranked by original frequency,
then canonical phrase. C_ct is fractional topic contribution, N_c all eligible country
papers, C_t all eligible papers containing the topic, and N the eligible corpus.
R_ct = (C_ct/N_c)/(C_t/N); log2(R_ct) is colored on [-2,2], with out-of-range values
retained in the table. Bubble area is proportional to C_ct. Fewer than five distinct
participating papers yields an open gray bubble; a dash denotes zero. Topics overlap.

## Caption

**Fig.7 | Historical research topics, prospective directions and country-level research
profiles within the sampled corpus.** (a,b) Original historical and prospective word
clouds, unchanged. (c) Fractional country/region paper contributions. (d) Research-topic
preferences relative to the eligible corpus, with bubble area indicating fractional
paper contributions. Country
statistics describe the collected corpus; future directions remain source-grounded hypotheses.

## 中文图注与限制

**图7｜样本语料中的历史研究主题、前瞻方向与国家科研画像。** a、b 为保持原样的
历史主题和未来方向词云；c 为按机构所属国家／地区分数计数的论文贡献；d 为相对于
同一有效语料总体的主题偏好，气泡面积表示论文贡献份额。

历史国家归属完整率为 {n/24914:.2%}，共 {24914-n:,} 篇未纳入 c–d。
机构国家以论文发表时的机构为依据；不是作者国籍。多国论文总权重为一，同国多机构
不重复计数。d 的主题可重叠，偏好不是质量。缺失不等于零贡献。
本语料主要来自 Nature Communications，不外推为各国整体科研实力。
机构名称未被解析时，仅从该作者的原始机构地址末尾读取明确国家／地区名称，并要求
地址与本地论文文本在 Unicode、空白和标点归一化后精确匹配；保留原文字符位置。
仅当已核验地址能完整解释国家计数增加时修正 API 国家数量，其余冲突保持未解决。

## Reproducibility

Run `python -m figure_pipeline.fig7_reference.country_extension audit|fetch|aggregate|render`.
Use the publication Python environment; metadata fetching is resumable. Tables preserve
all countries, attribution states, topic denominators and direction source links. Original
asset hashes, aggregation checks and same-renderer upper-image comparisons are in qa/.
Sources: https://help.openalex.org/data/authorships/ ;
https://help.openalex.org/api/filtering/ ;
https://www.naturalearthdata.com/downloads/50m-cultural-vectors/50m-admin-0-countries-2/
"""
    (out / "methods_and_caption.md").write_text(text, encoding="utf-8")


def render(out: Path) -> None:
    from .country_checks import verify

    protected = read(out / "qa/protected_inputs.json")
    assert all(digest(ROOT / p) == checksum for p, checksum in protected.items())
    verify(out)
    data = read(out / "data/plot_data.json")
    configure_fonts()
    for renderer in (panel_c, panel_d):
        renderer(data, out)
    target = compose(out, data)
    export(target, out)
    methods(data, out)
    assert all(digest(ROOT / p) == checksum for p, checksum in protected.items())
    print(f"Rendered {target}", flush=True)
