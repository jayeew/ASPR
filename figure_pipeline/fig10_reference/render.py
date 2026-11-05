"""Render the G02 MOF case from the archived graph facts."""

from pathlib import Path
import hashlib
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.axes import Axes
from matplotlib.patches import FancyBboxPatch, Ellipse
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/fig10_reference"
DATA = ROOT / "outputs/classic_claims_position_2025_2026/graph_position_v1"
ICONS = Path(__file__).parent / "icons"
NAVY = "#10213B"
ACCENT = "#E4AE4A"
COLORS = ["#4C78A8", ACCENT, "#D87562", "#B43E54"]
PALE = ["#EEF6FC", "#FCF5E6", "#FFF5ED", "#FAECEF"]


def claim_icon(ax: Axes, name: str, x: float, y: float, w: float, h: float) -> None:
    """Fit the high-resolution transparent artwork without stretching it."""
    with Image.open(ICONS / (name + ".png")) as source:
        art = source.convert("RGBA")
    pixels = np.array(art)
    rgb = pixels[:, :, :3].astype(float) / 255
    chromatic = np.ptp(rgb, axis=2) > .08
    tone = rgb @ np.array([.2126, .7152, .0722])
    accent = np.array(list(bytes.fromhex(COLORS[["C03","C04","C05","C06"].index(name)][1:]))) / 255
    mid = accent @ np.array([.2126, .7152, .0722])
    shade = np.where((tone < mid)[:, :, None], accent * tone[:, :, None] / mid,
                     accent + (1-accent) * (tone-mid)[:, :, None] / (1-mid))
    pixels[:, :, :3][chromatic] = np.clip(shade[chromatic]*255, 0, 255).astype(np.uint8)
    art = Image.fromarray(pixels)
    bounds = art.getchannel("A").point(lambda a: 255 if a > 35 else 0).getbbox()
    if bounds:
        art = art.crop(bounds)
    scale = min(w / art.width, h / art.height)
    width, height = art.width * scale, art.height * scale
    left, top = x + (w - width) / 2, y + (h - height) / 2
    ax.imshow(
        art,
        extent=(left, left + width, top + height, top),
        aspect="auto",
        interpolation="lanczos",
        zorder=3,
    )


def main() -> None:
    for filename in ("arial.ttf", "arialbd.ttf"):
        font = Path("/mnt/c/Windows/Fonts") / filename
        if font.exists():
            font_manager.fontManager.addfont(font)
    matplotlib.rcParams.update(
        {
            "font.family": ["Arial", "DejaVu Sans"],
            "axes.linewidth": 0.65,
            "axes.edgecolor": "#8295A7",
            "xtick.color": NAVY,
            "ytick.color": NAVY,
            "pdf.fonttype": 42,
        }
    )
    facts = {
        i: json.loads((DATA / i / "facts.json").read_text())
        for i in ["C03", "C04", "C05", "C06", "G02"]
    }
    ids = list(facts)[:4]
    fig = plt.figure(figsize=(15, 9.4), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 1150), ylim=(720, 0))
    ax.axis("off")

    def box(x, y, w, h, fill="white", stroke="#c6cadd", radius=4):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle=f"round,pad=0,rounding_size={radius}",
                fc=fill,
                ec=stroke,
                lw=0.8,
            )
        )

    def t(x, y, s, size=9, bold=False, col="#10213B", align="left"):
        ax.text(
            x,
            y,
            s,
            fontsize=size,
            weight="bold" if bold else "normal",
            color=col,
            ha=align,
            va="center",
            linespacing=1.28,
        )

    def panel(x, y, w, h, title, tw):
        box(x, y, w, h, "white", "#A9CDE4", 5)
        box(x + 1, y + 1, w - 2, 23, "#EAF7FF", "#EAF7FF", 3)
        t(x + 8, y + 12, title, 12, True, NAVY)

    def sub(x, y, w, label):
        box(x, y, w, 18, "#EAF7FF", "#EAF7FF")
        t(x + 5, y + 9, label, 9, True, NAVY)

    def icon(name, x, y, w, h):
        ax.imshow(
            Image.open(ICONS / (name + ".png")),
            extent=(x, x + w, y + h, y),
            aspect="auto",
            zorder=3,
        )

    def line(x, y, X, Y, color="#9ca5b5", width=1, style="-"):
        ax.plot([x, X], [y, Y], color=color, lw=width, ls=style, zorder=2)

    panel(4, 4, 1142, 143, "a    2025 Nobel Chemistry–related MOF claim set (G02)", 414)
    t(
        825,
        18,
        "G02: Four landmark claims covering key facets of MOFs",
        11,
        True,
        NAVY,
        "center",
    )
    icon("medal", 11, 51, 58, 58)
    t(74, 54, "Nobel Prize in\nChemistry 2025", 11, True)
    t(74, 105, "Metal–organic frameworks:\ndesign, construction\nand applications", 8)
    line(202, 38, 202, 135)
    t(214, 56, "Graph coverage window", 9, True)
    t(214, 77, "2023-01-07 — 2025-10-07", 8)
    t(214, 104, "Evaluation cutoff", 9, True)
    t(214, 124, "< 2025-10-08", 9)
    names = [
        "Framework assembly",
        "Gas uptake & release",
        "Stability & porosity",
        "Pore property tuning",
    ]
    descriptions = [
        "Metal-node geometry and\nlinker connectivity guide\nextended porous networks.",
        "Selected frameworks retain\ntheir structure during\nreversible gas sorption\nunder tested conditions.",
        "Zn₄O nodes and BDC linkers\nform MOF-5. Its pores persist\nafter solvent removal.",
        "Linker length tunes pore size.\nLinker functional groups\nmodify pore chemistry.",
    ]
    for j, i in enumerate(ids):
        x = 389 + 189 * j
        box(x, 41, 182, 99, PALE[j], COLORS[j])
        t(x + 6, 54, i + "  " + names[j], 8.5, True, COLORS[j])
        claim_icon(ax, i, x + 4, 72, 54, 45)
        t(x + 60, 92, descriptions[j], 7.6)
        t(x + 91, 130, facts[i]["claim_type"], 7, True, COLORS[j], "center")
    panel(
        4,
        158,
        1142,
        275,
        "b    C04 main case: gas adsorption, release and framework response",
        518,
    )
    sub(13, 190, 222, "b1    Target claim (C04)")
    sub(247, 190, 421, "b2    Local neighborhood: 10 nearest claims")
    sub(680, 190, 456, "b3    Representative neighbors and scientific correspondence")
    box(14, 216, 220, 68, "#F2F8FC")
    t(
        24,
        250,
        "Selected porous MOFs adsorb and\nrelease gases while remaining\nusable under tested conditions.",
        9.3,
        True,
        NAVY,
    )
    for j, (a, b) in enumerate(
        [
            ("Object", "Open-pore MOFs"),
            ("Operation", "Gas adsorption and release"),
            ("Result", "Framework remains usable"),
        ]
    ):
        y = 292 + 39 * j
        box(14, y, 220, 33, "#f6f6fa")
        t(23, y + 16, a, 8.4, True, NAVY)
        t(88, y + 16, b, 8.2)
    f = facts["C04"]
    ns = f["neighbors"]
    pos = {
        1: (320, 268),
        3: (397, 268),
        4: (559, 267),
        7: (495, 267),
        2: (309, 353),
        8: (384, 354),
        5: (497, 357),
        6: (551, 373),
        9: (592, 342),
        10: (634, 372),
    }
    for x, y, w, h, col, label in [
        (354, 266, 176, 69, "#cce7fa", "Flexible frameworks"),
        (545, 266, 178, 69, "#CFEAE5", "Stimulated uptake / release"),
        (348, 356, 153, 67, "#d4ecdc", "Selective capture"),
        (562, 360, 191, 75, "#ffe6c8", "Transport & related functions"),
    ]:
        ax.add_patch(
            Ellipse((x, y), w, h, fc=col, alpha=0.47, ec="#9599aa", ls="--", lw=0.8)
        )
        t(x, y - 25, label, 7.6, True, NAVY, "center")
    ix = {n["claim_id"]: n["semantic_rank"] for n in ns}
    for a, b in f["structure"]["historical_edges"]:
        line(*pos[ix[a]], *pos[ix[b]], "#8795a1", 1)
    for n in ns:
        k = n["semantic_rank"]
        x, y = pos[k]
        line(450, 309, x, y, "#5076bf", 0.9, "--")
        ax.scatter(x, y, s=480, fc="#deefff", ec="#6b94c5", zorder=4)
        ax.text(
            x,
            y,
            f"N{k}\n’{n['publication_date'][2:4]}",
            ha="center",
            va="center",
            fontsize=7,
            zorder=5,
        )
    ax.scatter(450, 309, s=660, marker="D", color=ACCENT, zorder=5)
    ax.text(
        450, 309, "C04", ha="center", va="center", color="white", fontsize=9, zorder=6
    )
    t(260, 413, "◆ Target claim     ○ Historical neighbor", 8, True, NAVY)
    line(450, 413, 474, 413)
    t(478, 413, "Historical (11)", 7.5)
    line(559, 413, 581, 413, "#5076bf", 1, "--")
    t(585, 413, "Insertion (10)", 7.5)
    evidence = [
        (
            2,
            "Strong",
            "Reversible formaldehyde capture\nand low-energy regeneration.",
            "Shared operation: capture and release.\nSpecific chemisorption mechanism.",
        ),
        (
            7,
            "Strong",
            "A flexible Zn-MOF film captures\nand releases CO₂ reversibly.",
            "Shared gas cycling and framework reuse.\nFilm, CO₂ and external stimuli are specific.",
        ),
        (
            1,
            "Boundary",
            "Sorption drives structural changes\nand negative gas adsorption.",
            "A usable framework can change shape.\nThe response depends on sorption history.",
        ),
        (
            10,
            "Peripheral",
            "Hollow MOF films increase\ngas permeability.",
            "Shared material and gas transport.\nPermeation measures passage through a film.",
        ),
    ]
    for j, (k, role, desc, note) in enumerate(evidence):
        y = 215 + 52 * j
        box(680, y, 456, 48, "#fafaff")
        box(682, y + 2, 53, 44, PALE[j])
        t(
            708,
            y + 14,
            "N" + str(k) + "  ’" + ns[k - 1]["publication_date"][2:4],
            8.5,
            True,
            NAVY,
            "center",
        )
        t(708, y + 33, role, 7, True, COLORS[j], "center")
        t(741, y + 24, desc, 7.7, True)
        line(925, y + 5, 925, y + 43)
        t(932, y + 24, note, 7.5)
    panel(4, 447, 655, 264, "c    Four claims occupy distinct local positions", 365)
    sub(
        11,
        480,
        467,
        "Claim and focus                 Key statistics               Representative neighborhood",
    )
    sub(485, 480, 167, "Similarity and components")
    topics = [
        "Ligand geometry and\ncoordination cages",
        "Reversible capture and\nstructural response",
        "MOF nanopatterning and\nlinker-induced rigidity",
        "Linker chemistry and\ntunable pore size",
    ]
    for j, i in enumerate(ids):
        y = 505 + j * 48
        m = facts[i]["metrics"]
        s = facts[i]["structure"]
        box(11, y, 467, 45, "#fafbfe")
        claim_icon(ax, i, 14, y + 7, 37, 30)
        t(56, y + 14, i, 8, True, COLORS[j])
        t(56, y + 31, names[j], 7.5, True)
        t(
            183,
            y + 15,
            f"NN {m['nearest_prior_similarity']:.3f}   Top 5 {m['mean_top5_similarity']:.3f}",
            7.6,
        )
        t(
            183,
            y + 31,
            f"Parts {s['historical_components_before']} → 1    Δpairs +{s['joint_newly_connected_historical_pairs']}",
            7.6,
        )
        t(335, y + 23, topics[j], 8)
    chart = fig.add_axes([0.452, 0.066, 0.091, 0.153])
    xs = list(range(4))
    chart.bar(
        [u - 0.15 for u in xs],
        [facts[i]["metrics"]["nearest_prior_similarity"] for i in ids],
        0.3,
        color="#4C78A8",
    )
    chart.bar(
        [u + 0.15 for u in xs],
        [facts[i]["metrics"]["mean_top5_similarity"] for i in ids],
        0.3,
        color=COLORS[2],
    )
    chart.set_ylim(0.5, 0.9)
    chart.set_xticks(xs, ids)
    chart.tick_params(labelsize=7)
    right = chart.twinx()
    right.plot(
        xs,
        [facts[i]["structure"]["historical_components_before"] for i in ids],
        "o--",
        color="#27354a",
        ms=3,
    )
    right.set_ylim(0, 12)
    right.tick_params(labelsize=7)
    t(490, 516, "■ Nearest", 7.7, True, "#4C78A8")
    t(550, 516, "■ Top-5 mean", 7.7, True, COLORS[2])
    t(490, 533, "○ Components before insertion", 7.2)
    panel(666, 447, 480, 264, "d    Joint neighborhood of the MOF claim set (G02)", 390)
    sub(674, 480, 180, "d1    Joint structure")
    sub(862, 480, 275, "d2    Shared neighbors / Jaccard")
    s = facts["G02"]["structure"]
    for j, (label, val) in enumerate(
        [
            ("Target claims", 4),
            ("Historical neighbors", s["historical_neighbor_count"]),
            ("Components before", s["historical_components_before"]),
            ("Components after", s["historical_components_after"]),
            ("Newly reachable pairs", s["joint_newly_connected_historical_pairs"]),
        ]
    ):
        y = 506 + 25 * j
        box(675, y, 177, 22, "#F4F8FA")
        t(681, y + 11, label, 8.3)
        t(839, y + 11, str(val), 10, True, NAVY, "right")
    loc = dict(zip(ids, [(886, 533), (954, 533), (954, 603), (886, 603)]))
    t(991, 509, "Pair       Shared   Jaccard", 7.6, True)
    for j, pair in enumerate(s["claim_pairs"]):
        a, b = pair["claim_ids"]
        n = pair["shared_neighbor_count"]
        line(*loc[a], *loc[b], COLORS[3] if n else "#c9cdd5", 1 + n, "-" if n else ":")
        t(990, 527 + 18 * j, a + "–" + b, 7.3)
        t(1072, 527 + 18 * j, str(n), 7.5)
        t(1102, 527 + 18 * j, f"{pair['jaccard']:.3f}", 7.5)
    for j, i in enumerate(ids):
        x, y = loc[i]
        ax.scatter(x, y, s=570, fc=PALE[j], ec=COLORS[j], zorder=4)
        ax.text(x, y, i, ha="center", va="center", fontsize=8, zorder=5)
    sub(674, 636, 462, "d3    Takeaway")
    box(674, 657, 462, 46, "#F2F8FC")
    t(
        684,
        680,
        "Assembly, gas cycling, stability and pore tuning retrieve distinct neighbors.\nJoint insertion connects all 36 historical nodes in the local graph.",
        9,
        True,
        NAVY,
    )
    fig.set_size_inches(15, 9.83)
    ax.set_position([0, 0, 1, 9.4 / 9.83])
    fig.text(.012, .969, "Fig. 10 | Local graph positions of four MOF claims associated with the 2025 Nobel Prize in Chemistry", fontsize=15, weight="bold", fontfamily="DejaVu Serif", color=NAVY)
    for typ, fmt in [
        ("Fig10.svg", "none"),
        ("Fig10_outlined.svg", "path"),
        ("Fig10_outlined.pdf", "path"),
        ("Fig10_outlined.png", "path"),
    ]:
        matplotlib.rcParams["svg.fonttype"] = fmt
        fig.savefig(OUT / typ, dpi=300, facecolor="white")
    plt.close(fig)
    (OUT / "figure_source_data.json").write_text(
        json.dumps(facts, ensure_ascii=False, indent=2)
    )
    (OUT / "Fig10_caption.md").write_text(
        "Figure 10. Local graph positions of four MOF claims associated with the 2025 Nobel Prize in Chemistry. (a) Case scope and cutoff. Icons illustrate concepts. C03 depicts a generic extended network. C04 illustrates reversible uptake under the tested conditions (doi:10.1002/anie.199717251). In the C05 cubic-net schematic, vertices represent Zn₄O nodes and edges represent benzene-1,4-dicarboxylate (BDC) linkers (doi:10.1038/46248). C06 compares pore apertures as linker length changes at fixed topology. Linker functionalization provides a separate means of changing pore chemistry (doi:10.1126/science.1067208). (b) C04 and its ten ranked semantic neighbors. Shaded regions are editorial topic groupings; solid lines show historical backbone edges and dashed lines show virtual insertion. (c) NN and Top 5 denote nearest and mean top-five cosine similarity. Δpairs counts newly reachable historical node pairs. (d) Lines in the claim-pair diagram summarize neighborhood overlap; dotted lines denote zero overlap. Joint insertion connects the 36 historical neighbors across 19 original components. The observations describe this local Nature-family corpus.\n"
    )
    names = [
        "Fig10.svg",
        "Fig10_outlined.svg",
        "Fig10_outlined.pdf",
        "Fig10_outlined.png",
        "Fig10_caption.md",
        "figure_source_data.json",
    ]
    (OUT / "delivery_manifest.json").write_text(
        json.dumps(
            {n: hashlib.sha256((OUT / n).read_bytes()).hexdigest() for n in names},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
