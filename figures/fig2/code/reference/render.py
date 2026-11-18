"""Reference-positioned mechanism diagrams, observed networks and metric cards."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from figure_pipeline.fig1_reference.svg import el, mix

from .data import METRICS, read
from .graphics import (
    BLUE,
    CLAIM_COLORS,
    COMMUNITIES,
    CYAN,
    INK,
    LINE,
    MUTED,
    NAVY,
    ORANGE,
    PURPLE,
    RED,
    Figure,
    Scene,
    badge,
    box,
    fit,
    icon,
    para,
)

CLAIM_SUMMARIES = [
    "PVH–BAT connection and social-stress susceptibility; β3AR signaling implicated.",
    "BAT β3AR expression and drug-induced stress susceptibility.",
    "BAT-dependent behavior and stress-associated IL-6 changes.",
    "PVH activation alters behavior and BAT signaling; 6-OHDA blocks effects.",
]


def claim_card(data: dict[str, Any], index: int, width: float, height: float) -> Scene:
    s = Scene(width, height, f"claim_{index}")
    color = CLAIM_COLORS[index]
    s.rect(
        0.5,
        0.5,
        width - 1,
        height - 1,
        mix(color, "#ffffff", 0.955),
        mix(color, "#ffffff", 0.55),
        3,
    )
    badge(s, 5, 5, f"C{index + 1}", color, 29, 17)
    fit(s, 42, 17, data["claims"][index]["claim_type"], width - 47, 9, color, True)
    if index == 0:
        s.text(7, 37, "Author claim (summary)", 8.8, NAVY, True)
        para(
            s,
            7,
            49,
            "PVH–BAT axis regulates stress behavior through sympathetic β3AR signaling.",
            width - 14,
            8.2,
        )
        s.text(7, 91, "Grounded claim (narrowed)", 8.8, NAVY, True)
        para(s, 7, 104, CLAIM_SUMMARIES[0], width - 14, 8.2)
        s.text(7, height - 9, "Sources: S1, S2, S3 · Partial support", 7.7, BLUE)
    else:
        para(s, 7, 30, CLAIM_SUMMARIES[index], width - 14, 8.2)
    return s


def manuscript(data: dict[str, Any]) -> Scene:
    s = Scene(153, 351, "manuscript")
    box(s, "Target manuscript")
    s.rect(7, 28, 139, 268, "#ffffff", "#cad9e6", 1)
    para(
        s,
        15,
        45,
        "PVH–BAT control of social-stress susceptibility",
        119,
        10.5,
        NAVY,
        True,
    )
    s.text(15, 100, "Du et al. (2026)", 8.2, MUTED)
    s.text(15, 112, "Communications Biology", 7.5, MUTED)
    # These grey rules are document iconography, not facsimile manuscript text.
    for i in range(30):
        width = [109, 102, 114, 95, 108, 113, 80][i % 7]
        s.rect(15, 123 + i * 5.1, width, 2.7, "#d7dde4")
    for y, label, color in [
        (142, "S1", BLUE),
        (185, "S2", ORANGE),
        (234, "S3", "#28956b"),
    ]:
        s.rect(17, y, 94, 7, mix(color, "#ffffff", 0.55))
        badge(s, 115, y - 4, label, color, 25, 16, 8)
    for i, color in enumerate(["#76a9da", "#efb486", "#76a9da"]):
        icon(s, "doc", 11 + i * 43, 309, 26, color)
    s.text(132, 333, "…", 15, BLUE)
    return s


def grounding(data: dict[str, Any]) -> Scene:
    s = Scene(120, 304, "grounding")
    box(s, "Extract and ground claims", title_size=9)
    labels = [
        ("doc", "Mine", f"{data['candidate_count']} candidates\nfrom manuscript spans"),
        (
            "network",
            "Consolidate",
            f"Merge candidates into\n{len(data['claims'])} contributions",
        ),
        ("shield", "Verify support", "Check claim scope\nagainst source spans"),
    ]
    for i, (kind, title, body) in enumerate(labels):
        y = 34 + 89 * i
        s.rect(5, y, 110, 65, "#fcfeff", LINE, 2)
        icon(s, kind, 12, y + 6, 19)
        fit(s, 43, y + 19, title, 65, 10, NAVY, True)
        para(s, 11, y + 41, body, 98, 8, MUTED)
        if i < 2:
            s.arrow(61, y + 68, 61, y + 89, BLUE, 1.3, 0)
    return s


def architecture_gear(data: dict[str, Any]) -> Scene:
    s = Scene(156, 304, "gear_arch")
    box(s, tint="#fff5ee", stroke="#f1d5c2")
    icon(s, "db", 4, 3, 27, ORANGE)
    fit(s, 39, 20, "GEAR evidence analysis", 111, 10, RED, True)
    s.rect(5, 42, 146, 92, "#fff9f3", "#f2d4bc", 3)
    fit(s, 12, 57, "Historical prior works", 132, 10, RED, True)
    for i, work in enumerate(data["works"]):
        x = 14 + i * 45
        s.rect(x - 3, 65, 36, 61, "white", "#f2dbca", 2)
        icon(s, "doc", x + 5, 71, 17, NAVY)
        s.text(x + 15, 103, work["alias"], 9, NAVY, True, anchor="middle")
        s.text(
            x + 15,
            116,
            str(work["record"]["payload"]["publication_year"]),
            8.5,
            MUTED,
            anchor="middle",
        )
    items = [
        ("search", "Search framing", "Normal + contrastive queries"),
        ("bars", "Historical comparison", "Compare scope and evidence"),
        ("doc", "Residual contribution", "Identify supported differences"),
    ]
    for i, (kind, title, body) in enumerate(items):
        y = 154 + i * 48
        s.rect(5, y, 146, 44, "#fffaf6", "#f0d8c4", 3)
        icon(s, kind, 11, y + 8, 19, ORANGE)
        fit(s, 37, y + 16, title, 108, 9.1, RED, True)
        para(s, 37, y + 29, body, 108, 7.6, MUTED)
    s.arrow(77, 135, 77, 153, ORANGE, 1.2, 0)
    return s


def architecture_graph() -> Scene:
    s = Scene(165, 304, "graph_arch")
    box(s)
    icon(s, "network", 8, 1, 25)
    s.text(47, 20, "Graph reasoning", 11, NAVY, True)
    s.rect(5, 36, 155, 89, "#f7fcff", LINE, 3)
    fit(s, 12, 50, "Prebuilt historical graph assets", 142, 9.5, NAVY, True)
    icon(s, "db", 9, 57, 23, "#5a91c9")
    fit(s, 42, 66, "Claim Graph + Paper Graph", 110, 8.5, NAVY, True)
    fit(s, 42, 78, "Embeddings · Communities · Indexes", 110, 7.2, MUTED)
    s.line(10, 88, 155, 88, LINE, 0.6)
    icon(s, "clock", 10, 96, 14)
    para(s, 29, 99, "Cutoff-compatible\nsnapshot", 64, 7.1, BLUE)
    icon(s, "lock", 95, 96, 12)
    para(s, 112, 99, "Read only\nat runtime", 43, 7.1, BLUE)
    s.arrow(84, 128, 84, 146, BLUE, 1.2, 0)
    s.rect(5, 147, 155, 152, "#f6fcff", LINE, 3)
    s.text(13, 163, "Runtime graph analysis", 9.5, NAVY, True)
    items = [
        ("network", "Target encoding", "Encode the target claim"),
        ("search", "Eligible neighbors", "Find eligible historical claims"),
        ("plus", "Temporary insertion", "Insert into the local graph"),
        ("bars", "Graph fact cards", "Compute 14 graph descriptors"),
    ]
    for i, (kind, title, body) in enumerate(items):
        y = 171 + i * 31
        icon(s, kind, 11, y + 1, 17)
        s.text(37, y + 8, title, 8.8, NAVY, True)
        para(s, 37, y + 20, body, 116, 7.2, MUTED)
        if i < 3:
            s.line(36, y + 29, 154, y + 29, LINE, 0.4)
    return s


def output_card(width: float, height: float, paper: bool = False) -> Scene:
    s = Scene(width, height, "output")
    box(s, tint="#f5f1ff", stroke="#d6c7f6")
    icon(s, "doc", 9, 9, 27, PURPLE)
    para(
        s,
        45,
        19,
        "Whole-paper report" if paper else "Claim synthesis",
        width - 50,
        10.1,
        PURPLE,
        True,
    )
    para(
        s,
        45,
        44,
        "Traceable report" if paper else "Per claim C1–C4",
        width - 50,
        7.8,
        MUTED,
    )
    labels = (
        [
            "Contributions",
            "Historical increment",
            "Knowledge relations",
            "Graph interpretation",
            "Limitations",
            "Evidence links",
        ]
        if paper
        else [
            "Supported scope",
            "Key finding",
            "Knowledge relation",
            "Assessment",
            "Limitation",
        ]
    )
    for i, text in enumerate(labels):
        yy = 66 + i * 17
        s.node(11, yy - 3, fill=PURPLE, stroke="none", radius=2)
        fit(s, 20, yy, text, width - 27, 8.6, NAVY)
    return s


def panel_a(fig: Figure, data: dict[str, Any]) -> None:
    s = fig.panel("a", "Integrated framework with grounded claims")
    s.text(
        657,
        24,
        "Independent evidence and graph reasoning over shared claims",
        8.7,
        BLUE,
    )
    # Mask the over-wide panel heading only through its reserved title width.
    for x, width, color in [(8, 454, BLUE), (474, 333, BLUE), (814, 213, PURPLE)]:
        s.rect(
            x,
            40,
            width,
            438,
            mix(color, "#ffffff", 0.985),
            mix(color, "#ffffff", 0.85),
            3,
            sw=0.5,
        )
        s.rect(x, 40, width, 35, mix(color, "#ffffff", 0.94), radius=3)
    s.text(22, 62, "Layer 1", 16, NAVY, True)
    s.text(105, 62, "Manuscript grounding", 12.5, NAVY, True)
    s.text(253, 62, "(from text spans to shared claims)", 8.2, BLUE)
    s.text(487, 61, "Layer 2", 15, BLUE, True)
    para(s, 563, 54, "Independent reasoning branches", 228, 10.2, NAVY, True)
    s.text(563, 68, "GEAR analysis and graph reasoning", 8.4, BLUE)
    s.text(830, 61, "Layer 3", 14, PURPLE, True)
    s.text(892, 60, "Synthesis and reporting", 10, PURPLE, True)
    fig.component(
        "a",
        "a_manuscript",
        12,
        80,
        manuscript(data),
        "Target manuscript and source anchors",
    )
    fig.component(
        "a",
        "a_grounding",
        172,
        80,
        grounding(data),
        "Mining, consolidation and support verification",
    )
    s.arrow(164, 254, 172, 254, BLUE, 1.5, 0)
    s.arrow(292, 254, 303, 254, BLUE, 1.5, 0)
    s.text(310, 94, "Shared grounded claim set", 10.2, NAVY, True)
    s.text(332, 107, "Core analysis unit", 8.2, BLUE)
    y = 114
    for i in range(4):
        height = 158 if i == 0 else 57
        fig.component(
            "a",
            f"a_claim_C{i + 1}",
            303,
            y,
            claim_card(data, i, 152, height),
            f"Grounded claim C{i + 1}",
        )
        y += height + 7
    fig.component(
        "a",
        "a_gear_branch",
        476,
        80,
        architecture_gear(data),
        "Independent GEAR analysis",
    )
    fig.component(
        "a",
        "a_graph_branch",
        640,
        80,
        architecture_graph(),
        "Independent Graph analysis",
    )
    # Shared claims fork into both branches. No GEAR-to-Graph edge is drawn.
    s.arrow(456, 254, 476, 254, ORANGE, 1.4, 0)
    s.path("M457 254 L468 254 L468 78 L636 78 L636 254", stroke=BLUE, width=1)
    s.arrow(636, 254, 640, 254, BLUE, 1.2, 0)
    j = Scene(258, 77, "joint_arch")
    j.rect(0.5, 0.5, 257, 76, "#f3fcff", "#75cce7", 3, sw=0.75)
    j.text(70, 14, "Per-claim interpretation", 9.5, NAVY, True)
    j.line(7, 23, 250, 23, "#b3dff0", 0.6)
    icon(j, "network", 10, 31, 26)
    fit(j, 49, 42, "Union neighborhood → Joint Graph analysis", 204, 10, NAVY, True)
    j.text(49, 59, "Multi-claim neighborhood (C1–C4)", 9, BLUE)
    fig.component(
        "a", "a_graph_interpretation", 546, 390, j, "Per-claim and joint Graph paths"
    )
    s.arrow(722, 385, 722, 390, BLUE, 1, 0)
    s.path("M799 375 H810 V439 H805", stroke=CYAN, width=0.9)
    s.arrow(810, 439, 802, 439, CYAN, 0.9, 0)
    # GEAR has its own output route; the graph interpretation enclosure excludes it.
    s.path("M530 385 V477 H434 V489", stroke=ORANGE, width=1.1)
    s.path("M679 468 V483 H732 V489", stroke=CYAN, width=1.1)
    fig.component(
        "a",
        "a_claim_output",
        889,
        100,
        output_card(133, 159),
        "Standard per-claim synthesis",
    )
    fig.component(
        "a",
        "a_paper_output",
        889,
        271,
        output_card(133, 181, True),
        "Study whole-paper report",
    )
    for y, label, color in [(149, "GEAR (Ci)", ORANGE), (178, "Graph (Ci)", BLUE)]:
        s.text(819, y - 7, label, 8.7, color, True)
        s.arrow(815, y, 885, y, color, 1.2, 0)
    ports = [
        "Manuscript",
        "Shared claims",
        "GEAR analyses",
        "Per-claim Graph",
        "Joint Graph",
        "Source catalog",
    ]
    for i, text in enumerate(ports):
        yy = 295 + i * 27
        color = [ORANGE, ORANGE, ORANGE, BLUE, CYAN, PURPLE][i]
        s.text(817, yy - 6, text, 7.8, color)
        s.arrow(816, yy, 885, yy, color, 0.95, 0)
    outputs = Scene(1019, 86, "outputs")
    outputs.rect(0.5, 0.5, 1018, 85, "#f3effc", "#e7e0f5", 3)
    outputs.text(29, 53, "Outputs", 19, PURPLE, True)
    outputs.rect(180, 9, 365, 66, "#fefcff", "#cdbbee", 3)
    outputs.text(193, 28, "GEAR (Ci)", 9, ORANGE, True)
    outputs.arrow(194, 35, 266, 35, ORANGE, 1, 0)
    outputs.text(193, 54, "Graph (Ci)", 9, BLUE, True)
    outputs.arrow(194, 61, 266, 61, BLUE, 1, 0)
    icon(outputs, "doc", 278, 23, 29, PURPLE)
    outputs.text(323, 35, "Claim-level synthesis", 13, PURPLE, True)
    para(
        outputs,
        323,
        52,
        "Combine independent evidence for the same claim",
        210,
        8.4,
        MUTED,
    )
    outputs.rect(561, 9, 442, 66, "#fefcff", "#cdbbee", 3)
    para(
        outputs,
        571,
        26,
        "Manuscript + shared claims\nGEAR + per-claim Graph + Joint Graph\nSource catalog",
        162,
        8.2,
        NAVY,
    )
    outputs.arrow(736, 42, 764, 42, PURPLE, 1.2, 0)
    icon(outputs, "doc", 772, 22, 30, PURPLE)
    outputs.text(816, 35, "Whole-paper report", 13, PURPLE, True)
    outputs.text(816, 54, "Integrated, source-linked analysis", 8.7, MUTED)
    fig.component("a", "a_outputs", 8, 489, outputs, "Two distinct output paths")


def example_strip(width: float) -> Scene:
    s = Scene(width, 24, "example")
    s.rect(0.4, 0.4, width - 0.8, 23.2, "#eaf5ff", radius=3)
    s.text(7, 16, "Example claim: C1 (Mechanism)", 10.8, NAVY, True)
    fit(
        s,
        183,
        16,
        "PVH–BAT control of social-stress susceptibility",
        width - 190,
        9,
        BLUE,
    )
    return s


def timeline(data: dict[str, Any]) -> Scene:
    s = Scene(352, 115, "timeline")
    box(s, "Retrieved prior works (selected)", "#fff5ef", "#f1d7c5", RED)
    years = [w["record"]["payload"]["publication_year"] for w in data["works"]]
    lo, hi = min(years), 2026
    s.arrow(18, 63, 308, 63, BLUE, 1, 0)
    for i, work in enumerate(data["works"]):
        year = work["record"]["payload"]["publication_year"]
        x = 27 + (year - lo) / (hi - lo) * 246
        s.node(x, 63, fill=CLAIM_COLORS[i], radius=4.5, stroke=ORANGE)
        s.text(x, 38, work["alias"], 10, NAVY, True, anchor="middle")
        s.text(x, 51, str(year), 8.5, MUTED, anchor="middle")
        text = {
            "W1": "DMH–BAT\nthermogenesis",
            "W2": "Hypothalamic\nenergy control",
            "W3": "PVH\nconnectivity",
        }[work["alias"]]
        para(s, x - 24, 82, text, 68, 7.8, MUTED)
    s.line(291, 26, 291, 106, ORANGE, 0.9, "3 2")
    icon(s, "doc", 316, 45, 22, "#8295bd")
    s.text(310, 87, "Target", 8, NAVY)
    s.text(310, 99, "2026", 8, MUTED)
    return s


def comparison() -> Scene:
    s = Scene(299, 214, "comparison")
    box(s, "Comparison with prior evidence", "#fff5ee", "#ecd6c5", RED)
    cols = [0, 79, 185, 299]
    s.rect(1, 24, 297, 24, "#e5f4fd")
    for x, text, width in [
        (5, "Aspect", 70),
        (84, "Prior evidence", 97),
        (190, "Target contribution (C1)", 104),
    ]:
        fit(s, x, 40, text, width, 8.2, NAVY, True)
    rows = [
        (
            "Object / problem",
            "BAT thermogenesis and energy balance [W1, W2]",
            "Social-stress susceptibility in mice [S1]",
        ),
        (
            "Method / mechanism",
            "DMH GLP-1R signaling; developmental PVH connectivity [W1, W3]",
            "PVH–BAT tracing and chemogenetic activation; β3AR implicated [S2, S3]",
        ),
        (
            "Conditions / scope",
            "Adult male rats [W1]; postnatal connectivity in mice [W3]",
            "Social-defeat mouse models; pathway mediation remains limited [S1–S3]",
        ),
    ]
    for i, row in enumerate(rows):
        y = 48 + i * 55
        s.rect(1, y, 297, 55, "#ffffff" if i % 2 else "#f9fcff")
        for j, text in enumerate(row):
            para(
                s,
                cols[j] + 5,
                y + 13,
                text,
                cols[j + 1] - cols[j] - 10,
                8.25 if j else 8.2,
                NAVY if j == 0 else INK,
                j == 0,
            )
        s.line(1, y + 55, 298, y + 55, LINE, 0.45)
    for x in cols[1:-1]:
        s.line(x, 24, x, 213, LINE, 0.5)
    return s


def reasoning_card(
    title: str, body: str, kind: str, color: str, height: float
) -> Scene:
    s = Scene(176, height, "reasoning")
    s.rect(
        0.5,
        0.5,
        175,
        height - 1,
        mix(color, "#ffffff", 0.965),
        mix(color, "#ffffff", 0.76),
        3,
    )
    icon(s, kind, 7, 10, 20, color)
    fit(s, 35, 16, title, 136, 10, color, True)
    para(s, 35, 30, body, 134, 8.25)
    return s


def panel_b(fig: Figure, data: dict[str, Any]) -> None:
    s = fig.panel("b", "GEAR historical evidence analysis", RED, "#fff5ed")
    fig.component("b", "b_example", 7, 36, example_strip(486), "C1 identity")
    search = Scene(126, 115, "search")
    box(search, "Retrieve prior works", "#fff5ee", "#eed7c3", RED, 10)
    for y, kind, title, body in [
        (29, "search", "Normal search", "Object, mechanism\nand author terms"),
        (
            72,
            "network",
            "Contrastive search",
            "Alternative routes\nand mechanisms",
        ),
    ]:
        icon(search, kind, 7, y + 1, 21)
        fit(search, 36, y + 10, title, 83, 9, NAVY, True)
        para(search, 36, y + 23, body, 83, 7.3, MUTED)
    fig.component("b", "b_search", 7, 68, search, "Search framing (method summary)")
    fig.component(
        "b", "b_timeline", 140, 68, timeline(data), "Actual historical works and dates"
    )
    s.text(
        145, 195, "Cutoff: 23 Jan 2026  |  Target publication: 23 Jan 2026", 8.2, RED
    )
    fig.component(
        "b", "b_comparison", 7, 202, comparison(), "Source-bound scientific comparison"
    )
    main = Scene(176, 105, "main_status")
    box(main, "Main C1 · Limited history", "#fff7f0", "#eed7c3", RED, 9.5)
    para(
        main,
        8,
        37,
        "Building blocks [W1, W3]; one relation remains unresolved. Any increment is scope-limited.",
        160,
        8.2,
    )
    fit(main, 8, 96, "GEAR: inconclusive", 160, 8.5, RED, True)
    fig.component(
        "b",
        "b_main_status",
        316,
        202,
        main,
        "Main C1: retrieval coverage is sufficient but one historical relation remains unresolved",
    )
    contrast = Scene(176, 125, "history_contrast")
    box(contrast, "X · Partial antecedent", "#f4f8ff", LINE, NAVY, 10)
    para(contrast, 8, 36, "IL-coated nanoparticles [X1]", 160, 8.4)
    para(
        contrast,
        8,
        64,
        "2014: polymer–gold micelles.\nTarget: PLGA + outer IL.",
        160,
        8.2,
    )
    fit(contrast, 8, 104, "GEAR: residual extension", 160, 8.2, NAVY, True)
    fit(contrast, 8, 117, "Prior source: abstract", 160, 7.8)
    fig.component(
        "b",
        "b_history_contrast",
        316,
        312,
        contrast,
        "X: saved partial-antecedent judgment based on the prior abstract",
    )
    s.text(13, 431, "Main C1: 3 of 6 prior works shown.", 8, MUTED)


def community_map(data: dict[str, Any]) -> dict[int, str]:
    ids = sorted(
        {
            n["community_id"]
            for n in data["historical"].values()
            if n["community_id"] is not None
        }
    )
    ids = data.get("display_community_order", ids)
    return {cid: COMMUNITIES[i] for i, cid in enumerate(ids)}


def scientific_network(
    data: dict[str, Any],
    layout: dict[str, list[float]],
    width: float,
    height: float,
    joint: bool = False,
) -> Scene:
    s = Scene(width, height, "joint_network" if joint else "local_network")
    box(
        s,
        "Union neighborhood (C1–C4)" if joint else "Local claim neighborhood",
        title_size=10.7,
    )
    colors = community_map(data)
    claims = data["claims"] if joint else data["claims"][:1]
    targets = {c["claim_id"]: c for c in claims}
    neighbor_ids = set(layout) - set(targets)
    positions = {
        n: [18 + xy[0] * (width - 66), 40 + xy[1] * (height - (85 if joint else 60))]
        for n, xy in layout.items()
    }
    historic_edges = (
        data["joint"]["historical_edges"]
        if joint
        else data["facts"][0]["neighbor_edges"]
    )
    insertion = (
        data["joint"]["insertion_edges"]
        if joint
        else [[claims[0]["claim_id"], n] for n in neighbor_ids]
    )
    for u, v in historic_edges:
        edge = s.line(*positions[u], *positions[v], "#a9b3c1", 0.65, opacity=0.8)
        edge.set("data-source-edge", f"{u}|{v}")
    for u, v in insertion:
        color = CLAIM_COLORS[list(targets).index(u)]
        edge = s.line(*positions[u], *positions[v], color, 0.65, "2 1.8", 0.72)
        edge.set("data-source-edge", f"{u}|{v}")
    for n in sorted(neighbor_ids):
        node = data["historical"][n]
        x, y = positions[n]
        if joint and n in data["joint"]["shared_historical_neighbors"]:
            s.add(
                el(
                    "circle",
                    cx=x,
                    cy=y,
                    r=7.8,
                    fill="none",
                    stroke="#a8b0bc",
                    stroke_width=1.5,
                )
            )
        s.node(
            x,
            y,
            node["claim_type"],
            colors.get(node["community_id"], "#b8bdc7"),
            4.4 if joint else 5.1,
            "#7184a2",
            0.55,
            n,
        )
        if not joint:
            s.text(x + 6, y - 4, data["aliases"][n], 6.8, MUTED)
    for i, (n, claim) in enumerate(targets.items()):
        x, y = positions[n]
        s.node(
            x,
            y,
            claim["claim_type"],
            "#ffffff",
            8.1 if joint else 8.8,
            CLAIM_COLORS[i],
            2.1,
            n,
        )
        label_y = y + ([-3, -9, 15, 4][i] if joint else 4)
        s.rect(x + 9, label_y - 9, 17, 12, "white", opacity=0.92)
        s.text(x + 10, label_y, data["aliases"][n], 11.5, CLAIM_COLORS[i], True)
    if joint:
        s.text(10, height - 24, "27 historical nodes · 4 target claims", 8, MUTED)
        s.text(10, height - 10, "No target-to-target edges are added.", 8.1, BLUE)
    return s


def role_legend(width: float, height: float, data: dict[str, Any]) -> Scene:
    s = Scene(width, height, "legend")
    box(s)
    s.text(6, 13, "Community (color)", 8.5, NAVY, True)
    for i, (cid, color) in enumerate(community_map(data).items()):
        x = 10 + i * 33
        s.node(x, 26, fill=color, radius=3.5, stroke="none")
        s.text(x + 5, 29, str(cid), 7.5, MUTED)
    s.text(6, 44, "Role (shape)", 8.5, NAVY, True)
    for i, role in enumerate(["FINDING", "METHOD", "MECHANISM", "RESOURCE", "THEORY"]):
        y = 57 + i * 14
        s.node(11, y, role, "#f8fbff", 3.5, BLUE, 0.8)
        s.text(21, y + 3, role.title(), 8, MUTED)
    s.line(6, 133, 25, 133, "#a8b2c0", 0.8)
    s.text(30, 136, "Historical", 7.6, MUTED)
    s.line(6, 143, 25, 143, BLUE, 0.8, "2 2")
    s.text(30, 146, "Inserted", 7.6, MUTED)
    return s


def atlas(data: dict[str, Any]) -> Scene:
    s = Scene(109, 149, "atlas")
    box(s, "Historical atlas (Fig.1)", title_size=8.5)
    a = data["atlas"]
    nodes = {n["claim_id"]: n for n in a["nodes"]}
    mapping = {cid: COMMUNITIES[i] for i, cid in enumerate(a["focus_communities"])}
    xs, ys = [n["x"] for n in a["layout"]], [n["y"] for n in a["layout"]]
    points = {
        n["claim_id"]: (
            5 + (n["x"] - min(xs)) / (max(xs) - min(xs)) * 99,
            27 + (n["y"] - min(ys)) / (max(ys) - min(ys)) * 98,
        )
        for n in a["layout"]
    }
    for u, v in a["edges"]:
        if u in points and v in points:
            s.line(*points[u], *points[v], "#c5cddd", 0.12, opacity=0.22)
    for n, xy in points.items():
        s.node(
            *xy,
            fill=mapping.get(nodes[n].get("community_id"), "#cbd1da"),
            radius=0.57,
            stroke="none",
            ident=f"atlas-{n}",
        )
    s.text(8, 139, "Observed context sample", 7.2, MUTED)
    return s


def parent_path(data: dict[str, Any]) -> Scene:
    s = Scene(116, 149, "parent_path")
    box(s, "Parent-paper path context", title_size=8.4)
    witness = data["witness"]
    s.text(14, 38, "Target P", 8.7, NAVY)
    s.text(71, 38, "P(H2)", 8.7, NAVY)
    for x, color in [(30, "#f4aacb"), (89, "#bac5d4")]:
        s.node(x, 53, fill=color, radius=5, stroke="#7c91ad")
        s.arrow(x, 59, 59, 96, BLUE, 0.9, 0)
    s.node(59, 102, fill="#f1c16d", radius=5, stroke="#b69c6d")
    s.text(40, 120, "Shared R", 8.3, NAVY)
    s.text(8, 138, "Citation context only", 7.7, MUTED)
    s.root.set("data-witness", str(witness["edges"]))
    return s


def metric_group(
    data: dict[str, Any], title: str, ids: list[int], width: float, height: float
) -> Scene:
    s = Scene(width, height, "metric_group")
    s.rect(0.5, 0.5, width - 1, height - 1, "white", LINE, 3)
    s.rect(1, 1, width - 2, 17, "#edf7ff", radius=3)
    fit(s, 6, 12.8, title, width - 12, 8.8, NAVY, True)
    for i, index in enumerate(ids):
        name, label = METRICS[index - 1]
        value = data["metrics"][name]
        formatted = (
            "N/A"
            if value is None
            else str(value)
            if isinstance(value, int)
            else f"{value:.3f}".rstrip("0").rstrip(".")
        )
        y = 27 + i * 13
        s.text(6, y, f"D{index}", 7.5, BLUE, True)
        fit(s, 28, y, label, width - 76, 8, INK)
        s.text(width - 7, y, formatted, 8.8, NAVY, True, anchor="end")
    return s


def insight(
    width: float, height: float, title: str, body: str, kind: str = "network"
) -> Scene:
    s = Scene(width, height, "insight")
    s.rect(0.5, 0.5, width - 1, height - 1, "#f2faff", LINE, 3)
    icon(s, kind, 7, 12, 23, BLUE)
    s.text(38, 15, title, 9.5, NAVY, True)
    para(s, 38, 26, body, width - 45, 8.1, MUTED)
    return s


def contrast_network(
    data: dict[str, Any],
    case: dict[str, Any],
    layout: dict[str, list[float]],
    width: float = 167,
) -> Scene:
    """Complete retained neighborhood; an empty result is never padded with nodes."""
    s = Scene(width, 149, "contrast_network")
    alias = case["alias"]
    box(s, f"{alias} · {case['title']}", title_size=9.7)
    if alias == data.get("detailed_profile", {}).get("alias"):
        s.rect(1, 1, width - 2, 147, "none", BLUE, 3, sw=1.4)
    fact = case["fact"]
    target = case["claim"]["claim_id"]
    nodes = {n["claim_id"]: n for n in fact["neighbors"]}
    positions = {
        n: (14 + xy[0] * (width - 40), 40 + xy[1] * 59) for n, xy in layout.items()
    }
    if not nodes:
        positions[target] = (width / 2 - 8, 68)
    colors = community_map(data)
    for u, v in fact["neighbor_edges"]:
        line = s.line(*positions[u], *positions[v], "#a9b3c1", 0.8)
        line.set("data-source-edge", f"{u}|{v}")
    for n, node in nodes.items():
        line = s.line(*positions[target], *positions[n], BLUE, 0.7, "2 1.8", 0.72)
        line.set("data-source-edge", f"{target}|{n}")
        s.node(
            *positions[n],
            node["claim_type"],
            colors.get(node["community_id"], "#b8bdc7"),
            4.5,
            "#7184a2",
            0.6,
            n,
        )
    x, y = positions[target]
    s.node(x, y, case["claim"]["claim_type"], "white", 7, BLUE, 2, target)
    s.text(x + 10, y + 4, alias, 10, BLUE, True)
    communities = {
        n["community_id"] for n in nodes.values() if n["community_id"] is not None
    }
    if nodes:
        fit(
            s,
            7,
            122,
            f"{len(nodes)} neighbors · {len(communities)} {'community' if len(communities) == 1 else 'communities'}",
            width - 14,
            8.3,
            NAVY,
            True,
        )
        effective = case["metrics"]["effective_community_count"]
        fit(s, 7, 139, f"Effective count: {effective:.2f}", width - 14, 8.2)
    else:
        fit(s, 7, 105, "0 eligible neighbors", width - 14, 8.5, NAVY, True)
        fit(s, 7, 122, "Descriptors: N/A", width - 14, 8.3)
        fit(s, 7, 139, "No Graph assessment saved", width - 14, 7.6)
    return s


def panel_c(fig: Figure, data: dict[str, Any], layouts: dict[str, Any]) -> None:
    s = fig.panel("c", "Single-claim structural profiles")
    profile = data["detailed_profile"]
    strip = Scene(506, 23, "detailed_example")
    strip.rect(0.5, 0.5, 505, 22, "#e5f2ff", LINE, 2)
    fit(
        strip,
        7,
        16,
        "Detailed example: X (Finding) · Ionic-liquid-coated gold–PLGA nanoparticles",
        492,
        10,
        NAVY,
        True,
    )
    strip.root.set("data-claim-id", profile["claim_id"])
    fig.component(
        "c",
        "c_example",
        7,
        36,
        strip,
        "X is the detailed profile; C1 is the single-community contrast",
        ["data/detailed_profile_X.json"],
    )
    main = {
        "alias": "C1",
        "title": "PVH–BAT stress",
        "claim": data["claims"][0],
        "fact": data["facts"][0],
        "metrics": data["metrics"],
    }
    fig.component(
        "c",
        "c_local_network",
        7,
        66,
        contrast_network(data, main, layouts["local"], 249),
        "Complete C1 insertion neighborhood",
    )
    for case, x in zip(data["contrasts"], [263]):
        fig.component(
            "c",
            f"c_contrast_{case['alias']}",
            x,
            66,
            contrast_network(data, case, layouts[case["alias"]], 249),
            case["selection_reason"],
        )
    s.text(
        11,
        227,
        "Rules: before cutoff · target excluded · cosine > 0.5 · at most 10 neighbors",
        8.3,
        BLUE,
    )
    s.text(11, 240, "14 descriptors · X (Finding)", 10, NAVY, True)
    fit(s, 270, 240, "Community-pair history: 10 of 10", 240, 8.2, MUTED)
    groups = [
        ("c_semantic", 7, 246, 250, 45, "1  Semantic proximity", [1, 2]),
        ("c_roles", 7, 295, 250, 32, "3  Contribution-role mixing", [8]),
        ("c_connectivity", 7, 331, 250, 58, "4  Local connectivity", [9, 10, 11]),
        (
            "c_communities",
            263,
            246,
            249,
            84,
            "2  Community composition & pair history",
            [3, 4, 5, 6, 7],
        ),
        (
            "c_paths",
            263,
            334,
            249,
            58,
            "5  Parent-paper path context · Context only",
            [12, 13, 14],
        ),
    ]
    # Keep all fourteen labels readable in the two-column arrangement.
    for ident, x, y, w, h, title, ids in groups:
        component = metric_group(profile, title, ids, w, h)
        component.root.set("data-claim-id", profile["claim_id"])
        fig.component(
            "c",
            ident,
            x,
            y,
            component,
            "X · " + title,
            ["data/detailed_profile_X.json"],
        )
    legend = Scene(505, 40, "encoding_legend")
    legend.rect(0.5, 0.5, 504, 39, "#f2faff", LINE, 3)
    for i, role in enumerate(["METHOD", "FINDING", "MECHANISM", "RESOURCE", "THEORY"]):
        x = 10 + i * 100
        legend.node(x, 12, role, "#c6dcf3", 3.7, "#526c88", 0.7)
        fit(legend, x + 8, 16, role.title(), 82, 7.8)
    fit(
        legend,
        7,
        33,
        "Color: community · Dashed: insertion · Structure is not a novelty score",
        491,
        8.2,
    )
    fig.component(
        "c",
        "c_encoding_legend",
        7,
        396,
        legend,
        "Shared node and edge encodings; graph coverage is not a novelty judgment",
    )


def joint_statistics(data: dict[str, Any]) -> Scene:
    s = Scene(192, 150, "joint_stats")
    box(s, "Joint graph statistics", title_size=11)
    j = data["joint"]
    rows = [
        ("Unique historical neighbors", str(j["historical_neighbor_count"])),
        ("Shared historical neighbors", str(len(j["shared_historical_neighbors"]))),
        (
            "Historical components\n(before → after)",
            f"{j['historical_components_before']} → {j['historical_components_after']}",
        ),
        (
            "Newly connected\nhistorical pairs",
            str(j["joint_newly_connected_historical_pairs"]),
        ),
    ]
    for i, (label, value) in enumerate(rows):
        y = 36 + i * 29
        para(s, 8, y, label, 127, 8.6)
        s.text(181, y + (6 if i > 1 else 1), value, 12, NAVY, True, anchor="end")
        if i < 3:
            s.line(7, y + 15, 185, y + 15, LINE, 0.4)
    return s


def panel_d(fig: Figure, data: dict[str, Any], layouts: dict[str, Any]) -> None:
    s = fig.panel("d", "Joint Graph analysis")
    s.text(43, 40, "Across the paper’s contributions", 10, BLUE)
    fig.component(
        "d",
        "d_joint_network",
        7,
        49,
        scientific_network(data, layouts["joint"], 288, 285, True),
        "Complete union neighborhood",
    )
    fig.component(
        "d", "d_statistics", 301, 49, joint_statistics(data), "Joint Graph counts"
    )
    interpretation = Scene(192, 130, "joint_interpretation")
    box(interpretation, "Interpretation", title_size=11)
    icon(interpretation, "network", 8, 25, 16)
    interpretation.text(37, 36, "Observed structure", 9.3, NAVY, True)
    para(
        interpretation,
        8,
        53,
        "Shared neighbors connect all four claims in the joint graph.",
        176,
        8.1,
    )
    icon(interpretation, "bulb", 8, 77, 15, ORANGE)
    interpretation.text(37, 88, "Joint interpretation", 9.3, NAVY, True)
    para(
        interpretation,
        8,
        104,
        "Complementary circuit, BAT and intervention evidence; IL-6 mediation remains unproven.",
        176,
        8.1,
    )
    fig.component(
        "d",
        "d_interpretation",
        301,
        204,
        interpretation,
        "Observed structure and scientific interpretation",
    )
    legend = Scene(131, 28, "joint_legend")
    legend.rect(0, 0, 131, 28, "#ffffff", opacity=0.94)
    legend.add(
        el("circle", cx=8, cy=8, r=5.8, fill="none", stroke="#a8b0bc", stroke_width=1.5)
    )
    legend.node(8, 8, radius=3.2, fill=BLUE, stroke="none")
    legend.text(19, 11, "Shared historical neighbor", 7.6, MUTED)
    legend.line(3, 23, 16, 23, BLUE, 0.8, "2 2")
    legend.text(19, 26, "Temporary insertion edge", 7.6, MUTED)
    fig.component(
        "d", "d_legend", 153, 74, legend, "Joint graph shared-neighbor and edge legend"
    )


def synthesis_core() -> Scene:
    s = Scene(128, 89, "synthesis_core")
    box(s, "Synthesis core", "#f5f0ff", "#d6c5f1", PURPLE, 10)
    icon(s, "gear", 9, 40, 25, "#8b9fbc")
    for i, text in enumerate(["Match scope", "Ground findings", "Retain disagreement"]):
        s.node(44, 38 + i * 17, fill=BLUE, radius=1.8, stroke="none")
        fit(s, 51, 41 + i * 17, text, 72, 8.2, NAVY)
    return s


def fusion_schema() -> Scene:
    s = Scene(207, 107, "fusion_schema")
    box(s, "Claim assessment · output schema", "#f4effe", "#d6c5f1", PURPLE, 9.4)
    rows = [
        ("Supported scope", "Claim-specific scope"),
        ("Key finding", "Text + evidence keys"),
        ("Overall assessment", "Recorded stance"),
        ("Limitation", "Unresolved evidence / scope"),
    ]
    for i, (label, field) in enumerate(rows):
        y = 35 + i * 19
        fit(s, 6, y, label, 91, 8.1, NAVY, True)
        fit(s, 101, y, field, 100, 7.8, MUTED)
        if i < 3:
            s.line(5, y + 5, 201, y + 5, "#ded4f3", 0.4)
    return s


def report_output(data: dict[str, Any]) -> Scene:
    s = Scene(177, 111, "report_output")
    box(s, "Innovation analysis report", "#f4effe", "#d6c5f1", PURPLE, 9.5)
    icon(s, "doc", 7, 29, 25, PURPLE)
    para(
        s,
        42,
        36,
        "PVH–BAT stress regulation: a working model; intermediate mechanisms remain unresolved.",
        128,
        8.2,
    )
    s.line(7, 78, 170, 78, "#dfd4f1", 0.6)
    fit(s, 7, 89, "Contributions · Historical increment", 163, 8, NAVY)
    fit(s, 7, 102, "Knowledge relations · Limitations", 163, 8, NAVY)
    return s


def source_appendix(data: dict[str, Any]) -> Scene:
    s = Scene(199, 111, "source_appendix")
    box(s, "Source appendix (traceable)", "#f4effe", "#d6c5f1", PURPLE, 9.5)
    s.text(7, 33, "Alias / type", 7.8, NAVY, True)
    s.text(76, 33, "Source excerpt", 7.8, NAVY, True)
    rows = [
        ("S3 · Manuscript", "“…connection between the PVH and BAT is polysynaptic.”"),
        ("W1 · Abstract", "DMH GLP-1 signaling and BAT thermogenesis (summary)"),
    ]
    for i, (label, body) in enumerate(rows):
        y = 48 + i * 34
        s.text(7, y, label, 7.8, NAVY)
        para(s, 76, y, body, 116, 7.6, MUTED)
        s.line(6, y + 25, 193, y + 25, "#e2d9f2", 0.4)
    return s


def panel_e(fig: Figure, data: dict[str, Any]) -> None:
    s = fig.panel("e", "Scope-aware synthesis and reporting", NAVY, "#f4effd")
    s.rect(7, 35, 505, 121, "#fbf9ff", "#dfd3f6", 3)
    branches = Scene(249, 107, "branch_judgments")
    box(branches, "Saved C1 branch assessments", title_size=10.2)
    fit(branches, 8, 36, "GEAR: unresolved", 233, 8.8, RED, True)
    fit(branches, 8, 52, "Scoped increment; firstness unresolved.", 233, 8.1)
    branches.line(8, 60, 241, 60, LINE, 0.6)
    fit(branches, 8, 76, "Graph: recognized", 233, 8.8, BLUE, True)
    fit(branches, 8, 94, "Mechanism claim, within provided context.", 233, 8.1)
    fig.component(
        "e",
        "e_branch_judgments",
        14,
        43,
        branches,
        "Observed independent C1 stances, before fusion",
    )
    method = Scene(225, 107, "synthesis_method")
    box(method, "Claim synthesis · method", "#f4effe", "#d6c5f1", PURPLE, 10.2)
    fit(method, 9, 37, "Match scope · Ground findings", 207, 8.7)
    fit(method, 9, 56, "Retain disagreement and limitations", 207, 8.5)
    para(method, 9, 80, "No saved per-claim fusion for this example.", 207, 8.3, PURPLE)
    fig.component(
        "e",
        "e_synthesis_method",
        281,
        43,
        method,
        "Generic synthesis mechanism; no invented per-claim fused output",
    )
    s.arrow(265, 96, 278, 96, PURPLE, 1.2, 0)
    s.rect(7, 162, 505, 130, "#fcfaff", "#dfd3f6", 3)
    s.text(14, 177, "Paper level (integrated and traceable)", 11.5, PURPLE, True)
    ports = Scene(107, 109, "paper_ports")
    ports.rect(0.5, 0.5, 106, 108, "#f7fbff", LINE, 2)
    labels = [
        "Manuscript",
        "Shared claims C1–C4",
        "GEAR analyses",
        "Per-claim Graph",
        "Joint Graph",
        "Source catalog",
    ]
    for i, label in enumerate(labels):
        color = [PURPLE, ORANGE, ORANGE, BLUE, CYAN, PURPLE][i]
        icon(ports, "doc" if i in [0, 1, 5] else "network", 5, 4 + i * 17, 10, color)
        fit(ports, 21, 13 + i * 17, label, 81, 8.1, NAVY)
    fig.component(
        "e", "e_paper_inputs", 13, 181, ports, "Six inputs to the whole-paper report"
    )
    s.arrow(121, 230, 134, 230, PURPLE, 1.2, 0)
    fig.component(
        "e",
        "e_report",
        135,
        180,
        report_output(data),
        "Translated summary of the actual whole-paper report",
    )
    fig.component(
        "e",
        "e_source_appendix",
        316,
        180,
        source_appendix(data),
        "Readable source excerpts and types",
    )
    provenance = Scene(505, 36, "provenance")
    provenance.rect(0.5, 0.5, 504, 35, "#f6faff", LINE, 3)
    icon(provenance, "link", 7, 9, 17, PURPLE)
    provenance.text(32, 22, "Provenance", 10, PURPLE, True)
    provenance.text(119, 14, "Finding → Evidence key → Evidence record", 8.4, NAVY)
    provenance.line(107, 7, 107, 29, "#c7b7eb", 0.7)
    provenance.text(119, 29, "Report citation → Source ID → Source excerpt", 8.4, NAVY)
    fig.component(
        "e", "e_provenance", 7, 300, provenance, "Two distinct traceability paths"
    )


def render(out: Path, only: str | None = None) -> None:
    data = read(out / "data/snapshot.json")
    layouts = read(out / "layouts/networks.json")
    fig = Figure(out)
    panel_a(fig, data)
    panel_b(fig, data)
    panel_c(fig, data, layouts)
    panel_d(fig, data, layouts)
    panel_e(fig, data)
    if only and only not in fig.panels and only not in fig.components:
        raise ValueError(f"Unknown panel/component: {only}")
    # Rendering is inexpensive; rebuilding SVG sources keeps the assembled master current.
    # Raster export honors --only, preserving other component and panel renderings.
    fig.save()
    print(
        f"Rendered {len(fig.components)} independent components and 5 panels.",
        flush=True,
    )
