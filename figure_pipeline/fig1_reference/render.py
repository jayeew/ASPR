"""Reference-proportioned publication artwork, assembled from independent SVGs."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import ConvexHull

from .data import COLORS, ROLE_COLORS, ROLES, bare, read, write
from .svg import INK, Scene, el, measure, mix, safe_id

PANEL_BOXES = {
    "a": [8, 46, 777, 562],
    "b": [8, 615, 1039, 487],
    "c": [793, 46, 254, 606],
    "d": [8, 1112, 590, 309],
    "e": [606, 1112, 440, 309],
}
TITLES = {
    "a": "Two-layer historical knowledge atlas",
    "b": "Local claim-insertion profiles",
    "c": "Graph populations",
    "d": "Community organization",
    "e": "Relation types and cross-layer context",
}
CASE_TITLES = {
    "A": "Semantic continuation",
    "B": "Cross-community combination",
    "C": "Role interleaving",
    "D": "Local connectivity reorganization",
}
GRADS = ["bluecard", "orangecard", "greencard", "redcard"]
CASE_COLORS = ["#0873c2", "#dc620a", "#138243", "#d3243b"]


class Figure:
    def __init__(self, out: Path) -> None:
        self.out = out
        self.data = read(out / "data/snapshot.json")
        self.components: dict[str, Scene] = {}
        self.entries: list[dict[str, Any]] = []
        self.panels: dict[str, Scene] = {}
        self.community_colors = {
            cid: COLORS[i]
            for i, cid in enumerate(self.data["atlas"]["focus_communities"])
        }

    def component(
        self, ident: str, w: float, h: float, title: str, source: str
    ) -> Scene:
        s = Scene(w, h, ident)
        s.root.set("data-component", ident)
        self.components[ident] = s
        self.entries.append(
            {
                "id": ident,
                "title": title,
                "width": w,
                "height": h,
                "source": source,
                "placements": [],
            }
        )
        return s

    def place(self, parent: Scene, child: Scene, x: float, y: float) -> None:
        parent.use(child, x, y)
        next(e for e in self.entries if e["id"] == child.ident)["placements"].append(
            {"parent": parent.ident, "x": x, "y": y}
        )

    def panel(self, key: str) -> Scene:
        _, _, w, h = PANEL_BOXES[key]
        panel = Scene(w, h, f"panel_{key}")
        bg = self.component(
            f"{key}_background", w, h, f"Panel {key} frame", "layout only"
        )
        if key == "b":
            bg.path(
                f"M4,0 H773 V37 H{w - 4} Q{w},37 {w},41 V{h - 4} Q{w},{h} {w - 4},{h} H4 Q0,{h} 0,{h - 4} V4 Q0,0 4,0 Z",
                "#ffffff",
                "#b5d9e9",
            )
            bg.rect(1, 1, 774, 36, "url(#bluewash)")
        else:
            bg.rect(0, 0, w, h, "#ffffff", "#90b8cf", 3)
            bg.rect(1, 1, w - 2, 36, "url(#bluewash)")
        self.place(panel, bg, 0, 0)
        label = self.component(
            f"{key}_label", 30, 30, f"Panel {key} label", "layout only"
        )
        label.text(1, 24, key, 27, "#000000", True, sans=True)
        self.place(panel, label, 8, 0)
        header = self.component(
            f"{key}_title", w - 46, 49 if key == "c" else 43, TITLES[key], "layout only"
        )
        fs = 20 if key in "ab" else 18 if key == "d" else 16.5
        header.text(0, 24, TITLES[key], fs, bold=True)
        if key == "c":
            header.text(0, 43, "and node composition", 18, bold=True)
        self.place(panel, header, 42, 0)
        self.panels[key] = panel
        return panel


def atlas(f: Figure) -> None:
    panel = f.panel("a")
    data = f.data["atlas"]
    network = f.component(
        "a_claim_network",
        620,
        326,
        "Historical Claim Graph",
        "snapshot.atlas.claim_nodes / claim_edges; backbone",
    )
    lookup = {n["claim_id"]: n for n in data["claim_nodes"]}
    # Envelopes are display annotations around actual points, not community membership changes.
    for cid in data["focus_communities"]:
        pts = np.array(
            [
                [n["x"] + 15, n["y"] + 8]
                for n in data["claim_nodes"]
                if n["community_id"] == cid
            ]
        )
        if len(pts) > 3:
            hull = pts[ConvexHull(pts).vertices]
            center = hull.mean(axis=0)
            hull = center + (hull - center) * 1.07
            mids = (hull + np.roll(hull, -1, axis=0)) / 2
            curve = (
                f"M{mids[-1, 0]:.2f},{mids[-1, 1]:.2f} "
                + " ".join(
                    f"Q{x:.2f},{y:.2f} {mx:.2f},{my:.2f}"
                    for (x, y), (mx, my) in zip(hull, mids)
                )
                + " Z"
            )
            network.add(
                el(
                    "path",
                    d=curve,
                    fill=f.community_colors[cid],
                    fill_opacity=".065",
                    stroke=f.community_colors[cid],
                    stroke_opacity=".45",
                    stroke_width=".75",
                    stroke_dasharray="3 3",
                )
            )
    for u, v in data["claim_edges"]:
        a, b = lookup[u], lookup[v]
        edge = network.line(
            a["x"] + 15,
            a["y"] + 8,
            b["x"] + 15,
            b["y"] + 8,
            "#adb8c2",
            0.65,
            opacity=0.57,
        )
        edge.set("id", "backbone-" + safe_id(u + "--" + v))
        edge.set("data-source", u)
        edge.set("data-target", v)
    for n in data["claim_nodes"]:
        color = f.community_colors.get(n["community_id"], "#bfc4c9")
        network.node(
            n["x"] + 15,
            n["y"] + 8,
            n["claim_type"],
            color,
            2.6 + 0.07 * min(n["backbone_degree"], 10),
            mix(color, "#41576a", 0.20),
            0.45,
            "claim-" + safe_id(n["claim_id"]),
        )
    for i, cid in enumerate(data["focus_communities"]):
        pts = np.array(
            [[n["x"], n["y"]] for n in data["claim_nodes"] if n["community_id"] == cid]
        )
        x, y = pts.mean(axis=0)
        if x + 36 < 210 and 270 < y + 49 < 337:
            y = 208  # Keep the community label above the left annotation card.
        network.rect(x + 7, y - 12, 24, 18, "#ffffff", radius=3, opacity=0.85)
        network.text(
            x + 11, y + 1, f"K{i + 1}", 14, mix(COLORS[i], "#000000", 0.30), True
        )
    f.place(panel, network, 25, 48)
    labels = f.component(
        "a_layer_annotations",
        195,
        300,
        "Claim layer annotations",
        "ontology; nodes=claims, color=community, shape=role",
    )
    labels.rect(0, 0, 185, 61, "#d8edfc", radius=6, opacity=0.97)
    labels.text(9, 20, "Claim Graph", 16, bold=True)
    labels.paragraph(9, 37, "Semantic relations and\ncitation context", 168, 12.5, 15)
    labels.rect(0, 231, 193, 58, "white", radius=4, opacity=0.96)
    labels.paragraph(
        9,
        244,
        "Claims as nodes\nColored by community\nShaped by contribution role",
        180,
        12.5,
        16,
    )
    f.place(panel, labels, 10, 43)
    paper_bg = f.component(
        "a_paper_background", 775, 204, "Paper layer background", "layout only"
    )
    paper_bg.path("M0,25 Q325,-20 775,12 V204 H0 Z", "url(#pinkwash)")
    f.place(panel, paper_bg, 1, 357)
    pn = f.component(
        "a_paper_network",
        590,
        165,
        "Citation-extended paper display",
        "snapshot.atlas.paper_layer; directed actual citations",
    )
    papers = data["paper_layer"]
    pmap = {p["work_id"]: p for p in papers["nodes"]}
    target_colors = {
        p["work_id"]: COLORS[ord(p["target_case"]) - ord("A")]
        for p in papers["nodes"]
        if p.get("target_case")
    }
    first_hop_nodes = {v for u, v in papers["edges"] if u in target_colors}
    for u, v in papers["edges"]:
        a, b = pmap[u], pmap[v]
        if u in target_colors:
            color, edge_class = target_colors[u], "target-outgoing"
        elif u in first_hop_nodes and any(
            root != v and [root, u] in papers["edges"] for root in target_colors
        ):
            color, edge_class = "#67717d", "second-hop"
        else:
            color, edge_class = mix("#67717d", "#ffffff", 0.18), "context"
        edge = pn.arrow(
            a["x"] * 0.88 + 58,
            a["y"] + 10,
            b["x"] * 0.88 + 58,
            b["y"] + 10,
            color,
            0.55,
            4.3,
        )
        edge.set("id", "citation-" + safe_id(u + "--" + v))
        edge.set("data-source", u)
        edge.set("data-target", v)
        edge.set("data-edge-class", edge_class)
    for p in papers["nodes"]:
        if p.get("target_case"):
            continue
        pn.node(
            p["x"] * 0.88 + 58,
            p["y"] + 10,
            fill="#c6c9d0",
            radius=4.3,
            stroke="#8190a4",
            ident="paper-" + safe_id(p["work_id"]),
        )
    f.place(panel, pn, 95, 387)
    ownership = f.component(
        "a_ownership_links",
        777,
        562,
        "Selected claim–paper ownership",
        "actual mapping; four independent overlays",
    )
    targetlayer = f.component(
        "a_case_markers",
        777,
        562,
        "Four independently inserted target claims",
        "snapshot.cases; role shape retained",
    )
    for i, c in enumerate(f.data["cases"]):
        pts = np.array(
            [
                [lookup[n["claim_id"]]["x"] + 40, lookup[n["claim_id"]]["y"] + 56]
                for n in c["neighbors"]
            ]
        )
        cx, cy = pts.mean(axis=0) + np.array(
            [[-8, 10], [22, -25], [-12, 23], [8, -9]][i]
        )
        c["atlas_target_xy"] = [float(cx), float(cy)]
        wid = bare(c["input"].get("openalex_work_id") or c["input"]["paper_id"])
        p = pmap[wid]
        px, py = p["x"] * 0.88 + 153, p["y"] + 397
        color = COLORS[i]
        ownership.path(
            f"M{cx},{cy} C{cx + 28},{cy + 125} {px - 26},{py - 70} {px},{py}",
            stroke=color,
            width=1.15,
            dash="4 3",
        )
        for x, y in pts:
            targetlayer.line(cx, cy, x, y, color, 0.95, "4 3", 0.8)
        targetlayer.node(
            cx, cy, c["claim"]["claim_type"], color, 8, INK, 1.7, "target-" + c["label"]
        )
        targetlayer.rect(cx + 9, cy - 19, 30, 17, "white", radius=2, opacity=0.9)
        targetlayer.text(cx + 11, cy - 6, "C" + c["label"], 14, bold=True, sans=True)
        targetlayer.node(px, py, fill=color, radius=8, stroke=INK, sw=1.5)
        targetlayer.text(px + 10, py - 8, "P" + c["label"], 14, bold=True, sans=True)
    f.place(panel, ownership, 0, 0)
    f.place(panel, targetlayer, 0, 0)
    text = f.component(
        "a_paper_annotations",
        136,
        172,
        "Paper layer annotations",
        "directed citations; selected ownership only",
    )
    text.rect(0, 0, 133, 65, "#f3dbe9", radius=5)
    text.text(9, 20, "Paper Graph", 16, bold=True)
    text.paragraph(9, 37, "Citation relations between papers", 119, 12.5, 15)
    text.paragraph(
        9,
        106,
        "Papers as nodes\nDirected citation edges\nParent-paper context",
        125,
        12,
        16,
    )
    f.place(panel, text, 9, 374)
    legend = f.component(
        "a_community_legend",
        125,
        181,
        "Community legend",
        "display aliases map to historical community IDs",
    )
    legend.rect(0, 0, 125, 181, "#ffffff", "#94c5e7", 5, 0.95)
    legend.text(6, 14, "Community (selected)", 11, bold=True)
    for i, color in enumerate(COLORS):
        legend.node(
            18, 27 + i * 17, fill=color, radius=5.8, stroke=mix(color, "#444444", 0.2)
        )
        legend.text(32, 31 + i * 17, f"K{i + 1}", 11)
    legend.node(18, 164, fill="#bfc4c9", radius=5.8)
    legend.text(32, 168, "Other / unassigned", 10.3)
    f.place(panel, legend, 646, 40)
    legend = f.component(
        "a_role_legend", 125, 137, "Claim role legend", "five roles; not community IDs"
    )
    legend.rect(0, 0, 125, 137, "#ffffff", "#94c5e7", 4, 0.95)
    legend.text(6, 16, "Claim role (node shape)", 11, bold=True)
    for i, role in enumerate(ROLES):
        legend.node(20, 32 + i * 21, role, "white", 6, INK, 0.9)
        legend.text(39, 36 + i * 21, role, 10.5)
    f.place(panel, legend, 646, 222)
    legend = f.component(
        "a_paper_legend",
        131,
        92,
        "Paper and citation legend",
        "parent papers and directed citation",
    )
    legend.rect(0, 0, 131, 92, "#ffffff", "#94c5e7", 4, 0.95)
    legend.node(14, 19, fill=COLORS[0], radius=8, stroke=INK, sw=1.2)
    legend.paragraph(33, 16, "Parent paper of target claim", 93, 10.5, 12)
    legend.node(14, 48, fill="#c6c9d0", radius=6)
    legend.text(33, 52, "Other paper", 11)
    legend.arrow(6, 75, 30, 75, gap=0)
    legend.text(34, 79, "Citation (directed)", 10.5)
    f.place(panel, legend, 641, 415)


def local_network(f: Figure, case: dict[str, Any], after: bool) -> Scene:
    label = case["label"]
    s = f.component(
        f"b_{label}_{'after' if after else 'before'}",
        115,
        105 if label in "AB" else 123,
        f"Case {label}: {'after' if after else 'before'} insertion",
        f"snapshot.cases[{label}]; full retained neighborhood; backbone",
    )
    s.text(3, 12, "After insertion" if after else "Before insertion", 11.7, bold=True)
    sy = 0.76 if label in "AB" else 1
    pos = {k: (v[0] + 13, v[1] * sy + 23) for k, v in case["layout"]["before"].items()}
    for u, v in case["edges"]:
        edge = s.line(*pos[u], *pos[v], "#8f99a5", 0.9)
        edge.set("id", f"local-{label}-{after}-" + safe_id(u + "--" + v))
        edge.set("data-source", u)
        edge.set("data-target", v)
    if after:
        tx, ty = case["layout"]["target"]
        tx, ty = tx + 13, ty * sy + 23
        for n in case["neighbors"]:
            color = (
                f.community_colors.get(n["community_id"], "#8a91a1")
                if label == "B"
                else COLORS[ord(label) - 65]
            )
            s.line(tx, ty, *pos[n["claim_id"]], color, 1, "4 3")
    for n in case["neighbors"]:
        color = (
            f.community_colors.get(n["community_id"], "#a8adb8")
            if label == "B"
            else "#b8bec7"
        )
        if label == "C":
            color = ROLE_COLORS[ROLES.index(n["claim_type"])]
        s.node(
            *pos[n["claim_id"]],
            n["claim_type"],
            color,
            4.3,
            "#81909d",
            0.7,
            "local-"
            + label
            + ("-after-" if after else "-before-")
            + safe_id(n["claim_id"]),
        )
    if after:
        s.node(
            tx, ty, case["claim"]["claim_type"], COLORS[ord(label) - 65], 7, INK, 1.6
        )
        s.rect(tx + 7, ty - 15, 23, 15, "white", radius=2, opacity=0.88)
        s.text(tx + 9, ty - 3, "C" + label, 12, bold=True)
    return s


def metric(
    f: Figure, case: dict[str, Any], key: str, title: str, value: str, width: float = 78
) -> Scene:
    s = f.component(
        f"b_{case['label']}_metric_{key}",
        width,
        64,
        title,
        f"snapshot.cases[{case['label']}].metrics.{key}",
    )
    color = COLORS[ord(case["label"]) - 65]
    s.rect(0, 0, width, 64, "#f8fcff", "#cce2ef", 2)
    s.paragraph(5, 12, title, width - 8, 9.8, 10.5, max_lines=3)
    s.rect(4, 38, width - 8, 22, mix(color, "#ffffff", 0.79), radius=2)
    s.text(width / 2, 53, value, 17, mix(color, "#000000", 0.5), True, anchor="middle")
    return s


def case_card(f: Figure, case: dict[str, Any]) -> Scene:
    label = case["label"]
    index = ord(label) - 65
    h = 198 if label in "AB" else 230
    card = Scene(503, h, f"case_{label}")
    bg = f.component(
        f"b_{label}_card_background", 503, h, f"Case {label} card frame", "layout only"
    )
    bg.rect(0, 0, 503, h, "white", mix(COLORS[index], "#ffffff", 0.6), 5)
    bg.rect(0, 0, 503, 25, f"url(#{GRADS[index]})", radius=5)
    bg.rect(10, 3, 68, 20, CASE_COLORS[index], radius=10)
    bg.text(44, 18, "Case " + label, 16, "white", True, anchor="middle")
    bg.text(88, 19, CASE_TITLES[label], 16.5, mix(COLORS[index], "#000000", 0.6), True)
    f.place(card, bg, 0, 0)
    claim = f.component(
        f"b_{label}_claim_excerpt",
        483,
        35,
        f"Case {label}: grounded claim excerpt",
        f"snapshot.cases[{label}].claim.claim_text; excerpts, not manuscript quotations",
    )
    text = case["claim"]["claim_text"]
    if label in "BC":
        text = text[:175].rsplit(" ", 1)[0] + "…"
    elif label == "D":
        text = text.split(";")[0] + "…"
    claim.paragraph(0, 13, text, 483, 12.3, 14, italic=True, max_lines=3)
    case["display_excerpt"] = text
    f.place(card, claim, 10, 29)
    top = 73
    for after, x in [(False, 9), (True, 132)]:
        f.place(card, local_network(f, case, after), x, top)
    sep = f.component(
        f"b_{label}_separators",
        245,
        121,
        f"Case {label} column separators",
        "layout only",
    )
    sep.line(121, 0, 121, 120, "#c6deeb", 0.7, "2 2")
    sep.line(244, 0, 244, 120, "#c6deeb", 0.7)
    f.place(card, sep, 0, top)
    m = case["metrics"]
    values = {
        "A": [
            (
                "nearest_prior_similarity",
                "Nearest-prior similarity",
                f"{m['nearest_prior_similarity']:.2f}",
            ),
            (
                "effective_community_count",
                "Effective community count",
                f"{m['effective_community_count']:.1f}",
            ),
            (
                "component_merge_count",
                "Merged local components",
                str(m["component_merge_count"]),
            ),
        ],
        "B": [
            (
                "nearest_prior_similarity",
                "Nearest-prior similarity",
                f"{m['nearest_prior_similarity']:.2f}",
            ),
            (
                "effective_community_count",
                "Effective community count",
                f"{m['effective_community_count']:.2f}",
            ),
            (
                "community_rao_stirling",
                "Community Rao–Stirling diversity",
                f"{m['community_rao_stirling']:.2f}",
            ),
        ],
        "C": [
            (
                "nearest_prior_similarity",
                "Nearest-prior similarity",
                f"{m['nearest_prior_similarity']:.2f}",
            ),
            (
                "cross_type_neighbor_share",
                "Cross-role neighbor share",
                f"{m['cross_type_neighbor_share']:.0%}",
            ),
        ],
        "D": [
            (
                "components_before",
                "Connected components",
                f"{m['components_before']} → {m['components_after']}",
            ),
            (
                "newly_connected_neighbor_pair_count",
                "Newly connected neighbor pairs",
                str(m["newly_connected_neighbor_pair_count"]),
            ),
            (
                "connected_pair_share",
                "Newly connected pair share",
                f"{m['connected_pair_share']:.1%}",
            ),
        ],
    }
    for j, (key, title, value) in enumerate(values[label]):
        f.place(
            card,
            metric(f, case, key, title, value, 112 if label == "C" else 77),
            255 + j * (119 if label == "C" else 81),
            top,
        )
    descriptions = {
        "A": "Ten neighbors already form one component. The insertion adds semantic links without merging components.",
        "B": f"VSIG10L connects {len(m['community_probabilities'])} historical communities. Structural diversity alone does not establish firstness.",
        "C": f"{len(m['community_probabilities'])} communities; {len(m['role_counts'])} neighbor roles. Semantic links describe association, not a causal chain.",
        "D": "Three local components become one; 33 historical node pairs become reachable through this claim.",
    }
    y = 146
    if label == "C":
        bar = f.component(
            "b_C_role_composition",
            238,
            41,
            "Neighbor role composition",
            "counts among all ten retained neighbors",
        )
        bar.text(0, 11, "Neighbor role composition", 10.5, bold=True)
        x = 0
        for role, color in zip(ROLES, ROLE_COLORS):
            count = m["role_counts"].get(role, 0)
            if count:
                width = 236 * count / 10
                bar.rect(x, 16, width, 14, color)
                bar.text(x + width / 2, 38, f"{count * 10}%", 10, anchor="middle")
                x += width
        f.place(card, bar, 255, 142)
        y = 183
    elif label == "D":
        y = 153
    note = f.component(
        f"b_{label}_interpretation",
        238,
        43,
        CASE_TITLES[label] + ": structural observation",
        "computed metrics; claim-scoped interpretation",
    )
    note.rect(0, 0, 238, 43, "#f0f7fc", radius=4)
    note.paragraph(8, 12, descriptions[label], 223, 11.3, 12.5, max_lines=3)
    f.place(card, note, 255, y)
    # Provenance remains in the source companion, not in the case-card artwork.
    f.panels[f"case_{label}"] = card
    return card


def cases(f: Figure) -> None:
    panel = f.panel("b")
    for i, c in enumerate(f.data["cases"]):
        card = case_card(f, c)
        panel.use(card, 10 if i % 2 == 0 else 524, 45 if i < 2 else 249)


def populations(f: Figure) -> None:
    panel = f.panel("c")
    stats = f.data["statistics"]
    card = f.component(
        "c_historical_population",
        240,
        163,
        "Historical claims (target set)",
        "claim_nodes; nature_targets; abstract-derived history",
    )
    card.rect(0, 0, 240, 163, "url(#bluewash)", "#8fbde5", 4)
    card.text(9, 19, "Historical claims (target set)", 14, bold=True)
    for y, value, desc in [
        (48, stats["target_papers"], "historical target papers"),
        (92, stats["papers_with_claims"], "papers with historical claims"),
        (137, stats["claims"], "historical claims · abstract-derived"),
    ]:
        card.text(37, y, f"{value:,}", 20, bold=True)
        card.text(37, y + 15, desc, 11.5)
    card.arrow(23, 31, 23, 69, "#a5adb7", 2, 0)
    card.arrow(23, 76, 23, 112, "#a5adb7", 2, 0)
    f.place(panel, card, 7, 57)
    card = f.component(
        "c_paper_population",
        240,
        131,
        "Citation-extended paper graph",
        "verified aggregate; clean distinct non-self citation edges",
    )
    card.rect(0, 0, 240, 131, "url(#bluewash)", "#8fbde5", 4)
    card.text(9, 19, "Citation-extended paper graph", 14, bold=True)
    card.text(9, 34, "(context set)", 13, bold=True)
    for y, value, desc in [
        (62, stats["paper_nodes"], "paper nodes"),
        (105, stats["paper_clean_edges"], "clean citation edges"),
    ]:
        card.text(37, y, f"{value:,}", 20, bold=True)
        card.text(37, y + 15, desc, 12)
    card.arrow(23, 46, 23, 82, "#a5adb7", 2, 0)
    f.place(panel, card, 7, 226)
    scope = f.component(
        "c_scope_note",
        225,
        31,
        "Population scopes",
        "two graph populations must not be summed",
    )
    scope.rect(0, 0, 225, 31, "#f9fafc", "#bacbd9", 3)
    scope.text(112, 13, "Two graph scopes; counts", 11.5, anchor="middle")
    scope.text(112, 26, "should not be summed.", 11.5, anchor="middle")
    f.place(panel, scope, 15, 363)
    roles = f.component(
        "c_role_bars",
        240,
        127,
        "Claim role composition",
        "all 70,034 historical claims; counts, linear scale",
    )
    roles.text(0, 16, "Claim role composition", 14, bold=True)
    roles.text(240, 16, "n = 70,034", 11, italic=True, anchor="end")
    maximum = max(stats["roles"].values())
    for i, (role, color) in enumerate(zip(ROLES, ROLE_COLORS)):
        y = 26 + i * 20
        value = stats["roles"][role]
        width = value / maximum * 120
        roles.text(0, y + 11, role, 10.5)
        roles.rect(66, y, width, 14, color)
        roles.text(69 + width, y + 12, f"{value:,}", 11.5)
    f.place(panel, roles, 7, 399)
    counts = f.component(
        "c_claims_per_paper",
        240,
        74,
        "Claims per parent paper",
        "all 24,919 historical targets; zero-claim papers retained",
    )
    counts.text(0, 13, "Claims per parent paper", 13.5, bold=True)
    for i, (n, value) in enumerate(
        sorted(((int(k), v) for k, v in stats["claims_per_paper"].items()))
    ):
        y = 20 + i * 12.7
        counts.text(11, y + 9, f"{n} claims", 10.7)
        counts.text(99, y + 9, f"{value:,}", 10.7, anchor="end")
        counts.rect(107, y + 1, 126, 7, "#edf4f8", radius=2)
        counts.rect(107, y + 1, value / 20512 * 126, 7, "#5298e6", radius=2)
    f.place(panel, counts, 7, 532)


def communities(f: Figure) -> None:
    panel = f.panel("d")
    stats = f.data["statistics"]
    dist = f.component(
        "d_size_distribution",
        221,
        266,
        "Community size distribution",
        "all assigned communities; empirical complementary cumulative distribution; log axes",
    )
    dist.text(4, 17, "Community size distribution", 12.5, bold=True)
    dist.text(4, 31, "Complementary CDF (CCDF)", 10, color="#506985")
    ox, oy, w, h = 38, 220, 88, 174
    dist.line(ox, oy, ox + w, oy, "#596576", 0.9)
    dist.line(ox, oy, ox, oy - h, "#596576", 0.9)
    for power in range(4):
        x = ox + power / 3.5 * w
        dist.line(x, oy, x, oy + 4, "#596576")
        dist.text(x, oy + 16, ["1", "10", "10²", "10³"][power], 10, anchor="middle")
    for power, label in [(0, "100"), (-1, "10"), (-2, "1"), (-3, "0.1")]:
        y = oy - h * (power + 3.5) / 3.5
        dist.line(ox - 4, y, ox, y, "#596576")
        dist.line(ox, y, ox + w, y, "#e0eaf3", 0.5)
        dist.text(ox - 6, y + 3, label, 10, anchor="end")
    coords = [
        (
            ox + math.log10(row["size"]) / 3.5 * w,
            oy - h * (math.log10(row["share_at_least_size"]) + 3.5) / 3.5,
        )
        for row in stats["community_ccdf"]
    ]
    path = f"M{coords[0][0]},{coords[0][1]}"
    for x, y in coords[1:]:
        path += f" V{y} H{x}"
    dist.path(path + f" V{oy} H{coords[0][0]} Z", fill="#e7f1fd")
    dist.path(path, stroke="#216bce", width=1.4)
    dist.text(84, 251, "Community size (claims)", 10.5, anchor="middle")
    group = el("g", transform="translate(11 150) rotate(-90)")
    text = el(
        "text",
        x=0,
        y=0,
        font_family="Times New Roman",
        font_size=10.5,
        fill=INK,
        text_anchor="middle",
    )
    text.text = "Communities ≥ size (%)"
    group.append(text)
    dist.add(group)
    metrics = [
        ("Communities", "2,422"),
        ("Median size", "2"),
        ("Maximum size", "2,544"),
        ("Unassigned claims", "14,872"),
    ]
    for i, (title, value) in enumerate(metrics):
        dist.text(137, 45 + i * 41, title, 10)
        dist.text(137, 62 + i * 41, value, 15, bold=True)
    dist.text(137, 194, "21.2% of claims", 9.8)
    f.place(panel, dist, 8, 37)
    bars = f.component(
        "d_community_roles",
        192,
        224,
        "Role composition of selected communities",
        "full community populations, not atlas sample",
    )
    bars.paragraph(
        0, 14, "Role composition of selected\ncommunities", 192, 11.2, 13, bold=True
    )
    for i, c in enumerate(stats["communities"][:6]):
        y = 42 + i * 23
        bars.text(0, y + 10, f"{c['display']} (n = {c['size']:,})", 10)
        x = 73
        for role, color in zip(ROLES, ROLE_COLORS):
            width = c["role_counts"].get(role, 0) / c["size"] * 112
            bars.rect(x, y, width, 15, color)
            x += width
    for x, val in [(73, "0"), (129, "0.5"), (185, "1.0")]:
        bars.text(x, 191, val, 10, anchor="middle")
    bars.text(129, 207, "Proportion", 10.5, anchor="middle")
    f.place(panel, bars, 235, 37)
    heat = f.component(
        "d_community_mixing",
        148,
        224,
        "Inter-community connectivity",
        "backbone undirected edge count; mirrored off-diagonal; no double counting",
    )
    heat.paragraph(
        0, 14, "Inter-community connectivity\n(highlighted communities)", 145, 10.5, 13
    )
    matrix = np.array(stats["community_mixing"])[:6, :6]
    maxval = matrix.max()
    for i in range(6):
        for j in range(6):
            value = matrix[i, j]
            frac = math.log10(value + 1) / math.log10(maxval + 1)
            heat.rect(
                22 + j * 17,
                45 + i * 19,
                17,
                19,
                mix("#f1f8fc", "#1260aa", frac),
                "white",
                sw=0.5,
            )
        heat.text(17, 58 + i * 19, f"K{i + 1}", 9, anchor="end")
        heat.text(30.5 + i * 17, 172, f"K{i + 1}", 9, anchor="middle")
    for i in range(70):
        heat.rect(127, 48 + i, 5, 1.1, mix("#1260aa", "#f1f8fc", i / 69))
    heat.text(131, 40, f"{maxval:,}", 8.8, anchor="middle")
    heat.text(131, 130, "0", 9, anchor="middle")
    heat.text(69, 190, "Edges · log(1 + count)", 10, anchor="middle")
    heat.text(69, 204, "Off-diagonal mirrored", 9.8, anchor="middle")
    f.place(panel, heat, 432, 37)
    legend = f.component(
        "d_role_legend",
        350,
        22,
        "Role colors for statistical charts",
        "five contribution roles",
    )
    x = 0
    for role, color in zip(ROLES, ROLE_COLORS):
        legend.rect(x, 5, 11, 11, color)
        legend.text(x + 15, 14, role, 9.8)
        x += {
            "FINDING": 65,
            "METHOD": 66,
            "MECHANISM": 86,
            "RESOURCE": 79,
            "THEORY": 58,
        }[role]
    f.place(panel, legend, 235, 245)
    note = f.component(
        "d_scope_note",
        346,
        26,
        "Community scope",
        "unassigned claims excluded from community count",
    )
    note.paragraph(
        0,
        11,
        "Colors and K labels match panel a. Community IDs and full counts are provided in the source data.",
        344,
        10,
        12,
    )
    f.place(panel, note, 235, 275)
    panel.line(227, 43, 227, 300, "#d8e7ef", 0.65)
    panel.line(427, 43, 427, 241, "#d8e7ef", 0.65)


def path_motif(f: Figure, kind: str, title: str) -> Scene:
    s = f.component(
        "e_path_" + kind,
        73,
        130,
        title,
        "relation syntax only; real witnesses in data/paper_path_witnesses.json",
    )
    s.paragraph(4, 12, title, 66, 10.5, 12)
    s.node(17, 48, fill="#ffc97d", radius=6, stroke="#df8734", sw=0.9)
    s.line(17, 55, 17, 72, "#708293", 0.7, "2 2")
    for x, y in [(17, 79), (59, 79)]:
        s.node(x, y, fill="#85b5ec", radius=6, stroke="#1f5b9b", sw=0.9)
    if kind == "direct":
        s.arrow(17, 79, 59, 79, INK, 0.9, 7)
        s.text(38, 106, "P* → Pᵢ", 11, anchor="middle")
    elif kind == "two_hop":
        s.node(38, 79, fill="#85b5ec", radius=5, stroke="#1f5b9b", sw=0.9)
        s.arrow(17, 79, 38, 79, INK, 0.8, 6)
        s.arrow(38, 79, 59, 79, INK, 0.8, 6)
        s.text(38, 106, "P* → R → Pᵢ", 10, anchor="middle")
    else:
        s.node(38, 47, fill="#85b5ec", radius=5, stroke="#1f5b9b", sw=0.9)
        s.arrow(17, 79, 38, 47, INK, 0.8, 6)
        s.arrow(59, 79, 38, 47, INK, 0.8, 6)
        s.text(38, 106, "P* → R ← Pᵢ", 10, anchor="middle")
    return s


def relations(f: Figure) -> None:
    panel = f.panel("e")
    stats = f.data["statistics"]
    heat = f.component(
        "e_role_heatmap",
        190,
        182,
        "Claim-role interaction frequencies",
        "historical semantic edges; earlier rows, later columns; column normalized",
    )
    heat.text(0, 15, "Claim-role interaction frequencies", 11.7, bold=True)
    matrix = np.array(stats["role_matrix_column_normalized"])
    for i, role in enumerate(ROLES):
        heat.text(58, 51 + i * 20, role, 9.2, anchor="end")
        for j in range(5):
            heat.rect(
                62 + j * 16,
                37 + i * 20,
                16,
                20,
                mix("#f0f8fc", "#145ca4", float(matrix[i, j]) ** 0.5),
                "white",
                sw=0.6,
            )
    for j, label in enumerate(["F", "M", "Me", "R", "T"]):
        heat.text(70 + j * 16, 151, label, 9, anchor="middle")
    for i in range(95):
        heat.rect(150, 38 + i, 6, 1.1, mix("#145ca4", "#f0f8fc", i / 94))
    heat.text(160, 46, "100%", 9)
    heat.text(160, 134, "0", 9)
    heat.text(88, 166, "Earlier role × later role", 10, anchor="middle")
    heat.text(88, 179, "Columns normalized", 10, anchor="middle")
    f.place(panel, heat, 9, 37)
    title = f.component(
        "e_path_title",
        228,
        24,
        "Citation-path motifs",
        "paper-level relation syntax, not causal relations",
    )
    title.text(0, 15, "Citation-path motifs (paper level)", 12, bold=True)
    f.place(panel, title, 206, 37)
    for j, (kind, name) in enumerate(
        [
            ("direct", "Direct citation"),
            ("two_hop", "Two-hop citation"),
            ("shared", "Shared reference"),
        ]
    ):
        f.place(panel, path_motif(f, kind, name), 207 + j * 73, 63)
    legend = f.component(
        "e_path_legend",
        226,
        35,
        "Path motif legend",
        "claim nodes, paper nodes, citations; schematic syntax",
    )
    legend.rect(0, 0, 226, 34, "#f9fcff", "#c1dceb", 3)
    legend.node(12, 11, fill="#ffc97d", radius=5, stroke="#df8734")
    legend.text(21, 15, "Claim", 10)
    legend.node(64, 11, fill="#85b5ec", radius=5, stroke="#1f5b9b")
    legend.text(73, 15, "Paper", 10)
    legend.arrow(108, 11, 131, 11, gap=0)
    legend.text(136, 15, "Citation", 10)
    legend.text(
        111, 29, "Schematic syntax; witnesses in source data", 9.5, anchor="middle"
    )
    f.place(panel, legend, 206, 187)
    counts = f.component(
        "e_edge_counts",
        424,
        79,
        "Edge counts across graph layers",
        "saved historical edge families; overlapping counts",
    )
    counts.rect(0, 0, 424, 79, "#ffffff", "#c1dceb", 3)
    counts.text(8, 17, "Edge counts (across graph layers)", 12, bold=True)
    for i, (text, color) in enumerate(
        [
            (f"Semantic claim edges: {stats['semantic_edges']:,}", "#47ac80"),
            (
                f"Saved paper-path annotations: {stats['saved_paper_path_annotations']:,}",
                "#f2a03b",
            ),
            (f"Local/community backbone edges: {stats['backbone_edges']:,}", "#e55870"),
        ]
    ):
        counts.node(14, 31 + i * 18, fill=color, radius=4, stroke=color, sw=0)
        counts.text(26, 35 + i * 18, text, 11)
    counts.path("M325,23 H329 V66 H325 M329,45 H338", stroke="#667589", width=0.8)
    counts.paragraph(345, 32, "Edge families overlap; do not sum.", 72, 10.5, 12)
    f.place(panel, counts, 9, 222)
    panel.line(199, 43, 199, 215, "#d8e7ef", 0.65)


def render(out: Path, only: str | None = None) -> Figure:
    # Retire the four annotations from both assembly layers and standalone exports.
    for label in "ABCD":
        for extension in ["svg", "pdf", "png"]:
            (out / "components" / f"b_{label}_provenance.{extension}").unlink(
                missing_ok=True
            )
        (out / "components/layers" / f"b_{label}_provenance.svg").unlink(
            missing_ok=True
        )
    f = Figure(out)
    atlas(f)
    cases(f)
    populations(f)
    communities(f)
    relations(f)
    title = f.component("figure_title", 1035, 39, "Figure title", "layout only")
    title.text(
        0,
        29,
        "Fig. 1 | Two-layer knowledge graph and local claim-insertion profiles",
        30.5,
        "#080808",
        True,
    )
    footer = f.component(
        "figure_caption_note",
        1036,
        43,
        "Figure scope note",
        "descriptive insertion, bounded historical graph",
    )
    footer.paragraph(
        0,
        13,
        "Historical atlas: 2023–2025 abstract-derived claims. Four full-text target claims are shown as independent insertions; network layouts are display coordinates. Semantic association and local connectivity do not establish scientific correctness or novelty.",
        1036,
        11.5,
        14,
    )
    figure = Scene(1055, 1491, "fig1")
    figure.rect(0, 0, 1055, 1491, "white")
    f.place(figure, title, 12, 0)
    for key in ["a", "b", "c", "d", "e"]:
        x, y, _, _ = PANEL_BOXES[key]
        figure.use(f.panels[key], x, y)
    f.place(figure, footer, 12, 1434)
    for ident, scene in f.components.items():
        if only and not (ident == only or ident.startswith(only + "_")):
            continue
        scene.save(out / "components" / "layers" / f"{ident}.svg")
        title_text = next(e["title"] for e in f.entries if e["id"] == ident)
        has_title = bool(scene.text_boxes) and scene.text_boxes[0]["text"] == title_text
        footer_text = "GEAR · Fig.1 · " + ident
        width = max(
            scene.width + 24,
            measure(footer_text, 10) + 24,
            0 if has_title else measure(title_text, 13, bold=True) + 24,
        )
        top = 12 if has_title else 38
        standalone = Scene(width, scene.height + top + 33, "standalone-" + ident)
        standalone.rect(0, 0, standalone.width, standalone.height, "white")
        if not has_title:
            standalone.paragraph(
                12, 19, title_text, standalone.width - 24, 13, 15, bold=True
            )
        standalone.use(scene, (standalone.width - scene.width) / 2, top)
        standalone.text(12, standalone.height - 10, footer_text, 10, "#61748a")
        standalone.save(out / "components" / f"{ident}.svg")
    for key, scene in f.panels.items():
        if only and key != only and not key.endswith(only):
            continue
        scene.save(out / "panels" / f"{key}.svg")
        scene.save(out / "layouts" / "templates" / f"{key}.svg")
    figure.save(out / "final/Fig1.svg", physical=True)
    figure.save(out / "layouts" / "templates" / "Fig1.svg", physical=True)
    write(out / "layouts/components.json", f.entries)
    write(
        out / "layouts/style.json",
        {
            "reference_size": [1055, 1491],
            "panel_boxes": PANEL_BOXES,
            "community_colors": COLORS,
            "role_colors": ROLE_COLORS,
            "fonts": ["Times New Roman", "Arial"],
            "layout_adjustments": [
                "Panel b has a stepped upper boundary so panel c does not overlap its heading.",
                "Case quotes use two or three lines; graph rows shift down to y=73 within each case.",
                "Bottom-right headings reduced to fit actual column width.",
                "Claim coordinates reserve two left annotation boxes; paper positions leave a clear left annotation gutter.",
            ],
        },
    )
    write(out / "qa/text_bounds.json", figure.text_boxes)
    component_overflow = []
    for ident, scene in f.components.items():
        component_overflow.extend(
            {"component": ident, **box}
            for box in scene.text_boxes
            if box["x"] < -0.5
            or box["y"] < -0.5
            or box["x"] + box["width"] > scene.width + 0.5
            or box["y"] + box["height"] > scene.height + 0.5
        )
    write(out / "qa/component_bounds.json", component_overflow)
    write(out / "data/display_cases.json", f.data["cases"])
    return f
