"""Draw Figure 3: python -m figure_pipeline.fig3_reference render --panel a."""

from __future__ import annotations

import argparse
import copy
import xml.etree.ElementTree as ET

import pandas as pd

from figure_pipeline.fig1_reference.export import convert, fonts
from figure_pipeline.fig1_reference.svg import Scene, measure
from . import charts, drawing, review_charts
from .settings import OUTPUT, PANELS, read


def render(only: str | None = None) -> None:
    fonts(OUTPUT)
    if only in (None,"a"):
        drawing.render_a()
    if only == "a":
        return
    summary=read(OUTPUT/"data/aggregates.json")
    frame=pd.read_csv(OUTPUT/"data/tables/paper_metrics.csv")
    if only in (None,"b"):
        charts.render_b(summary,frame)
    if only in (None,"c","f"):
        examples=read(OUTPUT/"data/examples.json")
        if only in (None,"c"): charts.render_c(summary,examples)
        if only in (None,"f"): charts.render_f(summary,frame,examples)
    if only in (None,"d"):
        charts.render_d(summary,frame)
    if only in (None,"e"):
        review_charts.render_e(read(OUTPUT/"data/review_aggregates.json"),pd.read_csv(OUTPUT/"data/tables/reviewer_pairs.csv"))
    if only in (None,"g"):
        charts.render_g(summary)
    if only is None:
        charts.supplementary_projections(frame)


def assemble(high_res: bool = True) -> None:
    import cairosvg
    fonts(OUTPUT)
    scene=Scene(2100,3150,"Fig3")
    scene.rect(0,0,2100,3150,"#ffffff")
    title="Fig. 3 | Effectiveness, evidence reliability and independent appraisal of innovation analysis"
    size=min(51,51*2050/measure(title,51,True))
    drawing.text(scene,25,62,title,size,"#101018",True)
    for ident,(x,y,_,_) in PANELS.items():
        root=ET.parse(OUTPUT/"panels"/f"{ident}.svg").getroot()
        layer=next(n for n in root if n.get("id")==f"panel_{ident}")
        group=ET.Element("{http://www.w3.org/2000/svg}g",{"transform":f"translate({x} {y})"})
        group.append(copy.deepcopy(layer));scene.add(group)
    interim = (not (OUTPUT/"data/aggregates.json").exists()
               or read(OUTPUT/"data/aggregates.json").get("interim",False)
               or read(OUTPUT/"data/review_aggregates.json").get("interim",False))
    footer = ("INTERIM — evaluations incomplete; panels have different completed sample sizes."
              if interim else "Fixed 200-paper cohort · Model-assisted source checking and independent LLM preference; not expert ground truth.")
    drawing.text(scene,30,3100,footer,21,"#9d3a20" if interim else drawing.INK)
    drawing.text(scene,30,3130,"Intervals preserve paper clustering. Reviewer-history analyses use their own observable denominators.",20)
    final=OUTPUT/"final/Fig3.svg";scene.save(final,physical=True);convert(final,1)
    cairosvg.svg2svg(url=str(final),write_to=str(OUTPUT/"final/Fig3_outlined.svg"))
    if high_res:
        for width in (6300,10500):
            cairosvg.svg2png(url=str(final),write_to=str(OUTPUT/"final"/f"Fig3_{width}px.png"),output_width=width,output_height=width*3//2)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("render","assemble","all"))
    parser.add_argument("--panel",choices=tuple(PANELS))
    parser.add_argument("--preview",action="store_true")
    args=parser.parse_args()
    if args.stage in ("render","all"):
        render(args.panel)
    if args.stage in ("assemble","all"):
        assemble(not args.preview)


if __name__=="__main__":
    main()
