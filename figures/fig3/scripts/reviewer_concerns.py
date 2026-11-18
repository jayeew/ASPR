"""Read-only source validation and standalone four-layer reviewer evidence figure."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle, Rectangle, FancyBboxPatch, PathPatch
from matplotlib.path import Path as MPath

RADII = dict(core=68, dimension=228, link=307, theme=355, concern_inner=362, concern=411, tick=417, maximum=450)
COLORS = ['#DC494C','#F39728','#4DA84C','#2689CD','#8953AA']
DIMS = [('Historical\noriginality',['T01'],801),('Advances in\nunderstanding\nand capability',['T02','T03','T04','T64'],774),('Knowledge\norganisation\nand integration',['T06','T75'],374),('Evidential\nbasis',['T46','T12','T16','T11','T32'],946),('Scientific\nvalue',['T05','T62','T74'],592)]
EXPECTED = dict(zip(['T01','T02','T03','T04','T64','T06','T75','T46','T12','T16','T11','T32','T05','T62','T74'],[801,510,396,81,222,372,10,728,590,578,492,407,559,91,10]))

def read_csv(path: Path) -> list[dict]:
    with path.open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))

def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def main() -> None:
    ap=argparse.ArgumentParser();ap.add_argument('--project-root',type=Path,default=next(p for p in Path(__file__).resolve().parents if (p / 'AGENTS.md').is_file()));ap.add_argument('--output-dir',type=Path);ap.add_argument('--dpi',type=int,default=600);ap.add_argument('--preview',action='store_true');args=ap.parse_args()
    src=args.project_root/'outputs/reviewer_open_coding_pilot_20261116/open_discovery'
    out=args.output_dir or src/'fig_reviewer_concerns_four_layer'
    if out.exists(): raise FileExistsError(f'Refusing to overwrite existing output: {out}')
    files=[src/n for n in ['paper_theme_matrix.csv','theme_statistics.csv','theme_to_theoretical_category.csv','innovation_dimensions_zh.md']]
    hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    matrix,stats,categories=[read_csv(f) for f in files[:3]]
    for f,rs in zip(files,[matrix,stats,categories]):print(f.name,len(rs),list(rs[0]))
    ids=[r['theme_id'] for r in stats];assert len(set(ids))==len(ids)==100
    assert {r['theme_id'] for r in categories}==set(ids)
    papers={}
    for row in matrix:
        dest=papers.setdefault(row['paper_id'],{t:False for t in ids})
        for t in ids:
            assert row[t] in ('0','1');dest[t]|=bool(int(row[t]))
    assert len(papers)==994
    roster=read_csv(src.parent/'sample.csv')
    status={r['paper_id']:r for r in read_csv(src.parent/'paper_coding_status.csv')}
    assert len(roster)==len({r['paper_id'] for r in roster})==1000
    assert set(papers).issubset({r['paper_id'] for r in roster})
    unavailable=[dict(pilot_id=r['pilot_id'],paper_id=r['paper_id'],title=r['title'],
        source_status=status[r['paper_id']]['source_status'],
        reason=status[r['paper_id']]['notes'],
        treatment='Included in roster denominator; unavailable for theme coding, not an observed absence')
        for r in roster if r['paper_id'] not in papers]
    assert len(unavailable)==6
    counts={t:sum(r[t] for r in papers.values()) for t in ids}
    assert all(counts[r['theme_id']]==int(r['present_papers']) for r in stats)
    assert all(counts[t]==n for t,n in EXPECTED.items())
    unions=[sum(any(r[t] for t in ts) for r in papers.values()) for _,ts,_ in DIMS]
    assert unions==[d[2] for d in DIMS]
    assigned={t:i for i,(_,ts,_) in enumerate(DIMS) for t in ts};assert len(assigned)==15
    slots={}
    for i,(_,ts,_) in enumerate(DIMS):
        for j,t in enumerate(ts):slots[i*20+int((j+.5)*20/len(ts))]=t
    remaining=iter(sorted(set(ids)-set(assigned),key=lambda t:int(t[1:])))
    slots={i:slots[i] if i in slots else next(remaining) for i in range(100)}
    assert set(slots.values())==set(ids) and len(set(slots.values()))==100
    from matplotlib import font_manager
    for font in ['times.ttf','timesbd.ttf','timesi.ttf']:
        font_manager.fontManager.addfont('/mnt/c/Windows/Fonts/'+font)
    plt.rcParams.update({'font.family':'Times New Roman','svg.fonttype':'none','pdf.fonttype':42})
    fig=plt.figure(figsize=(12.54,12.54));ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,1254),ylim=(1254,0));ax.set_aspect('equal');ax.axis('off')
    cx,cy=455,630
    def point(r: float,a: float) -> tuple[float,float]:
        a=math.radians(a);return cx+r*math.sin(a),cy-r*math.cos(a)
    def ring(a: float,b: float,ri: float,ro: float,color: str,alpha: float=1) -> None:
        ax.add_patch(Wedge((cx,cy),ro,a-90,b-90,width=ro-ri,facecolor=color,edgecolor='white',linewidth=1,alpha=alpha))
    def label(x: float,y: float,s: str,size: float=12,**kw) -> None:
        ax.text(x,y,s,fontsize=size,va='center',**kw)
    label(627,48,'Reviewer concerns informing the five dimensions',26,ha='center',weight='bold',color='#22252A')
    ax.plot([25,1229],[83,83],color='#83909A',lw=.6)
    for i,(name,ts,n) in enumerate(DIMS):
        a=i*72;b=a+72;c=COLORS[i]
        ring(a+.6,b-.6,68,228,c)
        ring(a,b,229,307,c,.055)
        x,y=point(157,a+36)
        label(x,y-19,name,13 if i in (1,2) else 15,ha='center',color='white',weight='bold',linespacing=1.05)
        label(x,y+37,f'{n}\n({n/1000:.1%})',16,ha='center',color='white',weight='bold')
        for j,t in enumerate(ts):
            mid=a+(j+.5)*72/len(ts);inner=a+36+(j-(len(ts)-1)/2)*8
            # Constant-width Bezier stroke: equal width at both ends and throughout.
            verts=[point(308,mid),point(276,mid),point(254,inner),point(228,inner)]
            codes=[MPath.MOVETO]+[MPath.CURVE4]*3
            ax.add_patch(PathPatch(MPath(verts,codes),facecolor='none',
                alpha=.23,edgecolor=c,lw=23.04,capstyle='butt'))
            ring(a+j*72/len(ts)+.5,a+(j+1)*72/len(ts)-.5,308,355,c)
            x,y=point(331,mid);label(x,y,t,17,ha='center',color='white',weight='bold')
    manifest=[]
    cat={r['theme_id']:r for r in categories}
    for k,t in slots.items():
        a=k*3.6;b=a+3.6;idx=assigned.get(t);c=COLORS[idx] if idx is not None else '#CFD2D5'
        ring(a+.15,b-.15,362,411,c)
        h=33*counts[t]/1000
        ax.add_patch(Wedge((cx,cy),417+h,a+.55-90,b-.55-90,width=h,facecolor=c,edgecolor='none'))
        manifest.append(dict(slot_index=k,theme_code=t,name_zh=cat[t]['name_zh'],selected=idx is not None,assigned_dimension_or_null=DIMS[idx][0].replace('\n',' ') if idx is not None else '',original_category=cat[t]['category_id'],angle_start=a,angle_end=b,paper_count=counts[t],percentage=counts[t]/1000*100,tick_height=h))
    ax.add_patch(Circle((cx,cy),451,fill=False,ec='#AEB4B9',lw=.7));ax.add_patch(Circle((cx,cy),68,fc='white',ec='#E5E5E5'))
    label(cx,cy-12,'1000',29,ha='center',weight='bold');label(cx,cy+23,'papers',19,ha='center',weight='bold')
    def box(y: float,h: float,title: str) -> None:
        ax.add_patch(FancyBboxPatch((923,y),308,h,boxstyle='round,pad=0,rounding_size=7',fc='white',ec='#83909A',lw=.9))
        ax.add_patch(Rectangle((923,y),308,35,fc='#405463'));label(1077,y+18,title,14,ha='center',color='white',weight='bold')
    box(119,365,'Four layers (inside → out)')
    layer=['Dimension layer: five analysis\ndimensions (paper-level unions)','Theme layer: 15 selected\nreviewer concerns','Concern layer: 100 coded concerns\n(15 highlighted; 85 gray)','Frequency layer: theme occurrence\nas a share of the 1000-paper roster']
    for j,s in enumerate(layer):
        y=190+j*79;ax.add_patch(Circle((951,y),15,fc='#536672'));label(951,y,str(j+1),16,ha='center',color='white',weight='bold');label(979,y,s,12)
    box(505,345,'Five dimensions (inner ring)')
    for i,(name,ts,n) in enumerate(DIMS):
        y=570+i*61;ax.add_patch(FancyBboxPatch((939,y-12),25,23,boxstyle='round,pad=0,rounding_size=3',fc=COLORS[i],ec='none'))
        name=['Historical originality','Advances in understanding\nand capability','Knowledge organisation\nand integration','Evidential basis','Scientific value'][i]
        label(976,y-6,name,11.8,weight='bold',linespacing=.95);label(976,y+22,f'{n} ({n/1000:.1%})',12)
    box(880,250,'Mark legend')
    for j,(c,s) in enumerate([('#8953AA','Selected concern (n = 15)'),('#CFD2D5','Other concern (n = 85)')]):
        y=942+j*44;ax.add_patch(Rectangle((940,y-9),27,18,fc=c));label(980,y,s,12)
    for j in range(5):ax.add_patch(Rectangle((940+j*6,1042-(j+1)*5),4,(j+1)*5,fc='#B4B8BC'))
    label(980,1031,'Frequency tick: proportion of\n1000 roster papers with the concern',11)
    ax.add_patch(PathPatch(MPath([(941,1092),(947,1078),(958,1078),(964,1092)],
        [MPath.MOVETO]+[MPath.CURVE4]*3),facecolor='none',edgecolor='#8953AA',
        lw=16,alpha=.23,capstyle='butt'))
    label(980,1088,'Theme–dimension conceptual link\n(not weighted)',11.5)
    label(627,1174,'Selected concerns are highlighted within the broader set of 100 coded reviewer concerns.',14,ha='center',style='italic')
    label(627,1205,'1000 roster papers: 994 analysable; 6 unavailable reviews. Equal sectors and links are not weights.',10.5,ha='center',color='#59636B')
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    outside=[t.get_text() for t in ax.texts if not fig.bbox.contains(*t.get_window_extent(renderer).get_points()[0]) or not fig.bbox.contains(*t.get_window_extent(renderer).get_points()[1])]
    assert not outside,outside
    out.mkdir(parents=True)
    stem=out/'reviewer_five_dimensions_four_layer'
    for ext in ['svg','pdf','png']:fig.savefig(stem.with_suffix('.'+ext),dpi=args.dpi,facecolor='white')
    if args.preview:fig.savefig(out/'preview.png',dpi=120)
    save_csv(out/'unavailable_reviews.csv',unavailable)
    save_csv(out/'outer_ring_slot_manifest.csv',manifest)
    save_csv(out/'selected_theme_counts.csv',[{'theme':t,'name_zh':cat[t]['name_zh'],'dimension':DIMS[assigned[t]][0].replace('\n',' '),'papers':counts[t],'percentage':round(counts[t]/1000*100,1)} for t in assigned])
    save_csv(out/'dimension_union_counts.csv',[{'dimension':name.replace('\n',' '),'themes':','.join(ts),'papers':n,'percentage':round(n/1000*100,1)} for name,ts,n in DIMS])
    assert all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==v for f,v in hashes.items())
    (out/'plot_validation.json').write_text(json.dumps(dict(unique_papers=994,roster_papers=1000,percentage_denominator=1000,unavailable_reviews=6,unique_themes=100,selected_themes=15,dimension_unions=unions,all_100_counts_match=True,outer_slots_unique=True,coloured_slots=15,grey_slots=85,links=15,text_outside_canvas=outside,input_hashes=hashes,inputs_unchanged=True,png_dpi=args.dpi),indent=2))
    (out/'caption.md').write_text('Coverage of reviewer concerns informing five dimensions of innovation analysis across a roster of 1000 papers (994 analysable; six with unavailable substantive independent reviews). Counts are paper-level unions of the specified themes. The 15 themes were selected by conceptual relevance, not frequency rank. Occurrences include praise, criticism and requests, not endorsements or evaluation weights. Equal sectors and equal-width links encode conceptual grouping only. The remaining 85 themes provide coding context; their placement does not assign them to a dimension. Outer tick lengths encode single-theme paper prevalence on a linear scale. Graph-based representation is a methodological extension of knowledge organisation and integration; evidential basis incorporates selected research-quality requirements. The sample spans six Nature Portfolio journals, excluding Nature itself. T11 concerns measurement and analysis verification, not reviewer reliability. All displayed percentages use the 1000-paper roster denominator. The six unavailable reviews are recorded separately and are not observed theme absences.\n')
    print(out)

if __name__=='__main__':main()
