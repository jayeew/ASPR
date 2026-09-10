"""Combine aggregate-only graph/Nature statistics into portable readable files."""
from __future__ import annotations
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/GRAPH_STATISTICS'


def flatten(value: Any, path: str='') -> Iterator[tuple[str, Any]]:
    if isinstance(value,dict):
        for key,item in value.items():
            yield from flatten(item,f'{path}.{key}' if path else str(key))
    elif isinstance(value,list):
        for index,item in enumerate(value):yield from flatten(item,f'{path}[{index}]')
        if not value:yield path,'[]'
    else:yield path,value


def main() -> None:
    names=['paper_graph','claim_graph','nature_corpus','runtime_graph']
    data={name:json.loads((OUT/f'{name}_statistics.json').read_text()) for name in names}
    atlas=ROOT/'outputs/FROM_WEB_v2/data/fig01_atlas_audit.json'
    data['figure1_display_subgraph']={'scope':'Frozen display subset; never the full database','statistics':json.loads(atlas.read_text())['statistics']}
    payload={'created_utc':datetime.now(timezone.utc).isoformat(),'scope':'Aggregate statistics only; no manuscript text, individual paper/claim IDs, edge lists, vectors or coordinates.','sections':data}
    (OUT/'ALL_GRAPH_NATURE_STATISTICS.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False))
    flat=list(flatten(payload))
    with (OUT/'ALL_STATISTICS.csv').open('w',newline='',encoding='utf-8-sig') as handle:
        writer=csv.writer(handle);writer.writerow(['统计路径/分组/指标','统计值'])
        for key,value in flat:
            text='' if value is None else str(value)
            if isinstance(value,str) and text.startswith(('=','+','-','@')):text="'"+text
            writer.writerow([key,text])
    intro='''# Claim Graph / Paper Graph / 全部 Nature 语料：统计总报告

本报告汇总实际本地资产和当前已保存运行结果；交付数据均为统计量，不含原始节点、边列表、论文正文、逐条claims、向量或坐标。统计时间、来源、分母及未计算项目见各部分；各部分有独立扫描窗口，不伪称全系统同一瞬间的事务快照。

## 四个范围务必分开

1. Paper Graph：包含目标论文及引用扩展文献的完整历史论文图。
2. Claim Graph：由历史目标论文抽取的主张图，不是所有Paper Graph论文都抽取了claim。
3. Nature语料：完整24,919篇历史论文，以及摘要/句子/claims/嵌入的处理覆盖。
4. 当前研究运行：原200篇与新增800篇分别统计；不与24,919篇历史语料混为一组。

Fig1的620 claims、519 papers展示子图仅单独列在机器统计中，不能替代全库总量。不同边族存在重叠，不能直接相加。旧历史分位数不能校准当前阈值检索；结构统计不等于科学创新或正确性。

本Markdown给出总览与各专题统计表；同目录HTML含全部统计的可展开浏览和搜索，JSON保留完整分组，CSV列出每项统计值，均不含逐篇/逐节点研究记录。

'''
    overview='## 核心规模（不同范围不能相加）\n\n|对象|统计量|\n|---|---:|\n'
    overview+=f"|完整Paper Graph节点|{data['paper_graph']['nodes']['raw_rows']:,}|\n|原始论文引用边|{data['paper_graph']['edges']['raw_rows']:,}|\n|排除自环后的唯一引用边|{data['paper_graph']['edges']['clean_distinct_nonself_edges']:,}|\n|历史Claim Graph节点|{data['claim_graph']['nodes']['rows']:,}|\n|历史语义主张边|{data['claim_graph']['edges']['semantic_claim_edges']['row_count']:,}|\n|用于局部拓扑/社区的backbone边|{data['claim_graph']['edges']['claim_backbone_edges']['row_count']:,}|\n|当前运行队列论文|{data['runtime_graph']['cohorts']['all1000']['roster_papers']:,}|\n|当前运行共享claims|{data['runtime_graph']['cohorts']['all1000']['shared_claims']:,}|\n\n"
    intro+=overview
    parts=[]
    for name in names:parts.append((OUT/f'{name}_summary.md').read_text())
    (OUT/'数据统计总报告.md').write_text(intro+'\n\n---\n\n'.join(parts))
    encoded=json.dumps(payload,ensure_ascii=False,allow_nan=False).replace('<','\\u003c')
    document='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ASPR 全量图谱与Nature统计</title><style>body{font:15px/1.6 system-ui,sans-serif;color:#252525;margin:24px auto;padding:0 20px;max-width:1280px}h1{color:#430258}header{background:#f4f3f7;padding:20px;border-radius:8px}input{font:inherit;padding:10px;width:min(90%,700px)}details{margin:8px 0 8px 14px;border-left:2px solid #ddd;padding-left:12px}summary{cursor:pointer;color:#365d8d;font-weight:600}table{border-collapse:collapse;width:100%;table-layout:fixed}td,th{padding:5px 9px;border-bottom:1px solid #ddd;text-align:left;overflow-wrap:anywhere}td:first-child{width:65%}small{color:#616161}button{padding:9px;margin:8px;cursor:pointer}</style><header><h1>Claim Graph · Paper Graph · 全部 Nature 语料</h1><p>全量汇总统计：节点、边、类型、时间、拓扑、覆盖、指标分布与分组。无原始论文/节点/边记录。</p><p>历史Nature语料、引用扩展Paper Graph、历史Claim Graph和当前200/800运行队列分别列示；缺失或未计算均保留说明。</p><input id="query" placeholder="搜索统计路径、指标或分组，例如 neighbor / missing / year"><button id="download">下载完整统计JSON</button><small id="count"></small></header><main id="tree"></main><section id="results" hidden></section><script id="data" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.getElementById('data').textContent);const root=document.getElementById('tree');
function leaf(value){return value===null?'NA / null':String(value)}
function build(parent,key,value){if(value!==null&&typeof value==='object'){const d=document.createElement('details'),s=document.createElement('summary');s.textContent=key+' · '+Object.keys(value).length+'项';d.append(s);let loaded=false;d.addEventListener('toggle',()=>{if(d.open&&!loaded){loaded=true;const entries=Object.entries(value);let offset=0;const more=document.createElement('button');function batch(){more.remove();for(const[k,v]of entries.slice(offset,offset+200))build(d,k,v);offset+=200;if(offset<entries.length){more.textContent='再显示200项（余'+(entries.length-offset)+'项）';d.append(more)}}more.onclick=batch;batch()}});parent.append(d)}else{const p=document.createElement('p');const b=document.createElement('b');b.textContent=key+': ';p.append(b,document.createTextNode(leaf(value)));parent.append(p)}}
for(const[k,v]of Object.entries(data))build(root,k,v);const all=[];function walk(v,p){if(v!==null&&typeof v==='object'){for(const[k,x]of Object.entries(v))walk(x,p?p+'.'+k:k)}else all.push([p,leaf(v)])}walk(data,'');
let timer;document.getElementById('query').addEventListener('input',e=>{clearTimeout(timer);timer=setTimeout(()=>{const q=e.target.value.toLowerCase().trim(),r=document.getElementById('results');r.replaceChildren();root.hidden=!!q;r.hidden=!q;if(!q)return;const matches=all.filter(x=>(x[0]+' '+x[1]).toLowerCase().includes(q));document.getElementById('count').textContent=matches.length+'项匹配（页面最多显示500项；完整数据已嵌入本文件）';const t=document.createElement('table');for(const row of matches.slice(0,500)){const tr=document.createElement('tr');for(const val of row){const td=document.createElement('td');td.textContent=val;tr.append(td)}t.append(tr)}r.append(t)},180)});
document.getElementById('download').onclick=()=>{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));a.download='ALL_GRAPH_NATURE_STATISTICS.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};
</script></html>'''
    (OUT/'全部图谱与Nature统计.html').write_text(document.replace('__DATA__',encoded))
    print(json.dumps({'aggregate_leaf_values':len(flat),'sections':names,'output':str(OUT)},ensure_ascii=False))


if __name__=='__main__':main()
