"""Combine completed uniform passes, preserving both raw model outputs."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent

def read(path: Path) -> dict | list:
    return json.loads(path.read_text())

def write(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main() -> None:
    core=read(HERE/'theme_codebook.json')
    supplement=read(HERE/'supplement_codebook.json')
    if isinstance(supplement,dict):
        if supplement.get('status')!='frozen_for_uniform_screening':
            raise ValueError('Supplementary definitions must be frozen before measurement')
        supplement=supplement['themes']
    core_ids={t['theme_id'] for t in core}
    supp_ids={t['theme_id'] for t in supplement}
    replacements=core_ids & supp_ids
    definitions={t['theme_id']:t for t in core}
    definitions.update({t['theme_id']:t for t in supplement})
    fullbook=[definitions[t] for t in sorted(definitions,key=lambda t:int(t[1:]))]
    reports=[json.loads(l) for l in (HERE/'raw_reports.jsonl').read_text().splitlines()]
    if len(reports)!=994 or len({r['pilot_id'] for r in reports})!=994:
        raise ValueError('Expected the same 994 eligible papers')
    combined=HERE/'complete_screening'
    combined.mkdir(exist_ok=True)
    write(combined/'theme_codebook.json',fullbook)
    link=combined/'raw_reports.jsonl'
    if not link.exists():link.symlink_to(Path('..')/'raw_reports.jsonl')
    discordances=[]; manifest=[]
    # Check all required files before constructing any final statistics input.
    for r in reports:
        for folder in ['theme_matches','supplement_matches']:
            if not (HERE/folder/f"{r['pilot_id']}.json").exists():
                raise ValueError(f"Incomplete {folder}: {r['pilot_id']}")
    for r in reports:
        pid=r['pilot_id'];cp=HERE/'theme_matches'/f'{pid}.json';sp=HERE/'supplement_matches'/f'{pid}.json'
        a,b=read(cp),read(sp)
        for result,ids in [(a,core_ids),(b,supp_ids)]:
            if result['pilot_id']!=pid or result['paper_id']!=r['paper_id'] or result['read_complete'] is not True:
                raise ValueError(f'Identity/incomplete result: {pid}')
            hits=[x['theme_id'] for x in result['matches']]
            if len(hits)!=len(set(hits)) or not set(hits)<=ids:
                raise ValueError(f'Invalid theme IDs: {pid}')
        bhits={x['theme_id'] for x in b['matches']}
        for x in a['matches']:
            if x['theme_id'] in replacements and x['theme_id'] not in bhits:
                discordances.append(dict(pilot_id=pid,theme_id=x['theme_id'],core_match=x,
                    reason='Narrow core positive not recovered by broader uniform replacement; semantic adjudication required'))
        merged=[dict(x,screening_pass='core') for x in a['matches'] if x['theme_id'] not in replacements]
        merged += [dict(x,screening_pass='supplement') for x in b['matches']]
        item=dict(pilot_id=pid,paper_id=r['paper_id'],read_complete=True,matches=merged,
            other_concerns=a['other_concerns'],source_warnings=list(dict.fromkeys(a['source_warnings']+b['source_warnings'])),
            artifact_kind='combination of two preserved model passes; not a new model response',
            core_result=str(cp),core_result_sha256=sha(cp),supplement_result=str(sp),supplement_result_sha256=sha(sp),
            core_input_fingerprint=a['input_fingerprint'],supplement_input_fingerprint=b['input_fingerprint'])
        write(combined/'theme_matches'/f'{pid}.json',item)
        manifest.append(dict(pilot_id=pid,core_sha256=sha(cp),supplement_sha256=sha(sp)))
    ledger=read(HERE/'semantic_adjustments.json')
    retained=[];superseded=[]
    for x in ledger['adjustments']:
        tid=x.get('theme_id',x.get('evidence',{}).get('theme_id'))
        (superseded if tid in replacements else retained).append(x)
    extra=HERE/'supplement_semantic_adjustments.json'
    if extra.exists():retained += read(extra)['adjustments']
    write(combined/'semantic_adjustments.json',dict(version=1,adjustments=retained))
    write(combined/'superseded_core_adjustments.json',dict(reason='These core theme definitions were uniformly replaced; prior records remain in original ledger',
        replaced_theme_ids=sorted(replacements),adjustments=superseded))
    (HERE/'override_discordances.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in discordances))
    write(HERE/'screening_combination_audit.json',dict(papers=len(reports),themes=len(fullbook),
        core_themes=len(core),supplement_targets=len(supplement),replacement_ids=sorted(replacements),
        new_theme_ids=sorted(supp_ids-core_ids),core_codebook_sha256=sha(HERE/'theme_codebook.json'),
        supplementary_codebook_sha256=sha(HERE/'supplement_codebook.json'),raw_outputs_modified=False,
        core_adjustments_retained=len(retained),core_adjustments_superseded=len(superseded),
        override_discordances=len(discordances),manifest=manifest))
    print(json.dumps(dict(papers=len(reports),themes=len(fullbook),replacement_ids=sorted(replacements),discordances=len(discordances))))

if __name__=='__main__':main()
