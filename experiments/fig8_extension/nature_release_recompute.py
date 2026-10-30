"""Offline standard-library recomputation from the portable released JSONL tables."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def compute(root: Path) -> dict[str, Any]:
    protocol=json.loads((root/'protocol.json').read_text())
    results={}
    papers=[json.loads(line) for line in (root/'papers.jsonl').read_text().splitlines()]
    for scenario,rule in protocol['scenarios'].items():
        countries: dict[str,float]=defaultdict(float)
        universities: dict[str,float]=defaultdict(float)
        paper_n=0
        for paper in papers:
            authors=[a for a in paper['authors'] if not rule['personal_only'] or not a['nonpersonal']]
            if rule['complete_case'] and any(s['identity_missing'] for a in authors for s in a['slots']):
                continue
            if not authors:
                raise ValueError('No eligible observed authors: '+paper['doi'])
            paper_n+=1
            for a in authors:
                slots=a['slots']
                for s in slots:
                    codes=s['country_codes'] or ['unknown']
                    for code in codes:
                        countries[code]+=1/len(authors)/len(slots)/len(codes)
                if rule['deduplicate_university']:
                    groups={}
                    for s in slots:
                        targets=s['university_targets']['direct_units']
                        group=next(iter(targets)) if len(targets)==1 else s['entity_id']
                        groups[group]=targets
                    maps=list(groups.values())
                else:
                    maps=[s['university_targets'][rule['hierarchy']] for s in slots]
                for targets in maps:
                    for ident,fraction in targets.items():
                        universities[ident]+=fraction/len(authors)/len(maps)
        assert abs(sum(countries.values())-paper_n)<1e-7
        results[scenario]={'paper_n':paper_n,'countries':dict(countries),'universities':dict(universities)}
    return results


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('release',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    result=compute(args.release)
    if args.output:
        args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    expected=args.release/'expected_results.json'
    if expected.exists():
        other=json.loads(expected.read_text())
        for scenario,data in result.items():
            assert data['paper_n']==other[scenario]['paper_n']
            for category in ['countries','universities']:
                assert set(data[category])==set(other[scenario][category])
                assert all(abs(v-other[scenario][category][k])<1e-9 for k,v in data[category].items())
    print('Offline reproduction passed:',len(result),'scenarios')


if __name__=='__main__':
    main()
