"""Checks specific to 2026 source binding and future-word-cloud counts."""
from pathlib import Path

import pytest

from figure_pipeline.fig7_reference.data import write
from figure_pipeline.fig7_reference.frontier_predict import validate
from figure_pipeline.fig7_reference.frontier_render import cloud_rows


def packet() -> dict:
    return {'packet_id':'P01','evidence_columns':['id','paper_id','claim','provenance'],
        'evidence':[['T','2026-A','New target finding.','2026_fulltext'],
                    ['H','2024-B','Historical observation.','2023_2025_abstract'],
                    ['X','2025-C','Unlinked historical finding.','2023_2025_abstract']],
        'graph_columns':['id','target','neighbors'], 'graph':[['G','T',[['H',.7]]]]}


def direction() -> dict:
    return {'direction':'Test the next mechanism.','observed_2026_basis':'New finding',
        'future_extension':'Test the mechanism experimentally.','graph_refs':['G'],
        'graph_basis':'A real neighbor.', 'citations':[{'source_id':'T','quote':'New target finding.'},
        {'source_id':'H','quote':'Historical observation.'}],
        'labels':[{'category':'Problem','phrase':'Resolve specific mechanism','display_label':'Resolve mechanism'}]}


def test_actual_2026_and_historical_endpoints_required() -> None:
    p=packet();d=direction()
    good,bad=validate(p,{'directions':[d]})
    assert len(good)==1 and not bad
    d['citations'][1]={'source_id':'X','quote':'Unlinked historical finding.'}
    good,bad=validate(p,{'directions':[d]})
    assert not good and 'graph_endpoints_not_cited' in bad[0]['errors']


def test_future_sources_cannot_be_historical_only_or_paraphrased() -> None:
    d=direction();d['citations'][0]['quote']='A rewritten claim'
    assert 'quote_not_exact' in validate(packet(),{'directions':[d]})[1][0]['errors']
    p=packet();p['evidence'][0][3]='2023_2025_abstract'
    assert 'requires_2026_and_historical_sources' in validate(p,{'directions':[direction()]})[1][0]['errors']


def test_tied_scores_do_not_invent_size_hierarchy(tmp_path: Path) -> None:
    d=direction();d['candidate_id']='P01-D01'
    # The same label twice within one direction must still count once.
    d['labels']*=2
    write(tmp_path/'data/directions.json',[d])
    write(tmp_path/'data/evidence.json',{'T':{'paper_id':'A'},'H':{'paper_id':'B'}})
    rows=cloud_rows(tmp_path)
    assert len(rows)==1 and rows[0]['frequency']==1
    assert rows[0]['source_paper_count']==2 and rows[0]['font_size_pt']==18


def test_source_union_not_sum_across_directions(tmp_path: Path) -> None:
    d=direction();d['candidate_id']='P01-D01'
    e={**d,'candidate_id':'P01-D02'}
    write(tmp_path/'data/directions.json',[d,e])
    write(tmp_path/'data/evidence.json',{'T':{'paper_id':'A'},'H':{'paper_id':'B'}})
    r=cloud_rows(tmp_path)[0]
    assert r['frequency']==2 and r['source_paper_count']==2


def test_conflicting_short_labels_are_not_silently_merged(tmp_path: Path) -> None:
    d=direction();d['candidate_id']='P01-D01'
    e={**direction(),'candidate_id':'P01-D02'}
    e['labels'][0]['display_label']='Different meaning'
    write(tmp_path/'data/directions.json',[d,e])
    write(tmp_path/'data/evidence.json',{})
    with pytest.raises(ValueError,match='conflicting display labels'):
        cloud_rows(tmp_path)


def test_mixed_historical_and_fulltext_sources_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import csv
    from figure_pipeline.fig7_reference import frontier_predict as module
    write(tmp_path/'packets/P01.json',packet())
    write(tmp_path/'sources/P01.json',{
        'T':{'claim_id':'target','paper_id':'2026-A','internal_support':'supported'},
        'H':{'claim_id':'historical','paper_id':'2024-B','community_id':7}})
    def fake_obtain(destination: Path, ident: str, *args: object) -> dict:
        if ident=='merge':
            return {'retained':[{'candidate_id':'P01-D01','labels':direction()['labels']}],
                    'excluded':[],'limitations':''}
        return {'directions':[direction()],'limitations':''}
    monkeypatch.setattr(module,'obtain',fake_obtain)
    module.predict(tmp_path,2)
    with (tmp_path/'tables/direction_sources.csv').open(encoding='utf-8-sig') as handle:
        rows=list(csv.DictReader(handle))
    assert len(rows)==2 and rows[0]['internal_support']=='supported'
    assert rows[1]['community_id']=='7' and rows[1]['source_id']=='P01:H'
