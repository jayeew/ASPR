from __future__ import annotations
import json
from pathlib import Path
import pytest
from figure_pipeline.benchmark_revision import operational, paired, transitions
from figure_pipeline.specs.contract import require_publication

ROOT=Path(__file__).resolve().parents[2]


def test_missing_card_does_not_become_zero_evidence() -> None:
    snapshot={'ROSTER':[{'paper_id':'p','journal_name':'J'}],'CLAIMS':[{'paper_id':'p','gear_card_available':False,'usable_compared_historical_passage':None,'graph_neighbor_count':10}]}
    row=operational(snapshot)[0]
    assert row['execution_status'] is None
    assert row['evidence_coverage_among_analyzed'] is None
    assert row['usable_evidence_claims']==0
    assert row['analyzed_claims']==0
    assert row['requested_claims']==1


def test_frozen_negative_effect_and_paired_denominators() -> None:
    snapshot=json.loads((ROOT/'outputs/FROM_WEB/data/fig04_fig06_fig07_snapshot.json').read_text())
    summary,points=paired(snapshot)
    assert [r['n'] for r in summary]==[189,152,152,189]
    assert summary[1]['delta'][0]==pytest.approx(-16.31299,abs=.01)
    assert summary[1]['delta'][2]<0
    assert summary[3]['delta'][1]<0<summary[3]['delta'][2]
    assert len({p['paper_id'] for p in points if p['metric']=='coverage_full_rate'})==189
    aligned=transitions(snapshot)
    assert len({(r['paper_id'],r['reference_id']) for r in aligned})==len(aligned)
    assert all(r['independent_correctness'] is None for r in aligned)


def test_publication_rejects_missing_fusion() -> None:
    with pytest.raises(ValueError,match='full_fusion_reports'):
        require_publication([5],{})
    require_publication([1],{'clean_atlas':True,'frozen_local_cases':True})


def test_publication_exports_only_requested_figures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import figure_pipeline.build as builder
    audit=tmp_path/'audit'; audit.mkdir()
    (audit/'artifact_attestations.json').write_text(json.dumps({'clean_atlas':True,'frozen_local_cases':True}))
    (audit/'manual_visual_review.json').write_text(json.dumps({'1':{'approved_for_publication':True}}))
    (tmp_path/'fig01_example.svg').write_text('<svg/>')
    (tmp_path/'fig03_unverified.svg').write_text('<svg/>')
    monkeypatch.setattr(builder,'OUT',tmp_path)
    monkeypatch.setattr('sys.argv',['build','--mode','publication','--figures','1'])
    builder.main()
    assert (tmp_path/'publication/fig01_example.svg').exists()
    assert not (tmp_path/'publication/fig03_unverified.svg').exists()
