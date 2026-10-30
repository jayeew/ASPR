"""Clinical rollups must preserve definition and publisher-link boundaries."""
from experiments.fig8_extension import nature_clinical_review as review
from experiments.fig8_extension.nature_boundary_rank import university_shares


def test_clinical_vs_academic_definition() -> None:
    h = object.__new__(review.ClinicalHierarchy)
    base = {'review_entity_id':'unit', 'clinical_rollup':{'kind':'health_system','target_id':'pku','official_url':review.LAB}}
    assert h.targets(base,'direct_units') == {'pku':1.0}
    assert h.targets(base,'academic_only') == {}
    academic = {'review_entity_id':'cuimc', 'clinical_rollup':{'kind':'academic','target_id':'columbia','official_url':review.CU}}
    assert h.targets(academic,'academic_only') == {'columbia':1.0}
    assert h.targets(academic,'exclude_all_reviewed_medical') == {}


def test_deduplication_changes_denominator() -> None:
    h = object.__new__(review.ClinicalHierarchy)
    slots = [{'review_entity_id':str(i),'review_entity_type':'research_institute',
              'clinical_rollup':{'kind':'academic','target_id':target,'official_url':review.CU}}
             for i,target in enumerate(['columbia','columbia','other'])]
    assert university_shares(slots,h,'direct_units') == {'columbia':2/3,'other':1/3}
    assert university_shares(slots,h,'university_deduplicated') == {'columbia':.5,'other':.5}


def test_incomplete_or_raw_fallback_not_eligible(monkeypatch) -> None:
    monkeypatch.setattr(review,'author_addresses',lambda a,p: (['address'],'openalex_raw_affiliation'))
    assert not review.eligible({}, {'author_affiliation_parse_complete':True})[1]
    monkeypatch.setattr(review,'author_addresses',lambda a,p: (['address'],'publisher_exact_author_name'))
    assert not review.eligible({}, {'author_affiliation_parse_complete':False})[1]
    assert review.eligible({}, {'author_affiliation_parse_complete':True})[1]
