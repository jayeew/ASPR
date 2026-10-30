"""Meaningful guards for closure identity matching and portable accounting."""
from __future__ import annotations

import json
from pathlib import Path

from experiments.fig8_extension import nature_release_review as review
from experiments.fig8_extension.nature_release_recompute import compute


def test_city_name_is_not_an_organization_match() -> None:
    records={'r':{'organization':{'id':'r','locations':[{'geonames_details':{'name':'Singapore','country_code':'SG'}}]},'receipt':'example'}}
    entity,evidence=review.candidate('Unverified Company, Singapore, Singapore',records,{'singapore':{'r'}},{},'SG')
    assert entity is None
    assert evidence['registry_exact_name_candidates']==[]


def test_exact_name_wrong_city_stays_unresolved() -> None:
    records={'r':{'organization':{'id':'r','locations':[{'geonames_details':{'name':'Boston','country_code':'US'}}]},'receipt':'example'}}
    entity,evidence=review.candidate('Example University, Chicago, IL, USA',records,{'example university':{'r'}},{},'US')
    assert entity is None
    assert evidence['accepted_country_city_candidates']==[]


def test_country_aliases() -> None:
    assert review.address_country('Independent consultant, Cambridge, Great Britain',review.COUNTRIES)=='GB'
    assert review.address_country('Hospital, Gaziantep, Turkey',review.COUNTRIES)=='TR'
    assert review.address_country('Company, Cambridge',review.COUNTRIES) is None


def test_individual_link_not_whole_page_failure(monkeypatch) -> None:
    monkeypatch.setattr(review,'author_addresses',lambda a,p: (['Department, University, USA'],'publisher_exact_author_name'))
    assert review.link({}, {'author_affiliation_parse_complete':False})[2]
    monkeypatch.setattr(review,'author_addresses',lambda a,p: (['Department, University, USA'],'openalex_raw_affiliation'))
    assert not review.link({}, {})[2]


def test_portable_unknown_mass_and_personal_denominator(tmp_path: Path) -> None:
    rule={'hierarchy':'direct_units','personal_only':True,'complete_case':False,'deduplicate_university':False}
    (tmp_path/'protocol.json').write_text(json.dumps({'scenarios':{'main':rule}}))
    known={'country_codes':['US'],'identity_missing':False,'entity_id':'u','university_targets':{'direct_units':{'u':1}}}
    unknown={'country_codes':[],'identity_missing':True,'entity_id':'?','university_targets':{'direct_units':{}}}
    paper={'doi':'example','authors':[{'nonpersonal':False,'slots':[known,unknown]},
        {'nonpersonal':False,'slots':[known]},{'nonpersonal':True,'slots':[unknown]}]}
    (tmp_path/'papers.jsonl').write_text(json.dumps(paper)+'\n')
    data=compute(tmp_path)['main']
    assert data['countries']=={'US':.75,'unknown':.25}
    assert data['universities']=={'u':.75}
    assert data['paper_n']==1
