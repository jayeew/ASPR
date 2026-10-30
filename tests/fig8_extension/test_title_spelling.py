from __future__ import annotations
import pytest
from experiments.fig8_extension.fig8b_qualification_build import match


def evidence(title: str, doi: str = '10.1/example', issn: str = '1234-5678', kind: str = 'research-article') -> dict:
    row={'work_id':'W1','doi':'10.1/example','title':'Tumor signalling and modelling','source_id':'S1','journal_name':'Journal'}
    item={'doi':doi,'title':title,'issns':[issn],'journal_titles':['Journal'],'kind':kind,'family':'archived_jats','categories':[]}
    return match(row,item,{'issn':'["1234-5678"]'})


def test_finite_spelling_with_doi_and_journal() -> None:
    assert evidence('Tumor signaling and modeling')['identity']=='matched'


@pytest.mark.parametrize('title,doi,issn',[
    ('Tumor signaling and modeling','10.1/other','1234-5678'),
    ('Tumor signaling and modeling','','1234-5678'),
    ('Tumor signaling and modeling','10.1/example','9999-0000'),
    ('Tumor signaling and modeling in mice','10.1/example','1234-5678'),
    ('Tumor signaling without modeling','10.1/example','1234-5678'),
])
def test_identity_differences_remain_held(title: str,doi: str,issn: str) -> None:
    assert evidence(title,doi,issn)['identity']!='matched'


def test_spelling_does_not_promote_review() -> None:
    assert evidence('Tumor signaling and modeling',kind='review')['class']=='nonprimary'
