from experiments.fig8_extension.fig8b_qualification_decide import journal_research_format, science_china_chemistry_format


def evidence(kind: str, categories: list[str]) -> dict:
    return {'identity':'matched','family':'publisher_springer','source_id':'https://openalex.org/S147836164',
            'kind':kind,'categories':categories,'class':'subtype_review_required','title':'A new catalyst'}


def test_scoped_literal_formats_do_not_infer_generic_short_format() -> None:
    assert science_china_chemistry_format(evidence('BriefCommunication',['Highlights'])) == 'nonprimary'
    assert not journal_research_format(evidence('BriefCommunication',['Highlights']))
    assert journal_research_format(evidence('BriefCommunication',['Communications']))
    assert not journal_research_format(evidence('BriefCommunication',['Brief Communications']))
    assert not science_china_chemistry_format({**evidence('Highlights',[]),'source_id':'other'})
    assert not science_china_chemistry_format(evidence('OriginalPaper',['Highlights']))
    assert not science_china_chemistry_format({**evidence('Highlights',[]),'class':'integrity_hold'})


def test_materials_uses_own_literal_formats_and_retains_other_holds() -> None:
    from experiments.fig8_extension.fig8b_qualification_decide import science_china_materials_format
    def material(kind: str, categories: list[str]) -> dict:
        return {**evidence(kind, categories), 'source_id': 'https://openalex.org/S2764782236'}
    assert journal_research_format(material('Letter',['Letters']))
    assert science_china_materials_format(material('BriefCommunication',['Highlights'])) == 'nonprimary'
    assert not journal_research_format(material('BriefCommunication',['Highlights']))
    assert not journal_research_format(material('BriefCommunication',[]))
    assert not science_china_materials_format(material('Letter',['Commentary']))
    assert not journal_research_format({**material('Letter',[]),'title_scope_flag':True})
    assert not science_china_materials_format(evidence('Letter',[]))


def test_previously_research_paper_with_new_explicit_highlight_conflict_is_held() -> None:
    from experiments.fig8_extension.fig8b_qualification_decide import decide
    highlight={**evidence('Highlights',[]),'id':'native-highlight','online_date':'2023-03-21','class':'unresolved_type'}
    original={**evidence('OriginalPaper',['Highlights','OriginalPaper']),'id':'internal-original','online_date':'2023-03-21','class':'research'}
    result=decide([highlight,original],[])
    assert result['status']=='pending_conflicting_primary_types'
    assert set(result['evidence_ids'])=={'native-highlight','internal-original'}
