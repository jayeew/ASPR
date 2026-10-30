from experiments.fig8_extension.country_accent_names import aliases_for, enrich


def test_unique_full_accent_name_requires_native_exact_address() -> None:
    authors=[{'name':'Dejan Lazarević','countries':[],'publisher_affiliation_source':'raw',
        'publisher_affiliation_slots':[{'text':'Institute, Milan, 20132 Italy','country_codes':[]}]}]
    source={'authors':[{'name':'Dejan Lazarevic','affiliations':[{'text':'Institute, Milan, 20132 Italy'}]}]}
    rules={'Institute, Milan, 20132 Italy':'IT'}
    result,slots,people=enrich(authors,source,'raw',rules,'rule','explicit_terminal_country_name')
    assert slots==people==1
    assert result[0]['name']=='Dejan Lazarević'
    assert result[0]['publisher_affiliation_slots'][0]['country_evidence']['native_author_name']=='Dejan Lazarevic'
    assert enrich(result,source,'raw',rules,'rule','explicit_terminal_country_name')[0]==result
    assert enrich(authors,source,'raw',{'Different address':'IT'},'rule','explicit_terminal_country_name')[1]==0


def test_ambiguous_abbreviated_reordered_and_nonlatin_names_stay_unmatched() -> None:
    assert aliases_for([{'name':'S Kumar'}],{'authors':[{'name':'Sushil Kumar'}]})=={}
    assert aliases_for([{'name':'A B'}],{'authors':[{'name':'B A'}]})=={}
    assert aliases_for([{'name':'José A'},{'name':'Jose A'}],{'authors':[{'name':'Jose A'}]})=={}
    assert aliases_for([{'name':'José A'}],{'authors':[{'name':'Jose A'},{'name':'José A'}]})=={}
    assert aliases_for([{'name':'Иван Й'}],{'authors':[{'name':'Иван И'}]})=={}
