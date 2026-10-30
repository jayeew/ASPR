from experiments.fig8_extension.fig8b_qualification_decide import imprint_dates, portfolio_label


def records(day='2024-01-02'):
    return [{'id':'archive','identity':'matched','family':'archived_jats','publishers':['Wiley'],
        'online_date':'','publication_dates':[day]},
        {'id':'date','identity':'matched','family':'crossref_publisher_deposit',
         'publishers':['Wiley'],'online_date':'2024-01-02'}]


def test_archival_imprint_requires_full_date_agreement():
    assert portfolio_label('',records(),2024)==('False',['archive','date'])
    assert portfolio_label('',records('2024-01-03'),2024)[0]=='unresolved'
    assert portfolio_label('',records('2024'),2024)[0]=='unresolved'
    assert portfolio_label('',records()[1:],2024)[0]=='unresolved'
    assert portfolio_label('True',records(),2024)[0]=='unresolved'
    values=records();values[1]['identity']='mismatch'
    assert portfolio_label('',values,2024)[0]=='unresolved'


def test_no_collection_or_partial_date_promoted():
    rows=[{'attributes':{},'year':'2024','month':'1','day':'2'},
          {'attributes':{'pub-type':'collection'},'year':'2024','month':'2','day':'1'},
          {'attributes':{},'year':'2024','month':'3','day':''}]
    assert imprint_dates(rows)==['2024-01-02']
