from experiments.fig8_extension.corroborated_roster_repair import reconcile


def test_duplicate_placeholder_removed_only_with_unique_native_identity():
    a={'name':'Alice Li','position':0,'author_id':'A1','corroborated_orcid':'0000-0002-1825-0097',
       'publisher_affiliation_slots':[{'text':'Lab, Canada','country_codes':['CA']}],
       'publisher_affiliation_source':'front.xml','countries':['CA']}
    placeholder={'name':'Alice Li','position':1,'author_id':None}
    source={'evidence_receipt':'front.xml','authors':[{'name':'Alice Li','nonpersonal':False,
        'orcid':a['corroborated_orcid'],'affiliations':[{'text':'Lab, Canada'}]}]}
    out,status=reconcile([a,placeholder],source)
    assert status=='publisher_roster_reconciled' and out==[a]
    assert reconcile(out,source)==(out,status)
    assert reconcile([a,{**a,'position':1}],source)[1]=='ambiguous_current_orcid'
    assert reconcile([a,{**a,'position':1,'corroborated_orcid':None}],source)[1]=='multiple_primary_records_for_native_author'
    assert reconcile([{**a,'nature_v1_slots':[]}],source)[1]=='nature_preserved'
    assert reconcile([a],{**source,'evidence_receipt':'other'})[1]=='existing_primary_source_conflict'
    assert reconcile([a],{**source,'byline_has_etal':True})[1]=='native_byline_incomplete_or_ambiguous'
