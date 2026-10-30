import json

from experiments.fig8_extension.country_incremental import identity_changed


def test_country_updates_do_not_reopen_but_identity_changes_do():
    a={'name':'A Li','author_id':'A1','position':0,'countries':['CN']}
    before=json.dumps([a])
    assert not identity_changed(before,json.dumps([{**a,'countries':['US'],'country_conflict_retained':True}]))
    assert not identity_changed(json.dumps(before),before)
    assert identity_changed(before,json.dumps([a,{**a,'position':1,'author_id':None}]))
    assert identity_changed(before,json.dumps([{**a,'corroborated_orcid':'0000-0002-1825-0097'}]))
    assert identity_changed(before,json.dumps([{**a,'publisher_affiliation_source':'new.xml'}]))
