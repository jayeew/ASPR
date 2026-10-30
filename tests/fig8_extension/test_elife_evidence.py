from experiments.fig8_extension.elife_evidence import evidence


def record() -> dict:
    return {'native': {'doi': '10.7554/eLife.1', 'title': 'A <i>study</i>',
                       'status': 'vor', 'stage': 'published', 'type': 'research-article',
                       'published': '2023-01-02T00:00:00Z', 'versionDate': '2026-01-01',
                       'elocationId': 'e1'},
            'candidate': {'doi': '10.7554/elife.1', 'journal': 'eLife', 'source_id': 'S1'},
            'raw_receipt': 'test.json.gz'}


def test_first_publication_not_latest_version() -> None:
    e = evidence(record())
    assert e['online_date'] == '2023-01-02'
    assert e['title'] == 'A study'


def test_preprint_path_not_auto_qualified() -> None:
    r = record()
    r['native']['elocationId'] = 'RP1'
    e = evidence(r)
    assert e['scope_review_required']
    assert not e['online_date']
    r['native']['status'] = 'preprint'
    assert evidence(r) is None


def test_journal_specific_formats_retain_holds() -> None:
    from experiments.fig8_extension.fig8b_qualification_decide import journal_research_format, type_class

    e = {'identity': 'matched', 'family': 'publisher_elife_api',
         'source_id': 'https://openalex.org/S1336409049', 'kind': 'short-report',
         'type_policy_receipt': 'policy.json', 'title': 'Experimental results'}
    assert journal_research_format(e)
    assert not journal_research_format({**e, 'scope_review_required': True})
    assert not journal_research_format({**e, 'title': 'A systematic review'})
    assert not journal_research_format({**e, 'source_id': 'another-journal'})
    assert type_class('insight', 'publisher_elife_api') == 'nonprimary'
    assert type_class('insight', 'other') == 'unresolved_type'


def test_v8_version_identity_can_resolve_only_rp_path_hold() -> None:
    from experiments.fig8_extension.fig8b_qualification_decide import resolve_elife_publication_path

    r = record()
    r['candidate']['doi'] = '10.7554/elife.1.3'
    r['native'].update(doiVersion='10.7554/eLife.1.3', elocationId='RP1')
    assert evidence(r) is None
    r.update(verified_v8=True, raw_sha256='confirmed-hash')
    new = evidence(r)
    assert new['online_date'] == '2023-01-02'
    old = {**new, 'native_api_version': 0, 'scope_review_required': True,
           'scope_note': 'Reviewed-preprint pathway requires historical journal publication date review',
           'class': 'subtype_review_required'}
    resolved = resolve_elife_publication_path([old, new])
    assert resolved[0]['class'] == 'research'
    assert not resolve_elife_publication_path([{**old, 'title_scope_flag': 'review'}, new])[0] == resolved[0]


def test_entity_only_title_decodes_without_losing_comparison_symbol() -> None:
    from experiments.fig8_extension.elife_text import plain

    assert plain('3D genomic features across &gt;50 diverse cell types') == '3D genomic features across >50 diverse cell types'
    assert plain('A <i>study</i> &amp; results') == 'A study & results'
    assert plain('A &amp;gt; B') == 'A &gt; B'
