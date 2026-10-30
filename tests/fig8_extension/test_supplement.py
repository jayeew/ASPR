from __future__ import annotations

import json
from pathlib import Path

from experiments.fig8_extension.fig8b_qualification_build import portfolio_label
from experiments.fig8_extension.supplement_apply import (
    abstract_ok,
    complete_byline,
    countries,
    enrich_authors,
    signature_match,
)
from experiments.fig8_extension.supplement_queue import connect, route, task
from experiments.fig8_extension.supplement_sources import jats_metadata, parse_publisher, publisher_url


def test_btpr_uses_reviewed_host_despite_stale_acs_registry() -> None:
    doi = '10.1002/btpr.3344'
    assert route('American Chemical Society', doi) == 'wiley'
    assert publisher_url(doi, 'acs') == 'https://aiche.onlinelibrary.wiley.com/doi/' + doi
    other = '10.1021/accountsmr.3c00112'
    assert route('American Chemical Society', other) == 'acs'
    assert publisher_url(other, 'acs').startswith('https://pubs.acs.org/')


def publisher_author(name: str = 'Alice Smith') -> dict:
    return {'name': name, 'nonpersonal': False, 'affiliations': [{'text': 'Example University, United States',
        'countries': [{'text': 'United States', 'code': 'US'}], 'identifiers': [], 'names': ['Example University']}],
        'unresolved_affiliation_refs': []}


def existing_author(name: str = 'Alice Smith') -> dict:
    return {'name': name, 'position': 0, 'countries': [], 'institution_ids_or_verified_local_entities': []}


def test_jats_preserves_author_affiliation_links_and_explicit_country() -> None:
    xml = b'''<article article-type="research-article" xml:lang="en"><front><journal-meta><issn>1234-5678</issn></journal-meta>
    <article-meta><article-id pub-id-type="doi">10.1/a</article-id><title-group><article-title>A title</article-title></title-group>
    <contrib-group><contrib contrib-type="author"><name><surname>Smith</surname><given-names>Alice</given-names></name><xref ref-type="aff" rid="a1"/></contrib></contrib-group>
    <aff id="a1"><institution>Example University</institution><country country="US">United States</country></aff>
    <abstract><title>Abstract</title><p>This is the actual abstract text.</p></abstract></article-meta></front></article>'''
    result = jats_metadata(xml)
    assert result['abstract'] == 'This is the actual abstract text.'
    assert result['authors'][0]['affiliations'][0]['countries'][0]['code'] == 'US'
    assert not result['authors'][0]['unresolved_affiliation_refs']


def test_generic_publisher_date_is_not_online_date() -> None:
    data = b'<meta name="citation_doi" content="10.1/a"><meta name="citation_publication_date" content="2024/01/01"><meta name="citation_title" content="A"><meta name="citation_article_type" content="Journal Article">'
    assert parse_publisher(data)['online_date'] == ''
    assert parse_publisher(data)['article_types'] == ['Journal Article']
    invalid = data + b'<meta name="citation_online_date" content="2024/99/99">'
    assert parse_publisher(invalid)['online_date'] == ''


def test_jats_structured_identifiers_are_not_affiliation_address_text() -> None:
    from lxml import etree
    from experiments.fig8_extension.supplement_sources import affiliation_text
    node = etree.fromstring(b'<aff><label>123</label><institution-id institution-id-type="ROR">https://ror.org/example</institution-id><institution-id institution-id-type="GRID">grid.123</institution-id><institution>CERN</institution>, Geneva, <country country="CH">Switzerland</country></aff>')
    before = etree.tostring(node)
    assert affiliation_text(node) == 'CERN, Geneva, Switzerland'
    assert etree.tostring(node) == before
    mixed = etree.fromstring(b'<aff><label>a</label>Department <institution>Institute</institution><institution-id>id</institution-id>, City, Country</aff>')
    assert affiliation_text(mixed) == 'Department Institute, City, Country'
    literal = etree.fromstring(b'<aff>123 Science Road, Institute, Country</aff>')
    assert affiliation_text(literal) == '123 Science Road, Institute, Country'


def test_missing_or_ambiguous_country_is_not_headquarters() -> None:
    author = publisher_author()
    assert countries(author['affiliations'], {'unitedstates': 'US'}) == ['US']
    author['affiliations'].append({'text': 'Unknown site', 'countries': []})
    assert countries(author['affiliations'], {}) == []
    assert countries([{'text': 'Hong Kong, China', 'countries': [{'text': 'China', 'code': 'CN'}]}], {'china': 'CN'}) == []


def test_author_identity_must_be_unique_not_fuzzy() -> None:
    source = {'authors': [publisher_author()], 'evidence_receipt': 'raw.xml'}
    result, _ = enrich_authors([existing_author('A. Smith')], source, {}, {}, {})
    assert result[0]['countries'] == []
    source['authors'].append(publisher_author())
    result, _ = enrich_authors([existing_author()], source, {}, {}, {})
    assert result[0]['countries'] == []


def test_exact_name_site_fills_identity_without_university_rollup() -> None:
    source = {'authors': [publisher_author()], 'evidence_receipt': 'raw.xml'}
    result, _ = enrich_authors([existing_author()], source, {},
        {'exampleuniversity': [{'id': 'I1', 'country_code': 'US'}]}, {'unitedstates': 'US'})
    assert result[0]['institution_ids_or_verified_local_entities'] == ['I1']
    assert result[0]['countries'] == ['US']
    assert result[0]['publisher_affiliation_slots'][0]['university_ownership_status'] == 'not_inferred_from_registry'
    repeated, _ = enrich_authors(result, source, {}, {'exampleuniversity': [{'id': 'I1', 'country_code': 'US'}]}, {'unitedstates': 'US'})
    assert json.dumps(repeated, sort_keys=True) == json.dumps(result, sort_keys=True)


def test_complete_byline_requires_old_author_subset_and_no_etal() -> None:
    source = {'authors': [publisher_author(), publisher_author('Bob Jones')], 'evidence_receipt': 'raw.xml'}
    result, complete = complete_byline([existing_author()], source)
    assert complete and len(result) == 2 and result[1]['name'] == 'Bob Jones'
    assert result[1]['author_id'] == ''
    assert complete_byline([existing_author('Other Person')], source)[1] is False
    assert complete_byline([existing_author()], {**source, 'byline_has_etal': True})[1] is False


def test_ledger_discovery_does_not_erase_completed_attempt_state(tmp_path: Path) -> None:
    db = connect(tmp_path)
    ident = task(db, 'qualification', '10.1/a', 'pmc', {'doi': '10.1/a'}, 'W1')
    db.execute("UPDATE tasks SET status='access_restricted',attempt_n=1 WHERE task_id=?", (ident,))
    task(db, 'qualification', '10.1/a', 'pmc', {'doi': '10.1/a', 'pmcids': ['PMC1']}, 'W1')
    assert db.execute('SELECT count(*) FROM tasks').fetchone()[0] == 1
    assert db.execute('SELECT status,attempt_n FROM tasks').fetchone() == ('access_restricted', 1)
    assert db.execute('SELECT count(*) FROM task_papers').fetchone()[0] == 1
    db.close()


def test_membership_empty_is_not_non_nature_without_dated_primary() -> None:
    assert portfolio_label('', [], 2024)[0] == 'unresolved'
    aps = {'id': 'a', 'identity': 'matched', 'family': 'aps', 'online_date': '2024-01-01'}
    assert portfolio_label('', [aps], 2024) == ('False', ['a'])
    assert portfolio_label('', [aps], 2023)[0] == 'unresolved'
    assert portfolio_label('True', [aps], 2024)[0] == 'unresolved'


def test_abstract_not_filled_on_title_or_journal_mismatch() -> None:
    paper = {'title': 'Real title', 'journal_name': 'Journal A'}
    item = {'title': ['Other title'], 'issns': ['1234-5678']}
    assert not signature_match(paper, item, {'1234-5678'})
    assert not abstract_ok('No abstract available' * 10)
    assert not abstract_ok(('Some duplicate content.' * 5) + '\n' + ('Some duplicate content.' * 5))


def test_pilot_rotates_all_field_year_strata(tmp_path: Path) -> None:
    import hashlib
    from collections import Counter

    from experiments.fig8_extension.supplement_queue import select_pilot

    db = connect(tmp_path)
    for field in ['biochemistry', 'materials', 'physics']:
        for year in [2023, 2024, 2025]:
            for n in range(80):
                doi = f'10.1/{field}-{year}-{n}'
                db.execute('INSERT INTO papers VALUES(' + ','.join('?' * 18) + ')',
                    (doi, doi, field, year, f'S{n%4}', 'Journal', 'Title', 'Elsevier', 'elsevier', 'p.parquet',
                     'pending_no_primary_type_date', '[]', 0, 0, 0, 0, 0, hashlib.sha256(doi.encode()).hexdigest()))
    db.commit()
    db.close()
    pilot = select_pilot(tmp_path)
    counts = Counter((r['field'], r['year']) for r in pilot)
    assert len(pilot) == 500 and len(counts) == 9
    assert max(counts.values()) - min(counts.values()) <= 1
    assert pilot == select_pilot(tmp_path)


def test_transient_source_failure_stops_after_three_attempts(tmp_path: Path, monkeypatch) -> None:
    from experiments.fig8_extension import supplement_sources as sources

    (tmp_path / 'fields/qualification').mkdir(parents=True)

    class Response:
        status_code = 503
        url = 'https://pubs.acs.org/doi/10.1/x'

        def __init__(self):
            self.headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    class Session:
        calls = 0

        def get(self, *args, **kwargs):
            self.calls += 1
            return Response()

    session = Session()
    monkeypatch.setattr(sources.LOCAL, 'session', session, raising=False)
    monkeypatch.setattr(sources, 'paced', lambda url: None)
    monkeypatch.setattr(sources.time, 'sleep', lambda value: None)
    result = sources.request({'doi': '10.1/x', 'route': 'acs'}, tmp_path, 'publisher')
    assert result['status'] == 'retry_exhausted' and session.calls == 3
    assert sources.request({'doi': '10.1/x', 'route': 'acs'}, tmp_path, 'publisher')['reused_receipt']
    assert session.calls == 3


def test_fractional_allocation_preserves_unknowns_and_does_not_rollup_ror() -> None:
    from experiments.fig8_extension.supplement_allocation import SCENARIOS, allocate

    author = existing_author()
    author['countries'] = ['US']
    author['institution_ids_or_verified_local_entities'] = ['EducationLabelOnly']
    paper = {'authors_json': json.dumps([author])}
    known = {name: set() for name in SCENARIOS}
    c, u, stats = allocate(paper, 'main_personal_direct', known)
    assert c == {'unknown': 1} and u == {'unknown': 1}
    assert stats['affiliation_structure_unverified_author_records'] == 1
    author['publisher_affiliation_slots'] = [{'institution_ids': ['EducationLabelOnly'], 'identity_status': 'exact', 'country_codes': ['US']}]
    c, u, _ = allocate({'authors_json': json.dumps([author])}, 'main_personal_direct', known)
    assert c == {'US': 1} and u == {'unknown': 1}
    c, _, stats = allocate(paper, 'complete_case', known)
    assert not c and stats['complete_case_excluded_papers'] == 1


def test_raw_primary_affiliation_uses_only_explicit_terminal_country() -> None:
    author = publisher_author()
    author['affiliations'] = [{'text': 'Department of X, Example University, United States', 'countries': [], 'identifiers': [], 'names': []}]
    source = {'authors': [author], 'evidence_receipt': 'raw.xml'}
    result, _ = enrich_authors([existing_author()], source, {}, {'exampleuniversity': [{'id': 'I1', 'country_code': 'US'}]}, {'united states': 'US'})
    assert result[0]['countries'] == ['US']
    assert result[0]['institution_ids_or_verified_local_entities'] == ['I1']
    author['affiliations'][0]['text'] = 'Example University, Boston'
    result, _ = enrich_authors([existing_author()], source, {}, {'exampleuniversity': [{'id': 'I1', 'country_code': 'US'}]}, {'united states': 'US'})
    assert result[0]['countries'] == [] and result[0]['institution_ids_or_verified_local_entities'] == []


def test_collective_only_paper_keeps_unknown_paper_mass() -> None:
    from experiments.fig8_extension.supplement_allocation import SCENARIOS, allocate

    paper = {'authors_json': json.dumps([{'name': 'Collective', 'nonpersonal_verified': True}])}
    known = {name: set() for name in SCENARIOS}
    c, u, stats = allocate(paper, 'main_personal_direct', known)
    assert c == {'unknown': 1} and u == {'unknown': 1}
    assert stats['no_personal_author_denominator_papers'] == 1


def test_official_branch_rule_requires_declared_parent_and_site() -> None:
    from experiments.fig8_extension.supplement_allocation import declared_relationship

    relation = {'H1': {'targets': {'direct_units': {'U1': 1}}, 'countries': ['US'], 'parent_names': {'U1': 'Example University'}}}
    slot = {'institution_ids': ['H1'], 'country_codes': ['US'], 'text': 'Hospital of Example University, United States'}
    assert declared_relationship(slot, 'main_personal_direct', relation) == {'U1': 1}
    assert declared_relationship({**slot, 'text': 'Other joint center'}, 'main_personal_direct', relation) == {}
    assert declared_relationship({**slot, 'country_codes': ['GB']}, 'main_personal_direct', relation) == {}


def test_official_oai_front_namespace_and_bulk_window() -> None:
    from datetime import datetime, timezone

    from experiments.fig8_extension.supplement_oai import (
        parse_oai,
        seconds_until_bulk_window,
    )

    xml = b'''<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><GetRecord><record><metadata>
    <article xmlns="https://jats.nlm.nih.gov/ns/archiving/1.4/" article-type="research-article"><front><journal-meta><issn>1234-5678</issn></journal-meta>
    <article-meta><article-id pub-id-type="doi">10.1/a</article-id><title-group><article-title>A</article-title></title-group></article-meta></front></article>
    </metadata></record></GetRecord></OAI-PMH>'''
    assert parse_oai(xml)['doi'] == ['10.1/a']
    # Thursday noon Eastern is a prohibited bulk window; Saturday is permitted.
    assert seconds_until_bulk_window(datetime(2026, 10, 22, 16, tzinfo=timezone.utc)) > 0
    assert seconds_until_bulk_window(datetime(2026, 10, 24, 16, tzinfo=timezone.utc)) == 0


def test_aaas_and_nas_require_matched_dated_archived_publisher_evidence() -> None:
    for publisher in ['American Association for the Advancement of Science', 'National Academy of Sciences']:
        e = {'id': 'publisher-proof', 'identity': 'matched', 'family': 'archived_jats',
             'online_date': '2024-06-01', 'publishers': [publisher]}
        assert portfolio_label('', [e], 2024) == ('False', ['publisher-proof'])
        assert portfolio_label('', [e], 2023)[0] == 'unresolved'
        assert portfolio_label('', [{**e, 'identity': 'mismatched'}], 2024)[0] == 'unresolved'
        assert portfolio_label('', [{**e, 'family': 'crossref_publisher_deposit'}], 2024)[0] == 'unresolved'
        assert portfolio_label('True', [e], 2024)[0] == 'unresolved'
