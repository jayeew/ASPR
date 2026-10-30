from experiments.fig8_extension.primary_country_resolution import index_authors, resolve


def test_primary_agreement_supersedes_only_lower_authority():
    a={'name':'Alice Li','country_source':'openalex_authorship_countries','countries':['US'],'country_conflict_retained':True,
       'publisher_affiliation_slots':[{'country_codes':['CA']}],'position':0}
    s={'authors':[{'name':'Alice Li','unresolved_affiliation_refs':[],'affiliations':[{'text':'University, Canada','countries':[]}]}],'evidence_receipt':'front.xml'}
    value,receipts=resolve(a,[s],{'canada':'CA'})
    assert value['countries']==['CA'] and not value['country_conflict_retained']
    assert receipts==['front.xml']
    assert resolve(a,[index_authors(s)],{'canada':'CA'})==(value,receipts)
    assert resolve(a,[index_authors({**s,'authors':s['authors']*2})],{'canada':'CA'})[1]==[]
    assert resolve({**a,'nature_v1_slots':[{}]},[s],{'canada':'CA'})[1]==[]
    assert resolve({**a,'country_source':'publisher_explicit_affiliation_countries'},[s],{'canada':'CA'})[1]==[]
    bad={'authors':[{'name':'Alice Li','unresolved_affiliation_refs':['missing'],'affiliations':[{'text':'University, Canada','countries':[]}]}],'evidence_receipt':'other.xml'}
    assert resolve(a,[s,bad],{'canada':'CA'})[1]==[]
    conflicting={'authors':[{'name':'Alice Li','unresolved_affiliation_refs':[],'affiliations':[{'text':'University, Japan','countries':[]}]}],'evidence_receipt':'other.xml'}
    assert resolve(a,[s,conflicting],{'canada':'CA','japan':'JP'})[1]==[]


def test_secondary_only_conflict_requires_agreeing_complete_primary_addresses():
    author = {'name': 'Alice Li', 'country_source': 'openalex_authorship_countries',
              'countries': ['US'], 'secondary_country_conflict': True,
              'secondary_country_observations': ['US'], 'position': 0,
              'publisher_affiliation_slots': [{'country_codes': ['CA']}]}
    source = {'authors': [{'name': 'Alice Li', 'unresolved_affiliation_refs': [],
                          'affiliations': [{'text': 'University, Canada', 'countries': []}]}],
              'evidence_receipt': 'front.xml'}
    updated, receipts = resolve(author, [index_authors(source)], {'canada': 'CA'})
    assert receipts == ['front.xml'] and updated['countries'] == ['CA']
    assert updated['secondary_country_conflict'] is False
    assert updated['secondary_country_observations'] == ['US']
    assert resolve(updated, [source], {'canada': 'CA'}) == (updated, [])
    assert resolve(author, [], {'canada': 'CA'}) == (author, [])
    assert resolve({**author, 'nature_v1_slots': []}, [source], {'canada': 'CA'})[1] == []
    assert resolve({**author, 'publisher_affiliation_slots': [{'country_codes': []}]},
                   [source], {'canada': 'CA'})[1] == []


def test_individual_html_links_do_not_certify_whole_byline():
    from experiments.fig8_extension.springer_author_cache import parse_country_authors
    markup = '''<meta name="citation_author" content="Li, Alice">
    <ul data-test="authors-list">
      <li><a data-test="author-name" data-author-popup="group-1" href="#group-1">The Collaboration</a></li>
      <li><a data-test="author-name">Alice Li</a><sup><a href="#Aff1">1</a></sup></li>
      <li><a data-test="author-name">Bob Jr.</a><sup><a href="#Aff9">9</a></sup></li>
    </ul><li id="Aff1"><p class="c-article-author-affiliation__address">University, Canada</p></li>'''
    source = {**parse_country_authors(markup), 'evidence_receipt': 'page.html'}
    assert source['byline_has_etal'] is True
    assert [a['country_link_certified'] for a in source['authors']] == [False, True, False]
    author = {'name': 'Alice Li', 'country_source': 'openalex_authorship_countries',
              'countries': ['US'], 'secondary_country_conflict': True,
              'publisher_affiliation_slots': [{'country_codes': ['CA']}]}
    assert resolve(author, [index_authors(source)], {'canada': 'CA'})[1] == ['page.html']
    duplicate = markup.replace('</ul>', '<li><a data-test="author-name">Alice Li</a></li></ul>')
    ambiguous = {**parse_country_authors(duplicate), 'evidence_receipt': 'page.html'}
    assert resolve(author, [index_authors(ambiguous)], {'canada': 'CA'})[1] == []


def test_only_explicit_present_address_note_is_excluded_and_retained():
    from experiments.fig8_extension.springer_author_cache import parse_country_authors
    markup = '''<ul data-test="authors-list"><li><a data-test="author-name">Alice Li</a>
      <sup><a href="#Aff1">1</a><a href="#nAff2">2</a></sup></li></ul>
      <li id="Aff1"><p class="c-article-author-affiliation__address">University, Canada</p></li>
      <li id="nAff2"><p class="js-present-address">Present address: University, USA</p></li>'''
    source = {**parse_country_authors(markup), 'evidence_receipt': 'page.html'}
    author = {'name': 'Alice Li', 'country_source': 'openalex_authorship_countries',
              'countries': ['CA'], 'secondary_country_conflict': True,
              'publisher_affiliation_slots': [{'country_codes': ['CA']}]}
    updated, receipts = resolve(author, [source], {'canada': 'CA', 'usa': 'US'})
    assert receipts == ['page.html'] and updated['countries'] == ['CA']
    assert updated['country_resolution_excluded_present_addresses'][0]['reference'] == '#nAff2'
    unlabelled = {**parse_country_authors(markup.replace('js-present-address', 'unclassified-note')), 'evidence_receipt': 'page.html'}
    assert resolve(author, [unlabelled], {'canada': 'CA', 'usa': 'US'})[1] == []
