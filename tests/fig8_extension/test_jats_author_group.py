from experiments.fig8_extension.supplement_sources import jats_metadata


def test_explicit_author_group_recovers_missing_attributes_but_not_editors():
    raw = b'''<article><front><journal-meta><journal-title-group><journal-title>Example</journal-title></journal-title-group></journal-meta><article-meta>
    <contrib-group content-type="author"><contrib><name><surname>Doe</surname><given-names>Jane</given-names></name><xref ref-type="aff" rid="a1"/></contrib>
    <contrib contrib-type="editor"><name><surname>Editor</surname></name></contrib></contrib-group>
    <contrib-group><contrib><name><surname>Unspecified</surname></name></contrib></contrib-group>
    <aff id="a1">Institute, France</aff></article-meta></front></article>'''
    result = jats_metadata(raw)
    assert [a['name'] for a in result['authors']] == ['Jane Doe']
    assert result['authors'][0]['affiliations'][0]['text']=='Institute, France'
    assert not result['authors'][0]['unresolved_affiliation_refs']


def test_single_group_affiliation_applies_only_without_competing_links() -> None:
    from lxml import etree
    from experiments.fig8_extension.supplement_sources import shared_group_affiliation
    group = etree.fromstring(b'''<contrib-group>
        <contrib id="a"/><contrib id="b"/>
        <aff id="one"><institution>University</institution>, <country>Poland</country></aff>
    </contrib-group>''')
    for author in group.findall('contrib'):
        assert shared_group_affiliation(author)['names'] == ['University']
    etree.SubElement(group[0], 'xref', {'ref-type': 'aff', 'rid': 'one'})
    assert shared_group_affiliation(group[1]) is None
    group[0].remove(group[0][0])
    etree.SubElement(group, 'aff').text = 'Another institution'
    assert shared_group_affiliation(group[0]) is None


def test_group_affiliation_does_not_leak_to_other_groups() -> None:
    raw = b'''<article><front><journal-meta/><article-meta>
      <contrib-group><contrib contrib-type="author"><name><surname>A</surname></name></contrib>
      <aff><institution>Institute A</institution></aff></contrib-group>
      <contrib-group><contrib contrib-type="author"><name><surname>B</surname></name></contrib></contrib-group>
      </article-meta></front></article>'''
    authors = jats_metadata(raw)['authors']
    assert authors[0]['affiliations'][0]['text'] == 'Institute A'
    assert authors[1]['affiliations'] == []
