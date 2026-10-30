from experiments.fig8_extension.pmc_cloud_metadata import assess_version


def example() -> tuple[dict, dict]:
    paper = {'pmcid': 'PMC1', 'doi': '10.1/example', 'title': 'Paper title'}
    metadata = {**paper, 'is_retracted': False, 'is_manuscript': False,
                'is_historical_ocr': False, 'is_pmc_openaccess': True,
                'license_code': 'CC BY', 'version': 2,
                'xml_url': 's3://pmc-oa-opendata/PMC1.2/PMC1.2.xml?md5=' + 'a' * 32}
    return metadata, paper


def test_final_version_requires_jats_identity_check() -> None:
    metadata, paper = example()
    result = assess_version(metadata, paper)
    assert result['status'] == 'published_xml_candidate_identity_check_required'
    assert result['qualification_implied'] is False


def test_manuscript_or_retraction_not_promoted() -> None:
    metadata, paper = example()
    assert assess_version({**metadata, 'is_manuscript': True}, paper)['status'].startswith('author_manuscript')
    assert assess_version({**metadata, 'is_retracted': True}, paper)['status'].startswith('retracted')
    del metadata['is_manuscript']
    assert assess_version(metadata, paper)['status'] == 'publication_version_unconfirmed'


def test_wrong_doi_or_xml_version_retained() -> None:
    metadata, paper = example()
    assert assess_version({**metadata, 'doi': '10.1/other'}, paper)['status'] == 'metadata_identity_mismatch'
    assert assess_version({**metadata, 'version': 1}, paper)['status'] == 'xml_location_identity_conflict'


def test_explicit_oa_without_license_code_can_supply_metadata() -> None:
    metadata, paper = example()
    metadata['license_code'] = None
    result = assess_version(metadata, paper)
    assert result['status'] == 'published_xml_candidate_identity_check_required'
    assert result['license_code'] is None
    assert result['license_code_status'] == 'unspecified_preserve_article_permissions'
    assert result['qualification_implied'] is False
    for oa in (False, None):
        assert assess_version({**metadata, 'is_pmc_openaccess': oa}, paper)['status'] == 'distribution_metadata_unconfirmed'
