from experiments.fig8_extension.mdpi_targeted_scope import identifier


def test_only_native_publisher_article_paths_supply_identifiers() -> None:
    assert identifier('https://www.mdpi.com/1422-0067/24/1/768/pdf?version=1672899391') == 'oai:mdpi.com:/1422-0067/24/1/768/'
    assert identifier('https://doi.org/10.3390/ijms24010768') is None
    assert identifier('https://www.mdpi.com.example.org/1422-0067/24/1/768') is None
    assert identifier('https://www.mdpi.com/1422-0067/24/1') is None
    assert identifier('https://www.mdpi.com/journal/ijms') is None
