from experiments.fig8_extension.cached_secondary_dispatch import archive_pmcid


def test_numeric_pmc_route_does_not_confuse_pubmed_ids() -> None:
    assert archive_pmcid('https://www.ncbi.nlm.nih.gov/pmc/articles/11129960/', 'www.ncbi.nlm.nih.gov') == 'PMC11129960'
    assert archive_pmcid('https://www.ncbi.nlm.nih.gov/pubmed/11129960', 'www.ncbi.nlm.nih.gov') == ''
    assert archive_pmcid('https://example.org/PMC11129960', 'example.org') == ''
    assert archive_pmcid('https://pmc.ncbi.nlm.nih.gov/articles/PMC11129960/', 'pmc.ncbi.nlm.nih.gov') == 'PMC11129960'
