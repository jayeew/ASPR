from experiments.fig8_extension.fig8b_qualification_decide import journal_research_format


def test_horticulture_research_report_is_journal_and_source_bound() -> None:
    e=dict(identity='matched',family='publisher_springer',kind='Research Report',
           journal_titles=['Horticulture, Environment, and Biotechnology'],issns=['2211-3460'],
           type_policy_receipt='supplement/target_500k/horticulture_reports/policy.json',
           categories=['Research Report','Report'],title='Cucumber genotype experiment')
    assert journal_research_format(e)
    for changes in [dict(kind='Report'),dict(journal_titles=['Another journal']),dict(type_policy_receipt=''),dict(categories=['Case Report']),dict(title='Study protocol for cucumber experiments')]:
        assert not journal_research_format({**e,**changes})
