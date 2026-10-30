from experiments.fig8_extension.postal_country_cleanup import remaining_papers


def test_partial_success_does_not_drop_remaining_country_slots() -> None:
    papers=[{'work_id':'W1','slots':[{'receipt':'one'},{'receipt':'one'}]},
            {'work_id':'W2','slots':[{'receipt':'two'}]},
            {'work_id':'W3','slots':[{'receipt':'three'}]}]
    audit=[{'work_id':'W1','receipt':'one','slots_filled':1},
           {'work_id':'W2','receipt':'two','slots_filled':1},
           {'work_id':'W3','receipt':'three','status':'identity_unconfirmed'}]
    assert remaining_papers(papers,audit)=={'W1','W3'}
