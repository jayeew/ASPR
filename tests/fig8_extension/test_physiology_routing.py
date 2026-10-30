from experiments.fig8_extension.supplement_queue import route


def test_physiology_does_not_use_physical_society_adapter() -> None:
    assert route('American Physical Society', '10.1152/ajpcell.00123.2023') == 'physiology'
    assert route('American Physiological Society') == 'physiology'
    assert route('American Physical Society', '10.1103/physrevb.109.054106') == 'aps'
    assert route('American Physical Society') == 'aps'
