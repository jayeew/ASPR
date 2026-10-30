from experiments.fig8_extension.russian_signed_self import classify


def test_explicit_moscow_self_and_documented_physics_faculty():
    assert classify('moscow state university, 119991, moscow, russia') == 'msu'
    assert classify('faculty of physics, moscow state university, 119991, moscow, russia') == 'msu'
    assert classify('national research nuclear university mephi (moscow engineering physics institute), 115409, moscow, russia') == 'mephi'


def test_branches_other_departments_joint_signatures_and_acronyms_are_held():
    for name in [
        'moscow state university, baku branch, baku, azerbaijan',
        'moscow state university, faculty of chemistry, 119991, moscow, russia',
        'faculty of physics, moscow state university, moscow, russia and kurchatov institute',
        'aix marseille univ, ird, ap-hm, mephi, marseille, france',
        'national research nuclear university, moscow, russia',
        'national research nuclear university mephi, tashkent, uzbekistan',
        'moscow state university, 119234, moscow, russia',
    ]:
        assert classify(name) is None
