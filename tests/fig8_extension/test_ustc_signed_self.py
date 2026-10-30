from experiments.fig8_extension.ustc_signed_self import classify


def test_explicit_university_self_signatures():
    assert classify('University of Science and Technology of China') == 'ustc'
    assert classify('University of Science and Technology of China, Hefei 230026, Anhui, China') == 'ustc'
    assert classify('University of Science and Technology of China (USTC), Hefei, China') == 'ustc'


def test_unrelated_universities_branches_and_joint_laboratories_are_held():
    for name in [
        'University of Electronic Science and Technology of China',
        'University of Science and Technology of China, Suzhou, China',
        'University of Science and Technology of China, First Affiliated Hospital, Hefei, China',
        'University of Science and Technology of China and IHEP, Hefei, China',
        'Hefei National Laboratory, University of Science and Technology of China, Hefei, China',
    ]:
        assert classify(name) is None
