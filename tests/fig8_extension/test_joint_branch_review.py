"""Branch attribution must stay publication-address specific."""
from __future__ import annotations

from experiments.fig8_extension.nature_joint_branch_review import (
    ReviewedHierarchy,
    branch,
)


def test_joint_label_and_email_do_not_identify_branch() -> None:
    assert branch('Peking-Tsinghua Center for Life Sciences, Beijing, China', 'I4210160507') is None
    assert branch('California NanoSystems Institute, Los Angeles, USA', 'I142207940') is None
    assert branch('Collaborative Innovation Center of Quantum Matter, Beijing. x@pku.edu.cn', 'I4210127494') is None


def test_same_address_explicit_host_identifies_branch() -> None:
    assert branch('State Key Laboratory of Membrane Biology, School of Life Sciences, Peking University, Beijing',
                  'I4391768054') == 'I20231570'
    assert branch('California NanoSystems Institute, University of California, Los Angeles, USA',
                  'I142207940') == 'I161318765'
    assert branch('California NanoSystems Institute, University of California, Santa Barbara, USA',
                  'I142207940') == 'I154570441'


def test_two_hosts_in_same_address_remain_unresolved() -> None:
    assert branch('Center for Life Sciences, Peking University and Tsinghua University, Beijing', 'I4210160507') is None


def test_same_entity_can_have_different_branch_assignments() -> None:
    h = object.__new__(ReviewedHierarchy)
    a = {'review_entity_id': 'joint', 'branch_review': {'target_id': 'UCLA'}}
    b = {'review_entity_id': 'joint', 'branch_review': {'target_id': 'UCSB'}}
    assert h.targets(a, 'direct_units') == {'UCLA': 1}
    assert h.targets(b, 'joint_equal_split') == {'UCSB': 1}
