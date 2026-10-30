"""Nonuniversity branches and contradicted publisher links cannot gain university credit."""
from __future__ import annotations

from experiments.fig8_extension.nature_cutoff_review import CutoffHierarchy, campus


def test_usda_or_cas_branch_never_counts_as_university() -> None:
    h=object.__new__(CutoffHierarchy)
    slot={'cutoff_branch':{'status':'verified_nonuniversity_branch'}}
    for scenario in ['direct_units','joint_equal_split','joint_two_partner_sensitivity','academic_only']:
        assert h.targets(slot,scenario)=={}


def test_publisher_conflict_is_quarantined_in_joint_sensitivity() -> None:
    h=object.__new__(CutoffHierarchy)
    slot={'cutoff_branch':{'status':'publisher_assignment_conflict'}}
    assert h.targets(slot,'joint_equal_split')=={}


def test_branch_requires_explicit_same_address_university() -> None:
    assert campus('QB3, Berkeley, CA, USA','QB3') is None
    assert campus('QB3, University of California, Berkeley, USA','QB3')=='https://openalex.org/I95457486'
    assert campus('University of California, Berkeley, USA','QB3') is None
    assert campus('QB3, University of California, Berkeley and University of California, San Francisco','QB3') is None
