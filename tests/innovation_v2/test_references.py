from gear.innovation.contracts import HumanPoint
from gear.innovation.evaluation import Match, score_details
from gear.innovation.references import bind_points, split_blocks


def point(**updates):
    row = {
        "reference_id": "r",
        "contribution_id": "g",
        "paper_id": "p",
        "reviewer_id": "1",
        "round_number": 1,
        "source_block_id": "B0001",
        "source_quote": "A specific contribution.",
        "target_text": "A specific contribution",
        "dimension": "identification",
        "stance": None,
        "reasons": [],
        "tier": "B",
        "version_status": "applicable",
        "exclusion_reason": "",
    }
    row.update(updates)
    return HumanPoint(**row)


def test_round_after_response_reopens_reviewer():
    blocks = split_blocks(
        "Round 1\nReviewer #1\nA specific contribution.\nAuthor response\nReviewer #1\nWe disagree.\nRound 2\nReviewer #1\nImproved."
    )
    assert [b["role"] for b in blocks] == ["reviewer", "author", "author", "reviewer"]
    assert blocks[-1]["round_number"] == 2


def test_unquoted_opinion_cannot_be_reference():
    blocks = [
        {
            "block_id": "B0001",
            "reviewer_id": "1",
            "round_number": 1,
            "role": "reviewer",
            "text": "A specific contribution.",
        }
    ]
    row = bind_points([point(source_quote="invented praise")], blocks, "p")[0]
    assert row.tier == "excluded"


def test_silence_not_negative_and_extra_predictions_not_false_positive():
    matches = [
        Match(
            reference_id="r",
            predicted_claim_id="c",
            scope="same",
            predicted_stance=None,
            reason_statuses=[],
            rationale="same",
        )
    ]
    scores = score_details([point()], matches)
    assert scores["coverage"] == 1
    assert scores["matched_stance_agreement"] is None
    assert "precision" not in scores


def test_review_disagreement_preserved_without_duplicate_coverage():
    refs = [
        point(tier="A", dimension="increment", stance="recognized"),
        point(
            reference_id="r2",
            reviewer_id="2",
            tier="A",
            dimension="increment",
            stance="challenged",
        ),
    ]
    matches = [
        Match(
            reference_id=p.reference_id,
            predicted_claim_id="c",
            scope="same",
            predicted_stance="recognized",
            reason_statuses=[],
            rationale="same",
        )
        for p in refs
    ]
    scores = score_details(refs, matches)
    assert scores["coverage"] == 1 and scores["reference_contribution_count"] == 1
    assert scores["matched_stance_agreement"] == 0.5


def test_grouping_cannot_merge_reviewers_or_dimensions(monkeypatch):
    from gear.config import GearConfig
    from gear.innovation import references

    monkeypatch.setattr(references, "assign_groups", lambda *a: [[0, 1]])
    rows = [
        point(tier="A", dimension="increment", stance="recognized"),
        point(
            reference_id="r2",
            reviewer_id="2",
            round_number=2,
            tier="A",
            dimension="increment",
            stance="challenged",
        ),
    ]
    assert set(references.retain_latest(GearConfig(), rows)) == {"r", "r2"}


def test_later_explicit_opinion_replaces_only_same_reviewers_point(monkeypatch):
    from gear.config import GearConfig
    from gear.innovation import references

    monkeypatch.setattr(references, "assign_groups", lambda *a: [[0, 1]])
    rows = [point(), point(reference_id="r2", round_number=2)]
    assert references.retain_latest(GearConfig(), rows) == ["r2"]


def test_identification_never_receives_review_stance_or_reasons(monkeypatch):
    import json

    from gear.config import GearConfig
    from gear.innovation.contracts import AnalysisResult, Assessment, Finding
    from gear.innovation.evaluation import identify
    from gear.model_client import LazyRoleClient

    captured = {}

    def generate(self, **kwargs):
        captured.update(json.loads(kwargs["user"]))
        return {
            "matches": [
                {
                    "reference_id": "r",
                    "predicted_claim_id": "c",
                    "scope": "same",
                    "rationale": "same contribution",
                }
            ]
        }

    monkeypatch.setattr(LazyRoleClient, "generate_json", generate)
    analysis = Assessment(
        claim_id="c",
        claim_text="A specific contribution",
        supported_scope="same",
        findings=[
            Finding(
                dimension="novelty", text="Uncertain firstness", evidence_keys=["p"]
            )
        ],
        overall_stance="unresolved",
        overall_reason="uncertain",
        limitations=[],
    )
    result = AnalysisResult(
        paper_id="p",
        system="gear",
        claim_fingerprint="x",
        status="complete",
        assessments=[analysis],
    )
    reference = point(
        tier="A",
        dimension="firstness",
        stance="recognized",
        reasons=["secret reviewer reasoning"],
    )
    identity = identify(GearConfig(), result, [reference], "evaluation_judge")
    assert identity["r"].scope == "same"
    text = json.dumps(captured)
    assert "secret reviewer reasoning" not in text and "recognized" not in text
    assert "Uncertain firstness" not in text


def test_nonpublic_cover_sheet_is_auditable_exclusion(tmp_path):
    from gear.config import GearConfig
    from gear.innovation.references import screen

    review = tmp_path / "review.md"
    review.write_text(
        "Previously reviewed at a journal not operating a transparent peer review scheme. "
        "Suitable for publication without further review."
    )
    paper = tmp_path / "paper.md"
    paper.write_text("Manuscript")
    result = screen("paper", review, paper, tmp_path / "screen", GearConfig())
    assert result.status.value == "complete"
    assert not result.retained_ids
    assert (tmp_path / "screen/exclusion.json").exists()


def test_unknown_unsegmented_review_remains_failure(tmp_path):
    from gear.config import GearConfig
    from gear.innovation.references import screen

    review = tmp_path / "review.md"
    review.write_text("Unrecognized headings and potentially substantive reviewer text")
    paper = tmp_path / "paper.md"
    paper.write_text("Manuscript")
    result = screen("paper", review, paper, tmp_path / "screen", GearConfig())
    assert result.status.value == "limited"
    assert not (tmp_path / "screen/exclusion.json").exists()
