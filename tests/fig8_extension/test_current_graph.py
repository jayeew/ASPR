from datetime import date

import pytest

from gear.review_contracts import GraphClaim, GraphFactCard, GraphNeighbor
from experiments.fig8_extension.current_graph import joint_metrics


def card(neighbors: list[str]) -> GraphFactCard:
    return GraphFactCard(claim=GraphClaim(claim_id='new::C1',paper_id='new',
        claim_type='FINDING',claim_text='A supported finding.',
        source_sentence_ids=['S001'],source_sentence_texts=['A supported finding.']),
        neighbors=[GraphNeighbor(claim_id=n,parent_paper_id='old',claim_text=n,
            claim_type='FINDING',publication_date=date(2024,1,1),
            cosine_similarity=.8,semantic_rank=i+1) for i,n in enumerate(neighbors)],metrics=[])


def test_joint_bridge_changes_only_historical_node_pairs() -> None:
    metrics,facts=joint_metrics([card(['a','b'])],[])
    assert metrics['component_merge_count']==1
    assert metrics['newly_connected_pair_ratio']==1
    assert metrics['efficiency_change']==pytest.approx(.5)
    assert facts['insertion_edges']==[['new::C1','a'],['new::C1','b']]


def test_no_neighbors_has_no_computable_pair_metrics() -> None:
    metrics,_=joint_metrics([card([])],[])
    assert metrics['graph_status']=='no_neighbors'
    assert metrics['neighbor_match_coverage']==0
    assert metrics['newly_connected_pair_ratio'] is None
    assert metrics['efficiency_change'] is None
