"""Native temporary insertion and union facts, without legacy run wrappers."""
from __future__ import annotations

from typing import Any

from .config import Config
from .retrieval import _LOCAL_LOCK, Retriever


class NativeGraph:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.runtime: Any = None

    def compute(self, data: dict[str, Any], claims: list[dict[str, Any]]) -> dict[str, Any]:
        from gear.claim_attribution import ClaimGraphRuntime
        from gear.innovation.joint_graph import joint_structure
        from gear.review_contracts import GraphClaim, InnovationPaperInput
        with _LOCAL_LOCK:
            if self.runtime is None:
                self.runtime = ClaimGraphRuntime(self.config.graph_assets, self.config.embedding_model, 10, .5)
            runtime = self.runtime
            row = data['paper']
            item = InnovationPaperInput(paper_id=row['paper_id'], paper_path=data['manuscript_path'],
                title=row['title'], doi=row.get('doi'), publication_date=row['publication_date'],
                cutoff_date=data['cutoff'], abstract_text='', abstract_source='fulltext_claim_extraction',
                openalex_work_id=row.get('openalex_work_id'), reference_work_ids=row.get('reference_work_ids', []))
            retriever = Retriever(self.config)
            local = retriever.local()
            # Reuse the retrieval encoder, never load another copy for graph insertion.
            if claims and 'encoder' not in local:
                retriever.recall(claims[0]['normalized_claim_text'],
                                 {'title': row['title'], 'doi': row.get('doi'), 'cutoff': data['cutoff']})
            vectors = local['encoder'].encode([c['normalized_claim_text'] for c in claims],
                       normalize_embeddings=True, convert_to_numpy=True).astype('float32') if claims else []
            cards = [runtime.insert_vector(GraphClaim(claim_id=c['claim_id'], paper_id=row['paper_id'],
                claim_type=c['claim_type'], claim_text=c['normalized_claim_text'],
                source_sentence_ids=c['source_span_ids'], source_sentence_texts=[c['manuscript_quote']]), item, vector)
                for c, vector in zip(claims, vectors)]
            neighbors = {n.claim_id: n for card in cards for n in card.neighbors}
            edges = []
            if neighbors:
                runtime._connections()
                rows = {int(runtime._claim_db.execute('SELECT claim_row FROM claim_nodes WHERE claim_id=?',
                        (n.claim_id,)).fetchone()[0]): n.claim_id for n in neighbors.values()}
                edges = [[rows[a], rows[b]] for a, b in sorted(runtime._neighbor_edges(list(neighbors.values())))]
            return {'cards': [c.model_dump(mode='json') for c in cards], 'joint': joint_structure(cards, edges)}

    def close(self) -> None:
        if self.runtime is not None:
            self.runtime.close()
