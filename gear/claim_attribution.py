"""Abstract Claim extraction and transient insertion into the Claim Graph."""

from __future__ import annotations

import json
import math
import re
import sqlite3
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np

from gear.claim_graph.contracts import InnovationClaimType
from gear.config import GearConfig

from .review_contracts import (
    BranchStatus,
    GraphBranchResult,
    GraphClaim,
    GraphFactCard,
    GraphNeighbor,
    InnovationPaperInput,
    MetricFact,
    NumberedSentence,
)
from gear.artifacts import write_jsonl, write_model
from gear.model_client import LazyRoleClient


SENTENCE_SPLIT = re.compile(r"(?<=[.!?。！？])\s+")
GRAPH_CLAIM_SYSTEM = """Extract 1-5 atomic contribution Claims from the numbered abstract.
Each Claim must describe one method, finding, mechanism, resource, or theory contribution explicitly
supported by the supplied sentences. Exclude background, motivation, generic importance and future
work. Bind every Claim to exact sentence IDs; do not use outside knowledge. Return JSON only."""


def number_abstract(abstract: str, paper_id: str) -> list[NumberedSentence]:
    parts = [part.strip() for part in SENTENCE_SPLIT.split(abstract) if part.strip()]
    return [NumberedSentence(sentence_id=f"{paper_id}::S{i:02d}", text=text) for i, text in enumerate(parts, 1)]


class AbstractClaimExtractor:
    def __init__(self, config: GearConfig) -> None:
        self.client = LazyRoleClient(config, "graph_claim")

    def extract(self, item: InnovationPaperInput) -> list[GraphClaim]:
        sentences = number_abstract(item.abstract_text, item.paper_id)
        raw = self.client.generate_json(
            system=GRAPH_CLAIM_SYSTEM,
            user=json.dumps({"paper_id": item.paper_id, "title": item.title, "sentences": [x.model_dump() for x in sentences]}, ensure_ascii=False),
            response_schema={
                "type": "object",
                "properties": {"claims": {"type": "array", "minItems": 1, "maxItems": 5, "items": {
                    "type": "object", "properties": {
                        "claim_type": {"type": "string", "enum": [x.value for x in InnovationClaimType]},
                        "claim_text": {"type": "string"},
                        "source_sentence_ids": {"type": "array", "items": {"type": "string"}},
                    }, "required": ["claim_type", "claim_text", "source_sentence_ids"], "additionalProperties": False,
                }}},
                "required": ["claims"], "additionalProperties": False,
            },
        )
        sentence_map = {row.sentence_id: row.text for row in sentences}
        output: list[GraphClaim] = []
        for index, value in enumerate(raw.get("claims", [])[:5], 1):
            ids = [str(x) for x in value.get("source_sentence_ids", []) if str(x) in sentence_map]
            text = str(value.get("claim_text", "")).strip()
            if not ids or not text:
                continue
            output.append(GraphClaim(
                claim_id=f"{item.paper_id}::GRAPH::{index:02d}", paper_id=item.paper_id,
                claim_type=InnovationClaimType(str(value.get("claim_type", "FINDING")).upper()),
                claim_text=text, source_sentence_ids=ids,
                source_sentence_texts=[sentence_map[x] for x in ids],
            ))
        if not output:
            raise ValueError("摘要没有产生可绑定的 Graph Claim")
        return output


class ClaimGraphRuntime:
    """Read static assets and compute a target insertion without mutating them."""

    def __init__(self, root: Path, embedding_model: Path, top_k: int = 10) -> None:
        self.root = root
        self.embedding_model = embedding_model
        self.top_k = top_k
        self._model: object | None = None
        self._faiss: object | None = None
        self._faiss_unavailable = False
        self._embedding_matrix: np.ndarray | None = None
        self._claim_db: sqlite3.Connection | None = None
        self._paper_db: sqlite3.Connection | None = None
        self._stats_db: sqlite3.Connection | None = None
        self._claim_texts: dict[str, str] | None = None
        self._paper_id_map: dict[str, str] | None = None
        self._centroids: np.ndarray | None = None
        self._community_rows: dict[int, int] | None = None

    def close(self) -> None:
        for connection in (self._claim_db, self._paper_db, self._stats_db):
            if connection is not None:
                connection.close()

    def insert(self, claim: GraphClaim, item: InnovationPaperInput) -> GraphFactCard:
        vector = self._encode(claim.claim_text)
        neighbors = self._neighbors(vector, item)
        metrics = self._metrics(neighbors, claim.claim_type)
        return GraphFactCard(
            claim=claim, neighbors=neighbors, metrics=metrics,
            community_ids=sorted({x.community_id for x in neighbors if x.community_id is not None}),
            notes=["临时插入未改写历史 Claim Graph。", "Graph 指标是结构事实，不直接构成创新性结论。"],
        )

    def _encode(self, text: str) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(str(self.embedding_model), trust_remote_code=True)
        array = self._model.encode([text], normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(array[0], dtype=np.float32)

    def _connections(self) -> None:
        if self._claim_db is None:
            self._claim_db = sqlite3.connect(self.root / "claim_graph_index.sqlite")
            self._claim_db.row_factory = sqlite3.Row
        if self._paper_db is None:
            self._paper_db = sqlite3.connect(self.root / "paper_graph_index.sqlite")
        stats = self.root / "claim_graph_runtime_statistics.sqlite"
        if self._stats_db is None and stats.exists():
            self._stats_db = sqlite3.connect(stats)

    def _index(self) -> object:
        if self._faiss is None and not self._faiss_unavailable:
            try:
                import faiss
            except ImportError:
                self._faiss_unavailable = True
            else:
                self._faiss = faiss.read_index(str(self.root / "claim_semantic_index.faiss"))
        return self._faiss

    def _semantic_search(self, vector: np.ndarray, requested: int) -> tuple[np.ndarray, np.ndarray]:
        index = self._index()
        if index is not None:
            distances, rows = index.search(vector.reshape(1, -1), requested)
            return distances[0], rows[0]
        if self._embedding_matrix is None:
            self._embedding_matrix = np.load(
                self.root / "claim_embedding_matrix.npy", mmap_mode="r"
            )
        scores = np.empty(len(self._embedding_matrix), dtype=np.float32)
        for start in range(0, len(self._embedding_matrix), 4096):
            end = min(start + 4096, len(self._embedding_matrix))
            scores[start:end] = np.asarray(self._embedding_matrix[start:end]) @ vector
        requested = min(requested, len(scores))
        selected = np.argpartition(scores, len(scores) - requested)[-requested:]
        selected = selected[np.argsort(scores[selected])[::-1]]
        return scores[selected], selected.astype(np.int64)

    def _index_size(self) -> int:
        index = self._index()
        if index is not None:
            return int(index.ntotal)
        if self._embedding_matrix is None:
            self._embedding_matrix = np.load(
                self.root / "claim_embedding_matrix.npy", mmap_mode="r"
            )
        return len(self._embedding_matrix)

    def _neighbors(self, vector: np.ndarray, item: InnovationPaperInput) -> list[GraphNeighbor]:
        self._connections()
        index_size = self._index_size()
        requested = min(500, index_size)
        distances, rows = self._semantic_search(vector, requested)
        while requested < index_size and self._eligible_count(rows, item) < self.top_k:
            requested = min(requested * 2, index_size)
            distances, rows = self._semantic_search(vector, requested)
        output: list[GraphNeighbor] = []
        rank = 0
        for cosine, row_id in zip(distances, rows):
            row = self._claim_db.execute("SELECT * FROM claim_nodes WHERE claim_row = ?", (int(row_id),)).fetchone()
            if row is None or str(row["parent_paper_id"]) == item.paper_id:
                continue
            published = date.fromisoformat(str(row["publication_date"])[:10])
            if published >= item.cutoff_date:
                continue
            claim_row = self._claim_text(str(row["claim_id"]))
            rank += 1
            parent_paper_id = str(row["parent_paper_id"])
            parent_openalex_id = self._parent_openalex_id(parent_paper_id)
            path = self._paper_path(item.reference_work_ids, parent_openalex_id)
            output.append(GraphNeighbor(
                claim_id=str(row["claim_id"]), parent_paper_id=parent_paper_id,
                parent_openalex_work_id=parent_openalex_id,
                claim_type=InnovationClaimType(str(row["claim_type"])),
                claim_text=claim_row, publication_date=published,
                cosine_similarity=float(cosine), semantic_rank=rank,
                community_id=row["community_id"], **path,
            ))
            if rank >= self.top_k:
                break
        return output

    def _eligible_count(self, rows: np.ndarray, item: InnovationPaperInput) -> int:
        count = 0
        for row_id in rows:
            row = self._claim_db.execute("SELECT parent_paper_id,publication_date FROM claim_nodes WHERE claim_row = ?", (int(row_id),)).fetchone()
            if row is None or str(row[0]) == item.paper_id:
                continue
            if date.fromisoformat(str(row[1])[:10]) < item.cutoff_date:
                count += 1
                if count >= self.top_k:
                    return count
        return count

    def _claim_text(self, claim_id: str) -> str:
        if self._claim_texts is None:
            import pandas as pd
            frame = pd.read_parquet(self.root / "claim_nodes.parquet", columns=["claim_id", "claim_text"])
            self._claim_texts = dict(zip(frame["claim_id"].astype(str), frame["claim_text"].astype(str)))
        return self._claim_texts.get(claim_id, "")

    def _parent_openalex_id(self, article_id: str) -> str:
        if self._paper_id_map is None:
            import pandas as pd
            frame = pd.read_parquet(
                self.root / "canonical_target_works.parquet",
                columns=["nature_article_id", "work_id"],
            ).dropna(subset=["nature_article_id", "work_id"])
            self._paper_id_map = dict(zip(
                frame["nature_article_id"].astype(str),
                frame["work_id"].astype(str),
            ))
        return self._paper_id_map.get(article_id, article_id)

    def _paper_path(self, target_refs: list[str], neighbor_paper: str) -> dict[str, object]:
        refs = {self._bare_openalex_id(value) for value in target_refs}
        neighbor_id = self._bare_openalex_id(neighbor_paper)
        direct = neighbor_id in refs
        neighbor_refs = {str(x[0]) for x in self._paper_db.execute("SELECT cited_work_id FROM paper_edges WHERE citing_work_id = ?", (neighbor_id,))}
        shared = refs & neighbor_refs
        salton = len(shared) / math.sqrt(max(len(refs) * len(neighbor_refs), 1))
        two_hop = 0
        for start in range(0, len(target_refs), 800):
            chunk = target_refs[start:start + 800]
            if not chunk:
                continue
            placeholders = ",".join("?" for _ in chunk)
            query = f"SELECT COUNT(*) FROM paper_edges WHERE citing_work_id IN ({placeholders}) AND cited_work_id = ?"
            two_hop += int(self._paper_db.execute(query, (*chunk, neighbor_id)).fetchone()[0])
        return {"direct_citation": direct, "two_hop_path_count": two_hop,
                "shared_reference_count": len(shared), "shared_reference_salton": salton}

    @staticmethod
    def _bare_openalex_id(value: str) -> str:
        return str(value).rstrip("/").rsplit("/", 1)[-1].upper()

    def _metrics(self, neighbors: list[GraphNeighbor], claim_type: InnovationClaimType) -> list[MetricFact]:
        similarities = [x.cosine_similarity for x in neighbors]
        communities = [x.community_id for x in neighbors if x.community_id is not None]
        weights: dict[int, float] = {}
        for neighbor in neighbors:
            if neighbor.community_id is not None:
                weights[neighbor.community_id] = weights.get(neighbor.community_id, 0.0) + max(neighbor.cosine_similarity, 0.0)
        weight_total = sum(weights.values())
        probabilities = {key: value / weight_total for key, value in weights.items()} if weight_total else {}
        component_sizes = self._neighbor_component_sizes(neighbors)
        possible_pairs = len(neighbors) * (len(neighbors) - 1) // 2
        disconnected_pairs = possible_pairs - sum(size * (size - 1) // 2 for size in component_sizes)
        values: dict[str, float | int] = {
            "nearest_prior_similarity": max(similarities, default=0.0),
            "mean_top5_similarity": float(np.mean(similarities[:5])) if similarities else 0.0,
            "effective_community_count": 1.0 / sum(value * value for value in probabilities.values()) if probabilities else 0.0,
            "community_rao_stirling": self._rao(probabilities),
            "first_observed_recent_nature_pair_share": self._new_pair_share(list(probabilities)),
            "community_pair_mean_surprisal": self._pair_surprisal(list(probabilities)),
            "neighbor_induced_density": self._neighbor_density(neighbors),
            "component_merge_count": max(len(component_sizes) - 1, 0),
            "newly_connected_neighbor_pair_count": disconnected_pairs,
            "cross_boundary_weight_share": self._cross_boundary_share(neighbors),
            "direct_citation_neighbor_count": sum(x.direct_citation for x in neighbors),
            "two_hop_neighbor_count": sum(x.two_hop_path_count > 0 for x in neighbors),
            "co_citation_neighbor_count": sum(x.shared_reference_count > 0 for x in neighbors),
            "cross_type_neighbor_count": sum(x.claim_type != claim_type for x in neighbors),
        }
        return [self._metric_fact(name, value, claim_type) for name, value in values.items()]

    def _rao(self, probabilities: dict[int, float]) -> float:
        if len(probabilities) < 2:
            return 0.0
        if self._centroids is None:
            import pandas as pd
            self._centroids = np.load(self.root / "community_centroid_matrix.npy", mmap_mode="r")
            frame = pd.read_parquet(self.root / "community_centroid_index.parquet")
            self._community_rows = dict(zip(frame["community_id"].astype(int), frame["centroid_row"].astype(int)))
        result = 0.0
        ordered = sorted(probabilities)
        for index, left in enumerate(ordered):
            for right in ordered[index + 1:]:
                if left not in self._community_rows or right not in self._community_rows:
                    continue
                similarity = float(np.dot(self._centroids[self._community_rows[left]], self._centroids[self._community_rows[right]]))
                result += 2.0 * probabilities[left] * probabilities[right] * float(np.clip(1.0 - similarity, 0.0, 1.0))
        return result

    def _new_pair_share(self, communities: list[int]) -> float:
        pairs = [(a, b) for i, a in enumerate(sorted(set(communities))) for b in sorted(set(communities))[i + 1:]]
        if not pairs or self._stats_db is None:
            return 0.0
        unseen = sum(self._stats_db.execute("SELECT 1 FROM community_pair_history WHERE community_a = ? AND community_b = ?", pair).fetchone() is None for pair in pairs)
        return unseen / len(pairs)

    def _pair_surprisal(self, communities: list[int]) -> float:
        pairs = [(a, b) for i, a in enumerate(sorted(set(communities))) for b in sorted(set(communities))[i + 1:]]
        if not pairs or self._stats_db is None:
            return 0.0
        counts = []
        for pair in pairs:
            row = self._stats_db.execute("SELECT pair_connector_count,community_a_claim_count,community_b_claim_count,historical_claim_count FROM community_pair_history WHERE community_a = ? AND community_b = ?", pair).fetchone()
            if row and int(row[1]) and int(row[2]) and int(row[3]):
                commonness = (int(row[0]) + 0.5) * int(row[3]) / (int(row[1]) * int(row[2]))
                counts.append(-math.log(commonness))
        return float(np.mean(counts)) if counts else 0.0

    def _neighbor_edges(self, neighbors: list[GraphNeighbor]) -> set[tuple[int, int]]:
        rows = []
        for item in neighbors:
            row = self._claim_db.execute("SELECT claim_row FROM claim_nodes WHERE claim_id = ?", (item.claim_id,)).fetchone()
            if row:
                rows.append(int(row[0]))
        if len(rows) < 2:
            return set()
        placeholders = ",".join("?" for _ in rows)
        query = f"SELECT source_row,target_row FROM semantic_backbone_adjacency WHERE source_row IN ({placeholders}) AND target_row IN ({placeholders})"
        return {tuple(sorted((int(a), int(b)))) for a, b in self._claim_db.execute(query, (*rows, *rows)) if a != b}

    def _neighbor_density(self, neighbors: list[GraphNeighbor]) -> float:
        n = len(neighbors)
        return len(self._neighbor_edges(neighbors)) / (n * (n - 1) / 2) if n > 1 else 0.0

    def _neighbor_component_sizes(self, neighbors: list[GraphNeighbor]) -> list[int]:
        rows = []
        for item in neighbors:
            row = self._claim_db.execute("SELECT claim_row FROM claim_nodes WHERE claim_id = ?", (item.claim_id,)).fetchone()
            if row:
                rows.append(int(row[0]))
        parent = {row: row for row in rows}
        def find(node: int) -> int:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node
        for left, right in self._neighbor_edges(neighbors):
            a, b = find(left), find(right)
            if a != b:
                parent[b] = a
        counts = Counter(find(row) for row in rows)
        return list(counts.values())

    @staticmethod
    def _cross_boundary_share(neighbors: list[GraphNeighbor]) -> float:
        total = sum(max(x.cosine_similarity, 0.0) for x in neighbors)
        if total == 0:
            return 0.0
        dominant = Counter(x.community_id for x in neighbors).most_common(1)[0][0]
        return sum(max(x.cosine_similarity, 0.0) for x in neighbors if x.community_id != dominant) / total

    def _metric_fact(self, name: str, value: float | int, claim_type: InnovationClaimType) -> MetricFact:
        percentile = None
        type_percentile = None
        direction = None
        if self._stats_db is not None:
            row = self._stats_db.execute("SELECT raw_percentile, metric_direction FROM metric_percentiles WHERE reference_scope = 'ALL' AND claim_type = '' AND metric_name = ? ORDER BY ABS(metric_value - ?) LIMIT 1", (name, float(value))).fetchone()
            if row:
                percentile, direction = float(row[0]), str(row[1])
            typed = self._stats_db.execute("SELECT raw_percentile FROM metric_percentiles WHERE reference_scope = 'CLAIM_TYPE' AND claim_type = ? AND metric_name = ? ORDER BY ABS(metric_value - ?) LIMIT 1", (claim_type.value, name, float(value))).fetchone()
            if typed:
                type_percentile = float(typed[0])
        return MetricFact(name=name, value=value, global_percentile=percentile, claim_type_percentile=type_percentile, direction=direction)


def run_graph_branch(item: InnovationPaperInput, output_dir: Path, config: GearConfig,
                     graph_root: Path, embedding_model: Path) -> GraphBranchResult:
    runtime = ClaimGraphRuntime(graph_root, embedding_model)
    extractor = AbstractClaimExtractor(config)
    try:
        return run_graph_branch_shared_runtime(
            item, output_dir, runtime=runtime, extractor=extractor
        )
    finally:
        runtime.close()


def run_graph_branch_shared_runtime(
    item: InnovationPaperInput,
    output_dir: Path,
    *,
    runtime: ClaimGraphRuntime,
    extractor: AbstractClaimExtractor,
) -> GraphBranchResult:
    """Run one virtual insertion while reusing model, FAISS and SQLite handles."""
    branch_dir = output_dir / "graph"
    write_model(output_dir / "innovation_input.json", item)
    try:
        claims = extractor.extract(item)
        cards = [runtime.insert(claim, item) for claim in claims]
        write_jsonl(branch_dir / "graph_claims.jsonl", claims)
        write_jsonl(branch_dir / "graph_fact_cards.jsonl", cards)
        _write_insertion_edges(branch_dir / "graph_insertion_edges.parquet", cards)
        result = GraphBranchResult(paper_id=item.paper_id, status=BranchStatus.COMPLETE, claims=claims, fact_cards=cards)
    except (ImportError, OSError, RuntimeError, ValueError, sqlite3.Error) as exc:
        result = GraphBranchResult(paper_id=item.paper_id, status=BranchStatus.LIMITED, limitations=[str(exc)])
    result.output_files = {"claims": str(branch_dir / "graph_claims.jsonl"), "fact_cards": str(branch_dir / "graph_fact_cards.jsonl"), "insertion_edges": str(branch_dir / "graph_insertion_edges.parquet")}
    write_model(branch_dir / "graph_branch_result.json", result)
    return result


def _write_insertion_edges(path: Path, cards: list[GraphFactCard]) -> None:
    import pandas as pd
    rows = []
    for card in cards:
        for neighbor in card.neighbors:
            rows.append({
                "target_claim_id": card.claim.claim_id,
                "historical_claim_id": neighbor.claim_id,
                "cosine_similarity": neighbor.cosine_similarity,
                "semantic_rank": neighbor.semantic_rank,
                "edge_type": "semantic_and_paper_path" if (
                    neighbor.direct_citation or neighbor.two_hop_path_count or neighbor.shared_reference_count
                ) else "semantic_only",
                "direct_citation": neighbor.direct_citation,
                "two_hop_path_count": neighbor.two_hop_path_count,
                "shared_reference_count": neighbor.shared_reference_count,
                "shared_reference_salton": neighbor.shared_reference_salton,
            })
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
