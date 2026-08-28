"""Lazy local dual-view scientific retrieval and reranking."""

from __future__ import annotations

import fcntl
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .contracts import RetrievedWork


class LocalScientificRanker:
    """Load BGE models only when a retrieval action actually needs them."""

    def __init__(self, recall_path: Path, reranker_path: Path) -> None:
        self.recall_path = Path(recall_path)
        self.reranker_path = Path(reranker_path)
        self._recall: Any = None
        self._reranker: Any = None
        self._gpu_lease: Any = None

    def rank(
        self,
        works: Sequence[RetrievedWork],
        *,
        whole_paper_view: str,
        purpose_view: str,
        recall_limit: int,
        rerank_top_k: int,
        output_limit: int,
    ) -> tuple[list[RetrievedWork], dict[str, tuple[float, float]]]:
        if not works:
            return [], {}
        documents = [self._document(work) for work in works]
        recall = self._load_recall()
        document_vectors = recall.encode(documents, return_dense=True)["dense_vecs"]
        scores_by_view: list[list[float]] = []
        recalled_ids: set[int] = set()
        for view in (whole_paper_view, purpose_view):
            query_vector = recall.encode([view], return_dense=True)["dense_vecs"][0]
            scores = [float(vector @ query_vector) for vector in document_vectors]
            scores_by_view.append(scores)
            recalled_ids.update(
                sorted(range(len(works)), key=scores.__getitem__, reverse=True)[
                    :recall_limit
                ]
            )
        candidate_ids = sorted(recalled_ids)
        reranker = self._load_reranker()
        reranked_by_view: list[dict[int, float]] = []
        selected_ids: set[int] = set()
        for view in (whole_paper_view, purpose_view):
            pairs = [[view, documents[index]] for index in candidate_ids]
            raw = (
                reranker.compute_score(pairs, normalize=True)
                if hasattr(reranker, "compute_score")
                else reranker.predict(pairs)
            )
            values = (
                [float(raw)]
                if not hasattr(raw, "__iter__") or isinstance(raw, (str, bytes))
                else [float(value) for value in raw]
            )
            score_map = {
                index: float(score) for index, score in zip(candidate_ids, values)
            }
            reranked_by_view.append(score_map)
            selected_ids.update(
                sorted(score_map, key=score_map.__getitem__, reverse=True)[
                    :rerank_top_k
                ]
            )
        ordered_ids = sorted(
            selected_ids,
            key=lambda index: max(
                reranked_by_view[0].get(index, float("-inf")),
                reranked_by_view[1].get(index, float("-inf")),
            ),
            reverse=True,
        )[:output_limit]
        result_scores: dict[str, tuple[float, float]] = {
            works[index].work_id: (
                max(scores_by_view[0][index], scores_by_view[1][index]),
                max(
                    reranked_by_view[0].get(index, float("-inf")),
                    reranked_by_view[1].get(index, float("-inf")),
                ),
            )
            for index in ordered_ids
        }
        return [works[index] for index in ordered_ids], result_scores

    def _load_recall(self) -> Any:
        if self._recall is None:
            if not self.recall_path.is_dir():
                raise FileNotFoundError(self.recall_path)
            import torch
            from FlagEmbedding import BGEM3FlagModel

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required for the local scientific ranker")

            self._acquire_gpu_lease()
            self._recall = BGEM3FlagModel(
                str(self.recall_path), use_fp16=True, devices=["cuda:0"]
            )
            # FlagEmbedding defers its device transfer until the first encode.
            # Make it explicit so the model is never used for CPU inference.
            self._recall.model.to("cuda:0")
            self._recall.model.half()
        return self._recall

    def _load_reranker(self) -> Any:
        if self._reranker is None:
            if not self.reranker_path.is_dir():
                raise FileNotFoundError(self.reranker_path)
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(
                str(self.reranker_path),
                device="cuda:0",
                model_kwargs={"torch_dtype": "float16"},
            )
        return self._reranker

    def _acquire_gpu_lease(self) -> None:
        """Limit model-bearing review processes on a shared CUDA device."""
        if self._gpu_lease is not None:
            return
        slot_count = int(os.environ.get("GEAR_GPU_MAX_PROCESSES", "0"))
        if slot_count <= 0:
            return
        timeout = float(os.environ.get("GEAR_GPU_LEASE_TIMEOUT_SECONDS", "3600"))
        lease_dir = Path(
            os.environ.get("GEAR_GPU_LEASE_DIR", "/tmp/aspr_gear_gpu_leases")
        )
        lease_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        deadline = time.monotonic() + timeout
        while True:
            for slot in range(slot_count):
                handle = (lease_dir / f"cuda0_slot_{slot}.lock").open(
                    "a+", encoding="utf-8"
                )
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    handle.close()
                    continue
                self._gpu_lease = handle
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"GPU lease unavailable after {timeout:.1f}s "
                    f"({slot_count} slots)"
                )
            time.sleep(0.25)

    @staticmethod
    def _document(work: RetrievedWork) -> str:
        return f"{work.title}\n{work.abstract}".strip()


__all__ = ["LocalScientificRanker"]
