"""Verify released assets and reconstruct joint reachability independently."""

from __future__ import annotations

import hashlib
import json
from itertools import combinations
from pathlib import Path

from gear.config import load_config
from gear.model_client import resolve_role_model

from .graph_position import DEST, OUT, ROOT, save
from .validate import protected_check


def main() -> None:
    assets = json.loads((OUT / "graph_file_inventory.json").read_text())
    assert all(
        Path(a["path"]).stat().st_mtime_ns == a["mtime_ns"]
        and Path(a["path"]).stat().st_size == a["bytes"]
        for a in assets
    )
    baseline = protected_check()
    assert baseline["valid"]
    from zipfile import ZipFile

    inputs = {}
    with ZipFile(OUT / "data_bundle.zip") as z:
        for n in ["claims", "groups"]:
            name = f"evaluation/{n}.jsonl"
            contents = (OUT / name).read_bytes()
            assert z.read(name) == contents
            inputs[name] = hashlib.sha256(contents).hexdigest()
    for p in DEST.glob("*/facts.json"):
        f = json.loads(p.read_text())
        s = f["structure"]
        import networkx as nx

        graph = nx.Graph()
        historical = {b for _, b in s["insertion_edges"]}
        graph.add_nodes_from(historical)
        graph.add_edges_from(s["historical_edges"])
        before = {
            tuple(sorted(x))
            for x in combinations(historical, 2)
            if nx.has_path(graph, *x)
        }
        graph.add_edges_from(s["insertion_edges"])
        after = {
            tuple(sorted(x))
            for x in combinations(historical, 2)
            if nx.has_path(graph, *x)
        }
        assert len(after - before) == s["joint_newly_connected_historical_pairs"]
    model = ROOT / "data/models/Qwen3-Embedding-4B"
    save(
        "run_manifest.json",
        {
            "inputs": inputs,
            "asset_manifest_sha256": hashlib.sha256(
                (OUT / "graph_file_inventory.json").read_bytes()
            ).hexdigest(),
            "embedding_files": [
                (p.name, p.stat().st_size, p.stat().st_mtime_ns)
                for p in sorted(model.glob("*"))
                if p.is_file()
            ],
            "role": resolve_role_model(load_config(), "graph_analysis"),
            "policy": {"top_k": 10, "strict_cosine_above": 0.5, "strict_date": True},
            "GEAR_calls": 0,
            "fusion_calls": 0,
            "external_retrieval_calls": 0,
            "protected": baseline,
            "checked_asset_files": len(assets),
            "pair_reconstruction": True,
            "completed_model_tasks": len(list(DEST.glob("*/analysis.json"))),
        },
    )


if __name__ == "__main__":
    main()
