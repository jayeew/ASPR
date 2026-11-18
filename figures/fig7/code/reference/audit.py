"""Read-only inventory of native assets used by the Fig7 forecast."""
from __future__ import annotations

from .data import compact, load_graph


def main() -> None:
    nodes, edges = load_graph()
    print(compact({
        'claims': len(nodes),
        'papers': nodes.parent_paper_id.nunique(),
        'eligible_edges': len(edges),
        'communities': nodes.community_id.nunique(),
        'unassigned_claims': int(nodes.community_id.isna().sum()),
        'material_groups': nodes.group.nunique(),
        'cutoff': nodes.date.max(),
        'missing_source_fragments': int(nodes.source_fragments.apply(len).eq(0).sum()),
    }))


if __name__ == '__main__':
    main()
