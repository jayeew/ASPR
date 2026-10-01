"""Claim-scoped GEAR scientific judgments through bounded independent CLI steps."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from . import models as m
from .evidence import EvidenceStore
from .materials import batches, catalog, packet

if TYPE_CHECKING:
    from .stages import Stages


async def analyze_claim(stages: Stages, claim: dict[str, Any], index: int) -> dict[str, Any]:
    from gear.innovation.analysis import COMMON, GEAR
    from gear.innovation.contracts import Assessment
    from gear.prior_art import (
        BATCH_RELATION_CLASSIFICATION_PROMPT,
        DIRECT_ANTECEDENT_VERIFICATION_PROMPT,
    )
    name = f'gear_{index:02d}'
    manuscript = stages.manuscript_material(claim, 6500)
    internal = await stages.ask('baseline', {'claim': claim, **manuscript}, m.ManuscriptSupport, name+'_internal',
        'Assess the actual manuscript methods/results supporting this claim. Author assertion is not proof. '
        'Preserve narrower supported_scope, exact source quotes and limitations. Missing selected passages '
        'mean unresolved, not internally_unsupported. Use internally_unsupported only with explicit contradictory evidence.')
    sources, relations, failures = [], [], []
    if internal['status'] != 'internally_unsupported':
        retrieved = await stages.search(name, {'claim': claim, 'supported_scope': internal['supported_scope'],
            'query_instruction': 'Include normal scientific comparisons and a contrastive search for existing methods '
                                 'already achieving this result, not only the target terminology.'})
        sources, failures = retrieved['sources'], retrieved['failures']
        # Read each selected work separately; keep one work's evidence out of other relation labels.
        async def classify(group: list[Any], i: int) -> Any:
            priors = [{'work_id': s['source_id'], **packet(catalog([s], 1500), claim, 3500)} for s in group]
            return await stages.ask('baseline', {'target_claim': claim, 'manuscript_support': internal,
                'prior_works': priors}, m.HistoricalRelations, f'{name}_relations_{i:05d}', BATCH_RELATION_CLASSIFICATION_PROMPT)
        rows = await stages.map(name+'_历史关系', batches(sources, 2, 60000), classify)
        relations = [r for value in rows for r in value['relations']]
        antecedents = [r for r in relations if r['relation'] == 'DIRECT_ANTECEDENT']
        async def verify(relation: Any, i: int) -> Any:
            source = [s for s in sources if s['source_id'] == relation['work_id']]
            return await stages.ask('baseline', {'target_claim': claim, 'supported_scope': internal['supported_scope'],
                **packet(catalog(source, 1500), claim, 6000)}, m.AntecedentVerification,
                f'{name}_antecedent_{i:05d}', DIRECT_ANTECEDENT_VERIFICATION_PROMPT)
        checks = await stages.map(name+'_先例独立判断', antecedents, verify)
        for relation, check in zip(antecedents, checks):
            relation['independent_verification_passed'] = check['confirmed']
            relation['verification'] = check
    evidence = {'manuscript_support': internal, 'historical_relations': relations, 'retrieval_failures': failures}
    EvidenceStore(stages.config.output/'evidence/gear'/stages.ident/name).add('GEAR:'+claim['claim_id'], 'gear_claim_card', evidence)
    result = await stages.ask('baseline', {'claim_id': claim['claim_id'], 'claim_text': claim['normalized_claim_text'],
        'sources': {'GEAR:'+claim['claim_id']: evidence}}, Assessment, name+'_assessment', COMMON+'\n'+GEAR, True)
    result.update(claim_id=claim['claim_id'], claim_text=claim['normalized_claim_text'])
    return {'analysis': result, 'sources': sources}
