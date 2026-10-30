"""Conservative, portable original-research eligibility decisions from explicit evidence."""
from __future__ import annotations

import hashlib
import re
from datetime import date

NONPRIMARY = {
    'news','newsviews','newsfeature','researchhighlight','review','reviewarticle',
    'comment','commentary','articlecommentary','editorial','correspondence','correction',
    'authorcorrection','publishercorrection','retraction','retractionnote','obituary',
    'careercolumn','booksarts','bookreview','worldview','careers','newsqanda','newsandviews',
    'researchbriefing','publishedErratum'.lower(),'erratum','systematicreview','metaanalysis',
    'perspective','perspectives','mattersarising','mattersarisingreply','reply',
    'meetingreport','interview','newsandcomment','researchsummary',
    'researchhighlights','feature','technologyfeature','careerfeature','newsinbrief',
    'researchwatch','clinicaltrialwatch','yearinreview','lettertotheeditor',
    'journalclub','naturebriefing','naturepodcast','naturevideo','podcast','naturecareerspodcast',
    'newsviewsforum','newsexplainer','careerqanda','careerguide','qanda','futures',
    'synopsis','viewpoint','focus','nobel','essay',
}
SPECIAL = {'protocol','protocolextension','studyprotocol','datadescriptor','datapaper',
           'casereport','casereports','registeredreport','analysis','briefcommunication',
           'briefreport','letter','technicalreport','resource','preregisteredreport'}


def key(value: str | None) -> str:
    return ''.join(c for c in (value or '').casefold() if c.isalnum())



NLM_EXPLICIT_STUDY_TYPES = {
    'randomizedcontrolledtrial', 'controlledclinicaltrial', 'clinicaltrial',
    'clinicaltrialphasei', 'clinicaltrialphaseii', 'clinicaltrialphaseiii',
    'clinicaltrialphaseiv', 'observationalstudy', 'comparativestudy',
    'evaluationstudy', 'clinicalstudy', 'validationstudy', 'multicenterstudy',
    'adaptiveclinicaltrial', 'pragmaticclinicaltrial', 'equivalencetrial',
    'clinicaltrialveterinary', 'randomizedcontrolledtrialveterinary',
    'observationalstudyveterinary', 'twinstudy',
}


def nlm_indexed_research(evidence: dict) -> bool:
    """Human-authorized MED alternative; never a generic Journal Article inference."""
    labels = {key(v) for v in evidence.get('categories', [])}
    excluded = NONPRIMARY | SPECIAL | {
        'clinicaltrialprotocol', 'dataset', 'datanote', 'datadescriptor',
        'resourceannouncement', 'retractedpublication', 'expressionofconcern',
        'duplicatepublication', 'casestudy',
    }
    return (
        evidence.get('family') == 'indexed_nlm_research'
        and evidence.get('identity') == 'matched'
        and evidence.get('index_source') == 'MED'
        and evidence.get('review_rule') == 'user_authorized_nlm_explicit_study_20261026'
        and str(evidence.get('pmid', '')).isdigit()
        and bool(evidence.get('raw_sha256')) and bool(evidence.get('record_sha256'))
        and bool(evidence.get('issns')) and bool(labels & NLM_EXPLICIT_STUDY_TYPES)
        and not labels & excluded
        and not evidence.get('scope_review_required')
        and not evidence.get('title_scope_flag')
        and not re.search(
            r'\b(protocol|case reports?|systematic review|meta.analysis|scoping review|'
            r'narrative review|data note|data descriptor|resource announcement)\b',
            evidence.get('title', ''), re.I)
    )


def virta_original_research(evidence: dict) -> bool:
    """Explicit original-article classification from the approved public VIRTA collection."""
    attributes = evidence.get('virta_attributes', {})
    expected = {'MuotoKoodi': '0', 'YleisoKoodi': '0',
                'EmojulkaisuntyyppiKoodi': '0', 'ArtikkelityyppiKoodi': '1',
                'VertaisarvioituKytkin': '1'}
    return (evidence.get('family') == 'institution_virta'
            and evidence.get('identity') == 'matched' and evidence.get('kind') == 'A1'
            and evidence.get('source_record_status') == '2'
            and evidence.get('type_policy_receipt') == '../../supplement/target_500k/virta_public/rule.json'
            and bool(evidence.get('record_sha256')) and bool(evidence.get('raw_sha256'))
            and bool(evidence.get('issns')) and bool(evidence.get('source_record_id'))
            and all(str(attributes.get(k, '')) in {'', value} for k, value in expected.items())
            and not evidence.get('scope_review_required') and not evidence.get('title_scope_flag'))


def ghent_original_research(evidence: dict) -> bool:
    """Explicit original subtype in the public institutional bibliography; never A1 alone."""
    return (evidence.get('family') == 'institution_ghent'
            and evidence.get('identity') == 'matched' and evidence.get('kind') == 'original'
            and evidence.get('source_publication_type') == 'journalArticle'
            and evidence.get('source_classification') in {'A1','A2','A3'}
            and evidence.get('source_wos_type') in {'','Article'}
            and evidence.get('source_publication_status') == 'published'
            and evidence.get('type_policy_receipt') == '../../supplement/target_500k/ghent_public/rule.json'
            and bool(evidence.get('record_sha256')) and bool(evidence.get('raw_sha256'))
            and bool(evidence.get('issns')) and bool(evidence.get('source_record_id'))
            and not evidence.get('scope_review_required') and not evidence.get('title_scope_flag'))


def public_institution_original_research(evidence: dict) -> bool:
    """Source-defined original subtypes, with curation and exact bibliographic identity."""
    common=(evidence.get('identity')=='matched' and bool(evidence.get('source_record_id'))
            and bool(evidence.get('raw_sha256')) and bool(evidence.get('record_sha256'))
            and not evidence.get('scope_review_required') and not evidence.get('title_scope_flag'))
    if not common:return False
    if evidence.get('family')=='institution_muenster':
        return (evidence.get('kind')=='Research article (journal)'
                and evidence.get('source_peer_reviewed') is True
                and evidence.get('source_publication_status')=='Published'
                and bool(evidence.get('journal_titles'))
                and evidence.get('type_policy_receipt')=='../../supplement/target_500k/muenster_public/rule.json')
    if evidence.get('family')=='institution_slovenia':
        return (evidence.get('kind')=='1.01 Original Scientific Article'
                and evidence.get('source_typology_code')=='1.01'
                and evidence.get('source_approved') is True and evidence.get('source_peer_reviewed') is True
                and evidence.get('source_publication_status')=='Objavljeno'
                and evidence.get('source_container_type')=='Revija' and bool(evidence.get('issns'))
                and evidence.get('type_policy_receipt')=='../../supplement/target_500k/slovenia_public/rule.json')
    return False


def springer_special_originalpaper(evidence: dict) -> bool:
    """A generic technical OriginalPaper tag cannot settle a special format."""
    labels = {key(v) for v in evidence.get('categories', [])}
    return (evidence.get('identity') == 'matched'
            and evidence.get('family') == 'publisher_springer'
            and 'originalpaper' in labels
            and (bool(labels & {'addendum', 'guideline', 'guidelines', 'images',
                               'neuroimages', 'caseimageincardiovascularultrasound',
                               'award', 'highlight'})
                 or bool(re.match(r'^addendum\s*(?:to\b|:)', evidence.get('title', ''), re.I))))


def aso_visual_abstract(evidence: dict) -> bool:
    """ASO's separately published companion abstracts, not the original article."""
    labels = {key(evidence.get('kind'))} | {key(v) for v in evidence.get('categories', [])}
    return (evidence.get('identity') == 'matched'
            and evidence.get('family') == 'publisher_springer'
            and evidence.get('class') in {'unresolved_type', 'nonprimary'}
            and not evidence.get('scope_review_required')
            and 'Annals of Surgical Oncology' in evidence.get('journal_titles', [])
            and bool(set(evidence.get('issns', [])) & {'1068-9265', '1534-4681'})
            and 'asovisualabstract' in labels and labels <= {'asovisualabstract', 'abstract'}
            and key(evidence.get('title')).startswith('asovisualabstract'))


def nature_reviews_tools_trade(evidence: dict) -> bool:
    """Reviewed non-primary format; policy and native checks in nature_tools_trade_policy."""
    journals = {
        'https://openalex.org/S160464432', 'https://openalex.org/S196724125',
        'https://openalex.org/S93373720', 'https://openalex.org/S62674001',
        'https://openalex.org/S8553189', 'https://openalex.org/S4210234670',
        'https://openalex.org/S26843219', 'https://openalex.org/S18618073',
        'https://openalex.org/S5407042882', 'https://openalex.org/S4210178125',
        'https://openalex.org/S4210174246',
    }
    return (evidence.get('identity') == 'matched'
            and evidence.get('family') in {'nature_catalogue', 'nature_page'}
            and (evidence.get('source_id') in journals
                 or (not evidence.get('source_id')
                     and '3005-0685' in evidence.get('issns', [])
                     and 'Nature Reviews Clean Technology' in evidence.get('journal_titles', [])))
            and key(evidence.get('kind')) == 'toolsofthetrade'
            and evidence.get('class') in {'unresolved_type', 'nonprimary'}
            and not evidence.get('scope_review_required')
            and not ({key(v) for v in evidence.get('categories', [])}
                     & {'researcharticle', 'originalresearch', 'originalarticle'}))


def literal_nonresearch_alias(evidence: dict) -> bool:
    """Known literal editorial/commentary aliases; preserve holds and other classes."""
    if (evidence.get('identity') != 'matched' or evidence.get('class') != 'unresolved_type'
            or evidence.get('scope_review_required')):
        return False
    kind = evidence.get('kind', '')
    return (aso_visual_abstract(evidence) or native_nonresearch_format(evidence)
            or aso_author_reflections(evidence)
            or laban_in_brief(evidence)
            or nature_reviews_tools_trade(evidence)
            or (key(kind) == 'opinion' and evidence.get('family') in {'publisher_mdpi_oai', 'publisher_frontiers'})
            or (key(kind) == 'qa' and '&' in kind)
            or (evidence.get('family') in {'nature_catalogue', 'nature_page'}
                and (key(kind) in {'reviewpaper', 'productnews'}
                     or (key(kind) in {'careerqa', 'newsqa'} and '&' in kind)))
            or (evidence.get('family') == 'archived_jats' and key(kind) == 'abstract')
            or (evidence.get('family') in {'nature_catalogue', 'nature_page'}
                and key(kind) == 'whereiwork'
                and evidence.get('source_id') == 'https://openalex.org/S137773608')
            or (evidence.get('family') in {'nature_catalogue', 'nature_page'}
                and key(kind) == 'primeview'
                and evidence.get('source_id') in {
                    'https://openalex.org/S4210231269',
                    'https://openalex.org/S4210201506'})
            or (evidence.get('family') == 'publisher_springer'
                and key(kind) in {'editorialnotes', 'invitedcommentary'}))


def aso_author_reflections(evidence: dict) -> bool:
    """SSO author instructions define this named format as companion commentary."""
    labels = {key(v) for v in evidence.get('categories', [])}
    return (evidence.get('identity') == 'matched'
            and evidence.get('family') == 'publisher_springer'
            and evidence.get('class') in {'unresolved_type', 'nonprimary'}
            and not evidence.get('scope_review_required')
            and key(evidence.get('kind')) == 'asoauthorreflections'
            and key(evidence.get('title')).startswith('asoauthorreflections')
            and evidence.get('journal_titles') == ['Annals of Surgical Oncology']
            and bool(set(evidence.get('issns', [])) & {'1068-9265', '1534-4681'})
            and labels <= {'asoauthorreflections', 'briefcommunication'})


def laban_in_brief(evidence: dict) -> bool:
    """Lab Animal's summaries of other papers, distinct from Brief Communication."""
    return (evidence.get('identity') == 'matched'
            and evidence.get('family') == 'nature_page'
            and key(evidence.get('kind')) == 'inbrief'
            and 'Lab Animal' in evidence.get('journal_titles', [])
            and bool(set(evidence.get('issns', [])) & {'1548-4475', '0093-7355'})
            and evidence.get('class') in {'unresolved_type', 'nonprimary'}
            and not evidence.get('scope_review_required')
            and not ({key(v) for v in evidence.get('categories', [])}
                     & {'article', 'researcharticle', 'originalresearch', 'originalarticle'}))


def native_nonresearch_format(evidence: dict) -> bool:
    """Explicit JATS commentary/report types and Polymer Journal review formats."""
    if (evidence.get('identity') != 'matched' or evidence.get('scope_review_required')
            or evidence.get('class') not in {'unresolved_type', 'nonprimary'}):
        return False
    kind = key(evidence.get('kind'))
    if evidence.get('family') == 'archived_jats':
        if kind == 'introduction':
            return not ({key(v) for v in evidence.get('categories', [])} & {'researcharticle', 'originalarticle', 'originalresearch'})
        return kind in {'discussion', 'inbrief', 'reviewerreport', 'editorreport', 'communitycomment'}
    return (evidence.get('family') == 'nature_page'
            and 'Polymer Journal' in evidence.get('journal_titles', [])
            and bool(set(evidence.get('issns', [])) & {'0032-3896', '1349-0540'})
            and kind in {'reviewpaper', 'focusreview'})


def type_class(kind: str, family: str, categories: list[str] | None = None) -> str:
    normalized=key(kind)
    if normalized == 'qa' and '&' in kind:
        normalized = 'qanda'
    if family in {'nature_catalogue', 'nature_page'}:
        if normalized in {'reviewpaper', 'productnews'}:
            return 'nonprimary'
        if normalized in {'careerqa', 'newsqa'} and '&' in kind:
            return 'nonprimary'
    groups={key(x) for x in (categories or [])}
    if family == 'archived_jats' and normalized in {'coverpicture', 'coverimage', 'frontcover',
            'backcover', 'insidefrontcover', 'insidebackcover', 'frontispiece'}:
        return 'nonprimary'
    if family == 'archived_jats' and normalized == 'introduction':
        return 'unresolved_type' if groups & {'researcharticle', 'originalarticle', 'originalresearch'} else 'nonprimary'
    if family == 'archived_jats' and normalized == 'abstract':
        return 'nonprimary'
    if family == 'archived_jats' and normalized in {'discussion', 'inbrief', 'reviewerreport', 'editorreport', 'communitycomment'}:
        return 'nonprimary'
    if family in {'publisher_mdpi_oai', 'publisher_frontiers'} and normalized == 'opinion':
        return 'nonprimary'
    if family == 'publisher_pensoft':
        if (groups | {normalized}) & {'conferenceabstract', 'conferenceabstracts'}:
            return 'nonprimary'
        if (groups | {normalized}) & {'monograph', 'taxonomyinventories', 'notesongeographicdistribution',
                                      'checklist', 'fskxfoodsafetyknowledge', 'pensoftunreviewedheading'}:
            return 'subtype_review_required'
    if family == 'publisher_elife_api' and normalized == 'insight':return 'nonprimary'
    if family=='publisher_frontiers_young_minds' and normalized in {'coreconcept','newdiscovery'}:return 'nonprimary'
    if normalized in NONPRIMARY:return 'nonprimary'
    if family=='publisher_springer' and normalized in {'reviewpaper', 'editorialnotes', 'invitedcommentary'}:return 'nonprimary'
    if family == 'publisher_sciopen' and normalized == 'minireview':return 'nonprimary'
    if normalized in {'expressionofconcern','retractedpublication'}:return 'integrity_hold'
    if groups & (NONPRIMARY|{'studyprotocol','protocol','casereport','datapaper'}):
        return 'subtype_review_required'
    if family=='publisher_jstage_html' and normalized in {'originalpaper','originalpapers','originalarticle','originalarticles','researchpaper','researchpapers','researcharticles'}:
        return 'research'
    if family=='publisher_jstage_pdf' and normalized in {'originalpaper','originalarticle','researchpaper'}:
        return 'research'
    if family=='publisher_repository_pdf_front' and normalized=='originalarticle':
        return 'research'
    if family=='publisher_cpl_native' and normalized in {'letter','expressletter'}:
        # CPL author policy explicitly defines these genres as original research.
        return 'research'
    if family=='publisher_cpb_native' and normalized=='rapidcommunication':
        # Native CPB genre, bound to its official peer-review/format policy.
        return 'research'
    if family=='publisher_cjl_native' and normalized in {'researchpaper','researchpapers','researcharticle','researcharticles','研究论文'}:
        return 'research'
    if family=='publisher_springer' and normalized in {'originalpaper','originalarticle','researchpaper'}:
        return 'research'
    if normalized in {'researcharticle','originalresearch','originalresearcharticle'}:return 'research'
    if normalized=='letter' and family=='aps':return 'research'
    if normalized=='article' and family in {'nature_catalogue','nature_page','nature_v1','aps'}:
        return 'research'
    if normalized in SPECIAL:return 'subtype_review_required'
    return 'unresolved_type'


APS_REVIEWED_SCIENTIFIC_CORRECTIONS = {'b3394ae0f5d0f22fbad463da', '59f589b406b41ade9d95007e', '82cb2c5964e539e26b6f65ab', 'd38c0c198a56b7d48fc4ebd9', '700dc64dbbbd28b6bb160e50', '6373517b11f421f4f7fd27ae', '05ec58ab2c01b39f77cf2148', '392d3260ef70d8875150df87', '179b847349da4ac5c5b16324', '3b9ee4bf89d33f09eb1d829c', '5c480d9e24fba857e5069c35', 'e9302068bcdccb674e42b253', '8bd2a3fcaf5c6e4bdf50fbe6', 'cf6bc8536cb7be6fae3b3e06'}


def reviewed_aps_scientific_correction(evidence: dict) -> bool:
    """Exact cached records reviewed in supplement/aps_scientific_corrections/review.json."""
    return (evidence.get("id") in APS_REVIEWED_SCIENTIFIC_CORRECTIONS
            and evidence.get("identity") == "matched" and evidence.get("family") == "aps"
            and evidence.get("kind") == "article" and evidence.get("class") == "subtype_review_required"
            and not evidence.get("scope_review_required"))


def reviewed_aps_tutorial(evidence: dict) -> bool:
    """Official journal-specific educational formats; see aps_tutorial_policy/review.json."""
    return (evidence.get('identity') == 'matched' and evidence.get('family') == 'aps'
            and evidence.get('aps_journal') in {'PRXQUANTUM', 'PRXENERGY', 'PRXLIFE'}
            and key(evidence.get('kind')) == 'tutorial'
            and evidence.get('class') in {'unresolved_type', 'subtype_review_required', 'nonprimary'})


def electronic_date(dates: list[dict]) -> tuple[str,str]:
    accepted=[]
    for item in dates:
        attrs=item['attributes']
        electronic=(attrs.get('pub-type') in {'epub','epub-ppub'} or
                    (attrs.get('publication-format')=='electronic' and attrs.get('date-type') in {None,'pub','publication'}))
        if not electronic:continue
        try:
            year=int(item.get('year') or 0)
            month=int(item.get('month') or 1);day=int(item.get('day') or 1)
            date(year,month,day)
        except (ValueError,TypeError):continue
        precision='day' if item.get('day') and item.get('month') else 'month' if item.get('month') else 'year'
        value=f'{year:04d}' + (f'-{month:02d}' if precision in {'month','day'} else '') + (f'-{day:02d}' if precision=='day' else '')
        accepted.append((value,precision))
    if not accepted:return '', 'missing_explicit_electronic_date'
    if len({x[0][:4] for x in accepted})>1:return '', 'conflicting_electronic_years'
    return min(accepted, key=lambda x:(-len(x[0]),x[0]))


def science_china_chemistry_format(evidence: dict) -> str:
    """Literal publisher formats, supported by the retained 2019 journal guide."""
    if evidence.get('identity') != 'matched' or evidence.get('family') != 'publisher_springer':
        return ''
    journal = evidence.get('source_id') == 'https://openalex.org/S147836164' or (
        not evidence.get('source_id') and 'Science China Chemistry' in evidence.get('journal_titles', [])
        and bool(set(evidence.get('issns', [])) & {'1674-7291', '1869-1870'}))
    if not journal or evidence.get('scope_review_required') or evidence.get('class') == 'integrity_hold':
        return ''
    labels = {key(evidence.get('kind'))} | {key(v) for v in evidence.get('categories', [])}
    if labels & {'highlight', 'highlights'} and labels <= {'highlight', 'highlights', 'briefcommunication'}:
        return 'nonprimary'
    if 'communications' in labels and labels <= {'communications', 'briefcommunication'}:
        return 'research'
    return ''


def science_china_materials_format(evidence: dict) -> str:
    """Use Materials' own official format guide, not Chemistry's definition."""
    if evidence.get('identity') != 'matched' or evidence.get('family') != 'publisher_springer':
        return ''
    journal = evidence.get('source_id') == 'https://openalex.org/S2764782236' or (
        not evidence.get('source_id') and 'Science China Materials' in evidence.get('journal_titles', [])
        and bool(set(evidence.get('issns', [])) & {'2095-8226', '2199-4501'}))
    if not journal or evidence.get('scope_review_required') or evidence.get('class') == 'integrity_hold':
        return ''
    labels = {key(evidence.get('kind'))} | {key(v) for v in evidence.get('categories', [])}
    if labels & {'highlight', 'highlights'} and labels <= {'highlight', 'highlights', 'briefcommunication'}:
        return 'nonprimary'
    if labels & {'letter', 'letters'} and labels <= {'letter', 'letters'}:
        return 'research'
    return ''


def spectrum_observation_format(evidence: dict) -> bool:
    """Journal-specific findings format; supplement/spectrum_observation_policy/rule.json."""
    return (evidence.get('identity') == 'matched'
            and evidence.get('family') == 'archived_jats'
            and evidence.get('class') in {'research', 'subtype_review_required', 'unresolved_type'}
            and not evidence.get('scope_review_required')
            and not evidence.get('title_scope_flag')
            and '2165-0497' in evidence.get('issns', [])
            and {key(v) for v in evidence.get('journal_titles', [])} == {'microbiologyspectrum'}
            and (evidence.get('doi') or '').startswith('10.1128/spectrum.')
            and key(evidence.get('kind')) == 'briefreport'
            and {key(v) for v in evidence.get('categories', [])} == {'observation'})


def native_named_research_format(evidence: dict) -> bool:
    """Named formats with explicit journal headings and official author guidance."""
    if evidence.get('family') != 'archived_jats' or evidence.get('identity') != 'matched':
        return False
    groups = {key(v) for v in evidence.get('categories', [])}
    issns = set(evidence.get('issns', []))
    kind = key(evidence.get('kind'))
    frontiers = (str(evidence.get('doi', '')).startswith('10.3389/')
                 and any(str(v).casefold().startswith('frontiers in ')
                         for v in evidence.get('journal_titles', []))
                 and any(key(v) in {'frontiersmediasa', 'frontiersmedia'} for v in evidence.get('publishers', []))
                 and kind == 'briefreport' and 'briefresearchreport' in groups)
    # Literal short research formats; source evidence is in native_short_formats/policies.json.
    named_short = kind == 'briefreport' and (
        (issns & {'2666-9102'} and 'shortcommunication' in groups)
        or (issns & {'2213-6711'} and groups == {'report'})
        or (issns & {'0950-1991','1477-9129'} and groups == {'researchreport'})
        or (issns & {'0021-9533','1477-9137'} and groups == {'shortreport'})
        or (issns & {'0003-2700','1520-6882'} and groups == {'technicalnote'}))
    return bool(frontiers or named_short or spectrum_observation_format(evidence) or
        (issns & {'2352-3042', '2352-4820'} and kind in {'briefreport', 'letter'}
         and 'rapidcommunication' in groups)
        or (issns & {'1367-4803', '1367-4811', '2635-0041'} and kind == 'briefreport'
            and groups & {'applicationnote', 'applicationnotes', 'applicationsnote', 'applicationsnotes'})
        or (issns & {'1467-5463', '1477-4054'} and kind == 'casereport'
            and groups & {'casestudy', 'casestudies'})
        or (issns & {'1467-7644', '1467-7652'} and kind == 'briefreport'
            and groups & {'briefcommunication', 'briefcommunications'}))


def pediatric_research_format(evidence: dict) -> bool:
    """Explicit basic/clinical/population study formats in Pediatric Research."""
    labels = {key(evidence.get('kind'))} | {key(v) for v in evidence.get('categories', [])}
    research = {'basicsciencearticle', 'clinicalresearcharticle', 'populationstudyarticle'}
    return (evidence.get('identity') == 'matched' and evidence.get('family') in {'nature_page', 'nature_catalogue'}
            and evidence.get('class') != 'integrity_hold' and not evidence.get('scope_review_required')
            and 'Pediatric Research' in evidence.get('journal_titles', [])
            and bool(set(evidence.get('issns', [])) & {'0031-3998', '1530-0447'})
            and bool(labels & research) and labels <= research | {'originalpaper', 'article'})


# Reviewed definitions: supplement/acs_rapid_formats/remaining_guides/reviewed_rules.json
ACS_RAPID_FORMATS = [('Journal of the American Chemical Society', {'0002-7863', '1520-5126'}, 'communication'),
 ('ACS Macro Letters', {'2161-1653'}, 'letter'),
 ('Organic Letters', {'1523-7052', '1523-7060'}, 'letter'),
 ('ACS Synthetic Biology', {'2161-5063'}, 'letter'),
 ('ACS Sustainable Chemistry & Engineering', {'2168-0485'}, 'letter'),
 ('Biochemistry', {'0006-2960', '1520-4995'}, 'communication'),
 ('ACS Medicinal Chemistry Letters', {'1948-5875'}, 'letter'),
 ('ACS Measurement Science Au', {'2694-250X'}, 'letter'),
 ('Analytical Chemistry', {'1520-6882', '0003-2700'}, 'letter'),
 ('JACS Au', {'2691-3704'}, 'letter'),
 ('ACS Sensors', {'2379-3694'}, 'letter'),
 ('ACS Infectious Diseases', {'2373-8227'}, 'letter'),
 ('ACS Photonics', {'2330-4022'}, 'letter'),
 ('ACS Chemical Biology', {'1554-8929', '1554-8937'}, 'letter'),
 ('Chemical Research in Toxicology', {'0893-228X', '1520-5010'}, 'communication'),
 ('Crystal Growth & Design', {'1528-7505', '1528-7483'}, 'communication'),
 ('ACS Bio & Med Chem Au', {'2694-2437'}, 'letter'),
 ('Bioconjugate Chemistry', {'1520-4812', '1043-1802'}, 'communication'),
 ('Journal of Chemical Theory and Computation', {'1549-9618', '1549-9626'}, 'letter'),
 ('ACS Chemical Neuroscience', {'1948-7193'}, 'letter'),
 ('Inorganic Chemistry', {'1520-510X', '0020-1669'}, 'communication'),
 ('ACS Environmental Au', {'2694-2518'}, 'letter'),
 ('ACS Applied Optical Materials', {'2771-9855'}, 'letter'),
 ('ACS Materials Au', {'2694-2461'}, 'letter'),
 ('ACS Materials Letters', {'2639-4979'}, 'letter'),
 ('ACS Nanoscience Au', {'2694-2496'}, 'letter'),
 ('Chemical & Biomedical Imaging', {'2832-3637'}, 'letter'),
 ('ACS Energy Letters', {'2380-8195'}, 'letter'),
 ('ACS Organic & Inorganic Au', {'2694-247X'}, 'letter'),
 ('ACS Physical Chemistry Au', {'2694-2445'}, 'letter'),
 ('ACS Applied Nano Materials', {'2574-0970'}, 'letter'),
 ('Journal of Chemical Information and Modeling', {'1549-9596', '1549-960X'}, 'letter'),
 ('ACS Pharmacology & Translational Science', {'2575-9108'}, 'letter'),
 ('ACS Applied Polymer Materials', {'2637-6105'}, 'letter'),
 ('ACS Catalysis', {'2155-5435'}, 'letter'),
 ('Precision Chemistry', {'2771-9316'}, 'letter'),
 ('ACS Applied Bio Materials', {'2576-6422'}, 'letter'),
 ('ACS Polymers Au', {'2694-2453'}, 'letter'),
 ('ACS Earth & Space Chemistry', {'2472-3452'}, 'letter'),
 ('Chem & Bio Engineering', {'2836-967X'}, 'letter'),
 ('Molecular Pharmaceutics', {'1543-8384', '1543-8392'}, 'communication'),
 ('ACS Biomaterials Science & Engineering', {'2373-9878'}, 'letter'),
 ('ACS Applied Energy Materials', {'2574-0962'}, 'letter')]


def acs_reviewed_rapid_format(evidence: dict) -> bool:
    """Exact native short formats corroborated by reviewed official ACS guides."""
    if (evidence.get('family') != 'archived_jats' or evidence.get('identity') != 'matched'
            or key(evidence.get('kind')) != 'rapidcommunication'
            or not (evidence.get('doi') or '').startswith('10.1021/')):
        return False
    labels = {key(v) for v in evidence.get('categories', [])}
    issns = set(evidence.get('issns', []))
    titles = {key(v) for v in evidence.get('journal_titles', [])}
    return any(issns & identifiers and key(title) in titles
               and len(labels) == 1 and labels <= {label, label + 's'}
               for title, identifiers, label in ACS_RAPID_FORMATS)


def complete_springer_short_format(evidence: dict) -> bool:
    """Two named research sections; retained official definitions in remaining_springer_short_formats/rules.json."""
    rules = {
        'Journal of Plant Pathology': ('2239-7264', '10.1007/s42161-'),
        'National Academy Science Letters': ('2250-1754', '10.1007/s40009-'),
    }
    titles = evidence.get('journal_titles', [])
    if (evidence.get('identity') != 'matched' or evidence.get('family') != 'publisher_springer'
            or len(titles) != 1 or titles[0] not in rules):
        return False
    issn, prefix = rules[titles[0]]
    return (issn in evidence.get('issns', [])
            and (evidence.get('doi') or '').startswith(prefix)
            and key(evidence.get('kind')) in {'briefcommunication', 'shortcommunication'}
            and {key(v) for v in evidence.get('categories', [])}
                == {'shortcommunication', 'briefcommunication'})


def elsevier_sco_study_report(evidence: dict) -> bool:
    """Native PLB short research reports with retained publisher and study evidence."""
    abstract = evidence.get('native_research_report_abstract', '')
    quote = evidence.get('native_research_report_quote', '')
    return (evidence.get('family') == 'publisher_elsevier'
            and evidence.get('native_pit') == 'sco'
            and evidence.get('native_journal_code') == 'PLB'
            and evidence.get('journal_titles') == ['Physics Letters B']
            and (evidence.get('doi') or '').startswith('10.1016/j.physletb.')
            and key(evidence.get('kind')) == 'shortcommunication'
            and {key(v) for v in evidence.get('categories', [])} == {'shortcommunication'}
            and evidence.get('type_policy_receipt') == '../../supplement/target_500k/plb_cached_type_review/rule.json'
            and bool(quote) and quote in abstract
            and evidence.get('native_abstract_sha256') == hashlib.sha256(abstract.encode()).hexdigest()
            and evidence.get('class') != 'integrity_hold')


def journal_research_format(evidence: dict) -> bool:
    """Reviewed journal-specific formats, without overriding content holds."""
    if evidence.get('identity') != 'matched':
        return False
    complete_short = complete_springer_short_format(evidence)
    horticulture_report = (
        evidence.get('family') == 'publisher_springer'
        and key(evidence.get('kind')) == 'researchreport'
        and evidence.get('journal_titles') == ['Horticulture, Environment, and Biotechnology']
        and '2211-3460' in evidence.get('issns', [])
        and evidence.get('type_policy_receipt') == 'supplement/target_500k/horticulture_reports/policy.json'
        and {key(v) for v in evidence.get('categories', [])} <= {'researchreport', 'report'}
    )
    acs_letter = acs_reviewed_rapid_format(evidence) or (
        evidence.get('identity') == 'matched' and evidence.get('family') == 'archived_jats'
        and key(evidence.get('kind')) == 'rapidcommunication'
        and {key(v) for v in evidence.get('categories', [])} == {'letter'}
        and bool(set(evidence.get('issns', [])) & {'1530-6984', '1530-6992', '1948-7185'})
    )
    pnas = (evidence.get('source_id') == 'https://openalex.org/S125754415' or
            bool(set(evidence.get('issns', [])) & {'0027-8424', '1091-6490'}) or
            ('Proceedings of the National Academy of Sciences of the United States of America'
             in evidence.get('journal_titles', []) and
             'National Academy of Sciences' in evidence.get('publishers', [])))
    elife = (evidence.get('identity') == 'matched'
             and evidence.get('family') == 'publisher_elife_api'
             and evidence.get('source_id') == 'https://openalex.org/S1336409049'
             and key(evidence.get('kind')) in {'shortreport', 'researchadvance'}
             and bool(evidence.get('type_policy_receipt')))
    bdj_taxonomy = (evidence.get('identity') == 'matched'
                    and evidence.get('family') == 'publisher_pensoft'
                    and bool(set(evidence.get('issns', [])) & {'1314-2836', '1314-2828'})
                    and 'taxonomyinventories' in {key(v) for v in evidence.get('categories', [])})
    molecular_journal = (evidence.get('source_id') == 'https://openalex.org/S197962172'
        or (not evidence.get('source_id')
            and 'Molecular Biology Reports' in evidence.get('journal_titles', [])
            and bool(set(evidence.get('issns', [])) & {'0301-4851', '1573-4978'})))
    molecular_brief = (molecular_journal
                       and evidence.get('family') == 'publisher_springer'
                       and key(evidence.get('kind')) == 'briefreport')
    metallurgy_journal = (evidence.get('source_id') == 'https://openalex.org/S165830345'
        or (not evidence.get('source_id')
            and 'Metallurgical and Materials Transactions A' in evidence.get('journal_titles', [])
            and bool(set(evidence.get('issns', [])) & {'1073-5623', '1543-1940'})))
    metallurgy_brief = (metallurgy_journal
                       and evidence.get('family') == 'publisher_springer'
                       and key(evidence.get('kind')) == 'briefcommunication')
    microbiology_journal = (evidence.get('source_id') == 'https://openalex.org/S2248630'
        or (not evidence.get('source_id')
            and 'Current Microbiology' in evidence.get('journal_titles', [])
            and bool(set(evidence.get('issns', [])) & {'0343-8651', '1432-0991'})))
    microbiology_short = (microbiology_journal
        and evidence.get('family') == 'publisher_springer'
        and (key(evidence.get('kind')) == 'shortcommunication'
             or (key(evidence.get('kind')) == 'briefcommunication'
                 and 'shortcommunication' in {key(v) for v in evidence.get('categories', [])})))
    chemistry_communication = science_china_chemistry_format(evidence) == 'research'
    materials_letter = science_china_materials_format(evidence) == 'research'
    polymer_original = (evidence.get('family') == 'nature_page'
        and key(evidence.get('kind')) in {'originalarticle', 'originalpaper'}
        and 'Polymer Journal' in evidence.get('journal_titles', [])
        and bool(set(evidence.get('issns', [])) & {'0032-3896', '1349-0540'}))
    # Elsevier official DTD table: FLA is original research; retained policy p57.
    # This application is limited to the checked NPB XML cohort.
    elsevier_full_length = (
        evidence.get('family') == 'publisher_elsevier'
        and key(evidence.get('kind')) == 'fulllengtharticle'
        and set(evidence.get('journal_titles', [])) == {'Nuclear Physics B'}
        and (evidence.get('doi') or '').startswith('10.1016/j.nuclphysb.')
        and {key(v) for v in evidence.get('categories', [])} == {'fulllengtharticle'}
        and evidence.get('class') != 'integrity_hold')
    # Founding editorial 10.1016/j.crmeth.2021.100013 distinguishes Reports
    # (short single-point methodological studies) from commentary/review formats.
    cell_methods_report = (
        evidence.get('family') == 'archived_jats'
        and key(evidence.get('kind')) == 'briefreport'
        and evidence.get('journal_titles') == ['Cell Reports Methods']
        and bool(set(evidence.get('issns', [])) & {'2667-2375'})
        and (evidence.get('doi') or '').startswith('10.1016/j.crmeth.')
        and {key(v) for v in evidence.get('categories', [])} == {'report'}
        and evidence.get('class') in {'subtype_review_required', 'research'})
    if elsevier_sco_study_report(evidence) or complete_short or horticulture_report or cell_methods_report or elsevier_full_length or pediatric_research_format(evidence) or native_named_research_format(evidence) or acs_letter or elife or bdj_taxonomy or molecular_brief or metallurgy_brief or microbiology_short or chemistry_communication or materials_letter or polymer_original:
        pass
    elif pnas:
        if (evidence.get('identity') != 'matched' or evidence.get('family') != 'archived_jats'
                or key(evidence.get('kind')) != 'briefreport'):
            return False
    else:
        formats = {
            'https://openalex.org/S106963461': {'briefcommunication'},
            'https://openalex.org/S127827428': {'briefcommunication', 'analysis', 'resource'},
            'https://openalex.org/S137905309': {'briefcommunication', 'analysis', 'letter', 'technicalreport'},
            'https://openalex.org/S4210214063': {'letter', 'briefcommunication', 'resource'},
            'https://openalex.org/S4210232714': {'letter', 'briefcommunication', 'resource'},
            'https://openalex.org/S151741590': {'resource', 'technicalreport'},
            'https://openalex.org/S2764671299': {'letter', 'briefcommunication', 'resource'},
            'https://openalex.org/S2764926557': {'analysis', 'briefcommunication', 'resource'},
            'https://openalex.org/S139253143': {'analysis', 'briefcommunication', 'resource', 'technicalreport'},
            'https://openalex.org/S103895331': {'letter'},
            'https://openalex.org/S41063453': {'letter', 'resource', 'technicalreport'},
            'https://openalex.org/S4210228084': {'analysis', 'briefcommunication', 'resource'},
            'https://openalex.org/S2298632': {'briefcommunication', 'resource', 'technicalreport'},
            'https://openalex.org/S203256638': {'briefcommunication', 'resource'},
        }.get(evidence.get('source_id'), set())
        if evidence.get('family') not in {'nature_page', 'nature_catalogue'} or key(evidence.get('kind')) not in formats:
            return False
    if evidence.get('scope_review_required') or evidence.get('title_scope_flag'):
        return False
    allowed = {key(evidence.get('kind'))} | ({'letter'} if acs_letter else set())
    if complete_short:
        allowed |= {'briefcommunication', 'shortcommunication'}
    if (native_named_research_format(evidence)
            and set(evidence.get('issns', [])) & {'1467-7644', '1467-7652'}):
        allowed |= {'briefcommunication', 'briefcommunications'}
    if {key(v) for v in evidence.get('categories', [])} & (NONPRIMARY | (SPECIAL - allowed) | {'expressionofconcern', 'retractedpublication'}):
        return False
    title = evidence.get('title', '')
    return not (key(title).startswith(('correction', 'retraction', 'erratum', 'expressionofconcern')) or
                re.search(r'\b(systematic review|meta.analysis|study protocol|case report|case series|scoping review|narrative review|umbrella review|bibliometric analysis)\b', title, re.I))


def expression_notice(evidence: dict) -> bool:
    """Recognize the notice itself, not an integrity flag on a research paper."""
    return (not evidence.get('scope_review_required')
            and key(evidence.get('kind')) == 'expressionofconcern'
            and key(evidence.get('title')).startswith('expressionofconcern'))


def nature_protocol_format(evidence: dict) -> bool:
    """Official nonprimary protocol formats; supplement/nature_protocol_policy/rule.json."""
    if (evidence.get('identity') != 'matched'
            or evidence.get('family') not in {'nature_catalogue', 'nature_page'}
            or evidence.get('class') not in {'subtype_review_required', 'unresolved_type', 'nonprimary'}
            or evidence.get('scope_review_required')
            or not (evidence.get('doi') or '').startswith('10.1038/s41596-')):
        return False
    journal = (evidence.get('source_id') == 'https://openalex.org/S109387254'
               or (not evidence.get('source_id')
                   and {key(v) for v in evidence.get('journal_titles', [])} == {'natureprotocols'}
                   and bool(set(evidence.get('issns', [])) & {'1754-2189', '1750-2799'})))
    labels = {key(v) for v in evidence.get('categories', [])}
    kind = key(evidence.get('kind'))
    return bool(journal and labels <= {'protocol', 'protocolextension', 'protocolupdate', 'reviewpaper'}
                and (kind in {'protocol', 'protocolextension', 'protocolupdate'}
                     or (kind == 'reviewpaper' and bool(labels & {'protocol', 'protocolextension', 'protocolupdate'}))))


def refine_nature_protocols(primary: list[dict]) -> list[dict]:
    """Corroborate generic archive type using the same manuscript's publisher type."""
    precise = [e for e in primary if nature_protocol_format(e)]
    output = []
    for item in primary:
        if (item.get('family') == 'archived_jats' and item.get('identity') == 'matched'
                and item.get('class') == 'research' and key(item.get('kind')) == 'researcharticle'
                and {key(v) for v in item.get('categories', [])} <= {'article'}
                and {key(v) for v in item.get('journal_titles', [])} == {'natureprotocols'}
                and (not item.get('issns') or bool(set(item['issns']) & {'1754-2189', '1750-2799'}))
                and not item.get('scope_review_required')):
            for native in precise:
                if (item.get('doi') and item.get('doi') == native.get('doi')
                        and key(item.get('title')) and key(item.get('title')) == key(native.get('title'))):
                    item = {**item, 'class': 'nonprimary', 'format_refined_by_evidence_id': native['id']}
                    break
        output.append(item)
    return [{**e, 'class': 'nonprimary'} if nature_protocol_format(e) else e for e in output]


def star_protocol_format(evidence: dict) -> bool:
    """Native STAR Protocols format; see supplement/star_protocol_policy/rule.json."""
    return (evidence.get('identity') == 'matched'
            and evidence.get('family') == 'archived_jats'
            and evidence.get('class') in {'subtype_review_required', 'unresolved_type', 'nonprimary'}
            and not evidence.get('scope_review_required')
            and '2666-1667' in evidence.get('issns', [])
            and {key(v) for v in evidence.get('journal_titles', [])} == {'starprotocols'}
            and {key(v) for v in evidence.get('categories', [])} == {'protocol'}
            and key(evidence.get('kind')) in {'researcharticle', 'protocol'})


def refine_star_protocols(primary: list[dict]) -> list[dict]:
    """Refine category-free legacy fronts only with the same paper's precise front."""
    precise = [e for e in primary if star_protocol_format(e)]
    output = []
    for item in primary:
        if (item.get('family') == 'archived_jats' and item.get('identity') == 'matched'
                and item.get('class') == 'research' and not item.get('categories')
                and not item.get('scope_review_required')
                and key(item.get('kind')) == 'researcharticle'):
            for source in precise:
                if (item.get('doi') and item.get('doi') == source.get('doi')
                        and key(item.get('title')) and key(item.get('title')) == key(source.get('title'))
                        and {key(v) for v in item.get('journal_titles', [])} == {'starprotocols'}
                        and (not item.get('issns') or '2666-1667' in item['issns'])):
                    item = {**item, 'class': 'nonprimary',
                            'format_refined_by_evidence_id': source['id']}
                    break
        output.append(item)
    return output


def plos_format_class(evidence: dict) -> str:
    """Official PLOS journal-specific formats, limited to the matched API source."""
    if evidence.get('family') != 'publisher_plos' or evidence.get('class') not in {'unresolved_type', 'subtype_review_required', 'integrity_hold'}:
        return ''
    rules = {
        '1545-7885': ({'shortreports', 'methodsandresources', 'methodsresources',
                      'metaresearcharticle', 'updatearticle', 'discoveryreport'},
                     {'primer', 'essay', 'communitypage', 'unsolvedmystery', 'formalcomment', 'consensusview'}),
        '1553-7358': ({'methods', 'software'}, {'education', 'tensimplerules', 'quicktips', 'collectionoverview', 'formalcomment'}),
        '1553-7404': ({'methods'}, {'viewpoints', 'opinionpiece', 'opinion'}),
        '1553-7374': ({'shortreports'}, {'pearls'}),
        '1935-2735': (set(), {'viewpoints', 'formalcomment', 'policyplatform'}),
        '1932-6203': (set(), {'registeredreportprotocol', 'studyprotocol'}),
    }
    matches = [rules[i] for i in set(evidence.get('issns', [])) if i in rules]
    if len(matches) != 1 or evidence.get('scope_review_required'):
        return ''
    if evidence.get('class') == 'integrity_hold':
        return 'nonprimary' if expression_notice(evidence) else ''
    kind = key(evidence.get('kind'))
    research, nonprimary = matches[0]
    if kind in nonprimary:
        return 'nonprimary'
    # Only explicitly nonresearch journal formats can resolve an existing subtype hold.
    if evidence.get('class') == 'subtype_review_required':
        return ''
    if kind not in research:
        return ''
    groups = {key(v) for v in evidence.get('categories', [])}
    title = evidence.get('title', '')
    if (evidence.get('title_scope_flag') or groups & (NONPRIMARY | SPECIAL | {'expressionofconcern', 'retractedpublication'})
            or key(title).startswith(('correction', 'retraction', 'erratum', 'expressionofconcern'))
            or re.search(r'\b(systematic review|meta.analysis|study protocol|case report|case series|scoping review|narrative review|umbrella review)\b', title, re.I)):
        return 'subtype_review_required'
    return 'research'


def reviewed_pnas_letter_content(primary: list[dict]) -> list[dict]:
    """Exclude only individually read pure commentary; never exclude Letters as a class."""
    family = 'reviewed_pnas_letter_content'
    originals = [e for e in primary if e.get('family') != family]
    sources = {e['id']: e for e in originals}
    if len(sources) != len(originals):
        return originals
    accepted, replaced = [], set()
    for review in primary:
        if review.get('family') != family:
            continue
        targets = review.get('reviewed_primary_evidence_ids', [])
        if (review.get('identity') != 'matched'
                or review.get('study_design_class') != 'nonprimary'
                or review.get('review_protocol') != 'pnas_letter_full_body_individual_1'
                or not (review.get('doi') or '').startswith('10.1073/pnas.')
                or not set(review.get('issns', [])) & {'0027-8424', '1091-6490'}
                or not all(review.get(k) for k in ('content_sha256', 'raw_sha256',
                           'content_locator', 'review_rule_receipt', 'content_review_reason'))
                or review.get('scope_review_required') or not targets
                or any(i not in sources or sources[i].get('identity') != 'matched'
                       or sources[i].get('doi') != review['doi']
                       or sources[i].get('family') != 'archived_jats'
                       or key(sources[i].get('kind')) != 'letter'
                       or sources[i].get('class') not in {'subtype_review_required', 'unresolved_type'}
                       or sources[i].get('scope_review_required') for i in targets)):
            continue
        replaced.update(targets)
        accepted.append({**review, 'class': 'nonprimary'})
    return [{**e, 'class': 'nonprimary'} if e['id'] in replaced else e
            for e in originals] + accepted


def reviewed_frontiers_content(primary: list[dict]) -> list[dict]:
    """Apply individually evidenced nonresearch reviews, never a Perspective-wide rule."""
    if not any(e.get('family') == 'reviewed_frontiers_content' for e in primary):
        return primary
    sources = {e['id']: e for e in primary if e.get('family') != 'reviewed_frontiers_content'}
    originals = [e for e in primary if e.get('family') != 'reviewed_frontiers_content']
    if len(sources) != len(originals):
        return originals
    accepted = []
    replaced = set()
    for review in primary:
        if review.get('family') != 'reviewed_frontiers_content':
            continue
        targets = review.get('reviewed_primary_evidence_ids', [])
        if (review.get('study_design_class') != 'nonprimary' or not targets
                or not review.get('content_sha256') or not review.get('content_locator')
                or not review.get('review_quotes') or not review.get('content_review_reason')
                or review.get('scope_review_required')
                or any(i not in sources or not frontiers_perspective_hold(sources[i])
                       or sources[i].get('doi') != review.get('doi')
                       or sources[i].get('scope_review_required') for i in targets)):
            continue
        accepted.append({**review, 'class': 'nonprimary'})
        replaced.update(targets)
    return [{**e, 'class': 'nonprimary'} if e['id'] in replaced else e
            for e in sources.values()] + accepted


def frontiers_methods_content_review(review: dict, sources: dict, targets: list, backing: list) -> bool:
    """Accept only individually reviewed, source-bound Frontiers Methods content."""
    doi = review.get('doi') or ''
    if (review.get('review_protocol') != 'native_frontiers_methods_design_1'
            or not doi.startswith('10.3389/')
            or review.get('review_rule_receipt') != 'supplement/frontiers_remaining_formats/methods_content_reviews.json'
            or review.get('format_policy_receipt') != 'supplement/frontiers_remaining_formats/methods_policy_review.json'
            or not all(review.get(k) for k in ('study_design_reason', 'raw_sha256', 'abstract_sha256', 'content_locator'))
            or not targets or not backing):
        return False
    if not all(i in sources and sources[i].get('doi') == doi for i in [*targets, *backing]):
        return False
    return (any(sources[i].get('family') in {'archived_jats', 'publisher_frontiers'}
                and any(str(t).casefold().startswith('frontiers in ') for t in sources[i].get('journal_titles', []))
                for i in backing)
            and all((sources[i].get('family') == 'archived_jats' and key(sources[i].get('kind')) == 'methodsarticle')
                    or (sources[i].get('family') == 'publisher_frontiers' and key(sources[i].get('kind')) == 'methods')
                    for i in targets))


def reviewed_study_design(primary: list[dict]) -> list[dict]:
    """Resolve only named subtype holds reviewed against matched primary content."""
    if not any(e.get('family') == 'reviewed_study_design' for e in primary):
        return primary
    sources = {e['id']: e for e in primary if e.get('family') != 'reviewed_study_design'}
    originals = [e for e in primary if e.get('family') != 'reviewed_study_design']
    if len(sources) != len(originals):
        return originals
    accepted = []
    replacements: dict[str, set[str]] = {}
    for review in primary:
        if review.get('family') != 'reviewed_study_design':
            continue
        targets = review.get('reviewed_primary_evidence_ids', [])
        backing = review.get('source_evidence_ids', [])
        classification = review.get('study_design_class', review.get('class'))
        mdpi_brief_review = (
            review.get('review_protocol') == 'native_mdpi_brief_report_design_1'
            and (review.get('doi') or '').startswith('10.3390/')
            and review.get('review_rule_receipt') == 'supplement/mdpi_brief_report_policy/content_reviews.json'
            and review.get('format_policy_receipt') == 'supplement/mdpi_brief_report_policy/policy.json'
            and bool(review.get('study_design_reason'))
        )
        native_content_review = (
            (review.get('review_protocol') in {'native_nature_individual_design_1', 'native_individual_design_1'}
             or mdpi_brief_review)
            and review.get('raw_sha256') and review.get('abstract_sha256')
            and review.get('content_locator') and review.get('review_rule_receipt')
            and bool(review.get('doi'))
            and all(sources.get(i, {}).get('doi') == review['doi'] for i in backing)
            and (any(sources.get(i, {}).get('family') == 'archived_jats' for i in backing)
                 or (mdpi_brief_review and any(sources.get(i, {}).get('family') == 'publisher_mdpi_oai' for i in backing)))
            and (review.get('review_protocol') == 'native_individual_design_1' or mdpi_brief_review
                 or any(sources.get(i, {}).get('family') in {'nature_catalogue', 'nature_page'} for i in backing))
            and all(sources.get(i, {}).get('doi') == review['doi']
                    and sources.get(i, {}).get('family') in {'archived_jats', 'nature_catalogue', 'nature_page', 'publisher_mdpi_oai'}
                    and key(sources.get(i, {}).get('kind')) in
                        ({'briefreport', 'researcharticle'} if mdpi_brief_review else {'article', 'researcharticle', 'originalpaper'})
                    and (not mdpi_brief_review or sources.get(i, {}).get('family') in {'archived_jats', 'publisher_mdpi_oai'})
                    for i in targets)
        )
        springer_content_review = (
            classification == 'nonprimary'
            and review.get('review_protocol') == 'springer_special_native_content_1'
            and review.get('review_rule_receipt') == 'supplement/springer_special_originalpaper/content_reviews.json'
            and all(review.get(k) for k in ('raw_sha256', 'content_sha256',
                                           'content_locator', 'review_quotes', 'study_design_reason'))
            and bool(targets) and bool(backing)
            and all(springer_special_originalpaper(sources.get(i, {}))
                    and sources[i].get('doi') == review.get('doi')
                    for i in [*targets, *backing])
        )
        native_content_review = (native_content_review or springer_content_review
                                 or frontiers_methods_content_review(review, sources, targets, backing))
        archive_nonprimary = (
            classification == 'nonprimary' and review.get('generic_archive_nonresearch_adjudication')
            and review.get('content_sha256') and review.get('content_locator')
            and any(plos_format_class(sources.get(i, {})) == 'nonprimary' for i in backing)
        )
        if (classification not in {'research', 'nonprimary'} or not targets or not backing
                or not (review.get('abstract_sha256') or review.get('content_sha256')) or review.get('scope_review_required')
                or any(i not in sources for i in [*targets, *backing])
                or not (native_content_review or any(sources[i].get('family') == 'publisher_plos' for i in backing))
                or any((sources[i].get('class') not in ({'subtype_review_required', 'unresolved_type', 'research'}
                                                      if classification == 'research' else {'subtype_review_required', 'unresolved_type'})
                        and not (native_content_review and sources[i].get('class') == 'research')
                        and not (archive_nonprimary and sources[i].get('family') == 'archived_jats'
                                 and sources[i].get('class') == 'research'
                                 and key(sources[i].get('kind')) == 'researcharticle'))
                       or sources[i].get('scope_review_required') for i in targets)):
            continue
        accepted.append({**review, 'class': classification})
        for ident in targets:
            replacements.setdefault(ident, set()).add(classification)
    result = []
    for source in originals:
        ident = source['id']
        values = replacements.get(ident, set())
        result.append({**source, 'class': next(iter(values))} if len(values) == 1 else source)
    return result + accepted


def resolve_elife_publication_path(primary: list[dict]) -> list[dict]:
    """Clear only the provisional RP-path hold corroborated by native v8 identity."""
    verified = [e for e in primary if e.get('family') == 'publisher_elife_api'
                and e.get('native_api_version') == 8 and e.get('native_version_doi')
                and e.get('raw_sha256') and e.get('native_status') == 'vor'
                and e.get('online_date') and not e.get('scope_review_required')]
    result = []
    for item in primary:
        provisional = (item.get('family') == 'publisher_elife_api'
                       and item.get('scope_note') == 'Reviewed-preprint pathway requires historical journal publication date review'
                       and item.get('scope_review_required') and not item.get('title_scope_flag'))
        matched = any(e.get('native_doi') == item.get('native_doi')
                      and key(e.get('title')) == key(item.get('title'))
                      and e.get('kind') == item.get('kind') for e in verified)
        if provisional and matched:
            item = {**item, 'scope_review_required': False,
                    'class': type_class(item['kind'], item['family'], item.get('categories'))}
        result.append(item)
    return result


def pnas_frontmatter(evidence: dict) -> bool:
    """Native PNAS magazine categories override generic research labels for review."""
    pnas = (evidence.get('source_id') == 'https://openalex.org/S125754415'
            or bool(set(evidence.get('issns', [])) & {'0027-8424', '1091-6490'}))
    categories = {key(value) for value in evidence.get('categories', [])}
    return bool(pnas and evidence.get('identity') == 'matched'
                and evidence.get('family') == 'archived_jats'
                and categories & {'coreconcepts', 'innerworkings', 'scienceandculture',
                                  'scienceculture', 'opinions', 'newsfeatures', 'profile', 'qnas'})


def adjudicate_pnas_magazine(primary: list[dict]) -> list[dict]:
    """Apply native-heading proofs only to explicitly reviewed archive evidence IDs."""
    targets = set()
    for proof in primary:
        if (pnas_frontmatter(proof) and proof.get('native_magazine_rule') == 'pnas_editorial_1300186110'
                and proof.get('native_heading_sha256') and proof.get('magazine_policy_receipt')):
            targets.update(proof.get('reviewed_primary_evidence_ids', []))
            targets.add(proof['id'])
    return [{**e, 'class': 'nonprimary', 'magazine_adjudicated': True}
            if e['id'] in targets and e.get('family') == 'archived_jats'
            and e.get('class') != 'integrity_hold' else e for e in primary]


def frontiers_reviewed_format(evidence: dict) -> str:
    """Two explicit Frontiers formats; policy receipt in supplement/frontiers_type_policy."""
    if (evidence.get('family') != 'publisher_frontiers' or evidence.get('identity') != 'matched'
            or not (evidence.get('doi') or '').startswith('10.3389/')
            or evidence.get('scope_review_required') or evidence.get('title_scope_flag')
            or evidence.get('class') == 'integrity_hold'):
        return ''
    kind = key(evidence.get('kind'))
    if kind == 'minireview':
        return 'nonprimary'
    if kind != 'briefresearchreport':
        return ''
    groups = {key(v) for v in evidence.get('categories', [])}
    if groups & (NONPRIMARY | SPECIAL | {'expressionofconcern', 'retractedpublication', 'minireview'}):
        return ''
    if re.search(r'\b(systematic review|meta.analysis|study protocol|case report|case series|scoping review|narrative review|umbrella review)\b', evidence.get('title', ''), re.I):
        return ''
    return 'research'


def frontiers_perspective_hold(evidence: dict) -> bool:
    """Frontiers permits original data in Perspectives; the label alone is insufficient."""
    if (evidence.get('identity') != 'matched' or evidence.get('class') == 'integrity_hold'
            or not (evidence.get('doi') or '').startswith('10.3389/')
            or key(evidence.get('kind')) not in {'perspective', 'perspectives'}):
        return False
    return (evidence.get('family') == 'publisher_frontiers'
            or (evidence.get('family') == 'archived_jats'
                and any(t.startswith('Frontiers in ') for t in evidence.get('journal_titles', []))
                and any(key(v) in {'frontiersmediasa', 'frontiersmedia'}
                        for v in evidence.get('publishers', []))))


def mdpi_reviewed_communication(evidence: dict) -> bool:
    """MDPI OAI rapid-communication maps explicitly to Communication in ListSets."""
    if (evidence.get('family') != 'publisher_mdpi_oai' or evidence.get('identity') != 'matched'
            or not evidence.get('doi', '').startswith('10.3390/')
            or key(evidence.get('kind')) != 'rapidcommunication'
            or evidence.get('scope_review_required') or evidence.get('title_scope_flag')
            or evidence.get('class') == 'integrity_hold'):
        return False
    groups = {key(v) for v in evidence.get('categories', [])}
    if groups & (NONPRIMARY | SPECIAL | {'expressionofconcern', 'retractedpublication'}):
        return False
    return not re.search(r'\b(systematic review|meta.analysis|study protocol|case report|case series|scoping review|narrative review|umbrella review)\b', evidence.get('title', ''), re.I)


def reviewed_case_report(primary: list[dict]) -> bool:
    """Apply an explicit user scope choice only to linked native case-report labels."""
    indexed = {e['id']: e for e in primary}
    for review in primary:
        if (review.get('family') != 'reviewed_comparison_scope'
                or review.get('scope_rule') != 'case_report_extension_20261026'
                or review.get('comparison_scope') != 'extension_only'):
            continue
        for ident in review.get('source_evidence_ids', []):
            native = indexed.get(ident, {})
            labels = {key(native.get('kind', ''))} | {key(v) for v in native.get('categories', [])}
            if (native.get('identity') == 'matched'
                    and native.get('family') in {'archived_jats', 'publisher_springer', 'publisher_mdpi_oai', 'publisher_frontiers', 'publisher_pensoft', 'publisher_oai_dc'}
                    and native.get('doi') and native.get('doi') == review.get('doi')
                    and labels & {'casereport', 'casereports'}
                    and not native.get('scope_review_required')):
                return True
    return False


def reviewed_data_note(primary: list[dict]) -> bool:
    """Apply an explicit user scope choice only to linked native Data Note labels in the two approved journals."""
    indexed = {e['id']: e for e in primary}
    for review in primary:
        if (review.get('family') != 'reviewed_comparison_scope'
                or review.get('scope_rule') != 'data_note_extension_20261026'
                or review.get('comparison_scope') != 'extension_only'):
            continue
        for ident in review.get('source_evidence_ids', []):
            native = indexed.get(ident, {})
            labels = {key(native.get('kind', ''))} | {key(v) for v in native.get('categories', [])}
            if (native.get('identity') == 'matched'
                    and native.get('family') in {'archived_jats', 'publisher_wellcome_xml'}
                    and set(native.get('issns', [])) & {'2398-502X', '1756-0500'}
                    and native.get('doi') and native.get('doi') == review.get('doi')
                    and labels & {'datanote', 'datanotes'}
                    and not native.get('scope_review_required')):
                return True
    return False


def reviewed_resource_announcement(primary: list[dict]) -> bool:
    """User-selected main-sample exclusion, backed by an identified native subtype."""
    indexed = {e['id']: e for e in primary}
    for review in primary:
        if (review.get('family') != 'reviewed_comparison_scope'
                or review.get('scope_rule') != 'mra_genome_sequences_extension_20261025'
                or review.get('comparison_scope') != 'extension_only'):
            continue
        for ident in review.get('source_evidence_ids', []):
            native = indexed.get(ident, {})
            if (native.get('identity') == 'matched'
                    and native.get('family') == 'archived_jats'
                    and native.get('doi') and native.get('doi') == review.get('doi')
                    and '2576-098X' in native.get('issns', [])
                    and native.get('kind') in {'brief-report', 'research-article'}
                    and 'Genome Sequences' in native.get('categories', [])
                    and not native.get('scope_review_required')):
                return True
    return False


CACHED_REVIEW_SELF_DESCRIPTION = re.compile(
    r'\b(?:in this (?:narrative )?review\s*,?\s*we|here\s*,?\s*we review|'
    r'this (?:narrative )?review (?:aims?|summari[sz]es?|discusses?|focuses?|'
    r'will focus|provides?|presents?|highlights?|outlines?|describes?|examines?|explores?|covers?))\b', re.I)
CACHED_REVIEW_MIXED_SCOPE = re.compile(
    r'\b(?:we (?:also )?(?:report|demonstrate|develop(?:ed)?|perform(?:ed)?|conduct(?:ed)?|'
    r'investigat(?:e|ed)|measur(?:e|ed)|test(?:ed)?|evaluat(?:e|ed)|introduce|present)|'
    r'our (?:results|experiments|findings|study)|new (?:data|results|experiments|model|framework)|'
    r'original (?:data|research)|meta.analysis|systematic review|scoping review|case report|'
    r'study protocol|bibliometric analysis)\b|'
    r'\b(?:we (?:also |further |additionally )?(?:used?|found|analy[sz](?:e|ed)|appl(?:y|ied)|calculat(?:e|ed)|collect(?:ed)?|assess(?:ed)?|validat(?:e|ed)|observ(?:e|ed)|show(?:ed)?|compar(?:e|ed)|identified|confirmed|established|provide(?:d)? new)|our (?:analysis|analyses|data|model|experiments|research)|first of its kind|new (?:approaches|analysis|analyses)|novel (?:data|analysis|analyses))\b', re.I)


def cached_author_review(evidence: dict, secondary: list[dict]) -> bool:
    """Archived author self-description plus MED Review; never publisher-type proof."""
    abstract = evidence.get('review_abstract', '')
    return (evidence.get('family') == 'cached_author_review_scope'
            and evidence.get('identity') == 'matched'
            and evidence.get('review_rule') == 'med_author_review_no_primary_1'
            and evidence.get('source') == 'MED' and bool(evidence.get('pmid'))
            and not evidence.get('scope_review_required')
            and bool(evidence.get('raw_sha256')) and bool(evidence.get('content_locator'))
            and bool(evidence.get('issns')) and bool(evidence.get('doi'))
            and 'Review' in evidence.get('indexed_types', [])
            and len(abstract.split()) >= 40
            and hashlib.sha256(abstract.encode()).hexdigest() == evidence.get('abstract_sha256')
            and bool(CACHED_REVIEW_SELF_DESCRIPTION.search(abstract))
            and not CACHED_REVIEW_MIXED_SCOPE.search(abstract + ' ' + evidence.get('title', ''))
            and not any(re.search(r'retract|concern|meta.analysis|systematic review|clinical trial|case reports|randomized', t, re.I)
                        for t in evidence.get('indexed_types', []))
            and any(s.get('receipt') == evidence.get('secondary_receipt')
                    and 'Review' in s.get('types', []) for s in secondary)
            and not any(re.search(r'retract|concern|meta.analysis|systematic review|clinical trial|case reports|randomized', t, re.I)
                        for s in secondary for t in s.get('types', [])))


def same_archive_missing_metadata(item: dict, precise: dict) -> bool:
    """Enrich an incomplete front record only from the same identified PMC article."""
    if (item.get('family') != 'archived_jats' or item.get('identity') != 'matched'
            or item.get('categories') or item.get('issns')
            or not precise.get('issns') or not native_named_research_format(precise)
            or not journal_research_format(precise)
            or key(item.get('kind')) != key(precise.get('kind'))
            or any(e.get('scope_review_required') or e.get('title_scope_flag')
                   for e in (item, precise))):
        return False
    journals = {key(v) for v in item.get('journal_titles', []) if key(v)}
    identifiers = []
    for source in (item, precise):
        location = str(source.get('url', '')) + ' ' + str(source.get('receipt', ''))
        identifiers.append(set(re.findall(r'\bPMC[0-9]+\b', location)))
    return bool(
        item.get('doi') and item['doi'] == precise.get('doi')
        and key(item.get('title')) and key(item.get('title')) == key(precise.get('title'))
        and journals and journals == {key(v) for v in precise.get('journal_titles', []) if key(v)}
        and len(identifiers[0]) == 1 and identifiers[0] == identifiers[1]
    )


def decide(evidence: list[dict], secondary: list[dict], duplicate_of: str = '') -> dict:
    if duplicate_of:
        version_proofs = [e for e in evidence if e.get('family') == 'repository_version_review'
                          and e.get('identity') == 'matched' and e.get('retained_work_id') == duplicate_of
                          and e.get('relation_type') == 'IsVariantFormOf'
                          and e.get('content_identity_confirmed') is True]
        if version_proofs:
            return {'status': 'excluded_duplicate_repository_version',
                    'reason': 'Repository-declared variant of retained publisher DOI; exact title/journal and matching abstract or complete author sequence; publisher identity corroborated',
                    'online_date': '', 'online_year': 0, 'type': 'repository_version',
                    'evidence_ids': [e['id'] for e in version_proofs]}
        return {'status':'excluded_duplicate_wrong_journal','reason':'Same DOI and title; publisher-deposited ISSNs identify the retained Work','online_date':'','online_year':0,'type':'','evidence_ids':[]}
    cached_reviews = [e for e in evidence if e.get('family') == 'cached_author_review_scope']
    evidence = [e for e in evidence if e.get('family') != 'cached_author_review_scope']
    # Any native source (including rejected identity) takes precedence over this fallback.
    accepted_reviews = [e for e in cached_reviews if cached_author_review(e, secondary)]
    if not evidence and accepted_reviews:
        return {'status': 'excluded_archived_author_review',
                'reason': 'Identity-matched MED Review index and archived author abstract explicitly describe a review; no primary type evidence or mixed-study signal; supplement/cached_author_review_scope/rule.json',
                'online_date': '', 'online_year': 0, 'type': 'Review (MED + archived author abstract)',
                'evidence_ids': [e['id'] for e in accepted_reviews]}
    primary=resolve_elife_publication_path(reviewed_study_design([e for e in evidence if e['identity']=='matched']))
    # CPB migrated records repeat the site's release timestamp across older articles.
    # Keep that raw evidence, but never use it to establish original online year.
    primary = [{**e, 'online_date': '', 'date_precision': 'unusable_site_migration_timestamp'}
               if e.get('family') == 'publisher_cpb_native'
               and e.get('native_online_dates') else e for e in primary]
    indexed_rule_used = any(nlm_indexed_research(e) for e in primary)
    primary = [{**e, 'class': 'research'} if nlm_indexed_research(e) else e for e in primary]
    virta_rule_used = any(virta_original_research(e) or ghent_original_research(e) or public_institution_original_research(e) for e in primary)
    ghent_rule_used = any(ghent_original_research(e) for e in primary)
    extra_institution_used = any(public_institution_original_research(e) for e in primary)
    primary = [{**e, 'class': 'research'} if virta_original_research(e) or ghent_original_research(e) or public_institution_original_research(e) else e for e in primary]
    primary = [{**e, 'class': 'nonprimary'} if literal_nonresearch_alias(e) else e for e in primary]
    nature_protocol_rule_used = any(nature_protocol_format(e) for e in primary)
    primary = refine_nature_protocols(primary)
    primary = refine_star_protocols(primary)
    star_protocol_rule_used = any(star_protocol_format(e) for e in primary)
    primary = [{**e, 'class': 'nonprimary'} if star_protocol_format(e) else e for e in primary]
    archived_notice = any(
        e.get('family') == 'archived_jats' and expression_notice(e)
        and e.get('separate_notice_rule') == 'native_related_article_object_of_concern'
        and (e.get('notice_target_dois') or e.get('notice_target_pmcids')) for e in primary)
    if archived_notice:
        primary = [{**e, 'class': 'nonprimary'} if e.get('class') == 'integrity_hold'
                   and expression_notice(e) else e for e in primary]
    primary = [{**e, 'class': frontiers_reviewed_format(e)} if frontiers_reviewed_format(e) else e for e in primary]
    primary = [{**e, 'class': 'subtype_review_required',
                'type_policy_receipt': 'supplement/frontiers_perspective_policy/review.json'}
               if frontiers_perspective_hold(e) else e for e in primary]
    primary = reviewed_frontiers_content(primary)
    primary = [{**e, 'class': 'research'} if mdpi_reviewed_communication(e) else e for e in primary]
    primary = [{**e, 'class': type_class(e.get('kind', ''), e['family'], e.get('categories'))}
               if e.get('family') == 'publisher_pensoft' and not e.get('scope_review_required')
               and not e.get('title_scope_flag') else e for e in primary]
    primary = [{**e, 'class': 'nonprimary'} if (science_china_chemistry_format(e) == 'nonprimary' or science_china_materials_format(e) == 'nonprimary') else e for e in primary]
    plos_notice = any(e.get('family') == 'publisher_plos' and expression_notice(e)
                      and plos_format_class(e) == 'nonprimary' for e in primary)
    plos_rule_used = any(plos_format_class(e) for e in primary)
    primary = [{**e, 'class': plos_format_class(e)} if plos_format_class(e) else e for e in primary]
    if plos_notice:
        primary = [{**e, 'class': 'nonprimary'} if e.get('class') == 'integrity_hold'
                   and expression_notice(e) else e for e in primary]
    format_rule_used = any(e.get('class') in {'subtype_review_required', 'unresolved_type'} and journal_research_format(e) for e in primary)
    format_rule_label = 'Brief Communication' if all(key(e.get('kind')) == 'briefcommunication' for e in primary if journal_research_format(e)) else 'research format'
    primary = [{**e, 'class': 'research'} if e.get('class') in {'subtype_review_required', 'unresolved_type'} and journal_research_format(e)
               else e for e in primary]
    named_formats = [e for e in primary if journal_research_format(e) and (
        native_named_research_format(e)
        or (e.get('journal_titles') == ['Cell Reports Methods']
            and e.get('categories') == ['Report']))]
    for index, item in enumerate(primary):
        if (item.get('family') != 'archived_jats' or item.get('categories')
                or item.get('class') not in {'subtype_review_required', 'unresolved_type'}):
            continue
        for precise in named_formats:
            pbj_same_identity = (
                not item.get('issns')
                and bool(set(precise.get('issns', [])) & {'1467-7644', '1467-7652'})
                and bool(item.get('doi')) and item.get('doi') == precise.get('doi')
                and bool(key(item.get('title'))) and key(item.get('title')) == key(precise.get('title'))
                and {key(v) for v in item.get('journal_titles', [])} == {'plantbiotechnologyjournal'}
                and {key(v) for v in precise.get('journal_titles', [])} == {'plantbiotechnologyjournal'}
            )
            spectrum_same_identity = (
                not item.get('issns') and spectrum_observation_format(precise)
                and bool(item.get('doi')) and item.get('doi') == precise.get('doi')
                and bool(key(item.get('title'))) and key(item.get('title')) == key(precise.get('title'))
                and {key(v) for v in item.get('journal_titles', [])} == {'microbiologyspectrum'}
                and not item.get('scope_review_required') and not item.get('title_scope_flag')
            )
            methods_same_identity = (
                not item.get('issns')
                and precise.get('journal_titles') == ['Cell Reports Methods']
                and item.get('journal_titles') == precise.get('journal_titles')
                and bool(set(precise.get('issns', [])) & {'2667-2375'})
                and bool(item.get('doi')) and item.get('doi') == precise.get('doi')
                and bool(key(item.get('title'))) and key(item.get('title')) == key(precise.get('title'))
                and not item.get('scope_review_required') and not item.get('title_scope_flag')
            )
            archive_same_identity = same_archive_missing_metadata(item, precise)
            refined = {**item, 'categories': precise['categories']}
            if archive_same_identity or pbj_same_identity or spectrum_same_identity or methods_same_identity:
                refined['issns'] = precise['issns']
            if (key(item.get('kind')) == key(precise.get('kind'))
                    and (archive_same_identity or pbj_same_identity or spectrum_same_identity or methods_same_identity or set(item.get('issns', [])) & set(precise.get('issns', [])))
                    and journal_research_format(refined)):
                primary[index] = {**item, 'class': 'research', 'format_refined_by_evidence_id': precise['id']}
                break
    primary = [{**e, "class": "research"} if reviewed_aps_scientific_correction(e) else e for e in primary]
    primary = [{**e, 'class': 'nonprimary'} if reviewed_aps_tutorial(e) else e for e in primary]
    reviewed_nature_meta = any(
        e.get('family')=='nature_v1' and key(e.get('kind'))=='article'
        and e.get('title_scope_flag') and re.search(r'\bmeta.analysis\b', e.get('title',''), re.I)
        and not re.search(r'\b(systematic review|study protocol|case report|case series|scoping review|narrative review|umbrella review)\b', e.get('title',''), re.I)
        and not e.get('scope_review_required') for e in primary)
    if reviewed_nature_meta:
        primary = [{**e, 'class':'research'} if e.get('class')=='subtype_review_required'
                   and e.get('title_scope_flag') and not e.get('scope_review_required')
                   and type_class(e.get('kind',''),e.get('family',''),e.get('categories'))=='research'
                   else e for e in primary]
    primary = reviewed_pnas_letter_content(adjudicate_pnas_magazine(primary))
    rejected=[e['id'] for e in evidence if e['identity']!='matched']
    result={'status':'pending_no_primary_type_date','reason':'No matched primary evidence','online_date':'','online_year':0,'type':'','evidence_ids':[e['id'] for e in primary], 'rejected_evidence_ids':rejected}
    if not primary:
        if rejected:
            result.update(status='pending_primary_identity_mismatch',reason='Available primary evidence fails exact title or journal identity')
        result['secondary_flags']=sorted({t for e in secondary for t in e['types'] if key(t) in NONPRIMARY|{'retractedpublication','expressionofconcern'}})
        if result['secondary_flags'] and not rejected:result.update(status='pending_secondary_nonprimary_or_integrity_flag',reason='Secondary exclusion flag requires primary confirmation')
        return result
    if (indexed_rule_used or virta_rule_used) and rejected:
        result.update(status='pending_primary_identity_mismatch',
                      reason=('Institutional classification does not resolve rejected native identity evidence'
                              if virta_rule_used else 'NLM alternative does not resolve rejected native identity evidence'))
        return result
    classes={e['class'] for e in primary}
    years={int(e['online_date'][:4]) for e in primary if re.match(r'^\d{4}(?:-|$)',e['online_date'])}
    result['type']=' | '.join(sorted({e['kind'] for e in primary}))
    if len(years)==1:
        result['online_year']=next(iter(years))
        result['online_date']=min([e['online_date'] for e in primary if e['online_date']],key=lambda s:(-len(s),s))
    if 'integrity_hold' in classes:
        result.update(status='pending_integrity_review',reason='Expression of concern or retracted-publication evidence');return result
    if any(pnas_frontmatter(e) and not e.get('magazine_adjudicated') for e in primary):
        result.update(status='pending_special_subtype_review',
                      reason='Native PNAS Front Matter category conflicts with generic article type; '
                             'research acceptance suspended; supplement/journal_type_policies/pnas_frontmatter_review.json')
        return result
    if 'nonprimary' in classes and 'research' in classes:
        result.update(status='pending_conflicting_primary_types',reason='Primary sources disagree on research versus nonprimary type');return result
    if 'nonprimary' in classes:
        result.update(status='excluded_primary_nonresearch',reason='Explicit primary nonresearch type' + ('; Nature Protocols scope: supplement/nature_protocol_policy/rule.json' if nature_protocol_rule_used else '') + ('; STAR Protocols native Protocol rule: supplement/star_protocol_policy/rule.json' if star_protocol_rule_used else '') + ('; journal-specific PLOS rule: supplement/plos_batch/type_policies/reviewed_rules.json' if plos_rule_used else ''));return result
    if len(years)>1:
        result.update(status='pending_conflicting_primary_online_years',reason='Explicit online years disagree');return result
    if years:
        year=next(iter(years));result['online_year']=year
        result['online_date']=min([e['online_date'] for e in primary if e['online_date']], key=lambda s:(-len(s),s))
        if year not in {2023,2024,2025}:
            result.update(status='excluded_primary_online_year_outside_window',reason='Explicit online year outside 2023-2025');return result
    reviewed_meta_receipts = {receipt for e in primary
                             if e.get('family') == 'reviewed_study_design' and e.get('class') == 'research'
                             for receipt in e.get('reviewed_secondary_meta_analysis_receipts', [])}
    reviewed_content_labels = {
        (item.get('receipt'), key(label))
        for e in primary
        if e.get('family') == 'reviewed_study_design' and e.get('class') == 'research'
        for item in e.get('reviewed_secondary_content_labels', [])
        if item.get('receipt') and item.get('original_research_component') and item.get('raw_sha256')
        for label in item.get('types', [])
        if key(label) in {'review', 'systematicreview', 'metaanalysis'}
    }
    secondary_hold=any(key(t) in NONPRIMARY|{'retractedpublication','expressionofconcern'}
                       and not (key(t) == 'metaanalysis' and e.get('receipt') in reviewed_meta_receipts)
                       and (e.get('receipt'), key(t)) not in reviewed_content_labels
                       for e in secondary for t in e['types'])
    if indexed_rule_used or virta_rule_used or any(e.get('review_protocol') in {'native_mdpi_brief_report_design_1', 'native_frontiers_methods_design_1'} for e in primary):
        secondary_hold = secondary_hold or any(
            key(t) in {'casereport', 'casereports', 'casestudy', 'studyprotocol', 'clinicaltrialprotocol',
                        'datapaper', 'datanote', 'datadescriptor', 'resourceannouncement', 'dataset', 'duplicatepublication'}
            for e in secondary for t in e['types'])
    if secondary_hold and 'research' in classes:
        result.update(status='pending_primary_secondary_type_conflict',reason='Research label conflicts with secondary nonprimary/integrity flag');return result
    research_evidence = [e for e in primary if e['class'] == 'research']
    generic_archive_only = bool(research_evidence) and all(
        e.get('family') == 'archived_jats'
        and key(e.get('kind')) == 'researcharticle'
        and {key(v) for v in e.get('categories', [])} <= {'article'}
        and not journal_research_format(e) for e in research_evidence)
    if generic_archive_only and any(key(t) == 'letter' for e in secondary for t in e['types']):
        result.update(status='pending_primary_secondary_type_conflict',
                      reason='Generic archived research label conflicts with indexed Letter; specific original-research subtype is unconfirmed')
        return result
    if reviewed_case_report(primary):
        if secondary_hold:
            result.update(status='pending_primary_secondary_type_conflict',
                          reason='Case report scope review retains secondary type/integrity conflict')
        else:
            result.update(status='excluded_case_report_main_scope',
                          reason='User-defined main comparison excludes explicit native case reports; '
                                 'retain in unified data for extension comparison subject to year and metadata checks; '
                                 'supplement/case_report_scope/rule.json')
        return result
    if reviewed_data_note(primary):
        if secondary_hold:
            result.update(status='pending_primary_secondary_type_conflict',
                          reason='Data Note scope review retains secondary type/integrity conflict')
        else:
            result.update(status='excluded_data_note_main_scope',
                          reason='User-defined main comparison excludes explicit native Data Notes in Wellcome Open Research and BMC Research Notes; '
                                 'retain in unified data for extension comparison subject to year and metadata checks; '
                                 'supplement/data_note_scope/rule.json')
        return result
    if reviewed_resource_announcement(primary):
        if secondary_hold:
            result.update(status='pending_primary_secondary_type_conflict',
                          reason='Resource scope review retains secondary type/integrity conflict')
        else:
            result.update(status='excluded_resource_announcement_main_scope',
                          reason='User-defined main comparison excludes original data resource announcements; '
                                 'retain in unified data for extension comparison, subject to online-year and metadata checks; '
                                 'supplement/mra_resource_scope/rule.json')
        return result
    if 'subtype_review_required' in classes:
        result.update(status='pending_special_subtype_review',reason='Protocol/data/case/analysis/short-format scope requires content-specific adjudication');return result
    if 'research' in classes and any(springer_special_originalpaper(e) for e in primary):
        result.update(status='pending_special_subtype_review',
                      reason='Springer OriginalPaper technical label coexists with a special native format; '
                             'original-research acceptance suspended pending format/content review; '
                             'supplement/springer_special_originalpaper/review.json')
        return result
    if 'research' in classes and years:
        result.update(status='included_verified_research',reason='Matched explicit research type and online year' + ('; Public institutional explicit original-research subtype, source-specific rule and curation; not publisher type verification; electronic year independently sourced; supplement/target_500k/muenster_public/rule.json and supplement/target_500k/slovenia_public/rule.json' if extra_institution_used else ('; Ghent public bibliography explicit original subtype in published peer-reviewed journal article (A1/A2/A3); institution-reported evidence, not publisher type verification; online year independently sourced; supplement/target_500k/ghent_public/rule.json' if ghent_rule_used else ('; Finnish official VIRTA approved A1 original-article classification, institution-reported evidence, not publisher type verification; online year independently sourced; supplement/target_500k/virta_public/rule.json' if virta_rule_used else ''))) + ('; user-authorized NLM/PubMed explicit study type and electronic date via Europe PMC MED; secondary indexing evidence, not publisher type verification; supplement/target_500k/explicit_indexed_research_types/rule.json' if indexed_rule_used else '') + ('; preserved previously reviewed Nature original-research scope (nature/papers.jsonl), title meta-analysis alone is not new contrary evidence' if reviewed_nature_meta else '') + (f'; journal-specific {format_rule_label} rule: supplement/journal_type_policies/research_formats.json' if format_rule_used else '') + ('; journal-specific PLOS rule: supplement/plos_batch/type_policies/reviewed_rules.json' if plos_rule_used else ''));return result
    if 'research' in classes:
        result.update(status='pending_primary_online_date',reason='Research type confirmed; no explicit consistent online year');return result
    result.update(status='pending_primary_type',reason='Primary evidence available but no accepted research subtype')
    return result

def imprint_dates(dates: list[dict]) -> list[str]:
    """Full publication dates only; these do not independently establish online dates."""
    result=set()
    for item in dates:
        attrs=item.get('attributes',{})
        if attrs.get('pub-type') not in {None,'ppub','epub','epub-ppub'}:continue
        if attrs.get('date-type') not in {None,'pub','publication'}:continue
        if not all(item.get(k) for k in ['year','month','day']):continue
        try:value=date(int(item['year']),int(item['month']),int(item['day']))
        except (ValueError,TypeError):continue
        result.add(value.isoformat())
    return sorted(result)


def portfolio_label(inherited: str, evidence: list[dict], year: int) -> tuple[str,list[str]]:
    """Dated archived imprint or own-publisher records; no inference from missing catalogue rows."""
    nonportfolio=[]
    online_dates={e.get('online_date','') for e in evidence if e.get('identity')=='matched' and e.get('online_date','').startswith(str(year)) and len(e.get('online_date',''))==10}
    terms=['elsevier','wiley','american physical society','american chemical society',
           'royal society of chemistry','iop publishing','institute of physics','frontiers',
           'mdpi','multidisciplinary digital','oxford university press','optica','edp sciences',
           'public library of science','plos','american institute of physics','aip publishing',
           'american association for the advancement of science','national academy of sciences']
    for item in evidence:
        if item['identity']!='matched':continue
        dated=item.get('online_date','').startswith(str(year))
        dates=item.get('publication_dates',[])
        imprints={d for d in dates if isinstance(d,str)}
        imprints.update(imprint_dates([d for d in dates if isinstance(d,dict)]))
        corroborated=item['family']=='archived_jats' and bool(imprints & online_dates)
        if not dated and not corroborated:continue
        publisher=' '.join(item.get('publishers',[])).casefold()
        explicit_statement = (item.get('publisher_statement_rule') == 'Explicit dated named published-by statement; copyright holder alone insufficient'
                              and bool(item.get('publisher_statement'))
                              and publisher in {'american association for cancer research', 'wolters kluwer', 's. karger', 'bmj'})
        if item['family']=='aps' or (item['family']=='archived_jats' and (any(t in publisher for t in terms) or explicit_statement)):
            nonportfolio.append(item['id'])
            if corroborated and not dated:
                nonportfolio.extend(e['id'] for e in evidence if e.get('identity')=='matched' and e.get('online_date') in imprints & online_dates)
    if inherited=='True' and nonportfolio:return 'unresolved',nonportfolio
    if inherited in {'True','False'}:return inherited,[]
    return ('False',nonportfolio) if nonportfolio else ('unresolved',[])

