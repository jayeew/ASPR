"""Validate cloud version metadata before acquiring a published JATS record."""
from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse

from .fig8b_qualification_decide import key


def assess_version(metadata: dict, paper: dict) -> dict:
    if (metadata.get('pmcid') != paper['pmcid']
            or str(metadata.get('doi', '')).lower() != paper['doi'].lower()
            or not key(paper['title']) or key(metadata.get('title')) != key(paper['title'])):
        return {'status': 'metadata_identity_mismatch'}
    if metadata.get('is_retracted') is not False:
        return {'status': 'retracted_or_integrity_status_unconfirmed'}
    if metadata.get('is_manuscript') is True:
        return {'status': 'author_manuscript_retained_separate_review',
                'reason': 'Do not substitute manuscript metadata for final publication type or date'}
    if metadata.get('is_manuscript') is not False or metadata.get('is_historical_ocr') is not False:
        return {'status': 'publication_version_unconfirmed'}
    if metadata.get('is_pmc_openaccess') is not True:
        return {'status': 'distribution_metadata_unconfirmed'}
    url = urlparse(metadata.get('xml_url', ''))
    version = metadata.get('version')
    expected = f"/{paper['pmcid']}.{version}/{paper['pmcid']}.{version}.xml"
    if url.scheme != 's3' or url.netloc != 'pmc-oa-opendata' or url.path != expected:
        return {'status': 'xml_location_identity_conflict'}
    checksums = parse_qs(url.query).get('md5', [])
    if len(checksums) != 1 or not re.fullmatch(r'[0-9a-fA-F]{32}', checksums[0]):
        return {'status': 'xml_checksum_missing_or_invalid'}
    return {'status': 'published_xml_candidate_identity_check_required',
            'url': 'https://pmc-oa-opendata.s3.amazonaws.com' + url.path,
            'md5': checksums[0].lower(), 'license_code': metadata.get('license_code'),
            'license_code_status': ('provided' if metadata.get('license_code')
                                    else 'unspecified_preserve_article_permissions'),
            'version': version, 'qualification_implied': False}
