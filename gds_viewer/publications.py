"""Small source-PDF catalog shared by the viewer and the boto3-only uploader."""
import json
import re

CATALOG_KEY = 'publications/catalog.json'


def decode_catalog(raw):
    if raw is None:
        return {'schema_version': 1, 'papers': []}
    catalog = json.loads(raw)
    if not isinstance(catalog, dict) or catalog.get('schema_version') != 1 or not isinstance(catalog.get('papers'), list):
        raise ValueError('Unsupported source-PDF catalog')
    seen = set()
    for paper in catalog['papers']:
        if not isinstance(paper, dict):
            raise ValueError('Invalid source-PDF catalog entry')
        digest = paper.get('sha256', '')
        pid = paper.get('publication_id', '')
        if (not re.fullmatch(r'[0-9a-f]{64}', digest)
                or pid != 'paper-' + digest[:16] or pid in seen
                or paper.get('key') != f'publications/files/{digest}.pdf'
                or not isinstance(paper.get('title'), str) or not paper['title'].strip()
                or not isinstance(paper.get('filename'), str)
                or not paper['filename'].lower().endswith('.pdf')
                or '/' in paper['filename'] or '\\' in paper['filename']
                or type(paper.get('size')) is not int or paper['size'] <= 0
                or not isinstance(paper.get('links'), list) or not paper['links']):
            raise ValueError('Invalid source-PDF catalog entry')
        for link in paper['links']:
            if (not isinstance(link, dict) or not isinstance(link.get('log_id'), str) or not link['log_id']
                    or not re.fullmatch(r'[0-9a-f]{64}', link.get('las_sha256', ''))):
                raise ValueError('Invalid source-PDF log binding')
        seen.add(pid)
    return catalog


def catalog_bytes(value):
    raw = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n').encode('utf-8')
    decode_catalog(raw)
    return raw


def select_papers(catalog, fingerprints):
    result = []
    for paper in catalog['papers']:
        matched = [link['log_id'] for link in paper['links']
                   if fingerprints.get(link['log_id']) == link['las_sha256']]
        if matched:
            result.append(dict(paper, selected_log_ids=list(dict.fromkeys(matched))))
    return result
