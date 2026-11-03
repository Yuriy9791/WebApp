#!/usr/bin/env python3
"""Upload one source PDF and its explicit log bindings; dry-run by default."""
import argparse
import base64
import csv
import hashlib
from io import StringIO
import json
from pathlib import Path
import re
import sys
import os

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import boto3
from botocore.exceptions import ClientError
from gds_viewer.cloudshell_check import resolve_connection
from gds_viewer.publications import CATALOG_KEY, decode_catalog, catalog_bytes
from gds_s3 import Store, sha


def active_metadata(store):
    pointer_raw, _ = store.get('current.json')
    if pointer_raw is None:
        raise ValueError('No active data release')
    pointer = json.loads(pointer_raw)
    release = pointer['release_id']
    if not re.fullmatch(r'v[0-9]{3,}', release) or pointer['manifest_key'] != f'releases/{release}/manifest.json':
        raise ValueError('Invalid active release pointer')
    raw, _ = store.get(pointer['manifest_key'])
    if raw is None or sha(raw) != pointer['manifest_sha256']:
        raise ValueError('Active manifest checksum mismatch')
    manifest = json.loads(raw)
    if manifest['schema_version'] != 1 or manifest['release_id'] != release:
        raise ValueError('Unsupported data manifest')
    entries = {entry['key']: entry for entry in manifest['files']}

    def read_csv(relative):
        key = f'releases/{release}/{relative}'
        entry = entries[key]
        data, _ = store.get(key)
        if data is None or len(data) != entry['size'] or sha(data) != entry['sha256']:
            raise ValueError('Metadata checksum mismatch: ' + relative)
        return list(csv.DictReader(StringIO(data.decode('utf-8-sig'))))

    return release, read_csv('metadata/records.csv'), read_csv('metadata/curves.csv')


def prepare_paper(input_file, records, curves):
    spec = json.loads(input_file.read_text(encoding='utf-8-sig'))
    pdf = (input_file.parent / spec['pdf_file']).resolve()
    data = pdf.read_bytes()
    if not pdf.name.lower().endswith('.pdf') or not data.startswith(b'%PDF-'):
        raise ValueError('Input must be a PDF file')
    title = spec['title'].strip()
    if not title or not isinstance(spec.get('links'), list) or not spec['links']:
        raise ValueError('Provide a title and explicit log bindings')
    links = {}
    summary = []
    for binding in spec['links']:
        candidates = [r for r in records if r['las_filename'] == binding['las_filename']]
        requested = binding.get('curves')
        if len(candidates) != 1 or not isinstance(requested, list) or not requested:
            raise ValueError('Specify one exact LAS filename and a nonempty curves list')
        record = candidates[0]
        matched = [c for c in curves if c['record_id'] == record['record_id']
                   and c['available_for_plot'].lower() == 'true' and c['mnemonic'] in requested]
        if {c['mnemonic'] for c in matched} != set(requested):
            raise ValueError('Unknown/unavailable curve in ' + binding['las_filename'])
        for curve in matched:
            links[curve['log_id']] = {'log_id': curve['log_id'], 'las_sha256': record['enriched_sha256']}
        summary.append({'las_filename': record['las_filename'], 'curves': requested})
    digest = hashlib.sha256(data).hexdigest()
    paper = {'publication_id': 'paper-' + digest[:16], 'title': title, 'filename': pdf.name,
             'key': f'publications/files/{digest}.pdf', 'sha256': digest, 'size': len(data),
             'links': [links[k] for k in sorted(links)]}
    decode_catalog(catalog_bytes({'schema_version': 1, 'papers': [paper]}))
    return paper, data, summary


def upload_publication(store, input_file, apply=False):
    release, records, curves = active_metadata(store)
    paper, data, bindings = prepare_paper(input_file, records, curves)
    old, etag = store.get(CATALOG_KEY)
    catalog = decode_catalog(old)
    existing = next((p for p in catalog['papers'] if p['publication_id'] == paper['publication_id']), None)
    if existing:
        if existing['sha256'] != paper['sha256']:
            raise ValueError('Publication identifier collision')
        merged = {(x['log_id'], x['las_sha256']): x for x in existing['links'] + paper['links']}
        paper['links'] = [merged[k] for k in sorted(merged)]
    catalog['papers'] = [p for p in catalog['papers'] if p['publication_id'] != paper['publication_id']] + [paper]
    catalog['papers'].sort(key=lambda p: p['publication_id'])
    raw = catalog_bytes(catalog)
    plan = {'mode': 'APPLY' if apply else 'DRY RUN', 'data_release': release,
            'publication_id': paper['publication_id'], 'title': paper['title'],
            'pdf_bytes': len(data), 'bindings': bindings, 'bucket': store.bucket,
            'region': store.region, 'data_release_unchanged': True}
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if not apply:
        return plan
    store.check_bucket()
    store.put_new(paper['key'], data)
    if old != raw:
        store.s3.put_object(Key=CATALOG_KEY, Body=raw, ContentType='application/json',
            CacheControl='no-cache, max-age=0', ServerSideEncryption='AES256',
            ChecksumSHA256=base64.b64encode(hashlib.sha256(raw).digest()).decode(),
            **({'IfMatch': etag} if etag else {'IfNoneMatch': '*'}), **store.base)
    if store.get(CATALOG_KEY)[0] != raw:
        raise ValueError('Publication catalog changed concurrently; inspect it before retrying')
    print('Source PDF uploaded and linked. Reload the app and click Build logs.', flush=True)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, help='JSON file containing pdf_file, title and links')
    parser.add_argument('--apply', action='store_true', help='Upload after reviewing the dry run')
    parser.add_argument('--bucket')
    parser.add_argument('--region')
    parser.add_argument('--profile', help='Existing local AWS profile; omit in CloudShell')
    parser.add_argument('--print-read-policy', action='store_true', help='Print the additional Heroku reader policy')
    args = parser.parse_args()
    if args.print_read_policy:
        if not args.bucket or not re.fullmatch(r'[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]', args.bucket):
            parser.error('--print-read-policy requires --bucket')
        print(json.dumps({'Version': '2012-10-17', 'Statement': [
            {'Sid': 'ReadGDSSourcePDFs', 'Effect': 'Allow', 'Action': ['s3:GetObject'],
             'Resource': [f'arn:aws:s3:::{args.bucket}/publications/*']}]}, indent=2))
        return
    if args.input is None:
        parser.error('--input is required')
    session = boto3.Session(profile_name=args.profile) if args.profile else boto3.Session()
    environ = dict(os.environ)
    if args.bucket:
        environ['GDS_S3_BUCKET'] = args.bucket
    bucket, region, _ = resolve_connection(environ, Path(__file__).resolve().parents[1], Path.home(), session)
    store = Store(bucket, args.region or region, session=session)
    upload_publication(store, args.input.resolve(), args.apply)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, FileNotFoundError, ClientError) as error:
        print('ERROR: ' + str(error), file=sys.stderr)
        sys.exit(1)
