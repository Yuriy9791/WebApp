#!/usr/bin/env python3
"""Upload a PDF for an existing CSV Reference; log/publication links stay in the CSV."""
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
from gds_viewer.publications import existing_references, reference_key, pdf_key
from gds_s3 import Store, sha


def active_references(store):
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

    return release, existing_references(read_csv('metadata/records.csv'), read_csv('metadata/metadata_preserved.csv'))


def upload_publication(store, pdf, reference, dry_run=False):
    release, references = active_references(store)
    canonical = references.get(reference_key(reference))
    if canonical is None:
        raise ValueError('Reference is absent from the existing CSV. Use --list-references to copy its exact value.')
    data = pdf.read_bytes()
    if not pdf.name.lower().endswith('.pdf') or not data.startswith(b'%PDF-'):
        raise ValueError('Input must be a PDF file')
    digest = hashlib.sha256(data).hexdigest()
    plan = {'mode': 'DRY RUN' if dry_run else 'UPLOAD', 'reference': canonical,
            'pdf': str(pdf), 'pdf_bytes': len(data), 'bucket': store.bucket,
            'key': pdf_key(canonical), 'existing_csv_links_unchanged': True}
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if not dry_run:
        store.check_bucket()
        store.s3.put_object(Key=pdf_key(canonical), Body=data, ContentType='application/pdf',
            ServerSideEncryption='AES256', Metadata={'sha256': digest},
            ChecksumSHA256=base64.b64encode(hashlib.sha256(data).digest()).decode(), **store.base)
        print('PDF uploaded. Its existing Reference in Publications is now the download link.', flush=True)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pdf', type=Path, help='PDF file to upload')
    parser.add_argument('--reference', help='Existing Reference copied from Publications / CSV')
    parser.add_argument('--list-references', action='store_true', help='List existing CSV references')
    parser.add_argument('--dry-run', action='store_true', help='Show the destination without uploading')
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
    if not args.list_references and (args.pdf is None or not args.reference):
        parser.error('Provide --pdf and --reference')
    session = boto3.Session(profile_name=args.profile) if args.profile else boto3.Session()
    environ = dict(os.environ)
    if args.bucket:
        environ['GDS_S3_BUCKET'] = args.bucket
    bucket, region, _ = resolve_connection(environ, Path(__file__).resolve().parents[1], Path.home(), session)
    store = Store(bucket, args.region or region, session=session)
    if args.list_references:
        _, references = active_references(store)
        for reference in references.values():
            print(reference)
        return
    upload_publication(store, args.pdf.resolve(), args.reference, args.dry_run)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, FileNotFoundError, ClientError) as error:
        print('ERROR: ' + str(error), file=sys.stderr)
        sys.exit(1)
