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
from zipfile import is_zipfile

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


def upload_batch(store, batch, dry_run=False):
    """Preflight a local file/Reference list, archive all originals and link PDFs."""
    config = json.loads(batch.read_text(encoding='utf-8-sig'))
    if config.get('schema_version') != 1 or not isinstance(config.get('files'), list) or not config['files']:
        raise ValueError('Batch requires schema_version 1 and a nonempty files list')
    root = batch.resolve().parent
    release, references = active_references(store)
    objects, seen_paths, linked = {}, set(), set()
    for entry in config['files']:
        if not isinstance(entry, dict):
            raise ValueError('Each batch entry must be an object')
        relative = entry['file']
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise ValueError('Batch file paths must be relative')
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or path in seen_paths:
            raise ValueError('Missing, duplicate or unsafe batch file: ' + relative)
        seen_paths.add(path)
        suffix = path.suffix.lower()
        if suffix not in {'.pdf', '.pptx'}:
            raise ValueError('Batch supports PDF and PPTX originals: ' + relative)
        data = path.read_bytes()
        if (suffix == '.pdf' and not data.startswith(b'%PDF-')) or (suffix == '.pptx' and not is_zipfile(path)):
            raise ValueError('Invalid document: ' + relative)
        digest = sha(data)
        if entry.get('sha256') != digest:
            raise ValueError('File checksum mismatch: ' + relative)
        reference = entry.get('reference', '')
        if reference:
            canonical = references.get(reference_key(reference))
            if canonical is None:
                raise ValueError('Reference is absent from the existing CSV: ' + reference)
            if suffix != '.pdf' or reference_key(canonical) in linked:
                raise ValueError('Each Reference must have exactly one PDF: ' + reference)
            linked.add(reference_key(canonical))
            objects[pdf_key(canonical)] = (path, digest, len(data))
        else:
            original_key = 'publications/originals/' + digest + suffix
            objects.setdefault(original_key, (path, digest, len(data)))
    # Check every existing destination before the first write. Never overwrite
    # a different article silently; retries skip already uploaded identical bytes.
    pending = []
    for key, (path, digest, size) in objects.items():
        current, _ = store.get(key)
        if current is not None and sha(current) != digest:
            raise ValueError('Existing PDF has different bytes: ' + key + '. Use the single-PDF command for an intentional replacement.')
        if current is None:
            pending.append((key, path, digest))
    plan = {'mode': 'DRY RUN' if dry_run else 'UPLOAD', 'release': release,
            'files': len(seen_paths), 'linked_references': len(linked),
            'new_objects': len(pending), 'unchanged_objects': len(objects) - len(pending),
            'bucket': store.bucket, 'existing_csv_links_unchanged': True}
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if not dry_run:
        store.check_bucket()
        for number, (key, path, digest) in enumerate(pending, 1):
            data = path.read_bytes()
            if sha(data) != digest:
                raise ValueError('File changed after preflight: ' + str(path))
            store.put_new(key, data)
            print(f'[{number}/{len(pending)}] {path.name}', flush=True)
        print(f'Done: {len(seen_paths)} originals stored; {len(linked)} existing References have PDF downloads.', flush=True)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pdf', type=Path, help='PDF file to upload')
    parser.add_argument('--reference', help='Existing Reference copied from Publications / CSV')
    parser.add_argument('--batch', type=Path, help='Prepared batch JSON: all originals and existing PDF References')
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
    if args.batch and (args.pdf or args.reference or args.list_references):
        parser.error('--batch cannot be combined with --pdf, --reference or --list-references')
    if not args.list_references and not args.batch and (args.pdf is None or not args.reference):
        parser.error('Provide --batch FILE.json, or --pdf and --reference')
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
    if args.batch:
        upload_batch(store, args.batch.resolve(), args.dry_run)
    else:
        upload_publication(store, args.pdf.resolve(), args.reference, args.dry_run)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, FileNotFoundError, ClientError) as error:
        print('ERROR: ' + str(error), file=sys.stderr)
        sys.exit(1)
