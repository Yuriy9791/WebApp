#!/usr/bin/env python3
"""Upload a verified release and activate it after the web app supports its policy."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
from urllib.parse import urlparse
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import boto3
from gds_viewer.cloudshell_check import resolve_connection
from gds_viewer.curve_policy import GLOBAL_POLICY, curve_types, plottable
from gds_s3 import Store, load_manifest


def web_status(app_url):
    parsed = urlparse(app_url)
    if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('Use the HTTPS application URL copied from your browser')
    with urlopen(app_url.rstrip('/') + '/readyz', timeout=30) as response:
        result = json.load(response)
    if result.get('supported_curve_policy') != GLOBAL_POLICY:
        raise ValueError('Deploy the updated gds-viewer-iteration-2 branch to Heroku first; /readyz does not support this catalog yet')
    return result


def validate_payload(payload, release):
    # The build already checked Parquet against LAS. Deployment verifies every
    # byte by SHA256 instead of installing an analytics runtime in CloudShell.
    manifest, _ = load_manifest(payload, release)
    if manifest.get('curve_selection') != GLOBAL_POLICY:
        raise ValueError('This deploy command expects a global curve type release')
    catalog = payload / 'releases' / release / 'metadata'

    def read(name):
        with (catalog / name).open(encoding='utf-8-sig', newline='') as stream:
            return list(csv.DictReader(stream))

    records = read('records.csv')
    curves = read('curves.csv')
    allowed = curve_types(read('curve_types_legacy.csv'))
    ids = {r['record_id'] for r in records}
    if len(ids) != len(records) or len(records) != manifest['counts']['las_files']:
        raise ValueError('Invalid record count or duplicate records')
    if len({c['log_id'] for c in curves}) != len(curves):
        raise ValueError('Duplicate curve IDs')
    files = {f['key'] for f in manifest['files']}
    for r in records:
        for path in (r['las_path'], r['parquet_path']):
            if f'releases/{release}/{path}' not in files:
                raise ValueError('Missing manifest member: ' + path)
    selected = []
    for c in curves:
        available = c['available_for_plot'].lower() == 'true'
        if c['record_id'] not in ids or available != plottable(c, allowed):
            raise ValueError('Orphan curve or incomplete global type catalog')
        if available:
            if not c['parquet_column']:
                raise ValueError('Missing Parquet mapping: ' + c['log_id'])
            selected.append(c)
    if not allowed or len(selected) != manifest['counts']['selected_curve_columns']:
        raise ValueError('Invalid selected curve count')
    return len(records), len(selected)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--payload', required=True, type=Path)
    p.add_argument('--release', default='v002')
    p.add_argument('--app-url', required=True, help='Current public HTTPS URL of your Heroku app')
    args = p.parse_args()
    records, curves = validate_payload(args.payload, args.release)
    print(f'Local release verified: {records} wells, {curves} curves', flush=True)
    web_status(args.app_url)
    bucket, region, _ = resolve_connection(os.environ, Path(__file__).resolve().parents[1], Path.home(), boto3.Session())
    print(f'Using existing storage: {bucket} ({region})', flush=True)
    store = Store(bucket, region)
    store.upload(args.payload, args.release)
    # activate() verifies every remote SHA256 before switching current.json.
    store.activate(args.payload, args.release)
    result = web_status(args.app_url)
    if result.get('release') != args.release or result.get('curves') != curves:
        raise ValueError('Release activated, but web readiness differs. Reload the app and inspect /readyz; do not re-upload')
    print(json.dumps(result, indent=2))
    print('DONE. Reload the dataset in the browser. Kurdistan-1 now contains GR and DT.')


if __name__ == '__main__':
    main()
