#!/usr/bin/env python3
"""Upload a verified release and activate it after the web app supports its policy."""
import argparse
import json
import os
from pathlib import Path
import sys
from urllib.parse import urlparse
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import boto3
from gds_viewer.cloudshell_check import resolve_connection
from gds_viewer.curve_policy import GLOBAL_POLICY
from gds_viewer.data import Repository, LocalStore
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--payload', required=True, type=Path)
    p.add_argument('--release', default='v002')
    p.add_argument('--app-url', required=True, help='Current public HTTPS URL of your Heroku app')
    args = p.parse_args()
    manifest, _ = load_manifest(args.payload, args.release)
    if manifest.get('curve_selection') != GLOBAL_POLICY:
        raise ValueError('This deploy command expects a global curve type release')
    repo = Repository(LocalStore(args.payload), args.release)
    snap = repo.active()
    for rid in snap.records:
        table = repo.table(snap, rid)
        for c in snap.curves.values():
            if c['record_id'] == rid and c['parquet_column'] not in table.column_names:
                raise ValueError('Missing Parquet column: ' + c['log_id'])
    print(f'Local release verified: {len(snap.records)} wells, {len(snap.curves)} curves', flush=True)
    web_status(args.app_url)
    bucket, region, _ = resolve_connection(os.environ, Path(__file__).resolve().parents[1], Path.home(), boto3.Session())
    print(f'Using existing storage: {bucket} ({region})', flush=True)
    store = Store(bucket, region)
    store.upload(args.payload, args.release)
    # activate() verifies every remote SHA256 before switching current.json.
    store.activate(args.payload, args.release)
    result = web_status(args.app_url)
    if result.get('release') != args.release or result.get('curves') != len(snap.curves):
        raise ValueError('Release activated, but web readiness differs. Reload the app and inspect /readyz; do not re-upload')
    print(json.dumps(result, indent=2))
    print('DONE. Reload the dataset in the browser. Kurdistan-1 now contains GR and DT.')


if __name__ == '__main__':
    main()
