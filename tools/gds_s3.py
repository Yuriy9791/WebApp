#!/usr/bin/env python3
"""GDS S3 release manager. No AWS keys, deletion or ACLs in this package."""
import argparse
import base64
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import re
import sys
from datetime import datetime, timezone

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n').encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def missing(error):
    return error.response['Error']['Code'] in {'404', 'NoSuchKey', 'NoSuchBucket', 'NotFound'}


def load_manifest(payload, release):
    mpath = payload / 'releases' / release / 'manifest.json'
    raw = mpath.read_bytes()
    manifest = json.loads(raw)
    require(manifest['schema_version'] == 1 and manifest['release_id'] == release, 'Invalid manifest')
    require(bool(re.fullmatch(r'batch-[0-9]{3,}', manifest['source_batch'])), 'Invalid source batch')
    seen = set()
    for item in manifest['files']:
        key = item['key']
        require(key not in seen, f'Duplicate key: {key}')
        seen.add(key)
        require(key.startswith((f'releases/{release}/', f"sources/{manifest['source_batch']}/")), f'Invalid key: {key}')
        path = payload / key
        require(path.resolve().is_relative_to(payload.resolve()) and not path.is_symlink(), f'Unsafe path: {key}')
        require(path.is_file() and path.stat().st_size == item['size'] and file_sha(path) == item['sha256'], f'Local hash/size mismatch: {key}')
    actual = {p.relative_to(payload).as_posix() for p in (payload / 'releases' / release).rglob('*') if p.is_file()}
    expected = {k for k in seen if k.startswith(f'releases/{release}/')} | {f'releases/{release}/manifest.json'}
    require(actual == expected, 'Release contains missing or unlisted local files')
    return manifest, raw


class Store:
    def __init__(self, bucket=None, region=None, session=None):
        self.session = session or boto3.Session(region_name=region)
        self.region = region or self.session.region_name
        require(self.region, 'AWS region not found. Use --region REGION or set AWS_REGION in CloudShell.')
        sts = self.session.client('sts', region_name=self.region)
        identity = sts.get_caller_identity()
        self.account = identity['Account']
        self.partition = identity['Arn'].split(':')[1]
        self.bucket = bucket or f'gds-viewer-data-{self.account}-{self.region}'
        require(bool(re.fullmatch(r'[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]', self.bucket)), 'Invalid bucket name')
        # The four previous application buckets must never be targeted by this kit.
        require(self.bucket not in {'for-metadata', 'transformed-for-visualization-data-1',
                                   'transformed-for-download-data', 'raw-from-published-data'}, 'Use a NEW GDS bucket')
        self.s3 = self.session.client('s3', region_name=self.region,
                                      config=Config(retries={'mode': 'standard', 'max_attempts': 5}))
        self.base = {'Bucket': self.bucket, 'ExpectedBucketOwner': self.account}

    def get(self, key):
        try:
            response = self.s3.get_object(Key=key, **self.base)
            try:
                return response['Body'].read(), response['ETag']
            finally:
                response['Body'].close()
        except ClientError as e:
            if missing(e):
                return None, None
            raise

    def put_new(self, key, body):
        current, _ = self.get(key)
        if current is not None:
            require(sha(current) == sha(body), f'Existing object has different bytes: {key}. Use a new release/batch.')
            return False
        args = dict(self.base, Key=key, Body=body, IfNoneMatch='*',
                    ServerSideEncryption='AES256', ChecksumSHA256=base64.b64encode(hashlib.sha256(body).digest()).decode(),
                    ContentType=mimetypes.guess_type(key)[0] or 'application/octet-stream',
                    Metadata={'sha256': sha(body)})
        try:
            self.s3.put_object(**args)
        except ClientError as e:
            if e.response['Error']['Code'] not in {'PreconditionFailed', '412', 'ConditionalRequestConflict', '409'}:
                raise
            current, _ = self.get(key)
            require(current is not None and sha(current) == sha(body), f'Concurrent upload conflict: {key}. Rerun after checking.')
        return True

    def keys(self, prefix):
        found = set()
        for page in self.s3.get_paginator('list_objects_v2').paginate(Prefix=prefix, **self.base):
            found.update(x['Key'] for x in page.get('Contents', []))
        return found

    def check_bucket(self):
        self.s3.head_bucket(**self.base)
        location = self.s3.get_bucket_location(**self.base).get('LocationConstraint') or 'us-east-1'
        if location == 'EU':
            location = 'eu-west-1'
        require(location == self.region, f'Bucket region is {location}; use --region {location}')
        version = self.s3.get_bucket_versioning(**self.base).get('Status')
        require(version == 'Enabled', 'S3 Versioning is not enabled. Run 01_create_storage.sh.')
        block = self.s3.get_public_access_block(**self.base)['PublicAccessBlockConfiguration']
        require(all(block.get(k) for k in ['BlockPublicAcls', 'IgnorePublicAcls', 'BlockPublicPolicy', 'RestrictPublicBuckets']), 'Public access block must be enabled')
        marker, _ = self.get('gds-storage.json')
        require(marker is not None and json.loads(marker).get('application') == 'GDS-Viewer', 'Bucket is not initialized by this kit')

    def provision(self, output):
        exists = True
        try:
            self.s3.head_bucket(**self.base)
        except ClientError as e:
            if not missing(e):
                raise
            exists = False
        if not exists:
            args = {'Bucket': self.bucket, 'ObjectOwnership': 'BucketOwnerEnforced'}
            if self.region != 'us-east-1':
                args['CreateBucketConfiguration'] = {'LocationConstraint': self.region}
            self.s3.create_bucket(**args)
            self.s3.get_waiter('bucket_exists').wait(**self.base)
        else:
            location = self.s3.get_bucket_location(**self.base).get('LocationConstraint') or 'us-east-1'
            require(('eu-west-1' if location == 'EU' else location) == self.region, 'Bucket exists in a different region')
            marker, _ = self.get('gds-storage.json')
            if marker is None:
                require(not self.keys(''), 'Existing bucket is not empty and is not a GDS bucket. Choose a new name.')
            else:
                require(json.loads(marker).get('application') == 'GDS-Viewer', 'Bucket marker belongs to another application')
        self.s3.put_public_access_block(PublicAccessBlockConfiguration={
            'BlockPublicAcls': True, 'IgnorePublicAcls': True, 'BlockPublicPolicy': True, 'RestrictPublicBuckets': True}, **self.base)
        self.s3.put_bucket_versioning(VersioningConfiguration={'Status': 'Enabled'}, **self.base)
        self.s3.put_bucket_encryption(ServerSideEncryptionConfiguration={
            'Rules': [{'ApplyServerSideEncryptionByDefault': {'SSEAlgorithm': 'AES256'}}]}, **self.base)
        self.s3.put_bucket_ownership_controls(OwnershipControls={'Rules': [{'ObjectOwnership': 'BucketOwnerEnforced'}]}, **self.base)
        self.put_new('gds-storage.json', json_bytes({'schema_version': 1, 'application': 'GDS-Viewer',
                     'prefixes': ['sources/', 'releases/'], 'active_release_pointer': 'current.json'}))
        output.mkdir(parents=True, exist_ok=True)
        (output / 'connection.json').write_bytes(json_bytes({'bucket': self.bucket, 'region': self.region,
                                                           'account': self.account, 'current_key': 'current.json'}))
        arn = f'arn:{self.partition}:s3:::{self.bucket}'
        read_policy = {'Version': '2012-10-17', 'Statement': [
            {'Sid': 'ReadGDSRelease', 'Effect': 'Allow', 'Action': ['s3:GetObject'],
             'Resource': [arn + '/current.json', arn + '/releases/*']}]}
        (output / 'heroku_read_policy.json').write_bytes(json_bytes(read_policy))
        print(f'Storage ready: s3://{self.bucket}/ ({self.region})')
        print(f'Connection and application read policy: {output}')

    def upload(self, payload, release):
        self.check_bucket()
        manifest, raw = load_manifest(payload, release)
        old, _ = self.get(f'releases/{release}/manifest.json')
        require(old is None or old == raw, f'Release {release} is already sealed with different content. Use a new release.')
        count = len(manifest['files'])
        added = 0
        for i, item in enumerate(manifest['files'], 1):
            added += self.put_new(item['key'], (payload / item['key']).read_bytes())
            if i % 20 == 0 or i == count:
                print(f'Upload: {i}/{count}; new objects: {added}', flush=True)
        # Manifest is the final seal: publish it only after all member bytes were checked/uploaded.
        self.put_new(f'releases/{release}/manifest.json', raw)
        print(f'Release {release} uploaded. current.json has not changed.')

    def verify(self, payload, release):
        self.check_bucket()
        manifest, raw = load_manifest(payload, release)
        remote, _ = self.get(f'releases/{release}/manifest.json')
        require(remote == raw, 'Remote manifest is missing or differs from the local manifest')
        expected = {i['key'] for i in manifest['files'] if i['key'].startswith(f'releases/{release}/')}
        expected.add(f'releases/{release}/manifest.json')
        require(self.keys(f'releases/{release}/') == expected, 'Unexpected or missing objects in release prefix')
        for i, item in enumerate(manifest['files'], 1):
            # Full download and SHA256 verification; do not trust ETag or user-defined metadata as a checksum.
            response = self.s3.get_object(Key=item['key'], **self.base)
            h, size = hashlib.sha256(), 0
            try:
                for part in iter(lambda: response['Body'].read(1024 * 1024), b''):
                    h.update(part)
                    size += len(part)
            finally:
                response['Body'].close()
            require(size == item['size'] and h.hexdigest() == item['sha256'], f'Remote hash/size mismatch: {item["key"]}')
            if i % 40 == 0 or i == len(manifest['files']):
                print(f'Verify SHA256: {i}/{len(manifest["files"])}', flush=True)
        print(f'VERIFIED {release}: {manifest["counts"]}')
        return manifest, raw

    def activate(self, payload, release):
        # Capture pointer before lengthy verification; CAS prevents concurrent activation overwrite.
        old, etag = self.get('current.json')
        previous = json.loads(old) if old else None
        manifest, raw = self.verify(payload, release)
        if previous and previous.get('release_id') == release and previous.get('manifest_sha256') == sha(raw):
            print(f'{release} is already active. No pointer update needed.')
            return
        pointer = {'schema_version': 1, 'release_id': release, 'manifest_key': f'releases/{release}/manifest.json',
                   'manifest_sha256': sha(raw), 'activated_at_utc': datetime.now(timezone.utc).isoformat(),
                   'previous_release_id': previous.get('release_id') if previous else None}
        body = json_bytes(pointer)
        conditions = {'IfMatch': etag} if etag else {'IfNoneMatch': '*'}
        self.s3.put_object(Key='current.json', Body=body, CacheControl='no-cache, max-age=0',
                           ContentType='application/json', ServerSideEncryption='AES256',
                           ChecksumSHA256=base64.b64encode(hashlib.sha256(body).digest()).decode(),
                           **conditions, **self.base)
        check, _ = self.get('current.json')
        require(check == body, 'Pointer changed concurrently after activation. Inspect status.')
        print(f'ACTIVE: {release}; s3://{self.bucket}/current.json')

    def status(self):
        self.check_bucket()
        raw, _ = self.get('current.json')
        print(f'Bucket: {self.bucket}\nRegion: {self.region}')
        print(raw.decode() if raw else 'No active release yet. Run upload, then activate.')


def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['create', 'upload', 'verify', 'activate', 'status', 'run', 'check-local'])
    p.add_argument('--region', default=os.environ.get('AWS_REGION') or os.environ.get('AWS_DEFAULT_REGION'))
    p.add_argument('--bucket', default=os.environ.get('GDS_BUCKET'))
    p.add_argument('--release', default='v001')
    p.add_argument('--payload', type=Path, default=root / 'payload')
    p.add_argument('--state', type=Path, default=root / 'state')
    args = p.parse_args()
    require(bool(re.fullmatch(r'v[0-9]{3,}', args.release)), 'Invalid release ID')
    if args.command == 'check-local':
        manifest, _ = load_manifest(args.payload, args.release)
        print(json.dumps(manifest['counts'], indent=2))
        print('LOCAL SHA256 OK')
        return
    # Saved values take precedence over environment defaults, explicit flags override saved values.
    config_file = args.state / 'connection.json'
    if config_file.exists():
        saved = json.loads(config_file.read_text())
        explicit_region = any(x == '--region' or x.startswith('--region=') for x in sys.argv[1:])
        explicit_bucket = any(x == '--bucket' or x.startswith('--bucket=') for x in sys.argv[1:])
        if not explicit_region:
            args.region = saved['region']
        if not explicit_bucket and not os.environ.get('GDS_BUCKET'):
            args.bucket = saved['bucket']
    store = Store(args.bucket, args.region)
    if args.command in {'create', 'run'}:
        store.provision(args.state)
    if args.command in {'upload', 'run'}:
        store.upload(args.payload, args.release)
    if args.command == 'verify':
        store.verify(args.payload, args.release)
    if args.command in {'activate', 'run'}:
        store.activate(args.payload, args.release)
    if args.command == 'status':
        store.status()


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, FileNotFoundError, ClientError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(1)
