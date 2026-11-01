"""Find the existing GDS storage for a read-only CloudShell check."""
import json
import os
from pathlib import Path
import sys

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from . import check_data


def bucket_region(client, bucket):
    try:
        response = client.head_bucket(Bucket=bucket)
    except ClientError as error:
        response = error.response
        # HeadBucket can disclose the real region even when ListBucket is denied.
        region = response.get('ResponseMetadata', {}).get('HTTPHeaders', {}).get('x-amz-bucket-region')
        if region and response.get('Error', {}).get('Code') in {'403', 'AccessDenied'}:
            return region
        raise
    region = response.get('BucketRegion') or response.get('ResponseMetadata', {}).get('HTTPHeaders', {}).get('x-amz-bucket-region')
    if not region:
        location = client.get_bucket_location(Bucket=bucket).get('LocationConstraint')
        region = 'eu-west-1' if location == 'EU' else location or 'us-east-1'
    return region


def resolve_connection(environ, app_dir, home_dir, session):
    explicit_bucket = environ.get('GDS_S3_BUCKET', '').strip()
    explicit_config = environ.get('GDS_STORAGE_CONFIG', '').strip()
    paths = [Path(explicit_config).expanduser()] if explicit_config else [
        Path(app_dir).parent / 'GDS_CloudShell_Kit/state/connection.json',
        Path(home_dir) / 'GDS_CloudShell_Kit/state/connection.json']
    if not explicit_bucket:
        for path in dict.fromkeys(paths):
            if path.is_file():
                value = json.loads(path.read_text())
                bucket, region = value.get('bucket'), value.get('region')
                if not isinstance(bucket, str) or not bucket or not isinstance(region, str) or not region:
                    raise ValueError(f'Invalid storage settings: {path}')
                return bucket, region, f'saved settings: {path}'
        if explicit_config:
            raise ValueError(f'Storage settings file not found: {explicit_config}')
    # This region is only the entry point to discovery, never part of an inferred name.
    initial_region = environ.get('AWS_REGION') or environ.get('AWS_DEFAULT_REGION') or session.region_name
    client = session.client('s3', region_name=initial_region, config=Config(
        connect_timeout=5, read_timeout=15, retries={'mode': 'standard', 'max_attempts': 3}))
    if explicit_bucket:
        return explicit_bucket, bucket_region(client, explicit_bucket), 'explicit GDS_S3_BUCKET'
    buckets = []
    for page in client.get_paginator('list_buckets').paginate(PaginationConfig={'PageSize': 1000}):
        buckets.extend(b['Name'] for b in page.get('Buckets', []) if b['Name'].startswith('gds-viewer-data-'))
    buckets = sorted(set(buckets))
    if len(buckets) != 1:
        found = ', '.join(buckets) if buckets else '(none)'
        raise ValueError('Cannot select exactly one GDS bucket. Found: ' + found +
                         '. Set GDS_S3_BUCKET to the exact existing name, or GDS_STORAGE_CONFIG to connection.json.')
    bucket = buckets[0]
    return bucket, bucket_region(client, bucket), 'existing bucket discovered in your AWS account'


def main():
    if not os.environ.get('GDS_LOCAL_ROOT'):
        bucket, region, origin = resolve_connection(os.environ, Path(__file__).resolve().parents[1], Path.home(), boto3.Session())
        os.environ['GDS_S3_BUCKET'] = bucket
        os.environ['AWS_REGION'] = region
        print(f'Storage settings: {origin}\nChecking bucket: {bucket}, region: {region}', flush=True)
    check_data.main()


if __name__ == '__main__':
    try:
        main()
    except ClientError as error:
        code = error.response.get('Error', {}).get('Code', 'AWS error')
        hints = {
            'NoSuchBucket': 'The selected bucket does not exist. Check the exact bucket name in your S3 console and set GDS_S3_BUCKET.',
            'NoSuchKey': 'A requested object is missing. Check current.json and whether the data release upload and activation completed.',
            'AccessDenied': 'AWS denied this request. For discovery you can instead supply GDS_STORAGE_CONFIG or GDS_S3_BUCKET; reading data requires s3:GetObject.',
        }
        print(f'ERROR [{code}]: {hints.get(code, error.response.get("Error", {}).get("Message", str(error)))}', file=sys.stderr)
        sys.exit(1)
    except (ValueError, FileNotFoundError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(1)
