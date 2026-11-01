"""Read-only, checksum-verified access to one immutable GDS release."""
from collections import OrderedDict
import csv
from dataclasses import dataclass
import hashlib
from io import BytesIO, StringIO
import json
import os
from pathlib import Path, PurePosixPath
import re
from threading import RLock
from urllib.parse import quote

import boto3
from botocore.config import Config
import pyarrow.parquet as pq


class DataError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise DataError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def safe_key(key):
    p = PurePosixPath(key)
    require(isinstance(key, str) and not p.is_absolute() and '..' not in p.parts and '\\' not in key, 'Invalid data path')
    return key


def read_csv(raw):
    return list(csv.DictReader(StringIO(raw.decode('utf-8-sig'))))


def is_true(v):
    return str(v).lower() == 'true'


def ages(value):
    return [p.strip() for p in re.split(r'[_;,]', value or '') if p.strip()] or ['Age not specified']


class LocalStore:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def path(self, key):
        target = (self.root / safe_key(key)).resolve()
        require(target.is_relative_to(self.root), 'Data path outside root')
        return target

    def read(self, key):
        return self.path(key).read_bytes()


class S3Store:
    def __init__(self, bucket, region):
        require(bool(bucket), 'Set GDS_S3_BUCKET to the NEW GDS bucket name')
        self.bucket = bucket
        self.client = boto3.client('s3', region_name=region, config=Config(
            signature_version='s3v4', retries={'mode': 'standard', 'max_attempts': 3},
            connect_timeout=5, read_timeout=15))

    def read(self, key):
        response = self.client.get_object(Bucket=self.bucket, Key=safe_key(key))
        try:
            return response['Body'].read()
        finally:
            response['Body'].close()

    def download_url(self, key, filename):
        return self.client.generate_presigned_url('get_object', Params={
            'Bucket': self.bucket, 'Key': safe_key(key),
            'ResponseContentDisposition': "attachment; filename*=UTF-8''" + quote(filename, safe='')},
            ExpiresIn=300)


@dataclass
class Snapshot:
    release: str
    digest: str
    manifest: dict
    records: dict
    curves: dict
    metadata: dict
    stratigraphy: dict

    @property
    def descriptor(self):
        return {'release': self.release, 'sha256': self.digest}


class Repository:
    def __init__(self, store, local_release=None):
        self.store = store
        self.local_release = local_release
        self.lock = RLock()
        self.snapshots = OrderedDict()
        self.tables = OrderedDict()

    @classmethod
    def from_env(cls):
        local = os.environ.get('GDS_LOCAL_ROOT')
        if local:
            return cls(LocalStore(local), os.environ.get('GDS_LOCAL_RELEASE', 'v001'))
        return cls(S3Store(os.environ.get('GDS_S3_BUCKET'), os.environ.get('AWS_REGION', 'us-east-2')))

    def active(self):
        if self.local_release:
            return self.snapshot(self.local_release)
        pointer = json.loads(self.store.read('current.json'))
        release = pointer['release_id']
        require(pointer['manifest_key'] == f'releases/{release}/manifest.json', 'Pointer path mismatch')
        return self.snapshot(release, pointer['manifest_sha256'])

    def file(self, snap, relative):
        key = safe_key(f'releases/{snap.release}/{relative}')
        entries = {x['key']: x for x in snap.manifest['files']}
        require(key in entries, 'File is absent from the release manifest')
        raw = self.store.read(key)
        entry = entries[key]
        require(len(raw) == entry['size'] and sha(raw) == entry['sha256'], f'Checksum mismatch: {relative}')
        return raw

    def snapshot(self, release, expected=None):
        require(bool(re.fullmatch(r'v[0-9]{3,}', release)), 'Invalid release ID')
        with self.lock:
            if expected and (release, expected) in self.snapshots:
                self.snapshots.move_to_end((release, expected))
                return self.snapshots[(release, expected)]
        raw = self.store.read(f'releases/{release}/manifest.json')
        digest = sha(raw)
        require(expected is None or digest == expected, 'Manifest hash differs from the active pointer')
        with self.lock:
            if (release, digest) in self.snapshots:
                return self.snapshots[(release, digest)]
        manifest = json.loads(raw)
        require(manifest['schema_version'] == 1 and manifest['release_id'] == release, 'Unsupported release manifest')
        snap = Snapshot(release, digest, manifest, {}, {}, {}, {})
        recs = read_csv(self.file(snap, 'metadata/records.csv'))
        curves = read_csv(self.file(snap, 'metadata/curves.csv'))
        metadata = read_csv(self.file(snap, 'metadata/metadata_preserved.csv'))
        require(len({r['record_id'] for r in recs}) == len(recs), 'Duplicate record IDs')
        snap.records = {r['record_id']: r for r in recs}
        require(all(c['record_id'] in snap.records for c in curves), 'Orphan curve record')
        require(len({c['log_id'] for c in curves}) == len(curves), 'Duplicate curve IDs')
        snap.curves = {c['log_id']: c for c in curves if is_true(c['available_for_plot'])}
        require(all(is_true(c['listed_in_legacy_csv']) and c.get('parquet_column') for c in snap.curves.values()), 'Unlisted or unmapped plot curve')
        snap.metadata = {r['csv_row']: r for r in metadata}
        snap.stratigraphy = json.loads(self.file(snap, 'stratigraphy/index.json'))
        require(len(recs) == manifest['counts']['las_files'], 'Record count mismatch')
        with self.lock:
            self.snapshots[(release, digest)] = snap
            while len(self.snapshots) > 8:
                self.snapshots.popitem(last=False)
        return snap

    def from_descriptor(self, descriptor):
        return self.snapshot(descriptor['release'], descriptor['sha256'])

    def table(self, snap, record_id):
        require(record_id in snap.records, 'Unknown record ID')
        key = (snap.release, snap.digest, record_id)
        with self.lock:
            if key in self.tables:
                self.tables.move_to_end(key)
                return self.tables[key]
        record = snap.records[record_id]
        # ParquetFile avoids a background dataset scanner for each in-memory file.
        table = pq.ParquetFile(BytesIO(self.file(snap, record['parquet_path']))).read(use_threads=False)
        require(table.num_rows == int(record['sample_count']) and 'depth' in table.column_names, 'Parquet shape mismatch')
        with self.lock:
            self.tables[key] = table
            # Cache is bounded per worker; no measurements go to browser Store components.
            while len(self.tables) > 24:
                self.tables.popitem(last=False)
        return table

    def intervals(self, snap, record_id):
        record = snap.records[record_id]
        result = []
        for entry in snap.stratigraphy.get('files', []):
            if entry['record_id'] != record_id:
                continue
            if entry['depth_scale'] != record['depth_scale'] or entry['depth_unit'] != record['las_depth_unit']:
                continue
            result.extend(read_csv(self.file(snap, entry['path'])))
        return result

    def curve_rows(self, snap, selected):
        selected = set(selected or []) & set(snap.records)
        result = []
        for c in snap.curves.values():
            if c['record_id'] not in selected:
                continue
            r = snap.records[c['record_id']]
            meta = [snap.metadata[x] for x in c['legacy_metadata_rows'].split(';') if x in snap.metadata]
            reference = ' | '.join(dict.fromkeys(m['Reference'].strip() for m in meta if m['Reference'].strip())) or r['Reference']
            result.append({'id': c['log_id'], 'record_id': c['record_id'], 'Well': r['well_name_original'],
                'Curve': c['mnemonic'], 'Unit': c['unit'] if c['unit'] not in {'', '_'} else 'Not specified',
                'Start': float(c['depth_first_valid']), 'End': float(c['depth_last_valid']),
                'Depth scale': r['depth_scale'], 'Depth unit': r['las_depth_unit'],
                'Age': r['Age'] or 'Age not specified', 'Reference': reference,
                'Notes': r['Special_mark'], 'Latitude': float(r['latitude']), 'Longitude': float(r['longitude'])})
        return result


def visible_records(snap, selected_ages=None, query='', curve_type=None):
    query = (query or '').casefold().strip()
    out = []
    for r in snap.records.values():
        if not is_true(r['map_allowed_preliminary']):
            continue
        if selected_ages and not set(ages(r['Age'])) & set(selected_ages):
            continue
        if query and query not in (r['well_name_original'] + ' ' + r['Reference']).casefold():
            continue
        if curve_type and curve_type not in r['curve_mnemonics'].split(';'):
            continue
        out.append(r)
    return out


def update_selection(current, event, valid_ids, toggle=False):
    selected = list(dict.fromkeys(x for x in (current or []) if x in valid_ids))
    for p in (event or {}).get('points', []):
        custom = p.get('customdata', [])
        rid = custom[0] if isinstance(custom, (list, tuple)) and custom else custom
        if rid not in valid_ids:
            continue
        if toggle and rid in selected:
            selected.remove(rid)
        elif rid not in selected:
            selected.append(rid)
    return selected
