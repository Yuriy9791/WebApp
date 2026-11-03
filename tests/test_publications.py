import csv
import hashlib
from importlib.util import module_from_spec, spec_from_file_location
from io import BytesIO
import json
from pathlib import Path
import sys
from unittest.mock import patch

import boto3
from botocore.exceptions import ClientError
from moto import mock_aws
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gds_viewer.data import Repository, LocalStore, S3Store
from gds_viewer.publications import CATALOG_KEY, catalog_bytes, decode_catalog
from gds_viewer.ui import make_app

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
spec = spec_from_file_location('upload_publication', TOOLS / 'upload_publication.py')
upload = module_from_spec(spec)
spec.loader.exec_module(upload)


def csv_file(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def payload(tmp_path):
    root = tmp_path / 'payload'
    release = root / 'releases/v001'
    records, curves = [], []
    for rid, filename, mnemonics in [('well-a', 'A.las', ['GR', 'DT']), ('well-b', 'B.las', ['GR'])]:
        data = ('Test LAS ' + filename).encode()
        las = release / 'las' / filename
        las.parent.mkdir(parents=True, exist_ok=True)
        las.write_bytes(data)
        records.append(dict(record_id=rid, las_filename=filename, well_name_original=rid,
            las_path='las/' + filename, parquet_path='curves/' + rid + '.parquet', sample_count='3',
            enriched_sha256=hashlib.sha256(data).hexdigest(), latitude='25', longitude='45',
            map_allowed_preliminary='True', Age='Jurassic', Reference='Reference ' + rid,
            Special_mark='', depth_scale='artificial', las_depth_unit='m', curve_mnemonics=';'.join(mnemonics)))
        arrays = {'depth': [0., 1., 2.]}
        for i, mnemonic in enumerate(mnemonics, 1):
            arrays[f'c{i:03}'] = [10., 20., 30.]
            curves.append(dict(record_id=rid, log_id=f'{rid}:column:{i}', column_index=str(i),
                mnemonic=mnemonic, unit='gAPI' if mnemonic == 'GR' else 'us/ft', role='curve',
                available_for_plot='True', listed_in_legacy_csv='True', valid_count='3',
                parquet_column=f'c{i:03}', legacy_metadata_rows='1' if rid == 'well-a' else '2',
                depth_first_valid='0', depth_last_valid='2'))
        parquet = release / 'curves' / (rid + '.parquet')
        parquet.parent.mkdir(exist_ok=True)
        pq.write_table(pa.table(arrays), parquet)
    csv_file(release / 'metadata/records.csv', records)
    csv_file(release / 'metadata/curves.csv', curves)
    csv_file(release / 'metadata/metadata_preserved.csv', [
        {'csv_row': '1', 'Reference': 'Reference well-a'}, {'csv_row': '2', 'Reference': 'Reference well-b'}])
    (release / 'stratigraphy').mkdir()
    (release / 'stratigraphy/index.json').write_text('{"files": []}')
    files = [{'key': f.relative_to(root).as_posix(), 'size': f.stat().st_size,
              'sha256': hashlib.sha256(f.read_bytes()).hexdigest()}
             for f in sorted(release.rglob('*')) if f.is_file()]
    manifest = {'schema_version': 1, 'release_id': 'v001', 'source_batch': 'batch-001',
                'curve_selection': 'accepted_catalog_available_for_plot',
                'counts': {'las_files': 2, 'metadata_rows': 2}, 'files': files}
    (release / 'manifest.json').write_text(json.dumps(manifest))
    return root, records, curves


def source_input(tmp_path, well='A.las', curve='GR', pdf=b'%PDF-1.4\nsource article\n%%EOF'):
    path = tmp_path / ('source_' + well + '_' + curve + '.json')
    filename = 'paper-' + hashlib.sha256(pdf).hexdigest()[:8] + '.pdf'
    (tmp_path / filename).write_bytes(pdf)
    path.write_text(json.dumps({'pdf_file': filename, 'title': 'Test source article',
                               'links': [{'las_filename': well, 'curves': [curve]}]}))
    return path


def attach_local(payload, input_file):
    root, records, curves = payload
    paper, data, _ = upload.prepare_paper(input_file, records, curves)
    pdf = root / paper['key']
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(data)
    (root / CATALOG_KEY).write_bytes(catalog_bytes({'schema_version': 1, 'papers': [paper]}))
    return paper, data


def test_older_releases_without_pdfs_still_work(payload):
    root, _, _ = payload
    repo = Repository(LocalStore(root), 'v001')
    assert repo.source_papers(repo.active(), ['well-a:column:1']) == []
    client = make_app(repo).server.test_client()
    assert client.get('/readyz').json['source_pdf_downloads'] is True
    assert client.get('/source-pdf/paper-0000000000000000').status_code == 404


def test_local_download_and_exact_log_binding(payload, tmp_path):
    paper, data = attach_local(payload, source_input(tmp_path))
    repo = Repository(LocalStore(payload[0]), 'v001')
    snap = repo.active()
    assert len(repo.source_papers(snap, ['well-a:column:1'])) == 1
    assert repo.source_papers(snap, ['well-a:column:2', 'well-b:column:1']) == []
    client = make_app(repo).server.test_client()
    response = client.get('/source-pdf/' + paper['publication_id'])
    assert response.status_code == 200 and response.data == data
    assert response.mimetype == 'application/pdf'
    assert 'attachment' in response.headers['Content-Disposition']
    assert response.headers['Cache-Control'] == 'no-store'
    snap.records['well-a']['enriched_sha256'] = '0' * 64
    assert repo.source_papers(snap, ['well-a:column:1']) == []


def test_pdf_checksum_mismatch_and_catalog_path_rejected(payload, tmp_path):
    paper, _ = attach_local(payload, source_input(tmp_path))
    (payload[0] / paper['key']).write_bytes(b'corrupt')
    client = make_app(Repository(LocalStore(payload[0]), 'v001')).server.test_client()
    assert client.get('/source-pdf/' + paper['publication_id']).status_code == 404
    paper['key'] = 'releases/v001/metadata/records.csv'
    with pytest.raises(ValueError):
        decode_catalog(json.dumps({'schema_version': 1, 'papers': [paper]}))


def test_download_link_in_actual_plot_callback(payload, tmp_path):
    paper, _ = attach_local(payload, source_input(tmp_path))
    repo = Repository(LocalStore(payload[0]), 'v001')
    app = make_app(repo)
    key = next(k for k in app.callback_map if 'plot-output.children' in k)
    response = app.server.test_client().post('/_dash-update-component', json={'output': key,
        'outputs': [{'id': 'plot-output', 'property': 'children'}, {'id': 'publications', 'property': 'children'}],
        'inputs': [{'id': 'build-logs', 'property': 'n_clicks', 'value': 1},
                   {'id': 'well-select', 'property': 'value', 'value': ['well-a']},
                   {'id': 'curve-table', 'property': 'selected_row_ids', 'value': ['well-a:column:1', 'well-a:column:2']}],
        'state': [{'id': 'release', 'property': 'data', 'value': repo.active().descriptor},
                  {'id': 'depth-start', 'property': 'value', 'value': None},
                  {'id': 'depth-end', 'property': 'value', 'value': None},
                  {'id': 'xscale', 'property': 'value', 'value': 'auto'},
                  {'id': 'xreverse', 'property': 'value', 'value': 'auto'}],
        'changedPropIds': ['build-logs.n_clicks']})
    assert response.status_code == 200
    result = response.json['response']
    assert result['plot-output']['children'][0]['props']['children']['type'] == 'Graph'
    rendered = json.dumps(result['publications']['children'])
    assert 'Download source PDF' in rendered
    assert '/source-pdf/' + paper['publication_id'] in rendered
    assert 'Some selected logs do not yet' in rendered


@mock_aws
def test_s3_dry_run_upload_merge_and_signed_download(payload, tmp_path):
    store = upload.Store('gds-source-pdf-tests', 'eu-north-1')
    store.provision(tmp_path / 'state')
    store.upload(payload[0], 'v001')
    store.activate(payload[0], 'v001')
    pointer = store.get('current.json')
    files = store.keys('')
    first = source_input(tmp_path)
    plan = upload.upload_publication(store, first)
    assert plan['mode'] == 'DRY RUN' and store.keys('') == files
    upload.upload_publication(store, first, apply=True)
    upload.upload_publication(store, source_input(tmp_path, curve='DT'), apply=True)
    upload.upload_publication(store, source_input(tmp_path, well='B.las'), apply=True)
    catalog = decode_catalog(store.get(CATALOG_KEY)[0])
    assert len(catalog['papers']) == 1 and len(catalog['papers'][0]['links']) == 3
    assert len(store.keys('publications/files/')) == 1
    assert store.get('current.json') == pointer
    repo = Repository(S3Store(store.bucket, store.region))
    assert len(repo.source_papers(repo.active(), ['well-a:column:1'])) == 1
    response = make_app(repo).server.test_client().get('/source-pdf/' + catalog['papers'][0]['publication_id'])
    assert response.status_code == 302
    assert 'X-Amz-Signature=' in response.headers['Location']
    assert 'response-content-disposition=' in response.headers['Location']


def test_bad_binding_rejected_before_upload(payload, tmp_path):
    with pytest.raises(ValueError, match='Unknown/unavailable'):
        upload.prepare_paper(source_input(tmp_path, curve='RESD'), payload[1], payload[2])


@mock_aws
def test_concurrent_catalog_write_is_not_overwritten(payload, tmp_path):
    store = upload.Store('gds-source-pdf-concurrent', 'eu-north-1')
    store.provision(tmp_path / 'state')
    store.upload(payload[0], 'v001')
    store.activate(payload[0], 'v001')
    upload.upload_publication(store, source_input(tmp_path), apply=True)
    competing = decode_catalog(store.get(CATALOG_KEY)[0])
    competing['papers'][0]['title'] = 'Concurrent update to preserve'
    raw = catalog_bytes(competing)
    original = store.s3.put_object

    def race(**kwargs):
        if kwargs['Key'] == CATALOG_KEY:
            original(Key=CATALOG_KEY, Body=raw, **store.base)
        return original(**kwargs)

    with patch.object(store.s3, 'put_object', side_effect=race):
        with pytest.raises(ClientError) as error:
            upload.upload_publication(store, source_input(tmp_path, well='B.las',
                pdf=b'%PDF-1.4\nother source\n%%EOF'), apply=True)
    assert error.value.response['Error']['Code'] == 'PreconditionFailed'
    assert store.get(CATALOG_KEY)[0] == raw
