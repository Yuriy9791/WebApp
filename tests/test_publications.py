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
from gds_viewer.publications import pdf_key, reference_id
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


def source_pdf(tmp_path, data=b'%PDF-1.4\nsource article\n%%EOF'):
    path = tmp_path / 'source.pdf'
    path.write_bytes(data)
    return path


def attach_local(payload, pdf, reference='Reference well-a'):
    target = payload[0] / pdf_key(reference)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(pdf.read_bytes())


def test_older_releases_without_pdfs_still_work(payload):
    repo = Repository(LocalStore(payload[0]), 'v001')
    assert not repo.source_pdf_exists('Reference well-a')
    client = make_app(repo).server.test_client()
    assert client.get('/readyz').json['source_pdf_downloads'] is True
    assert client.get('/source-pdf/' + reference_id('Reference well-a')).status_code == 404


def test_pdf_uses_existing_reference_for_all_related_logs(payload, tmp_path):
    pdf = source_pdf(tmp_path)
    attach_local(payload, pdf)
    repo = Repository(LocalStore(payload[0]), 'v001')
    snap = repo.active()
    # Both existing curves inherit the same CSV Reference, without any new log bindings.
    rows = repo.curve_rows(snap, ['well-a'])
    assert len(rows) == 2 and {r['Reference'] for r in rows} == {'Reference well-a'}
    assert repo.source_pdf_exists(rows[0]['Reference'])
    assert not repo.source_pdf_exists(repo.curve_rows(snap, ['well-b'])[0]['Reference'])
    client = make_app(repo).server.test_client()
    response = client.get('/source-pdf/' + reference_id('Reference well-a'))
    assert response.status_code == 200 and response.data == pdf.read_bytes()
    assert response.mimetype == 'application/pdf'
    assert 'attachment' in response.headers['Content-Disposition']
    assert response.headers['Cache-Control'] == 'no-store'
    assert client.get('/source-pdf/' + reference_id('Not in the CSV')).status_code == 404


def test_publication_name_itself_is_the_link_in_plot_callback(payload, tmp_path):
    attach_local(payload, source_pdf(tmp_path))
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
    nodes = result['publications']['children']
    assert len(nodes) == 2  # Existing heading and existing citation, no extra cards/buttons.
    anchor = nodes[1]['props']['children']
    assert anchor['type'] == 'A'
    assert anchor['props']['children'] == 'Reference well-a'
    assert anchor['props']['href'] == '/source-pdf/' + reference_id('Reference well-a')


@mock_aws
def test_s3_upload_reference_and_signed_download(payload, tmp_path):
    store = upload.Store('gds-reference-pdf-tests', 'eu-north-1')
    store.provision(tmp_path / 'state')
    store.upload(payload[0], 'v001')
    store.activate(payload[0], 'v001')
    pointer = store.get('current.json')
    files = store.keys('')
    pdf = source_pdf(tmp_path)
    plan = upload.upload_publication(store, pdf, 'Reference well-a', dry_run=True)
    assert plan['mode'] == 'DRY RUN' and store.keys('') == files
    upload.upload_publication(store, pdf, 'Reference well-a')
    assert store.get(pdf_key('Reference well-a'))[0] == pdf.read_bytes()
    assert store.keys('publications/') == {pdf_key('Reference well-a')}
    assert store.get('current.json') == pointer
    repo = Repository(S3Store(store.bucket, store.region))
    assert repo.source_pdf_exists('Reference well-a')
    assert not repo.source_pdf_exists('Reference well-b')
    response = make_app(repo).server.test_client().get('/source-pdf/' + reference_id('Reference well-a'))
    assert response.status_code == 302
    assert 'X-Amz-Signature=' in response.headers['Location']
    assert 'response-content-disposition=' in response.headers['Location']


@mock_aws
def test_unknown_reference_does_not_write_and_existing_values_are_listed(payload, tmp_path):
    store = upload.Store('gds-reference-validation', 'eu-north-1')
    store.provision(tmp_path / 'state')
    store.upload(payload[0], 'v001')
    store.activate(payload[0], 'v001')
    _, refs = upload.active_references(store)
    assert set(refs.values()) == {'Reference well-a', 'Reference well-b'}
    before = store.keys('')
    with pytest.raises(ValueError, match='absent from the existing CSV'):
        upload.upload_publication(store, source_pdf(tmp_path), 'Invented source')
    assert store.keys('') == before


def test_reference_lookup_is_stable_under_case_and_whitespace():
    assert pdf_key('  Liu et al 2017 N Iraq (Kurdistan) ') == pdf_key('liu et al 2017  n iraq (kurdistan)')
