import csv
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch
import pytest
from gds_viewer.data import Repository, LocalStore, DataError, S3Store, visible_records, update_selection
from gds_viewer.figures import build_figures
from gds_viewer.ui import make_app

ROOT=Path(os.environ.get('GDS_TEST_ROOT',Path(__file__).resolve().parents[2]/'GDS_CloudShell_Kit/payload'))
pytestmark=pytest.mark.skipif(not ROOT.exists(),reason='Set GDS_TEST_ROOT to the accepted CloudShell payload')

@pytest.fixture
def repo(): return Repository(LocalStore(ROOT),'v001')


def craigie(snap):
    return next(k for k,r in snap.records.items() if r['las_filename']=='1_Craigie, GeoArabia.las')


def test_metadata_and_allowed_curves(repo):
    s=repo.active(); assert len(s.records)==164 and len(s.metadata)==359 and len(s.curves)==265
    rows=repo.curve_rows(s,[craigie(s)])
    assert {r['Curve'] for r in rows}=={'GR','DT','NPHI','RHOB'}
    assert all(r['Depth scale']=='artificial' for r in rows)


def test_selection_uses_ids_not_coordinate_cross_product(repo):
    s=repo.active(); valid=list(s.records)[:3]
    assert update_selection([valid[0]],{'points':[{'customdata':[valid[1]]}]},set(valid))==valid[:2]
    assert update_selection(valid[:2],{'points':[{'customdata':[valid[0]]}]},set(valid),True)==[valid[1]]
    assert update_selection([],{'points':[{'customdata':['not-a-record']}]},set(valid))==[]


def test_filters_do_not_change_dataset(repo):
    s=repo.active()
    assert len(visible_records(s))==164
    assert len(visible_records(s,query='1_Craigie, GeoArabia'))==1
    assert all('GR' in r['curve_mnemonics'].split(';') for r in visible_records(s,curve_type='GR'))


def test_read_all_parquet_and_lazy_cache(repo):
    s=repo.active(); assert sum(repo.table(s,k).num_rows for k in s.records)==921550
    assert len(repo.tables)<=24
    rid=craigie(s)
    first=repo.table(s,rid)
    with patch.object(repo.store,'read',side_effect=AssertionError('Unexpected refetch')):
        assert repo.table(s,rid) is first


def test_all_curves_plot_and_null_gaps(repo):
    s=repo.active()
    figures=build_figures(repo,s,list(s.curves))
    assert sum(len(f.data) for f in figures)==265
    for f in figures:
        assert all(t.connectgaps is False for t in f.data)
        assert f.layout.yaxis.autorange=='reversed'
    ids=[c['log_id'] for c in s.curves.values() if c['record_id']==craigie(s)]
    fig=build_figures(repo,s,ids)[0]
    assert 'Artificial' in fig.layout.yaxis.title.text
    values=repo.table(s,craigie(s))[s.curves[ids[0]]['parquet_column']].to_pylist()
    assert list(fig.data[0].x)==values


def test_invalid_selection_and_depth_rejected(repo):
    s=repo.active()
    with pytest.raises(DataError): build_figures(repo,s,['not-real'])
    with pytest.raises(DataError): build_figures(repo,s,list(s.curves)[:1],100,0)
    with pytest.raises(DataError): repo.snapshot('../outside')
    with pytest.raises(DataError): repo.store.read('../outside')


def test_checksum_corruption_detected(repo):
    s=repo.active(); rid=craigie(s); original=repo.store.read
    def changed(key):
        raw=original(key)
        return b'corrupt' if key.endswith(s.records[rid]['parquet_path']) else raw
    with patch.object(repo.store,'read',side_effect=changed):
        with pytest.raises(DataError,match='Checksum'): repo.table(s,rid)
    with pytest.raises(DataError,match='Manifest hash'): repo.snapshot('v001','0'*64)


def test_http_layout_health_download_and_callbacks(repo):
    app=make_app(repo); client=app.server.test_client(); s=repo.active()
    assert client.get('/').status_code==200
    assert client.get('/_dash-layout').status_code==200
    assert client.get('/_dash-dependencies').status_code==200
    assert client.get('/readyz').json['records']==164
    rid=craigie(s); response=client.get('/download/v001/'+rid)
    assert response.status_code==200
    assert hashlib.sha256(response.data).hexdigest()==s.records[rid]['enriched_sha256']
    assert client.get('/download/v001/absent').status_code==404
    # Run the actual Dash callback endpoint, not only pure functions.
    key=next(k for k in app.callback_map if 'plot-output.children' in k)
    ids=[r['id'] for r in repo.curve_rows(s,[rid])]
    response=client.post('/_dash-update-component',json={'output':key,
        'outputs':[{'id':'plot-output','property':'children'},{'id':'publications','property':'children'}],
        'inputs':[{'id':'build-logs','property':'n_clicks','value':1},{'id':'well-select','property':'value','value':[rid]},{'id':'curve-table','property':'selected_row_ids','value':ids}],
        'state':[{'id':'release','property':'data','value':s.descriptor},{'id':'depth-start','property':'value','value':None},{'id':'depth-end','property':'value','value':None},{'id':'xscale','property':'value','value':'auto'},{'id':'xreverse','property':'value','value':'auto'}],
        'changedPropIds':['build-logs.n_clicks']})
    assert response.status_code==200
    result=response.json['response']['plot-output']['children']
    assert result[0]['props']['children']['type']=='Graph'


def test_s3_readonly_current_and_presigned_url():
    from moto import mock_aws
    import boto3
    with mock_aws():
        client=boto3.client('s3',region_name='us-east-2'); bucket='gds-viewer-test-data'
        client.create_bucket(Bucket=bucket,CreateBucketConfiguration={'LocationConstraint':'us-east-2'})
        for path in (ROOT/'releases/v001').rglob('*'):
            if path.is_file(): client.put_object(Bucket=bucket,Key=path.relative_to(ROOT).as_posix(),Body=path.read_bytes())
        m=(ROOT/'releases/v001/manifest.json').read_bytes()
        client.put_object(Bucket=bucket,Key='current.json',Body=json.dumps({'release_id':'v001','manifest_key':'releases/v001/manifest.json','manifest_sha256':hashlib.sha256(m).hexdigest()}).encode())
        repo=Repository(S3Store(bucket,'us-east-2')); s=repo.active()
        assert len(s.records)==164
        assert repo.table(s,craigie(s)).num_rows==4268
        app=make_app(repo); response=app.server.test_client().get('/download/v001/'+craigie(s))
        assert response.status_code==302 and 'X-Amz-Signature=' in response.headers['Location']


def test_table_select_filtered_curves_keeps_stable_ids(repo):
    app=make_app(repo); client=app.server.test_client(); s=repo.active(); rid=craigie(s)
    rows=repo.curve_rows(s,[rid]); displayed=[rows[-1],rows[0]]
    key=next(k for k in app.callback_map if 'curve-table.data' in k)
    response=client.post('/_dash-update-component',json={'output':key,
        'outputs':[{'id':'curve-table','property':'data'},{'id':'curve-table','property':'selected_row_ids'},{'id':'curve-table','property':'selected_rows'},{'id':'table-note','property':'children'},{'id':'curve-table','property':'tooltip_data'}],
        'inputs':[{'id':'well-select','property':'value','value':[rid]},{'id':'select-curves','property':'n_clicks','value':1},{'id':'clear-curves','property':'n_clicks','value':0}],
        'state':[{'id':'curve-table','property':'derived_virtual_data','value':displayed},{'id':'curve-table','property':'selected_row_ids','value':[]},{'id':'release','property':'data','value':s.descriptor}],
        'changedPropIds':['select-curves.n_clicks']})
    assert response.status_code==200
    table=response.json['response']['curve-table']
    assert set(table['selected_row_ids'])=={r['id'] for r in displayed}
    assert {table['data'][i]['id'] for i in table['selected_rows']}==set(table['selected_row_ids'])
