import hashlib
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from gds_viewer.curve_policy import GLOBAL_POLICY, curve_types, plottable
from gds_viewer.data import Repository, LocalStore, DataError, visible_records
from gds_viewer.figures import build_figures
from gds_viewer.ui import make_app


def test_global_type_rule_keeps_depth_and_empty_channels_out():
    allowed = curve_types([{'Type': ' dt '}, {'Type': 'DT'}, {'Type': 'GR'}, {'Type': ''}])
    assert allowed == ['DT', 'GR']
    c = dict(role='curve', mnemonic='dt', valid_count='4', listed_in_legacy_csv='False')
    assert plottable(c, allowed)
    assert not plottable(dict(c, valid_count='0'), allowed)
    assert not plottable(dict(c, role='auxiliary_depth'), allowed)
    assert not plottable(dict(c, mnemonic='TOC'), allowed)


@pytest.fixture
def expanded():
    root = os.environ.get('GDS_EXPANDED_TEST_ROOT')
    if not root:
        pytest.skip('Set GDS_EXPANDED_TEST_ROOT to the rebuilt v002 payload')
    repo = Repository(LocalStore(root), 'v002')
    return repo, repo.active()


def test_full_original_csv_types_and_complete_catalog(expanded):
    repo, s = expanded
    assert len(s.records) == 164 and len(s.curves) == 332
    assert set(s.allowed_curve_types) == {
        'DT', 'GNT', 'GR', 'NPHI', 'PERM', 'PHI', 'RESD', 'RESS',
        'RHOB', 'SFLU', 'SGR', 'SO', 'SONIC', 'SP', 'TOC'}
    assert s.manifest['curve_selection'] == GLOBAL_POLICY
    assert sum(not (c['listed_in_legacy_csv'].lower() == 'true') for c in s.curves.values()) > 0
    assert all(c['role'] == 'curve' and int(c['valid_count']) > 0 for c in s.curves.values())
    for rid, r in s.records.items():
        selected = [c for c in s.curves.values() if c['record_id'] == rid]
        assert int(r['plot_curve_count']) == len(selected)
        assert set(r['curve_mnemonics'].split(';')) == {c['mnemonic'] for c in selected}
        assert set(repo.table(s, rid).column_names) == {'depth'} | {c['parquet_column'] for c in selected}


def test_every_parquet_column_matches_unchanged_las(expanded):
    import lasio
    from io import StringIO
    repo, s = expanded
    total = 0
    for rid, r in s.records.items():
        raw = repo.file(s, r['las_path'])
        assert hashlib.sha256(raw).hexdigest() == r['enriched_sha256']
        las = lasio.read(StringIO(raw.decode('utf-8-sig')), mnemonic_case='preserve')
        table = repo.table(s, rid)
        np.testing.assert_array_equal(table['depth'].to_numpy(), las.data[:, 0])
        for c in s.curves.values():
            if c['record_id'] == rid:
                np.testing.assert_allclose(table[c['parquet_column']].to_numpy(),
                    las.data[:, int(c['column_index'])], rtol=0, atol=0, equal_nan=True)
                total += 1
    assert total == 332


def test_kurdistan_dt_in_table_plots_and_http_callback(expanded):
    repo, s = expanded
    rid = next(k for k, r in s.records.items() if r['las_filename'] == 'Kurdistan-1.las')
    rows = repo.curve_rows(s, [rid])
    assert {r['Curve'] for r in rows} == {'GR', 'DT'}
    dt = next(r for r in rows if r['Curve'] == 'DT')
    assert (dt['Start'], dt['End'], dt['Unit']) == (213.8, 5413.1, 'us/ft')
    assert rid in {r['record_id'] for r in visible_records(s, curve_type='DT')}
    fig = build_figures(repo, s, [r['id'] for r in rows])[0]
    assert [t.name for t in fig.data] == ['GR', 'DT']
    assert sum(v is not None for v in fig.data[1].x) == 51994
    app = make_app(repo)
    client = app.server.test_client()
    assert client.get('/readyz').json['supported_curve_policy'] == GLOBAL_POLICY
    layout = client.get('/_dash-layout').json

    def find_id(node, target):
        if isinstance(node, dict):
            if node.get('props', {}).get('id') == target:
                return node['props']
            for value in node.values():
                found = find_id(value, target)
                if found:
                    return found
        elif isinstance(node, list):
            for value in node:
                found = find_id(value, target)
                if found:
                    return found

    assert set(find_id(layout, 'curve-filter')['options']) == set(s.allowed_curve_types)
    key = next(k for k in app.callback_map if 'curve-table.data' in k)
    response = client.post('/_dash-update-component', json={
        'output': key,
        'outputs': [{'id': 'curve-table', 'property': p} for p in ['data', 'selected_row_ids', 'selected_rows']]
                   + [{'id': 'table-note', 'property': 'children'}, {'id': 'curve-table', 'property': 'tooltip_data'}],
        'inputs': [{'id': 'well-select', 'property': 'value', 'value': [rid]},
                   {'id': 'select-curves', 'property': 'n_clicks', 'value': 0},
                   {'id': 'clear-curves', 'property': 'n_clicks', 'value': 0}],
        'state': [{'id': 'curve-table', 'property': 'derived_virtual_data', 'value': None},
                  {'id': 'curve-table', 'property': 'selected_row_ids', 'value': []},
                  {'id': 'release', 'property': 'data', 'value': s.descriptor}],
        'changedPropIds': ['well-select.value']})
    assert response.status_code == 200
    assert {r['Curve'] for r in response.json['response']['curve-table']['data']} == {'GR', 'DT'}


def test_every_selected_curve_can_be_plotted(expanded):
    repo, s = expanded
    figures = build_figures(repo, s, list(s.curves))
    assert sum(len(f.data) for f in figures) == 332


def test_global_catalog_rejects_missing_supported_curve(expanded):
    repo, s = expanded
    import gds_viewer.data as data
    original = data.read_csv

    def hide_dt(raw):
        rows = original(raw)
        if rows and 'available_for_plot' in rows[0]:
            rows = [dict(r) for r in rows]
            next(r for r in rows if r['mnemonic'] == 'DT')['available_for_plot'] = 'False'
        return rows

    fresh = Repository(repo.store, 'v002')
    with patch.object(data, 'read_csv', side_effect=hide_dt):
        with pytest.raises(DataError, match='incomplete'):
            fresh.active()
