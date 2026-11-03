#!/usr/bin/env python3
"""Build all finite LAS channels whose types occur anywhere in the original CSV."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import tempfile
import zipfile
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gds_viewer.curve_policy import GLOBAL_POLICY, curve_types, plottable


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, values):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(values[0]))
        w.writeheader()
        w.writerows(values)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def build(archive, output, release, batch, type_catalog):
    import lasio
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    if not re.fullmatch(r'v[0-9]{3,}', release) or not re.fullmatch(r'batch-[0-9]{3,}', batch):
        raise ValueError('Use release v001, v002, ... and batch-001, batch-002, ...')
    if output.exists():
        raise ValueError(f'Output already exists: {output}. Choose a new directory.')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='gds-build-', dir=output.parent) as temp:
        temp = Path(temp)
        extracted = temp / 'input'
        with zipfile.ZipFile(archive) as z:
            for entry in z.infolist():
                target = (extracted / entry.filename).resolve()
                if not target.is_relative_to(extracted.resolve()) or (entry.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError(f'Unsafe ZIP entry: {entry.filename}')
            z.extractall(extracted)
        candidates = list(extracted.rglob('catalog/records.csv'))
        if len(candidates) != 1:
            raise ValueError('Expected exactly one catalog/records.csv in source archive')
        source = candidates[0].parent.parent
        records = rows(source / 'catalog/records.csv')
        curves = rows(source / 'catalog/curves.csv')
        metadata = rows(source / 'inputs/List_of_curves.csv')
        allowed = curve_types(rows(type_catalog))
        if not allowed:
            raise ValueError('Original CSV has no nonempty Type values')
        if not records or len({r['record_id'] for r in records}) != len(records):
            raise ValueError('Missing or duplicate record_id')
        known = {r['record_id'] for r in records}
        if any(c['record_id'] not in known for c in curves):
            raise ValueError('Orphan curve row')
        stage = temp / 'payload'
        rel = stage / 'releases' / release
        for sub in ['las', 'metadata', 'curves', 'stratigraphy', 'audit']:
            (rel / sub).mkdir(parents=True, exist_ok=True)
        srcdir = stage / 'sources' / batch
        srcdir.mkdir(parents=True)
        shutil.copyfile(archive, srcdir / 'GDS_Iteration_1_Data.zip')
        shutil.copyfile(source / 'inputs/List_of_curves.csv', rel / 'metadata/List_of_curves.csv')
        shutil.copyfile(source / 'catalog/metadata_preserved.csv', rel / 'metadata/metadata_preserved.csv')
        shutil.copyfile(type_catalog, rel / 'metadata/curve_types_legacy.csv')
        for f in (source / 'audit').iterdir():
            if f.is_file():
                shutil.copyfile(f, rel / 'audit' / f.name)
        shutil.copyfile(source / 'README_RU.txt', rel / 'audit/source_README_RU.txt')
        counts = dict(las_files=len(records), metadata_rows=len(metadata), parquet_files=0,
                      selected_curve_columns=0, records_with_plot_curves=0, total_samples=0,
                      allowed_curve_types=len(allowed))
        changes = []
        for i, r in enumerate(records, 1):
            rid = r['record_id']
            if not re.fullmatch(r'[A-Za-z0-9-]+', rid):
                raise ValueError(f'Invalid record ID: {rid}')
            if 'extended hakami' in r['well_name_original'].lower():
                raise ValueError('Excluded Extended Hakami record found')
            f = (source / r['las_path']).resolve()
            if not f.is_relative_to(source.resolve()) or f.name != r['las_filename']:
                raise ValueError('LAS path mismatch')
            if digest(f) != r['enriched_sha256']:
                raise ValueError(f'LAS hash mismatch: {f.name}')
            las = lasio.read(str(f), mnemonic_case='preserve')
            if las.data.shape[0] != int(r['sample_count']):
                raise ValueError(f'LAS sample count mismatch: {f.name}')
            lat, lon = float(r['latitude']), float(r['longitude'])
            if not math.isfinite(lat) or not -90 <= lat <= 90 or not math.isfinite(lon) or not -180 <= lon <= 180:
                raise ValueError(f'Invalid coordinates: {f.name}')
            if not math.isclose(float(las.well.LAT.value), lat, abs_tol=1e-7) or not math.isclose(float(las.well.LONG.value), lon, abs_tol=1e-7):
                raise ValueError(f'LAS/catalog coordinate mismatch: {f.name}')
            rc = [c for c in curves if c['record_id'] == rid]
            if len(rc) != las.data.shape[1]:
                raise ValueError(f'LAS/catalog column count mismatch: {f.name}')
            selected = []
            for c in rc:
                index = int(c['column_index'])
                if c['mnemonic'] != las.curves[index].original_mnemonic:
                    raise ValueError(f'LAS/catalog mnemonic mismatch: {f.name}:{index}')
                c['parquet_column'] = ''
                values = las.data[:, index]
                valid = np.isfinite(values)
                if int(valid.sum()) != int(c['valid_count']):
                    raise ValueError(f'LAS/catalog valid count mismatch: {f.name}:{index}')
                old_available = c['available_for_plot'].lower() == 'true'
                c['available_for_plot'] = str(plottable(c, allowed))
                c['type_allowed_by_legacy_csv'] = str(c['mnemonic'].strip().upper() in allowed)
                c['selection_status'] = ('depth_axis' if c['role'] != 'curve'
                    else 'type_not_in_legacy_csv' if c['type_allowed_by_legacy_csv'] == 'False'
                    else 'no_data' if not valid.any() else 'selected')
                if c['available_for_plot'] == 'True':
                    c['parquet_column'] = f'c{index:03d}'
                    selected.append(c)
                if old_available and c['available_for_plot'] != 'True':
                    raise ValueError(f'Previously available curve would be removed: {f.name}:{index}')
                if not old_available and c['available_for_plot'] == 'True':
                    changes.append({'record_id': rid, 'las_filename': f.name,
                                    'mnemonic': c['mnemonic'], 'column_index': index,
                                    'listed_for_this_record': c['listed_in_legacy_csv'],
                                    'selection_reason': GLOBAL_POLICY})
            r['curve_mnemonics'] = ';'.join(dict.fromkeys(c['mnemonic'] for c in selected))
            r['plot_curve_count'] = str(len(selected))
            r['curve_selection_source'] = GLOBAL_POLICY
            depth = las.data[:, 0]
            if not np.isfinite(depth).all():
                raise ValueError(f'Missing primary depth: {f.name}')
            arrays = {'depth': pa.array(depth, type=pa.float64())}
            for c in selected:
                values = las.data[:, int(c['column_index'])]
                if int(np.isfinite(values).sum()) != int(c['valid_count']):
                    raise ValueError(f'LAS/catalog valid count mismatch: {f.name}')
                arrays[c['parquet_column']] = pa.array(values, mask=~np.isfinite(values), type=pa.float64())
            table = pa.table(arrays).replace_schema_metadata({
                b'record_id': rid.encode(), b'depth_scale': r['depth_scale'].encode(),
                b'depth_unit': r['las_depth_unit'].encode(), b'source_sha256': digest(f).encode(),
                b'depth_source': b'LAS column_index=0; no resampling or unit conversion',
            })
            pq.write_table(table, rel / 'curves' / f'{rid}.parquet', compression='zstd')
            back = pq.read_table(rel / 'curves' / f'{rid}.parquet')
            if not back.equals(table):
                raise ValueError(f'Parquet round trip mismatch: {f.name}')
            shutil.copyfile(f, rel / 'las' / f.name)
            r['source_las_path'] = r['las_path']
            r['las_path'] = f'las/{f.name}'
            r['parquet_path'] = f'curves/{rid}.parquet'
            counts['parquet_files'] += 1
            counts['selected_curve_columns'] += len(selected)
            counts['records_with_plot_curves'] += bool(selected)
            counts['total_samples'] += len(depth)
            if i % 20 == 0 or i == len(records):
                print(f'Build: {i}/{len(records)} LAS', flush=True)
        if set(f.name for f in (source / 'las_with_coordinates').glob('*.las')) != {r['las_filename'] for r in records}:
            raise ValueError('Catalog and LAS file inventory differ')
        write_csv(rel / 'metadata/records.csv', records)
        write_csv(rel / 'metadata/curves.csv', curves)
        if changes:
            write_csv(rel / 'audit/added_plot_curves.csv', changes)
        write_json(rel / 'audit/curve_policy.json', {
            'policy': GLOBAL_POLICY, 'allowed_types': allowed,
            'type_catalog_sha256': digest(type_catalog), 'added_curve_columns': len(changes),
            'provenance': 'Per-record legacy metadata links and LAS measurements preserved',
        })
        write_json(rel / 'stratigraphy/index.json', {'schema_version': 1, 'status': 'not_supplied', 'files': []})
        write_json(srcdir / 'source.json', {'batch': batch, 'archive_sha256': digest(archive),
                   'archive': 'GDS_Iteration_1_Data.zip', 'coordinate_reference': 'all'})
        manifest = {'schema_version': 1, 'release_id': release, 'source_batch': batch,
                    'build_versions': {'lasio': lasio.__version__, 'numpy': np.__version__, 'pyarrow': pa.__version__},
                    'counts': counts, 'coordinate_reference': 'all', 'curve_selection': GLOBAL_POLICY,
                    'path_base': f'releases/{release}/',
                    'notes': ['All finite LAS channels matching original CSV types are selected globally.',
                              'Per-record legacy metadata links and flags retained as provenance.',
                              'Historical audit files describe source input; curve_policy.json describes this release.',
                              'One Parquet per LAS; unselected records contain only depth.',
                              'Stratigraphy was not supplied.'], 'files': []}
        for f in sorted(stage.rglob('*')):
            if f.is_file():
                manifest['files'].append({'key': f.relative_to(stage).as_posix(),
                                          'size': f.stat().st_size, 'sha256': digest(f)})
        write_json(rel / 'manifest.json', manifest)
        stage.rename(output)
        print(json.dumps(counts, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--release', default='v001')
    p.add_argument('--batch', default='batch-001')
    p.add_argument('--type-catalog', type=Path, required=True, help='Full original CSV with Type column')
    a = p.parse_args()
    build(a.archive.resolve(), a.output.resolve(), a.release, a.batch, a.type_catalog.resolve())


if __name__ == '__main__':
    main()
