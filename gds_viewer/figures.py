import hashlib
import html
import math
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .data import DataError, require

COLORS = {'GR': '#177245', 'SGR': '#177245', 'DT': '#c84f43', 'SONIC': '#c84f43',
          'NPHI': '#2874a6', 'PHI': '#2874a6', 'RHOB': '#b14382', 'TOC': '#d78d24'}
LOG_TYPES = {'PERM', 'RESD', 'RESS', 'RES', 'SFLU'}
REVERSED_TYPES = {'NPHI', 'PHI', 'DT', 'SONIC'}


def map_figure(records, selected=(), style='carto-positron'):
    selected = set(selected or [])
    fig = go.Figure(go.Scattermap(
        lat=[float(r['latitude']) for r in records], lon=[float(r['longitude']) for r in records],
        customdata=[[r['record_id'], html.escape(r['well_name_original']), r['curve_mnemonics'] or 'No selected curves'] for r in records],
        marker={'size': [13 if r['record_id'] in selected else 8 for r in records],
                'color': ['#007f86' if r['record_id'] in selected else '#c64343' for r in records]},
        mode='markers', hovertemplate='<b>%{customdata[1]}</b><br>%{lat:.5f}, %{lon:.5f}<br>%{customdata[2]}<extra></extra>'))
    fig.update_layout(map={'style': style, 'center': {'lat': 24.5, 'lon': 50.5}, 'zoom': 3.4},
        margin={'l': 0, 'r': 0, 't': 0, 'b': 0}, height=510, dragmode='select', clickmode='event',
        uirevision='gds-map', paper_bgcolor='white', font={'family': 'Arial'})
    return fig


def build_figures(repo, snap, log_ids, lower=None, upper=None, scale='auto', reverse='auto'):
    log_ids = list(dict.fromkeys(log_ids or []))
    require(log_ids, 'Select at least one curve in the table.')
    require(all(k in snap.curves for k in log_ids), 'Curve selection is no longer valid; reload the page.')
    if lower is not None and upper is not None:
        require(lower < upper, 'Depth start must be smaller than depth end.')
    figures = []
    # All selected curves are retained, grouped into readable sets of six tracks.
    for start in range(0, len(log_ids), 6):
        group = log_ids[start:start + 6]
        titles = [html.escape(snap.curves[k]['mnemonic']) + '<br>' + html.escape(snap.records[snap.curves[k]['record_id']]['well_name_original']) for k in group]
        fig = make_subplots(rows=1, cols=len(group), subplot_titles=titles, horizontal_spacing=0.09 if len(group) > 1 else 0)
        for col, key in enumerate(group, 1):
            curve = snap.curves[key]
            record = snap.records[curve['record_id']]
            table = repo.table(snap, curve['record_id'])
            depth = np.asarray(table['depth'].to_pylist(), dtype=float)
            values = np.asarray(table[curve['parquet_column']].to_pylist(), dtype=float)
            mask = np.ones(len(depth), dtype=bool)
            if lower is not None:
                mask &= depth >= lower
            if upper is not None:
                mask &= depth <= upper
            y, x = depth[mask], values[mask].copy()
            require(len(y) > 0, f'No samples in the selected depth interval: {record["well_name_original"]}')
            islog = scale == 'log' or (scale == 'auto' and curve['mnemonic'] in LOG_TYPES)
            if islog:
                x[x <= 0] = np.nan
            require(np.isfinite(x).any(), f'No plottable values for {curve["mnemonic"]}; try linear scale or another interval.')
            invert = reverse == 'yes' or (reverse == 'auto' and curve['mnemonic'] in REVERSED_TYPES)
            unit = curve['unit'] if curve['unit'] not in {'_', ''} else 'unit not specified'
            fig.add_trace(go.Scattergl(x=[float(v) if math.isfinite(v) else None for v in x], y=y.tolist(),
                mode='lines', line={'color': COLORS.get(curve['mnemonic'], '#376c85'), 'width': 1.4},
                connectgaps=False, name=curve['mnemonic'], hovertemplate='%{x:.5g}<br>Depth: %{y:.4g}<extra></extra>'), row=1, col=col)
            fig.update_xaxes(title_text=html.escape(unit), side='top', type='log' if islog else 'linear',
                autorange='reversed' if invert else True, showgrid=True, gridcolor='#e7edef', row=1, col=col)
            label = f'Artificial depth ({record["las_depth_unit"]} in LAS)' if record['depth_scale'] == 'artificial' else f'Depth ({record["las_depth_unit"]})'
            fig.update_yaxes(title_text=label, autorange='reversed', showgrid=True, gridcolor='#e7edef', row=1, col=col)
            for interval in repo.intervals(snap, curve['record_id']):
                top, base = float(interval['top']), float(interval['base'])
                require(top <= base, 'Invalid stratigraphy interval')
                label = interval.get('formation', '') or interval.get('period', '')
                hue = int(hashlib.sha256(label.encode()).hexdigest()[:4], 16) % 360
                fig.add_hrect(y0=top, y1=base, fillcolor=f'hsla({hue},45%,70%,0.2)', line_width=0,
                    annotation_text=html.escape(label), annotation_position='inside right', row=1, col=col)
        fig.update_layout(template='plotly_white', height=1050, width=max(500, 320 * len(group)),
            margin={'l': 65, 'r': 30, 't': 120, 'b': 40}, showlegend=False, hovermode='closest',
            font={'family': 'Arial', 'size': 11})
        figures.append(fig)
    return figures
