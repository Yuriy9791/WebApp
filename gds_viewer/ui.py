import logging
import re
from pathlib import Path
from urllib.parse import quote
from dash import Dash, Input, Output, State, ctx, dash_table, dcc, html
from dash.exceptions import PreventUpdate
from flask import abort, redirect, send_file
from botocore.exceptions import ClientError
from .data import Repository, LocalStore, DataError, ages, visible_records, update_selection
from .curve_policy import GLOBAL_POLICY
from .publications import split_references, reference_id, pdf_key, pdf_filename
from .figures import map_figure, build_figures

log = logging.getLogger(__name__)
COLUMNS = ['Well','Curve','Unit','Start','End','Depth scale','Depth unit','Age','Reference','Notes','Latitude','Longitude']


def publication(text, download=False):
    if download:
        return html.P(html.A(text, href='/source-pdf/' + reference_id(text), title='Download PDF'), className='reference')
    children, offset = [], 0
    for match in re.finditer(r'https?://[^\s<>]+', text or ''):
        children.extend([text[offset:match.start()], html.A(match.group(), href=match.group(), target='_blank', rel='noopener noreferrer')])
        offset = match.end()
    children.append((text or '')[offset:])
    return html.P(children, className='reference')


def make_app(repo=None):
    repo = repo or Repository.from_env()
    app = Dash(__name__, title='GDS Viewer', assets_folder=str(Path(__file__).resolve().parents[1] / 'assets'), suppress_callback_exceptions=True)

    def layout():
        try:
            snap = repo.active()
        except Exception:
            log.exception('Cannot open active GDS release')
            return html.Main([html.H1('GDS Viewer'),html.H2('Data connection is not ready'),html.P('Check GDS_S3_BUCKET, AWS_REGION and read access. Run check_s3.sh to diagnose.'),html.A('Retry',href='/')],className='shell')
        age_values = sorted({a for r in snap.records.values() for a in ages(r['Age'])})
        types = list(snap.allowed_curve_types) or sorted({c['mnemonic'] for c in snap.curves.values()})
        return html.Main([
            dcc.Store(id='release',data=snap.descriptor),
            html.Header([html.Div([html.Div('GDS',className='brand'),html.Div([html.H1('Geoscience Data Viewer'),html.P('Published well logs · Arabian Plate',className='subtitle')])],className='title-row'),html.Div([html.A('Reload dataset',href='/',className='text-link')],className='header-meta')]),
            dcc.Tabs(id='tabs',value='wells',children=[dcc.Tab(label='Wells',value='wells'),dcc.Tab(label='Logs',value='logs')]),
            html.Section([
                html.Div([html.Div([html.H2('Explore wells'),html.P('Click a point to add or remove a well. Drag a rectangle to add wells.',className='muted'),dcc.Loading(dcc.Graph(id='well-map',figure=map_figure(list(snap.records.values())),config={'displaylogo':False,'scrollZoom':True,'modeBarButtonsToAdd':['select2d']})),html.Div(id='map-count',className='muted')],className='map-panel'),
                    html.Aside([html.Label('Search well or publication'),dcc.Input(id='search',type='search',debounce=True,placeholder='Name or reference…',className='search'),html.Label('Geological age'),dcc.Dropdown(id='age-filter',options=age_values,value=[],multi=True,placeholder='All ages'),html.Label('Curve type'),dcc.Dropdown(id='curve-filter',options=types,placeholder='All curve types'),html.Label('Selected wells'),dcc.Dropdown(id='well-select',options=[{'label':r['well_name_original'],'value':r['record_id']} for r in snap.records.values()],value=[],multi=True,placeholder='Select on the map or search this list'),html.P('Use this list for records sharing the same map position.',className='hint'),html.Button('Clear wells',id='clear-wells',className='secondary'),html.Hr(),html.P('Coordinates: WGS 84 · all',className='hint'),html.P('All curve types from the original CSV are supported. Available curves come from LAS measurements.',className='hint')],className='filters')],className='explorer'),
                html.Div([html.H2('Available curves'),html.P(id='table-note',className='muted'),dash_table.DataTable(id='curve-table',data=[],columns=[{'name':c,'id':c,'type':'numeric' if c in {'Start','End','Latitude','Longitude'} else 'text'} for c in COLUMNS],row_selectable='multi',selected_row_ids=[],sort_action='native',filter_action='native',page_size=12,style_table={'overflowX':'auto'},style_cell={'fontFamily':'Arial','fontSize':12,'padding':'10px','textAlign':'left','minWidth':'90px','maxWidth':'250px','whiteSpace':'normal'},style_header={'backgroundColor':'#eef4f5','fontWeight':'bold','color':'#193c48'},tooltip_data=[],tooltip_duration=None),html.Div([html.Button('Select filtered curves',id='select-curves',className='secondary'),html.Button('Clear curves',id='clear-curves',className='secondary')],className='toolbar'),html.P('Choose curves, open Logs, then click Build logs.',className='hint')],className='table-panel')],id='wells-panel'),
            html.Section([html.Div([html.Div([html.H2('Well logs'),html.P('One track per curve; depth increases downwards.',className='muted')]),html.Button('Build logs',id='build-logs',className='primary')],className='section-heading'),html.Div([
                html.Div([html.Label('Depth start'),dcc.Input(id='depth-start',type='number',placeholder='Full interval')]),html.Div([html.Label('Depth end'),dcc.Input(id='depth-end',type='number',placeholder='Full interval')]),html.Div([html.Label('Horizontal scale'),dcc.Dropdown(id='xscale',options=[{'label':'Automatic','value':'auto'},{'label':'Linear','value':'linear'},{'label':'Logarithmic','value':'log'}],value='auto',clearable=False)]),html.Div([html.Label('Reverse horizontal axis'),dcc.Dropdown(id='xreverse',options=[{'label':'Automatic','value':'auto'},{'label':'Yes','value':'yes'},{'label':'No','value':'no'}],value='auto',clearable=False)])],className='plot-controls'),html.P('Depth bounds use each track’s original LAS scale. Artificial depth is labelled separately. Logarithmic axes omit values ≤ 0.',className='hint'),html.Div(id='selection-status',className='muted'),dcc.Loading(html.Div(id='plot-output',children=html.P('Select curves on the Wells tab to begin.',className='empty'))),html.Div(id='publications'),html.Div([html.H2('Selected well locations'),dcc.Graph(id='selected-map',figure=map_figure([]),config={'displaylogo':False})],className='selected-map')],id='logs-panel',style={'display':'none'}),
            html.Section([html.H2('Download LAS'),html.Div(id='downloads'),html.P(id='stratigraphy-status',className='hint')],className='downloads'),html.Footer('GDS Viewer · Iteration 2 · LAS measurements / CSV metadata',className='muted')],className='shell')
    app.layout = layout

    def source_documents(snap, chosen, rows):
        refs = list(dict.fromkeys(ref for row in rows if row['id'] in chosen
                                  for ref in split_references(row['Reference'])))
        children = [html.H2('Publications')]
        for reference in refs:
            try:
                available = repo.source_pdf_exists(reference)
            except ClientError:
                log.exception('Cannot check publication PDF')
                available = False
            children.append(publication(reference, download=available))
        return children

    @app.callback(Output('wells-panel','style'),Output('logs-panel','style'),Input('tabs','value'))
    def tabs(value):
        return {'display':'block' if value=='wells' else 'none'},{'display':'block' if value=='logs' else 'none'}

    @app.callback(Output('well-select','value'),Input('well-map','clickData'),Input('well-map','selectedData'),Input('clear-wells','n_clicks'),State('well-select','value'),State('release','data'),prevent_initial_call=True)
    def select_wells(click,box,clear,current,descriptor):
        snap=repo.from_descriptor(descriptor); prop=ctx.triggered[0]['prop_id']
        if prop.startswith('clear-wells'): return []
        event=click if prop.endswith('clickData') else box
        if event is None: raise PreventUpdate
        return update_selection(current,event,set(snap.records),toggle=prop.endswith('clickData'))

    @app.callback(Output('well-map','figure'),Output('map-count','children'),Input('age-filter','value'),Input('search','value'),Input('curve-filter','value'),Input('well-select','value'),State('release','data'))
    def draw_map(age,search,curve_type,selected,descriptor):
        snap=repo.from_descriptor(descriptor); records=visible_records(snap,age,search,curve_type)
        return map_figure(records,selected),f'{len(records)} wells on map · {len(selected or [])} selected'

    @app.callback(Output('curve-table','data'),Output('curve-table','selected_row_ids'),Output('curve-table','selected_rows'),Output('table-note','children'),Output('curve-table','tooltip_data'),Input('well-select','value'),Input('select-curves','n_clicks'),Input('clear-curves','n_clicks'),State('curve-table','derived_virtual_data'),State('curve-table','selected_row_ids'),State('release','data'))
    def fill_table(selected,select_click,clear_click,displayed,chosen,descriptor):
        snap=repo.from_descriptor(descriptor); rows=repo.curve_rows(snap,selected); valid={r['id'] for r in rows}
        if ctx.triggered_id=='select-curves': chosen=[r['id'] for r in (displayed if displayed is not None else rows) if r['id'] in valid]
        elif ctx.triggered_id=='clear-curves': chosen=[]
        else: chosen=[k for k in (chosen or []) if k in valid]
        note=f'{len(selected or [])} selected wells · {len(rows)} available curves'
        if selected and not rows: note+='. No supported curves with data; LAS files remain downloadable.'
        tooltip=[{c:{'value':str(r[c]),'type':'text'} for c in COLUMNS} for r in rows]
        selected_indices=[i for i,r in enumerate(rows) if r['id'] in chosen]
        return rows,chosen,selected_indices,note,tooltip

    @app.callback(Output('selection-status','children'),Input('curve-table','selected_row_ids'))
    def selection_status(ids): return f'{len(ids or [])} curves selected. Click Build logs after changing selection or depth bounds.'

    @app.callback(Output('plot-output','children'),Output('publications','children'),Input('build-logs','n_clicks'),Input('well-select','value'),Input('curve-table','selected_row_ids'),State('release','data'),State('depth-start','value'),State('depth-end','value'),State('xscale','value'),State('xreverse','value'),prevent_initial_call=True)
    def plot(n_clicks,wells,chosen,descriptor,lower,upper,scale,reverse):
        if ctx.triggered_id!='build-logs': return html.P('Selection changed. Click Build logs to update the plots.',className='empty'),[]
        try:
            snap=repo.from_descriptor(descriptor); rows=repo.curve_rows(snap,wells); allowed={r['id'] for r in rows}; chosen=[k for k in (chosen or []) if k in allowed]
            figures=build_figures(repo,snap,chosen,lower,upper,scale,reverse)
            return [html.Div(dcc.Graph(figure=f,responsive=True,style={'width':'100%','minWidth':f'{f.layout.width}px','height':f'{f.layout.height}px'},config={'displaylogo':False,'toImageButtonOptions':{'format':'png','filename':'GDS_logs'}}),className='log-scroll') for f in figures],source_documents(snap,chosen,rows)
        except DataError as error: return html.P(str(error),className='error'),[]
        except Exception:
            log.exception('Log rendering failed')
            return html.P('Could not read selected data. Check the server log and S3 connection.',className='error'),[]

    @app.callback(Output('downloads','children'),Output('stratigraphy-status','children'),Output('selected-map','figure'),Input('well-select','value'),State('release','data'))
    def downloads(selected,descriptor):
        snap=repo.from_descriptor(descriptor); records=[snap.records[k] for k in dict.fromkeys(selected or []) if k in snap.records]
        links=[html.A(r['las_filename'],href=f'/download/{snap.release}/{quote(r["record_id"],safe="")}',className='download-link') for r in records]
        status='Stratigraphy is not supplied for this release.' if not snap.stratigraphy.get('files') else 'Compatible stratigraphy intervals are displayed on log tracks.'
        return links or html.P('Select one or more wells.',className='muted'),status,map_figure(records,[r['record_id'] for r in records])

    @app.server.get('/healthz')
    def health(): return {'status':'ok'}

    @app.server.get('/readyz')
    def ready():
        try:
            snap=repo.active()
            return {'status':'ready','release':snap.release,'records':len(snap.records),'curves':len(snap.curves),
                    'supported_curve_policy': GLOBAL_POLICY, 'source_pdf_downloads': True}
        except Exception:
            log.exception('Readiness check failed')
            return {'status':'data_unavailable'},503

    @app.server.get('/download/<release>/<record_id>')
    def download(release,record_id):
        try:
            snap=repo.snapshot(release); record=snap.records.get(record_id)
            if record is None: abort(404)
            relative=record['las_path']; key=f'releases/{release}/{relative}'
            if key not in {x['key'] for x in snap.manifest['files']}: abort(404)
            if isinstance(repo.store,LocalStore):
                repo.file(snap,relative)
                return send_file(repo.store.path(key),as_attachment=True,download_name=record['las_filename'])
            response=redirect(repo.store.download_url(key,record['las_filename']),302)
            response.headers['Cache-Control']='no-store'
            return response
        except (DataError,FileNotFoundError): abort(404)

    @app.server.get('/source-pdf/<publication_id>')
    def download_source_pdf(publication_id):
        try:
            reference = repo.source_reference(publication_id)
            if reference is None:
                abort(404)
            key, filename = pdf_key(reference), pdf_filename(reference)
            if not repo.store.exists(key):
                abort(404)
            if isinstance(repo.store, LocalStore):
                response = send_file(repo.store.path(key), as_attachment=True,
                                     download_name=filename, mimetype='application/pdf')
            else:
                response = redirect(repo.store.download_url(key, filename), 302)
            response.headers['Cache-Control'] = 'no-store'
            return response
        except (DataError,FileNotFoundError):
            abort(404)
        except ClientError:
            log.exception('Cannot download publication PDF')
            abort(503)
    return app
