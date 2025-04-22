##%%writefile 'App.py'
import re
import json
import dash
import dash_table as dt
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash import Dash, html as dhtml, Input, Output, callback_context
from dash_table import DataTable
import plotly.express as px
from plotly import tools
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import boto3
from botocore import UNSIGNED
from botocore.client import Config
import botocore
import lasio
import requests

# ──────── Basic Auth Setup ────────────────────────────────────────────────────────
from dash_auth import BasicAuth

VALID_USERS = {
    "kaust": "seastar2024"
}
# ────────────────────────────────────────────────────────────────────────────────

##### settings ###########################################################################################################################################

name_app = "GDS-Viewer"
token = "pk.eyJ1IjoieXVyaXlrYXByaWVsb3YiLCJhIjoiY2t2YjBiNXl2NDV4YzJucXcwcXdtZHVveiJ9.JSi7Xwold-yTZieIc264Ww"
bucket_for_visualization = "transformed-for-visualization-data-1"
bucket_for_metadata = "for-metadata"
bucket_for_download = "transformed-for-download-data"
folders_name_for_visualization = ['csv/']
folders_name_for_download = ['las/']
list_metadata_files = ['List_of_curves.csv', 'List_of_data-new.csv']
list_metadata = ['Age', 'Name', 'Type', 'lat', 'lon', 'Depth_start', 'Depth_finish', 
                 'Special_mark', 'Reference']

geotime_list = {
    'Pleistocene': 12, 'Neogene': 11, 'Paleogene': 10, 'Cretaceous': 9,
    'Jurassic': 8, 'Triassic': 7, 'Permian': 6, 'Carboniferous': 5,
    'Devonian': 4, 'Silurian': 3, 'Ordovican': 2, 'Cambrian': 1, 'Precambrian': 0
}

list_mnemonics_log500 = ['']
list_mnemonics_log2000 = ['PERM']
list_mnemonics_RES = ['RESD', 'RESS', 'RES', 'SFLU']
list_mnemonics = ['SO', 'DT', 'RHOB', 'GR', 'SGR', 'SONIC', 'GNT', 'SP', 'DTC', 'TOC']

new_columns_name = ['Time', 'Lat', 'Lon', 'Depth_start, feet', 'Depth_finish, feet', 'Well_name']

color_curve = {
    'GR': 'green', 'SGR': 'green', 'GNT': 'green',
    'DT': 'red', 'SONIC': 'red', 'NPHI': 'blue', 'PHI': 'blue',
    'PERM': 'blue', 'RHOB': 'DeepPink', 'SP': 'CornflowerBlue',
    'SFLU': 'CornflowerBlue', 'RES': 'CornflowerBlue',
    'RESD': 'CornflowerBlue', 'RESS': 'CornflowerBlue', 'TOC': 'orange'
}

#### model ##################################################################################################################################################

def make_client_resource():
    client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    resource = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    return client, resource

def find_number_file_name(list_dir, key_word):
    for i, s1 in enumerate(list_dir):
        dirs = s1.split('/')
        if len(dirs) > 2 and dirs[2].split('.')[0] == key_word:
            return i

def find_number_lasfile_name(list_dir, key_word):
    for i, s1 in enumerate(list_dir):
        filename = s1.split('/')[-1]
        if filename.split('.las')[0] == key_word:
            return i
    return -1

def read_curves_csv(client, datadir, option, type_curve):
    keys_log = [obj['Key'] for obj in client.list_objects_v2(
                Bucket=datadir, Prefix=option)['Contents']]
    number_file = find_number_file_name(keys_log, type_curve)
    path_file = keys_log[number_file]
    obj = client.get_object(Bucket=datadir, Key=path_file)
    return pd.read_csv(obj['Body'])

def read_resource_metadata_csv(client, datadir, metadata_file_name, 
                               *args, make_change=False, num_col=None):
    keys_loc = [obj['Key'] for obj in client.list_objects_v2(
                Bucket=datadir, Prefix=metadata_file_name)['Contents']]
    obj = client.get_object(Bucket=datadir, Key=keys_loc[0])
    file_content = pd.read_csv(obj['Body'])
    if make_change:
        column = file_content.columns[num_col]
        file_content[column] = pd.Categorical(
            file_content[column].tolist(), categories=list(args)[0]
        )
        file_content = file_content.sort_values(by=column).reset_index(drop=True)
    return file_content

client, resource = make_client_resource()

curves_data = read_resource_metadata_csv(
    client, bucket_for_metadata, list_metadata_files[0]
)
table_data = read_resource_metadata_csv(
    client, bucket_for_metadata, list_metadata_files[1],
    geotime_list, make_change=True, num_col=0
)

wells_map = curves_data.drop_duplicates(subset=['lat', 'lon']).reset_index(drop=True)
Keys_las = [obj['Key'] for obj in client.list_objects_v2(
    Bucket=bucket_for_download, Prefix=folders_name_for_download[0]
)['Contents']]

#### view ################################################################################################################################################################

for_mapping_list = ['lat', 'lon', 'Name']
plotly_theme = 'seaborn'
dash_theme = dbc.themes.FLATLY

px.set_mapbox_access_token(token)
fig_map = px.scatter_mapbox(
    wells_map[for_mapping_list],
    title='Saudi Arabya Plate', lat="lat", lon="lon",
    hover_name=wells_map.Name, zoom=4,
    mapbox_style='satellite', height=800
)
fig_map.layout.template = plotly_theme
fig_map.update_layout(margin=dict(l=0, r=0, t=35, b=0), clickmode='event+select')
fig_map.update_traces(marker_size=8, marker_color='red')

Tab_map_view = [
    dbc.Row(
        [
            dbc.Col([
                html.Br(),
                dcc.Graph(id='basic-interactions', figure=fig_map)
            ], width=4, md={'size': 8, "offset": 1, 'order': 'first'}),

            dbc.Col([
                html.Br(),
                html.H5("Geologic Time", style={'textAlign': 'center'}),
                html.Br(),
                dbc.Card([
                    dcc.Checklist(
                        id='geo-time',
                        options=[{'label': x, 'value': x} for x in table_data['Geological_Time'].unique()],
                        value=table_data['Geological_Time'].unique(),
                        labelStyle={'padding': '0.3rem 1rem', 'display': 'block', 'cursor': 'pointer'},
                        inputStyle={"margin-right": "10px"}
                    ),
                ]),
            ], width=4, sm={'size': 2, "offset": 0, 'order': 2}),
        ]
    ),
    html.Br(),
    dbc.Row([
        dbc.Col(
            html.H5("Selected Wells", style={'textAlign': 'center'}),
            width=4, md={'size': 6, "offset": 2, 'order': 'first'}
        ),
    ]),
    html.Br(),
    dbc.Row([
        dbc.Col([
            html.Div(id='curves-table'),
            html.Br()
        ], width=4, md={'size': 6, "offset": 2, 'order': 'first'}),
    ]),
]

Tab_log_view = [
    dbc.Row(
        [
            dbc.Col(dbc.Container(html.Div(id='logs'), fluid=True),
                    width=4, lg={'size': 9, "offset": 0, 'order': 1}, md=12),

            dbc.Col([
                html.Br(), html.Br(),
                html.H5("LAS Files", style={'textAlign': 'center'}),
                html.Div(id='downloading'),
                html.Br(), html.Br(), html.Br(), html.Br(), html.Br(),
                html.Br(), html.Br(), html.Br(), html.Br(), html.Br(),
                html.Br(),
                dbc.Container(html.Div(id='choosen-wells')),
            ], width=4, xl={'size': 3, "offset": 0, 'order': 2}, md=12),
        ]
    ),
]

# ──────── App + Auth Initialization ─────────────────────────────────────────────────────────
app = Dash(__name__, external_stylesheets=[dash_theme])
auth = BasicAuth(app, VALID_USERS)
app.title = name_app
server = app.server
# ──────────────────────────────────────────────────────────────────────────────────────

app.layout = dbc.Container([
    dbc.Row([html.H3("GDS-Viewer Dashboard", style={'textAlign': 'center'})]),
    dbc.Row(dbc.Col([
        dbc.Tabs([
            dbc.Tab(Tab_map_view, label="Wells", activeTabClassName="fw-bold fst-italic"),
            dbc.Tab(Tab_log_view, label="Logs", activeTabClassName="fw-bold fst-italic"),
        ]),
    ], width=12)),
], fluid=True)

## Callbacks ##############################################################################################

@app.callback(
    Output('basic-interactions', 'figure'),
    Input('geo-time', 'value'),
    prevent_initial_call=True,
)
def update_display_wells(options_chosen):
    wells_map['intersection_time'] = wells_map['Age'].apply(
        lambda x: 1 if set(x.split('_')).intersection(options_chosen) else 0
    )
    wells_table = wells_map[wells_map['intersection_time'] != 0]
    fig = px.scatter_mapbox(
        wells_table, title='Saudi Arabya Plate', hover_name=wells_table.Name,
        lat="lat", lon="lon", zoom=4, mapbox_style='satellite', height=800
    )
    fig.layout.template = plotly_theme
    fig.update_layout(clickmode='event+select')
    fig.update_traces(marker_size=8, marker_color='red')
    return fig

@app.callback(
    Output('curves-table', 'children'),
    Input('basic-interactions', 'selectedData'),
    prevent_initial_call=True,
)
def display_click_data(clickData):
    if not clickData:
        return dash.no_update

    with open('data.json', 'w') as f:
        json.dump(clickData, f, indent=2)

    data_str = json.dumps(clickData)
    ys = re.findall(r"'lat':\s*(\d+\.\d+)", data_str)
    xs = re.findall(r"'lon':\s*(\d+\.\d+)", data_str)
    xs = [float(x) for x in xs]
    ys = [float(y) for y in ys]

    df_wells = curves_data[list_metadata]
    df_sel = df_wells[df_wells['lon'].isin(xs) & df_wells['lat'].isin(ys)]
    df_sel = df_sel.rename(columns={
        'Age': new_columns_name[0],
        'lat': new_columns_name[1],
        'lon': new_columns_name[2],
        'Depth_start': new_columns_name[3],
        'Depth_finish': new_columns_name[4],
        'Name': new_columns_name[5]
    })

    table = DataTable(
        id='curves-table_1',
        columns=[{'name': col, 'id': col} for col in df_sel.columns],
        data=df_sel.to_dict('records'),
        filter_action='native',
        style_cell={'textAlign': 'left', 'padding': '10px', 'backgroundColor': 'rgb(160,160,160)'},
        style_data={'color': 'grey', 'backgroundColor': 'white'},
        style_header={'backgroundColor': 'rgb(210,210,210)', 'color': 'grey', 'fontWeight': 'bold'},
        style_table={'height': '400px', 'overflowY': 'auto'},
        sort_action="native", sort_mode="multi",
        column_selectable="single", row_selectable="multi", row_deletable=True,
        selected_rows=[], page_action="none", page_current=0, page_size=10
    )
    return table

@app.callback(
    Output('logs', 'children'),
    Input('curves-table_1', "derived_virtual_data"),
    Input('curves-table_1', "derived_virtual_selected_rows"),
    prevent_initial_call=True,
)
def display_logs(rows, derived_virtual_selected_rows):
    if not derived_virtual_selected_rows:
        return dash.no_update

    df = pd.DataFrame(rows)
    sel = df.iloc[derived_virtual_selected_rows]
    cols_ = sel.shape[0]

    fig = tools.make_subplots(
        rows=1, cols=cols_,
        subplot_titles=[f"{sel.iloc[i]['Type']}, {sel.iloc[i]['Well_name']}" for i in range(cols_)]
    ).update_xaxes(side='top', ticklabelposition="inside", title_standoff=10)

    # aggregate formations & times
    formations = []
    times = []
    for i in range(cols_):
        rec = sel.iloc[i]
        data_curves = read_curves_csv(
            client, bucket_for_visualization, folders_name_for_visualization[0], rec['Type']
        )
        df_curve = data_curves[
            (data_curves['Well_name'] == rec['Well_name']) &
            (data_curves['lat'] == rec['Lat']) &
            (data_curves['lon'] == rec['Lon']) &
            (data_curves['DEPTH'] >= rec['Depth_start, feet']) &
            (data_curves['DEPTH'] <= rec['Depth_finish, feet'])
        ]
        formations += list(df_curve.Formation)
        times += list(df_curve.Time)

    formations = pd.unique(formations)
    times = pd.unique(times)
    colors = {f: f'rgba({60+i*10},{256//(i+1)},{i*15},0.2)' for i,f in enumerate(formations)}
    colors_t = {t: f'rgba({60+i*10},{256//(i+1)},{i*15},0.2)' for i,t in enumerate(times)}

    # plot each trace
    for i in range(cols_):
        rec = sel.iloc[i]
        data_curves = read_curves_csv(
            client, bucket_for_visualization, folders_name_for_visualization[0], rec['Type']
        )
        df_curve = data_curves[
            (data_curves['Well_name'] == rec['Well_name']) &
            (data_curves['lat'] == rec['Lat']) &
            (data_curves['lon'] == rec['Lon']) &
            (data_curves['DEPTH'] >= rec['Depth_start, feet']) &
            (data_curves['DEPTH'] <= rec['Depth_finish, feet'])
        ].drop_duplicates(subset=['DEPTH'])

        x = df_curve[df_curve.columns[1]]
        y = df_curve[df_curve.columns[0]]
        fig.add_trace(
            go.Scatter(
                x=x, y=y, mode='lines',
                line=dict(width=2),
                marker_color=color_curve[rec['Type']],
                name=f"{rec['Lat']}_{rec['Lon']}_{rec['Well_name']}_{rec['Type']}",
                hovertemplate=(
                    f"{rec['Type']}: %{{x:.2f}}<br>"
                    "Depth: %{y:.1f}<br>"
                    f"Well: {rec['Lat']}_{rec['Lon']}_{rec['Well_name']}<br>"
                    "Formation: %{customdata}<extra></extra>"
                ),
                customdata=df_curve.Formation
            ), row=1, col=i+1
        )

        # axis scaling
        if rec['Type'] in list_mnemonics_log500:
            fig.update_xaxes(type="log", range=[np.log10(1), np.log10(500)], row=1, col=i+1)
        elif rec['Type'] in list_mnemonics_log2000 + list_mnemonics_RES:
            fig.update_xaxes(type="log", range=[np.log10(1), np.log10(2000)], row=1, col=i+1)
        elif rec['Type'] in ['NPHI','PHI','SONIC','DT']:
            fig.update_xaxes(autorange='reversed', range=[40, -15], row=1, col=i+1)

    fig.update_layout(
        autosize=False, height=2500,
        margin=dict(l=10, r=20, t=70, b=0),
        yaxis=dict(autorange='reversed'),
        hovermode="y unified",
        template=plotly_theme
    )

    return dcc.Graph(id='logs_', figure=fig)

@app.callback(
    Output("downloading", "children"),
    Input('curves-table_1', "derived_virtual_data"),
    Input('curves-table_1', "derived_virtual_selected_rows"),
    prevent_initial_call=True,
)
def display_las(rows, derived_virtual_selected_rows):
    if not derived_virtual_selected_rows:
        return dash.no_update

    df = pd.DataFrame(rows)
    sel = df.iloc[derived_virtual_selected_rows]
    link_items = []
    for _, rec in sel.iterrows():
        name = f"{rec['Lat']}_{rec['Lon']}_{rec['Depth_start, feet']}_{rec['Depth_finish, feet']}_{rec['Well_name']}"
        idx = find_number_lasfile_name(Keys_las, name)
        if idx != -1:
            url = client.generate_presigned_url(
                ClientMethod='get_object',
                Params={'Bucket': bucket_for_download, 'Key': Keys_las[idx]}
            )
            link_items.append(dbc.ListGroupItem(name, href=url,
                className="list-group-item list-group-item-action list-group-item-secondary text-center"))
        else:
            link_items.append(dbc.ListGroupItem(name + " - No Las File",
                className="list-group-item list-group-item-action list-group-item-secondary text-center"))
    return dbc.ListGroup(link_items)

@app.callback(
    Output('choosen-wells', 'children'),
    Input('curves-table_1', "derived_virtual_data"),
    Input('curves-table_1', "derived_virtual_selected_rows")
)
def change_map(rows, derived_virtual_selected_rows):
    if not derived_virtual_selected_rows:
        return dash.no_update

    df = pd.DataFrame(rows)
    sel = df.iloc[derived_virtual_selected_rows]
    fig_map_ = px.scatter_mapbox(
        sel, hover_name=sel.Well_name, title='Selected Wells',
        lat="Lat", lon="Lon", zoom=4, mapbox_style='satellite'
    )
    fig_map_.update_layout(margin=dict(l=0, r=0, t=35, b=0), clickmode='event+select')
    fig_map_.update_traces(marker_size=9, marker_color='red')
    return dcc.Graph(id='scatter_plot', figure=fig_map_)

if __name__ == '__main__':
    app.run_server()
