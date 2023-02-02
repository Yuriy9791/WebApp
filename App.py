##%%writefile 'App.py'
import re
import json
import dash
import dash_table as dt
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash import Dash, html, Input, Output, callback_context
import dash_table as dt
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

##### settings ###########################################################################################################################################

name_app = "GDS-Viewer"
token = "pk.eyJ1IjoieXVyaXlrYXByaWVsb3YiLCJhIjoiY2t2YjBiNXl2NDV4YzJucXcwcXdtZHVveiJ9.JSi7Xwold-yTZieIc264Ww"
bucket_for_visualization="transformed-for-visualization-data-1"
bucket_for_metadata="for-metadata"
bucket_for_download="transformed-for-download-data"
folders_name_for_visualization = ['csv/']#['curves', 'stratigraphy']
folders_name_for_download = ['las/']
list_metadata_files = ['List_of_curves.csv', 'List_of_data-new.csv']
# for display information
list_metadata = ['Age', 'Name', 'Type', 'lat', 'lon', 'Depth_start', 'Depth_finish', 
                 'Special_mark','Reference']


geotime_list = dict()
geotime_list['Pleistocene'] = 12
geotime_list['Neogene'] = 11
geotime_list['Paleogene'] = 10
geotime_list['Cretaceous'] = 9
geotime_list['Jurassic'] = 8
geotime_list['Triassic'] = 7
geotime_list['Permian'] = 6
geotime_list['Carboniferous'] = 5
geotime_list['Devonian'] = 4
geotime_list['Silurian'] = 3
geotime_list['Ordovican'] = 2
geotime_list['Cambrian'] = 1
geotime_list['Precambrian'] = 0

# log curves with different axis scale
list_mnemonics_log500 = ['']
list_mnemonics_log2000 =  ['PERM']
list_mnemonics_RES = ['RESD', 'RESS', 'RES', 'SFLU']
list_mnemonics = ['SO', 'DT', 'RHOB', 'GR', 'SGR', 'SONIC', 'GNT', 'SP', 'DTC']

# changing in columns name
new_columns_name = ['Time', 'Lat', 'Lon', 'Depth_start, feet', 'Depth_finish, feet', 'Well_name']

# color curve
color_curve = {
               'GR': 'green',
               'SGR': 'green',
               'GNT': 'green',
               'DT': 'red',
               'SONIC': 'red',
               'NPHI': 'blue',
               'PHI': 'blue',
               'PERM': 'blue',
               'RHOB': 'DeepPink ',
               'SP':    'CornflowerBlue',
               'SFLU':  'CornflowerBlue',
               'RES' :  'CornflowerBlue',
               'RESD' :  'CornflowerBlue',
               'RESS' :  'CornflowerBlue',
                            
               }

#### model ##############################################################################################################################################################


def make_client_resource():
    """
    Connect with s3 aws.
    """
    client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    resource = boto3.client('s3', config=Config(signature_version=UNSIGNED))
                             
    return client, resource


def find_number_file_name(list_dir, key_word):
    for i in range(len(list_dir)):
        s1 = list_dir[i]
        dirs = s1.split('/')
        if len(dirs) > 2:
            mnemonics = dirs[2].split('.')[0]
            if mnemonics == key_word:
                return i


def find_number_lasfile_name(list_dir, key_word):
    j = -1
    for i in range(0, len(list_dir)):
        s1 = list_dir[i]
        
        filename = s1.split('/')[-1]
        mnemonics = filename.split('.las')[0]
        if mnemonics == key_word:
            j = i
            break
    
    return j

        
def read_curves_csv(client, datadir, option, type_curve):
    keys_log = [obj['Key'] for obj in client.list_objects_v2(
                Bucket=datadir, Prefix=option)['Contents']]
    
    number_file = find_number_file_name(keys_log, type_curve)
    path_file = keys_log[number_file]
    obj = client.get_object(Bucket = datadir,
                                Key = path_file
                                )
    return pd.read_csv(obj['Body'])


def read_resource_metadata_csv(client, datadir, metadata_file_name, 
                               *args, make_change = False, num_col = None):
    keys_loc = [obj['Key'] for obj in client.list_objects_v2(\
                Bucket=datadir, Prefix=metadata_file_name)['Contents']]
    
    obj = client.get_object(Bucket=datadir, Key=keys_loc[0])
    file_content = pd.read_csv(obj['Body'])
    
    if make_change:
        column = file_content.columns[num_col]
        file_content[column] = pd.Categorical(file_content[column].tolist(), 
                                              categories = list(args)[0])
        file_content = file_content.sort_values(by=column).reset_index(drop=True)
        
    return file_content


client, recourse = make_client_resource()

curves_data = read_resource_metadata_csv(client, bucket_for_metadata, list_metadata_files[0])
table_data = read_resource_metadata_csv(client, bucket_for_metadata, list_metadata_files[1], 
                                            geotime_list, make_change=True, num_col=0)


wells_map =  curves_data.copy() # curves_data[['Age','lat', 'lon', 'Name']]
wells_map = wells_map.drop_duplicates(subset=['lat', 'lon']).reset_index(drop=True)

Keys_las = [obj['Key'] for obj in client.list_objects_v2(Bucket=bucket_for_download, 
                                                                           Prefix=folders_name_for_download[0])\
                                                                           ['Contents']]

#### view ################################################################################################################################################################

for_maping_list = ['lat', 'lon', 'Name']
plotly_theme = 'seaborn'#'plotly_dark'#'ggplot2'#'plotly'#'simple_white' #
dash_theme = dbc.themes.FLATLY#CYBORG #SUPERHERO #

px.set_mapbox_access_token(token)
fig_map = px.scatter_mapbox(wells_map[for_maping_list], title='Saudi Arabya Plate',
                            lat="lat", lon="lon",  hover_name=wells_map.Name, zoom=4, mapbox_style='satellite', height= 800)
fig_map.layout.template = plotly_theme 
fig_map.update_layout(margin=dict(l=0, r=0, t=35, b=0),
                      clickmode='event+select')
fig_map.update_traces(marker_size=8, marker_color='red')

#fig_logs = tools.make_subplots(rows=1, cols=1).\
#                                  update_xaxes(side='top', ticklabelposition="inside",
#                                               title_standoff = 1)

Tab_map_view = [
                 dbc.Row(
                          [
                            dbc.Col([
                                      html.Br(),
                                      dcc.Graph(id='basic-interactions', figure=fig_map)
                                    ],
                                      width=4, md={'size': 8,  "offset": 1, 'order': 'first'}
                                   ),
                            dbc.Col(
                                     [
                                         html.Br(), 
                                         html.H5(children="Geologic Time", style = {'textAlign' : 'center'}),
                                         html.Br(),
                                         dbc.Card(
                                                   [
                                                     
                                                     dcc.Checklist(id='geo-time', 
                                                                  options=[
                                                                             {'label': x, 'value': x, 'disabled':False} for x in table_data['Geological_Time'].unique()
                                                                           ],
                                                                  value=table_data['Geological_Time'].unique(), 
                                                                  labelStyle={
                                                                  #'background':'#A5D6A7',
                                                                  'padding':'0.3rem 1rem',
                                                                   # 'border-radius':'5.5rem',  
                                                                  'display': 'block',
                                                                  'cursor': 'pointer'
                                                                             },
                                                                   inputStyle={"margin-right": "10px"}), 
                                                    ]
                                                  ),
                                                                             
                                     ], width=4, sm={'size': 2,  "offset": 0, 'order': 2}
                                   )
                              
                          ]
                        ), 
               
                html.Br(),
                dbc.Row([ dbc.Col(
                                   html.H5(children="Selected Wells", style = {'textAlign' : 'center'}), 
                                   width=4, md={'size': 6,  "offset": 2, 'order': 'first'}
                                  ),
                         ]),          
    
                html.Br(),
    
                dbc.Row(
                         [    
                              dbc.Col([
                                         html.Div(id='curves-table'),
                                         html.Br()
                                       ], 
                                       width=4, md={'size': 6,  "offset": 2, 'order': 'first'}),
                              
                         ]
                       ),
                                                  
               ]


Tab_log_view = [
                 
                 dbc.Row(
                          [
                           
                           dbc.Col(dbc.Container(html.Div(id='logs'), fluid=True), width=4, lg={'size': 8, "offset": 0, 'order': 1}, md=12),
                           dbc.Col( [
                                      html.Br(),
                                      html.Br(),
                                      html.H5(children="LAS Files", style = {'textAlign' : 'center'}),
                                      html.Div(id='downloading'),
                                      html.Br(),
                                      html.Br(),
                                      #html.H5(children="Choosen Wells", style = {'textAlign' : 'center'}),
                                      dbc.Container(html.Div(id='choosen-wells')),
                                      
                                    ], width=4, xl={'size': 4, "offset": 0, 'order': 2}, md=12),
                              
                              
                          ]
                         ),
    
                ]


app = dash.Dash(__name__, external_stylesheets=[dash_theme])
app.title = name_app
server = app.server

app.layout = dbc.Container([
                        dbc.Row([
                                 html.H3(children="GDS-Viewer Dashboard", style = {'textAlign' : 'center'}),
                                ]
                               ),
    
                         dbc.Row(
                                dbc.Col([
                                           dbc.Tabs(
                                                    [
                                                    #dbc.Tab(Tab_content, label="Content", activeTabClassName="fw-bold fst-italic"),
                                                     dbc.Tab(Tab_map_view, label="Wells", activeTabClassName="fw-bold fst-italic"),
                                                     dbc.Tab(Tab_log_view, label="Logs", activeTabClassName="fw-bold fst-italic"),
                                                    ]
                                                   ),
                                            ], width={'size': 12, 'offset': 0})),
        
      
                       ], fluid=True)


## Callbacks ##############################################################################################

@app.callback(
    Output(component_id='basic-interactions', component_property='figure'),
    Input(component_id='geo-time', component_property='value'),
    prevent_initial_call=True,    
)
def update_display_wells(options_chosen):
    
    wells_map['intersection_time'] = wells_map['Age'].apply(lambda x: 0 if set(x.split('_')).intersection(set((options_chosen)))==set()\
                                                           else 1)
    wells_table =  wells_map[wells_map['intersection_time'] !=0]  
    
    fig_map = px.scatter_mapbox(wells_table, title='Saudi Arabya Plate', hover_name=wells_table.Name,
                                lat="lat", lon="lon",  zoom=4, mapbox_style='satellite', height= 800)
    fig_map.layout.template = plotly_theme 
    fig_map.update_layout(clickmode='event+select')
    fig_map.update_traces(marker_size=8, marker_color='red')

    return fig_map


@app.callback(Output('curves-table', 'children'),
              Input('basic-interactions', 'selectedData'))
def display_click_data(clickData):
    if clickData:
        with open('data.json', 'w') as f:
            data = json.dumps(clickData, indent=2)
            json.dump(data, f)
        
        data = str(json.loads(json.dumps(clickData, indent=2)))
        
        ys = re.findall(r"'lat': \d\d.\d\d", data)
        ys_3 = re.findall(r"'lat': \d\d.\d", data)
        for y_3 in ys_3:
            if y_3 not in re.findall(r"'lat': \d\d.\d", " ".join(ys)):
                ys.append(y_3)
        xs = re.findall(r"'lon': \d\d.\d\d", data)
        xs_3 = re.findall(r"'lon': \d\d.\d", data)
        for x_3 in xs_3:
            if x_3 not in re.findall(r"'lon': \d\d.\d", " ".join(xs)):
                xs.append(x_3)
    
        x = []
        y = []
        for x_s, y_s in zip(xs, ys):
            if re.findall(r'\d\d.\d\d',x_s) !=[]:
                x_number = float(re.findall(r'\d\d.\d\d',x_s)[0])
            else:
                x_number = float(re.findall(r'\d\d.\d',x_s)[0])
            x.append(x_number)
            if re.findall(r'\d\d.\d\d',y_s) !=[]:
                y_number = float(re.findall(r'\d\d.\d\d',y_s)[0])
            else:
                y_number = float(re.findall(r'\d\d.\d',y_s)[0])
            y.append(y_number)
            
        
        well_curves = curves_data[list_metadata] #well_table_display[list_metadata] # [['Age', 'lat', 'lon', 'Depth_start', 'Depth_finish','Type', 'Name', 'Special_mark']]
        df_ = well_curves[(well_curves['lon'].isin(x)) & (well_curves['lat'].isin(y))]
        
        ## Rename
        df_ = df_.rename(columns={'Age': new_columns_name[0], 'lat':new_columns_name[1], 
                                  'lon':new_columns_name[2], 'Depth_start': new_columns_name[3], 
                                 'Depth_finish': new_columns_name[4], 'Name': new_columns_name[5]})
        
        table = DataTable(id='curves-table_1',
                          columns = [{'name': col, 'id': col} for col in df_.columns],
                          data = df_.to_dict('records'),
                          filter_action='native',
                          style_cell={'textAlign': 'left',
                                     'padding': '10px',
                                     'backgroundColor': 'rgb(160, 160, 160)'},
                          style_data={
                                      'color': 'grey',
                                      'backgroundColor': 'white',
                                                                            
                                     },
                          
                          style_header={
                                        'backgroundColor': 'rgb(210, 210, 210)',
                                        'color': 'grey',
                                        'fontWeight': 'bold'
                                        },
                          
                          style_table = {'height': '400px', 'overflowY': 'auto'},
                          sort_action="native",
                          sort_mode="multi",
                          column_selectable="single",
                          row_selectable="multi",
                          row_deletable=True,
                          selected_rows=[],
                          page_action="none",
                          page_current= 0,
                          page_size= 10,
                         )

        return table
                       

@app.callback(
              Output('logs', 'children'),
              #Output("downloaded", "children"),
              Input('curves-table_1', "derived_virtual_data"),
              Input('curves-table_1', "derived_virtual_selected_rows"),
              prevent_initial_call=True,
              )
def display_logs(rows, derived_virtual_selected_rows):
       
    if derived_virtual_selected_rows is None:
        derived_virtual_selected_rows = []
    
    if derived_virtual_selected_rows!=[]:
        df = pd.DataFrame(rows)
        selected_rows = df[df.index.isin(derived_virtual_selected_rows)]
        cols_ = selected_rows.shape[0]
        
        
        titles = [selected_rows.iloc[i:i+1]['Type'].values[0]+', '+
                  selected_rows.iloc[i:i+1]['Well_name'].values[0]
                  for i in range(cols_)]  
        fig = tools.make_subplots(rows=1, cols=cols_, subplot_titles=titles).\
                                  update_xaxes(side='top', ticklabelposition="inside",
                                               title_standoff = 10)
       
        formations = []
        times = []
        for i in range(0, cols_):
            type_curve = selected_rows.iloc[i:i+1]['Type'].values[0]
                ## Reading data from gds with appropriation type of curve. 
                ## Second variant - read at the beginning all data to the memmory
            data_curves = read_curves_csv(client, bucket_for_visualization, 
                                              folders_name_for_visualization[0], type_curve)
            columns_curves = data_curves.columns
            wellname = selected_rows.iloc[i:i+1][new_columns_name[-1]].values[0]
            lat =  selected_rows.iloc[i:i+1][new_columns_name[1]].values[0]
            lon =  selected_rows.iloc[i:i+1][new_columns_name[2]].values[0]
                
            start_d = selected_rows.iloc[i:i+1][new_columns_name[3]].values[0]
            stop_d = selected_rows.iloc[i:i+1][new_columns_name[4]].values[0]
                
                                        
            df_curve = data_curves[(data_curves['Well_name']==wellname) & 
                                (data_curves['lat']==lat) & 
                                (data_curves['lon']==lon) &
                                (data_curves['DEPTH']>=start_d) &
                                (data_curves['DEPTH']<=stop_d)]
                
                                     
            for f in list(df_curve.Formation):
                formations.append(f)
                
            for t in list(df_curve.Time):
                times.append(t)
                
        formations = pd.unique(formations)
        colors = {f: ('rgba(' + ','.join((str(60+i*10),str(256/(i+1)), str(i*15), str(0.2)))+')') 
                  for f,i in zip(formations, range(0, len(formations)))
                 }
        
        times = pd.unique(times)
        colors_t = {t: ('rgba(' + ','.join((str(60+i*10),str(256/(i+1)), str(i*15), str(0.2)))+')') 
                  for t,i in zip(times, range(0, len(times)))
                 }
        
                              
        appeared_formation=[]
        appeared_time=[]
        link_size = [2]
        for i in range(0, cols_):
                
                type_curve = selected_rows.iloc[i:i+1]['Type'].values[0]
                ## Reading data from gds with appropriation type of curve. 
                ## Second variant - read at the beginning all data to the memmory
                data_curves = read_curves_csv(client, bucket_for_visualization, 
                                              folders_name_for_visualization[0], type_curve)
                columns_curves = data_curves.columns
                wellname = selected_rows.iloc[i:i+1][new_columns_name[-1]].values[0]
                lat =  selected_rows.iloc[i:i+1][new_columns_name[1]].values[0]
                lon =  selected_rows.iloc[i:i+1][new_columns_name[2]].values[0]
                
                start_d = selected_rows.iloc[i:i+1][new_columns_name[3]].values[0]
                stop_d = selected_rows.iloc[i:i+1][new_columns_name[4]].values[0]
                
                                        
                df_curve = data_curves[(data_curves['Well_name']==wellname) & 
                                (data_curves['lat']==lat) & 
                                (data_curves['lon']==lon) &
                                (data_curves['DEPTH']>=start_d) &
                                (data_curves['DEPTH']<=stop_d)]
                
                df_curve = df_curve[~( (df_curve.duplicated(['DEPTH'])))]
                
                y = df_curve[columns_curves[0]]
                x = df_curve[columns_curves[1]]
                          
                name = str(lat)+'_'+str(lon)+'_'+ wellname + '_'+ type_curve
                fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(width = link_size[0]), 
                                         marker_color=color_curve[type_curve],
                                         name=str(lat)+'_'+str(lon)+'_'+ wellname + '_'+ type_curve, 
                                         hovertemplate=
                                                       type_curve+": %{x:.2f}<br><br>" +
                                                       "Depth: %{y:.1f}<br><br>" +
                                                       'Well: ' + str(lat)+'_'+str(lon)+'_'+wellname+"<br>" +
                                                       "<extra></extra>"), 1, i+1)
                #print(df_curve)
                columns = df_curve.columns
                formation_curve = pd.unique(df_curve.Formation)
                time_curve = pd.unique(df_curve.Time)
                # Formation visualization
                if len(formation_curve)==1 and (' ' in formation_curve) :
                    pass
                else:
                    k = 0
                    for f in formation_curve:
                        y_min = df_curve[df_curve['Formation'] ==f].values[0][0]
                        y_max = df_curve[df_curve['Formation'] ==f].values[-1][0]
                    
                        x_min = df_curve[columns[1]].dropna().values.min()
                        x_max = df_curve[columns[1]].dropna().values.max()
                        
                        x_min_ = x_min * 0.9
                        x_min_f =  x_min_ - (x_max - x_min_) * 0.5
                        x_max_f = x_min_
                        
                        if k==0:
                                                
                            x_f = [x_min_f + (x_max_f - x_min_f)/2]
                            dy = df_curve['DEPTH'].values[-1]
                            y_f_name = [y_min - 0.01 * dy]
                            fig.add_trace(go.Scatter(name = f, 
                                                 x = x_f, 
                                                 y = y_f_name,
                                                 mode='text',line=dict(color="black"),
                                                 text=['FORMATION'],
                                                 textposition="middle center", showlegend=False, 
                                                 ), 1, i+1                            
                                          )
                            k +=1
                        
                        fig.add_trace(go.Scatter(name = f, 
                                                 x = [x_min_f, x_min_f, 
                                                      x_max_f, x_max_f, x_min_f
                                                     ],#[x_min, x_min, x_max, x_max, x_min], 
                                                 y = [y_min, y_max, y_max, y_min, y_min], 
                                                 mode='lines', line=dict(color="black"),
                                                 fill="toself", fillcolor = colors[f], showlegend=False
                                                ), 1, i+1
                                      )
                        
                        fig.add_trace(go.Scatter(name = f, 
                                                 x = [x_min_f + (x_max_f - x_min_f)/2],#[x_min + (x_max-x_min)/2], 
                                                 y=[y_min + (y_max-y_min)/2],
                                                 mode='text',line=dict(color="black"),text=[f],
                                                 textposition="middle center", showlegend=False, 
                                                 ), 1, i+1                            
                                      )


                        appeared_formation.append(f)    
                                 
                # GeoTime visualization
                type_curve = ['GR', 'SGR']
                if len(time_curve)==1 and (' ' in time_curve) :
                    pass
                elif columns[1] in type_curve:
                    b = 0
                    for t in time_curve:
                        color_t = 'FloralWhite '
                        if t == 'CRETACEOUS':
                            color_t = 'DarkGoldenRod '
                            
                        y_min = df_curve[df_curve['Time'] ==t].values[0][0]
                        y_max = df_curve[df_curve['Time'] ==t].values[-1][0]
                    
                        x_min = df_curve[columns[1]].dropna().values.min()
                        x_max = df_curve[columns[1]].dropna().values.max()
                        
                        x_min_ = x_min * 0.9
                        x_min_t =  x_min_ - (x_max - x_min_) * 0.5 * 2
                        x_max_t = x_min_ - (x_max - x_min_) * 0.5
                        
                        if b == 0:
                            x_t = [x_min_t + (x_max_t-x_min_t)/2]
                            dy = df_curve['DEPTH'].values[-1]
                            y_t_name = [y_min - 0.01 * dy]
                            fig.add_trace(go.Scatter(name = t, 
                                                 x = [x_min_t + (x_max_t-x_min_t)/2], 
                                                 y = y_t_name,
                                                 mode='text',line=dict(color="black"),
                                                 text=['PERIOD'],
                                                 textposition="middle center", showlegend=False,
                                                  ), 1, i+1                            
                                      )
                            b +=1
                        
                        fig.add_trace(go.Scatter(name = t, 
                                                 x = [x_min_t, x_min_t, x_max_t, x_max_t, x_min_t],
                                                 y = [y_min, y_max, y_max, y_min, y_min], 
                                                 mode='lines', line=dict(color="black"),
                                                 fill="toself", fillcolor = color_t,
                                                 showlegend=False
                                                ), 1, i+1
                                      )
                        
                        fig.add_trace(go.Scatter(name = t, 
                                                 x = [x_min_t + (x_max_t-x_min_t)/2], 
                                                 y=[y_min + (y_max-y_min)/2],
                                                 mode='text',line=dict(color="black"),text=[t],
                                                 textposition="middle center", showlegend=False,
                                                  ), 1, i+1                            
                                      )
                        
                        appeared_time.append(t)    
                    
                
                if selected_rows.iloc[i:i+1]['Type'].values[0] in list_mnemonics_log500:
                    fig.update_yaxes(autorange="reversed")
                    fig.update_xaxes(type="log",range=[np.log10(1), np.log10(500)],  row=1, col=i+1)
                elif(selected_rows.iloc[i:i+1]['Type'].values[0]=='NPHI') or\
                    (selected_rows.iloc[i:i+1]['Type'].values[0]=='PHI') or\
                    (selected_rows.iloc[i:i+1]['Type'].values[0]=='SONIC') or\
                    (selected_rows.iloc[i:i+1]['Type'].values[0]=='DT'):
                    fig.update_yaxes(autorange="reversed")
                    fig.update_xaxes(autorange="reversed", range=[40, -15], row=1, col=i+1)
                elif(selected_rows.iloc[i:i+1]['Type'].values[0] in list_mnemonics_log2000) or\
                     selected_rows.iloc[i:i+1]['Type'].values[0] in list_mnemonics_RES:
                    fig.update_yaxes(autorange="reversed")
                    fig.update_xaxes(type="log",range=[np.log10(1), np.log10(2000)],  row=1, col=i+1)
                elif(selected_rows.iloc[i:i+1]['Type'].values[0] in list_mnemonics):
                    fig.update_yaxes(autorange="reversed")
                
                
        fig.update_layout(autosize=True,  
                          height=2500, 
                          #title_text="Curve Log",
                          margin=dict(l=10, r=20, t=70, b=0),
                          yaxis_range=[y.min(),y.max()], 
                          hovermode="y unified")
        fig.layout.template = plotly_theme
    
        return  dcc.Graph(id='logs_', figure = fig)#, pd.unique(link)


@app.callback(
              Output("downloading", "children"),
              Input('curves-table_1', "derived_virtual_data"),
              Input('curves-table_1', "derived_virtual_selected_rows"),
              prevent_initial_call=True,
              )
def display_las(rows, derived_virtual_selected_rows):
       
    if derived_virtual_selected_rows is None:
        derived_virtual_selected_rows = []
    
    if derived_virtual_selected_rows!=[]:
        df = pd.DataFrame(rows)
        selected_rows = df[df.index.isin(derived_virtual_selected_rows)]
        cols_ = selected_rows.shape[0]
        
        link = []
        name_well = []
        for i in range(0, cols_):
                wellname = selected_rows.iloc[i:i+1][new_columns_name[-1]].values[0]
                lat =  selected_rows.iloc[i:i+1][new_columns_name[1]].values[0]
                lon =  selected_rows.iloc[i:i+1][new_columns_name[2]].values[0]
                
                start = float("%.1f" % selected_rows.iloc[i:i+1][new_columns_name[3]].values[0])
                stop = float("%.1f" % selected_rows.iloc[i:i+1][new_columns_name[4]].values[0])
                
                name = ('_').join((str(lat), str(lon), str(start), str(stop), wellname))#str(lat)+'_'+str(lon)+'_'+ wellname
                
                numb = find_number_lasfile_name(Keys_las, name)#('_').join((str(lat), str(lon), str(start), str(stop), wellname)))
                
                if numb !=-1:
                    
                    url = client.generate_presigned_url(
                                                            ClientMethod='get_object',
                                                            Params={'Bucket': bucket_for_download,
                                                                    'Key': Keys_las[numb]
                                                                   }
                                                            )
                    response = requests.get(url, allow_redirects=True)
                        #filename = Keys_las[numb].split('/')[1]
                      
                
                else:
                    name = name + ' - No Las File'
                    url = 'none'
                    
                if name not in name_well:
                    name_well.append(name)
                    link.append(url)
    
        
                        
        return  dbc.ListGroup([dbc.ListGroupItem(name, href=url,
                                                 className="list-group-item list-group-item-action list-group-item-secondary text-center") if url!='none'
                               else dbc.ListGroupItem(name, className="list-group-item list-group-item-action list-group-item-secondary text-center")
                               for name,url in zip(name_well, link)])


@app.callback(             
              Output(component_id='choosen-wells', component_property='children'), 
              Input('curves-table_1', "derived_virtual_data"),
              Input('curves-table_1', "derived_virtual_selected_rows")
                            )
def change_map(rows, derived_virtual_selected_rows):
    
        
    if derived_virtual_selected_rows is None:
        derived_virtual_selected_rows = []
        pass
        #return scatter_plot_graph
    
    if derived_virtual_selected_rows!=[]:
        
        df = pd.DataFrame(rows)
        selected_rows = df[df.index.isin(derived_virtual_selected_rows)]
         
        fig_map_ = px.scatter_mapbox(selected_rows, 
                                     hover_name=selected_rows.Well_name, 
                                     title='Selected Wells',
                                     lat="Lat", lon="Lon",  
                                     zoom=4, 
                                     mapbox_style='satellite')
        fig_map_.layout.template = plotly_theme 
        fig_map_.update_layout(margin=dict(l=0, r=0, t=35, b=0),
                               clickmode='event+select')
        fig_map_.update_traces(marker_size=9, marker_color='red')
        scatter_plot_graph = dcc.Graph(id='scatter_plot', figure=fig_map_) 
        
        return scatter_plot_graph


if __name__ == '__main__':
    app.run_server()
