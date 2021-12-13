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

#### model ##############################################################################################################################################################

bucket_for_visualization="transformed-for-visualization-data-1"
bucket_for_metadata="for-metadata"
bucket_for_download="transformed-for-download-data"
folders_name_for_visualization = ['csv/']#['curves', 'stratigraphy']
folders_name_for_download = ['las/']
list_metadata_files = ['List_of_curves.csv', 'List_of_data.csv']
# for display information
list_metadata = ['Age', 'Name', 'Type', 'lat', 'lon', 'Depth_start', 'Depth_finish', 
                 'Special_mark','Reference']

geotime_list = [ 'Eocene', 'Late_Jurassic', 'Jurassic', 
                'Late Permian_Early Triassic', 'Late Carboniferous_Early Permian',]
# log curves with different axis scale
list_mnemonics_log500 = ['GR']
list_mnemonics_log2000 =  ['PERM']
list_mnemonics_RES = ['RESD', 'RESS',]
list_mnemonics = ['SO', 'DT', 'RHOB']

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
    print('listdir->', list_dir)
    j = -1
    for i in range(1, len(list_dir)):
        s1 = list_dir[i]
        
        filename = s1.split('/')[-1]
        mnemonics = filename.split('.LAS')[0]
        print('i', i, 's1->',s1, 'mnemon-',mnemonics, 'filename-', filename)
        if mnemonics == key_word:
            j = i
    
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

wells_map = curves_data[['Age','lat', 'lon', 'Name']] 
wells_map = wells_map.set_index(['lat']).drop_duplicates()
wells_map = wells_map.rename_axis('lat').reset_index()

Keys_las = [obj['Key'] for obj in client.list_objects_v2(Bucket=bucket_for_download, 
                                                                           Prefix=folders_name_for_download[0])\
                                                                           ['Contents']]

#### view ################################################################################################################################################################

for_maping_list = ['lat', 'lon', 'Name']
plotly_theme = 'seaborn'#'plotly_dark'#'ggplot2'#'plotly'#'simple_white' #
dash_theme = dbc.themes.FLATLY#SUPERHERO #'CYBORG'

px.set_mapbox_access_token(
                          "pk.eyJ1IjoieXVyaXlrYXByaWVsb3YiLCJhIjoiY2t2YjBiNXl2NDV4YzJucXcwcXdtZHVveiJ9.JSi7Xwold-yTZieIc264Ww"
                           )
fig_map = px.scatter_mapbox(wells_map[for_maping_list], lat="lat", lon="lon",  zoom=4, mapbox_style='satellite', height= 800)
fig_map.layout.template = plotly_theme 
fig_map.update_layout(clickmode='event+select')
fig_map.update_traces(marker_size=9, marker_color='red')

fig_logs = tools.make_subplots(rows=1, cols=1).\
                                  update_xaxes(side='top', ticklabelposition="inside",
                                               title_standoff = 1)

Tab_map_view = [
                 dbc.Row(
                          [
                            dbc.Col(dcc.Graph(id='basic-interactions', figure=fig_map), width=4, md={'size': 9,  "offset": 1, 'order': 'first'}),
                            dbc.Col(
                                     [
                                         html.Br(), html.Br(),
                                         html.H5(children="Age", style = {'textAlign' : 'center'}),
                                         dbc.Card(dcc.Checklist(id='geo-time', 
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
                                                  inputStyle={"margin-right": "10px"}
                                                                                                  
                                                 )),
                                                                             
                                     ]
                                   )
                              
                          ]
                        ), 
               
                dbc.Row(
                            [
                                html.H5(children="Selected Wells",style = {'textAlign' : 'center'}),
                                
                            ]
                ),          
    
                 dbc.Row(
                         [
                             
                             dbc.Col([ html.Br(),
                                       dbc.Card(html.Div(id='curves-table'))
                                     ], width=4, md={'size': 6, "offset": 3, 'order': 'first'}),
                         ]
                       ),
                                                  
               ]


Tab_log_view = [
                 
                 dbc.Row(
                          [
                           
                           dbc.Col(dbc.Container(html.Div(id='logs'), fluid=True), width=8, md={'size': 9, "offset": 1, 'order': 1}),
                           dbc.Col( [
                                      html.H5(children="LAS for download", style = {'textAlign' : 'center'}), 
                                      html.Div(id='downloading')
                                    ], width=4, md={'size': 2, "offset": 0, 'order': 2}),
                          ]
                         ),
    
                ]



app = dash.Dash(__name__, external_stylesheets=[dash_theme])
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
    [Input(component_id='geo-time', component_property='value')]
)
def update_display_wells(options_chosen):
    fig_map = px.scatter_mapbox(wells_map[wells_map['Age'].isin(options_chosen)], lat="lat", lon="lon",  zoom=4, mapbox_style='satellite', height= 800)
    fig_map.layout.template = plotly_theme 
    fig_map.update_layout(clickmode='event+select')
    fig_map.update_traces(marker_size=9, marker_color='red')
    
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
            
        
        well_curves = curves_data[['Age', 'lat', 'lon', 'Type', 'Name']]
        df_ = well_curves[(well_curves['lon'].isin(x)) & (well_curves['lat'].isin(y))]
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
                                      'width': '75px', 'minWidth': '75px', 'maxWidth': '75px',
                                      'overflow': 'hidden',
                                      'textOverflow': 'ellipsis'
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
        fig = tools.make_subplots(rows=1, cols=cols_).\
                                  update_xaxes(side='top', ticklabelposition="inside",
                                               title_standoff = 10)
        link = []
        for i in range(0, cols_):
                type_curve = selected_rows.iloc[i:i+1]['Type'].values[0]
                data_curves = read_curves_csv(client, bucket_for_visualization, 
                                              folders_name_for_visualization[0], type_curve)
                columns_curves = data_curves.columns
                wellname = selected_rows.iloc[i:i+1]['Name'].values[0]
                lat =  selected_rows.iloc[i:i+1]['lat'].values[0]
                lon =  selected_rows.iloc[i:i+1]['lon'].values[0]                
                                        
                y = data_curves[(data_curves['Well_name']==wellname) & 
                                (data_curves['lat']==lat) & 
                               (data_curves['lon']==lon)][columns_curves[0]]
                x = data_curves[(data_curves['Well_name']==wellname) & 
                                (data_curves['lat']==lat) & 
                               (data_curves['lon']==lon)][columns_curves[1]]
            
                name = str(lat)+'_'+str(lon)+'_'+wellname + '_'+ type_curve
                fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=name), 1, i+1)
            
                if selected_rows.iloc[i:i+1]['Type'].values[0] in list_mnemonics_log500:
                    fig.update_yaxes(autorange="reversed")
                    fig.update_xaxes(type="log",range=[np.log10(1), np.log10(500)],  row=1, col=i+1)
                elif(selected_rows.iloc[i:i+1]['Type'].values[0]=='NPHI') or\
                    (selected_rows.iloc[i:i+1]['Type'].values[0]=='PHI'):
                    fig.update_yaxes(autorange="reversed")
                    fig.update_xaxes(autorange="reversed", range=[40, -15], row=1, col=i+1)
                elif(selected_rows.iloc[i:i+1]['Type'].values[0] in list_mnemonics_log2000) or\
                     selected_rows.iloc[i:i+1]['Type'].values[0] in list_mnemonics_RES:
                    fig.update_yaxes(autorange="reversed")
                    fig.update_xaxes(type="log",range=[np.log10(1), np.log10(2000)],  row=1, col=i+1)
                elif(selected_rows.iloc[i:i+1]['Type'].values[0] in list_mnemonics):
                    fig.update_yaxes(autorange="reversed")
                
                numb = find_number_lasfile_name(Keys_las, ('_').join((str(lat), str(lon))))
                try:
                        url = client.generate_presigned_url(
                                                            ClientMethod='get_object',
                                                            Params={'Bucket': bucket_for_download,
                                                                    'Key': Keys_las[numb]
                                                                   }
                                                            )
                        response = requests.get(url, allow_redirects=True)
                        filename = Keys_las[numb].split('/')[1]
                        
                except botocore.exceptions.ClientError as e:
                    if e.response['Error']['Code'] == "404":
                        print("The object does not exist.")
                    else:
                        raise  
                link.append(html.A(name,id='lsa-link',href=url))            
    
        fig.update_layout(autosize=False,  height=2500, yaxis_range=[y.min(),y.max()])
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
                wellname = selected_rows.iloc[i:i+1]['Name'].values[0]
                lat =  selected_rows.iloc[i:i+1]['lat'].values[0]
                lon =  selected_rows.iloc[i:i+1]['lon'].values[0]                
                            
                name = str(lat)+'_'+str(lon)+'_'+wellname
                
                numb = find_number_lasfile_name(Keys_las, ('_').join((str(lat), str(lon))))
                print('huy->', ('_').join((str(lat), str(lon))))
                #print(numb)
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
                    name = name + '- No Las File'
                    url = 'none'
                if name not in name_well:
                    name_well.append(name)
                    link.append(url)
    
        return  dbc.ListGroup([dbc.ListGroupItem(name, href=url) if url!='none'
                               else dbc.ListGroupItem(name)
                               for name,url in zip(name_well, link)])

               
if __name__ == '__main__':
    app.run_server()
