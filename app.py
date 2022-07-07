# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 11:57:56 2022

@author: tuomas.poukkula
"""

import pandas as pd
import numpy as np
import requests
import plotly.graph_objs as go
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA
import dash_daq
from flask import Flask
import os
import io
from dash_extensions.enrich import callback_context,Dash  ,ALL, Output,dcc,html, Input, State
from dash.exceptions import PreventUpdate
from dash_bootstrap_templates import ThemeChangerAIO
import random
import dash_bootstrap_components as dbc
from datetime import datetime
import locale

# riippu ollaanko Windows vai Linux -ympäristössä, mitä locale-koodausta käytetään.

try:
    locale.setlocale(locale.LC_ALL, 'fi_FI')
except:
    locale.setlocale(locale.LC_ALL, 'fi-FI')

in_dev = False

MODELS = {
    
        'Satunnaismetsä': {'model':RandomForestRegressor,
                           'doc': 'https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html',

                           'constant_hyperparameters': {
                                                        'n_jobs':-1,
                                                        'random_state':42}
                           },
        'Adaboost': {'model':AdaBoostRegressor,
                     'doc':'https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html',
                     
                     'constant_hyperparameters':{'random_state':42,
                                                 }
                     },
        'K lähimmät naapurit':{'model':KNeighborsRegressor,
                               'doc':'https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html',

                               'constant_hyperparameters': {
                                                            }
                               },
        'Tukivektorikone':{'model':SVR,
                           'doc':'https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html',

                               'constant_hyperparameters': {
                                                            }
                               },
        'Gradient Boost':{'model':GradientBoostingRegressor,
                          'doc':'https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#sklearn.ensemble.GradientBoostingRegressor',
                          'constant_hyperparameters': {'random_state':42
                                                       }
                          }
        
    
    }
UNWANTED_PARAMS = ['verbose',
                   #'cache_size', 
                   'max_iter',
                   'warm_start',
                    'max_features',
                   'tol',
                   'subsample'
                
                   
                   ]
LESS_THAN_ONE = [
        
                   'alpha',
                   'validation_fraction',
                   
    
    ]

LESS_THAN_HALF = [
        
        
                   'min_weight_fraction_leaf',
                 
    
    ]

config_plots = {'locale':'fi',
#                 'editable':True,
                "modeBarButtonsToRemove":["sendDataToCloud"],
                'modeBarButtonsToAdd' : [
                               'drawline',
                               'drawopenpath',
                               'drawclosedpath',
                               'drawcircle',
                               'drawrect',
                               'eraseshape'
                               ],
               "displaylogo":False}

spinners = ['graph', 'cube', 'circle', 'dot' ,'default']

p_font_size = 28
graph_height = 800

p_style = {'font-family':'Messina Modern Book',
            'font-size':p_font_size,
           'text-align':'center'}

p_bold_style = {'font-family':'Cadiz Semibold',
            'font-size':p_font_size-5,
           'text-align':'left'}

h4_style = {'font-family':'Messina Modern Semibold',
            'font-size':'18px',
           'text-align':'center',
           'margin-bottom': '20px'}
h3_style = {'font-family':'Messina Modern Semibold',
            'font-size':'34px',
           'text-align':'center',
           'margin-bottom': '30px'}
h2_style = {'font-family':'Messina Modern Semibold',
            'font-size':'52px',
           'text-align':'center',
           'margin-bottom': '30px'}
h1_style = {'font-family':'Messina Modern Semibold',
            'font-size':'80px',
           'text-align':'center',
           'margin-bottom': '50px'}



dbc_css = ("https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates@V1.0.2/dbc.min.css")

external_stylesheets = [
                        # "https://stackpath.bootstrapcdn.com/bootswatch/4.5.2/superhero/bootstrap.min.css",
                        
                         dbc.themes.SUPERHERO,
                         dbc_css,
                         "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css",
                          'https://codepen.io/chriddyp/pen/brPBPO.css',
                          
                       ]


server = Flask(__name__)
server.secret_key = os.environ.get('secret_key','secret')
app = Dash(name = __name__, 
           prevent_initial_callbacks = False, 
           # transforms=[ServersideOutputTransform(),
           #             TriggerTransform()],
           server = server,
           external_scripts = ["https://raw.githubusercontent.com/plotly/plotly.js/master/dist/plotly-locale-fi.js",
                               "https://cdn.plot.ly/plotly-locale-fi-latest.js"],
            # meta_tags=[{'name': 'viewport',
            #                 'content': 'width=device-width, initial-scale=1, maximum-scale=1, minimum-scale=1,'}],
           external_stylesheets = external_stylesheets
          )
# app.scripts.config.serve_locally = True
#app.scripts.append_script({"external_url": "https://cdn.plot.ly/plotly-locale-fi-latest.js"})
app.title = 'Phillipsin vinouma'

def get_unemployment():
    

  url = 'https://statfin.stat.fi:443/PxWeb/api/v1/fi/StatFin/tyti/statfin_tyti_pxt_135z.px'
  headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
                                    'Content-Type':'application/json'
                                   }
  payload = {
  "query": [
    {
      "code": "Tiedot",
      "selection": {
        "filter": "item",
        "values": [
          "Tyottomyysaste"
        ]
      }
    }
  ],
  "response": {
    "format": "json-stat2"
  }
}
  data_json = requests.post(url,json=payload,headers=headers)
  data = data_json.json()

  df = pd.DataFrame(data['dimension']['Kuukausi']['category']['index'].keys(), columns = ['Aika'])
  df.Aika = pd.to_datetime(df.Aika.str.replace('M','-'))

  tiedot_df = pd.DataFrame(data['dimension']['Tiedot']['category']['label'].values(), columns = ['Tiedot'])

  df['index'] = tiedot_df['index'] = 0

  df = pd.merge(left = df, right = tiedot_df, on = 'index', how = 'outer').drop('index',axis=1).set_index('Aika')


  df['Työttömyysaste'] = data['value']

  return df

def get_inflation():

  url = 'https://statfin.stat.fi:443/PxWeb/api/v1/fi/StatFin/khi/statfin_khi_pxt_11xd.px'
  headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
                                    'Content-Type':'application/json'
                                   }
  payload = {
  "query": [
    {
      "code": "Tiedot",
      "selection": {
        "filter": "item",
        "values": [
          "pisteluku"
        ]
      }
    }
  ],
  "response": {
    "format": "json-stat2"
    }
  }

  data_json = requests.post(url,json=payload,headers=headers)
  
  data = data_json.json()
  

  hyödyke_df = pd.DataFrame(data['dimension']['Hyödyke']['category']['label'].values(), columns = ['Hyödyke'])
  tiedot_df = pd.DataFrame(data['dimension']['Tiedot']['category']['label'].values(), columns = ['Tiedot'])
  kuukausi_df = pd.DataFrame(data['dimension']['Kuukausi']['category']['index'].keys(), columns = ['Aika'])
  
  kuukausi_df.Aika = pd.to_datetime(kuukausi_df.Aika.str.replace('M','-'))

  hyödyke_df['index'] = kuukausi_df['index'] = tiedot_df['index'] = 0
  df = pd.merge(left = pd.merge(left = kuukausi_df, right = hyödyke_df, on = 'index', how = 'outer'), right = tiedot_df, on = 'index', how ='outer').drop('index',axis=1).set_index('Aika')

  df['Pisteluku'] = data['value']

  return df

def get_inflation_percentage():

  url = 'https://pxweb2.stat.fi:443/PxWeb/api/v1/fi/StatFin/khi/statfin_khi_pxt_122p.px'
  headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
                                    'Content-Type':'application/json'
                                   }
  payload = {
  "query": [],
  "response": {
    "format": "json-stat2"
  }
  }
  data_json = requests.post(url,json=payload,headers=headers)
  
  data = data_json.json()
  
  df = pd.DataFrame(data['dimension']['Kuukausi']['category']['index'].keys(), columns = ['Aika'])
  
  df.Aika = pd.to_datetime(df.Aika.str.replace('M','-'))


  df['Inflaatio'] = data['value']

  return df.set_index('Aika') 
  
def get_data():

  unemployment_df = get_unemployment()
  inflation_df = get_inflation()


  inflation_df = pd.pivot_table(inflation_df.reset_index(), columns = 'Hyödyke', index = 'Aika' )
  inflation_df.columns = [c[-1] for c in inflation_df.columns]

  data = pd.merge(left = unemployment_df.drop('Tiedot',axis=1).reset_index(), right = inflation_df.reset_index(), how = 'inner', on = 'Aika').set_index('Aika')
  data = data.dropna(axis=1)
  data =data.loc[:,~data.apply(lambda x: x.duplicated(),axis=1).all()].copy()
  data['prev'] = data['Työttömyysaste'].shift(1)
  data.dropna(axis=0, inplace=True)
  data['month'] = data.index.month
  data['change'] = data.Työttömyysaste - data.prev
  
  inflation_percentage_df = get_inflation_percentage()

  data = pd.merge(left = data.reset_index(), right = inflation_percentage_df.reset_index(), how = 'left', on = 'Aika').set_index('Aika')

  return data


def draw_phillips_curve():
    
  max_date = data.index.values[-1]
  max_date_str = data.index.strftime('%B %Y').values[-1]

  a, b = np.polyfit(np.log(data.Työttömyysaste), data.Inflaatio, 1)

  y = a * np.log(data.Työttömyysaste) +b 

  df = data.copy()
  df['log_inflation'] = y
  df = df.sort_values(by = 'log_inflation')
  
  

  hovertemplate = ['<b>{}</b><br>Työttömyysaste: {} %<br>Inflaatio: {} %'.format(df.index[i].strftime('%B %Y'), df.Työttömyysaste.values[i], df.Inflaatio.values[i]) for i in range(len(df))]

  return go.Figure(data=[
                  go.Scatter(y=df['Inflaatio'], 
                            x = df.Työttömyysaste, 
                            name = 'Inflaatio', 
                            mode = 'markers+text', 
                            text = [None if i != max_date else '<b>'+max_date_str+'</b>' for i in df.index],
                            textposition='top left',
                            textfont=dict(family='Cadiz Book',
                                          size = 18
                                          ),
                            showlegend=False,
                            hovertemplate=hovertemplate,
                            marker = dict(color = 'red', symbol='diamond', 
                                           size = 10
                                          )
                            ),
                go.Scatter(x = df.Työttömyysaste, 
                            y = df['log_inflation'], 
                            name = 'Logaritminen trendiviiva', 
                            mode = 'lines',
                            line = dict(width=5),
                            showlegend=True,
                            hovertemplate=[], 
                            marker = dict(color = 'blue'))
                  ],
            layout = go.Layout(
                               xaxis=dict(showspikes=True,
                                          title = dict(text='Työttömyysaste (%)', font=dict(size=22, family = 'Cadiz Semibold')), 
                                          tickformat = ' ',
                                          automargin=True,
                                          
                                          tickfont = dict(
                                                           size=18, 
                                                          family = 'Cadiz Semibold')
                                          ), 
                               yaxis=dict(showspikes=True,
                                          title = dict(text='Inflaatio (%)', font=dict(size=22, family = 'Cadiz Semibold')),
                                          tickformat = ' ', 
                                          automargin=True,
                                          tickfont = dict(
                                               size=18,
                                              family = 'Cadiz Semibold')
                                          ),
                                margin=dict(
                                    l=10,
                                    r=10,
                                    # b=100,
                                     # t=120,
                                     # pad=4
                                ),
                               height= graph_height,
                               template='seaborn',  
                               # autosize=True,
                                hoverlabel = dict(font=dict(size=20,family='Cadiz Book')),
                                legend = dict(font=dict(
                                                        size=18,
                                                        family = 'Cadiz Book'
                                                        ),
                                               orientation='h',
                                               # xanchor='center',
                                               # yanchor='top',
                                               # x=.85,
                                               # y=.99
                                              ),
                              
                               title = dict(text = 'Työttömyysaste vs.<br>Inflaatio<br>{} - {}<br>'.format(df.index.min().strftime('%B %Y'),df.index.max().strftime('%B %Y')),
                                            x=.5,
                                            font=dict(
                                                size=22,
                                                family = 'Cadiz Semibold'))
                              )
            )


def get_param_options(model_name):

  model = MODELS[model_name]['model']

  dict_list = {}

  text = str(model().__doc__).split('Parameters\n    ----------\n')[1].split('\n\n    Attributes\n')[0].replace('\n        ', '\n').splitlines()
  for t in text:
    try:
      if t[0]==' ' and ':' in t:
        
        param = t.split(':')[0].strip()
        if ' ' not in param:
          if '}' in t.split(':')[1]:

            param_list = [c.replace('{','').replace('"','').replace(',','').replace("'",'').replace('}','') for c in t.split(':')[1].split('} ')[0].strip().split() if '=' not in c]
            
            dict_list[param] = param_list
          else:
            param_type = t.split(':')[1].split(',')[0].strip().split()[0]
            dict_list[param] = param_type
        
    except:
      ''
  return dict_list

def test_results(test_df):
    
    past_df = data.loc[test_df.index,:].copy()
    
    past_df['Ennuste'] = np.round(test_df.Työttömyysaste,1)
    
    mape = mean_absolute_percentage_error(past_df.Työttömyysaste,past_df.Ennuste)
    past_df['n_feat'] = test_df.n_feat.values[0]
    
    past_df['mape'] = mape
    
    return past_df

def plot_test_results(df, chart_type = 'lines+bars'):
    
    # mape = round(100 * mean_absolute_percentage_error(df.Työttömyysaste, df.Ennuste),2)
    # mape = round(100*df.mape.values[0],1)
    hovertemplate = ['<b>{}</b>:<br>Toteutunut: {}<br>Ennuste: {}'.format(df.index[i].strftime('%B %Y'),df.Työttömyysaste.values[i], df.Ennuste.values[i]) for i in range(len(df))]
    
    if chart_type == 'lines+bars':
    
        return go.Figure(data=[go.Scatter(x=df.index.strftime('%B %Y'), 
                               y = df.Työttömyysaste, 
                               name = 'Toteutunut',
                               showlegend=True, 
                               mode = 'lines+markers+text',
                               text=[str(round(c,2))+' %' for c in df.Työttömyysaste], 
                               textposition='top center',
                               hovertemplate = hovertemplate,
                               textfont = dict(family='Cadiz Semibold', size = 18,color='green'), 
                               marker = dict(color='#008000',size=12),
                               line = dict(width=5)),
                    
                    go.Bar(x=df.index.strftime('%B %Y'), 
                           y = df.Ennuste, 
                           name = 'Ennuste',
                           showlegend=True, 
                           marker = dict(color='red'), 
                           text=[str(round(c,2))+' %' for c in df.Ennuste], 
                           textposition='inside',
                           hovertemplate = hovertemplate,
                           textfont = dict(family='Cadiz Semibold', size = 18)
                           )
                    ],layout=go.Layout(xaxis = dict(title = dict(text='Aika',font=dict(size=20, family = 'Cadiz Semibold')),
                                                    tickfont = dict(family = 'Cadiz Semibold', size = 18),
                                                    automargin=True
                                                    ),
                                       yaxis = dict(title = dict(text='Työttömyysaste (%)',font=dict(family='Cadiz Semibold',size=20)),
                                                    tickfont = dict(family = 'Cadiz Semibold', size = 18),
                                                    automargin=True
                                                    ),
                                       height = graph_height-200,
                                       margin=dict(
                                            l=10,
                                           r=10,
                                           # b=100,
                                            # t=120,
                                            # pad=4
                                       ),
                                       legend = dict(font=dict(size=16,family='Cadiz Book'),
                                                      orientation='h',
                                                      xanchor='center',
                                                      yanchor='top',
                                                       x=.47,
                                                       y=1.04
                                                     ),
                                       hoverlabel = dict(font_size = 20, font_family = 'Cadiz Book'),
                                       template = 'seaborn',
                                      # margin=dict(autoexpand=True),
                                       title = dict(text = 'Työttömyysasteen ennuste<br>kuukausittain',
                                                    x=.5,
                                                    font=dict(family='Cadiz Semibold',size=22)
                                                    )
                                       )
                                                    )
    elif chart_type == 'lines':
    
        return go.Figure(data=[go.Scatter(x=df.index, 
                                y = df.Työttömyysaste, 
                                name = 'Toteutunut',
                                showlegend=True, 
                                mode = 'lines+markers+text',
                                text=[str(round(c,2))+' %' for c in df.Työttömyysaste], 
                                textposition='top center',
                                hovertemplate = hovertemplate,
                                textfont = dict(family='Cadiz Semibold', size = 18,color='green'), 
                                marker = dict(color='#008000',size=10),
                                line = dict(width=2)),
                    
                    go.Scatter(x=df.index, 
                            y = df.Ennuste, 
                            name = 'Ennuste',
                            showlegend=True,
                            mode = 'lines+markers+text',
                            marker = dict(color='red',size=10), 
                            text=[str(round(c,2))+' %' for c in df.Ennuste], 
                            # textposition='inside',
                            hovertemplate = hovertemplate,
                            line = dict(width=2),
                            textfont = dict(family='Cadiz Semibold', size = 18,color='red'))
                    ],layout=go.Layout(xaxis = dict(title = dict(text='Aika',font=dict(size=20, family = 'Cadiz Semibold')),
                                                    tickfont = dict(family = 'Cadiz Semibold', size = 18),
                                                    automargin=True,
                                                    ),
                                        yaxis = dict(title = dict(text='Työttömyysaste (%)',font=dict(family='Cadiz Semibold',size=20)),
                                                    tickfont = dict(family = 'Cadiz Semibold', size = 18),
                                                    automargin=True
                                                    ),
                                        height = graph_height-200,
                                        legend = dict(font=dict(size=16,family='Cadiz Book'),
                                                       orientation='h',
                                                       xanchor='center',
                                                       yanchor='top',
                                                        x=.47,
                                                        y=1.04
                                                      ),
                                        margin=dict(
                                             l=10,
                                            r=10,
                                            # b=100,
                                             # t=120,
                                             # pad=4
                                        ),
                                        hoverlabel = dict(font_size = 20, font_family = 'Cadiz Book'),
                                        template = 'seaborn',
                                        title = dict(text = 'Työttömyysasteen ennuste<br>kuukausittain',
                                                    x=.5,
                                                    font=dict(family='Cadiz Semibold',size=22)
                                                    )
                                        )
                                                    )

    else:
        return go.Figure(data=[go.Bar(x=df.index.strftime('%B %Y'), 
                                    y = df.Työttömyysaste, 
                                    name = 'Toteutunut',
                           showlegend=True, 
                           marker = dict(color='green'), 
                           text=[str(round(c,2))+' %' for c in df.Työttömyysaste], 
                           textposition='inside',
                           hovertemplate = hovertemplate,
                           textfont = dict(family='Cadiz Semibold', size = 18)
                                    ),
                        
                        go.Bar(x=df.index.strftime('%B %Y'), 
                                y = df.Ennuste, 
                                name = 'Ennuste',
                           showlegend=True, 
                           marker = dict(color='red'), 
                           text=[str(round(c,2))+' %' for c in df.Ennuste], 
                           textposition='inside',
                           hovertemplate = hovertemplate,
                           textfont = dict(family='Cadiz Semibold', size = 18)
                                )
                        ],layout=go.Layout(xaxis = dict(title = dict(text='Aika',font=dict(size=20, family = 'Cadiz Semibold')),
                                                        tickfont = dict(family = 'Cadiz Semibold', size = 18),
                                                        automargin=True
                                                        ),
                                            yaxis = dict(title = dict(text='Työttömyysaste (%)',font=dict(family='Cadiz Semibold',size=20)),
                                                        tickfont = dict(family = 'Cadiz Semibold', size = 18),
                                                        automargin=True
                                                        ),
                                            height = graph_height-200,
                                            margin=dict(
                                                 l=10,
                                                r=10,
                                                # b=100,
                                                 # t=120,
                                                 # pad=4
                                            ),
                                            legend = dict(font=dict(size=16,family='Cadiz Book'),
                                                           orientation='h',
                                                           xanchor='center',
                                                           yanchor='top',
                                                            x=.47,
                                                            y=1.04
                                                          ),
                                            hoverlabel = dict(font_size = 20, font_family = 'Cadiz Book'),
                                            template = 'seaborn',
                                            title = dict(text = 'Työttömyysasteen ennuste<br>kuukausittain',
                                                        x=.5,
                                                        font=dict(family='Cadiz Semibold',size=22)
                                                        )
                                            )
                                                        )                                                   

                                                    
                                                    
                                                    
def plot_forecast_data(df, chart_type):
    
    
    hover_true = ['<b>{}</b><br>Työttömyysaste: {} %'.format(data.index[i].strftime('%B %Y'), data.Työttömyysaste.values[i]) for i in range(len(data))]
    hover_pred = ['<b>{}</b><br>Työttömyysaste: {} %'.format(df.index[i].strftime('%B %Y'), round(df.Työttömyysaste.values[i],1)) for i in range(len(df))]
    

    if chart_type == 'lines':
    
    
        return go.Figure(data=[go.Scatter(x=data.index, 
                                          y = data.Työttömyysaste, 
                                          name = 'Toteutunut',
                                          showlegend=True,
                                          mode="lines", 
                                          hovertemplate =  hover_true,##'<b>%{x}</b>: %{y}%',
                                          marker = dict(color='green')),
                    go.Scatter(x=df.index, 
                               y = np.round(df.Työttömyysaste,1), 
                               name = 'Ennuste',
                               showlegend=True,
                               mode="lines", 
                               hovertemplate = hover_pred,#'<b>%{x}</b>: %{y}%',
                               marker = dict(color='red'))
                    ],layout=go.Layout(xaxis = dict(title = dict(text = 'Aika',font=dict(
                         size=20, 
                        family = 'Cadiz Semibold')),
                        automargin=True,
                
                                                    tickfont = dict(family = 'Cadiz Semibold', 
                                                                     size = 18
                                                                    ),
                                                    rangeslider=dict(visible=True),
                                                    rangeselector=dict(
                buttons=list([
                    dict(count=3,
                         label="3kk",
                         step="month",
                         stepmode="backward"),
                    dict(count=6,
                         label="6kk",
                         step="month",
                         stepmode="backward"),
                    dict(count=1,
                         label="YTD",
                         step="year",
                         stepmode="todate"),
                    dict(count=1,
                         label="1v",
                         step="year",
                         stepmode="backward"),
                    dict(count=3,
                         label="3v",
                         step="year",
                         stepmode="backward"),
                    dict(count=5,
                         label="5v",
                         step="year",
                         stepmode="backward"),
                    dict(step="all",
                         label='MAX')
                ])
            ),
                                                    
                                                    ),
                                       height=graph_height,
                                       margin=dict(
                                            l=10,
                                           r=10,
                                           # b=100,
                                            # t=120,
                                            # pad=4
                                       ),
                                      hoverlabel = dict(
                                            font_size = 20, 
                                                         font_family = 'Cadiz Book'),
                                       legend = dict(orientation='h',
                                                     x=.5,
                                                     y=.01,
                                                     xanchor='center',
                                                      yanchor='bottom',
                                                     font=dict(
                                                    size=14,
                                                   family = 'Cadiz Semibold')),
                                       template = 'seaborn',
                                       yaxis = dict(title=dict(text = 'Työttömyysaste (%)',
                                                     font=dict(
                                                          size=20, 
                                                         family = 'Cadiz Semibold')),
                                                    automargin=True,
                                                     tickfont = dict(family = 'Cadiz Book', 
                                                                      size = 18
                                                                     )),
                                       title = dict(text = 'Työttömyysaste ja ennuste kuukausittain<br>{} - {}'.format(data.index.strftime('%B %Y').values[0],df.index.strftime('%B %Y').values[-1]),
                                                    x=.5,
                                                    font=dict(family='Cadiz Semibold',
                                                               size=22
                                                              )),
    
                                       ))


    else:
        
        
      
        return go.Figure(data=[go.Bar(x=data.index, 
                                          y = data.Työttömyysaste, 
                                          name = 'Toteutunut',
                                          showlegend=True,
                                          # mode="lines", 
                                          hovertemplate = hover_true,#'<b>%{x}</b>: %{y}%',
                                          marker = dict(color='green')),
                    go.Bar(x=df.index, 
                               y = np.round(df.Työttömyysaste,1), 
                               name = 'Ennuste',
                               showlegend=True,
                               # mode="lines", 
                               hovertemplate = hover_pred,#'<b>%{x}</b>: %{y}%',
                               marker = dict(color='red'))
                    ],layout=go.Layout(xaxis = dict(title = dict(text = 'Aika',
                                                                 font=dict(
                                                                     size=20, 
                                                                     family = 'Cadiz Semibold')),
                                                                 automargin=True,
                                                    tickfont = dict(family = 'Cadiz Semibold', 
                                                                    size = 18
                                                                    ),
                                                    rangeslider=dict(visible=True),
                                                    rangeselector=dict(
                buttons=list([
                    dict(count=3,
                          label="3kk",
                          step="month",
                          stepmode="backward"),
                    dict(count=6,
                          label="6kk",
                          step="month",
                          stepmode="backward"),
                    dict(count=1,
                          label="YTD",
                          step="year",
                          stepmode="todate"),
                    dict(count=1,
                          label="1v",
                          step="year",
                          stepmode="backward"),
                    dict(count=3,
                          label="3v",
                          step="year",
                          stepmode="backward"),
                    dict(count=5,
                          label="5v",
                          step="year",
                          stepmode="backward"),
                    dict(step="all",
                          label='MAX')
                ])
            ),
                                                    
                                                    ),
                                       height=graph_height,
                                       margin=dict(
                                            l=10,
                                           r=10,
                                           # b=100,
                                            # t=120,
                                            # pad=4
                                       ),
                                       template='seaborn',
                                       hoverlabel = dict(
                                           font_size = 20, 
                                           font_family = 'Cadiz Book'),
                                       legend = dict(orientation='h',
                                                     x=.5,
                                                     y=.01,
                                                     xanchor='center',
                                                      yanchor='bottom',
                                                     font=dict(
                                                    size=14,
                                                   family = 'Cadiz Semibold')),
                                       yaxis = dict(title=dict(text = 'Työttömyysaste (%)',
                                                     font=dict(
                                                          size=20, 
                                                         family = 'Cadiz Semibold')),
                                                    automargin=True,
                                                     tickfont = dict(family = 'Cadiz Book', 
                                                                      size = 18
                                                                     )
                                                     ),
                                       title = dict(text = 'Työttömyysaste ja ennuste kuukausittain<br>{} - {}'.format(data.index.strftime('%B %Y').values[0],df.index.strftime('%B %Y').values[-1]),
                                                    x=.5,
                                                    font=dict(family='Cadiz Semibold',
                                                               size=22
                                                              )),
    
                                       )) 
                                                    
                                                    


                                                       

                                                

def predict(df, model, features, feature_changes = None, length=6, use_pca = False, n_components=.99):
  
  feat = features.copy()
  feat.append('prev')
  feat.append('month')

  if feature_changes is None:
    feature_changes = pd.Series({f:0 for f in features})

  scl = StandardScaler()
  label = 'change'
  x = df[feat]
  y = df[label]

  n_feat  = len(features) 

  X = scl.fit_transform(x)

  pca = PCA(n_components = n_components, random_state = 42, svd_solver = 'full')

  if use_pca:
    
    X = pca.fit_transform(X)
    n_feat = len(pd.DataFrame(X).columns)
    
    
  model.fit(X,y)

  last_row = df.iloc[-1:,:].copy()

  last_row.index = last_row.index + pd.DateOffset(months=1)
  last_row.month = last_row.index.month

  last_row.prev = last_row.Työttömyysaste
  last_row.Työttömyysaste = np.nan
  last_row[features] = last_row[features] * (1 + feature_changes/100)

  scaled_features = scl.transform(last_row[feat])

  if use_pca:
    scaled_features = pca.transform(scaled_features)

  last_row.change = model.predict(scaled_features)

  last_row.Työttömyysaste = np.maximum(0, last_row.prev + last_row.change)

  results = []

  results.append(last_row)

  for _ in range(length-1):

    dff = results[-1].copy()
    dff.index = dff.index + pd.DateOffset(months=1)
    dff.month = dff.index.month
    dff.prev = dff.Työttömyysaste
    dff.Työttömyysaste = np.nan
    
    dff[features] = dff[features] * (1 + feature_changes/100)
    

    scaled_features = scl.transform(dff[feat])

    if use_pca:
      scaled_features = pca.transform(scaled_features)

    dff.change = model.predict(scaled_features)

    dff.Työttömyysaste = np.maximum(0, dff.prev + dff.change)
    results.append(dff)

  result = pd.concat(results)
  result['n_feat'] = n_feat
  
 
  return result


def apply_average(features, length = 4):

  return 100 * data[features].pct_change().iloc[-length:, :].mean()


data = get_data()
correlations_desc = data[data.columns[:-4]].corr()['Työttömyysaste'].iloc[1:].sort_values(ascending=False)
correlations_asc = data[data.columns[:-4]].corr()['Työttömyysaste'].iloc[1:].sort_values(ascending=True)
correlations_abs_desc = data[data.columns[:-4]].corr()['Työttömyysaste'].iloc[1:].abs().sort_values(ascending=False)
correlations_abs_asc = data[data.columns[:-4]].corr()['Työttömyysaste'].iloc[1:].abs().sort_values(ascending=True)
main_classes = sorted([c for c in data.columns[:-4] if len(c.split()[0])==2])
second_classes = sorted([c for c in data.columns[:-4] if len(c.split()[0])==4])
third_classes = sorted([c for c in data.columns[:-4] if len(c.split()[0])==6])
fourth_classes = sorted([c for c in data.columns[:-4] if len(c.split()[0])==8])
fifth_classes = sorted([c for c in data.columns[:-4] if len(c.split()[0])==10])

feature_options = [{'label':c, 'value':c} for c in data.columns[1:-4]]
corr_desc_options = [{'label':c, 'value':c} for c in correlations_desc.index]
corr_asc_options = [{'label':c, 'value':c} for c in correlations_asc.index]
corr_abs_desc_options = [{'label':c, 'value':c} for c in correlations_abs_desc.index]
corr_abs_asc_options = [{'label':c, 'value':c} for c in correlations_abs_asc.index]
main_class_options = [{'label':c, 'value':c} for c in main_classes]
second_class_options = [{'label':c, 'value':c} for c in second_classes]
third_class_options = [{'label':c, 'value':c} for c in third_classes]
fourth_class_options = [{'label':c, 'value':c} for c in fourth_classes]
fifth_class_options = [{'label':c, 'value':c} for c in fifth_classes]



initial_options = corr_abs_desc_options
initial_features = [[list(f.values())[0] for f in corr_abs_desc_options][i] for i in random.sample(range(len(corr_abs_desc_options)),6)]


def serve_layout():
    
    return dbc.Container(fluid=True, className = 'dbc', children=[
        html.Br(),        
        dbc.Row(
            [
                dbc.Col(
                    [
                    ThemeChangerAIO(aio_id="theme", 
                                        button_props={'title':'Vaihda väriteemaa',
                                                      'size':'lg',
                                                      'children' : 'Vaihda väriteemaa',
                                                      'color':'warning'},
                                        offcanvas_props={'title':"Valitse jokin alla olevista väriteemoista",
                                                         'scrollable':True},
                                        radio_props={"value":dbc.themes.SUPERHERO}),
                    html.Br(),
     
                
                    dbc.Button("Avaa pikaohje", 
                                 id="open-offcanvas", 
                                 n_clicks=0, 
                                 outline=True,
                                 size = 'lg',
                                 color = 'danger',
                                 className="me-1",
                                 style = {'font-style':'Cadiz Semibold'}),
                    dbc.Offcanvas(
                          [
                              
                          html.H3('Tässä on lyhyt ohjeistus sovelluksen käyttöön. Yksityiskohtaisempaa informaatiota löytyy Ohje ja esittely -välilehdeltä sekä jokaisen toiminnon omalta välilehdeltään.', 
                                  style = {'font-family':'Cadiz Semibold',
                                            'text-align':'left',
                                            'font-size':22,
                                            'margin-bottom':'30px'
                                            }
                                  ),
                              
                          html.H3('1. Valitse hyödykkeitä Hyödykkeiden valinta -välilehdellä', 
                                  style = {'font-family':'Cadiz Semibold',
                                            'text-align':'left',
                                            'font-size':20,
                                            'margin-bottom':'30px'
                                            }
                                  ),
                          
                          html.P(
                              "Valitse haluamasi hyödykkeet alasvetovalikosta. "
                              "Voit lajitella hyödykkeet haluamallasi tavalla. "
                              "Valitse käytetäänkö edellisten kuukausien muutoskeskiarvoja "
                              "tai vakiomuutosta kaikille valitsinta klikkaamalla. "
                              "Säädä olettu muutos liutin -valinnalla. "
                              "Hienosäädä yksittäisten hyödykkeiden muutoksia muokkaamalla laatikoiden arvoja.",
                               style = {'font-family':'Cadiz Book',
                                        'font-size':16,
                                         'text-align':'left'}
                          ),
                          html.Br(),
                          html.H3('2. Tutki valitsemiasi hyödykkeitä Tutkiva analyysi -välilehdellä', 
                                   style = {'font-family':'Cadiz Semibold',
                                            'margin-bottom':'30px',
                                            'font-size':20,
                                              'text-align':'left',
                                              
                                              }
                                  ),
                          
                          html.P(
                              "Tarkastele kuvaajien avulla valittujen hyödykkeiden suhdetta työttömyysasteeseen "
                              "tai hyödykkeiden suhteita toisiinsa. "
                              "Voit myös tarkastella indeksien, työttömyysasteen ja inflaation aikasarjoja.",
                               style = {'font-family':'Cadiz Book',
                                        'font-size':16,
                                          'text-align':'left'}
                          ),
                          html.Br(),
                          html.H3('3. Valitse menetelmä Menetelmän valinta -välilehdellä', 
                                   style = {'font-family':'Cadiz Semibold',
                                            'margin-bottom':'30px',
                                            'font-size':20,
                                              'text-align':'left'}
                                  ),
                          html.P(
                              "Valitse haluamasi koneoppimisalgoritmi alasvetovalikosta. "
                              "Säädä algoritmin hyperparametrit. "
                              "Valitse painikkeesta käytetäänkö pääkomponenttianalyysiä "
                              "ja niin tehtäessä valitse säilötyn variaation määrä liutin-valinnalla.",
                               style = {'font-family':'Cadiz Book',
                                        'font-size':16,
                                          'text-align':'left'}
                          ),
                          html.Br(),
                          html.H3('4. Testaa menetelmää Testaaminen-välilehdellä', 
                                   style = {'font-family':'Cadiz Semibold',
                                             'text-align':'left',
                                             'font-size':20,
                                             'margin-bottom':'30px'
                                             }
                                  ),
                          
                          html.P(
                              "Valitse testin pituus ja klikkaa testaa nappia. "
                              "Tarkastele testin kuvaajaa tai viedä tulokset Exceliin "
                              "klikkaamalla 'Lataa testitulokset koneelle -nappia'. "
                              "Voit palata edellisiin vaiheisiin ja kokeilla uudelleen eri hyödykkeillä ja menetelmillä.",
                              style = {'font-family':'Cadiz Book',
                                       'font-size':16,
                                        'text-align':'left'}
                          ),
                          html.Br(),
                          html.H3('5. Tee ennuste Ennustaminen-välilehdellä', 
                                   style = {'font-family':'Cadiz Semibold',
                                             'text-align':'left',
                                             'font-size':20,
                                             'margin-bottom':'30px'
                                             
                                             }
                                  ),
                          
                          html.P(
                              "Valitse ennusteen pituus ja klikkaa ennusta nappia. "
                              "Tarkastele ennusteen kuvaajaa tai viedä tulokset Exceliin "
                              "klikkaamalla 'Lataa ennustedata koneelle -nappia'. "
                              "Voit palata edellisiin vaiheisiin ja kokeilla uudelleen eri hyödykkeillä ja menetelmillä. "
                              "Voit myös säätää hyödykeindeksien oletettuja kuukausimuutoksia ja kokeilla uudestaan.",
                              style = {'font-family':'Cadiz Book',   
                                       'font-size':16,
                                         'text-align':'left'}
                          ),
                          
                          
                          
                        ],
                          id="offcanvas",
                          title="Pikaohje",
                          scrollable=True,
                          is_open=False,
                          style = {'font-style':'Cadiz Book',
                                   'font-size':'30px'}
                    )  
                
                ], xs =12, sm=12, md=12, lg=1, xl=1),
                
                dbc.Col([
                    
                      
                    html.H1('Phillipsin vinouma',
                             style=h1_style
                            ),
                  
                    html.H2('Työttömyyden ennustaminen hintatason muutoksilla',
                            style=h2_style),
                    
                    html.P('Valitse haluamasi välilehti alla olevia otsikoita klikkaamalla. ' 
                           'Vasemman yläkulman painikkeista saat näkyviin pikaohjeen '
                           'ja voit myös vaihtaa sivun väriteemaa.',
                           style = p_style)
                    ],xs =12, sm=12, md=12, lg=11, xl=11)
        ], justify='left'),
        html.Br(),
        
        html.Div(id = 'hidden_store_div',
                 children = [
                    
                    dcc.Store(id = 'features_values',data={f:0.0 for f in initial_features}),
                    dcc.Store(id = 'change_weights'), 
                    dcc.Store(id = 'method_selection_results'),
                    dcc.Store(id = 'test_data'),
                    dcc.Store(id = 'forecast_data'),
                    dcc.Download(id='forecast_download'),
                    dcc.Download(id='test_download')
        ]),
        
        dbc.Tabs(id ='tabs', children = [
            
            
            
            dbc.Tab(label='Ohje ja esittely',
                    tab_id = 'ohje',
                    tabClassName="flex-grow-1 text-center",
                    tab_style = {'font-size':'34px',
                                 'font-family':'Cadiz Semibold'},
                    style = {
                            
                              "maxHeight": "1024px",

                            "overflow": "auto"
                        },
                    
                    children = [
                        html.Div(children = [
                        dbc.Row(justify='center',
                                # style = {'margin' : '10px 10px 10px 10px'},
                                children=[
                            
                          dbc.Col(xs =12, sm=12, md=12, lg=8, xl=8, children =[
                         
                                  html.Br(),
                                  
                                  html.P('“The world is its own best model.”', 
                                        style = {
                                            'text-align':'center',
                                            'font-style': 'italic', 
                                            'font-family':'Messina Modern Book', 
                                              'font-size':p_font_size
                                            }),
                                  html.A([html.P('(Rodney Brooks)', 
                                        
                                        style={
                                            'textAlign':'center',
                                            'font-family':'Messina Modern Book', 
                                              'font-size':p_font_size-4
                                            })], href = 'https://en.wikipedia.org/wiki/Rodney_Brooks', target="_blank"),
                                  

                                  html.Br(),
                                  html.H3('Esittely',style=h3_style
                                          ),
                                  html.P('Tällä työkalulla voi vapaasti kehitellä koneoppimismenetelmän, joka pyrkii ennustamaan tulevien kuukausien työttömyysastetta ottaen huomioon aiemmat toteutuneet työttömyysasteet, kuukausittaiset työttömyyden kausivaihtelut sekä käyttäjän itse valitsemien hyödykkeiden kuluttajahintaindeksit. Näin on mahdollista tuottaa skenaariopohjaisia ennustemalleja kun tiettyjen hyödykkeiden tai yleisindeksin oletetaan muuttuvan tietyllä tavalla. Ennuste tehdään sillä oletuksella, että valittujen hyödykkeiden hintataso muuttuu käyttäjän syöttämän oletetun keskimääräisen kuukausimuutoksen mukaan. Lisäksi koneoppimismenetelmän voi valita ja algoritmien hyperparametrit voi säätää itse. Menetelmää voi testata valitulle aikavälille ennen ennusteen tekemistä. Hintaindeksien valintaa helpottaakseen, voi niiden keskinäisiä suhteita ja suhdetta työttömyysasteeseen tutkia sille varatulla välilehdellä. ', 
                                        style=p_style),
                                  html.P('Sovellus hyödyntää Tilastokeskuksen Statfin-rajapinnasta saatavaa työttömyys, -ja kuluttajahintaindeksidataa. Sovellus on avoimen lähdekoodin kehitysprojekti, joka on saatavissa kokonaisuudessaan tekijän Github-sivun kautta. Sovellus on kehitetty harrasteprojektina tutkivaan ja kokeelliseen käyttöön, ja siitä saataviin tuloksiin on suhtauduttava varauksella (ks. Vastuunvapautuslauseke alla).', 
                                        style=p_style),
                                  html.Br(),
                                  html.H3('Teoria',
                                          style=h3_style
                                          ),
                                  
                                  html.P('Inflaatio, eli yleisen hintatason nousu on vaikuttanut viimeisen vuoden aikana monen kotitalouden kulutustottumuksiin. Hintavakauden saavuttamiseksi, keskuspankkien on pyrittävä kiristämään rahapolitiikkaa ohjauskorkoa nostamalla, mikä tullee nostamaan asuntovelallisten korkokuluja. Vähemmän varallisuutta ohjautuu osakemarkkinoille, mikä ajaa pörssikursseja alaspäin. Samalla työmarkkinoilla on painetta palkankorotuksille, mikä voi pahimmillaan eskaloitua lakkoiluun. Vaikuttaisi siis siltä, että inflaatiolla on vain negatiivisia vaikutuksia, mutta näin ei aina ole. Nimittäin historian aika on havaittu makrotaloustieteellinen ilmiö, jonka mukaan lyhyellä aikavälillä inflaation ja työttömyyden välillä vallitsee ristiriita. Toisin sanoen korkean inflaation koetaan lyhyellä aikavälillä matalaa työttömyyttä. Selittäviä teorioita tälle ilmiölle on useita. Lyhyellä aikavälillä hintojen noustessa tuotanto nousee, koska tuottajat nostavat hyödykkeiden tuotantoa suurempien katteiden saavuttamiseksi. Tämä johtaa matalampaan työttömyyteen, koska tuotannon kasvattaminen johtaa uusiin rekrytointeihin, jolloin työttömien osuus työvoimasta pienenee. Toisin päin katsottuna, kun työttömyys on matala, markkinoilla on työn kysyntäpainetta, mikä nostaa palkkoja. Palkkojen nousu taas johtaa yleisen hintatason nousuun, koska hyödykkeiden tarjoajat voivat pyytää korkeampaa hintaa tuotteistaan ja palveluistaan. Elämme nyt sellaista hetkeä, jossa inflaatio viimeisen noin kymmenen vuoden huipussaan. Samalla työttömyys on paljon matalampi kuin se oli esimerkiksi vuonna 2015, jolloin inflaatio oli välillä jopa negatiivinen. Ajoittain voivat molemmat olla samaan aikaan matalia tai korkeita, mutta yleisesti ottaen lyhyellä ajalla toisen noustessa toisen trendikäyrä laskee. Tämä tunnetaan makrotaloustieteessä ns. Phillipsin käyränä, joka on sen vuonna 1958 kehittäneen taloustieteilijä Alban William Phillipsin mukaan.',
                                        style=p_style),
                                  
                                  html.H3('Phillipsin käyrä Suomen taloudessa kuukausittain', 
                                          style=h3_style),
                                  html.H4('(Lähde: Tilastokeskus)', 
                                          style=h4_style),
                                  
                                  ])
                                  ]
                                ),

                                  dbc.Row([
                                      dbc.Col([
                                             
                                              html.Div(
                                                  [dcc.Graph(figure= draw_phillips_curve(),
                                                        config = config_plots
                                                        )
                                                  ],style={'textAlign':'center'}
                                                   
                                                 
                                                      )
                                             
                                            ],xs =12, sm=12, md=12, lg=8, xl=8)
                                  ], justify = 'center', 
                                      # style = {'margin' : '10px 10px 10px 10px'}
                                    ),
                                 
                                  dbc.Row([
                                     
                                      dbc.Col([
                                          html.Br(),
                                          html.P('Yllä olevassa kuvaajassa on esitetty sirontakuviolla työttömyysaste ja inflaatio eri aikoina. Lisäksi siinä on nimetty viimeisin ajankohta, jolta on saatavissa sekä inflaatio että työttömyyslukema. Viemällä hiiren pisteiden päälle, näkee arvot sekä ajan. Logaritminen trendiviiva esittää aikoinaan Phillipsin tekemää empiiristä havaintoa, jossa inflaation ja työttömyysasteen välillä vallitsee negatiivinen korrelaatio. Kuvaajassa inflaation trendikäyrästä laskee työttömyyden kasvaessa.' ,
                                                style = {
                                                    'text-align':'center', 
                                                    'font-family':'Messina Modern Book', 
                                                      'font-size':p_font_size-2
                                                    }),
                                          html.P('Kyseessä on tunnettu taloustieteen teoria, jota on tosin vaikea soveltaa, koska ei ole olemassa sääntöjä, joilla voitaisiin helposti ennustaa työttömyyttä saatavilla olevien inflaatiota kuvaavien indikaattorien avulla. Mikäli sääntöjä on vaikea formuloida, niin niitä voi yrittää koneoppimisen avulla oppia historiadataa havainnoimalla. Voisiko siis olla olemassa tilastollisen oppimisen menetelmä, joka pystyisi oppimaan Phillipsin käyrän historiadatasta? Mikäli tämänlainen menetelmä olisi olemassa, pystyisimme ennustamaan lyhyen aikavälin työttömyyttä, kun oletamme hyödykkeiden hintatason muuttuvan skenaariomme mukaisesti.',
                                                style=p_style), 
                                          html.P('Phillipsin käyrälle on omistettu oma lukunsa kauppatieteellisen yliopistotutkinnon kansantaloustieteen pääsykoekirjassa, ja siitä seuraa yksi kymmenestä taloustieteen perusperiaatteesta.',
                                                style=p_style),
                                          html.Br(),
                                                                            
                                          html.H3('Taloustieteen kymmenes perusperiaate:',style = {'text-align':'center', 
                                                                                                   'font-family':'Messina Modern Semibold',
                                                                                                   'font-style': 'italic', 
                                                                                                   'font-weight': 'bold', 
                                                                                                   'font-size':'34px'}),
                                          
                                          html.P('Työttömyyden ja inflaation kesken vallitsee lyhyellä ajalla ristiriita. Täystyöllisyyttä ja vakaata hintatasoa on vaikea saavuttaa yhtä aikaa.', 
                                                style = {
                                                    'text-align':'center',
                                                    'font-style': 'italic', 
                                                    'font-family':'Messina Modern Book', 
                                                      'font-size':p_font_size
                                                    }),
                                          html.P('(Matti Pohjola, 2019, Taloustieteen oppikirja, s. 250, ISBN:978-952-63-5298-5)', 
                                                style={
                                                    'textAlign':'center',
                                                    'font-family':'Messina Modern Book', 
                                                      'font-size':p_font_size-4
                                                    }),
        
                                          html.Br(),
                                          html.P('Tämä sovellus perustuu inflaation ja työttömyyden välisen suhteen hyödyntämiseen. Sen esittäminen matemaattisesti olisi vaikeaa ja olisi altis poikkeustapauksille. Siksi lieneekin, kuten AI-pioneeri Rodney Brooks on ilmaissut, havinnoida ilmiötä ja ja pyrkiä oppimaan siitä. Tässä on sopiva paikka hyödyntää koneoppimista, vaikkakin soveltaminen ei olekaan yksiselitteistä. Siksi tällä työkalulla on mahdollista rakentaa kokeellisesti koneoppimisen ratkaisu, joka, perustuen Phillipsin teoriaan, pyrkii ennustamaan työttömyyttä valittujen hyödykkeiden hintatason olettamien avulla. Sovellus jakaantuu ennustajapiirteiden valintaan, tutkivaan analyysiin, menetelmän suunnitteluun, toteutuksen testaamiseen sekä lyhyen aikavälin ennusteen tekemiseen.',
                                                style=p_style), 
                                          html.P('Inflaatiota mitataan kuluttajahintaindeksin vuosimuutoksen avulla. Kuluttajahintaindeksi on Tilastokeskuksen koostama indikaattori, joka mittaa satoja hyödykkeitä sisältävän hyödykekorin hinnan muutosta johonkin perusvuoden (tässä tapauksessa vuoden 2010) hintatasoon nähden. Tässä sovelluksessa voi valita kuluttajahintaindeksin komponentteja eli hyödykeryhmien indeksien pistelukuja työttömyyden ennustamiseen. Ennuste perustuu siten käyttäjän itse koostamaan hyödykekoriin.',
                                                style=p_style), 
                                          html.P('Tutkivan analyysin avulla voi arvioida mitkä hyödykkeet sopisivat parhaiten ennusteen selittäjiksi. Tutkivassa analyysissa tarkastellaan muuttujien välisiä korrelaatiota. Tässä sovelluksessa voi tehdä korrelaatioperusteisen ennusteen. Pääsääntönä on valita ennustajiksi niitä hyödykeryhmiä, jotka korreloivat eniten työttömyyden kanssa sekä vähiten keskenään. Myös koneoppimisalgoritmin valinta ja sen uniikkien hyperparametrien säätäminen vaikutta tulokseen. Testin avulla saa informaatiota siitä, miten valittu menetelmä olisi onnistunut ennustamaan työttömyyttä aikaisempina ajankohtina.',
                                                style=p_style),
                                          html.Br(),
                                          
                                          
                                           html.H3('Sovelluksen käyttöhje',
                                                   style=h3_style
                                                   ),
                                           
                                           html.P('Seuraavaksi hieman ohjeistusta sovelluksen käyttöön. Jokainen osio olisi tehtävä vastaavassa järjestyksessä. Välilehteä valitsemalla pääset suorittamaan jokaisen vaiheen. Välilehdillä on vielä yksityiskohtaisemmat ohjeet.', 
                                                    style = p_style),
                                           html.Br(),
                                           html.P('1. Hyödykkeiden valinta. Valitse haluamasi hyödykeryhmät alasvetovalikosta. Näiden avulla ennustetaan työttömyyttä Phillipsin teorian mukaisesti.', 
                                                    style = p_style),
                                          html.P('2. Tutkiva analyysi. Voit tarkastella ja analysoida valitsemiasi hyödykkeitä. Voit tarvittaessa palata edelliseen vaiheeseen ja poistaa tai lisätä hyödykkeitä.',
                                                    style = p_style),
                                          html.P('3. Menetelmän valinta. Tässä osiossa valitsen koneoppimisalgoritmin sekä säädät hyperparametrit. Lisäksi voi valita hyödynnetäänkö pääkomponenttianalyysiä ja kuinka paljon variaatiota säilötään.',
                                                    style = p_style),
                                         html.P('4. Testaaminen. Voit valita menneen ajanjakson, jota malli pyrkii ennustamaan. Näin pystyt arvioimaan kuinka ennustemalli olisi toiminut jo toteutuneelle datalle.',
                                                    style = p_style),
                                         html.P('5. Ennusteen tekeminen. Voit nyt hyödyntää valitsemaasi menetelmää tehdäksesi ennusteen tulevaisuuteen. Valitse ennusteen pituus ja klikkaa ennusta. Ennusteen voi sitten viedä myös Exceliin. Ennustetta tehdessä hyödynnetään asettamiasi hyödykkeiden muutosarvoja.',
                                                    style = p_style),
                                          
                                          html.Br(),
                                          
                                          html.H3('Pääkomponenttianalyysistä',
                                                  style=h3_style
                                                  ),
                                          
                                          html.P('Pääkomponenttianalyysilla (englanniksi Principal Component Analysis, PCA) pyritään minimoimaan käytettyjen muuttujien määrää pakkaamalla ne sellaisiin kokonaisuuksiin, jotta hyödynnetty informaatio säilyy. Informaation säilyvyyttä mitataan selitetyllä varianssilla (eng. explained variance), joka tarkoittaa uusista pääkomponenteista luodun datan hajonnan säilyvyyttä alkuperäiseen dataan verrattuna. Tässä sovelluksessa selitetyn varianssin (tai säilytetyn variaation) osuuden voi valita itse, mikäli hyödyntää PCA:ta. Näin saatu pääkomponenttijoukko on siten pienin sellainen joukko, joka säilyttää vähintään valitun osuuden alkuperäisen datan hajonnasta. Näin PCA-algoritmi muodostaa juuri niin monta pääkomponenttia, jotta selitetyn varianssin osuus pysyy haluttuna.',
                                                  style=p_style),
                                          html.P('PCA on yleisesti hyödyllinen toimenpide silloin, kun valittuja muuttujia on paljon, milloin on myös mahdollista, että osa valituista muuttujista aiheuttaa datassa kohinaa, mikä taas johtaa heikompaan ennusteeseen.  Pienellä määrällä tarkasti harkittuja muuttujia PCA ei ole välttämätön.',
                                                  style=p_style),
                                          html.Br(),
                                       
                                          html.Br(),
                                          
                                          html.H3('Vastuuvapauslauseke',
                                                  style=h3_style),
                                          
                                          html.P("Sivun ja sen sisältö tarjotaan ilmaiseksi sekä sellaisena kuin se on saatavilla. Kyseessä on yksityishenkilön tarjoama palvelu eikä viranomaispalvelu tai kaupalliseen tarkoitukseen tehty sovellus. Sivulta saatavan informaation hyödyntäminen on päätöksiä tekevien tahojen omalla vastuulla. Palvelun tarjoaja ei ole vastuussa menetyksistä, oikeudenkäynneistä, vaateista, kanteista, vaatimuksista, tai kustannuksista taikka vahingosta, olivat ne mitä tahansa tai aiheutuivat ne sitten miten tahansa, jotka johtuvat joko suoraan tai välillisesti yhteydestä palvelun käytöstä. Huomioi, että tämä sivu on yhä kehityksen alla.",
                                                  style=p_style),
                                          html.Br(),
                                          html.H3('Tuetut selaimet ja tekniset rajoitukset',
                                                  style=h3_style),
                                          
                                          html.P("Sovellus on testattu toimivaksi Google Chromella ja Mozilla Firefoxilla. Edge- ja Internet Explorer -selaimissa sovellus ei toimi. Opera, Safari -ja muita selaimia ei ole testattu.",
                                                  style=p_style),
                                          html.Br(),
                                          html.Div(style={'text-align':'center'},children = [
                                              html.H3('Lähteet', 
                                                      style = h3_style),
                                              
                                              html.Label(['Tilastokeskus: ', 
                                                        html.A('Työvoimatutkimuksen tärkeimmät tunnusluvut, niiden kausitasoitetut aikasarjat sekä kausi- ja satunnaisvaihtelusta tasoitetut trendit', href = "https://statfin.stat.fi/PxWeb/pxweb/fi/StatFin/StatFin__tyti/statfin_tyti_pxt_135z.px/",target="_blank")
                                                      ],style=p_style),
                                              html.Br(),
                                              html.Label(['Tilastokeskus: ', 
                                                        html.A('Kuluttajahintaindeksi (2010=100)', href = "https://statfin.stat.fi/PxWeb/pxweb/fi/StatFin/StatFin__khi/statfin_khi_pxt_11xd.px/",target="_blank")
                                                      ],style=p_style),
                                              html.Br(),
                                              html.Label(['Tilastokeskus: ', 
                                                       html.A('Käsitteet ja määritelmät', href = "https://www.stat.fi/meta/kas/index.html",target="_blank")
                                                      ],style=p_style),
                                              html.Br(),
                                              html.Label(['Wikipedia: ', 
                                                        html.A('Phillipsin käyrä', href = "https://fi.wikipedia.org/wiki/Phillipsin_k%C3%A4yr%C3%A4",target="_blank")
                                                      ],style=p_style),
                                              html.Br(),
                                              html.Label(['Wikipedia: ', 
                                                        html.A('Pääkomponenttianalyysi', href = "https://fi.wikipedia.org/wiki/P%C3%A4%C3%A4komponenttianalyysi",target="_blank")
                                                      ],style=p_style),
                                              html.Br(),
                                              html.Label(['Wikipedia: ', 
                                                        html.A('Pearsonin korrelaatiokerroin (englanniksi)', href = "https://en.wikipedia.org/wiki/Pearson_correlation_coefficient",target="_blank")
                                                      ],style=p_style),
                                              html.Br(),
                                                html.Label(['Scikit-learn: ', 
                                                        html.A('Regressiotekniikat', href = "https://scikit-learn.org/stable/supervised_learning.html#supervised-learning",target="_blank")
                                                        ],style=p_style),
                                                html.Br(),
    
                                          ]),
                                          html.Br(),
                                          html.Br(),
                                          html.H3('Tekijä', style = h3_style),
                                          
                                          html.Div(style = {'textAlign':'center'},children = [
                                              html.I('Tuomas Poukkula', style = p_style),
                                         
                                              html.Br(),
                                              html.P("Data Scientist",
                                                     style = p_style)
                                              ]),
                                                     
                  
                                   
                                          ],xs =12, sm=12, md=12, lg=8, xl=8)
                                      ],justify ='center',
                                      # style={'margin': '10px 10px 10px 10px'}
                                      ),
                                  html.Br(),
                                  dbc.Row([
                                     

                                      dbc.Col([
                                          html.Div([
                                              html.A([
                                                  html.Img(
                                                      src=app.get_asset_url('256px-Linkedin_icon.png'),
                                                    
                                                      style={
                                                            'height' : '70px',
                                                            'width' : '70px',
                                                            'text-align':'right',
                                                            'float' : 'right',
                                                            'position' : 'center',
                                                            'padding-top' : 0,
                                                            'padding-right' : 0
                                                      }
                                                      )
                                          ], href='https://www.linkedin.com/in/tuomaspoukkula/',target = '_blank', 
                                                  style = {'textAlign':'center'})
                                          ],style={'textAlign':'center'})],xs =6, sm=6, md=6, lg=6, xl=6),
                                      dbc.Col([
                                          html.Div([
                                              html.A([
                                                  html.Img(
                                                      src=app.get_asset_url('Twitter-logo.png'),
                                                      style={
                                                          'height' : '70px',
                                                          'width' : '70px',
                                                            'text-align':'left',
                                                            'float' : 'left',
                                                            'position' : 'center',
                                                            'padding-top' : 0,
                                                            'padding-right' : 0
                                                      }
                                                      )
                                          ], href='https://twitter.com/TuomasPoukkula',target = '_blank', style = {'textAlign':'center'})
                                          ],style={'textAlign':'center'})],xs =6, sm=6, md=6, lg=6, xl=6),
                                      ],justify ='center',
                                      
                                      ),
                                  dbc.Row([
                                      dbc.Col([
                                      html.Div([
                                          html.A([
                                              html.Img(
                                                  src=app.get_asset_url('gofore_logo_orange.svg'),
                                                  style={
                                                     
                                                        'text-align':'center',
                                                        'float' : 'center',
                                                        'position' : 'center',
                                                        'padding-top' : '20px',
                                                         'padding-bottom' : '20px'
                                                  }
                                                  )
                                      ], href='https://gofore.com/',target = '_blank', style = {'textAlign':'center'})
                                      ],style={'textAlign':'center'})],xs =12, sm=12, md=12, lg=6, xl=6)
                                      
                                      ],
                                  justify='center'),
                                  dbc.Row([
                                     
                                      dbc.Col([
                                          html.Div(style = {'text-align':'center'},children = [
                                              
                                              html.Label(['Sovellus ', 
                                                      html.A('GitHub:ssa', href='https://github.com/tuopouk/skewedphillips')
                                                      ],style=p_style)
                                      ])
                                          ],xs =12, sm=12, md=12, lg=6, xl=6)
                                      ],
                                      justify ='center',
                                      
                                      ),
                                  html.Br()
                                  ])
                 
                        
                        
                        
                        ]

),
            
            dbc.Tab(label ='Hyödykkeiden valinta',
                    tab_id ='feature_tab',
                     tabClassName="flex-grow-1 text-center",
                    tab_style = {'font-size':'34px',
                                 'font-family':'Cadiz Semibold'},
                    style = {
                        #"position": "fixed",
                        "maxHeight": "1024px",
                       
                          # "height":"1024px",
                    
                        "overflow": "auto"
                    },
              children= [
        
                dbc.Row(children = [
                        
                        html.Br(),
                    
                        dbc.Col(children=[
                           
                            html.Br(),
                            html.P('Tässä osiossa valitaan hyödykkeitä, joita käytetään työttömyyden ennustamisessa.',
                                    style = p_style),
                            html.P('Voit valita alla olevasta valikosta hyödykkeitä, minkä jälkeen voit säätää niiden oletettavaa kuukausimuutosta syöttämällä lukeman alle ilmestyviin laatikkoihin.',
                                    style = p_style),
                            html.P('Voit myös säätää kaikille hyödykkeille saman kuukausimuutoksen tai hyödyntää toteutuneiden kuukausimuutosten keskiarvoja.',
                                    style = p_style),
                            html.P('Hyödykevalikon voi rajata tai lajitella sen yllä olevasta alasvetovalikosta. Valittavanasi on joko aakkosjärjestys, korrelaatiojärjestykset (Pearsonin korrelaatiokertoimen mukaan) tai rajaus Tilastokeskuksen hyödykehierarkian mukaan. Korrelaatiojärjestyksellä tässä viitataan jokaisen hyödykkeen hintaindeksin arvojen ja saman ajankohdan työtömyysasteiden välistä korrelaatiokerrointa, joka on laskettu Pearsonin metodilla. Nämä voi lajitella laskevaan tai nousevaan järjestykseen joko todellisen arvon mukaan (suurin positiivinen - pienin negatiivinen) tai itseisarvon (ilman etumerkkiä +/-) mukaan.',
                                    style = p_style),

                            html.Br(),
                            html.H3('Valitse ennustepiirteiksi hyödykeryhmiä valikosta',
                                    style=h3_style),
                            
                            dbc.DropdownMenu(id = 'sorting',
                                              #align_end=True,
                                              children = [
                                                 
                                                  dbc.DropdownMenuItem("Aakkosjärjestyksessä", id = 'alphabet',style={
                                                      'font-size':p_font_size, 
                                                      'font-family':'Cadiz Book'}),
                                                  dbc.DropdownMenuItem("Korrelaatio (laskeva)", id = 'corr_desc',style={
                                                      'font-size':p_font_size, 
                                                      'font-family':'Cadiz Book'}),
                                                  dbc.DropdownMenuItem("Korrelaatio (nouseva)", id = 'corr_asc',style={
                                                      'font-size':p_font_size, 
                                                      'font-family':'Cadiz Book'}),
                                                  dbc.DropdownMenuItem("Absoluuttinen korrelaatio (laskeva)", id = 'corr_abs_desc',style={
                                                      'font-size':p_font_size, 
                                                      'font-family':'Cadiz Book'}),
                                                  dbc.DropdownMenuItem("Absoluuttinen korrelaatio (nouseva)", id = 'corr_abs_asc',style={
                                                      'font-size':p_font_size, 
                                                      'font-family':'Cadiz Book'}),
                                                  dbc.DropdownMenuItem("Pääluokittain", id = 'main_class',style={
                                                      'font-size':p_font_size, 
                                                      'font-family':'Cadiz Book'}),
                                                  dbc.DropdownMenuItem("2. luokka", id = 'second_class',style={
                                                      'font-size':p_font_size, 
                                                      'font-family':'Cadiz Book'}),
                                                  dbc.DropdownMenuItem("3. luokka", id = 'third_class',style={
                                                      'font-size':p_font_size, 
                                                      'font-family':'Cadiz Book'}),
                                                  dbc.DropdownMenuItem("4. luokka", id = 'fourth_class',style={
                                                      'font-size':p_font_size, 
                                                      'font-family':'Cadiz Book'}),
                                                  dbc.DropdownMenuItem("5. luokka", id = 'fifth_class',style={
                                                      'font-size':p_font_size, 
                                                      'font-family':'Cadiz Book'})
                                                 
                                                 
                                                  ],
                                            label = "Absoluuttinen korrelaatio (laskeva)",
                                            color="secondary", 
                                            className="m-1",
                                            size="lg",
                                            style={
                                                'font-size':28, 
                                                'font-family':'Cadiz Book'}
                                            ),
                            
                            html.Br(),
                            dcc.Dropdown(id = 'feature_selection',
                                          options = initial_options,
                                          multi = True,
                                          value = list(initial_features),
                                          style = {'font-size':25, 'font-family':'Cadiz Book'},
                                          placeholder = 'Valitse hyödykkeitä'),
                            html.Br(),
                            
                            dbc.Alert("Valitse ainakin yksi hyödykeryhmä valikosta!", color="danger",
                                      dismissable=True, fade = True, is_open=False, id = 'alert', 
                                      style = {'text-align':'center', 'font-size':p_font_size, 'font-family':'Cadiz Semibold'}),
                            dash_daq.BooleanSwitch(id = 'select_all', 
                                                    label = dict(label = 'Valitse kaikki',
                                                                 style = {'font-size':p_font_size, 
                                                                          'fontFamily':'Cadiz Semibold'}), 
                                                    on = False, 
                                                    color = 'blue'),
                            html.Br(),
                            # html.Div(id = 'selections_div',children =[]),
                            
                
                            ],xs =12, sm=12, md=12, lg=12, xl=12
                        )
                        ],justify='center', 
                    # style = {'margin' : '10px 10px 10px 10px'}
                    ),
                    dbc.Row(id = 'selections_div', children = [
                        
                            dbc.Col([
                                
                                html.H3('Aseta arvioitu hyödykkeiden hintaindeksien keskimääräinen kuukausimuutos prosenteissa',
                                                        style = h3_style),
                                
                                dash_daq.BooleanSwitch(id = 'averaging', 
                                                        label = dict(label = 'Käytä toteutumien keskiarvoja',style = {'font-size':p_font_size, 'fontFamily':'Cadiz Semibold'}), 
                                                        on = True, 
                                                        color = 'blue')
                                
                                ],xs =12, sm=12, md=12, lg=6, xl=6),
      
                ],justify = 'center', 
                    
                ),
                    html.Br(),
                dbc.Row([
                        dbc.Col([
                            html.Div(id = 'slider_div'),
                            html.Br(),
                            html.Div(id='slider_prompt_div')
                            
                            ], xs =12, sm=12, md=12, lg=9, xl=9)
                        
                        ],justify = 'center', 
                  
                    ),
                dbc.Row(id = 'adjustments_div',
                        justify = 'left', 
                     
                        )
            ]
            ),


            dbc.Tab(label = 'Tutkiva analyysi',
                    tab_id = 'eda_tab',
                    tabClassName="flex-grow-1 text-center",
                    tab_style = {'font-size':'34px',
                                 'font-family':'Cadiz Semibold'},
                    style = {
                            
                            "maxHeight": "1024px",

                            "overflow": "auto"
                        },
                     children = [
                    
                     dbc.Row([
                         
                         dbc.Col([
                             html.Br(),
                              html.Br(),
                              html.P('Tässä osiossa työttömyysastetta sekä valittujen kuluttajahintaindeksin hyödykeryhmien keskinäistä suhdetta sekä muutosta ajassa. Alla voit nähdä kuinka eri hyödykeryhmien hintaindeksit korreloivat keskenään sekä työttömyysasteen kanssa. Voit myös havainnoida indeksien, inflaation sekä sekä työttömyysasteen aikasarjoja. Kuvattu korrelaatio perustuu Pearsonin korrelaatiokertoimeen.',
                                     style = p_style),
                              html.Br()
                              ],xs =12, sm=12, md=12, lg=9, xl=9)
                         ],
                             justify = 'center', 
                             style = {'textAlign':'center',
                                       # 'margin':'10px 10px 10px 10px'
                                      }
                             ),
                    
                     dbc.Row([
                                dbc.Col(children = [
                                   
                                        html.Div(id = 'corr_selection_div'),
                                        html.Br(),
                                        html.Div(id = 'eda_div'),
                                        html.Br(),
                                       
                                   
                                    ], xs =12, sm=12, md=12, lg=6, xl=6
                                ),
                                dbc.Col(children = [
                                   
                                        html.Div(id = 'feature_corr_selection_div'),
                                        html.Br(),
                                        html.Div(id = 'corr_div'),
                                   
                                    ], xs =12, sm=12, md=12, lg=6, xl=6, align ='start'
                                )
                         ],justify='center', 
                          # style = {'margin' : '10px 10px 10px 10px'}
                     ),

                    # html.Br(),
                     dbc.Row(id = 'timeseries_div',children=[
                        
                         dbc.Col(xs =12, sm=12, md=12, lg=6, xl=6,
                                 children = [
                                     html.Div(id = 'timeseries_selection'),
                                     html.Br(),
                                     html.Div(id='timeseries')
                                     ]),
                         dbc.Col(children = [
                             html.Div(style={'textAlign':'center'},children=[
                                 html.Br(),
                                 html.H3('Alla olevassa kuvaajassa on esitetty inflaatio ja työttömyys Suomessa kuukausittain.',
                                        style = h3_style),
                                 
                                 html.Div(id = 'employement_inflation_div',
                                          
                                          children=[dcc.Graph(id ='employement_inflation',
                                                     figure = go.Figure(data=[go.Scatter(x = data.index,
                                                                               y = data.Työttömyysaste,
                                                                               name = 'Työttömyysaste',
                                                                               mode = 'lines',
                                                                               marker = dict(color ='red')),
                                                                    go.Scatter(x = data.index,
                                                                               y = data.Inflaatio,
                                                                               name = 'Inflaatio',
                                                                               mode ='lines',
                                                                               marker = dict(color = 'purple'))],
                                                              layout = go.Layout(title = dict(text = 'Työttömyysaste ja inflaatio kuukausittain<br>{} - {}'.format(data.index.strftime('%B %Y').values[0],data.index.strftime('%B %Y').values[-1]),
                                                                                              x=.5,
                                                                                              font=dict(
                                                                                                  family='Cadiz Semibold',
                                                                                                   size=16
                                                                                                  )),
                                                                                 height=graph_height,
                                                                                 template = 'seaborn',
                                                                                 margin=dict(
                                                                                      l=10,
                                                                                     r=10,
                                                                                     # b=100,
                                                                                      # t=120,
                                                                                      # pad=4
                                                                                 ),
                                                                                 hoverlabel=dict(font=dict(family='Cadiz Book',size=14)),
                                                                                 legend = dict(orientation = 'h',
                                                                                                xanchor='center',
                                                                                                yanchor='top',
                                                                                                x=.5,
                                                                                                y=1.04,
                                                                                               font=dict(
                                                                                      size=12,
                                                                                     family='Cadiz Book')),
                                                                                 xaxis = dict(title=dict(text = 'Aika',
                                                                                                         font=dict(
                                                                                                              size=18, 
                                                                                                             family = 'Cadiz Semibold')), 
                                                                                              tickfont = dict(
                                                                                                  family = 'Cadiz Semibold', 
                                                                                                   size = 16
                                                                                                  ),
                                                                                              rangeslider=dict(
                                                                                                  visible=True
                                                                                                  ),
                                                                                              rangeselector=dict(
                                                                                                  buttons=list([
                                                                                                      dict(count=1,
                                                                                                           label="1kk",
                                                                                                           step="month",
                                                                                                           stepmode="backward"),
                                                                                                      dict(count=6,
                                                                                                           label="6kk",
                                                                                                           step="month",
                                                                                                           stepmode="backward"),
                                                                                                      dict(count=1,
                                                                                                           label="YTD",
                                                                                                           step="year",
                                                                                                           stepmode="todate"),
                                                                                                      dict(count=1,
                                                                                                           label="1v",
                                                                                                           step="year",
                                                                                                           stepmode="backward"),
                                                                                                      dict(count=3,
                                                                                                           label="3v",
                                                                                                           step="year",
                                                                                                           stepmode="backward"),
                                                                                                      dict(count=5,
                                                                                                           label="5v",
                                                                                                           step="year",
                                                                                                           stepmode="backward"),
                                                                                                      dict(step="all",label = 'MAX')
                                                                                                  ])
                                                                                              )
                                                                                              ),
                                                                                 yaxis = dict(title=dict(text = 'Arvo (%)',
                                                                                                        font=dict(
                                                                                                             size=18, 
                                                                                                            family = 'Cadiz Semibold')),
                                                                                             tickformat = ' ',
                                                                                             automargin=True,
                                                                                             tickfont = dict(
                                                                                                 family = 'Cadiz Semibold', 
                                                                                                  size = 16
                                                                                                 )
                                                                                             )
                                                                                
                                                                                 )
                                                                                 ),
                                           config=config_plots
                                           )])
                                 ])
                                
                            
                             ], xs =12, sm=12, md=12, lg=6, xl=6, align ='end')
                        
                         ],
                         justify='center', 
                         # style = {'margin' : '10px 10px 10px 10px'}
                         )
                
                     ]
                ),
            dbc.Tab(label='Menetelmän valinta',
                    tab_id ='hyperparam_tab',
                    tabClassName="flex-grow-1 text-center",
                    tab_style = {'font-size':'34px',
                                 'font-family':'Cadiz Semibold'},
                    style = {
                            
                            "maxHeight": "1024px",

                            "overflow": "auto"
                        },
                    
                    children = [
                        dbc.Row([
                            
                            dbc.Col([
                            
                                html.Br(),
                                html.P('Tässä osiossa voit valita koneoppimisalgoritmin, säätää sen hyperparametrit sekä halutessasi hyödyntää pääkomponenttianalyysia valitsemallasi tavalla.',
                                        style = p_style),
                                html.P('Valitse ensin algoritmi, minkä jälkeen alle ilmestyy sille ominaiset hyperparametrit, joita voit säätää sopiviksi. Hyperparametrit luetaan dynaamisesti suoraan Scikit-learn -kirjaston dokumentaatiosta, eikä niille siksi ole suomenkielistä käännöstä. Säätövalikoiden alta löytyy linkki valitun algoritmin dokumentaatiosivulle, jossa aiheesta voi lukea enemmän. Hyperparametrien säädölle ei ole yhtä ainutta tapaa, vaan eri arvoja on testattava iteratiivisesti.',
                                        style = p_style),
                                html.Br(),
                                html.P('Lisäksi voit valita hyödynnetäänkö pääkomponenttianalyysiä piirteiden karsimiseksi. Pääkompomponenttianalyysi on tilastollis-tekninen kohinanpoistomenetelmä, jolla pyritään parantamaan ennusteen laatua. Siinä valituista piirteistä muodostetaan lineaarikombinaatioita siten, että alkuperäisessä datassa oleva variaatio säilyy tietyn suhdeluvun verran muunnetussa aineistossa. Variaatio voi säätää haluamakseen. Kuten hyperparametrien tapauksessa, on tämäkin määrittely puhtaasti empiirinen.',
                                        style = p_style),
                                html.P('Mikäli hyperparametrin laatikon reunat ovat punaisena, niin arvo ei ole sopiva. Testaaminen ja ennustaminen epäonnistuvat, jos hyperparametreihin sovelleta sallittuja arvoja. Voit tarkastaa sallitut arvot mallin dokumentaatiosta.',
                                        style = p_style)
                            ],xs =12, sm=12, md=12, lg=9, xl=9)
                        ], justify = 'center', 
                            style = {'textAlign':'center',
                                      # 'margin':'10px 10px 10px 10px'
                                     }
                            ),
                        html.Br(),
                        dbc.Row([
                            dbc.Col(id = 'model_selection', children = [
                                
                                html.H3('Valitse algoritmi',style=h3_style),
                                
                                dcc.Dropdown(id = 'model_selector',
                                              value = 'Satunnaismetsä',
                                              multi = False,
                                              placeholder = 'Valitse algoritmi',
                                              style = {'font-size':p_font_size, 'font-family':'Cadiz Book','color': 'black'},
                                              options = [{'label': c, 'value': c} for c in MODELS.keys()]),
                                
                                html.Br(),
                                html.H3('Säädä hyperparametrit', style = h3_style),
                                
                                html.Div(id = 'hyperparameters_div')
                                
                                ], xs =12, sm=12, md=12, lg=9, xl=9),
                            dbc.Col(id = 'pca_selections', children = [
                                html.Br(),
                                dash_daq.BooleanSwitch(id = 'pca_switch', 
                                                                  label = dict(label = 'Käytä pääkomponenttianalyysia',style = {'font-size':30, 'fontFamily':'Cadiz Semibold','textAlign':'center'}), 
                                                                  on = False, 
                                                                  color = 'blue'),
                                html.Br(),
                                
                                
                                html.Div(id = 'ev_placeholder',children =[
                                    html.H3('Valitse säilytettävä variaatio', style = h3_style),
                                    
                                    dcc.Slider(id = 'ev_slider',
                                        min = .7, 
                                        max = .99, 
                                        value = .95, 
                                        step = .01,
                                        tooltip={"placement": "top", "always_visible": True},
                                        marks = {
                                                  .7: {'label':'70%', 'style':{'font-size':20, 'fontFamily':'Cadiz Semibold'}},
                                            .85: {'label':'85%', 'style':{'font-size':20, 'fontFamily':'Cadiz Semibold'}},
                                                  .99: {'label':'99%', 'style':{'font-size':20, 'fontFamily':'Cadiz Semibold'}}

                                                }
                                      ),
                                    html.Br(),
                                  html.Div(id = 'ev_slider_update', 
                                          children = [
                                              html.Div([html.P('Valitsit {} % säilytetyn variaation.'.format(95),
                                                                style = p_style)
                                                        ], style = {'display':'none'}
                                                      )
                                          ]
                                        )
                                  ]
                                    )
                                
                                ],xs =12, sm=12, md=12, lg=3, xl=3)
                        ],justify='left', 
                              # style = {'margin' : '10px 10px 10px 10px'}
                            ),
                        html.Br(),
                        
                        
                        ]
                    ),
            dbc.Tab(label='Testaaminen',
                    tab_id ='test_tab',
                    tabClassName="flex-grow-1 text-center",
                    tab_style = {'font-size':'34px',
                                 'font-family':'Cadiz Semibold'},
                    style = {
                            
                            "maxHeight": "1024px",

                            "overflow": "auto"
                        },
                    children = [
                        html.Br(),
                        dbc.Row([
                            dbc.Col([   
                                        html.H3('Menetelmän testaaminen', style = h3_style),
                                        
                                        html.P('Tässä osiossa voit testata kuinka hyvin valittu menetelmä olisi onnistunut ennustamaan menneiden kuukausien työttömyysasteen hyödyntäen valittuja piirteitä. Testattaessa valittu määrä kuukausia jätetään testidataksi, jota menetelmä pyrkii ennustamaan.',
                                               style = p_style),
                                        html.P('Tässä kohtaa hyödykeindeksien oletetaan toteutuvan sellaisinaan.',
                                               style = p_style),
                                        html.P('Tehtyäsi testin voit tarkastella viereistä tuloskuvaajaa tai viedä testidatan alle ilmestyvästä painikeesta Exceliin.',
                                              style=p_style)
                            
                            
                            ],xs =12, sm=12, md=12, lg=9, xl=9)
                            ], justify = 'center', 
                            style = {'textAlign':'center', 
                                      'margin':'10px 10px 10px 10p'
                                     }
                            ),
                        html.Br(),
                        dbc.Row(children = [
                            
                            dbc.Col(children = [
                                        html.Br(),

                                        html.Br(),
                                        html.H3('Valitse testidatan pituus',style = h3_style),
                                        
                                        dcc.Slider(id = 'test_slider',
                                                  min = 1,
                                                  max = 12,
                                                  value = 3,
                                                  step = 1,
                                                  tooltip={"placement": "top", "always_visible": True},
                                                 
                                                  marks = {1: {'label':'kuukausi', 'style':{'font-size':20, 'fontFamily':'Cadiz Semibold'}},
                                                          # 3:{'label':'3 kuukautta', 'style':{'font-size':20, 'fontFamily':'Cadiz Semibold','color':'white'}},
                                                        
                                                            6:{'label':'puoli vuotta', 'style':{'font-size':20, 'fontFamily':'Cadiz Semibold'}},
                                                          #  9:{'label':'yhdeksän kuukautta', 'style':{'font-size':20, 'fontFamily':'Cadiz Semibold','color':'white'}},
                                                         
                                                          12:{'label':'vuosi', 'style':{'font-size':20, 'fontFamily':'Cadiz Semibold'}},
                                                          

                                                            }
                                                  ),
                                      html.Br(),  
                                      html.Div(id = 'test_size_indicator', style = {'textAlign':'center'}),
                                      html.Br(),
                                      html.Div(id = 'test_button_div',children = [html.P('Valitse ensin hyödykkeitä.',style = {
                                          'text-align':'center', 
                                          'font-family':'Cadiz Semibold', 
                                            'font-size':p_font_size
                                          })], style = {'textAlign':'center'}),
                                      html.Br(),
                                      html.Div(id='test_download_button_div', style={'textAlign':'center'})
                                      
                                      ],xs = 12, sm = 12, md = 12, lg = 4, xl = 4
                                ),
                            dbc.Col([html.Div(id = 'test_results_div')],xs = 12, sm = 12, md = 12, lg = 8, xl = 8)
                            ], justify = 'center', 
                             # style = {'margin' : '10px 10px 10px 10px'}
                            ),
                 
                        
                        
                        
                        ]
                    ),
            dbc.Tab(label='Ennustaminen',
                    tab_id = 'forecast_tab',
                    tabClassName="flex-grow-1 text-center",
                    tab_style = {'font-size':'34px',
                                 'font-family':'Cadiz Semibold'},
                    style = {
                            
                            "maxHeight": "1024px",

                            "overflow": "auto"
                        },
                    children = [
                        html.Br(),
                        dbc.Row([
                            
                            dbc.Col([
                                        html.H3('Ennusteen tekeminen',style=h3_style),
                                        
                                        html.P('Tässä osiossa voit tehdä ennusteen valitulle ajalle. Ennustettaessa on käytössä Menetelmän valinta -välilehdellä tehdyt asetukset. Ennusteen tekemisessä hyödynnetään Hyödykkeiden valinta -välilehdellä tehtyjä oletuksia hyödykkeiden suhteellisesta hintakehityksestä.',
                                              style=p_style),
                                        html.P('Tehtyäsi ennusteen voit tarkastella viereistä ennusteen kuvaajaa tai viedä tulosdatan alle ilmestyvästä painikeesta Exceliin.',
                                              style=p_style),
                                        html.Br()
                                    ],xs =12, sm=12, md=12, lg=9, xl=9)
                            
                            
                            ], justify = 'center', 
                            style = {'textAlign':'center',
                                      'margin':'10px 10px 10px 10px'
                                     }),
                        html.Br(),
                        dbc.Row(children = [
                                    #dbc.Col(xs =12, sm=12, md=12, lg=3, xl=3, align = 'start'),
                                    dbc.Col(children = [
                                        html.Br(),

                                        html.H3('Valitse ennusteen pituus',
                                                style=h3_style),
                                        dcc.Slider(id = 'forecast_slider',
                                                  min = 2,
                                                  max = 12,
                                                  value = 3,
                                                  step = 1,
                                                  tooltip={"placement": "top", "always_visible": True},
                                                  marks = {2: {'label':'2 kuukautta', 'style':{'font-size':16, 'fontFamily':'Cadiz Semibold'}},
                                                          # 3: {'label':'kolme kuukautta', 'style':{'font-size':20, 'fontFamily':'Cadiz Semibold','color':'white'}},
                                                          6:{'label':'puoli vuotta', 'style':{'font-size':16, 'fontFamily':'Cadiz Semibold'}},
                                                          # 9:{'label':'yhdeksän kuukautta', 'style':{'font-size':20, 'fontFamily':'Cadiz Semibold','color':'white'}},
                                                          12:{'label':'vuosi', 'style':{'font-size':16, 'fontFamily':'Cadiz Semibold'}},
                                                        #  24:{'label':'kaksi vuotta', 'style':{'font-size':20, 'fontFamily':'Cadiz Semibold','color':'white'}}
                                                        
                                                     

                                                          }
                                                  ),
                                        html.Br(),
                                        html.Div(id = 'forecast_slider_indicator',style = {'textAlign':'center'}),
                                        html.Div(id = 'forecast_button_div',children = [html.P('Valitse ensin hyödykkeitä.',
                                                                                              style = p_style
                                                                                              )],style = {'textAlign':'center'})],
                                        xs =12, sm=12, md=12, lg=4, xl=4
                                        ),
                                    html.Br(),
                                    
                                    dbc.Col([dcc.Loading(id = 'forecast_results_div',type = spinners[random.randint(0,len(spinners)-1)])],
                                            xs = 12, sm = 12, md = 12, lg = 8, xl = 8)
                                    ], justify = 'center', 
                             # style = {'margin' : '10px 10px 10px 10px'}
                                    )
                                       
                            
                            
                            
                            
                              
                            
                        ]
                            
                            )


        ]
            
    ),

    
   ]
  )

@app.callback(

    [Output('adjustments_div','children'),
     Output('features_values','data')],
    [Input('slider','value'),
     Input('feature_selection','value')],
    [State('averaging','on')]    
    
)
def add_value_adjustments(slider_value, features, averaging):
    
    
    
    if averaging:
        
        mean_df = apply_average(features = features, length = slider_value)
        
        features_values = {feature:mean_df.loc[feature] for feature in features}
        
        row_children =[dbc.Col([html.Br(), 
                                html.P(feature,style={'font-family':'Messina Modern Semibold',
                                            'font-size':22}),
                                dcc.Input(id = {'type':'value_adjust', 'index':feature}, 
                                               value = round(mean_df.loc[feature],1), 
                                               type = 'number', 
                                               style={'font-family':'Messina Modern Semibold',
                                                           'font-size':22},
                                               step = .1)],xs =12, sm=12, md=4, lg=2, xl=2) for feature in features]
    else:
        
        features_values = {feature:slider_value for feature in features}
        
        row_children =[dbc.Col([html.Br(), 
                                html.P(feature,style={'font-family':'Messina Modern Semibold',
                                            'font-size':22}),
                                dcc.Input(id = {'type':'value_adjust', 'index':feature}, 
                                               value = slider_value, 
                                               type = 'number', 
                                               style ={'font-family':'Messina Modern Semibold',
                                                           'font-size':22},
                                               step = .1)],xs =12, sm=12, md=4, lg=2, xl=2) for feature in features]
    return row_children, features_values


@app.callback(

    Output('change_weights','data'),
    [Input({'type': 'value_adjust', 'index': ALL}, 'id'),
    State({'type': 'value_adjust', 'index': ALL}, 'value')],    
    
)
def store_weights(feature_changes, feature_change_values):
    
    if feature_changes is None:
        raise PreventUpdate
    
    weights_dict = {feature_changes[i]['index']:feature_change_values[i] for i in range(len(feature_changes))}
        
    return weights_dict


@app.callback(

    Output('slider_div','children'),
    [Input('averaging', 'on')
     ]
    
)
def update_slider_div(averaging):
    
    if averaging:
        
        return [html.H3('Valitse kuinka monen edeltävän kuukauden keskiarvoa käytetään', 
                        style = h3_style),
        html.Br(),
        dcc.Slider(id = 'slider',
                      min = 1,
                      max = 12,
                      value = 4,
                      step = 1,
                      tooltip={"placement": "top", "always_visible": True},
                       marks = {1:{'label':'kuukausi', 'style':{'font-size':20, 'fontFamily':'Cadiz Semibold'}},
                                # 3:{'label':'3 kuukautta', 'style':{'font-size':20, 'fontFamily':'Cadiz Semibold','color':'white'}},
                                6:{'label':'puoli vuotta', 'style':{'font-size':20, 'fontFamily':'Cadiz Semibold'}},
                                # 9:{'label':'yhdeksän kuukautta', 'style':{'font-size':20, 'fontFamily':'Cadiz Semibold','color':'white'}},
                                12:{'label':'vuosi', 'style':{'font-size':20, 'fontFamily':'Cadiz Semibold'}}   
                             }
                      
                    )]
        
    else:
        return [
            html.H3('Valitse kuinka iso suhteellista muutosta sovelletaan', 
                    style = h3_style),
            
            dcc.Slider(id = 'slider',
                          min = -10,
                          max = 10,
                          value = 0,
                          step = 0.1,
                          tooltip={"placement": "top", "always_visible": True},
                           marks = {-10:{'label':'-10%', 'style':{'font-size':20, 'fontFamily':'Cadiz Semibold'}},
                                    # 3:{'label':'3 kuukautta', 'style':{'font-size':20, 'fontFamily':'Cadiz Semibold','color':'white'}},
                                    0:{'label':'0%', 'style':{'font-size':20, 'fontFamily':'Cadiz Semibold'}},
                                    # 9:{'label':'yhdeksän kuukautta', 'style':{'font-size':20, 'fontFamily':'Cadiz Semibold','color':'white'}},
                                    10:{'label':'10%', 'style':{'font-size':20, 'fontFamily':'Cadiz Semibold'}}   
                                 }
                          
                        )
            
            ]


@app.callback(

    Output('alert', 'is_open'),
    [Input('feature_selection','value')]

)
def update_alert(features):
    
    return len(features) == 0

@app.callback(

    Output('hyperparameters_div','children'),
    [Input('model_selector','value')]    
    
)
def update_hyperparameter_selections(model_name):
    
    
    
    model = MODELS[model_name]['model']
    
        
    hyperparameters = model().get_params()
    
    type_dict ={}
    for i, c in enumerate(hyperparameters.values()):
      
        type_dict[i] =str(type(c)).split()[1].split('>')[0].replace("'",'')
        
    h_series = pd.Series(type_dict).sort_values()

    param_options = get_param_options(model_name)
       
    
    children = []
    
    for i in h_series.index:
        
        hyperparameter = list(hyperparameters.keys())[i]
        
        
        
        if hyperparameter not in UNWANTED_PARAMS:
        
            
            value = list(hyperparameters.values())[i]
            
            
            
            if type(value) == int:
                children.append(dbc.Col([html.P(hyperparameter+':', 
                                                 style=p_bold_style)],xs =12, sm=12, md=12, lg=12, xl=12))
                children.append(html.Br())
                children.append(dbc.Col([dcc.Slider(id = {'index':hyperparameter, 'type':'hyperparameter_tuner'},
                                   value = value,
                                   max = 4*(value+5),
                                   min = value,
                                   marks=None,                                
                                   tooltip={"placement": "bottom", "always_visible": True},
                                   step = 1),
                                  html.Br()
                                         ],xs =12, sm=12, md=12, lg=12, xl=12)
                                )
                
            elif type(value) == float:
                
                if hyperparameter in LESS_THAN_ONE:
                    
                    children.append(dbc.Col([html.P(hyperparameter+':', 
                                                     style=p_bold_style)],xs =12, sm=12, md=12, lg=2, xl=2)),
                    children.append(html.Br())
                    children.append(dbc.Col([dbc.Input(id = {'index':hyperparameter, 'type':'hyperparameter_tuner'},
                                                       min = 0,
                                                       max=0.99,
                                                       type = 'number',
                                                       style = {'font-family':'Cadiz Book','font-size':p_font_size},
                                                       value = value,
                                                       className="mb-3",
                                                       step=0.01),
                                                      html.Br()],xs =12, sm=12, md=12, lg=2, xl=2)
                                    )
                elif hyperparameter in LESS_THAN_HALF:
                        
                        children.append(dbc.Col([html.P(hyperparameter+':', 
                                                         style=p_bold_style)],xs =12, sm=12, md=12, lg=2, xl=2)),
                        children.append(html.Br())
                        children.append(dbc.Col([dbc.Input(id = {'index':hyperparameter, 'type':'hyperparameter_tuner'},
                                                           min = 0,
                                                           max=0.49,
                                                           type = 'number',
                                                           value = value,
                                                           className="mb-3",
                                                           style = {'font-family':'Cadiz Book','font-size':p_font_size},
                                                           step=0.01),
                                                          html.Br()],xs =12, sm=12, md=12, lg=2, xl=2)
                                        )
                    
                else:
                    children.append(dbc.Col([html.P(hyperparameter+':', 
                                                     style=p_bold_style)],xs =12, sm=12, md=12, lg=2, xl=2)),
                    children.append(html.Br())
                    children.append(dbc.Col([dbc.Input(id = {'index':hyperparameter, 'type':'hyperparameter_tuner'},
                                                       min = 0,
                                                       type = 'number',
                                                       value = value,
                                                       className="mb-3",
                                                       style = {'font-family':'Cadiz Book','font-size':p_font_size},
                                                       step=0.01),
                                                      html.Br()
                                             ],xs =12, sm=12, md=12, lg=2, xl=2)
                                    )
                    
                children.append(html.Br())
                
            
            elif type(value) == bool:
                children.append(dbc.Col([html.P(hyperparameter+':', 
                                                 style=p_bold_style)],xs =12, sm=12, md=12, lg=2, xl=2)),
                children.append(html.Br())
                children.append(dbc.Col([dbc.Switch(id = {'index':hyperparameter, 'type':'hyperparameter_tuner'}, 
                                          style= {'font-family':'Cadiz Book','font-size':p_font_size},
                                          value=value,
                                          ),
                                         html.Br()],xs =12, sm=12, md=12, lg=2, xl=2)
                                )
                children.append(html.Br())
                
            elif type(value) == str:
                    
                    if type(param_options[hyperparameter]) == list:
                        
                        
                        children.append(dbc.Col([html.P(hyperparameter+':', 
                                                         style=p_bold_style)],xs =12, sm=12, md=12, lg=2, xl=2)),
                        children.append(html.Br())
                        children.append(dbc.Col([dcc.Dropdown(id = {'index':hyperparameter, 'type':'hyperparameter_tuner'}, 
                                                  multi = False,
                                                  #label = hyperparameter,
                                                  style = {'font-family':'Cadiz Book','font-size':p_font_size},
                                                  options = [{'label':c, 'value': c} for c in param_options[hyperparameter] if c not in ['precomputed','poisson']],
                                                  value = value),
                                                 html.Br()],xs =12, sm=12, md=12, lg=2, xl=2)
                                    )
                        children.append(html.Br())
                        children.append(html.Br())
                       
     
    children.append(html.Br()) 
    children.append(html.Div(style = {'textAlign':'center'},
             children = [html.A('Katso dokumentaatio.', href = MODELS[model_name]['doc'], target="_blank",style = p_style)]))
    return dbc.Row(children, justify ='start')


@app.callback(

    Output('method_selection_results','data'),
    [Input('model_selector','value'),
    Input({'type': 'hyperparameter_tuner', 'index': ALL}, 'id'),
    Input({'type': 'hyperparameter_tuner', 'index': ALL}, 'value'),
    Input('pca_switch','on'),
    Input('ev_slider','value')]    
    
)
def store_method_selection_results(model_name, hyperparams, hyperparam_values,pca, explained_variance):
            
    hyperparam_grid  = {hyperparams[i]['index']:hyperparam_values[i] for i in range(len(hyperparams))}
    
    if pca:
    
        result_dict = {'model':model_name,
                       'pca':pca,
                       'explained_variance':explained_variance,
                       'hyperparam_grid':hyperparam_grid}
    else:
        
        result_dict = {'model':model_name,
                       'pca':pca,
                       'explained_variance':None,
                       'hyperparam_grid':hyperparam_grid}
      
    return result_dict

@app.callback(
    
      [Output('test_data','data'),
       Output('test_results_div','children'),
       Output('test_download_button_div','children')],
      [Input('test_button','n_clicks')],
      [State('test_slider', 'value'),
       State('features_values','data'),
       State('method_selection_results','data')

       ]
    
)
def update_test_results(n_clicks, 
                        test_size,
                        features_values,
                        method_selection_results
 
                        ):
    
    if n_clicks > 0:
    
        df = data.iloc[:len(data)-test_size,:].copy()
        
        features = sorted(list(features_values.keys()))
        
        model_name = method_selection_results['model']
             
        pca = method_selection_results['pca']
        explained_variance = method_selection_results['explained_variance']
        hyperparam_grid = method_selection_results['hyperparam_grid']
        
        model = MODELS[model_name]['model']
        constants = MODELS[model_name]['constant_hyperparameters'] 
        
        
        model_hyperparams = hyperparam_grid.copy()
        
        
        model_hyperparams.update(constants)
        
        
        model = model(**model_hyperparams)
        
        
        test_df = predict(df, model, features, feature_changes = None, length=test_size, use_pca=pca,n_components=explained_variance)

        test_result = test_results(test_df)

        
        mape = test_result.mape.values[0]
        
        led_color = 'red'
        
        if mape <=.25:
            led_color='orange'
        if mape <= .1:
            led_color='green'

        
        test_plot =[html.Br(),
                    dbc.Row([
                        
                        # dbc.Col([
                            html.Br(),
                             html.H3('Testin tulokset',
                                     style = h3_style),
                             
                             html.P('Alla olevassa kuvaajassa nähdään kuinka hyvin ennustemalli olisi ennustanut työttömyysasteen ajalle {} - {}.'.format(test_result.index.strftime('%B %Y').values[0],test_result.index.strftime('%B %Y').values[-1]),
                                    style = p_style),
                              html.Div([html.Br(),dbc.RadioItems(id = 'test_chart_type', 
                                          options = [{'label':'pylväät','value':'bars'},
                                                    {'label':'viivat','value':'lines'},
                                                    {'label':'viivat ja pylväät','value':'lines+bars'}],
                                          labelStyle={'display':'inline-block', 'padding':'10px','margin':'10px 10px 10px 10px','font-size':25,'font-family':'Cadiz Book'},
                                          className="btn-group",
                                          inputClassName="btn-check",
                                          labelClassName="btn btn-outline-warning",
                                          labelCheckedClassName="active",
                                        
                                          value = 'lines+bars'
                                        ),html.Br()
                                    ],
                                    style = {'textAlign':'right'}
                                  ),
                              html.Br(),
                              html.Div([dcc.Loading(id ='test_graph_div',type = spinners[random.randint(0,len(spinners)-1)])])
   
                        ],style= {'textAlign':'center'}),
                    html.Br(),
                    
                    dbc.Row(
                        
                        [
                            dbc.Col([html.Br(),
                                    html.H3('MAPE (%)', style = h3_style),
                                     dash_daq.LEDDisplay(backgroundColor='black',
                                                         size =70,
                                                         color = led_color,
                                                         style = {'textAlign':'center'},
                                                         value = round(100*mape,1))
                                     ],
                                    xs =12, sm=12, md=12, lg=6, xl=6
                                    ),
                            
                            dbc.Col([html.Br(),
                                     html.H3('Tarkkuus (%)', style = h3_style),
                                     dash_daq.LEDDisplay(backgroundColor='black',
                                                         size =70,
                                                         color = led_color,
                                                         style = {'textAlign':'center'},
                                                         value = round(100*(1-mape),1))
                                     ],
                                    xs =12, sm=12, md=12, lg=6, xl=6
                                    )
                            
                            ], justify ='center',style= {'textAlign':'center'}
                        
                        ),
                    html.Br(),
                    html.P('Keskimääräinen suhteellinen virhe (MAPE) on kaikkien ennustearvojen suhteellisten virheiden keskiarvo. Tarkkuus on tässä tapauksessa laskettu kaavalla 1 - MAPE.', 
                           style = p_style,
                           className="card-text"),
                    html.Br(),

                    
                    ]
             

        feat = features.copy()
        feat = ['Työttömyysaste','Ennuste','month','change','mape','n_feat']+feat
        
        button_children = dbc.Button(children=[html.I(className="fa fa-download mr-1"), ' Lataa testitulokset koneelle'],
                                       id='test_download_button',
                                       n_clicks=0,
                                       style = dict(fontSize=30,fontFamily='Cadiz Semibold',textAlign='center'),
                                       outline=True,
                                       size = 'lg',
                                       color = 'info'
                                       )
        
        return test_result[feat].reset_index().to_dict('records'), test_plot, button_children
    else:
        return [html.Div(),html.Div(),html.Div()]


        
@app.callback(
    
      [Output('forecast_data','data'),
       Output('forecast_results_div','children'),
       Output('forecast_download_button_div','children')],
      [Input('forecast_button','n_clicks')],
      [State('forecast_slider', 'value'),
       State('change_weights','data'),
       State('method_selection_results','data')

       ]
    
)
def update_forecast_results(n_clicks, 
                        forecast_size,
                        weights_dict,
                        method_selection_results
                        ):

    
    
    if n_clicks > 0:
        
        features = sorted(list(weights_dict.keys()))
        
        df = data.copy()
        
        model_name = method_selection_results['model']
             
        pca = method_selection_results['pca']
        explained_variance = method_selection_results['explained_variance']
        hyperparam_grid = method_selection_results['hyperparam_grid']
        
        model = MODELS[model_name]['model']
        constants = MODELS[model_name]['constant_hyperparameters'] 
        
        
        model_hyperparams = hyperparam_grid.copy()
        
        
        model_hyperparams.update(constants)
        
        
        model = model(**model_hyperparams)
        
        
        weights = pd.Series(weights_dict)
        forecast_df = predict(df, 
                              model, 
                              features, 
                              feature_changes = weights, 
                              length=forecast_size, 
                              use_pca=pca,
                              n_components=explained_variance)
        

        
        forecast_div =  html.Div([html.Br(),
                      html.H3('Ennustetulokset', 
                              style = h3_style),
                      
                      html.P('Alla olevassa kuvaajassa on esitetty toteuteet arvot sekä ennuste ajalle {} - {}.'.format(forecast_df.index.strftime('%B %Y').values[0],forecast_df.index.strftime('%B %Y').values[-1]),
                             style = p_style),
                      html.P('Voit valita alla olevista painikkeista joko pylväs, -tai viivadiagramin. Kuvaajan pituutta voi säätää alla olevasta liukuvaliskosta. Pituutta voi rajat myös vasemman yläkulman painikkeista.',
                             style = p_style),
                      
                      html.Div([
                      dbc.RadioItems(id = 'chart_type', 
                        options = [{'label':'pylväät','value':'bars'},
                                  {'label':'viivat','value':'lines'}],
                        labelStyle={'display':'inline-block', 'padding':'10px','margin':'10px 10px 10px 10px','font-size':25,'font-family':'Cadiz Book'},
                        className="btn-group",
                        inputClassName="btn-check",
                        labelClassName="btn btn-outline-warning",
                        labelCheckedClassName="active",                        
                        value = 'lines'
                      )
                      ],style={'textAlign':'right'}),
                      html.Br(),

          html.Div(id = 'forecast_graph_div'),
        
          html.Br()
          ],style={'textAlign':'center'})

          
          # ], justify='center')        
        forecast_download_button = dbc.Button(children=[html.I(className="fa fa-download mr-1"), ' Lataa ennustedata koneelle'],
                                 id='forecast_download_button',
                                 n_clicks=0,
                                 style=dict(fontSize=30,fontFamily='Cadiz Semibold',textlign='center'),
                                 outline=True,
                                 size = 'lg',
                                 color = 'info'
                                 )
        
        feat = features.copy()
        feat = ['Työttömyysaste','month','change','n_feat']+feat
        
        return [forecast_df[feat].reset_index().to_dict('records'), forecast_div, [html.Br(),forecast_download_button]]
    else:
        return [html.Div(),html.Div(),html.Div()]

@app.callback(
    Output("forecast_download", "data"),
    [Input("forecast_download_button", "n_clicks"),
    [State('forecast_data','data'),
     State('method_selection_results','data'),
     State('change_weights','data')
     ]
    ]
    
)
def download_forecast_data(n_clicks, df, method_selection_results, weights_dict):
    
    if n_clicks > 0:
        
        
        df = pd.DataFrame(df).set_index('Aika').copy()
        df.index = pd.to_datetime(df.index)       
        forecast_size = len(df)
        n_feat = df.n_feat.values[0]
        df.drop('n_feat',axis=1,inplace=True)
        
        df = df.rename(columns = {'change':'Ennustettu kuukausimuutos (prosenttiyksiköä)',
                                  'month':'Kuukausi',
                                  'prev': 'Edellisen kuukauden ennuste',
                                  'Työttömyysaste': 'Työttömyysaste (ennuste)'})
        
        
        features = sorted(list(weights_dict.keys()))
        
        weights_df = pd.DataFrame([weights_dict]).T
        weights_df.index.name = 'Hyödyke'
        weights_df.columns = ['Oletettu keskimääräinen kuukausimuutos (%)']
        
        model_name = method_selection_results['model']
             
        pca = method_selection_results['pca']
        explained_variance = method_selection_results['explained_variance']
        hyperparam_grid = method_selection_results['hyperparam_grid']
        
        if pca:
            metadict = {
                        'Pääkomponenttianalyysi' : pca,
                        'Selitetty variaatio': str(int(100*explained_variance))+'%',
                        'Pääkomponetteja': n_feat,
                        'Malli': model_name,
                        'Sovellettuja hyödykeryhmiä':len(features),
                        'Sovelletut hyödykeryhmät' : ',\n'.join(features),
                        'Ennusteen pituus': str(forecast_size)+' kuukautta'
                        
                        }

        else:
            metadict = {
                            'Pääkomponenttianalyysi' : pca,
                            'Malli': model_name,
                            'Sovellettuja hyödykeryhmiä':len(features),
                            'Sovelletut hyödykeryhmät' : ',\n'.join(features),
                            'Ennusteen pituus': str(forecast_size)+' kuukautta'
                            }
        
        metadata = pd.DataFrame([metadict]).T
        metadata.index.name = ''
        metadata.columns = ['Arvo']
        
        hyperparam_df = pd.DataFrame([hyperparam_grid]).T
        hyperparam_df.index.name = 'Hyperparametri'
        hyperparam_df.columns = ['Arvo']   
        hyperparam_df['Arvo'] = hyperparam_df['Arvo'].astype(str)

  
        data_ = data.copy().rename(columns={'change':'Muutos (prosenttiykköä)',
                                      'prev':'Edellisen kuukauden työttömyys -% ',
                                      'month':'Kuukausi'})
        
        
        xlsx_io = io.BytesIO()
        writer = pd.ExcelWriter(xlsx_io, engine='xlsxwriter')
        
        
        
        data_.to_excel(writer, sheet_name= 'Data')
        df.to_excel(writer, sheet_name= 'Ennustedata')
        weights_df.to_excel(writer, sheet_name= 'Indeksimuutokset')
        hyperparam_df.to_excel(writer, sheet_name= 'Hyperparametrit')
        metadata.to_excel(writer, sheet_name= 'Metadata')


        writer.save()
        

        
        output = xlsx_io.getvalue()

        return dcc.send_bytes(output, 'Ennustedata {} hyödykkeellä '.format(len(features))+datetime.now().strftime('%d_%m_%Y')+'.xlsx')
        
@app.callback(
    Output("test_download", "data"),
    [Input("test_download_button", "n_clicks"),
    State('test_data','data'),
    State('method_selection_results','data'),
    State('change_weights','data')
    ]
    
)
def download_test_data(n_clicks, df, method_selection_results, weights_dict):
    
    if n_clicks > 0:
        
        df = pd.DataFrame(df).set_index('Aika').copy()
        df.index = pd.to_datetime(df.index)
        mape = df.mape.values[0]
        test_size = len(df)
        n_feat = df.n_feat.values[0]
        df.drop('n_feat',axis=1,inplace=True)
        df.drop('mape',axis=1,inplace=True)
        df = df.rename(columns = {'change':'Ennustettu kuukausimuutos (prosenttiyksiköä)',
                                  'month':'Kuukausi',
                                  'prev': 'Edellisen kuukauden ennuste'})
        
        
        features = sorted(list(weights_dict.keys()))
        
        model_name = method_selection_results['model']
             
        pca = method_selection_results['pca']
        explained_variance = method_selection_results['explained_variance']
        hyperparam_grid = method_selection_results['hyperparam_grid']
        
        if pca:
            metadict = {'MAPE': str(round(100*mape,2))+'%',
                        'Pääkomponenttianalyysi' : pca,
                        'Selitetty variaatio': str(int(100*explained_variance))+'%',
                        'Pääkomponetteja': n_feat,
                        'Malli': model_name,
                        'Sovellettuja hyödykeryhmiä':len(features),
                        'Sovelletut hyödykeryhmät' : ',\n'.join(features),
                        'Testin pituus': str(test_size)+' kuukautta'
                        }
        else:
            metadict = {'MAPE': str(round(100*mape,2))+'%',
                            'Pääkomponenttianalyysi' : pca,
                            'Malli': model_name,
                            'Sovellettuja hyödykeryhmiä':len(features),
                            'Sovelletut hyödykeryhmät' : ',\n'.join(features),
                            'Testin pituus': str(test_size)+' kuukautta'
                            }
        
        metadata = pd.DataFrame([metadict]).T
        metadata.index.name = ''
        metadata.columns = ['Arvo']
        
        hyperparam_df = pd.DataFrame([hyperparam_grid]).T
        hyperparam_df.index.name = 'Hyperparametri'
        hyperparam_df.columns = ['Arvo']   
        hyperparam_df['Arvo'] = hyperparam_df['Arvo'].astype(str)

        
        xlsx_io = io.BytesIO()
        writer = pd.ExcelWriter(xlsx_io, engine='xlsxwriter')
        
        
        df.to_excel(writer, sheet_name= 'Testidata')
        metadata.to_excel(writer, sheet_name= 'Metadata')
        hyperparam_df.to_excel(writer, sheet_name= 'Mallin hyperparametrit')

        writer.save()
        

        
        output = xlsx_io.getvalue()

        return dcc.send_bytes(output, 'Testitulokset {} hyödykkeellä '.format(len(features))+datetime.now().strftime('%d_%m_%Y')+'.xlsx')


@app.callback(

    Output('test_graph_div', 'children'),
    
      [Input('test_chart_type','value')],
      [State('test_data','data')]
    
)
def update_test_chart_type(chart_type,df):
    
    
    
    df = pd.DataFrame(df).set_index('Aika')
    df.index = pd.to_datetime(df.index)

    
    df = df.reset_index().drop_duplicates(subset='Aika', keep='last').set_index('Aika')[['Työttömyysaste','Ennuste','mape']].dropna(axis=0)

    return [dcc.Graph(id ='test_graph', 
                     figure = plot_test_results(df,chart_type), 
                     config = config_plots)     ]



@app.callback(

    Output('forecast_graph_div', 'children'),
    
      [Input('chart_type','value')],
      [State('forecast_data','data')]
    
)
def update_forecast_chart_type(chart_type,df):
    
    df = pd.DataFrame(df).set_index('Aika')
    df.index = pd.to_datetime(df.index)

    return dcc.Graph(id = 'forecast_graph',
                    figure = plot_forecast_data(df, chart_type = chart_type), 
                    config = config_plots),

@app.callback(

    Output('ev_placeholder', 'style'),
    [Input('pca_switch', 'on')]
)    
def add_ev_slider(pca):
    
    return {False: {'margin' : '5px 5px 5px 5px', 'display':'none'},
           True: {'margin' : '5px 5px 5px 5px'}}[pca]

@app.callback(

    Output('ev_slider_update', 'children'),
    [Input('pca_switch', 'on'),
    Input('ev_slider', 'value')]

)
def update_ev_indicator(pca, explained_variance):
    
    return {False: [html.Div([html.P('Valitsit {} % säilytetyn variaation.'.format(int(100*explained_variance)),
                                                               style = p_style)
                                                       ], style = {'display':'none'}
                                                      )],
            True: [html.Div([html.P('Valitsit {} % säilytetyn variaation.'.format(int(100*explained_variance)),
                                                               style = p_style)
                                                       ]
                                                      )]}[pca]



@app.callback(
    Output('feature_selection','value'),
    [Input('select_all', 'on'),
    Input('feature_selection','options')]
)
def update_feature_list(on,options):
       
        
    if on:
        return [f['label'] for f in options]
    else:
        raise PreventUpdate
        
@app.callback(
    
    Output('select_all','on'),
    [Input('feature_selection','value'),
     State('feature_selection','options')]
    
)
def update_select_all_on(features,options):
    
    return len(features) == len(options)

@app.callback(
    [
     # Output('select_all','on'),
     Output('select_all','label'),
     Output('select_all','disabled')
     ],
    [Input('select_all', 'on')]
)    
def update_switch(on):
    
    if on:
        return {'label':'Kaikki hyödykkeet on valittu. Voit poistaa hyödykkeitä listasta klikkaamalla rasteista.',
                       'style':{'text-align':'center', 'font-size':p_font_size, 'font-family':'Cadiz Semibold'}
                      },True
    
    # used_selection = selection_options[label]
    
    # selection_values = sorted(list(pd.DataFrame(used_selection).sort_values(by='label').label.values))
            
    # if sorted(features) == selection_values:
        
    #     return True, {'label':'Kaikki hyödykkeet on valittu. Voit poistaa hyödykkeitä listasta klikkaamalla rasteista.',
    #                    'style':{'text-align':'center', 'font-size':20, 'font-family':'Cadiz Semibold'}
    #                   },True
    else:
        return dict(label = 'Valitse kaikki',style = {'font-size':p_font_size, 'fontFamily':'Cadiz Semibold'}),False



@app.callback(

    Output('test_button_div','children'),
    [

     Input('features_values','data')
     ]    
    
)
def add_test_button(features_values):
    
    if features_values is None:
        raise PreventUpdate 
        
        
    
    elif len(features_values) == 0:
        return [html.P('Valitse ensin hyödykkeitä',
                       style = p_style)]
    
    else:
               
        
        return dbc.Button('Testaa',
                           id='test_button',
                           n_clicks=0,
                           outline=False,
                           className="me-1",
                           size='lg',
                           color='success',
                           style = dict(fontSize=30,fontFamily='Cadiz Semibold')
                          )

@app.callback(
    Output('test_size_indicator','children'),
    [Input('test_slider','value')]
)
def update_test_size_indicator(value):
    
    return [html.Br(),html.P('Valitsit {} kuukautta testidataksi.'.format(value),
                             style = p_style)]

@app.callback(
    Output('forecast_slider_indicator','children'),
    [Input('forecast_slider','value')]
)
def update_forecast_size_indicator(value):
    
    return [html.Br(),html.P('Valitsit {} kuukauden ennusteen.'.format(value),
                             style = p_style)]




@app.callback(

    Output('timeseries_selection', 'children'),
    [
     
     Input('features_values','data')
     ]    
    
)
def update_timeseries_selections(features_values):
    
    features = sorted(list(features_values.keys()))
    
    return [
            html.Br(),
            html.H3('Tarkastele hyödykkeiden indeksin aikasarjoja',
                    style =h3_style),
            
            html.P('Tällä kuvaajalla voit tarkastella hyödykkeiden indeksikehitystä kuukausittain.',
                   style = p_style),
            html.H3('Valitse hyödyke',style = h3_style),
            dcc.Dropdown(id = 'timeseries_selection_dd',
                        options = [{'value':feature, 'label':feature} for feature in features],
                        value = [features[0]],
                        style = {
                            'font-size':p_font_size, 
                            'font-family':'Cadiz Book',
                            'color': 'black'},
                        multi = True)
            ]


@app.callback(

    Output('timeseries', 'children'),
    [Input('timeseries_selection_dd', 'value')]    
    
)
def update_time_series(values):
    
    traces = [go.Scatter(x = data.index, 
                         y = data[value],
                         showlegend=True,                         
                         name = ' '.join(value.split()[1:]),
                         mode = 'lines+markers') for value in values]
    return html.Div([dcc.Graph(figure=go.Figure(data=traces,
                                      layout = go.Layout(title = dict(text = 'Valittujen arvojen<br>indeksikehitys',
                                                                      x=.5,
                                                                      font=dict(
                                                                          family='Cadiz Semibold',
                                                                           size=20
                                                                          )),
                                                         
                                                         height=graph_height,
                                                         template = 'seaborn',
                                                         margin=dict(
                                                              l=10,
                                                             r=10,
                                                             # b=100,
                                                              # t=120,
                                                              # pad=4
                                                         ),
                                                         legend=dict(
                                                              orientation = 'h',
                                                                      # x=.1,
                                                                      # y=1,
                                                                      xanchor='center',
                                                                      yanchor='top',
                                                                     font=dict(
                                                              size=12,
                                                             family='Cadiz Book')),
                                                         xaxis = dict(title=dict(text = 'Aika',
                                                                                 font=dict(
                                                                                     size=18, 
                                                                                     family = 'Cadiz Semibold')),
                                                                      automargin=True,
                                                                      tickfont = dict(
                                                                          family = 'Cadiz Semibold', 
                                                                           size = 16
                                                                          )),
                                                         yaxis = dict(title=dict(text = 'Pisteluku (perusvuosi = 2010)',
                                                                                font=dict(
                                                                                     size=18, 
                                                                                    family = 'Cadiz Semibold')),
                                                                     tickformat = ' ',
                                                                     automargin=True,
                                                                     tickfont = dict(
                                                                         family = 'Cadiz Semibold', 
                                                                          size = 16
                                                                         ))
                                                         )
                                      ),
                     config = config_plots
                     )])




@app.callback(
    
    Output('forecast_button_div','children'),
    [
     
     Input('features_values','data')
     ]   
    
)
def add_predict_button(features_values):
    
    if features_values is None:
        raise PreventUpdate 
    
    elif len(features_values) == 0:
        return [html.P('Valitse ensin hyödykkeitä',
                       style = p_style)]
    
    
        
    else:
        return [dbc.Button('Ennusta',
                   id='forecast_button',
                   n_clicks=0,
                   outline=False,
                   className="me-1",
                   size='lg',
                   color='success',
                   style = dict(fontSize=30,fontFamily='Cadiz Semibold')
                   
                   ),
                html.Br(),
                html.Div(id = 'forecast_download_button_div',style={'textAlign':'center'})]





@app.callback(

    Output('slider_prompt_div','children'),
    [Input('slider', 'value'),
     State('averaging', 'on')]    
    
)
def update_slider_prompt(value, averaging):
    
        
    if averaging:
    
        return [html.Br(),html.P('Valitsit {} viimeisen kuukauden keskiarvot.'.format(value),
                      style = p_style),
                html.Br(),
                html.P('Voit vielä säätä yksittäisiä muutosarvoja laatikoihin kirjoittamalla tai tietokoneella työskenneltäessä laatioiden oikealla olevista nuolista.',
                       style = p_style)]
    else:
        return [html.Br(),html.P('Valitsit {} % keskimääräisen kuukausimuutoksen.'.format(value),
                      style = p_style),
                html.Br(),
                html.P('Voit vielä säätä yksittäisiä muutosarvoja laatikoihin kirjoittamalla tai oikealla olevista nuolista.',
                       style = p_style)]
        
 

@app.callback(

    Output('corr_selection_div', 'children'),
    [

     Input('features_values','data')
     ]
    
)
def update_corr_selection(features_values):
    
    features = sorted(list(features_values.keys()))

    return html.Div([
            html.Br(),
            html.H3('Tarkastele valitun hyödykkeen hintaindeksin suhdetta työttömyysasteeseen',
                    style=h3_style),
            
            html.P('Tällä kuvaajalla voit tarkastella valitun hyödykkeen hintaindeksin ja työttömyysasteen välistä korrelaatiota. Ennusteita tehtäessä on syytä valita piirteitä, jotka korreloivat vahvasti työttömyysasteen kanssa.',
                   style = p_style),
        html.H3('Valitse hyödyke', style = h3_style),
        dcc.Dropdown(id = 'corr_feature',
                        multi = True,
                        # clearable=False,
                        options = [{'value':feature, 'label':feature} for feature in features],
                        value = [features[0]],
                        style = {'font-size':20, 'font-family':'Cadiz Book'},
                        placeholder = 'Valitse hyödyke')
        ]
        )

@app.callback(

    Output('feature_corr_selection_div', 'children'),
    [

     Input('features_values','data')
     ]
    
)
def update_feature_corr_selection(features_values):
    
    features = sorted(list(features_values.keys()))
    
    return html.Div([
                html.Br(),
                html.H3('Tarkastele hyödykkeiden suhteita',
                        style=h3_style),
                html.Br(),
                html.P('Tällä kuvaajalla voit tarkastella hyödykkeiden välisiä korrelaatioita. Ennusteita tehtäessä on syytä valita piirteitä, jotka korreloivat heikosti keskenään.',
                       style = p_style),
        
        dbc.Row(justify = 'center',children=[
            dbc.Col([
                html.H3('Valitse hyödyke',style=h3_style),
                dcc.Dropdown(id = 'f_corr1',
                                multi = False,
                                options = [{'value':feature, 'label':feature} for feature in features],
                                value = features[0],
                                style = {'font-size':20, 'font-family':'Cadiz Book'},
                                placeholder = 'Valitse hyödyke')
        ],xs =12, sm=12, md=12, lg=6, xl=6),
        html.Br(),
            dbc.Col([
                html.H3('Valitse toinen hyödyke',
                        style=h3_style
                        ),
                dcc.Dropdown(id = 'f_corr2',
                                multi = False,
                                options = [{'value':feature, 'label':feature} for feature in features],
                                value = features[-1],
                                style = {'font-size':20, 'font-family':'Cadiz Book'},
                                placeholder = 'Valitse hyödyke')
            ],xs =12, sm=12, md=12, lg=6, xl=6)
        ])
        ])



@app.callback(

    Output('corr_div', 'children'),
    [Input('f_corr1','value'),
     Input('f_corr2','value')]    
    
)
def update_feature_correlation_plot(value1, value2):
    
    if value1 is None or value2 is None:
        raise PreventUpdate 
        
        
    a, b = np.polyfit(np.log(data[value1]), data[value2], 1)

    y = a * np.log(data[value1]) +b 

    df = data.copy()
    df['log_trend'] = y
    df = df.sort_values(by = 'log_trend')    
    
    corr_factor = round(sorted(data[[value1,value2]].corr().values[0])[0],2)
    
    traces = [go.Scatter(x = data[value1], 
                         y = data[value2], 
                         mode = 'markers',
                         name = ' ',#.join(value.split()[1:]),
                         showlegend=False,
                         marker = dict(color = 'purple', size = 10),
                         marker_symbol='star',
                         hovertemplate = "<b>Indeksiarvot:</b><br><b>{}</b>:".format(' '.join(value1.split()[1:]))+" %{x}"+"<br><b>"+"{}".format(' '.join(value2.split()[1:]))+"</b>: %{y}"
                         ),
                go.Scatter(x = df[value1], 
                            y = df['log_trend'], 
                            name = 'Logaritminen trendiviiva', 
                            mode = 'lines',
                            line = dict(width=5),                            
                            showlegend=True,
                            hovertemplate=[], 
                            marker = dict(color = 'orange'))
             ]
    
  
    
    return [
            html.Div([dcc.Graph(figure = go.Figure(data = traces,
          layout = go.Layout(title = dict(text = '<b>{}</b> vs.<br><b>{}</b><br>(Kokonaiskorrelaatio: {})'.format(' '.join(value1.split()[1:]), ' '.join(value2.split()[1:]), corr_factor), 
                                          x=.5, 
                                          font=dict(
                                              family='Cadiz Semibold',
                                               size=22
                                              )),
                             margin=dict(
                                  l=10,
                                 r=10,
                                 # b=100,
                                  # t=120,
                                  # pad=4
                             ),
                            xaxis= dict(title = dict(text='{} (pisteluku)'.format(' '.join(value1.split()[1:])), 
                                                     font=dict(
                                                         family='Cadiz Semibold',
                                                          size=20
                                                         )),
                                        automargin=True,
                                        tickfont = dict(
                                            family = 'Cadiz Semibold', 
                                             size = 18
                                            )),
                            height = graph_height,
                            legend = dict(font=dict(
                                 size=16,
                                family='Cadiz Book'),
                                          orientation='h'),
                            hoverlabel = dict(
                                 font_size = 20, 
                                font_family = 'Cadiz Book'),
                            template = 'seaborn',
                            yaxis = dict(title = dict(text='{} (pisteluku)'.format(' '.join(value2.split()[1:])), 
                                                      font=dict(
                                                          family='Cadiz Semibold',
                                                          size=20
                                                          )),
                                         automargin=True,
                                         tickfont = dict(
                                             family = 'Cadiz Semibold', 
                                             size = 18
                                             ))
                             )
          ),
                      config = config_plots)])]



@app.callback(

    Output('eda_div', 'children'),
    [Input('corr_feature','value')]    
    
)
def update_eda_plot(values):
        
        
    symbols = ['circle',
                 'square',
                 'diamond',
                 'cross',
                 'x',
                 'pentagon',
                 'hexagon',
                 'hexagon2',
                 'octagon',
                 'star',
                 'hexagram',
                 'hourglass',
                 'bowtie',
                 'asterisk',
                 'hash']
    
    
            
    traces = [go.Scatter(x = data[value], 
                         y = data['Työttömyysaste'], 
                         mode = 'markers',
                         name = ' '.join(value.split()[1:]).replace(',',',<br>')+' ({})'.format(round(sorted(data[['Työttömyysaste', value]].corr()[value].values)[0],1)),
                         showlegend=True,
                         marker = dict(size=10),
                         marker_symbol = symbols[random.randint(0,len(symbols)-1)],
                         hovertemplate = "<b>{}</b>:".format(value)+" %{x}"+"<br><b>Työttömyysaste</b>: %{y}"+" %"+"<br>(Korrelaatio: {:.2f})".format(sorted(data[['Työttömyysaste', value]].corr()[value].values)[0])) for value in values]

        
        
  
    
    return [dcc.Graph(figure = go.Figure(data = traces,
          layout = go.Layout(title = dict(text = 'Valitut hyödykkeet vs.<br>Työttömyysaste', 
                                          x=.5, 
                                          font=dict(
                                              family='Cadiz Semibold',
                                               size=22
                                              )
                                          ),
                            xaxis= dict(title = dict(text='Hyödykkeiden pisteluku', 
                                                     font=dict(
                                                         family='Cadiz Semibold',
                                                          size=20
                                                         )),
                                        automargin=True,
                                        tickfont = dict(
                                            family = 'Cadiz Semibold', 
                                             size = 18
                                            )
                                        ),
                            margin=dict(
                                 l=10,
                                r=10,
                                # b=100,
                                 # t=120,
                                 # pad=4
                            ),
                            height = graph_height,
                            legend = dict(
                                 orientation = 'h',
                                         # x=.1,
                                         # y=1,
                                         xanchor='center',
                                         yanchor='top',
                                        font=dict(
                                 size=16,
                                family='Cadiz Book')),
                            hoverlabel = dict(
                                 font_size = 20, 
                                font_family = 'Cadiz Book'),
                            template = 'seaborn',                            
                            yaxis = dict(title = dict(text='Työttömyysaste (%)', 
                                                      font=dict(
                                                          family='Cadiz Semibold',
                                                           size=20
                                                          )
                                                      ),
                                         automargin=True,
                                         tickfont = dict(
                                             family = 'Cadiz Semibold', 
                                              size = 18
                                             )
                                         )
                             )
          ),config = config_plots)]

@app.callback(
    [
     
     Output('feature_selection', 'options'),
     Output('sorting', 'label'),
     Output('feature_selection', 'value')
     
     ],
    [Input('alphabet', 'n_clicks'),
     Input('corr_desc', 'n_clicks'),
     Input('corr_asc', 'n_clicks'),
     Input('corr_abs_desc', 'n_clicks'),
     Input('corr_abs_asc', 'n_clicks'),
     Input('main_class', 'n_clicks'),
     Input('second_class', 'n_clicks'),
     Input('third_class', 'n_clicks'),
     Input('fourth_class', 'n_clicks'),
     Input('fifth_class', 'n_clicks')
    ]
)
def update_selections(*args):
    
    ctx = callback_context
    
    
    if not ctx.triggered:
        return corr_abs_asc_options, "Absoluuttinen korrelaatio (laskeva)",[f['value'] for f in corr_abs_asc_options[:4]]
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if button_id == 'alphabet':
        return feature_options, "Aakkosjärjestyksessä",[f['value'] for f in feature_options[:4]]
    elif button_id == 'corr_desc':
        return corr_desc_options, "Korrelaatio (laskeva)",[f['value'] for f in corr_desc_options[:4]]
    elif button_id == 'corr_asc':
        return corr_asc_options, "Korrelaatio (nouseva)",[f['value'] for f in corr_asc_options[:4]]
    elif button_id == 'corr_abs_desc':
        return corr_abs_desc_options, "Absoluuttinen korrelaatio (laskeva)",[f['value'] for f in corr_abs_desc_options[:4]]
    elif button_id == 'corr_abs_asc':
        return corr_abs_asc_options, "Absoluuttinen korrelaatio (nouseva)",[f['value'] for f in corr_abs_asc_options[:4]]
    elif button_id == 'main_class':
        return main_class_options, "Pääluokittain",[f['value'] for f in main_class_options[:4]]
    elif button_id == 'second_class':
        return second_class_options, "2. luokka",[f['value'] for f in second_class_options[:4]]
    elif button_id == 'third_class':
        return third_class_options, "3. luokka",[f['value'] for f in third_class_options[:4]]
    elif button_id == 'fourth_class':
        return fourth_class_options, "4. luokka",[f['value'] for f in fourth_class_options[:4]]
    else:
        return fifth_class_options, "5. luokka",[f['value'] for f in fifth_class_options[:4]]
    
@app.callback(
    Output("offcanvas", "is_open"),
    Input("open-offcanvas", "n_clicks"),
    [State("offcanvas", "is_open")],
)
def toggle_offcanvas(n1, is_open):
    if n1:
        return not is_open
    return is_open

app.layout = serve_layout

# Käynnistä sovellus.
if __name__ == "__main__":
    app.run_server(debug=in_dev, threaded = True)