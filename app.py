# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 11:57:56 2022

@author: tuomas.poukkula
"""

import pandas as pd
import numpy as np
import requests
import plotly.graph_objs as go
#from plotly.io.json import to_json_plotly
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.svm import SVR, LinearSVR
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.decomposition import PCA
# from dash import dcc
# from dash import html
from dash import dcc, html, ALL, MATCH
import dash_daq
from flask import Flask
import os
import base64
import io
import dash
from dash_extensions.enrich import Dash,ServersideOutput, Output, Input, State
from dash.exceptions import PreventUpdate
# import plotly.express as px
#import orjson
import random
import dash_bootstrap_components as dbc
from sklearn.metrics import silhouette_score
import time
from datetime import datetime
import io
import json
import locale

# riippu ollaanko Windows vai Linux -ympäristössä, mitä locale-koodausta käytetään.

try:
    locale.setlocale(locale.LC_ALL, 'fi_FI')
except:
    locale.setlocale(locale.LC_ALL, 'fi-FI')

in_dev = True

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
UNWANTED_PARAMS = ['verbose','cache_size', 'max_iter', 'warm_start', 'max_features','tol','subsample']

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

p_font_size = 18
graph_height = 800

external_stylesheets = [dbc.themes.SUPERHERO,
                        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css",
                        'https://codepen.io/chriddyp/pen/brPBPO.css'
                       ]


server = Flask(__name__)
server.secret_key = os.environ.get('secret_key','secret')
app = Dash(name = __name__, 
           prevent_initial_callbacks = False, 
           server = server,
           external_scripts = ["https://raw.githubusercontent.com/plotly/plotly.js/master/dist/plotly-locale-fi.js",
                               "https://cdn.plot.ly/plotly-locale-fi-latest.js"],
           # meta_tags = [{'name':'viewport',
           #              'content':'width=device-width, initial_scale=1.0, maximum_scale=1.2, minimum_scale=0.5'}],
           external_stylesheets = external_stylesheets
          )
app.scripts.config.serve_locally = False
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
                            textfont=dict(family='Arial', size = 16),
                            showlegend=False,
                            hovertemplate=hovertemplate,
                            marker = dict(color = 'red', symbol='diamond', size = 10)),
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
                                          title = dict(text='Työttömyysaste (%)', font=dict(size=16, family = 'Arial Black')), 
                                          tickformat = ' ', 
                                          tickfont = dict(size=16, family = 'Arial Black')), 
                               yaxis=dict(showspikes=True,
                                          title = dict(text='Inflaatio (%)', font=dict(size=16, family = 'Arial Black')),
                                          tickformat = ' ', 
                                          tickfont = dict(size=14,family = 'Arial Black')),
                               height= graph_height,
                               template='seaborn',  
                               autosize=True,
                               hoverlabel = dict(font_size = 14, font_family = 'Arial'),
                                legend = dict(font=dict(size=14),
                                               orientation='h',
                                               # xanchor='center',
                                               # yanchor='top',
                                               # x=.85,
                                               # y=.99
                                              ),
                              
                               title = dict(text = 'Työttömyysaste vs.<br>Inflaatio<br>{} - {}<br>'.format(df.index.min().strftime('%B %Y'),df.index.max().strftime('%B %Y')),x=.5,font=dict(size=20,family = 'Arial Black'))
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
    
    past_df['mape'] = mape
    
    return past_df

def plot_test_results(df, chart_type = 'lines+bars'):
    
    mape = round(100*df.mape.values[0],1)
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
                               textfont = dict(family='Arial Black', size = 14,color='green'), 
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
                           textfont = dict(family='Arial Black', size = 14,color='white')
                           )
                    ],layout=go.Layout(xaxis = dict(title = dict(text='Aika',font=dict(size=14, family = 'Arial Black')),
                                                    tickfont = dict(family = 'Arial Black', size = 14)
                                                    ),
                                       yaxis = dict(title = dict(text='Työttömyysaste (%)',font=dict(family='Arial Black',size=18)),
                                                    tickfont = dict(family = 'Arial Black', size = 16)
                                                    ),
                                       height = graph_height-300,
                                       legend = dict(font=dict(size=18),
                                                      # orientation='h',
                                                      # xanchor='center',
                                                      # yanchor='top',
                                                      # x=.08,
                                                      # y=1.2
                                                     ),
                                       hoverlabel = dict(font_size = 14, font_family = 'Arial'),
                                       template = 'seaborn',
                                      # margin=dict(autoexpand=True),
                                       title = dict(text = 'Työttömyysasteen ennuste kuukausittain<br>(MAPE: {} %)'.format(mape),
                                                    x=.5,
                                                    font=dict(family='Arial Black',size=20)
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
                                textfont = dict(family='Arial Black', size = 14,color='green'), 
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
                            textfont = dict(family='Arial Black', size = 14,color='red'))
                    ],layout=go.Layout(xaxis = dict(title = dict(text='Aika',font=dict(size=14, family = 'Arial Black')),
                                                    tickfont = dict(family = 'Arial Black', size = 14)
                                                    ),
                                        yaxis = dict(title = dict(text='Työttömyysaste (%)',font=dict(family='Arial Black',size=18)),
                                                    tickfont = dict(family = 'Arial Black', size = 16)
                                                    ),
                                        height = graph_height-300,
                                        legend = dict(font=dict(size=18),
                                                      # orientation='h',
                                                      # xanchor='center',
                                                      # yanchor='top',
                                                      # x=.08,
                                                      # y=1.2
                                                      ),
                                        hoverlabel = dict(font_size = 14, font_family = 'Arial'),
                                        template = 'seaborn',
                                      # margin=dict(autoexpand=True),
                                        title = dict(text = 'Työttömyysasteen ennuste kuukausittain<br>(MAPE: {} %)'.format(mape),
                                                    x=.5,
                                                    font=dict(family='Arial Black',size=20)
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
                           textfont = dict(family='Arial Black', size = 14,color='white')
                                    ),
                        
                        go.Bar(x=df.index.strftime('%B %Y'), 
                                y = df.Ennuste, 
                                name = 'Ennuste',
                           showlegend=True, 
                           marker = dict(color='red'), 
                           text=[str(round(c,2))+' %' for c in df.Ennuste], 
                           textposition='inside',
                           hovertemplate = hovertemplate,
                           textfont = dict(family='Arial Black', size = 14,color='white')
                                )
                        ],layout=go.Layout(xaxis = dict(title = dict(text='Aika',font=dict(size=14, family = 'Arial Black')),
                                                        tickfont = dict(family = 'Arial Black', size = 14)
                                                        ),
                                            yaxis = dict(title = dict(text='Työttömyysaste (%)',font=dict(family='Arial Black',size=18)),
                                                        tickfont = dict(family = 'Arial Black', size = 16)
                                                        ),
                                            height = graph_height-300,
                                            legend = dict(font=dict(size=18),
                                                          # orientation='h',
                                                          # xanchor='center',
                                                          # yanchor='top',
                                                          # x=.08,
                                                          # y=1.2
                                                          ),
                                            hoverlabel = dict(font_size = 14, font_family = 'Arial'),
                                            template = 'seaborn',
                                          # margin=dict(autoexpand=True),
                                            title = dict(text = 'Työttömyysasteen ennuste kuukausittain<br>(MAPE: {} %)'.format(mape),
                                                        x=.5,
                                                        font=dict(family='Arial Black',size=20)
                                                        )
                                            )
                                                        )                                                   

                                                    
                                                    
                                                    
def plot_forecast_data(df, chart_type):
    

    if chart_type == 'lines':
    
    
        return go.Figure(data=[go.Scatter(x=data.index, 
                                          y = data.Työttömyysaste, 
                                          name = 'Toteutunut',
                                          showlegend=True,
                                          mode="lines", 
                                          hovertemplate = '<b>%{x}</b>: %{y}%',
                                          marker = dict(color='green')),
                    go.Scatter(x=df.index, 
                               y = df.Työttömyysaste, 
                               name = 'Ennuste',
                               showlegend=True,
                               mode="lines", 
                               hovertemplate = '<b>%{x}</b>: %{y}%',
                               marker = dict(color='red'))
                    ],layout=go.Layout(xaxis = dict(title = dict(text = 'Aika',font=dict(size=18, family = 'Arial Black')),
                                                    tickfont = dict(family = 'Arial Black', size = 16),
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
                                       template='seaborn',
                                       hoverlabel = dict(font_size = 16, font_family = 'Arial'),
                                       legend = dict(font=dict(size=12)),
                                       yaxis = dict(title=dict(text = 'Työttömyysaste (%)',
                                                     font=dict(size=18, family = 'Arial Black')),
                                                     tickfont = dict(family = 'Arial', size = 16)),
                                       title = dict(text = 'Työttömyysaste ja ennuste kuukausittain<br>{} - {}'.format(data.index.strftime('%B %Y').values[0],df.index.strftime('%B %Y').values[-1]),
                                                    x=.5,
                                                    font=dict(family='Arial Black',size=20)),
    
                                       ))


    else:
        
        
      
        return go.Figure(data=[go.Bar(x=data.index, 
                                          y = data.Työttömyysaste, 
                                          name = 'Toteutunut',
                                          showlegend=True,
                                          # mode="lines", 
                                          hovertemplate = '<b>%{x}</b>: %{y}%',
                                          marker = dict(color='green')),
                    go.Bar(x=df.index, 
                               y = df.Työttömyysaste, 
                               name = 'Ennuste',
                               showlegend=True,
                               # mode="lines", 
                               hovertemplate = '<b>%{x}</b>: %{y}%',
                               marker = dict(color='red'))
                    ],layout=go.Layout(xaxis = dict(title = dict(text = 'Aika',font=dict(size=18, family = 'Arial Black')),
                                                    tickfont = dict(family = 'Arial Black', size = 16),
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
                                       template='seaborn',
                                       hoverlabel = dict(font_size = 16, font_family = 'Arial'),
                                       legend = dict(font=dict(size=12)),
                                       yaxis = dict(title=dict(text = 'Työttömyysaste (%)',
                                                     font=dict(size=18, family = 'Arial Black')),
                                                     tickfont = dict(family = 'Arial', size = 16)),
                                       title = dict(text = 'Työttömyysaste ja ennuste kuukausittain<br>{} - {}'.format(data.index.strftime('%B %Y').values[0],df.index.strftime('%B %Y').values[-1]),
                                                    x=.5,
                                                    font=dict(family='Arial Black',size=20)),
    
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

  X = scl.fit_transform(x)

  pca = PCA(n_components = n_components, random_state = 42, svd_solver = 'full')

  if use_pca:
    
    X = pca.fit_transform(X)

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

feature_options = [{'label':c, 'value':c} for c in data.columns[1:-4]]
corr_desc_options = [{'label':c, 'value':c} for c in correlations_desc.index]
corr_asc_options = [{'label':c, 'value':c} for c in correlations_asc.index]
corr_abs_desc_options = [{'label':c, 'value':c} for c in correlations_abs_desc.index]
corr_abs_asc_options = [{'label':c, 'value':c} for c in correlations_abs_asc.index]
main_class_options = [{'label':c, 'value':c} for c in main_classes]
second_class_options = [{'label':c, 'value':c} for c in second_classes]
third_class_options = [{'label':c, 'value':c} for c in third_classes]
fourth_class_options = [{'label':c, 'value':c} for c in fourth_classes]

initial_options = corr_abs_desc_options
initial_features = [[list(f.values())[0] for f in corr_abs_desc_options][i] for i in random.sample(range(len(corr_abs_desc_options)),6)]


def serve_layout():
    
    return html.Div(children=[
        
        html.Br(),
        html.H1('Phillipsin vinouma',style={'textAlign':'center', 'font-size':60}),
        html.Br(),
        html.P('Valitse haluamasi välilehti alla olevia otsikoita klikkaamalla.',
               style = {
                   'text-align':'center',
                    'font-style': 'italic', 
                   'font-family':'Arial', 
                    'font-size':p_font_size
                   }),
        html.Br(),
        
        dbc.Tabs([
            
            
            
            dbc.Tab(label='Ohje ja esittely',
                    tabClassName="flex-grow-1 text-center",
                    tab_style = {'font-size':28},
                    style = {
                            
                            "maxHeight": "1000px",

                            "overflow": "auto"
                        },
                    children = [
                       # html.Div([
                        dbc.Row(justify='center',
                                style = {'margin' : '10px 10px 10px 10px'},
                                children=[
                            
                          dbc.Col(xs =12, sm=12, md=12, lg=9, xl=9, children =[
                         
                                 html.Br(),
                                 html.H6('Taloustieteen kymmenes perusperiaate:',style = {'text-align':'center', 'font-family':'Arial Black','font-style': 'italic', 'font-weight': 'bold', 'font-size':20}),
                                 html.P('Työttömyyden ja inflaation kesken vallitsee lyhyellä ajalla ristiriita. Täystyöllisyyttä ja vakaata hintatasoa on vaikea saavuttaa yhtä aikaa.', 
                                        style = {
                                            'text-align':'center',
                                            'font-style': 'italic', 
                                            'font-family':'Arial', 
                                             'font-size':p_font_size
                                            }),
                                 html.P('(Matti Pohjola, 2019, Taloustieteen oppikirja, s. 250, ISBN:978-952-63-5298-5)', 
                                        style={
                                            'textAlign':'center',
                                            'font-family':'Arial', 
                                             'font-size':p_font_size-4
                                            }),
                                 html.Br(),
                                 html.H4('Esittely',style={'textAlign':'center','font-family':'Arial Black'}),
                                 html.Br(),
                                 html.P('Tällä työkalulla voi vapaasti kehitellä koneoppimismenetelmän, joka pyrkii ennustamaan tulevien kuukausien työttömyysastetta ottaen huomioon aiemmat toteutuneet työttömyysasteet, kuukausittaiset työttömyyden kausivaihtelut sekä käyttäjän itse valitsemien hyödykkeiden kuluttajahintaindeksit. Ennuste tehdään sillä oletuksella, että valittujen hyödykkeiden hintataso muuttuu käyttäjän syöttämän oletetun keskimääräisen kuukausimuutoksen mukaan. Lisäksi koneoppimismenetelmän voi valita ja algoritmien hyperparametrit voi säätää itse. Menetelmää voi testata valitulle aikavälille ennen ennusteen tekemistä. Hintaindeksien valintaa helpottaakseen, voi niiden keskinäisiä suhteita ja suhdetta työttömyysasteeseen tutkia sille varatulla välilehdellä. ', 
                                        style={
                                            'textAlign':'center',
                                            'font-family':'Arial', 
                                             'font-size':p_font_size
                                            }),
                                 html.P('Sovellus hyödyntää Tilastokeskuksen Statfin-rajapinnasta saatavaa työttömyys, -ja kuluttajahintaindeksidataa. Sovellus on avoimen lähdekoodin kehitysprojekti, joka on saatavissa kokonaisuudessaan tekijän Github-sivun kautta. Sovellus on kehitetty harrasteprojektina tutkivaan ja kokeelliseen käyttöön, ja siitä saataviin tuloksiin on suhtauduttava varauksella (ks. Vastuunvapautuslauseke alla).', 
                                        style={
                                            'textAlign':'center',
                                            'font-family':'Arial', 
                                             'font-size':p_font_size
                                            }),
                                 html.H4('Teoria',style={'textAlign':'center','font-family':'Arial Black'}),
                                 html.Br(),
                                 html.P('Inflaatio, eli yleisen hintatason nousu on vaikuttanut viimeisen vuoden aikana monen kotitalouden kulutustottumuksiin. Hintavakauden saavuttamiseksi, keskuspankkien on pyrittävä kiristämään rahapolitiikkaa ohjauskorkoa nostamalla, mikä tullee nostamaan asuntovelallisten korkokuluja. Vähemmän varallisuutta ohjautuu osakemarkkinoille, mikä ajaa pörssikursseja alaspäin. Samalla työmarkkinoilla on painetta palkankorotuksille, mikä voi pahimmillaan eskaloitua lakkoiluun. Vaikuttaisi siis siltä, että inflaatiolla on vain negatiivisia vaikutuksia, mutta näin ei aina ole. Nimittäin historian aika on havaittu makrotaloustieteellinen ilmiö, jonka mukaan lyhyellä aikavälillä inflaation ja työttömyyden välillä vallitsee ristiriita. Toisin sanoen korkean inflaation koetaan lyhyellä aikavälillä matalaa työttömyyttä. Selittäviä teorioita tälle ilmiölle on useita. Lyhyellä aikavälillä hintojen noustessa tuotanto nousee, koska tuottajat nostavat hyödykkeiden tuotantoa suurempien katteiden saavuttamiseksi. Tämä johtaa matalampaan työttömyyteen, koska tuotannon kasvattaminen johtaa uusiin rekrytointeihin, jolloin työttömien osuus työvoimasta pienenee. Toisin päin katsottuna, kun työttömyys on matala, markkinoilla on työn kysyntäpainetta, mikä nostaa palkkoja. Palkkojen nousu taas johtaa yleisen hintatason nousuun, koska hyödykkeiden tarjoajat voivat pyytää korkeampaa hintaa tuotteistaan ja palveluistaan. Elämme nyt sellaista hetkeä, jossa inflaatio viimeisen noin kymmenen vuoden huipussaan. Samalla työttömyys on paljon matalampi kuin se oli esimerkiksi vuonna 2015, jolloin inflaatio oli välillä jopa negatiivinen. Ajoittain voivat molemmat olla samaan aikaan matalia tai korkeita, mutta yleisesti ottaen lyhyellä ajalla toisen noustessa toisen trendikäyrä laskee. Tämä tunnetaan makrotaloustieteessä ns. Phillipsin käyränä, joka on sen vuonna 1958 kehittäneen taloustieteilijä Alban William Phillipsin mukaan.',
                                        style={
                                            'text-align':'center',
                                            'font-family':'Arial', 
                                             'font-size':p_font_size
                                            }),
                                 html.Br()
                                 ])
                                 ]
                                ),
                                 dbc.Row(
                                             
                                             [
                                            dbc.Col([

                                             html.H3('Phillipsin käyrä Suomen taloudessa kuukausittain (Lähde: Tilastokeskus)', style={'textAlign':'center','font-family':'Arial Black'}),
                                             html.Br(),
                                             html.Br(),
                                             html.Div(
                                                 [dcc.Graph(figure= draw_phillips_curve(),
                                                       config = config_plots
                                                       )
                                                  ]
                                                      ),
                                             html.Br(),
                                             html.Br(),
                                             html.P('Yllä olevassa kuvaajassa on esitetty sirontakuviolla työttömyysaste ja inflaatio eri aikoina. Lisäksi siinä on nimetty viimeisin ajankohta, jolta on saatavissa sekä inflaatio että työttömyyslukema. Viemällä hiiren pisteiden päälle, näkee arvot sekä ajan. Logaritminen trendiviiva esittää aikoinaan Phillipsin tekemää empiiristä havaintoa, jossa inflaation ja työttömyysasteen välillä vallitsee negatiivinen korrelaatio. Kuvaajassa inflaation trendikäyrästä laskee työttömyyden kasvaessa.' ,
                                                    style = {
                                                        'text-align':'center', 
                                                        'font-family':'Arial', 
                                                         'font-size':p_font_size-2
                                                        }),
                                            ],xs =12, sm=12, md=12, lg=9, xl=9)
                                 ], justify = 'center', 
                                      style = {'margin' : '5px 5px 5px 5px'}
                                    ),
                                 html.Br(),
                                 dbc.Row([
                                     
                                     dbc.Col([
                                     
                                         html.P('Kyseessä on tunnettu taloustieteen teoria, jota on tosin vaikea soveltaa, koska ei ole olemassa sääntöjä, joilla voitaisiin helposti ennustaa työttömyyttä saatavilla olevien inflaatiota kuvaavien indikaattorien avulla. Mikäli sääntöjä on vaikea formuloida, niin niitä voi yrittää koneoppimisen avulla oppia historiadataa havainnoimalla. Voisiko siis olla olemassa tilastollisen oppimisen menetelmä, joka pystyisi oppimaan Phillipsin käyrän historiadatasta? Mikäli tämänlainen menetelmä olisi olemassa, pystyisimme ennustamaan lyhyen aikavälin työttömyyttä, kun oletamme hyödykkeiden hintatason muuttuvan skenaariomme mukaisesti.',
                                                style={
                                                    'textAlign':'center',
                                                    'font-family':'Arial', 
                                                     'font-size':p_font_size
                                                    }), 
                                         html.Br(),
        
                                         html.Br(),
                                         html.P('Tällä työkalulla on mahdollista rakentaa koneoppimisen ratkaisu, joka, perustuen Phillipsin teoriaan, pyrkii ennustamaan työttömyyttä valittujen hyödykkeiden hintatason olettamien avulla. Sovellus jakaantuu ennustajapiirteiden valintaan, tutkivaan analyysiin, menetelmän suunnitteluun, toteutuksen testaamiseen sekä lyhyen aikavälin ennusteen tekemiseen.',
                                                style={
                                                    'textAlign':'center',
                                                    'font-family':'Arial', 
                                                     'font-size':p_font_size
                                                    }), 
                                         html.P('Inflaatiota mitataan kuluttajahintaindeksin vuosimuutoksen avulla. Kuluttajahintaindeksi on Tilastokeskuksen koostama indikaattori, joka mittaa satoja hyödykkeitä sisältävän hyödykekorin hinnan muutosta johonkin perusvuoden (tässä tapauksessa vuoden 2010) hintatasoon nähden. Tässä sovelluksessa voi valita kuluttajahintaindeksin komponentteja eli hyödykeryhmien indeksien pistelukuja työttömyyden ennustamiseen.',
                                                style={
                                                    'textAlign':'center',
                                                    'font-family':'Arial', 
                                                     'font-size':p_font_size
                                                    }), 
                                         html.P('Tutkivan analyysin avulla voi arvioida mitkä piirteet sopisivat parhaiten ennusteen selittäjiksi. Tutkivassa analyysissa tarkastellaan muuttujien välisiä korrelaatiota. Tässä sovelluksessa voi tehdä korrelaatioperusteisen ennusteen. Pääsääntönä on valita ennustajiksi niitä hyödykeryhmiä, jotka korreloivat eniten työttömyyden kanssa sekä vähiten keskenään. Myös koneoppimisalgoritmin valinta ja sen uniikkien hyperparametrien säätäminen vaikutta tulokseen. Testin avulla saa informaatiota siitä, miten valittu menetelmä olisi onnistunut ennustamaan työttömyyttä aikaisempina ajankohtina.',
                                                style={
                                                    'textAlign':'center',
                                                    'font-family':'Arial', 
                                                     'font-size':p_font_size
                                                    }),
                                         html.Br(),
   
                                         html.H4('Sovelluksen käyttöhje',style={'textAlign':'center','font-family':'Arial Black'}),
                                         html.Br(),
                                         html.P('Seuraavaksi hieman ohjeistusta sovelluksen käyttöön. Jokainen osio olisi tehtävä vastaavassa järjestyksessä. Välilehteä valitsemalla pääset suorittamaan jokaisen vaiheen. Välilehdillä on vielä yksityiskohtaisemmat ohjeet.', 
                                                style = {
                                                    'text-align':'center', 
                                                    'font-family':'Arial', 
                                                     'font-size':p_font_size
                                                    }),
                                         html.Br(),
                                         html.P('1. Piirteiden valinta. Valitse haluamasi hyödykeryhmät alasvetovalikosta. Näiden avulla ennustetaan työttömyyttä Phillipsin teorian mukaisesti.', 
                                                style = {
                                                    'text-align':'center', 
                                                    'font-family':'Arial', 
                                                     'font-size':p_font_size
                                                    }),
                                         html.P('2. Tutkiva analyysi. Voit tarkastella ja analysoida valitsemiasi piirteitä. Voit tarvittaessa palata edelliseen vaiheeseen ja poistaa tai lisätä piirteitä.',
                                                style = {
                                                    'text-align':'center', 
                                                    'font-family':'Arial', 
                                                     'font-size':p_font_size
                                                    }),
                                         html.P('3. Menetelmän valinta. Tässä osiossa valitsen koneoppimisalgoritmin sekä säädät hyperparametrit. Lisäksi voi valita hyödynnetäänkö pääkomponenttianalyysiä ja kuinka paljon variaatiota säilötään.',
                                                style = {
                                                    'text-align':'center', 
                                                    'font-family':'Arial', 
                                                     'font-size':p_font_size
                                                    }),
                                         html.P('4. Testaaminen. Voit valita menneen ajanjakson, jota malli pyrkii ennustamaan. Näin pystyt arvioimaan kuinka ennustemalli olisi toiminut jo toteutuneelle datalle.',
                                                style = {
                                                    'text-align':'center', 
                                                    'font-family':'Arial', 
                                                     'font-size':p_font_size
                                                    }),
                                         html.P('5. Ennusteen tekeminen. ',
                                                style = {
                                                    'text-align':'center', 
                                                    'font-family':'Arial', 
                                                    'font-size':p_font_size
                                                    }),
                                         # html.P('6. Valitse se osuus alkuperäisen datan variaatiosta, joka vähintään säilytetään PCA:ssa.',style = {'text-align':'center', 'font-family':'Arial Black', 'font-size':20}),
                                         # html.P('7. Klusteroi klikkaamalla "Klusteroi" -painiketta.',style = {'text-align':'center', 'font-family':'Arial Black', 'font-size':20}),
                                          html.H4('Pääkomponenttianalyysistä',style={'textAlign':'center','font-family':'Arial Black'}),
                                          html.Br(),
                                          html.P('Pääkomponenttianalyysilla (englanniksi Principal Component Analysis, PCA) pyritään minimoimaan käytettyjen muuttujien määrää pakkaamalla ne sellaisiin kokonaisuuksiin, jotta hyödynnetty informaatio säilyy. Informaation säilyvyyttä mitataan selitetyllä varianssilla (eng. explained variance), joka tarkoittaa uusista pääkomponenteista luodun datan hajonnan säilyvyyttä alkuperäiseen dataan verrattuna. Tässä sovelluksessa selitetyn varianssin (tai säilytetyn variaation) osuuden voi valita itse, mikäli hyödyntää PCA:ta. Näin saatu pääkomponenttijoukko on siten pienin sellainen joukko, joka säilyttää vähintään valitun osuuden alkuperäisen datan hajonnasta. Näin PCA-algoritmi muodostaa juuri niin monta pääkomponenttia, jotta selitetyn varianssin osuus pysyy haluttuna.',
                                                 style={'textAlign':'center','font-family':'Arial', 'font-size':p_font_size}),
                                          html.P('PCA on yleisesti hyödyllinen toimenpide silloin, kun valittuja muuttujia on paljon, milloin on myös mahdollista, että osa valituista muuttujista aiheuttaa datassa kohinaa, mikä taas johtaa heikompaan ennusteeseen.  Pienellä määrällä tarkasti harkittuja muuttujia PCA ei ole välttämätön.',
                                                 style={'textAlign':'center','font-family':'Arial', 'font-size':p_font_size}),
                                          html.Br(),
                                       
                                          html.Br(),
                                          html.H4('Vastuuvapauslauseke',style={'textAlign':'center','font-family':'Arial Black'}),
                                          html.Br(),
                                          html.P("Sivun ja sen sisältö tarjotaan ilmaiseksi sekä sellaisena kuin se on saatavilla. Kyseessä on yksityishenkilön tarjoama palvelu eikä viranomaispalvelu. Sivulta saatavan informaation hyödyntäminen on päätöksiä tekevien tahojen omalla vastuulla. Palvelun tarjoaja ei ole vastuussa menetyksistä, oikeudenkäynneistä, vaateista, kanteista, vaatimuksista, tai kustannuksista taikka vahingosta, olivat ne mitä tahansa tai aiheutuivat ne sitten miten tahansa, jotka johtuvat joko suoraan tai välillisesti yhteydestä palvelun käytöstä. Huomioi, että tämä sivu on yhä kehityksen alla.",
                                                 style={
                                                     'textAlign':'center',
                                                     'font-family':'Arial', 
                                                      'font-size':p_font_size
                                                     }),
                                          html.Br(),
                                          html.H4('Tuetut selaimet',style={'textAlign':'center','font-family':'Arial Black'}),
                                          html.Br(),
                                          html.P("Sovellus on testattu toimivaksi Google Chromella ja Mozilla Firefoxilla. Edge- ja Internet Explorer -selaimissa sovellus ei toimi. Opera, Safari -ja muita selaimia ei ole testattu.",
                                                 style={
                                                     'textAlign':'center',
                                                     'font-family':'Arial', 
                                                      'font-size':p_font_size
                                                     }),
                                          html.Br(),
                                         html.Div(style={'text-align':'center'},children = [
                                             html.H4('Lähteet', style = {'text-align':'center','font-family':'Arial Black'}),
                                              html.Br(),
                                              html.Label(['Tilastokeskus: ', 
                                                       html.A('Työvoimatutkimuksen tärkeimmät tunnusluvut, niiden kausitasoitetut aikasarjat sekä kausi- ja satunnaisvaihtelusta tasoitetut trendit', href = "https://statfin.stat.fi/PxWeb/pxweb/fi/StatFin/StatFin__tyti/statfin_tyti_pxt_135z.px/",target="_blank")
                                                      ],style={'textAlign':'center','font-family':'Arial', 'font-size':20}),
                                              html.Br(),
                                              html.Label(['Tilastokeskus: ', 
                                                       html.A('Kuluttajahintaindeksi (2010=100)', href = "https://statfin.stat.fi/PxWeb/pxweb/fi/StatFin/StatFin__khi/statfin_khi_pxt_11xd.px/",target="_blank")
                                                      ],style={'textAlign':'center','font-family':'Arial', 'font-size':20}),
                                              html.Br(),

                                              html.Label(['Wikipedia: ', 
                                                       html.A('Phillipsin käyrä', href = "https://fi.wikipedia.org/wiki/Phillipsin_k%C3%A4yr%C3%A4",target="_blank")
                                                      ],style={'textAlign':'center','font-family':'Arial', 'font-size':20}),
                                              html.Br(),
                                              html.Label(['Wikipedia: ', 
                                                       html.A('Pääkomponenttianalyysi', href = "https://fi.wikipedia.org/wiki/P%C3%A4%C3%A4komponenttianalyysi",target="_blank")
                                                      ],style={'textAlign':'center','font-family':'Arial', 'font-size':20}),
                                              html.Br(),
                                              html.Label(['Wikipedia: ', 
                                                       html.A('Pearsonin korrelaatiokerroin (englanniksi)', href = "https://en.wikipedia.org/wiki/Pearson_correlation_coefficient",target="_blank")
                                                      ],style={'textAlign':'center','font-family':'Arial', 'font-size':20}),
                                              html.Br(),
                                               html.Label(['Scikit-learn: ', 
                                                        html.A('Regressiotekniikat', href = "https://scikit-learn.org/stable/supervised_learning.html#supervised-learning",target="_blank")
                                                       ],style={'textAlign':'center','font-family':'Arial', 'font-size':20}),
                                               html.Br(),
    
                                          ]),
                                         html.Br(),
                                         html.Br(),
                                         html.H4('Tekijä', style = {'text-align':'center','font-family':'Arial Black'}),
                                         html.Br(),
                                         html.Div(style = {'textAlign':'center'},children = [
                                              html.I('Tuomas Poukkula', style = {'textAlign':'center','font-family':'Arial', 'font-size':20}),
                                         
                                              html.Br()
                                              ]),
                  
                                   
                                         ],xs =12, sm=12, md=12, lg=9, xl=9)
                                     ],justify ='center',
                                     style={'margin': '5px 5px 5px 5px'}
                                     ),
                                  dbc.Row([
                                     

                                      dbc.Col([
                                          html.Div([
                                              html.A([
                                                 html.Img(
                                                     src=app.get_asset_url('256px-Linkedin_icon.png'),
                                                    
                                                     style={
                                                           'height' : '50px',
                                                           'width' : '50px',
                                                           'text-align':'right',
                                                            'float' : 'right',
                                                            'position' : 'center',
                                                            'padding-top' : 0,
                                                            'padding-right' : 0
                                                     }
                                                     )
                                         ], href='https://www.linkedin.com/in/tuomaspoukkula/',target = '_blank', style = {'textAlign':'center'})
                                         ],style={'textAlign':'center'})],xs =6, sm=6, md=6, lg=6, xl=6),
                                      dbc.Col([
                                          html.Div([
                                              html.A([
                                                 html.Img(
                                                     src=app.get_asset_url('Twitter-logo.png'),
                                                     style={
                                                          'height' : '50px',
                                                          'width' : '50px',
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
                                      style={'margin': '5px 5px 5px 5px'}
                                      ),
                                 dbc.Row([
                                     
                                     dbc.Col([
                                         html.Div(style = {'text-align':'center'},children = [
                                             html.Br(),
                                             html.Label(['Sovellus ', 
                                                      html.A('GitHub:ssa', href='https://github.com/tuopouk/suomenavainalueet')
                                                     ],style={'textAlign':'center','font-family':'Arial', 'font-size':20})
                                     ])
                                         ],xs =12, sm=12, md=12, lg=6, xl=6)
                                     ],
                                     justify ='center',
                                     style={'margin': '5px 5px 5px 5px'}
                                     )
                 
                        
                        
                        
                        ]

),
            
            dbc.Tab(label ='Piirteiden valinta',
                    tabClassName="flex-grow-1 text-center",
                    tab_style = {'font-size':28},
                    style = {
                        #"position": "fixed",
                        "maxHeight": "1000px",
                          # "height":"1400px",
                    
                        "overflow": "auto"
                    },
              children= [
        
                dbc.Row(children = [
                        
                        html.Br(),
                    
                       dbc.Col(children=[
                           
                            html.Br(),
                            html.P('Tässä osiossa valitaan hyödykkeitä, joita käytetään työttömyyden ennustamisessa.',
                                   style = {'text-align':'center',
                                            'font-family':'Arial',
                                            'font-size':p_font_size}),
                            html.P('Voit valita alla olevasta valikosta hyödykkeitä, minkä jälkeen voit säätää niiden oletettavaa kuukausimuutosta syöttämällä lukeman alle ilmestyviin laatikkoihin.',
                                   style = {'text-align':'center',
                                            'font-family':'Arial',
                                            'font-size':p_font_size}),
                            html.P('Voit myös säätää kaikille hyödykkeille saman kuukausimuutoksen tai hyödyntää toteutuneiden kuukausimuutosten keskiarvoja.',
                                   style = {'text-align':'center',
                                            'font-family':'Arial',
                                            'font-size':p_font_size}),
                            html.P('Hyödykevalikon voi rajata tai lajitella sen yllä olevasta alasvetovalikosta. Valittavanasi on joko aakkosjärjestys, korrelaatiojärjestykset (Pearsonin korrelaatiokertoimen mukaan) tai rajaus Tilastokeskuksen hyödykehierarkian mukaan.',
                                   style = {'text-align':'center',
                                            'font-family':'Arial',
                                            'font-size':p_font_size}),

                            html.Br(),
                            html.H3('Valitse ennustepiirteiksi hyödykeryhmiä valikosta',
                                    style={'textAlign':'center',
                                           'font-family':'Arial Black'}),
                            html.Br(),
                            dbc.DropdownMenu(id = 'sorting',
                                             #align_end=True,
                                             children = [
                                                 
                                                 dbc.DropdownMenuItem("Aakkosjärjestyksessä", id = 'alphabet'),
                                                 dbc.DropdownMenuItem("Korrelaatio (laskeva)", id = 'corr_desc'),
                                                 dbc.DropdownMenuItem("Korrelaatio (nouseva)", id = 'corr_asc'),
                                                 dbc.DropdownMenuItem("Absoluuttinen korrelaatio (laskeva)", id = 'corr_abs_desc'),
                                                 dbc.DropdownMenuItem("Absoluuttinen korrelaatio (nouseva)", id = 'corr_abs_asc'),
                                                 dbc.DropdownMenuItem("Pääluokittain", id = 'main_class'),
                                                 dbc.DropdownMenuItem("2. luokka", id = 'second_class'),
                                                 dbc.DropdownMenuItem("3. luokka", id = 'third_class'),
                                                 dbc.DropdownMenuItem("4. luokka", id = 'fourth_class'),
                                                 
                                                 
                                                 ],
                                            label = "Absoluuttinen korrelaatio (laskeva)",
                 
                                            style={
                                                'font-size':22, 
                                                'font-family':'Arial',
                                                'color': 'black'}
                                            ),
                            
                            html.Br(),
                            dcc.Dropdown(id = 'feature_selection',
                                         options = initial_options,
                                         multi = True,
                                         value = list(initial_features),
                                         style = {'font-size':16, 'font-family':'Arial','color': 'black'},
                                         placeholder = 'Valitse hyödykkeitä.'),
                            html.Br(),
                            
                            dbc.Alert("Valitse ainakin yksi hyödykeryhmä valikosta!", color="danger",
                                      dismissable=True, fade = True, is_open=False, id = 'alert', 
                                      style = {'text-align':'center', 'font-size':18, 'font-family':'Arial Black'}),
                            dash_daq.BooleanSwitch(id = 'select_all', 
                                                   label = dict(label = 'Valitse kaikki',style = {'font-size':20, 'fontFamily':'Arial Black'}), 
                                                   on = False, 
                                                   color = 'blue'),
                            html.Br(),
                            html.Div(id = 'selections_div'),
                            
                            
                            
                            html.Div([
                                dcc.Store(id = 'used_selection',data=initial_options),
                                dcc.Store(id = 'selected_features',data=initial_features),
                                dcc.Store(id='test_data'),
                                dcc.Store(id = 'forecast_data'),
                                dcc.Download(id='forecast_download'),
                                dcc.Download(id='test_download')
                            ]),
                            
                            
                            ],xs =12, sm=12, md=12, lg=12, xl=12
                        )
                        ],justify='center', style = {'margin' : '10px 10px 10px 10px'}
                    )
                ]
                ),


            dbc.Tab(label = 'Tutkiva analyysi',
                    tabClassName="flex-grow-1 text-center",
                    tab_style = {'font-size':28},
                    style = {
                            
                            "maxHeight": "1000px",

                            "overflow": "auto"
                        },
                    children = [
                    
                    dbc.Row([html.Br(),
                             html.Br(),
                             html.P('Tässä osiossa työttömyysastetta sekä valittujen kuluttajahintaindeksin hyödykeryhmien keskinäistä suhdetta sekä muutosta ajassa. Alla voit nähdä kuinka eri hyödykeryhmien hintaindeksit korreloivat keskenään sekä työttömyysasteen kanssa. Voit myös havainnoida indeksien, inflaation sekä sekä työttömyysasteen aikasarjoja. Kuvattu korrelaatio perustuu Pearsonin korrelaatiokertoimeen.',
                                    style = {'font-family':'Arial',
                                              'font-size':p_font_size, 
                                             'text-align':'center'}),
                             html.Br()],
                            justify = 'center', style = {'textAlign':'center','margin':'10px 10px 10px 10px'}),
                    
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
                        ],justify='center', style = {'margin' : '5px 5px 5px 5px'}
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
                                html.H3('Alla olevassa kuvaajassa on esitetty inflaatio ja työttömyys Suomessa kuukausittain.',
                                       style = {'font-family':'Arial', 'text-aling':'center'}),
                                dcc.Graph(figure = go.Figure(data=[go.Scatter(x = data.index,
                                                                              y = data.Työttömyysaste,
                                                                              name = 'Työttömyysaste',
                                                                              mode = 'lines',
                                                                              marker = dict(color ='red')),
                                                                   go.Scatter(x = data.index,
                                                                              y = data.Inflaatio,
                                                                              name = 'Inflaatio',
                                                                              mode ='lines',
                                                                              marker = dict(color = 'purple'))],
                                                             layout = go.Layout(title = dict(text = 'Työttömyysaste ja inflaatio kuukausittain<br>{} - {}'.format(data.index.strftime('%B %Y').values[0],data.index.strftime('%B %Y').values[-1]),x=.5,font=dict(family='Arial Black',size=20)),
                                                                                height=graph_height,
                                                                                legend = dict(font=dict(size=12,family='Arial')),
                                                                                xaxis = dict(title=dict(text = 'Aika',font=dict(size=18, family = 'Arial Black')), 
                                                                                             tickfont = dict(family = 'Arial Black', size = 16),
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
                                                                                                       font=dict(size=18, family = 'Arial Black')),
                                                                                            tickformat = ' ',
                                                                                            tickfont = dict(family = 'Arial Black', size = 16)
                                                                                            )
                                                                                
                                                                                )
                                                                                ),
                                          config=config_plots
                                          )
                                ])
                                
                            
                            ], xs =12, sm=12, md=12, lg=6, xl=6, align ='end')
                        
                        ],
                        justify='center', style = {'margin' : '5px 5px 5px 5px'})
                
                    ]
                ),
            dbc.Tab(label='Menetelmän valinta',
                    tabClassName="flex-grow-1 text-center",
                    tab_style = {'font-size':28},
                    style = {
                            
                            "maxHeight": "1000px",

                            "overflow": "auto"
                        },
                    
                    children = [
                        dbc.Row([
                            html.Br(),
                            html.P('Tässä osiossa voit valita koneoppimisalgoritmin, säätää sen hypeparametrit sekä halutessasi hyödyntää pääkomponenttianalyysia valitsemallasi tavalla.',
                                   style = {'text-align':'center',
                                            'font-family':'Arial',
                                            'font-size':p_font_size}),
                            html.P('Valitse ensin algoritmi, minkä jälkeen alle ilmestyy sille ominaiset hyperparametrit, joita voit säätää sopiviksi. Hyperparametrit luetaan dynaamisesti suoraan Scikit-learn -kirjaston dokumentaatiosta, eikä niille siksi ole suomenkielistä käännöstä. Säätövalikoiden alta löytyy linkki valitun algoritmin dokumentaatiosivulle, jossa aiheesta voi lukea enemmän. Hyperparametrien säädölle ei ole yhtä ainutta tapaa, vaan eri arvoja on testattava iteratiivisesti.',
                                   style = {'text-align':'center',
                                            'font-family':'Arial',
                                            'font-size':p_font_size}),
                            html.Br(),
                            html.P('Lisäksi voit valita hyödynnetäänkö pääkomponenttianalyysiä piirteiden karsimiseksi. Pääkompomponenttianalyysi on tilastollis-tekninen kohinanpoistomenetelmä, jolla pyritään parantamaan ennusteen laatua. Siinä valituista piirteistä muodostetaan lineaarikombinaatioita siten, että alkuperäisessä datassa oleva variaatio säilyy tietyn suhdeluvun verran muunnetussa aineistossa. Variaatio voi säätää haluamakseen. Kuten hyperparametrien tapauksessa, on tämäkin määrittely puhtaasti empiirinen.',
                                   style = {'text-align':'center',
                                            'font-family':'Arial',
                                            'font-size':p_font_size})
                        ], justify = 'center', style = {'textAlign':'center','margin':'10px 10px 10px 10px'}),
                        html.Br(),
                        dbc.Row([
                            dbc.Col(id = 'model_selection', children = [
                                
                                html.H3('Valitse algoritmi.',style={'textAlign':'center'}),
                                html.Br(),
                                dcc.Dropdown(id = 'model_selector',
                                             value = 'Satunnaismetsä',
                                             multi = False,
                                             placeholder = 'Valitse algoritmi',
                                             style = {'font-size':16, 'font-family':'Arial','color': 'black'},
                                             options = [{'label': c, 'value': c} for c in MODELS.keys()]),
                                
                                html.Br(),
                                html.H3('Säädä hyperparametrit.', style = {'textAlign':'center'}),
                                html.Br(),
                                html.Div(id = 'hyperparameters_div')
                                
                                ], xs =12, sm=12, md=12, lg=8, xl=8),
                            dbc.Col(id = 'pca_selections', children = [
                                html.Br(),
                                dash_daq.BooleanSwitch(id = 'pca_switch', 
                                                                 label = dict(label = 'Käytä pääkomponenttianalyysia',style = {'font-size':20, 'fontFamily':'Arial Black','textAlign':'center'}), 
                                                                  on = False, 
                                                                  color = 'blue'),
                                html.Br(),
                                
                                
                                html.Div(id = 'ev_placeholder',children =[
                                    html.H4('Valitse säilytettävä variaatio', style = {'textAlign':'center'}),
                                    html.Br(),
                                    dcc.Slider(id = 'ev_slider',
                                       min = .7, 
                                       max = .99, 
                                       value = .95, 
                                       step = .01,
                                       tooltip={"placement": "top", "always_visible": True},
                                        marks = {
                                                 .7: {'label':'70%', 'style':{'font-size':20, 'fontFamily':'Arial Black','color':'white'}},
                                            .85: {'label':'85%', 'style':{'font-size':20, 'fontFamily':'Arial Black','color':'white'}},
                                                 .99: {'label':'99%', 'style':{'font-size':20, 'fontFamily':'Arial Black','color':'white'}}

                                                }
                                      ),
                                    html.Br(),
                                 html.Div(id = 'ev_slider_update', 
                                          children = [
                                              html.Div([html.P('Valitsit {} % säilytetyn variaation.'.format(95),
                                                               style = {'text-align':'center', 
                                                                         'font-size':20, 
                                                                        'font-family':'Arial Black'})
                                                       ], style = {'display':'none'}
                                                      )
                                          ]
                                        )
                                 ]
                                    )
                                
                                ],xs =12, sm=12, md=12, lg=4, xl=4)
                        ],justify='left', style = {'margin' : '5px 5px 5px 5px'}),
                        html.Br(),
                        
                        
                        ]),
            dbc.Tab(label='Testaaminen',
                    tabClassName="flex-grow-1 text-center",
                    tab_style = {'font-size':28},
                    style = {
                            
                            "maxHeight": "1200px",

                            "overflow": "auto"
                        },
                    children = [
                        html.Br(),
                        dbc.Row([
                            
                                       html.H4('Menetelmän testaaminen', style = {'textAlign':'center',
                                                                                     'font-family':'Arial Black'}),
                                       html.Br(),
                                       html.P('Tässä osiossa voit testata kuinka hyvin valittu menetelmä olisi onnistunut ennustamaan menneiden kuukausien työttömyysasteen hyödyntäen valittuja piirteitä. Testattaessa valittu määrä kuukausia jätetään testidataksi, jota menetelmä pyrkii ennustamaan.',style = {
                                            'font-size':p_font_size, 
                                           'font-family':'Arial',
                                           'text-align':'center'}),
                                       html.P('Tässä kohtaa hyödykeindeksien oletetaan toteutuvan sellaisinaan.',style = {
                                            'font-size':p_font_size, 
                                           'font-family':'Arial',
                                           'text-align':'center'}),
                                       html.P('Tehtyäsi testin voit tarkastella viereistä tuloskuvaajaa tai viedä testidatan alle ilmestyvästä painikeesta Exceliin.',
                                              style={'text-align':'center',
                                                     'font-family':'Arial',
                                                     'font-size':p_font_size})
                            
                            
                            
                            ], justify = 'center', style = {'textAlign':'center', 'margin':'10px 10px 10px 10p'}),
                        html.Br(),
                        dbc.Row(children = [
                            
                            dbc.Col(children = [
                                       html.Br(),

                                       html.Br(),
                                       html.H3('Valitse testidatan pituus.',style = {'textAlign':'center',
                                                                                     'font-family':'Arial Black'}),
                                       html.Br(),
                                       dcc.Slider(id = 'test_slider',
                                                 min = 1,
                                                 max = 12,
                                                 value = 3,
                                                 step = 1,
                                                 tooltip={"placement": "top", "always_visible": True},
                                                 
                                                  marks = {1: {'label':'kuukausi', 'style':{'font-size':18, 'fontFamily':'Arial Black','color':'white'}},
                                                          # 3:{'label':'3 kuukautta', 'style':{'font-size':20, 'fontFamily':'Arial Black','color':'white'}},
                                                        
                                                           6:{'label':'puoli vuotta', 'style':{'font-size':18, 'fontFamily':'Arial Black','color':'white'}},
                                                         #  9:{'label':'yhdeksän kuukautta', 'style':{'font-size':20, 'fontFamily':'Arial Black','color':'white'}},
                                                         
                                                          12:{'label':'vuosi', 'style':{'font-size':18, 'fontFamily':'Arial Black','color':'white'}},
                                                          

                                                           }
                                                 ),
                                      html.Br(),  
                                      html.Div(id = 'test_size_indicator', style = {'textAlign':'center'}),
                                      html.Br(),
                                      html.Div(id = 'test_button_div',children = [html.P('Valitse ensin piirteitä.',style = {
                                          'text-align':'center', 
                                          'font-family':'Arial Black', 
                                           'font-size':p_font_size
                                          })], style = {'textAlign':'center'}),
                                      html.Br(),
                                      html.Div(id='test_download_button_div', style={'textAlign':'center'})
                                      
                                      ],xs = 12, sm = 12, md = 12, lg = 4, xl = 4
                                ),
                            dbc.Col([html.Div(id = 'test_results_div')],xs = 12, sm = 12, md = 12, lg = 8, xl = 8, align='start')
                            ], justify = 'center', style = {'margin' : '10px 10px 10px 10px'}
                            ),
                 
                        
                        
                        
                        ]),
            dbc.Tab(label='Ennustaminen',
                    tabClassName="flex-grow-1 text-center",
                    tab_style = {'font-size':28},
                    style = {
                            
                            "maxHeight": "1000px",

                            "overflow": "auto"
                        },
                    children = [
                        html.Br(),
                        dbc.Row([
                            
                                       html.H4('Ennusteen tekeminen',style={'textAlign':'center',
                                                                            'font-family':'Arial Black'}),
                                       html.Br(),
                                       html.P('Tässä osiossa voit tehdä ennusteen valitulle ajalle. Ennustettaessa on käytössä Menetelmän valinta -välilehdellä tehdyt asetukset. Ennusteen tekemisessä hyödynnetään Piirteiden valinta -välilehdellä tehtyjä oletuksia hyödykkeiden suhteellisesta hintakehityksestä.',
                                              style={'text-align':'center',
                                                     'font-family':'Arial',
                                                     'font-size':p_font_size}),
                                       html.P('Tehtyäsi ennusteen voit tarkastella viereistä ennusteen kuvaajaa tai viedä tulosdatan alle ilmestyvästä painikeesta Exceliin.',
                                              style={'text-align':'center',
                                                     'font-family':'Arial',
                                                     'font-size':p_font_size}),
                                       html.Br(),
                            
                            
                            ], justify = 'center', style = {'textAlign':'center','margin':'10px 10px 10px 10px'}),
                        html.Br(),
                        dbc.Row(children = [
                                    #dbc.Col(xs =12, sm=12, md=12, lg=3, xl=3, align = 'start'),
                                    dbc.Col(children = [
                                       html.Br(),

                                       html.H3('Valitse ennusteen pituus.',style={'textAlign':'center', 'font-family':'Arial Black'}),
                                       dcc.Slider(id = 'forecast_slider',
                                                 min = 2,
                                                 max = 12,
                                                 value = 3,
                                                 step = 1,
                                                 tooltip={"placement": "top", "always_visible": True},
                                                 marks = {2: {'label':'2 kuukautta', 'style':{'font-size':16, 'fontFamily':'Arial Black','color':'white'}},
                                                         # 3: {'label':'kolme kuukautta', 'style':{'font-size':20, 'fontFamily':'Arial Black','color':'white'}},
                                                         6:{'label':'puoli vuotta', 'style':{'font-size':16, 'fontFamily':'Arial Black','color':'white'}},
                                                         # 9:{'label':'yhdeksän kuukautta', 'style':{'font-size':20, 'fontFamily':'Arial Black','color':'white'}},
                                                         12:{'label':'vuosi', 'style':{'font-size':16, 'fontFamily':'Arial Black','color':'white'}},
                                                       #  24:{'label':'kaksi vuotta', 'style':{'font-size':20, 'fontFamily':'Arial Black','color':'white'}}
                                                        
                                                     

                                                          }
                                                 ),
                                       html.Br(),
                                       html.Div(id = 'forecast_slider_indicator',style = {'textAlign':'center'}),
                                       html.Div(id = 'forecast_button_div',children = [html.P('Valitse ensin piirteitä.',
                                                                                              style = {
                                                                                                  'text-align':'center',
                                                                                                   'font-family':'Arial Black', 
                                                                                                    'font-size':p_font_size
                                                                                                    }
                                                                                              )],style = {'textAlign':'center'})],
                                        xs =12, sm=12, md=12, lg=4, xl=4
                                        ),
                                    html.Br(),
                                    
                                    dbc.Col([dcc.Loading(id = 'forecast_results_div',type = spinners[random.randint(0,len(spinners)-1)])],
                                            xs = 12, sm = 12, md = 12, lg = 8, xl = 8)
                                    ], justify = 'center', style = {'margin' : '10px 10px 10px 10px'}
                                    )
                                       
                            
                            
                            
                            
                                        # ],justify = 'left',style = {'margin' : '5px 5px 5px 5px'}
                                        # )
                            
                        ]
                            
                            )


        ]
            
    )
    
   ]
  )

@app.callback(

    Output('alert', 'is_open'),
    [Input('selected_features','data')]

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
                children.append(dbc.Col([html.H6(hyperparameter+':', style={'text-align':'left','font-size':15,'font-family':'Arial'})],xs =12, sm=12, md=12, lg=12, xl=12))
                children.append(html.Br())
                children.append(dbc.Col([dcc.Slider(id = {'index':hyperparameter, 'type':'hyperparameter_tuner'},
                                   value = value,
                                   max = 10*(value+1),
                                   min = value,
                                   marks=None,
                                   tooltip={"placement": "bottom", "always_visible": True},
                                   step = 1)
                                         ],xs =12, sm=12, md=12, lg=12, xl=12)
                                )
                
            elif type(value) == float:
                children.append(dbc.Col([html.H6(hyperparameter+':', style={'text-align':'left','font-size':15,'font-family':'Arial'})],xs =12, sm=12, md=12, lg=2, xl=2)),
                children.append(html.Br())
                children.append(dbc.Col([dbc.Input(id = {'index':hyperparameter, 'type':'hyperparameter_tuner'},
                                                   min = 0,
                                                   type = 'number',
                                                   #label = hyperparameter,
                                                   value = value,
                                                   step=0.01)],xs =12, sm=12, md=12, lg=2, xl=2)
                                )
                children.append(html.Br())
                
            
            elif type(value) == bool:
                children.append(dbc.Col([html.H6(hyperparameter+':', style={'text-align':'left','font-size':15,'font-family':'Arial'})],xs =12, sm=12, md=12, lg=2, xl=2)),
                children.append(html.Br())
                children.append(dbc.Col([dbc.Switch(id = {'index':hyperparameter, 'type':'hyperparameter_tuner'}, 
                                          style={'font-size':30},
                                          #label = hyperparameter,
                                          value=value,
                                          )],xs =12, sm=12, md=12, lg=2, xl=2)
                                )
                children.append(html.Br())
                
            elif type(value) == str:
                    
                    if type(param_options[hyperparameter]) == list:
                        children.append(dbc.Col([html.H6(hyperparameter+':', style={'text-align':'left','font-size':15,'font-family':'Arial'})],xs =12, sm=12, md=12, lg=2, xl=2)),
                        children.append(html.Br())
                        children.append(dbc.Col([dcc.Dropdown(id = {'index':hyperparameter, 'type':'hyperparameter_tuner'}, 
                                                  multi = False,
                                                  #label = hyperparameter,
                                                  style = {'font-family':'Arial','color': 'black'},
                                                  options = [{'label':c, 'value': c} for c in param_options[hyperparameter] if c not in ['precomputed','poisson']],
                                                  value = value)],xs =12, sm=12, md=12, lg=2, xl=2)
                                    )
                        children.append(html.Br())
                       
    # children.append(html.Br())   
    children.append(html.Br()) 
    children.append(html.Div(style = {'textAlign':'center'},
             children = [html.A('Katso dokumentaatio.', href = MODELS[model_name]['doc'], target="_blank",style = {'textAlign':'center','font-family':'Arial', 'font-size':20})]))
    return dbc.Row(children, justify ='start')


@app.callback(
    
      [Output('test_data','data'),
       Output('test_results_div','children'),
       Output('test_download_button_div','children')],
      [Input('test_button','n_clicks')],
      [State('test_slider', 'value'),
       State('model_selector','value'),
       # State('feature_selection','value'),
       State('selected_features','data'),
       #State('weights','data'),
       State({'type': 'hyperparameter_tuner', 'index': ALL}, 'id'),
       State({'type': 'hyperparameter_tuner', 'index': ALL}, 'value'),
       #State('forecast_slider','value'),
       State('pca_switch','on'),
       State('ev_slider','value')
       ]
    
)
def update_test_results(n_clicks, 
                        test_size, 
                        model_name, 
                        features, 
                        hyperparams,
                        hyperparam_values, 
                        pca, 
                        explained_variance):
    
    if n_clicks > 0:
    
        df = data.iloc[:len(data)-test_size,:].copy()
        
        model = MODELS[model_name]['model']
        # hyperparameters = MODELS[model_name]['tunable_hyperparameters']
        constants = MODELS[model_name]['constant_hyperparameters'] 
        
        
        hyperparam_grid  = {hyperparams[i]['index']:hyperparam_values[i] for i in range(len(hyperparams))}
        
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

        
        test_plot = html.Div(
                
                [   html.Br(),
                    dbc.Row([html.Br(),
                             html.H2('Testin tulokset', style = {'textAlign':'center', 'family':'Arial Black'}),
                             html.Br(),
                             html.P('Alla olevassa kuvaajassa nähdään kuinka hyvin ennustemalli olisi ennustanut työttömyysasteen ajalle {} - {}.'.format(test_result.index.strftime('%B %Y').values[0],test_result.index.strftime('%B %Y').values[-1]),
                                    style = {
                                        'text-align':'center', 
                                        'font-family':'Arial', 
                                         'font-size':p_font_size
                                        }),
                              html.Div([html.Br(),dbc.RadioItems(id = 'test_chart_type', 
                                          options = [{'label':'pylväät','value':'bars'},
                                                    {'label':'viivat','value':'lines'},
                                                    {'label':'viivat ja pylväät','value':'lines+bars'}],
                                          labelStyle={'display':'inline-block', 'padding':'10px','font-size':18},
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
                              dcc.Loading(id ='test_graph_div',type = spinners[random.randint(0,len(spinners)-1)])
                             # dcc.Graph(id ='test_graph', 
                             #           figure = plot_test_results(test_result), 
                             #           config = config_plots)
                             ]),
                    html.Br(),
                    
                    dbc.Row(
                        
                        [
                            dbc.Col([html.H4('MAPE (%)', style = {'textAlign':'center'}),
                                     dash_daq.LEDDisplay(backgroundColor='black',
                                                         size =50,
                                                         color = led_color,
                                                         style = {'textAlign':'center'},
                                                         value = round(100*mape,1))
                                     ],
                                    xs =12, sm=12, md=12, lg=6, xl=6
                                    ),
                            
                            dbc.Col([html.H4('Tarkkuus (%)', style = {'textAlign':'center'}),
                                     dash_daq.LEDDisplay(backgroundColor='black',
                                                         size =50,
                                                         color = led_color,
                                                         style = {'textAlign':'center'},
                                                         value = round(100*(1-mape),1))
                                     ],
                                    xs =12, sm=12, md=12, lg=6, xl=6
                                    )
                            
                            ], justify ='center'
                        
                        ),
                    html.Br(),
                    html.P('Keskimääräinen suhteellinen virhe (MAPE) on kaikkien ennustearvojen suhteellisten virheiden keskiarvo. Tarkkuus on tässä tapauksessa laskettu kaavalla 1 - MAPE.', 
                           style = {'text-align':'center',
                                    'font-family':'Arial',
                                    'font-size':20
                                    },
                           className="card-text"),
                    html.Br(),

                    
                    ], style= {'textAlign':'center'}
                
                )
        
                                
        
        test_result['PCA'] = str(pca)
        if pca:
            test_result['Selitetty varianssi'] = str(int(100*explained_variance))+'%'
        test_result['Malli'] = model_name
        test_result['Piirteet'] = ',\n'.join(features)
        test_result['Pituus'] = str(test_size)+' kuukautta'
        
        
        test_result = pd.concat([data,test_result]).reset_index()
        test_result['Hyperparametrit'] = json.dumps(hyperparam_grid)
        # test_result = pd.concat([test_result, pd.DataFrame([hyperparam_grid])],axis=1)
        
        button_children = dbc.Button(children=[html.I(className="fa fa-download mr-1"), 'Lataa testitulokset koneelle'],
                                       id='test_download_button',
                                       n_clicks=0,
                                       style = dict(fontSize=20,fontFamily='Arial Black',textAlign='center'),
                                       outline=True,
                                       size = 'lg',
                                       color = 'info'
                                       )

        return test_result.to_dict('records'), test_plot, button_children


        
@app.callback(
    
      [Output('forecast_data','data'),
       Output('forecast_results_div','children'),
       Output('forecast_download_button_div','children')],
      [Input('forecast_button','n_clicks')],
      [State('forecast_slider', 'value'),
       State('model_selector','value'),
       # State('feature_selection','value'),
       State('selected_features','data'),
       #State('weights','data'),
       State({'type': 'value_adjust', 'index': ALL}, 'id'),
       State({'type': 'value_adjust', 'index': ALL}, 'value'),
       #State('forecast_slider','value'),
       State({'type': 'hyperparameter_tuner', 'index': ALL}, 'id'),
       State({'type': 'hyperparameter_tuner', 'index': ALL}, 'value'),
       State('pca_switch','on'),
       State('ev_slider','value')
       ]
    
)
def update_forecast_results(n_clicks, 
                        forecast_size, 
                        model_name, 
                        features, 
                        feature_changes,
                        feature_change_values, 
                        hyperparams,
                        hyperparam_values,
                        pca, 
                        explained_variance):
    
    if n_clicks > 0:
    
        df = data.copy()
        
        model = MODELS[model_name]['model']
      
        constants = MODELS[model_name]['constant_hyperparameters'] 
        
        hyperparams_dict  = {hyperparams[i]['index']:hyperparam_values[i] for i in range(len(hyperparams))}
        
        hyperparam_grid = hyperparams_dict.copy()

        hyperparam_grid.update(constants)
        
        
        model = model(**hyperparam_grid)
        
        weights_dict = {feature_changes[i]['index']:feature_change_values[i] for i in range(len(feature_changes))}
        
        weights = pd.Series(weights_dict)
        forecast_df = predict(df, model, features, feature_changes = weights, length=forecast_size, use_pca=pca,n_components=explained_variance)
        
        
        forecast_df['Pääkomponenttianalyysi'] = pca
        if pca:
            forecast_df['Selitetty varianssi'] = explained_variance
        forecast_df['Malli'] = model_name
        forecast_df['Piirteet'] = ',\n'.join(features)
        forecast_df['Pituus'] = forecast_size
       
        
        forecast_df['Hyperparametrit'] = json.dumps(hyperparams_dict)
        
        forecast_df['Muutokset'] = json.dumps(weights_dict)
        
        forecast_div =  html.Div([html.Br(),
                      html.H3('Ennustetulokset', 
                              style = {'text-align':'center',
                                       'font-family':'Arial Black',
                                       }),
                      html.Br(),
                      html.P('Alla olevassa kuvaajassa on esitetty toteuteet arvot sekä ennuste ajalle {} - {}.'.format(forecast_df.index.strftime('%B %Y').values[0],forecast_df.index.strftime('%B %Y').values[-1]),
                             style = {
                                 'text-align':'center', 
                                 'font-family':'Arial', 
                                  'font-size':p_font_size
                                 }),
                      html.P('Voit valita alla olevista painikkeista joko pylväs, -tai viivadiagramin. Kuvaajan pituutta voi säätä alla olevasta liukuvaliskosta. Pituutta voi rajat myös vasemman yläkulman painikkeista.',
                             style = {
                                 'text-align':'center', 
                                 'font-family':'Arial', 
                                  'font-size':p_font_size
                                 }),
                      html.Br(),
                      html.Div([
                      dbc.RadioItems(id = 'chart_type', 
                        options = [{'label':'pylväät','value':'bars'},
                                  {'label':'viivat','value':'lines'}],
                        labelStyle={'display':'inline-block','font-size':18},
                        className="btn-group",
                        inputClassName="btn-check",
                        labelClassName="btn btn-outline-warning",
                        labelCheckedClassName="active",
                        
                       
                        value = 'lines'
                      )
                      ],style={'textAlign':'right'}),
                      html.Br(),
                  # ,
                  # style = {'textAlign':'right'}
                # ),
          dcc.Graph(id = 'forecast_graph',
                    figure = plot_forecast_data(forecast_df, chart_type='lines'), 
                    config = config_plots),
          html.Br()
          ],style={'textAlign':'center'})

          
          # ], justify='center')        
        forecast_download_button = dbc.Button(children=[html.I(className="fa fa-download mr-1"), 'Lataa ennustedata koneelle'],
                                 id='forecast_download_button',
                                 n_clicks=0,
                                 style=dict(fontSize=20,fontFamily='Arial Black',textlign='center'),
                                 outline=True,
                                 size = 'lg',
                                 color = 'info'
                                 )
        
        
        
        return forecast_df.reset_index().to_dict('records'), forecast_div, [html.Br(),forecast_download_button] 

@app.callback(
    Output("forecast_download", "data"),
    [Input("forecast_download_button", "n_clicks"),
    State('forecast_data','data')
    ]
    
)
def download_forecast_data(n_clicks, df):
    
    if n_clicks > 0:

        df = pd.DataFrame(df).set_index('Aika')
        df.index = pd.to_datetime(df.index)
        
        features = df.Piirteet.dropna().values[0].split(',\n')
        
        hyperparam_df = pd.DataFrame([json.loads(df.Hyperparametrit.values[0])]).T
        hyperparam_df.index.name = 'Hyperparametri'
        hyperparam_df.columns = ['Arvo']        
        hyperparam_df['Arvo'] = hyperparam_df['Arvo'].astype(str)
        
        value_df = pd.DataFrame([json.loads(df.Muutokset.values[0])]).T
        value_df.index.name = 'Hyödyke'
        value_df.columns = ['Indeksimuutos (%)']


        
        features = features + ['change','prev','month', 'Työttömyysaste']
        
        metadata = pd.DataFrame([{'Piirteet': df.Piirteet.dropna().values[0]+',\nEnnustettu muutos (prosenttiykköä),\nEdellisen kuukauden työttömyys -% (ennustettu),\nKuukausi',
                                  'Malli': df.Malli.values[0],
                                  'Ennusteen pituus': str(df.Pituus.values[0])+' kuukautta',
                                  'Pääkomponenttianalyysi':str(df.Pääkomponenttianalyysi.values[0])}])
        if 'Selitetty varianssi' in df.columns:
            metadata['Selitetty varianssi'] = str(int(100*df['Selitetty varianssi'].values[0]))+'%'
            
        metadata = metadata.T
        metadata.columns = ['Arvo']
        metadata.index.name = ''
        
        
        
        df = df[features].rename(columns={'change':'Ennustettu muutos (prosenttiykköä)',
                                      'prev':'Edellisen kuukauden työttömyys -% (ennustettu)',
                                      'month':'Kuukausi',
                                      'Työttömyysaste':'Työttömyysaste (ennuste %)'})
        data_ = data.copy().rename(columns={'change':'Muutos (prosenttiykköä)',
                                      'prev':'Edellisen kuukauden työttömyys -% ',
                                      'month':'Kuukausi'})
        
        
        xlsx_io = io.BytesIO()
        writer = pd.ExcelWriter(xlsx_io, engine='xlsxwriter')
        
        
        
        data_.to_excel(writer, sheet_name= 'Data')
        df.to_excel(writer, sheet_name= 'Ennustedata')
        value_df.to_excel(writer, sheet_name= 'Indeksimuutokset')
        hyperparam_df.to_excel(writer, sheet_name= 'Hyperparametrit')
        metadata.to_excel(writer, sheet_name= 'Metadata')


        writer.save()
        

        
        output = xlsx_io.getvalue()

        return dcc.send_bytes(output, 'Ennustedata {} piirteellä '.format(len(features))+datetime.now().strftime('%d_%m_%Y')+'.xlsx')
        
@app.callback(
    Output("test_download", "data"),
    [Input("test_download_button", "n_clicks"),
    State('test_data','data')
    ]
    
)
def download_test_data(n_clicks, df):
    
    if n_clicks > 0:
        
        df = pd.DataFrame(df).set_index('Aika')
        df.index = pd.to_datetime(df.index)

        
        features = df.Piirteet.dropna().values[0].split(',\n')
        
        metadata = df.loc[:,'mape':'Hyperparametrit'].dropna(axis=0).rename(columns={'mape':'MAPE'}).set_index('MAPE').drop_duplicates().T.iloc[:-1,:]
        metadata.index.name='MAPE'
       
        
        hyperparam_df = pd.DataFrame([json.loads(df.Hyperparametrit.values[0])]).T
        hyperparam_df.index.name = 'Hyperparametri'
        hyperparam_df.columns = ['Arvo']        
        hyperparam_df['Arvo'] = hyperparam_df['Arvo'].astype(str)
        
        hyperparam_df['Arvo'] = hyperparam_df['Arvo'].astype(str)
        
        columns = ['Työttömyysaste', 'Ennuste'] + features + ['prev','month','change']
        
        xlsx_io = io.BytesIO()
        writer = pd.ExcelWriter(xlsx_io, engine='xlsxwriter')
        
        df = df.reset_index().drop_duplicates(subset='Aika', keep='last').set_index('Aika')[columns].dropna(axis=0).rename(columns ={'prev':'Edellisen kuukauden ennustettu työttömyys -%','month':'Kuukausi','change':'Työttömyyden kuukausimuutos (prosenttiyksikköä)'})
        
        
        df.to_excel(writer, sheet_name= 'Testidata')
        metadata.to_excel(writer, sheet_name= 'Metadata')
        hyperparam_df.to_excel(writer, sheet_name= 'Mallin hyperparametrit')

        writer.save()
        

        
        output = xlsx_io.getvalue()

        return dcc.send_bytes(output, 'Testitulokset {} piirteellä '.format(len(features))+datetime.now().strftime('%d_%m_%Y')+'.xlsx')


@app.callback(

    Output('test_graph_div', 'children'),
    
      [Input('test_chart_type','value')],
      [State('test_data','data')]
    
)
def update_test_chart_type(chart_type,df):
    
    df = pd.DataFrame(df).set_index('Aika')
    df.index = pd.to_datetime(df.index)

    
    df = df.reset_index().drop_duplicates(subset='Aika', keep='last').set_index('Aika')[['Työttömyysaste','Ennuste','mape']].dropna(axis=0)

    return dcc.Graph(id ='test_graph', 
                     figure = plot_test_results(df,chart_type), 
                     config = config_plots)      



@app.callback(

    Output('forecast_graph', 'figure'),
    
      [Input('chart_type','value')],
      [State('forecast_data','data')]
    
)
def update_forecast_chart_type(chart_type,df):
    
    df = pd.DataFrame(df).set_index('Aika')
    df.index = pd.to_datetime(df.index)
    return plot_forecast_data(df, chart_type=chart_type)

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
                                                               style = {
                                                                   'text-align':'center',
                                                                    'font-size':p_font_size, 
                                                                   'font-family':'Arial Black'})
                                                       ], style = {'display':'none'}
                                                      )],
            True: [html.Div([html.P('Valitsit {} % säilytetyn variaation.'.format(int(100*explained_variance)),
                                                               style = {
                                                                   'text-align':'center', 
                                                                    'font-size':p_font_size, 
                                                                   'font-family':'Arial Black'})
                                                       ]
                                                      )]}[pca]



@app.callback(
    Output('feature_selection','value'),
    [Input('select_all', 'on')],
    [State('used_selection','data')]
)
def update_feature_list(on,used_selection):
       
        
    if on:
        
        return [f['value'] for f in used_selection]
    else:
        raise PreventUpdate

@app.callback(
    [Output('select_all','on'),
     Output('select_all','label'),
     Output('select_all','disabled')
     ],
    [

     Input('selected_features','data'),
     Input('used_selection','data')]
)    
def update_switch(features,used_selection):
            
    if len(features) == len(used_selection):
        
        return True, {'label':'Kaikki hyödykkeet on valittu. Voit poistaa hyödykkeitä listasta klikkaamalla rasteista.',
                       'style':{'text-align':'center', 'font-size':20, 'font-family':'Arial Black'}
                      },True
    else:
        return False, dict(label = 'Valitse kaikki',style = {'font-size':20, 'fontFamily':'Arial Black'}),False



@app.callback(

    Output('test_button_div','children'),
    [

     Input('selected_features','data')
     ]    
    
)
def add_test_button(values):
    
    if values is None:
        raise PreventUpdate 
    
    elif len(values) == 0:
        return [html.P('Valitse ensin piirteitä.',
                       style = {
                           'text-align':'center', 
                           'font-family':'Arial Black', 
                            'font-size':p_font_size
                           })]
    
    else:
        return dbc.Button('Testaa',
                           id='test_button',
                           n_clicks=0,
                           outline=False,
                           #className="btn btn-outline-info",
                           style = dict(fontSize=28,fontFamily='Arial Black')
                          )

@app.callback(
    Output('test_size_indicator','children'),
    [Input('test_slider','value')]
)
def update_test_size_indicator(value):
    
    return [html.Br(),html.P('Valitsit {} kuukautta testidataksi.'.format(value),
                             style = {
                                 'text-align':'center', 
                                  'font-size':p_font_size, 
                                 'font-family':'Arial Black', 
                                 #'color':'white'
                                 })]

@app.callback(
    Output('forecast_slider_indicator','children'),
    [Input('forecast_slider','value')]
)
def update_forecast_size_indicator(value):
    
    return [html.Br(),html.P('Valitsit {} kuukauden ennusteen.'.format(value),
                             style = {
                                 'text-align':'center', 
                                  'font-size':p_font_size, 
                                 'font-family':'Arial Black', 
                                 #'color':'white'
                                 })]




@app.callback(

    Output('timeseries_selection', 'children'),
    [
     # Input('feature_selection', 'value')
     Input('selected_features','data')
     ]    
    
)
def update_timeseries_selections(values):
    
    return [
            html.Br(),
            html.H2('Tarkastele hyödykkeiden indeksin aikasarjoja.',style = {'textAlign':'center', 'fontFamily':'Arial', 'color':'white'}),
            html.Br(),
            html.P('Tällä kuvaajalla voit tarkastella hyödykkeiden indeksikehitystä kuukausittain.',
                   style = {
                       'text-align':'center', 
                        'font-size':p_font_size, 
                       'font-family':'Arial', 
                       #'color':'white'
                       }),
            html.H4('Valitse hyödyke'),
            dcc.Dropdown(id = 'timeseries_selection_dd',
                        options = [{'value':c, 'label':c} for c in values],
                        value = [values[0]],
                        style = {
                            'font-size':p_font_size, 
                            'font-family':'Arial',
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
    return dcc.Graph(figure=go.Figure(data=traces,
                                      layout = go.Layout(title = dict(text = 'Valittujen arvojen indeksikehitys',x=.5,font=dict(family='Arial Black',size=20)),
                                                         height=graph_height,
                                                         legend=dict(font=dict(size=12,family='Arial')),
                                                         xaxis = dict(title=dict(text = 'Aika',font=dict(size=18, family = 'Arial Black')), 
                                                                      tickfont = dict(family = 'Arial Black', size = 16)),
                                                         yaxis = dict(title=dict(text = 'Pisteluku (perusvuosi = 2010)',
                                                                                font=dict(size=18, family = 'Arial Black')),
                                                                     tickformat = ' ',
                                                                     tickfont = dict(family = 'Arial Black', size = 16))
                                                         )
                                      ),
                     config = config_plots
                     )


@app.callback(

    [Output('adjustments', 'children')],
    [
     Input('slider', 'value'),
     State('averaging', 'on'), 
     State('common_change', 'on'), 
     State('selected_features','data')
     ]
    
)
def apply_changes(change, averaging, common_change, values):
    
    if change is None:
        raise PreventUpdate
    
    if averaging:
        
        average_length = change
        mean_df = apply_average(features = values, length = average_length)
        children=[]
        row_children =[dbc.Col([html.Br(), 
                                html.P(value,style={
                                        'font-family':'Arial',
                                        'font-size':p_font_size,
                                        
                                    }),
                                dcc.Input(id = {'type':'value_adjust', 'index':value}, 
                                               value = round(mean_df.loc[value],1), 
                                               type = 'number', 
                                               step = .1, placeholder=value)],xs =12, sm=12, md=4, lg=2, xl=2) for value in values]

               
        children.append(html.Div(children = dbc.Row(row_children, justify='left')))

                
        return children
    elif common_change:
        
        children=[]
            
        row_children =[dbc.Col([html.Br(), 
                                html.P(value,style={
                                        'font-family':'Arial',
                                        'font-size':p_font_size,
                                        
                                    }),
                                dcc.Input(id = {'type':'value_adjust', 'index':value}, 
                                               value = change, 
                                               type = 'number', 
                                               step = .1, placeholder=value)],xs =12, sm=12, md=4, lg=2, xl=2) for value in values]

               
        children.append(html.Div(children = dbc.Row(row_children, justify='left')))

                
        return children



@app.callback(
    
    Output('forecast_button_div','children'),
    [
     
     Input('selected_features','data')
     ]   
    
)
def add_predict_button(values):
    
    if values is None:
        raise PreventUpdate 
    
    elif len(values) == 0:
        return [html.P('Valitse ensin piirteitä.',
                       style = {
                           'text-align':'center',
                           'font-family':'Arial', 
                            'font-size':p_font_size
                           })]
    
    
        
    else:
        return [dbc.Button('Ennusta',
                   id='forecast_button',
                   n_clicks=0,
                   outline=False,
                   className="me-1",
                   size='lg',
                   color='success',
                   style = dict(fontSize=28,fontFamily='Arial Black')
                   
                   ),
                html.Br(),
                html.Div(id = 'forecast_download_button_div',style={'textAlign':'center'})]
    
@app.callback(
    Output('common_change','on'),
    [Input('averaging', 'on')]
    
    )
def disable_common_change(averaging):
    return not averaging
    
@app.callback(
    Output('averaging','on'),
    [Input('common_change', 'on')]
    
    )
def disable_averaging(common_change):
    return not common_change
    
@app.callback(

    [Output('range_div','children'),
     ],
    [Input('averaging', 'on'),
     Input('common_change','on')]
    
)
def open_slider(averaging, common_change):
    
    if averaging is None:
        raise PreventUpdate 
    
    if averaging:
        
        return dbc.Row([dbc.Col([
            
            html.H5('Valitse kuinka monen edeltävän kuukauden keskiarvoa käytetään.', style = {'text-align':'center', 'font-family':'Arial Black'}),
            html.Br(),
            dcc.Slider(id = 'slider',
                          min = 1,
                          max = 12,
                          value = 4,
                          step = 1,
                          tooltip={"placement": "top", "always_visible": True},
                           marks = {1:{'label':'kuukausi', 'style':{'font-size':20, 'fontFamily':'Arial Black','color':'white'}},
                                    # 3:{'label':'3 kuukautta', 'style':{'font-size':20, 'fontFamily':'Arial Black','color':'white'}},
                                    6:{'label':'puoli vuotta', 'style':{'font-size':20, 'fontFamily':'Arial Black','color':'white'}},
                                    # 9:{'label':'yhdeksän kuukautta', 'style':{'font-size':20, 'fontFamily':'Arial Black','color':'white'}},
                                    12:{'label':'vuosi', 'style':{'font-size':20, 'fontFamily':'Arial Black','color':'white'}}   
                                 }
                          
                        ),
            
                        ],xs =12, sm=12, md=12, lg=6, xl=6)],justify='center')
    elif common_change:
        return dbc.Row([dbc.Col([
            
            html.H5('Valitse kuinka iso suhteellista muutosta sovelletaan.', style = {'text-align':'center', 'font-family':'Arial Black'}),
            html.Br(),
            dcc.Slider(id = 'slider',
                          min = -10,
                          max = 10,
                          value = 0,
                          step = 0.1,
                          tooltip={"placement": "top", "always_visible": True},
                           marks = {-10:{'label':'-10%', 'style':{'font-size':20, 'fontFamily':'Arial Black','color':'white'}},
                                    # 3:{'label':'3 kuukautta', 'style':{'font-size':20, 'fontFamily':'Arial Black','color':'white'}},
                                    0:{'label':'0%', 'style':{'font-size':20, 'fontFamily':'Arial Black','color':'white'}},
                                    # 9:{'label':'yhdeksän kuukautta', 'style':{'font-size':20, 'fontFamily':'Arial Black','color':'white'}},
                                    10:{'label':'10%', 'style':{'font-size':20, 'fontFamily':'Arial Black','color':'white'}}   
                                 }
                          
                        ),
            
                        ],xs =12, sm=12, md=12, lg=6, xl=6)],justify='center')




@app.callback(

    Output('prompt','children'),
    [Input('slider', 'value'),
     Input('averaging', 'on'),
      Input('common_change','on')]    
    
)
def update_slider_prompt(value, averaging, common_change):
    
    if value is None:
        raise PreventUpdate
    
    if averaging:
    
        return [html.Br(),html.P('Valitsit {} viimeisen kuukauden keskiarvot.'.format(value),
                      style = {
                          'text-align':'center', 
                          'font-size':p_font_size, 
                          'font-family':'Arial'}),
                html.Br(),
                html.P('Voit vielä säätä yksittäisiä muutosarvoja laatikoihin kirjoittamalla tai oikealla olevista nuolista.',style = {'text-align':'center','font-size':p_font_size, 'font-family':'Arial Black'})]
    elif common_change:
        return [html.Br(),html.P('Valitsit {} % muutoksen.'.format(value),
                      style = {
                          'text-align':'center', 
                          'font-size':p_font_size, 
                          'font-family':'Arial'}),
                html.Br(),
                html.P('Voit vielä säätä yksittäisiä muutosarvoja laatikoihin kirjoittamalla tai oikealla olevista nuolista.',style = {'text-align':'center','font-size':p_font_size, 'font-family':'Arial Black'})]
        
    
 

@app.callback(

    Output('selected_features','data'),
    [Input('feature_selection','value')]
    
)
def store_selected_values(values):
    
    return values  

@app.callback(
    
    Output('selections_div','children'),
    
    [
     Input('selected_features','data')
     ]    
    
)
def add_adjustments(values):
    
    if values is not None or len(values)>0:
    
    
        children_avg = [html.H3('Aseta arvioitu piirteiden keskimääräinen kuukausimuutos prosenteissa.',
                                style = {'text-align':'center',
                                         'font-family':'Arial Black'}),
                    html.Br(),
                    dash_daq.BooleanSwitch(id = 'averaging', 
                                           label = dict(label = 'Käytä toteutumien keskiarvoja',style = {'font-size':20, 'fontFamily':'Arial Black'}), 
                                           on = True, 
                                           color = 'blue')
                    
                    ]
        
        children_change = [html.H3('tai aseta kaille yhtä suuri prosenttimuutos.',
                                   style = {'text-align':'center',
                                            'font-family':'Arial Black'}),
                    html.Br(),
                    dash_daq.BooleanSwitch(id = 'common_change', 
                                           label = dict(label = 'Käytä kaikille samaa muutosta',style = {'font-size':20, 'fontFamily':'Arial Black'}), 
                                           on = False, 
                                           color = 'blue')
                    
                    ]
    
        
        row_children =[dbc.Col([html.Br(), 
                                html.P(value,style = {'font-family':'Arial','font-size':p_font_size,'text-align':'center'}
                                       ),
                                dcc.Input(id = {'type':'value_adjust', 'index':value}, 
                                               value = 0.0, 
                                               type = 'number', 
                                               step = .1, placeholder=value)],xs =12, sm=12, md=4, lg=2, xl=2) for value in values]
        
        children = [dbc.Row([dbc.Col(children_avg,xs =12, sm=12, md=12, lg=6, xl=6),
                             dbc.Col(children_change,xs =12, sm=12, md=12, lg=6, xl=6)])]
        children.append(dbc.Row([html.Br(),html.Div(id='range_div'),html.Br(),html.Div(id = 'prompt')]))
                
        children.append(html.Div(id='adjustments',children = dbc.Row(row_children, justify='left')))
        
        return children

@app.callback(

    Output('corr_selection_div', 'children'),
    [

     Input('selected_features','data')
     ]
    
)
def update_corr_selection(values):
    
    if values is None:
        raise PreventUpdate

    
    return html.Div([
            html.Br(),
            html.H2('Tarkastele valitun piirteen suhdetta työttömyysasteeseen.',
                    style={'textAlign':'center',
                           'font-size':28}),
            html.Br(),
            html.P('Tällä kuvaajalla voit tarkastella valitun piirteen ja työttömyysasteen välistä korrelaatiota. Ennusteita tehtäessä on syytä valita piirteitä, jotka korreloivat vahvasti työttömyysasteen kanssa.',
                   style = {'font-family':'Arial','font-size':20,'text-align':'center'}),
        html.H4('Valitse hyödyke'),
        dcc.Dropdown(id = 'corr_feature',
                        multi = True,
                        # clearable=False,
                        options = [{'value':c, 'label':c} for c in values],
                        value = [values[0]],
                        style = {'font-size':16, 'font-family':'Arial','color': 'black'},
                        placeholder = 'Valitse hyödyke.')
        ]
        )

@app.callback(

    Output('feature_corr_selection_div', 'children'),
    [

     Input('selected_features','data')
     ]
    
)
def update_feature_corr_selection(values):
    

    
    return html.Div([
                html.Br(),
                html.H2('Tarkastele piirteiden suhteita.',
                        style={'textAlign':'center',
                                'font-size':28}),
                html.Br(),
                html.P('Tällä kuvaajalla voit tarkastella piirteiden välisiä korrelaatioita. Ennusteita tehtäessä on syytä valita piirteitä, jotka korreloivat heikosti keskenään.',
                       style = {'font-family':'Arial','font-size':p_font_size,'text-align':'center'}),
        
        dbc.Row(justify = 'center',children=[
            dbc.Col([
                html.H4('Valitse hyödyke'),
                dcc.Dropdown(id = 'f_corr1',
                                multi = False,
                                options = [{'value':c, 'label':c} for c in values],
                                value = values[0],
                                style = {'font-size':16, 'font-family':'Arial','color': 'black'},
                                placeholder = 'Valitse hyödyke.')
        ],xs =12, sm=12, md=12, lg=6, xl=6),
        #html.Br(),
            dbc.Col([
                html.H4('Valitse toinen hyödyke'),
                dcc.Dropdown(id = 'f_corr2',
                                multi = False,
                                options = [{'value':c, 'label':c} for c in values],
                                value = values[-1],
                                style = {'font-size':16, 'font-family':'Arial','color': 'black'},
                                placeholder = 'Valitse hyödyke.')
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
            dcc.Graph(figure = go.Figure(data = traces,
          layout = go.Layout(title = dict(text = '<b>{}</b> vs.<br><b>{}</b><br>(Kokonaiskorrelaatio: {})'.format(' '.join(value1.split()[1:]), ' '.join(value2.split()[1:]), corr_factor), x=.5, font=dict(family='Arial',size=22)),
                            xaxis= dict(title = dict(text='{} (pisteluku)'.format(' '.join(value1.split()[1:])), font=dict(family='Arial Black',size=16)),
                                        tickfont = dict(family = 'Arial Black', size = 16)),
                            height = graph_height,
                            legend = dict(font=dict(size=12,family='Arial'),
                                          orientation='h'),
                            hoverlabel = dict(font_size = 16, font_family = 'Arial'),
                            template = 'seaborn',
                            yaxis = dict(title = dict(text='{} (pisteluku)'.format(' '.join(value2.split()[1:])), font=dict(family='Arial Black',size=16)),
                                         tickfont = dict(family = 'Arial Black', size = 16))
                             )
          ),
                      config = config_plots)]



@app.callback(

    Output('eda_div', 'children'),
    [Input('corr_feature','value')]    
    
)
def update_eda_plot(values):
    
    if values is None:
        raise PreventUpdate 
        
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
                         name = ' '.join(value.split()[1:])+' ({})'.format(round(sorted(data[['Työttömyysaste', value]].corr()[value].values)[0],1)),
                         showlegend=True,
                         marker = dict(size=10),
                         marker_symbol = symbols[random.randint(0,len(symbols))],
                         hovertemplate = "<b>{}</b>:".format(value)+" %{x}"+"<br><b>Työttömyysaste</b>: %{y}"+" %"+"<br>(Korrelaatio: {:.2f})".format(sorted(data[['Työttömyysaste', value]].corr()[value].values)[0])) for value in values]

        
        
  
    
    return [

            html.Br(),
            dcc.Graph(figure = go.Figure(data = traces,
          layout = go.Layout(title = dict(text = 'Valitut hyödykkeet vs Työttömyysaste', x=.5, font=dict(family='Arial Black',size=22)),
                            xaxis= dict(title = dict(text='Hyödykkeiden pisteluku', font=dict(family='Arial Black',size=18)),
                                        tickfont = dict(family = 'Arial Black', size = 16)),
                            height = graph_height,
                            legend = dict(title ='<b>Hyödykkeet, suluissa korrelaatio</b>', font=dict(size=12,family='Arial')),
                            hoverlabel = dict(font_size = 16, font_family = 'Arial'),
                            template = 'seaborn',
                            yaxis = dict(title = dict(text='Työttömyysaste (%)', font=dict(family='Arial Black',size=18)),
                                         tickfont = dict(family = 'Arial Black', size = 16))
                             )
          ),
                      config = config_plots)]

@app.callback(
    [Output('feature_selection', 'options'),
     Output('sorting', 'label'),
     Output('used_selection', 'data')
     ],
    [Input('alphabet', 'n_clicks'),
     Input('corr_desc', 'n_clicks'),
     Input('corr_asc', 'n_clicks'),
     Input('corr_abs_desc', 'n_clicks'),
     Input('corr_abs_asc', 'n_clicks'),
     Input('main_class', 'n_clicks'),
     Input('second_class', 'n_clicks'),
     Input('third_class', 'n_clicks'),
     Input('fourth_class', 'n_clicks')
    ]
)
def update_selections(*args):
    
    ctx = dash.callback_context
    
    
    if not ctx.triggered:
        return corr_asc_options, "Absoluuttinen korrelaatio (laskeva)",corr_asc_options
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if button_id == 'alphabet':
        return feature_options, "Aakkosjärjestyksessä",feature_options
    elif button_id == 'corr_desc':
        return corr_desc_options, "Korrelaatio (laskeva)",corr_desc_options
    elif button_id == 'corr_asc':
        return corr_asc_options, "Korrelaatio (nouseva)",corr_asc_options
    elif button_id == 'corr_abs_desc':
        return corr_asc_options, "Absoluuttinen korrelaatio (laskeva)",corr_asc_options
    elif button_id == 'corr_abs_asc':
        return corr_asc_options, "Absoluuttinen korrelaatio (nouseva)",corr_asc_options
    elif button_id == 'main_class':
        return main_class_options, "Pääluokittain",main_class_options
    elif button_id == 'second_class':
        return second_class_options, "2. luokka",second_class_options
    elif button_id == 'third_class':
        return third_class_options, "3. luokka",third_class_options
    elif button_id == 'fourth_class':
        return fourth_class_options, "4. luokka",fourth_class_options
    


app.layout = serve_layout
if __name__ == "__main__":
    app.run_server(debug=in_dev,threaded=True)