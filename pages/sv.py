# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 11:57:56 2022

@author: tuomas.poukkula
"""
import dash 

dash.register_page(__name__,
                   title = 'Skev Phillips',
                   name = 'Skev Phillips',
                   image='sv.png',
                   description ='Prognos för arbetslösheten i Finland med förändringar i konsumentpriser')

import pandas as pd
import numpy as np
import requests
import plotly.graph_objs as go
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.decomposition import PCA
import dash_daq
import io
import math
import shap
from dash import html, dcc, callback, callback_context ,ALL, Output, Input, State
from dash.exceptions import PreventUpdate
import random
import dash_bootstrap_components as dbc
from datetime import datetime
import locale
from dash_iconify import DashIconify



np.seterr(invalid='ignore')

# riippu ollaanko Windows vai Linux -ympäristössä, mitä locale-koodausta käytetään.

try:
    locale.setlocale(locale.LC_ALL, 'sv-FI')
except:
    locale.setlocale(locale.LC_ALL, 'sv_FI')

in_dev = True

MODELS_sv = {
    
    
        'Beslutsträd':{'model':DecisionTreeRegressor,
                           'doc': 'https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html',
                           'video':"https://www.youtube.com/embed/UhY5vPfQIrA",
                           'explainer':shap.TreeExplainer,
                           'constant_hyperparameters': {
                                                        # 'n_jobs':-1,
                                                        'random_state':42}
                           },
    
        'Slumpmässig skog': {'model':RandomForestRegressor,
                           'doc': 'https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html',
                           'video':'https://www.youtube.com/embed/cIbj0WuK41w',
                           'explainer':shap.TreeExplainer,
                           'constant_hyperparameters': {
                                                        'n_jobs':-1,
                                                        'random_state':42}
                           },
        # SHAP-kirjasto ei tue AdaBoostia
        # 'Adaboost': {'model':AdaBoostRegressor,
        #              'doc':'https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html',
        #              'video':'',
        #              'explainer':shap.TreeExplainer,
        #              'constant_hyperparameters':{'random_state':42,
        #                                          }
        #              },
        # 'K Närmaste grannar':{'model':KNeighborsRegressor,
        #                         'doc':'https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html',
        #                         'video':'https://www.youtube.com/embed/jw5LhTWUoG4?list=PLRZZr7RFUUmXfON6dvwtkaaqf9oV_C1LF',
        #                         'explainer':shap.KernelExplainer,
        #                         'constant_hyperparameters': {
        #                                                     'n_jobs':-1
        #                                                     }
        #                         },
        # 'Stöd vektormaskin':{'model':SVR,
        #                     'doc':'https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html',
        #                     'video':"https://www.youtube.com/embed/_YPScrckx28",
        #                     'explainer':shap.KernelExplainer,
        #                         'constant_hyperparameters': {
        #                                                     }
        #                         },
        'Gradientförstärkning':{'model':GradientBoostingRegressor,
                          'doc':'https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html',
                          'video':"https://www.youtube.com/embed/TyvYZ26alZs",
                          'explainer':shap.TreeExplainer,
                          'constant_hyperparameters': {'random_state':42
                                                       }
                          },
        'XGBoost': {'model':XGBRegressor,
                           'doc': 'https://xgboost.readthedocs.io/en/stable/parameter.html',
                           'video':"https://www.youtube.com/embed/TyvYZ26alZs",
                           'explainer':shap.TreeExplainer,
                           'constant_hyperparameters': {
                                                        'n_jobs':-1,
                                                        'booster':'gbtree',
                                                        'nthread':1,
                                                        'random_state':42}
                           }
        # 'Stokastinen gradientin pudotus':{'model':SGDRegressor,
        #                   'doc':'https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html',
        #                   'video':"https://www.youtube.com/embed/TyvYZ26alZs",
        #                   'explainer':shap.LinearExplainer,
        #                   'constant_hyperparameters': {'random_state':42
        #                                                }
        #                   }
        
    
    }
UNWANTED_PARAMS = ['verbose',
                   'verbosity',
                   #'cache_size', 
                   'random_state',
                   'max_iter',
                   'warm_start',
                    'max_features',
                   'tol',
                   'enable_categorical',
                   'n_jobs',
                    # 'subsample',
                    'booster'
                   # 'alpha',
                   # 'l1_ratio'
                
                   
                   ]
LESS_THAN_ONE = [
        
                   'alpha',
                   'validation_fraction',
                   'colsample_bylevel',
                   'colsample_bynode',
                   'colsample_bytree',
                   'learning_rate',
                   'subsample'
                   # 'subsample',
                   
    
    ]

LESS_THAN_HALF = [
        
        
                   'min_weight_fraction_leaf',
                 
    
    ]



config_plots_sv = {'locale':'sv',
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

p_font_size = "1.3rem"
graph_height = 650

p_style = {
        # #'font-family':'Messina Modern Book',
            'font-size':p_font_size,
           'text-align':'break'}

p_center_style = {
        # #'font-family':'Messina Modern Book',
            'font-size':p_font_size,
           'text-align':'center'}


p_bold_style = {
        # #'font-family':'Cadiz Semibold',
            'font-size':"1.1rem",#p_font_size-3,
           'text-align':'left'}

h4_style = {
    # #'font-family':'Messina Modern Semibold',
            'font-size':"1.45rem",#'18px',
           'text-align':'center',
           'margin-bottom': '20px'}
h3_style = {
    # #'font-family':'Messina Modern Semibold',
            'font-size':"2.125rem",#'34px',
           'text-align':'center',
           'margin-bottom': '30px'}
h2_style = {
    # #'font-family':'Messina Modern Semibold',
            'font-size':"3.25rem",#'52px',
           'text-align':'center',
           'margin-bottom': '30px'}
h1_style = {
    # #'font-family':'Messina Modern Semibold',
            'font-size':"5rem",#'80px',
            'text-align':'center',
            'margin-bottom': '50px'}
# h1_style = "display-1 text-center fw-bold mb-5 mt-3"
footer = dbc.Card([
        html.Br(),
        
        dbc.Row([
            
            dbc.Col(dbc.NavLink(DashIconify(icon="logos:github"), href="https://github.com/tuopouk/skewedphillips",external_link=True, target='_blank',className="btn btn-link btn-floating btn-lg text-dark m-1"),className="mb-4" ,xl=1,lg=1,md=4,sm=4,xs=4),
            dbc.Col(dbc.NavLink(DashIconify(icon="line-md:twitter-x-alt"), href="https://twitter.com/TuomasPoukkula",external_link=True, target='_blank',className="btn btn-link btn-floating btn-lg text-dark m-1"),className="mb-4",xl=1,lg=1,md=4,sm=4,xs=4   ),
            dbc.Col(dbc.NavLink(DashIconify(icon="logos:linkedin"), href="https://www.linkedin.com/in/tuomaspoukkula/",external_link=True, target='_blank',className="btn btn-link btn-floating btn-lg text-dark m-1"),className="mb-4",xl=1,lg=1,md=4,sm=4,xs=4  )
            
            
            
            ],className ="d-flex justify-content-center align-items-center", justify='center',align='center')
    
    
    ],className ='card text-white bg-secondary mb-3')


def sv_set_color(x,y):
    
    
    if 'rbetslöshet' in x or x=='Nuvarande månad':
        return 'black'
    elif y < 0:
        
        return '#117733'
    elif y >= 0:
        return '#882255'



def sv_get_unemployment():
    

  url = 'https://statfin.stat.fi:443/PxWeb/api/v1/sv/StatFin/tyti/statfin_tyti_pxt_135z.px'
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

def sv_get_inflation():

  url = 'https://pxdata.stat.fi:443/PxWeb/api/v1/sv/StatFin/khi/statfin_khi_pxt_11xd.px'
  headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
                                    'Content-Type':'application/json'
                                   }
  payload = {
  "query": [
    {
      "code": "Hyödyke",
      "selection": {
        "filter": "item",
        "values": [
          "0",
          "01",
          "01.1",
          "01.1.1",
          "01.1.1.1",
          "01.1.1.1.1",
          "01.1.1.2",
          "01.1.1.2.1",
          "01.1.1.2.2",
          "01.1.1.3",
          "01.1.1.3.1",
          "01.1.1.3.2",
          "01.1.1.4",
          "01.1.1.4.2",
          "01.1.1.4.3",
          "01.1.1.4.4",
          "01.1.1.4.5",
          "01.1.1.5",
          "01.1.1.5.1",
          "01.1.1.5.2",
          "01.1.1.6",
          "01.1.1.6.1",
          "01.1.1.7",
          "01.1.1.7.1",
          "01.1.1.7.2",
          "01.1.1.8",
          "01.1.1.8.1",
          "01.1.2",
          "01.1.2.1",
          "01.1.2.1.1",
          "01.1.2.2",
          "01.1.2.2.1",
          "01.1.2.4",
          "01.1.2.4.1",
          "01.1.2.5",
          "01.1.2.5.2",
          "01.1.2.7",
          "01.1.2.7.1",
          "01.1.2.7.2",
          "01.1.2.7.3",
          "01.1.2.8",
          "01.1.2.8.1",
          "01.1.3",
          "01.1.3.1",
          "01.1.3.1.1",
          "01.1.3.1.2",
          "01.1.3.3",
          "01.1.3.3.1",
          "01.1.3.4",
          "01.1.3.4.1",
          "01.1.4",
          "01.1.4.1",
          "01.1.4.1.2",
          "01.1.4.3",
          "01.1.4.3.1",
          "01.1.4.4",
          "01.1.4.4.1",
          "01.1.4.4.2",
          "01.1.4.4.3",
          "01.1.4.5",
          "01.1.4.5.1",
          "01.1.4.5.2",
          "01.1.4.5.3",
          "01.1.4.5.4",
          "01.1.4.6",
          "01.1.4.6.1",
          "01.1.5",
          "01.1.5.1",
          "01.1.5.1.1",
          "01.1.5.2",
          "01.1.5.2.1",
          "01.1.5.2.2",
          "01.1.5.2.3",
          "01.1.5.2.4",
          "01.1.5.4",
          "01.1.5.4.1",
          "01.1.6",
          "01.1.6.1",
          "01.1.6.1.1",
          "01.1.6.1.2",
          "01.1.6.1.3",
          "01.1.6.1.4",
          "01.1.6.1.6",
          "01.1.6.1.7",
          "01.1.6.2",
          "01.1.6.2.1",
          "01.1.6.3",
          "01.1.6.3.1",
          "01.1.6.3.2",
          "01.1.6.4",
          "01.1.6.4.1",
          "01.1.7",
          "01.1.7.1",
          "01.1.7.1.1",
          "01.1.7.1.2",
          "01.1.7.1.3",
          "01.1.7.1.4",
          "01.1.7.2",
          "01.1.7.2.1",
          "01.1.7.3",
          "01.1.7.3.2",
          "01.1.7.4",
          "01.1.7.4.1",
          "01.1.7.4.3",
          "01.1.7.5",
          "01.1.7.5.1",
          "01.1.8",
          "01.1.8.1",
          "01.1.8.1.1",
          "01.1.8.2",
          "01.1.8.2.1",
          "01.1.8.3",
          "01.1.8.3.1",
          "01.1.8.4",
          "01.1.8.4.1",
          "01.1.8.4.2",
          "01.1.8.5",
          "01.1.8.5.1",
          "01.1.9",
          "01.1.9.1",
          "01.1.9.1.1",
          "01.1.9.2",
          "01.1.9.2.1",
          "01.1.9.3",
          "01.1.9.3.1",
          "01.1.9.4",
          "01.1.9.4.1",
          "01.1.9.5",
          "01.1.9.5.1",
          "01.2",
          "01.2.1",
          "01.2.1.1",
          "01.2.1.1.1",
          "01.2.1.2",
          "01.2.1.2.1",
          "01.2.1.3",
          "01.2.1.3.2",
          "01.2.2",
          "01.2.2.1",
          "01.2.2.1.1",
          "01.2.2.2",
          "01.2.2.2.1",
          "01.2.2.2.3",
          "01.2.2.3",
          "01.2.2.3.1",
          "02",
          "02.1",
          "02.1.1",
          "02.1.1.1",
          "02.1.1.1.1",
          "02.1.1.2",
          "02.1.1.2.1",
          "02.1.2",
          "02.1.2.1",
          "02.1.2.1.1",
          "02.1.2.2",
          "02.1.2.2.1",
          "02.1.2.3",
          "02.1.2.3.1",
          "02.1.2.4",
          "02.1.2.4.1",
          "02.1.3",
          "02.1.3.1",
          "02.1.3.1.1",
          "02.1.3.3",
          "02.1.3.3.1",
          "02.2",
          "02.2.0",
          "02.2.0.1",
          "02.2.0.1.1",
          "02.2.0.2",
          "02.2.0.2.1",
          "02.2.0.3",
          "02.2.0.3.1",
          "03",
          "03.1",
          "03.1.1",
          "03.1.1.1",
          "03.1.1.1.1",
          "03.1.2",
          "03.1.2.1",
          "03.1.2.1.1",
          "03.1.2.1.2",
          "03.1.2.1.3",
          "03.1.2.1.5",
          "03.1.2.1.6",
          "03.1.2.1.7",
          "03.1.2.2",
          "03.1.2.2.1",
          "03.1.2.2.2",
          "03.1.2.2.3",
          "03.1.2.2.4",
          "03.1.2.2.5",
          "03.1.2.2.6",
          "03.1.2.2.7",
          "03.1.2.3",
          "03.1.2.3.1",
          "03.1.2.3.2",
          "03.1.2.4",
          "03.1.2.4.1",
          "03.1.2.4.2",
          "03.1.3",
          "03.1.3.1",
          "03.1.3.1.1",
          "03.1.3.1.2",
          "03.1.3.2",
          "03.1.3.2.1",
          "03.1.4",
          "03.1.4.1",
          "03.1.4.1.1",
          "03.2",
          "03.2.1",
          "03.2.1.1",
          "03.2.1.1.2",
          "03.2.1.1.4",
          "03.2.1.2",
          "03.2.1.2.2",
          "03.2.1.2.3",
          "03.2.1.3",
          "03.2.1.3.2",
          "04",
          "04.1",
          "04.1.1",
          "04.1.1.1",
          "04.1.1.1.1",
          "04.1.2",
          "04.1.2.2",
          "04.1.2.2.2",
          "04.2",
          "04.2.1",
          "04.2.1.1",
          "04.2.1.1.1",
          "04.2.1.1.2",
          "04.2.2",
          "04.2.2.1",
          "04.2.2.1.1",
          "04.2.2.1.2",
          "04.2.3",
          "04.2.3.1",
          "04.2.3.1.1",
          "04.2.4",
          "04.2.4.1",
          "04.2.4.1.1",
          "04.3",
          "04.3.1",
          "04.3.1.1",
          "04.3.1.1.1",
          "04.3.1.1.2",
          "04.3.1.1.4",
          "04.3.1.1.7",
          "04.3.2",
          "04.3.2.3",
          "04.3.2.3.1",
          "04.4",
          "04.4.1",
          "04.4.1.1",
          "04.4.1.1.1",
          "04.4.2",
          "04.4.2.1",
          "04.4.2.1.1",
          "04.4.3",
          "04.4.3.1",
          "04.4.3.1.1",
          "04.4.4",
          "04.4.4.1",
          "04.4.4.1.1",
          "04.4.4.3",
          "04.4.4.3.1",
          "04.5",
          "04.5.1",
          "04.5.1.1",
          "04.5.1.1.1",
          "04.5.3",
          "04.5.3.1",
          "04.5.3.1.1",
          "04.5.5",
          "04.5.5.1",
          "04.5.5.1.1",
          "05",
          "05.1",
          "05.1.1",
          "05.1.1.1",
          "05.1.1.1.1",
          "05.1.1.1.2",
          "05.1.1.1.3",
          "05.1.1.1.4",
          "05.1.1.2",
          "05.1.1.2.1",
          "05.1.1.5",
          "05.1.1.5.1",
          "05.1.1.6",
          "05.1.1.6.1",
          "05.1.1.7",
          "05.1.1.7.2",
          "05.1.1.7.3",
          "05.1.2",
          "05.1.2.1",
          "05.1.2.1.1",
          "05.1.3",
          "05.1.3.1",
          "05.1.3.1.1",
          "05.2",
          "05.2.0",
          "05.2.0.1",
          "05.2.0.1.1",
          "05.2.0.2",
          "05.2.0.2.1",
          "05.2.0.2.2",
          "05.2.0.2.3",
          "05.2.0.3",
          "05.2.0.3.1",
          "05.2.0.3.2",
          "05.3",
          "05.3.1",
          "05.3.1.1",
          "05.3.1.1.1",
          "05.3.1.1.3",
          "05.3.1.2",
          "05.3.1.2.1",
          "05.3.1.2.2",
          "05.3.1.3",
          "05.3.1.3.3",
          "05.3.1.3.4",
          "05.3.1.5",
          "05.3.1.5.1",
          "05.3.2",
          "05.3.2.2",
          "05.3.2.2.1",
          "05.3.2.3",
          "05.3.2.3.1",
          "05.3.3",
          "05.3.3.1",
          "05.3.3.1.1",
          "05.4",
          "05.4.0",
          "05.4.0.1",
          "05.4.0.1.1",
          "05.4.0.1.3",
          "05.4.0.1.4",
          "05.4.0.1.5",
          "05.4.0.2",
          "05.4.0.2.1",
          "05.4.0.2.2",
          "05.4.0.3",
          "05.4.0.3.1",
          "05.4.0.4",
          "05.4.0.4.1",
          "05.5",
          "05.5.1",
          "05.5.1.1",
          "05.5.1.1.1",
          "05.5.2",
          "05.5.2.1",
          "05.5.2.1.1",
          "05.5.2.1.2",
          "05.5.2.2",
          "05.5.2.2.1",
          "05.5.2.2.2",
          "05.6",
          "05.6.1",
          "05.6.1.1",
          "05.6.1.1.1",
          "05.6.1.1.2",
          "05.6.1.2",
          "05.6.1.2.1",
          "05.6.1.2.2",
          "05.6.1.2.4",
          "05.6.1.2.5",
          "05.6.1.2.7",
          "05.6.1.2.8",
          "05.6.2",
          "05.6.2.1",
          "05.6.2.1.1",
          "06",
          "06.1",
          "06.1.1",
          "06.1.1.1",
          "06.1.1.1.1",
          "06.1.1.1.2",
          "06.1.1.1.3",
          "06.1.1.3",
          "06.1.1.3.1",
          "06.1.1.4",
          "06.1.1.4.1",
          "06.1.2",
          "06.1.2.1",
          "06.1.2.1.2",
          "06.1.3",
          "06.1.3.1",
          "06.1.3.1.1",
          "06.1.3.1.2",
          "06.1.3.2",
          "06.1.3.2.1",
          "06.2",
          "06.2.1",
          "06.2.1.1",
          "06.2.1.1.1",
          "06.2.1.2",
          "06.2.1.2.1",
          "06.2.2",
          "06.2.2.1",
          "06.2.2.1.1",
          "06.2.3",
          "06.2.3.1",
          "06.2.3.1.1",
          "06.2.3.3",
          "06.2.3.3.1",
          "06.3",
          "06.3.0",
          "06.3.0.1",
          "06.3.0.1.1",
          "07",
          "07.1",
          "07.1.1",
          "07.1.1.1",
          "07.1.1.1.1",
          "07.1.1.2",
          "07.1.1.2.1",
          "07.1.2",
          "07.1.2.1",
          "07.1.2.1.2",
          "07.1.3",
          "07.1.3.1",
          "07.1.3.1.1",
          "07.2",
          "07.2.1",
          "07.2.1.1",
          "07.2.1.1.1",
          "07.2.1.2",
          "07.2.1.2.1",
          "07.2.1.3",
          "07.2.1.3.1",
          "07.2.2",
          "07.2.2.1",
          "07.2.2.1.1",
          "07.2.2.2",
          "07.2.2.2.1",
          "07.2.2.4",
          "07.2.2.4.1",
          "07.2.3",
          "07.2.3.1",
          "07.2.3.1.1",
          "07.2.3.1.2",
          "07.2.3.2",
          "07.2.3.2.1",
          "07.2.4",
          "07.2.4.1",
          "07.2.4.1.1",
          "07.2.4.2",
          "07.2.4.2.1",
          "07.2.4.3",
          "07.2.4.3.1",
          "07.2.4.3.2",
          "07.3",
          "07.3.1",
          "07.3.1.1",
          "07.3.1.1.1",
          "07.3.2",
          "07.3.2.1",
          "07.3.2.1.1",
          "07.3.2.1.2",
          "07.3.2.2",
          "07.3.2.2.1",
          "07.3.3",
          "07.3.3.1",
          "07.3.3.1.1",
          "07.3.3.2",
          "07.3.3.2.1",
          "07.3.4",
          "07.3.4.1",
          "07.3.4.1.1",
          "08",
          "08.1",
          "08.1.0",
          "08.1.0.1",
          "08.1.0.1.1",
          "08.1.0.2",
          "08.1.0.2.1",
          "08.2",
          "08.2.0",
          "08.2.0.2",
          "08.2.0.2.1",
          "08.3",
          "08.3.0",
          "08.3.0.1",
          "08.3.0.1.1",
          "08.3.0.2",
          "08.3.0.2.1",
          "08.3.0.3",
          "08.3.0.3.1",
          "09",
          "09.1",
          "09.1.1",
          "09.1.1.1",
          "09.1.1.1.3",
          "09.1.1.2",
          "09.1.1.2.1",
          "09.1.1.2.2",
          "09.1.1.2.3",
          "09.1.1.2.4",
          "09.1.1.3",
          "09.1.1.3.1",
          "09.1.1.4",
          "09.1.1.4.1",
          "09.1.2",
          "09.1.2.1",
          "09.1.2.1.1",
          "09.1.2.1.2",
          "09.1.3",
          "09.1.3.1",
          "09.1.3.1.1",
          "09.1.3.2",
          "09.1.3.2.3",
          "09.1.4",
          "09.1.4.1",
          "09.1.4.1.1",
          "09.1.4.1.2",
          "09.1.4.2",
          "09.1.4.2.2",
          "09.1.4.3",
          "09.1.4.3.1",
          "09.1.5",
          "09.1.5.1",
          "09.1.5.1.1",
          "09.2",
          "09.2.1",
          "09.2.1.1",
          "09.2.1.1.1",
          "09.2.1.3",
          "09.2.1.3.1",
          "09.2.1.3.2",
          "09.2.2",
          "09.2.2.1",
          "09.2.2.1.1",
          "09.3",
          "09.3.1",
          "09.3.1.1",
          "09.3.1.1.1",
          "09.3.1.1.2",
          "09.3.1.2",
          "09.3.1.2.1",
          "09.3.1.2.3",
          "09.3.1.2.4",
          "09.3.2",
          "09.3.2.1",
          "09.3.2.1.1",
          "09.3.2.1.2",
          "09.3.2.2",
          "09.3.2.2.4",
          "09.3.3",
          "09.3.3.1",
          "09.3.3.1.2",
          "09.3.3.2",
          "09.3.3.2.1",
          "09.3.3.2.2",
          "09.3.3.3",
          "09.3.3.3.1",
          "09.3.3.3.2",
          "09.3.4",
          "09.3.4.2",
          "09.3.4.2.1",
          "09.3.4.2.2",
          "09.3.5",
          "09.3.5.1",
          "09.3.5.1.1",
          "09.4",
          "09.4.1",
          "09.4.1.1",
          "09.4.1.1.1",
          "09.4.1.1.2",
          "09.4.1.2",
          "09.4.1.2.1",
          "09.4.1.2.2",
          "09.4.1.2.5",
          "09.4.2",
          "09.4.2.1",
          "09.4.2.1.1",
          "09.4.2.1.2",
          "09.4.2.2",
          "09.4.2.2.1",
          "09.4.2.3",
          "09.4.2.3.3",
          "09.4.2.4",
          "09.4.2.4.1",
          "09.4.2.5",
          "09.4.2.5.1",
          "09.4.2.7",
          "09.4.2.7.1",
          "09.4.3",
          "09.4.3.1",
          "09.4.3.1.1",
          "09.5",
          "09.5.1",
          "09.5.1.1",
          "09.5.1.1.1",
          "09.5.1.1.2",
          "09.5.1.2",
          "09.5.1.2.2",
          "09.5.1.3",
          "09.5.1.3.1",
          "09.5.2",
          "09.5.2.1",
          "09.5.2.1.1",
          "09.5.2.1.2",
          "09.5.2.2",
          "09.5.2.2.1",
          "09.5.2.2.2",
          "09.5.3",
          "09.5.3.1",
          "09.5.3.1.1",
          "09.5.4",
          "09.5.4.1",
          "09.5.4.1.1",
          "09.5.4.2",
          "09.5.4.2.1",
          "09.6",
          "09.6.0",
          "09.6.0.1",
          "09.6.0.1.1",
          "09.6.0.2",
          "09.6.0.2.1",
          "09.6.0.2.2"
          # "10",
          # "10.2",
          # "10.2.0",
          # "10.2.0.1",
          # "10.2.0.1.2",
          # "10.5",
          # "10.5.0",
          # "10.5.0.1",
          # "10.5.0.1.1",
          # "11",
          # "11.1",
          # "11.1.1",
          # "11.1.1.1",
          # "11.1.1.1.1",
          # "11.1.1.1.2",
          # "11.1.1.1.3",
          # "11.1.1.1.4",
          # "11.1.1.1.5",
          # "11.1.1.1.6",
          # "11.1.1.1.7",
          # "11.1.1.1.8",
          # "11.1.1.1.9",
          # "11.1.1.2",
          # "11.1.1.2.1",
          # "11.1.1.3",
          # "11.1.1.3.1",
          # "11.1.1.4",
          # "11.1.1.4.1",
          # "11.1.1.4.2",
          # "11.1.1.4.3",
          # "11.1.2",
          # "11.1.2.1",
          # "11.1.2.1.1",
          # "11.2",
          # "11.2.0",
          # "11.2.0.1",
          # "11.2.0.1.1",
          # "11.2.0.2",
          # "11.2.0.2.1",
          # "11.2.0.3",
          # "11.2.0.3.1",
          # "12",
          # "12.1",
          # "12.1.1",
          # "12.1.1.1",
          # "12.1.1.1.1",
          # "12.1.1.2",
          # "12.1.1.2.1",
          # "12.1.1.2.3",
          # "12.1.1.3",
          # "12.1.1.3.1",
          # "12.1.2",
          # "12.1.2.1",
          # "12.1.2.1.1",
          # "12.1.2.2",
          # "12.1.2.2.1",
          # "12.1.2.3",
          # "12.1.2.3.1",
          # "12.1.3",
          # "12.1.3.2",
          # "12.1.3.2.1",
          # "12.1.3.2.2",
          # "12.1.3.3",
          # "12.1.3.3.1",
          # "12.1.3.3.2",
          # "12.1.3.3.3",
          # "12.1.3.3.4",
          # "12.3",
          # "12.3.1",
          # "12.3.1.1",
          # "12.3.1.1.2",
          # "12.3.1.1.3",
          # "12.3.2",
          # "12.3.2.1",
          # "12.3.2.1.1",
          # "12.3.2.1.2",
          # "12.3.2.2",
          # "12.3.2.2.1",
          # "12.3.2.3",
          # "12.3.2.3.3",
          # "12.4",
          # "12.4.0",
          # "12.4.0.1",
          # "12.4.0.1.1",
          # "12.4.0.2",
          # "12.4.0.2.1",
          # "12.5",
          # "12.5.2",
          # "12.5.2.1",
          # "12.5.2.1.1",
          # "12.5.3",
          # "12.5.3.1",
          # "12.5.3.1.1",
          # "12.5.4",
          # "12.5.4.1",
          # "12.5.4.1.1",
          # "12.6",
          # "12.6.2",
          # "12.6.2.1",
          # "12.6.2.1.1",
          # "12.7",
          # "12.7.0",
          # "12.7.0.1",
          # "12.7.0.1.2",
          # "12.7.0.4",
          # "12.7.0.4.1"
        ]
      }
    },
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


  payload = {
  "query": [
    {
      "code": "Hyödyke",
      "selection": {
        "filter": "item",
        "values": [
         
          "10",
          "10.2",
          "10.2.0",
          "10.2.0.1",
          "10.2.0.1.2",
          "10.5",
          "10.5.0",
          "10.5.0.1",
          "10.5.0.1.1",
          "11",
          "11.1",
          "11.1.1",
          "11.1.1.1",
          "11.1.1.1.1",
          "11.1.1.1.2",
          "11.1.1.1.3",
          "11.1.1.1.4",
          "11.1.1.1.5",
          "11.1.1.1.6",
          "11.1.1.1.7",
          "11.1.1.1.8",
          "11.1.1.1.9",
          "11.1.1.2",
          "11.1.1.2.1",
          "11.1.1.3",
          "11.1.1.3.1",
          "11.1.1.4",
          "11.1.1.4.1",
          "11.1.1.4.2",
          "11.1.1.4.3",
          "11.1.2",
          "11.1.2.1",
          "11.1.2.1.1",
          "11.2",
          "11.2.0",
          "11.2.0.1",
          "11.2.0.1.1",
          "11.2.0.2",
          "11.2.0.2.1",
          "11.2.0.3",
          "11.2.0.3.1",
          "12",
          "12.1",
          "12.1.1",
          "12.1.1.1",
          "12.1.1.1.1",
          "12.1.1.2",
          "12.1.1.2.1",
          "12.1.1.2.3",
          "12.1.1.3",
          "12.1.1.3.1",
          "12.1.2",
          "12.1.2.1",
          "12.1.2.1.1",
          "12.1.2.2",
          "12.1.2.2.1",
          "12.1.2.3",
          "12.1.2.3.1",
          "12.1.3",
          "12.1.3.2",
          "12.1.3.2.1",
          "12.1.3.2.2",
          "12.1.3.3",
          "12.1.3.3.1",
          "12.1.3.3.2",
          "12.1.3.3.3",
          "12.1.3.3.4",
          "12.3",
          "12.3.1",
          "12.3.1.1",
          "12.3.1.1.2",
          "12.3.1.1.3",
          "12.3.2",
          "12.3.2.1",
          "12.3.2.1.1",
          "12.3.2.1.2",
          "12.3.2.2",
          "12.3.2.2.1",
          "12.3.2.3",
          "12.3.2.3.3",
          "12.4",
          "12.4.0",
          "12.4.0.1",
          "12.4.0.1.1",
          "12.4.0.2",
          "12.4.0.2.1",
          "12.5",
          "12.5.2",
          "12.5.2.1",
          "12.5.2.1.1",
          "12.5.3",
          "12.5.3.1",
          "12.5.3.1.1",
          "12.5.4",
          "12.5.4.1",
          "12.5.4.1.1",
          "12.6",
          "12.6.2",
          "12.6.2.1",
          "12.6.2.1.1",
          "12.7",
          "12.7.0",
          "12.7.0.1",
          "12.7.0.1.2",
          "12.7.0.4",
          "12.7.0.4.1"
        ]
      }
    },
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
  df2 = pd.merge(left = pd.merge(left = kuukausi_df, right = hyödyke_df, on = 'index', how = 'outer'), right = tiedot_df, on = 'index', how ='outer').drop('index',axis=1).set_index('Aika')




  # df['name'] = [' '.join(c.split()[1:]) for c in df.Hyödyke]
  # df =df .reset_index()
  # df =df.drop_duplicates(subset=['Aika','name'],keep='first')
  # df = df.set_index('Aika')
  # df = df.drop('name',axis=1)

  return pd.concat([df,df2])

def sv_get_inflation_percentage():

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
  
def sv_get_data():
  unemployment_df = sv_get_unemployment()
  inflation_df = sv_get_inflation()

  inflation_df = pd.pivot_table(inflation_df.reset_index(), columns = 'Hyödyke', index = 'Aika' )
  inflation_df.columns = [c[-1] for c in inflation_df.columns]

  data = pd.merge(left = unemployment_df.drop('Tiedot',axis=1).reset_index(), right = inflation_df.reset_index(), how = 'outer', on = 'Aika').set_index('Aika')
  data.Työttömyysaste = data.Työttömyysaste.fillna(-1)
  data = data.dropna(axis=1)

  inflation_percentage_df = sv_get_inflation_percentage()

  data = pd.merge(left = data.reset_index(), right = inflation_percentage_df.reset_index(), how = 'inner', on = 'Aika').set_index('Aika').sort_index()

  data.Työttömyysaste = data.Työttömyysaste.replace(-1, np.nan)

  data['prev'] = data['Työttömyysaste'].shift(1)

  data['month'] = data.index.month
  data['change'] = data.Työttömyysaste - data.prev
  
  # data = data.T.drop_duplicates().T

  return data


data_sv = sv_get_data()


def sv_draw_phillips_curve():
    
  try:
      locale.setlocale(locale.LC_ALL, 'sv_FI')
  except:
      locale.setlocale(locale.LC_ALL, 'sv-FI')
      
      
  data_ = data_sv[(data_sv.Työttömyysaste.notna())&(data_sv.Inflaatio.notna())].copy()
    
  max_date = data_.index.values[-1]
  max_date_str = data_.index.strftime('%B %Y').values[-1]

  a, b = np.polyfit(np.log(data_.Työttömyysaste), data_.Inflaatio, 1)

  y = a * np.log(data_.Työttömyysaste) +b 

  df = data_.copy()
  df['log_inflation'] = y
  df = df.sort_values(by = 'log_inflation')
  
  

  hovertemplate = ['<b>{}</b><br>Arbetslöshet: {} %<br>Inflation: {} %'.format(df.index[i].strftime('%B %Y'), df.Työttömyysaste.values[i], df.Inflaatio.values[i]) for i in range(len(df))]

  return go.Figure(data=[
                  go.Scatter(y=df['Inflaatio'], 
                            x = df.Työttömyysaste, 
                            name = 'Inflation', 
                            mode = 'markers+text', 
                            text = [None if i != max_date else '<b>'+max_date_str+'</b>' for i in df.index],
                            textposition='top left',
                            textfont=dict(
                                 family='Cadiz Book',
                                          size = 18
                                          ),
                            showlegend=False,
                            hovertemplate=hovertemplate,
                            marker = dict(color = '#E66100', symbol='diamond', 
                                           size = 10
                                          )
                            ),
                go.Scatter(x = df.Työttömyysaste, 
                            y = df['log_inflation'], 
                            name = 'Logaritmisk<br>trendlinje', 
                            mode = 'lines',
                            line = dict(width=5),
                            showlegend=True,
                            hovertemplate=[], 
                            marker = dict(color = '#5D3A9B'))
                  ],
            layout = go.Layout(
                               xaxis=dict(showspikes=True,
                                          title = dict(text='Arbetslöshet (%)', font=dict(size=18, 
                                                                                             family = 'Cadiz Semibold'
                                                                                            )), 
                                          tickformat = ' ',
                                          automargin=True,
                                          
                                          tickfont = dict(
                                                           size=16, 
                                                           family = 'Cadiz Semibold'
                                                          )
                                          ), 
                               yaxis=dict(showspikes=True,
                                          title = dict(text='Inflation (%)', font=dict(size=18, 
                                                                                        family = 'Cadiz Semibold'
                                                                                       )
                                                       ),
                                          tickformat = ' ', 
                                          automargin=True,
                                          tickfont = dict(
                                               size=16,
                                               family = 'Cadiz Semibold'
                                              )
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
                                hoverlabel = dict(font=dict(size=18,
                                                             family='Cadiz Book'
                                                            )),
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
                              
                               title = dict(text = 'Arbetslöshet vs.<br>Inflation<br>{} - {}<br>'.format(df.index.min().strftime('%B %Y'),df.index.max().strftime('%B %Y')),
                                            x=.5,
                                            font=dict(
                                                size=20,
                                                 family = 'Cadiz Semibold'
                                                ))
                              )
            )
            
def sv_get_shap_values(model, explainer, X_train, X_test):


    if explainer.__name__ == 'Kernel':
        explainer = explainer(model.predict, X_train)
        shap_values = explainer.shap_values(X_test,normalize=False, n_jobs=-1)
        feature_names = X_test.columns
        shap_df = pd.DataFrame(shap_values, columns=feature_names)
        vals = np.abs(shap_df.values).mean(0)
        base_value = explainer.expected_value
        shap_importance = pd.DataFrame(list(zip(feature_names, vals)), columns=['col_name', 'feature_importance_vals']).set_index('col_name')
        shap_importance = shap_importance.sort_values(by=['feature_importance_vals'], ascending=False) 
        shap_importance.columns = ['SHAP']
        shap_importance.index = [i for i in shap_importance.index]
        shap_importance.index = shap_importance.index.str.replace('prev','Edellisen kuukauden työttömyysaste')
        shap_importance.index = shap_importance.index.str.replace('month','Kuukausi')
        shap_df['base'] = base_value
        return shap_importance, shap_df
    
    elif explainer.__name__ == 'Tree':
        explainer = explainer(model)
        shap_values = explainer(X_test)
        feature_names = shap_values.feature_names
        shap_df = pd.DataFrame(shap_values.values, columns=feature_names)
        vals = np.abs(shap_df.values).mean(0)
        try:
            base_value = explainer.expected_value[0]
        except:
            base_value = explainer.expected_value
        shap_importance = pd.DataFrame(list(zip(feature_names, vals)), columns=['col_name', 'feature_importance_vals']).set_index('col_name')
        shap_importance = shap_importance.sort_values(by=['feature_importance_vals'], ascending=False) 
        shap_importance.columns = ['SHAP']
        shap_importance.index = [i for i in shap_importance.index]
        shap_importance.index = shap_importance.index.str.replace('prev','Edellisen kuukauden työttömyysaste')
        shap_importance.index = shap_importance.index.str.replace('month','Kuukausi')
        shap_df['base'] = base_value
        return shap_importance, shap_df
    else:
        explainer = explainer(model,X_train)
        shap_values = explainer(X_test)
        feature_names = X_test.columns
        feature_names = shap_values.feature_names
        shap_df = pd.DataFrame(shap_values.values, columns=feature_names)
        vals = np.abs(shap_df.values).mean(0)
        base_value = explainer.expected_value
        shap_importance = pd.DataFrame(list(zip(feature_names, vals)), columns=['col_name', 'feature_importance_vals']).set_index('col_name')
        shap_importance = shap_importance.sort_values(by=['feature_importance_vals'], ascending=False) 
        shap_importance.columns = ['SHAP']
        shap_importance.index = [i for i in shap_importance.index]
        shap_importance.index = shap_importance.index.str.replace('prev','Edellisen kuukauden työttömyysaste')
        shap_importance.index = shap_importance.index.str.replace('month','Kuukausi')
        shap_df['base'] = base_value
        return shap_importance, shap_df
        

def sv_get_param_options(model_name):

  if model_name == 'XGBoost':
      
      return {'objective': ['reg:squarederror',
               # 'reg:squaredlogerror',
               'reg:pseudohubererror'
                          ],
              'eval_metric':['rmse','mae','mape','mphe'],
             'base_score': 'float',
             'booster': ['gbtree', 'gblinear', 'dart'],
             'colsample_bylevel': 'float',
             'colsample_bynode': 'float',
             'colsample_bytree': 'float',
             # 'enable_categorical': 'bool',
             'gamma': 'int',
             # 'gpu_id': None,
             # 'importance_type': None,
             # 'interaction_constraints': None,
             'learning_rate': 'float',
             # 'max_delta_step': 'int',
             # 'max_depth': 'int',
              'min_child_weight': 'int',
             # 'missing': np.nan,
             # 'monotone_constraints': None,
             'n_estimators': 'int',
             'n_jobs': 'int',
             # 'num_parallel_tree': 'int',
             # 'predictor': ['auto','cpu_predictor','gpu_predictor'],
             'random_state': 'int',
             'reg_alpha': 'int',
             'reg_lambda': 'int',
             'scale_pos_weight': 'int',
             'subsample': 'int',
             'tree_method': ['auto', 'exact', 'approx', 'hist'],
             'validate_parameters': 'bool',
             'verbosity': 'int'}

  model = MODELS_sv[model_name]['model']

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



def sv_plot_test_results(df, chart_type = 'lines+bars'):
    
    try:
        locale.setlocale(locale.LC_ALL, 'sv-FI')
    except:
        locale.setlocale(locale.LC_ALL, 'sv_FI')
    
   
    hovertemplate = ['<b>{}</b>:<br>Sant: {}<br>Förutsägt: {}'.format(df.index[i].strftime('%B %Y'),df.Työttömyysaste.values[i], df.Ennuste.values[i]) for i in range(len(df))]
    
    if chart_type == 'lines+bars':
    
        return go.Figure(data=[go.Scatter(x=df.index.strftime('%B %Y'), 
                               y = df.Työttömyysaste, 
                               name = 'Sant',
                               showlegend=True, 
                               mode = 'lines+markers+text',
                               text=[str(round(c,2))+' %' for c in df.Työttömyysaste], 
                               textposition='top center',
                               hovertemplate = hovertemplate,
                               textfont = dict(
                                    family='Cadiz Semibold', 
                                   size = 16), 
                               marker = dict(color='#1A85FF',size=12),
                               line = dict(width=5)),
                    
                    go.Bar(x=df.index.strftime('%B %Y'), 
                           y = df.Ennuste, 
                           name = 'Förutsägt',
                           showlegend=True, 
                           marker = dict(color='#D41159'), 
                           text=[str(round(c,2))+' %' for c in df.Ennuste], 
                           textposition='inside',
                           hovertemplate = hovertemplate,
                           textfont = dict(
                                family='Cadiz Semibold', 
                               size = 16)
                           )
                    ],layout=go.Layout(xaxis = dict(title = dict(text='Tid',font=dict(size=20, 
                                                                                       family = 'Cadiz Semibold'
                                                                                       )),
                                                    tickfont = dict(
                                                        family = 'Cadiz Semibold', 
                                                        size = 16),
                                                    automargin=True
                                                    ),
                                       yaxis = dict(title = dict(text='Arbetslöshet (%)',
                                                                 font=dict(
                                                                      family='Cadiz Semibold',
                                                                     size=16)),
                                                    tickfont = dict(
                                                        family = 'Cadiz Semibold', 
                                                        size = 16),
                                                    automargin=True
                                                    ),
                                       height = graph_height+100,
                                       margin=dict(
                                            l=10,
                                           r=10,
                                           # b=100,
                                            # t=120,
                                            # pad=4
                                       ),
                                       legend = dict(font=dict(size=16,
                                                                family='Cadiz Book'
                                                               ),
                                                      orientation='h',
                                                       xanchor='center',
                                                       yanchor='top',
                                                        x=.47,
                                                        y=1.04
                                                     ),
                                       hoverlabel = dict(font_size = 16, 
                                                         font_family = 'Cadiz Book'
                                                         ),
                                       template = 'seaborn',
                                      # margin=dict(autoexpand=True),
                                       title = dict(text = 'Prognos för arbetslöshetstal<br>per månad',
                                                    x=.5,
                                                    font=dict(
                                                         family='Cadiz Semibold',
                                                        size=20)
                                                    )
                                       )
                                                                                       )
                                                                                       
                                                    
    elif chart_type == 'lines':
    
        return go.Figure(data=[go.Scatter(x=df.index, 
                                y = df.Työttömyysaste, 
                                name = 'Sant',
                                showlegend=True, 
                                mode = 'lines+markers+text',
                                text=[str(round(c,2))+' %' for c in df.Työttömyysaste], 
                                textposition='top center',
                                hovertemplate = hovertemplate,
                                textfont = dict(
                                     family='Cadiz Semibold', 
                                    size = 18), 
                                marker = dict(color='#1A85FF',size=10),
                                line = dict(width=2)),
                    
                    go.Scatter(x=df.index, 
                            y = df.Ennuste, 
                            name = 'Förutsägt',
                            showlegend=True,
                            mode = 'lines+markers+text',
                            marker = dict(color='#D41159',size=10), 
                            text=[str(round(c,2))+' %' for c in df.Ennuste], 
                            # textposition='inside',
                            hovertemplate = hovertemplate,
                            line = dict(width=2),
                            textfont = dict(
                                 family='Cadiz Semibold', 
                                size = 16))
                    ],layout=go.Layout(xaxis = dict(title = dict(text='Tid',font=dict(size=16, 
                                                                                       family = 'Cadiz Semibold'
                                                                                       )),
                                                    tickfont = dict(
                                                        family = 'Cadiz Semibold', 
                                                        size = 16),
                                                    automargin=True,
                                                    ),
                                        yaxis = dict(title = dict(text='Arbetslöshet (%)',font=dict(
                                             family='Cadiz Semibold',
                                            size=16)),
                                                    tickfont = dict(
                                                        family = 'Cadiz Semibold',
                                                        size = 16),
                                                    automargin=True
                                                    ),
                                        height = graph_height+100,
                                        legend = dict(font=dict(size=16,
                                                                 family='Cadiz Book'
                                                                ),
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
                                        hoverlabel = dict(font_size = 16, 
                                                           font_family = 'Cadiz Book'
                                                          ),
                                        template = 'seaborn',
                                        title = dict(text = 'Prognos för arbetslöshetstal<br>per månad',
                                                    x=.5,
                                                    font=dict(
                                                         family='Cadiz Semibold',
                                                        size=20)
                                                    )
                                        ))
                                                    

    else:
        return go.Figure(data=[go.Bar(x=df.index.strftime('%B %Y'), 
                                    y = df.Työttömyysaste, 
                                    name = 'Sant',
                           showlegend=True, 
                           marker = dict(color='#1A85FF'), 
                           text=[str(round(c,2))+' %' for c in df.Työttömyysaste], 
                           textposition='inside',
                           hovertemplate = hovertemplate,
                           textfont = dict(
                                family='Cadiz Semibold', 
                               size = 16)
                                    ),
                        
                        go.Bar(x=df.index.strftime('%B %Y'), 
                                y = df.Ennuste, 
                                name = 'Förutsägt',
                           showlegend=True, 
                           marker = dict(color='#D41159'), 
                           text=[str(round(c,2))+' %' for c in df.Ennuste], 
                           textposition='inside',
                           hovertemplate = hovertemplate,
                           textfont = dict(
                                family='Cadiz Semibold', 
                               size = 16)
                                )
                        ],layout=go.Layout(xaxis = dict(title = dict(text='Tid',font=dict(size=20, 
                                                                                           family = 'Cadiz Semibold'
                                                                                           )),
                                                        tickfont = dict(
                                                            family = 'Cadiz Semibold', 
                                                            size = 18),
                                                        automargin=True
                                                        ),
                                            yaxis = dict(title = dict(text='Arbetslöshet (%)',font=dict(
                                                 family='Cadiz Semibold',
                                                size=16)),
                                                        tickfont = dict(
                                                         family = 'Cadiz Semibold', 
                                                        size = 16),
                                                        automargin=True
                                                        ),
                                            height = graph_height+100,
                                            margin=dict(
                                                 l=10,
                                                r=10,
                                                # b=100,
                                                 # t=120,
                                                 # pad=4
                                            ),
                                            legend = dict(font=dict(size=16,
                                                                     family='Cadiz Book'
                                                                    ),
                                                           orientation='h',
                                                            xanchor='center',
                                                            yanchor='top',
                                                             x=.47,
                                                             y=1.04
                                                          ),
                                            hoverlabel = dict(font_size = 18,
                                                              font_family = 'Cadiz Book'
                                                              ),
                                            template = 'seaborn',
                                            title = dict(text = 'Prognos för arbetslöshetstal<br>per Month',
                                                        x=.5,
                                                        font=dict(
                                                             family='Cadiz Semibold',
                                                            size=20)
                                                        )
                                            )
                                                        )                                                   

                                                    
                                                    
                                                    
def sv_plot_forecast_data(df, chart_type):
    
    try:
        locale.setlocale(locale.LC_ALL, 'sv-FI')
    except:
        locale.setlocale(locale.LC_ALL, 'sv_FI')
    
    
    hover_true = ['<b>{}</b><br>Arbetslöshet: {} %'.format(data_sv.index[i].strftime('%B %Y'), data_sv.Työttömyysaste.values[i]) for i in range(len(data_sv))]
    hover_pred = ['<b>{}</b><br>Arbetslöshet: {} %'.format(df.index[i].strftime('%B %Y'), round(df.Työttömyysaste.values[i],1)) for i in range(len(df))]
    

    if chart_type == 'lines':
    
    
        return go.Figure(data=[go.Scatter(x=data_sv.index, 
                                          y = data_sv.Työttömyysaste, 
                                          name = 'Sant',
                                          showlegend=True,
                                          mode="lines", 
                                          hovertemplate =  hover_true,##'<b>%{x}</b>: %{y}%',
                                          marker = dict(color='#1A85FF')),
                    go.Scatter(x=df.index, 
                               y = np.round(df.Työttömyysaste,1), 
                               name = 'Förutsägt',
                               showlegend=True,
                               mode="lines", 
                               hovertemplate = hover_pred,#'<b>%{x}</b>: %{y}%',
                               marker = dict(color='#D41159'))
                    ],layout=go.Layout(xaxis = dict(title = dict(text = 'Tid',font=dict(
                         size=16, 
                        family = 'Cadiz Semibold'
                        )),
                        automargin=True,
                
                                                    tickfont = dict(family = 'Cadiz Semibold', 
                                                                     size = 16
                                                                    ),
                                                    rangeslider=dict(visible=True),
                                                    rangeselector=dict(
                buttons=list([
                    dict(count=3,
                         label="3m",
                         step="month",
                         stepmode="backward"),
                    dict(count=6,
                         label="6m",
                         step="month",
                         stepmode="backward"),
                    dict(count=1,
                         label="YTD",
                         step="year",
                         stepmode="todate"),
                    dict(count=1,
                         label="1å",
                         step="year",
                         stepmode="backward"),
                    dict(count=3,
                         label="3å",
                         step="year",
                         stepmode="backward"),
                    dict(count=5,
                         label="5å",
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
                                            font_size = 16, 
                                                         font_family = 'Cadiz Book'
                                                         ),
                                       legend = dict(orientation='h',
                                                     x=.5,
                                                     y=.01,
                                                     xanchor='center',
                                                      yanchor='bottom',
                                                     font=dict(
                                                    size=14,
                                                   family = 'Cadiz Semibold'
                                                   )),
                                       template = 'seaborn',
                                       yaxis = dict(title=dict(text = 'Arbetslöshet (%)',
                                                     font=dict(
                                                          size=16, 
                                                         family = 'Cadiz Semibold'
                                                         )),
                                                    automargin=True,
                                                     tickfont = dict(
                                                         family = 'Cadiz Book', 
                                                                      size = 16
                                                                     )),
                                       title = dict(text = 'Arbetslöshetstal och prognos per månad<br>{} - {}'.format(data_sv.index.strftime('%B %Y').values[0],df.index.strftime('%B %Y').values[-1]),
                                                    x=.5,
                                                    font=dict(
                                                         family='Cadiz Semibold',
                                                               size=16
                                                              )),
    
                                       ))
    
    elif chart_type == 'area':
    
    
        return go.Figure(data=[go.Scatter(x=data_sv.index, 
                                          y = data_sv.Työttömyysaste, 
                                          name = 'Sant',
                                          showlegend=True,
                                          fill = 'tozeroy',
                                          mode="lines", 
                                          hovertemplate =  hover_true,##'<b>%{x}</b>: %{y}%',
                                          marker = dict(color='#1A85FF')),
                    go.Scatter(x=df.index, 
                               y = np.round(df.Työttömyysaste,1), 
                               name = 'Förutsägt',
                               showlegend=True,
                               fill = 'tozeroy',
                               mode="lines", 
                               hovertemplate = hover_pred,#'<b>%{x}</b>: %{y}%',
                               marker = dict(color='#D41159'))
                    ],layout=go.Layout(xaxis = dict(title = dict(text = 'Tid',font=dict(
                         size=16, 
                        family = 'Cadiz Semibold'
                        )),
                        automargin=True,
                
                                                    tickfont = dict(family = 'Cadiz Semibold', 
                                                                     size = 16
                                                                    ),
                                                    rangeslider=dict(visible=True),
                                                    rangeselector=dict(
                buttons=list([
                    dict(count=3,
                         label="3m",
                         step="month",
                         stepmode="backward"),
                    dict(count=6,
                         label="6m",
                         step="month",
                         stepmode="backward"),
                    dict(count=1,
                         label="YTD",
                         step="year",
                         stepmode="todate"),
                    dict(count=1,
                         label="1å",
                         step="year",
                         stepmode="backward"),
                    dict(count=3,
                         label="3å",
                         step="year",
                         stepmode="backward"),
                    dict(count=5,
                         label="5å",
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
                                            font_size = 16, 
                                                         font_family = 'Cadiz Book'
                                                         ),
                                       legend = dict(orientation='h',
                                                     x=.5,
                                                     y=.01,
                                                     xanchor='center',
                                                      yanchor='bottom',
                                                     font=dict(
                                                    size=14,
                                                   family = 'Cadiz Semibold'
                                                   )),
                                       template = 'seaborn',
                                       yaxis = dict(title=dict(text = 'Arbetslöshet (%)',
                                                     font=dict(
                                                          size=16, 
                                                         family = 'Cadiz Semibold'
                                                         )),
                                                    automargin=True,
                                                    rangemode='tozero',
                                                     tickfont = dict(
                                                         family = 'Cadiz Book', 
                                                                      size = 16
                                                                     )),
                                       title = dict(text = 'Arbetslöshetstal och prognos per månad<br>{} - {}'.format(data_sv.index.strftime('%B %Y').values[0],df.index.strftime('%B %Y').values[-1]),
                                                    x=.5,
                                                    font=dict(
                                                         family='Cadiz Semibold',
                                                               size=16
                                                              )),
    
                                       ))


    else:
        
        
      
        return go.Figure(data=[go.Bar(x=data_sv.index, 
                                          y = data_sv.Työttömyysaste, 
                                          name = 'Sant',
                                          showlegend=True,
                                          # mode="lines", 
                                          hovertemplate = hover_true,#'<b>%{x}</b>: %{y}%',
                                          marker = dict(color='#1A85FF')),
                    go.Bar(x=df.index, 
                               y = np.round(df.Työttömyysaste,1), 
                               name = 'Förutsägt',
                               showlegend=True,
                               # mode="lines", 
                               hovertemplate = hover_pred,#'<b>%{x}</b>: %{y}%',
                               marker = dict(color='#D41159'))
                    ],layout=go.Layout(xaxis = dict(title = dict(text = 'Tid',
                                                                 font=dict(
                                                                     size=16, 
                                                                     family = 'Cadiz Semibold'
                                                                     )),
                                                                 automargin=True,
                                                    tickfont = dict(family = 'Cadiz Semibold', 
                                                                    size = 16
                                                                    ),
                                                    rangeslider=dict(visible=True),
                                                    rangeselector=dict(
                buttons=list([
                    dict(count=3,
                          label="3m",
                          step="month",
                          stepmode="backward"),
                    dict(count=6,
                          label="6m",
                          step="month",
                          stepmode="backward"),
                    dict(count=1,
                          label="YTD",
                          step="year",
                          stepmode="todate"),
                    dict(count=1,
                          label="1å",
                          step="year",
                          stepmode="backward"),
                    dict(count=3,
                          label="3å",
                          step="year",
                          stepmode="backward"),
                    dict(count=5,
                          label="5å",
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
                                           font_size = 16, 
                                            font_family = 'Cadiz Book'
                                           ),
                                       legend = dict(orientation='h',
                                                     x=.5,
                                                     y=.01,
                                                     xanchor='center',
                                                      yanchor='bottom',
                                                     font=dict(
                                                    size=14,
                                                   family = 'Cadiz Semibold'
                                                   )),
                                       yaxis = dict(title=dict(text = 'Arbetslöshet (%)',
                                                     font=dict(
                                                          size=16, 
                                                         family = 'Cadiz Semibold'
                                                         )),
                                                    automargin=True,
                                                     tickfont = dict(
                                                         family = 'Cadiz Book', 
                                                                      size = 16
                                                                     )
                                                     ),
                                       title = dict(text = 'Arbetslöshetstal och prognos per månad<br>{} - {}'.format(data_sv.index.strftime('%B %Y').values[0],df.index.strftime('%B %Y').values[-1]),
                                                    x=.5,
                                                    font=dict(
                                                         family='Cadiz Semibold',
                                                               size=16
                                                              )),
    
                                       )) 
                                                    
                                                    


def sv_test(model, features, test_size, explainer, use_pca = False, n_components=.99):

  feat = features.copy()
  feat.append('prev')
  feat.append('month')
  
  cols = feat
  
  data_ = data_sv.iloc[1:,:].copy()
    
  data_ = data_[data_.Työttömyysaste.notna()]

  train_df = data_.iloc[:-test_size,:].copy()
  test_df = data_.iloc[-test_size:,:].copy()


  scl = StandardScaler()
  label = 'change'
  x = train_df[feat]
  y = train_df[label]

  n_feat  = len(features) 
  X = scl.fit_transform(x)
  pca = PCA(n_components = n_components, random_state = 42, svd_solver = 'full')

  if use_pca:
    
    X = pca.fit_transform(X)
    n_feat = len(pd.DataFrame(X).columns)
    cols = ['_ '+str(i+1)+'. PC' for i in range(n_feat)]


  model.fit(X,y)

  results = []
  scaled_features = []

  df = pd.DataFrame(test_df.iloc[0,:]).T
  
  
  X = scl.transform(df[feat])

  if use_pca:
    X = pca.transform(X)

  df['Ennustettu muutos'] = model.predict(X)
  df['Ennuste'] = np.maximum(0, df.prev + df['Ennustettu muutos'])

  results.append(df[feat+['Työttömyysaste', 'Ennuste','change', 'Ennustettu muutos']])

  scaled_features.append(pd.DataFrame(X, columns = cols))

  for i in test_df.index[1:]:

    df = pd.DataFrame(test_df.loc[i,feat]).T
    df['Työttömyysaste'] = test_df.loc[i,'Työttömyysaste']
    df['change'] = test_df.loc[i,'change']
    df['prev'] = results[-1].Ennuste.values[0]
    # df['month'] = test_df.loc[i,'month']
    X = scl.transform(df[feat])

    if use_pca:
      X = pca.transform(X)

    df['Ennustettu muutos'] = model.predict(X)
    df['Ennuste'] = np.maximum(0, df.prev + df['Ennustettu muutos'])

    results.append(df[feat+['Työttömyysaste', 'Ennuste','change', 'Ennustettu muutos']])

    scaled_features.append(pd.DataFrame(X, columns = cols))
  
  

  shap_df, local_shap_df  = sv_get_shap_values(model, explainer, X_train = pd.DataFrame(X, columns = cols), X_test = pd.concat(scaled_features))

  result = pd.concat(results)
  result['n_feat'] = n_feat
  result.Ennuste = np.round(result.Ennuste,1)
  result['mape'] = mean_absolute_percentage_error(result.Työttömyysaste, result.Ennuste)
  

  
  result.index.name ='Aika'
    
  result = result[['Työttömyysaste', 'Ennuste', 'change', 'Ennustettu muutos', 'prev','n_feat','mape','month']+features]
  

  return result, shap_df, local_shap_df                                                    

                                                

def sv_predict(model,explainer, features, feature_changes, length, use_pca = False, n_components=.99):
  
  df = data_sv.copy()
  df = df.iloc[1:,:]
  df = df[df.Työttömyysaste.notna()]
  
  feat = features.copy()
  feat.append('prev')
  feat.append('month')
  
  cols = feat

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
    cols = ['_ '+str(i+1)+'. PC' for i in range(n_feat)]
    
    
  model.fit(X,y)
  
  
  if data_sv.Työttömyysaste.isna().sum() > 0:
      last_row = data_sv.iloc[-1:,:].copy()
  else:

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
  scaled_features_shap = []

  results.append(last_row)
  scaled_features_shap.append(pd.DataFrame(scaled_features, columns = cols))
  

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
    scaled_features_shap.append(pd.DataFrame(scaled_features, columns = cols))

  result = pd.concat(results)
  result['n_feat'] = n_feat
  
  shap_df, local_shap_df = sv_get_shap_values(model, explainer, X_train = pd.DataFrame(X, columns = cols), X_test = pd.concat(scaled_features_shap))

  local_shap_df.index = result.index

  return result, shap_df, local_shap_df



def sv_apply_average(features, length = 4):

  return 100 * data_sv[features].pct_change().iloc[-length:, :].mean()





# Viimeiset neljä saraketta ovat prev, month, change ja inflaatio.
 
correlations_desc_sv = data_sv[data_sv.columns[:-4]].corr()['Työttömyysaste'].iloc[1:].sort_values(ascending=False)
correlations_asc_sv = data_sv[data_sv.columns[:-4]].corr()['Työttömyysaste'].iloc[1:].sort_values(ascending=True)
correlations_abs_desc_sv = data_sv[data_sv.columns[:-4]].corr()['Työttömyysaste'].iloc[1:].abs().sort_values(ascending=False)
correlations_abs_asc_sv = data_sv[data_sv.columns[:-4]].corr()['Työttömyysaste'].iloc[1:].abs().sort_values(ascending=True)


main_classes_sv = sorted([c for c in data_sv.columns[:-4] if len(c.split()[0])==2])
second_classes_sv = sorted([c for c in data_sv.columns[:-4] if c.split()[0].count('.')==1])
third_classes_sv = sorted([c for c in data_sv.columns[:-4] if c.split()[0].count('.')==2])
fourth_classes_sv = sorted([c for c in data_sv.columns[:-4] if c.split()[0].count('.')==3])
fifth_classes_sv = sorted([c for c in data_sv.columns[:-4] if c.split()[0].count('.')==4])

feature_options_sv = [{'label':c, 'value':c} for c in data_sv.columns[1:-4]]
corr_desc_options_sv = [{'label':c, 'value':c} for c in correlations_desc_sv.index]
corr_asc_options_sv = [{'label':c, 'value':c} for c in correlations_asc_sv.index]
corr_abs_desc_options_sv = [{'label':c, 'value':c} for c in correlations_abs_desc_sv.index]
corr_abs_asc_options_sv = [{'label':c, 'value':c} for c in correlations_abs_asc_sv.index]
main_class_options_sv = [{'label':c, 'value':c} for c in main_classes_sv]
second_class_options_sv = [{'label':c, 'value':c} for c in second_classes_sv]
third_class_options_sv = [{'label':c, 'value':c} for c in third_classes_sv]
fourth_class_options_sv = [{'label':c, 'value':c} for c in fourth_classes_sv]
fifth_class_options_sv = [{'label':c, 'value':c} for c in fifth_classes_sv]



initial_options_sv = feature_options_sv
example_basket_1 = [c for c in ['01.1.3 Fisk och skaldjur',
 '01.1.3.3 Torkad, rökt eller saltad fisk och skaldjur',
 '01.1.6.3.1 Frysta frukter och bär',
 '01.1.9.3.1 Babymat',
 '01.1.9.5.1 Övriga livsmedel, ej tidigare nämnda',
 '02.1.2.1.1 Druvviner',
 '03.1 Kläder',
 '03.1.2.2 Kläder för damer',
 '03.1.2.2.1 Kappor för damer',
 '03.1.2.4 Babykläder (från 0 till 2 år)',
 '03.2.1.3 Barnskor',
 '05.2.0.2.3 Lakan, örngott och påslakan',
 '05.6.1.1.1 Tvättmedel',
 '06.1.3 Behandlingsapparater och -utrustning',
 '06.1.3.1 Glasögon ock kontaktlinser',
 '07.3.1.1.1 Inrikes tågtrafik',
 '09 REKREATION OCH KULTUR',
 '09.4.2.3.3 Kabel- och betal-TV-avgifter',
 '09.6.0 Paketresor',
 '12.1.3.3.3 Kropps-, hand- och hårkrämer'] if c in data_sv.columns]

example_basket_2 = [c for c in data_sv.columns if c.split()[0] in ['01.1.3',
  '01.1.4.5',
  '01.1.9',
  '02.1.1.2.1',
  '02.1.2.4',
  '03.2.1.2',
  '03.2.1.3',
  '04.1.2',
  '04.1.2.2.2',
  '04.4.1.1',
  '04.5.5',
  '05.3.1.1.3',
  '05.4.0.1.3',
  '06.1.1.4',
  '07.3.1.1.1',
  '07.3.4.1',
  '08.2.0.2',
  '09.1.4.1.2',
  '10.2',
  '11.1.1.1.3',
  '11.1.1.1.8',
  '12.4.0']]



example_basket_3 = [c for c in data_sv.columns if c.split()[0] in ['01.1.1.2.1',
 '01.1.9.3',
 '01.2.2.3',
 '04.5.3.1.1',
 '05.3',
 '05.4.0',
 '05.4.0.4.1',
 '06.2.3.3',
 '07.3.1',
 '07.3.4',
 '09.1.3.1',
 '09.1.3.1.1',
 '09.2.1.3',
 '09.4.1',
 '09.4.1.1.2',
 '09.5.2.1.2',
 '11.2.0.2',
 '12',
 '12.1.1.1.1',
 '12.1.3.2',
 '12.3.2.2',
 '12.7.0.1']]

example_basket_1_options = [{'label':c, 'value':c} for c in example_basket_1]
example_basket_2_options = [{'label':c, 'value':c} for c in example_basket_2]
example_basket_3_options = [{'label':c, 'value':c} for c in example_basket_3]

initial_features_sv = example_basket_1



def layout():
    
    return html.Div([dbc.Container(fluid=True, className = 'dbc', children=[
        html.Br(),        
        dbc.Row(
            [
                
                
                dbc.Col([
                    
                    html.Br(),   
                    html.H1('Skev Phillips',
                             style=h1_style
                            ),
                  
                    html.H2('Prognos för arbetslösheten i Finland med förändringar i konsumentpriser',
                            style=h2_style),
                    
                    html.P('Välj önskad flik genom att klicka på rubrikerna nedan. '
                            'Knapparna i övre vänstra hörnet visar snabb hjälp '
                            'Du kan också ändra färgschemat på sidan.',
                           style = p_center_style)
                    ],xs =12, sm=12, md=12, lg=9, xl=9)
        ], justify='center'),
        html.Br(),
        
        html.Div(id = 'hidden_store_div_sv',
                 children = [
                    
                    dcc.Store(id = 'features_values_sv',data={f:0.0 for f in initial_features_sv}),
                    dcc.Store(id = 'change_weights_sv'), 
                    dcc.Store(id = 'method_selection_results_sv'),
                    dcc.Store(id ='shap_data_sv'),
                    dcc.Store(id ='local_shap_data_sv'),
                    dcc.Store(id = 'test_data_sv'),
                    dcc.Store(id = 'forecast_data_sv'),
                    dcc.Store(id = 'forecast_shap_data_sv'),
                    dcc.Store(id = 'local_forecast_shap_data_sv'),
                    dcc.Download(id='forecast_download_sv'),
                    dcc.Download(id='test_download_sv')
        ]),
        
        dbc.Tabs(id ='tabs_sv',
                 children = [
            
            
            
            dbc.Tab(label='Inledning och instruktioner',
                    tab_id = 'ohje_sv',
                    tabClassName="flex-grow-1 text-center",
                    tab_style = {'font-size':"1.5625rem",
                                 'font-weight': 'bold', 
                                 # #'font-family':'Cadiz Semibold'
                                 },
                    style = {
                            
                             
                               
                                "height":"100vh",

                            "overflowY": "scroll"
                        },
                    
                    children = [
                        html.Div(children = [
                        dbc.Row(justify='center',
                                # style = {'margin' : '10px 10px 10px 10px'},
                                children=[
                            
                          dbc.Col(xs =12, sm=12, md=12, lg=8, xl=8, children =[
                         
                                  html.Br(),
                                  
                                  html.Blockquote('“The world is its own best model.”', 
                                        style = {
                                            'text-align':'center',
                                            'font-style': 'italic', 
                                            #'font-family':'Messina Modern Book', 
                                              'font-size':p_font_size
                                            }),
                                  html.A([html.P('Rodney Brooks', 
                                        
                                        style={
                                            'textAlign':'center',
                                            'font-style': 'italic',
                                            #'font-family':'Messina Modern Book', 
                                              'font-size':"1rem"
                                            })], href = 'https://www.technologyreview.com/2019/08/21/133411/rodney-brooks/', target="_blank"),
                                  

                                  html.Br(),
                                  html.H3('Inledning',style=h3_style
                                          ),
                                  html.P("Stigande priser påverkar många finländska konsumenters liv och räntorna väntas stiga i takt med att centralbankerna känner press på att hålla inflationen tillbaka. Personer med bolån står inför svårare tider i takt med att räntorna stiger. Sammantaget verkar det som om det inte finns något bra med inflationen. Men är det verkligen så?",
                                        style = p_style),
                                  html.P("Inflationen har också ett positivt inslag, vilket är en nedgång i arbetslösheten på kort sikt. Denna så kallade Phillipskurvan är en empirisk observation som gjordes på 1950-talet av en ekonom Alban William Phillips. Observationen visar att det finns en konflikt mellan inflation och arbetslöshet på kort sikt. Denna idé presenteras i grafen nedan, som beskriver inflation och arbetslöshet råtta e samtidigt i Finland. Den fallande logaritmiska trendlinjen motsvarar Phillips observation. ",
                                        style = p_style),
                                 html.P("(Fortsätter efter diagrammet)",style={
                                             'font-style':'italic',
                                             'font-size':p_font_size,
                                            'text-align':'center'}
                                     ),
                                  html.H3("Månatlig Phillipskurva i Finlands ekonomi", 
                                          style=h3_style),
                                  html.H4("Källa: Statistikcentralen", 
                                          style=h4_style)
                                  
                                  ])
                                  ]
                                ),

                                  dbc.Row([
                                      dbc.Col([
                                             
                                               html.Div(
                                                    [dcc.Graph(id = 'phillips_sv',
                                                                figure= sv_draw_phillips_curve(),
                                                                config = config_plots_sv
                                                          )
                                                    ],
                                                   style={'textAlign':'center'}
                                                   
                                                 
                                                      )
                                             
                                            ],xs =12, sm=12, md=12, lg=8, xl=8)
                                  ], justify = 'center', 
                                      # style = {'margin' : '10px 10px 10px 10px'}
                                    ),
                                 
                                  dbc.Row([
                                     
                                      dbc.Col([
                                          html.Br(),
                                          html.P("Diagrammet ovan visar arbetslöshet och inflation vid olika tidpunkter med ett spridningsmönster. Den anger också det senaste datumet för både inflation och arbetslöshet. Håll musen över prickarna för att se värden och tid. Den logaritmiska trendlinjen representerar en empirisk observation gjord av Phillips, där det finns ett negativt samband mellan inflation och arbetslöshet. Indikatorn på inflationsutvecklingskurvan minskar i takt med att arbetslösheten ökar." ,
                                                style = {
                                                    'text-align':'center', 
                                                    #'font-family':'Messina Modern Book', 
                                                      'font-size':"1.1rem"
                                                    }),
                                          html.P("Det finns flera teorier som förklarar Phillipskurvan, beroende på om katalysatorn för fenomenet är en förändring i prisnivå eller arbetslöshet. Mot bakgrund av den höga arbetslösheten kräver utbuds- och efterfrågelagstiftningen lägre priser för att förbättra varuförsäljningen. Å andra sidan ökar produktionen på kort sikt i takt med att priserna stiger i takt med att producenterna ökar varuproduktionen för att uppnå högre marginaler. Detta leder till lägre arbetslöshet, eftersom ökad produktion leder till nya rekryteringar som kan göras till den arbetslösa befolkningen. Å andra sidan, när arbetslösheten är låg, finns det ett tryck på efterfrågan på arbetskraft på marknaden, vilket ökar lönerna. Löneökningarna leder å andra sidan till en ökning av den totala prisnivån, eftersom varuleverantörer kan begära högre priser på sina produkter och tjänster.",
                                                 style = p_style),
                                          html.P("Phillipskurvan kan också observeras intuitivt. År 2015 låg arbetslösheten till exempel flera gånger över 10% och inflationen låg kvar på noll och ibland till och med negativ. Coronaviruschocken 2020 orsakade en ökning av arbetslösheten, men bränslepriserna var till exempel mycket lägre än under första halvåret 2022. Det finns flera tillfällen i historien när arbetslöshet och inflation har förändrats i olika riktningar. Det har också funnits undantag, t.ex. under 1970-talets oljekris var båda höga, men om man tittar på historien finns det flera perioder då Phillipskurvan varit giltig. Det är viktigt att komma ihåg att Phillipskurvan endast är giltig på kort sikt, i vilket fall förutsägelser baserade på den inte bör göras för länge.", style = p_style),
                                          # html.P('Kyseessä on tunnettu taloustieteen teoria, jota on tosin vaikea soveltaa, koska ei ole olemassa sääntöjä, joilla voitaisiin helposti ennustaa työttömyyttä saatavilla olevien inflaatiota kuvaavien indikaattorien avulla. Mikäli sääntöjä on vaikea formuloida, niin niitä voi yrittää koneoppimisen avulla oppia historiadataa havainnoimalla. Voisiko siis olla olemassa tilastollisen oppimisen menetelmä, joka pystyisi oppimaan Phillipsin käyrän historiadatasta? Mikäli tämänlainen menetelmä olisi olemassa, pystyisimme ennustamaan lyhyen aikavälin työttömyyttä, kun oletamme hyödykkeiden hintatason muuttuvan skenaariomme mukaisesti.',
                                                # style=p_style), 
                                          html.P("Phillipskurvan har ett eget kapitel i Finlands Handelshögskolans examensbok för ekonomi och följer en av de tio grundläggande principerna för ekonomi.",
                                                style=p_style),
                                          html.Br(),
                                                                            
                                          html.H3("Ekonomins tionde grundprincip:",style = {'text-align':'center', 
                                                                                                   #'font-family':'Messina Modern Semibold',
                                                                                                   'font-style': 'italic', 
                                                                                                   'font-weight': 'bold', 
                                                                                                   'font-size':"2.125rem"}),
                                          
                                          html.Blockquote("Det finns en konflikt mellan arbetslöshet och inflation på kort sikt. Full sysselsättning och stabila prisnivåer är svåra att uppnå samtidigt.", 
                                                style = {
                                                    'text-align':'center',
                                                    'font-style': 'italic', 
                                                    #'font-family':'Messina Modern Book', 
                                                      'font-size':p_font_size
                                                    }),
                                          html.P('Matti Pohjola, 2019, Taloustieteen oppikirja, sidan 250, ISBN:978-952-63-5298-5', 
                                                style={
                                                    'textAlign':'center',
                                                    #'font-family':'Messina Modern Book', 
                                                      'font-size':"1rem"
                                                    }),
        
                                          html.Br(),
                                          html.P("Phillipskurvan är dock bara en teori som är lätt att observera genom att analysera historiska data, även om Phillipskurvan inte alltid materialiseras, skulle det ändå vara möjligt att använda förhållandet mellan inflation och arbetslöshet för att förutsäga arbetslöshet?",
                                                 style=p_style),
                                          html.P("Det är svårt att omvandla Phillipskurvan till en matematisk ekvation där vi genom att investera inflationen kan beräkna arbetslösheten. Detta gav mig idén att det kunde finnas en maskininlärningsmetod som kunde lära sig rådande lagar mellan inflation och arbetslöshet. Inflation är den årliga förändringen i konsumentprisindexet. Konsumentprisindexet består av flera gemensamma Det är viktigt att det finns ett stort utbud av produkter som uttrycker konsumtionsbehoven i samhället vid den tiden. Kan vissa av dessa varor påverka mer än andra? Skulle endast basprisindexet, arbetslöshetsgraden föregående månad och viss information om säsongsvariationer i arbetslösheten vara tillräckliga för prognoserna? Vilka varor ska jag välja? Vilken algoritm, vilka hyperparametrar? Jag lekte med tanken att det kan finnas någon kombination av råvarumix och metodik som kan ge åtminstone en tillfredsställande kortsiktig prognos. Därför ville jag skapa en app som vem som helst, oavsett akademisk bakgrund, kunde göra experiment som detta. ",
                                                 style=p_style),
                                          html.P("Efter flera iterationer blev resultatet ett program där man kan utforma en varukorg, välja en maskininlärningsmetod, testa kombinationen av dessa för att förutsäga redan realiserade värden och slutligen göra förutsägelser. Utöver det byggde jag upp möjligheten att justera hyperparametrar för maskininlärningsalgoritmer och använda huvudkomponentanalys för att eliminera irrelevanta egenskaper. ",
                                                 style=p_style),
                                          html.P("Nästa problem var svårigheten att tolka modellerna. Det finns en allmänt känd konflikt mellan noggrannhet och tolkning i maskininlärning. Enklare modeller, som linjär regression, är lättare att tolka än exempelvis slumpmässiga skogar, men slumpmässiga skogar kan ge bättre förutsägelser. Detta skapar ett problem med svarta lådor som måste lösas för att göra metoden mer trovärdig och transparent och därmed kunna användas i allmän planering och beslutsfattande. ",
                                                 style=p_style),
                                          html.P("Jag lade till Shapley-värden som agnostisk funktionalitet i programmet. Shapley-värden är ett koncept baserat på spelteori som bygger på beräkning av spelares bidrag i kooperativa spel (t.ex. enskilda spelares bidrag i en fotbollsmatch till resultatet). I maskininlärning används en liknande modell för att uppskatta prognosernas bidrag. Ett mer intressant forskningsproblem än att förutsäga arbetslösheten visade sig vara det som varor eller varukombinationer lyckas förutsäga arbetslösheten bäst! ",
                                                 style =p_style),
                                          html.P("Syftet var att hitta Phillips observation med hjälp av maskininlärning och kanske hitta formeln för Phillipskurvan.Fördelen med maskininlärning kommer från det faktum att det producerar sin egen syn på fenomenet genom att observera data som beskriver det. Det behöver bara observeras korrekt och tillräckligt ofta.",
                                                 style =p_style),
                                          html.P("Resultatet kan därför vara något annat än lagen i Phillipskurvan, eftersom maskininlärningsalgoritmer kan lära sig en helt annan dold lag. Kopplingen mellan komponenterna i inflationen och förändringen i arbetslösheten utnyttjas helt enkelt. Formeln som lärs är inte Phillipskurvan, utan en annan regel, en sned Phillipsobservation, möjligen en omvänd Phillips, eller en cur ve som innehåller dalar och kullar. Dessutom görs prognosen inte bara med prisindex, utan också med arbetslösheten för föregående månad och månadernas numeriska värden (t.ex. juni är 6). Med andra ord är Phillipskurvan i detta fall bara en teoretisk utgångspunkt, en gnista för forskning och en bakgrund för att på något sätt förklara arbetslösheten med inflationens komponenter. ",style=p_style),
                                          html.P("Så jag kodade denna kombination av data science blogg och applikation (jag vet inte vad de kallar det, 'blapp', 'bloglication'...), som använder uppgifterna från Statistikcentralens Statfin-gränssnitt från Konsumentprisindexet per råvara i förhållande till basåret 2010 och arbetslöshetstalet i Finland månadsvis. Varugrupper för vilka uppgifter inte finns tillgängliga för hela perioden har tagits bort från datauppsättningarna. Det finns fortfarande hundratals råvaror och varugrupper kvar att bygga prognoskomponenter. Icke-linjära maskininlärningsalgoritmer har valts som algoritmer alternativ eftersom de är bättre lämpade för detta fall än linjära modeller. ",
                                                 style =p_style),
                                          html.P("Ansökan är uppdelad i avsnitt, som uttrycks i flikar. Målet är att gå från vänster till höger i flikar och alltid iterera genom att ändra råvaror och metod. Ansökan grundar sig på hypotesen att månadsförändringen i arbetslöshetstalet för varje månad kan förklaras med arbetslöshetstalet för föregående månad, innevarande månad och rådande indexvärden för den valda varukorgen. När den månatliga arbetslöshetsförändringen erhålls genom en algoritm för maskininlärning kan den kombineras med föregående månads arbetslöshetstal, vilket resulterar i en prognos för innevarande månads arbetslöshetstal. Denna idé tillämpas rekursivt så långt man vill förutsäga. Denna lösning kräver att vi gör några antaganden om hur råvaruprisindex kommer att förändras. För att utvärdera det krävs en liten utforskande analys, till vilken det finns ett särskilt avsnitt i denna ansökan. Eftersom prognosen omfattar antaganden är det förmodligen bättre att tillämpa prognosmodellen under en kort tidsperiod (som Phillipskurvan ursprungligen var avsedd).",
                                                 style =p_style),
                                          html.P("Den slutliga prognosen bygger därför på användarindata antaganden om månatlig förändringstakt för utvalda varor. Du kan justera förändringstakten för varje vara individuellt, dra nytta av tidigare månaders genomsnitt eller ställa in samma förändringstakt för alla varor. Testerna utförs utifrån antagandet att de tidigare indexvärdena realiserades som sådana. nd dokumentation, test och prognosresultat kan exporteras som en Excel-fil för andra möjliga användningsfall. ",
                                                 style =p_style),
                                          
                                          
                                           html.H3('Instruktioner',
                                                   style=h3_style
                                                   ),
                                           
                                           html.P("Här är några instruktioner om hur du använder appen. Varje avsnitt ska göras i motsvarande ordning. Välj fliken för att slutföra varje steg. Flikarna har ännu mer detaljerade instruktioner. Dessutom finns det en knapp i appens övre vänstra hörn som öppnar snabbhjälpen på valfri flik.", 
                                                    style = p_style),
                                           html.Br(),
                                           html.P("1. Val av varor. Välj de varugrupper du vill ha från rullgardinsmenyn. Dessa används för att förutsäga arbetslöshet.", 
                                                    style = p_style),
                                          html.P("2. Exploratorisk analys. Du kan se och analysera de varor du valt. Om det behövs, gå tillbaka till föregående steg och ta bort eller lägga till tillgångar.",
                                                    style = p_style),
                                          html.P("3. Metodval. I det här avsnittet väljer du maskininlärningsalgoritmen och justerar hyperparametarna. Dessutom kan du välja om du vill använda huvudkomponentanalysen och hur mycket variation du ska lagra.",
                                                    style = p_style),
                                         html.P("4. Test. Du kan välja den tidsperiod som modellen vill förutsäga tidigare. Det gör att du kan uppskatta hur prognosmodellen skulle ha fungerat för redan realiserade data. I det här avsnittet kan du också se hur mycket varje prognosfunktion bidrog till att göra prognosen.",
                                                    style = p_style),
                                         html.P("5. Prognos. Du kan nu använda din valda metod för att göra en förutsägelse för framtiden. Välj längden på prognosen och klicka på prognosen. Du kan sedan exportera prognosen till Excel också. Prognosen baseras på förändringsvärdena för de varor du anger. Som i testavsnittet kan du också se vilka varor och egenskaper som påverkar förändringen i arbetslöshetstalet och vilka varuprisförändringar som bidrar till månatliga förändringar i arbetslöshetstalet.",
                                                    style = p_style),
                                          
                                          html.Br(),
                                          
                                          # html.H3('Pääkomponenttianalyysistä',
                                          #         style=h3_style
                                          #         ),
                                          
                                          # html.P('Pääkomponenttianalyysilla (englanniksi Principal Component Analysis, PCA) pyritään minimoimaan käytettyjen muuttujien määrää pakkaamalla ne sellaisiin kokonaisuuksiin, jotta hyödynnetty informaatio säilyy. Informaation säilyvyyttä mitataan selitetyllä varianssilla (eng. explained variance), joka tarkoittaa uusista pääkomponenteista luodun datan hajonnan säilyvyyttä alkuperäiseen dataan verrattuna. Tässä sovelluksessa selitetyn varianssin (tai säilytetyn variaation) osuuden voi valita itse, mikäli hyödyntää PCA:ta. Näin saatu pääkomponenttijoukko on siten pienin sellainen joukko, joka säilyttää vähintään valitun osuuden alkuperäisen datan hajonnasta. Näin PCA-algoritmi muodostaa juuri niin monta pääkomponenttia, jotta selitetyn varianssin osuus pysyy haluttuna.',
                                          #         style=p_style),
                                          # html.P('PCA on yleisesti hyödyllinen toimenpide silloin, kun valittuja muuttujia on paljon, milloin on myös mahdollista, että osa valituista muuttujista aiheuttaa datassa kohinaa, mikä taas johtaa heikompaan ennusteeseen.  Pienellä määrällä tarkasti harkittuja muuttujia PCA ei ole välttämätön.',
                                          #         style=p_style),
                                          # html.Br(),
                                                                                    
                                       
                                          # html.Br(),
                                          
                                          html.H3('Ansvarsfriskrivning',
                                                  style=h3_style),
                                          
                                          html.P("Webbplatsen och dess innehåll tillhandahålls kostnadsfritt och i mån av tillgång. Detta är en tjänst som tillhandahålls av en privat person, inte en officiell tjänst eller en ansökan gjord för kommersiella ändamål. Användningen av informationen som erhållits från webbplatsen är beslutsfattarnas ansvar. Tjänsteleverantören ansvarar inte för förlust, tvister, anspråk, kostnad eller skada, oavsett eller på något sätt, direkt eller indirekt från användningen av Tjänsten. Observera att denna sida fortfarande är under utveckling. ",
                                                  style=p_style),
                                          html.P("Denna webbplats använder endast nödvändiga cookies och användares personuppgifter samlas inte in för något ändamål.",
                                                  style=p_style),
                                          html.A([html.P("Se en tredjepartsrapport om efterlevnad av GDPR.", style = p_center_style)],
                                                 href = '/assets/report-skewedphillipsherokuappcom-11629005.pdf',
                                                 target = '_blank'),
                                          html.Br(),
                                          html.H3("Webbläsare som stöds och tekniska begränsningar",
                                                  style=h3_style),
                                          
                                          html.P("Appen har testats för att fungera med Google Chrome, Edge och Mozilla Firefox. I Internet Explorer fungerar appen inte. Opera, Safari och andra webbläsare har inte testats.",
                                                  style=p_style),
                                          html.P("Applikationen kan också laddas ner så kallad fristående version, så att den kan startas utan en webbläsare, t.ex. Windows eller Android. I Google Chrome, till höger om webbläsarens adressfält, bör det finnas en ikon som du kan ladda ner appen från. När du har laddat ner appen kan du hitta den på din egen enhet.",
                                                  style=p_style),
                                         
                                          html.Div(children = [
                                              html.H3('Referenser', 
                                                      style = h3_style),
                                              html.P("Här finns också en förteckning över datakällor och ytterligare läsningar relaterade till de ämnen som beskrivs.",
                                                     style =p_style),
                                              
                                              html.Label(['Statistikcentralens avgiftsfria statistikdatabaser: ', 
                                                        html.A('Arbetskraftsundersökningens viktigaste nyckeltal samt säsongrensade serier och de säsong- och slumpvariationsrensade trenderna', href = "https://statfin.stat.fi/PxWeb/pxweb/sv/StatFin/StatFin__tyti/statfin_tyti_pxt_135z.px/",target="_blank")
                                                      ],style=p_style),
                                              html.Br(),
                                              html.Label(['Statistikcentralens avgiftsfria statistikdatabaser: ', 
                                                        html.A('Konsumentprisindex (2010=100), månadsuppgifter', href = "https://statfin.stat.fi/PxWeb/pxweb/sv/StatFin/StatFin__khi/statfin_khi_pxt_11xd.px/",target="_blank")
                                                      ],style=p_style),
                                              html.Br(),
                                              html.Label(['Statistikcentralens avgiftsfria statistikdatabaser: ', 
                                                        html.A('Årsförändring av konsumentprisindexet, månadsuppgifter', href = "https://statfin.stat.fi/PxWeb/pxweb/sv/StatFin/StatFin__khi/statfin_khi_pxt_122p.px/",target="_blank")
                                                      ],style=p_style),
                                              html.Br(),
                                              html.Label(['Statistikcentralen: ', 
                                                       html.A('Begrepp', href = "https://www.stat.fi/meta/kas/index_sv.html",target="_blank")
                                                      ],style=p_style),
                                              html.Br(),
                                              html.Label(['Wikipedia: ', 
                                                        html.A('Phillipskurvan', href = "https://sv.wikipedia.org/wiki/Phillipskurvan",target="_blank")
                                                      ],style=p_style),
                                              html.Br(),
                                              html.Label(['Raha ja Talous: ', 
                                                        html.A('Phillipsin käyrä (på finska) ', href = "https://rahajatalous.wordpress.com/2012/11/15/phillipsin-kayra/",target="_blank")
                                                      ],style=p_style),
                                              html.Br(),
                                              html.Label(['Cato Journal: ', 
                                                        html.A('The Phillips Curve: A Poor Guide for Monetary Policy', href = "https://www.cato.org/cato-journal/winter-2020/phillips-curve-poor-guide-monetary-policy",target="_blank")
                                                      ],style=p_style),
                                              html.Br(),
                                              html.Label(['Ribeiro, Marco & Singh, Sameer & Guestrin, Carlos. (2016). : ', 
                                                        html.A('Model-Agnostic Interpretability of Machine Learning', href = "https://arxiv.org/pdf/1606.05386.pdf",target="_blank")
                                                      ],style=p_style),
                                              html.Br(),
                                              html.Label(['Lundberg, Scott & Lee, Su-In. (2017). : ', 
                                                        html.A('A Unified Approach to Interpreting Model Predictions', href = "https://www.researchgate.net/publication/317062430_A_Unified_Approach_to_Interpreting_Model_Predictions",target="_blank")
                                                      ],style=p_style),
                                              html.Br(),
                                              html.Label(['Wikipedia: ', 
                                                        html.A('Shapley values ', href = "https://en.wikipedia.org/wiki/Shapley_value",target="_blank")
                                                      ],style=p_style),
                                              html.Br(),
                                              html.Label(['Wikipedia: ', 
                                                        html.A('Principalkomponentanalys', href = "https://sv.wikipedia.org/wiki/Principalkomponentanalys",target="_blank")
                                                      ],style=p_style),
                                              html.Br(),
                                              html.Label(['Wikipedia: ', 
                                                        html.A('Pearsons korrelationskoefficient (på engelska)', href = "https://en.wikipedia.org/wiki/Pearson_correlation_coefficient",target="_blank")
                                                      ],style=p_style),
                                              html.Br(),
                                                html.Label(['Scikit-learn: ', 
                                                        html.A('Regression', href = "https://scikit-learn.org/stable/supervised_learning.html#supervised-learning",target="_blank")
                                                        ],style=p_style),
                                                html.Br(),
    
                                          ]),
                                          html.Br(),
                                          html.Br(),
                                          html.H3('Skriven av', style = h3_style),
                                          
                                          html.Div(style = {'textAlign':'center'},children = [
                                              html.I('Tuomas Poukkula', style = p_center_style),
                                         
                                              html.Br(),
                                              html.P("Data Scientist",
                                                     style = p_center_style),
                                              # html.P("Gofore Ltd",
                                              #        style = p_center_style),
                                              # html.A([html.P('Kontakt via e-post',style = p_center_style)],
                                              #        href = 'mailto:tuomas.poukkula@gofore.com?subject=Phillips: Frågor och svar',
                                              #        target='_blank')
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
                                                       src='/assets/256px-Linkedin_icon.png',
                                                    
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
                                                       src='/assets/Twitter-logo.png',
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
                                      # dbc.Col([
                                      # html.Div([
                                      #     html.A([
                                      #          html.Img(
                                      #              src='/assets/gofore_logo_orange.svg',
                                      #              style={
                                                     
                                      #                    'text-align':'center',
                                      #                    'float' : 'center',
                                      #                    'position' : 'center',
                                      #                    'padding-top' : '20px',
                                      #                     'padding-bottom' : '20px'
                                      #              }
                                      #              )
                                      # ], href='https://gofore.com/',target = '_blank', style = {'textAlign':'center'})
                                      # ],style={'textAlign':'center'})],xs =12, sm=12, md=12, lg=6, xl=6)
                                      
                                      ],
                                  justify='center'),
                                  dbc.Row([
                                     
                                      dbc.Col([
                                          html.Div(style = {'text-align':'center'},children = [
                                              
                                              html.Label(['Program på ', 
                                                      html.A('GitHub', href='https://github.com/tuopouk/skewedphillips')
                                                      ],style=p_style)
                                      ])
                                          ],xs =12, sm=12, md=12, lg=6, xl=6)
                                      ],
                                      justify ='center',
                                      
                                      ),
                                  html.Br()
                                  ]),
                        html.Br(),
                        footer
                 
                        
                        
                        
                        ]

),
            
            dbc.Tab(label ='Val av varor',
                    tab_id ='feature_tab_sv',
                     tabClassName="flex-grow-1 text-center",
                    tab_style = {'font-size':"1.5625rem",
                                 'font-weight': 'bold', 
                                 #'font-family':'Cadiz Semibold'
                                 },
                  style = {
                          
                           
                             
                              "height":"100vh",

                          "overflowY": "scroll"
                      },
              children= [
                  
                dbc.Row([
                    dbc.Col([
                        
                        html.Br(),
                        html.P("I detta avsnitt väljs de varor som används för att förutsäga arbetslöshet.",
                                style = p_style),
                        html.P("Du kan välja objekt från menyn nedan, sedan kan du justera deras förväntade månatliga förändring genom att ange ett nummer i rutorna nedan.",
                                style = p_style),
                        html.P("Du kan också justera samma månatliga förändring för alla tillgångar eller dra nytta av genomsnittet för faktiska månatliga förändringar.",
                                style = p_style),
                        html.P("Du kan välja eller sortera produktmenyn från rullgardinsmenyn ovan. Du kan välja antingen alfabetisk ordning, korrelationsordning (enligt Pearson korrelationskoefficient) eller avgränsning enligt Statistikcentralens varuhierarki. I korrelationsordningen avses här korrelationskoefficienten mellan värdena i prisindexet för varje vara och samtidigt arbetslöshetstalet, beräknat med Pearson-metoden. Dessa kan sorteras i fallande eller stigande ordning antingen efter det faktiska värdet (högsta positiva - lägsta negativa) eller det absoluta värdet (utan +/-). ",
                                style = p_style),
                        html.P("Det finns också några exempelkorgar som du kan komma åt genom att välja en i rullgardinsmenyn och använda kommandot Välj alla.",
                                style = p_style),
                    ],xs =12, sm=12, md=12, lg=9, xl=9)
                ], justify = 'center'),
                
        
                dbc.Row(children = [
                        
                        html.Br(),
                    
                        dbc.Col(children=[
                           
                            html.Br(),
                            html.Br(),
                            html.H3("Välj funktioner från rullgardinsmenyn",
                                    style=h3_style),
                            
                            dbc.DropdownMenu(id = 'sorting_sv',
                                              #align_end=True,
                                              children = [
                                                 
                                                  dbc.DropdownMenuItem("Alfabetisk ordning", id = 'alphabet_sv',style={
                                                      'font-size':"0.9rem", 
                                                      #'font-family':'Cadiz Book'
                                                      }
                                                      ),
                                                  dbc.DropdownMenuItem("Korrelation (fallande)", id = 'corr_desc_sv',style={
                                                      'font-size':"0.9rem", 
                                                      #'font-family':'Cadiz Book'
                                                      }
                                                      ),
                                                  dbc.DropdownMenuItem("Korrelation (stigande)", id = 'corr_asc_sv',style={
                                                      'font-size':"0.9rem", 
                                                      #'font-family':'Cadiz Book'
                                                      })
                                                      ,
                                                  dbc.DropdownMenuItem("Absolut korrelation (fallande)", id = 'corr_abs_desc_sv',style={
                                                      'font-size':"0.9rem", 
                                                      #'font-family':'Cadiz Book'
                                                      }
                                                      ),
                                                  dbc.DropdownMenuItem("Absolut korrelation (stigande)", id = 'corr_abs_asc_sv',style={
                                                      'font-size':"0.9rem", 
                                                      #'font-family':'Cadiz Book'
                                                      }
                                                      ),
                                                  dbc.DropdownMenuItem("Huvudklasser", id = 'main_class_sv',style={
                                                      'font-size':"0.9rem", 
                                                      #'font-family':'Cadiz Book'
                                                      }
                                                      ),
                                                  dbc.DropdownMenuItem("2. klass", id = 'second_class_sv',style={
                                                      'font-size':"0.9rem", 
                                                      #'font-family':'Cadiz Book'
                                                      }
                                                      ),
                                                  dbc.DropdownMenuItem("3. klass", id = 'third_class_sv',style={
                                                      'font-size':"0.9rem", 
                                                      #'font-family':'Cadiz Book'
                                                      }
                                                      ),
                                                  dbc.DropdownMenuItem("4. klass", id = 'fourth_class_sv',style={
                                                      'font-size':"0.9rem", 
                                                      #'font-family':'Cadiz Book'
                                                      }
                                                      ),
                                                  dbc.DropdownMenuItem("5. klass", id = 'fifth_class_sv',style={
                                                      'font-size':"0.9rem", 
                                                      #'font-family':'Cadiz Book'
                                                      }
                                                      ),
                                                  dbc.DropdownMenuItem("Example basket 1", id = 'example_basket_1_sv',style={
                                                      'font-size':"0.9rem", 
                                                      #'font-family':'Cadiz Book'
                                                      }
                                                      ),
                                                  dbc.DropdownMenuItem("Example basket 2", id = 'example_basket_2_sv',style={
                                                      'font-size':"0.9rem", 
                                                      #'font-family':'Cadiz Book'
                                                      }
                                                      ),
                                                  dbc.DropdownMenuItem("Example basket 3", id = 'example_basket_3_sv',style={
                                                      'font-size':"0.9rem", 
                                                      #'font-family':'Cadiz Book'
                                                      }
                                                      )
                                                 
                                                 
                                                  ],
                                            label = "Absolut korrelation (fallande)",
                                            color="secondary", 
                                            className="m-1",
                                            size="lg",
                                            style={
                                                'font-size':"0.9rem", 
                                                #'font-family':'Cadiz Book'
                                                }
                                            ),
                            
                            html.Br(),
                            dcc.Dropdown(id = 'feature_selection_sv',
                                          options = initial_options_sv,
                                          multi = True,
                                          value = list(initial_features_sv),
                                          style = {'font-size':"1rem", #'font-family':'Cadiz Book'
                                                   },
                                          placeholder = 'Välj råvaror'),
                            html.Br(),
                            
                            dbc.Alert("Välj minst en vara!", color="danger",
                                      dismissable=True, fade = True, is_open=False, id = 'alert_sv', 
                                      style = {'text-align':'center', 'font-size':p_font_size, #'font-family':'Cadiz Semibold'
                                               }),
                            dash_daq.BooleanSwitch(id = 'select_all_sv', 
                                                    label = dict(label = 'Select all',
                                                                 style = {'font-size':p_font_size, 
                                                                          # #'fontFamily':'Cadiz Semibold'
                                                                          }), 
                                                    on = False, 
                                                    color = 'blue'),
                            html.Br(),
                            # html.Div(id = 'selections_div',children =[]),
                            
                
                            ],xs =12, sm=12, md=12, lg=12, xl=12
                        )
                        ],justify='center', 
                    # style = {'margin' : '10px 10px 10px 10px'}
                    ),
                    dbc.Row(id = 'selections_div_sv', children = [
                        
                            dbc.Col([
                                
                                html.H3('Ange en uppskattning av genomsnittlig månadsindexförändring i procent',
                                                        style = h3_style),
                                
                                dash_daq.BooleanSwitch(id = 'averaging_sv', 
                                                        label = dict(label = 'Genomsnitt för användning',style = {'font-size':p_font_size, 
                                                                                                                      # 'font-family':'Cadiz Semibold'
                                                                                                                      }), 
                                                        on = True, 
                                                        color = 'blue')
                                
                                ],xs =12, sm=12, md=12, lg=6, xl=6),
      
                ],justify = 'center', 
                    
                ),
                    html.Br(),
                dbc.Row([
                        dbc.Col([
                            html.Div(id = 'slider_div_sv'),
                            html.Br(),
                            html.Div(id='slider_prompt_div_sv')
                            
                            ], xs =12, sm=12, md=12, lg=9, xl=9)
                        
                        ],justify = 'center', 
                  
                    ),
                dbc.Row(id = 'adjustments_div_sv',
                        justify = 'left', 
                     
                        ),
                html.Br(),
                footer
            ]
            ),


            dbc.Tab(label = 'Förberedande analys',
                    tab_id = 'eda_tab_sv',
                    tabClassName="flex-grow-1 text-center",
                    tab_style = {'font-size':"1.5625rem",
                                 'font-weight': 'bold', 
                                 #'font-family':'Cadiz Semibold'
                                 },
                    style = {
                            
                             
                               
                                "height":"100vh",

                            "overflowY": "scroll"
                        },
                     children = [
                    
                     dbc.Row([
                         
                         dbc.Col([
                             
                              html.Br(),
                              html.P("I det här avsnittet kan du se arbetslöshetstalet och förhållandet mellan de utvalda varugrupperna i konsumentprisindexet och förändringen över tiden. Nedan kan du se hur prisindexet för olika varugrupper korrelerar med varandra och med arbetslöshetstalet. Du kan också se tidsserier av index, inflation och arbetslöshetstal. Korrelationen beskrivs i s baserat på Pearson korrelationskoefficient. ",
                                     style = p_style),
                              html.Br()
                              ],xs =12, sm=12, md=12, lg=9, xl=9)
                         ],
                             justify = 'center'
                             ),
                    
                     dbc.Row([
                                dbc.Col(children = [
                                   
                                        html.Div(id = 'corr_selection_div_sv'),
                                        html.Br(),
                                        html.Div(id = 'eda_div_sv', 
                                                 children =[
                                                     html.Div([dbc.RadioItems(id = 'eda_y_axis_sv', 
                                                                 options = [{'label':'Arbetslöshet (%)','value':'Työttömyysaste'},
                                                                           {'label':'Förändring av månadsarbetslösheten (% enheter)','value':'change'}],
                                                                 labelStyle={'display':'inline-block', 'padding':'10px','margin':'10px 10px 10px 10px','font-size':"1.1rem",
                                                                             #'font-family':'Cadiz Book'
                                                                             'font-weight': 'bold'
                                                                             },
                                                                 className="btn-group",
                                                                 inputClassName="btn-check",
                                                                 labelClassName="btn btn-outline-secondary",
                                                                 labelCheckedClassName="active",
                                                               
                                                                 value = 'Työttömyysaste'
                                                               ) ],
                                                              style={'textAlign':'center'}), 
                                                     html.Div(id = 'commodity_unemployment_div_sv')
                                                     ]
                                                 ),
                                        html.Br(),
                                       
                                   
                                    ], xs =12, sm=12, md=12, lg=6, xl=6
                                ),
                                dbc.Col(children = [
                                   
                                        html.Div(id = 'feature_corr_selection_div_sv'),
                                        html.Br(),
                                        html.Div(id = 'corr_div_sv'),
                                   
                                    ], xs =12, sm=12, md=12, lg=6, xl=6, align ='start'
                                )
                         ],justify='center', 
                          # style = {'margin' : '10px 10px 10px 10px'}
                     ),

                    # html.Br(),
                     dbc.Row(id = 'timeseries_div_sv',children=[
                        
                         dbc.Col(xs =12, sm=12, md=12, lg=6, xl=6,
                                 children = [
                                     html.Div(id = 'timeseries_selection_sv'),
                                     html.Br(),
                                     html.Div(id='timeseries_sv')
                                     ]),
                         dbc.Col(children = [
                             html.Div(style={'textAlign':'center'},children=[
                                 html.Br(),
                                 html.H3("Figuren nedan visar månadsarbetslösheten och inflationen i Finland.",
                                        style = h3_style),
                                 
                                 html.Div(id = 'employment_inflation_div_sv',
                                          
                                          children=[dcc.Graph(id ='employment_inflation_sv',
                                                     figure = go.Figure(data=[go.Scatter(x = data_sv.index,
                                                                               y = data_sv.Työttömyysaste,
                                                                               name = 'Arbetslöshet',
                                                                               hovertemplate = '%{x}'+'<br>%{y}',
                                                                               mode = 'lines',
                                                                               marker = dict(color ='#000000')),
                                                                    go.Scatter(x = data_sv.index,
                                                                               y = data_sv.Inflaatio,
                                                                               name = 'Inflation',
                                                                               hovertemplate = '%{x}'+'<br>%{y}',
                                                                               mode ='lines',
                                                                               marker = dict(color = '#CC79A7'))],
                                                              layout = go.Layout(title = dict(text = 'Arbetslöshet och inflation per månad<br>{} - {}'.format(data_sv.index.strftime('%B %Y').values[0],data_sv.index.strftime('%B %Y').values[-1]),
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
                                                                                 hoverlabel=dict(font=dict(
                                                                                      family='Cadiz Book',
                                                                                     size=18)),
                                                                                 legend = dict(orientation = 'h',
                                                                                                xanchor='center',
                                                                                                yanchor='top',
                                                                                                x=.5,
                                                                                                y=1.04,
                                                                                               font=dict(
                                                                                      size=12,
                                                                                      family='Cadiz Book'
                                                                                     )),
                                                                                 xaxis = dict(title=dict(text = 'Tid',
                                                                                                         font=dict(
                                                                                                              size=18, 
                                                                                                             family = 'Cadiz Semibold'
                                                                                                             )), 
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
                                                                                                           label="1m",
                                                                                                           step="month",
                                                                                                           stepmode="backward"),
                                                                                                      dict(count=6,
                                                                                                           label="6m",
                                                                                                           step="month",
                                                                                                           stepmode="backward"),
                                                                                                      dict(count=1,
                                                                                                           label="YTD",
                                                                                                           step="year",
                                                                                                           stepmode="todate"),
                                                                                                      dict(count=1,
                                                                                                           label="1å",
                                                                                                           step="year",
                                                                                                           stepmode="backward"),
                                                                                                      dict(count=3,
                                                                                                           label="3å",
                                                                                                           step="year",
                                                                                                           stepmode="backward"),
                                                                                                      dict(count=5,
                                                                                                           label="5å",
                                                                                                           step="year",
                                                                                                           stepmode="backward"),
                                                                                                      dict(step="all",label = 'MAX')
                                                                                                  ])
                                                                                              )
                                                                                              ),
                                                                                 yaxis = dict(title=dict(text = 'Värde (%)',
                                                                                                        font=dict(
                                                                                                             size=18, 
                                                                                                            family = 'Cadiz Semibold'
                                                                                                            )),
                                                                                             tickformat = ' ',
                                                                                             automargin=True,
                                                                                             tickfont = dict(
                                                                                                 family = 'Cadiz Semibold', 
                                                                                                  size = 16
                                                                                                 )
                                                                                             )
                                                                                
                                                                                 )
                                                                                 ),
                                           config=config_plots_sv
                                           )])
                                 ])
                                
                            
                             ], xs =12, sm=12, md=12, lg=6, xl=6, align ='end')
                        
                         ],
                         justify='center', 
                         # style = {'margin' : '10px 10px 10px 10px'}
                         ),
                     html.Br(),
                     footer
                
                     ]
                ),
            dbc.Tab(label='Metodval',
                    tab_id ='hyperparam_tab_sv',
                    tabClassName="flex-grow-1 text-center",
                    tab_style = {'font-size':"1.5625rem",
                                 'font-weight': 'bold', 
                                 #'font-family':'Cadiz Semibold'
                                 },
                   style = {
                           
                            
                              
                               "height":"100vh",

                           "overflowY": "scroll"
                       },
                    
                    children = [
                        dbc.Row([
                            
                            dbc.Col([
                            
                                html.Br(),
                                html.P("I det här avsnittet kan du välja maskininlärningsalgoritm, justera dess hyperparametrar och, om du vill, använda huvudkomponentanalys som du vill.",
                                        style = p_style),
                                html.P("Välj först algoritmen, sedan visas dess specifika hyperparametrar, som du kan justera för att passa. Hyperparametrar läses automatiskt från Scikit-learn biblioteksdokumentation eller refereras från XGBoost-dokumentation och därför finns det ingen svensk översättning för dem. Under kontrollmenyerna hittar du en länk till dokumentationssidan för den valda algoritmen, där du kan läsa mer om ämnet. Det finns inte ett enda sätt att justera hyperparametrar, men olika värden måste testas iterativt.",
                                        style = p_style),
                                html.Br(),
                                html.P("Dessutom kan du välja om du vill använda principalkomponentanalys för att minimera funktioner. Huvudkomponentsanalysen är en statistisk och teknisk bullerdämpningsmetod som syftar till att förbättra prognosens kvalitet. Linjära kombinationer av de valda funktionerna bildas på ett sådant sätt att variationen i de ursprungliga uppgifterna förblir med ett visst förhållande i de ändrade uppgifterna. Du kan justera den förklarade variansen efter dina önskemål. Liksom för hyperparametrar är denna definition rent empirisk.",
                                        style = p_style),
                                html.P("Om kanterna i rutan hyperparameter är röda är värdet inte lämpligt. Testning och förutsägelse misslyckas om tillåtna värden tillämpas på hyperparametrar. Du kan kontrollera de tillåtna värdena i modelldokumentationen.",
                                        style = p_style)
                            ],xs =12, sm=12, md=12, lg=9, xl=9)
                        ], justify = 'center'
                            ),
                        html.Br(),
                        dbc.Row([
                            dbc.Col(id = 'model_selection_sv', children = [
                                
                                html.H3('Välj en algoritm',style=h3_style),
                                
                                dcc.Dropdown(id = 'model_selector_sv',
                                              value = 'Slumpmässig skog',
                                              multi = False,
                                              placeholder = 'Välj en algoritm',
                                              style = {'font-size':"0.9rem", #'font-family':'Cadiz Book'
                                                       },
                                              options = [{'label': c, 'value': c} for c in MODELS_sv.keys()]),
                                
                                html.Br(),
                                html.H3('Justera modellhyperparametrar', style = h3_style),
                                
                                html.Div(id = 'hyperparameters_div_sv')
                                
                                ], xs =12, sm=12, md=12, lg=9, xl=9),
                            dbc.Col(id = 'pca_selections_sv', children = [
                                html.Br(),
                                dash_daq.BooleanSwitch(id = 'pca_switch_sv', 
                                                                  label = dict(label = 'Använd principalkomponentanalys',style = {'font-size':30, 
                                                                                                                                # 'font-family':'Cadiz Semibold',
                                                                                                                                'textAlign':'center'}), 
                                                                  on = False, 
                                                                  color = 'blue'),
                                html.Br(),
                                html.P("Principalkomponentanalys är en bullerdämpningsmetod som kondenserar informationen om prognosfunktioner till huvudkomponenterna. Varje större komponent lagrar variationen av de ursprungliga uppgifterna, och den lagrade variationen av alla större komponenter uppgår till 100%.",
                                       style = p_style),
                                html.A([html.P('Titta på en kort introduktionsvideo om analys av huvudkomponenter.',
                                               style = p_style)],
                                       href = "https://www.youtube.com/embed/hJZHcmJBk1o",
                                       target = '_blank'),
                                
                                
                                html.Div(id = 'ev_placeholder_sv',children =[
                                    html.H3('Välj förklarad varians', style = h3_style),
                                    
                                    dcc.Slider(id = 'ev_slider_sv',
                                        min = .7, 
                                        max = .99, 
                                        value = .95, 
                                        step = .01,
                                        tooltip={"placement": "top", "always_visible": True},
                                        marks = {
                                                  .7: {'label':'70%', 'style':{'font-size':20, 
                                                                               # 'font-family':'Cadiz Semibold'
                                                                               }},
                                            .85: {'label':'85%', 'style':{'font-size':20, 
                                                                          # 'font-family':'Cadiz Semibold'
                                                                          }},
                                                  .99: {'label':'99%', 'style':{'font-size':20, 
                                                                                # #'fontFamily':'Cadiz Semibold'
                                                                                }}

                                                }
                                      ),
                                    html.Br(),
                                  html.Div(id = 'ev_slider_update_sv', 
                                          children = [
                                              html.Div([html.P('Du valde {} % förklarad varians.'.format(95),
                                                                style = p_center_style)
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
                        footer
                        
                        ]
                    ),
            dbc.Tab(label='Provning',
                    tab_id ='test_tab_sv',
                    tabClassName="flex-grow-1 text-center",
                    tab_style = {'font-size':"1.5625rem",
                                 'font-weight': 'bold', 
                                 #'font-family':'Cadiz Semibold'
                                 },
                    style = {
                            
                             
                               
                                "height":"100vh",

                            "overflowY": "scroll"
                        },
                    children = [
                        html.Br(),
                        dbc.Row([
                            dbc.Col([   
                                        # html.H3('Testa din metod', style = h3_style),
                                        # html.Br(),
                                        html.P("I det här avsnittet kan du testa hur väl den valda metoden skulle ha kunnat förutsäga arbetslösheten under de senaste månaderna med hjälp av de valda funktionerna. Vid testning lämnas det valda antalet månader kvar som testdata, vilket metoden syftar till att förutsäga.",
                                               style = p_style),
                                        html.P("Här antas indexvärdena vara som de var i verkligheten.",
                                               style = p_style),
                                        html.P("När du har slutfört testet kan du visa nästa resultatdiagram eller exportera testdata från knappen nedan till Excel.",
                                              style=p_style),
                                        html.P("Från graferna nedan kan du se hur varorna du valde påverkade den testade prognosen.",style=p_style),
                                        html.Br(),
                                        html.H3('Välj testlängd',style = h3_style),
                                        dcc.Slider(id = 'test_slider_sv',
                                                  min = 1,
                                                  max = 18,
                                                  value = 3,
                                                  step = 1,
                                                  tooltip={"placement": "top", "always_visible": True},
                                                 
                                                  marks = {1: {'label':'en månad', 'style':{'font-size':20, 
                                                                                            # 'font-family':'Cadiz Semibold'
                                                                                            }},
                                                          # 3:{'label':'3 kuukautta', 'style':{'font-size':20, #'fontFamily':'Cadiz Semibold','color':'white'}},
                                                        
                                                            6:{'label':'sex månader', 'style':{'font-size':20, 
                                                                                                # 'font-family':'Cadiz Semibold'
                                                                                                }},
                                                          #  9:{'label':'yhdeksän kuukautta', 'style':{'font-size':20, #'fontFamily':'Cadiz Semibold','color':'white'}},
                                                         
                                                          12:{'label':'ett år', 'style':{'font-size':20, 
                                                                                        # 'font-family':'Cadiz Semibold'
                                                                                        }},
                                                          18:{'label':'ett och ett halvt år', 'style':{'font-size':20, 
                                                                                        # 'font-family':'Cadiz Semibold'
                                                                                        }},
                                                          

                                                            }
                                                  ),
                                        html.Br()
                                        ],xs =12, sm=12, md=12, lg=9, xl=9),
                                ],justify='center'), 
                        dbc.Row([
                            dbc.Col([
                                        html.Div(id = 'test_size_indicator_sv', style = {'textAlign':'center'}),
                                        html.Br(),
                                        html.Div(id = 'test_button_div_sv',children = [html.P('Välj varor först.',style = {
                                            'text-align':'center', 
                                            #'font-family':'Cadiz Semibold', 
                                              'font-size':p_font_size
                                            })], style = {'textAlign':'center'}),
                                        html.Br(),
                                        html.Div(id='test_download_button_div_sv', style={'textAlign':'center'})
                                        
                                        
                            
                            
                            ],xs =12, sm=12, md=12, lg=9, xl=9)
                            ], justify = 'center',style={'text-align':'center'}
                            ),
                        html.Br(),
                        dbc.Row(children = [
                            dbc.Col([html.Div(id = 'test_results_div_sv')],xs = 12, sm = 12, md = 12, lg = 9, xl = 9),
                            
                            
                            ], justify = 'center', 
                            
                            ),
                        html.Br(),
                        dbc.Row([dbc.Col([html.Div(id = 'shap_selections_div_sv')],xs = 12, sm = 12, md = 12, lg = 9, xl = 9)],justify = 'center'),
                        dbc.Row(children = [
                            
                            dbc.Col([html.Div(id = 'shap_results_div_sv')],xs = 12, sm = 12, md = 12, lg = 6, xl = 6),
                            dbc.Col([html.Div(id = 'local_shap_results_div_sv')],xs = 12, sm = 12, md = 12, lg = 6, xl = 6),
                            
                            ], justify = 'center', align='start', 
                            
                            ),
                        html.Br(),
                        footer
                 
                        
                        
                        
                        ]
                    ),
            dbc.Tab(label='Prognos',
                    tab_id = 'forecast_tab_sv',
                    tabClassName="flex-grow-1 text-center",
                    tab_style = {'font-size':"1.5625rem",
                                 'font-weight': 'bold', 
                                 #'font-family':'Cadiz Semibold'
                                 },
                    style = {
                            
                             
                               
                                "height":"100vh",

                            "overflowY": "scroll"
                        },
                    children = [
                        html.Br(),
                        dbc.Row([
                            
                            dbc.Col([
                                        # html.H3('Prognoser',style=h3_style),
                                        # html.Br(),
                                        html.P("I det här avsnittet kan du göra en prognos för den valda tiden. När du förutspår aktiveras inställningarna på fliken Metodval. Prognosen baseras på antaganden som gjorts under fliken Produktval om den relativa prisutvecklingen på råvaror. "
                                                "Det är bra att notera att osäkerheten ökar ju längre prognosen görs. "
                                                "Dessutom kan du titta på modellen agnostik, som visar vilka varor och egenskaper som påverkar utvecklingen av arbetslöshetstalet och vilka varuprisförändringar som påverkar förändringen av arbetslöshetstalet på månadsbasis om användarnas antaganden uppfylls. ",
                                              style=p_style),
                                        html.P("När du har gjort prognosen kan du visa den intilliggande prognosgrafen eller exportera resultatdata från knappen nedan till Excel.",
                                              style=p_style),
                                        html.Br(),
                                        html.H3('Välj prognoslängd',
                                                style=h3_style),
                                        dcc.Slider(id = 'forecast_slider_sv',
                                                  min = 2,
                                                  max = 18,
                                                  value = 3,
                                                  step = 1,
                                                  tooltip={"placement": "top", "always_visible": True},
                                                  marks = {2: {'label':'två månader', 'style':{'font-size':16, 
                                                                                               # #'fontFamily':'Cadiz Semibold'
                                                                                               }},
                                                          # 3: {'label':'kolme kuukautta', 'style':{'font-size':20, #'fontFamily':'Cadiz Semibold','color':'white'}},
                                                          6:{'label':'sex månader', 'style':{'font-size':16, 
                                                                                              # #'fontFamily':'Cadiz Semibold'
                                                                                              }},
                                                          # 9:{'label':'yhdeksän kuukautta', 'style':{'font-size':20, #'fontFamily':'Cadiz Semibold','color':'white'}},
                                                          12:{'label':'ett år', 'style':{'font-size':16, 
                                                                                        # 'font-family':'Cadiz Semibold'
                                                                                        }},
                                                          18:{'label':'ett och ett halvt år', 'style':{'font-size':16, 
                                                                                        # 'font-family':'Cadiz Semibold'
                                                                                        }},
                                                        #  24:{'label':'kaksi vuotta', 'style':{'font-size':20, #'fontFamily':'Cadiz Semibold','color':'white'}}
                                                        
                                                     

                                                          }
                                                  ),
                                        html.Br(),
                                        html.Div(id = 'forecast_slider_indicator_sv',style = {'textAlign':'center'}),
                                        html.Div(id = 'forecast_button_div_sv',children = [html.P('Valitse commodities first.',
                                                                                              style = p_style
                                                                                              )],style = {'textAlign':'center'})
                                        
                                    ],xs =12, sm=12, md=12, lg=9, xl=9)
                            
                            
                            ], justify = 'center'),
                        html.Br(),
                        dbc.Row(children = [
                                    
                                    dbc.Col([dcc.Loading(id = 'forecast_results_div_sv',type = spinners[random.randint(0,len(spinners)-1)])],
                                            xs = 12, sm = 12, md = 12, lg = 8, xl = 8)
                                    ], justify = 'center', 
                             # style = {'margin' : '10px 10px 10px 10px'}
                                    ),
                        html.Br(),
                        dbc.Row([dbc.Col([html.Div(id = 'forecast_shap_selections_div_sv')],xs = 12, sm = 12, md = 12, lg = 9, xl = 9)],justify = 'center'),
                        dbc.Row([
                            
                            dbc.Col(id = 'forecast_shap_div_sv', xs = 12, sm = 12, md = 12, lg = 6, xl = 6),
                            dbc.Col(id = 'forecast_local_shap_div_sv', xs = 12, sm = 12, md = 12, lg = 6, xl = 6)
                            
                        ], justify ='center', align='start'),
                        html.Br(),
                        footer
                                       
                            
                            
                            
                            
                              
                            
                        ]
                            
                            )


        ]
            
    ),

    
   ]
  )])

@callback(
    
    [Output('shap_features_switch_sv', 'label'),
     Output('shap_features_switch_sv', 'disabled')],
    Input('shap_data_sv','data')
    
)
def sv_update_shap_switch(shap_data):
    
    shap_df = pd.DataFrame(shap_data)
    shap_df = shap_df.set_index(shap_df.columns[0])
    
    if 'Kuukausi' not in shap_df.index:
        return dict(label = 'Du använde principalkomponentanalys',
                     style = {'font-size':p_font_size,
                              'text-align':'center'
                              # #'fontFamily':'Cadiz Semibold'
                              }), True
    else:
        return dict(label = 'Visa endast varornas bidrag',
                     style = {'font-size':p_font_size, 
                              'text-align':'center'
                              # #'fontFamily':'Cadiz Semibold'
                              }), False
    
    
@callback(
    
    [Output('forecast_shap_features_switch_sv', 'label'),
     Output('forecast_shap_features_switch_sv', 'disabled')],
    Input('forecast_shap_data_sv','data')
    
)
def sv_updateforecast_shap_switch(shap_data):
    
    shap_df = pd.DataFrame(shap_data)
    shap_df = shap_df.set_index(shap_df.columns[0])
    
    if 'Kuukausi' not in shap_df.index:
        return dict(label = 'Du använde principalkomponentanalys',
                     style = {'font-size':p_font_size,
                              'text-align':'center'
                              # #'fontFamily':'Cadiz Semibold'
                              }), True
    else:
        return dict(label = 'Visa endast varornas bidrag',
                     style = {'font-size':p_font_size, 
                              'text-align':'center'
                              # #'fontFamily':'Cadiz Semibold'
                              }), False

@callback(

    [Output('adjustments_div_sv','children'),
     Output('features_values_sv','data')],
    [Input('slider_sv','value'),
     Input('feature_selection_sv','value')],
    [State('averaging_sv','on')]    
    
)
def sv_add_value_adjustments(slider_value, features, averaging):
    
    
    
    if averaging:
        
        mean_df = sv_apply_average(features = features, length = slider_value)
        
        features_values = {feature:mean_df.loc[feature] for feature in features}
        
        row_children =[dbc.Col([html.Br(), 
                                html.P(feature,style={#'font-family':'Messina Modern Semibold',
                                            'font-size':"1.1rem"}),
                                dcc.Input(id = {'type':'value_adjust_sv', 'index':feature}, 
                                               value = round(mean_df.loc[feature],1), 
                                               type = 'number', 
                                               style={#'font-family':'Messina Modern Semibold',
                                                           'font-size':"1.1rem"},
                                               step = .1)],xs =12, sm=12, md=4, lg=2, xl=2) for feature in features]
    else:
        
        features_values = {feature:slider_value for feature in features}
        
        row_children =[dbc.Col([html.Br(), 
                                html.P(feature,style={#'font-family':'Messina Modern Semibold',
                                            'font-size':"1.1rem"}),
                                dcc.Input(id = {'type':'value_adjust_sv', 'index':feature}, 
                                               value = slider_value, 
                                               type = 'number', 
                                               style ={#'font-family':'Messina Modern Semibold',
                                                           'font-size':"1.1rem"},
                                               step = .1)],xs =12, sm=12, md=4, lg=2, xl=2) for feature in features]
    return row_children, features_values


@callback(

    Output('change_weights_sv','data'),
    [Input({'type': 'value_adjust_sv', 'index': ALL}, 'id'),
    Input({'type': 'value_adjust_sv', 'index': ALL}, 'value')],    
    
)
def sv_store_weights(feature_changes, feature_change_values):
    
    if feature_changes is None:
        raise PreventUpdate
    
    weights_dict = {feature_changes[i]['index']:feature_change_values[i] for i in range(len(feature_changes))}
        
    return weights_dict


@callback(

    Output('slider_div_sv','children'),
    [Input('averaging_sv', 'on')
     ]
    
)
def sv_update_slider_div(averaging):
    
    if averaging:
        
        return [html.H3('Välj antalet senaste månader för genomsnittet', 
                        style = h3_style),
        html.Br(),
        dcc.Slider(id = 'slider_sv',
                      min = 1,
                      max = 12,
                      value = 4,
                      step = 1,
                      tooltip={"placement": "top", "always_visible": True},
                       marks = {1:{'label':'en månad', 'style':{'font-size':20, 
                                                                # #'fontFamily':'Cadiz Semibold'
                                                                }},
                                # 3:{'label':'3 kuukautta', 'style':{'font-size':20, #'fontFamily':'Cadiz Semibold','color':'white'}},
                                6:{'label':'sex månader', 'style':{'font-size':20, 
                                                                    # #'fontFamily':'Cadiz Semibold'
                                                                    }},
                                # 9:{'label':'yhdeksän kuukautta', 'style':{'font-size':20, #'fontFamily':'Cadiz Semibold','color':'white'}},
                                12:{'label':'et år', 'style':{'font-size':20, 
                                                              # #'fontFamily':'Cadiz Semibold'
                                                              }}   
                             }
                      
                    )]
        
    else:
        return [
            html.H3('Välj konstant månadsförändring', 
                    style = h3_style),
            
            dcc.Slider(id = 'slider_sv',
                          min = -20,
                          max = 20,
                          value = 0,
                          step = 0.1,
                          tooltip={"placement": "top", "always_visible": True},
                           marks = {
                                    # -30:{'label':'-30%', 'style':{'font-size':20, #'fontFamily':'Cadiz Semibold'
                                    #                                       }},
                                   -20:{'label':'-20%', 'style':{'font-size':20, #'fontFamily':'Cadiz Semibold'
                                                                  }},
                                    # 3:{'label':'3 kuukautta', 'style':{'font-size':20, #'fontFamily':'Cadiz Semibold','color':'white'}},
                                    0:{'label':'0%', 'style':{'font-size':20, #'fontFamily':'Cadiz Semibold'
                                                              }},
                                    # 9:{'label':'yhdeksän kuukautta', 'style':{'font-size':20, #'fontFamily':'Cadiz Semibold','color':'white'}},
                                    20:{'label':'20%', 'style':{'font-size':20, #'fontFamily':'Cadiz Semibold'
                                                                }},
                                    # 30:{'label':'30%', 'style':{'font-size':20, #'fontFamily':'Cadiz Semibold'
                                    #                             }} 
                                 }
                          
                        )
            
            ]


@callback(

    Output('alert_sv', 'is_open'),
    [Input('feature_selection_sv','value')]

)
def sv_update_alert(features):
    
    return len(features) == 0

@callback(

    Output('hyperparameters_div_sv','children'),
    [Input('model_selector_sv','value')]    
    
)
def sv_update_hyperparameter_selections(model_name):
    
    
    
    model = MODELS_sv[model_name]['model']
    
        
    hyperparameters = model().get_params()
    
    if model_name == 'XGBoost':
        
        hyperparameters = {'objective': 'reg:squarederror',
               'base_score': .5,
               'eval_metric':'rmse',
               'booster': 'gbtree',
               'colsample_bylevel': .99,
               'colsample_bynode': .99,
               'colsample_bytree': .99,
               # 'enable_categorical': False,
               'gamma': 0,
               # 'gpu_id': None,
               # 'importance_type': None,
               # 'interaction_constraints': None,
               'learning_rate': .3,
               # 'max_delta_step': 0,
               # 'max_depth': 0,
                'min_child_weight': 1,
               # 'missing': np.nan,
               # 'monotone_constraints': None,
               'n_estimators': 50,
               # 'n_jobs': -1,
               # 'num_parallel_tree': 1,
               # 'predictor': 'auto',
               # 'random_state': 42,
               'reg_alpha': 0,
               'reg_lambda': 1,
               'scale_pos_weight': 1,
               'subsample': .99,
               'tree_method': 'auto',
               'validate_parameters': False,
               'verbosity': 0}
    
    elif model_name=='Gradientförstärkning':
        hyperparameters.update({'subsample':.99})
    
    type_dict ={}
    for i, c in enumerate(hyperparameters.values()):
      
        type_dict[i] =str(type(c)).split()[1].split('>')[0].replace("'",'')
        
    h_series = pd.Series(type_dict).sort_values()

    param_options = sv_get_param_options(model_name)
       
    
    children = []
    
    for i in h_series.index:
        
        hyperparameter = list(hyperparameters.keys())[i]
        
        
        
        if hyperparameter not in UNWANTED_PARAMS:
        
            
            value = list(hyperparameters.values())[i]
            
            
            
            if type(value) == int:
                children.append(dbc.Col([html.P(hyperparameter+':', 
                                                 style=p_bold_style)],xs =12, sm=12, md=12, lg=12, xl=12))
                children.append(html.Br())
                children.append(dbc.Col([dcc.Slider(id = {'index':hyperparameter, 'type':'hyperparameter_tuner_sv'},
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
                    children.append(dbc.Col([dbc.Input(id = {'index':hyperparameter, 'type':'hyperparameter_tuner_sv'},
                                                       min = 0,
                                                       max=0.99,
                                                       type = 'number',
                                                       style = {
                                                           #'font-family':'Cadiz Book',
                                                           'font-size':p_font_size},
                                                       value = value,
                                                       className="mb-3",
                                                       step=0.01),
                                                      html.Br()],xs =12, sm=12, md=12, lg=2, xl=2)
                                    )
                elif hyperparameter in LESS_THAN_HALF:
                        
                        children.append(dbc.Col([html.P(hyperparameter+':', 
                                                         style=p_bold_style)],xs =12, sm=12, md=12, lg=2, xl=2)),
                        children.append(html.Br())
                        children.append(dbc.Col([dbc.Input(id = {'index':hyperparameter, 'type':'hyperparameter_tuner_sv'},
                                                           min = 0,
                                                           max=0.49,
                                                           type = 'number',
                                                           value = value,
                                                           className="mb-3",
                                                           style = {
                                                               #'font-family':'Cadiz Book',
                                                               'font-size':p_font_size},
                                                           step=0.01),
                                                          html.Br()],xs =12, sm=12, md=12, lg=2, xl=2)
                                        )

                    
                else:
                    children.append(dbc.Col([html.P(hyperparameter+':', 
                                                     style=p_bold_style)],xs =12, sm=12, md=12, lg=2, xl=2)),
                    children.append(html.Br())
                    children.append(dbc.Col([dbc.Input(id = {'index':hyperparameter, 'type':'hyperparameter_tuner_sv'},
                                                       min = 0,
                                                       type = 'number',
                                                       value = value,
                                                       className="mb-3",
                                                       style = {
                                                           #'font-family':'Cadiz Book',
                                                           'font-size':p_font_size},
                                                       step=0.01),
                                                      html.Br()
                                             ],xs =12, sm=12, md=12, lg=2, xl=2)
                                    )
                    
                children.append(html.Br())
                
            
            elif type(value) == bool:
                children.append(dbc.Col([html.P(hyperparameter+':', 
                                                 style=p_bold_style)],xs =12, sm=12, md=12, lg=2, xl=2)),
                children.append(html.Br())
                children.append(dbc.Col([dbc.Switch(id = {'index':hyperparameter, 'type':'hyperparameter_tuner_sv'}, 
                                          style= {
                                              #'font-family':'Cadiz Book',
                                              'font-size':p_font_size},
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
                        children.append(dbc.Col([dcc.Dropdown(id = {'index':hyperparameter, 'type':'hyperparameter_tuner_sv'}, 
                                                  multi = False,
                                                  #label = hyperparameter,
                                                  style = {
                                                      #'font-family':'Cadiz Book',
                                                      'font-size':"0.9rem"},
                                                  options = [{'label':c, 'value': c} for c in param_options[hyperparameter] if c not in ['precomputed','poisson']],
                                                  value = value),
                                                 html.Br()],xs =12, sm=12, md=12, lg=2, xl=2)
                                    )
                        children.append(html.Br())
                        children.append(html.Br())
                       
     
    children.append(html.Br()) 
    children.append(html.Div(style = {'textAlign':'center'},
             children = [html.A('Kolla en kort introduktionsvideo om algoritmen.', href = MODELS_sv[model_name]['video'], target="_blank",style = p_style),
                         html.Br(),
                         html.A('Kolla den tekniska dokumentationen.', href = MODELS_sv[model_name]['doc'], target="_blank",style = p_style),
                         ]))
    return dbc.Row(children, justify ='start')


@callback(

    Output('method_selection_results_sv','data'),
    [Input('model_selector_sv','value'),
    Input({'type': 'hyperparameter_tuner_sv', 'index': ALL}, 'id'),
    Input({'type': 'hyperparameter_tuner_sv', 'index': ALL}, 'value'),
    Input('pca_switch_sv','on'),
    Input('ev_slider_sv','value')]    
    
)
def sv_store_method_selection_results(model_name, hyperparams, hyperparam_values,pca, explained_variance):
            
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

@callback(
    
      [Output('test_data_sv','data'),
       Output('test_results_div_sv','children'),
       Output('test_download_button_div_sv','children'),
       Output('shap_data_sv','data'),
       Output('local_shap_data_sv','data')],
      [Input('test_button_sv','n_clicks')],
      [State('test_slider_sv', 'value'),
       State('features_values_sv','data'),
       State('method_selection_results_sv','data')

       ]
    
)
def sv_update_test_results(n_clicks, 
                        test_size,
                        features_values,
                        method_selection_results
 
                        ):
    
    if n_clicks > 0:
    
        try:
            locale.setlocale(locale.LC_ALL, 'sv-FI')
        except:
            locale.setlocale(locale.LC_ALL, 'sv_FI')
        
        
        features = sorted(list(features_values.keys()))
        
        model_name = method_selection_results['model']
             
        pca = method_selection_results['pca']
        explained_variance = method_selection_results['explained_variance']
        hyperparam_grid = method_selection_results['hyperparam_grid']
        
        model = MODELS_sv[model_name]['model']
        constants = MODELS_sv[model_name]['constant_hyperparameters'] 
        
        
        model_hyperparams = hyperparam_grid.copy()
        
        
        model_hyperparams.update(constants)
        
        
        model = model(**model_hyperparams)
        
        explainer = MODELS_sv[model_name]['explainer']
        
        
        test_result, shap_results, local_shap_df = sv_test(model, features, explainer = explainer, test_size=test_size, use_pca=pca,n_components=explained_variance)

        mape = test_result.mape.values[0]
        
        local_shap_df.index = test_result.index
        
        shap_data  = shap_results.reset_index().to_dict('records') 
        local_shap_data = local_shap_df.reset_index().to_dict('records')
        
        led_color = 'red'
        
        if mape <=.25:
            led_color='orange'
        if mape <= .1:
            led_color='green'

        
        test_plot =[html.Br(),
                    dbc.Row([
                        
                        # dbc.Col([
                            html.Br(),
                             html.H3('Hur gick det för oss?',
                                     style = h3_style),
                             html.P("(Fortsätter efter diagrammet)",style={
                                         'font-style':'italic',
                                         'font-size':p_font_size,
                                        'text-align':'center'}
                                 ),
                             html.P('Diagrammet nedan visar hur väl prognosmodellen skulle ha förutsett arbetslösheten från {} till {}.'.format(test_result.index.strftime('%B %Y').values[0],test_result.index.strftime('%B %Y').values[-1]),
                                    style = p_style),
                              html.Div([html.Br(),dbc.RadioItems(id = 'test_chart_type_sv', 
                                          options = [{'label':'stapeldiagram','value':'bars'},
                                                    {'label':'linjediagram','value':'lines'},
                                                    {'label':'stapel- och linjediagram','value':'lines+bars'}],
                                          labelStyle={'display':'inline-block', 'padding':'10px','margin':'10px 10px 10px 10px','font-size':"1.1rem",
                                                      #'font-family':'Cadiz Book'
                                                      'font-weight': 'bold'
                                                      },
                                          className="btn-group",
                                          inputClassName="btn-check",
                                          labelClassName="btn btn-outline-secondary",
                                          labelCheckedClassName="active",
                                        
                                          value = 'lines+bars'
                                        ),html.Br()
                                    ],
                                    style = {'textAlign':'right'}
                                  ),
                              html.Br(),
                              html.Div([dcc.Loading(id ='test_graph_div_sv',type = spinners[random.randint(0,len(spinners)-1)])])
   
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
                                     html.H3('Noggrannhet (%)', style = h3_style),
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
                    # html.P("Det genomsnittliga absoluta procentuella felet (MAPE) är genomsnittet av de relativa felen för alla prognosvärden. Noggrannheten i detta fall beräknas med formel 1 – MAPE.", 
                    #        style = p_style,
                    #        className="card-text"),
                    html.Div([
                        html.Label([html.A('Det genomsnittliga absoluta procentuella felet ( '),
                                html.A('MAPE', 
                                       href = 'https://en.wikipedia.org/wiki/Mean_absolute_percentage_error',
                                       target='_blank'),
                                html.A(' ) är genomsnittet av de relativa felen för alla prognosvärden. Noggrannheten i detta fall beräknas med formel 1 – MAPE.', 
                           )
                            ],
                                   style = {'font-size':'1.2rem'}
                           )
                        ],style = {'text-align':'center'},
                        className="card-text"),
                    html.Br(),

                    
                    ]
             

        feat = features.copy()
        feat = ['Työttömyysaste','Ennuste','prev','month','change','mape','n_feat', 'Ennustettu muutos']+feat
        
        button_children = dbc.Button(children=[html.I(className="fa fa-download mr-1"), ' Ladda ner testresultat'],
                                       id='test_download_button_sv',
                                       n_clicks=0,
                                       style = dict(fontSize=25,
                                                    # fontFamily='Cadiz Semibold',
                                                    textAlign='center'),
                                       outline=True,
                                       size = 'lg',
                                       color = 'info'
                                       )
        
        return test_result[feat].reset_index().to_dict('records'), test_plot, button_children, shap_data, local_shap_data
    else:
        return [html.Div(),html.Div(),html.Div(),html.Div(),html.Div()]


        
@callback(
    
      [Output('forecast_data_sv','data'),
       Output('forecast_shap_data_sv','data'),
       Output('local_forecast_shap_data_sv','data'),
       Output('forecast_results_div_sv','children'),
       Output('forecast_download_button_div_sv','children')],
      [Input('forecast_button_sv','n_clicks')],
      [State('forecast_slider_sv', 'value'),
       State('change_weights_sv','data'),
       State('method_selection_results_sv','data')

       ]
    
)
def sv_update_forecast_results(n_clicks, 
                        forecast_size,
                        weights_dict,
                        method_selection_results
                        ):

    
    
    if n_clicks > 0:
        
        try:
            locale.setlocale(locale.LC_ALL, 'sv-FI')
        except:
            locale.setlocale(locale.LC_ALL, 'sv_FI')
        
        features = sorted(list(weights_dict.keys()))
        
        
        
        model_name = method_selection_results['model']
             
        pca = method_selection_results['pca']
        explained_variance = method_selection_results['explained_variance']
        hyperparam_grid = method_selection_results['hyperparam_grid']
        
        model = MODELS_sv[model_name]['model']
        constants = MODELS_sv[model_name]['constant_hyperparameters'] 
        
        
        model_hyperparams = hyperparam_grid.copy()
        
        
        model_hyperparams.update(constants)
        
        
        model = model(**model_hyperparams)
        
        
        weights = pd.Series(weights_dict)
        
        explainer = MODELS_sv[model_name]['explainer']
        
        forecast_df, shap_df, local_shap_df = sv_predict(model,
                                 explainer,
                              features, 
                              feature_changes = weights, 
                              length=forecast_size, 
                              use_pca=pca,
                              n_components=explained_variance)
        

        
        forecast_div =  html.Div([html.Br(),
                      html.H3('Prognosresultat', 
                              style = h3_style),
                      
                      html.P('Diagrammet nedan visar de faktiska värdena och prognosen från {} till {}.'.format(forecast_df.index.strftime('%B %Y').values[0],forecast_df.index.strftime('%B %Y').values[-1]),
                             style = p_style),
                      html.P("Du kan välja antingen en kolumn, områdes eller ett linjediagram från knapparna nedan. Längden på tidsserien kan justeras från reglaget nedan. Du kan också begränsa längden genom att klicka på knapparna i övre vänstra hörnet.",
                             style = p_style),
                      html.P("(Fortsätter efter diagrammet)",style={
                                  'font-style':'italic',
                                  'font-size':p_font_size,
                                 'text-align':'center'}
                          ),
                      
                      html.Div([
                      dbc.RadioItems(id = 'chart_type_sv', 
                        options = [{'label':'stapeldiagram','value':'bars'},
                                  {'label':'linjediagram','value':'lines'},
                                  {'label':'områdesdiagram','value':'area'}],
                        labelStyle={'display':'inline-block', 'padding':'10px','margin':'10px 10px 10px 10px','font-size':"1.1rem",
                                    #'font-family':'Cadiz Book'
                                    'font-weight': 'bold'
                                    },
                        className="btn-group",
                        inputClassName="btn-check",
                        labelClassName="btn btn-outline-secondary",
                        labelCheckedClassName="active",                        
                        value = 'area'
                      )
                      ],style={'textAlign':'right'}),
                      html.Br(),

          html.Div(id = 'forecast_graph_div_sv'),
        
          html.Br()
          ])

          
          # ], justify='center')        
        forecast_download_button = dbc.Button(children=[html.I(className="fa fa-download mr-1"), ' Ladda ner prognosresultat'],
                                 id='forecast_download_button_sv',
                                 n_clicks=0,
                                 style=dict(fontSize=25,
                                            # fontFamily='Cadiz Semibold',
                                            textlign='center'),
                                 outline=True,
                                 size = 'lg',
                                 color = 'info'
                                 )
        
        feat = features.copy()
        feat = ['Työttömyysaste','month','change','n_feat','prev']+feat
        
        return [forecast_df[feat].reset_index().to_dict('records'),
                shap_df.reset_index().to_dict('records'),
                local_shap_df.reset_index().to_dict('records'),
                forecast_div, [html.Br(),forecast_download_button]]
    else:
        return [html.Div(),html.Div(),html.Div(),html.Div(),html.Div()]
    
@callback(

    [Output('shap_selections_div_sv','children'),
     Output('shap_results_div_sv','children'),
     Output('local_shap_results_div_sv','children')
     ],
    [Input('test_button_sv','n_clicks'),
     Input('shap_data_sv','data'),
     State('local_shap_data_sv','data')]    
    
)
def sv_update_shap_results(n_clicks, shap, local_shap_data):
    
    if shap is None or local_shap_data is None:
        raise PreventUpdate
        
    if n_clicks > 0:
        
        try:
            locale.setlocale(locale.LC_ALL, 'sv_FI')
        except:
            locale.setlocale(locale.LC_ALL, 'sv-FI')
    
        shap_df = pd.DataFrame(shap)
        
        shap_df = shap_df.set_index(shap_df.columns[0])
        
        local_shap_df = pd.DataFrame(local_shap_data)
        local_shap_df = local_shap_df.set_index(local_shap_df.columns[0])
        local_shap_df.index = pd.to_datetime(local_shap_df.index)
        
        options = [{'label':c.strftime('%B %Y'), 'value': c} for c in list(local_shap_df.index)]
        
        return [[
            
                    html.H3('Vilka funktioner var de viktigaste?',
                           style = h3_style),
                    html.P("Följande diagram visar den globala och lokala betydelsen av prognosfunktionerna som används för prognosen. "
                            "Globala signifikansvärden kan användas för att undersöka vilka egenskaper som i allmänhet är mest betydelsefulla för prognosen. "
                            "I stället kan lokala värden användas för att bedöma vilka faktorer som påverkats och hur en viss månads prognos påverkades. "
                            "Betydande värden presenteras som SHAP-värden, som beskriver bidraget av egenskaper till prognosen. "
                            "Förutom de utvalda råvaruindexen inkluderar prognosfunktionerna arbetslösheten föregående månad och månaden.",
                           style = p_style),
                    html.A([html.P("Se en kort introduktionsvideo om betydelsen av SHAP-värden för att förklara en modell.",
                                   style = p_center_style)], href="https://www.youtube.com/embed/Tg8aPwPPJ9c", target='_blank'),
                    html.A([html.P('Se även en icke-teknisk guide för tolkning av SHAP-analyser',
                                   style = p_center_style)], href="https://www.aidancooper.co.uk/a-non-technical-guide-to-interpreting-shap-analyses/", target='_blank'),
                    html.P("Grafens SHAP-värden multipliceras med 100 för att förbättra visualiseringen. "
                           "En SHAP-enhet på de här graferna motsvarar en hundradels procentenhet som används att beskriva arbetslöshets månadsförändring. ",
                           style = p_style),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                                html.Div(id = 'cut_off_div_sv'),
                                
                                html.Div(id = 'cut_off_indicator_sv'),
                                
                                ],xs =12, sm=12, md=12, lg=9, xl=9),
                        dbc.Col([
                                dash_daq.BooleanSwitch(id = 'shap_features_switch_sv', 
                                                        label = dict(label = "Visa endast varornas bidrag",
                                                                     style = {'font-size':p_font_size,
                                                                              'text-align':'center',
                                                                              # #'fontFamily':'Cadiz Semibold'
                                                                              }), 
                                                        on = False, 
                                                        color = 'red')
                                ],xs =12, sm=12, md=12, lg=3, xl=3)
                        ]),
                    html.Br()
                    ],
                    [html.Br(),
                        dbc.Card([
                        dbc.CardBody([
                            html.H3('Egenskapsimporter', className='card-title',
                                    style=h3_style),
                        
                        html.P("Diagrammet nedan visar medelvärdet av absoluta SHAP-värden. "
                            "De beskriver hur mycket funktionerna i genomsnitt påverkade prognoserna, oavsett vilken riktning som påverkades. "
                            "De beräknas som genomsnittet av de absoluta SHAP-värdena för de lokala egenskaperna. "
                            "Svart färg indikerar de triviala egenskaper som är den aktuella månaden och arbetslösheten föregående månad.",
                           style =p_style,
                           className="card-text"),
                     html.Br(),
                        dcc.Loading([dbc.Row(id = 'shap_graph_div_sv', justify = 'center')], type = random.choice(spinners))
                        ])
                        ])
                        ],
                
                    [html.Br(),
                        dbc.Card([
                        dbc.CardBody([
                            html.H3('Månatliga egenskapsimporter', className='card-title',
                                    style=h3_style),
                     html.P("Diagrammet nedan visar SHAP-värdena för funktionerna för en vald månad. "
                            "De representerar den riktning och intensitet som kännetecknade prognosen för den valda månaden. "
                            "Grönt belyser de faktorer som minskar månadsförändringen i arbetslösheten och rött belyser de egenskaper som ökar den. "
                            "Svart färg indikerar de triviala egenskaper som är den aktuella månaden och arbetslösheten föregående månad. "
                            "Den vertikala axeln visar namnen på funktionerna och deras värden för motsvarande tid med förändringens riktning jämfört med föregående månad med en ikon inom parentes. "
                            "Under grafen finns en formel som kan användas för att beräkna månatliga prognoser genom att använda de lokala SHAP-värden och det konstanta värde som modellen producerar.",
                            style =p_style,
                            className="card-text"),
                      html.Br(),
                        html.H3('Välj en månad', style =h3_style),
                                        dcc.Dropdown(id = 'local_shap_month_selection_sv',
                                                      options = options, 
                                                      style = {'font-size':16},
                                                      value = list(local_shap_df.index)[0],
                                                      multi=False ),
                                        html.Br(),
                                        
                                        html.Div(dcc.Loading(id = 'local_shap_graph_div_sv',
                                                              type = random.choice(spinners))
                                                 )
                                   ])
                        ])
                        ]]

    else:
        return [html.Div(),html.Div(),html.Div()]

@callback(

    [Output('forecast_shap_selections_div_sv','children'),
     Output('forecast_shap_div_sv','children'),
     Output('forecast_local_shap_div_sv','children')
     ],
    [Input('forecast_button_sv','n_clicks'),
     Input('forecast_shap_data_sv','data'),
     State('local_forecast_shap_data_sv','data')]    
    
)
def sv_update_forecast_shap_results(n_clicks, shap, local_shap_data):
    
    if shap is None or local_shap_data is None:
        raise PreventUpdate
        
    if n_clicks > 0:
        
        try:
            locale.setlocale(locale.LC_ALL, 'sv_FI')
        except:
            locale.setlocale(locale.LC_ALL, 'sv-FI')
    
        shap_df = pd.DataFrame(shap)
        
        shap_df = shap_df.set_index(shap_df.columns[0])
        
        local_shap_df = pd.DataFrame(local_shap_data)
        local_shap_df = local_shap_df.set_index(local_shap_df.columns[0])
        local_shap_df.index = pd.to_datetime(local_shap_df.index)
        
        options = [{'label':c.strftime('%B %Y'), 'value': c} for c in list(local_shap_df.index)]
        
        return [[
            
                    html.H3('Vilka egenskaper kommer att få störst effekt?',
                           style = h3_style),
                    html.P("Som i testavsnittet, när man förutspår arbetslösheten under de kommande månaderna, är det också möjligt att titta på vilka egenskaper som kommer att betyda mest och vilka som kommer att förklara förändringen i arbetslöshetstalet under en kommande månad. "
                            "Diagrammet ovan visar modellens prognos med antagandet att prisindexet för de varor som användaren valt ändras med den valda förändringstakten på månadsbasis. "
                            "De globala och lokala SHAP-värdena nedan gör det möjligt att se hur användarvalda månatliga varuspecifika förändringar påverkar prognosen. "
                            "Detta gör det möjligt att gå tillbaka till varuval sektionen, justera förändringshastigheten och försöka igen förutsäga med flera månatliga förändringar. ",
                           style = p_style),
                    # html.A([html.P("Se en kort introduktionsvideo om betydelsen av SHAP-värden för att förklara en modell.",
                    #                style = p_center_style)], href="https://www.youtube.com/embed/Tg8aPwPPJ9c", target='_blank'),
                    # html.A([html.P('Se även en icke-teknisk guide för tolkning av SHAP-analyser',
                    #                style = p_center_style)], href="https://www.aidancooper.co.uk/a-non-technical-guide-to-interpreting-shap-analyses/", target='_blank'),
                    html.P("Grafens SHAP-värden multipliceras med 100 för att förbättra visualiseringen. "
                           "En SHAP-enhet på de här graferna motsvarar en hundradels procentenhet som används att beskriva arbetslöshets månadsförändring. ",
                           style = p_style),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                                html.Div(id = 'forecast_cut_off_div_sv'),
                                
                                html.Div(id = 'forecast_cut_off_indicator_sv'),
                                
                                ],xs =12, sm=12, md=12, lg=9, xl=9),
                        dbc.Col([
                                dash_daq.BooleanSwitch(id = 'forecast_shap_features_switch_sv', 
                                                        label = dict(label = "Visa endast varornas bidrag",
                                                                     style = {'font-size':p_font_size,
                                                                              'text-align':'center',
                                                                              # #'fontFamily':'Cadiz Semibold'
                                                                              }), 
                                                        on = False, 
                                                        color = 'red')
                                ],xs =12, sm=12, md=12, lg=3, xl=3)
                        ]),
                    html.Br()
                    ],
                    [html.Br(),
                        dbc.Card([
                        dbc.CardBody([
                            html.H3('Egenskapsimporter', className='card-title',
                                    style=h3_style),
                        
                        html.P("De globala SHAP-värdena som presenteras i grafen nedan kan användas för att sortera utvalda råvaror och funktioner enligt deras inverkan på modellen. "
                               "Funktionen som fick det högsta globala signifikansvärdet bidrog mest som helhet oavsett förändringens riktning.",
                           style =p_style,
                           className="card-text"),
                     html.Br(),
                        dcc.Loading([dbc.Row(id = 'forecast_shap_graph_div_sv', justify = 'center')], type = random.choice(spinners))
                        ])
                        ])
                        ],
                
                    [html.Br(),
                        dbc.Card([
                        dbc.CardBody([
                            html.H3('Månatliga egenskapsimporter', className='card-title',
                                    style=h3_style),
                     html.P("Diagrammet nedan visar de lokala SHAP-värdena för funktionerna, som kan visas varje månad genom att välja önskad månad i rullgardinsmenyn. "
                            "Dessa värden anger vilka varuprisförändringar som kommer att minska eller öka arbetslösheten under den valda månaden. "
                            "Att dela en varas SHAP-värde med 100 resulterar i procentenheter i prisförändringen på en vara som bidrar till månadsförändringen i arbetslösheten. "
                            "Under grafen finns en formel för att beräkna arbetslösheten med hjälp av SHAP-värden.",
                            style =p_style,
                            className="card-text"),
                      html.Br(),
                        html.H3('Välj en månad', style =h3_style),
                                        dcc.Dropdown(id = 'forecast_local_shap_month_selection_sv',
                                                      options = options, 
                                                      style = {'font-size':16},
                                                      value = list(local_shap_df.index)[0],
                                                      multi=False ),
                                        html.Br(),
                                        
                                        html.Div(dcc.Loading(id = 'forecast_local_shap_graph_div_sv',
                                                              type = random.choice(spinners))
                                                 )
                                   ])
                        ])
                        ]]

    else:
        return [html.Div(),html.Div(),html.Div()]


@callback(

    Output('local_shap_graph_div_sv', 'children'),
    [Input('cut_off_sv', 'value'),
     Input('shap_features_switch_sv','on'),
     Input('local_shap_month_selection_sv','value'),
     Input('local_shap_data_sv','data')]
    
)
def sv_update_local_shap_graph(cut_off, only_commodities, date, local_shap_data):
    
    if local_shap_data is None:
        raise PreventUpdate
    
    try:
        locale.setlocale(locale.LC_ALL, 'sv_FI')
    except:
        locale.setlocale(locale.LC_ALL, 'sv-FI')    
    
    local_shap_df = pd.DataFrame(local_shap_data)
    local_shap_df = local_shap_df.set_index(local_shap_df.columns[0])
    local_shap_df.index = pd.to_datetime(local_shap_df.index)
    
    base_value = local_shap_df['base'].values[0]
    local_shap_df = local_shap_df.drop('base',axis=1)
    
    date = pd.to_datetime(date)
    
    
    date_str = date.strftime('%B %Y')
    prev_date = date - pd.DateOffset(months=1)
    prev_str = prev_date.strftime('%B %Y') + ' arbetslöshet'
    
    dff = local_shap_df.loc[date,:].copy()
    
  
    
    dff.index  = dff.index.str.replace('month','Nuvarande månad').str.replace('prev',prev_str)
    
    feature_values = {f:data_sv.loc[date,f] for f in data_sv.columns if f not in ['Työttömyysaste', 'change','prev','month','Inflaatio']}
    feature_values[prev_str] = data_sv.loc[date,'prev']
    feature_values['Nuvarande månad'] = data_sv.loc[date,'month']
    
    feature_values_1 = {f:data_sv.loc[date-pd.DateOffset(months=1),f] for f in data_sv.columns if f not in ['Työttömyysaste', 'change','prev','month','Inflaatio']}
    feature_values_1[prev_str] = data_sv.loc[date-pd.DateOffset(months=1),'prev']
    feature_values_1['Nuvarande månad'] = data_sv.loc[date-pd.DateOffset(months=1),'month']
    
    differences = {f:feature_values[f]-feature_values_1[f] for f in feature_values.keys()}
    changes={}
    
    # How the unemployment rate changed last month?
    changes[prev_str] = data_sv.loc[date-pd.DateOffset(months=1),'change']
    
    for d in differences.keys():
        if differences[d] >0:
            changes[d]='🔺'
        elif differences[d] <0:
            changes[d] = '🔽'
        else:
            changes[d] = '⇳'
          
    
    if only_commodities:
        dff = dff.loc[[i for i in dff.index if i not in ['Nuvarande månad', prev_str]]]
    
    
    dff = dff.sort_values(ascending = False)
    
   
    df = pd.Series(dff.iloc[cut_off+1:].copy().sum())
    
    
    
    # df.index = df.index.astype(str).str.replace('0', 'Muut {} piirrettä'.format(len(dff.iloc[cut_off+1:,:])))
    df.index = ['Andra {} funktioner'.format(len(dff.iloc[cut_off+1:]))]
    
    
    dff = pd.concat([dff.head(cut_off).copy(),df])
    dff = dff.loc[dff.index != 'Andra 0 funktioner']
    
    dff.index = dff.index.str.replace('_','')
    

    height = graph_height +200 + 10*len(dff)
    
    dff = np.round(dff*100,2)
   
    # dff = dff.sort_values()

    
    return html.Div([dcc.Graph(id = 'local_shap_graph_sv',
                     config = config_plots_sv,
                         figure = go.Figure(data=[go.Bar(y =['{} ({} {})'.format(i, round(feature_values[i],2),changes[i]) if i in feature_values.keys() else '{}: {}'.format(i,dff.loc[i]) for i in dff.index], 
                      x = dff.values,
                      orientation='h',
                      name = '',
                      
                      # marker_color = ['cyan' if i not in ['Month',prev_str] else 'black' for i in dff.index],
                       marker = dict(color = list(map(sv_set_color,dff.index,dff.values))),
                      text = dff.values,
                      hovertemplate = ['<b>{}</b><br><b>  SHAP värde</b>: {}<br><b>  Värde under nuvarande månad</b>: {} {}<br><b>  Värde under föregående månad</b>: {}'.format(i,dff.loc[i], round(feature_values[i],2),changes[i],round(feature_values_1[i],2)) if i in feature_values.keys() else '{}: {}'.format(i,dff.loc[i]) for i in dff.index],
                          textfont = dict(
                               family='Cadiz Semibold', 
                              size = 16))],
         layout=go.Layout(title = dict(text = 'Lokala funktionsbetydelser<br>SHAP värden: '+date_str,
                                                                     x=.5,
                                                                     font=dict(
                                                                          family='Cadiz Semibold',
                                                                          size=20
                                                                         )),
                                                      
                                                        
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
                                                                     # xanchor='center',
                                                                     # yanchor='top',
                                                                    font=dict(
                                                             size=12,
                                                             family='Cadiz Book'
                                                            )),
                                                        hoverlabel = dict(font=dict(
                                                             size=18,
                                                             family='Cadiz Book'
                                                            )),
                                                        height=height,#graph_height+200,
                                                        xaxis = dict(title=dict(text = 'SHAP värde',
                                                                                font=dict(
                                                                                    size=16, 
                                                                                    family = 'Cadiz Semibold'
                                                                                    )),
                                                                     automargin=True,
                                                                     # tickformat = ' ',
                                                                      # categoryorder='total descending',
                                                                     tickfont = dict(
                                                                         family = 'Cadiz Semibold', 
                                                                          size = 14
                                                                         )),
                                                        yaxis = dict(title=dict(text = 'Funktion: 🔺 = ökning, 🔽 = minskning, ⇳ = samma som föregående månad',
                                                                               font=dict(
                                                                                    size=16, 
                                                                                   family = 'Cadiz Semibold'
                                                                                   )),
                                                                    automargin=True,
                                                                    tickfont = dict(
                                                                        family = 'Cadiz Semibold', 
                                                                         size = 14
                                                                        ))
                                                        ))),
                     html.P('Prognos ≈ Föregående prognosad arbetslöshet + [ {} + SUM( SHAP värden ) ] / 100'.format(round(100*base_value,2)))
                     ])


@callback(

    Output('forecast_local_shap_graph_div_sv', 'children'),
    [Input('forecast_cut_off_sv', 'value'),
     Input('forecast_shap_features_switch_sv','on'),
     Input('forecast_local_shap_month_selection_sv','value'),
     Input('local_forecast_shap_data_sv','data'),
     State('forecast_data_sv','data')]
    
)
def sv_update_local_forecast_shap_graph(cut_off, only_commodities, date, local_shap_data, forecast_data):
    
    if local_shap_data is None:
        raise PreventUpdate
    
    try:
        locale.setlocale(locale.LC_ALL, 'sv_FI')
    except:
        locale.setlocale(locale.LC_ALL, 'sv-FI')    
        
    forecast_data = pd.DataFrame(forecast_data).set_index('Aika')
    forecast_data.index = pd.to_datetime(forecast_data.index )
    forecast_data = forecast_data.drop('n_feat',axis=1)    
    
    local_shap_df = pd.DataFrame(local_shap_data)
    local_shap_df = local_shap_df.set_index(local_shap_df.columns[0])
    local_shap_df.index = pd.to_datetime(local_shap_df.index)
    
    base_value = local_shap_df['base'].values[0]
    local_shap_df = local_shap_df.drop('base',axis=1)
    
    date = pd.to_datetime(date)
    
    
    date_str = date.strftime('%B %Y')
    prev_date = date - pd.DateOffset(months=1)
    prev_str = prev_date.strftime('%B %Y') + ' arbetslöshet'
    
    dff = local_shap_df.loc[date,:].copy()
    
  
    
    dff.index  = dff.index.str.replace('month','Nuvarande månad').str.replace('prev',prev_str)
    
    feature_values = {f:forecast_data.loc[date,f] for f in forecast_data.columns if f not in ['Työttömyysaste', 'change','prev','month','Inflaatio']}
    feature_values[prev_str] = forecast_data.loc[date,'prev']
    feature_values['Nuvarande månad'] = forecast_data.loc[date,'month']
    
    try:
        feature_values_1 = {f:forecast_data.loc[date-pd.DateOffset(months=1),f] for f in forecast_data.columns if f not in ['Työttömyysaste', 'change','prev','month','Inflaatio']}
        feature_values_1[prev_str] = forecast_data.loc[date-pd.DateOffset(months=1),'prev']
        feature_values_1['Nuvarande månad'] = forecast_data.loc[date-pd.DateOffset(months=1),'month']
    except:
        feature_values_1 = {f:data_sv.loc[date-pd.DateOffset(months=1),f] for f in data_sv.columns if f not in ['Työttömyysaste', 'change','prev','month','Inflaatio']}
        feature_values_1[prev_str] = data_sv.loc[date-pd.DateOffset(months=1),'prev']
        feature_values_1['Nuvarande månad'] = data_sv.loc[date-pd.DateOffset(months=1),'month']
    
        
    
    differences = {f:feature_values[f]-feature_values_1[f] for f in feature_values.keys()}
    changes={}
    
    # How the unemployment rate changed last month?
    try:
        changes[prev_str] = forecast_data.loc[date-pd.DateOffset(months=1),'change']
    except:
        changes[prev_str] = data_sv.loc[date-pd.DateOffset(months=1),'change']
    
    for d in differences.keys():
        if differences[d] >0:
            changes[d]='🔺'
        elif differences[d] <0:
            changes[d] = '🔽'
        else:
            changes[d] = '⇳'
          
    
    if only_commodities:
        dff = dff.loc[[i for i in dff.index if i not in ['Nuvarande månad', prev_str]]]
    
    
    dff = dff.sort_values(ascending = False)
    
   
    df = pd.Series(dff.iloc[cut_off+1:].copy().sum())
    
    
    
    # df.index = df.index.astype(str).str.replace('0', 'Muut {} piirrettä'.format(len(dff.iloc[cut_off+1:,:])))
    df.index = ['Andra {} funktioner'.format(len(dff.iloc[cut_off+1:]))]
    
    
    dff = pd.concat([dff.head(cut_off).copy(),df])
    dff = dff.loc[dff.index != 'Andra 0 funktioner']
    
    dff.index = dff.index.str.replace('_','')
    

    height = graph_height +200 + 10*len(dff)
    
    dff = np.round(dff*100,2)
   
    # dff = dff.sort_values()

    
    return html.Div([dcc.Graph(id = 'local_shap_graph_sv',
                     config = config_plots_sv,
                         figure = go.Figure(data=[go.Bar(y =['{} ({} {})'.format(i, round(feature_values[i],2),changes[i]) if i in feature_values.keys() else '{}: {}'.format(i,dff.loc[i]) for i in dff.index], 
                      x = dff.values,
                      orientation='h',
                      name = '',
                      
                      # marker_color = ['cyan' if i not in ['Month',prev_str] else 'black' for i in dff.index],
                       marker = dict(color = list(map(sv_set_color,dff.index,dff.values))),
                      text = dff.values,
                      hovertemplate = ['<b>{}</b><br><b>  SHAP värde</b>: {}<br><b>  Värde under nuvarande månad</b>: {} {}<br><b>  Värde under föregående månad</b>: {}'.format(i,dff.loc[i], round(feature_values[i],2),changes[i],round(feature_values_1[i],2)) if i in feature_values.keys() else '{}: {}'.format(i,dff.loc[i]) for i in dff.index],
                          textfont = dict(
                               family='Cadiz Semibold', 
                              size = 16))],
         layout=go.Layout(title = dict(text = 'Lokala funktionsbetydelser<br>SHAP värden: '+date_str,
                                                                     x=.5,
                                                                     font=dict(
                                                                          family='Cadiz Semibold',
                                                                          size=20
                                                                         )),
                                                      
                                                        
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
                                                                     # xanchor='center',
                                                                     # yanchor='top',
                                                                    font=dict(
                                                             size=12,
                                                             family='Cadiz Book'
                                                            )),
                                                        hoverlabel = dict(font=dict(
                                                             size=18,
                                                             family='Cadiz Book'
                                                            )),
                                                        height=height,#graph_height+200,
                                                        xaxis = dict(title=dict(text = 'SHAP värde',
                                                                                font=dict(
                                                                                    size=16, 
                                                                                    family = 'Cadiz Semibold'
                                                                                    )),
                                                                     automargin=True,
                                                                     # tickformat = ' ',
                                                                      # categoryorder='total descending',
                                                                     tickfont = dict(
                                                                         family = 'Cadiz Semibold', 
                                                                          size = 14
                                                                         )),
                                                        yaxis = dict(title=dict(text = 'Funktion: 🔺 = ökning, 🔽 = minskning, ⇳ = samma som föregående månad',
                                                                               font=dict(
                                                                                    size=16, 
                                                                                   family = 'Cadiz Semibold'
                                                                                   )),
                                                                    automargin=True,
                                                                    tickfont = dict(
                                                                        family = 'Cadiz Semibold', 
                                                                         size = 14
                                                                        ))
                                                        ))),
                     html.P('Prognos ≈ Föregående prognosad arbetslöshet + [ {} + SUM( SHAP värden ) ] / 100'.format(round(100*base_value,2)))
                     ])

    
@callback(

    Output('cut_off_indicator_sv','children'),
    [Input('cut_off_sv','value')]    
    
)
def sv_update_cut_off_indicator(cut_off):
    return [html.P('Du valde {} funktioner.'.format(cut_off).replace(' 1 funktioner',' en funktion'), style = p_center_style)]


@callback(

    Output('forecast_cut_off_indicator_sv','children'),
    [Input('forecast_cut_off_sv','value')]    
    
)
def sv_update_forecast_cut_off_indicator(cut_off):
    return [html.P('Du valde {} funktioner.'.format(cut_off).replace(' 1 funktioner',' en funktion'), style = p_center_style)]
    
@callback(

    Output('cut_off_div_sv','children'),
    [Input('shap_data_sv','data')]    
    
)
def sv_update_shap_slider(shap):
    if shap is None:
        raise PreventUpdate

    shap_df = pd.DataFrame(shap)
    
    
    return [html.P('Välj hur många funktioner som visas i diagrammet nedan.',
                       style = p_center_style),
                dcc.Slider(id = 'cut_off_sv',
                   min = 1, 
                   max = len(shap_df),
                   value = {True:len(shap_df), False: int(math.ceil(.2*len(shap_df)))}[len(shap_df)<=25],
                   step = 1,
                   marks=None,
                   tooltip={"placement": "top", "always_visible": True},
                   )]

@callback(

    Output('forecast_cut_off_div_sv','children'),
    [Input('forecast_shap_data_sv','data')]    
    
)
def sv_update_forecast_shap_slider(shap):
    if shap is None:
        raise PreventUpdate

    shap_df = pd.DataFrame(shap)
    
    
    return [html.P('Välj hur många funktioner som visas i diagrammet nedan.',
                       style = p_center_style),
                dcc.Slider(id = 'forecast_cut_off_sv',
                   min = 1, 
                   max = len(shap_df),
                   value = {True:len(shap_df), False: int(math.ceil(.2*len(shap_df)))}[len(shap_df)<=25],
                   step = 1,
                   marks=None,
                   tooltip={"placement": "top", "always_visible": True},
                   )]

@callback(

    Output('shap_graph_div_sv', 'children'),
    [Input('cut_off_sv', 'value'),
     Input('shap_features_switch_sv','on'),
     State('shap_data_sv','data')]
    
)
def sv_update_shap_graph(cut_off, only_commodities, shap):
    
    if shap is None:
        raise PreventUpdate
        
    
    shap_df = pd.DataFrame(shap)
    shap_df = shap_df.set_index(shap_df.columns[0])
    shap_df.index = shap_df.index.str.replace('Kuukausi','Månad')
    shap_df.index = shap_df.index.str.replace('Edellisen kuukauden työttömyysaste','Föregående arbetslöshetstal')
  
    
    
    if only_commodities:
        shap_df = shap_df.loc[[i for i in shap_df.index if i not in ['Månad', 'Föregående arbetslöshetstal']]]
    
    
    shap_df = shap_df.sort_values(by='SHAP', ascending = False)
    
   
    df = pd.DataFrame(shap_df.iloc[cut_off+1:,:].sum())
    df = df.T
    df.index = df.index.astype(str).str.replace('0', 'Andra {} funktioner'.format(len(shap_df.iloc[cut_off+1:,:])))
    
    
    shap_df = pd.concat([shap_df.head(cut_off),df])
    shap_df = shap_df.loc[shap_df.index != 'Andra 0 funktioner']
    
    shap_df.index = shap_df.index.str.replace('_','')
    

    height = graph_height +200 + 10*len(shap_df)
    
    
    return dcc.Graph(id = 'shap_graph_sv',
                     config = config_plots_sv,
                         figure = go.Figure(data=[go.Bar(y =shap_df.index, 
                      x = np.round(100*shap_df.SHAP,2),
                      orientation='h',
                      name = '',
                      marker_color = ['aquamarine' if i not in ['Månad','Föregående arbetslöshetstal'] else 'black' for i in shap_df.index],
                      # marker = dict(color = 'turquoise'),
                      text = np.round(100*shap_df.SHAP,2),
                      hovertemplate = '<b>%{y}</b>: %{x}',
                          textfont = dict(
                               family='Cadiz Semibold', 
                              size = 20))],
         layout=go.Layout(title = dict(text = 'Globala funktionsviktigheter<br>Genomsnittliga |SHAP - värden|',
                                                                     x=.5,
                                                                     font=dict(
                                                                          family='Cadiz Semibold',
                                                                          size=20
                                                                         )),
                                                      
                                                        
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
                                                                     # xanchor='center',
                                                                     # yanchor='top',
                                                                    font=dict(
                                                             size=12,
                                                             family='Cadiz Book'
                                                            )),
                                                        
                                                        height=height,#graph_height+200,
                                                        xaxis = dict(title=dict(text = 'Genomsnittligt |SHAP-värde|',
                                                                                font=dict(
                                                                                    size=18, 
                                                                                    family = 'Cadiz Semibold'
                                                                                    )),
                                                                     automargin=True,
                                                                     tickfont = dict(
                                                                         family = 'Cadiz Semibold', 
                                                                          size = 16
                                                                         )),
                                                        yaxis = dict(title=dict(text = 'Funktion',
                                                                               font=dict(
                                                                                    size=18, 
                                                                                   family = 'Cadiz Semibold'
                                                                                   )),
                                                                    tickformat = ' ',
                                                                     categoryorder='total ascending',
                                                                    automargin=True,
                                                                    tickfont = dict(
                                                                        family = 'Cadiz Semibold', 
                                                                         size = 16
                                                                        ))
                                                        )))
@callback(

    Output('forecast_shap_graph_div_sv', 'children'),
    [Input('forecast_cut_off_sv', 'value'),
     Input('forecast_shap_features_switch_sv','on'),
     State('forecast_shap_data_sv','data')]
    
)
def sv_update_forecast_shap_graph(cut_off, only_commodities, shap):
    
    if shap is None:
        raise PreventUpdate
        
    
    shap_df = pd.DataFrame(shap)
    shap_df = shap_df.set_index(shap_df.columns[0])
    shap_df.index = shap_df.index.str.replace('Kuukausi','Månad')
    shap_df.index = shap_df.index.str.replace('Edellisen kuukauden työttömyysaste','Föregående arbetslöshetstal')
  
    
    
    if only_commodities:
        shap_df = shap_df.loc[[i for i in shap_df.index if i not in ['Månad', 'Föregående arbetslöshetstal']]]
    
    
    shap_df = shap_df.sort_values(by='SHAP', ascending = False)
    
   
    df = pd.DataFrame(shap_df.iloc[cut_off+1:,:].sum())
    df = df.T
    df.index = df.index.astype(str).str.replace('0', 'Andra {} funktioner'.format(len(shap_df.iloc[cut_off+1:,:])))
    
    
    shap_df = pd.concat([shap_df.head(cut_off),df])
    shap_df = shap_df.loc[shap_df.index != 'Andra 0 funktioner']
    
    shap_df.index = shap_df.index.str.replace('_','')
    

    height = graph_height +200 + 10*len(shap_df)
    
    
    return dcc.Graph(id = 'shap_graph_sv',
                     config = config_plots_sv,
                         figure = go.Figure(data=[go.Bar(y =shap_df.index, 
                      x = np.round(100*shap_df.SHAP,2),
                      orientation='h',
                      name = '',
                      marker_color = ['aquamarine' if i not in ['Månad','Föregående arbetslöshetstal'] else 'black' for i in shap_df.index],
                      # marker = dict(color = 'turquoise'),
                      text = np.round(100*shap_df.SHAP,2),
                      hovertemplate = '<b>%{y}</b>: %{x}',
                          textfont = dict(
                               family='Cadiz Semibold', 
                              size = 20))],
         layout=go.Layout(title = dict(text = 'Globala funktionsviktigheter<br>Genomsnittliga |SHAP - värden|',
                                                                     x=.5,
                                                                     font=dict(
                                                                          family='Cadiz Semibold',
                                                                          size=20
                                                                         )),
                                                      
                                                        
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
                                                                     # xanchor='center',
                                                                     # yanchor='top',
                                                                    font=dict(
                                                             size=12,
                                                             family='Cadiz Book'
                                                            )),
                                                        
                                                        height=height,#graph_height+200,
                                                        xaxis = dict(title=dict(text = 'Genomsnittligt |SHAP-värde|',
                                                                                font=dict(
                                                                                    size=18, 
                                                                                    family = 'Cadiz Semibold'
                                                                                    )),
                                                                     automargin=True,
                                                                     tickfont = dict(
                                                                         family = 'Cadiz Semibold', 
                                                                          size = 16
                                                                         )),
                                                        yaxis = dict(title=dict(text = 'Funktion',
                                                                               font=dict(
                                                                                    size=18, 
                                                                                   family = 'Cadiz Semibold'
                                                                                   )),
                                                                    tickformat = ' ',
                                                                     categoryorder='total ascending',
                                                                    automargin=True,
                                                                    tickfont = dict(
                                                                        family = 'Cadiz Semibold', 
                                                                         size = 16
                                                                        ))
                                                        )))    

@callback(
    Output("forecast_download_sv", "data"),
    [Input("forecast_download_button_sv", "n_clicks")],
    [State('forecast_data_sv','data'),
     State('method_selection_results_sv','data'),
     State('change_weights_sv','data'),          
      State('forecast_shap_data_sv','data'),
      State('local_forecast_shap_data_sv','data'),
     ]
    
    
)
def sv_download_forecast_data(n_clicks, df, method_selection_results, weights_dict, shap_data, local_shap_data):
    
    if n_clicks > 0:
        
        
        df = pd.DataFrame(df).set_index('Aika').copy()
        df.index = pd.to_datetime(df.index)
        df.index.name = 'Tid'
        forecast_size = len(df)
        n_feat = df.n_feat.values[0]
        df.drop('n_feat',axis=1,inplace=True)
        
        df = df.rename(columns = {'change':'Beräknad månadsförändring (procentenheter)',
                                  'month':'Månad',
                                  'prev': 'Föregående prognos',
                                  'Työttömyysaste': 'Arbetslöshet (prognos)'})
        
        
        features = sorted(list(weights_dict.keys()))
        
        weights_df = pd.DataFrame([weights_dict]).T
        weights_df.index.name = 'Varor'
        weights_df.columns = ['Förväntad genomsnittlig månadsförändring (%)']
        
        model_name = method_selection_results['model']
             
        pca = method_selection_results['pca']
        explained_variance = method_selection_results['explained_variance']
        hyperparam_grid = method_selection_results['hyperparam_grid']
        
        if pca:
            metadict = {
                        'Principalkomponentanalys' : pca,
                        'Förklarad variation': str(int(100*explained_variance))+'%',
                        'Principalkomponenter': n_feat,
                        'Modell': model_name,
                        'Antal tillämpade egenskaper':len(features),
                        'Råvaror' : ',\n'.join(features),
                        'Prognosens längd': str(forecast_size)+' månader'
                        
                        }

        else:
            metadict = {
                            'Principalkomponentanalys' : pca,
                            'Modell': model_name,
                            'Antal tillämpade egenskaper':len(features),
                            'Råvaror' : ',\n'.join(features),
                            'Prognosens längd': str(forecast_size)+' månader'
                            }
        
        metadata = pd.DataFrame([metadict]).T
        metadata.index.name = ''
        metadata.columns = ['Värde']
        
        hyperparam_df = pd.DataFrame([hyperparam_grid]).T
        hyperparam_df.index.name = 'Hyperparameter'
        hyperparam_df.columns = ['Värde']   
        hyperparam_df['Värde'] = hyperparam_df['Värde'].astype(str)
        
        shap_df = pd.DataFrame(shap_data)
        shap_df = shap_df.set_index(shap_df.columns[0])
        shap_df.index.name = 'Funktion'
        shap_df.SHAP = np.round(100*shap_df.SHAP,2)
        shap_df.index = shap_df.index.str.replace('Kuukausi', 'Månad')
        shap_df.index = shap_df.index.str.replace('Edellisen kuukauden työttömyysaste', 'Arbetslöshet föregående månad')
        shap_df.index = shap_df.index.str.replace('_','')
        
        local_shap_df = pd.DataFrame(local_shap_data)
        local_shap_df = local_shap_df.set_index(local_shap_df.columns[0])
        local_shap_df.index = pd.to_datetime(local_shap_df.index)
        local_shap_df.index.name = 'Tid'
        local_shap_df = local_shap_df.rename(columns = {'month':'Månad',
                                  'prev': 'Arbetslöshet föregående månad'})
        local_shap_df = local_shap_df.multiply(100, axis=1)
        local_shap_df.columns = local_shap_df.columns.str.replace('_','')
        local_shap_df.drop('base',axis=1,inplace=True)

  
        data_ = data_sv.copy().rename(columns={'change':'Förändring (procentenheter)',
                                      'prev':'Tidigare arbetslöshetstal -%',
                                      'month':'Månad',
                                      'Työttömyysaste':'Arbetslöshet',
                                      'Inflaatio':'Inflation'})
        data_.index.name = 'Tid'
        
        
        xlsx_io = io.BytesIO()
        writer = pd.ExcelWriter(xlsx_io, engine='xlsxwriter')
        
        
        
        data_.to_excel(writer, sheet_name= 'Data')
        df.to_excel(writer, sheet_name= 'Prognosuppgifter')
        weights_df.to_excel(writer, sheet_name= 'Indexändringar')
        hyperparam_df.to_excel(writer, sheet_name= 'Hyperparametrar')
        shap_df.to_excel(writer, sheet_name= 'Funktion betydelser')
        local_shap_df.to_excel(writer, sheet_name= 'Månatliga betydelser')
        metadata.to_excel(writer, sheet_name= 'Metadata')
        
        
        workbook = writer.book
        workbook.set_properties(
        {
            "title": "Skewed Phillips",
            "subject": "Prognosresultat",
            "author": "Tuomas Poukkula",
            # "company": "Gofore Ltd.",
            "keywords": "XAI, Prediktiv analys",
            "comments": "Kolla aplikationen på: https://skewedphillips.herokuapp.com"
        }
        )
        


        writer.save()
        

        
        output = xlsx_io.getvalue()

        return dcc.send_bytes(output, 'Prognosdata med {} råvaror '.format(len(features)).replace(' 1 råvaror', ' en råvara')+datetime.now().strftime('%d_%m_%Y')+'.xlsx')
        
@callback(
    Output("test_download_sv", "data"),
    [Input("test_download_button_sv", "n_clicks"),
    State('test_data_sv','data'),
    State('method_selection_results_sv','data'),
    State('change_weights_sv','data'),
    State('shap_data_sv','data'),
    State('local_shap_data_sv','data')
    ]
    
)
def sv_download_test_data(n_clicks, 
                       df, 
                       method_selection_results, 
                        weights_dict, 
                       shap_data,
                       local_shap_data):
    
    if n_clicks > 0:
        
        df = pd.DataFrame(df).set_index('Aika').copy()
        df.index = pd.to_datetime(df.index)
        df.index.name = 'Tid'
        mape = df.mape.values[0]
        test_size = len(df)
        n_feat = df.n_feat.values[0]
        df.drop('n_feat',axis=1,inplace=True)
        df.drop('mape',axis=1,inplace=True)
        df = df.rename(columns = {'change':'Månadsförändring (procentenheter)',
                                  'month':'Månad',
                                  'Ennustettu muutos':'Beräknad månadsförändring (procentenheter)',
                                  'Ennuste':'Prognoserad arbetslöshet',
                                  'Työttömyysaste':'Arbetslöshet (%)',
                                  'prev': 'Föregående prognos'})
        
        
        features = sorted(list(weights_dict.keys()))
        
        model_name = method_selection_results['model']
             
        pca = method_selection_results['pca']
        explained_variance = method_selection_results['explained_variance']
        hyperparam_grid = method_selection_results['hyperparam_grid']
        
        if pca:
            metadict = {'MAPE': str(round(100*mape,2))+'%',
                        'Principalkomponentanalys' : pca,
                        'Förklarad variation': str(int(100*explained_variance))+'%',
                        'Principalkomponenter': n_feat,
                        'Modell': model_name,
                        'Antal tillämpade egenskaper':len(features),
                        'Råvaror' : ',\n'.join(features),
                        'Provningslängd': str(test_size)+' månader'
                        }
        else:
            metadict = {'MAPE': str(round(100*mape,2))+'%',
                            'Principalkomponentanalys' : pca,
                            'Modell': model_name,
                            'Antal tillämpade egenskaper':len(features),
                            'Råvaror' : ',\n'.join(features),
                            'Provningslängd': str(test_size)+' månader'
                            }
        
        metadata = pd.DataFrame([metadict]).T
        metadata.index.name = ''
        metadata.columns = ['Värde']
        
        hyperparam_df = pd.DataFrame([hyperparam_grid]).T
        hyperparam_df.index.name = 'Hyperparameter'
        hyperparam_df.columns = ['Värde']   
        hyperparam_df['Värde'] = hyperparam_df['Värde'].astype(str)
        
        shap_df = pd.DataFrame(shap_data)
        shap_df = shap_df.set_index(shap_df.columns[0])
        shap_df.index.name = 'Funktion'
        shap_df.SHAP = np.round(100*shap_df.SHAP,2)
        shap_df.index = shap_df.index.str.replace('Kuukausi', 'Månad')
        shap_df.index = shap_df.index.str.replace('Edellisen kuukauden työttömyysaste', 'Arbetslöshet föregående månad')
        shap_df.index = shap_df.index.str.replace('_','')
        
        local_shap_df = pd.DataFrame(local_shap_data)
        local_shap_df = local_shap_df.set_index(local_shap_df.columns[0])
        local_shap_df.index = pd.to_datetime(local_shap_df.index)
        local_shap_df.index.name = 'Tid'
        local_shap_df = local_shap_df.rename(columns = {'month':'Månad',
                                  'prev': 'Arbetslöshet föregående månad'})
        local_shap_df = local_shap_df.multiply(100, axis=1)
        local_shap_df.columns = local_shap_df.columns.str.replace('_','')
        local_shap_df.drop('base',axis=1,inplace=True)
        
        
        xlsx_io = io.BytesIO()
        writer = pd.ExcelWriter(xlsx_io, engine='xlsxwriter')
        
        
        df.to_excel(writer, sheet_name= 'Provningsdata')
        metadata.to_excel(writer, sheet_name= 'Metadata')
        hyperparam_df.to_excel(writer, sheet_name= 'Modellhyperparametrar')
        shap_df.to_excel(writer, sheet_name= 'Funktion betydelser')
        local_shap_df.to_excel(writer, sheet_name= 'Månatliga betydelser')
        
        workbook = writer.book
        workbook.set_properties(
        {
            "title": "Skewed Phillips",
            "subject": "Provningsresultat",
            "author": "Tuomas Poukkula",
            # "company": "Gofore Ltd.",
            "keywords": "XAI, Prediktiv analys",
            "comments": "Kolla aplikationen på: https://skewedphillips.herokuapp.com"
        }
        )

        writer.save()
        

        
        output = xlsx_io.getvalue()

        return dcc.send_bytes(output, 'Testresultat med {} råvaror '.format(len(features)).replace(' 1 råvaror', ' en råvara')+datetime.now().strftime('%d_%m_%Y')+'.xlsx')


@callback(

    Output('test_graph_div_sv', 'children'),
    
      [Input('test_chart_type_sv','value')],
      [State('test_data_sv','data')]
    
)
def sv_update_test_chart_type(chart_type,df):
    
    
    
    df = pd.DataFrame(df).set_index('Aika')
    df.index = pd.to_datetime(df.index)

    
    df = df.reset_index().drop_duplicates(subset='Aika', keep='last').set_index('Aika')[['Työttömyysaste','Ennuste','mape']].dropna(axis=0)

    return [dcc.Graph(id ='test_graph_sv', 
                     figure = sv_plot_test_results(df,chart_type), 
                     config = config_plots_sv)     ]



@callback(

    Output('forecast_graph_div_sv', 'children'),
    
      [Input('chart_type_sv','value')],
      [State('forecast_data_sv','data')]
    
)
def sv_update_forecast_chart_type(chart_type,df):
    
    df = pd.DataFrame(df).set_index('Aika')
    df.index = pd.to_datetime(df.index)

    return dcc.Graph(id = 'forecast_graph_sv',
                    figure = sv_plot_forecast_data(df, chart_type = chart_type), 
                    config = config_plots_sv),

@callback(

    Output('ev_placeholder_sv', 'style'),
    [Input('pca_switch_sv', 'on')]
)    
def sv_add_ev_slider(pca):
    
    return {False: {'margin' : '5px 5px 5px 5px', 'display':'none'},
           True: {'margin' : '5px 5px 5px 5px'}}[pca]

@callback(

    Output('ev_slider_update_sv', 'children'),
    [Input('pca_switch_sv', 'on'),
    Input('ev_slider_sv', 'value')]

)
def sv_update_ev_indicator(pca, explained_variance):
    
    return {False: [html.Div([html.P('Du valde {} % förklarad varians.'.format(int(100*explained_variance)),
                                                               style = p_center_style)
                                                       ], style = {'display':'none'}
                                                      )],
            True: [html.Div([html.P('Du valde {} % förklarad varians.'.format(int(100*explained_variance)),
                                                               style = p_center_style)
                                                       ]
                                                      )]}[pca]



@callback(
    Output('feature_selection_sv','value'),
    [Input('select_all_sv', 'on'),
    Input('feature_selection_sv','options')]
)
def sv_update_feature_list(on,options):
       
        
    if on:
        return [f['label'] for f in options]
    else:
        raise PreventUpdate
        
@callback(
    
    Output('select_all_sv','on'),
    [Input('feature_selection_sv','value'),
     State('feature_selection_sv','options')]
    
)
def sv_update_select_all_on(features,options):
    
    return len(features) == len(options)

@callback(
    [
     
     Output('select_all_sv','label'),
     Output('select_all_sv','disabled')
     ],
    [Input('select_all_sv', 'on')]
)    
def sv_update_switch(on):
    
    if on:
        return {'label':'Allt är valt. Du kan ta bort varor genom att klicka på krossen i listan.',
                       'style':{'text-align':'center', 'font-size':p_font_size,
                                #'font-family':'Cadiz Semibold'
                                }
                      },True
    

    else:
        return dict(label = 'Välj alla',style = {'font-size':p_font_size, 
                                                      # #'fontFamily':'Cadiz Semibold'
                                                      }),False



@callback(

    Output('test_button_div_sv','children'),
    [

     Input('features_values_sv','data')
     ]    
    
)
def sv_add_test_button(features_values):
    
    if features_values is None:
        raise PreventUpdate 
        
        
    
    elif len(features_values) == 0:
        return [html.P('Välj varor först',
                       style = p_style)]
    
    else:
               
        
        return dbc.Button('Testa',
                           id='test_button_sv',
                           n_clicks=0,
                           outline=False,
                           className="me-1",
                           size='lg',
                           color='success',
                           style = dict(fontSize=25,
                                        # fontFamily='Cadiz Semibold'
                                        )
                          )

@callback(
    Output('test_size_indicator_sv','children'),
    [Input('test_slider_sv','value')]
)
def sv_update_test_size_indicator(value):
    
    return [html.Br(),html.P('Du valde {} månader som testdata.'.format(value),
                             style = p_center_style)]

@callback(
    Output('forecast_slider_indicator_sv','children'),
    [Input('forecast_slider_sv','value')]
)
def sv_update_forecast_size_indicator(value):
    
    return [html.Br(),html.P('Du valde {} månader för prognoser.'.format(value),
                             style = p_center_style)]




@callback(

    Output('timeseries_selection_sv', 'children'),
    [
     
     Input('features_values_sv','data')
     ]    
    
)
def sv_update_timeseries_selections(features_values):
    
    features = sorted(list(features_values.keys()))
    
    return [
            html.Br(),
            html.H3('Visa varuindexets tidsserier',
                    style =h3_style),
            
            html.P("Använd denna graf för att se varuindexets utveckling månadsvis, vilket gör det lättare att bättre bedöma vilken typ av inflationsförväntningar som ska ingå i prognosen.",
                   style = p_style),
            html.H3("Välj en vara",style = h3_style),
            dcc.Dropdown(id = 'timeseries_selection_dd_sv',
                        options = [{'value':feature, 'label':feature} for feature in features],
                        value = [features[0]],
                        style = {
                            'font-size':"0.9rem", 
                            #'font-family':'Cadiz Book',
                            'color': 'black'},
                        multi = True)
            ]


@callback(

    Output('timeseries_sv', 'children'),
    [Input('timeseries_selection_dd_sv', 'value')]    
    
)
def sv_update_time_series(values):
    
    traces = [go.Scatter(x = data_sv.index, 
                         y = data_sv[value],
                         showlegend=True,   
                         hovertemplate = '%{x}'+'<br>%{y}',
                         name = ' '.join(value.split()[1:]),
                         mode = 'lines+markers') for value in values]
    return html.Div([dcc.Graph(figure=go.Figure(data=traces,
                                      layout = go.Layout(title = dict(text = "Index över <br>utvalda varor",
                                                                      x=.5,
                                                                      font=dict(
                                                                           family='Cadiz Semibold',
                                                                           size=20
                                                                          )),
                                                         
                                                         height=graph_height,
                                                         hoverlabel=dict(font=dict(
                                                              family='Cadiz Book',
                                                             size=18)),
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
                                                                      # xanchor='center',
                                                                      # yanchor='top',
                                                                     font=dict(
                                                              size=14,
                                                              family='Cadiz Book'
                                                             )),
                                                         xaxis = dict(title=dict(text = 'Tid',
                                                                                 font=dict(
                                                                                     size=18, 
                                                                                     family = 'Cadiz Semibold'
                                                                                     )),
                                                                      automargin=True,
                                                                      tickfont = dict(
                                                                          family = 'Cadiz Semibold', 
                                                                           size = 16
                                                                          )),
                                                         yaxis = dict(title=dict(text = "Indextal (basår = 2010)",
                                                                                font=dict(
                                                                                     size=18, 
                                                                                    family = 'Cadiz Semibold'
                                                                                    )),
                                                                     tickformat = ' ',
                                                                     automargin=True,
                                                                     tickfont = dict(
                                                                         family = 'Cadiz Semibold', 
                                                                          size = 16
                                                                         ))
                                                         )
                                      ),
                     config = config_plots_sv
                     )])




@callback(
    
    Output('forecast_button_div_sv','children'),
    [
     
     Input('features_values_sv','data')
     ]   
    
)
def sv_add_predict_button(features_values):
    
    if features_values is None:
        raise PreventUpdate 
    
    elif len(features_values) == 0:
        return [html.P('Välj varor först',
                       style = p_style)]
    
    
        
    else:
        return [dbc.Button('Prognosera',
                   id='forecast_button_sv',
                   n_clicks=0,
                   outline=False,
                   className="me-1",
                   size='lg',
                   color='success',
                   style = dict(fontSize=25,
                                # fontFamily='Cadiz Semibold'
                                )
                   
                   ),
                html.Br(),
                html.Div(id = 'forecast_download_button_div_sv',style={'textAlign':'center'})]





@callback(

    Output('slider_prompt_div_sv','children'),
    [Input('slider_sv', 'value'),
     State('averaging_sv', 'on')]    
    
)
def sv_update_slider_prompt(value, averaging):
    
        
    if averaging:
    
        return [html.Br(),html.P('Du valde genomsnittet för de senaste {} månaderna.'.format(value),
                      style = p_center_style),
                html.Br(),
                html.P('Du kan fortfarande justera enskilda värden på inmatningsrutan.',
                       style = p_center_style)]
    else:
        return [html.Br(),html.P('Du valde den beräknade månatliga förändringen till {} %.'.format(value),
                      style = p_center_style),
                html.Br(),
                html.P('Du kan fortfarande justera enskilda värden på inmatningsrutan.',
                       style = p_center_style)]
        
 

@callback(

    Output('corr_selection_div_sv', 'children'),
    [

     Input('features_values_sv','data')
     ]
    
)
def sv_update_corr_selection(features_values):
    
    features = sorted(list(features_values.keys()))

    return html.Div([
            html.Br(),
            html.H3("Se förhållandet mellan prisindexet för den valda råvaran och arbetslösheten",
                    style=h3_style),
            
            html.P("Använd denna graf för att visa förhållandet och korrelationen mellan prisindexet för den valda råvaran och arbetslösheten eller månadsförändringen. I teorin korrelerar en bra prediktiv funktion starkt med den förutsägbara variabeln. "
                   "För bättre visualisering visas trendlinjen i grafen endast när en vara väljs.",
                   style = p_style),
            html.P("(Fortsätter efter diagrammet)",style={
                        'font-style':'italic',
                        'font-size':p_font_size,
                       'text-align':'center'}
                ),
        html.H3('Välj en vara', style = h3_style),
        dcc.Dropdown(id = 'corr_feature_sv',
                        multi = True,
                        # clearable=False,
                        options = [{'value':feature, 'label':feature} for feature in features],
                        value = [features[0]],
                        style = {'font-size':"0.9rem", 
                                 #'font-family':'Cadiz Book'
                                 },
                        placeholder = 'Välj en vara')
        ]
        )

@callback(

    Output('feature_corr_selection_div_sv', 'children'),
    [

     Input('features_values_sv','data')
     ]
    
)
def sv_update_feature_corr_selection(features_values):
    
    features = sorted(list(features_values.keys()))
    
    return html.Div([
                html.Br(),
                html.H3('Visa varurelationer',
                        style=h3_style),
                html.Br(),
                html.P("Använd denna graf för att se relationerna och korrelationerna mellan varor. Om korrelationen mellan två råvaror är stark kan prognosen förbättras genom att ta bort den andra från prognosfunktionerna.",
                       style = p_style),
                html.P("(Fortsätter efter diagrammet)",style={
                            'font-style':'italic',
                            'font-size':p_font_size,
                           'text-align':'center'}
                    ),
        
        dbc.Row(justify = 'center',children=[
            dbc.Col([
                html.H3('Välj en vara',style=h3_style),
                dcc.Dropdown(id = 'f_corr1_sv',
                                multi = False,
                                options = [{'value':feature, 'label':feature} for feature in features],
                                value = features[0],
                                style = {'font-size':"0.9rem", 
                                         #'font-family':'Cadiz Book'
                                         },
                                placeholder = 'Välj en vara')
        ],xs =12, sm=12, md=12, lg=6, xl=6),
        html.Br(),
            dbc.Col([
                html.H3('Välj en annan vara',
                        style=h3_style
                        ),
                dcc.Dropdown(id = 'f_corr2_sv',
                                multi = False,
                                options = [{'value':feature, 'label':feature} for feature in features],
                                value = features[-1],
                                style = {'font-size':"0.9rem", 
                                         #'font-family':'Cadiz Book'
                                         },
                                placeholder = 'Välj en annan vara')
            ],xs =12, sm=12, md=12, lg=6, xl=6)
        ])
        ])



@callback(

    Output('corr_div_sv', 'children'),
    [Input('f_corr1_sv','value'),
     Input('f_corr2_sv','value')]    
    
)
def sv_update_feature_correlation_plot(value1, value2):
    
    if value1 is None or value2 is None:
        raise PreventUpdate 
        
        
    a, b = np.polyfit(np.log(data_sv[value1]), data_sv[value2], 1)

    y = a * np.log(data_sv[value1]) +b 

    df = data_sv.copy()
    df['log_trend'] = y
    df = df.sort_values(by = 'log_trend')    
    
    corr_factor = round(sorted(data_sv[[value1,value2]].corr().values[0])[0],2)
    
    traces = [go.Scatter(x = data_sv[value1], 
                         y = data_sv[value2], 
                         mode = 'markers',
                         name = ' ',#.join(value.split()[1:]),
                         showlegend=False,
                         marker = dict(color = 'purple', size = 10),
                         marker_symbol='star',
                         hovertemplate = "<b>Indexvärden:</b><br><b>{}</b>:".format(' '.join(value1.split()[1:]))+" %{x}"+"<br><b>"+"{}".format(' '.join(value2.split()[1:]))+"</b>: %{y}"
                         ),
                go.Scatter(x = df[value1], 
                            y = df['log_trend'], 
                            name = 'Logaritmisk trendlinje', 
                            mode = 'lines',
                            line = dict(width=5),                            
                            showlegend=True,
                            hovertemplate=[], 
                            marker = dict(color = 'orange'))
             ]
    
  
    
    return [
            html.Div([dcc.Graph(figure = go.Figure(data = traces,
          layout = go.Layout(title = dict(text = '<b>{}</b> vs.<br><b>{}</b><br>(Korrelation: {})'.format(' '.join(value1.split()[1:]), ' '.join(value2.split()[1:]), corr_factor), 
                                          x=.5, 
                                          font=dict(
                                               family='Cadiz Semibold',
                                               size=17
                                              )),
                             margin=dict(
                                  l=10,
                                 r=10,
                                 # b=100,
                                  # t=120,
                                  # pad=4
                             ),
                            xaxis= dict(title = dict(text='{} (Indextal)'.format(' '.join(value1.split()[1:])), 
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
                                 family='Cadiz Book'
                                ),
                                          orientation='h'),
                            hoverlabel = dict(
                                 font_size = 18, 
                                 font_family = 'Cadiz Book'
                                ),
                            template = 'seaborn',
                            yaxis = dict(title = dict(text='{} (Indextal)'.format(' '.join(value2.split()[1:])), 
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
                      config = config_plots_sv)])]



@callback(
    
    Output('commodity_unemployment_div_sv','children'),
    [Input('corr_feature_sv','value'),
     Input('eda_y_axis_sv','value')]
    
)
def sv_update_commodity_unemployment_graph(values, label):
    
    
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
                 # 'hash'
                 ]
    
    label_str = {'Työttömyysaste': 'Arbetslöshet (%)',
                 'change': 'Förändring av månadsarbetslösheten (% enheter)'}[label]     
            
    traces = [go.Scatter(x = data_sv[value], 
                         y = data_sv[label], 
                         mode = 'markers',
                         name = ' '.join(value.split()[1:]).replace(',',',<br>')+' ({})'.format(round(sorted(data_sv[[label, value]].corr()[value].values)[0],2)),
                         showlegend=True,
                         marker = dict(size=10),
                         marker_symbol = random.choice(symbols),
                         hovertemplate = "<b>{}</b>:".format(value)+" %{x}"+"<br><b>"+label_str+"</b>: %{y}"+"<br>(Korrelation: {:.2f})".format(sorted(data_sv[[label, value]].corr()[value].values)[0])) for value in values]
    text='Utvalda varor vs.<br>'+label_str
    if len(values)==1:
        data_ = data_sv[(data_sv[label].notna())].copy()
        
        for value in values:
        
            a, b = np.polyfit(np.log(data_[value]), data_[label], 1)
    
            y = a * np.log(data_[value].values) +b 
            
    
            df = data_[[value]].copy()
            df['log_inflation'] = y
            df = df.sort_values(by = 'log_inflation')
            traces.append(go.Scatter(x=df[value], 
                                      y =df['log_inflation'],
                                      showlegend=True,
                                      name = 'Logaritmisk<br>trendlinje',
                                      line = dict(width=5),
                                      hovertemplate=[]))
        text = f"{' '.join(values[0].split()[1:]).capitalize()} vs.<br>"+label_str
    
    return [dcc.Graph(figure = go.Figure(data = traces,
          layout = go.Layout(title = dict(text = text, 
                                          x=.5, 
                                          font=dict(
                                              family='Cadiz Semibold',
                                               size=20
                                              )
                                          ),
                            xaxis= dict(title = dict(text='Indextal', 
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
                                         # xanchor='center',
                                         # yanchor='top',
                                        font=dict(
                                 size=16,
                                 family='Cadiz Book'
                                )),
                            hoverlabel = dict(
                                 font_size = 18, 
                                 font_family = 'Cadiz Book'
                                ),
                            template = 'seaborn',                            
                            yaxis = dict(title = dict(text=label_str, 
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
          ),config = config_plots_sv)]

@callback(
    [
     
     Output('feature_selection_sv', 'options'),
     Output('sorting_sv', 'label')
     
     ],
    [Input('alphabet_sv', 'n_clicks'),
     Input('corr_desc_sv', 'n_clicks'),
     Input('corr_asc_sv', 'n_clicks'),
     Input('corr_abs_desc_sv', 'n_clicks'),
     Input('corr_abs_asc_sv', 'n_clicks'),
     Input('main_class_sv', 'n_clicks'),
     Input('second_class_sv', 'n_clicks'),
     Input('third_class_sv', 'n_clicks'),
     Input('fourth_class_sv', 'n_clicks'),
     Input('fifth_class_sv', 'n_clicks'),
     Input('example_basket_1_sv', 'n_clicks'),
     Input('example_basket_2_sv', 'n_clicks'),
     Input('example_basket_3_sv', 'n_clicks')
    ]
)
def sv_update_selections(*args):
    
    ctx = callback_context
    
    
    if not ctx.triggered:
        return feature_options_sv, "Alfabetisk ordning"#,[f['value'] for f in corr_abs_asc_options[:4]]
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if button_id == 'alphabet_sv':
        return feature_options_sv, "Alfabetisk ordning",#[f['value'] for f in feature_options[:4]]
    elif button_id == 'corr_desc_sv':
        return corr_desc_options_sv, "Korrelation (fallande)",#[f['value'] for f in corr_desc_options[:4]]
    elif button_id == 'corr_asc_sv':
        return corr_asc_options_sv, "Korrelation (stigande)",#[f['value'] for f in corr_asc_options[:4]]
    elif button_id == 'corr_abs_desc_sv':
        return corr_abs_desc_options_sv, "Absolut korrelation (fallande)"#,[f['value'] for f in corr_abs_desc_options[:4]]
    elif button_id == 'corr_abs_asc_sv':
        return corr_abs_asc_options_sv, "Absolut korrelation (stigande)",#[f['value'] for f in corr_abs_asc_options[:4]]
    elif button_id == 'main_class_sv':
        return main_class_options_sv, "Huvudklasser",#[f['value'] for f in main_class_options[:4]]
    elif button_id == 'second_class_sv':
        return second_class_options_sv, "2. klass",#[f['value'] for f in second_class_options[:4]]
    elif button_id == 'third_class_sv':
        return third_class_options_sv, "3. klass",#[f['value'] for f in third_class_options[:4]]
    elif button_id == 'fourth_class_sv':
        return fourth_class_options_sv, "4. klass"#,[f['value'] for f in fourth_class_options[:4]]
    elif button_id == 'fifth_class_sv':
        return fifth_class_options_sv, "5. klass"#,[f['value'] for f in fifth_class_options[:4]]
    elif button_id == 'example_basket_1_sv':
        return example_basket_1_options, "Exempelkorg 1"#,[f['value'] for f in fifth_class_options[:4]]
    elif button_id == 'example_basket_2_sv':
        return example_basket_2_options, "Exempelkorg 2"#,[f['value'] for f in fifth_class_options[:4]]
    else:
        return example_basket_3_options, "Exempelkorg 3"#,[f['value'] for f in fifth_class_options[:4]]
    
