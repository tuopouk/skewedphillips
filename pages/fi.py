# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 11:57:56 2022

@author: tuomas.poukkula
"""

import dash 

dash.register_page(__name__, 
                   path='/',
                   title = 'Phillipsin vinouma',
                   name = 'Phillipsin vinouma',
                   description = "Työttömyyden ennustaminen kuluttajahintojen muutoksilla",
                   image='fi.png',
                   redirect_from =['/assets','/assets/'])

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
    locale.setlocale(locale.LC_ALL, 'fi_FI')
except:
    locale.setlocale(locale.LC_ALL, 'fi-FI')



MODELS = {
    
    
    
        'Päätöspuu':{'model':DecisionTreeRegressor,
                           'doc': 'https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html',
                           'video':"https://www.youtube.com/embed/UhY5vPfQIrA",
                           'explainer':shap.TreeExplainer,
                           'constant_hyperparameters': {
                                                        # 'n_jobs':-1,
                                                        'random_state':42}
                           },
    
        'Satunnaismetsä': {'model':RandomForestRegressor,
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
        # 'K lähimmät naapurit':{'model':KNeighborsRegressor,
        #                         'doc':'https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html',
        #                         'video':'https://www.youtube.com/embed/jw5LhTWUoG4?list=PLRZZr7RFUUmXfON6dvwtkaaqf9oV_C1LF',
        #                         'explainer':shap.KernelExplainer,
        #                         'constant_hyperparameters': {
        #                                                     'n_jobs':-1
        #                                                     }
        #                         },
        # 'Tukivektorikone':{'model':SVR,
        #                     'doc':'https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html',
        #                     'video':"https://www.youtube.com/embed/_YPScrckx28",
        #                     'explainer':shap.KernelExplainer,
        #                         'constant_hyperparameters': {
        #                                                     }
        #                         },
        'Gradient Boost':{'model':GradientBoostingRegressor,
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

def set_color(x,y):
    
    
    if 'yöttömyys' in x or x=='Kuluva kuukausi':
        return 'black'
    elif y < 0:
        
        return '#117733'
    elif y >= 0:
        return '#882255'




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

  url = 'https://pxdata.stat.fi:443/PxWeb/api/v1/fi/StatFin/khi/statfin_khi_pxt_11xd.px'
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

  data = pd.merge(left = unemployment_df.drop('Tiedot',axis=1).reset_index(), right = inflation_df.reset_index(), how = 'outer', on = 'Aika').set_index('Aika')
  data.Työttömyysaste = data.Työttömyysaste.fillna(-1)
  data = data.dropna(axis=1)

  inflation_percentage_df = get_inflation_percentage()

  data = pd.merge(left = data.reset_index(), right = inflation_percentage_df.reset_index(), how = 'inner', on = 'Aika').set_index('Aika').sort_index()

  data.Työttömyysaste = data.Työttömyysaste.replace(-1, np.nan)

  data['prev'] = data['Työttömyysaste'].shift(1)

  data['month'] = data.index.month
  data['change'] = data.Työttömyysaste - data.prev
  
  # Näin poistetaan duplikaattisarakkeet arvojen perusteella.
  # Jos näin tehdään niin saatetaan kuitenkin poistaa sarake, jolla on
  # täsmälleen samat arvot kuin jollain toisella hyödykkeellä.
  # Esim. 04.2.1 Uuden asunnon hankinta ja 04.2.1.1 Osakehuoneistot ja kiinteistöt.
  
  # data = data.T.drop_duplicates().T

  return data


data = get_data()
  



def draw_phillips_curve():
    
  try:
      locale.setlocale(locale.LC_ALL, 'fi_FI')
  except:
      locale.setlocale(locale.LC_ALL, 'fi-FI')
      
  data_ = data[(data.Työttömyysaste.notna())&(data.Inflaatio.notna())].copy()
    
  max_date = data_.index.values[-1]
  max_date_str = data_.index.strftime('%B %Y').values[-1]

  a, b = np.polyfit(np.log(data_.Työttömyysaste), data_.Inflaatio, 1)

  y = a * np.log(data_.Työttömyysaste) +b 

  df = data_.copy()
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
                            textfont=dict(
                                 family='Cadiz Book',
                                          size = 18
                                          ),
                            showlegend=False,
                            hovertemplate=hovertemplate,
                            marker = dict( symbol='diamond', 
                                          color = '#E66100',
                                           size = 10
                                          )
                            ),
                go.Scatter(x = df.Työttömyysaste, 
                            y = df['log_inflation'], 
                            name = 'Logaritminen<br>trendiviiva', 
                            mode = 'lines',
                            line = dict(width=5),
                            showlegend=True,
                            hovertemplate=[], 
                            marker = dict(color = '#5D3A9B')
                            )
                  ],
            layout = go.Layout(
                               xaxis=dict(showspikes=True,
                                          title = dict(text='Työttömyysaste (%)', font=dict(size=16, 
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
                                          title = dict(text='Inflaatio (%)', font=dict(size=16, 
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
                              
                               title = dict(text = 'Työttömyysaste vs.<br>Inflaatio<br>{} - {}<br>'.format(df.index.min().strftime('%B %Y'),df.index.max().strftime('%B %Y')),
                                            x=.5,
                                            font=dict(
                                                size=20,
                                                 family = 'Cadiz Semibold'
                                                ))
                              )
            )
            
def get_shap_values(model, explainer, X_train, X_test):


    if explainer.__name__ == 'Kernel':
        explainer = explainer(model.predict, X_train)
        shap_values = explainer.shap_values(X_test,normalize=False, n_jobs=-1)
        feature_names = X_test.columns
        shap_df = pd.DataFrame(shap_values, columns=feature_names)
        vals = np.abs(shap_df.values).mean(0)
        base_value = explainer.expected_value
        # print(shap_values.sum(1) + explainer.expected_value - model.predict(X_test))   
        # print(model.predict(X_test))
        shap_importance = pd.DataFrame(list(zip(feature_names, vals)), columns=['col_name', 'feature_importance_vals']).set_index('col_name')
        shap_importance = shap_importance.sort_values(by=['feature_importance_vals'], ascending=False) 
        shap_importance.columns = ['SHAP']
        shap_importance.index = [i for i in shap_importance.index]
        shap_importance.index = shap_importance.index.str.replace('prev','Edellisen kuukauden työttömyysaste')
        shap_importance.index = shap_importance.index.str.replace('month','Kuukausi')
        shap_df['base'] = base_value
        return shap_importance,shap_df
    
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
        return shap_importance,shap_df
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
        return shap_importance,shap_df
        

def get_param_options(model_name):
    
    
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




def plot_test_results(df, chart_type = 'lines+bars'):
    
    
    try:
        locale.setlocale(locale.LC_ALL, 'fi_FI')
    except:
        locale.setlocale(locale.LC_ALL, 'fi-FI')
    
    
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
                               textfont = dict(
                                    family='Cadiz Semibold', 
                                   size = 16#,color='#1A85FF'
                                   ), 
                               marker = dict(color='#1A85FF',size=12),
                               line = dict(width=5)),
                    
                    go.Bar(x=df.index.strftime('%B %Y'), 
                           y = df.Ennuste, 
                           name = 'Ennuste',
                           showlegend=True, 
                           marker = dict(color='#D41159'), 
                           text=[str(round(c,2))+' %' for c in df.Ennuste], 
                           textposition='inside',
                           hovertemplate = hovertemplate,
                           textfont = dict(
                                family='Cadiz Semibold', 
                               size = 16)
                           )
                    ],layout=go.Layout(xaxis = dict(title = dict(text='Aika',font=dict(size=20, 
                                                                                       family = 'Cadiz Semibold'
                                                                                       )),
                                                    tickfont = dict(
                                                        family = 'Cadiz Semibold', 
                                                        size = 16),
                                                    automargin=True
                                                    ),
                                       yaxis = dict(title = dict(text='Työttömyysaste (%)',
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
                                       title = dict(text = 'Työttömyysasteen ennuste<br>kuukausittain',
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
                                name = 'Toteutunut',
                                showlegend=True, 
                                mode = 'lines+markers+text',
                                text=[str(round(c,2))+' %' for c in df.Työttömyysaste], 
                                textposition='top center',
                                hovertemplate = hovertemplate,
                                textfont = dict(
                                     family='Cadiz Semibold', 
                                    size = 16#,color='#1A85FF'
                                    ), 
                                marker = dict(color='#1A85FF',size=10),
                                line = dict(width=2)),
                    
                    go.Scatter(x=df.index, 
                            y = df.Ennuste, 
                            name = 'Ennuste',
                            showlegend=True,
                            mode = 'lines+markers+text',
                            marker = dict(color='#D41159',size=10), 
                            text=[str(round(c,2))+' %' for c in df.Ennuste], 
                            # textposition='inside',
                            hovertemplate = hovertemplate,
                            line = dict(width=2),
                            textfont = dict(
                                 family='Cadiz Semibold', 
                                size = 16#,color='#D41159'
                                )
                            )
                    ],layout=go.Layout(xaxis = dict(title = dict(text='Aika',font=dict(size=20, 
                                                                                       family = 'Cadiz Semibold'
                                                                                       )),
                                                    tickfont = dict(
                                                        family = 'Cadiz Semibold', 
                                                        size = 16),
                                                    automargin=True,
                                                    ),
                                        yaxis = dict(title = dict(text='Työttömyysaste (%)',font=dict(
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
                                        title = dict(text = 'Työttömyysasteen ennuste<br>kuukausittain',
                                                    x=.5,
                                                    font=dict(
                                                         family='Cadiz Semibold',
                                                        size=20)
                                                    )
                                        ))
                                                    

    else:
        return go.Figure(data=[go.Bar(x=df.index.strftime('%B %Y'), 
                                    y = df.Työttömyysaste, 
                                    name = 'Toteutunut',
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
                                name = 'Ennuste',
                           showlegend=True, 
                           marker = dict(color='#D41159'), 
                           text=[str(round(c,2))+' %' for c in df.Ennuste], 
                           textposition='inside',
                           hovertemplate = hovertemplate,
                           textfont = dict(
                                family='Cadiz Semibold', 
                               size = 16)
                                )
                        ],layout=go.Layout(xaxis = dict(title = dict(text='Aika',font=dict(size=20, 
                                                                                           family = 'Cadiz Semibold'
                                                                                           )),
                                                        tickfont = dict(
                                                            family = 'Cadiz Semibold', 
                                                            size = 16),
                                                        automargin=True
                                                        ),
                                            yaxis = dict(title = dict(text='Työttömyysaste (%)',font=dict(
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
                                            title = dict(text = 'Työttömyysasteen ennuste<br>kuukausittain',
                                                        x=.5,
                                                        font=dict(
                                                             family='Cadiz Semibold',
                                                            size=20)
                                                        )
                                            )
                                                        )                                                   

                                                    
                                                    
                                                    
def plot_forecast_data(df, chart_type):
    
    try:
        locale.setlocale(locale.LC_ALL, 'fi_FI')
    except:
        locale.setlocale(locale.LC_ALL, 'fi-FI')
    
    
    hover_true = ['<b>{}</b><br>Työttömyysaste: {} %'.format(data.index[i].strftime('%B %Y'), data.Työttömyysaste.values[i]) for i in range(len(data))]
    hover_pred = ['<b>{}</b><br>Työttömyysaste: {} %'.format(df.index[i].strftime('%B %Y'), round(df.Työttömyysaste.values[i],1)) for i in range(len(df))]
    

    if chart_type == 'lines':
    
    
        return go.Figure(data=[go.Scatter(x=data.index, 
                                          y = data.Työttömyysaste, 
                                          name = 'Toteutunut',
                                          showlegend=True,
                                          mode="lines", 
                                          hovertemplate =  hover_true,##'<b>%{x}</b>: %{y}%',
                                          marker = dict(color='#1A85FF')),
                    go.Scatter(x=df.index, 
                               y = np.round(df.Työttömyysaste,1), 
                               name = 'Ennuste',
                               showlegend=True,
                               mode="lines", 
                               hovertemplate = hover_pred,#'<b>%{x}</b>: %{y}%',
                               marker = dict(color='#D41159'))
                    ],layout=go.Layout(xaxis = dict(title = dict(text = 'Aika',font=dict(
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
                                            font_size = 18, 
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
                                       yaxis = dict(title=dict(text = 'Työttömyysaste (%)',
                                                     font=dict(
                                                          size=18, 
                                                         family = 'Cadiz Semibold'
                                                         )),
                                                    automargin=True,
                                                     tickfont = dict(
                                                         family = 'Cadiz Book', 
                                                                      size = 16
                                                                     )),
                                       title = dict(text = 'Työttömyysaste ja ennuste kuukausittain<br>{} - {}'.format(data.index.strftime('%B %Y').values[0],df.index.strftime('%B %Y').values[-1]),
                                                    x=.5,
                                                    font=dict(
                                                         family='Cadiz Semibold',
                                                               size=16
                                                              )),
    
                                       ))
                        
                        
    elif chart_type == 'area':
    
    
        return go.Figure(data=[go.Scatter(x=data.index, 
                                          y = data.Työttömyysaste, 
                                          name = 'Toteutunut',
                                          showlegend=True,
                                          mode="lines", 
                                          fill='tozeroy',
                                          hovertemplate =  hover_true,##'<b>%{x}</b>: %{y}%',
                                          marker = dict(color='#1A85FF')),
                    go.Scatter(x=df.index, 
                               y = np.round(df.Työttömyysaste,1), 
                               name = 'Ennuste',
                               showlegend=True,
                               mode="lines", 
                               fill='tozeroy',
                               hovertemplate = hover_pred,#'<b>%{x}</b>: %{y}%',
                               marker = dict(color='#D41159'))
                    ],layout=go.Layout(xaxis = dict(title = dict(text = 'Aika',font=dict(
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
                                            font_size = 18, 
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
                                       yaxis = dict(title=dict(text = 'Työttömyysaste (%)',
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
                                       title = dict(text = 'Työttömyysaste ja ennuste kuukausittain<br>{} - {}'.format(data.index.strftime('%B %Y').values[0],df.index.strftime('%B %Y').values[-1]),
                                                    x=.5,
                                                    font=dict(
                                                         family='Cadiz Semibold',
                                                               size=16
                                                              )),
    
                                       ))


    else:
        
        
      
        return go.Figure(data=[go.Bar(x=data.index, 
                                          y = data.Työttömyysaste, 
                                          name = 'Toteutunut',
                                          showlegend=True,
                                          # mode="lines", 
                                          hovertemplate = hover_true,#'<b>%{x}</b>: %{y}%',
                                          marker = dict(color='#1A85FF')),
                    go.Bar(x=df.index, 
                               y = np.round(df.Työttömyysaste,1), 
                               name = 'Ennuste',
                               showlegend=True,
                               # mode="lines", 
                               hovertemplate = hover_pred,#'<b>%{x}</b>: %{y}%',
                               marker = dict(color='#D41159'))
                    ],layout=go.Layout(xaxis = dict(title = dict(text = 'Aika',
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
                                       yaxis = dict(title=dict(text = 'Työttömyysaste (%)',
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
                                       title = dict(text = 'Työttömyysaste ja ennuste kuukausittain<br>{} - {}'.format(data.index.strftime('%B %Y').values[0],df.index.strftime('%B %Y').values[-1]),
                                                    x=.5,
                                                    font=dict(
                                                         family='Cadiz Semibold',
                                                               size=16
                                                              )),
    
                                       )) 
                                                    
                                                    


def test(model, features, test_size, explainer, use_pca = False, n_components=.99):

  feat = features.copy()
  feat.append('prev')
  feat.append('month')
  
  cols = feat
  
  data_ = data.iloc[1:,:].copy()
  
  
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
    cols = ['_ '+str(i+1)+'. pääkomponentti' for i in range(n_feat)]

  
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
  


  shap_df, local_shap_df = get_shap_values(model, explainer, X_train = pd.DataFrame(X, columns = cols), X_test = pd.concat(scaled_features))

  result = pd.concat(results)
  result['n_feat'] = n_feat
  result.Ennuste = np.round(result.Ennuste,1)
  result['mape'] = mean_absolute_percentage_error(result.Työttömyysaste, result.Ennuste)
  
  local_shap_df.index = result.index
  
  
  result.index.name ='Aika'
    
  result = result[['Työttömyysaste', 'Ennuste', 'change', 'Ennustettu muutos', 'prev','n_feat','mape','month']+features]
  

  return result, shap_df, local_shap_df         
                                          

                                                

def predict(model, explainer, features, feature_changes, length, use_pca = False, n_components=.99):
      
  
  
  df = data.copy()
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
    cols = ['_ '+str(i+1)+'. pääkomponentti' for i in range(n_feat)]
    
    
  model.fit(X,y)

  
  
  if data.Työttömyysaste.isna().sum() > 0:
      last_row = data.iloc[-1:,:].copy()
      
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
  
  shap_df, local_shap_df = get_shap_values(model, explainer, X_train = pd.DataFrame(X, columns = cols), X_test = pd.concat(scaled_features_shap))

  local_shap_df.index = result.index

  return result, shap_df, local_shap_df


def apply_average(features, length = 4):
 
    
  return 100 * data[features].pct_change().iloc[-length:, :].mean()





# Viimeiset neljä saraketta ovat prev, month, change ja inflaatio.

 
correlations_desc = data[data.columns[:-4]].corr()['Työttömyysaste'].iloc[1:].sort_values(ascending=False)
correlations_asc = data[data.columns[:-4]].corr()['Työttömyysaste'].iloc[1:].sort_values(ascending=True)
correlations_abs_desc = data[data.columns[:-4]].corr()['Työttömyysaste'].iloc[1:].abs().sort_values(ascending=False)
correlations_abs_asc = data[data.columns[:-4]].corr()['Työttömyysaste'].iloc[1:].abs().sort_values(ascending=True)

main_classes = sorted([c for c in data.columns[:-4] if len(c.split()[0])==2])
second_classes = sorted([c for c in data.columns[:-4] if c.split()[0].count('.')==1])
third_classes = sorted([c for c in data.columns[:-4] if c.split()[0].count('.')==2])
fourth_classes = sorted([c for c in data.columns[:-4] if c.split()[0].count('.')==3])
fifth_classes = sorted([c for c in data.columns[:-4] if c.split()[0].count('.')==4])

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



initial_options = feature_options


example_basket_1 = [c for c in ['01.1.3 Kala ja äyriäiset',
 '01.1.3.3 Kuivattu, savustettu tai suolattu kala ja äyriäiset',
 '01.1.6.3.1 Pakastetut hedelmät ja marjat',
 '01.1.9.3.1 Vauvanruoat',
 '01.1.9.5.1 Muut ruokavalmisteet, muualla luokittelemattomat',
 '02.1.2.1.1 Rypäleviinit',
 '03.1 Vaatetus',
 '03.1.2.2 Naisten vaatteet',
 '03.1.2.2.1 Naisten päällystakit',
 '03.1.2.4 Vauvojen vaatteet (0-2 vuotiaat)',
 '03.2.1.3 Lasten jalkineet',
 '05.2.0.2.3 Lakanat, tyynyliinat ja pussilakanat',
 '05.6.1.1.1 Pesuaineet',
 '06.1.3 Hoitolaitteet ja -välineet',
 '06.1.3.1 Silmälasit ja piilolinssit',
 '07.3.1.1.1 Kotimaan junaliikenne',
 '09 KULTTUURI JA VAPAA-AIKA',
 '09.4.2.3.3 Kaapeli ja maksu-TV tilausmaksut',
 '09.6.0 Valmismatkat',
 '12.1.3.3.3 Vartalo-, käsi- ja hiusvoiteet'] if c in data.columns]

example_basket_2 = [c for c in data.columns if c.split()[0] in ['01.1.3',
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



example_basket_3 = [c for c in data.columns if c.split()[0] in ['01.1.1.2.1',
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

initial_features = example_basket_1

example_basket_1_options = [{'label':c, 'value':c} for c in example_basket_1]
example_basket_2_options = [{'label':c, 'value':c} for c in example_basket_2]
example_basket_3_options = [{'label':c, 'value':c} for c in example_basket_3]


def layout():
    
    return html.Div([dbc.Container(fluid=True, className = 'dbc', children=[
        

        
        html.Br(),        
        dbc.Row(
            [
            
                
                dbc.Col([
                    
                    html.Br(),  
                    html.H1('Phillipsin vinouma',
                             style=h1_style
                            ),
                  
                    html.H2('Työttömyyden ennustaminen kuluttajahintojen muutoksilla',
                            style=h2_style),
                    
                    html.P('Valitse haluamasi välilehti alla olevia otsikoita klikkaamalla. ' 
                           'Vasemman yläkulman painikkeista saat näkyviin pikaohjeen '
                           'ja voit myös vaihtaa sivun väriteemaa.',
                           style = p_center_style)
                    ],xs =12, sm=12, md=12, lg=9, xl=9)
        ], justify='center'),
        html.Br(),
        
        html.Div(id = 'hidden_store_div',
                 children = [
                    
                    dcc.Store(id = 'features_values',data={f:0.0 for f in initial_features}),
                    dcc.Store(id = 'change_weights'), 
                    dcc.Store(id = 'method_selection_results'),
                    dcc.Store(id ='shap_data'),
                    dcc.Store(id ='local_shap_data'),
                    dcc.Store(id = 'test_data'),
                    dcc.Store(id = 'forecast_data'),
                    dcc.Store(id = 'forecast_shap_data'),
                    dcc.Store(id = 'local_forecast_shap_data'),
                    dcc.Download(id='forecast_download'),
                    dcc.Download(id='test_download')
        ]),
        
        dbc.Tabs(id ='tabs',
                 children = [
            
            
            
            dbc.Tab(label='Ohje ja esittely',
                    tab_id = 'ohje',
                    tabClassName="flex-grow-1 text-center",
                    tab_style = {'font-size':"1.5625rem",#'25px',
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
                                  html.H3('Johdanto',style=h3_style
                                          ),
                                  html.P('Hintojen nousu vaikuttaa monen suomalaisen kuluttujan elämään. Odotettavissa on korkojen nousuja keskuspankkien kokiessa painetta hillitä inflaatiota. Asuntovelallisille on luvassa vaikeampia aikoja korkojen noustessa. Kaiken kaikkiaan vaikuttaisi siltä, että inflaatiosta ei seuraa mitään hyvää. Mutta onko todella näin?',
                                        style = p_style),
                                  html.P('Inflaatiosta löytyy myös hopeareunus, joka on työttömyyden lasku lyhyellä aikavälillä. Tämä ns. Phillipsin käyrä on samannimisen taloustieteilijän Alban William Phillipsin 1950 -luvulla tekemä empiirinen havainto, jossa inflaation ja työttömyyden välillä vallitsee ristiriita lyhyellä ajalla. Tämä kyseinen idea on esitetty alla olevassa kuvaajassa, jossa on kuvattu inflaatio ja saman ajankohdan työttömyysaste Suomessa. Laskeva logaritminen trendiviiva vastaa Phillipsin havaintoa.',
                                        style = p_style),
                                  html.P("(Jatkuu kuvaajan jälkeen)",style={
                                              'font-style':'italic',
                                              'font-size':p_font_size,
                                             'text-align':'center'}
                                      ),
                                
                                  html.H3('Phillipsin käyrä Suomen taloudessa kuukausittain', 
                                          style=h3_style),
                                  html.H4('Lähde: Tilastokeskus', 
                                          style=h4_style),
                                  
                                  ])
                                  ]
                                ),

                                  dbc.Row([
                                      dbc.Col([
                                             
                                              html.Div(
                                                  [dcc.Graph(id = 'phillips',
                                                             figure= draw_phillips_curve(),
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
                                                    #'font-family':'Messina Modern Book', 
                                                      'font-size':"1.1rem"#p_font_size-2
                                                    }),
                                          html.P('Selittäviä teorioita Phillipsin käyrälle on useita riippuen siitä onko ilmiön katalyyttinä muutos hintatasossa vai työttömyydessä. Työttömyyden ollessa korkea, kysynnän ja tarjonnan laki vaatii hintojen alentamista hyödykkeiden menekin parantamiseksi. Toisaalta lyhyellä aikavälillä hintojen noustessa tuotanto nousee, koska tuottajat nostavat hyödykkeiden tuotantoa suurempien katteiden saavuttamiseksi. Tämä johtaa matalampaan työttömyyteen, koska tuotannon kasvattaminen johtaa uusiin rekrytointeihin, jotka voidaan tehdä työttömälle väestölle. Toisin päin katsottuna, kun työttömyys on matala, markkinoilla on työn kysyntäpainetta, mikä nostaa palkkoja. Palkkojen nousu taas johtaa yleisen hintatason nousuun, koska hyödykkeiden tarjoajat voivat pyytää korkeampaa hintaa tuotteistaan ja palveluistaan.',
                                                 style = p_style),
                                          html.P('Phillipsin käyrää voi havainnoida myös intuitiivisesti. Esimerkiksi vuonna 2015 työttömyysaste oli useaan otteeseen päälle kymmenen prosentin inflaation pysytellessä nollan tasolla ja ollen välillä jopa negatiivinen. Koronashokki vuonna 2020 aiheutti pompun työttömyysasteessa mutta esimerkiksi polttoaineen hinnat olivat paljon vuoden 2022 alkupuoliskoa matalammalla. Historiasta löytyy useita hetkiä, jolloin työttömyys ja inflaatio ovat muutuneet eri suuntaan. Poikkeuksiakin on ollut, esim. 1970-luvun öljykriisin aikaan molemmat olivat korkealla, mutta historiaa tarkasteltuna löytyy useita ajanjaksoja, joina Phillipsin käyrä on pätenyt. Onkin hyvä muistaa, että Phillipsin käyrä on voimassa vain lyhyellä aikavälillä, jolloin siihen perustuvia ennusteitakaan ei tulisi tehdä liian pitkälle ajalle.', style = p_style),
                                          # html.P('Kyseessä on tunnettu taloustieteen teoria, jota on tosin vaikea soveltaa, koska ei ole olemassa sääntöjä, joilla voitaisiin helposti ennustaa työttömyyttä saatavilla olevien inflaatiota kuvaavien indikaattorien avulla. Mikäli sääntöjä on vaikea formuloida, niin niitä voi yrittää koneoppimisen avulla oppia historiadataa havainnoimalla. Voisiko siis olla olemassa tilastollisen oppimisen menetelmä, joka pystyisi oppimaan Phillipsin käyrän historiadatasta? Mikäli tämänlainen menetelmä olisi olemassa, pystyisimme ennustamaan lyhyen aikavälin työttömyyttä, kun oletamme hyödykkeiden hintatason muuttuvan skenaariomme mukaisesti.',
                                                # style=p_style), 
                                          html.P('Phillipsin käyrälle on omistettu oma lukunsa kauppatieteellisen yliopistotutkinnon kansantaloustieteen pääsykoekirjassa, ja siitä seuraa yksi kymmenestä taloustieteen perusperiaatteesta.',
                                                style=p_style),
                                          html.Br(),
                                                                            
                                          html.H3('Taloustieteen kymmenes perusperiaate:',style = {'text-align':'center', 
                                                                                                   #'font-family':'Messina Modern Semibold',
                                                                                                   'font-style': 'italic', 
                                                                                                   'font-weight': 'bold', 
                                                                                                   'font-size':"2.125rem",#'34px'
                                                                                                   }),
                                          
                                          html.Blockquote('Työttömyyden ja inflaation kesken vallitsee lyhyellä ajalla ristiriita. Täystyöllisyyttä ja vakaata hintatasoa on vaikea saavuttaa yhtä aikaa.', 
                                                style = {
                                                    'text-align':'center',
                                                    'font-style': 'italic', 
                                                    #'font-family':'Messina Modern Book', 
                                                      'font-size':"1.2rem",#p_font_size
                                                    }),
                                          html.P('Matti Pohjola, 2019, Taloustieteen oppikirja, s. 250, ISBN:978-952-63-5298-5', 
                                                style={
                                                    'textAlign':'center',
                                                    #'font-family':'Messina Modern Book', 
                                                      'font-size':"1rem",#p_font_size-4
                                                    }),
        
                                          html.Br(),
                                          html.P('Phillipsin käyrä tosin on vain teoria, joka on helppo havainnoida historiadataa analysoimalla. Vaikka Phillipsin käyrä ei aina toteutuisikaan, niin olisiko kuitenkin mahdollista hyödyntää inflaation ja työttömyyden välistä suhdetta työttömyyden ennustamisessa?',
                                                 style=p_style),
                                          html.P('Phillipsin käyrää on vaikea muuttaa matemaattiseksi yhtälöksi, johon sijoittamalla inflaation saadaan laskettua työttömyysaste. Siitä sain ajatuksen, että voisiko olla olemassa koneoppimisen menetelmä, joka voisi oppia vallitsevat lainalaisuudet inflaation ja työttömyyden välillä.  Inflaatiohan on kuluttajahintaindeksin vuosimuutos. Kuluttajahintaindeksi muodostuu useista hyödykkeistä, jotka ilmaisevat yhteiskunnan sen aikaisia kulutustarpeita. Voisiko osa näistä hyödykkeistä vaikuttaa toisia enemmän? Riittäisikö ennustepiirteiksi vain perus kuluttajahintaindeksi, edellisen kuukauden työttömyysaste ja jokin tieto työttömyyden kausivaihtelusta? Mitä hyödykkeitä pitäisi valita? Mikä algoritmi, millä hyperparametreilla? Leikittelin ajatuksella, että voisi olla olemassa jokin hyödykeyhdistelmän ja metodologian kombinaatio, jolla saadaan tehtyä vähintään tyydyttävä lyhyen aikavälin ennuste. Halusinkin luoda sovelluksen, jolla kuka tahansa, akateemisesta taustasta riippumatta voisi tehdä tällaisia kokeiluja.',
                                                 style=p_style),
                                          html.P('Tuloksena syntyi usean iteraation jälkeen sovellus, jossa voi suunnitella hyödykekorin, valita koneoppimismenetelmän, testata näiden kombinaation kykyä ennustaa jo toteutuneita arvoja sekä lopulta tehdä ennusteita. Siihen päälle rakensin mahdollisuuden säätää koneoppimisalgoritmien hyperparametrit sekä hyödyntää pääkomponenttianalyysiä irrelevanttien piirteiden eliminoimiseksi.', 
                                                 style=p_style),
                                          html.P('Seuraavaksi ongelmaksi paljastui mallien vaikea tulkittavuus. Koneoppimisessa on yleisesti tunnettu tarkkuuden ja tulkittavuuden ristiriita. Yksinkertaisempia malleja, kuten lineaariregressio, on helpompi tulkita kuin esimerkiksi satunnaismetsää, mutta satunnaismetsä voi tuottaa paremman ennusteen. Tästä syntyy mustan laatikon ongelma, jota on syytä ratkaista, jotta menetelmästä saadaan uskottavampi ja läpinäkyvämpi ja jotta sitä voitaisiin näin hyödyntää yleisesti suunnittelussa ja päätöksenteossa. ',
                                                 style=p_style),
                                          html.P('Lisäsinkin sovellukseen agnostiseksi toiminnallisuudeksi Shapley arvojen tarkastelun. Shapley- arvot ovat peliteoriaan perustuva käsite, joka perustuu pelaajien kontribuutioiden laskemiseen yhteistyöpeleissä (esim. jalkapallopelin yksittäisten pelaajien kontribuutio lopputulokseen). Koneoppimisessa vastaavaa mallia hyödynnetään ennustepiirteiden ennustekontribuution arvioimiseen. Itse työttömyyden ennustamista mielenkiintoisemmaksi tutkimusongelmaksi muodostuikin, että mitkä hyödykkeet tai hyödykeyhdistelmät onnistuvat parhaiten ennustamaan työttömyyttä!',
                                                 style =p_style),
                                          html.P('Tarkoituksena oli etsiä Phillipsin havaintoa koneoppimisen avulla ja kenties löytää Phillipsin käyrän kaava. Koneoppimisen hyöty tulee siitä, että se tuottaa oman näkemyksensä ilmiöstä, sitä kuvaavaa dataa havainnoimalla. Kuten AI-pioneeri Rodney Brooks on sanonut, "maailma on itsensä paras malli, se on aina päivitetty, ja sisältää kaikki tarvittavat yksityiskohdat. Sitä pitää vain havainnoida oikein ja tarpeeksi usein."'  ,
                                                 style =p_style),
                                          html.P('Tulos voi siten olla jotain muuta kuin Phillipsin käyrän lainalaisuus, sillä koneoppivat algoritmit voivat oppia jonkin aivan toisen piilevän lainalaisuuden. Tässä vain hyödynnetään inflaation komponettien ja työttömyysasteen muutoksen välistä yhteyttä. Opittu kaava ei olekaan Phillipsin käyrä, vaan jokin toinen sääntö, vinouma Phillipsin havainnossa, mahdollisesti käänteinen Phillips tai käyrä, joka sisältää laaksoja ja kukkuloita. Lisäksi ennustetta ei tehdä ainoastaan hintaindekseillä vaan myös edellisen kuukauden työttömyysasteella sekä kuukausien numeerisilla arvoilla (esim. kesäkuu on 6). Nämä tekijät saattavat merkitä ennusteen kannalta paljon hintaindeksejä enemmän. Toisin sanoen, Phillipsin käyrä onkin tässä tapauksessa vain teoreettinen lähtökohta, kipinä tutkimukselle ja tausta sille, että käy jotenkin järkeen selittää työttömyyttä inflaation komponenteilla.',style=p_style),
                                          html.P('Koodasin siten tämän datatieteen blogin ja sovelluksen yhdistelmän (en tiedä miksi sellaista kutsutaan, "bläppi", "bloglikaatio",...), joka hyödyntää Tilastokeskuksen Statfin-rajapinnan tarjoamaa dataa Suomen kuluttajahintaindeksistä hyödykkeittäin perusvuoteen 2010 suhteutettuna, sekä Suomen työttömyysastetta kuukausittain. Dataseteistä on poistettu ne hyödykeryhmät, jolta ei löydy dataa koko tarkasteluajalta. Jäljelle jää silti satoja hyödykkeitä ja hyödykeryhmiä, joista voi rakentaan ennusteen komponentteja. Algoritmivaihtoehdoiksi on valittu epälineaarisia koneoppimisalgoritmeja, koska ne soveltuvat tähän tapaukseen lineaarisia malleja paremmin.',
                                                 style =p_style),
                                          html.P('Sovellus on jaettu osiin, jotka on ilmaistu välilehdillä. Tarkoitus on edetä välilehdittäin vasemmalta oikealle ja iteroida aina hyödykkeitä ja menetelmää muuttamalla. Sovellus perustuu siihen hypoteesiin, että jokaisen kuukauden työttömyysasteen kuukausimuutos voidaan selittää edellisen kuukauden työttömyysasteella, sen hetkisellä kuukaudella sekä valitun hyödykekorin vallitsevilla indeksiarvoilla. Kun työttömyyden kuukausimuutos saadaan koneoppivan algoritmin tuloksena, voidaan se laskea yhteen edellisen kuukauden työttömyysasteen kanssa, jolloin saadaan tulokseksi ennuste vallitsevan kuukauden työttömyysasteelle. Tätä ajatusta sovelletaan rekursiivisesti niin pitkälle kuin halutaan ennustaa. Tämä ratkaisu vaatii sen, että teemme jotain oletuksia kuinka hyödykkeiden hintaindeksit tulevat muuttumaan. Sen arvioimiseksi vaaditaan hieman tutkivaa analyysia, jolle on omistettu osio tässä sovelluksessa. Koska ennusteeseen liittyy oletuksia, lienee parempi soveltaa ennustemallia lyhyelle ajalle (kuten Phillipsin käyrä alunperinkin oli tarkoitettu).',
                                                 style =p_style),
                                          html.P('Lopullinen ennuste perustuu siis käyttäjän syöttämiin oletuksiin valittujen hyödykkeiden kuukausittaisesta muutosnopeudesta. Muutosnopeuden voi säätää jokaiselle hyödykkeelle erikseen, hyödyntää edeltävien kuukausien keskiarvoja tai asettaa kaikille hyödykkeille saman muutosnopeuden. Testit tehdään sillä oletuksella, että aiemmat indeksiarvot toteutuivat sellaisinaan. Raportoinnin ja dokumentaation parantamiseksi, testi, -ja ennustetulokset saa vietyä ulos Excel-tiedostona muita mahdollisia käyttötapauksia varten.',
                                                 style =p_style),
                                          
                                          
                                           html.H3('Sovelluksen käyttöhje',
                                                   style=h3_style
                                                   ),
                                           
                                           html.P('Seuraavaksi hieman ohjeistusta sovelluksen käyttöön. Jokainen osio olisi tehtävä vastaavassa järjestyksessä. Välilehteä valitsemalla pääset suorittamaan jokaisen vaiheen. Välilehdillä on vielä yksityiskohtaisemmat ohjeet. Lisäksi sovelluksen vasemmassa yläkulmassa löytyy painike, josta saa avattua pikaohjeen millä tahansa välilehdellä.', 
                                                    style = p_style),
                                           html.Br(),
                                           html.P('1. Hyödykkeiden valinta. Valitse haluamasi hyödykeryhmät alasvetovalikosta. Näiden avulla ennustetaan työttömyyttä Phillipsin teorian mukaisesti.', 
                                                    style = p_style),
                                          html.P('2. Tutkiva analyysi. Voit tarkastella ja analysoida valitsemiasi hyödykkeitä. Voit tarvittaessa palata edelliseen vaiheeseen ja poistaa tai lisätä hyödykkeitä.',
                                                    style = p_style),
                                          html.P('3. Menetelmän valinta. Tässä osiossa valitset koneoppimisalgoritmin sekä säädät hyperparametrit. Lisäksi voi valita hyödynnetäänkö pääkomponenttianalyysiä ja kuinka paljon variaatiota säilötään.',
                                                    style = p_style),
                                         html.P('4. Testaaminen. Voit valita menneen ajanjakson, jota malli pyrkii ennustamaan. Näin pystyt arvioimaan kuinka ennustemalli olisi toiminut jo toteutuneelle datalle. Tässä osiossa voi myös tarkastella kuinka paljon kukin ennustepiirre vaikutti ennusteen tekemiseen.',
                                                    style = p_style),
                                         html.P('5. Ennusteen tekeminen. Voit nyt hyödyntää valitsemaasi menetelmää tehdäksesi ennusteen tulevaisuuteen. Valitse ennusteen pituus ja klikkaa ennusta. Ennusteen voi sitten viedä myös Exceliin. Ennustetta tehdessä hyödynnetään asettamiasi hyödykkeiden muutosarvoja. Kuten testiosiossa, voit myös tarkastella mitkä hyödykkeet ja piirteet vaikuttavat työttömyysasteen muutokseen ja minkä hyödykkeiden hintamuutokset kontribuoivat työttömyysasteen kuukausimuutoksiin.',
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
                                          
                                          html.H3('Vastuuvapauslauseke',
                                                  style=h3_style),
                                          
                                          html.P("Sivun ja sen sisältö tarjotaan ilmaiseksi sekä sellaisena kuin se on saatavilla. Kyseessä on yksityishenkilön tarjoama palvelu eikä viranomaispalvelu tai kaupalliseen tarkoitukseen tehty sovellus. Sivulta saatavan informaation hyödyntäminen on päätöksiä tekevien tahojen omalla vastuulla. Palvelun tarjoaja ei ole vastuussa menetyksistä, oikeudenkäynneistä, vaateista, kanteista, vaatimuksista, tai kustannuksista taikka vahingosta, olivat ne mitä tahansa tai aiheutuivat ne sitten miten tahansa, jotka johtuvat joko suoraan tai välillisesti yhteydestä palvelun käytöstä. Huomioi, että tämä sivu on yhä kehityksen alla.",
                                                  style=p_style),
                                          
                                          html.P("Tämä sivusto hyödyntää vain välttämättömiä toiminnallisia evästeitä eikä käyttäjien henkilötietoja kerätä mihinkään tarkoitukseen.",
                                                  style=p_style),
                                          html.A([html.P('Katso kolmannen osapuolen tuottama raportti GDPR-yhteensopivuudesta.', style = p_center_style)],
                                                 href = '/assets/report-skewedphillipsherokuappcom-11629005.pdf',
                                                 target = '_blank'),
                                          html.Br(),
                                          html.H3('Tuetut selaimet ja tekniset rajoitukset',
                                                  style=h3_style),
                                          
                                          html.P("Sovellus on testattu toimivaksi Google Chromella, Edgellä ja Mozilla Firefoxilla. Internet Explorer -selaimessa sovellus ei toimi. Opera, Safari -ja muita selaimia ei ole testattu.",
                                                  style=p_style),
                                          html.P("Sovelluksesta voi myös ladata ns. standalone-version, joten sen voi käynnistää ilman selainta esim. Windowsilla tai Androidilla. Esimerkiksi Google Chromessa selaimen osoiterivin oikealla puolella pitäisi olla ikoni, josta klikkaamalla sovelluksen voi ladata. Lataamisen jälkeen sovellus löytyy omalta laitteelta.",
                                                  style=p_style),
                                          
                                          
                                          html.Br(),
                                          html.Div(children = [
                                              html.H3('Lähteet', 
                                                      style = h3_style),
                                              html.P('Tässä on vielä listattu datalähteet sekä lisälukemista kuvattuihin aiheisiin liittyen.',
                                                     style =p_style),
                                              
                                              html.Label(['Tilastokeskuksen maksuttomat tilastotietokannat: ', 
                                                        html.A('Työvoimatutkimuksen tärkeimmät tunnusluvut, niiden kausitasoitetut aikasarjat sekä kausi- ja satunnaisvaihtelusta tasoitetut trendit', href = "https://statfin.stat.fi/PxWeb/pxweb/fi/StatFin/StatFin__tyti/statfin_tyti_pxt_135z.px/",target="_blank")
                                                      ],style=p_style),
                                              html.Br(),
                                              html.Label(['Tilastokeskuksen maksuttomat tilastotietokannat: ', 
                                                        html.A('Kuluttajahintaindeksi (2010=100)', href = "https://statfin.stat.fi/PxWeb/pxweb/fi/StatFin/StatFin__khi/statfin_khi_pxt_11xd.px/",target="_blank")
                                                      ],style=p_style),
                                              html.Br(),
                                              html.Label(['Tilastokeskuksen maksuttomat tilastotietokannat: ', 
                                                        html.A('Kuluttajahintaindeksin vuosimuutos, kuukausitiedot', href = "https://statfin.stat.fi/PxWeb/pxweb/fi/StatFin/StatFin__khi/statfin_khi_pxt_122p.px/",target="_blank")
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
                                              html.Label(['Raha ja Talous: ', 
                                                        html.A('Phillipsin käyrä', href = "https://rahajatalous.wordpress.com/2012/11/15/phillipsin-kayra/",target="_blank")
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
                                                        html.A('A Unified Approach to Interpreting Model (Predict)ions', href = "https://www.researchgate.net/publication/317062430_A_Unified_Approach_to_Interpreting_Model_Predictions",target="_blank")
                                                      ],style=p_style),
                                              html.Br(),
                                              html.Label(['Wikipedia: ', 
                                                        html.A('Shapley-arvot (englanniksi)', href = "https://en.wikipedia.org/wiki/Shapley_value",target="_blank")
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
                                              html.I('Tuomas Poukkula', style = p_center_style),
                                         
                                              html.Br(),
                                              html.P("Data Scientist",
                                                     style = p_center_style),
                                              # html.P("Gofore Oyj",
                                              #        style = p_center_style),
                                              # html.A([html.P('Ota yhteyttä sähköpostilla',style = p_center_style)],
                                              #        href = 'mailto:tuomas.poukkula@gofore.com?subject=Phillips: Palaute ja keskustelu',
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
                                      #         html.Img(
                                      #             src='/assets/gofore_logo_orange.svg',
                                      #             style={
                                                     
                                      #                   'text-align':'center',
                                      #                   'float' : 'center',
                                      #                   'position' : 'center',
                                      #                   'padding-top' : '20px',
                                      #                    'padding-bottom' : '20px'
                                      #             }
                                      #             )
                                      # ], href='https://gofore.com/',target = '_blank', style = {'textAlign':'center'})
                                      # ],style={'textAlign':'center'})],xs =12, sm=12, md=12, lg=6, xl=6)
                                      
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
                                  ]),
                                footer
                 
                        
                        
                        
                        ]

),
            
            dbc.Tab(label ='Hyödykkeiden valinta',
                    tab_id ='feature_tab',
                     tabClassName="flex-grow-1 text-center",
                    tab_style = {'font-size':"1.5625rem",#'25px',
                                 'font-weight': 'bold', 
                                 #'font-family':'Cadiz Semibold'
                                 },
                  style = {
                          
                           
                             
                              "height":"100vh",

                          "overflowY": "scroll"
                      },
              children= [
        
                dbc.Row(children = [
                        
                        
                        dbc.Row([
                            dbc.Col([
                                html.Br(),
                                html.P('Tässä osiossa valitaan hyödykkeitä, joita käytetään työttömyyden ennustamisessa.',
                                        style = p_style),
                                html.P('Voit valita alla olevasta valikosta hyödykkeitä, minkä jälkeen voit säätää niiden oletettavaa kuukausimuutosta syöttämällä lukeman alle ilmestyviin laatikkoihin.',
                                        style = p_style),
                                html.P('Voit myös säätää kaikille hyödykkeille saman kuukausimuutoksen tai hyödyntää toteutuneiden kuukausimuutosten keskiarvoja.',
                                        style = p_style),
                                html.P('Hyödykevalikon voi rajata tai lajitella sen yllä olevasta alasvetovalikosta. Valittavanasi on joko aakkosjärjestys, korrelaatiojärjestykset (Pearsonin korrelaatiokertoimen mukaan) tai rajaus Tilastokeskuksen hyödykehierarkian mukaan. Korrelaatiojärjestyksellä tässä viitataan jokaisen hyödykkeen hintaindeksin arvojen ja saman ajankohdan työtömyysasteiden välistä korrelaatiokerrointa, joka on laskettu Pearsonin metodilla. Nämä voi lajitella laskevaan tai nousevaan järjestykseen joko todellisen arvon mukaan (suurin positiivinen - pienin negatiivinen) tai itseisarvon (ilman etumerkkiä +/-) mukaan.',
                                        style = p_style),
                                html.P("Käytössäsi on myös muutama esimerkkikori, jotka saat käyttöösi valitsemalla sellaisen alasvetovalikosta ja hyödyntämällä valitse kaikki -komentoa.",
                                        style = p_style)
                                
                                ],xs =12, sm=12, md=12, lg=9, xl=9)
                        ], justify = 'center'),
                    
                        dbc.Col(children=[
                           
                            html.Br(),
                            html.Br(),
                            html.H3('Valitse ennustepiirteiksi hyödykeryhmiä valikosta',
                                    style=h3_style),
                            
                            dbc.DropdownMenu(id = 'sorting',
                                              #align_end=True,
                                              children = [
                                                 
                                                  dbc.DropdownMenuItem("Aakkosjärjestyksessä", id = 'alphabet',style={
                                                      'font-size':"0.9rem", 
                                                      #'font-family':'Cadiz Book'
                                                      }
                                                      ),
                                                  dbc.DropdownMenuItem("Korrelaatio (laskeva)", id = 'corr_desc',style={
                                                      'font-size':"0.9rem", 
                                                      #'font-family':'Cadiz Book'
                                                      }
                                                      ),
                                                  dbc.DropdownMenuItem("Korrelaatio (nouseva)", id = 'corr_asc',style={
                                                      'font-size':"0.9rem", 
                                                      #'font-family':'Cadiz Book'
                                                      })
                                                      ,
                                                  dbc.DropdownMenuItem("Absoluuttinen korrelaatio (laskeva)", id = 'corr_abs_desc',style={
                                                      'font-size':"0.9rem", 
                                                      #'font-family':'Cadiz Book'
                                                      }
                                                      ),
                                                  dbc.DropdownMenuItem("Absoluuttinen korrelaatio (nouseva)", id = 'corr_abs_asc',style={
                                                      'font-size':"0.9rem", 
                                                      #'font-family':'Cadiz Book'
                                                      }
                                                      ),
                                                  dbc.DropdownMenuItem("Pääluokittain", id = 'main_class',style={
                                                      'font-size':"0.9rem", 
                                                      #'font-family':'Cadiz Book'
                                                      }
                                                      ),
                                                  dbc.DropdownMenuItem("2. luokka", id = 'second_class',style={
                                                      'font-size':"0.9rem", 
                                                      #'font-family':'Cadiz Book'
                                                      }
                                                      ),
                                                  dbc.DropdownMenuItem("3. luokka", id = 'third_class',style={
                                                      'font-size':"0.9rem", 
                                                      #'font-family':'Cadiz Book'
                                                      }
                                                      ),
                                                  dbc.DropdownMenuItem("4. luokka", id = 'fourth_class',style={
                                                      'font-size':"0.9rem", 
                                                      #'font-family':'Cadiz Book'
                                                      }
                                                      ),
                                                  dbc.DropdownMenuItem("5. luokka", id = 'fifth_class',style={
                                                      'font-size':"0.9rem", 
                                                      #'font-family':'Cadiz Book'
                                                      }
                                                      ),
                                                  dbc.DropdownMenuItem("Esimerkkikori 1", id = 'example_basket_1',style={
                                                      'font-size':"0.9rem", 
                                                      #'font-family':'Cadiz Book'
                                                      }
                                                      ),
                                                  dbc.DropdownMenuItem("Esimerkkikori 2", id = 'example_basket_2',style={
                                                      'font-size':"0.9rem", 
                                                      #'font-family':'Cadiz Book'
                                                      }
                                                      ),
                                                  dbc.DropdownMenuItem("Esimerkkikori 3", id = 'example_basket_3',style={
                                                      'font-size':"0.9rem", 
                                                      #'font-family':'Cadiz Book'
                                                      }
                                                      )
                                                 
                                                 
                                                  ],
                                            label = "Absoluuttinen korrelaatio (laskeva)",
                                            color="secondary", 
                                            className="m-1",
                                            size="lg",
                                            style={
                                                'font-size':"0.9rem", 
                                                #'font-family':'Cadiz Book'
                                                }
                                            ),
                            
                            html.Br(),
                            dcc.Dropdown(id = 'feature_selection',
                                          options = initial_options,
                                          multi = True,
                                          value = list(initial_features),
                                          style = {'font-size':"1rem", #'font-family':'Cadiz Book'
                                                   },
                                          placeholder = 'Valitse hyödykkeitä'),
                            html.Br(),
                            
                            dbc.Alert("Valitse ainakin yksi hyödykeryhmä valikosta!", color="danger",
                                      dismissable=True, fade = True, is_open=False, id = 'alert', 
                                      style = {'text-align':'center', 'font-size':p_font_size, #'font-family':'Cadiz Semibold'
                                               }),
                            dash_daq.BooleanSwitch(id = 'select_all', 
                                                    label = dict(label = 'Valitse kaikki',
                                                                 style = {'font-size':p_font_size, 
                                                                          # #'fontFamily':'Cadiz Semibold'
                                                                          }), 
                                                    on = False, 
                                                    color = 'blue'),
                            html.Br(),
                            # html.Div(id = 'selections_div',children =[]),
                            
                
                            ],xs =12, sm=12, md=12, lg=12, xl=12
                        )
                        ],justify='center'
                    ),
                    dbc.Row(id = 'selections_div', children = [
                        
                            dbc.Col([
                                
                                html.H3('Aseta arvioitu hyödykkeiden hintaindeksien keskimääräinen kuukausimuutos prosenteissa',
                                                        style = h3_style),
                                
                                dash_daq.BooleanSwitch(id = 'averaging', 
                                                        label = dict(label = 'Käytä toteutumien keskiarvoja',style = {'font-size':p_font_size, 
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
                            html.Div(id = 'slider_div'),
                            html.Br(),
                            html.Div(id='slider_prompt_div')
                            
                            ], xs =12, sm=12, md=12, lg=9, xl=9)
                        
                        ],justify = 'center', 
                  
                    ),
                dbc.Row(id = 'adjustments_div',
                        justify = 'left', 
                     
                        ),
                html.Br(),
                footer
            ]
            ),


            dbc.Tab(label = 'Tutkiva analyysi',
                    tab_id = 'eda_tab',
                    tabClassName="flex-grow-1 text-center",
                    tab_style = {'font-size':"1.5625rem",#'25px',
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
                              html.P('Tässä osiossa voit tarkastella työttömyysastetta sekä valittujen kuluttajahintaindeksin hyödykeryhmien keskinäistä suhdetta sekä muutosta ajassa. Alla voit nähdä kuinka eri hyödykeryhmien hintaindeksit korreloivat keskenään sekä työttömyysasteen kanssa. Voit myös havainnoida indeksien, inflaation sekä sekä työttömyysasteen aikasarjoja. Kuvattu korrelaatio perustuu Pearsonin korrelaatiokertoimeen.',
                                     style = p_style),
                              html.Br()
                              ],xs =12, sm=12, md=12, lg=9, xl=9)
                         ],
                             justify = 'center', 
                             ),
                    
                     dbc.Row([
                                dbc.Col(children = [
                                   
                                        html.Div(id = 'corr_selection_div'),
                                        html.Br(),
                                        html.Div(id = 'eda_div',
                                                 children = 
                                                     
                                                     
                                                     [html.Div([dbc.RadioItems(id = 'eda_y_axis', 
                                                                 options = [{'label':'Työttömyysaste (%)','value':'Työttömyysaste'},
                                                                           {'label':'Työttömyysasteen kuukausimuutos (%-yksikköä)','value':'change'}],
                                                                 labelStyle={'display':'inline-block', 'padding':'10px','margin':'10px 10px 10px 10px','font-size':"1.1rem",
                                                                             #'font-family':'Cadiz Book'
                                                                             'font-weight': 'bold'
                                                                             },
                                                                 className="btn-group",
                                                                 inputClassName="btn-check",
                                                                 labelClassName="btn btn-outline-secondary",
                                                                 labelCheckedClassName="active",
                                                               
                                                                 value = 'Työttömyysaste'
                                                               ) ],style={'textAlign':'center'}), 
                                                      html.Div(id = 'commodity_unemployment_div')]
                                                     
                                                     
                                                     ),
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
                                 
                                 html.Div(id = 'unemployment_inflation_div',
                                          
                                          children=[dcc.Graph(id ='employement_inflation',
                                                     figure = go.Figure(data=[go.Scatter(x = data.index,
                                                                               y = data.Työttömyysaste,
                                                                               name = 'Työttömyysaste',
                                                                               mode = 'lines',
                                                                               hovertemplate = '%{x}'+'<br>%{y}',
                                                                               marker = dict(color ='#000000')),
                                                                    go.Scatter(x = data.index,
                                                                               y = data.Inflaatio,
                                                                               name = 'Inflaatio',
                                                                               hovertemplate = '%{x}'+'<br>%{y}',
                                                                               mode ='lines',
                                                                               marker = dict(color = '#CC79A7'))],
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
                                                                                 xaxis = dict(title=dict(text = 'Aika',
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
                                           config=config_plots
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
            dbc.Tab(label='Menetelmän valinta',
                    tab_id ='hyperparam_tab',
                    tabClassName="flex-grow-1 text-center",
                    tab_style = {'font-size':"1.5625rem",#'25px',
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
                        ], justify = 'center'
                            ),
                        html.Br(),
                        dbc.Row([
                            dbc.Col(id = 'model_selection', children = [
                                
                                html.H3('Valitse algoritmi',style=h3_style),
                                
                                dcc.Dropdown(id = 'model_selector',
                                              value = 'Satunnaismetsä',
                                              multi = False,
                                              placeholder = 'Valitse algoritmi',
                                              style = {'font-size':"0.9rem", #'font-family':'Cadiz Book'
                                                       },
                                              options = [{'label': c, 'value': c} for c in MODELS.keys()]),
                                
                                html.Br(),
                                html.H3('Säädä hyperparametrit', style = h3_style),
                                
                                html.Div(id = 'hyperparameters_div')
                                
                                ], xs =12, sm=12, md=12, lg=9, xl=9),
                            dbc.Col(id = 'pca_selections', children = [
                                html.Br(),
                                dash_daq.BooleanSwitch(id = 'pca_switch', 
                                                                  label = dict(label = 'Käytä pääkomponenttianalyysia',style = {'font-size':"1.9rem", 
                                                                                                                                # 'font-family':'Cadiz Semibold',
                                                                                                                                'textAlign':'center'}), 
                                                                  on = False, 
                                                                  color = 'blue'),
                                html.Br(),
                                html.P('Pääkomponenttianalyysi on kohinanpoistomenetelmä, jolla saadaan tiivistettyä ennustepiirteiden informaatio pääkomponentteihin. Jokainen pääkomponentti säilöö alkuperäisen datan variaatiota ja kaikkien pääkomponettien säilötty variaatio summautuu sataan prosenttiin.',
                                       style = p_style),
                                html.A([html.P('Katso lyhyt esittelyvideo pääkomponenttianalyysistä.',
                                               style = p_center_style)],
                                       href = "https://www.youtube.com/embed/hJZHcmJBk1o",
                                       target = '_blank'),
                                
                                
                                html.Div(id = 'ev_placeholder',children =[
                                    html.H3('Valitse säilytettävä variaatio', style = h3_style),
                                    
                                    dcc.Slider(id = 'ev_slider',
                                        min = .7, 
                                        max = .99, 
                                        value = .95, 
                                        step = .01,
                                        tooltip={"placement": "top", "always_visible": True},
                                        marks = {
                                                  .7: {'label':'70%', 'style':{'font-size':"1.2rem", 
                                                                               # 'font-family':'Cadiz Semibold'
                                                                               }},
                                            .85: {'label':'85%', 'style':{'font-size':"1.2rem", 
                                                                          # 'font-family':'Cadiz Semibold'
                                                                          }},
                                                  .99: {'label':'99%', 'style':{'font-size':"1.2rem", 
                                                                                # #'fontFamily':'Cadiz Semibold'
                                                                                }}

                                                }
                                      ),
                                    html.Br(),
                                  html.Div(id = 'ev_slider_update', 
                                          children = [
                                              html.Div([html.P('Valitsit {} % säilytetyn variaation.'.format(95),
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
            dbc.Tab(label='Testaaminen',
                    tab_id ='test_tab',
                    tabClassName="flex-grow-1 text-center",
                    tab_style = {'font-size':"1.5625rem",#'25px',
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
                                        # html.H3('Menetelmän testaaminen', style = h3_style),
                                        # html.Br(),
                                        html.P('Tässä osiossa voit testata kuinka hyvin valittu menetelmä olisi onnistunut ennustamaan menneiden kuukausien työttömyysasteen hyödyntäen valittuja piirteitä. Testattaessa valittu määrä kuukausia jätetään testidataksi, jota menetelmä pyrkii ennustamaan.',
                                               style = p_style),
                                        html.P('Tässä kohtaa hyödykeindeksien oletetaan toteutuvan sellaisinaan.',
                                               style = p_style),
                                        html.P('Tehtyäsi testin voit tarkastella viereistä tuloskuvaajaa tai viedä testidatan alle ilmestyvästä painikeesta Exceliin.',
                                              style=p_style),
                                        html.P('Testitulosten alapuolella olevista kuvaajista voit tarkastella kuinka valitsemasi hyödykkeet vaikuttivat testattuun ennusteeseen.',
                                                style=p_style),
                                        html.Br()
                                    ],xs =12, sm=12, md=12, lg=9, xl=9),
                            ],justify='center'),
                        dbc.Row([
                            dbc.Col([ 
                                        html.H3('Valitse testidatan pituus',style = h3_style),
                                        dcc.Slider(id = 'test_slider',
                                                  min = 1,
                                                  max = 18,
                                                  value = 3,
                                                  step = 1,
                                                  tooltip={"placement": "top", "always_visible": True},
                                                 
                                                  marks = {1: {'label':'kuukausi', 'style':{'font-size':"1.2rem", 
                                                                                            # 'font-family':'Cadiz Semibold'
                                                                                            }},
                                                          # 3:{'label':'3 kuukautta', 'style':{'font-size':20, #'fontFamily':'Cadiz Semibold','color':'white'}},
                                                        
                                                            6:{'label':'puoli vuotta', 'style':{'font-size':"1.2rem", 
                                                                                                # 'font-family':'Cadiz Semibold'
                                                                                                }},
                                                          #  9:{'label':'yhdeksän kuukautta', 'style':{'font-size':20, #'fontFamily':'Cadiz Semibold','color':'white'}},
                                                         
                                                          12:{'label':'vuosi', 'style':{'font-size':"1.2rem", 
                                                                                        # 'font-family':'Cadiz Semibold'
                                                                                        }},
                                                          18:{'label':'puolitoista vuotta', 'style':{'font-size':"1.2rem", 
                                                                                        # 'font-family':'Cadiz Semibold'
                                                                                        }},
                                                          

                                                            }
                                                  ),
                                        html.Br(),  
                                        html.Div(id = 'test_size_indicator', style = {'textAlign':'center'}),
                                        html.Br(),
                                        html.Div(id = 'test_button_div',children = [html.P('Valitse ensin hyödykkeitä.',style = {
                                            'text-align':'center', 
                                            #'font-family':'Cadiz Semibold', 
                                              'font-size':p_font_size
                                            })]),
                                        html.Br(),
                                        html.Div(id='test_download_button_div', style={'textAlign':'center'})
                                        
                                        
                            
                            
                            ],xs =12, sm=12, md=12, lg=9, xl=9)
                            ], justify = 'center', style={'text-align':'center'}
                            ),
                        html.Br(),
                        dbc.Row(children = [
                            dbc.Col([html.Div(id = 'test_results_div')],xs = 12, sm = 12, md = 12, lg = 9, xl = 9),
                            
                            
                            ], justify = 'center', 
                            
                            ),
                        html.Br(),
                        dbc.Row([dbc.Col([html.Div(id = 'shap_selections_div')],xs = 12, sm = 12, md = 12, lg = 9, xl = 9)],justify = 'center'),
                        dbc.Row(children = [
                            
                            dbc.Col([html.Div(id = 'shap_results_div')],xs = 12, sm = 12, md = 12, lg = 6, xl = 6),
                            dbc.Col([html.Div(id = 'local_shap_results_div')],xs = 12, sm = 12, md = 12, lg = 6, xl = 6),
                            
                            ], justify = 'center',align='start'
                            
                            ),
                        html.Br(),
                        footer
                 
                        
                        
                        
                        ]
                    ),
            dbc.Tab(label='Ennustaminen',
                    tab_id = 'forecast_tab',
                    tabClassName="flex-grow-1 text-center",
                    tab_style = {'font-size':"1.5625rem",#'25px',
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
                                        # html.H3('Ennusteen tekeminen',style=h3_style),
                                        # html.Br(),
                                        html.P('Tässä osiossa voit tehdä ennusteen valitulle ajalle. Ennustettaessa on käytössä Menetelmän valinta -välilehdellä tehdyt asetukset. Ennusteen tekemisessä hyödynnetään Hyödykkeiden valinta -välilehdellä tehtyjä oletuksia hyödykkeiden suhteellisesta hintakehityksestä. '
                                               'On hyvä huomioida, että ennuste on sitä epävarmempi mitä pitemmälle ajalle sitä tekee. '
                                               'Lisäksi voit myös tarkastella mallin agnostiikkaa, joka kertoo sen mitkä hyödykkeet ja piirteet tulevat vaikuttamaan työttömyysasteen kehitykseen ja minkä hyödykkeiden hintojen nousut ja laskut tulevat vaikuttamaan työttömyysasteen muutokseen kuukausittain mikäli käyttäjän tekemät oletukset toteutuvat.',
                                              style=p_style),
                                        html.P('Tehtyäsi ennusteen voit tarkastella viereistä ennusteen kuvaajaa tai viedä tulosdatan alle ilmestyvästä painikeesta Exceliin.',
                                              style=p_style),
                                        html.Br(),
                                        
                                        html.H3('Valitse ennusteen pituus',
                                                style=h3_style),
                                        dcc.Slider(id = 'forecast_slider',
                                                  min = 2,
                                                  max = 18,
                                                  value = 3,
                                                  step = 1,
                                                  tooltip={"placement": "top", "always_visible": True},
                                                  marks = {2: {'label':'2 kuukautta', 'style':{'font-size':"1rem"#16, 
                                                                                               # #'fontFamily':'Cadiz Semibold'
                                                                                               }},
                                                          # 3: {'label':'kolme kuukautta', 'style':{'font-size':20, #'fontFamily':'Cadiz Semibold','color':'white'}},
                                                          6:{'label':'puoli vuotta', 'style':{'font-size':"1rem"#16, 
                                                                                              # #'fontFamily':'Cadiz Semibold'
                                                                                              }},
                                                          # 9:{'label':'yhdeksän kuukautta', 'style':{'font-size':20, #'fontFamily':'Cadiz Semibold','color':'white'}},
                                                          12:{'label':'vuosi', 'style':{'font-size':"1rem"#16, 
                                                                                        # 'font-family':'Cadiz Semibold'
                                                                                        }},
                                                          18:{'label':'puolitoista vuotta', 'style':{'font-size':"1rem"#16, 
                                                                                        # 'font-family':'Cadiz Semibold'
                                                                                        }},
                                                        #  24:{'label':'kaksi vuotta', 'style':{'font-size':20, #'fontFamily':'Cadiz Semibold','color':'white'}}
                                                        
                                                     

                                                          }
                                                  ),
                                        html.Br(),
                                        html.Div(id = 'forecast_slider_indicator',style = {'textAlign':'center'}),
                                        html.Div(id = 'forecast_button_div',children = [html.P('Valitse ensin hyödykkeitä.',
                                                                                              style = p_style
                                                                                              )],style = {'textAlign':'center'})
                                        
                                    ],xs =12, sm=12, md=12, lg=9, xl=9)
                            
                            
                            ], justify = 'center'),
                        html.Br(),
                        dbc.Row(children = [
                                    
                                    dbc.Col([dcc.Loading(id = 'forecast_results_div',type = spinners[random.randint(0,len(spinners)-1)])],
                                            xs = 12, sm = 12, md = 12, lg = 8, xl = 8)
                                    ], justify = 'center', 
                             # style = {'margin' : '10px 10px 10px 10px'}
                                    ),
                        html.Br(),
                        dbc.Row([dbc.Col([html.Div(id = 'forecast_shap_selections_div')],xs = 12, sm = 12, md = 12, lg = 9, xl = 9)],justify = 'center'),
                        dbc.Row([
                            
                            dbc.Col(id = 'forecast_shap_div', xs = 12, sm = 12, md = 12, lg = 6, xl = 6),
                            dbc.Col(id = 'forecast_local_shap_div', xs = 12, sm = 12, md = 12, lg = 6, xl = 6)
                            
                        ], justify ='center', align='start'),
                        html.Br(),
                        footer
                                       
                            
                            
                            
                            
                              
                            
                        ]
                            
                            )


        ]
            
    )
       
    
   ]
  )])

@callback(
    
    [Output('shap_features_switch', 'label'),
     Output('shap_features_switch', 'disabled')],
    Input('shap_data','data')
    
)
def update_shap_switch(shap_data):
    
    shap_df = pd.DataFrame(shap_data)
    shap_df = shap_df.set_index(shap_df.columns[0])
    
    if 'Kuukausi' not in shap_df.index:
        return dict(label = 'Käytit pääkomponenttianalyysiä',
                     style = {'font-size':p_font_size,
                              'text-align':'center'
                              # #'fontFamily':'Cadiz Semibold'
                              }), True
    else:
        return dict(label = 'Näytä vain hyödykkeiden kontribuutio',
                     style = {'font-size':p_font_size, 
                              'text-align':'center'
                              # #'fontFamily':'Cadiz Semibold'
                              }), False
    
    
@callback(
    
    [Output('forecast_shap_features_switch', 'label'),
     Output('forecast_shap_features_switch', 'disabled')],
    Input('forecast_shap_data','data')
    
)
def update_forecast_shap_switch(shap_data):
    
    shap_df = pd.DataFrame(shap_data)
    shap_df = shap_df.set_index(shap_df.columns[0])
    
    if 'Kuukausi' not in shap_df.index:
        return dict(label = 'Käytit pääkomponenttianalyysiä',
                     style = {'font-size':p_font_size,
                              'text-align':'center'
                              # #'fontFamily':'Cadiz Semibold'
                              }), True
    else:
        return dict(label = 'Näytä vain hyödykkeiden kontribuutio',
                     style = {'font-size':p_font_size, 
                              'text-align':'center'
                              # #'fontFamily':'Cadiz Semibold'
                              }), False

@callback(

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
                                html.P(feature,style={#'font-family':'Messina Modern Semibold',
                                            'font-size':"1.1rem"#22
                                            }),
                                dcc.Input(id = {'type':'value_adjust', 'index':feature}, 
                                               value = round(mean_df.loc[feature],1), 
                                               type = 'number', 
                                               style={#'font-family':'Messina Modern Semibold',
                                                           'font-size':"1.1rem"#22
                                                           },
                                               step = .1)],xs =12, sm=12, md=4, lg=2, xl=2) for feature in features]
    else:
        
        features_values = {feature:slider_value for feature in features}
        
        row_children =[dbc.Col([html.Br(), 
                                html.P(feature,style={#'font-family':'Messina Modern Semibold',
                                            'font-size':"1.1rem"#22
                                            }),
                                dcc.Input(id = {'type':'value_adjust', 'index':feature}, 
                                               value = slider_value, 
                                               type = 'number', 
                                               style ={#'font-family':'Messina Modern Semibold',
                                                           'font-size':"1.1rem"#22
                                                           },
                                               step = .1)],xs =12, sm=12, md=4, lg=2, xl=2) for feature in features]
    return row_children, features_values


@callback(

    Output('change_weights','data'),
    [Input({'type': 'value_adjust', 'index': ALL}, 'id'),
    Input({'type': 'value_adjust', 'index': ALL}, 'value')],    
    
)
def store_weights(feature_changes, feature_change_values):
    
    if feature_changes is None:
        raise PreventUpdate
    
    weights_dict = {feature_changes[i]['index']:feature_change_values[i] for i in range(len(feature_changes))}
        
    return weights_dict


@callback(

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
                       marks = {1:{'label':'kuukausi', 'style':{'font-size':"1.1rem"#20, 
                                                                # #'fontFamily':'Cadiz Semibold'
                                                                }},
                                # 3:{'label':'3 kuukautta', 'style':{'font-size':20, #'fontFamily':'Cadiz Semibold','color':'white'}},
                                6:{'label':'puoli vuotta', 'style':{'font-size':"1.1rem"#20, 
                                                                    # #'fontFamily':'Cadiz Semibold'
                                                                    }},
                                # 9:{'label':'yhdeksän kuukautta', 'style':{'font-size':20, #'fontFamily':'Cadiz Semibold','color':'white'}},
                                12:{'label':'vuosi', 'style':{'font-size':"1.1rem"#20, 
                                                              # #'fontFamily':'Cadiz Semibold'
                                                              }}   
                             }
                      
                    )]
        
    else:
        return [
            html.H3('Valitse kuinka isoa suhteellista kuukausimuutosta sovelletaan', 
                    style = h3_style),
            
            dcc.Slider(id = 'slider',
                          min = -20,
                          max = 20,
                          value = 0,
                          step = 0.1,
                          tooltip={"placement": "top", "always_visible": True},
                           marks = {
                                    # -30:{'label':'-30%', 'style':{'font-size':20, #'fontFamily':'Cadiz Semibold'
                                    #                                       }},
                                   -20:{'label':'-20%', 'style':{'font-size':"1.1rem"#22, #'fontFamily':'Cadiz Semibold'
                                                                  }},
                                    # 3:{'label':'3 kuukautta', 'style':{'font-size':20, #'fontFamily':'Cadiz Semibold','color':'white'}},
                                    0:{'label':'0%', 'style':{'font-size':"1.1rem"#22, #'fontFamily':'Cadiz Semibold'
                                                              }},
                                    # 9:{'label':'yhdeksän kuukautta', 'style':{'font-size':20, #'fontFamily':'Cadiz Semibold','color':'white'}},
                                    20:{'label':'20%', 'style':{'font-size':"1.1rem"#22, #'fontFamily':'Cadiz Semibold'
                                                                }},
                                    # 30:{'label':'30%', 'style':{'font-size':20, #'fontFamily':'Cadiz Semibold'
                                    #                             }} 
                                 }
                          
                        )
            
            ]


@callback(

    Output('alert', 'is_open'),
    [Input('feature_selection','value')]

)
def update_alert(features):
    
    return len(features) == 0

@callback(

    Output('hyperparameters_div','children'),
    [Input('model_selector','value')]    
    
)
def update_hyperparameter_selections(model_name):
    
    
    
    model = MODELS[model_name]['model']
    
        
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
    
    elif model_name=='Gradient Boost':
        hyperparameters.update({'subsample':.99})
    
    
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
                        children.append(dbc.Col([dbc.Input(id = {'index':hyperparameter, 'type':'hyperparameter_tuner'},
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
                    children.append(dbc.Col([dbc.Input(id = {'index':hyperparameter, 'type':'hyperparameter_tuner'},
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
                children.append(dbc.Col([dbc.Switch(id = {'index':hyperparameter, 'type':'hyperparameter_tuner'}, 
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
                        children.append(dbc.Col([dcc.Dropdown(id = {'index':hyperparameter, 'type':'hyperparameter_tuner'}, 
                                                  multi = False,
                                                  #label = hyperparameter,
                                                  style = {
                                                      #'font-family':'Cadiz Book',
                                                      'font-size':"0.9rem"#p_font_size-3
                                                      },
                                                  options = [{'label':c, 'value': c} for c in param_options[hyperparameter] if c not in ['precomputed','poisson']],
                                                  value = value),
                                                 html.Br()],xs =12, sm=12, md=12, lg=2, xl=2)
                                    )
                        children.append(html.Br())
                        children.append(html.Br())
                       
     
    children.append(html.Br()) 
    children.append(html.Div(style = {'textAlign':'center'},
             children = [html.A('Katso lyhyt esittelyvideo käytetystä algoritmista.', href = MODELS[model_name]['video'], target="_blank",style = p_style),
                         html.Br(),
                         html.A('Katso mallin tekninen dokumentaatio.', href = MODELS[model_name]['doc'], target="_blank",style = p_style),
                         ]))
    return dbc.Row(children, justify ='start')


@callback(

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

@callback(
    
      [Output('test_data','data'),
       Output('test_results_div','children'),
       Output('test_download_button_div','children'),
       Output('shap_data','data'),
       Output('local_shap_data','data')],
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
    
        
        try:
            locale.setlocale(locale.LC_ALL, 'fi_FI')
        except:
            locale.setlocale(locale.LC_ALL, 'fi-FI')

    
        
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
        
        explainer = MODELS[model_name]['explainer']
        
        
        test_result, shap_results, local_shap_df = test(model, features, explainer = explainer, test_size=test_size, use_pca=pca,n_components=explained_variance)

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
                             html.H3('Miten testi onnistui?',
                                     style = h3_style),
                             
                             html.P('Alla olevassa kuvaajassa nähdään kuinka hyvin ennustemalli olisi ennustanut työttömyysasteen ajalle {} - {}.'.format(test_result.index.strftime('%B %Y').values[0],test_result.index.strftime('%B %Y').values[-1]),
                                    style = p_style),
                             html.P("(Jatkuu kuvaajan jälkeen)",style={
                                         'font-style':'italic',
                                         'font-size':p_font_size,
                                        'text-align':'center'}
                                 ),
                              html.Div([html.Br(),dbc.RadioItems(id = 'test_chart_type', 
                                          options = [{'label':'pylväät','value':'bars'},
                                                    {'label':'viivat','value':'lines'},
                                                    {'label':'viivat ja pylväät','value':'lines+bars'}],
                                          labelStyle={'display':'inline-block', 'padding':'10px','margin':'10px 10px 10px 10px','font-size':"1.1rem",#18,
                                                      'font-weight': 'bold'
                                                      #'font-family':'Cadiz Book'
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
                    # html.P('Keskimääräinen suhteellinen virhe (MAPE) on kaikkien ennustearvojen suhteellisten virheiden keskiarvo. Tarkkuus on tässä tapauksessa laskettu kaavalla 1 - MAPE.', 
                    #        style = p_center_style,
                    #        className="card-text"),
                    html.Div([
                        html.Label([html.A('Keskimääräinen suhteellinen virhe ( ', 
                                           # style = {'font-size':'1.2rem'}
                                           ),
                                html.A('MAPE', 
                                       href = 'https://en.wikipedia.org/wiki/Mean_absolute_percentage_error',
                                       target='_blank',
                                       # style = {'font-size':'1.2rem'}
                                       ),
                                html.A(' ) on kaikkien ennustearvojen suhteellisten virheiden keskiarvo. Tarkkuus on tässä tapauksessa laskettu kaavalla 1 - MAPE.',
                                       # style = {'font-size':'1.2rem'}
                           )
                            ],
                                   style ={'font-size':'1.2rem'}
                           )
                        ],style = {'text-align':'center'},
                        className="card-text"),
                    html.Br(),

                    
                    ]
             

        feat = features.copy()
        feat = ['Työttömyysaste','Ennuste','prev','month','change','mape','n_feat', 'Ennustettu muutos']+feat
        
        button_children = dbc.Button(children=[html.I(className="fa fa-download mr-1"), ' Lataa testitulokset koneelle'],
                                       id='test_download_button',
                                       n_clicks=0,
                                       style = dict(fontSize=25,
                                                    # fontFamily='Cadiz Semibold',
                                                    textAlign='center'),
                                       outline=True,
                                       size = 'lg',
                                       color = 'info'
                                       )
        
       
        
        return test_result[feat].reset_index().to_dict('records'),test_plot, button_children,shap_data, local_shap_data
    else:
        return [html.Div(),html.Div(),html.Div(),html.Div(),html.Div()]


        
@callback(
    
      [Output('forecast_data','data'),
       Output('forecast_shap_data','data'),
       Output('local_forecast_shap_data','data'),
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
        
        try:
            locale.setlocale(locale.LC_ALL, 'fi_FI')
        except:
            locale.setlocale(locale.LC_ALL, 'fi-FI')
        
        features = sorted(list(weights_dict.keys()))
        
        
        
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
        
        explainer = MODELS[model_name]['explainer']
        
        forecast_df, shap_df, local_shap_df = predict(model, 
                              explainer,
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
                      html.P('Voit valita alla olevista painikkeista joko pylväs, alue, -tai viivadiagramin. Kuvaajan pituutta voi säätää alla olevasta liukuvaliskosta. Pituutta voi rajat myös vasemman yläkulman painikkeista.',
                             style = p_style),
                      html.P("(Jatkuu kuvaajan jälkeen)",style={
                                  'font-style':'italic',
                                  'font-size':p_font_size,
                                 'text-align':'center'}
                          ),
                      
                      html.Div([
                      dbc.RadioItems(id = 'chart_type', 
                        options = [{'label':'pylväät','value':'bars'},
                                  {'label':'viivat','value':'lines'},
                                  {'label':'alue','value':'area'}],
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

          html.Div(id = 'forecast_graph_div'),
        
          html.Br()
          ])

          
          # ], justify='center')        
        forecast_download_button = dbc.Button(children=[html.I(className="fa fa-download mr-1"), ' Lataa ennustedata koneelle'],
                                 id='forecast_download_button',
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
                forecast_div,
                [html.Br(),
                 forecast_download_button]
                ]
    else:
        return [html.Div(),html.Div(),html.Div(),html.Div(),html.Div()]
    
@callback(

    [Output('shap_selections_div','children'),
     Output('shap_results_div','children'),
     Output('local_shap_results_div','children')
     ],
    [Input('test_button','n_clicks'),
     Input('shap_data','data'),
     State('local_shap_data','data')]    
    
)
def update_shap_results(n_clicks, shap, local_shap_data):
        
    if shap is None or local_shap_data is None:
        raise PreventUpdate
        
    if n_clicks > 0:
        
        try:
            locale.setlocale(locale.LC_ALL, 'fi_FI')
        except:
            locale.setlocale(locale.LC_ALL, 'fi-FI')
        
    
        shap_df = pd.DataFrame(shap)
        
        shap_df = shap_df.set_index(shap_df.columns[0])
        
        
        local_shap_df = pd.DataFrame(local_shap_data)
        local_shap_df = local_shap_df.set_index(local_shap_df.columns[0])
        local_shap_df.index = pd.to_datetime(local_shap_df.index)
        
        options = [{'label':c.strftime('%B %Y'), 'value': c} for c in list(local_shap_df.index)]

         
        return [[
            
                    html.H3('Mitkä ennustepiirteet vaikuttivat eniten?',
                           style = h3_style),
                    html.P('Oheisissa kuvaajissa on esitetty käytettyjen ennustepiirteiden globaalit ja lokaalit tärkeydet ennusteelle. '
                           'Globaaleilla merkitysarvoilla voidaan tarkastella mitkä piirteet yleisesti ottaen ovat merkitsevimpiä ennusteelle. '
                           'Sen sijaan lokaaleilla arvoilla voidaan arvioida, mitkä tekijät nostivat tai laskivat ennusteen arvoa tiettynä kuukautena. '
                           'Merkitysarvot on esitetty ns. SHAP-arvoina, jotka kuvaavat piirteiden kontribuutiota ennusteelle. '
                           'Ennustepiirteisiin kuuluvat valittujen hyödykeindeksien lisäksi edellisen kuukauden työttömyysaste sekä kuukausi.',
                           style = p_style),
                    html.A([html.P('Katso lyhyt esittely SHAP -arvojen merkityksestä mallin selittämisessä.',
                                   style = p_center_style)], href="https://www.youtube.com/embed/Tg8aPwPPJ9c", target='_blank'),
                    html.A([html.P('Katso myös ei-tekninen selittävä blogi SHAP - arvoista.',
                                   style = p_center_style)], href="https://www.aidancooper.co.uk/a-non-technical-guide-to-interpreting-shap-analyses/", target='_blank'),
                    html.P('Kuvaajan SHAP-arvot on kerrottu sadalla visualisoinnin parantamiseksi. '
                           'Yksi SHAP-yksikkö vastaa siten yhtä sadasosaa prosenttiyksiköistä, jolla kuvataan työttömyysasteen kuukausimuutosta. ',
                           style = p_style),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                                html.Div(id = 'cut_off_div'),
                                
                                html.Div(id = 'cut_off_indicator'),
                                
                                ],xs =12, sm=12, md=12, lg=9, xl=9),
                        dbc.Col([
                                dash_daq.BooleanSwitch(id = 'shap_features_switch', 
                                                        label = dict(label = 'Näytä vain hyödykkeiden kontribuutio',
                                                                     style = {'font-size':p_font_size,
                                                                              'text-align':'center',
                                                                              # #'fontFamily':'Cadiz Semibold'
                                                                              }), 
                                                        on = False, 
                                                        color = '#D41159')
                                ],xs =12, sm=12, md=12, lg=3, xl=3)
                        ]),
                    html.Br()
                    ],
                    
                    [html.Br(),
                        dbc.Card([
                        dbc.CardBody([
                            html.H3('Piirteiden tärkeydet', className='card-title',
                                    style=h3_style),
                            html.P("Alla olevassa kuvaajassa on esitetty keskimääräiset absoluuttiset SHAP-arvot. "
                           "Ne kuvaavat kuinka paljon piirteet keskimäärin vaikuttivat ennusteisiin, riippumatta vaikutuksen suunnasta. "
                           "Ne on laskettu piirteiden lokaalien absoluuttisten SHAP - arvojen keskiarvona. "
                           "Mustalla on merkitty triviaalit piirteet eli kuluva kuukausi ja edellisen kuukauden työttömyysaste. ",
                            style =p_style,
                           className="card-text"),
                            html.Br(),
                            dcc.Loading([dbc.Row(id = 'shap_graph_div', justify = 'center')], type = random.choice(spinners))
                            ])
                        ])
                        ],
                
                    [html.Br(),
                     dbc.Card([
                         dbc.CardBody([
                             html.H3('Piirteiden merkitykset kuukausittain',className='card-title',
                                     style=h3_style),
                             html.P("Alla olevassa kuvaajassa on esitetty piirteiden lokaalit SHAP-arvot valitulle kuukaudelle. "
                                    "Ne kuvaavat suuntaa ja voimakkuutta, joka piirteillä oli valitun kuukauden ennusteeseen. "
                                    "Vihreällä värillä on korostettu työttömyyden kuukausimuutosta laskevat tekijät ja punaisella sitä nostavat piirteet. "
                                    "Mustalla on merkitty triviaalit piirteet eli kuluva kuukausi ja edellisen kuukauden työttömyysaste. "
                                    "Pystyakselilla on esitetty piirteiden nimet sekä suluissa niiden arvo valittuna ajankohtana ja muutoksen suunta edelliseen kuukauteen nähden kuvaavalla ikonilla. "
                                    "Kuvaajan alapuolella on esitetty kaava, jolla kuukausiennusteet voidaan laskea SHAP-arvoilla ja mallin tuottamalla vakioarvolla. ",
                                    style =p_style,className="card-text"),

                     
                            html.Br(),
                            html.H3('Valitse kuukausi', style =h3_style),
                                        dcc.Dropdown(id = 'local_shap_month_selection',
                                                      options = options, 
                                                      style = {'font-size':"1rem"},
                                                      value = list(local_shap_df.index)[0],
                                                      multi=False ),
                                        html.Br(),
                                        
                                        html.Div(dcc.Loading(id = 'local_shap_graph_div',
                                                              type = random.choice(spinners)))
                                    
                                    ])
                         ])
                                    ]]

    else:
        return [html.Div(),html.Div(),html.Div()]
    
    
@callback(

    [Output('forecast_shap_selections_div','children'),
     Output('forecast_shap_div','children'),
     Output('forecast_local_shap_div','children')
     ],
    [Input('forecast_button','n_clicks'),
     Input('forecast_shap_data','data'),
     State('local_forecast_shap_data','data')]    
    
)
def update_forecast_shap_results(n_clicks, shap, local_shap_data):
        
    if shap is None or local_shap_data is None:
        raise PreventUpdate
        
    if n_clicks > 0:
        
        try:
            locale.setlocale(locale.LC_ALL, 'fi_FI')
        except:
            locale.setlocale(locale.LC_ALL, 'fi-FI')
        
    
        shap_df = pd.DataFrame(shap)
        
        shap_df = shap_df.set_index(shap_df.columns[0])
        
        
        local_shap_df = pd.DataFrame(local_shap_data)
        local_shap_df = local_shap_df.set_index(local_shap_df.columns[0])
        
        
        
        local_shap_df.index = pd.to_datetime(local_shap_df.index)
        
        options = [{'label':c.strftime('%B %Y'), 'value': c} for c in list(local_shap_df.index)]

         
        return [[
            
                    html.H3('Mitkä ennustepiirteet tulevat vaikuttamaan eniten?',
                           style = h3_style),
                    html.P("Kuten testausosiossakin, ennustettaessa tulevien kuukausien työttömyttä voidaan myös tarkastella mitkä piirteet tulevat merkitsemään eniten ja mitkä piirteet tulevat selittämään jonkin tulevan kuukauden työttömyysasteen muutosta. "
                           "Yllä oleva kuvaaja osoittaa mallin tuottaman ennusteen sillä oletuksella, että käyttäjän valitsemien hyödykkeiden hintaindeksit muuttuvat valitulla muutosnopeudella kuukausittain. "
                           "Alla olevien globaalien ja lokaalien SHAP-arvojen avulla onkin mahdollista tarkastella miten käyttäjän valitsemat hyödykekohtaiset kuukausimuutokset vaikuttavat ennusteeseen. "
                           "Tässä on siten mahdollista palata hyödykkeiden valinta -osioon, säätää muutosnopeutta ja kokeilla uudestaan ennustaa useilla kuukausimuutoksilla.",
                           style = p_style),
                    # html.A([html.P('Katso lyhyt esittely SHAP -arvojen merkityksestä mallin selittämisessä.',
                    #                style = p_style)], href="https://www.youtube.com/embed/Tg8aPwPPJ9c", target='_blank'),
                    # html.A([html.P('Katso myös ei-tekninen selittävä blogi SHAP - arvoista.',
                    #                style = p_style)], href="https://www.aidancooper.co.uk/a-non-technical-guide-to-interpreting-shap-analyses/", target='_blank'),
                    html.P('Kuvaajan SHAP-arvot on kerrottu sadalla visualisoinnin parantamiseksi. '
                           'Yksi SHAP-yksikkö vastaa siten yhtä sadasosaa prosenttiyksiköistä, jolla kuvataan työttömyysasteen kuukausimuutosta. ',
                           style = p_style),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                                html.Div(id = 'forecast_cut_off_div'),
                                
                                html.Div(id = 'forecast_cut_off_indicator'),
                                
                                ],xs =12, sm=12, md=12, lg=9, xl=9),
                        dbc.Col([
                                dash_daq.BooleanSwitch(id = 'forecast_shap_features_switch', 
                                                        label = dict(label = 'Näytä vain hyödykkeiden kontribuutio',
                                                                     style = {'font-size':p_font_size,
                                                                              'text-align':'center',
                                                                              # #'fontFamily':'Cadiz Semibold'
                                                                              }), 
                                                        on = False, 
                                                        color = '#D41159')
                                ],xs =12, sm=12, md=12, lg=3, xl=3)
                        ]),
                    html.Br()
                    ],
                    
                    [html.Br(),
                        dbc.Card([
                        dbc.CardBody([
                            html.H3('Piirteiden tärkeydet', className='card-title',
                                    style=h3_style),
                            html.P("Alla olevassa kuvaajassa esitettyjen globaalien SHAP-arvojen avulla voidaan valitut hyödykkeet ja piirteet laittaa mallin painottamaan vaikuttavusjärjestykseen. "
                                   "Suurimman globaalin merkitsevyysarvon saanut piirre merkitsi kokonaisuudessa eniten riippumatta muutoksen suunnasta. ",
                            style =p_style,
                           className="card-text"),
                            html.Br(),
                            dcc.Loading([dbc.Row(id = 'forecast_shap_graph_div', justify = 'center')], type = random.choice(spinners))
                            ])
                        ])
                        ],
                
                    [html.Br(),
                     dbc.Card([
                         dbc.CardBody([
                             html.H3('Piirteiden merkitykset kuukausittain',className='card-title',
                                     style=h3_style),
                             html.P("Alla olevassa kuvaajassa on esitetty piirteiden lokaalit SHAP-arvot, joita voi tarkastella kuukausittain valitsemalla haluttu kuukausi alasvetovalikosta. "
                                    "Nämä arvot kertovat minkä hyödykkeiden hintojen nousut laskevat tai nostavat työttömyyttä valittuna kuukautena. "
                                    "Jakamalla hyödykkeen SHAP-arvo sadalla saadaan tulokseksi prosenttiyksiköissä se muutos, jonka hyödykkeen hinnan muutos kontribuoi työttömyyden kuukausimuutokseen. "
                                    "Kuvaajan alla on esitetty kaava, jolla työttömyysaste voidaan laskea SHAP-arvojen avulla.",
                                    style =p_style,className="card-text"),

                     
                            html.Br(),
                            html.H3('Valitse kuukausi', style =h3_style),
                                        dcc.Dropdown(id = 'forecast_local_shap_month_selection',
                                                      options = options, 
                                                      style = {'font-size':"1rem"},
                                                      value = list(local_shap_df.index)[0],
                                                      multi=False ),
                                        html.Br(),
                                        
                                        html.Div(dcc.Loading(id = 'forecast_local_shap_graph_div',
                                                              type = random.choice(spinners)))
                                    
                                    ])
                         ])
                                    ]]

    else:
        return [html.Div(),html.Div(),html.Div()]
    
@callback(

    Output('cut_off_indicator','children'),
    [Input('cut_off','value')]    
    
)
def update_cut_off_indicator(cut_off):
    return [html.P('Valitsit {} piirrettä.'.format(cut_off).replace(' 1 piirrettä',' yhden piirteen'), style = p_center_style)]

@callback(

    Output('forecast_cut_off_indicator','children'),
    [Input('forecast_cut_off','value')]    
    
)
def update_forecast_cut_off_indicator(cut_off):
    return [html.P('Valitsit {} piirrettä.'.format(cut_off).replace(' 1 piirrettä',' yhden piirteen'), style = p_center_style)]
    
@callback(

    Output('cut_off_div','children'),
    [Input('shap_data','data')]    
    
)
def update_shap_slider(shap):
    if shap is None:
        raise PreventUpdate

    shap_df = pd.DataFrame(shap)
    
    
    return [html.P('Valitse kuinka monta piirrettä näytetään kuvaajassa',
                       style = p_center_style),
                dcc.Slider(id = 'cut_off',
                   min = 1, 
                   max = len(shap_df),
                   value = {True:len(shap_df), False: int(math.ceil(.2*len(shap_df)))}[len(shap_df)<=25],
                   step = 1,
                   marks=None,
                   tooltip={"placement": "top", "always_visible": True},
                   )]

@callback(

    Output('forecast_cut_off_div','children'),
    [Input('forecast_shap_data','data')]    
    
)
def update_forecast_shap_slider(shap):
    if shap is None:
        raise PreventUpdate

    shap_df = pd.DataFrame(shap)
    
    
    return [html.P('Valitse kuinka monta piirrettä näytetään kuvaajassa',
                       style = p_center_style),
                dcc.Slider(id = 'forecast_cut_off',
                   min = 1, 
                   max = len(shap_df),
                   value = {True:len(shap_df), False: int(math.ceil(.2*len(shap_df)))}[len(shap_df)<=25],
                   step = 1,
                   marks=None,
                   tooltip={"placement": "top", "always_visible": True},
                   )]

@callback(

    Output('shap_graph_div', 'children'),
    [Input('cut_off', 'value'),
     Input('shap_features_switch','on'),
     State('shap_data','data')]
    
)
def update_shap_graph(cut_off, only_commodities, shap):
    
    if shap is None:
        raise PreventUpdate
        
    
    shap_df = pd.DataFrame(shap)
    shap_df = shap_df.set_index(shap_df.columns[0])
  
    
    
    if only_commodities:
        shap_df = shap_df.loc[[i for i in shap_df.index if i not in ['Kuukausi', 'Edellisen kuukauden työttömyysaste']]]
    
    
    shap_df = shap_df.sort_values(by='SHAP', ascending = False)
    
   
    df = pd.DataFrame(shap_df.iloc[cut_off+1:,:].sum())
    df = df.T
    df.index = df.index.astype(str).str.replace('0', 'Muut {} piirrettä'.format(len(shap_df.iloc[cut_off+1:,:])))
    
    
    shap_df = pd.concat([shap_df.head(cut_off),df])
    shap_df = shap_df.loc[shap_df.index != 'Muut 0 piirrettä']
    shap_df.index = shap_df.index.str.replace('_','')
    

    height = graph_height +200 + 10*len(shap_df)
    
    
    return dcc.Graph(id = 'shap_graph',
                     config = config_plots,
                         figure = go.Figure(data=[go.Bar(y =shap_df.index, 
                      x = np.round(100*shap_df.SHAP,2),
                      orientation='h',
                      name = '',
                      marker_color = ['aquamarine' if i not in ['Kuukausi','Edellisen kuukauden työttömyysaste'] else 'black' for i in shap_df.index],
                      # marker = dict(color = 'turquoise'),
                      text = np.round(100*shap_df.SHAP,2),
                      hovertemplate = '<b>%{y}</b>: %{x}',
                          textfont = dict(
                               family='Cadiz Semibold', 
                              size = 16))],
         layout=go.Layout(title = dict(text = 'Piirteiden globaalit merkitykset<br>Keskimääräiset |SHAP - arvot|',
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
                                                        xaxis = dict(title=dict(text = 'Keskimääräinen |SHAP - arvo|',
                                                                                font=dict(
                                                                                    size=18, 
                                                                                    family = 'Cadiz Semibold'
                                                                                    )),
                                                                     automargin=True,
                                                                     tickfont = dict(
                                                                         family = 'Cadiz Semibold', 
                                                                          size = 14
                                                                         )),
                                                        yaxis = dict(title=dict(text = 'Ennustepiirre',
                                                                               font=dict(
                                                                                    size=18, 
                                                                                   family = 'Cadiz Semibold'
                                                                                   )),
                                                                    tickformat = ' ',
                                                                     categoryorder='total ascending',
                                                                    automargin=True,
                                                                    tickfont = dict(
                                                                        family = 'Cadiz Semibold', 
                                                                         size = 14
                                                                        ))
                                                        )))

@callback(

    Output('forecast_shap_graph_div', 'children'),
    [Input('forecast_cut_off', 'value'),
     Input('forecast_shap_features_switch','on'),
     State('forecast_shap_data','data')]
    
)
def update_forecast_shap_graph(cut_off, only_commodities, shap):
    
    if shap is None:
        raise PreventUpdate
        
    
    shap_df = pd.DataFrame(shap)
    shap_df = shap_df.set_index(shap_df.columns[0])
  
    
    
    if only_commodities:
        shap_df = shap_df.loc[[i for i in shap_df.index if i not in ['Kuukausi', 'Edellisen kuukauden työttömyysaste']]]
    
    
    shap_df = shap_df.sort_values(by='SHAP', ascending = False)
    
   
    df = pd.DataFrame(shap_df.iloc[cut_off+1:,:].sum())
    df = df.T
    df.index = df.index.astype(str).str.replace('0', 'Muut {} piirrettä'.format(len(shap_df.iloc[cut_off+1:,:])))
    
    
    shap_df = pd.concat([shap_df.head(cut_off),df])
    shap_df = shap_df.loc[shap_df.index != 'Muut 0 piirrettä']
    
    shap_df.index = shap_df.index.str.replace('_','')
    

    height = graph_height +200 + 10*len(shap_df)
    
    
    return dcc.Graph(id = 'forecast_shap_graph',
                     config = config_plots,
                         figure = go.Figure(data=[go.Bar(y =shap_df.index, 
                      x = np.round(100*shap_df.SHAP,2),
                      orientation='h',
                      name = '',
                      marker_color = ['aquamarine' if i not in ['Kuukausi','Edellisen kuukauden työttömyysaste'] else 'black' for i in shap_df.index],
                      # marker = dict(color = 'turquoise'),
                      text = np.round(100*shap_df.SHAP,2),
                      hovertemplate = '<b>%{y}</b>: %{x}',
                          textfont = dict(
                               family='Cadiz Semibold', 
                              size = 16))],
         layout=go.Layout(title = dict(text = 'Piirteiden globaalit merkitykset<br>Keskimääräiset |SHAP - arvot|',
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
                                                        xaxis = dict(title=dict(text = 'Keskimääräinen |SHAP - arvo|',
                                                                                font=dict(
                                                                                    size=18, 
                                                                                    family = 'Cadiz Semibold'
                                                                                    )),
                                                                     automargin=True,
                                                                     tickfont = dict(
                                                                         family = 'Cadiz Semibold', 
                                                                          size = 14
                                                                         )),
                                                        yaxis = dict(title=dict(text = 'Ennustepiirre',
                                                                               font=dict(
                                                                                    size=18, 
                                                                                   family = 'Cadiz Semibold'
                                                                                   )),
                                                                    tickformat = ' ',
                                                                     categoryorder='total ascending',
                                                                    automargin=True,
                                                                    tickfont = dict(
                                                                        family = 'Cadiz Semibold', 
                                                                         size = 14
                                                                        ))
                                                        )))

    
@callback(

    Output('local_shap_graph_div', 'children'),
    [Input('cut_off', 'value'),
     Input('shap_features_switch','on'),
     Input('local_shap_month_selection','value'),
     Input('local_shap_data','data')]
    
)
def update_local_shap_graph(cut_off, only_commodities, date, local_shap_data):
    
    if local_shap_data is None:
        raise PreventUpdate
    
    try:
        locale.setlocale(locale.LC_ALL, 'fi_FI')
    except:
        locale.setlocale(locale.LC_ALL, 'fi-FI')   
        
    
    
    local_shap_df = pd.DataFrame(local_shap_data)
    local_shap_df = local_shap_df.set_index(local_shap_df.columns[0])
    local_shap_df.index = pd.to_datetime(local_shap_df.index)
    
    base_value = local_shap_df['base'].values[0]
    local_shap_df = local_shap_df.drop('base',axis=1)
    
    date = pd.to_datetime(date)
    
    
    date_str = date.strftime('%B %Y')
    prev_date = date - pd.DateOffset(months=1)
    prev_str = prev_date.strftime('%B %Y') + ' työttömyysaste'
    
    
    
    dff = local_shap_df.loc[date,:].copy()
    
  
    
    dff.index  = dff.index.str.replace('month','Kuluva kuukausi').str.replace('prev',prev_str)
    
    feature_values = {f:data.loc[date,f] for f in data.columns if f not in ['Työttömyysaste', 'change','prev','month','Inflaatio']}
    feature_values[prev_str] = data.loc[date,'prev']
    feature_values['Kuluva kuukausi'] = data.loc[date,'month']
    
    feature_values_1 = {f:data.loc[date-pd.DateOffset(months=1),f] for f in data.columns if f not in ['Työttömyysaste', 'change','prev','month','Inflaatio']}
    feature_values_1[prev_str] = data.loc[date-pd.DateOffset(months=1),'prev']
    feature_values_1['Kuluva kuukausi'] = data.loc[date-pd.DateOffset(months=1),'month']
    
    differences = {f:feature_values[f]-feature_values_1[f] for f in feature_values.keys()}
    changes={}
    
    # How the unemployment rate changed last month?
    changes[prev_str] = data.loc[date-pd.DateOffset(months=1),'change']
    
    for d in differences.keys():
        if differences[d] >0:
            changes[d]='🔺'
        elif differences[d] <0:
            changes[d] = '🔽'
        else:
            changes[d] = '⇳'
    
    if only_commodities:
        dff = dff.loc[[i for i in dff.index if i not in ['Kuluva kuukausi', prev_str]]]
    
    
    dff = dff.sort_values(ascending = False)
    
   
    df = pd.Series(dff.iloc[cut_off+1:].copy().sum())
    
    
    
    # df.index = df.index.astype(str).str.replace('0', 'Muut {} piirrettä'.format(len(dff.iloc[cut_off+1:,:])))
    df.index = ['Muut {} piirrettä'.format(len(dff.iloc[cut_off+1:]))]
    
    
    dff = pd.concat([dff.head(cut_off).copy(),df])
    dff = dff.loc[dff.index != 'Muut 0 piirrettä']
    dff.index = dff.index.str.replace('_','')

    height = graph_height +200 + 10*len(dff)
    
    dff = np.round(dff*100,2)
   
    # dff = dff.sort_values()

    
    return html.Div([dcc.Graph(id = 'local_shap_graph',
                     config = config_plots,
                         figure = go.Figure(data=[go.Bar(y =['{} ({} {})'.format(i, feature_values[i],changes[i]) if i in feature_values.keys() else i for i in dff.index], 
                      x = dff.values,
                      orientation='h',
                      name = '',
                      # marker_color = ['cyan' if i not in ['Kuukausi',prev_str] else 'black' for i in dff.index],
                       marker = dict(color = list(map(set_color,dff.index,dff.values))),
                      
                      text = dff.values,
                      hovertemplate = ['<b>{}</b><br><b>  SHAP-arvo</b>: {}<br><b>  Tarkasteltavan kuukauden arvo</b>: {} {}<br><b>  Edeltävän kuukauden arvo</b>: {}'.format(i,dff.loc[i], feature_values[i],changes[i],round(feature_values_1[i],2)) if i in feature_values.keys() else '{}: {}'.format(i,dff.loc[i]) for i in dff.index],
                          textfont = dict(
                               family='Cadiz Semibold', 
                              size = 16))],
         layout=go.Layout(title = dict(text = 'Lokaalit piirteiden tärkeydet<br>SHAP arvot: '+date_str,
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
                                                        xaxis = dict(title=dict(text = 'SHAP - arvo',
                                                                                font=dict(
                                                                                    size=14, 
                                                                                    family = 'Cadiz Semibold'
                                                                                    )),
                                                                     automargin=True,
                                                                     # tickformat = ' ',
                                                                      # categoryorder='total descending',
                                                                     tickfont = dict(
                                                                         family = 'Cadiz Semibold', 
                                                                          size = 16
                                                                         )),
                                                        yaxis = dict(title=dict(text = 'Ennustepiirre: 🔺 = arvo kasvoi, 🔽 = arvo laski, ⇳ = arvo pysyi samana edelliseen kuukauteen nähden',
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
                     html.P('Ennuste ≈ Edeltävä ennustettu työttömyysaste + [ {} + SUM( SHAP-arvot ) ] / 100'.format(round(100*base_value,2)))
                     ])

@callback(

    Output('forecast_local_shap_graph_div', 'children'),
    [Input('forecast_cut_off', 'value'),
     Input('forecast_shap_features_switch','on'),
     Input('forecast_local_shap_month_selection','value'),
     Input('local_forecast_shap_data','data'),
     State('forecast_data','data')]
    
)
def update_local_forecast_shap_graph(cut_off, only_commodities, date, local_shap_data, forecast_data):
    
    if local_shap_data is None:
        raise PreventUpdate
    
    try:
        locale.setlocale(locale.LC_ALL, 'fi_FI')
    except:
        locale.setlocale(locale.LC_ALL, 'fi-FI')   
        
    
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
    prev_str = prev_date.strftime('%B %Y') + ' työttömyysaste'
    
       
    dff = local_shap_df.loc[date,:].copy()
    
    
    dff.index  = dff.index.str.replace('month','Kuluva kuukausi').str.replace('prev',prev_str)
    
    feature_values = {f:forecast_data.loc[date,f] for f in forecast_data.columns if f not in ['Työttömyysaste', 'change','prev','month','Inflaatio']}
    feature_values[prev_str] = forecast_data.loc[date,'prev']
    feature_values['Kuluva kuukausi'] = forecast_data.loc[date,'month']
    
    
    
    try:
        feature_values_1 = {f:forecast_data.loc[date-pd.DateOffset(months=1),f] for f in forecast_data.columns if f not in ['Työttömyysaste', 'change','prev','month','Inflaatio']}
        feature_values_1[prev_str] = forecast_data.loc[date-pd.DateOffset(months=1),'prev']
        feature_values_1['Kuluva kuukausi'] = forecast_data.loc[date-pd.DateOffset(months=1),'month']
    except:    
        feature_values_1 = {f:data.loc[date-pd.DateOffset(months=1),f] for f in data.columns if f not in ['Työttömyysaste', 'change','prev','month','Inflaatio']}
        feature_values_1[prev_str] = data.loc[date-pd.DateOffset(months=1),'prev']
        feature_values_1['Kuluva kuukausi'] = data.loc[date-pd.DateOffset(months=1),'month']
    
    differences = {f:feature_values[f]-feature_values_1[f] for f in feature_values.keys()}
    changes={}
    
    # How the unemployment rate changed last month?
    try:
        changes[prev_str] = forecast_data.loc[date-pd.DateOffset(months=1),'change']
    except:
        changes[prev_str] = data.loc[date-pd.DateOffset(months=1),'change']
    
    for d in differences.keys():
        if differences[d] >0:
            changes[d]='🔺'
        elif differences[d] <0:
            changes[d] = '🔽'
        else:
            changes[d] = '⇳'
    
    if only_commodities:
        dff = dff.loc[[i for i in dff.index if i not in ['Kuluva kuukausi', prev_str]]]
    
    
    dff = dff.sort_values(ascending = False)
    
   
    df = pd.Series(dff.iloc[cut_off+1:].copy().sum())
    
    
    
    # df.index = df.index.astype(str).str.replace('0', 'Muut {} piirrettä'.format(len(dff.iloc[cut_off+1:,:])))
    df.index = ['Muut {} piirrettä'.format(len(dff.iloc[cut_off+1:]))]
    
    
    dff = pd.concat([dff.head(cut_off).copy(),df])
    dff = dff.loc[dff.index != 'Muut 0 piirrettä']
    dff.index = dff.index.str.replace('_','')

    height = graph_height +200 + 10*len(dff)
    
    dff = np.round(dff*100,2)
   
    # dff = dff.sort_values()

    
    return html.Div([dcc.Graph(id = 'local_shap_graph',
                     config = config_plots,
                         figure = go.Figure(data=[go.Bar(y =['{} ({} {})'.format(i, round(feature_values[i],2),changes[i]) if i in feature_values.keys() else '{}: {}'.format(i,dff.loc[i]) for i in dff.index], 
                      x = dff.values,
                      orientation='h',
                      name = '',
                      # marker_color = ['cyan' if i not in ['Kuukausi',prev_str] else 'black' for i in dff.index],
                       marker = dict(color = list(map(set_color,dff.index,dff.values))),
                      
                      text = dff.values,
                      hovertemplate = ['<b>{}</b><br><b>  SHAP-arvo</b>: {}<br><b>  Tarkasteltavan kuukauden arvo</b>: {} {}<br><b>  Edeltävän kuukauden arvo</b>: {}'.format(i,dff.loc[i], round(feature_values[i],2),changes[i],round(feature_values_1[i],2)) if i in feature_values.keys() else '{}: {}'.format(i,dff.loc[i]) for i in dff.index],
                          textfont = dict(
                               family='Cadiz Semibold', 
                              size = 16))],
         layout=go.Layout(title = dict(text = 'Lokaalit piirteiden tärkeydet<br>SHAP arvot: '+date_str,
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
                                                        xaxis = dict(title=dict(text = 'SHAP - arvo',
                                                                                font=dict(
                                                                                    size=14, 
                                                                                    family = 'Cadiz Semibold'
                                                                                    )),
                                                                     automargin=True,
                                                                     # tickformat = ' ',
                                                                      # categoryorder='total descending',
                                                                     tickfont = dict(
                                                                         family = 'Cadiz Semibold', 
                                                                          size = 16
                                                                         )),
                                                        yaxis = dict(title=dict(text = 'Ennustepiirre: 🔺 = arvo kasvoi, 🔽 = arvo laski, ⇳ = arvo pysyi samana edelliseen kuukauteen nähden',
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
                     html.P('Ennuste ≈ Edeltävä ennustettu työttömyysaste + [ {} + SUM( SHAP-arvot ) ] / 100'.format(round(100*base_value,2)))
                     ])


@callback(
    Output("forecast_download", "data"),
    [Input("forecast_download_button", "n_clicks")],
    [State('forecast_data','data'),
      State('method_selection_results','data'),
      State('change_weights','data'),
      State('forecast_shap_data','data'),
      State('local_forecast_shap_data','data'),
     ]
    
  
)
def download_forecast_data(n_clicks, df, method_selection_results, weights_dict, shap_data, local_shap_data):
    
    
    
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
        
        
        shap_df = pd.DataFrame(shap_data)
        shap_df = shap_df.set_index(shap_df.columns[0])
        shap_df.index.name = 'Piirre'
        shap_df.SHAP = np.round(100*shap_df.SHAP,2)
        shap_df.index = shap_df.index.str.replace('_','')
        
        local_shap_df = pd.DataFrame(local_shap_data)
        local_shap_df = local_shap_df.set_index(local_shap_df.columns[0])
        local_shap_df.index = pd.to_datetime(local_shap_df.index)
        local_shap_df.index.name = 'Aika'
        local_shap_df = local_shap_df.rename(columns = {'month':'Kuukausi',
                                  'prev': 'Edellisen kuukauden työttömyysaste'})
        local_shap_df = local_shap_df.multiply(100, axis=1)
        local_shap_df.columns = local_shap_df.columns.str.replace('_','')
        local_shap_df.drop('base',axis=1,inplace=True)
        
        

  
        data_ = data.copy().rename(columns={'change':'Muutos (prosenttiykköä)',
                                      'prev':'Edellisen kuukauden työttömyys -% ',
                                      'month':'Kuukausi'})
        
        
        xlsx_io = io.BytesIO()
        writer = pd.ExcelWriter(xlsx_io, engine='xlsxwriter')
        
        
        
        data_.to_excel(writer, sheet_name= 'Data')
        df.to_excel(writer, sheet_name= 'Ennustedata')
        weights_df.to_excel(writer, sheet_name= 'Indeksimuutokset')
        hyperparam_df.to_excel(writer, sheet_name= 'Hyperparametrit')
        shap_df.to_excel(writer, sheet_name= 'Mallin piirteiden vaikuttavuus')
        local_shap_df.to_excel(writer, sheet_name= 'Vaikuttavuus kuukausittain')
        metadata.to_excel(writer, sheet_name= 'Metadata')
        
        workbook = writer.book
        workbook.set_properties(
        {
            "title": "Skewed Phillips",
            "subject": "Ennustetulokset",
            "author": "Tuomas Poukkula",
            # "company": "Gofore Ltd.",
            "keywords": "XAI, Predictive analytics",
            "comments": "Katso sovellus täältä: https://skewedphillips.herokuapp.com"
        }
        )
        
        

        writer.save()
        

        
        output = xlsx_io.getvalue()

        return dcc.send_bytes(output, 'Ennustedata {} hyödykkeellä '.format(len(features)).replace(' 1 hyödykkeellä', ' yhdellä hyödykkeellä')+datetime.now().strftime('%d_%m_%Y')+'.xlsx')
        
@callback(
    Output("test_download", "data"),
    [Input("test_download_button", "n_clicks"),
    State('test_data','data'),
    State('method_selection_results','data'),
    State('change_weights','data'),
    State('shap_data','data'),
    State('local_shap_data','data')
    ]
    
)
def download_test_data(n_clicks, 
                       df, 
                       method_selection_results, 
                        weights_dict, 
                       shap_data,
                       local_shap_data):
    
    if n_clicks > 0:
        
        df = pd.DataFrame(df).set_index('Aika').copy()
        df.index = pd.to_datetime(df.index)
        mape = df.mape.values[0]
        test_size = len(df)
        n_feat = df.n_feat.values[0]
        df.drop('n_feat',axis=1,inplace=True)
        df.drop('mape',axis=1,inplace=True)
        df = df.rename(columns = {'change':'Kuukausimuutos (prosenttiyksiköä)',
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
        
        shap_df = pd.DataFrame(shap_data)
        shap_df = shap_df.set_index(shap_df.columns[0])
        shap_df.index.name = 'Piirre'
        shap_df.SHAP = np.round(100*shap_df.SHAP,2)
        shap_df.index = shap_df.index.str.replace('_','')
        
        local_shap_df = pd.DataFrame(local_shap_data)
        local_shap_df = local_shap_df.set_index(local_shap_df.columns[0])
        local_shap_df.index = pd.to_datetime(local_shap_df.index)
        local_shap_df.index.name = 'Aika'
        local_shap_df = local_shap_df.rename(columns = {'month':'Kuukausi',
                                  'prev': 'Edellisen kuukauden työttömyysaste'})
        local_shap_df = local_shap_df.multiply(100, axis=1)
        local_shap_df.columns = local_shap_df.columns.str.replace('_','')
        local_shap_df.drop('base',axis=1,inplace=True)
        
        
        
        xlsx_io = io.BytesIO()
        writer = pd.ExcelWriter(xlsx_io, engine='xlsxwriter')
        
        
        df.to_excel(writer, sheet_name= 'Testidata')
        metadata.to_excel(writer, sheet_name= 'Metadata')
        hyperparam_df.to_excel(writer, sheet_name= 'Mallin hyperparametrit')
        shap_df.to_excel(writer, sheet_name= 'Mallin piirteiden vaikuttavuus')
        local_shap_df.to_excel(writer, sheet_name= 'Vaikuttavuus kuukausittain')
        
        
        workbook = writer.book
        workbook.set_properties(
        {
            "title": "Skewed Phillips",
            "subject": "Testitulokset",
            "author": "Tuomas Poukkula",
            # "company": "Gofore Ltd.",
            "keywords": "XAI, Predictive analytics",
            "comments": "Katso sovellus täältä: https://skewedphillips.herokuapp.com"
        }
        )

        writer.save()
        

        
        output = xlsx_io.getvalue()

        return dcc.send_bytes(output, 'Testitulokset {} hyödykkeellä '.format(len(features)).replace(' 1 hyödykkeellä', ' yhdellä hyödykkeellä')+datetime.now().strftime('%d_%m_%Y')+'.xlsx')


@callback(

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



@callback(

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

@callback(

    Output('ev_placeholder', 'style'),
    [Input('pca_switch', 'on')]
)    
def add_ev_slider(pca):
    
    return {False: {'margin' : '5px 5px 5px 5px', 'display':'none'},
           True: {'margin' : '5px 5px 5px 5px'}}[pca]

@callback(

    Output('ev_slider_update', 'children'),
    [Input('pca_switch', 'on'),
    Input('ev_slider', 'value')]

)
def update_ev_indicator(pca, explained_variance):
    
    return {False: [html.Div([html.P('Valitsit {} % säilytetyn variaation.'.format(int(100*explained_variance)),
                                                               style = p_center_style)
                                                       ], style = {'display':'none'}
                                                      )],
            True: [html.Div([html.P('Valitsit {} % säilytetyn variaation.'.format(int(100*explained_variance)),
                                                               style = p_center_style)
                                                       ]
                                                      )]}[pca]



@callback(
    Output('feature_selection','value'),
    [Input('select_all', 'on'),
    Input('feature_selection','options')]
)
def update_feature_list(on,options):
       
        
    if on:
        return [f['label'] for f in options]
    else:
        raise PreventUpdate
        
@callback(
    
    Output('select_all','on'),
    [Input('feature_selection','value'),
     State('feature_selection','options')]
    
)
def update_select_all_on(features,options):
    
    return len(features) == len(options)

@callback(
    [
     
     Output('select_all','label'),
     Output('select_all','disabled')
     ],
    [Input('select_all', 'on')]
)    
def update_switch(on):
    
    if on:
        return {'label':'Kaikki hyödykkeet on valittu. Voit poistaa hyödykkeitä listasta klikkaamalla rasteista.',
                       'style':{'text-align':'center', 'font-size':p_font_size,
                                #'font-family':'Cadiz Semibold'
                                }
                      },True
    

    else:
        return dict(label = 'Valitse kaikki',style = {'font-size':p_font_size, 
                                                      # #'fontFamily':'Cadiz Semibold'
                                                      }),False



@callback(

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
                           style = dict(fontSize=25,
                                        # fontFamily='Cadiz Semibold'
                                        )
                          )

@callback(
    Output('test_size_indicator','children'),
    [Input('test_slider','value')]
)
def update_test_size_indicator(value):
    
    return [html.Br(),html.P('Valitsit {} kuukautta testidataksi.'.format(value),
                             style = p_center_style)]

@callback(
    Output('forecast_slider_indicator','children'),
    [Input('forecast_slider','value')]
)
def update_forecast_size_indicator(value):
    
    return [html.Br(),html.P('Valitsit {} kuukauden ennusteen.'.format(value),
                             style = p_center_style)]




@callback(

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
            
            html.P('Tällä kuvaajalla voit tarkastella hyödykkeiden indeksikehitystä kuukausittain. Näin on helpompi arvioida paremmin millaista inflaatio-odotusta syöttää ennusteelle.',
                   style = p_style),
            html.H3('Valitse hyödyke',style = h3_style),
            dcc.Dropdown(id = 'timeseries_selection_dd',
                        options = [{'value':feature, 'label':feature} for feature in features],
                        value = [features[0]],
                        style = {
                            'font-size':"0.9rem", 
                            #'font-family':'Cadiz Book',
                            'color': 'black'},
                        multi = True)
            ]


@callback(

    Output('timeseries', 'children'),
    [Input('timeseries_selection_dd', 'value')]    
    
)
def update_time_series(values):
    

    
    traces = [go.Scatter(x = data.index, 
                         y = data[value],
                         showlegend=True,                         
                         name = ' '.join(value.split()[1:]),
                         hovertemplate = '%{x}'+'<br>%{y}',
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
                                                                      # xanchor='center',
                                                                      # yanchor='top',
                                                                     font=dict(
                                                              size=14,
                                                              family='Cadiz Book'
                                                             )),
                                                         hoverlabel=dict(font=dict(
                                                              family='Cadiz Book',
                                                             size=18)),
                                                         xaxis = dict(title=dict(text = 'Aika',
                                                                                 font=dict(
                                                                                     size=18, 
                                                                                     family = 'Cadiz Semibold'
                                                                                     )),
                                                                      automargin=True,
                                                                      tickfont = dict(
                                                                          family = 'Cadiz Semibold', 
                                                                           size = 16
                                                                          )),
                                                         yaxis = dict(title=dict(text = 'Pisteluku (perusvuosi = 2010)',
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
                     config = config_plots
                     )])




@callback(
    
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
                   style = dict(fontSize=25,
                                # fontFamily='Cadiz Semibold'
                                )
                   
                   ),
                html.Br(),
                html.Div(id = 'forecast_download_button_div',style={'textAlign':'center'})]





@callback(

    Output('slider_prompt_div','children'),
    [Input('slider', 'value'),
     State('averaging', 'on')]    
    
)
def update_slider_prompt(value, averaging):
    
        
    if averaging:
    
        return [html.Br(),html.P('Valitsit {} viimeisen kuukauden keskiarvot.'.format(value),
                      style = p_center_style),
                html.Br(),
                html.P('Voit vielä säätä yksittäisiä muutosarvoja laatikoihin kirjoittamalla tai tietokoneella työskenneltäessä laatioiden oikealla olevista nuolista.',
                       style = p_center_style)]
    else:
        return [html.Br(),html.P('Valitsit {} % keskimääräisen kuukausimuutoksen.'.format(value),
                      style = p_center_style),
                html.Br(),
                html.P('Voit vielä säätä yksittäisiä muutosarvoja laatikoihin kirjoittamalla tai tietokoneella työskenneltäessä laatioiden oikealla olevista nuolista.',
                       style = p_center_style)]
        
 

@callback(

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
            
            html.P('Tällä kuvaajalla voit tarkastella valitun hyödykkeen hintaindeksin ja työttömyysasteen tai kuukausimuutoksen välistä suhdetta ja korrelaatiota. Teoriassa hyvä ennustepiirre korreloi vahvasti ennustettavan muuttujan kanssa. '
                   'Visualisoinnin parantamiseksi, trendiviiva näytetään kuvaajassa ainoastaan kun yksi vain hyödyke on valittuna.',
                   style = p_style),
            html.P("(Jatkuu kuvaajan jälkeen)",style={
                        'font-style':'italic',
                        'font-size':p_font_size,
                       'text-align':'center'}
                ),
        html.H3('Valitse hyödyke', style = h3_style),
        dcc.Dropdown(id = 'corr_feature',
                        multi = True,
                        # clearable=False,
                        options = [{'value':feature, 'label':feature} for feature in features],
                        value = [features[0]],
                        style = {'font-size':"0.9rem", 
                                 #'font-family':'Cadiz Book'
                                 },
                        placeholder = 'Valitse hyödyke')
        ]
        )

@callback(

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
                html.P('Tällä kuvaajalla voit tarkastella hyödykkeiden keskinäisiä suhteita ja korrelaatioita. Mikäli korrelaatio kahden hyödykkeen välillä on vahva, voi ennuste parantua toisen poistamalla ennustepiirteistä.',
                       style = p_style),
                html.P("(Jatkuu kuvaajan jälkeen)",style={
                            'font-style':'italic',
                            'font-size':p_font_size,
                           'text-align':'center'}
                    ),
        
        dbc.Row(justify = 'center',children=[
            dbc.Col([
                html.H3('Valitse hyödyke',style=h3_style),
                dcc.Dropdown(id = 'f_corr1',
                                multi = False,
                                options = [{'value':feature, 'label':feature} for feature in features],
                                value = features[0],
                                style = {'font-size':"0.9rem", 
                                         #'font-family':'Cadiz Book'
                                         },
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
                                style = {'font-size':"0.9rem", 
                                         #'font-family':'Cadiz Book'
                                         },
                                placeholder = 'Valitse hyödyke')
            ],xs =12, sm=12, md=12, lg=6, xl=6)
        ])
        ])



@callback(

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
                                               size=17
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
                                 family='Cadiz Book'
                                ),
                                          orientation='h'),
                            hoverlabel = dict(
                                 font_size = 18, 
                                 font_family = 'Cadiz Book'
                                ),
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






@callback(
    
    Output('commodity_unemployment_div','children'),
    [Input('corr_feature','value'),
     Input('eda_y_axis','value')]
    
)
def update_commodity_unemployment_graph(values, label):
    
    
    symbols = [
                'circle',
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
                  'bowtie'
                
                 ]
    
    label_str = {'Työttömyysaste': 'Työttömyysaste (%)',
                 'change': 'Työttömyysasteen kuukausimuutos (%-yksikköä)'}[label]     
            
    traces = [go.Scatter(x = data[value], 
                         y = data[label], 
                         mode = 'markers',
                         name = ' '.join(value.split()[1:]).replace(',',',<br>')+' ({})'.format(round(sorted(data[[label, value]].corr()[value].values)[0],2)),
                         showlegend=True,
                         marker = dict(size=10),
                         marker_symbol = random.choice(symbols),
                         hovertemplate = "<b>{}</b>:".format(value)+" %{x}"+"<br><b>"+label_str+"</b>: %{y}"+"<br>(Korrelaatio: {:.2f})".format(sorted(data[[label, value]].corr()[value].values)[0])) for value in values]
    text='Valitut hyödykkeet vs.<br>'+label_str
    if len(values)==1:
        data_ = data[(data[label].notna())].copy()
        
        for value in values:
        
            a, b = np.polyfit(np.log(data_[value]), data_[label], 1)
    
            y = a * np.log(data_[value].values) +b 
            
    
            df = data_[[value]].copy()
            df['log_inflation'] = y
            df = df.sort_values(by = 'log_inflation')
            traces.append(go.Scatter(x=df[value], 
                                      y =df['log_inflation'],
                                      showlegend=True,
                                      name = 'Logaritminen<br>trendiviiva',
                                      line = dict(width=5),
                                      hovertemplate=[]))
        text = f"{' '.join(values[0].split()[1:]).capitalize()} vs.<br>"+label_str
    
    
    return [dcc.Graph(figure = go.Figure(data = traces,
          layout = go.Layout(title = dict(text = text, 
                                          x=.5, 
                                          font=dict(
                                              family='Cadiz Semibold',
                                               size=18
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
          ),config = config_plots)]

@callback(
    [
     
     Output('feature_selection', 'options'),
     Output('sorting', 'label'),
     # Output('feature_selection', 'value')
     
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
     Input('fifth_class', 'n_clicks'),
     Input('example_basket_1', 'n_clicks'),
     Input('example_basket_2', 'n_clicks'),
     Input('example_basket_3', 'n_clicks')
    ]
)
def update_selections(*args):
    
    ctx = callback_context
    
    
    if not ctx.triggered:
        return feature_options, "Aakkosjärjestyksessä"#,[f['value'] for f in corr_abs_asc_options[:4]]
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if button_id == 'alphabet':
        return feature_options, "Aakkosjärjestyksessä"#,[f['value'] for f in feature_options[:4]]
    elif button_id == 'corr_desc':
        return corr_desc_options, "Korrelaatio (laskeva)"#,[f['value'] for f in corr_desc_options[:4]]
    elif button_id == 'corr_asc':
        return corr_asc_options, "Korrelaatio (nouseva)"#,[f['value'] for f in corr_asc_options[:4]]
    elif button_id == 'corr_abs_desc':
        return corr_abs_desc_options, "Absoluuttinen korrelaatio (laskeva)"#,[f['value'] for f in corr_abs_desc_options[:4]]
    elif button_id == 'corr_abs_asc':
        return corr_abs_asc_options, "Absoluuttinen korrelaatio (nouseva)"#,[f['value'] for f in corr_abs_asc_options[:4]]
    elif button_id == 'main_class':
        return main_class_options, "Pääluokittain"#,[f['value'] for f in main_class_options[:4]]
    elif button_id == 'second_class':
        return second_class_options, "2. luokka"#,[f['value'] for f in second_class_options[:4]]
    elif button_id == 'third_class':
        return third_class_options, "3. luokka"#,[f['value'] for f in third_class_options[:4]]
    elif button_id == 'fourth_class':
        return fourth_class_options, "4. luokka"#,[f['value'] for f in fourth_class_options[:4]]
    elif button_id == 'fifth_class':
        return fifth_class_options, "5. luokka"#,[f['value'] for f in fifth_class_options[:4]]
    elif button_id == 'example_basket_1':
        return example_basket_1_options, "Esimerkkikori 1"#,[f['value'] for f in fifth_class_options[:4]]
    elif button_id == 'example_basket_2':
        return example_basket_2_options, "Esimerkkikori 2"#,[f['value'] for f in fifth_class_options[:4]]
    else:
        return example_basket_3_options, "Esimerkkikori 3"#,[f['value'] for f in fifth_class_options[:4]]
    
    
