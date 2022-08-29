# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 11:57:56 2022

@author: tuomas.poukkula
"""
import dash 

dash.register_page(__name__,
                   title = 'Skewed Phillips',
                   name = 'Skewed Phillips',
                   image='en.png',
                   description ="Forecasting Finland's Unemployment Rate with Consumer Price Changes")
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
from sklearn.decomposition import PCA
import dash_daq
from xgboost import XGBRegressor
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
    locale.setlocale(locale.LC_ALL, 'en_US')
except:
    locale.setlocale(locale.LC_ALL, 'en-US')

in_dev = True

MODELS_en = {
    
    
        'Decision Tree':{'model':DecisionTreeRegressor,
                           'doc': 'https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html',
                           'video':"https://www.youtube.com/embed/UhY5vPfQIrA",
                           'explainer':shap.TreeExplainer,
                           'constant_hyperparameters': {
                                                        # 'n_jobs':-1,
                                                        'random_state':42}
                           },
    
        'Random Forest': {'model':RandomForestRegressor,
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
        # 'K Nearest Neighbors':{'model':KNeighborsRegressor,
        #                        'doc':'https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html',
        #                         'video':'https://www.youtube.com/embed/jw5LhTWUoG4?list=PLRZZr7RFUUmXfON6dvwtkaaqf9oV_C1LF',
        #                         'explainer':shap.KernelExplainer,
        #                        'constant_hyperparameters': {
        #                                                    'n_jobs':-1
        #                                                     }
        #                        },
        # 'Support Vector Machine':{'model':SVR,
        #                    'doc':'https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html',
        #                    'video':"https://www.youtube.com/embed/_YPScrckx28",
        #                     'explainer':shap.KernelExplainer,
        #                        'constant_hyperparameters': {
        #                                                     }
        #                        },
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



config_plots_en = {'locale':'en',
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
            dbc.Col(dbc.NavLink(DashIconify(icon="logos:twitter"), href="https://twitter.com/TuomasPoukkula",external_link=True, target='_blank',className="btn btn-link btn-floating btn-lg text-dark m-1"),className="mb-4",xl=1,lg=1,md=4,sm=4,xs=4   ),
            dbc.Col(dbc.NavLink(DashIconify(icon="logos:linkedin"), href="https://www.linkedin.com/in/tuomaspoukkula/",external_link=True, target='_blank',className="btn btn-link btn-floating btn-lg text-dark m-1"),className="mb-4",xl=1,lg=1,md=4,sm=4,xs=4  )
            
            
            
            ],className ="d-flex justify-content-center align-items-center", justify='center',align='center')
    
    
    ],className ='card text-white bg-secondary mb-3')


def en_set_color(x,y):
    
    
    if 'nemployment' in x or x=='Current Month':
        return 'black'
    elif y < 0:
        
        return '#32CD32'
    elif y >= 0:
        return '#E34234'


def en_get_unemployment():
    

  url = 'https://statfin.stat.fi:443/PxWeb/api/v1/en/StatFin/tyti/statfin_tyti_pxt_135z.px'
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

def en_get_inflation():

  url = 'https://statfin.stat.fi:443/PxWeb/api/v1/en/StatFin/khi/statfin_khi_pxt_11xd.px'
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
  # df['name'] = [' '.join(c.split()[1:]) for c in df.Hyödyke]
  # df =df .reset_index()
  # df =df.drop_duplicates(subset=['Aika','name'],keep='first')
  # df = df.set_index('Aika')
  # df = df.drop('name',axis=1)

  return df

def en_get_inflation_percentage():

  url = 'https://pxweb2.stat.fi:443/PxWeb/api/v1/en/StatFin/khi/statfin_khi_pxt_122p.px'
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
  



def en_get_data():
  unemployment_df = en_get_unemployment()
  inflation_df = en_get_inflation()

  inflation_df = pd.pivot_table(inflation_df.reset_index(), columns = 'Hyödyke', index = 'Aika' )
  inflation_df.columns = [c[-1] for c in inflation_df.columns]

  data = pd.merge(left = unemployment_df.drop('Tiedot',axis=1).reset_index(), right = inflation_df.reset_index(), how = 'outer', on = 'Aika').set_index('Aika')
  data.Työttömyysaste = data.Työttömyysaste.fillna(-1)
  data = data.dropna(axis=1)

  inflation_percentage_df = en_get_inflation_percentage()

  data = pd.merge(left = data.reset_index(), right = inflation_percentage_df.reset_index(), how = 'inner', on = 'Aika').set_index('Aika').sort_index()

  data.Työttömyysaste = data.Työttömyysaste.replace(-1, np.nan)

  data['prev'] = data['Työttömyysaste'].shift(1)

  data['month'] = data.index.month
  data['change'] = data.Työttömyysaste - data.prev
  
  
  
  # data = data.T.drop_duplicates().T

  return data


data_en = en_get_data()


def en_draw_phillips_curve():
    
  try:
      locale.setlocale(locale.LC_ALL, 'en_US')
  except:
      locale.setlocale(locale.LC_ALL, 'en-US')  
     
  data_ = data_en[(data_en.Työttömyysaste.notna())&(data_en.Inflaatio.notna())].copy()
    
  max_date = data_.index.values[-1]
  max_date_str = data_.index.strftime('%B %Y').values[-1]
  
  
  a, b = np.polyfit(np.log(data_.Työttömyysaste), data_.Inflaatio, 1)

  y = a * np.log(data_.Työttömyysaste) +b 

  df = data_.copy()
  df['log_inflation'] = y
  df = df.sort_values(by = 'log_inflation')
  
  

  hovertemplate = ['<b>{}</b><br>Unemployment rate: {} %<br>Inflation: {} %'.format(df.index[i].strftime('%B %Y'), df.Työttömyysaste.values[i], df.Inflaatio.values[i]) for i in range(len(df))]

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
                            marker = dict(color = 'red', symbol='diamond', 
                                           size = 10
                                          )
                            ),
                go.Scatter(x = df.Työttömyysaste, 
                            y = df['log_inflation'], 
                            name = 'Logarithmic<br>trendline', 
                            mode = 'lines',
                            line = dict(width=5),
                            showlegend=True,
                            hovertemplate=[], 
                            marker = dict(color = 'blue'))
                  ],
            layout = go.Layout(
                               xaxis=dict(showspikes=True,
                                          title = dict(text='Unemployment rate (%)', font=dict(size=18, 
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
                              
                               title = dict(text = 'Unemployment rate vs.<br>Inflation<br>{} - {}<br>'.format(df.index.min().strftime('%B %Y'),df.index.max().strftime('%B %Y')),
                                            x=.5,
                                            font=dict(
                                                size=20,
                                                 family = 'Cadiz Semibold'
                                                ))
                              )
            )
            
def en_get_shap_values(model, explainer, X_train, X_test):


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
        

def en_get_param_options(model_name):
    
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
        
  model = MODELS_en[model_name]['model']

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



def en_plot_test_results(df, chart_type = 'lines+bars'):
    
    
    try:
        locale.setlocale(locale.LC_ALL, 'en_US')
    except:
        locale.setlocale(locale.LC_ALL, 'en-US')
    
    hovertemplate = ['<b>{}</b>:<br>True: {}<br>Predicted: {}'.format(df.index[i].strftime('%B %Y'),df.Työttömyysaste.values[i], df.Ennuste.values[i]) for i in range(len(df))]
    
    if chart_type == 'lines+bars':
    
        return go.Figure(data=[go.Scatter(x=df.index.strftime('%B %Y'), 
                               y = df.Työttömyysaste, 
                               name = 'True',
                               showlegend=True, 
                               mode = 'lines+markers+text',
                               text=[str(round(c,2))+' %' for c in df.Työttömyysaste], 
                               textposition='top center',
                               hovertemplate = hovertemplate,
                               textfont = dict(
                                    family='Cadiz Semibold', 
                                   size = 16,color='green'), 
                               marker = dict(color='#008000',size=12),
                               line = dict(width=5)),
                    
                    go.Bar(x=df.index.strftime('%B %Y'), 
                           y = df.Ennuste, 
                           name = 'Predicted',
                           showlegend=True, 
                           marker = dict(color='red'), 
                           text=[str(round(c,2))+' %' for c in df.Ennuste], 
                           textposition='inside',
                           hovertemplate = hovertemplate,
                           textfont = dict(
                                family='Cadiz Semibold', 
                               size = 16)
                           )
                    ],layout=go.Layout(xaxis = dict(title = dict(text='Time',font=dict(size=20, 
                                                                                       family = 'Cadiz Semibold'
                                                                                       )),
                                                    tickfont = dict(
                                                        family = 'Cadiz Semibold', 
                                                        size = 16),
                                                    automargin=True
                                                    ),
                                       yaxis = dict(title = dict(text='Unemployment rate (%)',
                                                                 font=dict(
                                                                      family='Cadiz Semibold',
                                                                     size=16)),
                                                    tickfont = dict(
                                                        family = 'Cadiz Semibold', 
                                                        size = 18),
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
                                       title = dict(text = 'Unemployment Rate Forecast<br>per Month',
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
                                name = 'True',
                                showlegend=True, 
                                mode = 'lines+markers+text',
                                text=[str(round(c,2))+' %' for c in df.Työttömyysaste], 
                                textposition='top center',
                                hovertemplate = hovertemplate,
                                textfont = dict(
                                     family='Cadiz Semibold', 
                                    size = 16,color='green'), 
                                marker = dict(color='#008000',size=10),
                                line = dict(width=2)),
                    
                    go.Scatter(x=df.index, 
                            y = df.Ennuste, 
                            name = 'Predicted',
                            showlegend=True,
                            mode = 'lines+markers+text',
                            marker = dict(color='red',size=10), 
                            text=[str(round(c,2))+' %' for c in df.Ennuste], 
                            # textposition='inside',
                            hovertemplate = hovertemplate,
                            line = dict(width=2),
                            textfont = dict(
                                 family='Cadiz Semibold', 
                                size = 16,color='red'))
                    ],layout=go.Layout(xaxis = dict(title = dict(text='Time',font=dict(size=16, 
                                                                                       family = 'Cadiz Semibold'
                                                                                       )),
                                                    tickfont = dict(
                                                        family = 'Cadiz Semibold', 
                                                        size = 16),
                                                    automargin=True,
                                                    ),
                                        yaxis = dict(title = dict(text='Unemployment Rate (%)',font=dict(
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
                                        title = dict(text = 'Unemployment Rate Forecast<br>per Month',
                                                    x=.5,
                                                    font=dict(
                                                         family='Cadiz Semibold',
                                                        size=20)
                                                    )
                                        ))
                                                    

    else:
        return go.Figure(data=[go.Bar(x=df.index.strftime('%B %Y'), 
                                    y = df.Työttömyysaste, 
                                    name = 'True',
                           showlegend=True, 
                           marker = dict(color='green'), 
                           text=[str(round(c,2))+' %' for c in df.Työttömyysaste], 
                           textposition='inside',
                           hovertemplate = hovertemplate,
                           textfont = dict(
                                family='Cadiz Semibold', 
                               size = 16)
                                    ),
                        
                        go.Bar(x=df.index.strftime('%B %Y'), 
                                y = df.Ennuste, 
                                name = 'Predicted',
                           showlegend=True, 
                           marker = dict(color='red'), 
                           text=[str(round(c,2))+' %' for c in df.Ennuste], 
                           textposition='inside',
                           hovertemplate = hovertemplate,
                           textfont = dict(
                                family='Cadiz Semibold', 
                               size = 16)
                                )
                        ],layout=go.Layout(xaxis = dict(title = dict(text='Time',font=dict(size=20, 
                                                                                           family = 'Cadiz Semibold'
                                                                                           )),
                                                        tickfont = dict(
                                                            family = 'Cadiz Semibold', 
                                                            size = 16),
                                                        automargin=True
                                                        ),
                                            yaxis = dict(title = dict(text='Unemployment Rate (%)',font=dict(
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
                                            title = dict(text = 'Unemployment Rate Forecast<br>per Month',
                                                        x=.5,
                                                        font=dict(
                                                             family='Cadiz Semibold',
                                                            size=20)
                                                        )
                                            )
                                                        )                                                   

                                                    
                                                    
                                                    
def en_plot_forecast_data(df, chart_type):
    
    try:
        locale.setlocale(locale.LC_ALL, 'en_US')
    except:
        locale.setlocale(locale.LC_ALL, 'en-US')
    
    hover_true = ['<b>{}</b><br>Unemployment Rate: {} %'.format(data_en.index[i].strftime('%B %Y'), data_en.Työttömyysaste.values[i]) for i in range(len(data_en))]
    hover_pred = ['<b>{}</b><br>Unemployment Rate: {} %'.format(df.index[i].strftime('%B %Y'), round(df.Työttömyysaste.values[i],1)) for i in range(len(df))]
    

    if chart_type == 'lines':
    
    
        return go.Figure(data=[go.Scatter(x=data_en.index, 
                                          y = data_en.Työttömyysaste, 
                                          name = 'True',
                                          showlegend=True,
                                          mode="lines", 
                                          hovertemplate =  hover_true,##'<b>%{x}</b>: %{y}%',
                                          marker = dict(color='green')),
                    go.Scatter(x=df.index, 
                               y = np.round(df.Työttömyysaste,1), 
                               name = 'Predicted',
                               showlegend=True,
                               mode="lines", 
                               hovertemplate = hover_pred,#'<b>%{x}</b>: %{y}%',
                               marker = dict(color='red'))
                    ],layout=go.Layout(xaxis = dict(title = dict(text = 'Time',font=dict(
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
                         label="1y",
                         step="year",
                         stepmode="backward"),
                    dict(count=3,
                         label="3y",
                         step="year",
                         stepmode="backward"),
                    dict(count=5,
                         label="5y",
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
                                       yaxis = dict(title=dict(text = 'Unemployment Rate (%)',
                                                     font=dict(
                                                          size=16, 
                                                         family = 'Cadiz Semibold'
                                                         )),
                                                    automargin=True,
                                                     tickfont = dict(
                                                         family = 'Cadiz Book', 
                                                                      size = 16
                                                                     )),
                                       title = dict(text = 'Unemployment Rate and Forecast per Month<br>{} - {}'.format(data_en.index.strftime('%B %Y').values[0],df.index.strftime('%B %Y').values[-1]),
                                                    x=.5,
                                                    font=dict(
                                                         family='Cadiz Semibold',
                                                               size=16
                                                              )),
    
                                       ))
    elif chart_type == 'area':
    
    
        return go.Figure(data=[go.Scatter(x=data_en.index, 
                                          y = data_en.Työttömyysaste, 
                                          name = 'True',
                                          showlegend=True,
                                          fill = 'tozeroy',
                                          mode="lines", 
                                          hovertemplate =  hover_true,##'<b>%{x}</b>: %{y}%',
                                          marker = dict(color='green')),
                    go.Scatter(x=df.index, 
                               y = np.round(df.Työttömyysaste,1), 
                               name = 'Predicted',
                               showlegend=True,
                               fill = 'tozeroy',
                               mode="lines", 
                               hovertemplate = hover_pred,#'<b>%{x}</b>: %{y}%',
                               marker = dict(color='red'))
                    ],layout=go.Layout(xaxis = dict(title = dict(text = 'Time',font=dict(
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
                         label="1y",
                         step="year",
                         stepmode="backward"),
                    dict(count=3,
                         label="3y",
                         step="year",
                         stepmode="backward"),
                    dict(count=5,
                         label="5y",
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
                                       yaxis = dict(title=dict(text = 'Unemployment Rate (%)',
                                                     font=dict(
                                                          size=16, 
                                                         family = 'Cadiz Semibold'
                                                         )),
                                                    rangemode='tozero',
                                                    automargin=True,
                                                     tickfont = dict(
                                                         family = 'Cadiz Book', 
                                                                      size = 16
                                                                     )),
                                       title = dict(text = 'Unemployment Rate and Forecast per Month<br>{} - {}'.format(data_en.index.strftime('%B %Y').values[0],df.index.strftime('%B %Y').values[-1]),
                                                    x=.5,
                                                    font=dict(
                                                         family='Cadiz Semibold',
                                                               size=16
                                                              )),
    
                                       ))


    else:
        
        
      
        return go.Figure(data=[go.Bar(x=data_en.index, 
                                          y = data_en.Työttömyysaste, 
                                          name = 'True',
                                          showlegend=True,
                                          # mode="lines", 
                                          hovertemplate = hover_true,#'<b>%{x}</b>: %{y}%',
                                          marker = dict(color='green')),
                    go.Bar(x=df.index, 
                               y = np.round(df.Työttömyysaste,1), 
                               name = 'Predicted',
                               showlegend=True,
                               # mode="lines", 
                               hovertemplate = hover_pred,#'<b>%{x}</b>: %{y}%',
                               marker = dict(color='red'))
                    ],layout=go.Layout(xaxis = dict(title = dict(text = 'Time',
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
                          label="1y",
                          step="year",
                          stepmode="backward"),
                    dict(count=3,
                          label="3y",
                          step="year",
                          stepmode="backward"),
                    dict(count=5,
                          label="5y",
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
                                       yaxis = dict(title=dict(text = 'Unemployment Rate (%)',
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
                                       title = dict(text = 'Unemployment Rate and Forecast per month<br>{} - {}'.format(data_en.index.strftime('%B %Y').values[0],df.index.strftime('%B %Y').values[-1]),
                                                    x=.5,
                                                    font=dict(
                                                         family='Cadiz Semibold',
                                                               size=16
                                                              )),
    
                                       )) 
                                                    
                                                    


def en_test(model, features, test_size, explainer, use_pca = False, n_components=.99):

  feat = features.copy()
  feat.append('prev')
  feat.append('month')
  
  cols = feat
  
  data_ = data_en.iloc[1:,:].copy()
  
  
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
  
  

  shap_df, local_shap_df = en_get_shap_values(model, explainer, X_train = pd.DataFrame(X, columns = cols), X_test = pd.concat(scaled_features))

  result = pd.concat(results)
  result['n_feat'] = n_feat
  result.Ennuste = np.round(result.Ennuste,1)
  result['mape'] = mean_absolute_percentage_error(result.Työttömyysaste, result.Ennuste)
  

  
  result.index.name ='Aika'
    
  result = result[['Työttömyysaste', 'Ennuste', 'change', 'Ennustettu muutos', 'prev','n_feat','mape','month']+features]
  

  return result, shap_df, local_shap_df                                                    

                                                

def en_predict(model,explainer, features, feature_changes, length, use_pca = False, n_components=.99):
  
  df = data_en.copy()
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
  
  if data_en.Työttömyysaste.isna().sum() > 0:
      last_row = data_en.iloc[-1:,:].copy()
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
  shap_df, local_shap_df = en_get_shap_values(model, explainer, X_train = pd.DataFrame(X, columns = cols), X_test = pd.concat(scaled_features_shap))
  
  local_shap_df.index = result.index

  return result, shap_df, local_shap_df



def en_apply_average(features, length = 4):

  return 100 * data_en[features].pct_change().iloc[-length:, :].mean()





# Viimeiset neljä saraketta ovat prev, month, change ja inflaatio.
 
correlations_desc_en = data_en[data_en.columns[:-4]].corr()['Työttömyysaste'].iloc[1:].sort_values(ascending=False)
correlations_asc_en = data_en[data_en.columns[:-4]].corr()['Työttömyysaste'].iloc[1:].sort_values(ascending=True)
correlations_abs_desc_en = data_en[data_en.columns[:-4]].corr()['Työttömyysaste'].iloc[1:].abs().sort_values(ascending=False)
correlations_abs_asc_en = data_en[data_en.columns[:-4]].corr()['Työttömyysaste'].iloc[1:].abs().sort_values(ascending=True)

main_classes_en = sorted([c for c in data_en.columns[:-4] if len(c.split()[0])==2])
second_classes_en = sorted([c for c in data_en.columns[:-4] if c.split()[0].count('.')==1])
third_classes_en = sorted([c for c in data_en.columns[:-4] if c.split()[0].count('.')==2])
fourth_classes_en = sorted([c for c in data_en.columns[:-4] if c.split()[0].count('.')==3])
fifth_classes_en = sorted([c for c in data_en.columns[:-4] if c.split()[0].count('.')==4])

feature_options_en = [{'label':c, 'value':c} for c in data_en.columns[1:-4]]
corr_desc_options_en = [{'label':c, 'value':c} for c in correlations_desc_en.index]
corr_asc_options_en = [{'label':c, 'value':c} for c in correlations_asc_en.index]
corr_abs_desc_options_en = [{'label':c, 'value':c} for c in correlations_abs_desc_en.index]
corr_abs_asc_options_en = [{'label':c, 'value':c} for c in correlations_abs_asc_en.index]
main_class_options_en = [{'label':c, 'value':c} for c in main_classes_en]
second_class_options_en = [{'label':c, 'value':c} for c in second_classes_en]
third_class_options_en = [{'label':c, 'value':c} for c in third_classes_en]
fourth_class_options_en = [{'label':c, 'value':c} for c in fourth_classes_en]
fifth_class_options_en = [{'label':c, 'value':c} for c in fifth_classes_en]



initial_options_en = feature_options_en
initial_features_en = [c for c in ['01.1.3 Fish and seafood',
 '01.1.3.3 Dried, smoked or salted fish and seafood',
 '01.1.6.3.1 Frozen fruit and berries',
 '01.1.9.3.1 Baby food',
 '01.1.9.5.1 Other food products n.e.c.',
 '02.1.2.1.1 Wine from grapes',
 '03.1 Clothing',
 '03.1.2.2 Garments for women',
 "03.1.2.2.1 Women's overcoats and jackets",
 '03.1.2.4 Garments for infants (0 to 2 years)',
 '03.2.1.3 Footwear for children',
 '05.2.0.2.3 Sheets, pillowcases and quilt covers',
 '05.6.1.1.1 Detergents',
 '06.1.3 Therapeutic appliances and equipment',
 '06.1.3.1 Spectacles and contact lenses',
 '07.3.1.1.1 Domestic rail transport',
 '09 RECREATION AND CULTURE',
 '09.4.2.3.3 Subscription to cable TV  and pay-TV',
 '09.6.0 Package holidays',
 '12.1.3.3.3 Body, hand and hair lotions'] if c in data_en.columns]


def layout():
    
    return html.Div([dbc.Container(fluid=True, className = 'dbc', children=[
        html.Br(),        
        dbc.Row(
            [
                
                dbc.Col([
                    
                    html.Br(),   
                    html.H1('Skewed Phillips',
                             style=h1_style
                            ),
                  
                    html.H2("Forecasting Finland's Unemployment Rate with Consumer Price Changes",
                            style=h2_style),
                    
                    html.P('Select the desired tab by clicking on the titles below. ' 
                           'The buttons in the upper left corner show quick help '
                           'and you can also change the color scheme of the page.',
                           style = p_center_style)
                    ],xs =12, sm=12, md=12, lg=9, xl=9)
        ], justify='center'),
        html.Br(),
        
        html.Div(id = 'hidden_store_div_en',
                 children = [
                    
                    dcc.Store(id = 'features_values_en',data={f:0.0 for f in initial_features_en}),
                    dcc.Store(id = 'change_weights_en'), 
                    dcc.Store(id = 'method_selection_results_en'),
                    dcc.Store(id ='shap_data_en'),
                    dcc.Store(id ='local_shap_data_en'),
                    dcc.Store(id = 'test_data_en'),
                    dcc.Store(id = 'forecast_data_en'),                    
                    dcc.Store(id = 'forecast_shap_data_en'),
                    dcc.Store(id = 'local_forecast_shap_data_en'),
                    dcc.Download(id='forecast_download_en'),
                    dcc.Download(id='test_download_en')
        ]),
        
        dbc.Tabs(id ='tabs_en',
                 children = [
            
            
            
            dbc.Tab(label='Introduction and Instructions',
                    tab_id = 'ohje_en',
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
                                  html.H3('Introduction',style=h3_style
                                          ),
                                  html.P('Rising prices affect the lives of many Finnish consumers, and interest rates are expected to rise as central banks feel pressure to contain inflation. People with mortgages are facing more difficult times as interest rates rise. All in all, it would seem that there is nothing good about inflation. But is this really the case?',
                                        style = p_style),
                                  html.P("Inflation also has a silver lining, which is a fall in unemployment in the short term. This so-called The Phillips curve is an empirical observation made in the 1950s by an economist Alban William Phillips. The observation states that there is a conflict between inflation and unemployment in the short term. This idea is presented in the graph below, which describes inflation and unemployment rate at the same time in Finland. The descending logarithmic trend line corresponds to Phillips's observation.",
                                        style = p_style),
                                  html.P("(Continues after the chart)",style={
                                              'font-style':'italic',
                                              'font-size':p_font_size,
                                             'text-align':'center'}
                                      ),
                                
                                  html.H3('Monthly Phillips Curve in Finland', 
                                          style=h3_style),
                                  html.H4('Source: Statistics Finland', 
                                          style=h4_style)
                                  
                                  ])
                                  ]
                                ),

                                  dbc.Row([
                                      dbc.Col([
                                             
                                               html.Div(
                                                    [dcc.Graph(id = 'phillips_en',
                                                                figure= en_draw_phillips_curve(),
                                                                config = config_plots_en
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
                                          html.P('The graph above shows the unemployment rate and inflation at different times with a scatter pattern. It also identifies the latest date for both inflation and unemployment. Hover your mouse over the dots to see the values and time. The logarithmic trend line represents an empirical observation made by Phillips, in which there is a negative correlation between inflation and the unemployment rate. The indicator on the inflation trend curve decreases as unemployment increases.' ,
                                                style = {
                                                    'text-align':'center', 
                                                    #'font-family':'Messina Modern Book', 
                                                      'font-size':"1.1rem"
                                                    }),
                                          html.P('There are several theories explaining the Phillips curve, depending on whether the catalyst for the phenomenon is a change in price level or unemployment. In the face of high unemployment, the law of supply and demand requires lower prices to improve commodity sales. On the other hand, in the short term, as prices rise, output increases as producers increase commodity production to achieve higher margins. This leads to lower unemployment, as increased production leads to new recruitments that can be made to the unemployed population. On the other hand, when unemployment is low, there is labour demand pressure on the market, which increases wages. The rise in wages, on the other hand, leads to an increase in the overall price level, as suppliers of goods can ask for higher prices for their products and services.',
                                                 style = p_style),
                                          html.P('The Phillips curve can also be observed intuitively. For example, in 2015, the unemployment rate was several times above 10% with inflation remaining at zero and sometimes even negative. The coronavirus shock in 2020 caused a bump in the unemployment rate, but fuel prices, for example, were much lower than in the first half of 2022. There are several moments in history when unemployment and inflation have changed in different directions. There have also been exceptions, e.g. During the oil crisis of the 1970s, both were high, but if you look at history, there are several periods when the Phillips curve has been valid. It is important to remember that the Phillips curve is valid only in the short term, in which case predictions based on it should not be made for too long.', style = p_style),
                                          # html.P('Kyseessä on tunnettu taloustieteen teoria, jota on tosin vaikea soveltaa, koska ei ole olemassa sääntöjä, joilla voitaisiin helposti ennustaa työttömyyttä saatavilla olevien inflaatiota kuvaavien indikaattorien avulla. Mikäli sääntöjä on vaikea formuloida, niin niitä voi yrittää koneoppimisen avulla oppia historiadataa havainnoimalla. Voisiko siis olla olemassa tilastollisen oppimisen menetelmä, joka pystyisi oppimaan Phillipsin käyrän historiadatasta? Mikäli tämänlainen menetelmä olisi olemassa, pystyisimme ennustamaan lyhyen aikavälin työttömyyttä, kun oletamme hyödykkeiden hintatason muuttuvan skenaariomme mukaisesti.',
                                                # style=p_style), 
                                          html.P('The Phillips curve has its own chapter in the Economics Admission Examination Book of the University Degree in Economics and follows one of the ten basic principles of economics.',
                                                style=p_style),
                                          html.Br(),
                                                                            
                                          html.H3("Tenth Priciple of Economics:",style = {'text-align':'center', 
                                                                                                   #'font-family':'Messina Modern Semibold',
                                                                                                   'font-style': 'italic', 
                                                                                                   'font-weight': 'bold', 
                                                                                                   'font-size':"2.125rem"}),
                                          
                                          html.Blockquote('There is a conflict between unemployment and inflation in the short term. Full employment and stable price levels are difficult to achieve at the same time.', 
                                                style = {
                                                    'text-align':'center',
                                                    'font-style': 'italic', 
                                                    #'font-family':'Messina Modern Book', 
                                                      'font-size':p_font_size
                                                    }),
                                          html.P('Matti Pohjola, 2019, Taloustieteen oppikirja, page 250, ISBN:978-952-63-5298-5', 
                                                style={
                                                    'textAlign':'center',
                                                    #'font-family':'Messina Modern Book', 
                                                      'font-size':"1rem"
                                                    }),
        
                                          html.Br(),
                                          html.P('However, the Phillips curve is just a theory that is easy to observe by analyzing historical data. Even if the Phillips curve does not always materialise, would it still be possible to use the relationship between inflation and unemployment in predicting unemployment?',
                                                 style=p_style),
                                          html.P("It is difficult to convert the Phillips curve into a mathematical equation in which by investing inflation we can calculate the unemployment rate. This gave me the idea that there could be a machine learning method that could learn the prevailing laws between inflation and unemployment. Inflation is the annual change in the consumer price index. The consumer price index consists of several commodities that express the consumption needs of society at that time. Could some of these commodities affect more than others? Would only the basic consumer price index, the unemployment rate of the previous month and some information on seasonal fluctuations in unemployment be sufficient for forecast characteristics? Which commodities should I choose? What algorithm, what hyperparameters? I was toying with the idea that there might be some combination of commodity mix and methodology that can produce at least a satisfactory short-term forecast. That's why I wanted to create an app that anyone, regardless of academic background, could do experiments like this.",
                                                 style=p_style),
                                          html.P("After several iterations, the result was an application where you can design a basket of goods, choose a machine learning method, test the ability of the combination of these to predict already realized values and finally make predictions. On top of that, I built the possibility to adjust the hyperparameters of machine learning algorithms and utilize principal component analysis to eliminate irrelevant features. ",
                                                 style = p_style),
                                          html.P("The next problem was the difficulty of interpreting the models. There is a commonly known conflict between accuracy and interpretability in machine learning. Simpler models, such as linear regression, are easier to interpret than, for example, random forests, but random forests can produce better predictions. This creates a black box problem that needs to be solved in order to make the method more credible and transparent and thus to be used in general planning and decision-making. ",
                                                 style=p_style),
                                          html.P("I added Shapley values as agnostic functionality to the application. Shapley values are a concept based on game theory that is based on calculating the contributions of players in cooperative games (e.g. the contribution of individual players in a football game to the outcome). In machine learning, a similar model is used to estimate the prediction contribution of forecast traits. A more interesting research problem than predicting unemployment itself turned out to be that which commodities or combinations of commodities succeed in predicting unemployment best!",
                                                 style =p_style),
                                          html.P("The purpose was to find Phillips' observation using machine learning and perhaps find the formula of the Phillips curve. The benefit of machine learning comes from the fact that it produces its own view of the phenomenon by observing the data describing it. As AI pioneer Rodney Brooks has said, 'The world is the best model of itself, it is always updated, and contains all the necessary details, it just needs to be observed correctly and often enough.'"  ,
                                                 style =p_style),
                                          html.P('The result may therefore be something other than the law of the Phillips curve, as machine-learning algorithms can learn a completely different hidden law. The link between the components of inflation and the change in the unemployment rate is simply exploited. The formula learned is not the Phillips curve, but another rule, an oblique Phillips observation, possibly an inverse Phillips, or a curve containing valleys and hills. In addition, the forecast is made not only with price indices, but also with the unemployment rate of the previous month and the numerical values of the months (e.g. June is 6). In other words, the Phillips curve in this case is only a theoretical starting point, a spark for research and a background for somehow explaining unemployment with the components of inflation.',style=p_style),
                                          html.P("So I coded this combination of data science blog and application (I don't know what they call it, 'blapp', 'bloglication'...), which utilizes the data provided by Statistics Finland's Statfin interface from the Finnish Consumer Price Index by commodity relative to the base year 2010, and the unemployment rate in Finland monthly. Commodity groups for which data is not available for the whole period have been removed from the data sets. There are still hundreds of commodities and commodity groups left to build forecast components. Non-linear machine learning algorithms have been chosen as algorithm alternatives because they are better suited to this case than linear models.",
                                                 style =p_style),
                                          html.P('The application is divided into sections, which are expressed in tabs. The goal is to go from left to right in tabs and always iterate by changing the commodities and method. The application is based on the hypothesis that the monthly change in the unemployment rate of each month can be explained by the unemployment rate of the previous month, the current month and the prevailing index values of the selected commodity basket. When the monthly change in unemployment is obtained as a result of a machine learning algorithm, it can be combined with the unemployment rate of the previous month, resulting in a forecast for the unemployment rate of the current month. This idea is applied recursively as far as one wishes to predict. This solution requires that we make some assumptions about how commodity price indices will change. To evaluate it, a little exploratory analysis is required, to which there is a dedicated section in this application. Since the forecast involves assumptions, it is probably better to apply the forecast model for a short period of time (as the Phillips curve was originally intended).',
                                                 style =p_style),
                                          html.P("The final forecast is therefore based on user input assumptions about the monthly rate of change of selected commodities. You can adjust the change rate for each commodity individually, take advantage of previous months' averages, or set the same change rate for all commodities. The tests are performed on the assumption that the previous index values were realised as such. To improve reporting and documentation, test and forecast results can be exported as an Excel file for other possible use cases.",
                                                 style =p_style),
                                          
                                          
                                           html.H3('Instructions',
                                                   style=h3_style
                                                   ),
                                           
                                           html.P("Here's some instructions on how to use the app. Each section should be done in a corresponding order. Select the tab to complete each step. The tabs have even more detailed instructions. In addition, in the upper left corner of the app there is a button that opens the Quick Help on any tab.", 
                                                    style = p_style),
                                           html.Br(),
                                           html.P("1. Choice of goods. Select the commodity groups you want from the drop-down menu. These are used to predict unemployment.", 
                                                    style = p_style),
                                          html.P("2. Exploratory analysis. You can view and analyze the commodities you chose. If necessary, go back to the previous step and remove or add assets.",
                                                    style = p_style),
                                          html.P('3. Method selection. In this section you select the machine learning algorithm and adjust the hyperparameters. In addition, you can choose whether to utilize the main component analysis and how much variation to store.',
                                                    style = p_style),
                                         html.P('4. Test. You can select the time period in the past that the model seeks to predict. This allows you to estimate how the forecast model would have worked for already realized data. In this section you can also see how much each forecast feature contributed to making the forecast.',
                                                    style = p_style),
                                         html.P('5. Forecast. You can now use your chosen method to make a prediction for the future. Select the length of the forecast and click on the forecast. You can then export the prediction to Excel as well. The forecast is based on the change values of the commodities you set. As in the test section, you can also see which commodities and features affect the change in unemployment rates and which commodity price changes contribute to monthly changes in unemployment rates.',
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
                                          
                                          html.H3('Disclaimer',
                                                  style=h3_style),
                                          
                                          html.P("The site and its contents are provided free of charge and as available. This is a service provided by a private person, not an official service or an application made for commercial purposes. The use of the information obtained from the site is the responsibility of decision-makers. The Service Provider shall not be liable for any loss, litigation, claim, cost or damage, whatever or in any way, arising directly or indirectly from the use of the Service. Please note that this page is still under development.",
                                                  style=p_style),
                                          html.P("This website uses only crucial necessary cookies and users' personal data is not collected for any purpose.",
                                                  style=p_style),
                                          html.A([html.P('See a third-party report on GDPR compliance.', style = p_center_style)],
                                                 href = '/assets/report-skewedphillipsherokuappcom-11629005.pdf',
                                                 target = '_blank'),
                                          html.Br(),
                                          html.H3('Supported browsers and technical limitations',
                                                  style=h3_style),
                                          
                                          html.P("The app has been tested to work with Google Chrome, Edge and Mozilla Firefox. In Internet Explorer, the application does not work. Opera, Safari and other browsers have not been tested.",
                                                  style=p_style),
                                          html.P("The application can also be downloaded so-called. standalone version, so it can be started without a browser e.g. Windows or Android. For example, in Google Chrome, on the right side of the browser's address bar, there should be an icon from which to download the app. After downloading the app, you can find it on your own device.",
                                                  style=p_style),
                                       
                                          html.Br(),
                                          html.Div(children = [
                                              html.H3('References', 
                                                      style = h3_style),
                                              html.P('Here is also a list of data sources and additional reading related to the topics described.',
                                                     style =p_style),
                                              
                                              html.Label(['Statistics Finland’s free-of-charge statistical databases: ', 
                                                        html.A('Key indicators of the Labour Force Survey and their seasonal adjusted series and trends adjusted for random and seasonal variation', href = "https://statfin.stat.fi/PxWeb/pxweb/en/StatFin/StatFin__tyti/statfin_tyti_pxt_135z.px/",target="_blank")
                                                      ],style=p_style),
                                              html.Br(),
                                              html.Label(['Statistics Finland’s free-of-charge statistical databases: ', 
                                                        html.A('Consumer Price Index (2010=100), monthly data', href = "https://statfin.stat.fi/PxWeb/pxweb/en/StatFin/StatFin__khi/statfin_khi_pxt_11xd.px/",target="_blank")
                                                      ],style=p_style),
                                              html.Br(),
                                              html.Label(['Statistics Finland’s free-of-charge statistical databases: ', 
                                                        html.A('Annual changes of Consumer Price Index, monthly data', href = "https://statfin.stat.fi/PxWeb/pxweb/en/StatFin/StatFin__khi/statfin_khi_pxt_122p.px/",target="_blank")
                                                      ],style=p_style),
                                              html.Br(),
                                              html.Label(['Statistics Finland: ', 
                                                       html.A('Concepts', href = "https://www.stat.fi/meta/kas/index_en.html",target="_blank")
                                                      ],style=p_style),
                                              html.Br(),
                                              html.Label(['Wikipedia: ', 
                                                        html.A('Phillips curve', href = "https://en.wikipedia.org/wiki/Phillips_curve",target="_blank")
                                                      ],style=p_style),
                                              html.Br(),
                                              html.Label(['Raha ja Talous: ', 
                                                        html.A('Phillipsin käyrä (in Finnish) ', href = "https://rahajatalous.wordpress.com/2012/11/15/phillipsin-kayra/",target="_blank")
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
                                                        html.A('Principal component analysis', href = "https://en.wikipedia.org/wiki/Principal_component_analysis",target="_blank")
                                                      ],style=p_style),
                                              html.Br(),
                                              html.Label(['Wikipedia: ', 
                                                        html.A("Pearson's correlation coefficient", href = "https://en.wikipedia.org/wiki/Pearson_correlation_coefficient",target="_blank")
                                                      ],style=p_style),
                                              html.Br(),
                                                html.Label(['Scikit-learn: ', 
                                                        html.A('Regression', href = "https://scikit-learn.org/stable/supervised_learning.html#supervised-learning",target="_blank")
                                                        ],style=p_style),
                                                html.Br(),
    
                                          ]),
                                          html.Br(),
                                          html.Br(),
                                          html.H3('Written by', style = h3_style),
                                          
                                          html.Div(style = {'textAlign':'center'},children = [
                                              html.I('Tuomas Poukkula', style = p_center_style),
                                         
                                              html.Br(),
                                              html.P("Data Scientist",
                                                     style = p_center_style),
                                              html.P("Gofore Ltd",
                                                     style = p_center_style),
                                              html.A([html.P('Contact via e-mail',style = p_center_style)],
                                                     href = 'mailto:tuomas.poukkula@gofore.com?subject=Phillips: Q&A',
                                                     target='_blank')
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
                                      dbc.Col([
                                      html.Div([
                                          html.A([
                                               html.Img(
                                                   src='/assets/gofore_logo_orange.svg',
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
                                              
                                              html.Label(['Application on ', 
                                                      html.A('GitHub', href='https://github.com/tuopouk/skewedphillips')
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
            
            dbc.Tab(label ='Choice of Goods',
                    tab_id ='feature_tab_en',
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
        
                dbc.Row(children = [
                        
                        dbc.Row([
                            dbc.Col([
                                
                                html.Br(),
                                html.P('This section selects the commodities used to predict unemployment.',
                                        style = p_style),
                                html.P('You can select items from the menu below, then you can adjust their expected monthly change by entering a number in the boxes below.',
                                        style = p_style),
                                html.P('You can also adjust the same monthly change for all assets or take advantage of averages of actual monthly changes.',
                                        style = p_style),
                                html.P("You can select or sort the product menu from the drop-down menu above. You can choose either alphabetical order, correlation order (according to Pearson's correlation coefficient) or delineation according to Statistics Finland's commodity hierarchy. The correlation order here refers to the correlation coefficient between the values of the price index of each commodity and the unemployment rates at the same time, calculated using the Pearson method. These can be sorted in descending or ascending order by either the actual value (highest positive - lowest negative) or the absolute value (without +/-).",
                                        style = p_style),

                                html.Br(),
                                
                                ],xs =12, sm=12, md=12, lg=9, xl=9)
                        ], justify = 'center'),
                    
                        dbc.Col(children=[
                           
                            html.Br(),
                            html.Br(),
                            html.H3('Select features from the dropdown menu',
                                    style=h3_style),
                            
                            dbc.DropdownMenu(id = 'sorting_en',
                                              #align_end=True,
                                              children = [
                                                 
                                                  dbc.DropdownMenuItem("Alphabetical order", id = 'alphabet_en',style={
                                                      'font-size':"0.9rem", 
                                                      #'font-family':'Cadiz Book'
                                                      }
                                                      ),
                                                  dbc.DropdownMenuItem("Correlation (descending)", id = 'corr_desc_en',style={
                                                      'font-size':"0.9rem", 
                                                      #'font-family':'Cadiz Book'
                                                      }
                                                      ),
                                                  dbc.DropdownMenuItem("Correlation (ascending)", id = 'corr_asc_en',style={
                                                      'font-size':"0.9rem", 
                                                      #'font-family':'Cadiz Book'
                                                      })
                                                      ,
                                                  dbc.DropdownMenuItem("Absolute correlation (descending)", id = 'corr_abs_desc_en',style={
                                                      'font-size':"0.9rem", 
                                                      #'font-family':'Cadiz Book'
                                                      }
                                                      ),
                                                  dbc.DropdownMenuItem("Absolute correlation (ascending)", id = 'corr_abs_asc_en',style={
                                                      'font-size':"0.9rem", 
                                                      #'font-family':'Cadiz Book'
                                                      }
                                                      ),
                                                  dbc.DropdownMenuItem("Main classes", id = 'main_class_en',style={
                                                      'font-size':"0.9rem", 
                                                      #'font-family':'Cadiz Book'
                                                      }
                                                      ),
                                                  dbc.DropdownMenuItem("2. class", id = 'second_class_en',style={
                                                      'font-size':"0.9rem", 
                                                      #'font-family':'Cadiz Book'
                                                      }
                                                      ),
                                                  dbc.DropdownMenuItem("3. class", id = 'third_class_en',style={
                                                      'font-size':"0.9rem", 
                                                      #'font-family':'Cadiz Book'
                                                      }
                                                      ),
                                                  dbc.DropdownMenuItem("4. class", id = 'fourth_class_en',style={
                                                      'font-size':"0.9rem", 
                                                      #'font-family':'Cadiz Book'
                                                      }
                                                      ),
                                                  dbc.DropdownMenuItem("5. class", id = 'fifth_class_en',style={
                                                      'font-size':"0.9rem", 
                                                      #'font-family':'Cadiz Book'
                                                      }
                                                      )
                                                 
                                                 
                                                  ],
                                            label = "Absolute correlation (descending)",
                                            color="secondary", 
                                            className="m-1",
                                            size="lg",
                                            style={
                                                'font-size':"0.9rem", 
                                                #'font-family':'Cadiz Book'
                                                }
                                            ),
                            
                            html.Br(),
                            dcc.Dropdown(id = 'feature_selection_en',
                                          options = initial_options_en,
                                          multi = True,
                                          value = list(initial_features_en),
                                          style = {'font-size':"1rem", #'font-family':'Cadiz Book'
                                                   },
                                          placeholder = 'Select Commodities'),
                            html.Br(),
                            
                            dbc.Alert("Select at least one commodity!", color="danger",
                                      dismissable=True, fade = True, is_open=False, id = 'alert_en', 
                                      style = {'text-align':'center', 'font-size':p_font_size, #'font-family':'Cadiz Semibold'
                                               }),
                            dash_daq.BooleanSwitch(id = 'select_all_en', 
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
                    dbc.Row(id = 'selections_div_en', children = [
                        
                            dbc.Col([
                                
                                html.H3('Set an estimation of average monthly index change in percentages',
                                                        style = h3_style),
                                
                                dash_daq.BooleanSwitch(id = 'averaging_en', 
                                                        label = dict(label = 'Use averaging',style = {'font-size':p_font_size, 
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
                            html.Div(id = 'slider_div_en'),
                            html.Br(),
                            html.Div(id='slider_prompt_div_en')
                            
                            ], xs =12, sm=12, md=12, lg=9, xl=9)
                        
                        ],justify = 'center', 
                  
                    ),
                dbc.Row(id = 'adjustments_div_en',
                        justify = 'left', 
                     
                        ),
                html.Br(),
                footer
            ]
            ),


            dbc.Tab(label = 'Exploratory Analysis',
                    tab_id = 'eda_tab_en',
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
                              html.P("In this section you can view the unemployment rate and the relationship between the selected commodity groups in the Consumer Price Index and the change over time. Below you can see how the price indices of different commodity groups correlate with each other and with the unemployment rate. You can also observe time series of indices, inflation and unemployment rates. The correlation described is based on Pearson's correlation coefficient.",
                                     style = p_style),
                              html.Br()
                              ],xs =12, sm=12, md=12, lg=9, xl=9)
                         ],
                             justify = 'center'
                             ),
                    
                     dbc.Row([
                                dbc.Col(children = [
                                   
                                        html.Div(id = 'corr_selection_div_en'),
                                        html.Br(),
                                        html.Div(id = 'eda_div_en', 
                                                 children =[
                                                     html.Div([dbc.RadioItems(id = 'eda_y_axis_en', 
                                                                 options = [{'label':'Unemployment rate (%)','value':'Työttömyysaste'},
                                                                           {'label':'Monthly unemployment rate change (% units)','value':'change'}],
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
                                                     html.Div(id = 'commodity_unemployment_div_en')
                                                     ]
                                                 ),
                                        html.Br(),
                                       
                                   
                                    ], xs =12, sm=12, md=12, lg=6, xl=6
                                ),
                                dbc.Col(children = [
                                   
                                        html.Div(id = 'feature_corr_selection_div_en'),
                                        html.Br(),
                                        html.Div(id = 'corr_div_en'),
                                   
                                    ], xs =12, sm=12, md=12, lg=6, xl=6, align ='start'
                                )
                         ],justify='center', 
                          # style = {'margin' : '10px 10px 10px 10px'}
                     ),

                    # html.Br(),
                     dbc.Row(id = 'timeseries_div_en',children=[
                        
                         dbc.Col(xs =12, sm=12, md=12, lg=6, xl=6,
                                 children = [
                                     html.Div(id = 'timeseries_selection_en'),
                                     html.Br(),
                                     html.Div(id='timeseries_en')
                                     ]),
                         dbc.Col(children = [
                             html.Div(style={'textAlign':'center'},children=[
                                 html.Br(),
                                 html.H3('The figure below shows the monthly unemployment rate and inflation in Finland.',
                                        style = h3_style),
                                 
                                 html.Div(id = 'employment_inflation_div_en',
                                          
                                          children=[dcc.Graph(id ='employment_inflation_en',
                                                     figure = go.Figure(data=[go.Scatter(x = data_en.index,
                                                                               y = data_en.Työttömyysaste,
                                                                               name = 'Unemployment Rate',
                                                                               mode = 'lines',
                                                                               hovertemplate = '%{x}'+'<br>%{y}',
                                                                               marker = dict(color ='red')),
                                                                    go.Scatter(x = data_en.index,
                                                                               y = data_en.Inflaatio,
                                                                               name = 'Inflation',
                                                                               hovertemplate = '%{x}'+'<br>%{y}',
                                                                               mode ='lines',
                                                                               marker = dict(color = 'purple'))],
                                                              layout = go.Layout(title = dict(text = 'Unemployment Rate and Inflation per Month<br>{} - {}'.format(data_en.index.strftime('%B %Y').values[0],data_en.index.strftime('%B %Y').values[-1]),
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
                                                                                 xaxis = dict(title=dict(text = 'Time',
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
                                                                                                           label="1y",
                                                                                                           step="year",
                                                                                                           stepmode="backward"),
                                                                                                      dict(count=3,
                                                                                                           label="3y",
                                                                                                           step="year",
                                                                                                           stepmode="backward"),
                                                                                                      dict(count=5,
                                                                                                           label="5y",
                                                                                                           step="year",
                                                                                                           stepmode="backward"),
                                                                                                      dict(step="all",label = 'MAX')
                                                                                                  ])
                                                                                              )
                                                                                              ),
                                                                                 yaxis = dict(title=dict(text = 'Value (%)',
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
                                           config=config_plots_en
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
            dbc.Tab(label='Method Selection',
                    tab_id ='hyperparam_tab_en',
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
                                html.P('In this section you can select the machine learning algorithm, adjust its hyperparameters and, if you wish, utilize principal component analysis as you choose.',
                                        style = p_style),
                                html.P('First select the algorithm, then below will appear its specific hyperparameters, which you can adjust to fit. Hyperparameters are read automatically from Scikit-learn library documentation or referenced from XGBoost documentation. Under the control menus you will find a link to the documentation page of the selected algorithm, where you can read more about the topic. There is not one single way to adjust hyperparameters, but different values must be tested iteratively.',
                                        style = p_style),
                                html.Br(),
                                html.P('In addition, you can choose whether to utilize principal component analysis to minimize features. The main component analysis is a statistical and technical noise reduction method aimed at improving the quality of the forecast. Linear combinations of the selected features are formed in such a way that the variation in the original data remains by a certain ratio in the modified data. You can adjust this explained variance to your liking. As in the case of hyperparameters, this definition is purely empirical.',
                                        style = p_style),
                                html.P('If the edges of the hyperparameter box are red, then the value is not appropriate. Testing and prediction fails if allowed values are applied to hyperparameters. You can check the allowed values in the model documentation.',
                                        style = p_style)
                            ],xs =12, sm=12, md=12, lg=9, xl=9)
                        ], justify = 'center'
                            ),
                        html.Br(),
                        dbc.Row([
                            dbc.Col(id = 'model_selection_en', children = [
                                
                                html.H3('Choose an algorithm',style=h3_style),
                                
                                dcc.Dropdown(id = 'model_selector_en',
                                              value = 'Random Forest',
                                              multi = False,
                                              placeholder = 'Choose an algorithm',
                                              style = {'font-size':"0.9rem", #'font-family':'Cadiz Book'
                                                       },
                                              options = [{'label': c, 'value': c} for c in MODELS_en.keys()]),
                                
                                html.Br(),
                                html.H3('Adjust the model hyperparameters', style = h3_style),
                                
                                html.Div(id = 'hyperparameters_div_en')
                                
                                ], xs =12, sm=12, md=12, lg=9, xl=9),
                            dbc.Col(id = 'pca_selections_en', children = [
                                html.Br(),
                                dash_daq.BooleanSwitch(id = 'pca_switch_en', 
                                                                  label = dict(label = 'Use Principal Component Analysis',style = {'font-size':30, 
                                                                                                                                # 'font-family':'Cadiz Semibold',
                                                                                                                                'textAlign':'center'}), 
                                                                  on = False, 
                                                                  color = 'blue'),
                                html.Br(),
                                html.P('Principal component analysis is a noise removal method that condenses the information of forecast features into the main components. Each major component stores the variation of the original data, and the stored variation of all major components totals 100%.',
                                       style = p_style),
                                html.A([html.P('Watch a short introduction video on principal component analysis.',
                                               style = p_style)],
                                       href = "https://www.youtube.com/embed/hJZHcmJBk1o",
                                       target = '_blank'),
                                
                                
                                html.Div(id = 'ev_placeholder_en',children =[
                                    html.H3('Select explained variance', style = h3_style),
                                    
                                    dcc.Slider(id = 'ev_slider_en',
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
                                  html.Div(id = 'ev_slider_update_en', 
                                          children = [
                                              html.Div([html.P('You selected {} % explained variance.'.format(95),
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
            dbc.Tab(label='Test',
                    tab_id ='test_tab_en',
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
                                        # html.H3('Test your method', style = h3_style),
                                        # html.Br(),
                                        html.P('In this section you can test how well the chosen method would have been able to predict the unemployment rate of the past months using the selected features. When testing, the selected number of months is left as test data, which the method seeks to predict.',
                                               style = p_style),
                                        html.P('At this point, commodity indices are assumed to materialise as they stand.',
                                               style = p_style),
                                        html.P('After you have completed the test, you can view the next result graph or export the test data from the button below to Excel.',
                                              style=p_style),
                                        html.P("From the graphs below the test results, you can see how the commodities you selected affected the tested forecast.",style=p_style),
                                        html.Br()
                                        ],xs =12, sm=12, md=12, lg=9, xl=9),
                                ],justify='center'),
                        dbc.Row([
                            dbc.Col([
                                        html.H3('Select test length',style = h3_style),
                                        dcc.Slider(id = 'test_slider_en',
                                                  min = 1,
                                                  max = 18,
                                                  value = 3,
                                                  step = 1,
                                                  tooltip={"placement": "top", "always_visible": True},
                                                 
                                                  marks = {1: {'label':'a month', 'style':{'font-size':"1.2rem", 
                                                                                            # 'font-family':'Cadiz Semibold'
                                                                                            }},
                                                          # 3:{'label':'3 kuukautta', 'style':{'font-size':20, #'fontFamily':'Cadiz Semibold','color':'white'}},
                                                        
                                                            6:{'label':'six months', 'style':{'font-size':"1.2rem", 
                                                                                                # 'font-family':'Cadiz Semibold'
                                                                                                }},
                                                          #  9:{'label':'yhdeksän kuukautta', 'style':{'font-size':20, #'fontFamily':'Cadiz Semibold','color':'white'}},
                                                         
                                                          12:{'label':'a year', 'style':{'font-size':"1.2rem", 
                                                                                        # 'font-family':'Cadiz Semibold'
                                                                                        }},
                                                          18:{'label':'one and a half years', 'style':{'font-size':"1.2rem", 
                                                                                        # 'font-family':'Cadiz Semibold'
                                                                                        }},
                                                          

                                                            }
                                                  ),
                                        html.Br(),  
                                        html.Div(id = 'test_size_indicator_en', style = {'textAlign':'center'}),
                                        html.Br(),
                                        html.Div(id = 'test_button_div_en',children = [html.P('Select commodities first.',style = {
                                            'text-align':'center', 
                                            #'font-family':'Cadiz Semibold', 
                                              'font-size':p_font_size
                                            })], style = {'textAlign':'center'}),
                                        html.Br(),
                                        html.Div(id='test_download_button_div_en', style={'textAlign':'center'})
                                        
                                        
                            
                            
                            ],xs =12, sm=12, md=12, lg=9, xl=9)
                            ], justify = 'center', style={'text-align':'center'}
                            ),
                        html.Br(),
                        dbc.Row(children = [
                            dbc.Col([html.Div(id = 'test_results_div_en')],xs = 12, sm = 12, md = 12, lg = 9, xl = 9),
                            
                            
                            ], justify = 'center', 
                            
                            ),
                        html.Br(),
                        dbc.Row([dbc.Col([html.Div(id = 'shap_selections_div_en')],xs = 12, sm = 12, md = 12, lg = 9, xl = 9)],justify = 'center'),
                        dbc.Row(children = [
                            
                            dbc.Col([html.Div(id = 'shap_results_div_en')],xs = 12, sm = 12, md = 12, lg = 6, xl = 6),
                            dbc.Col([html.Div(id = 'local_shap_results_div_en')],xs = 12, sm = 12, md = 12, lg = 6, xl = 6),
                            
                            ], justify = 'center', align='start', 
                            
                            ),
                        html.Br(),
                        footer
                        
                        
                        ]
                    ),
            dbc.Tab(label='Forecast',
                    tab_id = 'forecast_tab_en',
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
                                        # html.H3('Forecasting',style=h3_style),
                                        # html.Br(),
                                        
                                        html.P('In this section you can make a forecast for the selected time. When predicting, the settings made on the Method Selection tab are enabled. The forecast is based on the assumptions made in the Product Selection tab about the relative price development of commodities. '
                                               'It is good to note that the uncertainty increases the further the projection is made. '
                                               "In addition, you can also look at the agnostics of the model, which shows which goods and features will affect the development of the unemployment rate and which commodities' price changes will affect the change in the unemployment rate on a monthly basis if the assumptions made by the user are fulfilled.",
                                              style=p_style),
                                        html.P('Once you have made the forecast, you can view the adjacent forecast graph or export the result data from the button below to Excel.',
                                              style=p_style),
                                        html.Br(),
                                        html.H3('Select forecast length',
                                                style=h3_style),
                                        dcc.Slider(id = 'forecast_slider_en',
                                                  min = 2,
                                                  max = 18,
                                                  value = 3,
                                                  step = 1,
                                                  tooltip={"placement": "top", "always_visible": True},
                                                  marks = {2: {'label':'two months', 'style':{'font-size':16, 
                                                                                               # #'fontFamily':'Cadiz Semibold'
                                                                                               }},
                                                          # 3: {'label':'kolme kuukautta', 'style':{'font-size':20, #'fontFamily':'Cadiz Semibold','color':'white'}},
                                                          6:{'label':'six months', 'style':{'font-size':16, 
                                                                                              # #'fontFamily':'Cadiz Semibold'
                                                                                              }},
                                                          # 9:{'label':'yhdeksän kuukautta', 'style':{'font-size':20, #'fontFamily':'Cadiz Semibold','color':'white'}},
                                                          12:{'label':'a year', 'style':{'font-size':16, 
                                                                                        # 'font-family':'Cadiz Semibold'
                                                                                        }},
                                                          18:{'label':'one and a half years', 'style':{'font-size':16, 
                                                                                        # 'font-family':'Cadiz Semibold'
                                                                                        }},
                                                        #  24:{'label':'kaksi vuotta', 'style':{'font-size':20, #'fontFamily':'Cadiz Semibold','color':'white'}}
                                                        
                                                     

                                                          }
                                                  ),
                                        html.Br(),
                                        html.Div(id = 'forecast_slider_indicator_en',style = {'textAlign':'center'}),
                                        html.Div(id = 'forecast_button_div_en',children = [html.P('Valitse commodities first.',
                                                                                              style = p_style
                                                                                              )],style = {'textAlign':'center'})
                                        
                                        
                                        
                                    ],xs =12, sm=12, md=12, lg=9, xl=9)
                            
                            
                            ], justify = 'center'),
                        html.Br(),
                        dbc.Row(children = [
                                    
                                    dbc.Col([dcc.Loading(id = 'forecast_results_div_en',type = spinners[random.randint(0,len(spinners)-1)])],
                                            xs = 12, sm = 12, md = 12, lg = 8, xl = 8)
                                    ], justify = 'center'
                                    ),
                        html.Br(),
                        dbc.Row([dbc.Col([html.Div(id = 'forecast_shap_selections_div_en')],xs = 12, sm = 12, md = 12, lg = 9, xl = 9)],justify = 'center'),
                        dbc.Row([
                            
                            dbc.Col(id = 'forecast_shap_div_en', xs = 12, sm = 12, md = 12, lg = 6, xl = 6),
                            dbc.Col(id = 'forecast_local_shap_div_en', xs = 12, sm = 12, md = 12, lg = 6, xl = 6)
                            
                        ], justify ='center', align='start'),
                        html.Br(),
                        footer
                                       
                            
                            
                            
                            
                              
                            
                        ]
                            
                            ),
            


        ]
            
    ),

    
   ]
  )])

@callback(
    
    [Output('shap_features_switch_en', 'label'),
     Output('shap_features_switch_en', 'disabled')],
    Input('shap_data_en','data')
    
)
def en_update_shap_switch(shap_data):
    
    shap_df = pd.DataFrame(shap_data)
    shap_df = shap_df.set_index(shap_df.columns[0])
    
    if 'Kuukausi' not in shap_df.index:
        return dict(label = 'You used Principal Component Analysis',
                     style = {'font-size':p_font_size,
                              'text-align':'center'
                              # #'fontFamily':'Cadiz Semibold'
                              }), True
    else:
        return dict(label = 'Show only the contribution of commodities',
                     style = {'font-size':p_font_size, 
                              'text-align':'center'
                              # #'fontFamily':'Cadiz Semibold'
                              }), False
    
    
@callback(
    
    [Output('forecast_shap_features_switch_en', 'label'),
     Output('forecast_shap_features_switch_en', 'disabled')],
    Input('forecast_shap_data_en','data')
    
)
def en_update_forecast_shap_switch(shap_data):
    
    shap_df = pd.DataFrame(shap_data)
    shap_df = shap_df.set_index(shap_df.columns[0])
    
    if 'Kuukausi' not in shap_df.index:
        return dict(label = 'You used Principal Component Analysis',
                     style = {'font-size':p_font_size,
                              'text-align':'center'
                              # #'fontFamily':'Cadiz Semibold'
                              }), True
    else:
        return dict(label = 'Show only the contribution of commodities',
                     style = {'font-size':p_font_size, 
                              'text-align':'center'
                              # #'fontFamily':'Cadiz Semibold'
                              }), False    

@callback(

    [Output('adjustments_div_en','children'),
     Output('features_values_en','data')],
    [Input('slider_en','value'),
     Input('feature_selection_en','value')],
    [State('averaging_en','on')]    
    
)
def en_add_value_adjustments(slider_value, features, averaging):
    
    
    
    if averaging:
        
        mean_df = en_apply_average(features = features, length = slider_value)
        
        features_values = {feature:mean_df.loc[feature] for feature in features}
        
        row_children =[dbc.Col([html.Br(), 
                                html.P(feature,style={#'font-family':'Messina Modern Semibold',
                                            'font-size':"1.1rem"}),
                                dcc.Input(id = {'type':'value_adjust_en', 'index':feature}, 
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
                                dcc.Input(id = {'type':'value_adjust_en', 'index':feature}, 
                                               value = slider_value, 
                                               type = 'number', 
                                               style ={#'font-family':'Messina Modern Semibold',
                                                           'font-size':"1.1rem"},
                                               step = .1)],xs =12, sm=12, md=4, lg=2, xl=2) for feature in features]
    return row_children, features_values


@callback(

    Output('change_weights_en','data'),
    [Input({'type': 'value_adjust_en', 'index': ALL}, 'id'),
    Input({'type': 'value_adjust_en', 'index': ALL}, 'value')],    
    
)
def en_store_weights(feature_changes, feature_change_values):
    
    if feature_changes is None:
        raise PreventUpdate
    
    weights_dict = {feature_changes[i]['index']:feature_change_values[i] for i in range(len(feature_changes))}
        
    return weights_dict


@callback(

    Output('slider_div_en','children'),
    [Input('averaging_en', 'on')
     ]
    
)
def en_update_slider_div(averaging):
    
    if averaging:
        
        return [html.H3('Select the number of latest months for averaging', 
                        style = h3_style),
        html.Br(),
        dcc.Slider(id = 'slider_en',
                      min = 1,
                      max = 12,
                      value = 4,
                      step = 1,
                      tooltip={"placement": "top", "always_visible": True},
                       marks = {1:{'label':'a month', 'style':{'font-size':20, 
                                                                # #'fontFamily':'Cadiz Semibold'
                                                                }},
                                # 3:{'label':'3 kuukautta', 'style':{'font-size':20, #'fontFamily':'Cadiz Semibold','color':'white'}},
                                6:{'label':'six months', 'style':{'font-size':20, 
                                                                    # #'fontFamily':'Cadiz Semibold'
                                                                    }},
                                # 9:{'label':'yhdeksän kuukautta', 'style':{'font-size':20, #'fontFamily':'Cadiz Semibold','color':'white'}},
                                12:{'label':'a year', 'style':{'font-size':20, 
                                                              # #'fontFamily':'Cadiz Semibold'
                                                              }}   
                             }
                      
                    )]
        
    else:
        return [
            html.H3('Select the constant monthly change', 
                    style = h3_style),
            
            dcc.Slider(id = 'slider_en',
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

    Output('alert_en', 'is_open'),
    [Input('feature_selection_en','value')]

)
def en_update_alert(features):
    
    return len(features) == 0

@callback(

    Output('hyperparameters_div_en','children'),
    [Input('model_selector_en','value')]    
    
)
def en_update_hyperparameter_selections(model_name):
    
    
    
    model = MODELS_en[model_name]['model']
    
        
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

    param_options = en_get_param_options(model_name)
       
    
    children = []
    
    for i in h_series.index:
        
        hyperparameter = list(hyperparameters.keys())[i]
        
        
        
        if hyperparameter not in UNWANTED_PARAMS:
        
            
            value = list(hyperparameters.values())[i]
            
            
            
            if type(value) == int:
                children.append(dbc.Col([html.P(hyperparameter+':', 
                                                 style=p_bold_style)],xs =12, sm=12, md=12, lg=12, xl=12))
                children.append(html.Br())
                children.append(dbc.Col([dcc.Slider(id = {'index':hyperparameter, 'type':'hyperparameter_tuner_en'},
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
                    children.append(dbc.Col([dbc.Input(id = {'index':hyperparameter, 'type':'hyperparameter_tuner_en'},
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
                        children.append(dbc.Col([dbc.Input(id = {'index':hyperparameter, 'type':'hyperparameter_tuner_en'},
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
                    children.append(dbc.Col([dbc.Input(id = {'index':hyperparameter, 'type':'hyperparameter_tuner_en'},
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
                children.append(dbc.Col([dbc.Switch(id = {'index':hyperparameter, 'type':'hyperparameter_tuner_en'}, 
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
                        children.append(dbc.Col([dcc.Dropdown(id = {'index':hyperparameter, 'type':'hyperparameter_tuner_en'}, 
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
             children = [html.A('Check a short introduction video about the algorithm.', href = MODELS_en[model_name]['video'], target="_blank",style = p_style),
                         html.Br(),
                         html.A('Check the technical documentation.', href = MODELS_en[model_name]['doc'], target="_blank",style = p_style),
                         ]))
    return dbc.Row(children, justify ='start')


@callback(

    Output('method_selection_results_en','data'),
    [Input('model_selector_en','value'),
    Input({'type': 'hyperparameter_tuner_en', 'index': ALL}, 'id'),
    Input({'type': 'hyperparameter_tuner_en', 'index': ALL}, 'value'),
    Input('pca_switch_en','on'),
    Input('ev_slider_en','value')]    
    
)
def en_store_method_selection_results(model_name, hyperparams, hyperparam_values,pca, explained_variance):
            
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
    
      [Output('test_data_en','data'),
       Output('test_results_div_en','children'),
       Output('test_download_button_div_en','children'),
       Output('shap_data_en','data'),
       Output('local_shap_data_en','data')],
      [Input('test_button_en','n_clicks')],
      [State('test_slider_en', 'value'),
       State('features_values_en','data'),
       State('method_selection_results_en','data')

       ]
    
)
def en_update_test_results(n_clicks, 
                        test_size,
                        features_values,
                        method_selection_results
 
                        ):
    
    if n_clicks > 0:
    
        try:
            locale.setlocale(locale.LC_ALL, 'en_US')
        except:
            locale.setlocale(locale.LC_ALL, 'en-US')
        
        
        features = sorted(list(features_values.keys()))
        
        model_name = method_selection_results['model']
             
        pca = method_selection_results['pca']
        explained_variance = method_selection_results['explained_variance']
        hyperparam_grid = method_selection_results['hyperparam_grid']
        
        model = MODELS_en[model_name]['model']
        constants = MODELS_en[model_name]['constant_hyperparameters'] 
        
        
        model_hyperparams = hyperparam_grid.copy()
        
        
        model_hyperparams.update(constants)
        
        
        model = model(**model_hyperparams)
        
        explainer = MODELS_en[model_name]['explainer']
        
        
        test_result, shap_results, local_shap_df = en_test(model, features, explainer = explainer, test_size=test_size, use_pca=pca,n_components=explained_variance)

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
                             html.H3('How did we do?',
                                     style = h3_style),
                             
                             html.P('The graph below shows how well the forecast model would have predicted the unemployment rate from {} to {}.'.format(test_result.index.strftime('%B %Y').values[0],test_result.index.strftime('%B %Y').values[-1]),
                                    style = p_style),
                             html.P("(Continues after the chart)",style={
                                         'font-style':'italic',
                                         'font-size':p_font_size,
                                        'text-align':'center'}
                                 ),
                              html.Div([html.Br(),dbc.RadioItems(id = 'test_chart_type_en', 
                                          options = [{'label':'bars','value':'bars'},
                                                    {'label':'lines','value':'lines'},
                                                    {'label':'lines and bars','value':'lines+bars'}],
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
                              html.Div([dcc.Loading(id ='test_graph_div_en',type = spinners[random.randint(0,len(spinners)-1)])])
   
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
                                     html.H3('Accuracy (%)', style = h3_style),
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
                    html.P('The mean absolute percentage error (MAPE) is the average of the relative errors of all forecast values. The accuracy in this case is calculated by the formula: 1 - MAPE.', 
                           style = p_style,
                           className="card-text"),
                    html.Br(),

                    
                    ]
             

        feat = features.copy()
        feat = ['Työttömyysaste','Ennuste','prev','month','change','mape','n_feat', 'Ennustettu muutos']+feat
        
        button_children = dbc.Button(children=[html.I(className="fa fa-download mr-1"), ' Download Test Results'],
                                       id='test_download_button_en',
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
    
      [Output('forecast_data_en','data'),
       Output('forecast_shap_data_en','data'),
       Output('local_forecast_shap_data_en','data'),
       Output('forecast_results_div_en','children'),
       Output('forecast_download_button_div_en','children')],
      [Input('forecast_button_en','n_clicks')],
      [State('forecast_slider_en', 'value'),
       State('change_weights_en','data'),
       State('method_selection_results_en','data')

       ]
    
)
def en_update_forecast_results(n_clicks, 
                        forecast_size,
                        weights_dict,
                        method_selection_results
                        ):

    
    
    if n_clicks > 0:
        
        try:
            locale.setlocale(locale.LC_ALL, 'en_US')
        except:
            locale.setlocale(locale.LC_ALL, 'en-US')
        
        features = sorted(list(weights_dict.keys()))
        
        
        
        model_name = method_selection_results['model']
             
        pca = method_selection_results['pca']
        explained_variance = method_selection_results['explained_variance']
        hyperparam_grid = method_selection_results['hyperparam_grid']
        
        model = MODELS_en[model_name]['model']
        constants = MODELS_en[model_name]['constant_hyperparameters'] 
        
        
        model_hyperparams = hyperparam_grid.copy()
        
        
        model_hyperparams.update(constants)
        
        
        model = model(**model_hyperparams)
        
        
        weights = pd.Series(weights_dict)
        
        explainer = MODELS_en[model_name]['explainer']
        
        forecast_df, shap_df, local_shap_df = en_predict(model, 
                              explainer,   
                              features, 
                              feature_changes = weights, 
                              length=forecast_size, 
                              use_pca=pca,
                              n_components=explained_variance)
        

        
        forecast_div =  html.Div([html.Br(),
                      html.H3('Forecast results', 
                              style = h3_style),
                      
                      html.P('The graph below shows the actual values and the forecast from {} to {}.'.format(forecast_df.index.strftime('%B %Y').values[0],forecast_df.index.strftime('%B %Y').values[-1]),
                             style = p_style),
                      html.P('You can select either a column, area, or a line diagram from the buttons below. The length of the time series can be adjusted from the slider below. You can also limit the length by clicking the buttons in the upper left corner.',
                             style = p_style),
                      html.P("(Continues after the chart)",style={
                                  'font-style':'italic',
                                  'font-size':p_font_size,
                                 'text-align':'center'}
                          ),
                      
                      html.Div([
                      dbc.RadioItems(id = 'chart_type_en', 
                        options = [{'label':'bars','value':'bars'},
                                  {'label':'lines','value':'lines'},
                                  {'label':'area','value':'area'}],
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

          html.Div(id = 'forecast_graph_div_en'),
        
          html.Br()
          ])

          
          # ], justify='center')        
        forecast_download_button = dbc.Button(children=[html.I(className="fa fa-download mr-1"), ' Download Forecast Results'],
                                 id='forecast_download_button_en',
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

    [Output('shap_selections_div_en','children'),
     Output('shap_results_div_en','children'),
     Output('local_shap_results_div_en','children')
     ],
    [Input('test_button_en','n_clicks'),
     Input('shap_data_en','data'),
     State('local_shap_data_en','data')]    
    
)
def en_update_shap_results(n_clicks, shap, local_shap_data):
        
    if shap is None or local_shap_data is None:
        raise PreventUpdate
        
    if n_clicks > 0:
        
        try:
            locale.setlocale(locale.LC_ALL, 'en_US')
        except:
            locale.setlocale(locale.LC_ALL, 'en-US')
        
    
        shap_df = pd.DataFrame(shap)
        
        shap_df = shap_df.set_index(shap_df.columns[0])
        
        
        local_shap_df = pd.DataFrame(local_shap_data)
        local_shap_df = local_shap_df.set_index(local_shap_df.columns[0])
        local_shap_df.index = pd.to_datetime(local_shap_df.index)
        
        options = [{'label':c.strftime('%B %Y'), 'value': c} for c in list(local_shap_df.index)]

         
        return [[
            
                    html.H3('Which features were the most important?',
                           style = h3_style),
                    html.P("The following graphs show the global and local importance of the forecast features used for the forecast. "
                            "Global significance values can be used to examine which features in general are most significant for the forecast. "
                            "Instead, local values can be used to assess which factors affected and how to a particular month's forecast. "
                            "Significant values are presented as SHAP values, which describe the contribution of traits to the forecast. "
                            "In addition to the selected commodity indices, the forecast features include the unemployment rate of the previous month and the month.",
                           style = p_style),
                    html.A([html.P('Watch a short introduction video about the importance of SHAP values in machine learning.',
                                   style = p_center_style)], href="https://www.youtube.com/embed/Tg8aPwPPJ9c", target='_blank'),
                    html.A([html.P('See also a Non-Technical Guide to Interpreting SHAP Analyses',
                                   style = p_center_style)], href="https://www.aidancooper.co.uk/a-non-technical-guide-to-interpreting-shap-analyses/", target='_blank'),
                    html.P("The graph's SHAP values are multiplied by 100 to improve visualization. "
                           "Thereby one SHAP unit corresponds to one hundredth of a percentage point that is used to describe unemployment rate's monthly change. ",
                           style = p_style),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                                html.Div(id = 'cut_off_div_en'),
                                
                                html.Div(id = 'cut_off_indicator_en'),
                                
                                ],xs =12, sm=12, md=12, lg=9, xl=9),
                        dbc.Col([
                                dash_daq.BooleanSwitch(id = 'shap_features_switch_en', 
                                                        label = dict(label = 'Show only the contribution of commodities',
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
                            html.H3('Feature importances', className='card-title',
                                    style=h3_style),
                        html.P("The graph below shows the mean absolute SHAP values. "
                            "They describe how much the features on average affected forecasts, regardless of the direction of impact. "
                            "They are calculated as the average of the absolute SHAP values of the local characteristics. "
                            "Black color indicates the trivial features which are the current month and the unemployment rate of the previous month. "
                            ,
                           style =p_style,
                           className='card-text'
                           ),
                     html.Br(),
                        dcc.Loading([dbc.Row(id = 'shap_graph_div_en', justify = 'center')], type = random.choice(spinners))
                        
                            ])
                        ])
                    ],
                
                    [html.Br(),
                        dbc.Card([
                        dbc.CardBody([
                            html.H3('Monthly feature importances', className='card-title',
                                    style=h3_style),
                     html.P("The graph below shows the SHAP values of the features for a selected month. "
                            "They represent the direction and intensity that characterised the forecast for the selected month. "
                            "Green highlights the factors that reduce the monthly change in unemployment and red highlights the features that increase it. "
                            "Black color indicates the trivial features which are the current month and the unemployment rate of the previous month. "
                            "The vertical axis shows the names of the features as well as their values of the corresponding time and the direction of change compared to the previous month with an icon. "
                            "Below the graph is a formula that can be used to calculate monthly forecasts by the use of SHAP values and the constant value produced by the model.",
                             style =p_style,
                            className="card-text"),
                      html.Br(),
                        html.H3('Select a month', style =h3_style),
                                        dcc.Dropdown(id = 'local_shap_month_selection_en',
                                                      options = options, 
                                                      style = {'font-size':16},
                                                      value = list(local_shap_df.index)[0],
                                                      multi=False ),
                                        html.Br(),
                                        
                                        html.Div(dcc.Loading(id = 'local_shap_graph_div_en',
                                                              type = random.choice(spinners))),
                                    
                                    ])
                        ])
                                    
                                    ]]

    else:
        return [html.Div(),html.Div(),html.Div()]
    
    
@callback(

    [Output('forecast_shap_selections_div_en','children'),
      Output('forecast_shap_div_en','children'),
      Output('forecast_local_shap_div_en','children')
      ],
    [Input('forecast_button_en','n_clicks'),
      Input('forecast_shap_data_en','data'),
      State('local_forecast_shap_data_en','data')]    
    
)
def en_update_forecast_shap_results(n_clicks, shap, local_shap_data):
    
    
        
    if shap is None or local_shap_data is None:
        raise PreventUpdate
        
    if n_clicks > 0:
        
        try:
            locale.setlocale(locale.LC_ALL, 'en_US')
        except:
            locale.setlocale(locale.LC_ALL, 'en-US')
        
    
        shap_df = pd.DataFrame(shap)
        
        shap_df = shap_df.set_index(shap_df.columns[0])
        
        
        local_shap_df = pd.DataFrame(local_shap_data)
        local_shap_df = local_shap_df.set_index(local_shap_df.columns[0])
        local_shap_df.index = pd.to_datetime(local_shap_df.index)
        
        options = [{'label':c.strftime('%B %Y'), 'value': c} for c in list(local_shap_df.index)]

         
        return [[
            
                    html.H3('Which features will have the greatest impact?',
                            style = h3_style),
                    html.P("As in the testing section, when predicting unemployment in the coming months, it is also possible to look at which features will matter the most and which features will explain the change in the unemployment rate in a coming month. "
                            "The graph above shows the forecast produced by the model on the assumption that the price indices of the commodities selected by the user change at the selected rate of change on a monthly basis. "
                            "The global and local SHAP values below make it possible to see how user-selected monthly commodity-specific changes affect the forecast. "
                            "This makes it possible to go back to the commodity selection section, adjust the change rate, and try again predicting with several monthly changes. ",
                            style = p_style),
                    # html.A([html.P('Watch a short introduction video about the importance of SHAP values in machine learning.',
                    #                style = p_style)], href="https://www.youtube.com/embed/Tg8aPwPPJ9c", target='_blank'),
                    # html.A([html.P('See also a Non-Technical Guide to Interpreting SHAP Analyses',
                    #                style = p_style)], href="https://www.aidancooper.co.uk/a-non-technical-guide-to-interpreting-shap-analyses/", target='_blank'),
                    html.P("The graph's SHAP values are multiplied by 100 to improve visualization. "
                           "Thereby one SHAP unit corresponds to one hundredth of a percentage point that is used to describe unemployment rate's monthly change. ",
                           style = p_style),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                                html.Div(id = 'forecast_cut_off_div_en'),
                                
                                html.Div(id = 'forecast_cut_off_indicator_en'),
                                
                                ],xs =12, sm=12, md=12, lg=9, xl=9),
                        dbc.Col([
                                dash_daq.BooleanSwitch(id = 'forecast_shap_features_switch_en', 
                                                        label = dict(label = 'Show only the contribution of commodities',
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
                            html.H3('Feature importances', className='card-title',
                                    style=h3_style),
                        html.P("The global SHAP values presented in the graph below can be used to sort the selected commodities and features according to their impact on the model. "
                                "The feature that received the highest global significance value contibuted the most as a whole regardless of the direction of change. ",
                            style =p_style,
                            className='card-text'
                            ),
                      html.Br(),
                        dcc.Loading([dbc.Row(id = 'forecast_shap_graph_div_en', justify = 'center')], type = random.choice(spinners))
                        
                            ])
                        ])
                    ],
                
                    [html.Br(),
                        dbc.Card([
                        dbc.CardBody([
                            html.H3('Monthly feature importances', className='card-title',
                                    style=h3_style),
                      html.P("The graph below shows the local SHAP values of the features, which can be viewed monthly by selecting the desired month from the drop-down menu. "
                            "These values indicate which commodity price changes will decrease or increase unemployment in the selected month. "
                            "Dividing the SHAP value of a commodity by 100 results in percentage points in the change in the price of a commodity that contributes to the monthly change in unemployment. "
                            "Below of the graph there is a formula for calculating the unemployment rate using SHAP values.",
                              style =p_style,
                            className="card-text"),
                      html.Br(),
                        html.H3('Select a month', style =h3_style),
                                        dcc.Dropdown(id = 'forecast_local_shap_month_selection_en',
                                                      options = options, 
                                                      style = {'font-size':16},
                                                      value = list(local_shap_df.index)[0],
                                                      multi=False ),
                                        html.Br(),
                                        
                                        html.Div(dcc.Loading(id = 'forecast_local_shap_graph_div_en',
                                                              type = random.choice(spinners))),
                                    
                                    ])
                        ])
                                    
                                    ]]

    else:
        return [html.Div(),html.Div(),html.Div()]    
    

@callback(

    Output('local_shap_graph_div_en', 'children'),
    [Input('cut_off_en', 'value'),
     Input('shap_features_switch_en','on'),
     Input('local_shap_month_selection_en','value'),
     Input('local_shap_data_en','data')]
    
)
def en_update_local_shap_graph(cut_off, only_commodities, date, local_shap_data):
    
    if local_shap_data is None:
        raise PreventUpdate
    
    try:
        locale.setlocale(locale.LC_ALL, 'en_US')
    except:
        locale.setlocale(locale.LC_ALL, 'en-US')    
    
    local_shap_df = pd.DataFrame(local_shap_data)
    local_shap_df = local_shap_df.set_index(local_shap_df.columns[0])
    local_shap_df.index = pd.to_datetime(local_shap_df.index)
    
    base_value = local_shap_df['base'].values[0]
    local_shap_df = local_shap_df.drop('base',axis=1)
    
    date = pd.to_datetime(date)
    
    
    date_str = date.strftime('%B %Y')
    prev_date = date - pd.DateOffset(months=1)
    prev_str = prev_date.strftime('%B %Y') + ' unemployment rate'
    
    dff = local_shap_df.loc[date,:].copy()
    
  
    
    dff.index  = dff.index.str.replace('month','Current Month').str.replace('prev',prev_str)
    
    feature_values = {f:data_en.loc[date,f] for f in data_en.columns if f not in ['Työttömyysaste', 'change','prev','month','Inflaatio']}
    feature_values[prev_str] = data_en.loc[date,'prev']
    feature_values['Current Month'] = data_en.loc[date,'month']
    
    feature_values_1 = {f:data_en.loc[date-pd.DateOffset(months=1),f] for f in data_en.columns if f not in ['Työttömyysaste', 'change','prev','month','Inflaatio']}
    feature_values_1[prev_str] = data_en.loc[date-pd.DateOffset(months=1),'prev']
    feature_values_1['Current Month'] = data_en.loc[date-pd.DateOffset(months=1),'month']
    
    differences = {f:feature_values[f]-feature_values_1[f] for f in feature_values.keys()}
    changes={}
    
    # How the unemployment rate changed last month?
    changes[prev_str] = data_en.loc[date-pd.DateOffset(months=1),'change']
    
    for d in differences.keys():
        if differences[d] >0:
            changes[d]='🔺'
        elif differences[d] <0:
            changes[d] = '🔽'
        else:
            changes[d] = '⇳'
          
    
    if only_commodities:
        dff = dff.loc[[i for i in dff.index if i not in ['Current Month', prev_str]]]
    
    
    dff = dff.sort_values(ascending = False)
    
   
    df = pd.Series(dff.iloc[cut_off+1:].copy().sum())
    
    
    
    # df.index = df.index.astype(str).str.replace('0', 'Muut {} piirrettä'.format(len(dff.iloc[cut_off+1:,:])))
    df.index = ['Other {} features'.format(len(dff.iloc[cut_off+1:]))]
    
    
    dff = pd.concat([dff.head(cut_off).copy(),df])
    dff = dff.loc[dff.index != 'Other 0 features']
    dff.index = dff.index.str.replace('_','')
    

    height = graph_height +200 + 10*len(dff)
    
    dff = np.round(dff*100,2)
   
    # dff = dff.sort_values()

    
    return html.Div([dcc.Graph(id = 'local_shap_graph_en',
                     config = config_plots_en,
                         figure = go.Figure(data=[go.Bar(y =['{} ({} {})'.format(i, feature_values[i],changes[i]) if i in feature_values.keys() else i for i in dff.index], 
                      x = dff.values,
                      orientation='h',
                      name = '',
                      
                      # marker_color = ['cyan' if i not in ['Month',prev_str] else 'black' for i in dff.index],
                       marker = dict(color = list(map(en_set_color,dff.index,dff.values))),
                      text = dff.values,
                      hovertemplate = ['<b>{}</b><br><b>  SHAP value</b>: {}<br><b>  Value on the current month</b>: {} {}<br><b>  Value on the previous month</b>: {}'.format(i,dff.loc[i], feature_values[i],changes[i],round(feature_values_1[i],2)) if i in feature_values.keys() else '{}: {}'.format(i,dff.loc[i]) for i in dff.index],
                          textfont = dict(
                               family='Cadiz Semibold', 
                              size = 16))],
         layout=go.Layout(title = dict(text = 'Local Feature Importances<br>SHAP values: '+date_str,
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
                                                        xaxis = dict(title=dict(text = 'SHAP value',
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
                                                        yaxis = dict(title=dict(text = 'Feature: 🔺 = increase, 🔽 = decrease, ⇳ = same as previous month',
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
                     html.P('Forecast ≈ Previous predicted unemployment rate + [ {} + SUM( SHAP values ) ] / 100'.format(round(100*base_value,2)))
                     ])

@callback(

    Output('forecast_local_shap_graph_div_en', 'children'),
    [Input('forecast_cut_off_en', 'value'),
     Input('forecast_shap_features_switch_en','on'),
     Input('forecast_local_shap_month_selection_en','value'),
     Input('local_forecast_shap_data_en','data'),
     State('forecast_data_en','data')]
    
)
def en_update_forecast_local_shap_graph(cut_off, only_commodities, date, local_shap_data, forecast_data):
    
    if local_shap_data is None:
        raise PreventUpdate
    
    try:
        locale.setlocale(locale.LC_ALL, 'en_US')
    except:
        locale.setlocale(locale.LC_ALL, 'en-US')    
        
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
    prev_str = prev_date.strftime('%B %Y') + ' unemployment rate'
    
    dff = local_shap_df.loc[date,:].copy()
    
  
    
    dff.index  = dff.index.str.replace('month','Current Month').str.replace('prev',prev_str)
    
    feature_values = {f:forecast_data.loc[date,f] for f in forecast_data.columns if f not in ['Työttömyysaste', 'change','prev','month','Inflaatio']}
    feature_values[prev_str] = forecast_data.loc[date,'prev']
    feature_values['Current Month'] = forecast_data.loc[date,'month']
    
    try:
        feature_values_1 = {f:forecast_data.loc[date-pd.DateOffset(months=1),f] for f in forecast_data.columns if f not in ['Työttömyysaste', 'change','prev','month','Inflaatio']}
        feature_values_1[prev_str] = forecast_data.loc[date-pd.DateOffset(months=1),'prev']
        feature_values_1['Current Month'] = forecast_data.loc[date-pd.DateOffset(months=1),'month']
    except:
        feature_values_1 = {f:data_en.loc[date-pd.DateOffset(months=1),f] for f in data_en.columns if f not in ['Työttömyysaste', 'change','prev','month','Inflaatio']}
        feature_values_1[prev_str] = data_en.loc[date-pd.DateOffset(months=1),'prev']
        feature_values_1['Current Month'] = data_en.loc[date-pd.DateOffset(months=1),'month']
    
    differences = {f:feature_values[f]-feature_values_1[f] for f in feature_values.keys()}
    changes={}
    
    # How the unemployment rate changed last month?
    try:
        changes[prev_str] = forecast_data.loc[date-pd.DateOffset(months=1),'change']
    except:
        changes[prev_str] = data_en.loc[date-pd.DateOffset(months=1),'change']
    
    for d in differences.keys():
        if differences[d] >0:
            changes[d]='🔺'
        elif differences[d] <0:
            changes[d] = '🔽'
        else:
            changes[d] = '⇳'
          
    
    if only_commodities:
        dff = dff.loc[[i for i in dff.index if i not in ['Current Month', prev_str]]]
    
    
    dff = dff.sort_values(ascending = False)
    
   
    df = pd.Series(dff.iloc[cut_off+1:].copy().sum())
    
    
    
    # df.index = df.index.astype(str).str.replace('0', 'Muut {} piirrettä'.format(len(dff.iloc[cut_off+1:,:])))
    df.index = ['Other {} features'.format(len(dff.iloc[cut_off+1:]))]
    
    
    dff = pd.concat([dff.head(cut_off).copy(),df])
    dff = dff.loc[dff.index != 'Other 0 features']
    dff.index = dff.index.str.replace('_','')
    

    height = graph_height +200 + 10*len(dff)
    
    dff = np.round(dff*100,2)
   
    # dff = dff.sort_values()

    
    return html.Div([dcc.Graph(id = 'local_shap_graph_en',
                     config = config_plots_en,
                         figure = go.Figure(data=[go.Bar(y =['{} ({} {})'.format(i, round(feature_values[i],2),changes[i]) if i in feature_values.keys() else '{}: {}'.format(i,dff.loc[i]) for i in dff.index], 
                      x = dff.values,
                      orientation='h',
                      name = '',
                      
                      # marker_color = ['cyan' if i not in ['Month',prev_str] else 'black' for i in dff.index],
                       marker = dict(color = list(map(en_set_color,dff.index,dff.values))),
                      text = dff.values,
                      hovertemplate = ['<b>{}</b><br><b>  SHAP value</b>: {}<br><b>  Value on the current month</b>: {} {}<br><b>  Value on the previous month</b>: {}'.format(i,dff.loc[i], round(feature_values[i],2),changes[i],round(feature_values_1[i],2)) if i in feature_values.keys() else '{}: {}'.format(i,dff.loc[i]) for i in dff.index],
                          textfont = dict(
                               family='Cadiz Semibold', 
                              size = 16))],
         layout=go.Layout(title = dict(text = 'Local Feature Importances<br>SHAP values: '+date_str,
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
                                                        xaxis = dict(title=dict(text = 'SHAP value',
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
                                                        yaxis = dict(title=dict(text = 'Feature: 🔺 = increase, 🔽 = decrease, ⇳ = same as previous month',
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
                     html.P('Forecast ≈ Previous predicted unemployment rate + [ {} + SUM( SHAP values ) ] / 100'.format(round(100*base_value,2)))
                     ])
    
@callback(

    Output('cut_off_indicator_en','children'),
    [Input('cut_off_en','value')]    
    
)
def en_update_cut_off_indicator(cut_off):
    return [html.P('You chose {} features.'.format(cut_off).replace(' 1 features',' one feature'), style = p_center_style)]


@callback(

    Output('forecast_cut_off_indicator_en','children'),
    [Input('forecast_cut_off_en','value')]    
    
)
def en_update_forecast_cut_off_indicator(cut_off):
    return [html.P('You chose {} features.'.format(cut_off).replace(' 1 features',' one feature'), style = p_center_style)]
    
@callback(

    Output('cut_off_div_en','children'),
    [Input('shap_data_en','data')]    
    
)
def en_update_shap_slider(shap):
    if shap is None:
        raise PreventUpdate

    shap_df = pd.DataFrame(shap)
    
    
    return [html.P('Select how many features are shown in the graph below.',
                       style = p_center_style),
                dcc.Slider(id = 'cut_off_en',
                   min = 1, 
                   max = len(shap_df),
                   value = {True:len(shap_df), False: int(math.ceil(.2*len(shap_df)))}[len(shap_df)<=25],
                   step = 1,
                   marks=None,
                   tooltip={"placement": "top", "always_visible": True},
                   )]

@callback(

    Output('forecast_cut_off_div_en','children'),
    [Input('forecast_shap_data_en','data')]    
    
)
def en_forecast_update_shap_slider(shap):
    if shap is None:
        raise PreventUpdate

    shap_df = pd.DataFrame(shap)
    
    
    return [html.P('Select how many features are shown in the graph below.',
                        style = p_center_style),
                dcc.Slider(id = 'forecast_cut_off_en',
                    min = 1, 
                    max = len(shap_df),
                    value = {True:len(shap_df), False: int(math.ceil(.2*len(shap_df)))}[len(shap_df)<=25],
                    step = 1,
                    marks=None,
                    tooltip={"placement": "top", "always_visible": True},
                    )]

@callback(

    Output('shap_graph_div_en', 'children'),
    [Input('cut_off_en', 'value'),
     Input('shap_features_switch_en','on'),
     State('shap_data_en','data')]
    
)
def en_update_shap_graph(cut_off, only_commodities, shap):
    
    if shap is None:
        raise PreventUpdate
        
    
    shap_df = pd.DataFrame(shap)
    shap_df = shap_df.set_index(shap_df.columns[0])
    shap_df.index = shap_df.index.str.replace('Kuukausi','Month')
    shap_df.index = shap_df.index.str.replace('Edellisen kuukauden työttömyysaste','Previous Unemployment Rate')
  
    
    
    if only_commodities:
        shap_df = shap_df.loc[[i for i in shap_df.index if i not in ['Month', 'Previous Unemployment Rate']]]
    
    
    shap_df = shap_df.sort_values(by='SHAP', ascending = False)
    
   
    df = pd.DataFrame(shap_df.iloc[cut_off+1:,:].sum())
    df = df.T
    df.index = df.index.astype(str).str.replace('0', 'Other {} features'.format(len(shap_df.iloc[cut_off+1:,:])))
    
    
    shap_df = pd.concat([shap_df.head(cut_off),df])
    shap_df = shap_df.loc[shap_df.index != 'Other 0 features']
    shap_df.index = shap_df.index.str.replace('_','')

    height = graph_height +200 + 10*len(shap_df)
    
    
    return dcc.Graph(id = 'shap_graph_en',
                     config = config_plots_en,
                         figure = go.Figure(data=[go.Bar(y =shap_df.index, 
                      x = np.round(100*shap_df.SHAP,2),
                      orientation='h',
                      name = '',
                      marker_color = ['aquamarine' if i not in ['Month','Previous Unemployment Rate'] else 'black' for i in shap_df.index],
                      # marker = dict(color = 'turquoise'),
                      text = np.round(100*shap_df.SHAP,2),
                      hovertemplate = '<b>%{y}</b>: %{x}',
                          textfont = dict(
                               family='Cadiz Semibold', 
                              size = 20))],
         layout=go.Layout(title = dict(text = 'Global Feature Importances<br>Mean |SHAP values|',
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
                                                        xaxis = dict(title=dict(text = 'Mean |SHAP value|',
                                                                                font=dict(
                                                                                    size=18, 
                                                                                    family = 'Cadiz Semibold'
                                                                                    )),
                                                                     automargin=True,
                                                                     tickfont = dict(
                                                                         family = 'Cadiz Semibold', 
                                                                          size = 16
                                                                         )),
                                                        yaxis = dict(title=dict(text = 'Feature',
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

    Output('forecast_shap_graph_div_en', 'children'),
    [Input('forecast_cut_off_en', 'value'),
     Input('forecast_shap_features_switch_en','on'),
     State('forecast_shap_data_en','data')]
    
)
def en_update_forecast_shap_graph(cut_off, only_commodities, shap):
    
    if shap is None:
        raise PreventUpdate
        
    
    shap_df = pd.DataFrame(shap)
    shap_df = shap_df.set_index(shap_df.columns[0])
    shap_df.index = shap_df.index.str.replace('Kuukausi','Month')
    shap_df.index = shap_df.index.str.replace('Edellisen kuukauden työttömyysaste','Previous Unemployment Rate')
  
    
    
    if only_commodities:
        shap_df = shap_df.loc[[i for i in shap_df.index if i not in ['Month', 'Previous Unemployment Rate']]]
    
    
    shap_df = shap_df.sort_values(by='SHAP', ascending = False)
    
   
    df = pd.DataFrame(shap_df.iloc[cut_off+1:,:].sum())
    df = df.T
    df.index = df.index.astype(str).str.replace('0', 'Other {} features'.format(len(shap_df.iloc[cut_off+1:,:])))
    
    
    shap_df = pd.concat([shap_df.head(cut_off),df])
    shap_df = shap_df.loc[shap_df.index != 'Other 0 features']
    shap_df.index = shap_df.index.str.replace('_','')

    height = graph_height +200 + 10*len(shap_df)
    
    
    return dcc.Graph(id = 'shap_graph_en',
                     config = config_plots_en,
                         figure = go.Figure(data=[go.Bar(y =shap_df.index, 
                      x = np.round(100*shap_df.SHAP,2),
                      orientation='h',
                      name = '',
                      marker_color = ['aquamarine' if i not in ['Month','Previous Unemployment Rate'] else 'black' for i in shap_df.index],
                      # marker = dict(color = 'turquoise'),
                      text = np.round(100*shap_df.SHAP,2),
                      hovertemplate = '<b>%{y}</b>: %{x}',
                          textfont = dict(
                               family='Cadiz Semibold', 
                              size = 20))],
         layout=go.Layout(title = dict(text = 'Global Feature Importances<br>Mean |SHAP values|',
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
                                                        xaxis = dict(title=dict(text = 'Mean |SHAP value|',
                                                                                font=dict(
                                                                                    size=18, 
                                                                                    family = 'Cadiz Semibold'
                                                                                    )),
                                                                     automargin=True,
                                                                     tickfont = dict(
                                                                         family = 'Cadiz Semibold', 
                                                                          size = 16
                                                                         )),
                                                        yaxis = dict(title=dict(text = 'Feature',
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
    Output("forecast_download_en", "data"),
    [Input("forecast_download_button_en", "n_clicks")],
    [State('forecast_data_en','data'),
     State('method_selection_results_en','data'),
     State('change_weights_en','data'),     
      State('forecast_shap_data_en','data'),
      State('local_forecast_shap_data_en','data'),
     ]
    
    
)
def en_download_forecast_data(n_clicks, df, method_selection_results, weights_dict, shap_data, local_shap_data):
    
    if n_clicks > 0:
        
        
        df = pd.DataFrame(df).set_index('Aika').copy()
        df.index = pd.to_datetime(df.index)
        df.index.name = 'Time'
        forecast_size = len(df)
        n_feat = df.n_feat.values[0]
        df.drop('n_feat',axis=1,inplace=True)
        
        df = df.rename(columns = {'change':'Forecasted monthly change (percentage units)',
                                  'month':'Month',
                                  'prev': 'Previous forecast',
                                  'Työttömyysaste': 'Unemployment Rate (forecast)'})
        
        
        features = sorted(list(weights_dict.keys()))
        
        weights_df = pd.DataFrame([weights_dict]).T
        weights_df.index.name = 'Commodity'
        weights_df.columns = ['Expected average monthly change (%)']
        
        model_name = method_selection_results['model']
             
        pca = method_selection_results['pca']
        explained_variance = method_selection_results['explained_variance']
        hyperparam_grid = method_selection_results['hyperparam_grid']
        
        if pca:
            metadict = {
                        'Principal Component Analysis' : pca,
                        'Explained Variance': str(int(100*explained_variance))+'%',
                        'Principal Components': n_feat,
                        'Model': model_name,
                        'Number of applied features':len(features),
                        'Commodities' : ',\n'.join(features),
                        'Forecast length': str(forecast_size)+' months'
                        
                        }

        else:
            metadict = {
                            'Principal Component Analysis' : pca,
                            'Model': model_name,
                            'Number of applied features':len(features),
                            'Commodities' : ',\n'.join(features),
                            'Forecast length': str(forecast_size)+' months'
                            }
        
        metadata = pd.DataFrame([metadict]).T
        metadata.index.name = ''
        metadata.columns = ['Value']
        
        hyperparam_df = pd.DataFrame([hyperparam_grid]).T
        hyperparam_df.index.name = 'Hyperparameter'
        hyperparam_df.columns = ['Value']   
        hyperparam_df['Value'] = hyperparam_df['Value'].astype(str)
        
        shap_df = pd.DataFrame(shap_data)
        shap_df = shap_df.set_index(shap_df.columns[0])
        shap_df.index.name = 'Feature'
        shap_df.SHAP = np.round(100*shap_df.SHAP,2)
        shap_df.index = shap_df.index.str.replace('Kuukausi', 'Month')
        shap_df.index = shap_df.index.str.replace('Edellisen kuukauden työttömyysaste', 'Previous Unemployment Rate')
        shap_df.index = shap_df.index.str.replace('_','')
        
        local_shap_df = pd.DataFrame(local_shap_data)
        local_shap_df = local_shap_df.set_index(local_shap_df.columns[0])
        local_shap_df.index = pd.to_datetime(local_shap_df.index)
        local_shap_df.index.name = 'Time'
        local_shap_df = local_shap_df.rename(columns = {'month':'Month',
                                  'prev': 'Previous Unemployment Rate'})
        local_shap_df = local_shap_df.multiply(100, axis=1)
        local_shap_df.columns = local_shap_df.columns.str.replace('_','')
        local_shap_df.drop('base',axis=1,inplace=True)

  
        data_ = data_en.copy().rename(columns={'change':'Change (percentage units)',
                                      'prev':'Previous unemployment rate -% ',
                                      'month':'Month',
                                      'Työttömyysaste':'Unemployment Rate',
                                      'Inflaatio':'Inflation'})
        data_.index.name = 'Time'
        
        
        xlsx_io = io.BytesIO()
        writer = pd.ExcelWriter(xlsx_io, engine='xlsxwriter')
        
        
        
        data_.to_excel(writer, sheet_name= 'Data')
        df.to_excel(writer, sheet_name= 'Forecast data')
        weights_df.to_excel(writer, sheet_name= 'Index changes')
        hyperparam_df.to_excel(writer, sheet_name= 'Hyperparameters')
        shap_df.to_excel(writer, sheet_name= 'Feature importances')
        local_shap_df.to_excel(writer, sheet_name= 'Monthly importances')
        metadata.to_excel(writer, sheet_name= 'Metadata')
        
        
        workbook = writer.book
        workbook.set_properties(
        {
            "title": "Skewed Phillips",
            "subject": "Forecast Results",
            "author": "Tuomas Poukkula",
            "company": "Gofore Ltd.",
            "keywords": "XAI, Predictive analytics",
            "comments": "Check out the app on: https://skewedphillips.herokuapp.com"
        }
        )
        writer.save()
        

        
        output = xlsx_io.getvalue()

        return dcc.send_bytes(output, 'Forecast data with {} commodities '.format(len(features)).replace(' 1 commodities',' one commodity')+datetime.now().strftime('%d_%m_%Y')+'.xlsx')
        
@callback(
    Output("test_download_en", "data"),
    [Input("test_download_button_en", "n_clicks"),
    State('test_data_en','data'),
    State('method_selection_results_en','data'),
    State('change_weights_en','data'),
    State('shap_data_en','data'),
    State('local_shap_data_en','data')
    ]
    
)
def en_download_test_data(n_clicks, 
                       df, 
                       method_selection_results, 
                        weights_dict, 
                       shap_data,
                       local_shap_data):
    
    if n_clicks > 0:
        
        df = pd.DataFrame(df).set_index('Aika').copy()
        df.index = pd.to_datetime(df.index)
        df.index.name = 'Time'
        mape = df.mape.values[0]
        test_size = len(df)
        n_feat = df.n_feat.values[0]
        df.drop('n_feat',axis=1,inplace=True)
        df.drop('mape',axis=1,inplace=True)
        df = df.rename(columns = {'change':'Monthly change (percentage units)',
                                  'month':'Month',
                                  'Ennustettu muutos':'Forecasted monthly change (percentage units)',
                                  'Ennuste':'Forecasted unemployment rate',
                                  'Työttömyysaste':'Unemployment Rate (%)',
                                  'prev': 'Previous forecast'})
        
        
        features = sorted(list(weights_dict.keys()))
        
        model_name = method_selection_results['model']
             
        pca = method_selection_results['pca']
        explained_variance = method_selection_results['explained_variance']
        hyperparam_grid = method_selection_results['hyperparam_grid']
        
        if pca:
            metadict = {'MAPE': str(round(100*mape,2))+'%',
                        'Principal Component Analysis' : pca,
                        'Explained Variance': str(int(100*explained_variance))+'%',
                        'Principal components': n_feat,
                        'Model': model_name,
                        'Number of applied features':len(features),
                        'Commodities' : ',\n'.join(features),
                        'Test length': str(test_size)+' months'
                        }
        else:
            metadict = {'MAPE': str(round(100*mape,2))+'%',
                            'Principal Component Analysis' : pca,
                            'Model': model_name,
                            'Number of applied features':len(features),
                            'Commodities' : ',\n'.join(features),
                            'Test length': str(test_size)+' months'
                            }
        
        metadata = pd.DataFrame([metadict]).T
        metadata.index.name = ''
        metadata.columns = ['Value']
        
        hyperparam_df = pd.DataFrame([hyperparam_grid]).T
        hyperparam_df.index.name = 'Hyperparameter'
        hyperparam_df.columns = ['Value']   
        hyperparam_df['Value'] = hyperparam_df['Value'].astype(str)
        
        shap_df = pd.DataFrame(shap_data)
        shap_df = shap_df.set_index(shap_df.columns[0])
        shap_df.index.name = 'Feature'
        shap_df.SHAP = np.round(100*shap_df.SHAP,2)
        shap_df.index = shap_df.index.str.replace('Kuukausi', 'Month')
        shap_df.index = shap_df.index.str.replace('Edellisen kuukauden työttömyysaste', 'Previous Unemployment Rate')
        shap_df.index = shap_df.index.str.replace('_','')
        
        local_shap_df = pd.DataFrame(local_shap_data)
        local_shap_df = local_shap_df.set_index(local_shap_df.columns[0])
        local_shap_df.index = pd.to_datetime(local_shap_df.index)
        local_shap_df.index.name = 'Time'
        local_shap_df = local_shap_df.rename(columns = {'month':'Month',
                                  'prev': 'Previous Unemployment Rate'})
        local_shap_df = local_shap_df.multiply(100, axis=1)
        local_shap_df.columns = local_shap_df.columns.str.replace('_','')
        local_shap_df.drop('base',axis=1,inplace=True)
        
        xlsx_io = io.BytesIO()
        writer = pd.ExcelWriter(xlsx_io, engine='xlsxwriter')
        
        
        df.to_excel(writer, sheet_name= 'Test data')
        metadata.to_excel(writer, sheet_name= 'Metadata')
        hyperparam_df.to_excel(writer, sheet_name= 'Model hyperparameters')
        shap_df.to_excel(writer, sheet_name= 'Feature importances')
        local_shap_df.to_excel(writer, sheet_name= 'Monthly importances')
        
        workbook = writer.book
        workbook.set_properties(
        {
            "title": "Skewed Phillips",
            "subject": "Test Results",
            "author": "Tuomas Poukkula",
            "company": "Gofore Ltd.",
            "keywords": "XAI, Predictive analytics",
            "comments": "Check out the app on: https://skewedphillips.herokuapp.com"
        }
        )

        writer.save()
        

        
        output = xlsx_io.getvalue()

        return dcc.send_bytes(output, 'Test results with {} commodities '.format(len(features)).replace(' 1 commodities',' one commodity')+datetime.now().strftime('%d_%m_%Y')+'.xlsx')


@callback(

    Output('test_graph_div_en', 'children'),
    
      [Input('test_chart_type_en','value')],
      [State('test_data_en','data')]
    
)
def en_update_test_chart_type(chart_type,df):
    
    
    
    df = pd.DataFrame(df).set_index('Aika')
    df.index = pd.to_datetime(df.index)

    
    df = df.reset_index().drop_duplicates(subset='Aika', keep='last').set_index('Aika')[['Työttömyysaste','Ennuste','mape']].dropna(axis=0)

    return [dcc.Graph(id ='test_graph_en', 
                     figure = en_plot_test_results(df,chart_type), 
                     config = config_plots_en)     ]



@callback(

    Output('forecast_graph_div_en', 'children'),
    
      [Input('chart_type_en','value')],
      [State('forecast_data_en','data')]
    
)
def en_update_forecast_chart_type(chart_type,df):
    
    df = pd.DataFrame(df).set_index('Aika')
    df.index = pd.to_datetime(df.index)

    return dcc.Graph(id = 'forecast_graph_en',
                    figure = en_plot_forecast_data(df, chart_type = chart_type), 
                    config = config_plots_en),

@callback(

    Output('ev_placeholder_en', 'style'),
    [Input('pca_switch_en', 'on')]
)    
def en_add_ev_slider(pca):
    
    return {False: {'margin' : '5px 5px 5px 5px', 'display':'none'},
           True: {'margin' : '5px 5px 5px 5px'}}[pca]

@callback(

    Output('ev_slider_update_en', 'children'),
    [Input('pca_switch_en', 'on'),
    Input('ev_slider_en', 'value')]

)
def en_update_ev_indicator(pca, explained_variance):
    
    return {False: [html.Div([html.P('You chose {} % explained variance.'.format(int(100*explained_variance)),
                                                               style = p_center_style)
                                                       ], style = {'display':'none'}
                                                      )],
            True: [html.Div([html.P('You chose {} % explained variance.'.format(int(100*explained_variance)),
                                                               style = p_center_style)
                                                       ]
                                                      )]}[pca]



@callback(
    Output('feature_selection_en','value'),
    [Input('select_all_en', 'on'),
    Input('feature_selection_en','options')]
)
def en_update_feature_list(on,options):
       
        
    if on:
        return [f['label'] for f in options]
    else:
        raise PreventUpdate
        
@callback(
    
    Output('select_all_en','on'),
    [Input('feature_selection_en','value'),
     State('feature_selection_en','options')]
    
)
def en_update_select_all_on(features,options):
    
    return len(features) == len(options)

@callback(
    [
     
     Output('select_all_en','label'),
     Output('select_all_en','disabled')
     ],
    [Input('select_all_en', 'on')]
)    
def en_update_switch(on):
    
    if on:
        return {'label':'Everything is selected. You can remove commodities by clicking the crosess on the list.',
                       'style':{'text-align':'center', 'font-size':p_font_size,
                                #'font-family':'Cadiz Semibold'
                                }
                      },True
    

    else:
        return dict(label = 'Select all',style = {'font-size':p_font_size, 
                                                      # #'fontFamily':'Cadiz Semibold'
                                                      }),False



@callback(

    Output('test_button_div_en','children'),
    [

     Input('features_values_en','data')
     ]    
    
)
def en_add_test_button(features_values):
    
    if features_values is None:
        raise PreventUpdate 
        
        
    
    elif len(features_values) == 0:
        return [html.P('Select commodities first',
                       style = p_style)]
    
    else:
               
        
        return dbc.Button('Test',
                           id='test_button_en',
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
    Output('test_size_indicator_en','children'),
    [Input('test_slider_en','value')]
)
def en_update_test_size_indicator(value):
    
    return [html.Br(),html.P('You selected {} months as test data.'.format(value),
                             style = p_center_style)]

@callback(
    Output('forecast_slider_indicator_en','children'),
    [Input('forecast_slider_en','value')]
)
def en_update_forecast_size_indicator(value):
    
    return [html.Br(),html.P('You selected {} months for forecasting.'.format(value),
                             style = p_center_style)]




@callback(

    Output('timeseries_selection_en', 'children'),
    [
     
     Input('features_values_en','data')
     ]    
    
)
def en_update_timeseries_selections(features_values):
    
    features = sorted(list(features_values.keys()))
    
    return [
            html.Br(),
            html.H3('View commodity index time series',
                    style =h3_style),
            
            html.P('Use this graph to view the index development of commodities on a monthly basis. This will make it easier to better assess what kind of inflation expectations to feed into the forecast.',
                   style = p_style),
            html.H3('Select a commodity',style = h3_style),
            dcc.Dropdown(id = 'timeseries_selection_dd_en',
                        options = [{'value':feature, 'label':feature} for feature in features],
                        value = [features[0]],
                        style = {
                            'font-size':"0.9rem", 
                            #'font-family':'Cadiz Book',
                            'color': 'black'},
                        multi = True)
            ]


@callback(

    Output('timeseries_en', 'children'),
    [Input('timeseries_selection_dd_en', 'value')]    
    
)
def en_update_time_series(values):
    
    traces = [go.Scatter(x = data_en.index, 
                         y = data_en[value],
                         showlegend=True,   
                         hovertemplate = '%{x}'+'<br>%{y}',
                         name = ' '.join(value.split()[1:]),
                         mode = 'lines+markers') for value in values]
    return html.Div([dcc.Graph(figure=go.Figure(data=traces,
                                      layout = go.Layout(title = dict(text = 'Indexes of<br>selected commodities',
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
                                                         hoverlabel = dict(font=dict(size=18,
                                                                                      family='Cadiz Book'
                                                                                     )),
                                                         xaxis = dict(title=dict(text = 'Time',
                                                                                 font=dict(
                                                                                     size=18, 
                                                                                     family = 'Cadiz Semibold'
                                                                                     )),
                                                                      automargin=True,
                                                                      tickfont = dict(
                                                                          family = 'Cadiz Semibold', 
                                                                           size = 16
                                                                          )),
                                                         yaxis = dict(title=dict(text = 'Point figure (base year = 2010)',
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
                     config = config_plots_en
                     )])




@callback(
    
    Output('forecast_button_div_en','children'),
    [
     
     Input('features_values_en','data')
     ]   
    
)
def en_add_predict_button(features_values):
    
    if features_values is None:
        raise PreventUpdate 
    
    elif len(features_values) == 0:
        return [html.P('Select commodities first',
                       style = p_style)]
    
    
        
    else:
        return [dbc.Button('Forecast',
                   id='forecast_button_en',
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
                html.Div(id = 'forecast_download_button_div_en',style={'textAlign':'center'})]





@callback(

    Output('slider_prompt_div_en','children'),
    [Input('slider_en', 'value'),
     State('averaging_en', 'on')]    
    
)
def en_update_slider_prompt(value, averaging):
    
        
    if averaging:
    
        return [html.Br(),html.P('You selected the average of the latest {} months.'.format(value),
                      style = p_center_style),
                html.Br(),
                html.P('You can still adjust individual values on the input boxes.',
                       style = p_style)]
    else:
        return [html.Br(),html.P('You selected the estimated monthly change to be  {} %.'.format(value),
                      style = p_center_style),
                html.Br(),
                html.P('You can still adjust individual values on the input boxes.',
                       style = p_style)]
        
 

@callback(

    Output('corr_selection_div_en', 'children'),
    [

     Input('features_values_en','data')
     ]
    
)
def en_update_corr_selection(features_values):
    
    features = sorted(list(features_values.keys()))

    return html.Div([
            html.Br(),
            html.H3('View the relationship between the price index of the selected commodity and the unemployment rate',
                    style=h3_style),
            
            html.P('Use this graph to view the relationship and correlation between the price index of the selected commodity and the unemployment rate or monthly change. In theory, a good predictive feature correlates strongly with the predictable variable. '
                   'For better visualisation, the trendline is shown in the graph only when one commodity is selected.',
                   style = p_style),
            html.P("(Continues after the chart)",style={
                        'font-style':'italic',
                        'font-size':p_font_size,
                       'text-align':'center'}
                ),
        html.H3('Select a commodity', style = h3_style),
        dcc.Dropdown(id = 'corr_feature_en',
                        multi = True,
                        # clearable=False,
                        options = [{'value':feature, 'label':feature} for feature in features],
                        value = [features[0]],
                        style = {'font-size':"0.9rem", 
                                 #'font-family':'Cadiz Book'
                                 },
                        placeholder = 'Select a commodity')
        ]
        )

@callback(

    Output('feature_corr_selection_div_en', 'children'),
    [

     Input('features_values_en','data')
     ]
    
)
def en_update_feature_corr_selection(features_values):
    
    features = sorted(list(features_values.keys()))
    
    return html.Div([
                html.Br(),
                html.H3('View commodity relations',
                        style=h3_style),
                html.Br(),
                html.P('Use this graph to view the relationships and correlations between commodities. If the correlation between two commodities is strong, the forecast can be improved by removing the other from the forecast features.',
                       style = p_style),
                html.P("(Continues after the chart)",style={
                            'font-style':'italic',
                            'font-size':p_font_size,
                           'text-align':'center'}
                    ),
        
        dbc.Row(justify = 'center',children=[
            dbc.Col([
                html.H3('Select a commodity',style=h3_style),
                dcc.Dropdown(id = 'f_corr1_en',
                                multi = False,
                                options = [{'value':feature, 'label':feature} for feature in features],
                                value = features[0],
                                style = {'font-size':"0.9rem", 
                                         #'font-family':'Cadiz Book'
                                         },
                                placeholder = 'Select a commodity')
        ],xs =12, sm=12, md=12, lg=6, xl=6),
        html.Br(),
            dbc.Col([
                html.H3('Select another commodity',
                        style=h3_style
                        ),
                dcc.Dropdown(id = 'f_corr2_en',
                                multi = False,
                                options = [{'value':feature, 'label':feature} for feature in features],
                                value = features[-1],
                                style = {'font-size':"0.9rem", 
                                         #'font-family':'Cadiz Book'
                                         },
                                placeholder = 'Select another commodity')
            ],xs =12, sm=12, md=12, lg=6, xl=6)
        ])
        ])



@callback(

    Output('corr_div_en', 'children'),
    [Input('f_corr1_en','value'),
     Input('f_corr2_en','value')]    
    
)
def en_update_feature_correlation_plot(value1, value2):
    
    if value1 is None or value2 is None:
        raise PreventUpdate 
        
        
    a, b = np.polyfit(np.log(data_en[value1]), data_en[value2], 1)

    y = a * np.log(data_en[value1]) +b 

    df = data_en.copy()
    df['log_trend'] = y
    df = df.sort_values(by = 'log_trend')    
    
    corr_factor = round(sorted(data_en[[value1,value2]].corr().values[0])[0],2)
    
    traces = [go.Scatter(x = data_en[value1], 
                         y = data_en[value2], 
                         mode = 'markers',
                         name = ' ',#.join(value.split()[1:]),
                         showlegend=False,
                         marker = dict(color = 'purple', size = 10),
                         marker_symbol='star',
                         hovertemplate = "<b>Index values:</b><br><b>{}</b>:".format(' '.join(value1.split()[1:]))+" %{x}"+"<br><b>"+"{}".format(' '.join(value2.split()[1:]))+"</b>: %{y}"
                         ),
                go.Scatter(x = df[value1], 
                            y = df['log_trend'], 
                            name = 'Logarithmic trendline', 
                            mode = 'lines',
                            line = dict(width=5),                            
                            showlegend=True,
                            hovertemplate=[], 
                            marker = dict(color = 'orange'))
             ]
    
  
    
    return [
            html.Div([dcc.Graph(figure = go.Figure(data = traces,
          layout = go.Layout(title = dict(text = '<b>{}</b> vs.<br><b>{}</b><br>(Correlation: {})'.format(' '.join(value1.split()[1:]), ' '.join(value2.split()[1:]), corr_factor), 
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
                            xaxis= dict(title = dict(text='{} (Point figure)'.format(' '.join(value1.split()[1:])), 
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
                            yaxis = dict(title = dict(text='{} (Point figure)'.format(' '.join(value2.split()[1:])), 
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
                      config = config_plots_en)])]




@callback(
    
    Output('commodity_unemployment_div_en','children'),
    [Input('corr_feature_en','value'),
     Input('eda_y_axis_en','value')]
    
)
def en_update_commodity_unemployment_graph(values, label):
    
    
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
    
    label_str = {'Työttömyysaste': 'Unemployment rate (%)',
                 'change': 'Monthly unemployment rate change (% units)'}[label]     
            
    traces = [go.Scatter(x = data_en[value], 
                         y = data_en[label], 
                         mode = 'markers',
                         name = ' '.join(value.split()[1:]).replace(',',',<br>')+' ({})'.format(round(sorted(data_en[[label, value]].corr()[value].values)[0],2)),
                         showlegend=True,
                         marker = dict(size=10),
                         marker_symbol = random.choice(symbols),
                         hovertemplate = "<b>{}</b>:".format(value)+" %{x}"+"<br><b>"+label_str+"</b>: %{y}"+"<br>(Correlation: {:.2f})".format(sorted(data_en[[label, value]].corr()[value].values)[0])) for value in values]
    
    if len(values)==1:
        data_ = data_en[(data_en[label].notna())].copy()
        
        for value in values:
        
            a, b = np.polyfit(np.log(data_[value]), data_[label], 1)
    
            y = a * np.log(data_[value].values) +b 
            
    
            df = data_[[value]].copy()
            df['log_inflation'] = y
            df = df.sort_values(by = 'log_inflation')
            traces.append(go.Scatter(x=df[value], 
                                      y =df['log_inflation'],
                                      showlegend=True,
                                      name = 'Logarithmic<br>trendline',
                                      line = dict(width=5),
                                      hovertemplate=[]))
    
    return [dcc.Graph(figure = go.Figure(data = traces,
          layout = go.Layout(title = dict(text = 'Selected commodities vs.<br>'+label_str, 
                                          x=.5, 
                                          font=dict(
                                              family='Cadiz Semibold',
                                               size=20
                                              )
                                          ),
                            xaxis= dict(title = dict(text='Point figure', 
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
          ),config = config_plots_en)]

@callback(
    [
     
     Output('feature_selection_en', 'options'),
     Output('sorting_en', 'label'),
     # Output('feature_selection', 'value')
     
     ],
    [Input('alphabet_en', 'n_clicks'),
     Input('corr_desc_en', 'n_clicks'),
     Input('corr_asc_en', 'n_clicks'),
     Input('corr_abs_desc_en', 'n_clicks'),
     Input('corr_abs_asc_en', 'n_clicks'),
     Input('main_class_en', 'n_clicks'),
     Input('second_class_en', 'n_clicks'),
     Input('third_class_en', 'n_clicks'),
     Input('fourth_class_en', 'n_clicks'),
     Input('fifth_class_en', 'n_clicks')
    ]
)
def en_update_selections(*args):
    
    ctx = callback_context
    
    
    if not ctx.triggered:
        return feature_options_en, "Alphabetical order"#,[f['value'] for f in corr_abs_asc_options[:4]]
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if button_id == 'alphabet_en':
        return feature_options_en, "Alphabetical order",#[f['value'] for f in feature_options[:4]]
    elif button_id == 'corr_desc_en':
        return corr_desc_options_en, "Correlation (descending)",#[f['value'] for f in corr_desc_options[:4]]
    elif button_id == 'corr_asc_en':
        return corr_asc_options_en, "Correlation (ascending)",#[f['value'] for f in corr_asc_options[:4]]
    elif button_id == 'corr_abs_desc_en':
        return corr_abs_desc_options_en, "Absolute correlation (descending)"#,[f['value'] for f in corr_abs_desc_options[:4]]
    elif button_id == 'corr_abs_asc_en':
        return corr_abs_asc_options_en, "Absolute correlation (ascending)",#[f['value'] for f in corr_abs_asc_options[:4]]
    elif button_id == 'main_class_en':
        return main_class_options_en, "Main classes",#[f['value'] for f in main_class_options[:4]]
    elif button_id == 'second_class_en':
        return second_class_options_en, "2. class",#[f['value'] for f in second_class_options[:4]]
    elif button_id == 'third_class_en':
        return third_class_options_en, "3. class",#[f['value'] for f in third_class_options[:4]]
    elif button_id == 'fourth_class_en':
        return fourth_class_options_en, "4. class"#,[f['value'] for f in fourth_class_options[:4]]
    else:
        return fifth_class_options_en, "5. class",#[f['value'] for f in fifth_class_options[:4]]
    
