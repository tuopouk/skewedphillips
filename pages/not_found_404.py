# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 08:39:50 2022

@author: tuomas.poukkula
"""

from dash import html
import dash

dash.register_page(__name__, path="/404")


layout = html.Div([
    html.Br(),
    html.Br(),
    html.Br(),
    html.H1('Oops!'),
    
    html.H2('Siirry osoitteeseen:'),
    html.A([html.P('https://skewedphillips.herokuapp.com/')],
           href = 'https://skewedphillips.herokuapp.com/'),
    html.Br(),
    html.H2('Please go to:'),
    html.A([html.P('https://skewedphillips.herokuapp.com/en')],
           href = 'https://skewedphillips.herokuapp.com/en'),
    html.H2('Gå till:'),
    html.A([html.P('https://skewedphillips.herokuapp.com/sv')],
           href = 'https://skewedphillips.herokuapp.com/sv'),
    
    ])