# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 08:39:50 2022

@author: tuomas.poukkula
"""

from dash import html
import dash

dash.register_page(__name__, path="/404")


layout = html.Div([
    
    html.H1('Oops!'),
    
    html.H2('Päivitä sivu tai siirry osoitteeseen:'),
    html.A([html.P('https://skewedphillips.herokuapp.com/')],
           href = 'https://skewedphillips.herokuapp.com/'),
    html.Br(),
    html.H2('Please refresh the page or go to:'),
    html.A([html.P('https://skewedphillips.herokuapp.com/')],
           href = 'https://skewedphillips.herokuapp.com/'),
    html.H2('Uppdatera sidan eller gå till:'),
    html.A([html.P('https://skewedphillips.herokuapp.com/')],
           href = 'https://skewedphillips.herokuapp.com/'),
    
    ])