import dash  
from dash import Dash, Input, Output, html
import dash_bootstrap_components as dbc 
from flask import Flask
import os
from dash_bootstrap_templates import ThemeChangerAIO


dbc_css = ("https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates@V1.0.2/dbc.min.css")

external_stylesheets = [
                        # "https://stackpath.bootstrapcdn.com/bootswatch/4.5.2/superhero/bootstrap.min.css",
                        
                          dbc.themes.SUPERHERO,
                          dbc_css,
                          dbc.icons.BOOTSTRAP,
                          "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css",
                          'https://codepen.io/chriddyp/pen/brPBPO.css',
                          
                        ]


server = Flask(__name__)
server.secret_key = os.environ.get('secret_key','secret')
app = Dash(name = __name__, 
            prevent_initial_callbacks = False, 
            use_pages=True,
            server = server,
            external_scripts = ["https://raw.githubusercontent.com/plotly/plotly.js/master/dist/plotly-locale-fi.js",
                                "https://cdn.plot.ly/plotly-locale-fi-latest.js",
                                "https://cdn.plot.ly/plotly-locale-sv-latest.js"],
            # meta_tags=[{'name': 'viewport',
            #                 'content': 'width=device-width, initial-scale=1, maximum-scale=1, minimum-scale=1,'}],
            external_stylesheets = external_stylesheets
          )


app.index_string = '''<!DOCTYPE html>
<html>
<head>
<title>Phillipsin vinouma</title>
<link rel="manifest" href="./assets/manifest.json" />
{%metas%}
{%favicon%}
{%css%}
</head>
<script type="module">
    import 'https://cdn.jsdelivr.net/npm/@pwabuilder/pwaupdate';
    const el = document.createElement('pwa-update');
    document.body.appendChild(el);
</script>
<body>
<script>
  if ('serviceWorker' in navigator) {
    window.addEventListener('load', ()=> {
      navigator
      .serviceWorker
      .register('./assets/sw01.js')
      .then(()=>console.log("Ready."))
      .catch(()=>console.log("Err..."));
    });
  }
</script>
{%app_entry%}
<footer>
{%config%}
{%scripts%}
{%renderer%}
</footer>
</body>
</html>
'''

app.title = 'Skewed Phillips'


theme_changer = dbc.Row(
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
                                radio_props={"value":dbc.themes.SUPERHERO})
            ])
        ])

navbar = dbc.Navbar(
    
     dbc.Container([
    
    dbc.Row([
        
        dbc.Col([
            html.A([
                html.Img(src = dash.get_asset_url('gofore_logo_white.svg'),
                         height="40px")
                ],
                href = 'https://gofore.com/', 
                target='_blank')
            ])
        ], align='center',className = "d-flex justify-content-start"),
                    dbc.Row([
                    dbc.Col(
                         dbc.Collapse(
                            dbc.Nav([
                                dbc.NavbarBrand("by: Tuomas Poukkula",style={'font-style':'italic'}, className="ms-2"),
                                dbc.NavItem(dbc.NavLink(html.I(className="bi bi-github"), href="https://github.com/tuopouk/skewedphillips",external_link=True, target='_blank') ),
                                dbc.NavItem(dbc.NavLink(html.I(className="bi bi bi-twitter"), href="https://twitter.com/TuomasPoukkula",external_link=True) ),
                                dbc.NavItem(dbc.NavLink(html.I(className="bi bi-linkedin"), href="https://www.linkedin.com/in/tuomaspoukkula/",external_link=True) ),
                                
                            ]
                            ),
                            id="navbar-collapse",
                            is_open=False,
                            navbar=True
                         )
                    )
                ],
                align="center"),
    dbc.Row([
        dbc.Col([dbc.DropdownMenu(id ='dd_menu',
                                  # size="lg",
                                  # menu_variant="dark",
                                  children =
                                        [
                                            dbc.DropdownMenuItem('FI',id = 'fi',  href='/'),
                                            dbc.DropdownMenuItem('EN',id = 'en',  href='/en'),
                                             dbc.DropdownMenuItem('SV',id = 'sv',  href='/sv')
                                        ],
            # nav=True,
            label="FI"
        )])
        
        ], align = 'center', className = "d-flex justify-content-end"),
 
    # className ='dbc'
     # className="mb-2"
    
      ],className='dbc', fluid=True
       ),
                            
    color="primary",
    dark=True
    )
        


app.layout = dbc.Container(
    [navbar,html.Br(), theme_changer,  dash.page_container],
    fluid=True,
    className='dbc'
)

@dash.callback(
    Output('dd_menu','label'), 
    [Input('fi','n_clicks'),
     Input('en','n_clicks'),
      Input('sv','n_clicks')
     
     ]
)
def update_label(*args):
    
    ctx = dash.callback_context

    if not ctx.triggered:
        button_id = "fi"
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == 'fi':
        return 'FI'
    elif button_id == 'en':
        return 'EN'
    else:
        return 'SV'
    
@dash.callback(

     [Output(ThemeChangerAIO.ids.button("theme"), "title"),
     Output(ThemeChangerAIO.ids.button("theme"), "children"),
      Output(ThemeChangerAIO.ids.offcanvas("theme"), "title")
      ],
    [Input('fi','n_clicks'),
     Input('en','n_clicks'),
      Input('sv','n_clicks')]
    
)
def change_theme_changer_language(*args):
    
    ctx = dash.callback_context

    if not ctx.triggered:
        button_id = "fi"
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == 'fi':
        return 'Vaihda väriteemaa', 'Vaihda väriteemaa', "Valitse jokin alla olevista väriteemoista"
    elif button_id == 'en':
        return 'Change Color Theme', 'Change Color Theme', "Select a Color Theme"
    else:
        return "Ändra färgtema", "Ändra färgtema", "Välj ett färgtema"
    


if __name__ == "__main__":
    app.run_server(debug=False)