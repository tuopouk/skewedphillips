import dash  
from dash import Dash, Input, Output, html, State
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

p_font_size = 22
p_style = {
        # #'font-family':'Messina Modern Book',
            'font-size':p_font_size,
           'text-align':'center'}


quick_help_button = dbc.Button("Avaa pikaohje", 
             id="quick_help_button", 
             n_clicks=0, 
             outline=False,
              size = 'sm',
             color = 'danger',
             className="me-1",
             # style = {'font-style':'Cadiz Semibold'}
             ),
quick_help_offcanvas = dbc.Offcanvas(
      
      id="quick_help_offcanvas",
      title="Pikaohje",
      scrollable=True,
      is_open=False,
      style = {
          # 'font-style':'Cadiz Book',
              'background-color':'white',
              'color':'black',
               'font-size':'30px'}
)

theme_changer = ThemeChangerAIO(aio_id="theme", 
                                button_props={'title':'Vaihda väriteemaa',
                                                 'size':'sm',
                                              'children' : 'Vaihda väriteemaa',
                                              'outline':False,
                                              'color':'warning'},
                                offcanvas_props={'title':"Valitse jokin alla olevista väriteemoista",
                                                 
                                                  'scrollable':True},
                                radio_props={"value":dbc.themes.SUPERHERO})



navbar = dbc.Navbar(
    
      dbc.Container([
    
    dbc.Row([
        
        dbc.Col([
            html.A([
                html.Img(src = dash.get_asset_url('gofore_logo_white.svg'),
                          height="60px")
                ],
                href = 'https://gofore.com/', 
                target='_blank')
            ]),#xl = 4, lg = 4, md = 12, sm = 12),
   
        dbc.Col([
            dbc.NavItem(theme_changer),
            
            ], align='center'),#xl = 4, lg = 4, md = 6, sm = 6),
        dbc.Col([
            dbc.NavItem(quick_help_button)
            ], align='center'),#xl = 4, lg = 4, md = 6, sm = 6),
 
        
        ], align='center',
        className = "d-flex justify-content-start"
        ),
    


        dbc.Row([
                    dbc.Col(
                        [
                         
                          dbc.Collapse(
                            dbc.Nav([
                                dbc.NavbarBrand("by: Tuomas Poukkula",style={'font-style':'italic'}, className="ms-2"),
                                dbc.NavItem(dbc.NavLink(html.I(className="bi bi-github"), href="https://github.com/tuopouk/skewedphillips",external_link=True, target='_blank') ),
                                dbc.NavItem(dbc.NavLink(html.I(className="bi bi bi-twitter"), href="https://twitter.com/TuomasPoukkula",external_link=True, target='_blank') ),
                                dbc.NavItem(dbc.NavLink(html.I(className="bi bi-linkedin"), href="https://www.linkedin.com/in/tuomaspoukkula/",external_link=True, target='_blank') ),
                                
                            ]
                            ),
                            id="navbar-collapse",
                            is_open=False,
                            navbar=True
                          )
                          ]
                    )
                ],align="center", className = "d-flex justify-content-end"),
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
            )], align = 'center')
            
              ], align = 'center', className = "d-flex justify-content-end"),

    
      ],className='d-flex justify-content-between', fluid=True
        ),
                            
    color="primary",
    dark=True,
    className = 'navbar fixed-top'
    )
        


app.layout = dbc.Container(
    [navbar,
     quick_help_offcanvas,
     html.Br(),
     html.Br(),
     html.Br(),
     html.Br(),
     dash.page_container],
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
    Output('quick_help_button','children'), 
    [Input('fi','n_clicks'),
     Input('en','n_clicks'),
      Input('sv','n_clicks')
     
     ]
)
def update_quick_help_button_label(*args):
    
    ctx = dash.callback_context

    if not ctx.triggered:
        button_id = "fi"
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == 'fi':
        return 'Avaa pikaohje'
    elif button_id == 'en':
        return 'Open Quick Help'
    else:
        return 'Öppna snabbhjälp'
    
    
@dash.callback(
    [Output('quick_help_offcanvas','title'),
     Output('quick_help_offcanvas','children')], 
    [Input('fi','n_clicks'),
     Input('en','n_clicks'),
      Input('sv','n_clicks')
     
     ]
)
def update_quick_help_offcanvas(*args):
    
    ctx = dash.callback_context

    if not ctx.triggered:
        button_id = "fi"
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == 'fi':
        return 'Pikaohje',[
                    
                html.H3('Tässä on lyhyt ohjeistus sovelluksen käyttöön. Yksityiskohtaisempaa informaatiota löytyy Ohje ja esittely -välilehdeltä sekä jokaisen toiminnon omalta välilehdeltään.', 
                        style = {
                            # #'font-family':'Cadiz Semibold',
                                  'text-align':'left',
                                  
                                  'font-size':22,
                                  'margin-bottom':'30px'
                                  }
                        ),
                    
                html.H3('1. Valitse hyödykkeitä Hyödykkeiden valinta -välilehdellä', 
                        style = {
                            # #'font-family':'Cadiz Semibold',
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
                     style = {
                         # #'font-family':'Cadiz Book',
                              'font-size':p_font_size-2,
                               'text-align':'left'}
                ),
                html.Br(),
                html.H3('2. Tutki valitsemiasi hyödykkeitä Tutkiva analyysi -välilehdellä', 
                         style = {
                             # #'font-family':'Cadiz Semibold',
                                  'margin-bottom':'30px',
                                  'font-size':20,
                                    'text-align':'left',
                                    
                                    }
                        ),
                
                html.P(
                    "Tarkastele kuvaajien avulla valittujen hyödykkeiden suhdetta työttömyysasteeseen "
                    "tai hyödykkeiden suhteita toisiinsa. "
                    "Voit myös tarkastella indeksien, työttömyysasteen ja inflaation aikasarjoja.",
                     style = {
                         # #'font-family':'Cadiz Book',
                              'font-size':p_font_size-2,
                                'text-align':'left'}
                ),
                html.Br(),
                html.H3('3. Valitse menetelmä Menetelmän valinta -välilehdellä', 
                         style = {
                             # #'font-family':'Cadiz Semibold',
                                  'margin-bottom':'30px',
                                  'font-size':20,
                                    'text-align':'left'}
                        ),
                html.P(
                    "Valitse haluamasi koneoppimisalgoritmi alasvetovalikosta. "
                    "Säädä algoritmin hyperparametrit. "
                    "Valitse painikkeesta käytetäänkö pääkomponenttianalyysiä "
                    "ja niin tehtäessä valitse säilötyn variaation määrä liutin-valinnalla.",
                     style = {
                         # #'font-family':'Cadiz Book',
                              'font-size':p_font_size-2,
                                'text-align':'left'}
                ),
                html.Br(),
                html.H3('4. Testaa menetelmää Testaaminen-välilehdellä', 
                         style = {
                             # #'font-family':'Cadiz Semibold',
                                   'text-align':'left',
                                   'font-size':20,
                                   'margin-bottom':'30px'
                                   }
                        ),
                
                html.P(
                    "Valitse testin pituus ja klikkaa testaa nappia. "
                    "Tarkastele testin kuvaajaa tai viedä tulokset Exceliin "
                    "klikkaamalla 'Lataa testitulokset koneelle -nappia'. "
                    "Voit palata edellisiin vaiheisiin ja kokeilla uudelleen eri hyödykkeillä ja menetelmillä."
                    " "
                    " Voit myös tutkia Shapley-arvojen avulla mitkä piirteet ja hyödykkeet vaikuttivat eniten ennustetulokseen.",
                    style = {
                        # #'font-family':'Cadiz Book',
                             'font-size':p_font_size-2,
                              'text-align':'left'}
                ),
                html.Br(),
                html.H3('5. Tee ennuste Ennustaminen-välilehdellä', 
                         style = {
                             # #'font-family':'Cadiz Semibold',
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
                    style = {
                        # #'font-family':'Cadiz Book',   
                             'font-size':p_font_size-2,
                               'text-align':'left'}
                ),
                
                
                
              ]
    elif button_id == 'en':
        return "Quick Help", [
            
        html.H3('For more detailed information, see the Help and Introduction tab and the separate tab for each action.', 
                style = {
                    # #'font-family':'Cadiz Semibold',
                          'text-align':'left',
                          
                          'font-size':22,
                          'margin-bottom':'30px'
                          }
                ),
            
        html.H3('1. Select commodities on the Choice of Goods tab', 
                style = {
                    # #'font-family':'Cadiz Semibold',
                          'text-align':'left',
                          'font-size':20,
                          'margin-bottom':'30px'
                          }
                ),
        
        html.P(
            "Select the commodities you want from the drop-down menu. "
            "You can sort the goods any way you want. "
            "Select whether to use change averages from previous months "
            "or a standard change for all by clicking on the selector. "
            "Adjust the default change using the slider option. "
            "Fine-tune changes in individual commodities by modifying the values in the boxes.",
             style = {
                 # #'font-family':'Cadiz Book',
                      'font-size':p_font_size-2,
                       'text-align':'left'}
        ),
        html.Br(),
        html.H3('2. Explore the products you select in the Exploratory Analysis tab', 
                 style = {
                     # #'font-family':'Cadiz Semibold',
                          'margin-bottom':'30px',
                          'font-size':20,
                            'text-align':'left',
                            
                            }
                ),
        
        html.P(
            "View the relationship between selected goods and unemployment rate using graphs "
            "or the relationship between commodities. "
            "You can also view time series for indices, unemployment rate, and inflation.",
             style = {
                 # #'font-family':'Cadiz Book',
                      'font-size':p_font_size-2,
                        'text-align':'left'}
        ),
        html.Br(),
        html.H3('3. Select a method on the Method Selection tab', 
                 style = {
                     # #'font-family':'Cadiz Semibold',
                          'margin-bottom':'30px',
                          'font-size':20,
                            'text-align':'left'}
                ),
        html.P(
            "Select the desired machine learning algorithm from the drop-down menu. "
            "Adjust the algorithm's hyperparameters. "
            "Select whether to use principal component analysis"
            "and when doing so, select the amount of explained variance with the slider selection.",
             style = {
                 # #'font-family':'Cadiz Book',
                      'font-size':p_font_size-2,
                        'text-align':'left'}
        ),
        html.Br(),
        html.H3('4. Test method on Test tab', 
                 style = {
                     # #'font-family':'Cadiz Semibold',
                           'text-align':'left',
                           'font-size':20,
                           'margin-bottom':'30px'
                           }
                ),
        
        html.P(
            "Select the test length and click the test button. "
            "View the test graph or export the results to Excel "
            "by clicking the 'Download test results' button. "
            "You can go back to the previous steps and try again with different commodities and methods."
            " "
            " You can also use Shapley values to determine which features and commodities contributed most to your forecast result.",
            style = {
                # #'font-family':'Cadiz Book',
                     'font-size':p_font_size-2,
                      'text-align':'left'}
        ),
        html.Br(),
        html.H3('5. Make a forecast on the Forecast tab', 
                 style = {
                     # #'font-family':'Cadiz Semibold',
                           'text-align':'left',
                           'font-size':20,
                           'margin-bottom':'30px'
                           
                           }
                ),
        
        html.P(
            "Select the forecast length and click the forecast button. "
            "View the forecast graph or export the results to Excel "
            "by clicking the 'Download forecast results' button."
            "You can go back to the previous steps and try again with different commodities and methods. "
            "You can also adjust expected monthly changes in commodity indices and try again.",
            style = {
                # #'font-family':'Cadiz Book',   
                     'font-size':p_font_size-2,
                       'text-align':'left'}
        ),
        
        
        
      ]
    else:
        return "Snabbhjälp", [
            
        html.H3('Mer detaljerad information finns i fliken Hjälp och Introduktion och den separata fliken för varje åtgärd.', 
                style = {
                    # #'font-family':'Cadiz Semibold',
                          'text-align':'left',
                          
                          'font-size':22,
                          'margin-bottom':'30px'
                          }
                ),
            
        html.H3('1. Välj varor på fliken Val av varor', 
                style = {
                    # #'font-family':'Cadiz Semibold',
                          'text-align':'left',
                          'font-size':20,
                          'margin-bottom':'30px'
                          }
                ),
        
        html.P(
            "Välj de varor du vill ha från rullgardinsmenyn. "
              "Du kan sortera varorna som du vill."
              "Välj om förändringsmedel från tidigare månader ska användas "
              "eller en standardändring för alla genom att klicka på väljaren. "
              "Justera standardändringen med skjutreglaget."
              "Finjustera förändringar i enskilda varor genom att ändra värdena i rutorna.",
             style = {
                 # #'font-family':'Cadiz Book',
                      'font-size':p_font_size-2,
                       'text-align':'left'}
        ),
        html.Br(),
        html.H3('2. Utforska de produkter du väljer på fliken Exploratorisk analys', 
                 style = {
                     # #'font-family':'Cadiz Semibold',
                          'margin-bottom':'30px',
                          'font-size':20,
                            'text-align':'left',
                            
                            }
                ),
        
        html.P(
            "Se förhållandet mellan utvalda varor och arbetslöshet med hjälp av grafer "
          "eller förhållandet mellan varor."
          "Du kan också se tidsserier för index, arbetslöshet och inflation.",
             style = {
                 # #'font-family':'Cadiz Book',
                      'font-size':p_font_size-2,
                        'text-align':'left'}
        ),
        html.Br(),
        html.H3('3. Välj en metod på fliken Metodval', 
                 style = {
                     # #'font-family':'Cadiz Semibold',
                          'margin-bottom':'30px',
                          'font-size':20,
                            'text-align':'left'}
                ),
        html.P(
            "Välj önskad maskininlärningsalgoritm från rullgardinsmenyn. "
              "Justera algoritmens hyperparametrar."
              "Välj om huvudkomponentanalys ska användas"
              "och när du gör det väljer du mängden förklarad varians med skjutreglagets val.",
             style = {
                 # #'font-family':'Cadiz Book',
                      'font-size':p_font_size-2,
                        'text-align':'left'}
        ),
        html.Br(),
        html.H3("4. Testa din metod på fliken Provning", 
                 style = {
                     # #'font-family':'Cadiz Semibold',
                           'text-align':'left',
                           'font-size':20,
                           'margin-bottom':'30px'
                           }
                ),
        
        html.P(
            "Välj testlängden och klicka på testknappen. "
              "Visa testgrafen eller exportera resultaten till Excel "
              "genom att klicka på knappen 'Hämta testresultat'."
              "Du kan gå tillbaka till de tidigare stegen och försöka igen med olika råvaror och metoder."
              " "
              "Du kan också använda Shapley-värden för att avgöra vilka funktioner och råvaror som bidrog mest till ditt prognosresultat.",
            style = {
                # #'font-family':'Cadiz Book',
                     'font-size':p_font_size-2,
                      'text-align':'left'}
        ),
        html.Br(),
        html.H3("5. Gör en prognos på fliken Prognos", 
                 style = {
                     # #'font-family':'Cadiz Semibold',
                           'text-align':'left',
                           'font-size':20,
                           'margin-bottom':'30px'
                           
                           }
                ),
        
        html.P(
            "Välj prognoslängden och klicka på prognosknappen. "
          "Visa prognosgrafen eller exportera resultaten till Excel "
          "genom att klicka på knappen 'Hämta prognosresultat'."
          "Du kan gå tillbaka till de tidigare stegen och försöka igen med olika råvaror och metoder. "
          "Du kan också justera förväntade månatliga förändringar i råvaruindex och försöka igen.",
            style = {
                # #'font-family':'Cadiz Book',   
                     'font-size':p_font_size-2,
                       'text-align':'left'}
        ),
        
        
        
      ]    

    
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
   
    
@dash.callback(
    Output("quick_help_offcanvas", "is_open"),
    Input("quick_help_button", "n_clicks"),
    [State("quick_help_offcanvas", "is_open")],
)
def toggle_offcanvas(n1, is_open):
    if n1:
        return not is_open
    return is_open



if __name__ == "__main__":
    app.run_server(debug=False)