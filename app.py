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
             style = {'font-weight': 'bold'}
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
                                button_props={'title':'Vaihda v??riteemaa',
                                                 'size':'sm',
                                              'children' : 'Vaihda v??riteemaa',
                                              'outline':False,
                                              'style':{'font-weight': 'bold'},
                                              'color':'success'},
                                offcanvas_props={'title':"Valitse jokin alla olevista v??riteemoista",
                                                 
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
                                dbc.NavItem(id = 'email',children = [dbc.NavLink(html.I(className="bi bi-envelope"), href="mailto:tuomas.poukkula@gofore.com?subject=Skewed Phillips",external_link=True, target='_blank')] ),
                                
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
                label="??? / A"
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
        # button_id = "fi"
        return '??? / A'
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
        return '??ppna snabbhj??lp'
    
    
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
                    
                html.H3('T??ss?? on lyhyt ohjeistus sovelluksen k??ytt????n. Yksityiskohtaisempaa informaatiota l??ytyy Ohje ja esittely -v??lilehdelt?? sek?? jokaisen toiminnon omalta v??lilehdelt????n.', 
                        style = {
                            # #'font-family':'Cadiz Semibold',
                                  'text-align':'left',
                                  
                                  'font-size':22,
                                  'margin-bottom':'30px'
                                  }
                        ),
                    
                html.H3('1. Valitse hy??dykkeit?? Hy??dykkeiden valinta -v??lilehdell??', 
                        style = {
                            # #'font-family':'Cadiz Semibold',
                                  'text-align':'left',
                                  'font-size':20,
                                  'margin-bottom':'30px'
                                  }
                        ),
                
                html.P(
                    "Valitse haluamasi hy??dykkeet alasvetovalikosta. "
                    "Voit lajitella hy??dykkeet haluamallasi tavalla. "
                    "Valitse k??ytet????nk?? edellisten kuukausien muutoskeskiarvoja "
                    "tai vakiomuutosta kaikille valitsinta klikkaamalla. "
                    "S????d?? olettu muutos liutin -valinnalla. "
                    "Hienos????d?? yksitt??isten hy??dykkeiden muutoksia muokkaamalla laatikoiden arvoja.",
                     style = {
                         # #'font-family':'Cadiz Book',
                              'font-size':p_font_size-2,
                               'text-align':'left'}
                ),
                html.Br(),
                html.H3('2. Tutki valitsemiasi hy??dykkeit?? Tutkiva analyysi -v??lilehdell??', 
                         style = {
                             # #'font-family':'Cadiz Semibold',
                                  'margin-bottom':'30px',
                                  'font-size':20,
                                    'text-align':'left',
                                    
                                    }
                        ),
                
                html.P(
                    "Tarkastele kuvaajien avulla valittujen hy??dykkeiden suhdetta ty??tt??myysasteeseen "
                    "tai hy??dykkeiden suhteita toisiinsa. "
                    "Voit my??s tarkastella indeksien, ty??tt??myysasteen ja inflaation aikasarjoja.",
                     style = {
                         # #'font-family':'Cadiz Book',
                              'font-size':p_font_size-2,
                                'text-align':'left'}
                ),
                html.Br(),
                html.H3('3. Valitse menetelm?? Menetelm??n valinta -v??lilehdell??', 
                         style = {
                             # #'font-family':'Cadiz Semibold',
                                  'margin-bottom':'30px',
                                  'font-size':20,
                                    'text-align':'left'}
                        ),
                html.P(
                    "Valitse haluamasi koneoppimisalgoritmi alasvetovalikosta. "
                    "S????d?? algoritmin hyperparametrit. "
                    "Valitse painikkeesta k??ytet????nk?? p????komponenttianalyysi?? "
                    "ja niin teht??ess?? valitse s??il??tyn variaation m????r?? liutin-valinnalla.",
                     style = {
                         # #'font-family':'Cadiz Book',
                              'font-size':p_font_size-2,
                                'text-align':'left'}
                ),
                html.Br(),
                html.H3('4. Testaa menetelm???? Testaaminen-v??lilehdell??', 
                         style = {
                             # #'font-family':'Cadiz Semibold',
                                   'text-align':'left',
                                   'font-size':20,
                                   'margin-bottom':'30px'
                                   }
                        ),
                
                html.P(
                    "Valitse testin pituus ja klikkaa testaa nappia. "
                    "Tarkastele testin kuvaajaa tai vied?? tulokset Exceliin "
                    "klikkaamalla 'Lataa testitulokset koneelle -nappia'. "
                    "Voit palata edellisiin vaiheisiin ja kokeilla uudelleen eri hy??dykkeill?? ja menetelmill??."
                    " "
                    " Voit my??s tutkia Shapley-arvojen avulla mitk?? piirteet ja hy??dykkeet vaikuttivat eniten ennustetulokseen.",
                    style = {
                        # #'font-family':'Cadiz Book',
                             'font-size':p_font_size-2,
                              'text-align':'left'}
                ),
                html.Br(),
                html.H3('5. Tee ennuste Ennustaminen-v??lilehdell??', 
                         style = {
                             # #'font-family':'Cadiz Semibold',
                                   'text-align':'left',
                                   'font-size':20,
                                   'margin-bottom':'30px'
                                   
                                   }
                        ),
                
                html.P(
                    "Valitse ennusteen pituus ja klikkaa ennusta nappia. "
                    "Tarkastele ennusteen kuvaajaa tai vied?? tulokset Exceliin "
                    "klikkaamalla 'Lataa ennustedata koneelle -nappia'. "
                    "Voit palata edellisiin vaiheisiin ja kokeilla uudelleen eri hy??dykkeill?? ja menetelmill??. "
                    "Voit my??s s????t???? hy??dykeindeksien oletettuja kuukausimuutoksia ja kokeilla uudestaan.",
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
        return "Snabbhj??lp", [
            
        html.H3('Mer detaljerad information finns i fliken Hj??lp och Introduktion och den separata fliken f??r varje ??tg??rd.', 
                style = {
                    # #'font-family':'Cadiz Semibold',
                          'text-align':'left',
                          
                          'font-size':22,
                          'margin-bottom':'30px'
                          }
                ),
            
        html.H3('1. V??lj varor p?? fliken Val av varor', 
                style = {
                    # #'font-family':'Cadiz Semibold',
                          'text-align':'left',
                          'font-size':20,
                          'margin-bottom':'30px'
                          }
                ),
        
        html.P(
            "V??lj de varor du vill ha fr??n rullgardinsmenyn. "
              "Du kan sortera varorna som du vill."
              "V??lj om f??r??ndringsmedel fr??n tidigare m??nader ska anv??ndas "
              "eller en standard??ndring f??r alla genom att klicka p?? v??ljaren. "
              "Justera standard??ndringen med skjutreglaget."
              "Finjustera f??r??ndringar i enskilda varor genom att ??ndra v??rdena i rutorna.",
             style = {
                 # #'font-family':'Cadiz Book',
                      'font-size':p_font_size-2,
                       'text-align':'left'}
        ),
        html.Br(),
        html.H3('2. Utforska de produkter du v??ljer p?? fliken Exploratorisk analys', 
                 style = {
                     # #'font-family':'Cadiz Semibold',
                          'margin-bottom':'30px',
                          'font-size':20,
                            'text-align':'left',
                            
                            }
                ),
        
        html.P(
            "Se f??rh??llandet mellan utvalda varor och arbetsl??shet med hj??lp av grafer "
          "eller f??rh??llandet mellan varor."
          "Du kan ocks?? se tidsserier f??r index, arbetsl??shet och inflation.",
             style = {
                 # #'font-family':'Cadiz Book',
                      'font-size':p_font_size-2,
                        'text-align':'left'}
        ),
        html.Br(),
        html.H3('3. V??lj en metod p?? fliken Metodval', 
                 style = {
                     # #'font-family':'Cadiz Semibold',
                          'margin-bottom':'30px',
                          'font-size':20,
                            'text-align':'left'}
                ),
        html.P(
            "V??lj ??nskad maskininl??rningsalgoritm fr??n rullgardinsmenyn. "
              "Justera algoritmens hyperparametrar."
              "V??lj om huvudkomponentanalys ska anv??ndas"
              "och n??r du g??r det v??ljer du m??ngden f??rklarad varians med skjutreglagets val.",
             style = {
                 # #'font-family':'Cadiz Book',
                      'font-size':p_font_size-2,
                        'text-align':'left'}
        ),
        html.Br(),
        html.H3("4. Testa din metod p?? fliken Provning", 
                 style = {
                     # #'font-family':'Cadiz Semibold',
                           'text-align':'left',
                           'font-size':20,
                           'margin-bottom':'30px'
                           }
                ),
        
        html.P(
            "V??lj testl??ngden och klicka p?? testknappen. "
              "Visa testgrafen eller exportera resultaten till Excel "
              "genom att klicka p?? knappen 'H??mta testresultat'."
              "Du kan g?? tillbaka till de tidigare stegen och f??rs??ka igen med olika r??varor och metoder."
              " "
              "Du kan ocks?? anv??nda Shapley-v??rden f??r att avg??ra vilka funktioner och r??varor som bidrog mest till ditt prognosresultat.",
            style = {
                # #'font-family':'Cadiz Book',
                     'font-size':p_font_size-2,
                      'text-align':'left'}
        ),
        html.Br(),
        html.H3("5. G??r en prognos p?? fliken Prognos", 
                 style = {
                     # #'font-family':'Cadiz Semibold',
                           'text-align':'left',
                           'font-size':20,
                           'margin-bottom':'30px'
                           
                           }
                ),
        
        html.P(
            "V??lj prognosl??ngden och klicka p?? prognosknappen. "
          "Visa prognosgrafen eller exportera resultaten till Excel "
          "genom att klicka p?? knappen 'H??mta prognosresultat'."
          "Du kan g?? tillbaka till de tidigare stegen och f??rs??ka igen med olika r??varor och metoder. "
          "Du kan ocks?? justera f??rv??ntade m??natliga f??r??ndringar i r??varuindex och f??rs??ka igen.",
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
        return 'Vaihda v??riteemaa', 'Vaihda v??riteemaa', "Valitse jokin alla olevista v??riteemoista"
    elif button_id == 'en':
        return 'Change Color Theme', 'Change Color Theme', "Select a Color Theme"
    else:
        return "??ndra f??rgtema", "??ndra f??rgtema", "V??lj ett f??rgtema"
   
    
@dash.callback(
    Output("quick_help_offcanvas", "is_open"),
    Input("quick_help_button", "n_clicks"),
    [State("quick_help_offcanvas", "is_open")],
)
def toggle_offcanvas(n1, is_open):
    if n1:
        return not is_open
    return is_open

@dash.callback(
    Output('email','children'),
    [Input('dd_menu','label')]
    )
def update_email_topic(label):
        

    if label == 'FI':
        return [dbc.NavLink(html.I(className="bi bi-envelope"), href="mailto:tuomas.poukkula@gofore.com?subject=Phillipsin vinouma",external_link=True, target='_blank')]
    elif label == 'EN':
        return [dbc.NavLink(html.I(className="bi bi-envelope"), href="mailto:tuomas.poukkula@gofore.com?subject=Skewed Phillips",external_link=True, target='_blank')]
    elif label == 'SV':
        return [dbc.NavLink(html.I(className="bi bi-envelope"), href="mailto:tuomas.poukkula@gofore.com?subject=Skev Phillips",external_link=True, target='_blank')]
    else:
       return [dbc.NavLink(html.I(className="bi bi-envelope"), href="mailto:tuomas.poukkula@gofore.com?subject=https://skewedphillips.herokuapp.com/",external_link=True, target='_blank')] 

if __name__ == "__main__":
    app.run_server(debug=False)