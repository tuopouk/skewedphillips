# skewedphillips

# Phillipsin vinouma

Tein sovelluksen, jolla voi kokeilla eri algoritmien kykyä ennusta Suomen työttömyysastetta Tilastokeskuksen julkaisemien kuluttajahintaindeksien avulla. Ajatus työttömyyden ennustamisesta kuluttajahintojen avulla perustuu 'Phillipsin käyrä' nimiseen makrotaloustieteen teoriaan, jonka mukaan lyhyellä ajalla inflaatio ja työttömyys ovat ristiriidassa. Inflaatiota mitataan kuluttajahintaindeksin vuosimuutoksella ja kuluttajahintaindeksi sisältää satoja hyödykkeitä ja palveluita. Tässä sovelluksessa voi tehdä ennustesovelluksen käyttäjän määrittelemällä hyödykekorilla.

Sovelluksessa voi valita hyödykkeet ja algoritmin, säätää hyperparametrit ja asettaa oletuksen kuluttajahintojen muutoksesta.

Aineistona toimii Tilastokeskuksen avoimet tietoaineistot kuluttajahintaindeksistä ja työttömyysasteesta

https://statfin.stat.fi/PxWeb/pxweb/fi/StatFin/StatFin__tyti/statfin_tyti_pxt_135z.px/
https://statfin.stat.fi/PxWeb/pxweb/fi/StatFin/StatFin__khi/statfin_khi_pxt_11xd.px/

Sovellus löytyy osoitteesta: 

https://skewedphillips.herokuapp.com/

Sovellus toimii PWA:na, joten sen voi ladata standalone-versiona esim. Androidille tai Windowsille.

# Skewed Phillips

I made a Dash application in which you can train machine learning models to predict unemployment rate in Finland based on Finnish consumer price indexes of different products. The app utilizes Finnish open data and is mostly targeted towards Finnish audience. Other languages might be added in the future. 

You can select which products' price indexes you use and the application uses dynamic callbacks to register the products. Then you need to select a machine learning algorithm and tune its hyperparameters. Again, Dash dynamic callbacks are used to get information out of the hyperparameters. Then you can test your model's performance on a selected time and analyze the impact of each feature with Shapley values. You can iterate to find the best set of products and methodology. You can also apply principal component analysis. You can also do some exploratory data analysis on how each product behaves in accordance with unemployment and how the products' price indexes cross correlate. Finally you can make a prediction for a given time based on your inputs of expected change of consumer price indexes. You can apply averaging, same change for every product or just set each product's expected change individually. The test and forecast results can be exported to Excel files.

This app implements PWA functionality if used with https. I also added the ability to change Bootstrap themes, so one can customize the color and font scheme on the go.
