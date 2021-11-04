import dash
from dash import dcc
import dash_bootstrap_components as dbc
from dash import html
import plotly.express as px
import pandas as pd

from PriceChangeForecast import PriceChangeForecast
from Portfolios import Portfolios_Analysis
from PricePrediction import PricePrediction



# Since we're adding callbacks to elements that don't exist in the app.layout,
# Dash will raise an exception to warn us that we might be
# doing something wrong.
# In this case, we're adding the elements through a callback, so we can ignore
# the exception.
app = dash.Dash(__name__, 
suppress_callback_exceptions=True,
external_stylesheets=[dbc.themes.BOOTSTRAP])

price_change_page = PriceChangeForecast(app)
portfolio_analysis = Portfolios_Analysis(app)
price_pred_page = PricePrediction(app)

app.layout = html.Div([
    html.H1('On Chain Analytics Dashboard'),
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

index_page = html.Div([
    dcc.Link('View Price Forecast', href='/price-forecast'),
    html.Br(),

    dcc.Link('View Price Change Forecast', href='/price-change-forecast'),
    html.Br(),

    dcc.Link('View Portfolio Analysis', href='/portfolio-analysis'),
    html.Br(),
])

# Update the index
@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/price-change-forecast':
        return price_change_page.layout
    elif pathname == '/price-forecast':
        return price_pred_page.layout
    elif pathname == '/portfolio-analysis':
        return portfolio_analysis.layout
    else:
        return index_page
    # You could also return a 404 "URL not found" page here


if __name__ == '__main__':
    app.run_server(debug=True)



# page_2_layout = html.Div([
#     html.H1('Page 2'),
#     dcc.RadioItems(
#         id='page-2-radios',
#         options=[{'label': i, 'value': i} for i in ['Orange', 'Blue', 'Red']],
#         value='Orange'
#     ),
#     html.Div(id='page-2-content'),
#     html.Br(),
#     dcc.Link('Go to Page 1', href='/page-1'),
#     html.Br(),
#     dcc.Link('Go back to home', href='/')
# ])

# @app.callback(dash.dependencies.Output('page-2-content', 'children'),
#               [dash.dependencies.Input('page-2-radios', 'value')])
# def page_2_radios(value):
#     return 'You have selected "{}"'.format(value)