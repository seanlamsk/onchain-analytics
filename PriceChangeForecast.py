import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

class PriceChangeForecast:

    def __init__(self,app):
        self.app = app

        self.card = dbc.Card(
            dbc.CardBody(
                [
                    html.H4("Title", id="card-title"),
                    html.H2("100", id="card-value"),
                    html.P("Description", id="card-description")
                ]
            )
        )

        self.layout = html.Div([
            html.H1('Price Change Forecast',id='price-change-title'),
            html.Br(),
            html.Div(id='price-change-content',
            children=[
                dcc.Dropdown(
                    id='pricechange-coin-dropdown',
                    options=[
                        {'label': 'Bitcoin', 'value': 'btc'},
                        {'label': 'Ethereum', 'value': 'eth'},
                        {'label': 'Litecoin', 'value': 'ltc'}
                    ],
                    value='btc'
                ),

            ]),
            html.Br(),
            html.H3(id='selected-coin'),
            html.Br(),
            html.Div([
                dbc.Row([
                    dbc.Col([self.card]), dbc.Col([self.card])
                ])
            ]),
            
            
            html.Br(),


            dcc.Link('Go back to home', href='/'),
        ])

        @self.app.callback(
                    Output('selected-coin', 'children'),
                    [Input('pricechange-coin-dropdown', 'value')])
        def coin_dropdown(value):
            return 'Forecast for {}'.format(value.upper())