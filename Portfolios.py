import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import efficientfrontier

# fig, minvolport, optriskport = efficientfrontier.plotEfficientFrontierfromRawCSV()

class Portfolios_Analysis:

    def __init__(self,app):
        self.app = app
        fig, minvolport, optriskport = efficientfrontier.plotEfficientFrontierfromRawCSV()

        self.cardMVP = dbc.Card(
            dbc.CardBody(
                [
                    html.H4("Minimum Volatility Portfolio", id="card-title"),
                    html.Ul(id = "holdingsMVP"),
                    html.Li("BTC: %{minvolport['btc']}", id = "MVPBTC"),
                    html.Li("ETH: %{minvolport['eth']}", id ="MVPETH" ),
                    html.Li("LTC %{minvolport['ltc']}", id = "MVPLTC")
                ]
            )
        )

        self.cardORP = dbc.Card(
            dbc.CardBody(
                [
                    html.H4("Optimal Returns Portfolio", id="card-title2"),
                    html.Ul(id = "holdingsORP"),
                    html.Li("BTC: %{optriskport[`btc`]}", id = "ORPBTC"),
                    html.Li("ETH: %{optriskport[`eth`]}", id = "ORPETH"),
                    html.Li("LTC %{optriskport[`ltc`]}", id = "ORPLTC")
                ]
            )
        )

        self.layout = html.Div([
            html.H1('Portfolio Analysis',id='price-change-title'),
            html.Br(),
            html.Div([
                dbc.Row([
                    dbc.Col([self.cardORP]), dbc.Col([self.cardMVP])
                ])
            ]),
            html.Br(),
            html.Br(),
            dcc.Graph(figure = fig),
            
            
            html.Br(),


            dcc.Link('Go back to home', href='/'),
        ])

        @self.app.callback(
                    Output('selected-portfolio', 'children'),
                    [Input('portfolio', 'value')])
        def portfolio(value):
            return 'Forecast for {}'.format(value.upper())