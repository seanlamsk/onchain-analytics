import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import efficientfrontier

# fig, minvolport, optriskport = efficientfrontier.plotEfficientFrontierfromRawCSV()


class Portfolios_Analysis:

    def __init__(self, app):
        self.app = app
        fig, minvolport, optriskport = efficientfrontier.plotEfficientFrontierfromRawCSV()
        btcmvp = minvolport['btc weight'] * 100
        ethmvp = minvolport['eth weight'] * 100
        ltcmvp = minvolport['ltc weight'] * 100
        btcorp = optriskport['btc weight'] * 100
        ethorp = optriskport['eth weight'] * 100
        ltcorp = optriskport['ltc weight'] * 100
        self.cardMVP = dbc.Card(
            dbc.CardBody(
                [
                    html.H4("Minimum Volatility Portfolio", id="card-title"),
                    html.Ul(id="holdingsMVP"),
                    html.Li("BTC: {}%".format(btcmvp.round(3)), id="MVPBTC"),
                    html.Li("ETH: {}%".format(ethmvp.round(3)), id="MVPETH"),
                    html.Li("LTC {}%".format(ltcmvp.round(3)), id="MVPLTC")
                ]
            )
        )

        self.cardORP = dbc.Card(
            dbc.CardBody(
                [
                    html.H4("Optimal Returns Portfolio", id="card-title2"),
                    html.Ul(id="holdingsORP"),
                    html.Li("BTC: {}%".format(btcorp.round(3)), id="ORPBTC"),
                    html.Li("ETH: {}%".format(ethorp.round(3)), id="ORPETH"),
                    html.Li("LTC {}%".format(ltcorp.round(3)), id="ORPLTC")
                ]
            )
        )

        self.layout = html.Div([
            html.H1('Portfolio Analysis', id='price-change-title'),
            html.Br(),
            html.Div([
                dbc.Row([
                    dbc.Col([self.cardORP]), dbc.Col([self.cardMVP])
                ])
            ]),
            html.Br(),
            html.Br(),
            dcc.Graph(figure=fig),


            html.Br(),


            # dcc.Link('Go back to home', href='/'),
        ])

        @self.app.callback(
            Output('selected-portfolio', 'children'),
            [Input('portfolio', 'value')])
        def portfolio(value):
            return 'Forecast for {}'.format(value.upper())
