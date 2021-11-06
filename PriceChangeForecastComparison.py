import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from datetime import date
from datetime import date, timedelta
import pandas as pd
import re

class PriceChangeForecastComparison:

    def __init__(self,app):
        #################################
        # MODEL INITIALISATION
        #################################

        DEFAULT_COIN = 'btc'

        self.app = app

        self.predictions = {}
        self.corrs = pd.read_csv('models/predictions/correlation.csv', index_col=0)
        
        # Loading trained models
        for coin in ['btc','eth','ltc']:
            k = f'LSTM-{coin}'
            # store prediction from test set in memory
            prediction = pd.read_csv(f"models/predictions/lstm_price_change_pred_{coin}.csv").to_numpy()

            # map date to index of test partition
            # test partition start from latest BTC halving: 2020-05-11 to 2020-09-10
            start = date(2020, 9, 11)
            end = date(2020, 9, 21)
            delta = end - start
            dates_range = pd.date_range(start,end-timedelta(days=1),freq='d').astype(str).to_list()

            date2id = dict(zip(dates_range,range(len(prediction)-delta.days,len(prediction),1)))
            # print(len(prediction),len(dates_range),date2id)
            self.predictions[k] = (pd.read_csv(f"models/predictions/lstm_price_change_pred_{coin}.csv").to_numpy(), date2id)

            # RF PREDICTIONS
            r = f'RF-{coin}'
            prediction_rf = pd.read_csv(f"models/predictions/rf_price_change_pred_{coin}.csv").to_numpy()
            date2id_rf = dict(zip(dates_range,range(0,len(prediction_rf))))
            self.predictions[r] = (prediction_rf, date2id_rf)


        #################################
        # HTML ELEMENTS
        #################################
        
        def create_card(html_elements):
            return dbc.Card(
                dbc.CardBody(
                    html_elements
                )
            )

        corr_card = create_card(
                [
                    html.H4("Price Correlation", id="price-corr"),
                    html.H2("1", id="price-corr-value"),
                ]
            )

        lstm_left_card = create_card(
                [
                    html.H4("LSTM Model Forecast", id="lstm-left"),
                    html.H2("...", id="lstm-left-value"),
                ]
            )

        lstm_right_card = create_card(
                [
                    html.H4("LSTM Model Forecast", id="lstm-right"),
                    html.H2("...", id="lstm-right-value"),
                ]
            )

        rf_left_card = create_card(
                [
                    html.H4("RF Model Forecast", id="rf-left"),
                    html.H2("...", id="rf-left-value"),
                ]
            )

        rf_right_card = create_card(
                [
                    html.H4("RF Model Forecast", id="rf-right"),
                    html.H2("...", id="rf-right-value"),
                ]
            )
        
        slider_left = create_card(
            [
                html.H6('Date of Prediction:'),
                dcc.Slider(id='date-selector-left',
                       min=0,
                       max=9,
                       marks={i: '{}'.format(dates_range[i])
                              for i in range(0, len(dates_range))},
                       value=0,
                       ),
            ]
        )

        slider_right = create_card(
            [
                html.H6('Date of Prediction:'),
                dcc.Slider(id='date-selector-right',
                       min=0,
                       max=9,
                       marks={i: '{}'.format(dates_range[i])
                              for i in range(0, len(dates_range))},
                       value=0,
                       ),
            ]
        )

        self.layout = html.Div([
            html.H3('Price Change Forecast',id='price-change-title'),
            html.Br(),
            html.H2('Comparison',id='compare-header'),
            html.Br(),
            dcc.Link('Go back to price change forecast', href='/price-change-forecast'),
            html.Br(),
            html.Br(),
            # Grid layout
            html.Div([
                dbc.Row([
                    dbc.Col([
                        dbc.Row([
                            dcc.Dropdown(
                                id='pricechange-coin-dropdown-left',
                                options=[
                                    {'label': 'Bitcoin', 'value': 'btc'},
                                    {'label': 'Ethereum', 'value': 'eth'},
                                    {'label': 'Litecoin', 'value': 'ltc'}
                                ],
                                value=DEFAULT_COIN
                            )
                        ]),
                        dbc.Row([
                            dbc.Col([slider_left])
                        ])
                    ]),
                    dbc.Col([
                        dbc.Row([
                            dcc.Dropdown(
                                id='pricechange-coin-dropdown-right',
                                options=[
                                    {'label': 'Bitcoin', 'value': 'btc'},
                                    {'label': 'Ethereum', 'value': 'eth'},
                                    {'label': 'Litecoin', 'value': 'ltc'}
                                ],
                                value=DEFAULT_COIN
                            )
                        ]),
                        dbc.Row([
                            dbc.Col([slider_right])
                        ])
                    ]),
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Row([
                            dbc.Col([rf_left_card]), dbc.Col([lstm_left_card])
                        ])
                    ]),
                    dbc.Col([
                        dbc.Row([
                            dbc.Col([rf_right_card]), dbc.Col([lstm_right_card])
                        ])
                    ]),
                ]),
                html.Br(),
                html.Br(),
                dbc.Row([
                    dbc.Col([
                        corr_card
                    ], style={'text-align': 'center'})
                ]),
                html.Br(),
                html.Br()

            ]),
            
            
            html.Br(),


            dcc.Link('Go back to home', href='/'),
        ])
        #################################
        # HTML CALLBACKS CENTER
        #################################
        def format_output(label):
            label = re.sub('[(],]', '', label)
            l = label.split(" ")[0][1:-1]
            u = label.split(" ")[1][1:-1]
            return f"{l}% to {u}%"

        @self.app.callback(
                    [Output('compare-header', 'children'), Output('price-corr-value', 'children')],
                    [Input('pricechange-coin-dropdown-left', 'value'),Input('pricechange-coin-dropdown-right', 'value')])
        def coin_dropdown(coinleft,coinright):
            return f'Comparison: {coinleft.upper()} vs {coinright.upper()}', self.corrs[coinleft].loc[coinright]

        #################################
        # HTML CALLBACKS LEFT
        #################################

        @self.app.callback(
                    Output('selected-coin-left', 'children'),
                    [Input('pricechange-coin-dropdown-left', 'value')])
        def coin_dropdown(value):
            return 'Forecast for {}'.format(value.upper())

        @self.app.callback(
                    [Output('lstm-left-value', 'children'),  Output('rf-left-value', 'children')],
                    [Input('pricechange-coin-dropdown-left', 'value'),Input('date-selector-left', 'value')])
        def select_date(coin,date_index):
            selected_date = dates_range[date_index]

            prediction = self.predictions[f"LSTM-{coin}"][0]
            id_mapping = self.predictions[f"LSTM-{coin}"][1]
            ind = id_mapping[selected_date]

            prediction_rf = self.predictions[f"RF-{coin}"][0]
            id_mapping_rf = self.predictions[f"RF-{coin}"][1]
            ind_rf = id_mapping_rf[selected_date]
            return format_output(prediction[ind][1]), format_output(prediction_rf[ind_rf][1])

        #################################
        # HTML CALLBACKS RIGHT
        #################################

        @self.app.callback(
                    Output('selected-coin-right', 'children'),
                    [Input('pricechange-coin-dropdown-right', 'value')])
        def coin_dropdown(value):
            return 'Forecast for {}'.format(value.upper())

        @self.app.callback(
                    [Output('lstm-right-value', 'children'), Output('rf-right-value', 'children')],
                    [Input('pricechange-coin-dropdown-right', 'value'),Input('date-selector-right', 'value')])
        def select_date(coin,date_index):
            selected_date = dates_range[date_index]

            prediction = self.predictions[f"LSTM-{coin}"][0]
            id_mapping = self.predictions[f"LSTM-{coin}"][1]
            ind = id_mapping[selected_date]

            prediction_rf = self.predictions[f"RF-{coin}"][0]
            id_mapping_rf = self.predictions[f"RF-{coin}"][1]
            ind_rf = id_mapping_rf[selected_date]
            return format_output(prediction[ind][1]), format_output(prediction_rf[ind_rf][1])

