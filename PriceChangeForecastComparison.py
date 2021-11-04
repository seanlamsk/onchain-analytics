import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from datetime import date
from datetime import date, timedelta
import pandas as pd


class PriceChangeForecastComparison:

    def __init__(self,app):
        #################################
        # MODEL INITIALISATION
        #################################

        DEFAULT_COIN = 'btc'

        self.app = app

        self.predictions = {}
        
        # Loading trained models
        for coin in ['btc','eth','ltc']:
            k = f'LSTM-{coin}'
            # store prediction from test set in memory
            prediction = pd.read_csv(f"models/predictions/lstm_price_change_pred_{coin}.csv").to_numpy()

            # map date to index of test partition
            # test partition start from latest BTC halving: 2020-05-11 to 2020-09-10
            start = date(2020, 8, 1)
            end = date(2020, 9, 11)
            delta = end - start
            dates_range = pd.date_range(start,end-timedelta(days=1),freq='d').astype(str).to_list()

            date2id = dict(zip(dates_range,range(len(prediction)-delta.days,len(prediction),1)))
            # print(len(prediction),len(dates_range),date2id)
            self.predictions[k] = (pd.read_csv(f"models/predictions/lstm_price_change_pred_{coin}.csv").to_numpy(), date2id)
        
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
                    html.H4("Price Correlation"),
                    html.H2("100", id="corr-value"),
                ]
            )

        lstm_left_card = create_card(
                [
                    html.H4("LSTM Model Forecast", id="lstm-left"),
                    html.H2("100", id="lstm-left-value"),
                ]
            )

        lstm_right_card = create_card(
                [
                    html.H4("LSTM Model Forecast", id="lstm-right"),
                    html.H2("100", id="lstm-right-value"),
                ]
            )
        
        date_picker_left = create_card(
                [
                    html.H4("Forecast", id="selected-coin-left"),
                    dcc.DatePickerSingle(
                        id='selected-date-left',
                        min_date_allowed=start,
                        max_date_allowed=end-timedelta(days=1),
                        initial_visible_month=start,
                        date=start
                    ),
                ]
            )

        date_picker_right = create_card(
                [
                    html.H4("Forecast", id="selected-coin-right"),
                    dcc.DatePickerSingle(
                        id='selected-date-right',
                        min_date_allowed=start,
                        max_date_allowed=end-timedelta(days=1),
                        initial_visible_month=start,
                        date=start
                    ),
                ]
            )

        self.layout = html.Div([
            html.H1('Price Change Forecast',id='price-change-title'),
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
                            dbc.Col([date_picker_left])
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
                            dbc.Col([date_picker_right])
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
                html.Br(),
                dbc.Row([
                    dbc.Col([
                        dbc.Row([
                            dbc.Col([lstm_left_card]), dbc.Col([lstm_left_card])
                        ])
                    ]),
                    dbc.Col([
                        dbc.Row([
                            dbc.Col([lstm_right_card]), dbc.Col([lstm_right_card])
                        ])
                    ]),
                ])

            ]),
            
            
            html.Br(),


            dcc.Link('Go back to home', href='/'),
        ])
        #################################
        # HTML CALLBACKS CENTER
        #################################

        @self.app.callback(
                    Output('compare-header', 'children'),
                    [Input('pricechange-coin-dropdown-left', 'value'),Input('pricechange-coin-dropdown-right', 'value')])
        def coin_dropdown(coinleft,coinright):
            return f'Comparison: {coinleft.upper()} vs {coinright.upper()}'

        #################################
        # HTML CALLBACKS LEFT
        #################################

        @self.app.callback(
                    Output('selected-coin-left', 'children'),
                    [Input('pricechange-coin-dropdown-left', 'value')])
        def coin_dropdown(value):
            return 'Forecast for {}'.format(value.upper())

        @self.app.callback(
                    Output('lstm-left-value', 'children'),
                    [Input('pricechange-coin-dropdown-left', 'value'),Input('selected-date-left', 'date')])
        def select_date(coin,selected_date):
            prediction = self.predictions[f"LSTM-{coin}"][0]
            id_mapping = self.predictions[f"LSTM-{coin}"][1]
            ind = id_mapping[selected_date]
            return prediction[ind][1].upper()

        #################################
        # HTML CALLBACKS RIGHT
        #################################

        @self.app.callback(
                    Output('selected-coin-right', 'children'),
                    [Input('pricechange-coin-dropdown-right', 'value')])
        def coin_dropdown(value):
            return 'Forecast for {}'.format(value.upper())

        @self.app.callback(
                    Output('lstm-right-value', 'children'),
                    [Input('pricechange-coin-dropdown-right', 'value'),Input('selected-date-right', 'date')])
        def select_date(coin,selected_date):
            prediction = self.predictions[f"LSTM-{coin}"][0]
            id_mapping = self.predictions[f"LSTM-{coin}"][1]
            ind = id_mapping[selected_date]
            return prediction[ind][1].upper()

