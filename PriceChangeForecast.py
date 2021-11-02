import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from datetime import date
from datetime import date, timedelta
import pandas as pd

from models.LSTM_price_change import LSTMPriceChangeModel

class PriceChangeForecast:

    def __init__(self,app):
        #################################
        # MODEL INITIALISATION
        #################################

        DEFAULT_COIN = 'btc'

        self.app = app

        self.models = {
            "LSTM-btc": LSTMPriceChangeModel("btc_metrics_raw.csv"),
            "LSTM-eth": LSTMPriceChangeModel("eth_metrics_raw.csv"),
            "LSTM-ltc": LSTMPriceChangeModel("ltc_metrics_raw.csv"),
        }

        self.select_model(DEFAULT_COIN)

        self.predictions = {}
        
        # Loading trained models
        for k,model in self.models.items():
            coin = k.split('-')[1]
            model.init()
            model.load_model(f'models/saved_models/{coin}/lstm_price_change_{coin}.hp5')
            
            # store prediction from test set in memory
            self.predictions[k] = model.predict(model.X_test,return_label=False)

        # map date to index of test partition
        # test partition start from latest BTC halving: 2020-05-11 to 2020-09-10
        start = date(2020, 8, 1)
        end = date(2020, 9, 11)
        delta = end - start
        dates_range = pd.date_range(start,end-timedelta(days=1),freq='d').astype(str).to_list()

        date2id = dict(zip(dates_range,range(len(model.X_test)-delta.days,len(model.X_test),1)))
        
        #################################
        # HTML ELEMENTS
        #################################
        
        def create_card(html_elements):
            return dbc.Card(
                dbc.CardBody(
                    html_elements
                )
            )

        avg_card = create_card(
                [
                    html.H4("Average", id="avg"),
                    html.H2("100", id="card-value"),
                ]
            )

        lstm_card = create_card(
                [
                    html.H4("LSTM Model Forecast", id="lstm"),
                    html.H2("100", id="lstm-value"),
                ]
            )
        
        date_picker_card = create_card(
                [
                    html.H4("Forecast", id="date_picker_h"),
                    dcc.DatePickerSingle(
                        id='selected-date',
                        min_date_allowed=start,
                        max_date_allowed=end,
                        initial_visible_month=start,
                        date=start
                    ),
                ]
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
                    value=DEFAULT_COIN
                ),

            ]),
            html.Br(),
            html.H3(id='selected-coin'),
            html.Br(),
            # Grid layout
            html.Div([
                dbc.Row([
                    dbc.Col([avg_card]), dbc.Col([date_picker_card])
                ]),
                dbc.Row([
                    dbc.Col([lstm_card]), dbc.Col([avg_card])
                ])

            ]),
            
            
            html.Br(),


            dcc.Link('Go back to home', href='/'),
        ])

        #################################
        # HTML CALLBACKS
        #################################

        @self.app.callback(
                    Output('selected-coin', 'children'),
                    [Input('pricechange-coin-dropdown', 'value')])
        def coin_dropdown(value):
            return 'Forecast for {}'.format(value.upper())

        @self.app.callback(
                    Output('lstm-value', 'children'),
                    [Input('pricechange-coin-dropdown', 'value'),Input('selected-date', 'date')])
        def select_date(coin,selected_date):
            ind = date2id[selected_date]
            prediction = self.predictions[f"LSTM-{coin}"][ind]
            return prediction.upper()

    def select_model(self,dropdown_option):
        if dropdown_option == 'btc':
            self.active_lstm_model = self.models['LSTM-btc']
        elif dropdown_option == 'eth':
            self.active_lstm_model = self.models['LSTM-eth']
        elif dropdown_option == 'ltc':
            self.active_lstm_model = self.models['LSTM-ltc']
