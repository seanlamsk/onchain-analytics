import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from datetime import date
from datetime import date, timedelta
import pandas as pd


class PriceChangeForecast:

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
        
            # RF PREDICTIONS
            r = f'RF-{coin}'
            prediction_rf = pd.read_csv(f"models/predictions/rf_price_change_pred_{coin}.csv").to_numpy()
            
            start_rf = date(2020, 9, 11)
            end_rf = date(2020, 9, 21)
            delta_rf = end_rf - start_rf
            dates_range_rf = pd.date_range(start_rf,end_rf-timedelta(days=1),freq='d').astype(str).to_list()
            print('dates_range_rf',dates_range_rf)
            date2id_rf = dict(zip(dates_range_rf,range(0,len(prediction))))
            print('date2id_rf',date2id_rf)
            # print(len(prediction),len(dates_range),date2id)
            self.predictions[r] = (pd.read_csv(f"models/predictions/rf_price_change_pred_{coin}.csv").to_numpy(), date2id_rf)
        
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
        
        rf_card = create_card(
                [
                    html.H4("Random Forest Model Forecast", id="rf"),
                    html.H2("...", id="rf-value"),
                ]
            )

        date_picker_card = create_card(
                [
                    html.H4("Forecast", id="date_picker_h"),
                    dcc.DatePickerSingle(
                        id='selected-date',
                        min_date_allowed=start_rf,
                        max_date_allowed=end_rf-timedelta(days=1),
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
            dcc.Link('Compare coins', href='/price-change-forecast-compare'),
            html.H3(id='selected-coin'),
            html.Br(),
            # Grid layout
            html.Div([
                dbc.Row([
                    dbc.Col([avg_card]), dbc.Col([date_picker_card])
                ]),
                dbc.Row([
                    dbc.Col([lstm_card]), dbc.Col([rf_card])
                ])

            ]),
            
            
            html.Br(),


            dcc.Link('Go back to home', href='/'),
        ])

        #################################
        # HTML CALLBACKS
        #################################

# change here to add output for rf-value
        @self.app.callback(
                    Output('selected-coin', 'children'),
                    [Input('pricechange-coin-dropdown', 'value')])
        def coin_dropdown(value):
            return 'Forecast for {}'.format(value.upper())

        @self.app.callback(
                    [
                        Output('lstm-value', 'children'),
                        Output('rf-value', 'children')
                    ],
                    [Input('pricechange-coin-dropdown', 'value'),Input('selected-date', 'date')])
        def select_date(coin,selected_date):
            prediction = self.predictions[f"LSTM-{coin}"][0]
            id_mapping = self.predictions[f"LSTM-{coin}"][1]
            ind = id_mapping[selected_date]
            
            prediction_rf = self.predictions[f"RF-{coin}"][0]
            id_mapping_rf = self.predictions[f"RF-{coin}"][1]
            ind_rf = id_mapping_rf[selected_date]
            print(prediction[ind][1].upper(), prediction_rf[ind_rf][1].upper())
            return prediction[ind][1].upper(), prediction_rf[ind_rf][1].upper()

