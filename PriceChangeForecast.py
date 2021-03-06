import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from dash import dash_table
from datetime import date, timedelta
import pandas as pd
import re


class PriceChangeForecast:

    def __init__(self, app):
        #################################
        # MODEL INITIALISATION
        #################################

        DEFAULT_COIN = 'btc'

        self.app = app

        self.predictions = {}

        # Loading trained models
        for coin in ['btc', 'eth', 'ltc']:
            k = f'LSTM-{coin}'
            # store prediction from test set in memory
            prediction = pd.read_csv(
                f"models/predictions/lstm_price_change_pred_{coin}.csv").to_numpy()

            # map date to index of test partition
            # test partition start from latest BTC halving: 2020-05-11 to 2020-09-10

            # lstm
            start = date(2020, 9, 11)
            end = date(2020, 9, 21)
            delta = end - start
            dates_range = pd.date_range(
                start, end-timedelta(days=1), freq='d').astype(str).to_list()

            date2id = dict(zip(dates_range, range(
                len(prediction)-delta.days, len(prediction), 1)))
            # print(len(prediction),len(dates_range),date2id)
            self.predictions[k] = (prediction, date2id)

            # RF PREDICTIONS
            r = f'RF-{coin}'
            prediction_rf = pd.read_csv(
                f"models/predictions/rf_price_change_pred_{coin}.csv").to_numpy()
            date2id_rf = dict(zip(dates_range, range(0, len(prediction_rf))))
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

        lstm_card = create_card(
            [
                html.H4("LSTM Model Forecast", id="lstm"),
                html.H2("...", id="lstm-value"),
            ]
        )

        rf_card = create_card(
            [
                html.H4("Random Forest Model Forecast", id="rf"),
                html.H2("...", id="rf-value"),
            ]
        )

        slider_card = create_card(
            [
                html.H6('Date of Prediction:'),
                dcc.Slider(id='date-selector-pc',
                           min=0,
                           max=9,
                           marks={i: '{}'.format(dates_range[i])
                                  for i in range(0, len(dates_range))},
                           value=0,
                           ),
            ]
        )

        self.layout = html.Div([
            html.H3('Price Change Forecast', id='price-change-title'),
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
                    dbc.Col([
                        dash_table.DataTable(
                            id='data-table',
                        )
                    ]),
                    dbc.Col([
                        dbc.Row([
                            dbc.Col([slider_card])
                        ]),
                        dbc.Row([
                            dbc.Col([lstm_card]), dbc.Col([rf_card])
                        ])
                    ])
                ])
            ]),


            html.Br(),


            # dcc.Link('Go back to home', href='/'),
        ])

        #################################
        # HTML CALLBACKS
        #################################

        def format_output(label):
            label = re.sub('[(],]', '', label)
            l = float(label.split(" ")[0][1:-1])
            u = float(label.split(" ")[1][1:-1])
            return f"{l}% to {u}%"

        @self.app.callback(
            [Output('data-table', 'data'), Output('data-table', 'columns')],
            [Input('pricechange-coin-dropdown', 'value')])
        def update_table(coin):
            N = 10

            lstm_pred = self.predictions[f"LSTM-{coin}"][0]
            lstm_values = [format_output(i[1]) for i in lstm_pred][-N:]

            start = date(2020, 9, 11)
            end = date(2020, 9, 21)
            dates = pd.date_range(
                start, end-timedelta(days=1), freq='d').astype(str).to_list()

            rf_pred = self.predictions[f"RF-{coin}"][0]
            rf_values = [format_output(i[1]) for i in rf_pred][-N:]

            # print(len(lstm_values),len(dates),len(rf_values))

            data = pd.DataFrame({
                "Dates": dates,
                "LSTM": lstm_values,
                "RF": rf_values
            })
            return data.to_dict('records'), [{"name": i, "id": i} for i in data.columns]

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
            [Input('pricechange-coin-dropdown', 'value'), Input('date-selector-pc', 'value')])
        def select_date(coin, date_index):
            selected_date = dates_range[date_index]

            prediction = self.predictions[f"LSTM-{coin}"][0]
            id_mapping = self.predictions[f"LSTM-{coin}"][1]
            ind = id_mapping[selected_date]

            prediction_rf = self.predictions[f"RF-{coin}"][0]
            id_mapping_rf = self.predictions[f"RF-{coin}"][1]
            ind_rf = id_mapping_rf[selected_date]
            return format_output(prediction[ind][1]), format_output(prediction_rf[ind_rf][1])
