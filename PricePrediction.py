import sys
sys.path.insert(0,'..')
import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash_core_components import Dropdown, Graph, Slider
import plotly.express as px
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import pandas as pd

from models.arima_model import ArimaModel
from models.LSTM_PricePrediction import LSTMModel


class PricePrediction:

    def __init__(self,app):
        self.app = app
        self.coin = 'btc'
        self.value = 1
        self.prediction = pd.read_csv(f'models/predictions/{self.coin}_price_pred_prediction.csv')
        self.actual = pd.read_csv(f'models/predictions/{self.coin}_price_pred_actual.csv')
        self.stats = pd.read_csv('models/predictions/price_pred_stats.csv', index_col=0)

        def create_card(html_elements):
            return dbc.Card(
                dbc.CardBody(
                    html_elements
                )
            )      

        arima_pred_card =  create_card(
            [
                html.H4('ARIMA Price Prediction'),
                html.Div(id='Arima_pred'),
            ]
        )

        lstm_pred_card =  create_card(
            [
                html.H4('LSTM Price Prediction'),
                html.Div(id='Lstm_pred'),
            ]
        )

        returns_card =  create_card(
            [
                html.H4('Annualised returns'),
                html.Div(id='returns'),
            ]
        )

        volatility_card =  create_card(
            [
                html.H4('Annualised volatility'),
                html.Div(id='volatility'),
            ]
        )

        slider_card =  create_card(
            [
                Slider(id='date-selector',
                    min=0, 
                    max=9,
                    marks = {i-1: '{}'.format(self.prediction['Date'][i]) for i in range(1, len(self.prediction['Date']))},
                    value = 0,
                    ),
            ]
        )

        dropdown_card =  create_card(
            [
                Dropdown(
                    id = 'crypto-select',
                    options = [
                        {'label': 'Bitcoin', 'value': 'Bitcoin'},
                        {'label': 'Ethereum', 'value': 'Ethereum'},
                        {'label': 'Litecoin', 'value': 'Litecoin'}
                    ],
                    value = 'selected_coin',
                    searchable=False,
                    clearable=False,
                    placeholder="Bitcoin",
                    # style = dict(
                    #             width='90%',
                    #             display='inline-block',
                    #             verticalAlign="middle"
                    #         )
            ),
            
            ]
        )

        self.layout = html.Div([
            html.H1('Price Prediction Models'),
            html.Br(),
            html.Div([
                dbc.Row([
                    dbc.Col([dropdown_card])
                ]), 
                dbc.Row([
                    dbc.Col([slider_card])
                ]),
                dbc.Row([
                    dbc.Col([arima_pred_card]), dbc.Col([lstm_pred_card])
                ]),
                dbc.Row([
                    dbc.Col([returns_card]), dbc.Col([volatility_card])
                ]),]),
            html.H4('Price Prediction Graph'),
            Graph(
                id = 'comparison-graph'
            ),
            html.Div(id='test'),
            html.Br(),
            # dcc.Link('Go to Page 2', href='/page-2'),
            html.Br(),
            dcc.Link('Go back to home', href='/'),
        ])

        @self.app.callback(dash.dependencies.Output('comparison-graph', 'figure'),
                    [dash.dependencies.Input('crypto-select', 'value')])
        def update_graph(value):
            self.coin = 'btc'
            if value == 'Ethereum':
                self.coin = 'eth'
            elif value == 'Litecoin':
                self.coin = 'ltc'
            self.prediction = pd.read_csv(f'models/predictions/{self.coin}_price_pred_prediction.csv')
            self.actual = pd.read_csv(f'models/predictions/{self.coin}_price_pred_actual.csv')
            traceArima = go.Scatter(x=self.prediction['Date'], y = self.prediction[f'ARIMA'], mode = 'lines', name = "ARIMA")
            traceLstm = go.Scatter(x=self.prediction['Date'], y = self.prediction[f'LSTM'], mode = 'lines', name = "LSTM")
            traceActual = go.Scatter(x=self.actual["Date"], y = self.actual[f'close'], mode = 'lines', name = "Actual Price")
            df = [traceArima, traceLstm, traceActual]
            layout = go.Layout(title = 'Price Prediction of LSTM and ARIMA with ' + value)
            figure = go.Figure(data=df, layout=layout)
            return figure

        @self.app.callback(dash.dependencies.Output('returns', 'children'),
                    [dash.dependencies.Input('crypto-select', 'value')])
        def update_returns(value):
            coin = 'btc'
            if value == 'Ethereum':
                coin = 'eth'
            elif value == 'Litecoin':
                coin = 'ltc'
            return self.stats.loc[coin, 'Annualised Returns']

        @self.app.callback(dash.dependencies.Output('volatility', 'children'),
                    [dash.dependencies.Input('crypto-select', 'value')])
        def update_volatility(value):
            coin = 'btc'
            if value == 'Ethereum':
                coin = 'eth'
            elif value == 'Litecoin':
                coin = 'ltc'
            return self.stats.loc[coin, 'Volatility']
        # @self.app.callback(dash.dependencies.Output('test', 'children'),
        #             [dash.dependencies.Input('date-selector', 'value')])
        # def up(value):
        #     return value

        @self.app.callback(dash.dependencies.Output('Arima_pred', 'children'), dash.dependencies.Output('date-selector', 'marks'),
                    dash.dependencies.Input('date-selector', 'value'),
                    dash.dependencies.Input('crypto-select', 'value')
                    )
        def update_pred_arima(value1, value2):
            self.value = value1+1
            return (self.prediction.iloc[self.value]['ARIMA'], {i-1: '{}'.format(self.prediction['Date'][i]) for i in range(1, len(self.prediction['Date']))})
            
        @self.app.callback(dash.dependencies.Output('Lstm_pred', 'children'),
                    dash.dependencies.Input('date-selector', 'value'),
                    dash.dependencies.Input('crypto-select', 'value')
                    )
        def update_pred_lstm(value1, value2):
            self.value = value1+1
            return self.prediction.iloc[self.value]['LSTM']
