import sys
sys.path.insert(0,'..')
import dash
import dash_html_components as html
import dash_core_components as dcc
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
        self.ARIMA_pred = ArimaModel('btc_metrics_raw.csv', 10, './models/saved_models/btc/arimamodel.pkl', 'load').arima_pred_future()
        self.LSTM_pred = LSTMModel('btc_metrics_raw.csv', './models/saved_models/btc/lstm_price_predictor.hp5', 'load').forecast()

        self.layout = html.Div([
            html.H1('Price Prediction Models'),
            Dropdown(
                id = 'crypto-select',
                options = [
                    {'label': 'Bitcoin', 'value': 'btc_metrics_raw.csv'},
                    {'label': 'Ethereum', 'value': 'eth_metrics_raw.csv'},
                    {'label': 'Litecoin', 'value': 'ltc_metrics_raw.csv'}
                ],
                value = 'modelname',
                searchable=False,
                clearable=False,
                placeholder="Bitcoin"
            ),
            Slider(id='date-selector',
                min=0, 
                max=6,
                marks = {i: '{}'.format(self.ARIMA_pred.index[i]) for i in range(len(self.ARIMA_pred.index))},
                value = 0,
                ),
            html.H4('ARIMA Price Prediction'),
            html.Div(id='Arima_pred'),
            html.H4('LSTM Price Prediction'),
            html.Div(id='Lstm_pred'),
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
            name_coin = 'Bitcoin'
            arima_model_name = './models/saved_models/btc/arimamodel.pkl'
            lstm_model_name = './models/saved_models/btc/lstm_price_predictor.hp5'
            if value == "modelname":
                value = "btc_metrics_raw.csv"
            if value == 'eth_metrics_raw.csv':
                arima_model_name = './models/saved_models/eth/arimamodel.pkl'
                lstm_model_name = './models/saved_models/eth/lstm_price_predictor.hp5'
                name_coin = 'Ethereum'
            elif value == 'ltc_metrics_raw.csv':
                arima_model_name = './models/saved_models/ltc/arimamodel.pkl'
                lstm_model_name = './models/saved_models/ltc/lstm_price_predictor.hp5'
                name_coin = 'Litecoin'
            Arima_past_seven = ArimaModel(value, 10, arima_model_name, 'load').arima_pred_past_seven()
            Arima_past_seven = Arima_past_seven.to_frame().reset_index()
            LSTM_past_seven = LSTMModel(value, lstm_model_name, 'load').past_seven_days_forecast()
            LSTM_past_seven = LSTM_past_seven.reset_index().rename(columns={0: 'LSTM'}) 
            original_df = pd.read_csv(value)[-7:]
            df = pd.merge(Arima_past_seven, LSTM_past_seven, on='Date')
            traceArima = go.Scatter(x=df["Date"], y = df["ARIMA"], mode = 'lines', name = "ARIMA")
            traceLstm = go.Scatter(x=df["Date"], y = df["LSTM"], mode = 'lines', name = "LSTM")
            traceOG = go.Scatter(x=original_df["Date"], y = original_df["close"], mode = 'lines', name = "Actual Price")
            df = [traceArima, traceLstm, traceOG]
            layout = go.Layout(title = 'Comparison Between LSTM and ARIMA with ' + name_coin)
            figure = go.Figure(data=df, layout=layout)
            return figure
        
        @self.app.callback(dash.dependencies.Output('test', 'children'),
            [dash.dependencies.Input('crypto-select', 'value')])
        def update_pred_obj(value):
            self.ARIMA_pred = ArimaModel('btc_metrics_raw.csv', 10, './models/saved_models/btc/arimamodel.pkl', 'load').arima_pred_future()
            self.LSTM_pred = LSTMModel('btc_metrics_raw.csv', './models/saved_models/btc/lstm_price_predictor.hp5', 'load').forecast()
            if value == 'eth_metrics_raw.csv':
                self.ARIMA_pred = ArimaModel('eth_metrics_raw.csv', 10, './models/saved_models/eth/arimamodel.pkl', 'load').arima_pred_future()
                self.LSTM_pred = LSTMModel('eth_metrics_raw.csv', './models/saved_models/eth/lstm_price_predictor.hp5', 'load').forecast()
            elif value == 'ltc_metrics_raw.csv':
                self.ARIMA_pred = ArimaModel('ltc_metrics_raw.csv', 10, './models/saved_models/ltc/arimamodel.pkl', 'load').arima_pred_future()
                self.LSTM_pred = LSTMModel('ltc_metrics_raw.csv', './models/saved_models/ltc/lstm_price_predictor.hp5', 'load').forecast()

        @self.app.callback(dash.dependencies.Output('Arima_pred', 'children'),
                    [dash.dependencies.Input('date-selector', 'value')])
        def update_pred_arima(value):
            return self.ARIMA_pred[value]
            
        @self.app.callback(dash.dependencies.Output('Lstm_pred', 'children'),
                    [dash.dependencies.Input('date-selector', 'value')])
        def update_pred_lstm(value):
            return self.LSTM_pred[0][value]