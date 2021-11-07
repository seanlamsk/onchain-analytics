import pandas as pd
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import plotly.express as px
from dash_core_components import Dropdown, Graph, Slider
import dash_bootstrap_components as dbc
from dash import html, dcc
import dash
import sys
import re
sys.path.insert(0, '..')


class PricePrediction:

    def __init__(self, app):
        self.app = app
        self.coin = 'btc'
        self.value = 1
        self.prediction = pd.read_csv(
            f'models/predictions/{self.coin}_price_pred_prediction.csv')
        self.actual = pd.read_csv(
            f'models/predictions/{self.coin}_price_pred_actual.csv')
        self.ma = pd.read_csv(
            f'models/predictions/{self.coin}_price_pred_ma.csv')
        self.stats = pd.read_csv(
            'models/predictions/price_pred_stats.csv', index_col=0)
        self.previous_coin = 'Bitcoin'

        def create_card(html_elements):
            return dbc.Card(
                dbc.CardBody(
                    html_elements
                )
            )

        arima_pred_card = create_card(
            [
                html.H4('ARIMA Price Prediction'),
                dbc.Row([
                    dbc.Col(html.Div(id='Arima_pred')),
                    dbc.Col(html.Div(id='Arima_change', style={
                        "color": 'Black'
                    })),
                ]
                )
            ]
        )

        lstm_pred_card = create_card(
            [
                html.H4('LSTM Price Prediction'),
                dbc.Row([
                    dbc.Col(html.Div(id='Lstm_pred')),
                    dbc.Col(html.Div(id='Lstm_change', style={
                        "color": 'Black'
                    })),
                ]
                )
            ]
        )

        returns_card = create_card(
            [
                html.H4('Annualised returns'),
                html.Div(id='returns'),
            ]
        )

        volatility_card = create_card(
            [
                html.H4('Annualised volatility'),
                html.Div(id='volatility'),
            ]
        )

        slider_card = create_card(
            [
                html.H6('Date of Prediction:'),
                Slider(id='date-selector',
                       min=0,
                       max=9,
                       marks={i-1: '{}'.format(self.prediction['Date'][i])
                              for i in range(1, len(self.prediction['Date']))},
                       value=0,
                       ),
            ]
        )

        dropdown_card = create_card(
            [
                Dropdown(
                    id='crypto-select',
                    options=[
                        {'label': 'Bitcoin', 'value': 'Bitcoin'},
                        {'label': 'Ethereum', 'value': 'Ethereum'},
                        {'label': 'Litecoin', 'value': 'Litecoin'}
                    ],
                    value='selected_coin',
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

        checkbox_card = create_card(
            [
                dbc.Row([
                    dbc.Col(
                        dcc.Checklist(
                            id='checklist_select',
                            options=[
                                {'label': "20d MA", 'value': '20dMA'},
                                {'label': "50d MA", 'value': '50dMA'},
                            ],
                            value=[],
                            labelStyle={'display': 'inline-block'}
                        )
                    ),
                    dbc.Col([
                        html.H6('Plot Price Level: '),
                        dbc.Input(
                            id='threshold', placeholder='Enter Price Level', type='text', debounce=True),
                    ])
                ])
            ]
        )

        self.layout = html.Div([
            html.H3('Price Prediction Models'),
            html.Br(),
            html.Div([
                dbc.Row([
                    dbc.Col([
                        dbc.Row(Graph(
                            id='comparison-graph'
                        ),),
                        dbc.Row([
                            checkbox_card
                        ])
                    ]
                    ),
                    dbc.Col([
                        dbc.Row([
                            dbc.Col([dropdown_card])
                        ]),
                        dbc.Row([
                            dbc.Col([slider_card])
                        ]),
                        dbc.Row([
                            dbc.Col([arima_pred_card]), dbc.Col(
                                [lstm_pred_card])
                        ]),
                        dbc.Row([
                            dbc.Col([returns_card]), dbc.Col([volatility_card])
                        ]), ]
                    )
                ]),
            ]),
        ])

        @self.app.callback(dash.dependencies.Output('comparison-graph', 'figure'), dash.dependencies.Output('threshold', 'valid'),
                           dash.dependencies.Input('crypto-select', 'value'),
                           dash.dependencies.Input(
                               'checklist_select', 'value'),
                           dash.dependencies.Input('threshold', 'value'),
                           )
        def update_graph(value, value2, value3):
            self.coin = 'btc'
            name = value
            if value == 'selected_coin':
                name = 'Bitcoin'
            if value == 'Ethereum':
                self.coin = 'eth'
            elif value == 'Litecoin':
                self.coin = 'ltc'
            self.prediction = pd.read_csv(
                f'models/predictions/{self.coin}_price_pred_prediction.csv')
            self.actual = pd.read_csv(
                f'models/predictions/{self.coin}_price_pred_actual.csv')
            self.ma = pd.read_csv(
                f'models/predictions/{self.coin}_price_pred_ma.csv')
            traceArima = go.Scatter(
                x=self.prediction['Date'], y=self.prediction['ARIMA'], mode='lines', name="ARIMA")
            traceLstm = go.Scatter(
                x=self.prediction['Date'], y=self.prediction['LSTM'], mode='lines', name="LSTM")
            traceActual = go.Scatter(
                x=self.actual["Date"], y=self.actual['close'], mode='lines', name="Actual Price")
            tracema20 = go.Scatter(
                x=self.ma["Date"], y=self.ma['MA_20'], name="20d MA", line=dict(dash='dot'))
            tracema50 = go.Scatter(
                x=self.ma["Date"], y=self.ma['MA_50'], name="50d MA", line=dict(dash='dot'))
            df = [traceArima, traceLstm, traceActual]
            if ('20dMA' in value2):
                df.append(tracema20)
            if ('50dMA' in value2):
                df.append(tracema50)
            layout = go.Layout(
                title=f'Price Graph of {name} (with Price Prediction)')
            figure = go.Figure(data=df, layout=layout)
            valid = True
            if self.previous_coin == name:
                try:
                    value3 = int(value3)
                    figure.add_hline(y=value3, line_width=2, line_dash='dash')
                    valid = True
                except:
                    try:
                        value3 = float(value3)
                        figure.add_hline(
                            y=value3, line_width=2, line_dash='dash')
                        valid = True
                    except:
                        valid = False
            else:
                self.previous_coin = name
                figure.add_hline(y='', line_width=2, line_dash='dash')
            return figure, valid

        @self.app.callback(dash.dependencies.Output('returns', 'children'),
                           [dash.dependencies.Input('crypto-select', 'value')])
        def update_returns(value):
            coin = 'btc'
            if value == 'Ethereum':
                coin = 'eth'
            elif value == 'Litecoin':
                coin = 'ltc'
            return f"{self.stats.loc[coin, 'Annualised Returns']:.2f}%"

        @self.app.callback(dash.dependencies.Output('volatility', 'children'),
                           [dash.dependencies.Input('crypto-select', 'value')])
        def update_volatility(value):
            coin = 'btc'
            if value == 'Ethereum':
                coin = 'eth'
            elif value == 'Litecoin':
                coin = 'ltc'
            return f"{self.stats.loc[coin, 'Volatility']:.2f}"

        @self.app.callback(dash.dependencies.Output('Arima_pred', 'children'),
                           dash.dependencies.Output('date-selector', 'marks'),
                           dash.dependencies.Output(
                               'Arima_change', 'children'),
                           dash.dependencies.Output('Arima_change', 'style'),
                           dash.dependencies.Input('date-selector', 'value'),
                           dash.dependencies.Input('crypto-select', 'value')
                           )
        def update_pred_arima(value1, value2):
            self.value = value1+1
            pred_value = self.prediction.iloc[self.value]['ARIMA']
            change = pred_value - self.actual.iloc[-1]['close']
            percent_change = change/self.actual.iloc[-1]['close']*100
            if percent_change > 0:
                colorarima = 'Green'
                symbol = '+'
            elif percent_change < 0:
                colorarima = 'Red'
                symbol = ''
            else:
                colorarima = 'Black'
                symbol = ''
            return_value = f'{symbol}{change:,.2f} ({symbol}{percent_change:,.2f}%)'
            return (f"US$ {pred_value:,.2f}", {i-1: '{}'.format(self.prediction['Date'][i]) for i in range(1, len(self.prediction['Date']))}, return_value, {"color": colorarima})

        @self.app.callback(dash.dependencies.Output('Lstm_pred', 'children'),
                           dash.dependencies.Output('Lstm_change', 'children'),
                           dash.dependencies.Output('Lstm_change', 'style'),
                           dash.dependencies.Input('date-selector', 'value'),
                           dash.dependencies.Input('crypto-select', 'value')
                           )
        def update_pred_lstm(value1, value2):
            self.value = value1+1
            pred_value = self.prediction.iloc[self.value]['LSTM']
            change = pred_value - self.actual.iloc[-1]['close']
            percent_change = change/self.actual.iloc[-1]['close']*100
            if percent_change > 0:
                colorlstm = 'Green'
                symbol = '+'
            elif percent_change < 0:
                colorlstm = 'Red'
                symbol = ''
            else:
                colorlstm = 'Black'
                symbol = ''
            return_value = f'{symbol}{change:,.2f} ({symbol}{percent_change:,.2f}%)'
            return (f"US$ {pred_value:,.2f}", return_value, {"color": colorlstm})


# go.Scatter(x=month, y=low_2000, name='Low 2000',
#                          line=dict(color='royalblue', width=4, dash='dot')))
