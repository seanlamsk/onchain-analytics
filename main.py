import dash
# from dash import dcc
import dash_bootstrap_components as dbc
from dash import html, dcc
import plotly.express as px
import pandas as pd
import re

from PriceChangeForecast import PriceChangeForecast
from PriceChangeForecastComparison import PriceChangeForecastComparison
from Portfolios import Portfolios_Analysis
from PricePrediction import PricePrediction


# Since we're adding callbacks to elements that don't exist in the app.layout,
# Dash will raise an exception to warn us that we might be
# doing something wrong.
# In this case, we're adding the elements through a callback, so we can ignore
# the exception.
app = dash.Dash(__name__,
                suppress_callback_exceptions=True,
                external_stylesheets=[dbc.themes.BOOTSTRAP])

price_change_page = PriceChangeForecast(app)
price_change_comparison_page = PriceChangeForecastComparison(app)
portfolio_analysis = Portfolios_Analysis(app)
price_pred_page = PricePrediction(app)


def trend_of_value(coin, model):
    df = pd.read_csv(f'models/predictions/{coin}_price_pred_prediction.csv')
    last_pred = df.iloc[-1][model]
    last_actual = df.iloc[0][model]
    price_diff = last_pred - last_actual
    trend_value = 'BULLISH' if price_diff > 0 else 'BEARISH' if price_diff < 0 else 'NEUTRAL'
    trend_color = 'Green' if price_diff > 0 else 'Red' if price_diff < 0 else 'Gray'
    return (trend_value, trend_color)


def create_card(html_elements):
    return dbc.Card(
        dbc.CardBody(
            html_elements
        )
    )

def format_pc_output(label):
    label = re.sub('[(],]', '', label)
    l = float(label.split(" ")[0][1:-1])
    u = float(label.split(" ")[1][1:-1])
    return f"{l}% to {u}%"

def price_change_pred(coin):
    
    k = f'LSTM-{coin}'
    # store prediction from test set in memory
    prediction = pd.read_csv(f"models/predictions/lstm_price_change_pred_{coin}.csv").to_numpy()

    # RF PREDICTIONS
    prediction_rf = pd.read_csv(f"models/predictions/rf_price_change_pred_{coin}.csv").to_numpy()

    return format_pc_output(prediction[-1:][0][1]),format_pc_output(prediction_rf[-1:][0][1])


arima_card = create_card(
    [
        html.H2('ARIMA'),
        dbc.Row([
            dbc.Col(
                html.H4('Bitcoin:')),
            dbc.Col(
                html.H4(trend_of_value('btc', 'ARIMA')[0], style={"color": trend_of_value('btc', 'ARIMA')[1]})),
        ]),
        dbc.Row([
            dbc.Col(
                html.H4('Ethereum:')),
            dbc.Col(
                html.H4(trend_of_value('eth', 'ARIMA')[0], style={"color": trend_of_value('eth', 'ARIMA')[1]})),
        ]),
        dbc.Row([
            dbc.Col(
                html.H4('Litecoin:')),
            dbc.Col(
                html.H4(trend_of_value('ltc', 'ARIMA')[0], style={"color": trend_of_value('ltc', 'ARIMA')[1]})),
        ])
    ]
)


lstm_card = create_card(
    [
        html.H2('LSTM'),
        dbc.Row([
                dbc.Col(
                    html.H4('Bitcoin:')),
                dbc.Col(
                    html.H4(trend_of_value('btc', 'LSTM')[0], style={"color": trend_of_value('btc', 'LSTM')[1]})),
                ]),
        dbc.Row([
                dbc.Col(
                    html.H4('Ethereum:')),
                dbc.Col(
                    html.H4(trend_of_value('eth', 'LSTM')[0], style={"color": trend_of_value('eth', 'LSTM')[1]})),
                ]),
        dbc.Row([
                dbc.Col(
                    html.H4('Litecoin:')),
                dbc.Col(
                    html.H4(trend_of_value('ltc', 'LSTM')[0], style={"color": trend_of_value('ltc', 'LSTM')[1]})),
                ])
    ])


rf_pc_card = create_card(
    [
        html.H2('RF Price Change Forecast'),
        dbc.Row([
                dbc.Col(
                    html.H4('Bitcoin:')),
                dbc.Col(
                    html.H4(price_change_pred('btc')[1])),
                ]),
        dbc.Row([
                dbc.Col(
                    html.H4('Ethereum:')),
                dbc.Col(
                    html.H4(price_change_pred('eth')[1])),
                ]),
        dbc.Row([
                dbc.Col(
                    html.H4('Litecoin:')),
                dbc.Col(
                    html.H4(price_change_pred('ltc')[1])),
                ])
])    

lstm_pc_card = create_card(
    [
        html.H2('LSTM Price Change Forecast'),
        dbc.Row([
                dbc.Col(
                    html.H4('Bitcoin:')),
                dbc.Col(
                    html.H4(price_change_pred('btc')[0])),
                ]),
        dbc.Row([
                dbc.Col(
                    html.H4('Ethereum:')),
                dbc.Col(
                    html.H4(price_change_pred('eth')[0])),
                ]),
        dbc.Row([
                dbc.Col(
                    html.H4('Litecoin:')),
                dbc.Col(
                    html.H4(price_change_pred('ltc')[0])),
                ])
])


nav = dbc.Nav(
    [
        dbc.NavItem(dbc.NavLink('Home', href='/')),
        dbc.NavItem(dbc.NavLink(
            'Price Prediction Dashboard', href='/price-forecast')),
        dbc.NavItem(dbc.NavLink('Price Change Dashboard',
                    href='/price-change-forecast')),
        dbc.NavItem(dbc.NavLink('Portfolio Dashboard',
                    href='/portfolio-analysis')),
    ],
    justified=True,
    pills=True,
    style={
        # 'background-color': '#2b367a'
    },
)

navbar = dbc.Navbar(
    dbc.Container(
        [
            html.A([
                dbc.Row(
                    [
                        dbc.Col(
                            html.Img(src="/assets/img/icon.png", height='150px')),
                        dbc.Col(dbc.NavbarBrand("On Chain Analytics Dashboard", className='ms-2', 
                                style={'font-size': "40px"}), style={'verticalAlign': 'left'}),
                    ],
                    align='center',
                    className='g-0',
                    style={'size': 10}
                ),
                dbc.Row(
                    dbc.Col(nav, style={}),
                    style={
                        'size': 10,
                        'width': '90%'
                    }
                )
            ],
                style={"textDecoration": "none", "font-family":"Garamond", "font-weight":600},
            )
        ]
    ),
    color="#9BC4C4",
    dark=False,
)


app.layout = html.Div([
    html.Div([
        dbc.Row([
            dbc.Col([navbar])
        ]),
        # dbc.Row([
        #     nav
        # ])
    ],
    ),
    html.Br(),
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

index_page = html.Div([
    html.H1('Overview Dashboard'),
    html.Div([
        dbc.Row([
            dbc.Col([arima_card]),
            dbc.Col([lstm_card]),
        ]),
        dbc.Row([
            dbc.Col([rf_pc_card]),
            dbc.Col([lstm_pc_card]),
        ])
    ])
])

# Update the index


@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/price-change-forecast':
        return price_change_page.layout
    elif pathname == '/price-change-forecast-compare':
        return price_change_comparison_page.layout
    elif pathname == '/price-forecast':
        return price_pred_page.layout
    elif pathname == '/portfolio-analysis':
        return portfolio_analysis.layout
    else:
        return index_page
    # You could also return a 404 "URL not found" page here

if __name__ == '__main__':
    app.run_server(debug=True)

