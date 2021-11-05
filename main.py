import dash
# from dash import dcc
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash import html
import plotly.express as px
import pandas as pd

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
                            html.Img(src="/assets/img/icon.png", height='50px')),
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
                style={"textDecoration": "none"},
            )
        ]
    ),
    color="#0e153d",
    dark=True,
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
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

index_page = html.Div([
    html.H1('Overview Dashboard'),
    html.Div([
        dbc.Row([
            dbc.Col([arima_card]),
            dbc.Col([lstm_card]),
        ])
    ])
])

# index_page = html.Div([
#     dcc.Link('View Price Forecast', href='/price-forecast'),
#     html.Br(),

#     dcc.Link('View Price Change Forecast', href='/price-change-forecast'),
#     html.Br(),

#     dcc.Link('View Portfolio Analysis', href='/portfolio-analysis'),
#     html.Br(),
# ])

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


# page_2_layout = html.Div([
#     html.H1('Page 2'),
#     dcc.RadioItems(
#         id='page-2-radios',
#         options=[{'label': i, 'value': i} for i in ['Orange', 'Blue', 'Red']],
#         value='Orange'
#     ),
#     html.Div(id='page-2-content'),
#     html.Br(),
#     dcc.Link('Go to Page 1', href='/page-1'),
#     html.Br(),
#     dcc.Link('Go back to home', href='/')
# ])

# @app.callback(dash.dependencies.Output('page-2-content', 'children'),
#               [dash.dependencies.Input('page-2-radios', 'value')])
# def page_2_radios(value):
#     return 'You have selected "{}"'.format(value)
