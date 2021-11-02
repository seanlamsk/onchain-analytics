import numpy as np
import pandas as pd
from pandas_datareader import data
import matplotlib.pyplot as plt
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.validators.scatter.marker import SymbolValidator

def readDataCreateDF():
    # Variables must be declared in the order below
    # btc,eth,ltc,df = readDataCreateDF()
    btc = pd.read_csv('btc_metrics_raw.csv', parse_dates = ['Date'])
    eth = pd.read_csv('eth_metrics_raw.csv', parse_dates = ['Date'])
    ltc = pd.read_csv('ltc_metrics_raw.csv', parse_dates = ['Date'])
    df = pd.DataFrame()
    return btc,eth,ltc,df
# btc,eth,ltc,df = readDataCreateDF()


def btc_df(btc,df):
    #Making the btc df, also sets the index as date
    btc['close_pct'] = btc['close'].pct_change().apply(lambda x: np.log(1+x))
    # btc.head()
    df['date'] = btc['Date']
    df['btc'] = btc['close_pct']
    return btc
# btc = btc_df(btc,df)

def eth_df(eth,df):
    #Making the eth df
    eth['close_pct'] = eth['close'].pct_change().apply(lambda x: np.log(1+x))
    # eth.head()
    df['eth'] = eth['close_pct']
    return eth
# eth = eth_df(eth,df)

def ltc_df(ltc,df):
    #Making the ltc df
    ltc['close_pct'] = ltc['close'].pct_change().apply(lambda x: np.log(1+x))
    # ltc.head()
    df['ltc'] = ltc['close_pct']
    return ltc
# ltc = ltc_df(ltc,df)

def get_var(df):
    """Takes in df of all 3 coins and returns 3 variance values: btc_var,eth_var,ltc_var"""
    #Variance
    btc_rs = df['btc']
    eth_rs = df['eth']
    ltc_rs = df['ltc']
    btc_var = btc_rs.var()
    eth_var = eth_rs.var()
    ltc_var = ltc_rs.var()
    print('Variance: \n' , 'BTC:', btc_var, '\n', 'ETH:', eth_var, '\n', 'LTC:', ltc_var, '\n')
    return btc_var,eth_var,ltc_var

def getVolatility(btc_var,eth_var,ltc_var):
    """Takes in var of all 3 coins and returns 3 volatility values: btc_vol,eth_vol,ltc_vol"""
    #Volatility
    btc_vol = np.sqrt(btc_var * 365)
    eth_vol = np.sqrt(eth_var * 365)
    ltc_vol = np.sqrt(ltc_var * 365)
    print('Volatility: \n' , 'BTC:', btc_vol, '\n', 'ETH:', eth_vol, '\n', 'LTC:', ltc_vol, '\n')
    return btc_vol,eth_vol,ltc_vol

#Covariance
def getCovariance(df):
    """Takes in df and returns a covariance matrix"""
    cov_matrix = df.cov()
    return cov_matrix


#Correlation
def getCorrelation(df):
    """Takes in df and returns correlation matrix"""
    corr_matrix = df.corr()
    return corr_matrix
# corr_matrix

#Expected returns on 40% BTC, 30% ETH, 30% LTC portfolio
# w = [0.4,0.3,0.3]
# e_r_ind = df.mean()
# e_r = (e_r_ind*w).sum()

def plotEfficientFrontier(df, cov_matrix):
    
    #Setting index as date
    df = df.set_index('date')
    ind_er = df.resample('Y').last().mean()
    # ind_er

    ann_sd = df.std().apply(lambda x: x*np.sqrt(365))
    # ann_sd

    w = [0.4, 0.3, 0.3]
    port_er = (w*ind_er).sum()
    # port_er

    ann_sd = df.apply(lambda x: np.log(1+x)).std().apply(lambda x: x*np.sqrt(365))
    # ann_sd

    tokens = pd.concat([ind_er, ann_sd], axis=1) # Creating a table for visualising returns and volatility of assets
    tokens.columns = ['Returns', 'Volatility']
    # tokens

    p_ret = [] # Define an empty array for portfolio returns
    p_vol = [] # Define an empty array for portfolio volatility
    p_weights = [] # Define an empty array for asset weights

    num_tokens = len(df.columns)
    num_portfolios = 10000

    for portfolio in range(num_portfolios):
        weights = np.random.random(num_tokens)
        weights = weights/np.sum(weights)
        p_weights.append(weights)
        returns = np.dot(weights, ind_er) # Returns are the product of individual expected returns of asset and its 
                                        # weights 
        p_ret.append(returns.round(6))
        var = cov_matrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum()# Portfolio Variance
        sd = np.sqrt(var) # Daily standard deviation
        ann_sd = sd*np.sqrt(250) # Annual standard deviation = volatility
        p_vol.append(ann_sd.round(5))

    data = {'Returns':p_ret, 'Volatility':p_vol}

    for counter, symbol in enumerate(df.columns.tolist()):
        # print(counter, symbol)
        data[symbol+' weight'] = [w[counter] for w in p_weights]

    portfolios  = pd.DataFrame(data)
    # portfolios.plot.scatter(x='Volatility', y='Returns', marker='o', s=10, alpha=0.3, grid=True, figsize=[10,10])
    
    fig = px.scatter(portfolios, x= 'Volatility', y = 'Returns', title = 'Portfolios', hover_data = {'btc weight':':.2f', 'eth weight':':.2f', 'ltc weight':':.2f'})
    min_vol_port = portfolios.iloc[portfolios['Volatility'].idxmin()]
    rf = 0.01 # risk factor
    optimal_risky_port = portfolios.iloc[((portfolios['Returns']-rf)/portfolios['Volatility']).idxmax()]
    fig.add_trace(
    go.Scatter(
        mode='markers',
        x=[optimal_risky_port['Volatility']],
        y=[optimal_risky_port['Returns']],
        name = "Optimal Risky Portfolio",
        hovertemplate = "Volatility: %{x} <br> Returns: %{y} <br> BTC: %{optimal_risky_port[`btc`]}% <br> ETH: %{optimal_risky_port[`eth`]}% <br> %{optimal_risky_port[`ltc`]}%" ,
        marker=dict(
            color='Red',
            size=30,
            line=dict(
                color='Black',
                width=3
            )
        ),
        showlegend=True
    )
    )

    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=[min_vol_port['Volatility']],
            y=[min_vol_port['Returns']],
            name = "Minimal Volatility Portfolio",
            hovertemplate = "Volatility: %{x} <br> Returns: %{y} <br> BTC: %{min_vol_port[`btc`]}% <br> ETH: %{min_vol_port[`eth`]}% <br> %{min_vol_port[`ltc`]}%" ,
            marker=dict(
                color='Green',
                size=30,
                line=dict(
                    color='Black',
                    width=3
                )
            ),
            showlegend=True
        )
    )
    
    return  fig, min_vol_port, optimal_risky_port
    # return 
    

def plotEfficientFrontierfromRawCSV():
    btc,eth,ltc,df = readDataCreateDF()
    btc = btc_df(btc,df)
    eth = eth_df(eth,df)
    ltc = ltc_df(ltc,df)
    # btc_var,eth_var,ltc_var = get_var(df)
    # btc_vol, eth_vol, ltc_vol = getVolatility(btc_var,eth_var,ltc_var)
    # corr = getCorrelation(df)
    cov = getCovariance(df)
    fig, minvolport, optriskport = plotEfficientFrontier(df, cov)
    return fig, minvolport, optriskport