import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta

from models.LSTM_price_change import LSTMPriceChangeModel
from models.arima_model import ArimaModel
from models.LSTM_PricePrediction import LSTMModel

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
coins = ['btc','eth','ltc']

# LSTM change prediction
# for coin in coins:
#     model = LSTMPriceChangeModel(f"{coin}_metrics_raw.csv",N_PERIOD=10)
#     model.init()
#     model.fit(f"models/saved_models/{coin}/lstm_price_change_{coin}.hp5")
#     predictions = model.predict(model.X_test,return_label=False)
#     pd.DataFrame(predictions).to_csv(f"models/predictions/lstm_price_change_pred_{coin}.csv", header=None)

#     del model


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
dateparse = lambda dates: pd.datetime.strptime(dates[:dates.find('+')], '%Y-%m-%d %H:%M:%S')
df_btc = pd.read_csv('btc_metrics_raw.csv', index_col='Date', parse_dates=True, date_parser=dateparse)
df_eth = pd.read_csv('eth_metrics_raw.csv', index_col='Date', parse_dates=True, date_parser=dateparse)
df_ltc = pd.read_csv('ltc_metrics_raw.csv', index_col='Date', parse_dates=True, date_parser=dateparse)

# LSTM Price Change
def add_into_df (df, input):
    df.loc[-1] = input  # adding a row
    df.index = df.index + 1  # shifting index
    df.sort_index(inplace=True)

def annualised_HV (df):
    return np.sqrt(np.log(df['close'] / df['close'].shift(1)).var()) * np.sqrt(252)

def annualised_return (df):
    n = relativedelta(df.index[-1], df.index[0]).years
    return ((1 + df['close'][-1])**(1/n) - 1) * 100

#Annualised volatility for whole period
btc_HV = annualised_HV(df_btc)
eth_HV = annualised_HV(df_eth)
ltc_HV = annualised_HV(df_ltc)

btc_returns = annualised_return(df_btc)
eth_returns = annualised_return(df_eth)
ltc_returns = annualised_return(df_ltc)

data = {'Volatility': [btc_HV, eth_HV, ltc_HV], "Annualised Returns": [btc_returns, eth_returns, ltc_returns]}
stats_df = pd.DataFrame(data).round(2)
stats_df.index = ['btc', 'eth', 'ltc']

for coin in coins:
    Lstm_obj = LSTMModel(f'{coin}_metrics_raw.csv', f'models/saved_models/{coin}/lstm_price_predictor.hp5')#, 'load')
    Arima_Obj = ArimaModel(f'{coin}_metrics_raw.csv', 10, f'models/saved_models/{coin}/arimamodel.pkl')#, 'load')

    dateparse = lambda dates: pd.datetime.strptime(dates[:dates.find('+')], '%Y-%m-%d %H:%M:%S')
    df = pd.read_csv(f'{coin}_metrics_raw.csv', index_col='Date', parse_dates=True, date_parser=dateparse)

    df_close_last_30 = df['close'][-30:].reset_index()

    Lstm_pred = Lstm_obj.forecast().reset_index().rename(columns={0: 'LSTM', 'index': 'Date'})

    Arima_pred = Arima_Obj.arima_pred_future().reset_index().rename(columns={'index': 'Date'})

    add_into_df(Lstm_pred, df_close_last_30.iloc[-1, :].to_list())

    add_into_df(Arima_pred, df_close_last_30.iloc[-1, :].to_list())

    pred_df=Arima_pred.merge(Lstm_pred, on='Date').round(2)
    pred_df.to_csv(f'models/predictions/{coin}_price_pred_prediction.csv')

    actual_df = df_close_last_30.round(2)
    actual_df.to_csv(f'models/predictions/{coin}_price_pred_actual.csv')
    
#Write CSV

stats_df.to_csv('models/predictions/price_pred_stats.csv')

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------






