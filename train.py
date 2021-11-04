import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta

from models.LSTM_price_change import LSTMPriceChangeModel
from models.arima_model import ArimaModel
from models.LSTM_PricePrediction import LSTMModel

dateparse = lambda dates: pd.datetime.strptime(dates[:dates.find('+')], '%Y-%m-%d %H:%M:%S')
df_btc = pd.read_csv('btc_metrics_raw.csv', index_col='Date', parse_dates=True, date_parser=dateparse)
df_eth = pd.read_csv('eth_metrics_raw.csv', index_col='Date', parse_dates=True, date_parser=dateparse)
df_ltc = pd.read_csv('ltc_metrics_raw.csv', index_col='Date', parse_dates=True, date_parser=dateparse)

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# LSTM change prediction
# model = LSTMPriceChangeModel("btc_metrics_raw.csv")
# model.init()
# model.fit("models/saved_models/btc/lstm_price_change_btc.hp5")

# model2 = LSTMPriceChangeModel("ltc_metrics_raw.csv")
# model2.init()
# model2.fit("models/saved_models/ltc/lstm_price_change_ltc.hp5")

# model3 = LSTMPriceChangeModel("eth_metrics_raw.csv")
# model3.init()
# model3.fit("models/saved_models/eth/lstm_price_change_eth.hp5")


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
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


Lstm_obj1 = LSTMModel('btc_metrics_raw.csv', 'models/saved_models/btc/lstm_price_predictor.hp5')
Lstm_obj2 = LSTMModel('eth_metrics_raw.csv', 'models/saved_models/eth/lstm_price_predictor.hp5')
Lstm_obj3 = LSTMModel('ltc_metrics_raw.csv', 'models/saved_models/ltc/lstm_price_predictor.hp5')

# ARIMA
Arima_Obj1 = ArimaModel('btc_metrics_raw.csv', 10, 'models/saved_models/btc/arimamodel.pkl')
Arima_Obj2 = ArimaModel('eth_metrics_raw.csv', 10, 'models/saved_models/eth/arimamodel.pkl')
Arima_Obj3 = ArimaModel('ltc_metrics_raw.csv', 10, 'models/saved_models/ltc/arimamodel.pkl')

#Create CSV for LSTM and ARIMA
#Pred next 10days
df_btc_close_last_30 = df_btc['close'][-30:].reset_index()
df_eth_close_last_30 = df_eth['close'][-30:].reset_index()
df_ltc_close_last_30 = df_ltc['close'][-30:].reset_index()

Lstm_pred_btc = Lstm_obj1.forecast().reset_index().rename(columns={0: 'LSTM_btc', 'index': 'Date'})
Lstm_pred_eth = Lstm_obj2.forecast().reset_index().rename(columns={0: 'LSTM_eth', 'index': 'Date'})
Lstm_pred_ltc = Lstm_obj3.forecast().reset_index().rename(columns={0: 'LSTM_ltc', 'index': 'Date'})

Arima_pred_btc = Arima_Obj1.arima_pred_future().reset_index().rename(columns={'index': 'Date', 'ARIMA': 'ARIMA_btc'})
Arima_pred_eth = Arima_Obj2.arima_pred_future().reset_index().rename(columns={'index': 'Date', 'ARIMA': 'ARIMA_eth'})
Arima_pred_ltc = Arima_Obj3.arima_pred_future().reset_index().rename(columns={'index': 'Date', 'ARIMA': 'ARIMA_ltc'})

#Add last data to df
add_into_df(Lstm_pred_btc, df_btc_close_last_30.iloc[-1, :].to_list())
add_into_df(Lstm_pred_eth, df_eth_close_last_30.iloc[-1, :].to_list())
add_into_df(Lstm_pred_ltc, df_ltc_close_last_30.iloc[-1, :].to_list())

add_into_df(Arima_pred_btc, df_btc_close_last_30.iloc[-1, :].to_list())
add_into_df(Arima_pred_eth, df_eth_close_last_30.iloc[-1, :].to_list())
add_into_df(Arima_pred_ltc, df_ltc_close_last_30.iloc[-1, :].to_list())

#Annualised volatility for whole period
btc_HV = annualised_HV(df_btc)
eth_HV = annualised_HV(df_eth)
ltc_HV = annualised_HV(df_ltc)

btc_returns = annualised_return(df_btc)
eth_returns = annualised_return(df_eth)
ltc_returns = annualised_return(df_ltc)

pred_df = Lstm_pred_btc.merge(Lstm_pred_eth, on='Date').merge(Lstm_pred_ltc, on='Date').merge(Arima_pred_btc, on='Date').merge(Arima_pred_eth, on='Date').merge(Arima_pred_ltc, on='Date').round(2)

actual_df = df_btc_close_last_30.merge(df_eth_close_last_30, on='Date').merge(df_ltc_close_last_30, on='Date').rename(columns={'close_x': 'close_btc', 'close_y': 'close_eth', 'close': 'close_ltc'}).round(2)

data = {'Volatilty': [btc_HV, eth_HV, ltc_HV], "Annualised Returns": [btc_returns, eth_returns, ltc_returns]}
stats_df = pd.DataFrame(data).round(2)
stats_df.index = ['btc', 'eth', 'ltc']

#Write CSV
pred_df.to_csv('models/predictions/price_pred_prediction.csv')
actual_df.to_csv('models/predictions/price_pred_actual.csv')
stats_df.to_csv('models/predictions/price_pred_stats.csv')

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------












