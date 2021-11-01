import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15,7)
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
import warnings
warnings.filterwarnings('ignore')
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults

class ArimaModel:
  def __init__(self, dir, N_Pred_Period, model_name, mode='new'):
    if mode == 'new':
      self.df = self.create_df(dir)['close']
      self.model = self.arima_modeling(model_name)
      self.n = N_Pred_Period
    else:
      self.df = self.create_df(dir)['close']
      self.model = self.load_model(model_name)
      self.n = N_Pred_Period
    
  def ad_test(self):
    df = adfuller(self.df, autolag = 'AIC')
    print('ADF: ', df[0])
    print('P-Value: ', df[1])
    print('Num of Lags: ', df[2])
    print('Num of Observations used for ADF Regression and Critical Values Calculation: ', df[3])
    print('Critical Values: ')
    for key, val in df[4].items():
      print('\t', key, ': ', val)

  def create_df(self, dir):
    dateparse = lambda dates: pd.datetime.strptime(dates[:dates.find('+')], '%Y-%m-%d %H:%M:%S')
    df=pd.read_csv(dir, index_col='Date', parse_dates=True, date_parser=dateparse)
    df.dropna(inplace=True)
    start_date = dt.datetime(2012, 11, 28)
    first_date = df.index[0].to_pydatetime('%Y-%m-%d')
    if start_date < first_date:
      start_date = first_date
    df = df.loc[start_date:]
    return df
  
  def auto_arima_test(self):
    stepwise_fit =auto_arima(self.df, trace=True, suppress_warnings=True)
    get_parametes = stepwise_fit.get_params()
    best_order=get_parametes['order']
    return best_order
    
  def arima_modeling(self, model_name):
    configuration = self.auto_arima_test()
    model = ARIMA(self.df, order = configuration)
    model_fit = model.fit()
    model_fit.save(model_name)
    return model_fit
    
  def arima_pred(self):
    start = self.df.index[-1].to_pydatetime('%Y-%m-%d') + dt.timedelta(days=1)
    end = start + dt.timedelta(days=self.n)
    index_future_dates = pd.date_range(start = start, end = end)
    pred = self.model.predict(start = 0, end=len(self.df)+self.n, typ = 'levels').rename('ARIMA Next ' + str(self.n) + ' days Predictions')
    new_index = np.concatenate((self.df.index, index_future_dates), axis=None)
    pred.index = new_index
    return pred

  def load_model(self, model_name):
    model = ARIMAResults.load(model_name)
    return model

# Arima_Obj = ArimaModel('../btc_metrics_raw.csv', 30, '../pretrained_models/arimamodeltest.pkl')
# pred = ArimaModel('../btc_metrics_raw.csv', 30, '../pretrained_models/arimamodeltest.pkl', 'load').arima_pred()
# print(pred)
# pred.plot(figsize=(12,5), legend=True)