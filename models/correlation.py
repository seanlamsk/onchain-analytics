import pandas as pd

btc = pd.read_csv('../btc_metrics_raw.csv')
ltc = pd.read_csv('../ltc_metrics_raw.csv')
eth = pd.read_csv('../eth_metrics_raw.csv')

result = pd.concat([btc['close'], ltc['close'], eth['close']], axis=1, join="inner")
result.columns = ['btc', 'ltc', 'eth']

correlation = result.corr()
# correlation.to_csv('predictions/correlation.csv')