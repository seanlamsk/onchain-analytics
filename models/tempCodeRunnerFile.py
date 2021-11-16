red = ArimaModel('../btc_metrics_raw.csv', 10, '../models/saved_models/btcarimamodel.pkl', 'load').arima_pred_past_seven()
pred = pred.to_frame().reset_index().rename(columns={'ARIMA': 'Price'}) 
print(pred)