from models.LSTM_price_change import LSTMPriceChangeModel

# LSTM change prediction
model = LSTMPriceChangeModel("btc_metrics_raw.csv")
model.init()
model.fit("models/saved_models/btc/lstm_price_change_btc.hp5")

model2 = LSTMPriceChangeModel("ltc_metrics_raw.csv")
model2.init()
model2.fit("models/saved_models/ltc/lstm_price_change_ltc.hp5")

model3 = LSTMPriceChangeModel("eth_metrics_raw.csv")
model3.init()
model3.fit("models/saved_models/eth/lstm_price_change_eth.hp5")