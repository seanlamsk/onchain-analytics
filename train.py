from models.LSTM_price_change import LSTMPriceChangeModel
import pandas as pd

coins = ['btc','eth','ltc']

# LSTM change prediction
for coin in coins:
    model = LSTMPriceChangeModel(f"{coin}_metrics_raw.csv",N_PERIOD=10)
    model.init()
    model.fit(f"models/saved_models/{coin}/lstm_price_change_{coin}.hp5")
    predictions = model.predict(model.X_test,return_label=False)
    pd.DataFrame(predictions).to_csv(f"models/predictions/lstm_price_change_pred_{coin}.csv", header=None)

    del model
