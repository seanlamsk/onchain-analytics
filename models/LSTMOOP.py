import numpy as np
import pandas as pd
import seaborn as sns
import math
from scipy.stats import norm
from functools import reduce
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout , GRU
import datetime as dt

class LSTMModel:

  def __init__(self, dir, model_name, mode = 'new', N_PERIOD = 1, timestep = 10, number_of_days_pred = 20):
    self.targets = ['close']
    self.third_halving = '2020-05-11'
    if mode == 'new':
      self.df = self.create_df(dir)
      self.n = number_of_days_pred
      self.N_PERIOD = N_PERIOD
      self.timestep = timestep
      self.top_features = self.selectFeatures()
      self.X_test, self.y_test, self.X_train, self.y_train = self.train_test_prep()
      self.model = self.create_model(64, LSTM)
      self.fit_model = self.fit_model(model_name)
    else:
      self.df = self.create_df(dir)
      self.n = number_of_days_pred
      self.N_PERIOD = N_PERIOD
      self.timestep = timestep
      self.top_features = self.selectFeatures()
      self.X_test, self.y_test, self.X_train, self.y_train = self.train_test_prep()
      self.model = self.load_model(model_name)
      
  def create_df(self, dir):
    df=pd.read_csv(dir)
    start_date = pd.to_datetime('2012-11-28 00:00:00+00:00')
    first_date = pd.to_datetime(str(df['Date'][0]))
    if start_date < first_date:
      start_date = first_date
    start_date = start_date.strftime('%Y-%m-%d')
    df = df.loc[start_date:]
    self.start_date = start_date
    return df

  def selectFeatures(self):
    df = self.df
    shift_min = -self.N_PERIOD
    shift_max = self.N_PERIOD
    shifts = range(shift_min,shift_max+1)
    eligible_features = []

    for cc_target in self.targets:
      numeric_cols = list(df.select_dtypes(include=['float','int']).columns)
      if cc_target in numeric_cols:
        numeric_cols.remove(cc_target)
      for variable in numeric_cols:
        corrs = self.cross_corr(df[cc_target], df[variable], shifts)
        sorted_corr_shift = list(map(lambda x: x - self.N_PERIOD , sorted(range(len(corrs)), key=lambda k: corrs[k], reverse=True)))
        #select best leading indicator
        top_n = 4
        threshold_shift = 5
        threshold_corr = 0.5       
        eligible = reduce(lambda prev,shift: bool( shift+self.N_PERIOD <= threshold_shift and abs(sorted_corr_shift[shift+self.N_PERIOD]) >= threshold_corr) , sorted_corr_shift[:top_n])
        if eligible:
          eligible_features.append(variable)
    eligible_features.append('close')
    eligible_features.append('Date')
    return df[eligible_features]

  # Cross correlation
  def cross_corr(self,target,var,shifts):
    cross_corrs = []
    for s in shifts:
      cross_corrs.append(target.corr(var.shift(s)))
    return cross_corrs

  def train_test_prep (self):

    self.top_features['Date'] = self.top_features['Date'].astype('datetime64[ns]')
    self.top_features = self.top_features.set_index('Date')
    self.date_index = self.top_features.index

    self.top_features['TOMORROW_CLOSE'] = self.top_features['close'].shift(-self.N_PERIOD,fill_value=0)
    self.top_features.drop(self.top_features.tail(self.N_PERIOD).index,inplace=True) 

    #Determine train and test dataset
    train_size = int(len(self.top_features.loc[:self.third_halving]))
    train_dataset, test_dataset,  = self.top_features.iloc[:train_size],self.top_features.iloc[train_size:] 

    # Split train data to X and y
    X_train = train_dataset.drop('TOMORROW_CLOSE', axis = 1)
    y_train = train_dataset.loc[:,['TOMORROW_CLOSE']]

    # Split test data to X and y
    X_test = test_dataset.drop('TOMORROW_CLOSE', axis = 1)
    y_test = test_dataset.loc[:,['TOMORROW_CLOSE']]

    # Different scaler for input and output
    scaler_x = MinMaxScaler(feature_range = (-1,1))
    scaler_y = MinMaxScaler(feature_range = (-1,1))

    # Fit the scaler using available training data
    input_scaler = scaler_x.fit(X_train)
    output_scaler = scaler_y.fit(y_train)
    self.output_scaler = output_scaler

    # Apply the scaler to training data
    train_y_norm = output_scaler.transform(y_train)
    train_x_norm = input_scaler.transform(X_train)

    # Apply the scaler to test data
    test_y_norm = output_scaler.transform(y_test)
    test_x_norm = input_scaler.transform(X_test)

    X_test, y_test = self.threeD_dataset(test_x_norm, test_y_norm, self.timestep)
    X_train, y_train = self.threeD_dataset(train_x_norm, train_y_norm, self.timestep)

    return X_test, y_test, X_train, y_train

  #change to 3d datasets
  def threeD_dataset (self, X, y, time_steps = 1):
      Xs, ys = [], []
      
      for i in range(len(X)-time_steps):
          v = X[i:i+time_steps, :]
          Xs.append(v)
          ys.append(y[i+time_steps])
          
      return np.array(Xs), np.array(ys)

  # Create LSTM 

  def create_model(self, units, m):
      model = Sequential()
      # First layer of LSTM
      model.add(m (units = units, return_sequences = True, 
                  input_shape = [self.X_train.shape[1], self.X_train.shape[2]]))
      model.add(Dropout(0.2)) 
      # Second layer of LSTM
      model.add(m (units = units))                 
      model.add(Dropout(0.2))
      model.add(Dense(units = 1)) 
      #Compile model
      model.compile(loss='mse', optimizer='adam')
      return model

  #Fit model
  def fit_model(self, model_name):
      #save_checkpoint = tf.keras.callbacks.ModelCheckpoint('saved_models/lstm_price_predictor',
      #                              save_format='tf',
        #                              monitor='val_accuracy',
        #                            verbose=1,
          #                          save_weights_only=False,
            #                       save_best_only=True,
          #                       mode='max',
            #                    save_freq='epoch')
      early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss',
                                                patience = 11)

      # Set shuffle equal to False due to importance of oder for this dataset
      history = self.model.fit(self.X_train, self.y_train, epochs = 100, validation_split = 0.2,
                      batch_size = 32, shuffle = False, callbacks = [early_stop])
      tf.keras.models.save_model(self.model, model_name, save_format="h5")
      return history

  # def load_model(self,file):
  #     model = tf.keras.models.load_model("saved_model/lstm_price_predictor.hp5")

  def plot_history_future(self, prediction):
      
      plt.figure(figsize=(10, 6))
      
      y_train = self.output_scaler.inverse_transform(self.y_train)
      range_history = len(y_train)
      range_future = list(range(range_history, range_history + len(prediction)))

      plt.plot(np.arange(range_history), np.array(y_train), label='History')
      plt.plot(range_future, np.array(prediction),label='Prediction')

      plt.title('History and prediction for LSTM')
      plt.legend(loc='upper left')
      plt.xlabel('Date')
      plt.ylabel('Close value (US $)')

  def forecast(self):
      # Scale the forecast input with the scaler fit on the training data
      X_new = self.top_features[-self.n:]
      Y_new = X_new.loc[:,['TOMORROW_CLOSE']]
      X_new = X_new.drop('TOMORROW_CLOSE', axis = 1 )


      scaler_x = MinMaxScaler(feature_range = (-1,1))
      scaler_y = MinMaxScaler(feature_range = (-1,1))
      input_scaler = scaler_x.fit(X_new)
      output_scaler = scaler_y.fit(Y_new)
      X = input_scaler.transform(X_new)
      # # Reshape forecast data as 3D input
      Xs = []
      for i in range(len(X) - self.timestep):
          v = X[i:i+self.timestep, :]
          Xs.append(v)
          
      X_transformed = np.array(Xs)

      # Make prediction for forecast data using LSTM model 
      prediction = self.model.predict(X_transformed)
      prediction_actual = output_scaler.inverse_transform(prediction)
      return_pred = pd.DataFrame(prediction_actual)
      start_date = self.top_features.index[-1].to_pydatetime('%Y-%m-%d') + dt.timedelta(days=2)
      end_date = start_date + dt.timedelta(days=len(X) - self.timestep -1)
      index_future_dates = pd.date_range(start = start_date, end = end_date)
      return_pred.index = index_future_dates
      return return_pred

  def past_seven_days_forecast(self):
      n = 7
      X_new = self.top_features[-(2*n+self.timestep):-n]
      Y_new = X_new.loc[:,['TOMORROW_CLOSE']]
      X_new = X_new.drop('TOMORROW_CLOSE', axis = 1 )


      scaler_x = MinMaxScaler(feature_range = (-1,1))
      scaler_y = MinMaxScaler(feature_range = (-1,1))
      input_scaler = scaler_x.fit(X_new)
      output_scaler = scaler_y.fit(Y_new)
      X = input_scaler.transform(X_new)
      # # Reshape forecast data as 3D input
      Xs = []
      for i in range(len(X) - self.timestep):
          v = X[i:i+self.timestep, :]
          Xs.append(v)
          
      X_transformed = np.array(Xs)

      # Make prediction for forecast data using LSTM model 
      prediction = self.model.predict(X_transformed)
      # print(prediction)
      prediction_actual = output_scaler.inverse_transform(prediction)
      return_pred = pd.DataFrame(prediction_actual)
      return_pred.index = self.date_index[-n:]
      return return_pred

  def load_model(self, model_name):
    model = tf.keras.models.load_model(model_name)
    return model
        

# Lstm_obj = LSTMModel('../btc_metrics_raw.csv', '../pretrained_models/lstm_price_predictor.hp5')

# LSTM_obj1 = LSTMModel('../btc_metrics_raw.csv','../pretrained_models/lstm_price_predictor.hp5', 'load')
# prediction = LSTM_obj1.forecast()
# print(prediction)

# In[4]:


#Lstm_obj.past_seven_days_forecast()


# In[5]:


#prediction = Lstm_obj.forecast()
## Lstm_obj.plot_history_future(prediction)
#prediction


# In[6]:


#Lstm_obj.top_features


# In[8]:


#model = tf.keras.models.load_model("saved_model/lstm_price_predictor.hp5")
#LSTM_obj2 = LSTMModel('btc_metrics_raw.csv', 1, 10, 20, model, 'load')


# In[9]:


#prediction = LSTM_obj2.forecast()
## Lstm_obj.plot_history_future(prediction)
#prediction





