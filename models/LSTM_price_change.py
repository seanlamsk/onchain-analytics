import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import numpy as np
import math
from scipy.stats import norm
from functools import reduce
from sklearn import preprocessing
from numpy import asarray
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras import backend as K

import tensorflow as tf

from sklearn.metrics import classification_report


class LSTMPriceChangeModel:
    def __init__(self, file_name, N_PERIOD=1, N_BINS=7):

        #GLOBAL PARAMETERS
        self.N_PERIOD = N_PERIOD #lookahead target label
        self.N_BINS = N_BINS #number of target variable bins

        """# Dataset Setup
        *   create target variables (% change bins ahead of n_period)
        *   drop rows without available lookahead prices
        """
        self.df = pd.read_csv(file_name,index_col="Date")
        #target response variable - bin of N period change in future
        self.df['daily_change'] = self.df['close'].pct_change(-self.N_PERIOD)
        self.df['target_raw'] = pd.qcut(self.df['daily_change'],self.N_BINS).astype(str)
        self.df.dropna(inplace=True) #drop rows with no available lookahead prices
        
        self.le = preprocessing.LabelEncoder()
        self.le.fit(self.df['target_raw'])
        self.df['target'] = self.le.transform(self.df['target_raw']) #encoded bins 

        self.label_classes = self.le.classes_

        self.model = None

    def cross_corr(self,target,var,shifts):
        cross_corrs = []
        for s in shifts:
            cross_corrs.append(target.corr(var.shift(s)))
        return cross_corrs

    def feature_selection(self, top_n = 5, threshold_shift = 5, threshold_corr = 0.5):
        """# Feature Selection

        Select best features based on cross correlation.

        Features that have highest cross correlation implies higher chance of success in being a leading indicator
        """
        shift_min = -self.N_PERIOD
        shift_max = self.N_PERIOD
        shifts = range(shift_min,shift_max+1) #in days

        targets = ['close']

        self.eligible_features = []

        for cc_target in targets:

            numeric_cols = list(self.df.select_dtypes(include=['float','int']).columns)
            exclusions = ['daily_change','target']

            if cc_target in numeric_cols:
                numeric_cols.remove(cc_target)

            for col in exclusions:
                numeric_cols.remove(col)

            fig, axes = plt.subplots(7,3,figsize=(20,20))
            axes = axes.flatten()
            for variable , ax in zip(numeric_cols,axes):
                corrs = self.cross_corr(self.df[cc_target],self.df[variable],shifts)
                sorted_corr_shift = list(map(lambda x: x - self.N_PERIOD , sorted(range(len(corrs)), key=lambda k: corrs[k], reverse=True)))

                #select best leading indicator
                top_n = top_n
                threshold_shift = threshold_shift
                threshold_corr = threshold_corr
                
                eligible = reduce(lambda prev,shift: bool( shift+self.N_PERIOD <= threshold_shift and abs(sorted_corr_shift[shift+self.N_PERIOD]) >= threshold_corr) , sorted_corr_shift[:top_n])

                # print(f"{variable} eligible? {eligible} , ","Best cross corr shifts: ",sorted_corr_shift[:top_n])
                if eligible:
                    self.eligible_features.append(variable)
    
    def create_3d_features(self,X, time_steps = 1):
            Xs = []
            
            for i in range(len(X)-time_steps):
                v = X[i:i+time_steps, :]
                Xs.append(v)
                
            return np.array(Xs)

    def init(self):
        """# LSTM

        ## Dataset prep
        """

        #perform feature selection
        self.feature_selection()
        print("Eligible features",self.eligible_features)

        df2 = self.df.copy()

        X = df2[self.eligible_features]
        y = pd.get_dummies(df2['target'])

        X.tail()

        y.tail()

        #Split to train and test by halving dates
        #1st and 2nd halving - train, 3rd halving - test
        # df[:"2020-05-10"], df["2020-05-11":]
        X_train = X[:"2020-05-10"]
        X_test = X["2020-05-11":]
        y_train = y[:"2020-05-10"].to_numpy()
        y_test = y["2020-05-11":].to_numpy()

        X_train.shape, X_test.shape, y_train.shape,y_test.shape

        # Different scaler for input and output
        scaler_x = MinMaxScaler(feature_range = (-1,1))

        # Fit the scaler using available training data
        input_scaler = scaler_x.fit(X_train)

        # Apply the scaler to training data
        X_train = input_scaler.transform(X_train)

        # Apply the scaler to test data
        X_test = input_scaler.transform(X_test)


        X_train3d = self.create_3d_features(X_train,self.N_PERIOD)
        X_test3d = self.create_3d_features(X_test,self.N_PERIOD)

        #Finalise
        self.X_train = X_train3d
        self.X_test = X_test3d

        self.y_train = y_train[:len(y_train)-self.N_PERIOD]
        self.y_test = y_test[:len(y_test)-self.N_PERIOD]

        """## Model """

        self.model = self.lstm_classifier(10,(self.X_train.shape[1],self.X_train.shape[2]),self.y_train.shape[1])

    def f1_metric(self,y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        recall = true_positives / (possible_positives + K.epsilon())
        f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
        return f1_val

    def lstm_classifier(self,h,input_shape,n_classes):    
        model = Sequential()
        # First layer of LSTM
        model.add(layers.LSTM(units = h, return_sequences = True, 
                    input_shape = input_shape))
        model.add(layers.Dropout(0.2)) 
        # Second layer of LSTM
        model.add(layers.LSTM(units = h))                 
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(units = n_classes,activation='softmax')) 
        #Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy',self.f1_metric])
        return model

    def fit(self,save_dir,epochs=1000):
        """## Model Fit"""
        # #Callbacks
        # save_checkpoint = tf.keras.callbacks.ModelCheckpoint('saved_models/lstm_price_change',
        #                                  save_format='tf',
        #                                  monitor='val_accuracy',
        #                                  verbose=1,
        #                                  save_weights_only=False,
        #                                  save_best_only=True,
        #                                  mode='max',
        #                                  save_freq='epoch')

        # Stop training when val_accuracy has stopped improving for more than p epochs
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='accuracy',
                                            patience=50,
                                            verbose=1,
                                            mode='max',
                                            restore_best_weights=True)

        self.history = self.model.fit(self.X_train,self.y_train,epochs=epochs,validation_split=0.05,callbacks=[early_stop])

        tf.keras.models.save_model(self.model, save_dir, save_format="h5")


    def load_model(self,file):
    #    self.model.load_weights(file)
        self.model = tf.keras.models.load_model(file,custom_objects={"f1_metric":self.f1_metric})


    def predict(self,X,return_label=False):
        yhat = self.model.predict(X)
        pred_label = np.argmax(yhat,axis=-1)
        bin = self.le.inverse_transform(pred_label) 

        predicted =  bin if not return_label else pred_label
        return predicted


# def plot_metric(history,metric,training_only=False):
#     legend = ['training data'] if training_only else ['training data', 'testing data']
#     plt.plot(history.history[metric])
#     if not training_only:
#         plt.plot(history.history['val_' + metric])
#         plt.title('Model '+metric)
#         plt.ylabel(metric)
#         plt.xlabel('epoch')
#         plt.legend(legend, loc='upper left')
#         plt.show()

# plot_metric(history,'accuracy')
# plot_metric(history,'loss')
# plot_metric(history,'f1_metric')

# pred_y = model.predict(X_test)
# # pred_bin = np.where(pred_y<0.5,0,1)
# pred_label = np.argmax(pred_y,axis=-1)
# y_label = np.argmax(y_test,axis=-1)

# print(classification_report(y_label,pred_label))

# pd.DataFrame(y_label).value_counts().plot(kind='bar')

# pd.DataFrame(pred_label).value_counts().plot(kind='bar')

# """## Visualisation"""

# def predict(model,X,label_encoder,return_label=False):
#   yhat = model.predict(X)
#   pred_label = np.argmax(yhat,axis=-1)
#   bin = le.inverse_transform(pred_label) 
  
#   predicted =  bin if not return_label else pred_label
#   return predicted

# len(X_test),len(df2["2020-05-11":][:len(df2["2020-05-11":])-N_PERIOD])

# le.inverse_transform(list(range(1,7)))

# test_frame = df2["2020-05-11":][:len(df2["2020-05-11":])-N_PERIOD].copy()
# test_frame['predicted'] = predict(model,X_test,le,True)

# test_frame

# test_frame.index = pd.to_datetime(test_frame.index)

# plt.figure(figsize=(16, 6))
# plt.plot(test_frame.index, test_frame['predicted'], label='predicted')
# plt.plot(test_frame.index, test_frame['target'], label='actual')
# plt.legend()


