import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import numpy as np
import math
from scipy.stats import norm
from functools import reduce
from sklearn import preprocessing
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import pickle

pred_list = []

for i in range(1,11):
  #GLOBAL PARAMETER
  #parameters for all models!
  N_PERIOD = i #lookahead target label
  N_BINS = 3 #number of target variable bins

  df = pd.read_csv('../eth_metrics_raw.csv',index_col="Date")
  new_df = df.copy()
  #target response variable - bin of N period change in future
  df['daily_change'] = df['close'].pct_change(-N_PERIOD)
  # df['target_raw'] = pd.cut(df['daily_change'],N_BINS).astype(str)
  df['target_raw'] = pd.qcut(df['daily_change'],N_BINS).astype(str)
  df.dropna(inplace=True) #drop rows with no available lookahead prices

  le = preprocessing.LabelEncoder()
  le.fit(df['target_raw'])
  df['target'] = le.transform(df['target_raw']) #encoded bins 

  label_classes = le.classes_

  def cross_corr(target,var,shifts):
    cross_corrs = []
    for s in shifts:
      cross_corrs.append(target.corr(var.shift(s)))
    return cross_corrs

  shift_min = -N_PERIOD
  shift_max = N_PERIOD
  shifts = range(shift_min,shift_max+1) #in days

  targets = ['close']

  eligible_features = []

  for cc_target in targets:

    numeric_cols = list(df.select_dtypes(include=['float','int']).columns)
    if cc_target in numeric_cols:
      numeric_cols.remove(cc_target)

    fig, axes = plt.subplots(7,3,figsize=(20,20))
    axes = axes.flatten()
    for variable , ax in zip(numeric_cols,axes):
      corrs = cross_corr(df[cc_target],df[variable],shifts)
      sorted_corr_shift = list(map(lambda x: x - N_PERIOD , sorted(range(len(corrs)), key=lambda k: corrs[k], reverse=True)))

      #select best leading indicator
      top_n = 5
      threshold_shift = 5
      threshold_corr = 0.5
      
      eligible = reduce(lambda prev,shift: bool( shift+N_PERIOD <= threshold_shift and abs(corrs[shift+N_PERIOD]) >= threshold_corr) , sorted_corr_shift[:top_n])

      print(f"{variable} eligible? {eligible} , ","Best cross corr shifts: ",sorted_corr_shift[:top_n])
      if eligible:
        eligible_features.append(variable)

      ax.plot(shifts,corrs)
      ax.set_xlabel('Day shift')
      ax.set_ylabel('Correlation')
      ax.set_title(f'Cross Corr: {cc_target} , {variable}')

  if 'daily_change' in eligible_features:
    eligible_features.remove('daily_change')
  if 'target' in eligible_features:
    eligible_features.remove('target')
  print('el:', eligible_features)


  # model input creation
  X = df[eligible_features]

  y = df['target'].to_numpy()
  # y = np_utils.to_categorical(y,num_classes=len(label_classes))


  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  X_train = X_train.to_numpy()
  X_test = X_test.to_numpy()

  # Decision Tree baseline model
  dt = DecisionTreeClassifier(random_state=0)
  dt.fit(X_train, y_train)
  y_pred_dt = dt.predict(X_test)
  # print(classification_report(y_test,y_pred_dt))

  # Random Forest model
  rf = RandomForestClassifier(n_estimators = 1000, random_state = 42)
  rf.fit(X_train, y_train)
  # y_pred_rf = rf.predict(X_test)
  # print(classification_report(y_test,y_pred_dt))
  # predictions to record in csv
  print('TO BE',new_df.loc['2020-09-10 00:00:00+00:00'][eligible_features].to_numpy())
  # print('AS IS', df[eligible_features].tail(N_PERIOD+10))
  ui_pred = rf.predict([new_df.loc['2020-09-10 00:00:00+00:00'][eligible_features].to_numpy()])
  results = pd.DataFrame(le.inverse_transform(ui_pred))
  print(results)
  pred_list.append(results.iloc[0][0])
  print(pred_list)

  # Save model as pickle file
  # pkl_filename = "{}.pkl".format(N_PERIOD)
  # with open(pkl_filename, 'wb') as file:
  #     pickle.dump(rf, file)

  # # Load from file
  # with open(pkl_filename, 'rb') as file:
  #     pickle_model = pickle.load(file)
  print('- END -')

pred_df = pd.DataFrame({'predictions': pred_list})
print(pred_df)
# pred_df.to_csv('eth.csv')
