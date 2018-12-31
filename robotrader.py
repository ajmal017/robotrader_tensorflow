#%%
import datetime

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
from util_functions import *

days_predict = 1
test_size = 0.4

etfToSymbol = {

    'momentum': 'MTUM',
    'quality': 'QUAL',
    'value': 'VLUE',
}
symbols = list(etfToSymbol.values())
#%% download data
dataDict = excel_to_dataDict('historical_data_robotrader.xlsx')
# dataDict=import_data_from_fw(symbols)
# %% plot returns
returnsAll = get_returns(dataDict, days_predict, plot=True)

# %% get Input/Target
targetMatrix = get_target(dataDict=dataDict, symbols=list(etfToSymbol.values()), days_predict=days_predict)
inputMatrix = get_input(dataDict=dataDict)
# Clean nan data
inputMatrixClean = inputMatrix.dropna(axis=1)
inputMatrixClean.fillna(0,inplace=True)
targetMatrix = targetMatrix.T[inputMatrixClean.index].T

#%% split to train
x_train, x_test = train_test_split(inputMatrixClean.fillna(0), test_size=test_size, shuffle=False)
y_train, y_test = train_test_split(targetMatrix.fillna(0), test_size=test_size, shuffle=False)

print('input have %d features' % (len(inputMatrixClean.columns)))
print('train  have %d samples \ntest %d samples' % (len(x_train), len(x_test)))

'''
#%% SVC simple => predict class

from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
imputer = preprocessing.Imputer()  # drop columns with nans
scaler = preprocessing.StandardScaler()
# pca = PCA(kernel="rbf", n_components=5, n_jobs=4)
classifier = SVC(gamma=2, C=1)

pipelineObject = Pipeline([('imputer', imputer),
                           ('scaler', scaler),
                           #('pca', pca),
                           ('classifier', classifier)])


pipelineObject.fit(x_train.dropna(),y_train)
train_accuracy= hamming_score(y_train, pipelineObject.predict(x_train))
test_accuracy= hamming_score(y_test, pipelineObject.predict(x_test))
#% train hamming_score =0.887   test hamming_score =0.501
print('train hamming_score =%.3f   test hamming_score =%.3f'%(train_accuracy,test_accuracy))
#%% decission tree
from sklearn.tree import DecisionTreeClassifier
imputer = preprocessing.Imputer()  # drop columns with nans
scaler = preprocessing.StandardScaler()
# pca = PCA(kernel="rbf", n_components=5, n_jobs=4)
classifier = DecisionTreeClassifier()

pipelineObject = Pipeline([('imputer', imputer),
                           ('scaler', scaler),
                           #('pca', pca),
                           ('classifier', classifier)])


pipelineObject.fit(x_train.dropna(),y_train)
train_accuracy= hamming_score(y_train, pipelineObject.predict(x_train))
test_accuracy= hamming_score(y_test, pipelineObject.predict(x_test))
# train hamming_score =1.000   test hamming_score =0.529
print('train hamming_score =%.3f   test hamming_score =%.3f'%(train_accuracy,test_accuracy))
#%% xgboost classiffier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
imputer = preprocessing.Imputer()  # drop columns with nans
scaler = preprocessing.StandardScaler()
# pca = PCA(kernel="rbf", n_components=5, n_jobs=4)
classifier = XGBClassifier()

pipelineObject = Pipeline([('imputer', imputer),
                           ('scaler', scaler),
                           #('pca', pca),
                           ('classifier', classifier)])


pipelineObject.fit(x_train,y_train)
train_accuracy= hamming_score(y_train, pipelineObject.predict(x_train))
test_accuracy= hamming_score(y_test, pipelineObject.predict(x_test))
# train hamming_score =0.830   test hamming_score =0.483
print('train hamming_score =%.3f   test hamming_score =%.3f' % (train_accuracy,test_accuracy))
# %% probabilistic = weight => backtest
output_train = pipelineObject.predict_proba(x_train)
output_test = pipelineObject.predict_proba(x_test)
# %% get backtest
returns_train = returnsAll.T[x_train.index].T[symbols]
returns_test = returnsAll.T[x_test.index].T[symbols]
backtest_train_returns = returns_train * output_train
backtest_test_returns = returns_test * output_test
# %% plot it

backtest_total = backtest_train_returns.append(backtest_test_returns).dropna()
test_date = returns_test.index[0]
plt.close()
pnl_total = backtest_total.sum(axis=1).cumsum()
pnl_total.plot()
pnl_total[test_date:].plot()
plt.legend(['train_sample', 'test_sample'])
plt.show()

# %% benchmark it with buy and hold
plt.close()
pnl_total.plot()
pnl_total[test_date:].plot()

benchmark_total = returnsAll / returnsAll.shape[1]
pnl_benchmark = benchmark_total.sum(axis=1).cumsum()
pnl_benchmark.plot()
plt.legend(['train_sample', 'test_sample', 'buy & hold'])
plt.show()

# %% add commission to backtest to see lose profitability
# The average ETF carries an expense ratio of 0.44%
commission_average_etf = 0.44 / 100
basic_fee = 2
capital = 1000000  # play with capital to get min balance

# http://guides.wsj.com/personal-finance/investing/how-to-choose-an-exchange-traded-fund-etf/

backtest_train_returns_commission = (backtest_train_returns) * capital - basic_fee - (
    backtest_train_returns) * capital * commission_average_etf
backtest_test_returns_commission = (backtest_test_returns) * capital - basic_fee - (
    backtest_test_returns) * capital * commission_average_etf

backtest_total_commission = backtest_train_returns_commission.append(backtest_test_returns_commission).dropna()
plt.close()
pnl_total_commission = backtest_total_commission.sum(axis=1).cumsum()
pnl_total_commission.plot()
pnl_total_commission[test_date:].plot()
plt.close()
pnl_total_commission.plot()
pnl_total_commission[test_date:].plot()

benchmark_total = (returnsAll / returnsAll.shape[1]) * capital
pnl_benchmark = benchmark_total.sum(axis=1).cumsum()
pnl_benchmark.plot()
plt.legend(['train_sample', 'test_sample', 'buy & hold'])
plt.show()

'''
#%% NN basic classiffier skflow
# TensorFlow and tf.keras

# one hot encoding output
binarizer = LabelBinarizer()
y_train_binarized = binarizer.fit_transform(y_train.values)
y_test_binarized = binarizer.fit_transform(y_test.values)
normalizer = MinMaxScaler()
x_train_normalized = normalizer.fit_transform(x_train)
x_test_normalized = normalizer.fit_transform(x_test)
# %%
from tensorflow import keras

features = x_train.shape[1]
classes = len(symbols)


model = keras.Sequential([
    keras.layers.Dense(x_train.shape[1]),
    keras.layers.Dense(int(x_train.shape[1] * 2), activation=tf.nn.sigmoid),
    keras.layers.Dense(int(x_train.shape[1] * 2), activation=tf.nn.sigmoid),
    keras.layers.Dense(3, activation=tf.nn.sigmoid)
])

#%%
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# %%
model.fit(x_train_normalized, y_train_binarized, epochs=100)

# %%
predictions_train = pd.DataFrame(model.predict(x_train.values), index=x_train.index, columns=symbols)

predictions_test = pd.DataFrame(model.predict(x_test.values), index=x_test.index, columns=symbols)
predictions_train_normalized = (predictions_train).div(predictions_train.sum(axis=1), axis=0)
predictions_test_normalized = (predictions_test).div(predictions_test.sum(axis=1), axis=0)

# %%
# %% get backtest
returns_train = returnsAll.T[x_train.index].T[symbols]
returns_test = returnsAll.T[x_test.index].T[symbols]
backtest_train_returns = returns_train * predictions_train_normalized
backtest_test_returns = returns_test * predictions_test_normalized
# %% plot it

backtest_total = backtest_train_returns.append(backtest_test_returns).dropna()
test_date = returns_test.index[0]
plt.close()
pnl_total = backtest_total.sum(axis=1).cumsum()
pnl_total.plot()
pnl_total[test_date:].plot()

benchmark_total = returnsAll / returnsAll.shape[1]
pnl_benchmark = benchmark_total.sum(axis=1).cumsum()
pnl_benchmark.plot()
plt.legend(['train_sample', 'test_sample', 'buy & hold'])
plt.show()
