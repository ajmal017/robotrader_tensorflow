#%%
import datetime

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
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
x_train, x_test = train_test_split(inputMatrixClean.fillna(0), test_size=test_size)
y_train, y_test = train_test_split(targetMatrix.fillna(0), test_size=test_size)

print('input have %d features' % (len(inputMatrixClean.columns)))
print('train  have %d samples \ntest %d samples' % (len(x_train), len(x_test)))
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
#get backtest





#%% NN basic classiffier skflow
# TensorFlow and tf.keras

# one hot encoding output
binarizer = LabelBinarizer()
y_train_binarized = binarizer.fit_transform(y_train.values)
y_test_binarized = binarizer.fit_transform(y_test.values)
# %%
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(x_train.shape[1]),
    keras.layers.Dense(x_train.shape[1] / 2, activation=tf.nn.relu),
    keras.layers.Dense(3, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# %%
model.fit(x_train.values, y_train_binarized, epochs=100)

# %%
predictions_train = model.predict(x_train.values)
predictions_test = model.predict(x_test.values)
#%%

train_accuracy= hamming_score(y_train, pipelineObject.predict(x_train))
test_accuracy= hamming_score(y_test, pipelineObject.predict(x_test))

print('train hamming_score =%.3f   test hamming_score =%.3f'%(train_accuracy,test_accuracy))
