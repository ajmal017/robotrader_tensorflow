#%%
import datetime

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



days_predict = 1
etfToSymbol = {

    'momentum': 'MTUM',
    'quality': 'QUAL',
    'value': 'VLUE',
}
#%% download data
'''
from tradeasystems_connector.util.instrument_util import getInstrumentList
from tradeasystems_connector.util.persist_util import dataDict_to_excel

import user_settings
from tradeasystems_connector.manager_trader import ManagerTrader
from tradeasystems_connector.model.asset_type import AssetType
from tradeasystems_connector.model.currency import Currency
from tradeasystems_connector.model.instrument import *
from tradeasystems_connector.model.period import Period


manager = ManagerTrader(user_settings=user_settings)

fromDate = datetime.datetime(year=2013, day=18, month=7)
toDate = datetime.date.today()  # datetime.datetime(year=2018, day=20, month=11)

instrumentList = getInstrumentList(symbolList=list(etfToSymbol.values()), currency=Currency.usd,
                                   asset_type=AssetType.etf)
instrumentList.append(vix)
instrumentList.append(sp500_etf)
instrumentList.append(eur_usd)
instrumentList.append(t_bond)

dataDict = manager.getDataDictOfMatrix(instrumentList=instrumentList, ratioList=[], fromDate=fromDate,
                                            toDate=toDate)
#%%
for key in dataDict.keys():
    dataDict[key] = dataDict[key].dropna(axis=0)


dataDict_to_excel(dataDict, 'historical_data_robotrader.xlsx')

# %% cached
'''
def excel_to_dataDict(filePath):
    reader = pd.ExcelFile(filePath)
    sheets = reader.sheet_names
    outputDict = {}
    for sheet in sheets:
        outputDict[sheet] = reader.parse(sheet)
    return outputDict


dataDict = excel_to_dataDict('historical_data_robotrader.xlsx')
#%%
closeMatrix =  dataDict['close'].copy()
returnsAll = closeMatrix.divide(closeMatrix.shift(days_predict)) - 1
returnsAll.cumsum().plot()
plt.title('cumsum Returns of data period= %d days'%days_predict)
plt.show()
targetMatrix = returnsAll.shift(-days_predict)#to get the target return


#%% target Matrix
target_returns = targetMatrix[list(etfToSymbol.values())]
columns_list =list(target_returns.columns)
new_columns = [position for position in range(len(columns_list))]
target_returns_position = pd.DataFrame(target_returns.values,columns=new_columns,index = target_returns.index)
targetMatrix =target_returns_position.idxmax(axis=1)
taget_labels = target_returns.idxmax(axis=1)
#%% input
inputMatrix  = dataDict['close'].copy()
#%

#% add other columns
for otherMatrix in dataDict.keys():
    if otherMatrix == 'close':
        continue
    for column in dataDict[otherMatrix].columns:
        name = '%s_%s'%(otherMatrix,column)
        inputMatrix[name] = dataDict[otherMatrix][column]
input_columns = list(inputMatrix.columns)
#%
#add delay and returns past
delay_range = range(1,5)
for column in input_columns:
    for delay in delay_range:
        # inputMatrix['%s_%d'%(column,delay)]= inputMatrix[column].shift(delay)
        inputMatrix['returns_%s_%d' % (column, delay)] = inputMatrix[column].divide(inputMatrix[column].shift(delay)) - 1

#% add moving average
title = 'sma'
period_range = [20,40,60,200]
for column in input_columns:
    for period in period_range:
        inputMatrix['%s%s_%d'%(title,column,period)]= (inputMatrix[column].rolling(period).mean())
## volatility
title = 'std'
period_range = [20,40,60,200]
for column in input_columns:
    for period in period_range:
        inputMatrix['%s%s_%d'%(title,column,period)]= (inputMatrix[column].rolling(period).std())



#%% split to train
test_size = 0.4
#%%
inputMatrixClean = inputMatrix.dropna(axis=1)
inputMatrixClean.fillna(0,inplace=True)
targetMatrix = targetMatrix.T[inputMatrixClean.index].T
#%%
x_train, x_test = train_test_split(inputMatrixClean.fillna(0), test_size=test_size)
y_train, y_test = train_test_split(targetMatrix.fillna(0), test_size=test_size)

print('input have %d features' % (len(inputMatrixClean.columns)))
print('train  have %d samples \ntest %d samples' % (len(x_train), len(x_test)))
#%%

def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    '''
    Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
    https://stackoverflow.com/q/32239577/395857
    '''
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        # print('\nset_true: {0}'.format(set_true))
        # print('set_pred: {0}'.format(set_pred))
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)) )
        # print('tmp_a: {0}'.format(tmp_a))
        acc_list.append(tmp_a)
    return np.mean(acc_list)


#%% SVC simple
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
print('train hamming_score =%.3f   test hamming_score =%.3f'%(train_accuracy,test_accuracy))
#%% NN basic classiffier skflow
# TensorFlow and tf.keras
import tensorflow as tf


model = tf.estimator.DNNClassifier(
    feature_columns=list(x_train.columns),
    # Two hidden layers of 10 nodes each.
    hidden_units=[int(len(x_train.columns)/2), int(len(x_train.columns)/2)],
    # The model must choose between 3 classes.
    n_classes=(int(y_train.max().max()+1)))

#%%

train_accuracy= hamming_score(y_train, pipelineObject.predict(x_train))
test_accuracy= hamming_score(y_test, pipelineObject.predict(x_test))

print('train hamming_score =%.3f   test hamming_score =%.3f'%(train_accuracy,test_accuracy))
