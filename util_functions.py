import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def excel_to_dataDict(filePath):
    reader = pd.ExcelFile(filePath)
    sheets = reader.sheet_names
    outputDict = {}
    for sheet in sheets:
        outputDict[sheet] = reader.parse(sheet)
    return outputDict


def calculateSharpe(returns_series):
    num = returns_series.mean() * np.sqrt(252)
    den = returns_series.std()
    sharpe = num / den
    return sharpe

def import_data_from_fw(symbols, appended_instruments=None, save_file=False):
    import datetime
    from tradeasystems_connector.util.instrument_util import getInstrumentList
    from tradeasystems_connector.util.persist_util import dataDict_to_excel

    import user_settings
    from tradeasystems_connector.manager_trader import ManagerTrader
    from tradeasystems_connector.model.asset_type import AssetType
    from tradeasystems_connector.model.currency import Currency

    from tradeasystems_connector.model.period import Period

    manager = ManagerTrader(user_settings=user_settings)

    fromDate = datetime.datetime(year=2013, day=18, month=7)
    toDate = datetime.date.today()  # datetime.datetime(year=2018, day=20, month=11)

    instrumentList = getInstrumentList(symbolList=symbols, currency=Currency.usd,
                                       asset_type=AssetType.etf)
    # appended instruments
    if appended_instruments is None:
        from tradeasystems_connector.model.instrument import vix, sp500_etf, eur_usd, t_bond
        instrumentList.append(vix)
        instrumentList.append(sp500_etf)
        instrumentList.append(eur_usd)
        instrumentList.append(t_bond)
    else:
        instrumentList += appended_instruments

    dataDict = manager.getDataDictOfMatrix(instrumentList=instrumentList, ratioList=[], fromDate=fromDate,
                                           toDate=toDate)
    # % save data to excel
    if save_file:
        for key in dataDict.keys():
            dataDict[key] = dataDict[key].dropna(axis=0)

        dataDict_to_excel(dataDict, 'historical_data_robotrader.xlsx')
    return dataDict


def get_input(dataDict, delay_range=range(1, 5), sma_period_range=[20, 40, 60, 200],
              std_period_range=[20, 40, 60, 200]):
    inputMatrix = dataDict['close'].copy()
    # %

    # % add other columns
    for otherMatrix in dataDict.keys():
        if otherMatrix == 'close':
            continue
        for column in dataDict[otherMatrix].columns:
            name = '%s_%s' % (otherMatrix, column)
            inputMatrix[name] = dataDict[otherMatrix][column]
    input_columns = list(inputMatrix.columns)
    # %
    # add delay and returns past
    # delay_range = range(1, 5)
    for column in input_columns:
        for delay in delay_range:
            # inputMatrix['%s_%d'%(column,delay)]= inputMatrix[column].shift(delay)
            inputMatrix['returns_%s_%d' % (column, delay)] = inputMatrix[column].divide(
                inputMatrix[column].shift(delay)) - 1

    # % add moving average
    title = 'sma'
    # sma_period_range = [20, 40, 60, 200]
    for column in input_columns:
        for period in sma_period_range:
            inputMatrix['%s%s_%d' % (title, column, period)] = (inputMatrix[column].rolling(period).mean())
    ## volatility
    title = 'std'
    # std_period_range = [20, 40, 60, 200]
    for column in input_columns:
        for period in std_period_range:
            inputMatrix['%s%s_%d' % (title, column, period)] = (inputMatrix[column].rolling(period).std())

    ## you can add or remove what you want

    return inputMatrix


def get_target(dataDict, symbols, days_predict=1):
    returnsAll = get_returns(dataDict=dataDict, days_predict=days_predict, plot=False)
    targetMatrix = returnsAll.shift(-days_predict)  # to get the target return

    # % target Matrix
    target_returns = targetMatrix[symbols]
    columns_list = list(target_returns.columns)
    new_columns = [position for position in range(len(columns_list))]
    target_returns_position = pd.DataFrame(target_returns.values, columns=new_columns, index=target_returns.index)
    targetMatrix = target_returns_position.idxmax(axis=1)
    return targetMatrix


def get_returns(dataDict, days_predict=1, plot=True):
    closeMatrix = dataDict['close'].copy()
    returnsAll = closeMatrix.divide(closeMatrix.shift(days_predict)) - 1
    if plot:
        returnsAll.cumsum().plot()
        plt.title('cumsum Returns of data period= %d days' % days_predict)
        plt.show()
    return returnsAll


def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    '''
    Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
    https://stackoverflow.com/q/32239577/395857
    '''
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set(np.where(y_true[i])[0])
        set_pred = set(np.where(y_pred[i])[0])
        # print('\nset_true: {0}'.format(set_true))
        # print('set_pred: {0}'.format(set_pred))
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred)) / \
                    float(len(set_true.union(set_pred)))
        # print('tmp_a: {0}'.format(tmp_a))
        acc_list.append(tmp_a)
    return np.mean(acc_list)


def get_backtest_returns(returns_df, prediction_proba, plot=True):
    output = returns_df * prediction_proba
    if plot:
        plt.close()
        pnl_train = output.sum(axis=1).cumsum()
        pnl_train.plot()
    return output
