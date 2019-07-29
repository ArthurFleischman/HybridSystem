import numpy as np
import pandas as pd


def prepro(dim, data):
    '''
    preprocessing and preparing data 
    '''
    series = pd.Series(data)
    lag = pd.concat([series.shift(i) for i in range(dim+1)], axis=1)
    train_len = int(np.floor(0.6*len(data)))
    val_len = int(np.floor(0.8*len(data)))

    x_train = lag.iloc[dim:train_len, 1:dim+1]
    y_train = lag.iloc[dim:train_len, 0]

    x_test = lag.iloc[train_len:, 1:dim+1]
    y_test = lag.iloc[train_len:, 0]

    x_val = lag.iloc[train_len:val_len, 1:dim + 1]
    y_val = lag.iloc[train_len:val_len, 0]

    x_test = lag.iloc[val_len:, 1:dim+1]
    y_test = lag.iloc[val_len:, 0]

    return (x_train, y_train, x_val,y_val,x_test,y_test)
