from sklearn.neural_network import MLPRegressor, MLPClassifier
import numpy as numpy
from prepro import prepro
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import quandl


def mymlpr():
    data = quandl.get("EOD/DIS", authtoken="-L9Dut7Vm4mKvxKxyP_H",
                      start_date="2016-12-27", end_date="2017-12-27")

    n_neuronios = 50
    x_train, y_train, x_val, y_val, x_test, y_test = prepro(12, data["Open"])

    model = MLPRegressor(hidden_layer_sizes=(n_neuronios),
                         activation='relu', alpha=0.01)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    '''
    x = range(len(y_test))
    plt.plot(x, y_test, label='Real')
    plt.plot(x, pred, label='MLPR')
    plt.title("The Walt Disney Company (DIS) Stock Prices")
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    '''
    return pred


if __name__ == '__main__':
    mymlpr()
