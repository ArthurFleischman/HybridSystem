from sklearn.neural_network import MLPRegressor, MLPClassifier
import numpy as numpy
from prepro import prepro
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import quandl


def mymlpr(qdata):
    data = qdata

    n_neuronios = 50
    x_train, y_train, x_val, y_val, x_test, y_test = prepro(12, data["Open"])

    model = MLPRegressor(hidden_layer_sizes=(n_neuronios),
                         activation='relu', alpha=0.01)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    return pred


if __name__ == '__main__':
    mymlpr()
