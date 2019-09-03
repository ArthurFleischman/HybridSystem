from matplotlib import pyplot as plt
from prepro import prepro
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error as MSE
import numpy as np
from gridSVR import gridSVR as grd
import quandl
from Pso import pso


def mysvr(qdata):
    #data = np.loadtxt('airlines2.txt')
    data = qdata
    (x_train, y_train, x_val, y_val, x_test, y_test) = prepro(12, data["Open"])

    print('initializing grid...')
    best_model, best_predicts, best_error, best_param = grd(
        x_train, y_train, x_val, y_val)
    print('grid ready')

    best_predicts = best_model.predict(x_test)
    best_error = MSE(best_predicts, y_test)

    print('initializing pso...')
    g_best = pso(30, 50, x_train, y_train, x_val, y_val)
    print('pso ready')

    print('initializing svr')
    model = SVR(C=g_best[0], gamma=g_best[1], epsilon=g_best[2])
    print('svr ready')

    model.fit(x_train, y_train)
    predicts = model.predict(x_test)

    return predicts, best_predicts


if __name__ == '__main__':
    mysvr()
