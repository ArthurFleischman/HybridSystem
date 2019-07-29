import numpy as np
from math import floor, ceil
from matplotlib import pyplot as plt
import quandl
from prepro import prepro


def myar():
    data = quandl.get("EOD/DIS", authtoken="-L9Dut7Vm4mKvxKxyP_H",
                      returns="numpy", start_date="2016-12-27", end_date="2017-12-27")
    x_train, y_train, x_val, y_val, x_test, y_test = prepro(12, data["Open"])

    #ones = np.matrix([1]*len(x_train)).T
    #xt = np.append(x_train, ones, axis=1)
    in1 = np.linalg.pinv(x_train)
    yt = y_train.T

    coef = in1.dot(yt)

    #ones1 = np.matrix([1]*len(x_test)).T
    #ni = np.append(x_test, ones1, axis=1)
    out = x_test.dot(coef.T)
    '''
    nout = y_test.T
    x = range(len(out))
    plt.plot(x, out, label='AR')
    plt.plot(x, nout, label='Real')
    plt.title("The Walt Disney Company (DIS) Stock Prices")
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    '''
    return out, y_test


if __name__ == '__main__':
    myar()
