import numpy as np
from math import floor, ceil
from matplotlib import pyplot as plt
import quandl


def myar():

    #data = np.loadtxt('airlines2.txt')
    data = quandl.get("EOD/DIS", authtoken="-L9Dut7Vm4mKvxKxyP_H",
                      returns="numpy", start_date="2016-12-27", end_date="2017-12-27")

    window = 12  # line size -1
    newdata = division(data["Open"])
    len80 = len(newdata[0])
    len20 = len(newdata[1])

    inputs = np.matrix([newdata[0][z:z+window]
                        for z in range(len80-1) if z+window <= len80-1])
    ones = np.matrix([1]*len(inputs)).T
    inputs = np.append(inputs, ones, axis=1)
    inputs1 = np.linalg.pinv(inputs)
    # sai a inversa de inputs como inputs1
    outputs = np.matrix([newdata[0][window+z] for z in range(len80-1)
                         if z+window <= len80-1])  # outputs dos 80%
    outputs = outputs.T
    coef = inputs1.dot(outputs)  # coef dos 80%
    # sai coeficientes

    newinputs = np.matrix([newdata[1][z:z + window]
                           for z in range(len20 - 1) if z + window <= len20 - 1])
    ones1 = np.matrix([1]*len(newinputs)).T
    newinputs = np.append(newinputs, ones1, axis=1)
    outputs1 = newinputs.dot(coef)
    # 20% * coeficientes

    # 'Y's dos 20%
    newoutputs = np.matrix([newdata[1][window + z]
                            for z in range(len20 - 1) if z + window <= len20 - 1])
    newoutputs = newoutputs.T
    plot(outputs1, newoutputs)
    return outputs1


def division(data1):

    list_80 = []
    list_20 = []
    newlist = []

    for w in range(floor(len(data1) * 0.8)):
        list_80.append(data1[w])
    newlist.append(list_80)
    for w in range(ceil(len(list_80)), len(data1)):
        list_20.append(data1[w])
    newlist.append(list_20)
    return newlist


def plot(*args):
    x = range(len(args[0]))
    plt.plot(x, args[0], label='AR')
    plt.plot(x, args[1], label='Real')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.title("The Walt Disney Company (DIS) Stock Prices")
    plt.show()


if __name__ == '__main__':
    myar()
