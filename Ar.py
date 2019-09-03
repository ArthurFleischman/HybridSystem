import numpy as np
from math import floor, ceil
from prepro import prepro


def myar(qdata):
    data = qdata
    x_train, y_train, x_val, y_val, x_test, y_test = prepro(12, data["Open"])

    in1 = np.linalg.pinv(x_train)
    yt = y_train.T

    coef = in1.dot(yt)
    out = x_test.dot(coef.T)
    return out, y_test


# if __name__ == '__main__':
#     df = pd.DataFrame(myar())
#     print(df)
