from Svr import mysvr
from Ar import myar
from MLPR import mymlpr
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import quandl


data = quandl.get("EOD/DIS", authtoken="-L9Dut7Vm4mKvxKxyP_H",
                  returns="numpy", start_date="2016-12-27", end_date="2017-12-27")
x, i = myar(data)
y, z = mysvr(data)
w = mymlpr(data)
data_sum = pd.DataFrame([x, y, z, w])
df = data_sum.median()
q = range(len(df))
plt.plot(q, y, label='PSO-SVR')
plt.plot(q, z, label='GRID-SVR')
plt.plot(q, w, label='MLPR')
plt.plot(q, x, label='AR')
plt.plot(q, df, label='Hibrido')
plt.plot(q, i, label='Real')
plt.title("The Walt Disney Company (DIS) Stock Prices")
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
