from Svr import mysvr
from AR1 import myar
from MLPR import mymlpr
from matplotlib import pyplot as plt
import numpy as np

x, i = myar()
y, z = mysvr()
w = mymlpr()
new = x+y+z+w
new = new/4
q = range(len(new))
plt.plot(q, x, label='AR')
plt.plot(q, y, label='PSO-SVR')
plt.plot(q, z, label='GRID-SVR')
plt.plot(q, w, label='MLPR')
plt.plot(q, x, label='AR')
plt.plot(q, new, label='Hibrido')
plt.plot(q, i, label='Real')
plt.title("The Walt Disney Company (DIS) Stock Prices")
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
