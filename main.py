import numpy as np
import matplotlib.pyplot as plt
import math

x_name = input("輸入 x 軸名稱")
y_name = input("輸入 y 軸名稱")



x, y = np.loadtxt(f'data.csv', delimiter = ',', usecols = (0, 1), unpack = True)

A = np.vstack([x, np.ones(len(x))]).T
m, c = np.linalg.lstsq(A, y, rcond = -1)[0] # 計算回歸直線
print(m, c)

fig, (graph) = plt.subplots(1, 1, sharex = False, sharey = True, figsize = (4.5, 4.5))

(graph).set_xlabel(x_name, fontsize = 14)#字體大小
(graph).set_ylabel(y_name, fontsize = 14)
(graph).grid(color = 'red', linestyle = '--', linewidth = 1)
(graph).set_xlim(0, np.max(x) * 1.1) #xy軸極值大小
(graph).set_ylim(0, np.max(y) * 1.1)
(graph).plot(x, y, color = 'blue', marker = 'o', markersize = 4, linestyle = '', label = "Original Data")
(graph).plot(x, m * x + c, color = 'orange', linewidth = 2, label = "Fitted Line")
(graph).legend(loc = 'upper left')

fig.savefig('graph.svg')
fig.savefig('graph.png')