import numpy as np
import matplotlib.pyplot as plt
import math



x, y = np.loadtxt(f'test/graph/interference_data.csv', delimiter = ',', usecols = (0, 1), unpack = True)

A = np.vstack([x, np.ones(len(x))]).T
m, c = np.linalg.lstsq(A, y, rcond = -1)[0]
print(m, c)

fig, (graph) = plt.subplots(1, 1, sharex = False, sharey = True, figsize = (4.5, 4.5))

(graph).set_xlabel("incidence ", fontsize = 14)
(graph).set_ylabel("refraction", fontsize = 14)
(graph).grid(color = 'red', linestyle = '--', linewidth = 1)
(graph).set_xlim(0, np.max(x) * 1.1)
(graph).set_ylim(0, np.max(y) * 1.1)
(graph).plot(x, y, color = 'blue', marker = 'o', markersize = 4, linestyle = '', label = "Original Data")
(graph).plot(x, m * x + c, color = 'orange', linewidth = 2, label = "Fitted Line")
(graph).legend(loc = 'upper left')

fig.savefig('graph.svg')
fig.savefig('graph.png')
fig.show()