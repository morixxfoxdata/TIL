from IPython.core.pylabtools import figsize
import numpy as np
import matplotlib.pyplot as plt
import os


figsize(12.5, 3.5)
path = os.path.dirname(__file__)
count_data = np.loadtxt(path + "/data/txtdata.csv")
n_count_data = len(count_data)

plt.bar(np.arange(n_count_data), count_data, color="#348ABD")
plt.xlim(0, n_count_data)
plt.show()