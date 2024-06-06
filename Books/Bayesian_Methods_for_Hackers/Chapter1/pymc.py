import pymc as pm
from IPython.core.pylabtools import figsize
import numpy as np
import matplotlib.pyplot as plt
import os


path = os.path.dirname(__file__)
count_data = np.loadtxt(path + "/data/txtdata.csv")

