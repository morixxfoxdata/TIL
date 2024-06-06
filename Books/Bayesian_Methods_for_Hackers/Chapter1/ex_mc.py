import pymc as pm
from IPython.core.pylabtools import figsize
import numpy as np
import matplotlib.pyplot as plt
import os


path = os.path.dirname(__file__)
count_data = np.loadtxt(path + "/data/txtdata.csv")
n_count_data = len(count_data)

alpha = 1 / count_data.mean()
lambda_1 = pm.Exponential("lambda_1", alpha)
lambda_2 = pm.Exponential("lambda_2", alpha)

tau = pm.DiscreteUniform("tau", lower=0, upper=n_count_data)

print("Random output:", tau.random(), tau.random(), tau.random())