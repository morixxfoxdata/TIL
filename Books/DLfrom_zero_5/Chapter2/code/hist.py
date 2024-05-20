import os
import numpy as np
# print(os.path.dirname(__file__))
def export_xs():
    path = os.path.join(os.path.dirname(__file__), 'height.txt')
    xs = np.loadtxt(path)
    return xs
# print(xs.shape)     # (25000,)

# plt.hist(xs, bins='auto', density=True)
# plt.xlabel('Height(cm)')
# plt.ylabel('Probability Density')
# plt.show()


