import matplotlib.pyplot as plt
import numpy as np

a = np.arange(6).reshape(2, 3)
print(a)
# [[0 1 2]
#  [3 4 5]]

np.savetxt('./data/np_savetxt.csv', a, fmt="%.5f")