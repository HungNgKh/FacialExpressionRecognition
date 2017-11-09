import matplotlib.pyplot as plt
import numpy as np

arr = np.load('../saved/recorded/accuracy.npy')

arr = arr.T[1]

plt.plot(range(1, np.size(arr) + 1), arr)
plt.show()

