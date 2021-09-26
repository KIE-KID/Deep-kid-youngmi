import numpy as np
import matplotlib.pylab as plt

def identity_function(x):
    return x

x = np.arange(-5.0, 5.0, 0.1)
y = identity_function(x)
plt.plot(x, y)
plt.ylim(-5.5, 5.5)
plt.show()
