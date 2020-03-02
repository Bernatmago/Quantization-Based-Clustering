from math import log, floor
import numpy as np

from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Quantization process
# Map all data points to a m dimensional histogram B
# Determine de number of bins per dimension: p
# Determine size of histogram bin in each dimension size_dim

# No se de on surt m (number of dimensions) entenc que depen del dataset
# Bins per dimension: p
bins_per_dim = int(floor(log(X.shape[0], X.shape[-1])))

print("Bins per dimension:", bins_per_dim)

# Size of each dimenion
# (Max value - Min value) / bins per dim
size_dims = np.amax(X, axis=0) - np.amin(X, axis=0) / bins_per_dim

print(size_dims)

# Ara quantizar (volem saber a quin index assignar quin valor
# Operacio simbol raro (es fa per a cada dimensio)
    # Si valor punt a dimensio l es mes gran que el minim (valor - minim) / tamany dimensio l
    # Si valor punt a dimensio l es igual al minim aleshores 1
    # Arrodonir cap a dalt
data_mins = np.amin(X, axis=0)

def strange_op(data_point):
    mask = (data_point == data_mins)
    data_point[mask] = 1
    data_point[~mask] = data_point[~mask] - data_mins[~mask]
    return np.ceil(data_point).astype(int)

strange_result = np.apply_along_axis(strange_op, axis=1, arr=X)
print(strange_result)
