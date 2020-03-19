import numpy as np
from math import log, floor
import heapq


class QBCA:
    def __init__(self):
        self.n_dims = 0
        self.bins_per_dim = 0
        self.bin_sizes = []
        self.bins = []
        self.data_mins = []

    def __init_bins(self, X):
        self.n_dims = X.shape[-1]
        self.bins_per_dim = int(floor(log(X.shape[0], self.n_dims)))
        self.bin_sizes = (np.amax(X, axis=0) - np.amin(X, axis=0)) / self.bins_per_dim
        self.data_mins = np.amin(X, axis=0)
        self.bins = np.zeros([self.bins_per_dim] * self.n_dims, dtype=np.uint32)

    def __quantize_sample(self, data_point):
        mask = (data_point == self.data_mins)
        data_point[mask] = 1
        data_point[~mask] = data_point[~mask] - self.data_mins[~mask]
        data_point = np.ceil(data_point / self.bin_sizes).astype(int)
        p_array = np.full_like(data_point, self.bins_per_dim)
        dim_exp = np.arange(data_point.shape[-1] - 1, -1, -1)
        p_array = np.power(p_array, dim_exp)
        # Jo crec que hi ha un error a la formula
        bin_idx = np.sum(np.multiply(data_point - 1, p_array))  # data_point[-1]
        return bin_idx

    def __quantization(self, X):
        self.__init_bins(X)
        sample_bins = np.apply_along_axis(self.__quantize_sample, axis=1, arr=X)
        bin_idxs = np.unravel_index(sample_bins, self.bins.shape)
        bin_idxs = list(zip(*bin_idxs))
        for idx in bin_idxs:
            self.bins[idx] += 1

    def __center_initialization(self):
        # Empty max heap H
        h = heapq._heapify_max([])
        # Seed list L
        l = []
        # Calculate number of points in each histogram bin
        # Process all non zero bins
        while h is not 'empty':
            # get next bin of h
            # flag = true
            # For each neightboring bin
                # if size neightbor is bigger flag = false
                # (IF any neightbor bigger dont add b into L)
            # if flag then add bin into seed list
        # Case if not enough
        # Later more
            pass

    def fit(self, X):
        self.__quantization(X)
        self.__center_initialization()
        # CCI (init)
        # CCA (assign)


if __name__ == '__main__':
    from sklearn import datasets

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    q = QBCA()
    q.fit(X)
