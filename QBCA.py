import numpy as np
from math import log, floor
import heapq
from itertools import combinations, permutations, product


class QBCA:
    def __init__(self):
        self.n_dims = 0
        self.bins_per_dim = 0
        self.bin_sizes = []
        self.bins = []
        self.bins_points = []
        self.data_mins = []
        self.bins_shape = ()
        self.seeds = []

    def __init_bins(self, X):
        self.n_dims = X.shape[-1]
        self.bins_per_dim = int(floor(log(X.shape[0], self.n_dims)))
        self.bin_sizes = (np.amax(X, axis=0) - np.amin(X, axis=0)) / self.bins_per_dim
        self.data_mins = np.amin(X, axis=0)
        self.bins = np.zeros(self.bins_per_dim ** self.n_dims, dtype=np.uint32)
        self.bins_points = [[] for _ in range(self.bins_per_dim ** self.n_dims)]
        self.bins_shape = tuple([self.bins_per_dim] * self.n_dims)
        self.neigh_idx = list(product([-1, 0, 1], repeat=self.n_dims))
        self.neigh_idx.remove(tuple(np.zeros(self.n_dims)))
        self.neigh_idx = np.array(self.neigh_idx)

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
        bin_idxs = np.apply_along_axis(self.__quantize_sample, axis=1, arr=X)
        for idx, p in zip(bin_idxs, X):
            self.bins[idx] += 1
            self.bins_points[idx].append(p)

    def __get_neigh_idx(self, idx):
        unr = np.unravel_index(idx, (self.bins_shape))
        # Use offsets to get neighbors
        n_idx = self.neigh_idx + unr
        # Remove invalid
        n_idx = n_idx[~(n_idx < 0).any(1)]
        n_idx = n_idx[~(n_idx >= self.n_dims).any(1)]
        n_bins = []
        for i in n_idx:
            # b = np.ravel_multi_index(i, (self.bins_shape))
            print(i)
            n_bins.append(np.ravel_multi_index(i, (self.bins_shape)))
        return np.array(n_bins)

        print(1)

    def __bin_cardinality(self, bin_tuple):
        return bin_tuple[0]

    def __center_initialization(self, n_seeds):
        self.seeds = np.zeros((n_seeds, self.n_dims))
        s_bins = []
        # Process all non zero bins
        nozero_idx = np.where(self.bins != 0)
        nozero_bins = self.bins[nozero_idx]

        # To make heap work as max heap changed value symbol
        h = [(-x[0], x[1]) for x in np.vstack((nozero_bins, nozero_idx)).T]
        heapq.heapify(h)
        h_copy = h[:]

        while len(h) > 0:
            a = heapq.heappop(h)
            print(a)
            n_idxs = self.__get_neigh_idx(a[1])
            flag = True
            for n_idx in n_idxs:
                if self.bins[n_idx] > (-a[0]):
                    flag = False
            if flag:
                s_bins.append(a)

        # If need more seeds
        if len(s_bins) < n_seeds:
            # Select k − | L | histogram bins with the largest cardinalities not in L
            s_bins.extend([x for x in h_copy if x not in s_bins][:(n_seeds - len(s_bins))])
            pass
        # For the histogram bins with the largest cardinalities in L
        s_bins.sort(key=self.__bin_cardinality)
        for s_idx, (_, s_bin) in enumerate(s_bins[: n_seeds]):
            # Cluster all the points in each histogram bin as one cluster
            # The center is the seed
            self.seeds[s_idx] = np.mean(np.vstack(self.bins_points[s_bin]), axis=0)

    def fit(self, X, n_seeds=3):
        self.__quantization(X)
        self.__center_initialization(n_seeds)
        # CCI (init)
        # CCA (assign)


if __name__ == '__main__':
    from sklearn import datasets

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    q = QBCA()
    q.fit(X)

# bin_idxs = np.unravel_index(sample_bins, self.bins.shape)
# bin_idxs = list(zip(*bin_idxs))
# for idx, p in zip(bin_idxs, X):
#     self.bins[idx] += 1
#     self.bins_points
