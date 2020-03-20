import numpy as np
from math import log, floor, sqrt
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
        self.seeds_prev = []
        self.seed_points = []

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

    def __get_nonzero_bins(self):
        nozero_idx = np.where(self.bins != 0)
        nozero_bins = self.bins[nozero_idx]
        return nozero_bins, nozero_idx[0]

    def __get_neigh_idx(self, idx):
        unr = np.unravel_index(idx, self.bins_shape)
        # Use offsets to get neighbors
        n_idx = self.neigh_idx + unr
        # Remove invalid
        n_idx = n_idx[~(n_idx < 0).any(1)]
        n_idx = n_idx[~(n_idx >= self.n_dims).any(1)]
        n_bins = []
        for i in n_idx:
            # b = np.ravel_multi_index(i, (self.bins_shape))
            n_bins.append(np.ravel_multi_index(i, self.bins_shape))
        return np.array(n_bins)

    def __bin_cardinality(self, bin_tuple):
        return bin_tuple[0]

    def __compute_center(self, points):
        a = np.mean(np.vstack(points), axis=0)
        return np.mean(np.vstack(points), axis=0)

    def __center_initialization(self, n_seeds):
        self.seeds = np.zeros((n_seeds, self.n_dims))
        s_bins = []
        # Process all non zero bins
        nozero_bins, nozero_idx = self.__get_nonzero_bins()
        # To make heap work as max heap changed value symbol
        h = [(-x[0], x[1]) for x in np.vstack((nozero_bins, nozero_idx)).T]
        heapq.heapify(h)
        h_copy = h[:]

        while len(h) > 0:
            a = heapq.heappop(h)
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
        # For the histogram bins with the largest cardinalities in L
        s_bins.sort(key=self.__bin_cardinality)
        for s_idx, (_, s_bin) in enumerate(s_bins[: n_seeds]):
            # Compute center with bin points
            self.seeds[s_idx] = self.__compute_center(self.bins_points[s_bin])

    def __max_coords_dim(self,dim, idx):
        return self.bin_sizes[dim] * (idx +1)

    def __min_coords_dim(self,dim, idx):
        return self.bin_sizes[dim] * idx

    def __value_bin_dim(self,seed, idx, dim):
        if seed[dim] < self.__min_coords_dim(dim, idx[dim]):
            return self.__min_coords_dim(dim, idx[dim])
        elif seed[dim] > self.__max_coords_dim(dim, idx[dim]):
            return self.__max_coords_dim(dim, idx[dim])
        else:
            return seed[dim]

    def __min_distance(self, seed, bin_idx):
        # seed = np_array of n dims
        # bin_idx to look at bin stuff
        # bin = self.bins[bin_idx]
        b = np.zeros(self.n_dims)
        bin_idx = np.unravel_index(bin_idx, self.bins_shape)
        for d in range(self.n_dims):
                b[d] = self.__value_bin_dim(seed, bin_idx, d)
        return sqrt(np.sum(np.power(seed-b, 2)))

    def __max_distance(self, seed, bin_idx):
        lb = np.zeros(self.n_dims)
        bin_idx = np.unravel_index(bin_idx, self.bins_shape)
        for d in range(self.n_dims):
            t = (self.__min_coords_dim(d, bin_idx[d]) + self.__max_coords_dim(d, bin_idx[d])) / 2
            if seed[d] >= t:
                lb[d] = t
            else:
                lb[d] = self.__max_coords_dim(d, bin_idx[d])
        return sqrt(np.sum(np.power(seed-lb, 2)))

    def __center_assignment(self, n_seeds):
        self.seed_points = [[] for _ in range(len(self.seeds))]
        self.seeds_prev = np.copy(self.seeds)

        nonzero_bins, nonzero_idxs = self.__get_nonzero_bins()
        for i, b_idx in enumerate(nonzero_idxs):
            candidates = []
            lowest_max_dist = None
            for s in self.seeds:
                max_dist = self.__max_distance(s, b_idx)
                if not lowest_max_dist or max_dist < lowest_max_dist:
                    lowest_max_dist = max_dist

            for s_idx, s in enumerate(self.seeds):
                min_dist = self.__min_distance(s, b_idx)
                if min_dist <= lowest_max_dist:
                    candidates.append(s_idx)

            for p in self.bins_points[b_idx]:
                lowest_dist = None
                for s_idx in candidates:
                    dist = np.linalg.norm(self.seeds[s_idx]-p)
                    if not lowest_dist or dist < lowest_max_dist:
                        lowest_dist = dist
                        candidate_idx = s_idx
                self.seed_points[candidate_idx].append(p)
        # Recompute centers
        for s_idx, _ in enumerate(self.seeds):
            self.seeds[s_idx] = self.__compute_center(self.seed_points[s_idx])

    def __end_operation(self):
        # for each seed
        o1 = []
        for s_prev, s in zip(self.seeds_prev, self.seeds):
            o2 = np.zeros(self.n_dims)
            o1.append(np.sum(s-s_prev) ** 2)
        d = len(self.seeds)
        return np.sum(o1) / len(self.seeds)

    def __output_points_preds(self, X):
        X_out = np.zeros(X.shape)
        y = np.zeros(X.shape[0])
        acc_len = 0
        for seed, s_points in enumerate(self.seed_points):
            X_out[:len(s_points), :] = s_points
            y[:len(s_points)] = seed
            acc_len += len(s_points)
        return X_out, y

        pass
    def fit(self, X, n_seeds=3, thr=0.1):
        end_value = thr+1
        self.__quantization(X)
        self.__center_initialization(n_seeds)
        while end_value > thr:
            self.__center_assignment(n_seeds)
            end_value = self.__end_operation()
        # The order of x is not maintained
        X_out, y = self.__output_points_preds(X)
        pass

# MIRWAR BUG CENTRESS MORTS I TAL


if __name__ == '__main__':
    from sklearn import datasets
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    q = QBCA()
    q.fit(X)


