import numpy as np
from math import log, floor, sqrt
import heapq
from itertools import product


class QBCA:
    def __init__(self, n_seeds=3, thr=0.001, max_iter=50, verbose=False):
        """"Initialize the algortihm"""
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
        self.dist_count = 0
        self.n_iter_ = 0
        self.n_seeds = n_seeds
        self.thr = thr
        self.max_iter = max_iter
        self.verbose = verbose

    def __init_bins(self, X):
        """"Initialize all the quantization related variables"""
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
        self.dist_count = 0

    def __quantize_sample(self, data_point):
        """Given a data point return its correspondent bin index"""
        mask = (data_point == self.data_mins)
        data_point[mask] = 1
        data_point[~mask] = data_point[~mask] - self.data_mins[~mask]
        data_point = np.ceil(data_point / self.bin_sizes).astype(int)
        p_array = np.full_like(data_point, self.bins_per_dim)
        dim_exp = np.arange(data_point.shape[-1] - 1, -1, -1)
        p_array = np.power(p_array, dim_exp)
        bin_idx = np.sum(np.multiply(data_point - 1, p_array))
        return bin_idx

    def __get_nonzero_bins(self):
        """Get all the non empty bins and their indexes"""
        nozero_idx = np.where(self.bins != 0)
        nozero_bins = self.bins[nozero_idx]
        return nozero_bins, nozero_idx[0]

    def __get_neigh_idx(self, idx):
        """Get all the neighboring bins indexes given an index"""
        unr = np.unravel_index(idx, self.bins_shape)
        n_idx = self.neigh_idx + unr
        n_idx = n_idx[~(n_idx < 0).any(1)]
        n_idx = n_idx[~(n_idx >= self.bins_per_dim).any(1)]
        n_bins = []
        for i in n_idx:
            n_bins.append(np.ravel_multi_index(i, self.bins_shape))
        return np.array(n_bins)

    def __bin_cardinality(self, bin_tuple):
        """Get bin cardinality from a tuple"""
        return bin_tuple[0]

    def __compute_center(self, points):
        """Compute the center from a set of points"""
        return np.mean(np.vstack(points), axis=0)

    def __heaapify__bins(self, b, b_idx):
        """Get bins as a heap based on their value"""
        h = [(-x[0], x[1]) for x in np.vstack((b, b_idx)).T]
        heapq.heapify(h)
        return h

    def __max_coords_dim(self, dim, idx):
        """Get the maximum coordinate value from a bin in a given dimension"""
        return self.bin_sizes[dim] * (idx + 1)

    def __min_coords_dim(self, dim, idx):
        """Get the minimum coordinate value from a bin in a given dimension"""
        return self.bin_sizes[dim] * idx

    def __value_bin_dim(self, seed, idx, dim):
        """Get the value from a bin in a given dimension"""
        if seed[dim] < self.__min_coords_dim(dim, idx[dim]):
            return self.__min_coords_dim(dim, idx[dim])
        elif seed[dim] > self.__max_coords_dim(dim, idx[dim]):
            return self.__max_coords_dim(dim, idx[dim])
        else:
            return seed[dim]

    def __min_distance(self, seed, bin_idx):
        """Compute minimum distance between a seed and a bin"""
        b = np.zeros(self.n_dims)
        bin_idx = np.unravel_index(bin_idx, self.bins_shape)
        for d in range(self.n_dims):
            b[d] = self.__value_bin_dim(seed, bin_idx, d)
        self.dist_count += 1
        return sqrt(np.sum(np.power(seed - b, 2)))

    def __max_distance(self, seed, bin_idx):
        """Compute the maximum distance between a seed and a bin"""
        lb = np.zeros(self.n_dims)
        bin_idx = np.unravel_index(bin_idx, self.bins_shape)
        for d in range(self.n_dims):
            t = (self.__min_coords_dim(d, bin_idx[d]) + self.__max_coords_dim(d, bin_idx[d])) / 2
            if seed[d] >= t:
                lb[d] = self.__min_coords_dim(d, bin_idx[d])
            else:
                lb[d] = self.__max_coords_dim(d, bin_idx[d])
        self.dist_count += 1
        return sqrt(np.sum(np.power(seed - lb, 2)))

    def __initial_seeds(self, n_seeds, h):
        """Compute the first seeds from a given bin heap"""
        h_copy = h[:]
        s_bins = []
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
            s_bins.extend([x for x in h_copy if x not in s_bins][:(n_seeds - len(s_bins))])
        s_bins.sort(key=self.__bin_cardinality)
        return s_bins

    def __lowest_max_dist(self, b_idx):
        """Compute lowest max dist of a given bin"""
        lowest_max_dist = None
        for s in self.seeds:
            max_dist = self.__max_distance(s, b_idx)
            if not lowest_max_dist or max_dist < lowest_max_dist:
                lowest_max_dist = max_dist
        return lowest_max_dist

    def __bin_candidates(self, b_idx, lowest_max_dist):
        """Get the seed candidates for a bin given the lowest max distance"""
        candidates = []
        for s_idx, s in enumerate(self.seeds):
            min_dist = self.__min_distance(s, b_idx)
            if min_dist <= lowest_max_dist:
                candidates.append(s_idx)
        return candidates

    def __assign_bin_points(self, b_idx, candidates):
        """Assign points fom a bin to their respective seeds"""
        for p in self.bins_points[b_idx]:
            lowest_dist = None
            candidate_idx = None
            for s_idx in candidates:
                dist = np.linalg.norm(self.seeds[s_idx] - p)
                if not lowest_dist or dist < lowest_dist:
                    lowest_dist = dist
                    candidate_idx = s_idx
            self.seed_points[candidate_idx].append(p)

    def __assign_bin_seeds(self, b_idx):
        """Assign a bin to its correspondent seeds"""
        lowest_max_dist = self.__lowest_max_dist(b_idx)
        candidates = self.__bin_candidates(b_idx, lowest_max_dist)
        self.__assign_bin_points(b_idx, candidates)

    def __end_operation(self):
        """Compute the ending criteria based on center changes"""
        seed_self_dist = []
        for s_prev, s in zip(self.seeds_prev, self.seeds):
            seed_self_dist.append(np.sum(s - s_prev) ** 2)
        return np.sum(seed_self_dist) / len(self.seeds)

    def __output_points_preds(self, X):
        """Output datapoints with their predictions"""
        x_out = np.zeros(X.shape)
        y = np.zeros(X.shape[0], dtype=int)
        acc_len = 0
        for seed, s_points in enumerate(self.seed_points):
            x_out[acc_len:(acc_len + len(s_points)), :] = s_points
            y[acc_len:(acc_len + len(s_points))] = seed
            acc_len += len(s_points)
        return x_out, y

    def __quantization(self, X):
        """Quantize the input data"""
        self.__init_bins(X)
        bin_idxs = np.apply_along_axis(self.__quantize_sample, axis=1, arr=X)
        for idx, p in zip(bin_idxs, X):
            self.bins[idx] += 1
            self.bins_points[idx].append(p)

    def __center_initialization(self, n_seeds):
        """Initialize centers from the initial quantization given a number of seeds"""
        self.seeds = np.zeros((n_seeds, self.n_dims))
        nozero_bins, nozero_idx = self.__get_nonzero_bins()
        h = self.__heaapify__bins(nozero_bins, nozero_idx)
        l = self.__initial_seeds(n_seeds, h)
        for s_idx, (_, s_bin) in enumerate(l[: n_seeds]):
            # Compute center with bin points
            self.seeds[s_idx] = self.__compute_center(self.bins_points[s_bin])

    def __center_assignment(self):
        """Assign points to their respective seeds"""
        self.seed_points = [[] for _ in range(len(self.seeds))]
        self.seeds_prev = np.copy(self.seeds)
        _, nonzero_idxs = self.__get_nonzero_bins()

        for b_idx in nonzero_idxs:
            self.__assign_bin_seeds(b_idx)
        # Recompute centers
        for s_idx, _ in enumerate(self.seeds):
            self.seeds[s_idx] = self.__compute_center(self.seed_points[s_idx])

    def fit(self, x):
        """Start clustering process untill ending criteria is met"""
        self.n_iter_ = 0
        end_value = self.thr + 1
        self.__quantization(x)
        self.__center_initialization(self.n_seeds)
        while end_value > self.thr and self.n_iter_ < self.max_iter:
            self.__center_assignment()
            end_value = self.__end_operation()
            if self.verbose: print(end_value)
            self.n_iter_ += 1
        return self

    def fit_predict(self, x):
        """Cluster data and output the final predictions"""
        self.fit(x)
        return self.__output_points_preds(x)

    def predict(self, x):
        """Output data predictions"""
        return self.__output_points_preds(x)


if __name__ == '__main__':
    from sklearn import datasets

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    q = QBCA()
    q.fit(X)
