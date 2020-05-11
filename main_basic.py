from sklearn.decomposition import PCA
from sklearn import datasets
import pandas as pd
from QBCA import QBCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import time
from utils import *
from evaluate import dunn_index
if __name__ == '__main__':
    X, y = load_data('satimage', n_dims=5)
    n_clusters = 5
    thr = 0.0001

    q = QBCA()
    km = KMeans(n_clusters=n_clusters, tol=thr, precompute_distances=False, algorithm='full', n_init=1)

    t = time.time()
    y_k = km.fit_predict(X)
    print('Kmeans took {} seconds'.format(time.time() - t))
    print('{} distance computations ({} iter)'.format(km.n_iter_ * X.shape[0], km.n_iter_))
    # print('Dunn index: {}'.format(dunn_index(cluster_list(X, y_k))))
    print('Shilouette: {}'.format(silhouette_score(X, y_k)))

    t = time.time()
    X_2, y = q.fit(X, n_seeds=n_clusters, thr=thr)
    print('Qbca took {} seconds'.format(time.time() - t))
    print('{} distance computations ({} iter)'.format(q.dist_count, q.n_iter_))
    # print('Dunn index: {}'.format(dunn_index(cluster_list(X_2, y))))
    print('Shilouette: {}'.format(silhouette_score(X_2, y)))
    # b = np.isin(X_2, X)
    plot_comparison(X, y_k, X_2, y, 'K-Means', 'QBCA')