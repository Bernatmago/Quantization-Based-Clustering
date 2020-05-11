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
    n_clusters = [2, 3, 5, 10]
    algorithms = ['K-Means', 'QBCA']
    thr = 0.0001
    n_reps = 5
    experiments_shilouette = []
    experiments_distances = []
    for alg in algorithms:
        algorithm_shilouette = []
        algorithm_distances = []
        for nc in n_clusters:
            sr = np.zeros(n_reps)
            dr = np.zeros(n_reps)
            tr = np.zeros(n_reps)
            for r in range(n_reps):
                X, y = load_data('satimage', n_dims=5)
                t = time.time()
                if alg == 'K-Means':
                    clf = KMeans(n_clusters=nc, tol=thr, precompute_distances=False, algorithm='full', n_init=1)
                    y = clf.fit_predict(X)
                else:
                    clf = QBCA(n_seeds=nc, thr=thr)
                    X, y = clf.fit_predict(X)

                if alg == 'K-Means':
                    n_dist = clf.n_iter_ * X.shape[0]
                else:
                    n_dist = clf.dist_count
                sr[r] = silhouette_score(X, y)
                dr[r] = n_dist
                tr[r] = time.time() - t
                # print('Dunn index: {}'.format(dunn_index(cluster_list(X, y_k))))

            algorithm_shilouette.append(np.mean(sr))
            algorithm_distances.append(np.mean(dr))
            print('{} with {} clusters took {} seconds (mean of {} reps)'.format(alg, nc, np.mean(tr), n_reps))
            print('Mean {} distance computations'.format(algorithm_distances[-1]))
            print('Mean Shilouette: {}'.format(algorithm_shilouette[-1]))

        experiments_shilouette.append(algorithm_shilouette)
        experiments_distances.append(algorithm_distances)

    plot_metrics(experiments_shilouette, n_clusters, algorithms, 'Silhouette', 'N-Clusters')
    plot_metrics(experiments_distances, n_clusters, algorithms, 'Dist_calculations', 'N-Clusters')
    # plot_comparison(X, y_k, X_2, y, 'K-Means', 'QBCA')
