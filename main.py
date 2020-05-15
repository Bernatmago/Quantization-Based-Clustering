from QBCA import QBCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import time
from utils import *
from evaluate import dunn_index

if __name__ == '__main__':
    n_clusters = [2, 3, 4, 5]
    algorithms = ['K-Means', 'QBCA']
    thr = 0.0001
    max_iter = 50
    n_reps = 10
    n_dims = 5
    images = ['totoro', 'cybertruck']
    datasets = ['test', 'satimage']
    datasets = datasets + images
    for dataset in datasets:
        print('{} Running experiments for {} {}'.format('-'*10, dataset, '-'*10))
        experiments_shilouette = []
        experiments_distances = []
        if dataset in images:
            thr = 0.01
        else:
            thr = 0.0001
        # experiments_dunn = []
        for alg in algorithms:
            algorithm_shilouette = []
            algorithm_distances = []
            algorithm_dunn = []
            for nc in n_clusters:
                shil_reps = np.zeros(n_reps)
                dist_reps = np.zeros(n_reps)
                # dunn_reps = np.zeros(n_reps)
                time_reps = np.zeros(n_reps)
                iter_reps = np.zeros(n_reps)
                for r in range(n_reps):
                    if dataset in images:
                        X, _ = image_as_dataset(dataset, im_height=100)
                    else:
                        X, _ = load_data(dataset, n_dims=n_dims, has_labels=True)
                    t = time.time()
                    if alg == 'K-Means':
                        clf = KMeans(n_clusters=nc, tol=thr, max_iter=max_iter, precompute_distances=False, algorithm='full', n_init=1)
                        y = clf.fit_predict(X)
                    else:
                        clf = QBCA(n_seeds=nc, thr=thr, max_iter=max_iter)
                        X, y = clf.fit_predict(X)

                    if alg == 'K-Means':
                        # n_dist = clf.n_iter_ * X.shape[0]
                        n_dist = clf.n_iter_ * X.shape[0]
                    else:
                        n_dist = clf.dist_count
                    shil_reps[r] = silhouette_score(X, y)
                    dist_reps[r] = n_dist
                    # dunn_reps[r] = dunn_index(cluster_list(X, y))
                    time_reps[r] = time.time() - t
                    iter_reps[r] = clf.n_iter_

                    del clf

                algorithm_shilouette.append(np.mean(shil_reps))
                algorithm_distances.append(np.mean(dist_reps))
                # algorithm_dunn.append(np.mean(dunn_reps))
                print('{} with {} clusters took {} seconds (mean of {} reps)'.format(alg, nc, np.mean(time_reps), n_reps))
                print('Mean {} iterations'.format(np.mean(iter_reps)))
                print('Mean {} distance computations'.format(algorithm_distances[-1]))
                print('Mean Shilouette: {}'.format(algorithm_shilouette[-1]))
                # print('Mean Dunn: {}'.format(algorithm_dunn[-1]))

            experiments_shilouette.append(algorithm_shilouette)
            experiments_distances.append(algorithm_distances)
            # experiments_dunn.append(algorithm_dunn)

        plot_metrics(experiments_shilouette, n_clusters, algorithms, 'Silhouette', 'N-Clusters', dataset)
        # plot_metrics(experiments_dunn, n_clusters, algorithms, 'Dunn', 'N-Clusters', dataset)
        plot_metrics(experiments_distances, n_clusters, algorithms, 'Dist_C', 'N-Clusters', dataset)
