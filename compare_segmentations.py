from QBCA import QBCA
from sklearn.cluster import KMeans
import time
from utils import *

if __name__ == '__main__':
    n_clusters = 3
    thr = 0.01
    max_iter = 50
    dataset = 'cybertruck'
    algorithms = ['K-Means', 'QBCA']
    X, im = image_as_dataset(dataset, im_height=150)

    q = QBCA(n_seeds=n_clusters, thr=thr, max_iter=max_iter)
    km = KMeans(n_clusters=n_clusters, tol=thr, max_iter=50, precompute_distances=False, algorithm='full', n_init=1)

    t = time.time()
    y_k = km.fit_predict(X)
    print('Kmeans took {} seconds'.format(time.time() - t))

    t = time.time()
    X_2, y = q.fit_predict(X)
    print('Qbca took {} seconds'.format(time.time() - t))

    c1 = np.array(km.cluster_centers_)
    c2 = np.array(q.seeds)
    plot_segmented(X, y_k, X_2, y, c1, c2, im.shape, algorithms, dataset)
