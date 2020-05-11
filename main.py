from sklearn.decomposition import PCA
from sklearn import datasets
import pandas as pd
from QBCA import QBCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
if __name__ == '__main__':
    # iris = datasets.load_iris()
    # X = iris.data
    # y = iris.target
    df = pd.read_csv('data/satimage_csv.csv')
    X = df.to_numpy()[:, :-1]
    pca = PCA(n_components=2)
    X = pca.fit_transform(X)
    n_clusters = 5
    q = QBCA()
    km = KMeans(n_clusters=n_clusters)
    X_2, y = q.fit(X, n_seeds=n_clusters)
    y_k = km.fit_predict(X)
    b = np.isin(X_2, X)
    fig, axs = plt.subplots(2, 1)
    axs[0].set_title('Kmeans')
    axs[0].scatter(X[:, 0], X[:, 1], c=y_k)
    axs[1].set_title('Qbca')
    axs[1].scatter(X_2[:, 0], X_2[:, 1], c=y)
    plt.show()
    print(1)