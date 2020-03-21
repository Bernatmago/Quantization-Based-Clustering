from sklearn.decomposition import PCA
from sklearn import datasets
from QBCA import QBCA
import matplotlib.pyplot as plt
if __name__ == '__main__':
    iris = datasets.load_iris()
    X = iris.data
    # y = iris.target

    pca = PCA(n_components=2)
    X = pca.fit_transform(X)

    q = QBCA()
    X_2, y = q.fit(X, n_seeds=2)

    fig, axs = plt.subplots(2, 1)
    axs[0].set_title('Original')
    axs[0].scatter(X[:, 0], X[:, 1])
    axs[1].set_title('Clustered')
    axs[1].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
    plt.show()